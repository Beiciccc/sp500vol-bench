"""C1 BERT-base + S1 truncation strategy.

Truncates each filing to the first ``max_length`` tokens (default 512), encodes
via BERT-base, predicts realised volatility from CLS embedding via the
VolatilityHead. One submodel per horizon — matches HAR-RV / Block B pattern.

This module is intentionally self-contained: the fit/predict interface mirrors
VolatilityForecaster so the existing scripts/train.py dispatcher can drive it.
Training loop includes optional checkpoint resume via training.checkpoint.
"""

from __future__ import annotations

import functools
import math
import os
import pickle
from collections.abc import Iterable
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sp500vol.models.base import VolatilityForecaster
from sp500vol.models.classical_text._text_dataset import load_texts
from sp500vol.models.neural_text import _train_utils as train_utils
from sp500vol.models.neural_text.encoders import CLSEncoder, EncoderConfig
from sp500vol.models.neural_text.heads import VolatilityHead

_EPSILON = 1e-6
_FORCE_PRETOKENIZE_ENV = "SP500VOL_FORCE_PRETOKENIZE"
_TOKENIZER_THREADS_ENV = "SP500VOL_TOKENIZER_THREADS"
_TOKENIZATION_BATCH_SIZE_ENV = "SP500VOL_TOKENIZATION_BATCH_SIZE"
_PRETOK_REUSE_ENV = "SP500VOL_PRETOK_REUSE"  # cross-horizon token reuse; ON unless falsy


def _transformer_blocks(encoder):
    """Locate the ModuleList of transformer blocks inside a CLSEncoder wrapper for the
    top-4 freeze (HPO arm). CLSEncoder holds the HF model at `.encoder`; BERT/RoBERTa/
    Longformer expose their block list at `<hf>.encoder.layer`. Returns None if unfound."""
    import torch.nn as _nn

    hf = getattr(encoder, "encoder", encoder)  # unwrap CLSEncoder -> HF AutoModel
    for path in (("encoder", "layer"), ("transformer", "layer"), ("layers",), ("h",)):
        obj = hf
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, _nn.ModuleList) and len(obj) > 0:
            return obj
    return None


class _TextDataset(Dataset):
    """Pairs filing text with its (log-transformed) target."""

    def __init__(self, texts: list[str], targets: np.ndarray) -> None:
        if len(texts) != len(targets):
            raise ValueError("texts and targets length mismatch")
        self.texts = texts
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, float]:
        return self.texts[idx], float(self.targets[idx])


class _TokenizedTextDataset(Dataset):
    """Pairs pre-tokenised filing tensors with their targets. Sequences are stored RAGGED
    (variable length, truncation-only — no padding to max_length); the DataLoader's
    `_collate_tokenized` dynamically pads each batch to its own longest sequence, so short
    documents are never padded out to the full window."""

    def __init__(
        self,
        *,
        input_ids: list[torch.Tensor],
        attention_mask: list[torch.Tensor],
        targets: np.ndarray,
    ) -> None:
        if len(input_ids) != len(attention_mask):
            raise ValueError("input_ids and attention_mask length mismatch")
        if len(input_ids) != len(targets):
            raise ValueError("token tensors and targets length mismatch")
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "target": float(self.targets[idx]),
        }


class BertS1(VolatilityForecaster):
    """Per-horizon BERT-S1 regressor (truncated input → CLS → MLP → softplus)."""

    name = "C1_bert_s1"

    def __init__(
        self,
        *,
        pretrained: str = "bert-base-uncased",
        max_length: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 2.0e-5,
        weight_decay: float = 0.01,
        batch_size: int = 8,
        max_epochs: int = 3,
        early_stopping: bool = True,
        es_patience: int = 1,
        es_min_delta: float = 0.0,
        mixed_precision: str = "no",
        warmup_ratio: float = 0.0,
        grad_accumulation_steps: int = 1,
        objective: str = "mse",
        freeze_mode: str = "none",
        head_lr_mult: float = 1.0,
        device: str = "auto",
        log_target: bool = True,
        pretokenize: bool = False,
        tokenization_batch_size: int = 128,
        tokenizer_threads: int | None = None,
        checkpoint: bool = True,
        checkpoint_dir: str | Path | None = None,
        seed: int | None = None,
        strategy: str | None = None,
        dataloader_num_workers: int = 0,
        dataloader_persistent_workers: bool = False,
        dataloader_pin_memory: bool | None = None,
        dataloader_prefetch_factor: int | None = None,
    ) -> None:
        self.encoder_cfg = EncoderConfig(pretrained=pretrained, max_length=max_length)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.early_stopping = bool(early_stopping)
        self.es_patience = int(es_patience)
        self.es_min_delta = float(es_min_delta)
        self.mixed_precision = str(mixed_precision)
        self.warmup_ratio = float(warmup_ratio)
        self.grad_accumulation_steps = max(1, int(grad_accumulation_steps))
        self.objective = str(objective)
        self.freeze_mode = str(freeze_mode)
        self.head_lr_mult = float(head_lr_mult)
        self.log_target = bool(log_target)
        self.pretokenize = bool(pretokenize)
        self.tokenization_batch_size = max(1, int(tokenization_batch_size))
        self.tokenizer_threads = _optional_positive_int(tokenizer_threads)
        self.device = _resolve_device(device)
        self.checkpoint = bool(checkpoint)
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.seed = None if seed is None else int(seed)
        self.strategy = strategy or _infer_strategy(self.name)
        # DataLoader worker knobs (default 0 => legacy single-process loading).
        self.dl_num_workers = max(0, int(dataloader_num_workers))
        self.dl_persistent_workers = bool(dataloader_persistent_workers) and self.dl_num_workers > 0
        self._dl_pin_cfg = dataloader_pin_memory
        self.dl_prefetch_factor = (
            int(dataloader_prefetch_factor)
            if dataloader_prefetch_factor is not None and self.dl_num_workers > 0
            else None
        )
        self.models_: dict[Any, dict[str, Any]] = {}
        self.val_curves_: dict[Any, list[dict[str, Any]]] = {}

    def _dl_pin_memory(self) -> bool:
        """Pinned host buffers: explicit config wins, else CUDA-only (legacy default)."""
        if self._dl_pin_cfg is None:
            return self.device.type == "cuda"
        return bool(self._dl_pin_cfg)

    def _loader_kwargs(self) -> dict[str, Any]:
        """Shared DataLoader knobs. persistent_workers/prefetch_factor are dropped when
        num_workers==0 (torch rejects them) so legacy single-process loaders are
        unaffected."""
        kwargs: dict[str, Any] = {
            "num_workers": self.dl_num_workers,
            "pin_memory": self._dl_pin_memory(),
        }
        if self.dl_num_workers > 0:
            kwargs["persistent_workers"] = self.dl_persistent_workers
            if self.dl_prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.dl_prefetch_factor
        return kwargs

    # --- forecaster API --------------------------------------------------

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        df = _require_dataframe(X_train, name="X_train")
        target = np.asarray(y_train, dtype=float)
        if len(df) != len(target):
            raise ValueError(f"X_train has {len(df)} rows but y_train has {len(target)} values")

        texts = load_texts(df)
        horizons = _horizons(df)
        text_paths = df["text_path"].astype(str).tolist()
        val_texts, val_target, val_horizons, val_text_paths = self._prepare_val(X_val, y_val)
        # Tokenise each unique filing once across the horizon loop (and across val epochs);
        # scoped to this fit() call so the (large) cache never reaches model.pkl.
        if not (
            getattr(self, "_tok_cache_pinned", False)
            and getattr(self, "_tok_cache", None) is not None
        ):
            self._tok_cache = (
                self._new_token_cache()
                if self._use_pretokenize() and _pretok_reuse_enabled()
                else None
            )
        self.models_ = {}
        self.val_curves_ = {}
        try:
            for horizon in sorted(set(horizons.tolist())):
                mask = horizons == horizon
                h_val_texts, h_val_target, h_val_paths = _select_val(
                    val_texts, val_target, val_horizons, val_text_paths, horizon
                )
                self._fit_one(
                    horizon,
                    texts=[t for t, keep in zip(texts, mask, strict=True) if keep],
                    target=target[mask],
                    text_paths=[p for p, keep in zip(text_paths, mask, strict=True) if keep],
                    val_texts=h_val_texts,
                    val_target=h_val_target,
                    val_text_paths=h_val_paths,
                )
        finally:
            if not getattr(self, "_tok_cache_pinned", False):
                self._tok_cache = None

    def _prepare_val(self, X_val, y_val):
        """Load validation texts/targets/horizons/text_paths, or (None,)*4 when absent."""
        if X_val is None or y_val is None:
            return None, None, None, None
        df = _require_dataframe(X_val, name="X_val")
        if len(df) == 0:
            return None, None, None, None
        target = np.asarray(y_val, dtype=float)
        if len(df) != len(target):
            raise ValueError(f"X_val has {len(df)} rows but y_val has {len(target)} values")
        return load_texts(df), target, _horizons(df), df["text_path"].astype(str).tolist()

    def predict(self, X) -> np.ndarray:
        df = _require_dataframe(X, name="X")
        texts = load_texts(df)
        horizons = _horizons(df)
        text_paths = df["text_path"].astype(str).tolist()
        # Pin RAYON explicitly here (not only indirectly via _predict_one -> _build_modules)
        # so a predict-only / reloaded-model process tokenises multi-threaded, matching the
        # explicit pins in BertS2._predict_one and GatedFusion.predict.
        self._configure_tokenizer_runtime()
        # Tokenise each unique test filing once across horizons; scoped to this call.
        if not (
            getattr(self, "_tok_cache_pinned", False)
            and getattr(self, "_tok_cache", None) is not None
        ):
            self._tok_cache = (
                self._new_token_cache()
                if self._use_pretokenize() and _pretok_reuse_enabled()
                else None
            )
        preds = np.empty(len(df), dtype=float)
        try:
            for horizon in sorted(set(horizons.tolist())):
                if horizon not in self.models_:
                    raise ValueError(f"no {self.name} model fitted for horizon_days={horizon!r}")
                mask = horizons == horizon
                horizon_texts = [t for t, keep in zip(texts, mask, strict=True) if keep]
                horizon_paths = [p for p, keep in zip(text_paths, mask, strict=True) if keep]
                predict_start = monotonic()
                train_utils.log_predict_horizon_start(
                    self, horizon=horizon, n_rows=len(horizon_texts)
                )
                preds[mask] = self._predict_one(horizon, horizon_texts, text_paths=horizon_paths)
                train_utils.log_predict_horizon_done(
                    self,
                    horizon=horizon,
                    n_rows=len(horizon_texts),
                    secs=monotonic() - predict_start,
                )
        finally:
            if not getattr(self, "_tok_cache_pinned", False):
                self._tok_cache = None
        return preds

    def save(self, path: Path) -> None:
        if not self.models_:
            raise RuntimeError(f"{self.name} must be fitted before save")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "encoder_cfg": self.encoder_cfg,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping": self.early_stopping,
            "es_patience": self.es_patience,
            "es_min_delta": self.es_min_delta,
            "mixed_precision": self.mixed_precision,
            "warmup_ratio": self.warmup_ratio,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "log_target": self.log_target,
            "pretokenize": self.pretokenize,
            "tokenization_batch_size": self.tokenization_batch_size,
            "tokenizer_threads": self.tokenizer_threads,
            "checkpoint": self.checkpoint,
            "seed": self.seed,
            "strategy": self.strategy,
            "models": {
                horizon: {
                    "encoder_state": entry["encoder_state"],
                    "head_state": entry["head_state"],
                }
                for horizon, entry in self.models_.items()
            },
        }
        with save_path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BertS1:
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        model = cls(
            pretrained=state["encoder_cfg"].pretrained,
            max_length=state["encoder_cfg"].max_length,
            hidden_dim=int(state["hidden_dim"]),
            dropout=float(state["dropout"]),
            lr=float(state["lr"]),
            weight_decay=float(state["weight_decay"]),
            batch_size=int(state["batch_size"]),
            max_epochs=int(state["max_epochs"]),
            early_stopping=bool(state.get("early_stopping", True)),
            es_patience=int(state.get("es_patience", 1)),
            es_min_delta=float(state.get("es_min_delta", 0.0)),
            mixed_precision=str(state.get("mixed_precision", "no")),
            warmup_ratio=float(state.get("warmup_ratio", 0.0)),
            grad_accumulation_steps=int(state.get("grad_accumulation_steps", 1)),
            log_target=bool(state["log_target"]),
            pretokenize=bool(state.get("pretokenize", False)),
            tokenization_batch_size=int(state.get("tokenization_batch_size", 128)),
            tokenizer_threads=state.get("tokenizer_threads"),
            checkpoint=bool(state.get("checkpoint", True)),
            seed=state.get("seed"),
            strategy=state.get("strategy"),
        )
        model.models_ = {
            horizon: {
                "encoder_state": entry["encoder_state"],
                "head_state": entry["head_state"],
            }
            for horizon, entry in state["models"].items()
        }
        return model

    # --- internals -------------------------------------------------------

    def _build_modules(self) -> tuple[CLSEncoder, VolatilityHead]:
        self._configure_tokenizer_runtime()
        encoder = CLSEncoder(self.encoder_cfg).to(self.device)
        # When log_target=True we train on log(RV) (can be negative) so the
        # head must allow negative outputs — softplus would prevent learning.
        # When log_target=False we predict RV directly and want positivity.
        head = VolatilityHead(
            encoder.hidden_size,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            eps=_EPSILON,
            positive=not self.log_target,
        ).to(self.device)
        return encoder, head

    def _configure_tokenizer_runtime(self) -> None:
        threads = self.tokenizer_threads or _env_positive_int(_TOKENIZER_THREADS_ENV)
        if threads is None:
            return
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        os.environ["RAYON_NUM_THREADS"] = str(threads)

    def _fit_one(
        self,
        horizon: Any,
        texts: list[str],
        target: np.ndarray,
        *,
        text_paths: list[str] | None = None,
        val_texts: list[str] | None = None,
        val_target: np.ndarray | None = None,
        val_text_paths: list[str] | None = None,
    ) -> None:
        n_train = len(texts)
        horizon_start = monotonic()
        train_utils.log_fit_horizon_start(self, horizon=horizon, n_train=n_train)
        loaded_state = train_utils.maybe_load_horizon_checkpoint(
            self, horizon=horizon, n_train=n_train
        )
        if loaded_state is not None:
            self.models_[horizon] = loaded_state
            checkpoint_path = train_utils.horizon_checkpoint_path(self, horizon)
            if checkpoint_path is None:
                raise RuntimeError("loaded checkpoint without checkpoint path")
            train_utils.log_fit_horizon_skip(
                self, horizon=horizon, n_train=n_train, checkpoint_path=checkpoint_path
            )
            return

        encoder, head = self._build_modules()
        target_log = _maybe_log(target, log_target=self.log_target)
        if self._use_pretokenize():
            dataset = self._make_token_dataset(texts, text_paths, target_log, encoder=encoder)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=_tokenized_collate_fn(encoder),
                **self._loader_kwargs(),
            )
        else:
            dataset = _TextDataset(texts, target_log)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=lambda batch: _collate(batch, encoder),
            )

        if self.freeze_mode != "none":
            for prm in encoder.parameters():
                prm.requires_grad = False
            if self.freeze_mode == "top4":
                blocks = _transformer_blocks(encoder)
                if blocks is None:
                    raise ValueError(f"freeze_mode=top4 unsupported for {type(encoder).__name__}")
                for blk in list(blocks)[-4:]:
                    for prm in blk.parameters():
                        prm.requires_grad = True
            elif self.freeze_mode != "encoder":
                raise ValueError(f"unknown freeze_mode {self.freeze_mode!r}")
        enc_params = [prm for prm in encoder.parameters() if prm.requires_grad]
        param_groups = ([{"params": enc_params, "lr": self.lr}] if enc_params else []) + [
            {"params": list(head.parameters()), "lr": self.lr * self.head_lr_mult}
        ]
        optimiser = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        if self.objective == "mse":
            loss_fn = nn.MSELoss()
        else:  # pre-registered HPO objective grid; log-space QLIKE requires log targets
            from sp500vol.models.neural_text.hpo_objectives import make_objective

            if not self.log_target:
                raise ValueError("objective='qlike' requires log_target=True")
            loss_fn = make_objective(self.objective)
        scheduler = train_utils.make_scheduler(
            optimiser,
            steps_per_epoch=len(loader),
            max_epochs=self.max_epochs,
            grad_accumulation_steps=self.grad_accumulation_steps,
            warmup_ratio=self.warmup_ratio,
        )

        def forward_batch(batch):
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            targets = batch["targets"].to(self.device)
            emb = encoder(ids, mask)
            return head(emb), targets

        def snapshot_state():
            return {
                "encoder_state": {
                    k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()
                },
                "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }

        val_eval = None
        if val_texts:
            val_target_arr = np.asarray(val_target, dtype=float)
            val_target_log = _maybe_log(val_target_arr, log_target=self.log_target)

            def val_eval():
                return self._eval_val(
                    encoder,
                    head,
                    val_texts,
                    val_target_arr,
                    val_target_log,
                    text_paths=val_text_paths,
                )

        best_state, curve = train_utils.run_training_loop(
            self,
            horizon=horizon,
            loader=loader,
            optimiser=optimiser,
            loss_fn=loss_fn,
            forward_batch=forward_batch,
            snapshot_state=snapshot_state,
            modules=[encoder, head],
            val_eval=val_eval,
            scheduler=scheduler,
            grad_accumulation_steps=self.grad_accumulation_steps,
            max_epochs=self.max_epochs,
            early_stopping=self.early_stopping,
            patience=self.es_patience,
            min_delta=self.es_min_delta,
        )
        self.models_[horizon] = best_state
        self.val_curves_[horizon] = curve
        train_utils.save_horizon_checkpoint(
            self,
            horizon=horizon,
            n_train=n_train,
            state=self.models_[horizon],
        )
        train_utils.log_fit_horizon_done(
            self, horizon=horizon, n_train=n_train, secs=monotonic() - horizon_start
        )

    @torch.inference_mode()
    def _eval_val(
        self, encoder, head, texts, target_raw, target_log, *, text_paths=None
    ) -> tuple[float, float]:
        """Validation loss (training scale) + R^2 (RV scale) for early stopping."""
        raw = self._raw_outputs(encoder, head, texts, text_paths=text_paths)
        if raw.size == 0:
            return math.nan, math.nan
        val_loss = float(np.mean((raw - target_log) ** 2))
        preds_rv = _maybe_exp(raw, log_target=self.log_target)
        return val_loss, _r2(target_raw, preds_rv)

    @torch.inference_mode()
    def _raw_outputs(self, encoder, head, texts, *, text_paths=None) -> np.ndarray:
        """Head outputs (pre-exp, i.e. on the training scale) for the live modules."""
        text_list = list(texts)
        if self._use_pretokenize():
            dataset = self._make_token_dataset(
                text_list, text_paths, np.zeros(len(text_list), dtype=np.float32), encoder=encoder
            )
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=_tokenized_collate_fn(encoder),
                **self._loader_kwargs(),
            )
            batches: Iterable[dict[str, torch.Tensor]] = loader
        else:
            batches = _tokenized_prediction_batches(
                text_list, encoder=encoder, batch_size=self.batch_size
            )
        out: list[np.ndarray] = []
        for batch in batches:
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            emb = encoder(ids, mask)
            out.append(head(emb).detach().float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=float)

    @torch.inference_mode()
    def _predict_one(self, horizon: Any, texts: Iterable[str], *, text_paths=None) -> np.ndarray:
        encoder, head = self._build_modules()
        encoder.load_state_dict(self.models_[horizon]["encoder_state"])
        head.load_state_dict(self.models_[horizon]["head_state"])
        encoder.eval()
        head.eval()

        preds: list[float] = []
        text_list = list(texts)
        if self._use_pretokenize():
            dataset = self._make_token_dataset(
                text_list, text_paths, np.zeros(len(text_list), dtype=np.float32), encoder=encoder
            )
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=_tokenized_collate_fn(encoder),
                **self._loader_kwargs(),
            )
            batches: Iterable[dict[str, torch.Tensor]] = loader
        else:
            batches = _tokenized_prediction_batches(
                text_list,
                encoder=encoder,
                batch_size=self.batch_size,
            )
        for batch in batches:
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            emb = encoder(ids, mask)
            raw = head(emb).detach().float().cpu().numpy()
            preds.extend(_maybe_exp(raw, log_target=self.log_target).tolist())
        return np.asarray(preds, dtype=float)

    def _use_pretokenize(self) -> bool:
        return self.pretokenize or _env_flag(_FORCE_PRETOKENIZE_ENV)

    def _runtime_tokenization_batch_size(self) -> int:
        return _env_positive_int(_TOKENIZATION_BATCH_SIZE_ENV) or self.tokenization_batch_size

    def _new_token_cache(self):
        """Per-call token cache keyed by text_path. BertS2/S3/S4 override this to
        return a chunk-layout cache (_ChunkTokenCache)."""
        return _TokenCache()

    def _make_token_dataset(
        self, texts: list[str], text_paths: list[str] | None, targets: np.ndarray, *, encoder
    ) -> _TokenizedTextDataset:
        """Build a pretokenised dataset, reusing the per-call _TokenCache (keyed by
        text_path) when present so each unique filing is tokenised once; otherwise fall
        back to per-call _pretokenized_dataset (byte-identical output)."""
        cache = getattr(self, "_tok_cache", None)
        if cache is not None and text_paths is not None:
            ids, mask = cache.gather(
                text_paths,
                texts,
                encoder=encoder,
                tokenization_batch_size=self._runtime_tokenization_batch_size(),
            )
            return _TokenizedTextDataset(input_ids=ids, attention_mask=mask, targets=targets)
        return _pretokenized_dataset(
            texts,
            targets,
            encoder=encoder,
            tokenization_batch_size=self._runtime_tokenization_batch_size(),
        )


# --- shared helpers --------------------------------------------------------


def _require_dataframe(value, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame for neural text models")
    return value


def _horizons(df: pd.DataFrame) -> np.ndarray:
    if "horizon_days" not in df.columns:
        raise ValueError("DataFrame must include 'horizon_days'")
    return df["horizon_days"].astype(int).to_numpy()


def _select_val(
    val_texts: list[str] | None,
    val_target: np.ndarray | None,
    val_horizons: np.ndarray | None,
    val_text_paths: list[str] | None,
    horizon: Any,
) -> tuple[list[str] | None, np.ndarray | None, list[str] | None]:
    """Slice the validation set to one horizon, or (None, None, None) when absent."""
    if val_texts is None or val_horizons is None:
        return None, None, None
    vmask = val_horizons == horizon
    sel_texts = [t for t, keep in zip(val_texts, vmask, strict=True) if keep]
    if not sel_texts:
        return None, None, None
    sel_paths = (
        [p for p, keep in zip(val_text_paths, vmask, strict=True) if keep]
        if val_text_paths is not None
        else None
    )
    return sel_texts, np.asarray(val_target, dtype=float)[vmask], sel_paths


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination on the RV scale (matches evaluation metrics)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return math.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0.0:
        return math.nan
    return 1.0 - ss_res / ss_tot


def _collate(batch: list[tuple[str, float]], encoder: CLSEncoder) -> dict[str, torch.Tensor]:
    texts = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    tok = encoder.tokenize(texts)
    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "targets": targets,
    }


def _collate_tokenized(
    batch: list[dict[str, torch.Tensor | float]],
    *,
    pad_id: int,
    pad_to_multiple_of: int | None,
    padding_side: str,
) -> dict[str, torch.Tensor]:
    """Dynamically pad a batch of ragged sequences to the batch-longest length (rounded up
    to pad_to_multiple_of), on the tokenizer's padding side. Padding positions are masked
    out by attention_mask, so the CLS output is numerically identical to fixed max_length
    padding — a pure throughput win. Takes only primitives (no encoder/tokenizer) so it is
    picklable for multi-worker DataLoaders via functools.partial."""
    seqs = [item["input_ids"] for item in batch]
    masks = [item["attention_mask"] for item in batch]
    max_len = max(int(s.numel()) for s in seqs)
    if pad_to_multiple_of and max_len % pad_to_multiple_of:
        max_len += pad_to_multiple_of - (max_len % pad_to_multiple_of)
    n = len(seqs)
    input_ids = torch.full((n, max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)
    for i, (seq, mask) in enumerate(zip(seqs, masks, strict=True)):
        length = int(seq.numel())
        if padding_side == "left":
            input_ids[i, max_len - length :] = seq
            attention_mask[i, max_len - length :] = mask
        else:
            input_ids[i, :length] = seq
            attention_mask[i, :length] = mask
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": torch.tensor([item["target"] for item in batch], dtype=torch.float32),
    }


def _tokenized_collate_fn(encoder: CLSEncoder):
    """Picklable dynamic-padding collate bound to an encoder's pad settings (primitives
    only — the encoder/tokenizer is NOT captured, so multi-worker DataLoaders can pickle it)."""
    pad_id = encoder.tokenizer.pad_token_id
    return functools.partial(
        _collate_tokenized,
        pad_id=0 if pad_id is None else int(pad_id),
        pad_to_multiple_of=encoder.pad_to_multiple_of,
        padding_side=encoder.tokenizer.padding_side,
    )


def _pretok_reuse_enabled() -> bool:
    """Cross-horizon/epoch token-cache reuse; ON unless SP500VOL_PRETOK_REUSE is falsy."""
    return os.environ.get(_PRETOK_REUSE_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


class _TokenCache:
    """Tokenise each unique filing ONCE per fit()/predict() call, keyed by text_path, and
    gather fixed-width [N, L] tensors per horizon. Tokenisation is a pure function of
    (text, tokenizer, max_length) — horizon- and weight-independent — so the cached
    tensors are byte-identical to _pretokenized_dataset's. Avoids re-tokenising the same
    filing once per horizon (training/predict) and once per epoch (val early-stopping).
    Held only for the duration of one fit()/predict() call; never pickled into model.pkl."""

    def __init__(self) -> None:
        self._memo: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def gather(
        self,
        text_paths: list[str],
        texts: list[str],
        *,
        encoder: CLSEncoder,
        tokenization_batch_size: int,
        pad_to: int | None = None,
    ) -> tuple[list[torch.Tensor] | torch.Tensor, list[torch.Tensor] | torch.Tensor]:
        """Returns per-filing (input_ids, attention_mask) gathered in input order. With
        `pad_to=None` (default, BertS1 dynamic-padding path) the rows are RAGGED lists of
        variable-length tensors. With `pad_to=L` (fusion path, which stacks downstream) the
        rows are right-padded and stacked into fixed [N, L] tensors — byte-identical to the
        legacy max_length output."""
        max_length = encoder.cfg.max_length
        if not text_paths:
            if pad_to is None:
                return [], []
            empty = torch.empty((0, pad_to), dtype=torch.long)
            return empty, empty.clone()
        # Tokenise only the unique, not-yet-cached filings (dedups within this call and
        # across horizons/epochs). Same tokenizer args as _pretokenized_dataset: truncation
        # only (padding=False) — each filing stored RAGGED at its true length; the collate
        # pads per-batch. So short 8-Ks are never inflated to the full window.
        need = list(dict.fromkeys(tp for tp in text_paths if tp not in self._memo))
        if need:
            tp_to_text: dict[str, str] = {}
            for tp, tx in zip(text_paths, texts, strict=True):
                tp_to_text.setdefault(tp, tx)
            for start in range(0, len(need), tokenization_batch_size):
                batch_tps = need[start : start + tokenization_batch_size]
                tok = encoder.tokenizer(
                    [tp_to_text[tp] for tp in batch_tps],
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_attention_mask=True,
                )
                for j, tp in enumerate(batch_tps):
                    self._memo[tp] = (
                        torch.tensor(tok["input_ids"][j], dtype=torch.long),
                        torch.tensor(tok["attention_mask"][j], dtype=torch.long),
                    )
        ids = [self._memo[tp][0] for tp in text_paths]
        mask = [self._memo[tp][1] for tp in text_paths]
        if pad_to is None:
            return ids, mask
        # Fixed-width [N, pad_to] for stacking consumers (fusion). Right-pad with the
        # tokenizer's pad id; truncation already bounded every row to <= max_length.
        pad_id = encoder.tokenizer.pad_token_id
        pad_id = 0 if pad_id is None else int(pad_id)
        ids_t = torch.full((len(ids), pad_to), pad_id, dtype=torch.long)
        mask_t = torch.zeros((len(mask), pad_to), dtype=torch.long)
        for i, (seq, msk) in enumerate(zip(ids, mask, strict=True)):
            length = min(int(seq.numel()), pad_to)
            ids_t[i, :length] = seq[:length]
            mask_t[i, :length] = msk[:length]
        return ids_t, mask_t


def _pretokenized_dataset(
    texts: list[str],
    targets: np.ndarray,
    *,
    encoder: CLSEncoder,
    tokenization_batch_size: int,
) -> _TokenizedTextDataset:
    input_ids: list[torch.Tensor] = []
    attention_mask: list[torch.Tensor] = []
    for start in range(0, len(texts), tokenization_batch_size):
        chunk = texts[start : start + tokenization_batch_size]
        # Truncation only (padding=False): store each filing at its true length and let the
        # collate pad per-batch. return_tensors omitted because ragged rows can't stack.
        tok = encoder.tokenizer(
            chunk,
            max_length=encoder.cfg.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )
        for ids, mask in zip(tok["input_ids"], tok["attention_mask"], strict=True):
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_mask.append(torch.tensor(mask, dtype=torch.long))
    return _TokenizedTextDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        targets=targets,
    )


def _tokenized_prediction_batches(
    texts: list[str],
    *,
    encoder: CLSEncoder,
    batch_size: int,
) -> Iterable[dict[str, torch.Tensor]]:
    for start in range(0, len(texts), batch_size):
        tok = encoder.tokenize(texts[start : start + batch_size])
        yield {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _maybe_log(values: np.ndarray, *, log_target: bool) -> np.ndarray:
    if not log_target:
        return values
    safe = np.where(values >= 0.0, values, np.nan)
    return np.log(safe + _EPSILON)


def _maybe_exp(values: np.ndarray, *, log_target: bool) -> np.ndarray:
    if not log_target:
        return np.maximum(values, _EPSILON)
    max_log = math.log(np.finfo(float).max) - 1.0
    clipped = np.clip(values, math.log(_EPSILON), max_log)
    return np.maximum(np.exp(clipped) - _EPSILON, _EPSILON)


def _optional_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        return None
    return out


def _env_positive_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    out = int(raw)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return out


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _infer_strategy(model_name: str) -> str:
    if model_name == "C4_longformer":
        return "S5"
    if model_name.endswith("_s1"):
        return "S1"
    if model_name.endswith("_s2"):
        return "S2"
    if model_name.endswith("_s3"):
        return "S3"
    if model_name.endswith("_s4"):
        return "S4"
    return model_name
