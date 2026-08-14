"""D2 Gated fusion: HAR-RV price features ⊕ FinBERT text embedding.

Architecture (per horizon):

    price_feats  = [log rv_1d, log rv_5d, log rv_22d]      (3-dim, standardised)
    text_emb     = FinBERT CLS embedding of truncated filing (S1)  (768-dim)
    g            = sigmoid(W_g · [price_proj ; text_proj])          (gate in [0,1])
    fused        = g * price_proj + (1 - g) * text_proj
    y_hat        = head(fused)

The gate learns, per sample, how much to trust price vs text. This directly
tests H5 (joint > either modality alone). We freeze nothing — the FinBERT
encoder is fine-tuned jointly, same as the C-block.

Training reuses the BertS1 tokenisation pipeline for the text branch; the price
branch reads the three HAR-RV features already present in the aligned parquet.
"""

from __future__ import annotations

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
from sp500vol.models.neural_text.bert_s1 import (
    _EPSILON,
    _TOKENIZER_THREADS_ENV,
    _env_positive_int,
    _maybe_exp,
    _maybe_log,
    _pretok_reuse_enabled,
    _r2,
    _resolve_device,
    _TokenCache,
)
from sp500vol.models.neural_text.encoders import CLSEncoder, EncoderConfig
from sp500vol.models.neural_text.heads import VolatilityHead

_PRICE_FEATURES = ("feature_rv_1d", "feature_rv_5d", "feature_rv_22d")


class _GatedFusion(nn.Module):
    """Project price + text into a shared space and gate-combine them."""

    def __init__(self, text_dim: int, price_dim: int = 3, proj_dim: int = 128) -> None:
        super().__init__()
        self.price_proj = nn.Sequential(nn.Linear(price_dim, proj_dim), nn.GELU())
        self.text_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.GELU())
        self.gate = nn.Linear(proj_dim * 2, proj_dim)

    def forward(self, price: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        p = self.price_proj(price)
        t = self.text_proj(text)
        g = torch.sigmoid(self.gate(torch.cat([p, t], dim=-1)))
        return g * p + (1.0 - g) * t


class _TokenizedFusionDataset(Dataset):
    """Pre-tokenised text ([N, L] fixed-width) + standardised price + target. Tokenisation
    is done once up-front (via the cross-horizon/epoch token cache), so the train loader's
    collate does NO tokenisation and is picklable for multi-worker loading."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        price: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        if not (input_ids.shape[0] == attention_mask.shape[0] == len(price) == len(targets)):
            raise ValueError("input_ids, attention_mask, price, targets length mismatch")
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.price = torch.as_tensor(np.asarray(price), dtype=torch.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "price": self.price[idx],
            "target": float(self.targets[idx]),
        }


class GatedFusion(VolatilityForecaster):
    """D2 — gated fusion of HAR-RV price features and FinBERT text (S1 truncation)."""

    name = "D2_gated_fusion"

    def __init__(
        self,
        *,
        pretrained: str = "ProsusAI/finbert",
        max_length: int = 512,
        proj_dim: int = 128,
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
        log_target: bool = True,
        device: str = "auto",
        checkpoint: bool = True,
        checkpoint_dir: str | Path | None = None,
        seed: int | None = None,
        strategy: str | None = None,
        pretokenize: bool = False,
        tokenization_batch_size: int = 512,
        tokenizer_threads: int | None = None,
        dataloader_num_workers: int = 0,
        dataloader_persistent_workers: bool = False,
        dataloader_pin_memory: bool | None = None,
        dataloader_prefetch_factor: int | None = None,
    ) -> None:
        self.encoder_cfg = EncoderConfig(pretrained=pretrained, max_length=max_length)
        self.proj_dim = int(proj_dim)
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
        self.log_target = bool(log_target)
        self.device = _resolve_device(device)
        self.checkpoint = bool(checkpoint)
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.seed = None if seed is None else int(seed)
        self.strategy = strategy or self.name
        self.pretokenize = bool(pretokenize)
        self.tokenization_batch_size = max(1, int(tokenization_batch_size))
        self.tokenizer_threads = None if tokenizer_threads is None else int(tokenizer_threads)
        # DataLoader worker knobs (default 0 => legacy single-process loading).
        self.dl_num_workers = max(0, int(dataloader_num_workers))
        self.dl_persistent_workers = (
            bool(dataloader_persistent_workers) and self.dl_num_workers > 0
        )
        self._dl_pin_cfg = dataloader_pin_memory
        self.dl_prefetch_factor = (
            int(dataloader_prefetch_factor)
            if dataloader_prefetch_factor is not None and self.dl_num_workers > 0
            else None
        )
        self.price_mean_: dict[Any, np.ndarray] = {}
        self.price_std_: dict[Any, np.ndarray] = {}
        self.models_: dict[Any, dict[str, Any]] = {}
        self.val_curves_: dict[Any, list[dict[str, Any]]] = {}

    def _dl_pin_memory(self) -> bool:
        if self._dl_pin_cfg is None:
            return self.device.type == "cuda"
        return bool(self._dl_pin_cfg)

    def _loader_kwargs(self) -> dict[str, Any]:
        """DataLoader knobs; persistent_workers/prefetch_factor dropped when num_workers==0."""
        kwargs: dict[str, Any] = {
            "num_workers": self.dl_num_workers,
            "pin_memory": self._dl_pin_memory(),
        }
        if self.dl_num_workers > 0:
            kwargs["persistent_workers"] = self.dl_persistent_workers
            if self.dl_prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.dl_prefetch_factor
        return kwargs

    def _new_token_cache(self) -> _TokenCache:
        return _TokenCache()

    def _configure_tokenizer_runtime(self) -> None:
        """Pin RAYON threads + enable fast-tokeniser parallelism for the up-front
        pre-tokenisation (mirrors BertS1) so it is not single-threaded."""
        threads = self.tokenizer_threads or _env_positive_int(_TOKENIZER_THREADS_ENV)
        if threads is None:
            return
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        os.environ["RAYON_NUM_THREADS"] = str(threads)

    def _fusion_tokens(
        self, text_paths: list[str] | None, texts: list[str], encoder: CLSEncoder
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed-width [N, L] (input_ids, attention_mask) for the text branch, reusing the
        per-call _TokenCache when present so each unique filing is tokenised once across
        horizons/epochs; else tokenise here (max_length, byte-identical to the cache). CLS
        pooling is padding-invariant under the attention mask, so max_length padding gives
        the same text embedding as the old dynamic-padding encode."""
        cache = getattr(self, "_tok_cache", None)
        if cache is not None and text_paths is not None:
            # pad_to=max_length keeps the fixed-width [N, L] contract the fusion dataset and
            # its pure-stack collate expect (the BertS1 path instead leaves rows ragged for
            # dynamic per-batch padding). Byte-identical to the legacy output.
            return cache.gather(
                text_paths,
                texts,
                encoder=encoder,
                tokenization_batch_size=self.tokenization_batch_size,
                pad_to=encoder.cfg.max_length,
            )
        tok = encoder.tokenizer(
            texts,
            max_length=encoder.cfg.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        return tok["input_ids"], tok["attention_mask"]

    # --- forecaster API --------------------------------------------------

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        df = _require_df(X_train)
        target = np.asarray(y_train, dtype=float)
        texts = load_texts(df)
        horizons = df["horizon_days"].astype(int).to_numpy()
        text_paths = df["text_path"].astype(str).tolist()
        val = self._prepare_val(X_val, y_val)
        self._configure_tokenizer_runtime()
        # Tokenise each unique filing once across horizons AND val epochs; scoped to fit().
        self._tok_cache = self._new_token_cache() if _pretok_reuse_enabled() else None
        self.models_ = {}
        self.val_curves_ = {}
        try:
            for horizon in sorted(set(horizons.tolist())):
                mask = horizons == horizon
                val_texts, val_price, val_target, val_paths = _select_val_fusion(val, horizon)
                self._fit_one(
                    horizon,
                    texts=[t for t, keep in zip(texts, mask, strict=True) if keep],
                    price=_price_matrix(df[mask]),
                    target=target[mask],
                    text_paths=[p for p, keep in zip(text_paths, mask, strict=True) if keep],
                    val_texts=val_texts,
                    val_price=val_price,
                    val_target=val_target,
                    val_text_paths=val_paths,
                )
        finally:
            self._tok_cache = None

    def _prepare_val(self, X_val, y_val) -> dict[str, Any] | None:
        if X_val is None or y_val is None:
            return None
        df = _require_df(X_val)
        if len(df) == 0:
            return None
        return {
            "texts": load_texts(df),
            "price": _price_matrix(df),
            "target": np.asarray(y_val, dtype=float),
            "horizons": df["horizon_days"].astype(int).to_numpy(),
            "text_paths": df["text_path"].astype(str).tolist(),
        }

    def predict(self, X) -> np.ndarray:
        df = _require_df(X)
        texts = load_texts(df)
        horizons = df["horizon_days"].astype(int).to_numpy()
        text_paths = df["text_path"].astype(str).tolist()
        self._configure_tokenizer_runtime()
        self._tok_cache = self._new_token_cache() if _pretok_reuse_enabled() else None
        preds = np.empty(len(df), dtype=float)
        try:
            for horizon in sorted(set(horizons.tolist())):
                if horizon not in self.models_:
                    raise ValueError(f"no {self.name} model for horizon_days={horizon!r}")
                mask = horizons == horizon
                horizon_texts = [t for t, keep in zip(texts, mask, strict=True) if keep]
                horizon_price = _price_matrix(df[mask])
                horizon_paths = [p for p, keep in zip(text_paths, mask, strict=True) if keep]
                predict_start = monotonic()
                train_utils.log_predict_horizon_start(
                    self, horizon=horizon, n_rows=len(horizon_texts)
                )
                preds[mask] = self._predict_one(
                    horizon, texts=horizon_texts, price=horizon_price, text_paths=horizon_paths
                )
                train_utils.log_predict_horizon_done(
                    self,
                    horizon=horizon,
                    n_rows=len(horizon_texts),
                    secs=monotonic() - predict_start,
                )
        finally:
            self._tok_cache = None
        return preds

    def save(self, path: Path) -> None:
        if not self.models_:
            raise RuntimeError(f"{self.name} must be fitted before save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "encoder_cfg": self.encoder_cfg,
            "proj_dim": self.proj_dim,
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
            "log_target": self.log_target,
            "checkpoint": self.checkpoint,
            "seed": self.seed,
            "strategy": self.strategy,
            "price_mean": self.price_mean_,
            "price_std": self.price_std_,
            "models": self.models_,
        }
        with Path(path).open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> GatedFusion:
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        model = cls(
            pretrained=state["encoder_cfg"].pretrained,
            max_length=state["encoder_cfg"].max_length,
            proj_dim=int(state["proj_dim"]),
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
            log_target=bool(state["log_target"]),
            checkpoint=bool(state.get("checkpoint", True)),
            seed=state.get("seed"),
            strategy=state.get("strategy"),
        )
        model.price_mean_ = state["price_mean"]
        model.price_std_ = state["price_std"]
        model.models_ = state["models"]
        return model

    # --- internals -------------------------------------------------------

    def _build_modules(self) -> tuple[CLSEncoder, _GatedFusion, VolatilityHead]:
        encoder = CLSEncoder(self.encoder_cfg).to(self.device)
        fusion = _GatedFusion(encoder.hidden_size, price_dim=3, proj_dim=self.proj_dim).to(
            self.device
        )
        head = VolatilityHead(
            self.proj_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            eps=_EPSILON,
            positive=not self.log_target,
        ).to(self.device)
        return encoder, fusion, head

    def _standardise_fit(self, horizon: Any, price: np.ndarray) -> np.ndarray:
        # log-transform the (positive) RV features, then z-score
        safe = np.log(np.clip(price, _EPSILON, None))
        mean = safe.mean(axis=0)
        std = safe.std(axis=0) + 1e-8
        self.price_mean_[horizon] = mean
        self.price_std_[horizon] = std
        return (safe - mean) / std

    def _standardise_apply(self, horizon: Any, price: np.ndarray) -> np.ndarray:
        safe = np.log(np.clip(price, _EPSILON, None))
        return (safe - self.price_mean_[horizon]) / self.price_std_[horizon]

    def _fit_one(
        self,
        horizon: Any,
        texts: list[str],
        price: np.ndarray,
        target: np.ndarray,
        *,
        text_paths: list[str] | None = None,
        val_texts: list[str] | None = None,
        val_price: np.ndarray | None = None,
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
            try:
                self.models_[horizon] = loaded_state["model_state"]
                self.price_mean_[horizon] = np.asarray(loaded_state["price_mean"], dtype=float)
                self.price_std_[horizon] = np.asarray(loaded_state["price_std"], dtype=float)
            except KeyError:
                loaded_state = None
            else:
                checkpoint_path = train_utils.horizon_checkpoint_path(self, horizon)
                if checkpoint_path is None:
                    raise RuntimeError("loaded checkpoint without checkpoint path")
                train_utils.log_fit_horizon_skip(
                    self, horizon=horizon, n_train=n_train, checkpoint_path=checkpoint_path
                )
                return

        encoder, fusion, head = self._build_modules()
        price_std = self._standardise_fit(horizon, price)
        input_ids, attention_mask = self._fusion_tokens(text_paths, texts, encoder)
        dataset = _TokenizedFusionDataset(
            input_ids, attention_mask, price_std, _maybe_log(target, log_target=self.log_target)
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate_fusion_tokenized,
            **self._loader_kwargs(),
        )
        params = list(encoder.parameters()) + list(fusion.parameters()) + list(head.parameters())
        optimiser = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if getattr(self, "objective", "mse") == "mse":
            loss_fn = nn.MSELoss()
        else:
            from sp500vol.models.neural_text.hpo_objectives import make_objective
            loss_fn = make_objective(self.objective)
        scheduler = train_utils.make_scheduler(
            optimiser,
            steps_per_epoch=len(loader),
            max_epochs=self.max_epochs,
            grad_accumulation_steps=1,
            warmup_ratio=self.warmup_ratio,
        )

        def forward_batch(batch):
            text_emb = encoder(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
            fused = fusion(batch["price"].to(self.device), text_emb)
            return head(fused), batch["targets"].to(self.device)

        def snapshot_state():
            return {
                "encoder_state": {
                    k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()
                },
                "fusion_state": {
                    k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()
                },
                "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }

        val_eval = None
        if val_texts:
            val_price_std = self._standardise_apply(horizon, np.asarray(val_price, dtype=float))
            val_target_arr = np.asarray(val_target, dtype=float)
            val_target_log = _maybe_log(val_target_arr, log_target=self.log_target)

            def val_eval():
                return self._eval_val_fusion(
                    encoder,
                    fusion,
                    head,
                    val_texts,
                    val_price_std,
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
            modules=[encoder, fusion, head],
            val_eval=val_eval,
            scheduler=scheduler,
            grad_accumulation_steps=1,
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
            state={
                "model_state": self.models_[horizon],
                "price_mean": self.price_mean_[horizon],
                "price_std": self.price_std_[horizon],
            },
        )
        train_utils.log_fit_horizon_done(
            self, horizon=horizon, n_train=n_train, secs=monotonic() - horizon_start
        )

    @torch.inference_mode()
    def _eval_val_fusion(
        self, encoder, fusion, head, texts, price_std, target_raw, target_log, *, text_paths=None
    ) -> tuple[float, float]:
        raw = self._raw_outputs_fusion(
            encoder, fusion, head, texts, price_std, text_paths=text_paths
        )
        if raw.size == 0:
            return float("nan"), float("nan")
        val_loss = float(np.mean((raw - target_log) ** 2))
        preds_rv = _maybe_exp(raw, log_target=self.log_target)
        return val_loss, _r2(target_raw, preds_rv)

    @torch.inference_mode()
    def _raw_outputs_fusion(
        self, encoder, fusion, head, texts, price_std, *, text_paths=None
    ) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty(0, dtype=float)
        ids, mask = self._fusion_tokens(text_paths, text_list, encoder)
        out: list[np.ndarray] = []
        for start in range(0, len(text_list), self.batch_size):
            sl = slice(start, start + self.batch_size)
            chunk_price = torch.tensor(price_std[sl], dtype=torch.float32).to(self.device)
            text_emb = encoder(ids[sl].to(self.device), mask[sl].to(self.device))
            fused = fusion(chunk_price, text_emb)
            out.append(head(fused).detach().float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=float)

    @torch.inference_mode()
    def _predict_one(
        self, horizon: Any, texts: Iterable[str], price: np.ndarray, *, text_paths=None
    ) -> np.ndarray:
        encoder, fusion, head = self._build_modules()
        encoder.load_state_dict(self.models_[horizon]["encoder_state"])
        fusion.load_state_dict(self.models_[horizon]["fusion_state"])
        head.load_state_dict(self.models_[horizon]["head_state"])
        encoder.eval()
        fusion.eval()
        head.eval()

        price_std = self._standardise_apply(horizon, price)
        text_list = list(texts)
        if not text_list:
            return np.empty(0, dtype=float)
        ids, mask = self._fusion_tokens(text_paths, text_list, encoder)
        preds: list[float] = []
        for start in range(0, len(text_list), self.batch_size):
            sl = slice(start, start + self.batch_size)
            chunk_price = torch.tensor(price_std[sl], dtype=torch.float32).to(self.device)
            text_emb = encoder(ids[sl].to(self.device), mask[sl].to(self.device))
            fused = fusion(chunk_price, text_emb)
            raw = head(fused).detach().float().cpu().numpy()
            preds.extend(_maybe_exp(raw, log_target=self.log_target).tolist())
        return np.asarray(preds, dtype=float)


def _require_df(value) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("GatedFusion requires a pandas DataFrame")
    for col in (*_PRICE_FEATURES[1:], "horizon_days", "text_path"):
        if col not in value.columns and col != "feature_rv_1d":
            raise ValueError(f"GatedFusion requires column {col!r}")
    return value


def _select_val_fusion(
    val: dict[str, Any] | None, horizon: Any
) -> tuple[list[str] | None, np.ndarray | None, np.ndarray | None, list[str] | None]:
    """Slice the prepared validation dict to one horizon, or (None,)*4 when absent."""
    if val is None:
        return None, None, None, None
    vmask = val["horizons"] == horizon
    sel_texts = [t for t, keep in zip(val["texts"], vmask, strict=True) if keep]
    if not sel_texts:
        return None, None, None, None
    sel_paths = [p for p, keep in zip(val["text_paths"], vmask, strict=True) if keep]
    return sel_texts, val["price"][vmask], val["target"][vmask], sel_paths


def _price_matrix(df: pd.DataFrame) -> np.ndarray:
    if "feature_rv_1d" in df.columns:
        rv_1d = df["feature_rv_1d"].to_numpy(dtype=float)
    else:
        rv_1d = np.sqrt(252.0) * df["feature_return_1d"].abs().to_numpy(dtype=float)
    return np.column_stack(
        [
            rv_1d,
            df["feature_rv_5d"].to_numpy(dtype=float),
            df["feature_rv_22d"].to_numpy(dtype=float),
        ]
    )


def _collate_fusion_tokenized(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Picklable collate over PRE-TOKENISED rows (pure stack, no encoder/tokeniser) so the
    fusion train loader can use num_workers>0."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "price": torch.stack([b["price"] for b in batch]),
        "targets": torch.tensor([b["target"] for b in batch], dtype=torch.float32),
    }
