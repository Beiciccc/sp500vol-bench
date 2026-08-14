"""D3 — gated fusion of HAR-RV price features and a FROZEN decoder-LLM embedding.

Architecture mirrors D2 (gated_fusion) exactly, swapping the fine-tuned FinBERT
CLS text branch for a *frozen* gte-Qwen2-style embedding:

    price_feats = [log rv_1d, log rv_5d, log rv_22d]   (3-dim, standardised)
    text_emb    = frozen-LLM last-token embedding of the filing  (e.g. 3584-dim)
    g           = sigmoid(W_g[price_proj ; text_proj])
    fused       = g * price_proj + (1 - g) * text_proj
    y_hat       = head(fused)

Only the gate/projection/head train; the 7B encoder is frozen and its embeddings
are precomputed once and cached (see :mod:`sp500vol.models.neural_text.qwen_llm`).
This makes D3 the apples-to-apples instruction-tuned-LLM analogue of D2 and tests
whether a modern decoder representation changes the gating story.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sp500vol.models.fusion.gated_fusion import _GatedFusion, _price_matrix
from sp500vol.models.neural_text import _train_utils as train_utils
from sp500vol.models.neural_text.bert_s1 import (
    _EPSILON,
    _horizons,
    _maybe_exp,
    _maybe_log,
    _r2,
    _require_dataframe,
)
from sp500vol.models.neural_text.heads import VolatilityHead
from sp500vol.models.neural_text.qwen_llm import _FrozenLLMForecaster


def _select_val_fusion_llm(
    val: dict[str, Any] | None, horizon: Any
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if val is None:
        return None, None, None
    vmask = val["horizons"] == horizon
    if not bool(vmask.any()):
        return None, None, None
    return val["emb"][vmask], val["price"][vmask], val["target"][vmask]


class D3LLMFusion(_FrozenLLMForecaster):
    """D3 — HAR-RV price features gate-fused with a frozen decoder-LLM embedding."""

    name = "D3_llm_fusion"

    def __init__(
        self,
        *,
        pretrained: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
        proj_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, **kwargs)
        self.proj_dim = int(proj_dim)
        self.price_mean_: dict[Any, np.ndarray] = {}
        self.price_std_: dict[Any, np.ndarray] = {}

    # --- forecaster API --------------------------------------------------

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        df = _require_dataframe(X_train, name="X_train")
        target = np.asarray(y_train, dtype=float)
        if len(df) != len(target):
            raise ValueError(f"X_train has {len(df)} rows but y_train has {len(target)} values")
        emb = self._encode(df)
        price = _price_matrix(df)
        horizons = _horizons(df)
        val = self._prepare_val(X_val, y_val)
        self.models_ = {}
        self.val_curves_ = {}
        for horizon in sorted(set(horizons.tolist())):
            mask = horizons == horizon
            v_emb, v_price, v_target = _select_val_fusion_llm(val, horizon)
            self._fit_one(horizon, emb[mask], price[mask], target[mask], v_emb, v_price, v_target)

    def _prepare_val(self, X_val, y_val) -> dict[str, Any] | None:
        if X_val is None or y_val is None:
            return None
        df = _require_dataframe(X_val, name="X_val")
        if len(df) == 0:
            return None
        return {
            "emb": self._encode(df),
            "price": _price_matrix(df),
            "target": np.asarray(y_val, dtype=float),
            "horizons": _horizons(df),
        }

    def predict(self, X) -> np.ndarray:
        df = _require_dataframe(X, name="X")
        emb = self._encode(df)
        price = _price_matrix(df)
        horizons = _horizons(df)
        preds = np.empty(len(df), dtype=float)
        for horizon in sorted(set(horizons.tolist())):
            if horizon not in self.models_:
                raise ValueError(f"no {self.name} model fitted for horizon_days={horizon!r}")
            mask = horizons == horizon
            predict_start = monotonic()
            train_utils.log_predict_horizon_start(self, horizon=horizon, n_rows=int(mask.sum()))
            preds[mask] = self._predict_one(horizon, emb[mask], price[mask])
            train_utils.log_predict_horizon_done(
                self, horizon=horizon, n_rows=int(mask.sum()), secs=monotonic() - predict_start
            )
        return preds

    # --- internals -------------------------------------------------------

    def _build_modules(self) -> tuple[_GatedFusion, VolatilityHead]:
        if self.embedding_dim_ is None:
            raise RuntimeError("embedding_dim_ unknown; encode texts before building the fusion")
        fusion = _GatedFusion(
            text_dim=self.embedding_dim_, price_dim=3, proj_dim=self.proj_dim
        ).to(self.device)
        head = VolatilityHead(
            self.proj_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            eps=_EPSILON,
            positive=not self.log_target,
        ).to(self.device)
        return fusion, head

    def _standardise_fit(self, horizon: Any, price: np.ndarray) -> np.ndarray:
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
        emb: np.ndarray,
        price: np.ndarray,
        target: np.ndarray,
        val_emb: np.ndarray | None = None,
        val_price: np.ndarray | None = None,
        val_target: np.ndarray | None = None,
    ) -> None:
        n_train = len(emb)
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

        fusion, head = self._build_modules()
        price_std = self._standardise_fit(horizon, price)
        target_log = _maybe_log(target, log_target=self.log_target)
        dataset = TensorDataset(
            torch.from_numpy(emb.astype(np.float32)),
            torch.from_numpy(price_std.astype(np.float32)),
            torch.from_numpy(target_log.astype(np.float32)),
        )
        # pin_memory speeds the H2D copy of the (large) cached-embedding batches; workers
        # stay at 0 since __getitem__ is a trivial resident-tensor index (IPC would dominate).
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )
        params = list(fusion.parameters()) + list(head.parameters())
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
            emb_b, price_b, target_b = batch
            fused = fusion(price_b.to(self.device), emb_b.to(self.device))
            return head(fused), target_b.to(self.device)

        def snapshot_state():
            return {
                "fusion_state": {
                    k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()
                },
                "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }

        val_eval = None
        if val_emb is not None and len(val_emb):
            val_price_std = self._standardise_apply(horizon, np.asarray(val_price, dtype=float))
            val_target_arr = np.asarray(val_target, dtype=float)
            val_target_log = _maybe_log(val_target_arr, log_target=self.log_target)

            def val_eval():
                raw = self._fusion_forward(fusion, head, val_emb, val_price_std)
                if raw.size == 0:
                    return float("nan"), float("nan")
                val_loss = float(np.mean((raw - val_target_log) ** 2))
                preds_rv = _maybe_exp(raw, log_target=self.log_target)
                return val_loss, _r2(val_target_arr, preds_rv)

        best_state, curve = train_utils.run_training_loop(
            self,
            horizon=horizon,
            loader=loader,
            optimiser=optimiser,
            loss_fn=loss_fn,
            forward_batch=forward_batch,
            snapshot_state=snapshot_state,
            modules=[fusion, head],
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
    def _fusion_forward(self, fusion, head, emb: np.ndarray, price_std: np.ndarray) -> np.ndarray:
        emb_t = torch.from_numpy(np.asarray(emb, dtype=np.float32))
        price_t = torch.from_numpy(np.asarray(price_std, dtype=np.float32))
        out: list[np.ndarray] = []
        for start in range(0, len(emb_t), self.batch_size):
            e = emb_t[start : start + self.batch_size].to(self.device)
            p = price_t[start : start + self.batch_size].to(self.device)
            fused = fusion(p, e)
            out.append(head(fused).detach().float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=float)

    @torch.inference_mode()
    def _predict_one(self, horizon: Any, emb: np.ndarray, price: np.ndarray) -> np.ndarray:
        fusion, head = self._build_modules()
        fusion.load_state_dict(self.models_[horizon]["fusion_state"])
        head.load_state_dict(self.models_[horizon]["head_state"])
        fusion.eval()
        head.eval()
        price_std = self._standardise_apply(horizon, price)
        raw = self._fusion_forward(fusion, head, emb, price_std)
        return _maybe_exp(raw, log_target=self.log_target)

    def save(self, path: Path) -> None:
        if not self.models_:
            raise RuntimeError(f"{self.name} must be fitted before save")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "encoder_cfg": self.encoder_cfg,
            "embedding_dim": self.embedding_dim_,
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
            "instruction": self.instruction,
            "normalize_emb": self.normalize_emb,
            "encode_batch_size": self.encode_batch_size,
            "cache_embeddings": self.cache_embeddings,
            "checkpoint": self.checkpoint,
            "seed": self.seed,
            "strategy": self.strategy,
            "price_mean": self.price_mean_,
            "price_std": self.price_std_,
            "models": self.models_,
        }
        with save_path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> D3LLMFusion:
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
            es_patience=int(state.get("es_patience", 3)),
            es_min_delta=float(state.get("es_min_delta", 0.0)),
            mixed_precision=str(state.get("mixed_precision", "no")),
            warmup_ratio=float(state.get("warmup_ratio", 0.0)),
            log_target=bool(state["log_target"]),
            instruction=state.get("instruction"),
            normalize_emb=bool(state.get("normalize_emb", True)),
            encode_batch_size=int(state.get("encode_batch_size", 8)),
            cache_embeddings=bool(state.get("cache_embeddings", True)),
            seed=state.get("seed"),
            strategy=state.get("strategy"),
        )
        model.embedding_dim_ = None if state.get("embedding_dim") is None else int(
            state["embedding_dim"]
        )
        model.price_mean_ = state["price_mean"]
        model.price_std_ = state["price_std"]
        model.models_ = state["models"]
        return model
