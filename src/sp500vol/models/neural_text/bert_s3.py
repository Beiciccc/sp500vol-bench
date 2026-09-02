"""C-block + S3 chunk-attention with learnable attention pooling.

Same chunking pipeline as S2 (sliding-window, max_chunks), but instead of a
flat mean over chunk CLS embeddings we learn an attention weight per chunk:

    score_i   = v · tanh(W · cls_i)          (additive attention)
    alpha_i   = softmax(score_i) over real chunks (padded chunks masked to -inf)
    pooled    = sum_i alpha_i · cls_i

This lets the model up-weight the informative chunks (e.g. the Risk Factors
section deep in a 10-K) instead of diluting them in a flat mean. It is the key
contrast against S2 mean-pool in the AB1 long-doc strategy ablation.
"""

from __future__ import annotations

import functools
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sp500vol.models.neural_text.bert_s1 import _maybe_exp
from sp500vol.models.neural_text.bert_s2 import (
    BertS2,
    _ChunkedTextDataset,
    _collate_chunks,
    encode_real_chunks,
)
from sp500vol.models.neural_text.encoders import CLSEncoder


class _ChunkAttentionPool(nn.Module):
    """Additive (Bahdanau-style) attention over chunk embeddings."""

    def __init__(self, hidden_size: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, emb: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        """emb: (B, K, H); chunk_mask: (B, K) with 1=real, 0=pad → (B, H)."""
        scores = self.score(torch.tanh(self.proj(emb))).squeeze(-1)  # (B, K)
        scores = scores.masked_fill(chunk_mask == 0, float("-inf"))
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, K, 1)
        # Guard: a row with zero real chunks would give all -inf → nan; clamp.
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return (emb * alpha).sum(dim=1)  # (B, H)


class BertS3(BertS2):
    """C-family + S3 chunk-attention strategy.

    Inherits the chunk tokenisation / dataloader machinery from BertS2 and only
    overrides the pooling step (mean → learnable attention).
    """

    name = "C1_bert_s3"

    def __init__(self, *, attn_dim: int = 128, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attn_dim = int(attn_dim)
        # One attention pool per horizon, created lazily in _build_modules.
        self._attn_pools: dict[Any, _ChunkAttentionPool] = {}

    def _build_modules(self):
        encoder, head = super()._build_modules()
        pool = _ChunkAttentionPool(encoder.hidden_size, attn_dim=self.attn_dim).to(self.device)
        # Stash on the instance so _forward_chunked + optimiser can reach it.
        self._current_pool = pool
        return encoder, head

    def _forward_chunked(self, batch: dict[str, torch.Tensor], encoder: CLSEncoder) -> torch.Tensor:
        ids = batch["input_ids"].to(self.device)  # (B, K, L)
        mask = batch["attention_mask"].to(self.device)  # (B, K, L)
        chunk_counts = batch["chunk_counts"].to(self.device)  # (B,)
        _b, k, _length = ids.shape

        emb = encode_real_chunks(encoder, ids, mask, chunk_counts)  # (B, K, H), pad slots = 0

        chunk_mask = (
            torch.arange(k, device=self.device).unsqueeze(0) < chunk_counts.unsqueeze(1)
        ).long()  # (B, K)
        return self._current_pool(emb, chunk_mask)  # (B, H)

    def _extra_modules(self) -> list[nn.Module]:
        # The attention pool must be trained, validated and checkpointed alongside
        # the encoder/head; the shared BertS2._fit_one picks it up via this hook.
        return [self._current_pool]

    def _snapshot_chunked(self, encoder, head) -> dict[str, Any]:
        return {
            "encoder_state": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
            "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            "pool_state": {
                k: v.detach().cpu().clone() for k, v in self._current_pool.state_dict().items()
            },
        }

    @torch.inference_mode()
    def _predict_one(self, horizon: Any, texts, *, text_paths=None) -> Any:
        encoder, head = self._build_modules()
        pool = self._current_pool
        encoder.load_state_dict(self.models_[horizon]["encoder_state"])
        head.load_state_dict(self.models_[horizon]["head_state"])
        pool.load_state_dict(self.models_[horizon]["pool_state"])
        encoder.eval()
        head.eval()
        pool.eval()

        text_list = list(texts)
        cache = getattr(self, "_tok_cache", None)
        prebuilt = (
            cache.gather(
                text_paths,
                text_list,
                encoder=encoder,
                chunk_stride=self.chunk_stride,
                max_chunks=self.max_chunks,
                tokenization_batch_size=self._runtime_tokenization_batch_size(),
            )
            if cache is not None and text_paths is not None
            else None
        )
        dataset = _ChunkedTextDataset(
            text_list,
            np.zeros(len(text_list), dtype=np.float32),
            encoder,
            chunk_stride=self.chunk_stride,
            max_chunks=self.max_chunks,
            tokenization_batch_size=self._runtime_tokenization_batch_size(),
            prebuilt_items=prebuilt,
        )
        pad_id = encoder.tokenizer.pad_token_id or 0
        max_length = int(encoder.cfg.max_length)  # bind int, never the encoder (worker-pickle safe)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=functools.partial(
                _collate_chunks, pad_id=pad_id, max_length=max_length, max_chunks=self.max_chunks
            ),
            **self._loader_kwargs(),
        )
        preds: list[float] = []
        for batch in loader:
            emb = self._forward_chunked(batch, encoder)
            raw = head(emb).detach().float().cpu().numpy()
            preds.extend(_maybe_exp(raw, log_target=self.log_target).tolist())
        return np.asarray(preds, dtype=float)

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
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "log_target": self.log_target,
            "chunk_stride": self.chunk_stride,
            "max_chunks": self.max_chunks,
            "attn_dim": self.attn_dim,
            "models": self.models_,
        }
        with save_path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BertS3:
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
            grad_accumulation_steps=int(state.get("grad_accumulation_steps", 1)),
            log_target=bool(state["log_target"]),
            chunk_stride=int(state.get("chunk_stride", 256)),
            max_chunks=int(state.get("max_chunks", 16)),
            attn_dim=int(state.get("attn_dim", 128)),
        )
        model.models_ = state["models"]
        return model


class FinBertS3(BertS3):
    """C2 FinBERT + S3 chunk-attention."""

    name = "C2_finbert_s3"

    def __init__(self, *, pretrained: str = "ProsusAI/finbert", **kwargs) -> None:
        super().__init__(pretrained=pretrained, **kwargs)
