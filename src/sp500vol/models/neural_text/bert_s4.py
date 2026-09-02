"""C-block + S4 hierarchical chunk encoder strategy.

S4 keeps the S2 sliding-window chunk pipeline, then models interactions between
chunk CLS embeddings with a shallow TransformerEncoder before mask-aware document
pooling. This is a true two-level hierarchy: token-level transformer per chunk,
then chunk-level self-attention across the filing.
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


class BertS4(BertS2):
    """C-family + S4 hierarchical strategy."""

    name = "C1_bert_s4"

    def __init__(
        self,
        *,
        chunk_num_heads: int = 8,
        chunk_encoder_layers: int = 1,
        chunk_ff_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.chunk_num_heads = int(chunk_num_heads)
        self.chunk_encoder_layers = int(chunk_encoder_layers)
        self.chunk_ff_dim = None if chunk_ff_dim is None else int(chunk_ff_dim)

    def _build_modules(self):
        encoder, head = super()._build_modules()
        self._current_chunk_encoder = _build_chunk_encoder(
            hidden_size=encoder.hidden_size,
            num_heads=self.chunk_num_heads,
            num_layers=self.chunk_encoder_layers,
            ff_dim=self.chunk_ff_dim,
            dropout=self.dropout,
        ).to(self.device)
        return encoder, head

    def _forward_chunked(self, batch: dict[str, torch.Tensor], encoder: CLSEncoder) -> torch.Tensor:
        ids = batch["input_ids"].to(self.device)  # (B, K, L)
        mask = batch["attention_mask"].to(self.device)  # (B, K, L)
        chunk_counts = batch["chunk_counts"].to(self.device)  # (B,)
        _b, k, _length = ids.shape

        emb = encode_real_chunks(encoder, ids, mask, chunk_counts)  # (B, K, H), pad slots = 0

        valid_chunks = torch.arange(k, device=self.device).unsqueeze(0) < chunk_counts.unsqueeze(1)
        encoded = self._current_chunk_encoder(emb, src_key_padding_mask=~valid_chunks)
        weights = valid_chunks.float().unsqueeze(-1)
        denom = chunk_counts.clamp(min=1).float().unsqueeze(-1)
        return (encoded * weights).sum(dim=1) / denom

    def _extra_modules(self) -> list[nn.Module]:
        # The chunk-level transformer is trained/validated/checkpointed with the
        # encoder/head via the shared BertS2._fit_one hook.
        return [self._current_chunk_encoder]

    def _snapshot_chunked(self, encoder, head) -> dict[str, Any]:
        return {
            "encoder_state": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
            "chunk_encoder_state": {
                k: v.detach().cpu().clone()
                for k, v in self._current_chunk_encoder.state_dict().items()
            },
            "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        }

    @torch.inference_mode()
    def _predict_one(self, horizon: Any, texts, *, text_paths=None) -> Any:
        encoder, head = self._build_modules()
        chunk_encoder = self._current_chunk_encoder
        encoder.load_state_dict(self.models_[horizon]["encoder_state"])
        chunk_encoder.load_state_dict(self.models_[horizon]["chunk_encoder_state"])
        head.load_state_dict(self.models_[horizon]["head_state"])
        encoder.eval()
        chunk_encoder.eval()
        head.eval()

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
            "chunk_num_heads": self.chunk_num_heads,
            "chunk_encoder_layers": self.chunk_encoder_layers,
            "chunk_ff_dim": self.chunk_ff_dim,
            "models": self.models_,
        }
        with save_path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BertS4:
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
            chunk_num_heads=int(state.get("chunk_num_heads", 8)),
            chunk_encoder_layers=int(state.get("chunk_encoder_layers", 1)),
            chunk_ff_dim=state.get("chunk_ff_dim"),
        )
        model.models_ = state["models"]
        return model


class FinBertS4(BertS4):
    """C2 FinBERT + S4 hierarchical chunk encoder."""

    name = "C2_finbert_s4"

    def __init__(self, *, pretrained: str = "ProsusAI/finbert", **kwargs) -> None:
        super().__init__(pretrained=pretrained, **kwargs)


def _build_chunk_encoder(
    *,
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    ff_dim: int | None,
    dropout: float,
) -> nn.TransformerEncoder:
    nhead = _compatible_num_heads(hidden_size, max(1, int(num_heads)))
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_size,
        nhead=nhead,
        dim_feedforward=int(ff_dim or hidden_size * 4),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))


def _compatible_num_heads(hidden_size: int, requested: int) -> int:
    for nhead in range(min(hidden_size, requested), 0, -1):
        if hidden_size % nhead == 0:
            return nhead
    return 1
