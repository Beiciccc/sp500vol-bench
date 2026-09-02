"""Transformer encoder wrappers.

Thin layer over HuggingFace AutoModel/AutoTokenizer so the rest of the codebase
doesn't import transformers directly. All encoders return CLS-pooled embeddings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

_TORCH_COMPILE_ENV = "SP500VOL_TORCH_COMPILE"


def _torch_compile_enabled() -> bool:
    """Off by default; enable via env after a smoke validates numerics + speed on one
    fine-tune model. torch.compile gives ~15-25% step speedup on A100 bf16 for the
    512-token BERT-family encoders."""
    return os.environ.get(_TORCH_COMPILE_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class EncoderConfig:
    pretrained: str
    max_length: int = 512


class CLSEncoder(nn.Module):
    """Wraps a HuggingFace encoder and exposes a single CLS embedding."""

    # Subclasses whose forward path graph-breaks under torch.compile (e.g. Longformer's
    # dynamic global_attention_mask) set this False to stay eager.
    _supports_compile = True

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained)
        self.encoder = AutoModel.from_pretrained(cfg.pretrained)
        self.hidden_size: int = self.encoder.config.hidden_size
        # Compile the forward FUNCTION, not self.encoder, so the module's state_dict keys
        # stay clean (no `_orig_mod.` prefix) and checkpoints / model.pkl stay compatible
        # whether or not compile is enabled.
        self._compiled_fwd = None
        if _torch_compile_enabled() and self._supports_compile:
            self._compiled_fwd = torch.compile(self._raw_forward)

    @property
    def pad_to_multiple_of(self) -> int | None:
        """Round dynamic-padding lengths up to this multiple. None = pad to the exact
        batch-longest (BERT-family: no constraint). Longformer overrides this with its
        attention window, which the model requires the sequence length to be a multiple of."""
        return None

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Truncating + dynamic-padding tokeniser (S1 strategy). Pads to the batch-longest
        (rounded to pad_to_multiple_of) rather than a fixed max_length, so short documents
        (e.g. 8-Ks, median ~930 tok) are not padded out to the full window."""
        return self.tokenizer(
            texts,
            max_length=self.cfg.max_length,
            truncation=True,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

    def _raw_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS pooling — first token of the last hidden state. Some models
        # (e.g. RoBERTa, FinBERT) work the same way; for sentence-pair models
        # we'd use pooler_output instead, but CLS-only is the cleaner default
        # for regression.
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        fn = self._compiled_fwd or self._raw_forward
        return fn(input_ids, attention_mask)


class LongformerEncoder(CLSEncoder):
    """Longformer CLS encoder with the first token marked as global attention."""

    # The dynamic global_attention_mask triggers graph breaks/recompiles under
    # torch.compile, so Longformer stays eager.
    _supports_compile = False

    @property
    def pad_to_multiple_of(self) -> int | None:
        """Longformer requires the sequence length to be a multiple of its attention
        window; padding to that multiple lets the model skip its internal re-padding and
        keeps the CLS global-attention mask aligned. (longformer-base-4096 → 512.)"""
        window = self.encoder.config.attention_window
        return int(max(window)) if isinstance(window, (list, tuple)) else int(window)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        global_attention_mask = torch.zeros_like(attention_mask)
        if global_attention_mask.shape[1] > 0:
            global_attention_mask[:, 0] = 1
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        return out.last_hidden_state[:, 0, :]
