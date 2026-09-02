"""D1 Concat-MLP fusion baseline."""

from __future__ import annotations

import torch
from torch import nn

from sp500vol.models.fusion.gated_fusion import GatedFusion
from sp500vol.models.neural_text.bert_s1 import _EPSILON
from sp500vol.models.neural_text.encoders import CLSEncoder
from sp500vol.models.neural_text.heads import VolatilityHead


class _ConcatFusion(nn.Module):
    """Project price/text branches separately, then concatenate without a gate."""

    def __init__(self, text_dim: int, price_dim: int = 3, proj_dim: int = 128) -> None:
        super().__init__()
        self.price_proj = nn.Sequential(nn.Linear(price_dim, proj_dim), nn.GELU())
        self.text_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.GELU())

    def forward(self, price: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.price_proj(price), self.text_proj(text)], dim=-1)


class ConcatMLP(GatedFusion):
    """D1 — naive concatenation fusion of HAR-RV price features and FinBERT text."""

    name = "D1_concat_mlp"

    def _build_modules(self) -> tuple[CLSEncoder, _ConcatFusion, VolatilityHead]:
        encoder = CLSEncoder(self.encoder_cfg).to(self.device)
        fusion = _ConcatFusion(encoder.hidden_size, price_dim=3, proj_dim=self.proj_dim).to(
            self.device
        )
        head = VolatilityHead(
            self.proj_dim * 2,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            eps=_EPSILON,
            positive=not self.log_target,
        ).to(self.device)
        return encoder, fusion, head
