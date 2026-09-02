"""Regression heads on top of encoder embeddings.

Volatility head: encoder embedding → MLP → positive scalar via softplus.
Softplus output is preferred over raw linear so we never produce a negative
volatility prediction. QLIKE requires strictly positive variances.
"""

from __future__ import annotations

import torch
from torch import nn


class VolatilityHead(nn.Module):
    """MLP regressor for volatility forecasting.

    Two output modes:
      - ``positive=True`` (default): raw → softplus → strictly positive scalar.
        Use when training directly on realised volatility (positive target).
      - ``positive=False``: raw linear output. Use when training on
        ``log(volatility)`` — softplus would conflict with negative targets.
    """

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        eps: float = 1e-6,
        positive: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.positive = bool(positive)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.softplus = nn.Softplus(beta=1.0) if self.positive else None

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        raw = self.net(embedding).squeeze(-1)
        if self.softplus is None:
            return raw
        return self.softplus(raw) + self.eps
