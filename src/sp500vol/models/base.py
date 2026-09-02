"""Abstract base class for all forecasting models.

Every model in Block A/B/C/D inherits from this so that the training
loop and evaluator can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class VolatilityForecaster(ABC):
    """All A/B/C/D models implement this interface."""

    name: str = "base"

    @abstractmethod
    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        """Fit the model on training data, with optional validation set."""
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return point predictions for the given inputs."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model state. Must allow `load` round-trip."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> VolatilityForecaster:
        """Restore from saved state."""
        ...
