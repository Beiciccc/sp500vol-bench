"""A1 Naive Historical Volatility (NHV).

Predict realised volatility over the next H trading days as equal to the
trailing 22-day realised volatility computed at filing time. No fitting —
the strict lower-bound baseline for H1.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sp500vol.models.base import VolatilityForecaster

_EPSILON = 1e-12
_MAX_HV_VOL = 5.0


class NaiveHV(VolatilityForecaster):
    """A1 — horizon-aware historical-volatility baseline. Zero parameters."""

    name = "A1_hv"

    def __init__(self) -> None:
        self.fitted_ = False

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        del y_train, X_val, y_val
        _require_column(X_train, "feature_rv_22d")
        self.fitted_ = True

    def predict(self, X) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("NaiveHV must be fitted before predict")
        _require_column(X, "feature_rv_22d")
        return _horizon_historical_volatility(X, epsilon=_EPSILON, max_vol=_MAX_HV_VOL)

    def save(self, path: Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as fh:
            pickle.dump({"fitted": self.fitted_}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> NaiveHV:
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        model = cls()
        model.fitted_ = bool(state.get("fitted", False))
        return model


def _require_column(value, column: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"NaiveHV requires a pandas DataFrame; got {type(value)}")
    if column not in value.columns:
        raise ValueError(f"NaiveHV requires column {column!r}")


def _horizon_historical_volatility(
    value: pd.DataFrame,
    *,
    epsilon: float = _EPSILON,
    max_vol: float | None = None,
) -> np.ndarray:
    """Return a simple horizon-aware trailing-RV fallback.

    Available price features are 1d/5d/22d. For Block A's 5/10/20-day labels,
    use 5d for short horizons, 22d for long horizons, and interpolate in
    variance space for the middle horizon. If a feature is missing, fall back to
    the required 22d value.
    """
    _require_column(value, "feature_rv_22d")
    rv22 = value["feature_rv_22d"].to_numpy(dtype=float)
    rv5 = value["feature_rv_5d"].to_numpy(dtype=float) if "feature_rv_5d" in value else rv22
    rv1 = value["feature_rv_1d"].to_numpy(dtype=float) if "feature_rv_1d" in value else rv5

    if "horizon_days" not in value:
        out = rv22.copy()
    else:
        horizons = value["horizon_days"].to_numpy(dtype=float)
        out = np.empty(len(value), dtype=float)
        short = horizons <= 1
        mid_short = (horizons > 1) & (horizons <= 5)
        long = horizons >= 20
        middle = ~(short | mid_short | long)
        out[short] = rv1[short]
        out[mid_short] = rv5[mid_short]
        out[long] = rv22[long]
        if middle.any():
            weight = np.clip((horizons[middle] - 5.0) / 15.0, 0.0, 1.0)
            variance = (1.0 - weight) * np.square(rv5[middle]) + weight * np.square(rv22[middle])
            out[middle] = np.sqrt(variance)

    out = np.where(np.isfinite(out), out, epsilon)
    out = np.maximum(out, epsilon)
    if max_vol is not None:
        out = np.minimum(out, max_vol)
    return out
