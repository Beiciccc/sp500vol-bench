"""B1: Bag-of-Words + Ridge regression.

Replicates Kogan et al. (2009): vectorise filing text via top-N unigrams,
predict log-RV via linear ridge regression, one ridge per horizon (matches the
HAR-RV setup for fair comparison).

Performance: a filing's 3 horizon rows share the same text, so the corpus is
vectorised ONCE over the unique filings (a single shared vectoriser) and the
per-horizon ridges reuse that matrix; alpha CV runs across cores. This is ~3x
fewer tokenizations than vectorising per horizon and uses all cores for CV.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge

from sp500vol.models.base import VolatilityForecaster
from sp500vol.models.classical_text._fit_utils import (
    build_ridge,
    fit_ridge_cv,
    maybe_exp,
    maybe_log,
    unique_filing_index,
)
from sp500vol.models.classical_text._text_dataset import load_texts

_DEFAULT_MAX_FEATURES = 5000
# Default ridge alpha CV grid: 0.1 to 10000. 5-fold CV picks alpha per horizon.
_DEFAULT_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0)
_MIN_PREDICTION = 0.02
# Cap predictions in real volatility space; see module docstring rationale.
_MAX_PREDICTION = 5.0


class BoWRidge(VolatilityForecaster):
    """Bag-of-words + Ridge with one shared vectoriser and per-horizon ridges."""

    name = "B1_bow_ridge"

    def __init__(
        self,
        *,
        max_features: int = _DEFAULT_MAX_FEATURES,
        ridge_alpha: float | None = None,
        log_target: bool = True,
        alpha_grid: tuple[float, ...] | None = None,
    ) -> None:
        """If ``ridge_alpha`` is None, 5-fold CV picks alpha per horizon over
        ``alpha_grid``; if set, a plain Ridge with that fixed alpha is used."""
        self.max_features = int(max_features)
        self.ridge_alpha = None if ridge_alpha is None else float(ridge_alpha)
        self.log_target = bool(log_target)
        self.alpha_grid = tuple(alpha_grid) if alpha_grid else _DEFAULT_ALPHA_GRID
        self.vectorizer_: CountVectorizer | None = None
        self.models_: dict[Any, Ridge] = {}

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        del X_val, y_val
        df = _require_dataframe(X_train, name="X_train")
        target = np.asarray(y_train, dtype=float)
        if len(df) != len(target):
            raise ValueError(f"X_train has {len(df)} rows but y_train has {len(target)} values")

        matrix, inverse = self._fit_vectorizer(df)
        horizons = _horizons(df)
        self.models_ = {}
        for horizon in sorted(set(horizons.tolist())):
            mask = horizons == horizon
            x_h = matrix[inverse[mask]]
            y_h = maybe_log(target[mask], log_target=self.log_target)
            if self.ridge_alpha is None:
                ridge = fit_ridge_cv(x_h, y_h, self.alpha_grid)
            else:
                ridge = build_ridge(self.ridge_alpha)
                ridge.fit(x_h, y_h)
            self.models_[horizon] = ridge

    def predict(self, X) -> np.ndarray:
        df = _require_dataframe(X, name="X")
        if self.vectorizer_ is None:
            raise RuntimeError(f"{self.name} must be fitted before predict")
        first_idx, inverse = unique_filing_index(df)
        texts = load_texts(df.iloc[first_idx].reset_index(drop=True), persist_new=False)
        matrix = self.vectorizer_.transform(texts)
        horizons = _horizons(df)
        preds = np.empty(len(df), dtype=float)
        for horizon in sorted(set(horizons.tolist())):
            if horizon not in self.models_:
                raise ValueError(f"no {self.name} model fitted for horizon_days={horizon!r}")
            mask = horizons == horizon
            raw = self.models_[horizon].predict(matrix[inverse[mask]])
            preds[mask] = _maybe_exp(raw, log_target=self.log_target)
        return preds

    def save(self, path: Path) -> None:
        if not self.models_:
            raise RuntimeError(f"{self.name} must be fitted before save")
        state = {
            "max_features": self.max_features,
            "ridge_alpha": self.ridge_alpha,
            "log_target": self.log_target,
            "vectorizer": self.vectorizer_,
            "models": self.models_,
        }
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BoWRidge:
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        raw_alpha = state["ridge_alpha"]
        model = cls(
            max_features=int(state["max_features"]),
            ridge_alpha=None if raw_alpha is None else float(raw_alpha),
            log_target=bool(state["log_target"]),
        )
        model.vectorizer_ = state["vectorizer"]
        model.models_ = dict(state["models"])
        return model

    # --- internals -------------------------------------------------------

    def _build_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(
            lowercase=True,
            max_features=self.max_features,
            token_pattern=r"(?u)\b[a-z]{2,}\b",
        )

    def _fit_vectorizer(self, df: pd.DataFrame):
        """Vectorise the unique filings once; return (matrix, per-row inverse)."""
        first_idx, inverse = unique_filing_index(df)
        texts = load_texts(df.iloc[first_idx].reset_index(drop=True))
        self.vectorizer_ = self._build_vectorizer()
        matrix = self.vectorizer_.fit_transform(texts)
        return matrix, inverse


def _require_dataframe(value, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame for classical text models")
    return value


def _horizons(df: pd.DataFrame) -> np.ndarray:
    if "horizon_days" not in df.columns:
        raise ValueError("DataFrame must include 'horizon_days' for per-horizon fitting")
    return df["horizon_days"].astype(int).to_numpy()


def _maybe_exp(values: np.ndarray, *, log_target: bool) -> np.ndarray:
    return maybe_exp(
        values, log_target=log_target, min_pred=_MIN_PREDICTION, max_pred=_MAX_PREDICTION
    )
