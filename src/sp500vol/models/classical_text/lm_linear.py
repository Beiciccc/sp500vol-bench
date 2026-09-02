"""B3: Loughran-McDonald dictionary proportions + Ridge regression.

Per-filing features are the 8 L-M category proportions (count_category /
total_tokens). This is the cleanest direct test of "sentiment dictionary alone,
no fancy representation, beats price baselines or not".

Performance: a filing's 3 horizon rows share the same text, so proportions are
computed ONCE per unique filing (deduped on accession) and fanned out across
cores in contiguous chunks. Proportions are per-text (no corpus dependency), so
this is bit-identical to the prior per-row serial computation, just ~3x fewer
tokenizations and multi-core.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

from sp500vol.features.lm_dictionary import LM_CATEGORIES, LoughranMcDonaldDictionary
from sp500vol.models.base import VolatilityForecaster
from sp500vol.models.classical_text._fit_utils import (
    build_ridge,
    fit_ridge_cv,
    maybe_exp,
    maybe_log,
    text_n_jobs,
    unique_filing_index,
)
from sp500vol.models.classical_text._text_dataset import load_texts

_DEFAULT_ALPHA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)  # smaller grid for 8-feature L-M
_MIN_PREDICTION = 0.02
_MAX_PREDICTION = 5.0  # see bow_ridge.py for rationale
# Below this many unique filings, featurise serially (joblib spawn overhead).
_FEATURISE_MIN_ROWS = 1000
_FEATURISE_CHUNKS = 10


class LMLinear(VolatilityForecaster):
    """L-M proportions → Ridge, one submodel per horizon."""

    name = "B3_lm_linear"

    def __init__(
        self,
        *,
        dictionary: LoughranMcDonaldDictionary | None = None,
        ridge_alpha: float | None = None,
        log_target: bool = True,
        alpha_grid: tuple[float, ...] | None = None,
    ) -> None:
        self.dictionary = dictionary or LoughranMcDonaldDictionary.mock()
        self.ridge_alpha = None if ridge_alpha is None else float(ridge_alpha)
        self.log_target = bool(log_target)
        self.alpha_grid = tuple(alpha_grid) if alpha_grid else _DEFAULT_ALPHA_GRID
        self.feature_names: tuple[str, ...] = LM_CATEGORIES
        self.models_: dict[Any, Ridge] = {}

    # --- forecaster API --------------------------------------------------

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        del X_val, y_val
        df = _require_dataframe(X_train, name="X_train")
        target = np.asarray(y_train, dtype=float)
        if len(df) != len(target):
            raise ValueError(f"X_train has {len(df)} rows but y_train has {len(target)} values")

        features = self._featurise(df, persist_new=True)
        horizons = _horizons(df)
        self.models_ = {}
        for horizon in sorted(set(horizons.tolist())):
            mask = horizons == horizon
            self._fit_one(horizon, features[mask], target[mask])

    def predict(self, X) -> np.ndarray:
        df = _require_dataframe(X, name="X")
        features = self._featurise(df, persist_new=False)
        horizons = _horizons(df)
        preds = np.empty(len(df), dtype=float)
        for horizon in sorted(set(horizons.tolist())):
            if horizon not in self.models_:
                raise ValueError(f"no {self.name} model fitted for horizon_days={horizon!r}")
            mask = horizons == horizon
            raw = self.models_[horizon].predict(features[mask])
            preds[mask] = _maybe_exp(raw, log_target=self.log_target)
        return preds

    def save(self, path: Path) -> None:
        if not self.models_:
            raise RuntimeError(f"{self.name} must be fitted before save")
        state = {
            "feature_names": self.feature_names,
            "ridge_alpha": self.ridge_alpha,
            "log_target": self.log_target,
            "models": self.models_,
            "dictionary_by_category": {k: tuple(v) for k, v in self.dictionary.by_category.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> LMLinear:
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        dictionary = LoughranMcDonaldDictionary(
            by_category={k: frozenset(v) for k, v in state["dictionary_by_category"].items()}
        )
        raw_alpha = state["ridge_alpha"]
        model = cls(
            dictionary=dictionary,
            ridge_alpha=None if raw_alpha is None else float(raw_alpha),
            log_target=bool(state["log_target"]),
        )
        model.feature_names = tuple(state["feature_names"])
        model.models_ = dict(state["models"])
        return model

    # --- internals -------------------------------------------------------

    def _featurise(self, df: pd.DataFrame, *, persist_new: bool) -> np.ndarray:
        """Featurise each unique filing once (deduped), fanned out by chunk."""
        first_idx, inverse = unique_filing_index(df)
        uniq_texts = load_texts(
            df.iloc[first_idx].reset_index(drop=True),
            persist_new=persist_new,
        )

        if len(uniq_texts) < _FEATURISE_MIN_ROWS:
            uniq_rows = [self._row_features(t) for t in uniq_texts]
        else:
            index_chunks = np.array_split(np.arange(len(uniq_texts)), _FEATURISE_CHUNKS)
            text_chunks = [
                [uniq_texts[i] for i in idx.tolist()] for idx in index_chunks if idx.size
            ]
            results = Parallel(n_jobs=text_n_jobs())(
                delayed(self._featurise_chunk)(chunk) for chunk in text_chunks
            )
            uniq_rows = [row for chunk_rows in results for row in chunk_rows]

        uniq_arr = np.asarray(uniq_rows, dtype=float)
        return uniq_arr[inverse]

    def _featurise_chunk(self, texts: list[str]) -> list[list[float]]:
        return [self._row_features(t) for t in texts]

    def _row_features(self, text: str) -> list[float]:
        proportions = self.dictionary.proportions(text)
        return [proportions.get(cat, 0.0) for cat in self.feature_names]

    def _fit_one(self, horizon: Any, features: np.ndarray, target: np.ndarray) -> None:
        y = maybe_log(target, log_target=self.log_target)
        if self.ridge_alpha is None:
            ridge = fit_ridge_cv(features, y, self.alpha_grid)
        else:
            ridge = build_ridge(self.ridge_alpha)
            ridge.fit(features, y)
        self.models_[horizon] = ridge


def _require_dataframe(value, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame for L-M models")
    return value


def _horizons(df: pd.DataFrame) -> np.ndarray:
    if "horizon_days" not in df.columns:
        raise ValueError("DataFrame must include 'horizon_days'")
    return df["horizon_days"].astype(int).to_numpy()


def _maybe_exp(values: np.ndarray, *, log_target: bool) -> np.ndarray:
    return maybe_exp(
        values, log_target=log_target, min_pred=_MIN_PREDICTION, max_pred=_MAX_PREDICTION
    )
