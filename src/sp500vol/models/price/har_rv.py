"""A2 HAR-RV (Corsi 2009): the primary price-only baseline for H1.

HAR-RV regresses log(RV_t+1) on log(RV_t), log(RV_t-5..t), log(RV_t-22..t)
plus a constant. Simple linear model — runs on CPU in seconds.

Reference:
  Corsi, F. (2009). A simple approximate long-memory model of realized
  volatility. Journal of Financial Econometrics 7(2):174-196.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sp500vol.models.base import VolatilityForecaster

_ANNUALISATION_FACTOR = 252.0
_EXPECTED_ARRAY_DIMENSIONS = 2
_FEATURE_NAMES = ("feature_rv_1d", "feature_rv_5d", "feature_rv_22d")


class HARRV(VolatilityForecaster):
    """Heterogeneous Autoregressive model of Realised Volatility."""

    name = "A2_har_rv"

    def __init__(
        self, *, epsilon: float = 1e-12, log_transform: bool = True, smearing: bool = True
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.log_transform = bool(log_transform)
        # Duan (1983) smearing retransformation. A log-OLS forecast exp(E[log RV|X]) is the
        # conditional MEDIAN; for the right-skewed RV the MEAN is larger by ~exp(sigma^2/2),
        # so an uncorrected log-HAR systematically UNDER-forecasts (QLIKE penalises this).
        # The smearing factor S = mean(exp(train residuals)) rescales exp(Xb) to an unbiased
        # mean forecast. Only applies in log space; a no-op (S=1) when log_transform=False.
        self.smearing = bool(smearing)
        self.feature_names = list(_FEATURE_NAMES)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.smear_: float = 1.0
        self.coefs_: dict[Any, np.ndarray] = {}
        self.intercepts_: dict[Any, float] = {}
        self.smears_: dict[Any, float] = {}
        self._uses_horizon_models = False

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        """Fit one HAR-RV OLS model, or one model per ``horizon_days`` value."""
        del X_val, y_val
        features, horizons = self._extract_features_and_horizons(X_train)
        target = self._as_1d_float_array(y_train, name="y_train")
        if len(target) != len(features):
            raise ValueError(
                f"X_train has {len(features)} rows but y_train has {len(target)} values"
            )

        if horizons is None:
            self.coef_, self.intercept_, self.smear_ = self._fit_one(features, target)
            self.coefs_ = {}
            self.intercepts_ = {}
            self.smears_ = {}
            self._uses_horizon_models = False
            return

        self.coef_ = None
        self.intercept_ = None
        self.coefs_ = {}
        self.intercepts_ = {}
        self.smears_ = {}
        for horizon in sorted(set(horizons.tolist())):
            mask = horizons == horizon
            coef, intercept, smear = self._fit_one(features[mask], target[mask])
            self.coefs_[horizon] = coef
            self.intercepts_[horizon] = intercept
            self.smears_[horizon] = smear
        self._uses_horizon_models = True

    def predict(self, X) -> np.ndarray:
        """Return positive finite realised-volatility forecasts."""
        features, horizons = self._extract_features_and_horizons(X)

        if self._uses_horizon_models:
            if horizons is None:
                raise ValueError("X must include horizon_days for a horizon-specific HARRV")

            predictions = np.empty(len(features), dtype=float)
            for horizon in sorted(set(horizons.tolist())):
                if horizon not in self.coefs_:
                    raise ValueError(f"no HARRV model fitted for horizon_days={horizon!r}")
                mask = horizons == horizon
                predictions[mask] = self._predict_one(
                    features[mask],
                    self.coefs_[horizon],
                    self.intercepts_[horizon],
                    self.smears_.get(horizon, 1.0),
                )
            return predictions

        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("HARRV must be fitted before predict")
        return self._predict_one(features, self.coef_, self.intercept_, self.smear_)

    def save(self, path: Path) -> None:
        """Persist model state via pickle."""
        if self._uses_horizon_models:
            coefs = self.coefs_
            intercepts = self.intercepts_
            smears = self.smears_
        else:
            if self.coef_ is None or self.intercept_ is None:
                raise RuntimeError("HARRV must be fitted before save")
            coefs = {None: self.coef_}
            intercepts = {None: self.intercept_}
            smears = {None: self.smear_}

        state = {
            "feature_names": self.feature_names,
            "coefs": coefs,
            "intercepts": intercepts,
            "smears": smears,
            "epsilon": self.epsilon,
            "log_transform": self.log_transform,
            "smearing": self.smearing,
            "uses_horizon_models": self._uses_horizon_models,
        }
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as file:
            pickle.dump(state, file)

    @classmethod
    def load(cls, path: Path) -> HARRV:
        """Restore a model saved by :meth:`save`."""
        load_path = Path(path)
        with load_path.open("rb") as file:
            state = pickle.load(file)

        model = cls(
            epsilon=float(state.get("epsilon", 1e-12)),
            log_transform=bool(state.get("log_transform", True)),
            smearing=bool(state.get("smearing", False)),  # legacy models had no smearing
        )
        model.feature_names = list(state.get("feature_names", _FEATURE_NAMES))
        model._uses_horizon_models = bool(state.get("uses_horizon_models", False))

        coefs = {key: np.asarray(value, dtype=float) for key, value in state["coefs"].items()}
        intercepts = {key: float(value) for key, value in state["intercepts"].items()}
        # Backward-compatible: pre-smearing checkpoints default to S=1 (no correction).
        smears = {key: float(value) for key, value in state.get("smears", {}).items()}

        if model._uses_horizon_models:
            model.coefs_ = coefs
            model.intercepts_ = intercepts
            model.smears_ = {k: smears.get(k, 1.0) for k in coefs}
            model.coef_ = None
            model.intercept_ = None
        else:
            model.coef_ = coefs[None]
            model.intercept_ = intercepts[None]
            model.smear_ = smears.get(None, 1.0)
            model.coefs_ = {}
            model.intercepts_ = {}
        return model

    def _fit_one(self, features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
        transformed_features = self._transform_features(features)
        transformed_target = self._transform_target(target)
        valid = np.isfinite(transformed_target) & np.isfinite(transformed_features).all(axis=1)
        if not valid.any():
            raise ValueError("no valid finite rows available to fit HARRV")

        design = np.column_stack([np.ones(valid.sum()), transformed_features[valid]])
        params, *_ = np.linalg.lstsq(design, transformed_target[valid], rcond=None)
        # Duan smearing factor S = mean(exp(residual)) on the training fit; 1.0 unless we
        # are in log space and smearing is enabled. Corrects the median→mean retransformation
        # bias so the exponentiated forecast targets E[RV|X] rather than the median.
        smear = 1.0
        if self.log_transform and self.smearing:
            residual = transformed_target[valid] - design @ params
            smear = float(np.mean(np.exp(residual)))
        return params[1:].astype(float), float(params[0]), smear

    def _predict_one(
        self,
        features: np.ndarray,
        coef: np.ndarray,
        intercept: float,
        smear: float = 1.0,
    ) -> np.ndarray:
        transformed_features = self._transform_features(features)
        if not np.isfinite(transformed_features).all():
            raise ValueError("X contains invalid HAR-RV feature values")

        raw_prediction = transformed_features @ coef + intercept
        if self.log_transform:
            max_log = np.log(np.finfo(float).max) - 1.0
            raw_prediction = np.clip(raw_prediction, np.log(self.epsilon), max_log)
            prediction = np.exp(raw_prediction) * smear - self.epsilon
        else:
            prediction = raw_prediction
        return np.maximum(prediction, self.epsilon).astype(float)

    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        if not self.log_transform:
            return features
        safe_features = np.where(features >= 0.0, features, np.nan)
        return np.log(safe_features + self.epsilon)

    def _transform_target(self, target: np.ndarray) -> np.ndarray:
        if not self.log_transform:
            return target
        safe_target = np.where(target >= 0.0, target, np.nan)
        return np.log(safe_target + self.epsilon)

    def _extract_features_and_horizons(self, x) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(x, pd.DataFrame):
            features = self._extract_dataframe_features(x)
            horizons = None
            if "horizon_days" in x.columns:
                horizons = np.array(
                    [self._normalise_horizon(value) for value in x["horizon_days"]],
                    dtype=object,
                )
            return features, horizons

        features = self._extract_array_features(x)
        return features, None

    @staticmethod
    def _extract_dataframe_features(x: pd.DataFrame) -> np.ndarray:
        missing = [name for name in _FEATURE_NAMES[1:] if name not in x.columns]
        if "feature_rv_1d" in x.columns:
            daily = x["feature_rv_1d"].to_numpy(dtype=float)
        elif "feature_return_1d" in x.columns:
            daily = np.sqrt(_ANNUALISATION_FACTOR) * np.abs(
                x["feature_return_1d"].to_numpy(dtype=float)
            )
        else:
            missing.append("feature_rv_1d or feature_return_1d")

        if missing:
            raise ValueError(f"X missing required HAR-RV features: {missing}")

        return np.column_stack(
            [
                daily,
                x["feature_rv_5d"].to_numpy(dtype=float),
                x["feature_rv_22d"].to_numpy(dtype=float),
            ]
        )

    @staticmethod
    def _extract_array_features(x) -> np.ndarray:
        features = np.asarray(x, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != _EXPECTED_ARRAY_DIMENSIONS:
            raise ValueError("X must be a 2D array or a pandas DataFrame")
        if features.shape[1] < len(_FEATURE_NAMES):
            raise ValueError("numpy X must contain columns for rv_1d, rv_5d, and rv_22d")
        return features[:, : len(_FEATURE_NAMES)]

    @staticmethod
    def _as_1d_float_array(values, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            array = array.reshape(-1)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        return array

    @staticmethod
    def _normalise_horizon(value: Any) -> Any:
        if pd.isna(value):
            raise ValueError("horizon_days must not contain missing values")
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
