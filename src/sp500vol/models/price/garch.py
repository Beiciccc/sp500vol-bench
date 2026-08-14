"""A3/A4 GARCH(1,1) / EGARCH(1,1) baselines via the `arch` package.

Per-firm GARCH model:
  1. At fit time, identify the latest training filing date for each ticker
     in X_train. Fit one GARCH model per ticker on log returns up to that date.
  2. At predict time, for each unique (ticker, feature_window_end) pair,
     refit GARCH on returns up to feature_window_end (so we never use
     future information). Forecast next H days; sum variances; convert to
     annualised volatility.

Refitting per (ticker, date) is expensive (~50-200ms each). The cache
deduplicates across horizons so we do at most one fit per filing.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sp500vol.models.base import VolatilityForecaster
from sp500vol.models.price._refit_parallel import parallel_refit_predict
from sp500vol.models.price.hv import _horizon_historical_volatility

_ANNUALISATION_FACTOR = 252.0
_EPSILON = 1e-12
_HORIZON_TIE_BREAKER_WEIGHT = 1e-6
_MAX_FORECAST_VOL = 5.0
_MIN_FORECAST_VOL = 0.01
_MIN_HISTORY = 100  # minimum return observations before GARCH fit attempted
_PERCENT_SCALE = 100.0  # arch_model expects percent returns
_SIMULATION_REPS = 200  # for EGARCH / asymmetric vol: forecast via simulation


class GARCH(VolatilityForecaster):
    """Per-firm GARCH(p, q) baseline. Subclassed by EGARCH below."""

    name = "A3_garch"
    _VOL_MODEL = "GARCH"
    _FORECAST_METHOD = "analytic"  # analytic works for h>1 on standard GARCH

    def __init__(self, *, market_returns_df: pd.DataFrame, p: int = 1, q: int = 1) -> None:
        if "ticker" not in market_returns_df.columns or "date" not in market_returns_df.columns:
            raise ValueError("market_returns_df must have ticker / date / log_return columns")
        self.market_returns = _prepare_returns(market_returns_df)
        self.p = int(p)
        self.q = int(q)
        # Cached training fits for diagnostics; the predict path refits per filing.
        self.train_summary_: dict[str, dict[str, float]] = {}

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        del y_train, X_val, y_val
        self.train_summary_ = {}
        # The per-ticker training-MLE summary below is diagnostics only — predict() refits per
        # filing and NO metric/prediction reads train_summary_. Skipped by default to avoid a
        # serial single-core MLE loop over every ticker; set SP500VOL_PRICE_FIT_DIAGNOSTICS=1 to
        # populate it. Results are unaffected either way.
        if os.environ.get("SP500VOL_PRICE_FIT_DIAGNOSTICS", "0").strip().lower() not in {
            "1", "true", "yes", "on",
        }:
            return
        df = _require_dataframe(X_train, name="X_train")
        train_end = pd.to_datetime(df["filing_time_utc"]).dt.tz_convert(None).dt.normalize().max()
        for ticker in sorted(df["ticker"].unique()):
            returns = self._returns_up_to(ticker, train_end)
            if len(returns) < _MIN_HISTORY:
                continue
            result = self._fit_arch(returns)
            if result is None:
                continue
            self.train_summary_[ticker] = {
                "n_obs": float(len(returns)),
                "loglikelihood": float(result.loglikelihood),
            }

    def predict(self, X) -> np.ndarray:
        df = _require_dataframe(X, name="X")
        params = {
            "vol_model": self._VOL_MODEL,
            "p": self.p,
            "q": self.q,
            "forecast_method": self._FORECAST_METHOD,
            "sim_reps": _SIMULATION_REPS,
            "min_history": _MIN_HISTORY,
        }
        # Parallel per-(ticker, feature_window_end) refit+forecast (joblib,
        # by-ticker). Identical semantics to the serial loop; one fit per filing,
        # all horizons forecast from it.
        preds = parallel_refit_predict(df, self.market_returns, _garch_refit_one, params)

        # NaN-safe and simulation-safe: QLIKE is unstable for effectively zero
        # or explosive variance forecasts, so use a horizon-aware historical
        # volatility fallback when arch returns implausible values.
        fallback = _horizon_historical_volatility(
            df,
            epsilon=_MIN_FORECAST_VOL,
            max_vol=_MAX_FORECAST_VOL,
        )
        mask = ~np.isfinite(preds) | (preds <= _MIN_FORECAST_VOL) | (preds > _MAX_FORECAST_VOL)
        preds[mask] = fallback[mask]
        preds = (1.0 - _HORIZON_TIE_BREAKER_WEIGHT) * preds + (
            _HORIZON_TIE_BREAKER_WEIGHT * fallback
        )
        return np.clip(preds, _MIN_FORECAST_VOL, _MAX_FORECAST_VOL)

    def save(self, path: Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "p": self.p,
            "q": self.q,
            "vol_model": self._VOL_MODEL,
            "train_summary": self.train_summary_,
        }
        with save_path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> GARCH:
        raise NotImplementedError(
            "GARCH.load is not supported: predict requires the market_returns_df "
            "to refit per filing; instantiate a fresh model with the data instead."
        )

    # --- internals -------------------------------------------------------

    def _returns_up_to(self, ticker: str, cutoff: pd.Timestamp) -> pd.Series:
        rows = self.market_returns.get(ticker)
        if rows is None:
            return pd.Series(dtype=float)
        return rows.loc[:cutoff].dropna()

    def _fit_arch(self, returns: pd.Series):
        from arch import arch_model

        try:
            am = arch_model(
                returns * _PERCENT_SCALE,
                vol=self._VOL_MODEL,
                p=self.p,
                q=self.q,
                dist="normal",
                rescale=False,
            )
            return am.fit(disp="off", show_warning=False)
        except Exception:  # arch can throw assorted convergence errors
            return None

    def _forecast(self, result, *, horizon: int):
        """Forecast variance H steps ahead. Some vol families require simulation
        for horizon > 1 (EGARCH/asymmetric); standard GARCH supports analytic.
        """
        if self._FORECAST_METHOD == "analytic":
            return result.forecast(horizon=horizon, reindex=False)
        return result.forecast(
            horizon=horizon,
            method=self._FORECAST_METHOD,
            simulations=_SIMULATION_REPS,
            reindex=False,
        )


class EGARCH(GARCH):
    """A4 — EGARCH(1,1) with asymmetric leverage. Same interface as GARCH."""

    name = "A4_egarch"
    _VOL_MODEL = "EGARCH"
    _FORECAST_METHOD = "simulation"  # analytic h>1 not supported for EGARCH


def _garch_refit_one(
    series: pd.Series,
    feature_end: pd.Timestamp,
    horizons: list[int],
    params: dict,
) -> dict[int, float]:
    """Refit one GARCH/EGARCH on returns up to feature_end; forecast each horizon.

    Module-level (picklable) so joblib workers can call it. Mirrors the serial
    fit+forecast: one fit per (ticker, feature_end), all horizons forecast from
    it. The analytic path is deterministic; the EGARCH simulation path is seeded
    per (date, horizon) for reproducibility.
    """
    from arch import arch_model

    returns = series.loc[:feature_end].dropna()
    if len(returns) < params["min_history"]:
        return dict.fromkeys(horizons, float("nan"))
    try:
        model = arch_model(
            returns.to_numpy() * _PERCENT_SCALE,
            vol=params["vol_model"],
            p=params["p"],
            q=params["q"],
            dist="normal",
            rescale=False,
        )
        result = model.fit(disp="off", show_warning=False)
    except Exception:  # arch can throw assorted convergence errors
        return dict.fromkeys(horizons, float("nan"))

    analytic = params["forecast_method"] == "analytic"
    out: dict[int, float] = {}
    for horizon in horizons:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if analytic:
                    forecast = result.forecast(horizon=horizon, reindex=False)
                else:
                    seed = (int(pd.Timestamp(feature_end).value // 1_000_000_000) + horizon) % (
                        2**32
                    )
                    np.random.seed(seed)
                    forecast = result.forecast(
                        horizon=horizon,
                        method=params["forecast_method"],
                        simulations=params["sim_reps"],
                        reindex=False,
                    )
            total_var = float(forecast.variance.values[0].sum()) / (_PERCENT_SCALE**2)
            out[horizon] = float(np.sqrt(_ANNUALISATION_FACTOR / horizon * total_var))
        except Exception:
            out[horizon] = float("nan")
    return out


def _prepare_returns(df: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.tz_localize(None).dt.normalize()
    for ticker, group in work.groupby("ticker", sort=False):
        series = group.set_index("date")["log_return"].astype(float).sort_index()
        out[str(ticker)] = series
    return out


def _require_dataframe(value, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame for GARCH-family models")
    return value
