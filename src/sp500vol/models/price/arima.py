"""A5 ARIMA on log-RV baseline.

Per-firm ARIMA fit on the log of daily backward-looking RV series. At predict
time, refit per (ticker, feature_window_end) on returns up to that date, then
forecast log-RV one step ahead. We use a fixed (1, 0, 1) order by default —
order selection by AIC is implemented but optional.
"""

from __future__ import annotations

import math
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
_MAX_FORECAST_VOL = 5.0
_MIN_HISTORY = 60
_RV_WINDOW = 22  # backward RV window for the modelled series


class ARIMAVol(VolatilityForecaster):
    """ARIMA on log of trailing 22-day realised volatility."""

    name = "A5_arima"

    def __init__(
        self,
        *,
        market_returns_df: pd.DataFrame,
        order: tuple[int, int, int] = (1, 0, 1),
    ) -> None:
        if "ticker" not in market_returns_df.columns or "date" not in market_returns_df.columns:
            raise ValueError("market_returns_df must have ticker / date / log_return columns")
        self.market_returns = _prepare_returns(market_returns_df)
        self.order = tuple(int(v) for v in order)
        self.fit_summary_: dict[str, dict[str, float]] = {}

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        del y_train, X_val, y_val
        self.fit_summary_ = {}
        # Diagnostics-only per-ticker MLE summary (predict() refits per filing; no metric reads
        # fit_summary_). Skipped by default; SP500VOL_PRICE_FIT_DIAGNOSTICS=1 re-enables it.
        if os.environ.get("SP500VOL_PRICE_FIT_DIAGNOSTICS", "0").strip().lower() not in {
            "1", "true", "yes", "on",
        }:
            return
        df = _require_dataframe(X_train, name="X_train")
        train_end = pd.to_datetime(df["filing_time_utc"]).dt.tz_convert(None).dt.normalize().max()
        for ticker in sorted(df["ticker"].unique()):
            log_rv = self._log_rv_up_to(ticker, train_end)
            if len(log_rv) < _MIN_HISTORY:
                continue
            result = self._fit_arima(log_rv)
            if result is None:
                continue
            self.fit_summary_[ticker] = {
                "n_obs": float(len(log_rv)),
                "aic": float(result.aic),
            }

    def predict(self, X) -> np.ndarray:
        df = _require_dataframe(X, name="X")
        params = {
            "order": self.order,
            "rv_window": _RV_WINDOW,
            "min_history": _MIN_HISTORY,
            "annualisation": _ANNUALISATION_FACTOR,
            "epsilon": _EPSILON,
        }
        # Parallel per-(ticker, feature_window_end) refit+forecast (joblib,
        # by-ticker). One fit per filing; horizon-specific forecasts are derived
        # from the first H predicted log-RV steps.
        preds = parallel_refit_predict(df, self.market_returns, _arima_refit_one, params)

        # Non-finite fallback to horizon-aware trailing RV.
        if (~np.isfinite(preds)).any():
            fallback = _horizon_historical_volatility(
                df,
                epsilon=_EPSILON,
                max_vol=_MAX_FORECAST_VOL,
            )
            mask = ~np.isfinite(preds)
            preds[mask] = np.maximum(fallback[mask], _EPSILON)
        return np.clip(preds, _EPSILON, _MAX_FORECAST_VOL)

    def save(self, path: Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"order": self.order, "fit_summary": self.fit_summary_}
        with save_path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> ARIMAVol:
        raise NotImplementedError(
            "ARIMAVol.load is not supported: predict refits per filing. "
            "Instantiate a fresh model with the market data instead."
        )

    # --- internals -------------------------------------------------------

    def _log_rv_up_to(self, ticker: str, cutoff: pd.Timestamp) -> pd.Series:
        rows = self.market_returns.get(ticker)
        if rows is None:
            return pd.Series(dtype=float)
        returns = rows.loc[:cutoff].dropna()
        if len(returns) < _RV_WINDOW + 1:
            return pd.Series(dtype=float)
        rv = np.sqrt(_ANNUALISATION_FACTOR / _RV_WINDOW * (returns**2).rolling(_RV_WINDOW).sum())
        log_rv = np.log(rv.dropna() + _EPSILON)
        return log_rv

    def _fit_arima(self, log_rv: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ARIMA(log_rv, order=self.order, enforce_stationarity=False).fit()
        except Exception:
            return None


def _arima_refit_one(
    series: pd.Series,
    feature_end: pd.Timestamp,
    horizons: list[int],
    params: dict,
) -> dict[int, float]:
    """Refit one ARIMA on log-RV up to feature_end; forecast each horizon.

    Module-level (picklable) for joblib workers. Mirrors the serial fit+forecast.
    The per-horizon value is the RMS of the first H predicted volatility levels,
    preserving the annualised-volatility scale while making the baseline
    horizon-aware.
    """
    from statsmodels.tsa.arima.model import ARIMA

    returns = series.loc[:feature_end].dropna()
    if len(returns) < params["rv_window"] + 1:
        return dict.fromkeys(horizons, float("nan"))
    rolling_var = (returns**2).rolling(params["rv_window"]).sum()
    rv = np.sqrt(params["annualisation"] / params["rv_window"] * rolling_var)
    log_rv = np.log(rv.dropna() + params["epsilon"])
    if len(log_rv) < params["min_history"]:
        return dict.fromkeys(horizons, float("nan"))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ARIMA(log_rv, order=params["order"], enforce_stationarity=False).fit()
    except Exception:
        return dict.fromkeys(horizons, float("nan"))

    max_horizon = max(int(h) for h in horizons)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast_log_rv = np.asarray(result.forecast(steps=max_horizon), dtype=float)
    max_log = math.log(np.finfo(float).max) - 1.0
    forecast_log_rv = np.clip(forecast_log_rv, math.log(params["epsilon"]), max_log)
    forecast_vol = np.exp(forecast_log_rv)

    out: dict[int, float] = {}
    for horizon in horizons:
        h = int(horizon)
        vol = float(np.sqrt(np.mean(np.square(forecast_vol[:h]))))
        out[h] = max(vol, params["epsilon"])
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
        raise TypeError(f"{name} must be a pandas DataFrame for ARIMAVol")
    return value
