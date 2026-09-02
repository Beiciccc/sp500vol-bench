"""Parallel per-(ticker, feature_window_end) refit driver for the econometric
price baselines (GARCH / EGARCH / ARIMA).

Each baseline refits a fresh model on the return history up to every distinct
filing's feature-window end (no look-ahead). Those refits are independent across
(ticker, date), so we fan them out BY TICKER with joblib: one worker owns a
ticker's full return series and runs all of its refits, then results are mapped
back to the original row order. This preserves the exact serial semantics
(deterministic for analytic forecasts) while using all cores — turning the
~128k single-core refits per model from hours into minutes.

A model supplies a module-level `refit_one(series, feature_end, horizons, params)`
returning ``{horizon: forecast_vol}``; it must be picklable (top-level function).
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

RefitOne = Callable[[pd.Series, pd.Timestamp, list[int], dict], dict[int, float]]


def _ticker_batch(
    refit_one: RefitOne,
    params: dict,
    ticker: str,
    dates: np.ndarray,
    values: np.ndarray,
    requests: list[tuple[pd.Timestamp, list[int]]],
) -> tuple[str, dict[tuple[pd.Timestamp, int], float]]:
    """Run all of one ticker's refits in a worker process."""
    series = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
    out: dict[tuple[pd.Timestamp, int], float] = {}
    for feature_end, horizons in requests:
        vols = refit_one(series, feature_end, horizons, params)
        for h in horizons:
            out[(feature_end, h)] = float(vols.get(h, np.nan))
    return ticker, out


def parallel_refit_predict(
    df: pd.DataFrame,
    returns_by_ticker: dict[str, pd.Series],
    refit_one: RefitOne,
    params: dict,
    *,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Predict one row per ``df`` index via parallel per-(ticker, date) refits.

    Returns a float array aligned to ``df`` row order; rows whose ticker has no
    return series, or whose refit fails, come back as NaN for the caller to
    fall back on (matching the serial implementations).
    """
    work: list[tuple] = []
    tickers = df["ticker"].astype(str).to_numpy()
    fwe = pd.to_datetime(df["feature_window_end"]).dt.tz_localize(None).dt.normalize().to_numpy()
    horizons_col = df["horizon_days"].astype(int).to_numpy()

    keyframe = pd.DataFrame({"ticker": tickers, "fwe": fwe, "horizon": horizons_col})
    for ticker, grp in keyframe.groupby("ticker", sort=False):
        series = returns_by_ticker.get(str(ticker))
        if series is None or series.empty:
            continue
        requests = [
            (pd.Timestamp(fe), sorted({int(h) for h in hg["horizon"]}))
            for fe, hg in grp.groupby("fwe")
        ]
        work.append((str(ticker), series.index.to_numpy(), series.to_numpy(dtype=float), requests))

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_ticker_batch)(refit_one, params, *item) for item in work
    )

    lookup: dict[tuple[str, pd.Timestamp, int], float] = {}
    for ticker, out in results:
        for (fe, h), vol in out.items():
            lookup[(ticker, pd.Timestamp(fe), h)] = vol

    preds = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        preds[i] = lookup.get((str(tickers[i]), pd.Timestamp(fwe[i]), int(horizons_col[i])), np.nan)
    return preds
