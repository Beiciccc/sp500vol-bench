"""Day-clustered Diebold-Mariano inference (canonical implementation).

Reviewer-verified defect being fixed: the original DM tests run HAC(lag=h-1) over
OBSERVATION order, but ~10-25 same-day filings share the same market shocks, so the
effective sample size is far smaller than n_obs and t-stats are inflated ~2x.

Fix (exact spec, shared across all remediation tasks):
  1. per-observation loss differential d_i = lossA_i - lossB_i;
  2. group by CALENDAR DAY (effective_trading_day; fallback filing_time_utc date)
     -> daily mean differential d_t over the days present, sorted by day;
  3. run dm_test on the daily-mean lossA series vs daily-mean lossB series
     (equal weight per day), with h = label horizon in TRADING DAYS (5/10/20) so
     HAC lag = h-1 now counts DAYS of genuine label overlap; n = number of days;
  4. day-block moving bootstrap: blocks of h consecutive DAYS on the daily
     differential.

Import from anywhere as:
    sys.path.insert(0, "scripts/analysis"); from clustered_dm import ...
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sp500vol.evaluation.dm_test import dm_test  # noqa: E402

__all__ = ["daily_mean", "dm_test_clustered", "mbb_ci_daily"]


def _day_index(days):
    """Normalise a day-key array to sortable calendar-day values."""
    d = pd.Series(np.asarray(days))
    if not np.issubdtype(d.dtype, np.datetime64):
        d = pd.to_datetime(d, utc=True, errors="coerce")
    # calendar day only (drops intraday time from filing_time_utc fallback)
    return d.dt.normalize().to_numpy()


def daily_mean(x, days):
    """Group x by calendar day; return (daily means sorted by day, sorted unique days)."""
    x = np.asarray(x, dtype=np.float64)
    day = _day_index(days)
    df = pd.DataFrame({"day": day, "x": x})
    g = df.groupby("day", sort=True)["x"].mean()
    return g.to_numpy(), g.index.to_numpy()


def dm_test_clustered(lossA, lossB, days, h):
    """Day-clustered DM test.

    Args:
        lossA, lossB: per-observation loss series (same length/order).
        days: per-observation calendar-day key (effective_trading_day, or
              filing_time_utc date as fallback).
        h: label horizon in TRADING DAYS (5/10/20) -> HAC lag = h-1 on the
           daily series.

    Returns:
        (dm_stat, p_value, n_days). Positive stat = lossA (first arg) WORSE.
    """
    lossA = np.asarray(lossA, dtype=np.float64)
    lossB = np.asarray(lossB, dtype=np.float64)
    if len(lossA) != len(lossB):
        raise ValueError("lossA and lossB must have equal length")
    dA, _ = daily_mean(lossA, days)
    dB, _ = daily_mean(lossB, days)
    stat, p = dm_test(dA, dB, h=int(h))
    return float(stat), float(p), int(len(dA))


def mbb_ci_daily(d, days, h, *, B=2000, seed=2026, alpha=0.05):
    """Moving-block bootstrap CI for the mean loss differential, on the DAILY series.

    Blocks of h consecutive DAYS on the daily mean differential.

    Args:
        d: per-observation loss differential (lossA - lossB).
        days: per-observation calendar-day key.
        h: horizon in trading days -> block length in days.

    Returns:
        (mean_daily_d, ci_lo, ci_hi). NaN CI when there are < 2*h days.
    """
    dd, _ = daily_mean(d, days)
    n = len(dd)
    L = max(int(h), 1)
    if n < 2 * L:
        return float(np.mean(dd)) if n else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / L))
    means = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n, size=nb)
        idx = (starts[:, None] + np.arange(L)[None, :]) % n
        means[b] = dd[idx.ravel()[:n]].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(dd)), float(lo), float(hi)
