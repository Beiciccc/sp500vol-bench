"""Diebold-Mariano (1995) test for equal predictive accuracy.

Used to compare each text-based model against the HAR-RV baseline.
Reports the two-sided p-value; significance markers in result tables:
  † p<0.05, ‡ p<0.01.

Reference:
  Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy.
  Journal of Business & Economic Statistics 13(3):253-263.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

MIN_OBSERVATIONS = 2


def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    h: int = 1,
) -> tuple[float, float]:
    """Diebold-Mariano test, two-sided.

    Args:
        loss_a: per-observation loss series for model A (e.g. squared error).
        loss_b: per-observation loss series for model B.
        h: forecast horizon. Used to set HAC lag = h - 1.

    Returns:
        (dm_statistic, p_value)
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = len(d)
    if n < MIN_OBSERVATIONS:
        raise ValueError("need at least 2 observations")

    mean_d = float(np.mean(d))
    lag = max(h - 1, 0)
    var_d = _hac_variance(d, lag=lag)
    if var_d <= 0:
        # Degenerate HAC variance (near-constant loss differential): the test is
        # undefined. mean_d ~ 0 with zero variance means the models are identical
        # (p=1). Otherwise return NaN — NOT a spurious p=0 that would falsely count
        # as significant in the nsig tally.
        if np.isclose(mean_d, 0.0):
            return 0.0, 1.0
        return float("nan"), float("nan")
    # Harvey-Leybourne-Newbold (1997) small-sample correction + Student-t(n-1)
    # reference (vs the asymptotic normal): scales the statistic down for short
    # overlapping forecast series and uses a heavier-tailed null, so a small n
    # cannot manufacture significance.
    hln_factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if hln_factor <= 0:
        return float("nan"), float("nan")
    dm_stat = (mean_d / np.sqrt(var_d / n)) * (hln_factor**0.5)
    p = 2 * float(stats.t.sf(abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p)


def _hac_variance(x: np.ndarray, *, lag: int) -> float:
    """Newey-West HAC long-run variance estimate."""
    n = len(x)
    x_centered = x - np.mean(x)
    gamma_0 = float(np.dot(x_centered, x_centered) / n)
    if lag == 0:
        return gamma_0
    out = gamma_0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = float(np.dot(x_centered[k:], x_centered[:-k]) / n)
        out += 2 * w * gamma_k
    return out
