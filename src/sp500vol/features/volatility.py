"""Realised volatility label construction.

Convention: RV_H(t) = sqrt( (252/H) * sum_{i=t}^{t+H-1} r_i^2 )
where r_i are daily log returns. Annualised; sqrt-scaled so units are
comparable across horizons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALISATION_FACTOR = 252


def realised_volatility(
    log_returns: pd.Series,
    horizon: int,
) -> pd.Series:
    """Forward-looking annualised realised volatility.

    For each t, computes RV over the H-day window STARTING at t (inclusive).
    Returns are aligned by trading day; horizon is in trading days.

    Args:
        log_returns: daily log returns indexed by trading day.
        horizon: window length in trading days (e.g. 5, 10, 20).

    Returns:
        Series indexed by trading day. The last H-1 days have no full
        forward window and return NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    # Sum of squared returns over the FORWARD window [t, t+H-1]
    squared = log_returns**2
    # Reverse rolling: pandas rolling sums backward; we need forward.
    # Trick: reverse, rolling sum, reverse back.
    forward_sum = squared[::-1].rolling(window=horizon, min_periods=horizon).sum()[::-1]
    rv = np.sqrt(ANNUALISATION_FACTOR / horizon * forward_sum)
    return rv
