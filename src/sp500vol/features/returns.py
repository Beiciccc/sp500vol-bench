"""Log-return computation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series.

    Args:
        prices: indexed by trading day. Use adjusted close to handle splits/divs.

    Returns:
        Series of log returns. First element is NaN.
    """
    return np.log(prices / prices.shift(1))
