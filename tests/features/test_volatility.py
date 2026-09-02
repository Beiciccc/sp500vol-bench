"""Realised volatility computation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sp500vol.features.volatility import realised_volatility


def test_zero_returns_give_zero_vol() -> None:
    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    r = pd.Series(np.zeros(n), index=idx)
    rv = realised_volatility(r, horizon=5)
    valid = rv.dropna()
    assert (valid == 0).all()


def test_constant_volatility_window_sane() -> None:
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    r = pd.Series(rng.normal(scale=0.01, size=n), index=idx)
    rv5 = realised_volatility(r, horizon=5)
    rv20 = realised_volatility(r, horizon=20)
    # Annualised vol should be roughly sigma * sqrt(252) for IID returns
    expected = 0.01 * np.sqrt(252)
    assert abs(rv5.mean() - expected) < 0.05
    assert abs(rv20.mean() - expected) < 0.05


def test_last_rows_are_nan_when_forward_window_incomplete() -> None:
    n = 30
    horizon = 5
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    r = pd.Series(np.random.normal(size=n), index=idx)
    rv = realised_volatility(r, horizon=horizon)
    # Last (horizon-1) entries have no complete forward window
    assert rv.iloc[-(horizon - 1) :].isna().all()
