"""Diebold-Mariano test sanity checks."""

from __future__ import annotations

import numpy as np

from sp500vol.evaluation.dm_test import dm_test


def test_equal_losses_give_zero_statistic() -> None:
    rng = np.random.default_rng(0)
    loss = rng.normal(size=200) ** 2
    stat, p = dm_test(loss, loss)
    assert abs(stat) < 1e-9
    assert p == 1.0 or np.isnan(p) or p > 0.99


def test_model_a_strictly_worse_yields_significant_p() -> None:
    rng = np.random.default_rng(1)
    n = 500
    loss_a = rng.normal(loc=1.0, scale=0.2, size=n) ** 2
    loss_b = rng.normal(loc=0.5, scale=0.2, size=n) ** 2  # B better
    _, p = dm_test(loss_a, loss_b)
    assert p < 0.01


def test_horizon_lag_changes_variance_estimate() -> None:
    rng = np.random.default_rng(2)
    loss_a = rng.normal(size=400) ** 2
    loss_b = rng.normal(size=400) ** 2
    _, p_h1 = dm_test(loss_a, loss_b, h=1)
    _, p_h20 = dm_test(loss_a, loss_b, h=20)
    # With more lag the HAC variance changes; p-values typically differ
    assert p_h1 != p_h20 or (np.isnan(p_h1) and np.isnan(p_h20))
