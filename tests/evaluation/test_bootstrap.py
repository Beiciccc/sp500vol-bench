"""Block bootstrap sanity checks."""

from __future__ import annotations

import numpy as np

from sp500vol.evaluation.bootstrap import block_bootstrap_ci
from sp500vol.evaluation.metrics import mae


def test_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(42)
    n = 400
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(scale=0.1, size=n)
    point, lo, hi = block_bootstrap_ci(y_true, y_pred, mae, n_iter=200, block_size=20, rng=rng)
    assert lo <= point <= hi


def test_ci_tightens_with_more_iterations() -> None:
    rng = np.random.default_rng(43)
    n = 400
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(scale=0.1, size=n)
    _, lo1, hi1 = block_bootstrap_ci(y_true, y_pred, mae, n_iter=50, block_size=20, rng=rng)
    _, lo2, hi2 = block_bootstrap_ci(y_true, y_pred, mae, n_iter=2000, block_size=20, rng=rng)
    # More iterations shouldn't dramatically expand the CI
    assert (hi2 - lo2) < (hi1 - lo1) * 1.5
