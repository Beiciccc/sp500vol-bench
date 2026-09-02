"""Block bootstrap confidence intervals for forecast metrics.

For time-series data, IID bootstrap underestimates variance. We use a
non-overlapping block bootstrap with block size equal to the forecast horizon
(or larger), which preserves short-range dependence.

Reference:
  Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap.
  Journal of the American Statistical Association 89(428):1303-1313.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_iter: int = 1000,
    block_size: int = 20,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower, upper) at (1-alpha) coverage.

    Args:
        y_true / y_pred: equal-length arrays of true labels and predictions.
        metric_fn: e.g. sp500vol.evaluation.metrics.qlike (takes y_true, y_pred).
        n_iter: number of bootstrap samples.
        block_size: contiguous block length. Default 20 matches the 20-day horizon.
        alpha: 1 - coverage. 0.05 → 95% CI.
        rng: optional numpy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("length mismatch")
    if block_size > n:
        raise ValueError("block_size > sample size")

    point = metric_fn(y_true, y_pred)
    n_blocks = n // block_size
    if n_blocks * block_size != n:
        # Trim to exact multiple
        y_true = y_true[: n_blocks * block_size]
        y_pred = y_pred[: n_blocks * block_size]

    boot = np.empty(n_iter)
    for i in range(n_iter):
        idx_blocks = rng.integers(0, n_blocks, size=n_blocks)
        sample_idx = np.concatenate(
            [np.arange(b * block_size, (b + 1) * block_size) for b in idx_blocks]
        )
        boot[i] = metric_fn(y_true[sample_idx], y_pred[sample_idx])

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return float(point), lo, hi
