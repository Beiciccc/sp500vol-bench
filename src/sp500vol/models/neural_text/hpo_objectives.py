"""QLIKE-aligned training objective (pre-registered HPO arm, search dim `objective`).

The evaluation loss is QLIKE on variance units, QLIKE(y, f) = y/f - log(y/f) - 1.
Models emit z = log RV-hat (the repo-wide convention), so we parameterise
f = exp(z) and evaluate in log space:

    u = log y - z          (log accuracy ratio)
    QLIKE = exp(u) - u - 1

which is division-free, strictly convex in z, zero-minimised at z = log y, and
numerically safe once u is clamped (exp overflow guard; gradients stay bounded).
Labels are floored at EPS before the log. Gradient clipping at the optimiser
level is still recommended (the spec mandates clip_grad_norm).

`qlike_loss` is the torch training objective; `qlike_np` the numpy reference
used by the unit tests and any CPU-side audit.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8
U_CLAMP = 30.0  # exp(30) ~ 1e13: far beyond any sane log-ratio, prevents overflow


def qlike_np(log_y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Numpy reference: elementwise QLIKE in log parameterisation."""
    u = np.clip(np.asarray(log_y, float) - np.asarray(z, float), -U_CLAMP, U_CLAMP)
    return np.exp(u) - u - 1.0


def qlike_loss(z, log_y):
    """Torch QLIKE loss (mean). z: model output = log RV-hat; log_y: log label."""
    import torch
    u = torch.clamp(log_y - z, min=-U_CLAMP, max=U_CLAMP)
    return (torch.exp(u) - u - 1.0).mean()


def make_objective(name: str):
    """Return a torch loss fn(z, log_y) for the pre-registered objective grid.

    'mse'   — squared error on log RV (the fixed-recipe objective, unchanged).
    'qlike' — the aligned objective above.
    """
    import torch

    if name == "mse":
        return lambda z, log_y: torch.mean((z - log_y) ** 2)
    if name == "qlike":
        return qlike_loss
    raise ValueError(f"unknown objective '{name}' (pre-registered grid: mse, qlike)")
