"""Evaluation metrics for volatility forecasts.

Reports MAE, RMSE, R², and QLIKE (Patton 2011) — the proper scoring rule
for variance forecasts. QLIKE is preferred for volatility because it is robust
to noise in the variance proxy.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def qlike(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    """Patton (2011) QLIKE loss for variance forecasts.

    QLIKE(σ²_true, σ²_pred) = σ²_true / σ²_pred − log(σ²_true / σ²_pred) − 1

    Args:
        y_true: TRUE variance (NOT volatility — square if needed).
        y_pred: PREDICTED variance.

    Returns:
        Mean QLIKE loss. Lower is better; 0 means perfect.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if (yt <= 0).any() or (yp <= 0).any():
        # Variance must be strictly positive
        yt = np.clip(yt, eps, None)
        yp = np.clip(yp, eps, None)
    ratio = yt / yp
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, R², and QLIKE.

    Note: y_true / y_pred should be VOLATILITY (sqrt of variance); QLIKE
    internally squares them. Adjust if you store variance directly.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "qlike": qlike(y_true**2, y_pred**2),
    }
