"""Metric correctness tests."""

from __future__ import annotations

import numpy as np

from sp500vol.evaluation.metrics import all_metrics, mae, qlike, r_squared, rmse


def test_mae_zero_when_perfect() -> None:
    y = np.array([0.1, 0.2, 0.3])
    assert mae(y, y) == 0.0


def test_rmse_zero_when_perfect() -> None:
    y = np.array([0.1, 0.2, 0.3])
    assert rmse(y, y) == 0.0


def test_r2_one_when_perfect() -> None:
    y = np.array([0.1, 0.2, 0.3, 0.4])
    assert r_squared(y, y) == 1.0


def test_qlike_zero_when_perfect() -> None:
    y = np.array([0.04, 0.09, 0.16])  # variances
    assert qlike(y, y) == 0.0


def test_qlike_positive_for_mispredictions() -> None:
    y_true = np.array([0.04, 0.09, 0.16])
    y_pred = np.array([0.05, 0.07, 0.20])
    assert qlike(y_true, y_pred) > 0


def test_all_metrics_returns_expected_keys() -> None:
    y_true = np.array([0.10, 0.15, 0.20, 0.25])
    y_pred = np.array([0.11, 0.14, 0.22, 0.24])
    out = all_metrics(y_true, y_pred)
    assert set(out.keys()) == {"mae", "rmse", "r2", "qlike"}
