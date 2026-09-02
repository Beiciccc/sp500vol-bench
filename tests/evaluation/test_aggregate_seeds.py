"""Unit tests for the multi-seed aggregation helpers (scripts/analysis/aggregate_seeds.py).

Covers the pure functions that turn per-seed metrics + per-seed DM stats into the
cross-seed mean±std / nsig summary — the statistics that back the variance bands.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "aggregate_seeds.py"
    spec = importlib.util.spec_from_file_location("aggregate_seeds", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agg = _load_module()


def test_aggregate_values_basic():
    m, s, n = agg.aggregate_values([1.0, 2.0, 3.0])
    assert (m, n) == (2.0, 3)
    assert s == pytest.approx(1.0)  # sample std of 1,2,3


def test_aggregate_values_single_has_zero_std():
    assert agg.aggregate_values([5.0]) == (5.0, 0.0, 1)


def test_aggregate_values_drops_none_and_nan():
    m, _, n = agg.aggregate_values([1.0, None, float("nan"), 3.0])
    assert (m, n) == (2.0, 2)


def test_aggregate_values_empty_is_nan():
    m, s, n = agg.aggregate_values([])
    assert math.isnan(m) and math.isnan(s) and n == 0


def test_aggregate_dm_worse_all_significant():
    out = agg.aggregate_dm([(14.0, 0.001), (13.0, 0.002), (15.0, 0.0005)])
    assert out["mean"] == pytest.approx(14.0)
    assert out["n"] == 3 and out["n_sig"] == 3
    assert out["direction"] == "WORSE"
    assert out["std"] == pytest.approx(1.0)


def test_aggregate_dm_better_partial_significant():
    out = agg.aggregate_dm([(-2.4, 0.01), (-1.8, 0.07), (-2.1, 0.03)])
    assert out["mean"] == pytest.approx(-2.1)
    assert out["n"] == 3 and out["n_sig"] == 2  # 0.07 is not < 0.05
    assert out["direction"] == "BETTER"


def test_aggregate_dm_drops_nonfinite_and_handles_empty():
    out = agg.aggregate_dm([(float("inf"), 0.0), (12.0, 0.01)])
    assert out["n"] == 1 and out["mean"] == pytest.approx(12.0)
    empty = agg.aggregate_dm([])
    assert empty["n"] == 0 and empty["n_sig"] == 0 and math.isnan(empty["mean"])


def test_losses():
    y = np.array([0.2, 0.3])
    np.testing.assert_allclose(agg.se_loss(y, y), [0.0, 0.0])
    np.testing.assert_allclose(agg.qlike_loss(y, y), [0.0, 0.0], atol=1e-9)
    # QLIKE penalises under- and over-prediction positively
    assert (agg.qlike_loss(y, y * 2) > 0).all()
    assert (agg.qlike_loss(y, y * 0.5) > 0).all()
