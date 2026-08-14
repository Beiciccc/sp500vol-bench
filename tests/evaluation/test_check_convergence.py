"""Unit tests for the convergence gate (scripts/analysis/check_convergence.py).

Covers classify_horizon — the pure function that turns a val-curve shape into a
CONVERGED / UNDERFIT_HIT_CAP / DIVERGED verdict — plus classify_run end-to-end
on a synthetic run dir.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "check_convergence.py"
    spec = importlib.util.spec_from_file_location("check_convergence", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load()


def _curve(vals, *, best_idx=None):
    """Build epoch records from val_loss list; is_best marks the running min unless overridden."""
    recs = []
    running_min = float("inf")
    for i, v in enumerate(vals, start=1):
        is_best = (best_idx == i) if best_idx is not None else (v is not None and v < running_min)
        if v is not None and v < running_min:
            running_min = v
        recs.append(
            {"epoch": i, "train_loss": 1.0 / i, "val_loss": v, "val_r2": -1.0, "is_best": is_best}
        )
    return recs


def test_early_stop_is_converged():
    # val improves to epoch 4 then worsens; early-stopped at 7 < cap 15
    recs = _curve([1.0, 0.7, 0.5, 0.4, 0.45, 0.46, 0.47])
    verdict, d = cc.classify_horizon(recs, max_epochs=15)
    assert verdict == "CONVERGED"
    assert d["min_val_epoch"] == 4 and d["max_seen"] == 7


def test_stopped_under_cap_is_converged():
    recs = _curve([1.0, 0.6, 0.55])  # only 3 epochs, cap 15 -> stopped early
    assert cc.classify_horizon(recs, max_epochs=15)[0] == "CONVERGED"


def test_hit_cap_still_improving_is_underfit():
    # every epoch is the best, last == cap -> still descending when cut off
    recs = _curve([1.0 - 0.05 * i for i in range(15)])
    verdict, d = cc.classify_horizon(recs, max_epochs=15)
    assert verdict == "UNDERFIT_HIT_CAP"
    assert d["min_val_epoch"] == 15 and d["max_seen"] == 15


def test_hit_cap_but_plateaued_is_converged():
    # reached cap but best was epoch 6 (last epochs flat, no improvement) -> converged
    vals = [1.0, 0.8, 0.6, 0.5, 0.45, 0.42] + [0.43] * 9
    recs = _curve(vals, best_idx=6)
    assert cc.classify_horizon(recs, max_epochs=15)[0] == "CONVERGED"


def test_nan_train_loss_is_diverged():
    recs = _curve([1.0, 0.7, 0.5])
    recs[2]["train_loss"] = None
    out = cc.classify_horizon(recs, max_epochs=15)
    assert out[0] == "DIVERGED" and out[1]["reason"] == "nan_train_loss"


def test_nan_val_after_finite_is_diverged():
    recs = _curve([1.0, 0.7, None])
    out = cc.classify_horizon(recs, max_epochs=15)
    assert out[0] == "DIVERGED" and out[1]["reason"] == "nan_val_loss"


def test_val_rebound_is_diverged():
    # min 0.4; last two epochs 1.3, 1.5 both > 0.4*3=1.2 -> several-times-min blow-up
    recs = _curve([1.0, 0.4, 1.3, 1.5], best_idx=2)
    out = cc.classify_horizon(recs, max_epochs=15)
    assert out[0] == "DIVERGED" and out[1]["reason"] == "val_rebound"


def test_single_point_spike_not_diverged():
    # one bad final point (1.5) but the prior epoch (0.42) is fine -> not persistent
    recs = _curve([1.0, 0.4, 0.42, 1.5], best_idx=2)
    assert cc.classify_horizon(recs, max_epochs=15)[0] == "CONVERGED"


def test_small_rebound_not_flagged():
    # final 0.45 vs min 0.4 -> within tolerance, early-stopped converged
    recs = _curve([1.0, 0.4, 0.42, 0.45], best_idx=2)
    assert cc.classify_horizon(recs, max_epochs=15)[0] == "CONVERGED"


def test_patience_tail_rebound_converged():
    # Real-data pattern: val reaches a healthy min then rises ~1.9x over the early-stop
    # patience tail (best weights restored). This is CONVERGED, NOT diverged — the
    # regression that previously mis-flagged it (1.5x trip point) is fixed at 3x.
    recs = _curve([1.0, 0.5, 0.4, 0.7, 0.76], best_idx=3)  # last two 1.75x / 1.9x of min
    assert cc.classify_horizon(recs, max_epochs=15)[0] == "CONVERGED"


def test_no_val_signal_is_suspect():
    # fixed-epoch run with no validation: every is_best=True, all val_loss None
    recs = [
        {"epoch": i, "train_loss": 1.0 / i, "val_loss": None, "val_r2": None, "is_best": True}
        for i in range(1, 16)
    ]
    verdict, d = cc.classify_horizon(recs, max_epochs=15)
    assert verdict == "SUSPECT" and d["reason"] == "no_val_signal"


def test_empty_is_no_data():
    assert cc.classify_horizon([], max_epochs=15)[0] == "NO_DATA"


def test_classify_run_worst_horizon(tmp_path):
    run = tmp_path / "C4_longformer_full_long_form_seed2026"
    run.mkdir()
    curves = {
        "5": _curve([1.0, 0.6, 0.5, 0.45, 0.46, 0.47]),          # CONVERGED
        "10": _curve([1.0 - 0.05 * i for i in range(15)]),        # UNDERFIT_HIT_CAP
        "20": _curve([1.0, 0.4, 1.3, 1.6], best_idx=2),           # DIVERGED (>=3x persistent rebound)
    }
    (run / "val_curves.json").write_text(json.dumps(curves), encoding="utf-8")
    (run / "config.json").write_text(
        json.dumps({"model_config": {"training": {"max_epochs": 15}}}), encoding="utf-8"
    )
    res = cc.classify_run(run)
    assert res["verdict"] == "DIVERGED"  # worst of the three horizons
    assert res["max_epochs"] == 15


def test_classify_run_none_without_curve(tmp_path):
    run = tmp_path / "A2_har_rv_full_long_form_seed2026"
    run.mkdir()
    assert cc.classify_run(run) is None  # non-neural, nothing to gate


def test_classify_run_neural_missing_curve_is_suspect(tmp_path):
    # neural config (has training.max_epochs) but NO val_curves.json — e.g. a fully
    # checkpoint-resumed run. Must be SUSPECT, never silently CONVERGED.
    run = tmp_path / "C2_finbert_s1_full_long_form_seed2026"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps({"model_config": {"training": {"max_epochs": 15}}}), encoding="utf-8"
    )
    res = cc.classify_run(run)
    assert res is not None and res["verdict"] == "SUSPECT"


def test_classify_run_missing_horizon_is_suspect(tmp_path):
    # curve present but horizon 20 absent (resume gap) -> SUSPECT
    run = tmp_path / "C2_finbert_s1_full_long_form_seed2026"
    run.mkdir()
    curves = {"5": _curve([1.0, 0.6, 0.5, 0.45]), "10": _curve([1.0, 0.6, 0.5, 0.45])}
    (run / "val_curves.json").write_text(json.dumps(curves), encoding="utf-8")
    (run / "config.json").write_text(
        json.dumps({"model_config": {"training": {"max_epochs": 15}}}), encoding="utf-8"
    )
    assert cc.classify_run(run)["verdict"] == "SUSPECT"
