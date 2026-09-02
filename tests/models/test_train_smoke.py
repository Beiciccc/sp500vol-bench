"""Unit tests for train.py's smoke helpers — the tiny-subset + epoch-cap logic
behind `--smoke`. Verifies the subset still covers every (split, horizon) so a
smoke run exercises all code paths, and the cap never *raises* epochs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_train():
    path = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load_train()


def _df(per_group: int = 50) -> pd.DataFrame:
    rows = [
        {"split": split, "horizon_days": hz, "x": i}
        for split in ("train", "val", "test")
        for hz in (5, 10, 20)
        for i in range(per_group)
    ]
    return pd.DataFrame(rows)


def test_smoke_subset_covers_every_split_horizon():
    out = tr._smoke_subset(_df(), 8)
    sizes = out.groupby(["split", "horizon_days"]).size()
    assert len(sizes) == 9  # 3 splits x 3 horizons, none dropped
    assert (sizes == 8).all()  # exactly the cap per group


def test_smoke_subset_head_does_not_pad():
    df = pd.DataFrame(
        [{"split": "train", "horizon_days": 5, "x": i} for i in range(3)]
    )  # only 3 rows < cap
    assert len(tr._smoke_subset(df, 100)) == 3


def test_cap_smoke_epochs_caps():
    cfg = {"training": {"max_epochs": 15}}
    tr._cap_smoke_epochs(cfg, 2)
    assert cfg["training"]["max_epochs"] == 2


def test_cap_smoke_epochs_never_raises():
    cfg = {"training": {"max_epochs": 1}}
    tr._cap_smoke_epochs(cfg, 2)
    assert cfg["training"]["max_epochs"] == 1  # min(2, 1) — never lifts above config


def test_cap_smoke_epochs_also_caps_patience():
    # so the smoke can actually early-stop within its 2 epochs and exercise that branch
    cfg = {"training": {"max_epochs": 15, "es_patience": 3}}
    tr._cap_smoke_epochs(cfg, 2)
    assert cfg["training"]["max_epochs"] == 2
    assert cfg["training"]["es_patience"] == 1


def test_cap_smoke_epochs_no_training_key_is_noop():
    cfg = {"vectoriser": {"max_features": 5000}}
    tr._cap_smoke_epochs(cfg, 2)  # classical model: no training block
    assert "training" not in cfg
