"""Tests for the pinned chronological split convention in train.py.

Splits are assigned by the filing's EFFECTIVE TRADING DAY, day-inclusive on both
ends (see configs/base.yaml). This guards the 2019-12-31 -> 2020-01-01 boundary
that previously dropped 6 rows into 'unused' under the raw-UTC-timestamp rule.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd


def _load_train_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_for_tests", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = _load_train_module()


def _row(etd: str) -> dict:
    return {
        "effective_trading_day": pd.Timestamp(etd),
        "label_realised_vol": 0.2,
        "horizon_days": 5,
        "accession": etd,
    }


def test_assign_splits_is_day_inclusive_on_boundaries() -> None:
    data = pd.DataFrame(
        [
            _row("2010-01-01"),  # train start day -> train
            _row("2019-12-31"),  # train END day -> train (day-inclusive)
            _row("2020-01-01"),  # val start day -> val
            _row("2021-12-31"),  # val END day -> val (day-inclusive)
            _row("2022-01-03"),  # test start -> test
            _row("2025-12-23"),  # test -> test
        ]
    )
    out = train._assign_splits(data, "full")
    by_day = out.set_index("effective_trading_day")["split"].to_dict()

    assert by_day[pd.Timestamp("2010-01-01")] == "train"
    assert by_day[pd.Timestamp("2019-12-31")] == "train"  # inclusive end boundary
    assert by_day[pd.Timestamp("2020-01-01")] == "val"
    assert by_day[pd.Timestamp("2021-12-31")] == "val"  # inclusive end boundary
    assert by_day[pd.Timestamp("2022-01-03")] == "test"
    assert by_day[pd.Timestamp("2025-12-23")] == "test"
    assert len(out) == 6  # nothing dropped to 'unused'


def test_assign_splits_keeps_boundary_day_filing_out_of_unused() -> None:
    # A filing on the train-end ET day must land in train, not the unused gap.
    out = train._assign_splits(pd.DataFrame([_row("2019-12-31")]), "full")
    assert out["split"].tolist() == ["train"]
    assert (out["split"] == "unused").sum() == 0
