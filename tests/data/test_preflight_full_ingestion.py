"""Tests for full-ingestion preflight coverage accounting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd


def _load_preflight_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "preflight_full_ingestion.py"
    spec = importlib.util.spec_from_file_location("preflight_full_ingestion_for_tests", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load_preflight_module()


def test_ticker_intervals_preserves_non_contiguous_membership_rows() -> None:
    membership = pd.DataFrame(
        [
            {
                "ticker": "DELL",
                "cik": "0001571996",
                "member_from": pd.Timestamp("2010-01-04"),
                "member_to": pd.Timestamp("2013-10-28"),
            },
            {
                "ticker": "DELL",
                "cik": "0001571996",
                "member_from": pd.Timestamp("2024-09-23"),
                "member_to": pd.NaT,
            },
        ]
    )

    intervals = preflight._ticker_intervals(
        membership,
        pd.Timestamp("2010-01-01"),
        pd.Timestamp("2025-12-31"),
    )

    assert [
        (i.member_from.date().isoformat(), i.member_to.date().isoformat()) for i in intervals
    ] == [
        ("2010-01-04", "2013-10-28"),
        ("2024-09-23", "2025-12-31"),
    ]


def test_render_report_distinguishes_tickers_pairs_and_intervals() -> None:
    overlapping = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "cik": "0000000001",
                "member_from": pd.Timestamp("2020-01-01"),
                "member_to": pd.Timestamp("2020-01-31"),
            },
            {
                "ticker": "AAA",
                "cik": "0000000002",
                "member_from": pd.Timestamp("2020-02-01"),
                "member_to": pd.Timestamp("2020-02-28"),
            },
        ]
    )
    intervals = preflight._ticker_intervals(
        overlapping,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-12-31"),
    )

    report = preflight._render_report(
        cfg={"date_range": {"start": "2020-01-01", "end": "2020-12-31"}},
        config_path=Path("configs/data/full.yaml"),
        membership_path=Path("data/universe/sp500_membership.parquet"),
        raw_membership=overlapping,
        overlapping=overlapping,
        intervals=intervals,
        active_sizes=[],
        availability_df=pd.DataFrame(),
        failed_or_partial=pd.DataFrame(),
        risky=pd.DataFrame(),
        generated_at=pd.Timestamp("2026-06-06", tz="UTC").to_pydatetime(),
        market_skipped=False,
    )

    assert "| unique_ticker_count | 1 |" in report
    assert "| unique_ticker_cik_count | 2 |" in report
    assert "| membership_interval_count | 2 |" in report
    assert "| checked_interval_count | 2 |" in report
