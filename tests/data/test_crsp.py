"""Tests for CRSP readers: point-in-time CIK linking + daily returns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sp500vol.data.crsp import load_cik_links, load_crsp_returns, resolve_cik


def _links(tmp_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            # Baker Hughes M&A succession: two disjoint CIKs split by date window.
            {
                "permno": 75034,
                "cik": "0000808362",
                "link_start": pd.Timestamp("1987-01-01"),
                "link_end": pd.Timestamp("2017-07-04"),
            },
            {
                "permno": 75034,
                "cik": "0001701605",
                "link_start": pd.Timestamp("2017-07-05"),
                "link_end": pd.NaT,
            },
            # DPS override (open-ended).
            {
                "permno": 92618,
                "cik": "0001418135",
                "link_start": pd.Timestamp("2008-05-07"),
                "link_end": pd.NaT,
            },
            {
                "permno": 76226,
                "cik": "0000813828",
                "link_start": pd.Timestamp("1994-10-01"),
                "link_end": pd.Timestamp("2025-08-06"),
            },
            {
                "permno": 69550,
                "cik": "0000069499",
                "link_start": pd.Timestamp("1973-02-23"),
                "link_end": pd.Timestamp("2015-02-26"),
            },
            {
                "permno": 69550,
                "cik": "0001623613",
                "link_start": pd.Timestamp("2015-02-27"),
                "link_end": pd.Timestamp("2020-11-16"),
            },
            {
                "permno": 27983,
                "cik": "0000108772",
                "link_start": pd.Timestamp("1962-01-31"),
                "link_end": pd.Timestamp("2019-07-31"),
            },
            {
                "permno": 27983,
                "cik": "0001770450",
                "link_start": pd.Timestamp("2019-08-01"),
                "link_end": pd.NaT,
            },
            {
                "permno": 70500,
                "cik": "0000804055",
                "link_start": pd.Timestamp("1986-11-21"),
                "link_end": pd.Timestamp("2010-10-01"),
            },
            {
                "permno": 70500,
                "cik": "0001491675",
                "link_start": pd.Timestamp("2010-10-02"),
                "link_end": pd.Timestamp("2016-05-27"),
            },
            {
                "permno": 20626,
                "cik": "0000029915",
                "link_start": pd.Timestamp("1950-06-01"),
                "link_end": pd.Timestamp("2017-08-31"),
            },
        ]
    )
    path = tmp_path / "crsp_cik_links.parquet"
    df.to_parquet(path, index=False)
    return load_cik_links(path)


def test_resolve_cik_point_in_time_succession(tmp_path: Path) -> None:
    links = _links(tmp_path)
    # Before the merger boundary -> Baker Hughes Inc; after -> Baker Hughes Co.
    assert resolve_cik(75034, pd.Timestamp("2015-01-01"), links) == "0000808362"
    assert resolve_cik(75034, pd.Timestamp("2020-01-01"), links) == "0001701605"


def test_resolve_cik_dps_override(tmp_path: Path) -> None:
    links = _links(tmp_path)
    assert resolve_cik(92618, pd.Timestamp("2015-01-01"), links) == "0001418135"


def test_resolve_cik_manual_reorg_overrides(tmp_path: Path) -> None:
    links = _links(tmp_path)
    assert resolve_cik(76226, pd.Timestamp("2019-12-04"), links) == "0000813828"
    assert resolve_cik(69550, pd.Timestamp("2014-12-31"), links) == "0000069499"
    assert resolve_cik(69550, pd.Timestamp("2015-03-02"), links) == "0001623613"
    assert resolve_cik(27983, pd.Timestamp("2019-07-31"), links) == "0000108772"
    assert resolve_cik(27983, pd.Timestamp("2019-08-01"), links) == "0001770450"
    assert resolve_cik(70500, pd.Timestamp("2010-10-01"), links) == "0000804055"
    assert resolve_cik(70500, pd.Timestamp("2010-10-04"), links) == "0001491675"
    assert resolve_cik(20626, pd.Timestamp("2017-08-31"), links) == "0000029915"


def test_resolve_cik_open_link_covers_future(tmp_path: Path) -> None:
    links = _links(tmp_path)
    # NaT link_end (open link, CCM 'E' sentinel) covers any later date.
    assert resolve_cik(92618, pd.Timestamp("2025-12-31"), links) == "0001418135"


def test_resolve_cik_none_outside_window_or_unknown(tmp_path: Path) -> None:
    links = _links(tmp_path)
    assert resolve_cik(75034, pd.Timestamp("1900-01-01"), links) is None
    assert resolve_cik(99999, pd.Timestamp("2015-01-01"), links) is None


def test_load_cik_links_normalizes_cik_and_permno(tmp_path: Path) -> None:
    path = tmp_path / "links.parquet"
    pd.DataFrame(
        [{"permno": 10107, "cik": "789019", "link_start": "1986-03-13", "link_end": None}]
    ).to_parquet(path, index=False)
    links = load_cik_links(path)
    assert links["cik"].iloc[0] == "0000789019"  # zero-padded to 10
    assert links["permno"].dtype.kind == "i"


def test_load_crsp_returns_schema_and_upper(tmp_path: Path) -> None:
    path = tmp_path / "market_returns.parquet"
    pd.DataFrame(
        {"ticker": ["aapl"], "date": [pd.Timestamp("2020-01-02")], "log_return": [0.0123]}
    ).to_parquet(path, index=False)
    out = load_crsp_returns(path)
    assert list(out.columns) == ["ticker", "date", "log_return"]
    assert out["ticker"].tolist() == ["AAPL"]
