"""Safety checks around dataset-build market inputs and outputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _load_build_dataset_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("build_dataset_for_tests", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_dataset = _load_build_dataset_module()


def test_unique_tickers_deduplicates_time_varying_membership_rows() -> None:
    universe = pd.DataFrame(
        {
            "ticker": ["aapl", "MSFT", " AAPL ", "msft", None, ""],
            "cik": ["1", "2", "1", "2", "3", "4"],
        }
    )

    assert build_dataset._unique_tickers(universe) == ["AAPL", "MSFT"]


def test_resolve_universe_supports_sp500_membership_source(tmp_path: Path) -> None:
    membership_path = tmp_path / "sp500_membership.parquet"
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "member_from": "2019-01-01",
                "member_to": None,
                "source": "fixture",
            },
            {
                "ticker": "MSFT",
                "cik": "0000789019",
                "member_from": "2026-01-01",
                "member_to": None,
                "source": "fixture",
            },
        ]
    ).to_parquet(membership_path, index=False)

    universe = build_dataset._resolve_universe(
        {
            "date_range": {"start": "2010-01-01", "end": "2025-12-31"},
            "universe": {
                "source": "time_varying_sp500",
                "membership_table": str(membership_path),
            },
        }
    )

    assert universe["ticker"].tolist() == ["AAPL"]


def test_fetch_market_data_passes_deduped_tickers_to_fetcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fetch_ohlcv(
        tickers: list[str],
        start: str,
        end: str,
        cache_path: Path | None = None,
        timeout: float | None = None,
        *,
        crsp_store: Path | None = None,
    ) -> pd.DataFrame:
        captured["tickers"] = tickers
        captured["start"] = start
        captured["end"] = end
        captured["cache_path"] = cache_path
        captured["crsp_store"] = crsp_store
        return pd.DataFrame(
            {
                "ticker": tickers,
                "date": [pd.Timestamp("2020-01-02")] * len(tickers),
                "open": [1.0] * len(tickers),
                "high": [1.0] * len(tickers),
                "low": [1.0] * len(tickers),
                "close": [1.0] * len(tickers),
                "adj_close": [1.0] * len(tickers),
                "volume": [100] * len(tickers),
            }
        )

    monkeypatch.setattr(build_dataset, "fetch_ohlcv", fake_fetch_ohlcv)
    cfg = {
        "name": "unit",
        "date_range": {"start": "2020-01-01", "end": "2020-01-31"},
        "market": {
            "source": "crsp",
            "cache_root": str(tmp_path / "market"),
            "crsp_store": str(tmp_path / "crsp_daily.parquet"),
        },
    }

    build_dataset._fetch_market_data(cfg, ["AAPL", "MSFT", "aapl", " MSFT "])

    assert captured["tickers"] == ["AAPL", "MSFT"]
    assert captured["crsp_store"] is not None


def test_filing_date_filter_requires_matching_cik_window() -> None:
    universe = build_dataset._normalise_universe_for_filing_filter(
        pd.DataFrame(
            [
                {
                    "ticker": "MYL",
                    "cik": "0000069499",
                    "member_from": "2010-01-04",
                    "member_to": "2015-02-26",
                },
                {
                    "ticker": "MYL",
                    "cik": "0001623613",
                    "member_from": "2015-02-27",
                    "member_to": "2020-11-16",
                },
            ]
        )
    )

    assert build_dataset._is_in_universe_on_filing_date(
        "MYL", "0000069499", pd.Timestamp("2014-12-31"), universe
    )
    assert not build_dataset._is_in_universe_on_filing_date(
        "MYL", "0001623613", pd.Timestamp("2014-12-31"), universe
    )
    assert build_dataset._is_in_universe_on_filing_date(
        "MYL", "0001623613", pd.Timestamp("2016-03-01"), universe
    )
    assert not build_dataset._is_in_universe_on_filing_date(
        "MYL", "0000069499", pd.Timestamp("2016-03-01"), universe
    )


def test_assert_no_duplicate_ticker_dates_rejects_duplicate_keys() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "aapl"],
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02 12:00")],
        }
    )

    with pytest.raises(ValueError, match="market_returns has duplicate ticker/date rows"):
        build_dataset._assert_no_duplicate_ticker_dates(frame, name="market_returns")


def test_compute_market_returns_rejects_duplicate_ohlcv_keys() -> None:
    ohlcv = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02")],
            "adj_close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="ohlcv has duplicate ticker/date rows"):
        build_dataset._compute_market_returns(ohlcv)


def test_market_source_meta_records_crsp_source() -> None:
    meta = build_dataset._market_source_meta(
        {
            "market": {
                "source": "crsp",
                "crsp_store": "./data/market/crsp/daily_returns.parquet",
                "returns_store": "./data/market/crsp/market_returns.parquet",
            }
        }
    )

    assert meta == {
        "source": "crsp",
        "crsp_store": "./data/market/crsp/daily_returns.parquet",
        "returns_store": "./data/market/crsp/market_returns.parquet",
    }


class _NoopLog:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None


def test_filter_marketless_single_day_filings_drops_spin_off_rows() -> None:
    filings = pd.DataFrame(
        [
            {
                "ticker": "ASIX",
                "cik": "0001673985",
                "form": "8-K",
                "item_subtype": None,
                "filing_time_utc": pd.Timestamp("2016-10-03", tz="UTC"),
                "text_path": "asix.txt",
            },
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "form": "10-Q",
                "item_subtype": None,
                "filing_time_utc": pd.Timestamp("2016-10-03", tz="UTC"),
                "text_path": "aapl.txt",
            },
        ]
    )
    market_returns = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "date": [pd.Timestamp("2016-10-03")],
            "log_return": [0.01],
        }
    )
    universe = pd.DataFrame(
        [
            {
                "ticker": "ASIX",
                "member_from": pd.Timestamp("2016-10-03"),
                "member_to": pd.Timestamp("2016-10-03"),
            },
            {
                "ticker": "AAPL",
                "member_from": pd.Timestamp("2010-01-01"),
                "member_to": pd.Timestamp("2025-12-31"),
            },
        ]
    )

    out = build_dataset._filter_marketless_single_day_filings(
        filings,
        market_returns,
        universe,
        log=_NoopLog(),
    )

    assert out["ticker"].tolist() == ["AAPL"]


def test_filter_marketless_single_day_filings_rejects_real_market_gap() -> None:
    filings = pd.DataFrame(
        [
            {
                "ticker": "MISSING",
                "cik": "0000000001",
                "form": "10-Q",
                "item_subtype": None,
                "filing_time_utc": pd.Timestamp("2020-01-02", tz="UTC"),
                "text_path": "missing.txt",
            }
        ]
    )
    market_returns = pd.DataFrame(
        columns=["ticker", "date", "log_return"],
    )
    universe = pd.DataFrame(
        [
            {
                "ticker": "MISSING",
                "member_from": pd.Timestamp("2020-01-01"),
                "member_to": pd.Timestamp("2020-01-31"),
            }
        ]
    )

    with pytest.raises(ValueError, match="missing market data for non-spin-off tickers"):
        build_dataset._filter_marketless_single_day_filings(
            filings,
            market_returns,
            universe,
            log=_NoopLog(),
        )
