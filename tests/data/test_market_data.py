"""Tests for the CRSP-backed market-data reader.

``fetch_ohlcv`` reads the CRSP OHLCV store (built by ``scripts/ingest_wrds.py``),
filters to the requested tickers and date range, and returns the locked schema.
There is no network access and no yfinance/Tiingo fallback.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sp500vol.data import market_data

_OHLCV_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


def _crsp_store(tmp_path: Path) -> Path:
    """Write a small OHLCV-shaped CRSP store and return its path."""
    rows = [
        {
            "ticker": ticker,
            "date": day,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adj_close": 1.0,
            "volume": 100,
        }
        for ticker in ["AAPL", "MSFT"]
        for day in pd.date_range("2019-04-01", "2019-04-05", freq="D")
    ]
    store = tmp_path / "daily_returns.parquet"
    pd.DataFrame(rows)[_OHLCV_COLUMNS].to_parquet(store, index=False)
    return store


def test_fetch_ohlcv_returns_locked_schema(tmp_path: Path) -> None:
    store = _crsp_store(tmp_path)
    out = market_data.fetch_ohlcv(["AAPL"], "2019-04-01", "2019-04-30", crsp_store=store)
    assert list(out.columns) == _OHLCV_COLUMNS
    assert out["ticker"].unique().tolist() == ["AAPL"]


def test_fetch_ohlcv_filters_by_ticker(tmp_path: Path) -> None:
    store = _crsp_store(tmp_path)
    out = market_data.fetch_ohlcv(["MSFT"], "2019-04-01", "2019-04-30", crsp_store=store)
    assert set(out["ticker"]) == {"MSFT"}


def test_fetch_ohlcv_filters_by_date_range(tmp_path: Path) -> None:
    store = _crsp_store(tmp_path)
    out = market_data.fetch_ohlcv(["AAPL"], "2019-04-02", "2019-04-03", crsp_store=store)
    assert out["date"].min() == pd.Timestamp("2019-04-02")
    assert out["date"].max() == pd.Timestamp("2019-04-03")


def test_fetch_ohlcv_missing_ticker_returns_empty_with_schema(tmp_path: Path) -> None:
    store = _crsp_store(tmp_path)
    out = market_data.fetch_ohlcv(["NOPE"], "2019-04-01", "2019-04-30", crsp_store=store)
    assert out.empty
    assert list(out.columns) == _OHLCV_COLUMNS


def test_fetch_ohlcv_raises_when_no_tickers(tmp_path: Path) -> None:
    store = _crsp_store(tmp_path)
    with pytest.raises(ValueError, match="no tickers"):
        market_data.fetch_ohlcv([], "2019-04-01", "2019-04-30", crsp_store=store)


def test_fetch_ohlcv_raises_when_store_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CRSP market store"):
        market_data.fetch_ohlcv(
            ["AAPL"], "2019-04-01", "2019-04-30", crsp_store=tmp_path / "absent.parquet"
        )


def test_fetch_ohlcv_cache_short_circuits_store(tmp_path: Path) -> None:
    cache = tmp_path / "cache.parquet"
    pd.DataFrame(
        [
            {
                "ticker": "X",
                "date": pd.Timestamp("2020-01-01"),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "adj_close": 1.0,
                "volume": 1,
            }
        ]
    )[_OHLCV_COLUMNS].to_parquet(cache, index=False)
    # store path is absent — must not be touched because cache exists
    out = market_data.fetch_ohlcv(
        ["AAPL"],
        "2019-04-01",
        "2019-04-30",
        cache_path=cache,
        crsp_store=tmp_path / "absent.parquet",
    )
    assert out["ticker"].tolist() == ["X"]
