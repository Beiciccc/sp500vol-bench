"""Market data (OHLCV) sourced from the CRSP daily file.

The canonical market source is the CRSP daily store built by
``scripts/ingest_wrds.py`` from the WRDS S&P 500 constituents file. CRSP
``DlyClose`` is split-adjusted and ``DlyRet`` is the split/dividend-adjusted
total return; the authoritative log-return store is
``(ticker, date, log_return = log1p(DlyRet))`` (see ``sp500vol.data.crsp``).

This module reads the OHLCV-shaped CRSP store and filters it to the requested
tickers and date range — there is NO network access and no yfinance/Tiingo
fallback. The output ``ticker`` column is the point-in-time membership symbol so
it joins to filings, which are keyed by the point-in-time ticker.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from sp500vol.utils.paths import data_path

_OHLCV_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


def _default_crsp_store() -> Path:
    """Default CRSP OHLCV store (resolved lazily; honours SP500VOL_DATA_ROOT)."""
    return data_path("market", "crsp") / "daily_returns.parquet"


def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    cache_path: Path | None = None,
    timeout: float | None = None,
    *,
    crsp_store: Path | None = None,
) -> pd.DataFrame:
    """Return daily OHLCV for the given tickers from the CRSP store.

    Reads the OHLCV-shaped CRSP parquet (split-adjusted ``DlyClose``), filters to
    the requested tickers and the ``[start, end]`` window, and returns columns
    ``[ticker, date, open, high, low, close, adj_close, volume]``. No network
    access. ``timeout`` is accepted for signature compatibility and ignored.
    """
    del timeout  # no network; kept for backward-compatible signature
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)

    unique_tickers = _dedupe_tickers(tickers)
    if not unique_tickers:
        raise ValueError("no tickers supplied")

    store = crsp_store if crsp_store is not None else _default_crsp_store()
    if not store.exists():
        raise FileNotFoundError(
            f"CRSP market store not found: {store} — run scripts/ingest_wrds.py first"
        )

    frame = pd.read_parquet(store)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    mask = (
        frame["ticker"].isin(unique_tickers)
        & (frame["date"] >= start_ts)
        & (frame["date"] <= end_ts)
    )
    out = frame.loc[mask, _OHLCV_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out


def _dedupe_tickers(tickers: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw in tickers:
        if pd.isna(raw):
            continue
        ticker = str(raw).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        unique.append(ticker)
    return unique
