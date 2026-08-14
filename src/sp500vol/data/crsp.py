"""CRSP/WRDS data readers: point-in-time CIK linking + daily returns.

Replaces the Wikipedia/yfinance sources for the S&P 500 2010-2025 universe.

- CRSP ``DlyRet`` is the split/dividend-adjusted *total* return, so the canonical
  log return is ``log_return = log1p(DlyRet)`` — no price differencing, no
  yfinance adjusted-close.
- CIK comes from the CRSP/Compustat-Merged (CCM) linking table via a
  POINT-IN-TIME join: a PERMNO can map to different CIKs across M&A boundaries
  (e.g. Baker Hughes Inc -> Co), so a flat PERMNO->CIK dict would mis-attribute
  filings across the merger date.

Canonical artifacts are built by ``scripts/ingest_wrds.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CIK_LINKS_COLUMNS = ("permno", "cik", "link_start", "link_end")
RETURNS_COLUMNS = ("ticker", "date", "log_return")


def load_cik_links(path: Path) -> pd.DataFrame:
    """Load the point-in-time PERMNO->CIK link table (built from CCM)."""
    links = pd.read_parquet(path)
    links["link_start"] = pd.to_datetime(links["link_start"])
    links["link_end"] = pd.to_datetime(links["link_end"])  # NaT = open link
    links["permno"] = links["permno"].astype(int)
    links["cik"] = links["cik"].astype(str).str.zfill(10)
    return links


def resolve_cik(permno: int, date: pd.Timestamp, links: pd.DataFrame) -> str | None:
    """Resolve a PERMNO to the CIK active on ``date`` (point-in-time).

    A PERMNO can have multiple disjoint primary links across M&A boundaries; the
    date selects the correct one. Returns None if no link window covers ``date``.
    """
    d = pd.Timestamp(date).normalize()
    sub = links[links["permno"] == int(permno)]
    if sub.empty:
        return None
    covers = (sub["link_start"] <= d) & (sub["link_end"].isna() | (sub["link_end"] >= d))
    hit = sub.loc[covers]
    if hit.empty:
        return None
    # Defensive: if >1 window matches (should not for primary links), take latest start.
    return str(hit.sort_values("link_start").iloc[-1]["cik"])


def load_crsp_returns(path: Path) -> pd.DataFrame:
    """Load the canonical ``(ticker, date, log_return)`` store (from DlyRet)."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df[list(RETURNS_COLUMNS)]
