"""Point-in-time alignment of disclosures to market outcomes.

CRITICAL: this is the single most important correctness module in the repo.
Any look-ahead leak here invalidates the entire paper. See
tests/data/test_alignment.py for the canonical correctness tests — those must
pass before any large-scale training run.

Alignment rule:
  1. For each filing F with accepted timestamp t_f, compute effective trading
     day d_f via trading_calendar.effective_trading_day.
  2. For each horizon H ∈ {5, 10, 20}, label = realised_vol over the H
     trading days strictly after d_f.
  3. Features available to the model at prediction time = ONLY information
     observable strictly before t_f.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sp500vol.data.trading_calendar import (
    effective_trading_day,
    get_schedule,
    next_trading_day,
    trading_days_window,
)
from sp500vol.features.volatility import ANNUALISATION_FACTOR


@dataclass(frozen=True)
class AlignedFiling:
    """One filing x one horizon = one labeled training example."""

    cik: str
    ticker: str
    form: str  # "10-K" | "10-Q" | "8-K"
    item_subtype: str | None  # for 8-K (e.g. "2.02"); None otherwise
    filing_time_utc: datetime
    effective_trading_day: pd.Timestamp
    horizon_days: int  # 5 | 10 | 20
    label_realised_vol: float  # annualised RV in the [d_f, d_f+H) window
    label_window_start: pd.Timestamp
    label_window_end: pd.Timestamp
    text_path: Path  # path to parsed text
    metadata_path: Path  # path to filing metadata JSON


def align_filings_to_labels(
    filings: pd.DataFrame,
    market_data: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Produce the aligned (filing x horizon) training table.

    Args:
        filings: columns [cik, ticker, form, item_subtype, filing_time_utc, text_path]
        market_data: columns [ticker, date, log_return]; date is a trading day
        horizons: e.g. [5, 10, 20]

    Returns:
        DataFrame with one row per (filing, horizon) tuple. Columns include
        AlignedFiling fields plus any features pre-computable at d_f.
    """
    _validate_inputs(filings, market_data, horizons)

    market = market_data.copy()
    market["date"] = market["date"].map(_normalise_day)
    if market.duplicated(["ticker", "date"]).any():
        duplicates = market.loc[market.duplicated(["ticker", "date"], keep=False)]
        duplicate_keys = duplicates[["ticker", "date"]].drop_duplicates().to_dict("records")
        raise ValueError(f"duplicate market rows for ticker/date keys: {duplicate_keys}")

    market = market.sort_values(["ticker", "date"])
    returns_by_ticker = {
        str(ticker): group.set_index("date")["log_return"].astype(float).sort_index()
        for ticker, group in market.groupby("ticker", sort=False)
    }

    rows: list[dict[str, object]] = []
    for _, filing in filings.iterrows():
        ticker = str(filing["ticker"])
        if ticker not in returns_by_ticker:
            raise ValueError(f"missing market data for ticker {ticker}")

        filing_ts = _as_utc_timestamp(filing["filing_time_utc"])
        effective_day = effective_trading_day(filing_ts.to_pydatetime())
        feature_window_end = _last_market_close_before(filing_ts)
        returns = returns_by_ticker[ticker]

        for horizon in horizons:
            label_start = next_trading_day(effective_day)
            label_days = trading_days_window(label_start, horizon=horizon)
            label_returns = returns.reindex(label_days)
            if label_returns.isna().any():
                continue

            label_realised_vol = float(
                np.sqrt(ANNUALISATION_FACTOR / horizon * np.square(label_returns).sum())
            )
            rows.append(
                {
                    "cik": filing["cik"],
                    "ticker": ticker,
                    "form": filing["form"],
                    "item_subtype": filing.get("item_subtype"),
                    "filing_time_utc": filing_ts.to_pydatetime(),
                    "effective_trading_day": effective_day,
                    "horizon_days": int(horizon),
                    "label_realised_vol": label_realised_vol,
                    "label_window_start": pd.Timestamp(label_days[0]),
                    "label_window_end": pd.Timestamp(label_days[-1]),
                    "text_path": filing["text_path"],
                    "metadata_path": filing.get("metadata_path"),
                    "feature_window_end": feature_window_end,
                    "feature_return_1d": _return_at(returns, feature_window_end),
                    "feature_rv_5d": _backward_realised_vol(returns, feature_window_end, window=5),
                    "feature_rv_22d": _backward_realised_vol(
                        returns,
                        feature_window_end,
                        window=22,
                    ),
                    **_optional_filing_fields(filing),
                }
            )

    return pd.DataFrame(rows)


def _validate_inputs(
    filings: pd.DataFrame,
    market_data: pd.DataFrame,
    horizons: list[int],
) -> None:
    required_filing_cols = {"cik", "ticker", "form", "item_subtype", "filing_time_utc", "text_path"}
    required_market_cols = {"ticker", "date", "log_return"}
    missing_filing_cols = required_filing_cols.difference(filings.columns)
    missing_market_cols = required_market_cols.difference(market_data.columns)
    if missing_filing_cols:
        raise ValueError(f"filings missing required columns: {sorted(missing_filing_cols)}")
    if missing_market_cols:
        raise ValueError(f"market_data missing required columns: {sorted(missing_market_cols)}")
    if not horizons:
        raise ValueError("horizons must not be empty")
    if any(int(horizon) < 1 for horizon in horizons):
        raise ValueError("all horizons must be >= 1")


def _normalise_day(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None).normalize()


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _last_market_close_before(filing_time_utc: pd.Timestamp) -> pd.Timestamp | None:
    sched = get_schedule()
    market_closes = sched["market_close"]
    previous_closes = market_closes[market_closes < filing_time_utc]
    if previous_closes.empty:
        return None
    return pd.Timestamp(previous_closes.index[-1]).normalize()


def _return_at(returns: pd.Series, day: pd.Timestamp | None) -> float:
    if day is None or pd.isna(day) or day not in returns.index:
        return float("nan")
    return float(returns.loc[day])


def _backward_realised_vol(
    returns: pd.Series,
    end_day: pd.Timestamp | None,
    *,
    window: int,
) -> float:
    if end_day is None or pd.isna(end_day):
        return float("nan")
    history = returns.loc[:end_day].dropna().tail(window)
    if len(history) < window:
        return float("nan")
    return float(np.sqrt(ANNUALISATION_FACTOR / window * np.square(history).sum()))


def _optional_filing_fields(filing: pd.Series) -> dict[str, object]:
    optional_fields = [
        "accession",
        "filing_date",
        "token_count",
        "primary_document_url",
        "parse_warnings",
        "sections_json",
    ]
    return {field: filing[field] for field in optional_fields if field in filing.index}
