"""CRITICAL correctness tests for point-in-time alignment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sp500vol.data.alignment import align_filings_to_labels
from sp500vol.data.trading_calendar import get_schedule
from sp500vol.features.volatility import ANNUALISATION_FACTOR

_ET = ZoneInfo("America/New_York")
_TICKER = "AAPL"


def _et_datetime(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_ET)


def _filings(filing_time: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "ticker": _TICKER,
                "form": "8-K",
                "item_subtype": "2.02",
                "filing_time_utc": filing_time.astimezone(ZoneInfo("UTC")),
                "text_path": Path("data/interim/aapl_8k.txt"),
                "metadata_path": Path("data/interim/aapl_8k.json"),
            }
        ]
    )


def _market_data(start: str = "2023-06-20", end: str = "2023-08-15") -> pd.DataFrame:
    days = get_schedule().loc[start:end].index
    returns = np.arange(1, len(days) + 1, dtype=float) / 1_000
    return pd.DataFrame({"ticker": _TICKER, "date": days, "log_return": returns})


def test_after_hours_filing_uses_next_trading_day() -> None:
    aligned = align_filings_to_labels(
        _filings(_et_datetime(2023, 7, 7, 18)),
        _market_data(),
        horizons=[5],
    )

    row = aligned.iloc[0]
    assert row["effective_trading_day"] == pd.Timestamp("2023-07-10")
    assert row["label_window_start"] == pd.Timestamp("2023-07-11")


def test_filing_on_market_holiday_uses_next_session() -> None:
    aligned = align_filings_to_labels(
        _filings(_et_datetime(2023, 7, 4, 11)),
        _market_data(),
        horizons=[5],
    )

    row = aligned.iloc[0]
    assert row["effective_trading_day"] == pd.Timestamp("2023-07-05")
    assert row["label_window_start"] == pd.Timestamp("2023-07-06")


def test_intraday_filing_before_close_uses_same_day() -> None:
    aligned = align_filings_to_labels(
        _filings(_et_datetime(2023, 7, 6, 11)),
        _market_data(),
        horizons=[5],
    )

    row = aligned.iloc[0]
    assert row["effective_trading_day"] == pd.Timestamp("2023-07-06")
    assert row["label_window_start"] == pd.Timestamp("2023-07-07")


def test_label_window_uses_only_post_effective_day_returns() -> None:
    market = _market_data()
    market.loc[market["date"] == pd.Timestamp("2023-07-06"), "log_return"] = 9.99
    market.loc[market["date"] == pd.Timestamp("2023-07-07"), "log_return"] = 0.01

    aligned = align_filings_to_labels(
        _filings(_et_datetime(2023, 7, 6, 11)),
        market,
        horizons=[1],
    )

    expected = np.sqrt(ANNUALISATION_FACTOR * 0.01**2)
    assert aligned.iloc[0]["label_window_start"] == pd.Timestamp("2023-07-07")
    assert aligned.iloc[0]["label_realised_vol"] == expected


def test_features_use_only_pre_filing_information() -> None:
    market = _market_data()
    expected_return = 0.02
    market.loc[market["date"] == pd.Timestamp("2023-07-05"), "log_return"] = expected_return
    market.loc[market["date"] == pd.Timestamp("2023-07-06"), "log_return"] = 9.99

    aligned = align_filings_to_labels(
        _filings(_et_datetime(2023, 7, 6, 11)),
        market,
        horizons=[1],
    )

    row = aligned.iloc[0]
    assert row["feature_window_end"] == pd.Timestamp("2023-07-05")
    assert row["feature_return_1d"] == expected_return


def test_after_close_features_may_include_same_day_close() -> None:
    market = _market_data()
    expected_return = 0.03
    market.loc[market["date"] == pd.Timestamp("2023-07-06"), "log_return"] = expected_return

    aligned = align_filings_to_labels(
        _filings(_et_datetime(2023, 7, 6, 18)),
        market,
        horizons=[1],
    )

    row = aligned.iloc[0]
    assert row["feature_window_end"] == pd.Timestamp("2023-07-06")
    assert row["feature_return_1d"] == expected_return


def test_horizon_window_extends_by_trading_days_not_calendar_days() -> None:
    aligned = align_filings_to_labels(
        _filings(_et_datetime(2023, 7, 3, 11)),
        _market_data(),
        horizons=[3],
    )

    row = aligned.iloc[0]
    assert row["label_window_start"] == pd.Timestamp("2023-07-05")
    assert row["label_window_end"] == pd.Timestamp("2023-07-07")
