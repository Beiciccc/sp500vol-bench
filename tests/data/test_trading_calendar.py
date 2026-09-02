"""Trading-calendar contract tests for point-in-time alignment."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from sp500vol.data.trading_calendar import (
    effective_trading_day,
    next_trading_day,
    trading_days_window,
)

_ET = ZoneInfo("America/New_York")


def _et_datetime(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_ET)


def test_after_hours_filing_uses_next_trading_day() -> None:
    filing_time = _et_datetime(2023, 7, 7, 18)
    assert effective_trading_day(filing_time) == pd.Timestamp("2023-07-10")


def test_filing_on_market_holiday_uses_next_session() -> None:
    filing_time = _et_datetime(2023, 7, 4, 11)
    assert effective_trading_day(filing_time) == pd.Timestamp("2023-07-05")


def test_intraday_filing_before_close_uses_same_day() -> None:
    filing_time = _et_datetime(2023, 7, 6, 11)
    assert effective_trading_day(filing_time) == pd.Timestamp("2023-07-06")


def test_filing_at_close_uses_next_session() -> None:
    filing_time = _et_datetime(2023, 7, 6, 16)
    assert effective_trading_day(filing_time) == pd.Timestamp("2023-07-07")


def test_next_trading_day_skips_weekends() -> None:
    assert next_trading_day(pd.Timestamp("2023-07-07")) == pd.Timestamp("2023-07-10")


def test_trading_days_window_skips_market_holidays() -> None:
    window = trading_days_window(pd.Timestamp("2023-07-03"), horizon=3)
    assert list(window) == [
        pd.Timestamp("2023-07-03"),
        pd.Timestamp("2023-07-05"),
        pd.Timestamp("2023-07-06"),
    ]
