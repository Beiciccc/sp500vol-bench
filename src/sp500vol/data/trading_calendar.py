"""Trading-calendar utilities.

CRITICAL: any function that maps wall-clock filing timestamps to trading days
MUST go through this module. Look-ahead bias originates here when written carelessly.

Convention used throughout SP500Vol-Bench:
  - "filing_time" = SEC-reported accepted timestamp (UTC, includes after-hours)
  - "effective_trading_day" = the trading session on which the filing can first
    be acted on using daily data. Filings before the local close use the same
    session; filings at/after the close, or on non-trading days, use the next
    session.
  - Alignment code decides the label window explicitly from this effective day.
"""

from __future__ import annotations

from datetime import datetime, time
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

# S&P 500 labels use the regular US equities holiday/session calendar.
_CALENDAR_NAME = "NYSE"
_EASTERN = ZoneInfo("America/New_York")


@lru_cache(maxsize=1)
def get_schedule(start: str = "2010-01-01", end: str = "2027-01-01") -> pd.DataFrame:
    """Return US equities trading-day schedule between start and end inclusive."""
    cal = mcal.get_calendar(_CALENDAR_NAME)
    return cal.schedule(start_date=start, end_date=end)


def is_trading_day(date: pd.Timestamp) -> bool:
    """True if date is a US equities trading day."""
    sched = get_schedule()
    return pd.Timestamp(date).normalize() in sched.index


def next_trading_day(after: pd.Timestamp) -> pd.Timestamp:
    """Strictly next trading day after `after` (timestamp, not just date)."""
    sched = get_schedule()
    after_day = pd.Timestamp(after).tz_localize(None).normalize()
    future_days = sched.index[sched.index > after_day]
    if future_days.empty:
        raise ValueError(f"no trading day found after {after_day.date()}")
    return pd.Timestamp(future_days[0]).normalize()


def effective_trading_day(
    filing_time: datetime,
    *,
    cutoff_local_time: str = "16:00",
) -> pd.Timestamp:
    """Map a filing timestamp to its effective trading day.

    Args:
        filing_time: SEC accepted timestamp (timezone-aware).
        cutoff_local_time: filings accepted at or after this time (US/Eastern)
            push effective day to the next session. Default 16:00 ET (market close).

    Returns:
        Effective trading day (timezone-naive, normalized to date).
    """
    filing_ts = pd.Timestamp(filing_time)
    if filing_ts.tzinfo is None:
        raise ValueError("filing_time must be timezone-aware")

    filing_et = filing_ts.tz_convert(_EASTERN)
    local_day = filing_et.tz_localize(None).normalize()
    sched = get_schedule()

    if local_day not in sched.index:
        return _first_trading_day_on_or_after(local_day)

    configured_cutoff = _parse_local_time(cutoff_local_time)
    configured_cutoff_dt = pd.Timestamp.combine(local_day.date(), configured_cutoff).tz_localize(
        _EASTERN
    )
    market_close_et = pd.Timestamp(sched.loc[local_day, "market_close"]).tz_convert(_EASTERN)
    cutoff_dt = min(configured_cutoff_dt, market_close_et)

    if filing_et < cutoff_dt:
        return local_day
    return next_trading_day(local_day)


def trading_days_window(start: pd.Timestamp, *, horizon: int) -> pd.DatetimeIndex:
    """Return the next `horizon` trading days starting from `start` inclusive."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    start_day = pd.Timestamp(start).tz_localize(None).normalize()
    sched = get_schedule()
    window = sched.index[sched.index >= start_day][:horizon]
    if window.empty or pd.Timestamp(window[0]).normalize() != start_day:
        raise ValueError(f"start is not a trading day: {start_day.date()}")
    if len(window) < horizon:
        raise ValueError(f"not enough trading days from {start_day.date()} for horizon {horizon}")
    return pd.DatetimeIndex(window).normalize()


def _first_trading_day_on_or_after(date: pd.Timestamp) -> pd.Timestamp:
    """First trading day on or after a normalized date."""
    sched = get_schedule()
    day = pd.Timestamp(date).tz_localize(None).normalize()
    future_days = sched.index[sched.index >= day]
    if future_days.empty:
        raise ValueError(f"no trading day found on or after {day.date()}")
    return pd.Timestamp(future_days[0]).normalize()


def _parse_local_time(value: str) -> time:
    try:
        hour, minute = value.split(":", maxsplit=1)
        return time(hour=int(hour), minute=int(minute))
    except ValueError as exc:
        raise ValueError("cutoff_local_time must use HH:MM format") from exc
