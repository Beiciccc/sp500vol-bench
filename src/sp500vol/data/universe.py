"""Time-varying equity-index membership.

Locked design decision #2: use historical constituent snapshots to avoid
survivorship bias. We need to know, for each filing date, whether the filing's
issuer was in the study universe at that time.

Source decision:
  - Main universe: point-in-time S&P 500 membership built from WRDS CRSP
    S&P 500 Index Constituents, with CIK links resolved through CCM.
  - See design/03_universe_source_decision.md for the locked rationale.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_USER_AGENT = "SP500Vol-Bench/0.0.1 (set EDGAR_USER_AGENT to your name and email; SEC requires it)"
MEMBERSHIP_REQUIRED_COLUMNS = ("ticker", "cik", "member_from", "member_to")
CIK_WIDTH = 10


def validate_membership_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the membership schema.

    Required columns are ticker, cik, member_from, and member_to. Dates are
    normalized to midnight; member_to may be missing for current members.
    """
    missing_cols = set(MEMBERSHIP_REQUIRED_COLUMNS).difference(table.columns)
    if missing_cols:
        raise ValueError(f"membership table missing columns: {sorted(missing_cols)}")

    validated = table.copy()
    validated["ticker"] = _normalize_ticker(validated["ticker"])
    validated["cik"] = _normalize_cik(validated["cik"])
    validated["member_from"] = _parse_membership_dates(
        validated["member_from"],
        column="member_from",
        allow_open=False,
    )
    validated["member_to"] = _parse_membership_dates(
        validated["member_to"],
        column="member_to",
        allow_open=True,
    )

    if validated["ticker"].isna().any():
        rows = _row_numbers(validated["ticker"].isna())
        raise ValueError(f"membership table has blank ticker values at rows: {rows}")

    invalid_cik = ~validated["cik"].str.fullmatch(r"\d{10}", na=False)
    if invalid_cik.any():
        rows = _row_numbers(invalid_cik)
        raise ValueError(f"membership table has invalid CIK values at rows: {rows}")

    invalid_range = validated["member_to"].notna() & (
        validated["member_to"] < validated["member_from"]
    )
    if invalid_range.any():
        rows = _row_numbers(invalid_range)
        raise ValueError(f"membership table has invalid date ranges at rows: {rows}")

    _validate_non_overlapping_intervals(validated)
    return validated


def load_membership_table(path: Path) -> pd.DataFrame:
    """Load the time-varying membership table.

    Returns:
        DataFrame with columns [ticker, cik, member_from, member_to].
        member_to is NaT if currently a member.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table = pd.read_parquet(path)
    elif suffix == ".csv":
        table = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported membership table format: {path.suffix}")

    return validate_membership_table(table)


def is_member_on(ticker: str, date: pd.Timestamp, table: pd.DataFrame) -> bool:
    """True if `ticker` was a universe member on `date`."""
    day = pd.Timestamp(date).normalize()
    ticker_rows = table[table["ticker"].astype(str).str.upper() == ticker.upper()]
    if ticker_rows.empty:
        return False
    active = (ticker_rows["member_from"] <= day) & (
        ticker_rows["member_to"].isna() | (ticker_rows["member_to"] >= day)
    )
    return bool(active.any())


def members_on(date: pd.Timestamp, table: pd.DataFrame) -> list[str]:
    """Return all universe tickers active on the given date."""
    day = pd.Timestamp(date).normalize()
    active = (table["member_from"] <= day) & (
        table["member_to"].isna() | (table["member_to"] >= day)
    )
    return sorted(table.loc[active, "ticker"].astype(str).str.upper().unique().tolist())


def load_sec_ticker_map(
    cache_path: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
) -> pd.DataFrame:
    """Load SEC ticker-to-CIK lookup, caching the public JSON locally."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        request = urllib.request.Request(SEC_TICKER_URL, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
        cache_path.write_bytes(payload)

    records = json.loads(cache_path.read_text(encoding="utf-8"))
    table = pd.DataFrame.from_records(list(records.values()))
    table = table.rename(columns={"cik_str": "cik"})
    table["ticker"] = table["ticker"].astype(str).str.upper()
    table["cik"] = table["cik"].astype(int).astype(str).str.zfill(10)
    return table[["ticker", "cik", "title"]].sort_values("ticker").reset_index(drop=True)


def resolve_explicit_universe(
    tickers: list[str],
    *,
    cache_path: Path,
    user_agent: str = DEFAULT_USER_AGENT,
) -> pd.DataFrame:
    """Resolve explicit tickers to SEC CIKs."""
    ticker_map = load_sec_ticker_map(cache_path, user_agent=user_agent)
    requested = pd.DataFrame({"ticker": [ticker.upper() for ticker in tickers]})
    resolved = requested.merge(ticker_map, on="ticker", how="left")
    missing = resolved.loc[resolved["cik"].isna(), "ticker"].tolist()
    if missing:
        raise ValueError(f"could not resolve SEC CIKs for tickers: {missing}")
    return resolved


def _normalize_ticker(values: pd.Series) -> pd.Series:
    tickers = values.astype("string").str.strip().str.upper()
    return tickers.mask(tickers.eq(""))


def _normalize_cik(values: pd.Series) -> pd.Series:
    cik = values.astype("string").str.strip()
    cik = cik.mask(cik.eq(""))
    cik = cik.str.replace(r"\.0$", "", regex=True)
    return cik.str.zfill(CIK_WIDTH)


def _parse_membership_dates(
    values: pd.Series,
    *,
    column: str,
    allow_open: bool,
) -> pd.Series:
    blank = _blank_mask(values)
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    invalid = parsed.isna() & ~(blank & allow_open)
    if invalid.any():
        rows = _row_numbers(invalid)
        raise ValueError(f"membership table has invalid {column} values at rows: {rows}")
    return parsed.dt.tz_localize(None).dt.normalize()


def _blank_mask(values: pd.Series) -> pd.Series:
    as_text = values.astype("string").str.strip()
    return values.isna() | as_text.isna() | as_text.eq("")


def _validate_non_overlapping_intervals(table: pd.DataFrame) -> None:
    sorted_table = table.sort_values(["ticker", "cik", "member_from"], kind="stable")
    for (ticker, cik), group in sorted_table.groupby(["ticker", "cik"], sort=False):
        previous_end: pd.Timestamp | None = None
        previous_start: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            current_start = row.member_from
            current_end = row.member_to
            if previous_end is not None and (
                pd.isna(previous_end) or current_start <= previous_end
            ):
                raise ValueError(
                    "membership table has overlapping intervals for "
                    f"{ticker}/{cik}: "
                    f"{_format_date(previous_start)}-{_format_date(previous_end)} overlaps "
                    f"{_format_date(current_start)}-{_format_date(current_end)}"
                )
            previous_start = current_start
            previous_end = current_end


def _format_date(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "open"
    return pd.Timestamp(value).date().isoformat()


def _row_numbers(mask: pd.Series) -> list[int]:
    return [position for position, is_invalid in enumerate(mask.to_numpy()) if is_invalid]
