"""Preflight checks before starting full S&P 500 SP500Vol-Bench ingestion.

This script intentionally does not call EDGAR. It checks the point-in-time
study universe and market-data availability so the full run can
fail early on universe/data-source problems.

Usage:
    python scripts/preflight_full_ingestion.py
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.data.crsp import load_crsp_returns
from sp500vol.data.universe import load_membership_table, members_on
from sp500vol.utils import configure_logging, get_logger, resolve_data_path

DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "full.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "full_ingestion_preflight.md"
PARTIAL_TOLERANCE_DAYS = 10
LOW_COVERAGE_THRESHOLD = 0.70


@dataclass(frozen=True)
class TickerInterval:
    ticker: str
    cik: str
    member_from: pd.Timestamp
    member_to: pd.Timestamp
    interval_count: int


@dataclass(frozen=True)
class AvailabilityResult:
    ticker: str
    cik: str
    member_from: str
    member_to: str
    interval_count: int
    status: str
    rows: int
    expected_business_days: int
    coverage_ratio: float | None
    first_date: str
    last_date: str
    issue: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--membership", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--skip-market", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")
    log = get_logger("preflight_full_ingestion")

    cfg = _load_config(args.config)
    membership_path = args.membership or _repo_path(cfg["universe"]["membership_table"])
    date_start, date_end = _date_range(cfg)

    raw_membership = _read_membership_raw(membership_path)
    membership = load_membership_table(membership_path)
    overlapping = _overlapping_membership(membership, date_start, date_end)
    intervals = _ticker_intervals(overlapping, date_start, date_end)
    if args.max_tickers is not None:
        intervals = intervals[: args.max_tickers]

    log.info(
        "Loaded full-ingestion universe",
        config=str(args.config),
        membership=str(membership_path),
        membership_intervals=len(overlapping),
        checked_intervals=len(intervals),
        tickers=overlapping["ticker"].nunique(),
        ticker_cik_pairs=overlapping[["ticker", "cik"]].drop_duplicates().shape[0],
    )

    if args.skip_market:
        availability = [_skipped_availability(interval) for interval in intervals]
    else:
        returns = load_crsp_returns(resolve_data_path(cfg["market"]["returns_store"]))
        availability = [
            _check_crsp_availability(interval, returns=returns, log=log) for interval in intervals
        ]

    availability_df = pd.DataFrame([result.__dict__ for result in availability])
    failed_or_partial = _failed_or_partial(availability_df)
    risky = _risky_tickers(
        overlapping,
        date_end=date_end,
        availability_df=availability_df,
    )
    active_sizes = _active_universe_size_by_year(membership, date_start, date_end)

    report = _render_report(
        cfg=cfg,
        config_path=args.config,
        membership_path=membership_path,
        raw_membership=raw_membership,
        overlapping=overlapping,
        intervals=intervals,
        active_sizes=active_sizes,
        availability_df=availability_df,
        failed_or_partial=failed_or_partial,
        risky=risky,
        generated_at=datetime.now(UTC),
        market_skipped=args.skip_market,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    log.info(
        "Wrote full-ingestion preflight report",
        out=str(args.out),
        failed_or_partial=len(failed_or_partial),
        risky=len(risky),
    )
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config: {path}")
    cfg.pop("defaults", None)
    return cfg


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _date_range(cfg: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    date_start = pd.Timestamp(cfg["date_range"]["start"]).normalize()
    date_end = pd.Timestamp(cfg["date_range"]["end"]).normalize()
    if date_end < date_start:
        raise ValueError(f"invalid date range: {date_start.date()} > {date_end.date()}")
    return date_start, date_end


def _read_membership_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported membership table format: {path.suffix}")


def _overlapping_membership(
    membership: pd.DataFrame,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
) -> pd.DataFrame:
    active = (membership["member_from"] <= date_end) & (
        membership["member_to"].isna() | (membership["member_to"] >= date_start)
    )
    overlapping = membership.loc[active].copy()
    if overlapping.empty:
        raise ValueError(f"no universe members overlap {date_start.date()}-{date_end.date()}")
    return overlapping.sort_values(["ticker", "cik", "member_from"], kind="stable")


def _ticker_intervals(
    membership: pd.DataFrame,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
) -> list[TickerInterval]:
    intervals: list[TickerInterval] = []
    clipped = membership.copy()
    clipped["member_from"] = clipped["member_from"].clip(lower=date_start)
    clipped["member_to"] = clipped["member_to"].fillna(date_end).clip(upper=date_end)
    clipped = clipped.sort_values(["ticker", "cik", "member_from", "member_to"], kind="stable")
    for row in clipped.itertuples(index=False):
        intervals.append(
            TickerInterval(
                ticker=str(row.ticker),
                cik=str(row.cik),
                member_from=pd.Timestamp(row.member_from),
                member_to=pd.Timestamp(row.member_to),
                interval_count=1,
            )
        )
    return intervals


def _active_universe_size_by_year(
    membership: pd.DataFrame,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in range(date_start.year, date_end.year + 1):
        as_of = pd.Timestamp(year=year, month=12, day=31)
        as_of = min(max(as_of, date_start), date_end)
        rows.append(
            {
                "year": year,
                "as_of": _fmt_date(as_of),
                "active_tickers": len(members_on(as_of, membership)),
            }
        )
    return rows


def _check_crsp_availability(
    interval: TickerInterval,
    *,
    returns: pd.DataFrame,
    log: Any,
) -> AvailabilityResult:
    start = interval.member_from.normalize()
    end = interval.member_to.normalize()
    expected_business_days = len(pd.bdate_range(start=start, end=end))
    # Coverage of this ticker's membership window in the CRSP returns store (the
    # ingest source), so a green preflight means build_dataset can label it.
    ticker_dates = returns.loc[returns["ticker"] == interval.ticker, "date"]
    dates = pd.to_datetime(ticker_dates, errors="coerce").dt.normalize().dropna()
    dates = dates[(dates >= start) & (dates <= end)]
    if dates.empty:
        log.warning("no CRSP rows in membership window", ticker=interval.ticker)
        return _availability_result(
            interval,
            "failed",
            0,
            expected_business_days,
            None,
            None,
            "no rows in CRSP store for membership window",
        )

    first = pd.Timestamp(dates.min()).normalize()
    last = pd.Timestamp(dates.max()).normalize()
    rows = len(dates)
    coverage_ratio = rows / expected_business_days if expected_business_days else None
    issues = _availability_issues(start, end, first, last, coverage_ratio)
    status = "partial" if issues else "ok"
    return _availability_result(
        interval,
        status,
        rows,
        expected_business_days,
        first,
        last,
        "; ".join(issues),
        coverage_ratio=coverage_ratio,
    )


def _availability_issues(
    start: pd.Timestamp,
    end: pd.Timestamp,
    first: pd.Timestamp,
    last: pd.Timestamp,
    coverage_ratio: float | None,
) -> list[str]:
    issues: list[str] = []
    if first > start + pd.Timedelta(days=PARTIAL_TOLERANCE_DAYS):
        issues.append(f"first row starts {(first - start).days} days after membership start")
    if last < end - pd.Timedelta(days=PARTIAL_TOLERANCE_DAYS):
        issues.append(f"last row ends {(end - last).days} days before membership end")
    if coverage_ratio is not None and coverage_ratio < LOW_COVERAGE_THRESHOLD:
        issues.append(f"low business-day coverage ({coverage_ratio:.1%})")
    return issues


def _availability_result(
    interval: TickerInterval,
    status: str,
    rows: int,
    expected_business_days: int,
    first: pd.Timestamp | None,
    last: pd.Timestamp | None,
    issue: str,
    *,
    coverage_ratio: float | None = None,
) -> AvailabilityResult:
    return AvailabilityResult(
        ticker=interval.ticker,
        cik=interval.cik,
        member_from=_fmt_date(interval.member_from),
        member_to=_fmt_date(interval.member_to),
        interval_count=interval.interval_count,
        status=status,
        rows=rows,
        expected_business_days=expected_business_days,
        coverage_ratio=coverage_ratio,
        first_date=_fmt_date(first) if first is not None else "",
        last_date=_fmt_date(last) if last is not None else "",
        issue=issue,
    )


def _skipped_availability(interval: TickerInterval) -> AvailabilityResult:
    expected_business_days = len(pd.bdate_range(start=interval.member_from, end=interval.member_to))
    return _availability_result(
        interval,
        "skipped",
        0,
        expected_business_days,
        None,
        None,
        "market check skipped by CLI flag",
    )


def _failed_or_partial(availability_df: pd.DataFrame) -> pd.DataFrame:
    if availability_df.empty:
        return availability_df
    return availability_df[availability_df["status"].isin(["failed", "partial"])].copy()


def _risky_tickers(
    membership: pd.DataFrame,
    *,
    date_end: pd.Timestamp,
    availability_df: pd.DataFrame,
) -> pd.DataFrame:
    active_at_end = set(members_on(date_end, membership))
    ever_active = set(membership["ticker"].astype(str).str.upper())
    ended_tickers = ever_active - active_at_end
    problem_status = _problem_status_by_ticker(availability_df)

    rows: list[dict[str, object]] = []
    for ticker in sorted(ended_tickers | set(problem_status)):
        reasons: list[str] = []
        if ticker in ended_tickers:
            reasons.append("membership_ended_before_config_end")
        if ticker in problem_status:
            reasons.append(f"crsp_{problem_status[ticker]}")
        rows.append({"ticker": ticker, "risk_reasons": ", ".join(reasons)})
    return pd.DataFrame(rows)


def _problem_status_by_ticker(availability_df: pd.DataFrame) -> dict[str, str]:
    if availability_df.empty:
        return {}
    problem = availability_df[availability_df["status"].isin(["failed", "partial"])]
    return dict(zip(problem["ticker"].astype(str), problem["status"].astype(str), strict=False))


def _render_report(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    membership_path: Path,
    raw_membership: pd.DataFrame,
    overlapping: pd.DataFrame,
    intervals: list[TickerInterval],
    active_sizes: list[dict[str, object]],
    availability_df: pd.DataFrame,
    failed_or_partial: pd.DataFrame,
    risky: pd.DataFrame,
    generated_at: datetime,
    market_skipped: bool,
) -> str:
    cik_missing_count = _cik_missing_count(raw_membership)
    summary_rows = [
        ("config", str(config_path)),
        ("membership_table", str(membership_path)),
        ("date_range", f"{cfg['date_range']['start']} to {cfg['date_range']['end']}"),
        ("unique_ticker_count", str(overlapping["ticker"].nunique())),
        (
            "unique_ticker_cik_count",
            str(overlapping[["ticker", "cik"]].drop_duplicates().shape[0]),
        ),
        ("cik_missing_count", str(cik_missing_count)),
        ("membership_interval_count", str(len(overlapping))),
        ("checked_interval_count", str(len(intervals))),
        ("crsp_failed_or_partial_count", str(len(failed_or_partial))),
        ("risky_historical_renamed_delisted_count", str(len(risky))),
    ]
    lines = [
        "# Full S&P 500 Ingestion Preflight",
        "",
        f"- Generated UTC: {generated_at.isoformat()}",
        "- EDGAR ingestion: not run",
        f"- CRSP market availability: {'skipped' if market_skipped else 'checked'}",
        "",
        "## Summary",
        "",
        _markdown_table(["metric", "value"], summary_rows),
        "",
        "## Active Universe Size by Year",
        "",
        _markdown_table(["year", "as_of", "active_tickers"], active_sizes),
        "",
        "## Failed or Partial Market-Data Tickers",
        "",
        _availability_table(failed_or_partial),
        "",
        "## Risky Historical / Renamed / Delisted Tickers",
        "",
        _dataframe_table(risky, ["ticker", "risk_reasons"]),
        "",
        "## CRSP Market-Data Availability by Ticker",
        "",
        _availability_table(availability_df),
        "",
    ]
    return "\n".join(lines)


def _cik_missing_count(table: pd.DataFrame) -> int:
    if "cik" not in table.columns:
        return len(table)
    cik = table["cik"].astype("string").str.strip()
    return int((cik.isna() | cik.eq("")).sum())


def _availability_table(table: pd.DataFrame) -> str:
    columns = [
        "ticker",
        "cik",
        "member_from",
        "member_to",
        "interval_count",
        "status",
        "rows",
        "expected_business_days",
        "coverage_ratio",
        "first_date",
        "last_date",
        "issue",
    ]
    return _dataframe_table(table, columns)


def _dataframe_table(table: pd.DataFrame, columns: list[str]) -> str:
    if table.empty:
        return "_None._"
    rows = table.loc[:, columns].to_dict("records")
    return _markdown_table(columns, rows)


def _markdown_table(
    headers: list[str],
    rows: list[dict[str, object]] | list[tuple[object, ...]],
) -> str:
    rendered_rows = [_row_values(row, headers) for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rendered_rows:
        lines.append("| " + " | ".join(_escape_md_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _row_values(row: dict[str, object] | tuple[object, ...], headers: list[str]) -> list[object]:
    if isinstance(row, dict):
        return [row.get(header, "") for header in headers]
    return list(row)


def _escape_md_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _fmt_date(value: pd.Timestamp | datetime | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
