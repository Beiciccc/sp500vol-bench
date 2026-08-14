"""Build the SP500Vol-Bench dataset (sample or full S&P 500).

Usage:
    python scripts/build_dataset.py --config configs/data/sample.yaml
    python scripts/build_dataset.py --config configs/data/full.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.data.alignment import align_filings_to_labels
from sp500vol.data.crsp import load_crsp_returns
from sp500vol.data.edgar_fetcher import EdgarClient, FilingMetadata
from sp500vol.data.market_data import fetch_ohlcv
from sp500vol.data.parser import ParsedFiling, parse_filing
from sp500vol.data.pipeline_state import (
    ArtifactRecord,
    RunManifest,
    WorkItemLedger,
    append_failure_log,
    current_git_sha,
    hash_config,
    utc_now_iso,
    write_run_manifest,
)
from sp500vol.data.universe import is_member_on, load_membership_table, resolve_explicit_universe
from sp500vol.features.returns import log_returns
from sp500vol.utils import (
    configure_logging,
    data_path,
    data_root,
    get_logger,
    resolve_data_path,
)

DEFAULT_HORIZONS = [5, 10, 20]
MARKET_START_BUFFER_DAYS = 45
MARKET_END_BUFFER_DAYS = 90
STATE_DIR_NAME = "_state"
DUPLICATE_EXAMPLE_LIMIT = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    configure_logging("INFO")
    log = get_logger("build_dataset")
    log.info("Building dataset", config=str(args.config))

    cfg = _load_config(args.config)
    dataset_name = str(cfg["name"])
    run_id = f"build_dataset_{dataset_name}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    state_dir = data_path("processed", dataset_name, STATE_DIR_NAME)
    state_dir.mkdir(parents=True, exist_ok=True)

    universe = _resolve_universe(cfg)
    tickers = _unique_tickers(universe)
    log.info(
        "Resolved universe",
        universe_rows=len(universe),
        ticker_count=len(tickers),
        tickers=tickers,
    )

    started_at_utc = utc_now_iso()
    filings = asyncio.run(
        _fetch_and_parse_filings(cfg, universe, run_id=run_id, state_dir=state_dir)
    )
    log.info("Parsed filings", rows=len(filings))
    if filings.empty:
        raise RuntimeError("No filings parsed; cannot build aligned dataset")

    ohlcv = _fetch_market_data(cfg, tickers)
    market_returns = _load_market_returns(cfg, tickers)
    log.info("Fetched market data", ohlcv_rows=len(ohlcv), return_rows=len(market_returns))
    filings = _filter_marketless_single_day_filings(filings, market_returns, universe, log=log)

    aligned = align_filings_to_labels(filings, market_returns, horizons=DEFAULT_HORIZONS)
    if aligned.empty:
        raise RuntimeError("Alignment produced no labeled rows")

    _write_outputs(
        cfg,
        universe,
        filings,
        ohlcv,
        market_returns,
        aligned,
        args.config,
        run_id=run_id,
        started_at_utc=started_at_utc,
    )
    log.info(
        "Dataset build complete",
        dataset=dataset_name,
        filings=len(filings),
        aligned_rows=len(aligned),
        output=str(data_path("processed", dataset_name)),
    )
    return 0


def _load_config(path: Path) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config: {path}")
    cfg.pop("defaults", None)
    return cfg


def _resolve_universe(cfg: dict) -> pd.DataFrame:
    universe_cfg = cfg["universe"]
    source = universe_cfg["source"]
    if source == "explicit":
        return resolve_explicit_universe(
            list(universe_cfg["tickers"]),
            cache_path=REPO_ROOT / "data" / "universe" / "sec_company_tickers.json",
        )
    if source in {"time_varying_sp500", "time_varying_index"}:
        membership_path = _repo_path(universe_cfg["membership_table"])
        membership = load_membership_table(membership_path)
        date_start = pd.Timestamp(cfg["date_range"]["start"])
        date_end = pd.Timestamp(cfg["date_range"]["end"])
        active = (membership["member_from"] <= date_end) & (
            membership["member_to"].isna() | (membership["member_to"] >= date_start)
        )
        universe = membership.loc[active].copy()
        if universe.empty:
            raise ValueError(f"no universe members overlap {date_start.date()}-{date_end.date()}")
        return universe.sort_values(["ticker", "cik", "member_from"]).reset_index(drop=True)
    raise NotImplementedError(f"unsupported universe source: {source}")


def _unique_tickers(universe: pd.DataFrame) -> list[str]:
    """Return a stable, normalized, de-duplicated ticker list from a universe table."""
    if "ticker" not in universe.columns:
        raise ValueError("universe table missing ticker column")
    return _dedupe_tickers(universe["ticker"].tolist())


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


async def _fetch_and_parse_filings(
    cfg: dict,
    universe: pd.DataFrame,
    *,
    run_id: str,
    state_dir: Path,
) -> pd.DataFrame:
    edgar_cfg = cfg["edgar"]
    date_start = pd.Timestamp(cfg["date_range"]["start"])
    date_end = pd.Timestamp(cfg["date_range"]["end"])
    forms = list(cfg["forms"])
    cache_root = resolve_data_path(edgar_cfg["cache_root"])
    interim_root = data_path("interim", str(cfg["name"]))
    ledger_path = state_dir / "work_items.jsonl"
    failure_log_path = state_dir / "failure_log.jsonl"
    ledger = WorkItemLedger.load(ledger_path)

    max_concurrency = int(edgar_cfg.get("max_concurrency", 4))
    in_flight = max(1, int(edgar_cfg.get("in_flight", max(2 * max_concurrency, 16))))
    fetch_universe = list(universe[["ticker", "cik"]].drop_duplicates().itertuples(index=False))
    async with EdgarClient(
        cache_root=cache_root,
        max_concurrency=max_concurrency,
        rate_limit_per_sec=float(edgar_cfg.get("rate_limit_per_sec", 8)),
    ) as client:
        work_items = await _collect_filing_work_items(
            client,
            fetch_universe=[(str(firm.ticker), str(firm.cik)) for firm in fetch_universe],
            forms=forms,
            universe=universe,
            date_start=date_start,
            date_end=date_end,
            ledger=ledger,
            failure_log_path=failure_log_path,
            run_id=run_id,
            in_flight=in_flight,
        )
        rows = await _fetch_parsed_work_items(
            client,
            work_items=work_items,
            ledger=ledger,
            failure_log_path=failure_log_path,
            run_id=run_id,
            interim_root=interim_root,
            in_flight=in_flight,
        )

    ledger.save(ledger_path)

    if not rows:
        return pd.DataFrame()

    filings = (
        pd.DataFrame(rows)
        .sort_values(["ticker", "filing_time_utc", "accession"])
        .reset_index(drop=True)
    )
    filings["filing_time_utc"] = pd.to_datetime(filings["filing_time_utc"], utc=True)
    return filings


async def _collect_filing_work_items(
    client: EdgarClient,
    *,
    fetch_universe: list[tuple[str, str]],
    forms: list[str],
    universe: pd.DataFrame,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    ledger: WorkItemLedger,
    failure_log_path: Path,
    run_id: str,
    in_flight: int,
) -> dict[str, tuple[str, FilingMetadata]]:
    work_items: dict[str, tuple[str, FilingMetadata]] = {}
    filing_universe = _normalise_universe_for_filing_filter(universe)
    listing_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    for ticker, cik in fetch_universe:
        listing_queue.put_nowait((ticker, cik))

    async def _list_one(ticker: str, cik: str) -> None:
        try:
            all_meta = await client.list_filings(cik, forms)
        except Exception as exc:
            _record_listing_failure(
                ledger,
                failure_log_path=failure_log_path,
                run_id=run_id,
                ticker=ticker,
                cik=cik,
                exc=exc,
            )
            return

        for meta in all_meta:
            filing_date = pd.Timestamp(meta.filing_date)
            if date_start <= filing_date <= date_end and _is_in_universe_on_filing_date(
                ticker,
                cik,
                filing_date,
                filing_universe,
            ):
                work_items.setdefault(meta.accession, (ticker, meta))

    async def _listing_worker() -> None:
        while True:
            try:
                ticker, cik = listing_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _list_one(ticker, cik)
            finally:
                listing_queue.task_done()

    listing_workers = min(in_flight, listing_queue.qsize())
    if listing_workers:
        await asyncio.gather(*(_listing_worker() for _ in range(listing_workers)))
    return work_items


async def _fetch_parsed_work_items(
    client: EdgarClient,
    *,
    work_items: dict[str, tuple[str, FilingMetadata]],
    ledger: WorkItemLedger,
    failure_log_path: Path,
    run_id: str,
    interim_root: Path,
    in_flight: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fetch_queue = _build_fetch_queue(work_items, ledger, interim_root, rows)
    rows_lock = asyncio.Lock()

    async def _fetch_worker() -> None:
        while True:
            try:
                ticker, meta = fetch_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                row = await _process_filing_work_item(
                    client,
                    meta,
                    ticker=ticker,
                    ledger=ledger,
                    failure_log_path=failure_log_path,
                    run_id=run_id,
                    interim_root=interim_root,
                )
                if row is not None:
                    async with rows_lock:
                        rows.append(row)
            finally:
                fetch_queue.task_done()

    fetch_workers = min(in_flight, fetch_queue.qsize())
    if fetch_workers:
        await asyncio.gather(*(_fetch_worker() for _ in range(fetch_workers)))
    return rows


def _build_fetch_queue(
    work_items: dict[str, tuple[str, FilingMetadata]],
    ledger: WorkItemLedger,
    interim_root: Path,
    rows: list[dict[str, object]],
) -> asyncio.Queue[tuple[str, FilingMetadata]]:
    fetch_queue: asyncio.Queue[tuple[str, FilingMetadata]] = asyncio.Queue()
    for ticker, meta in sorted(
        work_items.values(),
        key=lambda item: (item[1].filing_date, item[1].accession, item[0]),
    ):
        item_id = f"filing:{meta.accession}"
        if ledger.status_for(item_id) == "done":
            cached_row = _cached_filing_row(meta, ticker, interim_root)
            if cached_row is not None:
                rows.append(cached_row)
                continue
        fetch_queue.put_nowait((ticker, meta))
    return fetch_queue


async def _process_filing_work_item(
    client: EdgarClient,
    meta: FilingMetadata,
    *,
    ticker: str,
    ledger: WorkItemLedger,
    failure_log_path: Path,
    run_id: str,
    interim_root: Path,
) -> dict[str, object] | None:
    item_id = f"filing:{meta.accession}"
    try:
        html_path = await client.fetch_document(meta)
        parsed = await asyncio.to_thread(parse_filing, html_path, meta.form)
        text_path, metadata_path = _write_parsed_filing(
            parsed,
            meta,
            ticker=ticker,
            interim_root=interim_root,
        )
        ledger.mark_done(
            item_id,
            run_id=run_id,
            stage="fetch_parse",
            ticker=ticker,
            cik=meta.cik,
            accession=meta.accession,
            attempt=1,
        )
        return _filing_row(meta, parsed, ticker, text_path, metadata_path)
    except Exception as exc:
        _record_fetch_failure(
            ledger,
            failure_log_path=failure_log_path,
            run_id=run_id,
            ticker=ticker,
            meta=meta,
            exc=exc,
        )
        return None


def _record_listing_failure(
    ledger: WorkItemLedger,
    *,
    failure_log_path: Path,
    run_id: str,
    ticker: str,
    cik: str,
    exc: Exception,
) -> None:
    item_id = f"listing:{cik}"
    append_failure_log(
        failure_log_path,
        run_id=run_id,
        stage="edgar_listing",
        item_id=item_id,
        ticker=ticker,
        cik=cik,
        error_type=type(exc).__name__,
        message=str(exc),
        retryable=True,
        attempt=1,
    )
    ledger.mark_failed(
        item_id,
        run_id=run_id,
        stage="edgar_listing",
        ticker=ticker,
        cik=cik,
        error_type=type(exc).__name__,
        message=str(exc),
        retryable=True,
        attempt=1,
    )


def _record_fetch_failure(
    ledger: WorkItemLedger,
    *,
    failure_log_path: Path,
    run_id: str,
    ticker: str,
    meta: FilingMetadata,
    exc: Exception,
) -> None:
    item_id = f"filing:{meta.accession}"
    append_failure_log(
        failure_log_path,
        run_id=run_id,
        stage="fetch_parse",
        item_id=item_id,
        ticker=ticker,
        cik=meta.cik,
        accession=meta.accession,
        error_type=type(exc).__name__,
        message=str(exc),
        retryable=True,
        attempt=1,
    )
    ledger.mark_failed(
        item_id,
        run_id=run_id,
        stage="fetch_parse",
        ticker=ticker,
        cik=meta.cik,
        accession=meta.accession,
        error_type=type(exc).__name__,
        message=str(exc),
        retryable=True,
        attempt=1,
    )


def _is_in_universe_on_filing_date(
    ticker: str,
    cik: str,
    filing_date: pd.Timestamp,
    universe: pd.DataFrame,
) -> bool:
    if {"member_from", "member_to", "cik"}.issubset(universe.columns):
        d = pd.Timestamp(filing_date).normalize()
        ticker_key = str(ticker).strip().upper()
        cik_key = str(cik).strip().zfill(10)
        hit = universe[(universe["ticker"] == ticker_key) & (universe["cik"] == cik_key)]
        if hit.empty:
            return False
        covers = (hit["member_from"] <= d) & (hit["member_to"].isna() | (hit["member_to"] >= d))
        return bool(covers.any())
    if {"member_from", "member_to"}.issubset(universe.columns):
        return is_member_on(ticker, filing_date, universe)
    return True


def _normalise_universe_for_filing_filter(universe: pd.DataFrame) -> pd.DataFrame:
    if not {"member_from", "member_to", "cik"}.issubset(universe.columns):
        return universe
    out = universe.copy()
    out["ticker"] = out["ticker"].astype("string").str.strip().str.upper()
    out["cik"] = out["cik"].astype("string").str.strip().str.zfill(10)
    out["member_from"] = pd.to_datetime(out["member_from"], errors="coerce")
    out["member_to"] = pd.to_datetime(out["member_to"], errors="coerce")
    return out


def _cached_filing_row(
    meta: FilingMetadata,
    ticker: str,
    interim_root: Path,
) -> dict[str, object] | None:
    metadata_path = interim_root / meta.form / meta.cik / f"{meta.accession}.json"
    text_path = interim_root / meta.form / meta.cik / f"{meta.accession}.txt"
    if not metadata_path.exists() or not text_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parsed = ParsedFiling(
        form=meta.form,
        sections=metadata.get("sections", {}),
        full_text=text_path.read_text(encoding="utf-8"),
        token_count=int(metadata.get("token_count", 0)),
        parse_warnings=list(metadata.get("parse_warnings", [])),
    )
    return _filing_row(meta, parsed, ticker, text_path, metadata_path)


def _write_parsed_filing(
    parsed: ParsedFiling,
    meta: FilingMetadata,
    *,
    ticker: str,
    interim_root: Path,
) -> tuple[Path, Path]:
    out_dir = interim_root / parsed.form / meta.cik
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / f"{meta.accession}.txt"
    metadata_path = out_dir / f"{meta.accession}.json"
    text_path.write_text(parsed.full_text, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "ticker": ticker,
                "cik": meta.cik,
                "accession": meta.accession,
                "form": meta.form,
                "filing_date": meta.filing_date,
                "accepted_datetime": meta.accepted_datetime,
                "primary_document_url": meta.primary_document_url,
                "items": meta.items,
                "sections": parsed.sections,
                "token_count": parsed.token_count,
                "parse_warnings": parsed.parse_warnings,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return text_path, metadata_path


def _filing_row(
    meta: FilingMetadata,
    parsed: ParsedFiling,
    ticker: str,
    text_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    return {
        "cik": meta.cik,
        "ticker": ticker,
        "form": meta.form,
        "item_subtype": ",".join(meta.items) if meta.items else None,
        "filing_time_utc": _accepted_timestamp(meta).to_pydatetime(),
        "filing_date": meta.filing_date,
        "accession": meta.accession,
        "primary_document_url": meta.primary_document_url,
        "text_path": str(text_path),
        "metadata_path": str(metadata_path),
        "token_count": parsed.token_count,
        "parse_warnings": json.dumps(parsed.parse_warnings),
        "sections_json": json.dumps(parsed.sections),
    }


def _accepted_timestamp(meta: FilingMetadata) -> pd.Timestamp:
    if meta.accepted_datetime:
        timestamp = pd.Timestamp(meta.accepted_datetime)
    else:
        timestamp = pd.Timestamp(f"{meta.filing_date} 16:00:00")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _market_window(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Date range padded by feature/label buffers so edge windows have data."""
    date_start = pd.Timestamp(cfg["date_range"]["start"]) - pd.Timedelta(
        days=MARKET_START_BUFFER_DAYS
    )
    date_end = pd.Timestamp(cfg["date_range"]["end"]) + pd.Timedelta(days=MARKET_END_BUFFER_DAYS)
    return date_start, date_end


def _fetch_market_data(cfg: dict, tickers: list[str]) -> pd.DataFrame:
    market_cfg = cfg["market"]
    if market_cfg["source"] != "crsp":
        raise NotImplementedError(f"unsupported market source: {market_cfg['source']}")

    date_start, date_end = _market_window(cfg)
    cache_root = resolve_data_path(market_cfg["cache_root"])
    cache_path = cache_root / f"{cfg['name']}_ohlcv.parquet"
    crsp_store = resolve_data_path(market_cfg["crsp_store"])
    ohlcv = fetch_ohlcv(
        _dedupe_tickers(tickers),
        start=date_start.strftime("%Y-%m-%d"),
        end=date_end.strftime("%Y-%m-%d"),
        cache_path=cache_path,
        crsp_store=crsp_store,
    )
    _assert_no_duplicate_ticker_dates(ohlcv, name="ohlcv")
    return ohlcv


def _load_market_returns(cfg: dict, tickers: list[str]) -> pd.DataFrame:
    """Load log1p(DlyRet) returns from the CRSP store, filtered to the universe.

    CRSP DlyRet is the split/dividend-adjusted *total* return — the correct input
    for realised-vol labels (DlyClose is split-only adjusted), so it is used
    directly instead of differencing adj_close.
    """
    market_cfg = cfg["market"]
    returns_path = resolve_data_path(market_cfg["returns_store"])
    date_start, date_end = _market_window(cfg)
    returns = load_crsp_returns(returns_path)
    universe_tickers = set(_dedupe_tickers(tickers))
    mask = (
        returns["ticker"].isin(universe_tickers)
        & (returns["date"] >= date_start)
        & (returns["date"] <= date_end)
    )
    out = returns.loc[mask].sort_values(["ticker", "date"]).reset_index(drop=True)
    _assert_no_duplicate_ticker_dates(out, name="market_returns")
    return out


def _filter_marketless_single_day_filings(
    filings: pd.DataFrame,
    market_returns: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    log: object,
) -> pd.DataFrame:
    """Drop expected zero-label spin-off rows before strict alignment.

    CRSP membership can contain one-day spin-off placeholders with a valid CIK
    but no tradeable return history under that ticker. Those intervals cannot
    produce labels and should contribute zero aligned rows. Longer membership
    intervals without market data remain a hard failure.
    """
    if filings.empty:
        return filings

    filing_tickers = set(_dedupe_tickers(filings["ticker"].tolist()))
    market_tickers = set(_dedupe_tickers(market_returns["ticker"].tolist()))
    missing_tickers = sorted(filing_tickers.difference(market_tickers))
    if not missing_tickers:
        return filings

    tolerated = [
        ticker
        for ticker in missing_tickers
        if _has_only_single_business_day_membership(universe, ticker)
    ]
    hard_missing = [ticker for ticker in missing_tickers if ticker not in set(tolerated)]
    if hard_missing:
        raise ValueError("missing market data for non-spin-off tickers: " + ", ".join(hard_missing))

    if not tolerated:
        return filings

    ticker_key = filings["ticker"].astype("string").str.strip().str.upper()
    drop_mask = ticker_key.isin(tolerated)
    dropped_rows = int(drop_mask.sum())
    if dropped_rows:
        log.info(
            "Dropped filings for single-day universe tickers without CRSP market data",
            tickers=tolerated,
            rows=dropped_rows,
        )
    return filings.loc[~drop_mask].reset_index(drop=True)


def _has_only_single_business_day_membership(universe: pd.DataFrame, ticker: str) -> bool:
    required = {"ticker", "member_from", "member_to"}
    if missing := required.difference(universe.columns):
        raise ValueError(f"universe missing required columns: {sorted(missing)}")

    ticker_key = universe["ticker"].astype("string").str.strip().str.upper()
    rows = universe.loc[ticker_key == str(ticker).strip().upper()]
    if rows.empty:
        return False

    for row in rows.itertuples(index=False):
        start = pd.Timestamp(row.member_from).normalize()
        if pd.isna(start) or pd.isna(row.member_to):
            return False
        end = pd.Timestamp(row.member_to).normalize()
        if pd.isna(end) or end < start:
            return False
        if len(pd.bdate_range(start=start, end=end)) != 1:
            return False
    return True


def _compute_market_returns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    _assert_no_duplicate_ticker_dates(ohlcv, name="ohlcv")
    frames = []
    for ticker, group in ohlcv.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        returns = log_returns(group.set_index("date")["adj_close"]).dropna()
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": returns.index,
                    "log_return": returns.to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "log_return"])
    market_returns = pd.concat(frames, ignore_index=True)
    _assert_no_duplicate_ticker_dates(market_returns, name="market_returns")
    return market_returns


def _assert_no_duplicate_ticker_dates(frame: pd.DataFrame, *, name: str) -> None:
    """Fail fast if a market data product has duplicate ticker/date keys."""
    missing = {"ticker", "date"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    if frame.empty:
        return

    keys = frame.loc[:, ["ticker", "date"]].copy()
    keys["ticker"] = keys["ticker"].astype("string").str.strip().str.upper()
    keys["date"] = pd.to_datetime(keys["date"], errors="coerce", utc=True).dt.tz_convert(None)
    keys["date"] = keys["date"].dt.normalize()

    invalid_ticker = keys["ticker"].isna() | keys["ticker"].eq("")
    if invalid_ticker.any():
        rows = keys.index[invalid_ticker].tolist()[:20]
        raise ValueError(f"{name} has missing/blank ticker values at rows: {rows}")

    invalid_date = keys["date"].isna()
    if invalid_date.any():
        rows = keys.index[invalid_date].tolist()[:20]
        raise ValueError(f"{name} has missing/invalid date values at rows: {rows}")

    duplicate_mask = keys.duplicated(["ticker", "date"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = keys.loc[duplicate_mask, ["ticker", "date"]].drop_duplicates()
        examples = [
            f"{row.ticker}/{pd.Timestamp(row.date).date().isoformat()}"
            for row in duplicate_keys.head(DUPLICATE_EXAMPLE_LIMIT).itertuples(index=False)
        ]
        suffix = (
            ""
            if len(duplicate_keys) <= DUPLICATE_EXAMPLE_LIMIT
            else f" ... (+{len(duplicate_keys) - DUPLICATE_EXAMPLE_LIMIT} more)"
        )
        raise ValueError(
            f"{name} has duplicate ticker/date rows for {len(duplicate_keys)} keys: "
            f"{', '.join(examples)}{suffix}"
        )


def _write_outputs(
    cfg: dict,
    universe: pd.DataFrame,
    filings: pd.DataFrame,
    ohlcv: pd.DataFrame,
    market_returns: pd.DataFrame,
    aligned: pd.DataFrame,
    config_path: Path,
    *,
    run_id: str,
    started_at_utc: str,
) -> None:
    dataset_name = str(cfg["name"])
    processed_root = data_path("processed", dataset_name)
    processed_root.mkdir(parents=True, exist_ok=True)

    _assert_no_duplicate_ticker_dates(ohlcv, name="ohlcv")
    _assert_no_duplicate_ticker_dates(market_returns, name="market_returns")

    universe.to_parquet(processed_root / "universe.parquet", index=False)
    filings.to_parquet(processed_root / "filings.parquet", index=False)
    ohlcv.to_parquet(processed_root / "ohlcv.parquet", index=False)
    market_returns.to_parquet(processed_root / "market_returns.parquet", index=False)
    aligned.to_parquet(processed_root / "aligned_filings.parquet", index=False)

    meta = {
        "run_id": run_id,
        "dataset": dataset_name,
        "source": _market_source_meta(cfg)["source"],
        "market": _market_source_meta(cfg),
        "built_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": hash_config(cfg),
        "git_sha": current_git_sha(REPO_ROOT),
        "tickers": _unique_tickers(universe),
        "date_range": cfg["date_range"],
        "forms": cfg["forms"],
        "horizons": DEFAULT_HORIZONS,
        "counts": {
            "universe_rows": len(universe),
            "filings": len(filings),
            "ohlcv_rows": len(ohlcv),
            "market_return_rows": len(market_returns),
            "aligned_rows": len(aligned),
        },
    }
    (processed_root / "_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts = [
        ArtifactRecord.from_path(processed_root / "universe.parquet", base_dir=data_root()),
        ArtifactRecord.from_path(processed_root / "filings.parquet", base_dir=data_root()),
        ArtifactRecord.from_path(processed_root / "ohlcv.parquet", base_dir=data_root()),
        ArtifactRecord.from_path(processed_root / "market_returns.parquet", base_dir=data_root()),
        ArtifactRecord.from_path(processed_root / "aligned_filings.parquet", base_dir=data_root()),
        ArtifactRecord.from_path(processed_root / "_meta.json", base_dir=data_root()),
    ]
    write_run_manifest(
        processed_root,
        RunManifest(
            run_id=run_id,
            config_hash=hash_config(cfg),
            git_sha=current_git_sha(REPO_ROOT),
            started_at_utc=started_at_utc,
            finished_at_utc=utc_now_iso(),
            counts=meta["counts"],
            artifacts=artifacts,
        ),
    )


def _market_source_meta(cfg: dict) -> dict[str, str]:
    market_cfg = cfg.get("market", {})
    return {
        "source": str(market_cfg.get("source", "")),
        "crsp_store": str(market_cfg.get("crsp_store", "")),
        "returns_store": str(market_cfg.get("returns_store", "")),
    }


def _repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


if __name__ == "__main__":
    sys.exit(main())
