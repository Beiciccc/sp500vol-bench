"""Async SEC EDGAR API client with rate-limit compliance.

SEC EDGAR Fair Access policy:
  - Max ~10 requests per second per IP
  - User-Agent header REQUIRED, must identify requester
  - https://www.sec.gov/os/accessing-edgar-data

Caching: every raw HTTP response is written to data/raw/{form}/{cik}/{accession}.html
plus a sidecar JSON with response headers. Never re-fetch a file that exists.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiohttp

EDGAR_BASE = "https://www.sec.gov"
SUBMISSIONS_BASE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
DEFAULT_USER_AGENT = "SP500Vol-Bench/0.0.1 Your Name your.email@example.com"


@dataclass(frozen=True)
class FilingMetadata:
    cik: str
    accession: str
    form: str  # "10-K" | "10-Q" | "8-K"
    filing_date: str  # YYYY-MM-DD
    accepted_datetime: str  # ISO 8601 with timezone
    primary_document_url: str
    items: list[str] | None  # 8-K item codes if applicable


class EdgarClient:
    """Async EDGAR client with SEC-compliant rate limiting and on-disk cache."""

    def __init__(
        self,
        cache_root: Path,
        user_agent: str | None = None,
        max_concurrency: int = 8,
        rate_limit_per_sec: float = 10.0,
    ) -> None:
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        ua = user_agent or os.environ.get("EDGAR_USER_AGENT") or DEFAULT_USER_AGENT
        self.user_agent = ua
        self.max_concurrency = max_concurrency
        self.rate_limit_per_sec = rate_limit_per_sec
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._rate_lock = asyncio.Lock()
        self._next_request_at = 0.0

    async def __aenter__(self) -> EdgarClient:
        timeout = aiohttp.ClientTimeout(total=90)
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def list_filings(self, cik: str, forms: list[str]) -> list[FilingMetadata]:
        """List all filings for a CIK filtered by form types."""
        cik_int = int(cik)
        submission_json = await self._get_json(SUBMISSIONS_BASE.format(cik=cik_int))
        filings = _records_from_recent_filings(submission_json.get("filings", {}).get("recent", {}))

        for file_meta in submission_json.get("filings", {}).get("files", []):
            file_name = file_meta.get("name")
            if not file_name:
                continue
            url = f"https://data.sec.gov/submissions/{file_name}"
            older_json = await self._get_json(url)
            filings.extend(_records_from_recent_filings(older_json))

        allowed_forms = set(forms)
        out: list[FilingMetadata] = []
        seen: set[str] = set()
        for filing in filings:
            accession = str(filing.get("accessionNumber", ""))
            form = str(filing.get("form", ""))
            primary_document = str(filing.get("primaryDocument", ""))
            if (
                not accession
                or accession in seen
                or form not in allowed_forms
                or not primary_document
            ):
                continue

            seen.add(accession)
            accession_no_dashes = accession.replace("-", "")
            primary_document_url = (
                f"{EDGAR_BASE}/Archives/edgar/data/{cik_int}/{accession_no_dashes}/"
                f"{primary_document}"
            )
            out.append(
                FilingMetadata(
                    cik=str(cik_int).zfill(10),
                    accession=accession,
                    form=form,
                    filing_date=str(filing.get("filingDate", "")),
                    accepted_datetime=str(filing.get("acceptanceDateTime", "")),
                    primary_document_url=primary_document_url,
                    items=_parse_items(filing.get("items")),
                )
            )

        return sorted(out, key=lambda meta: (meta.filing_date, meta.accession))

    async def fetch_document(self, meta: FilingMetadata) -> Path:
        """Download a single filing document; return cache path."""
        out_dir = self.cache_root / meta.form / meta.cik
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"{meta.accession}.html"
        metadata_path = out_dir / f"{meta.accession}.json"
        if html_path.exists():
            return html_path

        content = await self._get_bytes(meta.primary_document_url)
        html_path.write_bytes(content)
        metadata_path.write_text(
            json.dumps(
                {
                    "cik": meta.cik,
                    "accession": meta.accession,
                    "form": meta.form,
                    "filing_date": meta.filing_date,
                    "accepted_datetime": meta.accepted_datetime,
                    "primary_document_url": meta.primary_document_url,
                    "items": meta.items,
                    "fetched_at_utc": datetime.now(UTC).isoformat(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return html_path

    async def _get_json(self, url: str) -> dict:
        payload = await self._get_bytes(url)
        return json.loads(payload.decode("utf-8"))

    async def _get_bytes(self, url: str) -> bytes:
        if self._session is None:
            raise RuntimeError("EdgarClient must be used as an async context manager")

        async with self._semaphore:
            await self._respect_rate_limit()
            async with self._session.get(url) as response:
                response.raise_for_status()
                return await response.read()

    async def _respect_rate_limit(self) -> None:
        interval = 1.0 / self.rate_limit_per_sec
        loop = asyncio.get_running_loop()
        async with self._rate_lock:
            now = loop.time()
            wait = self._next_request_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_request_at = now + interval


def _records_from_recent_filings(filings: dict) -> list[dict[str, object]]:
    if not filings:
        return []
    keys = list(filings.keys())
    if not keys:
        return []
    length = len(filings[keys[0]])
    return [{key: filings[key][idx] for key in keys} for idx in range(length)]


def _parse_items(value: object) -> list[str] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
