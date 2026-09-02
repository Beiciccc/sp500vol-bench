"""Concurrency and resume tests for EDGAR fetch/parse ingestion."""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def _load_build_dataset_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "build_dataset_for_concurrency_tests",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_dataset = _load_build_dataset_module()
MAX_TEST_IN_FLIGHT = 5
MAX_TEST_NETWORK_CONCURRENCY = 2


class FakeEdgarState:
    def __init__(
        self,
        metadata_by_cik: dict[str, list[object]],
        *,
        fail_once: Iterable[str] = (),
    ) -> None:
        self.metadata_by_cik = metadata_by_cik
        self.fail_once = set(fail_once)
        self.fetch_calls: Counter[str] = Counter()
        self.worker_active = 0
        self.worker_peak = 0
        self.network_active = 0
        self.network_peak = 0


class FakeEdgarClient:
    state: ClassVar[FakeEdgarState]

    def __init__(
        self,
        cache_root: Path,
        user_agent: str | None = None,
        max_concurrency: int = 8,
        rate_limit_per_sec: float = 10.0,
    ) -> None:
        del user_agent, rate_limit_per_sec
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.state = type(self).state
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def __aenter__(self) -> FakeEdgarClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def list_filings(self, cik: str, forms: list[str]) -> list[object]:
        await asyncio.sleep(0)
        allowed_forms = set(forms)
        return [
            meta for meta in self.state.metadata_by_cik.get(cik, []) if meta.form in allowed_forms
        ]

    async def fetch_document(self, meta: object) -> Path:
        accession = str(meta.accession)
        self.state.fetch_calls[accession] += 1
        self.state.worker_active += 1
        self.state.worker_peak = max(self.state.worker_peak, self.state.worker_active)
        try:
            async with self._semaphore:
                self.state.network_active += 1
                self.state.network_peak = max(
                    self.state.network_peak,
                    self.state.network_active,
                )
                try:
                    await asyncio.sleep(0.01)
                    if accession in self.state.fail_once:
                        self.state.fail_once.remove(accession)
                        raise RuntimeError(f"injected fetch failure for {accession}")
                    path = self.cache_root / str(meta.form) / str(meta.cik) / f"{accession}.html"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        path.write_text(_html(accession), encoding="utf-8")
                    return path
                finally:
                    self.state.network_active -= 1
        finally:
            self.state.worker_active -= 1


@pytest.fixture(autouse=True)
def fake_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(build_dataset, "EdgarClient", FakeEdgarClient)
    monkeypatch.setattr(build_dataset, "resolve_data_path", Path)
    monkeypatch.setattr(
        build_dataset,
        "data_path",
        lambda *parts: tmp_path.joinpath(*(str(part) for part in parts)),
    )
    return tmp_path


def test_concurrent_fetch_matches_serial_and_dedupes_accessions(tmp_path: Path) -> None:
    metadata_by_cik = _metadata_by_cik()
    cfg = _cfg(tmp_path, max_concurrency=1, in_flight=1)
    serial_state = FakeEdgarState(metadata_by_cik)
    FakeEdgarClient.state = serial_state

    serial = _run_fetch(cfg, _universe(), tmp_path, run_id="serial")

    _clear_dataset_dirs(tmp_path, str(cfg["name"]))
    cfg = _cfg(tmp_path, max_concurrency=2, in_flight=4)
    concurrent_state = FakeEdgarState(metadata_by_cik)
    FakeEdgarClient.state = concurrent_state

    concurrent = _run_fetch(cfg, _universe(), tmp_path, run_id="concurrent")

    assert set(concurrent["accession"]) == set(serial["accession"])
    assert not concurrent["accession"].duplicated().any()
    assert_frame_equal(concurrent, serial)


def test_resume_skips_done_cached_filings_and_refetches_only_failed(tmp_path: Path) -> None:
    metadata_by_cik = _metadata_by_cik()
    cfg = _cfg(tmp_path, max_concurrency=2, in_flight=3)
    first_state = FakeEdgarState(metadata_by_cik, fail_once={"0000000001-20-000003"})
    FakeEdgarClient.state = first_state

    first = _run_fetch(cfg, _universe(), tmp_path, run_id="first")
    assert "0000000001-20-000003" not in set(first["accession"])

    second_state = FakeEdgarState(metadata_by_cik)
    FakeEdgarClient.state = second_state
    second = _run_fetch(cfg, _universe(), tmp_path, run_id="second")

    assert set(second["accession"]) == _expected_accessions()
    assert not second["accession"].duplicated().any()
    assert second_state.fetch_calls == Counter({"0000000001-20-000003": 1})


def test_worker_pool_and_client_concurrency_are_bounded(tmp_path: Path) -> None:
    metadata_by_cik = _metadata_by_cik(extra_count=8)
    cfg = _cfg(
        tmp_path,
        max_concurrency=MAX_TEST_NETWORK_CONCURRENCY,
        in_flight=MAX_TEST_IN_FLIGHT,
    )
    state = FakeEdgarState(metadata_by_cik)
    FakeEdgarClient.state = state

    filings = _run_fetch(cfg, _universe(), tmp_path, run_id="bounded")

    assert len(filings) == len(_expected_accessions(extra_count=8))
    assert state.worker_peak <= MAX_TEST_IN_FLIGHT
    assert state.network_peak <= MAX_TEST_NETWORK_CONCURRENCY


def _run_fetch(
    cfg: dict,
    universe: pd.DataFrame,
    tmp_path: Path,
    *,
    run_id: str,
) -> pd.DataFrame:
    state_dir = tmp_path / "processed" / str(cfg["name"]) / "_state"
    return asyncio.run(
        build_dataset._fetch_and_parse_filings(
            cfg,
            universe,
            run_id=run_id,
            state_dir=state_dir,
        )
    )


def _cfg(tmp_path: Path, *, max_concurrency: int, in_flight: int) -> dict:
    return {
        "name": "unit",
        "date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "forms": ["8-K"],
        "edgar": {
            "cache_root": str(tmp_path / "raw"),
            "max_concurrency": max_concurrency,
            "rate_limit_per_sec": 10,
            "in_flight": in_flight,
        },
    }


def _metadata_by_cik(*, extra_count: int = 0) -> dict[str, list[object]]:
    cik_one = "0000000001"
    cik_two = "0000000002"
    rows = {
        cik_one: [
            _meta(cik_one, "0000000001-20-000001", "2020-01-02"),
            _meta(cik_one, "0000000001-20-000002", "2020-02-03"),
            _meta(cik_one, "0000000001-20-000003", "2020-03-04"),
            _meta(cik_one, "0000000001-20-000002", "2020-02-03"),
        ],
        cik_two: [
            _meta(cik_two, "0000000002-20-000001", "2020-01-05"),
            _meta(cik_two, "0000000002-20-000002", "2020-04-06"),
            _meta(cik_two, "0000000002-20-000003", "2020-05-07"),
        ],
    }
    for index in range(extra_count):
        accession = f"{cik_two}-20-{index + 10:06d}"
        rows[cik_two].append(_meta(cik_two, accession, "2020-06-01"))
    return rows


def _expected_accessions(*, extra_count: int = 0) -> set[str]:
    accessions = {
        "0000000001-20-000001",
        "0000000001-20-000002",
        "0000000001-20-000003",
        "0000000002-20-000001",
        "0000000002-20-000002",
        "0000000002-20-000003",
    }
    accessions.update(f"0000000002-20-{index + 10:06d}" for index in range(extra_count))
    return accessions


def _meta(cik: str, accession: str, date: str) -> object:
    return build_dataset.FilingMetadata(
        cik=cik,
        accession=accession,
        form="8-K",
        filing_date=date,
        accepted_datetime=f"{date}T12:00:00.000Z",
        primary_document_url=f"https://example.test/{accession}.html",
        items=["8.01"],
    )


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "cik": ["0000000001", "0000000002"],
            "member_from": [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-01")],
            "member_to": [pd.NaT, pd.NaT],
        }
    )


def _clear_dataset_dirs(tmp_path: Path, dataset_name: str) -> None:
    for root in [
        tmp_path / "raw",
        tmp_path / "interim" / dataset_name,
        tmp_path / "processed" / dataset_name,
    ]:
        shutil.rmtree(root, ignore_errors=True)


def _html(accession: str) -> str:
    words = " ".join(f"word{index}" for index in range(140))
    return (
        "<html><body>"
        f"<p>Item 8.01 {accession} {words}</p>"
        f"<p>Item 9.01 exhibit details {words}</p>"
        "</body></html>"
    )
