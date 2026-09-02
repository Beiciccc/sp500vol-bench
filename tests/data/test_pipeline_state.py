"""Tests for pipeline resume state helpers."""

from __future__ import annotations

import json
from pathlib import Path

from sp500vol.data.pipeline_state import (
    ArtifactRecord,
    RunManifest,
    WorkItemLedger,
    append_failure_log,
    hash_config,
    load_failure_log,
    read_jsonl,
    write_jsonl,
    write_run_manifest,
)

COMPACTED_STATE_COUNT = 2


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ledger.jsonl"
    rows = [
        {"item_id": "aapl-0001", "status": "done", "attempt": 1},
        {"item_id": "msft-0002", "status": "skipped", "attempt": 2},
    ]

    write_jsonl(path, rows)

    assert read_jsonl(path) == rows


def test_read_jsonl_ignores_torn_final_line(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ledger.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"item_id":"a","status":"done"}\n{"item_id"', encoding="utf-8")

    assert read_jsonl(path) == [{"item_id": "a", "status": "done"}]


def test_read_jsonl_raises_mid_file_corruption(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ledger.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"item_id":"a","status":"done"}\n{"item_id"\n{"item_id":"b","status":"done"}\n',
        encoding="utf-8",
    )

    try:
        read_jsonl(path)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("mid-file JSON corruption should raise")


def test_work_item_ledger_marks_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / "work_items.jsonl"
    ledger = WorkItemLedger()

    ledger.mark_done(
        "aapl-0001",
        run_id="run-1",
        stage="fetch",
        ticker="AAPL",
        cik="0000320193",
        accession="0000320193-24-000001",
        attempt=1,
        timestamp_utc="2024-01-01T00:00:00+00:00",
    )
    ledger.mark_failed(
        "msft-0002",
        run_id="run-1",
        stage="parse",
        ticker="MSFT",
        error_type="ValueError",
        message="bad filing",
        retryable=True,
        attempt=2,
        timestamp_utc="2024-01-01T00:01:00+00:00",
    )
    ledger.mark_skipped(
        "goog-0003",
        run_id="run-1",
        stage="fetch",
        ticker="GOOG",
        message="outside window",
        timestamp_utc="2024-01-01T00:02:00+00:00",
    )
    ledger.save(path)

    resumed = WorkItemLedger.load(path)

    assert resumed.status_for("aapl-0001") == "done"
    assert resumed.status_for("msft-0002") == "failed"
    assert resumed.status_for("goog-0003") == "skipped"
    assert resumed.done_item_ids() == {"aapl-0001"}
    assert resumed.failed_item_ids() == {"msft-0002"}
    assert resumed.skipped_item_ids() == {"goog-0003"}
    assert resumed.remaining_items(["aapl-0001", "msft-0002", "goog-0003", "nvda-0004"]) == [
        "msft-0002",
        "nvda-0004",
    ]
    assert resumed.remaining_items(
        ["aapl-0001", "msft-0002", "goog-0003", "nvda-0004"],
        retry_failed=False,
    ) == ["nvda-0004"]


def test_work_item_ledger_load_uses_latest_state(tmp_path: Path) -> None:
    path = tmp_path / "work_items.jsonl"
    write_jsonl(
        path,
        [
            {
                "item_id": "aapl-0001",
                "status": "failed",
                "timestamp_utc": "2024-01-01T00:00:00+00:00",
            },
            {
                "item_id": "aapl-0001",
                "status": "done",
                "timestamp_utc": "2024-01-01T00:02:00+00:00",
            },
        ],
    )

    ledger = WorkItemLedger.load(path)

    assert ledger.status_for("aapl-0001") == "done"


def test_work_item_ledger_appends_one_line_per_mark(tmp_path: Path) -> None:
    path = tmp_path / "work_items.jsonl"
    ledger = WorkItemLedger.load(path)

    ledger.mark_failed(
        "aapl-0001",
        run_id="run-1",
        stage="fetch",
        ticker="AAPL",
        error_type="TimeoutError",
        message="timeout",
        retryable=True,
        attempt=1,
        timestamp_utc="2024-01-01T00:00:00+00:00",
    )
    ledger.mark_done(
        "aapl-0001",
        run_id="run-1",
        stage="fetch",
        ticker="AAPL",
        attempt=2,
        timestamp_utc="2024-01-01T00:01:00+00:00",
    )

    rows = path.read_text(encoding="utf-8").splitlines()
    resumed = WorkItemLedger.load(path)

    assert len(rows) == COMPACTED_STATE_COUNT
    assert resumed.status_for("aapl-0001") == "done"


def test_work_item_ledger_save_compacts_atomically(tmp_path: Path) -> None:
    path = tmp_path / "work_items.jsonl"
    ledger = WorkItemLedger.load(path)

    ledger.mark_failed(
        "aapl-0001",
        run_id="run-1",
        stage="fetch",
        error_type="TimeoutError",
        message="timeout",
        retryable=True,
        attempt=1,
        timestamp_utc="2024-01-01T00:00:00+00:00",
    )
    ledger.mark_done(
        "aapl-0001",
        run_id="run-1",
        stage="fetch",
        attempt=2,
        timestamp_utc="2024-01-01T00:01:00+00:00",
    )
    ledger.mark_done(
        "msft-0002",
        run_id="run-1",
        stage="fetch",
        attempt=1,
        timestamp_utc="2024-01-01T00:02:00+00:00",
    )

    ledger.save(path)

    rows = read_jsonl(path)
    assert len(rows) == COMPACTED_STATE_COUNT
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert WorkItemLedger.load(path).status_for("aapl-0001") == "done"


def test_failure_log_appends_records(tmp_path: Path) -> None:
    path = tmp_path / "failures.jsonl"

    first = append_failure_log(
        path,
        run_id="run-1",
        stage="fetch",
        item_id="aapl-0001",
        ticker="AAPL",
        cik="0000320193",
        accession="0000320193-24-000001",
        error_type="TimeoutError",
        message="SEC request timed out",
        retryable=True,
        attempt=1,
        timestamp_utc="2024-01-01T00:00:00+00:00",
    )
    second = append_failure_log(
        path,
        run_id="run-1",
        stage="parse",
        item_id="msft-0002",
        ticker="MSFT",
        cik="0000789019",
        accession="0000789019-24-000002",
        error_type="ValueError",
        message="empty text",
        retryable=False,
        attempt=2,
        timestamp_utc="2024-01-01T00:01:00+00:00",
    )

    rows = read_jsonl(path)
    failures = load_failure_log(path)

    assert rows == [first.to_dict(), second.to_dict()]
    assert failures[0].retryable is True
    assert failures[1].message == "empty text"


def test_manifest_write_includes_hashes_and_counts(tmp_path: Path) -> None:
    artifact = tmp_path / "features.parquet"
    artifact.write_text("feature rows\n", encoding="utf-8")
    artifact_record = ArtifactRecord.from_path(artifact, base_dir=tmp_path)
    manifest = RunManifest(
        run_id="run-1",
        config_hash=hash_config({"forms": ["10-K", "8-K"], "start": "2020-01-01"}),
        git_sha="abc123",
        started_at_utc="2024-01-01T00:00:00+00:00",
        finished_at_utc="2024-01-01T00:05:00+00:00",
        counts={"work_items": 2, "failed": 1},
        artifacts=[artifact_record],
    )

    manifest_path = write_run_manifest(tmp_path / "run-1", manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == "run_manifest.json"
    assert payload["run_id"] == "run-1"
    assert payload["config_hash"] == hash_config({"start": "2020-01-01", "forms": ["10-K", "8-K"]})
    assert payload["git_sha"] == "abc123"
    assert payload["started_at_utc"] == "2024-01-01T00:00:00+00:00"
    assert payload["finished_at_utc"] == "2024-01-01T00:05:00+00:00"
    assert payload["counts"] == {"failed": 1, "work_items": 2}
    assert payload["artifacts"] == [artifact_record.to_dict()]
    assert payload["artifacts"][0]["path"] == "features.parquet"
    assert payload["artifacts"][0]["sha256"]
