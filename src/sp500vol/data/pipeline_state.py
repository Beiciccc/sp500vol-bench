"""Persistence helpers for resumable data-pipeline runs.

This module intentionally contains only side-effect helpers for small state
files. It does not wire itself into dataset construction, so callers can adopt
the ledger, failure log, and manifest independently.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self, cast

JsonDict = dict[str, Any]
WorkStatus = Literal["done", "failed", "skipped"]

_MANIFEST_NAME = "run_manifest.json"
_FAILURE_FIELDS = (
    "run_id",
    "stage",
    "item_id",
    "ticker",
    "cik",
    "accession",
    "error_type",
    "message",
    "retryable",
    "attempt",
    "timestamp_utc",
)


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp suitable for JSON state files."""
    return datetime.now(UTC).isoformat()


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read a JSONL file into dictionaries.

    Missing files are treated as empty state, which is useful when resuming a
    run before any ledger or failure log has been written.
    """
    if not path.exists():
        return []

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    nonblank_lines = [
        (line_number, line.strip())
        for line_number, line in enumerate(raw_lines, start=1)
        if line.strip()
    ]
    last_line_number = nonblank_lines[-1][0] if nonblank_lines else None

    rows: list[JsonDict] = []
    for line_number, stripped in nonblank_lines:
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            if line_number == last_line_number:
                break
            raise
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write dictionaries to a JSONL file, replacing any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_dumps(dict(row)))
            handle.write("\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append a single dictionary to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_dumps(dict(row)))
        handle.write("\n")


@dataclass(frozen=True, slots=True, kw_only=True)
class FailureRecord:
    """One append-only failure-log entry."""

    run_id: str
    stage: str
    item_id: str
    ticker: str | None = None
    cik: str | None = None
    accession: str | None = None
    error_type: str
    message: str
    retryable: bool
    attempt: int
    timestamp_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> JsonDict:
        """Return the JSON payload in the public failure-log schema order."""
        values = asdict(self)
        return {field_name: values[field_name] for field_name in _FAILURE_FIELDS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Build a failure record from a decoded JSON object."""
        return cls(
            run_id=_required_str(payload, "run_id"),
            stage=_required_str(payload, "stage"),
            item_id=_required_str(payload, "item_id"),
            ticker=_optional_str(payload.get("ticker")),
            cik=_optional_str(payload.get("cik")),
            accession=_optional_str(payload.get("accession")),
            error_type=_required_str(payload, "error_type"),
            message=_required_str(payload, "message"),
            retryable=bool(payload["retryable"]),
            attempt=int(payload["attempt"]),
            timestamp_utc=_required_str(payload, "timestamp_utc"),
        )


def append_failure_log(
    path: Path,
    *,
    run_id: str,
    stage: str,
    item_id: str,
    error_type: str,
    message: str,
    retryable: bool,
    attempt: int,
    ticker: str | None = None,
    cik: str | None = None,
    accession: str | None = None,
    timestamp_utc: str | None = None,
) -> FailureRecord:
    """Append one failure record to a JSONL log and return the written entry."""
    record = FailureRecord(
        run_id=run_id,
        stage=stage,
        item_id=item_id,
        ticker=ticker,
        cik=cik,
        accession=accession,
        error_type=error_type,
        message=message,
        retryable=retryable,
        attempt=attempt,
        timestamp_utc=timestamp_utc or utc_now_iso(),
    )
    append_jsonl(path, record.to_dict())
    return record


def load_failure_log(path: Path) -> list[FailureRecord]:
    """Load all failure-log records from JSONL."""
    return [FailureRecord.from_dict(row) for row in read_jsonl(path)]


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Manifest entry for one produced artifact."""

    path: str
    sha256: str | None = None
    size_bytes: int | None = None

    @classmethod
    def from_path(cls, path: Path, *, base_dir: Path | None = None) -> Self:
        """Create an artifact record with path, SHA-256 hash, and byte size."""
        resolved = path.resolve()
        display_path = _relative_or_absolute(resolved, base_dir.resolve() if base_dir else None)
        return cls(
            path=display_path,
            sha256=sha256_file(resolved),
            size_bytes=resolved.stat().st_size,
        )

    def to_dict(self) -> JsonDict:
        """Return a JSON-serializable artifact record."""
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunManifest:
    """Top-level run manifest written as ``run_manifest.json``."""

    config_hash: str
    started_at_utc: str
    finished_at_utc: str
    counts: Mapping[str, int]
    artifacts: Iterable[ArtifactRecord | Mapping[str, Any]]
    run_id: str | None = None
    git_sha: str | None = None

    def to_dict(self) -> JsonDict:
        """Return the manifest payload."""
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "git_sha": self.git_sha,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "counts": dict(self.counts),
            "artifacts": [_artifact_to_dict(artifact) for artifact in self.artifacts],
        }


def write_run_manifest(run_dir_or_path: Path, manifest: RunManifest) -> Path:
    """Write ``run_manifest.json`` and return its path.

    If ``run_dir_or_path`` ends with ``.json`` it is treated as the exact output
    file; otherwise the manifest is written under ``run_manifest.json`` in that
    directory.
    """
    manifest_path = _manifest_path(run_dir_or_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def hash_config(config: Any) -> str:
    """Return a deterministic SHA-256 hash for a config object."""
    if isinstance(config, bytes):
        payload = config
    elif isinstance(config, str):
        payload = config.encode("utf-8")
    else:
        payload = _json_dumps(_normalise_json(config)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha(cwd: Path | None = None) -> str | None:
    """Return the current git commit SHA if git metadata is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkItemState:
    """Latest known state for one resumable work item."""

    item_id: str
    status: WorkStatus
    run_id: str | None = None
    stage: str | None = None
    ticker: str | None = None
    cik: str | None = None
    accession: str | None = None
    error_type: str | None = None
    message: str | None = None
    retryable: bool | None = None
    attempt: int | None = None
    timestamp_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> JsonDict:
        """Return a JSON-serializable work item state."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Build a work item state from a decoded JSON object."""
        return cls(
            item_id=_required_str(payload, "item_id"),
            status=_work_status(payload["status"]),
            run_id=_optional_str(payload.get("run_id")),
            stage=_optional_str(payload.get("stage")),
            ticker=_optional_str(payload.get("ticker")),
            cik=_optional_str(payload.get("cik")),
            accession=_optional_str(payload.get("accession")),
            error_type=_optional_str(payload.get("error_type")),
            message=_optional_str(payload.get("message")),
            retryable=_optional_bool(payload.get("retryable")),
            attempt=_optional_int(payload.get("attempt")),
            timestamp_utc=_required_str(payload, "timestamp_utc"),
        )


class WorkItemLedger:
    """In-memory ledger for work item resume state."""

    def __init__(
        self,
        states: Iterable[WorkItemState] = (),
        *,
        append_path: Path | None = None,
    ) -> None:
        self._states: dict[str, WorkItemState] = {}
        for state in states:
            self._states[state.item_id] = state
        self._append_path = append_path
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load the latest state per item from a JSONL ledger."""
        return cls(
            (WorkItemState.from_dict(row) for row in read_jsonl(path)),
            append_path=path,
        )

    def save(self, path: Path) -> None:
        """Atomically compact the latest state per item into a JSONL file."""
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        write_jsonl(tmp_path, (state.to_dict() for state in self.records()))
        tmp_path.replace(path)
        self._append_path = path

    def records(self) -> list[WorkItemState]:
        """Return current ledger states sorted by item id for stable output."""
        return sorted(self._states.values(), key=lambda state: state.item_id)

    def get(self, item_id: str) -> WorkItemState | None:
        """Return the latest state for an item, if present."""
        return self._states.get(item_id)

    def status_for(self, item_id: str) -> WorkStatus | None:
        """Return the latest status for an item, if present."""
        state = self.get(item_id)
        return state.status if state else None

    def mark_done(
        self,
        item_id: str,
        *,
        run_id: str | None = None,
        stage: str | None = None,
        ticker: str | None = None,
        cik: str | None = None,
        accession: str | None = None,
        attempt: int | None = None,
        timestamp_utc: str | None = None,
    ) -> WorkItemState:
        """Mark a work item as done."""
        return self._mark(
            item_id,
            status="done",
            run_id=run_id,
            stage=stage,
            ticker=ticker,
            cik=cik,
            accession=accession,
            attempt=attempt,
            timestamp_utc=timestamp_utc,
        )

    def mark_failed(
        self,
        item_id: str,
        *,
        run_id: str | None = None,
        stage: str | None = None,
        ticker: str | None = None,
        cik: str | None = None,
        accession: str | None = None,
        error_type: str | None = None,
        message: str | None = None,
        retryable: bool | None = None,
        attempt: int | None = None,
        timestamp_utc: str | None = None,
    ) -> WorkItemState:
        """Mark a work item as failed."""
        return self._mark(
            item_id,
            status="failed",
            run_id=run_id,
            stage=stage,
            ticker=ticker,
            cik=cik,
            accession=accession,
            error_type=error_type,
            message=message,
            retryable=retryable,
            attempt=attempt,
            timestamp_utc=timestamp_utc,
        )

    def mark_skipped(
        self,
        item_id: str,
        *,
        run_id: str | None = None,
        stage: str | None = None,
        ticker: str | None = None,
        cik: str | None = None,
        accession: str | None = None,
        message: str | None = None,
        attempt: int | None = None,
        timestamp_utc: str | None = None,
    ) -> WorkItemState:
        """Mark a work item as skipped."""
        return self._mark(
            item_id,
            status="skipped",
            run_id=run_id,
            stage=stage,
            ticker=ticker,
            cik=cik,
            accession=accession,
            message=message,
            attempt=attempt,
            timestamp_utc=timestamp_utc,
        )

    def done_item_ids(self) -> set[str]:
        """Return item ids currently marked done."""
        return self._ids_with_status("done")

    def failed_item_ids(self) -> set[str]:
        """Return item ids currently marked failed."""
        return self._ids_with_status("failed")

    def skipped_item_ids(self) -> set[str]:
        """Return item ids currently marked skipped."""
        return self._ids_with_status("skipped")

    def remaining_items(self, item_ids: Iterable[str], *, retry_failed: bool = True) -> list[str]:
        """Return items that should still be attempted on resume."""
        terminal = {"done", "skipped"} if retry_failed else {"done", "failed", "skipped"}
        return [item_id for item_id in item_ids if self.status_for(item_id) not in terminal]

    def _mark(
        self,
        item_id: str,
        *,
        status: WorkStatus,
        run_id: str | None = None,
        stage: str | None = None,
        ticker: str | None = None,
        cik: str | None = None,
        accession: str | None = None,
        error_type: str | None = None,
        message: str | None = None,
        retryable: bool | None = None,
        attempt: int | None = None,
        timestamp_utc: str | None = None,
    ) -> WorkItemState:
        state = WorkItemState(
            item_id=item_id,
            status=status,
            run_id=run_id,
            stage=stage,
            ticker=ticker,
            cik=cik,
            accession=accession,
            error_type=error_type,
            message=message,
            retryable=retryable,
            attempt=attempt,
            timestamp_utc=timestamp_utc or utc_now_iso(),
        )
        with self._lock:
            self._states[item_id] = state
            if self._append_path is not None:
                append_jsonl(self._append_path, state.to_dict())
        return state

    def _ids_with_status(self, status: WorkStatus) -> set[str]:
        return {item_id for item_id, state in self._states.items() if state.status == status}


def _manifest_path(run_dir_or_path: Path) -> Path:
    if run_dir_or_path.suffix == ".json":
        return run_dir_or_path
    return run_dir_or_path / _MANIFEST_NAME


def _artifact_to_dict(artifact: ArtifactRecord | Mapping[str, Any]) -> JsonDict:
    if isinstance(artifact, ArtifactRecord):
        return artifact.to_dict()
    return dict(artifact)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, default=_json_default, separators=(",", ":"), sort_keys=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalise_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if is_dataclass(value):
        return _normalise_json(asdict(value))
    return value


def _relative_or_absolute(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return str(path)
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if value is None:
        raise ValueError(f"{key} must not be null")
    return str(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _work_status(value: Any) -> WorkStatus:
    if value in {"done", "failed", "skipped"}:
        return cast(WorkStatus, value)
    raise ValueError(f"unknown work item status: {value!r}")
