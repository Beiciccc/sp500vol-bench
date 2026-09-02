"""Capture run environment for reproducibility manifest.

Every results/runs/{run_id}/env.json must be writable from `capture_env()`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return bool(out.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _hash_pip_freeze() -> str | None:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return hashlib.sha256(out.stdout.encode()).hexdigest()[:16]
    except subprocess.SubprocessError:
        return None


def _gpu_info() -> list[dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [
            {
                "name": torch.cuda.get_device_name(i),
                "total_mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    except ImportError:
        return []


def capture_env() -> dict[str, Any]:
    """Return a dict snapshot of the run environment.

    Keys: timestamp_utc, git_sha, git_dirty, python_version, platform, gpu, pip_hash.
    Cloud instance info (provider, instance_type, $/hr) read from env vars
    set by .env (see .env.example).
    """
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": _gpu_info(),
        "pip_freeze_hash": _hash_pip_freeze(),
        "cloud_provider": _env("CLOUD_PROVIDER"),
        "instance_type": _env("INSTANCE_TYPE"),
        "hourly_rate_usd": _env("HOURLY_RATE_USD"),
    }


def _env(key: str) -> str | None:
    import os

    return os.environ.get(key)


def write_env_snapshot(run_dir: Path) -> None:
    """Write env.json into the given run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps(capture_env(), indent=2))
