"""Cumulative GPU-hour and cost tracker.

Append-only log per run. Used to track budget against Plan B + Light Compression
target of ~$780-1080. See design/02_compute_plan.md.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import monotonic


class CostTracker:
    """Track wall-clock seconds and accumulated cloud cost for a run.

    Example:
        tracker = CostTracker(run_dir=Path("results/runs/xyz"), hourly_rate_usd=1.80)
        with tracker.timed("training"):
            train_model(...)
        tracker.write_summary()
    """

    def __init__(self, run_dir: Path, hourly_rate_usd: float | None = None) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.hourly_rate_usd = (
            hourly_rate_usd
            if hourly_rate_usd is not None
            else float(os.environ.get("HOURLY_RATE_USD", "0") or "0")
        )
        self._segments: list[dict] = []

    @contextmanager
    def timed(self, label: str) -> Iterator[None]:
        t0 = monotonic()
        try:
            yield
        finally:
            elapsed = monotonic() - t0
            self._segments.append({"label": label, "seconds": elapsed})

    @property
    def total_seconds(self) -> float:
        return sum(s["seconds"] for s in self._segments)

    @property
    def total_cost_usd(self) -> float:
        return self.total_seconds / 3600.0 * self.hourly_rate_usd

    def write_summary(self) -> None:
        summary = {
            "segments": self._segments,
            "total_seconds": self.total_seconds,
            "total_gpu_hours": round(self.total_seconds / 3600.0, 3),
            "hourly_rate_usd": self.hourly_rate_usd,
            "total_cost_usd": round(self.total_cost_usd, 3),
        }
        (self.run_dir / "cost.json").write_text(json.dumps(summary, indent=2))
