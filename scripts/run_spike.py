"""Prototype spike: sample data build plus A2 HAR-RV baseline.

The original spike plan also includes the cheapest FinBERT variant. That model
configuration is not implemented yet, so this script enforces the current
minimum paid-run gate: data build, price baseline training, evaluation, and a
written report. It exits non-zero if any stage fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.utils.paths import data_path  # noqa: E402

RUN_ID = "A2_har_rv_sample_combined_seed2026"
REPORT_PATH = REPO_ROOT / "results" / "spike_report.md"


def main() -> int:
    _run(["scripts/build_dataset.py", "--config", "configs/data/sample.yaml"])
    _run(
        [
            "scripts/train.py",
            "--model",
            "A2_har_rv",
            "--dataset",
            "sample",
            "--disclosure",
            "combined",
            "--seed",
            "2026",
        ]
    )
    _run(["scripts/evaluate.py", "--run-id", RUN_ID])
    _write_report()
    print(f"Wrote {REPORT_PATH}")
    return 0


def _run(args: list[str]) -> None:
    command = [sys.executable, *args]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _write_report() -> None:
    processed_meta = json.loads(
        (data_path("processed", "sample") / "_meta.json").read_text(encoding="utf-8")
    )
    run_dir = REPO_ROOT / "results" / "runs" / RUN_ID
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    test_metrics = [row for row in metrics if row["split"] == "test"]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Sample Spike Report",
                "",
                f"- Generated at UTC: {datetime.now(UTC).isoformat()}",
                "- Dataset: sample",
                f"- Filings: {processed_meta['counts']['filings']}",
                f"- Aligned rows: {processed_meta['counts']['aligned_rows']}",
                "- Model: A2_har_rv",
                "- Disclosure subset: combined",
                "",
                "## Test Metrics",
                "",
                "| Horizon | n | MAE | RMSE | R2 | QLIKE |",
                "|---:|---:|---:|---:|---:|---:|",
                *[_metric_row(row) for row in test_metrics],
                "",
                "## Gate Status",
                "",
                "- Sample data pipeline: PASS",
                "- A2 HAR-RV train/evaluate: PASS",
                "- FinBERT S1: NOT RUN - model config not implemented yet",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _metric_row(row: dict) -> str:
    return (
        f"| {row['horizon_days']} | {row['n']} | "
        f"{row['mae']:.6f} | {row['rmse']:.6f} | {row['r2']:.6f} | {row['qlike']:.6f} |"
    )


if __name__ == "__main__":
    raise SystemExit(main())
