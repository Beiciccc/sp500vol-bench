"""Compare metrics across multiple training runs.

Loads metrics.json from each run_dir under results/runs/ and prints a
side-by-side table. Optionally filters by split / disclosure subset.

Usage:
    python scripts/compare_runs.py --dataset dry_run_medium
    python scripts/compare_runs.py --dataset dry_run_medium --split test --disclosure combined
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results" / "runs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dry_run_medium")
    parser.add_argument("--split", default="test")
    parser.add_argument("--disclosure", default="combined")
    parser.add_argument("--metrics", nargs="+", default=["mae", "rmse", "r2", "qlike"])
    args = parser.parse_args()

    rows: list[dict] = []
    for run_dir in sorted(RUNS_DIR.glob(f"*_{args.dataset}_*_seed*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        model_id, _, _ = _parse_run_name(run_dir.name, args.dataset)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for row in metrics:
            if row["split"] != args.split:
                continue
            if row["disclosure_subset"] != args.disclosure:
                continue
            rows.append({"model": model_id, "horizon": int(row["horizon_days"]), **row})

    if not rows:
        print(f"No matching runs found for dataset={args.dataset} split={args.split}")
        return 1

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="model",
        columns="horizon",
        values=args.metrics,
        aggfunc="first",
    )
    pivot = pivot.reindex(sorted(pivot.index, key=_model_sort_key))
    print(f"\nDataset={args.dataset} split={args.split} disclosure={args.disclosure}\n{'=' * 60}")
    print(pivot.round(4).to_string())
    return 0


def _parse_run_name(name: str, dataset: str) -> tuple[str, str, int]:
    """Best-effort parse of {model_id}_{dataset}_{disclosure}_seed{N}."""
    parts = name.split(f"_{dataset}_")
    if len(parts) != 2:
        return name, "unknown", -1
    model_id, rest = parts
    pieces = rest.rsplit("_seed", maxsplit=1)
    disclosure = pieces[0] if len(pieces) > 1 else "unknown"
    seed = int(pieces[1]) if len(pieces) > 1 and pieces[1].isdigit() else -1
    return model_id, disclosure, seed


def _model_sort_key(model_id: str) -> tuple[str, str]:
    """Sort A1/A2/.../B1/.../C1/... preserving block order."""
    if not model_id or len(model_id) < 2:
        return ("Z", model_id)
    block = model_id[0]
    return (block, model_id)


if __name__ == "__main__":
    raise SystemExit(main())
