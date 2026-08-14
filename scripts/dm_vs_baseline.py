"""Diebold-Mariano test of all available runs vs a baseline run.

Loads predictions.parquet from every run under results/runs/, aligns on
(ticker, accession, horizon_days), and runs DM with squared-error loss vs
the chosen baseline (default A2_har_rv).

Usage:
    python scripts/dm_vs_baseline.py --dataset dry_run_medium
    python scripts/dm_vs_baseline.py --dataset dry_run_medium --baseline A1_hv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.evaluation.dm_test import dm_test

RUNS_DIR = REPO_ROOT / "results" / "runs"
KEY_COLS = ["ticker", "accession", "horizon_days"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dry_run_medium")
    parser.add_argument("--disclosure", default="combined")
    parser.add_argument("--split", default="test")
    parser.add_argument("--baseline", default="A2_har_rv")
    parser.add_argument("--seed", default="seed2026")
    args = parser.parse_args()

    baseline_dir = RUNS_DIR / f"{args.baseline}_{args.dataset}_{args.disclosure}_{args.seed}"
    if not baseline_dir.exists():
        print(f"baseline run not found: {baseline_dir}")
        return 1

    baseline_preds = (
        pd.read_parquet(baseline_dir / "predictions.parquet")
        .query("split == @args.split")[
            [*KEY_COLS, "prediction_realised_vol", "label_realised_vol", "filing_time_utc"]
        ]
        .rename(columns={"prediction_realised_vol": "pred_baseline"})
    )

    pattern = f"*_{args.dataset}_{args.disclosure}_{args.seed}"
    challengers = sorted(d for d in RUNS_DIR.glob(pattern) if d.is_dir() and d != baseline_dir)
    if not challengers:
        print(f"no challenger runs found matching {pattern}")
        return 1

    print(
        f"\n=== Diebold-Mariano vs {args.baseline}"
        f" (dataset={args.dataset}, disclosure={args.disclosure}, split={args.split}) ==="
    )
    print("Positive DM = challenger WORSE than baseline. p<0.05 = significant.\n")

    for run_dir in challengers:
        preds_path = run_dir / "predictions.parquet"
        if not preds_path.exists():
            continue
        challenger_id = run_dir.name.split(f"_{args.dataset}")[0]
        challenger = (
            pd.read_parquet(preds_path)
            .query("split == @args.split")[[*KEY_COLS, "prediction_realised_vol"]]
            .rename(columns={"prediction_realised_vol": "pred_challenger"})
        )
        merged = baseline_preds.merge(challenger, on=KEY_COLS)
        if merged.empty:
            print(f"--- {challenger_id} vs {args.baseline}: no overlap")
            continue

        print(f"--- {challenger_id} vs {args.baseline} ---")
        for horizon in sorted(merged["horizon_days"].unique()):
            sub = merged[merged["horizon_days"] == horizon].sort_values("filing_time_utc")
            y = sub["label_realised_vol"].to_numpy()
            loss_c = (sub["pred_challenger"].to_numpy() - y) ** 2
            loss_b = (sub["pred_baseline"].to_numpy() - y) ** 2
            stat, p = dm_test(loss_c, loss_b, h=int(horizon))
            tag = "** p<0.01" if p < 0.01 else ("* p<0.05" if p < 0.05 else "  ns    ")
            verdict = "WORSE" if stat > 0 else "BETTER"
            mae_c = float(np.mean(np.abs(sub["pred_challenger"] - y)))
            mae_b = float(np.mean(np.abs(sub["pred_baseline"] - y)))
            print(
                f"  h={int(horizon):2d}: DM={stat:+6.3f} p={p:.4f}  {tag}  "
                f"-> {challenger_id} {verdict}   "
                f"MAE(C/B)={mae_c:.4f}/{mae_b:.4f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
