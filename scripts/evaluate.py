"""Evaluate a finished training run with full statistical protocol.

Usage:
    python scripts/evaluate.py --run-id 20260601-120000_C2_finbert_s3_combined_seed2026
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.evaluation.bootstrap import block_bootstrap_ci
from sp500vol.evaluation.metrics import all_metrics, mae, qlike, r_squared, rmse
from sp500vol.utils import configure_logging, get_logger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline", default="A2_har_rv", help="DM test reference baseline")
    parser.add_argument("--bootstrap-iter", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    configure_logging("INFO")
    log = get_logger("evaluate")
    log.info("Evaluating", run_id=args.run_id, baseline=args.baseline)

    run_dir = _resolve_run_dir(args.run_id)
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    evaluation = {
        "run_id": args.run_id,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "baseline": args.baseline,
        "bootstrap_iter": args.bootstrap_iter,
        "bootstrap_seed": args.bootstrap_seed,
        "metrics": _metrics_by_group(
            predictions,
            bootstrap_iter=args.bootstrap_iter,
            bootstrap_seed=args.bootstrap_seed,
        ),
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2),
        encoding="utf-8",
    )
    log.info("Evaluation written", output=str(run_dir / "evaluation.json"))
    return 0


def _resolve_run_dir(run_id: str) -> Path:
    candidate = Path(run_id)
    if candidate.is_absolute() or candidate.exists():
        run_dir = candidate
    else:
        run_dir = REPO_ROOT / "results" / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if not (run_dir / "predictions.parquet").exists():
        raise FileNotFoundError(run_dir / "predictions.parquet")
    return run_dir


def _metrics_by_group(
    predictions: pd.DataFrame,
    *,
    bootstrap_iter: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_cols = ["split", "disclosure_subset", "horizon_days"]
    rng = np.random.default_rng(bootstrap_seed)
    for keys, group in predictions.groupby(group_cols, dropna=False):
        y_true = group["label_realised_vol"].to_numpy(dtype=float)
        y_pred = group["prediction_realised_vol"].to_numpy(dtype=float)
        metric_values = all_metrics(y_true, y_pred)
        horizon = int(keys[2])
        rows.append(
            {
                "split": keys[0],
                "disclosure_subset": keys[1],
                "horizon_days": horizon,
                "n": len(group),
                **metric_values,
                "bootstrap_ci": _bootstrap_ci_by_metric(
                    y_true,
                    y_pred,
                    n_iter=bootstrap_iter,
                    block_size=horizon,
                    rng=rng,
                ),
            }
        )
    return rows


def _bootstrap_ci_by_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_iter: int,
    block_size: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    metric_fns = {
        "mae": mae,
        "rmse": rmse,
        "r2": r_squared,
        "qlike": lambda yt, yp: qlike(yt**2, yp**2),
    }
    out: dict[str, dict[str, float]] = {}
    for name, metric_fn in metric_fns.items():
        point, lower, upper = block_bootstrap_ci(
            y_true,
            y_pred,
            metric_fn,
            n_iter=n_iter,
            block_size=block_size,
            rng=rng,
        )
        out[name] = {"point": point, "lower": lower, "upper": upper}
    return out


if __name__ == "__main__":
    sys.exit(main())
