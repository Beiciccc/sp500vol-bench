"""Aggregate per-seed runs into mean±std metrics + variance-band Diebold-Mariano.

For each (disclosure, model, horizon) this collects the seeds' test-split point
metrics (MAE/RMSE/R2/QLIKE) and a per-seed Diebold-Mariano statistic (squared-error
and QLIKE loss) vs the baseline (default A2_har_rv), then reports the cross-seed
mean +/- std and how many seeds reach p<0.05.

This is the multi-seed robustness view requested for the rerun:
  * the std band reflects run-to-run (training-seed) variability;
  * DM is recomputed PER SEED (challenger_seed vs the deterministic baseline) and
    then summarised, NOT pooled across seeds — so no single lucky seed can
    manufacture significance, and "nsig/n" shows how consistent the verdict is.

Robust to partial completion: a model with only 1-2 seeds present is aggregated
over whatever exists (n_seeds is reported), so the table can be inspected while
the box is still running.

Usage:
    python scripts/analysis/aggregate_seeds.py --dataset full \
        --disclosures long_form event_driven combined --seeds 2026 2027 2028
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.evaluation.dm_test import dm_test

RUNS_DIR = REPO_ROOT / "results" / "runs"
KEY = ["ticker", "accession", "horizon_days"]
HORIZONS = (5, 10, 20)
EPS = 1e-8
METRIC_KEYS = ("mae", "rmse", "r2", "qlike")


# --- losses -----------------------------------------------------------------


def se_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    return (np.asarray(f, dtype=float) - np.asarray(y, dtype=float)) ** 2


def qlike_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE on VARIANCE, matching evaluation.metrics.all_metrics
    (which squares vol before QLIKE). y, f are VOLATILITY here and are squared to
    variance, so the per-seed DM tests the same QLIKE the headline table reports."""
    yv = np.clip(np.asarray(y, dtype=float) ** 2, EPS, None)
    fv = np.clip(np.asarray(f, dtype=float) ** 2, EPS, None)
    return yv / fv - np.log(yv / fv) - 1.0


# --- pure aggregation helpers (unit-tested) ---------------------------------


def aggregate_values(values: list) -> tuple[float, float, int]:
    """Return (mean, sample-std, n) over the finite values (std=0 when n==1)."""
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan"), 0
    spread = stdev(vals) if len(vals) > 1 else 0.0
    return mean(vals), spread, len(vals)


def aggregate_dm(per_seed: list[tuple[float, float]]) -> dict:
    """Summarise per-seed (dm_stat, p) into mean/std + #significant + direction."""
    finite = [(d, p) for d, p in per_seed if d is not None and np.isfinite(d)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "n_sig": 0, "direction": "-"}
    dms = [d for d, _ in finite]
    spread = stdev(dms) if len(dms) > 1 else 0.0
    n_sig = sum(1 for _, p in finite if p < 0.05)
    direction = "WORSE" if mean(dms) > 0 else "BETTER"
    return {
        "mean": mean(dms),
        "std": spread,
        "n": len(finite),
        "n_sig": n_sig,
        "direction": direction,
    }


# --- IO ---------------------------------------------------------------------


def _run_dir(model: str, dataset: str, disclosure: str, seed: int) -> Path:
    return RUNS_DIR / f"{model}_{dataset}_{disclosure}_seed{seed}"


def _load_test_metrics(run_dir: Path) -> dict[int, dict]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(r["horizon_days"]): {k: r.get(k) for k in METRIC_KEYS}
        for r in rows
        if r.get("split") == "test"
    }


def _load_test_preds(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "predictions.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[df["split"] == "test"]


def _discover_models(dataset: str, disclosure: str, seeds: list[int], baseline: str) -> list[str]:
    found: set[str] = set()
    for seed in seeds:
        suffix = f"_{dataset}_{disclosure}_seed{seed}"
        for run_dir in RUNS_DIR.glob(f"*{suffix}"):
            if run_dir.is_dir():
                model = run_dir.name[: -len(suffix)]
                if model != baseline:
                    found.add(model)
    return sorted(found)


# --- per-model aggregation --------------------------------------------------


def _per_seed_dm(challenger: pd.DataFrame, baseline: pd.DataFrame, horizon: int) -> dict | None:
    """SE-DM and QLIKE-DM for one (challenger, baseline) pair at one horizon."""
    bcols = baseline[[*KEY, "prediction_realised_vol", "label_realised_vol", "filing_time_utc"]]
    bcols = bcols.rename(columns={"prediction_realised_vol": "pb"})
    ccols = challenger[[*KEY, "prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "pc"}
    )
    # one_to_one: a silent many-to-many fan-out would inflate the DM sample n and
    # corrupt the statistic — fail loudly instead of miscounting.
    merged = bcols.merge(ccols, on=KEY, validate="one_to_one").sort_values("filing_time_utc")
    sub = merged[merged["horizon_days"] == horizon]
    if len(sub) < 2:
        return None
    y = sub["label_realised_vol"].to_numpy()
    pc = sub["pc"].to_numpy()
    pb = sub["pb"].to_numpy()
    return {
        "se": dm_test(se_loss(y, pc), se_loss(y, pb), h=horizon),
        "qlike": dm_test(qlike_loss(y, pc), qlike_loss(y, pb), h=horizon),
    }


def _aggregate_model(
    model: str, dataset: str, disclosure: str, seeds: list[int], base_by_seed: dict
) -> list[dict]:
    base_fallback = next(iter(base_by_seed.values()), None)
    metrics: dict[int, dict[str, list]] = {h: {k: [] for k in METRIC_KEYS} for h in HORIZONS}
    dm_se: dict[int, list] = {h: [] for h in HORIZONS}
    dm_qlike: dict[int, list] = {h: [] for h in HORIZONS}
    n_seeds = 0

    for seed in seeds:
        run_dir = _run_dir(model, dataset, disclosure, seed)
        seed_metrics = _load_test_metrics(run_dir)
        preds = _load_test_preds(run_dir)
        if not seed_metrics and preds is None:
            continue
        n_seeds += 1
        for horizon, vals in seed_metrics.items():
            if horizon in metrics:
                for k in METRIC_KEYS:
                    metrics[horizon][k].append(vals.get(k))
        baseline = base_by_seed.get(seed, base_fallback)
        if preds is None or baseline is None:
            continue
        for horizon in HORIZONS:
            dm = _per_seed_dm(preds, baseline, horizon)
            if dm is not None:
                dm_se[horizon].append(dm["se"])
                dm_qlike[horizon].append(dm["qlike"])

    if n_seeds == 0:
        return []

    rows = []
    for horizon in HORIZONS:
        r2_m, r2_s, _ = aggregate_values(metrics[horizon]["r2"])
        ql_m, ql_s, _ = aggregate_values(metrics[horizon]["qlike"])
        mae_m, mae_s, _ = aggregate_values(metrics[horizon]["mae"])
        rmse_m, rmse_s, _ = aggregate_values(metrics[horizon]["rmse"])
        se = aggregate_dm(dm_se[horizon])
        ql = aggregate_dm(dm_qlike[horizon])
        rows.append(
            {
                "disclosure": disclosure,
                "model": model,
                "horizon": horizon,
                "n_seeds": n_seeds,
                "r2_mean": r2_m,
                "r2_std": r2_s,
                "qlike_mean": ql_m,
                "qlike_std": ql_s,
                "mae_mean": mae_m,
                "mae_std": mae_s,
                "rmse_mean": rmse_m,
                "rmse_std": rmse_s,
                "se_dm_mean": se["mean"],
                "se_dm_std": se["std"],
                "se_dm_nsig": se["n_sig"],
                "se_dm_n": se["n"],
                "se_dm_dir": se["direction"],
                "qlike_dm_mean": ql["mean"],
                "qlike_dm_std": ql["std"],
                "qlike_dm_nsig": ql["n_sig"],
                "qlike_dm_n": ql["n"],
                "qlike_dm_dir": ql["direction"],
            }
        )
    return rows


# --- output -----------------------------------------------------------------


def _fmt(m: float, s: float) -> str:
    return "—" if not np.isfinite(m) else f"{m:.4f}±{s:.4f}"


def _write_md(df: pd.DataFrame, path: Path, baseline: str, seeds: list[int]) -> None:
    lines = [
        f"# Multi-seed aggregate (seeds {seeds}) vs {baseline}",
        "",
        "Per (model, horizon): cross-seed mean±std of test metrics, and Diebold-Mariano",
        "recomputed per seed (challenger vs baseline). `sig` = seeds with p<0.05 over seeds run.",
        "Positive DM = challenger WORSE than baseline.",
        "",
    ]
    for disclosure in df["disclosure"].drop_duplicates():
        sub = df[df["disclosure"] == disclosure].sort_values(["model", "horizon"])
        lines.append(f"## {disclosure}")
        lines.append("")
        lines.append(
            "| model | h | seeds | R² (mean±std) | QLIKE (mean±std) "
            "| SE-DM (mean±std) | sig | QLIKE-DM (mean±std) | sig |"
        )
        lines.append("|---|--:|--:|--:|--:|--:|:--|--:|:--|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['model']} | {int(r['horizon'])} | {int(r['n_seeds'])} "
                f"| {_fmt(r['r2_mean'], r['r2_std'])} | {_fmt(r['qlike_mean'], r['qlike_std'])} "
                f"| {_fmt(r['se_dm_mean'], r['se_dm_std'])} "
                f"| {int(r['se_dm_nsig'])}/{int(r['se_dm_n'])} {r['se_dm_dir']} "
                f"| {_fmt(r['qlike_dm_mean'], r['qlike_dm_std'])} "
                f"| {int(r['qlike_dm_nsig'])}/{int(r['qlike_dm_n'])} {r['qlike_dm_dir']} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="full")
    parser.add_argument(
        "--disclosures", nargs="+", default=["long_form", "event_driven", "combined"]
    )
    parser.add_argument("--baseline", default="A2_har_rv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--models", nargs="*", default=None, help="default: auto-discover")
    parser.add_argument(
        "--out-md", type=Path, default=REPO_ROOT / "results/tables/seed_aggregate.md"
    )
    parser.add_argument(
        "--out-csv", type=Path, default=REPO_ROOT / "results/tables/seed_aggregate.csv"
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for disclosure in args.disclosures:
        models = args.models or _discover_models(
            args.dataset, disclosure, args.seeds, args.baseline
        )
        base_by_seed = {}
        for seed in args.seeds:
            preds = _load_test_preds(_run_dir(args.baseline, args.dataset, disclosure, seed))
            if preds is not None:
                base_by_seed[seed] = preds
        for model in models:
            rows.extend(_aggregate_model(model, args.dataset, disclosure, args.seeds, base_by_seed))

    if not rows:
        print("no runs found to aggregate")
        return 1
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    _write_md(df, args.out_md, args.baseline, args.seeds)
    print(f"aggregated {len(df)} cells over seeds {args.seeds} -> {args.out_csv} , {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
