"""ROW 3 — validation-tuned challenger arm (round-3 review remediation).

Defuses the "fixed recipe manufactured the null" objection
(results/REVIEW_ROUND3_FRESH_PANEL.md, EXPERIMENT-FREEZE row 3): a disclosed,
reduced-form hyperparameter grid over the two strongest fine-tuned challengers,
tuned on the VALIDATION split only; test predictions are written once, for the
selected config per (model, disclosure).

Grid (disclosed):
  * C2 FinBERT-S1 on long_form AND event_driven — lr in {5e-6, 1e-5, 2e-5},
    max 5 epochs, early stopping on validation loss (patience 1, min_delta 0),
    checkpoint selection by best-val epoch; batch/precision/warmup/weight-decay
    exactly as the archived recipe (configs/models/C2_finbert_s1.yaml).
  * D2 gated fusion on long_form — same 3 lrs + early stopping.

Selection: per (model, disclosure) the config minimising POOLED VALIDATION
QLIKE — vol-unit convention, qlike(y_true^2, y_pred^2) over all val rows and
horizons — is selected; its predictions (standard schema, all splits incl.
test) are written to
  results/runs/C2t_finbert_s1_full_<disclosure>_seed<seed>
  results/runs/D2t_gated_fusion_full_long_form_seed<seed>
and EVERY config's val-QLIKE + epochs-ran go to the tuning-audit CSV
  results/tables/row3_tuning_grid.csv
so the paper can show the sweep, not just the winner.

Usage:
  # full grid then selection (single GPU):
  python scripts/experiments/row3_tuned/tune_challengers.py --stage all
  # split across GPUs (see launch_row3_gpu.sh): shard i of n, then select once:
  CUDA_VISIBLE_DEVICES=0 python ... --stage train --shard 0 --num-shards 2
  CUDA_VISIBLE_DEVICES=1 python ... --stage train --shard 1 --num-shards 2
  python ... --stage select
  # CPU tiny dry run (proves loop + early stopping + selection + writing):
  CUDA_VISIBLE_DEVICES="" python ... --stage all --tag dryrun --limit 100 \
      --epochs 3 --batch-size 8 --precision no --max-length 64 --dl-workers 0
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.evaluation.metrics import qlike
from sp500vol.utils import (
    CostTracker,
    configure_logging,
    get_logger,
    seed_everything,
    write_env_snapshot,
)

RUNS_DIR = REPO_ROOT / "results" / "runs"
TABLES_DIR = REPO_ROOT / "results" / "tables"

# --- disclosed grid ---------------------------------------------------------
ARMS = [
    # (base model_id, disclosure, tuned model_id for the selected run dir)
    ("C2_finbert_s1", "long_form", "C2t_finbert_s1"),
    ("C2_finbert_s1", "event_driven", "C2t_finbert_s1"),
    ("D2_gated_fusion", "long_form", "D2t_gated_fusion"),
]
LRS = (5e-6, 1e-5, 2e-5)
# Row-3 protocol overrides (everything else stays as the archived recipe):
TUNED_TRAINING = {
    "max_epochs": 5,
    "early_stopping": True,
    "es_patience": 1,
    "es_min_delta": 0.0,
}
SELECTION_CRITERION = "min pooled validation QLIKE (vol-unit), all horizons"


def _load_train_module():
    """Import scripts/train.py (the archived-recipe pipeline) for its helpers:
    _load_yaml/_load_dataset/_filter_disclosure/_assign_splits/_drop_invalid_rows/
    _validate_trainable/_build_model/_metrics_by_group/_prediction_columns/_feature_rv_1d."""
    spec = importlib.util.spec_from_file_location("row3_train", REPO_ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def grid_configs() -> list[tuple[str, str, str, float]]:
    """Fixed-order config list: (base_model, disclosure, tuned_model, lr)."""
    return [(m, d, t, lr) for (m, d, t) in ARMS for lr in LRS]


def grid_run_id(model: str, dataset: str, disclosure: str, lr: float, seed: int) -> str:
    return f"{model}_{dataset}_{disclosure}_lr{_lr_tag(lr)}_seed{seed}"


def tuned_run_id(tuned_model: str, dataset: str, disclosure: str, seed: int, tag: str) -> str:
    rid = f"{tuned_model}_{dataset}_{disclosure}_seed{seed}"
    return f"{rid}_{tag}" if tag else rid


def _grid_root(tag: str) -> Path:
    return RUNS_DIR / (f"row3_grid_{tag}" if tag else "row3_grid")


def _fingerprint(args, lr: float) -> dict:
    """What must match for a stored grid run to count as done (resume safety —
    a dry-run artefact can never satisfy the full run, and vice versa)."""
    return {
        "lr": lr,
        "dataset": args.dataset,
        "seed": args.seed,
        "tuned_training": TUNED_TRAINING,
        "limit": args.limit,
        "epochs_override": args.epochs,
        "batch_size_override": args.batch_size,
        "precision_override": args.precision,
        "max_length_override": args.max_length,
    }


def _limit_filings(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """Dry-run subset: first n unique filings (chronological) PER SPLIT, keeping
    all their horizon rows — so train/val/test and every horizon stay populated."""
    kept = []
    for _, grp in data.groupby("split"):
        acc = (
            grp[["accession", "filing_time_utc"]]
            .drop_duplicates()
            .sort_values(["filing_time_utc", "accession"])["accession"]
            .head(n)
        )
        kept.append(grp[grp["accession"].isin(set(acc))])
    return (
        pd.concat(kept)
        .sort_values(["filing_time_utc", "accession", "horizon_days"])
        .reset_index(drop=True)
    )


def _prepare_data(train_mod, args, disclosure: str) -> pd.DataFrame:
    data = train_mod._load_dataset(args.dataset)
    data = train_mod._filter_disclosure(data, disclosure)
    data = train_mod._assign_splits(data, args.dataset)
    data = train_mod._drop_invalid_rows(data)
    if args.limit:
        data = _limit_filings(data, args.limit)
    train_mod._validate_trainable(data)
    return data


def _tuned_cfg(train_mod, model_id: str, lr: float, args) -> dict:
    """Archived recipe + row-3 overrides (lr, epochs, early stopping); dry-run
    knobs (--epochs/--batch-size/--precision/--max-length/--dl-workers) last."""
    cfg = train_mod._load_yaml(REPO_ROOT / "configs" / "models" / f"{model_id}.yaml")
    tr = cfg.setdefault("training", {})
    tr["lr"] = lr
    tr.update(TUNED_TRAINING)
    if args.epochs is not None:
        tr["max_epochs"] = args.epochs
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size
    if args.precision is not None:
        tr["mixed_precision"] = args.precision
    if args.dl_workers is not None:
        tr["dataloader_num_workers"] = args.dl_workers
        tr["dataloader_persistent_workers"] = args.dl_workers > 0
    if args.max_length is not None:
        cfg.setdefault("encoder", {})["max_length"] = args.max_length
    return cfg


def _val_qlike_summary(predictions: pd.DataFrame) -> tuple[float, dict[str, float], int]:
    """Pooled + per-horizon validation QLIKE in the VOL-UNIT convention
    (qlike squares vol into variance) — the paper's primary convention."""
    val = predictions[predictions["split"] == "val"]
    if val.empty:
        raise ValueError("no validation rows — cannot select by val QLIKE")
    yt = val["label_realised_vol"].to_numpy(dtype=float)
    yp = val["prediction_realised_vol"].to_numpy(dtype=float)
    pooled = qlike(yt**2, yp**2)
    by_h: dict[str, float] = {}
    for horizon, grp in val.groupby("horizon_days"):
        y_true = grp["label_realised_vol"].to_numpy(dtype=float)
        y_pred = grp["prediction_realised_vol"].to_numpy(dtype=float)
        by_h[str(int(horizon))] = qlike(y_true**2, y_pred**2)
    return pooled, by_h, len(val)


def _curve_stats(val_curves: dict) -> tuple[dict[str, int], dict[str, int]]:
    """(epochs_ran, best_epoch) per horizon from the recorded val curves."""
    epochs_ran: dict[str, int] = {}
    best_epoch: dict[str, int] = {}
    for horizon, curve in (val_curves or {}).items():
        key = str(int(horizon)) if not isinstance(horizon, str) else horizon
        epochs_ran[key] = len(curve)
        best = [e["epoch"] for e in curve if e.get("is_best")]
        best_epoch[key] = int(best[-1]) if best else len(curve)
    return epochs_ran, best_epoch


def run_config(  # noqa: PLR0915  (single linear train->predict->summarise flow, mirrors train.py main)
    train_mod, model_id: str, disclosure: str, lr: float, args
) -> dict:
    """Train ONE grid config end-to-end (archived pipeline conventions), write a
    self-contained grid run dir, and return its summary. Resumes iff a stored
    summary carries an identical fingerprint."""
    log = get_logger("row3")
    rid = grid_run_id(model_id, args.dataset, disclosure, lr, args.seed)
    run_dir = _grid_root(args.tag) / rid
    summary_path = run_dir / "row3_summary.json"
    fingerprint = _fingerprint(args, lr)

    if summary_path.exists() and (run_dir / "predictions.parquet").exists() and not args.force:
        stored = json.loads(summary_path.read_text(encoding="utf-8"))
        if stored.get("fingerprint") == fingerprint:
            log.info("row3 config already done — skipping", run_id=rid)
            return stored
        log.info("row3 stale grid run (fingerprint mismatch) — retraining", run_id=rid)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_env_snapshot(run_dir)
    tracker = CostTracker(run_dir=run_dir)
    seed_everything(args.seed)  # re-seed per config: results identical under any sharding

    cfg = _tuned_cfg(train_mod, model_id, lr, args)
    log.info(
        "row3 config start",
        run_id=rid,
        lr=lr,
        max_epochs=cfg["training"]["max_epochs"],
        batch_size=cfg["training"].get("batch_size"),
        limit=args.limit,
    )

    with tracker.timed("training"):
        data = _prepare_data(train_mod, args, disclosure)
        model = train_mod._build_model(
            model_id, cfg, dataset=args.dataset, run_dir=run_dir, seed=args.seed
        )
        train_rows = data[data["split"] == "train"].copy()
        val_rows = data[data["split"] == "val"].copy()
        model.fit(
            train_rows,
            train_rows["label_realised_vol"].to_numpy(),
            X_val=val_rows,
            y_val=val_rows["label_realised_vol"].to_numpy() if not val_rows.empty else None,
        )

        val_curves = getattr(model, "val_curves_", None) or {}
        (run_dir / "val_curves.json").write_text(
            json.dumps({str(k): v for k, v in val_curves.items()}, indent=2, default=str),
            encoding="utf-8",
        )

        predictions = data.copy()
        predictions["prediction_realised_vol"] = model.predict(predictions)
        predictions["run_id"] = rid
        predictions["model_id"] = model_id
        predictions["dataset"] = args.dataset
        predictions["seed"] = args.seed
        predictions["disclosure_subset"] = disclosure
        predictions["feature_rv_1d"] = train_mod._feature_rv_1d(predictions)
        prediction_cols = train_mod._prediction_columns(predictions)
        predictions[prediction_cols].to_parquet(run_dir / "predictions.parquet", index=False)

        metrics = train_mod._metrics_by_group(predictions)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "model": model_id,
                    "dataset": args.dataset,
                    "disclosure": disclosure,
                    "seed": args.seed,
                    "model_config": cfg,
                    "row3": {"grid_lr": lr, "tuned_training": TUNED_TRAINING},
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    tracker.write_summary()
    val_ql, val_ql_h, n_val = _val_qlike_summary(predictions)
    epochs_ran, best_epoch = _curve_stats(val_curves)
    tr = cfg["training"]
    summary = {
        "run_id": rid,
        "model_id": model_id,
        "disclosure": disclosure,
        "dataset": args.dataset,
        "seed": args.seed,
        "lr": lr,
        "batch_size": tr.get("batch_size"),
        "max_epochs": tr.get("max_epochs"),
        "es_patience": tr.get("es_patience"),
        "es_min_delta": tr.get("es_min_delta"),
        "warmup_ratio": tr.get("warmup_ratio"),
        "weight_decay": tr.get("weight_decay"),
        "mixed_precision": tr.get("mixed_precision"),
        "val_qlike": val_ql,
        "val_qlike_by_h": val_ql_h,
        "n_val_rows": n_val,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "gpu_hours": round(tracker.total_seconds / 3600.0, 3),
        "fingerprint": fingerprint,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(
        "row3 config done",
        run_id=rid,
        val_qlike=round(val_ql, 6),
        epochs_ran=epochs_ran,
    )

    # Free GPU memory before the next sequential config.
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return summary


def stage_train(train_mod, args) -> None:
    configs = grid_configs()
    if args.num_shards > 1:
        configs = configs[args.shard :: args.num_shards]
    print(
        f"=== ROW3 train: {len(configs)} config(s) "
        f"(shard {args.shard}/{args.num_shards}, tag={args.tag or '-'}) ===",
        flush=True,
    )
    for model_id, disclosure, _, lr in configs:
        run_config(train_mod, model_id, disclosure, lr, args)


def _load_summary(args, model_id: str, disclosure: str, lr: float) -> dict:
    rid = grid_run_id(model_id, args.dataset, disclosure, lr, args.seed)
    path = _grid_root(args.tag) / rid / "row3_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"grid run {rid} has no row3_summary.json — run --stage train (all shards) first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _write_selected_run(train_mod, args, arm, best: dict) -> str:
    """Copy the winning grid config's outputs into the standard-schema tuned run
    dir (C2t_/D2t_ naming) with run_id/model_id rewritten."""
    base_model, disclosure, tuned_model = arm
    rid = tuned_run_id(tuned_model, args.dataset, disclosure, args.seed, args.tag)
    src_dir = _grid_root(args.tag) / best["run_id"]
    out_dir = RUNS_DIR / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(src_dir / "predictions.parquet")
    predictions["run_id"] = rid
    predictions["model_id"] = tuned_model
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)

    metrics = train_mod._metrics_by_group(predictions)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for aux in ("val_curves.json", "cost.json", "env.json"):
        if (src_dir / aux).exists():
            shutil.copy2(src_dir / aux, out_dir / aux)

    src_cfg = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "model": tuned_model,
                "base_model": base_model,
                "dataset": args.dataset,
                "disclosure": disclosure,
                "seed": args.seed,
                "model_config": src_cfg.get("model_config"),
                "row3_selection": {
                    "criterion": SELECTION_CRITERION,
                    "grid_lrs": list(LRS),
                    "tuned_training": TUNED_TRAINING,
                    "selected_lr": best["lr"],
                    "selected_grid_run": best["run_id"],
                    "val_qlike": best["val_qlike"],
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return rid


def stage_select(train_mod, args) -> None:
    print(f"=== ROW3 select: {len(ARMS)} arms x {len(LRS)} lrs (tag={args.tag or '-'}) ===")
    csv_rows: list[dict] = []
    horizons: list[str] = []
    for arm in ARMS:
        base_model, disclosure, _tuned_model = arm
        summaries = [_load_summary(args, base_model, disclosure, lr) for lr in LRS]
        best = min(summaries, key=lambda s: s["val_qlike"])
        rid = _write_selected_run(train_mod, args, arm, best)
        print(f"\n{base_model} / {disclosure} -> {rid} (selected lr={_lr_tag(best['lr'])})")
        print(f"  {'lr':>8} {'val_qlike':>12} {'epochs_ran':>14} {'best_epoch':>14} sel")
        for s in summaries:
            horizons = sorted(s["val_qlike_by_h"], key=int)
            eps = "/".join(str(s["epochs_ran"].get(h, "-")) for h in horizons)
            bst = "/".join(str(s["best_epoch"].get(h, "-")) for h in horizons)
            mark = "*" if s is best else ""
            print(f"  {_lr_tag(s['lr']):>8} {s['val_qlike']:>12.6f} {eps:>14} {bst:>14}  {mark}")
            csv_rows.append({**s, "selected": s is best, "tuned_run_id": rid if s is best else ""})

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = f"row3_tuning_grid_{args.tag}.csv" if args.tag else "row3_tuning_grid.csv"
    csv_path = TABLES_DIR / csv_name
    fields = [
        "model_id", "disclosure", "dataset", "seed", "lr", "batch_size", "max_epochs",
        "es_patience", "es_min_delta", "warmup_ratio", "weight_decay", "mixed_precision",
        "val_qlike",
        *[f"val_qlike_h{h}" for h in horizons],
        *[f"epochs_ran_h{h}" for h in horizons],
        *[f"best_epoch_h{h}" for h in horizons],
        "n_val_rows", "gpu_hours", "selected", "tuned_run_id", "run_id",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for s in csv_rows:
            row = {k: s.get(k) for k in fields if k in s}
            for h in horizons:
                row[f"val_qlike_h{h}"] = s["val_qlike_by_h"].get(h)
                row[f"epochs_ran_h{h}"] = s["epochs_ran"].get(h)
                row[f"best_epoch_h{h}"] = s["best_epoch"].get(h)
            row["selected"] = bool(s["selected"])
            writer.writerow(row)
    print(f"\ntuning-audit CSV -> {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["train", "select", "all"])
    parser.add_argument("--dataset", default="full")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shard", type=int, default=0, help="train: this worker's shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="train: total workers (GPUs)")
    parser.add_argument("--tag", default="", help="isolate outputs (e.g. dryrun); '' = real run")
    parser.add_argument("--force", action="store_true", help="retrain even if a grid run is done")
    # Dry-run knobs — leave ALL unset for the real (disclosed-grid) run:
    parser.add_argument("--limit", type=int, default=None, help="first N filings per split")
    parser.add_argument("--epochs", type=int, default=None, help="override max_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch_size")
    parser.add_argument("--precision", default=None, help="override mixed_precision (e.g. no)")
    parser.add_argument("--max-length", type=int, default=None, help="override encoder max_length")
    parser.add_argument("--dl-workers", type=int, default=None, help="override dataloader workers")
    args = parser.parse_args()

    configure_logging("INFO")
    train_mod = _load_train_module()
    if args.stage in ("train", "all"):
        stage_train(train_mod, args)
    if args.stage in ("select", "all"):
        if args.stage == "all" and args.num_shards > 1:
            raise SystemExit("--stage all cannot be sharded; run select separately")
        stage_select(train_mod, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
