"""Predict-only pass for a selected HPO winner: REAL test rows included.

Loads the per-horizon checkpoints an ASHA trial already trained
(results/hpo/<task>/<rid>/checkpoints/horizon_<h>.pt), prepares the FULL panel
WITHOUT the search-stage test firewall, and writes raw predictions for every
row (train / val / test with their true split labels) to
results/hpo/<task>/<rid>/predictions_fulltest.parquet.

# =============================================================================
# DISCIPLINE — READ BEFORE TOUCHING THIS FILE
#
# This script generates predictions ONLY. It is deliberately structured so it
# CANNOT leak test performance:
#   * it never calls model.fit() — a missing/mismatched checkpoint is FATAL,
#     never a silent retrain;
#   * it computes, prints, and stores NO accuracy / error / loss statistic of
#     any kind, on ANY split — not on train, not on val, not on test;
#   * the output parquet carries exactly five pre-registered columns
#     [accession, horizon_days, split, label_realised_vol,
#      prediction_realised_vol]; labels pass through UNTOUCHED so the later
#     controlled step can consume them — storing them is fine, comparing them
#     to predictions here is not;
#   * the only numbers this script prints are row COUNTS, config values, and
#     integrity hashes.
# The single pre-registered test evaluation happens later, in a separate,
# controlled step. Do not add any label-vs-prediction computation here.
# =============================================================================
"""
from __future__ import annotations

# --- thread caps BEFORE numpy/torch are imported (the box overrides these by
# --- exporting its own values; setdefault never clobbers an existing export).
import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "2")

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))


def _load_asha():
    """Import the pre-registered ASHA harness by file path (same-dir, non-package).

    Reused from it: TASKS, rungs_for, trial_cfg, prepare_data, _train_mod,
    HPO_ROOT — i.e. exactly the code paths the search itself ran, so the
    config build and the firewall fingerprints are identical by construction.
    None of its scoring helpers are ever called from here.
    """
    spec = importlib.util.spec_from_file_location("asha_hpo_for_predict", HERE / "asha_hpo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OUT_COLS = [
    "accession",
    "horizon_days",
    "split",
    "label_realised_vol",
    "prediction_realised_vol",
]


def _rid(task: str, trial: int, rung: int, seed: int) -> str:
    rid = f"{task}_trial{trial:03d}_rung{rung}"
    if seed != 2026:
        rid += f"_s{seed}"
    return rid


def _fatal(msg: str) -> None:
    raise SystemExit(f"[predict_winner_test] FATAL: {msg}")


def _load_full_panel(train_mod, disclosure: str):
    """FULL panel: the exact loader chain asha_hpo.prepare_data runs
    (train.py: _load_dataset -> _filter_disclosure -> _assign_splits ->
    _drop_invalid_rows) with the two firewall lines (test-drop + val_select
    relabel) deliberately NOT executed. All rows keep their TRUE split labels.
    """
    data = train_mod._load_dataset("full")
    data = train_mod._filter_disclosure(data, disclosure)
    data = train_mod._assign_splits(data, "full")
    data = train_mod._drop_invalid_rows(data)
    return data


def _meta_diff(expected: dict, actual: dict) -> str:
    keys = sorted(set(expected) | set(actual))
    lines = []
    for k in keys:
        e, a = expected.get(k, "<absent>"), actual.get(k, "<absent>")
        if e != a:
            lines.append(f"    {k}: expected={e!r}  checkpoint={a!r}")
    return "\n".join(lines) or "    (dicts differ but no key-level diff found)"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Predict-only pass for an HPO winner: loads the trial's per-horizon "
            "checkpoints and writes predictions for ALL rows (train/val/test, true "
            "split labels) to results/hpo/<task>/<rid>/predictions_fulltest.parquet. "
            "Generates predictions only — computes no accuracy/error statistic on any split."
        )
    )
    ap.add_argument("--task", required=True, choices=["T1a", "T1c"])
    ap.add_argument("--trial", type=int, required=True, help="trial id from trials.json")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rung", type=int, default=2)
    ap.add_argument(
        "--smoke",
        type=int,
        default=None,
        metavar="N",
        help="cheap validation run: keep N rows per (split, horizon); writes "
        "predictions_fulltest_smoke.parquet instead of the real artifact",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="preflight only: parse trial, build cfg, prepare data, verify firewall "
        "fingerprints and checkpoint fingerprints, then STOP before predicting. "
        "Exit 0 if everything is in place, exit 3 listing what is missing.",
    )
    args = ap.parse_args()

    asha = _load_asha()  # imports torch/pandas AFTER the thread caps above
    import numpy as np

    from sp500vol.models.neural_text import _train_utils as train_utils
    from sp500vol.utils import seed_everything

    problems: list[str] = []

    def _problem(msg: str) -> None:
        if args.dry_run:
            problems.append(msg)
            print(f"[dry-run] MISSING/MISMATCH: {msg}")
        else:
            _fatal(msg)

    t = asha.TASKS[args.task]
    rungs = asha.rungs_for(args.task)
    if not 0 <= args.rung < len(rungs):
        _fatal(f"--rung {args.rung} out of range for {args.task} (rungs={rungs})")
    max_epochs = rungs[args.rung]
    rid = _rid(args.task, args.trial, args.rung, args.seed)
    run_dir = asha.HPO_ROOT / args.task / rid
    ckpt_dir = run_dir / "checkpoints"
    print(f"[predict_winner_test] rid={rid}")
    print(f"[predict_winner_test] run_dir={run_dir}")

    # --- trial dict: same source of truth as asha_hpo (trials.json) ----------
    trials_path = asha.HPO_ROOT / args.task / "trials.json"
    if not trials_path.exists():
        _fatal(f"{trials_path} not found — run the plan stage / sync results first")
    trials = {tr["trial"]: tr for tr in json.loads(trials_path.read_text())}
    if args.trial not in trials:
        _fatal(f"trial {args.trial} not in {trials_path} (have {sorted(trials)})")
    trial = trials[args.trial]
    print(f"[predict_winner_test] trial cfg: { {k: v for k, v in trial.items() if k != 'trial'} }")

    if not run_dir.exists():
        _problem(f"run dir {run_dir} does not exist (was this trial trained?)")
    if not ckpt_dir.exists():
        _problem(f"checkpoint dir {ckpt_dir} does not exist")

    # --- model: identical construction path to run_trial ---------------------
    seed_everything(args.seed)
    train_mod = asha._train_mod()
    cfg = asha.trial_cfg(train_mod, t["model"], trial, max_epochs)
    model = train_mod._build_model(t["model"], cfg, dataset="full", run_dir=run_dir, seed=args.seed)
    if t.get("fusion"):  # parity with run_trial (fusion ctors predate the objective kwarg)
        model.objective = trial["objective"]
    if not getattr(model, "checkpoint", False) or getattr(model, "checkpoint_dir", None) is None:
        _fatal("model was built without checkpointing — nothing to load")

    # --- data: FULL panel, firewall BYPASSED (true split labels kept) --------
    full = _load_full_panel(train_mod, t["disclosure"])
    n_test_full = int((full["split"] == "test").sum())
    split_counts = full["split"].value_counts().to_dict()
    print(f"[predict_winner_test] full panel rows by split: {split_counts}")
    if n_test_full == 0:
        _fatal("full panel contains no split=='test' rows — nothing to do")

    # --- firewall fingerprints: recompute with the ORIGINAL prepare_data -----
    # (proves this process sees byte-identical data to the training run)
    fw, manifest, n_test_dropped = asha.prepare_data(
        train_mod, t["model"], t["disclosure"], rung0_frac=t.get("rung0_frac"), rung=args.rung
    )
    if n_test_dropped != n_test_full:
        _fatal(
            f"test-row count mismatch between full prep ({n_test_full}) and "
            f"prepare_data ({n_test_dropped}) — data loading drifted"
        )
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("manifest_sha256") != manifest:
            _problem(
                f"manifest mismatch: summary.json has {summary.get('manifest_sha256')}, "
                f"recomputed {manifest} — the trial trained on different data"
            )
        if int(summary.get("n_test_dropped", -1)) != n_test_full:
            _problem(
                f"n_test mismatch: summary.json dropped {summary.get('n_test_dropped')} "
                f"test rows, full panel has {n_test_full}"
            )
        else:
            print(f"[predict_winner_test] manifest + n_test ({n_test_full}) match summary.json")
    else:
        _problem(f"{summary_path} not found — cannot verify the trial's data fingerprint")

    # --- per-horizon train-row counts (checkpoint fingerprint needs n_train) -
    fw_train = fw[fw["split"] == "train"]
    n_train_by_h = fw_train.groupby(fw_train["horizon_days"].astype(int)).size().to_dict()
    horizons = sorted(full["horizon_days"].astype(int).unique().tolist())
    print(f"[predict_winner_test] horizons={horizons}  n_train per horizon={n_train_by_h}")

    # --- checkpoint loading ---------------------------------------------------
    # Mirrors _train_utils.maybe_load_horizon_checkpoint / save_horizon_checkpoint:
    # payload = {"meta": checkpoint_meta(...), "state": {"encoder_state", "head_state"}}
    # at checkpoint_dir/horizon_<h>.pt. STRICT variant: where fit() would fall back
    # to retraining on any mismatch, this script aborts instead.
    ckpt_paths: dict[int, Path] = {}
    for h in horizons:
        if h not in n_train_by_h:
            _fatal(f"horizon {h} has no train rows — no checkpoint can exist for it")
        path = train_utils.horizon_checkpoint_path(model, h)
        if path is None:
            _fatal(f"could not derive checkpoint path for horizon {h}")
        ckpt_paths[h] = path
        if not path.exists():
            _problem(f"checkpoint missing: {path}")
            continue
        payload = train_utils._torch_load(path)
        if not isinstance(payload, dict) or "meta" not in payload or "state" not in payload:
            _problem(f"checkpoint payload invalid (no meta/state): {path}")
            continue
        expected = train_utils.checkpoint_meta(model, horizon=h, n_train=int(n_train_by_h[h]))
        if payload["meta"] != expected:
            _problem(
                f"checkpoint fingerprint mismatch for horizon {h} ({path}):\n"
                + _meta_diff(expected, payload["meta"])
            )
            continue
        if not isinstance(payload["state"], dict):
            _problem(f"checkpoint state invalid for horizon {h}: {path}")
            continue
        model.models_[h] = payload["state"]
        print(f"[predict_winner_test] loaded horizon {h} checkpoint: {path.name}")

    if args.dry_run:
        if problems:
            print(f"[dry-run] {len(problems)} problem(s) — see above")
            return 3
        print("[dry-run] all artifacts present, fingerprints match — ready to predict")
        return 0

    # --- token cache: pin one cache across the per-horizon predict loop ------
    # (same mechanism run_trial pins; tokenisation is horizon-independent)
    if hasattr(model, "_new_token_cache"):
        model._tok_cache = model._new_token_cache()
        model._tok_cache_pinned = True

    # --- predict ALL rows ------------------------------------------------------
    pred = full.copy()
    if args.smoke is not None:
        pred = (
            pred.groupby(["split", "horizon_days"], group_keys=False)
            .head(args.smoke)
            .reset_index(drop=True)
        )
        print(f"[predict_winner_test] SMOKE: kept {len(pred)} rows "
              f"({args.smoke} per (split, horizon))")
    pred["prediction_realised_vol"] = model.predict(pred)

    # --- output guards (counts + finiteness only — never label-vs-prediction) -
    out = pred[OUT_COLS].copy()
    if list(out.columns) != OUT_COLS:
        _fatal(f"output columns drifted: {list(out.columns)}")
    n_test_out = int((out["split"] == "test").sum())
    if n_test_out == 0:
        _fatal("output contains no split=='test' rows")
    if args.smoke is None and n_test_out != n_test_full:
        _fatal(f"output test-row count {n_test_out} != full panel test-row count {n_test_full}")
    if not np.isfinite(out["prediction_realised_vol"].to_numpy(dtype=float)).all():
        _fatal("non-finite predictions in output")

    fname = "predictions_fulltest_smoke.parquet" if args.smoke is not None else "predictions_fulltest.parquet"
    out_path = run_dir / fname
    out.to_parquet(out_path, index=False)

    meta = {
        "rid": rid,
        "task": args.task,
        "trial": args.trial,
        "seed": args.seed,
        "rung": args.rung,
        "epochs_cap": max_epochs,
        "smoke": args.smoke,
        "manifest_sha256": manifest,
        "rows_out_by_split": out["split"].value_counts().to_dict(),
        "n_test_rows": n_test_out,
        "horizons": horizons,
        "checkpoints": {str(h): str(p) for h, p in ckpt_paths.items()},
        "output": str(out_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_name = fname.replace(".parquet", "_meta.json")
    (run_dir / meta_name).write_text(json.dumps(meta, indent=2))

    print(f"[predict_winner_test] wrote {len(out)} rows -> {out_path}")
    print(f"[predict_winner_test] rows by split: {meta['rows_out_by_split']}")
    print("[predict_winner_test] predictions only — no statistic was computed on any split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
