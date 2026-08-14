"""ROW 2B — FIRM-DEMEANED-TARGET retraining of C2 FinBERT S1 (GPU).

Round-3 remediation (results/REVIEW_ROUND3_FRESH_PANEL.md, MUST-RUN row 2,
"levels-only objective manufactures the identity confound"): retrain the
headline encoder on y' = log RV - firm_mean_train(log RV), so the model can
only earn loss reduction from WITHIN-FIRM variation — a firm-dummy solution
is worth zero by construction.

Mechanics (zero changes to the model classes):
  - firm_mean_train m(firm, horizon) = mean over TRAIN-SPLIT rows only of
    log(label + 1e-6); firms absent from train fall back to the global train
    mean per horizon (disclosed; counts in config.json).
  - The neural models take LEVEL targets and log internally with eps=1e-6, so
    demeaning is applied multiplicatively: y_input = y * exp(-m). Internally
    log(y*exp(-m) + 1e-6) = log(y + 1e-6*exp(m)) - m — exactly the demeaned
    log target up to a negligible eps rescale (disclosed in config.json).
    Val targets are demeaned the same way, so early stopping selects on
    demeaned-scale val loss (consistent objective).
  - At predict time the firm mean is added back (pred_level = model_output *
    exp(m), identical to exp(raw + m)), so predictions.parquet stays in level
    units and flows through the standard M1 analysis unchanged.

Reduced form (disclosed): seed 2026 only; disclosures long_form and
event_driven. Run dirs: C2dm_finbert_s1_full_<disclosure>_seed2026.

Usage:
    python scripts/experiments/row2_demeaned/train_demeaned.py --disclosure long_form
    # CPU dry run (tiny subset, capped epochs, batch 8):
    ... train_demeaned.py --disclosure long_form --limit 8 --limit-epochs 1
    # data-pipeline-only test (no torch model built):
    ... train_demeaned.py --disclosure long_form --check-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.utils import (  # noqa: E402
    CostTracker,
    configure_logging,
    get_logger,
    seed_everything,
    write_env_snapshot,
)

# reuse the canonical pipeline from scripts/train.py (single source of truth)
_spec = importlib.util.spec_from_file_location("train_base", REPO_ROOT / "scripts" / "train.py")
train_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_base)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _textcache import ensure_texts_available  # noqa: E402

_EPS = 1e-6  # matches sp500vol.models.neural_text _EPSILON (internal log eps)


def dm_run_model_id(model: str) -> str:
    """C2_finbert_s1 -> C2dm_finbert_s1 (block prefix + 'dm')."""
    block, rest = model.split("_", 1)
    return f"{block}dm_{rest}"


def build_firm_means(
    data: pd.DataFrame, firm_col: str
) -> tuple[dict[tuple, float], dict[int, float], pd.DataFrame]:
    """Per-(firm, horizon) mean of log(label+eps) from TRAIN-SPLIT rows ONLY.

    Returns (firm_means, global_means_per_horizon, train_frame_used).
    """
    train = data[data["split"] == "train"].copy()
    if train.empty:
        raise ValueError("no train rows — cannot build firm means")
    # HARD train-only guard: the frame the means are computed from contains
    # train rows exclusively (no val/test labels can enter the mean).
    assert set(train["split"].unique()) == {"train"}, "firm-mean frame leaked non-train rows"
    train["_logy"] = np.log(train["label_realised_vol"].to_numpy(dtype=float) + _EPS)
    fm = train.groupby([firm_col, "horizon_days"], sort=False)["_logy"].mean()
    gm = train.groupby("horizon_days", sort=False)["_logy"].mean()
    firm_means = {(str(k[0]), int(k[1])): float(v) for k, v in fm.items()}
    global_means = {int(k): float(v) for k, v in gm.items()}
    return firm_means, global_means, train


def row_means(
    data: pd.DataFrame,
    firm_means: dict[tuple, float],
    global_means: dict[int, float],
    firm_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row firm mean m (train-only table; global-train-mean fallback).

    Returns (m, fallback_mask)."""
    firms = data[firm_col].astype(str).to_numpy()
    hs = data["horizon_days"].astype(int).to_numpy()
    m = np.empty(len(data), dtype=float)
    fallback = np.zeros(len(data), dtype=bool)
    for i, (f, h) in enumerate(zip(firms, hs, strict=True)):
        key = (f, int(h))
        if key in firm_means:
            m[i] = firm_means[key]
        else:
            m[i] = global_means[int(h)]
            fallback[i] = True
    return m, fallback


def assert_train_only(data: pd.DataFrame, firm_means: dict, firm_col: str) -> None:
    """Prove the firm-mean join is train-only; print the assertion evidence."""
    # (1) recomputing on a frame with ALL val/test rows dropped is identical
    dropped = data[data["split"] == "train"]
    fm2, _, _ = build_firm_means(dropped, firm_col)
    assert fm2 == firm_means, "firm means change when val/test rows are removed -> leakage"
    # (2) for every (firm, horizon) that ALSO has val/test rows, the mean computed
    #     on the FULL frame (train+val+test) must differ — proves the table used
    #     is not the full-frame (leaky) one. Only decidable when splits overlap
    #     in firms (tiny --limit subsets may have zero overlap).
    nontrain = data[data["split"] != "train"]
    overlap = {
        (str(f), int(h))
        for f, h in zip(nontrain[firm_col], nontrain["horizon_days"], strict=True)
    } & set(firm_means)
    if overlap:
        full = data.copy()
        full["_logy"] = np.log(full["label_realised_vol"].to_numpy(dtype=float) + _EPS)
        fm_full = {
            (str(k[0]), int(k[1])): float(v)
            for k, v in full.groupby([firm_col, "horizon_days"], sort=False)["_logy"]
            .mean()
            .items()
        }
        diffs = sum(1 for k in overlap if not np.isclose(firm_means[k], fm_full[k]))
        assert diffs > 0, "train-only means identical to full-frame means — suspicious"
        print(
            f"ASSERT train-only firm-mean join: PASS — means from "
            f"{int((data['split'] == 'train').sum())} train rows only; "
            f"{len(nontrain)} val/test rows untouched; {diffs}/{len(overlap)} "
            f"(firm,horizon) cells with val/test rows differ from the full-frame "
            f"(leaky) version, confirming no val/test labels entered."
        )
    else:
        print(
            "ASSERT train-only firm-mean join: PASS — recomputation on the "
            "train-only frame is identical; no (firm,horizon) overlap between "
            "train and val/test in this subset, so the leaky-version contrast "
            "is vacuous (expected only under tiny --limit subsets)."
        )


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="C2_finbert_s1")
    parser.add_argument("--dataset", default="full")
    parser.add_argument(
        "--disclosure", required=True, choices=["long_form", "event_driven", "combined"]
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--firm-col", default="cik")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="CPU dry run: keep N rows per (split, horizon), cap epochs, batch<=8.",
    )
    parser.add_argument("--limit-epochs", type=int, default=1)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Data-pipeline test only: build the demeaned target + one batch and "
        "assert the firm-mean join is train-only. No torch model is built.",
    )
    args = parser.parse_args()

    configure_logging("INFO")
    log = get_logger("row2.train_demeaned")
    seed_everything(args.seed)

    dm_model = dm_run_model_id(args.model)
    run_id = f"{dm_model}_{args.dataset}_{args.disclosure}_seed{args.seed}"
    if args.limit:
        run_id += f"_limit{args.limit}"

    # --- canonical data pipeline (identical to scripts/train.py) -------------
    model_cfg = train_base._load_yaml(
        REPO_ROOT / "configs" / "models" / f"{args.model}.yaml"
    )
    data = train_base._load_dataset(args.dataset)
    data = train_base._filter_disclosure(data, args.disclosure)
    data = train_base._assign_splits(data, args.dataset)
    data = train_base._drop_invalid_rows(data)
    if args.limit:
        data = train_base._smoke_subset(data, args.limit)
        train_base._cap_smoke_epochs(model_cfg, args.limit_epochs)
        tr_cfg = model_cfg.get("training", {})
        tr_cfg["batch_size"] = min(8, int(tr_cfg.get("batch_size", 8)))
        tr_cfg["mixed_precision"] = "no"
        tr_cfg["dataloader_num_workers"] = 0
        tr_cfg["dataloader_persistent_workers"] = False
        tr_cfg["dataloader_pin_memory"] = False
        tr_cfg["dataloader_prefetch_factor"] = None
        log.info("limit mode", rows=len(data), epochs=args.limit_epochs)
    train_base._validate_trainable(data)

    # --- firm-demeaned target --------------------------------------------------
    firm_means, global_means, train_frame = build_firm_means(data, args.firm_col)
    assert_train_only(data, firm_means, args.firm_col)
    m, fallback = row_means(data, firm_means, global_means, args.firm_col)
    n_fb = {
        s: int(fallback[(data["split"] == s).to_numpy()].sum())
        for s in ("train", "val", "test")
    }
    assert n_fb["train"] == 0, "a train row fell back to the global mean — join bug"
    print(
        f"firm means: {len(firm_means)} (firm,horizon) cells from "
        f"{len(train_frame)} train rows; global-mean fallback rows "
        f"(firms unseen in train): val={n_fb['val']} test={n_fb['test']}"
    )

    y_level = data["label_realised_vol"].to_numpy(dtype=float)
    y_demeaned_level = y_level * np.exp(-m)  # log(y')+... == log(y)-m up to eps

    if args.check_only:
        tr_mask = (data["split"] == "train").to_numpy()
        batch_df = data[tr_mask].iloc[:8]
        batch_m = m[tr_mask][:8]
        batch_target = np.log(y_demeaned_level[tr_mask][:8] + _EPS)
        ref = np.log(y_level[tr_mask][:8] + _EPS) - batch_m
        assert np.allclose(batch_target, ref, atol=1e-3), "demeaned batch != log(y)-m"
        print(
            "CHECK-ONLY batch (first 8 train rows):\n"
            + pd.DataFrame(
                {
                    "cik": batch_df[args.firm_col].to_numpy(),
                    "horizon": batch_df["horizon_days"].to_numpy(),
                    "label_rv": np.round(y_level[tr_mask][:8], 4),
                    "firm_mean_logrv": np.round(batch_m, 4),
                    "demeaned_log_target": np.round(batch_target, 4),
                }
            ).to_string(index=False)
        )
        print(f"CHECK-ONLY PASS: demeaned target == log(label)-firm_mean_train "
              f"(max |diff| = {float(np.max(np.abs(batch_target - ref))):.2e})")
        return 0

    text_source = ensure_texts_available(str(data["text_path"].iloc[0]))
    log.info("text source", source=text_source)

    # --- standard run-dir plumbing (train.py conventions) -----------------------
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_env_snapshot(run_dir)
    tracker = CostTracker(run_dir=run_dir)
    log.info("training start", model=dm_model, run_dir=str(run_dir),
             disclosure=args.disclosure, seed=args.seed)

    with tracker.timed("training"):
        model = train_base._build_model(
            args.model, model_cfg, dataset=args.dataset, run_dir=run_dir, seed=args.seed
        )
        tr_mask = (data["split"] == "train").to_numpy()
        val_mask = (data["split"] == "val").to_numpy()
        train_rows = data[tr_mask].copy()
        val_rows = data[val_mask].copy()
        model.fit(
            train_rows,
            y_demeaned_level[tr_mask],
            X_val=val_rows,
            y_val=y_demeaned_level[val_mask] if not val_rows.empty else None,
        )

        val_curves = getattr(model, "val_curves_", None)
        if val_curves:
            (run_dir / "val_curves.json").write_text(
                json.dumps({str(k): v for k, v in val_curves.items()}, indent=2,
                           default=str),
                encoding="utf-8",
            )

        predictions = data.copy()
        # model output is exp(demeaned log-RV); add the firm mean back
        # (exp(raw)*exp(m) == exp(raw+m)) so predictions stay in LEVEL units.
        predictions["prediction_realised_vol"] = model.predict(predictions) * np.exp(m)
        predictions["run_id"] = run_id
        predictions["model_id"] = dm_model
        predictions["dataset"] = args.dataset
        predictions["seed"] = args.seed
        predictions["disclosure_subset"] = args.disclosure
        predictions["feature_rv_1d"] = train_base._feature_rv_1d(predictions)
        cols = train_base._prediction_columns(predictions)
        predictions[cols].to_parquet(run_dir / "predictions.parquet", index=False)
        model.save(run_dir / "model.pkl")

        # firm-mean table for audit/reuse (train-only by construction)
        pd.DataFrame(
            [
                {"firm": k[0], "horizon_days": k[1], "mean_log_rv_train": v}
                for k, v in firm_means.items()
            ]
        ).to_parquet(run_dir / "firm_means_train.parquet", index=False)

        metrics = train_base._metrics_by_group(predictions)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                              encoding="utf-8")
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "model": dm_model,
                    "base_model": args.model,
                    "dataset": args.dataset,
                    "disclosure": args.disclosure,
                    "seed": args.seed,
                    "limit": args.limit,
                    "model_config": model_cfg,
                    "demeaning": {
                        "target": "log RV - firm_mean_train(log RV)",
                        "firm_col": args.firm_col,
                        "per_horizon": True,
                        "eps": _EPS,
                        "mechanics": (
                            "multiplicative: y_input = y * exp(-m); model logs "
                            "internally (eps=1e-6) so the training target equals "
                            "log(y + eps*exp(m)) - m, the demeaned log target up "
                            "to a negligible eps rescale; predictions restored to "
                            "level units via * exp(m) before writing."
                        ),
                        "n_firm_horizon_cells": len(firm_means),
                        "n_train_rows_used": int(len(train_frame)),
                        "fallback_global_train_mean_rows": n_fb,
                        "reduced_form_note": (
                            "seed 2026 only, disclosures long_form + event_driven "
                            "(row-2 reduced form, disclosed)"
                        ),
                    },
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        log.info("outputs written", predictions=len(predictions))

    tracker.write_summary()
    log.info("training done", total_cost_usd=tracker.total_cost_usd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
