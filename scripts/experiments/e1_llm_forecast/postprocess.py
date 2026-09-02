"""E1 / C6+D4 — turn raw LLM outputs into standard run dirs.

Subcommands:

  build-runs     raw outputs -> results/runs/{C6_llmtext,D4_llmfused}_full_<disc>_seed2026
                 for disc in {long_form, event_driven, combined}, following the
                 NEW-MODEL OUTPUT CONVENTION (schema copied from an A2/B2 run):
                   predictions.parquet  — VAL+TEST rows only. Train rows are ABSENT by
                     design: the LLM is never fit, and the downstream M1 combiner
                     (forecast_combination.log_combo) only needs val (weights) + test
                     (evaluation). Anything that requires train rows must skip C6/D4.
                   metrics.json — LIST of dicts {split,disclosure_subset,horizon_days,
                     n,mae,rmse,r2,qlike}; 6 rows (val,test x 5/10/20). qlike is
                     VARIANCE-unit: qlike(y^2, yhat^2), verified to reproduce B2 rows.
                   config.json  — model_id, source LLM, parse/clip/fill statistics.
                 Predictions are clipped to [0.03, 3.0] annualized (LLM garbage guard);
                 the clipping rate is reported. Filings whose output never parsed are
                 filled per --on-missing (default rv22 = shrink-to-price fallback,
                 counted in config.json; use drop to omit them instead).

        python postprocess.py build-runs --raw-dir raw_outputs/ \\
            [--out-root results/runs] [--on-missing rv22|drop]

  pilot-eval     go/no-go readout after `run_inference.py run --pilot 500`:
                 parse rate, clip rate, vol-unit QLIKE vs the A1_hv naive baseline on
                 the same filings, spearman(pred, label). See go_no_go.md for gates.

        python postprocess.py pilot-eval --raw-dir raw_pilot/ \\
            --manifest manifest_valtest.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

CLIP_LO, CLIP_HI = 0.03, 3.0
EPS = 1e-8
HORIZONS = (5, 10, 20)
DISCLOSURES = ("long_form", "event_driven", "combined")
VARIANT_MODEL = {"c6_text": "C6_llmtext", "d4_fused": "D4_llmfused",
                 "c6_dateonly": "C6_dateonly", "c6_datefirm": "C6_datefirm"}
PRED_COLS = [  # exact column order of existing run predictions (B2/A2 schema)
    "run_id", "model_id", "dataset", "seed", "disclosure_subset", "split", "ticker",
    "form", "item_subtype", "accession", "filing_time_utc", "effective_trading_day",
    "horizon_days", "label_realised_vol", "prediction_realised_vol",
    "feature_rv_1d", "feature_rv_5d", "feature_rv_22d", "text_path", "metadata_path",
]


def qlike_var(y_vol: np.ndarray, f_vol: np.ndarray) -> np.ndarray:
    """VARIANCE-unit qlike, exactly as in existing metrics.json (verified vs B2)."""
    a = np.clip(np.asarray(y_vol, float) ** 2, EPS, None)
    b = np.clip(np.asarray(f_vol, float) ** 2, EPS, None)
    return a / b - np.log(a / b) - 1.0


def qlike_vol(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """VOL-unit qlike (same convention as forecast_combination.qlike / go-no-go)."""
    y = np.clip(np.asarray(y, float), EPS, None)
    f = np.clip(np.asarray(f, float), EPS, None)
    return y / f - np.log(y / f) - 1.0


def load_raw(raw_dir: str) -> pd.DataFrame:
    parts = sorted(Path(raw_dir).glob("part-*.parquet"))
    if not parts:
        raise SystemExit(f"no part-*.parquet under {raw_dir}")
    raw = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    raw = raw.drop_duplicates(subset=["text_path", "variant"], keep="last")
    return raw


def metrics_rows(pred: pd.DataFrame, disc: str) -> list[dict]:
    rows = []
    for split in sorted(pred["split"].unique()):
        for h in HORIZONS:
            g = pred[(pred["split"] == split) & (pred["horizon_days"] == h)]
            if not len(g):
                continue
            y = g["label_realised_vol"].to_numpy()
            f = g["prediction_realised_vol"].to_numpy()
            rows.append({
                "split": split, "disclosure_subset": disc, "horizon_days": int(h),
                "n": int(len(g)),
                "mae": float(np.abs(y - f).mean()),
                "rmse": float(np.sqrt(((y - f) ** 2).mean())),
                "r2": float(1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                "qlike": float(qlike_var(y, f).mean()),
            })
    return rows


def build_runs(args) -> None:
    raw = load_raw(args.raw_dir)
    print(f"raw outputs: {len(raw)} rows, variants {sorted(raw['variant'].unique())}, "
          f"parse_ok {raw['parse_ok'].mean():.3f}")
    out_root = Path(args.out_root)
    suffix = getattr(args, "model_suffix", "") or ""
    for variant, model_id in VARIANT_MODEL.items():
        model_id = model_id + suffix
        rv = raw[raw["variant"] == variant]
        if not len(rv):
            print(f"[skip] no raw rows for variant {variant}")
            continue
        for disc in DISCLOSURES:
            a2 = pd.read_parquet(
                f"results/runs/A2_har_rv_full_{disc}_seed2026/predictions.parquet")
            base = a2[a2["split"].isin(["val", "test"])].copy()
            base = base.merge(
                rv[["text_path", "vol_5d", "vol_10d", "vol_20d", "parse_ok"]],
                on="text_path", how="inner")  # inner: only filings sent to the LLM
            if not len(base):
                print(f"[skip] {model_id} {disc}: no overlap with raw outputs")
                continue
            volmap = {5: "vol_5d", 10: "vol_10d", 20: "vol_20d"}
            pred = np.full(len(base), np.nan)
            for h, col in volmap.items():
                ix = base["horizon_days"] == h
                pred[ix.to_numpy()] = base.loc[ix, col].to_numpy()
            n_all = len(base)
            n_miss = int(np.isnan(pred).sum())
            if args.on_missing == "rv22":
                pred = np.where(np.isnan(pred), base["feature_rv_22d"].to_numpy(), pred)
            valid = ~np.isnan(pred)
            n_clip = int(((pred < CLIP_LO) | (pred > CLIP_HI))[valid].sum())
            pred = np.clip(pred, CLIP_LO, CLIP_HI)
            base["prediction_realised_vol"] = pred
            if args.on_missing == "drop":
                base = base[~base["prediction_realised_vol"].isna()]
            run_id = f"{model_id}_full_{disc}_seed2026"
            base["run_id"], base["model_id"] = run_id, model_id
            base["dataset"], base["seed"] = "full", 2026
            base["disclosure_subset"] = disc
            out = base[PRED_COLS].reset_index(drop=True)
            run_dir = out_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            out.to_parquet(run_dir / "predictions.parquet", index=False)
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics_rows(out, disc), indent=2))
            cfg = {
                "model_id": model_id,
                "note": ("Generative-LLM realised-vol forecaster (E1). Zero-shot "
                         "prompting, one JSON forecast per filing covering h=5/10/20. "
                         "VAL+TEST only (no training; M1 combiner needs val+test only). "
                         + ("Text-only prompt (C-block comparable)." if variant == "c6_text"
                            else "Text + HAR lags in prompt (D-block comparable).")),
                "variant": variant,
                "llm": str(rv["model_name"].iloc[0]),
                "prompt_cap_tokens": 6000,
                "clip_range": [CLIP_LO, CLIP_HI],
                "on_missing": args.on_missing,
                "stats": {
                    "n_rows": int(len(out)),
                    "n_filings": int(out["text_path"].nunique()),
                    "parse_fail_rows": n_miss,
                    "parse_fail_rate": round(n_miss / n_all, 4),
                    "clipped_rows": n_clip,
                    "clipped_rate": round(n_clip / max(n_all - n_miss, 1), 4),
                },
            }
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
            print(f"wrote {run_dir}  rows={len(out)} filings={out['text_path'].nunique()} "
                  f"parse_fail={n_miss} clip_rate={cfg['stats']['clipped_rate']:.3f}")


def pilot_eval(args) -> None:
    """Go/no-go readout (criteria in go_no_go.md). Pilot rows are TEST only."""
    raw = load_raw(args.raw_dir)
    man = pd.read_parquet(args.manifest)
    a1 = pd.read_parquet("results/runs/A1_hv_full_combined_seed2026/predictions.parquet")
    a1 = a1[a1["split"] == "test"][
        ["text_path", "horizon_days", "label_realised_vol", "prediction_realised_vol"]
    ].rename(columns={"prediction_realised_vol": "f_a1"})
    print(f"pilot filings: {raw['text_path'].nunique()}  "
          f"(manifest match {raw['text_path'].isin(man['text_path']).mean():.2f})")
    report = {}
    for variant, g in raw.groupby("variant"):
        parse_rate = float(g["parse_ok"].mean())
        long = g.melt(id_vars=["text_path"], value_vars=["vol_5d", "vol_10d", "vol_20d"],
                      var_name="hcol", value_name="f_llm")
        long["horizon_days"] = long["hcol"].map({"vol_5d": 5, "vol_10d": 10, "vol_20d": 20})
        parsed = long.dropna(subset=["f_llm"])
        clip_rate = float(((parsed["f_llm"] < CLIP_LO) | (parsed["f_llm"] > CLIP_HI)).mean())
        j = parsed.merge(a1, on=["text_path", "horizon_days"], how="inner")
        j["f_llm"] = j["f_llm"].clip(CLIP_LO, CLIP_HI)
        y = j["label_realised_vol"].to_numpy()
        ql_llm = float(qlike_vol(y, j["f_llm"].to_numpy()).mean())
        ql_a1 = float(qlike_vol(y, j["f_a1"].to_numpy()).mean())
        rho = float(sps.spearmanr(j["f_llm"], y).statistic)
        gates = {
            "parse_rate>=0.95": parse_rate >= 0.95,
            "clip_rate<=0.05": clip_rate <= 0.05,
            "qlike_beats_A1_hv": ql_llm < ql_a1,
            "spearman>0.15": rho > 0.15,
        }
        report[variant] = {
            "n_joined_rows": int(len(j)), "parse_rate": round(parse_rate, 4),
            "clip_rate": round(clip_rate, 4),
            "qlike_vol_llm": round(ql_llm, 4), "qlike_vol_A1_hv": round(ql_a1, 4),
            "spearman_pred_label": round(rho, 4),
            "gates": gates, "GO": all(gates.values()),
        }
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-runs")
    b.add_argument("--raw-dir", required=True)
    b.add_argument("--out-root", default="results/runs")
    b.add_argument("--model-suffix", default="",
                   help="append to model_id/run_id, e.g. _yi34 for a second LLM family")
    b.add_argument("--on-missing", choices=["rv22", "drop"], default="rv22",
                   help="rv22: fill unparsed filings with feature_rv_22d (counted); "
                        "drop: omit those rows")
    p = sub.add_parser("pilot-eval")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--manifest", required=True)
    args = ap.parse_args()
    build_runs(args) if args.cmd == "build-runs" else pilot_eval(args)


if __name__ == "__main__":
    main()
