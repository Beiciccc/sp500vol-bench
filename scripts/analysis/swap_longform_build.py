"""E-lf STEP 1/3 — build the SWAPPED LONG-FORM panel manifest (CPU only).

Pre-registration: configs/prereg_swap_lf_and_anon.md §E-lf (tag prereg-ea-v1.0).

Mechanism is the COMMITTED matched-firm swap, reused verbatim — the pairing is
computed by importing scripts/analysis/matched_firm_swap.matched_swap (the exact
function the committed table ran), NOT re-derived:

  * within each calendar day (withindate_placebo.day_key on effective_trading_day),
    firms are paired by nearest VALIDATION-period mean RV (rv_map computed from the
    val split of the SAME horizon, exactly as matched_firm_swap.main does);
  * a firm's k-th filing swaps with its partner's k-th filing; odd firm / unmatched
    extras stay unswapped;
  * deterministic — matched_swap contains no RNG (the committed convention;
    "seed 2026 / redraw count identical to committed" is trivially satisfied:
    one deterministic draw).

Trick that guarantees verbatim reuse: matched_swap swaps a float64 array between
paired rows, so we feed it np.arange(n) — the returned array IS the row
permutation (exact for n < 2^53), which we then apply to text_path/accession.

This script exchanges DOCUMENT POINTERS, not forecasts: the manifest maps each
val/test long-form row (split, horizon, accession) to its partner's accession
(and text_path, for reference). The swap is per-horizon (rv_map differs by
horizon), applied on val AND test — the committed placebo convention.

DISCIPLINE: this script touches NO test label statistic. Pairing uses VAL labels
only; the only test-split quantities are row counts and swap fractions (which are
asserted equal to the committed matched_firm_swap.csv swap_frac_test — G2 anchor).

Outputs (under the data root, $SP500VOL_DATA_ROOT, default /path/to/data-root/sp500vol-data):
  processed/swap_lf/swap_manifest_long_form.parquet
  processed/swap_lf/swap_manifest_long_form_meta.json   (build stats + gate asserts)

Run from repo root:
  .venv/bin/python scripts/analysis/swap_longform_build.py            # real build
  .venv/bin/python scripts/analysis/swap_longform_build.py --smoke 200
"""
from __future__ import annotations

# --- thread caps BEFORE numpy/pandas (<=4 cores local discipline) ------------
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)  # fc.load & committed tables use repo-root-relative paths
sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import forecast_combination as fc  # noqa: E402
import matched_firm_swap as mfs  # noqa: E402  (THE committed pairing function)
from withindate_placebo import day_key  # noqa: E402  (THE committed day key)

from sp500vol.utils.paths import data_path  # noqa: E402

DISC = "long_form"
SORT = fc.SORT
HORIZONS = fc.HORIZONS
COMMITTED_SWAP_CSV = "results/tables/matched_firm_swap.csv"
# the three frozen arms re-inferred in step 2 (prereg §E-lf)
ARMS = ("B2_tfidf_ridge", "C2_finbert_s1", "C5_qwen3")

PANEL_COLS = ["split", "ticker", "accession", "horizon_days", "label_realised_vol",
              "filing_time_utc", "effective_trading_day", "text_path"]


def load_panel() -> pd.DataFrame:
    """The A2 HAR long-form panel — the exact row universe of the committed
    matched_firm_swap merge (asserted below: every arm merge preserves it)."""
    har = fc.load("A2_har_rv", DISC)
    missing = [c for c in PANEL_COLS if c not in har.columns]
    if missing:
        raise SystemExit(f"[build] FATAL: A2 parquet lacks columns {missing}")
    p = har[PANEL_COLS].copy()
    if p.text_path.isna().any():
        raise SystemExit("[build] FATAL: null text_path in A2 panel")
    return p


def build_manifest(panel: pd.DataFrame, horizons=HORIZONS,
                   min_rows=(100, 30)) -> tuple[pd.DataFrame, list[dict]]:
    """Pure function: (panel) -> (manifest, level-preservation stats).

    Committed pairing per (horizon, split): rv_map/g_mean from the VAL split of
    that horizon; matched_swap applied separately on val and test rows sorted by
    the committed SORT (filing_time_utc, ticker, accession).
    """
    out, stats = [], []
    for h in horizons:
        dv = (panel[(panel.horizon_days == h) & (panel.split == "val")]
              .sort_values(SORT, kind="mergesort").reset_index(drop=True))
        dt = (panel[(panel.horizon_days == h) & (panel.split == "test")]
              .sort_values(SORT, kind="mergesort").reset_index(drop=True))
        if len(dv) < min_rows[0] or len(dt) < min_rows[1]:
            raise SystemExit(f"[build] FATAL: too few rows at h={h} "
                             f"(val={len(dv)}, test={len(dt)})")
        rv_map = dv.groupby("ticker")["label_realised_vol"].mean()
        g_mean = float(dv.label_realised_vol.mean())
        for split, d in (("val", dv), ("test", dt)):
            days = day_key(d)
            idx = np.arange(len(d), dtype=np.float64)
            permuted, frac = mfs.matched_swap(idx, d.ticker, days, rv_map, g_mean)
            perm = permuted.astype(np.int64)
            # exactness + structure guards on the committed function's output
            if not np.array_equal(permuted, perm.astype(np.float64)):
                raise SystemExit("[build] FATAL: index round-trip inexact")
            if not np.array_equal(np.sort(perm), np.arange(len(d))):
                raise SystemExit("[build] FATAL: matched_swap output not a permutation")
            if not np.array_equal(perm[perm], np.arange(len(d))):
                raise SystemExit("[build] FATAL: matched_swap output not an involution")
            rv_self = d.ticker.map(rv_map).fillna(g_mean).to_numpy(dtype=float)
            man = pd.DataFrame({
                "split": split,
                "horizon_days": int(h),
                "day": days,
                "ticker": d.ticker.to_numpy(),
                "accession": d.accession.to_numpy(),
                "partner_ticker": d.ticker.to_numpy()[perm],
                "partner_accession": d.accession.to_numpy()[perm],
                "swapped": perm != np.arange(len(d)),
                "rv_val_self": rv_self,
                "rv_val_partner": rv_self[perm],
                "text_path_orig": d.text_path.to_numpy(),
                "text_path_partner": d.text_path.to_numpy()[perm],
            })
            out.append(man)
            sw = man[man.swapped]
            dd = (sw.rv_val_self - sw.rv_val_partner).abs()
            ld = (np.log(sw.rv_val_self.clip(lower=fc.EPS))
                  - np.log(sw.rv_val_partner.clip(lower=fc.EPS))).abs()
            spread = float(rv_map.std())
            stats.append({
                "split": split, "h": int(h), "n_rows": int(len(d)),
                "n_swapped": int(sw.shape[0]), "swap_frac": float(frac),
                "pair_absdiff_rv_median": float(dd.median()),
                "pair_absdiff_rv_mean": float(dd.mean()),
                "pair_absdiff_rv_p90": float(dd.quantile(0.90)),
                "pair_absdiff_rv_max": float(dd.max()),
                "pair_absdiff_logrv_median": float(ld.median()),
                "xsec_std_val_rv": spread,
                "pair_absdiff_over_xsec_std": float(dd.median() / spread) if spread > 0 else float("nan"),
            })
    return pd.concat(out, ignore_index=True), stats


def assert_committed_fractions(stats: list[dict]) -> dict:
    """G2 anchor: TEST swap fractions must equal the committed table's
    swap_frac_test bit-for-bit (identical pairing function + identical inputs)."""
    committed = pd.read_csv(COMMITTED_SWAP_CSV)
    lf = committed[committed.disc == DISC]
    checks = {}
    for s in stats:
        if s["split"] != "test":
            continue
        ref = lf[lf.h == s["h"]].swap_frac_test.unique()
        if len(ref) != 1:
            raise SystemExit(f"[build] FATAL: committed swap_frac_test not unique at h={s['h']}")
        ok = abs(float(ref[0]) - s["swap_frac"]) < 1e-12
        checks[f"test_h{s['h']}"] = {"built": s["swap_frac"], "committed": float(ref[0]), "match": ok}
        if not ok:
            raise SystemExit(f"[build] FATAL: swap_frac mismatch h={s['h']}: "
                             f"built {s['swap_frac']} vs committed {ref[0]}")
        # n_test rows must also match the committed cell counts
        nref = lf[lf.h == s["h"]].n_test.unique()
        if len(nref) != 1 or int(nref[0]) != s["n_rows"]:
            raise SystemExit(f"[build] FATAL: n_test mismatch h={s['h']}: "
                             f"panel {s['n_rows']} vs committed {nref}")
    return checks


def assert_arm_coverage(panel: pd.DataFrame) -> dict:
    """The committed swap pairs on the (HAR ∩ arm) merged panel per cell. A single
    shared manifest is valid iff each arm's predictions cover the HAR row set
    exactly (merge preserves rows) — asserted here for the three E-lf arms."""
    out = {}
    key = ["ticker", "accession", "horizon_days"]
    base = panel[panel.split.isin(["val", "test"])]
    for arm in ARMS:
        txt = fc.load(arm, DISC)[key]
        merged = base.merge(txt.drop_duplicates(), on=key, how="inner")
        ok = len(merged) == len(base)
        out[arm] = {"panel_rows": int(len(base)), "merged_rows": int(len(merged)), "covers": ok}
        if not ok:
            raise SystemExit(f"[build] FATAL: arm {arm} does not cover the HAR panel "
                             f"({len(merged)}/{len(base)}) — per-arm manifests required")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="keep first N rows per (split,horizon); writes *_smoke outputs; "
                         "committed-fraction / arm-coverage asserts are SKIPPED (subset "
                         "pairing differs by construction)")
    ap.add_argument("--out", type=str, default=None,
                    help="override manifest parquet path (default: "
                         "$SP500VOL_DATA_ROOT/processed/swap_lf/swap_manifest_long_form.parquet)")
    args = ap.parse_args()

    panel = load_panel()
    n_by_split = panel.split.value_counts().to_dict()
    print(f"[build] A2 long_form panel rows by split: {n_by_split}")

    if args.smoke is not None:
        panel = (panel.sort_values(SORT, kind="mergesort")
                 .groupby(["split", "horizon_days"], group_keys=False)
                 .head(args.smoke).reset_index(drop=True))
        print(f"[build] SMOKE: kept {len(panel)} rows ({args.smoke} per (split, horizon))")

    manifest, stats = build_manifest(panel)

    # uniqueness guard: one row per (split, horizon, accession)
    dup = manifest.duplicated(["split", "horizon_days", "accession"]).sum()
    if dup:
        raise SystemExit(f"[build] FATAL: {dup} duplicate (split,h,accession) manifest rows")

    print(f"\n[build] level-preservation (paired |Δ val-RV|) — the G2 material:")
    sdf = pd.DataFrame(stats)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sdf.round(6).to_string(index=False))

    checks = {"committed_swap_frac": None, "arm_coverage": None}
    if args.smoke is None:
        checks["committed_swap_frac"] = assert_committed_fractions(stats)
        checks["arm_coverage"] = assert_arm_coverage(panel)
        print("\n[build] G2 anchor: test swap fractions + n_test match committed "
              "matched_firm_swap.csv exactly — PASS")
        print(f"[build] arm coverage over HAR panel — PASS for {', '.join(ARMS)}")
    else:
        print("\n[build] SMOKE: committed-fraction / arm-coverage asserts SKIPPED")

    suffix = "" if args.smoke is None else "_smoke"
    out_path = (Path(args.out) if args.out else
                data_path("processed", "swap_lf", f"swap_manifest_long_form{suffix}.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

    meta = {
        "prereg": "configs/prereg_swap_lf_and_anon.md §E-lf (prereg-ea-v1.0)",
        "smoke": args.smoke,
        "panel_rows_by_split": {k: int(v) for k, v in n_by_split.items()},
        "manifest_rows": int(len(manifest)),
        "manifest_sha256": sha,
        "pairing_source": "scripts/analysis/matched_firm_swap.matched_swap (imported verbatim)",
        "level_preservation_stats": stats,
        "gate_checks": checks,
        "arms": list(ARMS),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n[build] wrote {len(manifest)} manifest rows -> {out_path}")
    print(f"[build] manifest sha256 = {sha}")
    print(f"[build] meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
