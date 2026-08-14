"""NON-PREREGISTERED EXPLORATORY — same-lineage B2 anonymisation sensitivity
annex. NOT part of anon_arm.{csv,md}; NO branch adjudication; the main text
may cite it ONLY as a sensitivity remark.

WHY THIS FILE EXISTS: the registered B2 anon arms (ED and LF) EXITED at G1
(2026-07-16 ruling; see anon_score.py's exit-ruling block and the anon_arm
tables' b2 disclosures): the box control could not reproduce the committed
June predictions (official g1_control_b2*_boxvenv.json; exact-match 0), the
committed (env x cache) pair is unreconstructible (env.json carries no
package versions), and the text-store lineage has drifted (the box-control
and local-control TF-IDF fits hold different vocabulary sets, idf max|diff|
7.5). HOWEVER the box produced a ctrl/masked PAIR under ONE env and ONE text
cache — the SAME lineage — so the ctrl-vs-masked CONTRAST is internally
self-consistent even though neither leg ties to the committed anchor. That
contrast is what this annex reports, as exploration only.

Layout mirrors the registered table's 6 cells (3 horizons x 2 references):
ctrl rel% / masked rel% / share / day-clustered DM with a DESCRIPTIVE p —
NO Holm familying, NO branch mapping, NO single-shot discipline (the file is
regenerated; rows of other channels already present in the csv are kept).

Usage (from repo root; --box-ctrl-dir is REQUIRED — the box control run dir
is staged outside the repo and its location is deliberately not hardcoded):
  .venv/bin/python scripts/analysis/anon_annex_samelineage.py --channel ed \
      --box-ctrl-dir  <staging>/runs/B2_tfidf_ridge_anonctrl_full_event_driven_seed2026 \
      --masked-dir    <staging>/runs/B2_tfidf_ridge_anonmask_full_event_driven_seed2026
  # lf: same, once the box LF masked run exists (anonctrl_full_long_form dir
  # is already staged; pass the LF pair with --channel lf).
Outputs: results/tables/anon_annex_samelineage.{csv,md}
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")

import clustered_dm as cdm
import forecast_combination as fc

KEY = ["ticker", "accession", "horizon_days"]
EPS = 1e-8
HORIZONS = (5, 10, 20)
OUT_CSV = REPO / "results/tables/anon_annex_samelineage.csv"
OUT_MD = REPO / "results/tables/anon_annex_samelineage.md"
DECLARATION = ("NON-PREREGISTERED EXPLORATORY — SAME-LINEAGE B2 ANONYMISATION "
               "SENSITIVITY ANNEX. NOT PART OF anon_arm.{csv,md}; NO BRANCH "
               "ADJUDICATION; CITABLE IN THE MAIN TEXT ONLY AS A SENSITIVITY "
               "REMARK.")
G1_JSON = {"ed": "g1_control_b2_boxvenv.json",
           "lf": "g1_control_b2_lf_boxvenv.json"}
DISC = {"ed": "event_driven", "lf": "long_form"}


def _fatal(msg: str) -> None:
    raise SystemExit(f"[annex] FATAL: {msg}")


def ols(y, X):  # verbatim crossfamily_llama70.py / anon_score.py
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def m1_cells_from_preds(pred_pq: Path, disc: str) -> dict[int, dict]:
    """Per-horizon M1 vs both references — the anon_score.m1_cells /
    crossfamily_llama70.py block VERBATIM, with the text run loaded from an
    arbitrary predictions.parquet instead of a results/runs name."""
    a2 = fc.load("A2_har_rv", disc)[KEY + ["split", "label_realised_vol",
                                           "prediction_realised_vol",
                                           "effective_trading_day"]] \
        .rename(columns={"prediction_realised_vol": "fh"})
    p = pd.read_parquet(pred_pq)
    t = p[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "ft"})
    out = {}
    for h in HORIZONS:
        m = a2[a2.horizon_days == h].merge(t[t.horizon_days == h], on=KEY).dropna()
        v, te = m[m.split == "val"], m[m.split == "test"]
        y = te.label_realised_vol.values
        fR, fU, g = fc.log_combo(v.label_realised_vol.values, v.fh.values,
                                 v.ft.values, te.fh.values, te.ft.values)
        qR, qU = fc.qlike(y, fR), fc.qlike(y, fU)
        rel = 100 * np.mean(qR - qU) / np.mean(qR)
        dm, pv, nd = cdm.dm_test_clustered(qU, qR, te.effective_trading_day.values, h)
        fm = v.groupby("ticker").label_realised_vol.mean()
        gmean = v.label_realised_vol.mean()
        fid_v = v.ticker.map(fm).fillna(gmean).values
        fid_t = te.ticker.map(fm).fillna(gmean).values
        L = lambda x: np.log(np.clip(x, EPS, None))
        ly = L(v.label_realised_vol.values)
        bR = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v)]))
        bU = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v),
                                      L(v.ft.values)]))
        fRf = np.exp(bR[0] + bR[1] * L(te.fh.values) + bR[2] * L(fid_t))
        fUf = np.exp(bU[0] + bU[1] * L(te.fh.values) + bU[2] * L(fid_t)
                     + bU[3] * L(te.ft.values))
        qRf, qUf = fc.qlike(y, fRf), fc.qlike(y, fUf)
        relf = 100 * np.mean(qRf - qUf) / np.mean(qRf)
        dmf, pf, _ = cdm.dm_test_clustered(qUf, qRf, te.effective_trading_day.values, h)
        out[h] = dict(n_test=len(te), n_days=nd, rel_har=rel, dm_har=dm, p_har=pv,
                      rel_firm=relf, dm_firm=dmf, p_firm=pf, g_text=float(g))
    return out


def share(masked_rel: float, ctrl_rel: float) -> float:
    if not np.isfinite(ctrl_rel) or ctrl_rel <= 0:
        return float("nan")
    return 1.0 - masked_rel / ctrl_rel


def compute_channel(channel: str, box_ctrl_dir: Path, masked_dir: Path) -> pd.DataFrame:
    disc = DISC[channel]
    ctrl_pq = box_ctrl_dir / "predictions.parquet"
    mask_pq = masked_dir / "predictions.parquet"
    for pq in (ctrl_pq, mask_pq):
        if not pq.exists():
            _fatal(f"missing predictions: {pq}")
    print(f"[annex:{channel}] ctrl  = {ctrl_pq}")
    print(f"[annex:{channel}] mask  = {mask_pq}")
    ctrl = m1_cells_from_preds(ctrl_pq, disc)
    mask = m1_cells_from_preds(mask_pq, disc)
    # official-G1 context (descriptive; the reason this pair is annex-only)
    g1p = REPO / "results/anon" / G1_JSON[channel]
    g1 = json.loads(g1p.read_text()) if g1p.exists() else {}
    rows = []
    for h in HORIZONS:
        rows.append({
            "channel": channel, "disc": disc, "model": "B2_tfidf_ridge", "h": h,
            "n_test": ctrl[h]["n_test"], "n_days": ctrl[h]["n_days"],
            "rel_har_ctrl": ctrl[h]["rel_har"],
            "rel_har_masked": mask[h]["rel_har"],
            "share_har": share(mask[h]["rel_har"], ctrl[h]["rel_har"]),
            "dm_har_masked": mask[h]["dm_har"],
            "p_har_masked_descriptive": mask[h]["p_har"],
            "rel_firm_ctrl": ctrl[h]["rel_firm"],
            "rel_firm_masked": mask[h]["rel_firm"],
            "share_firm": share(mask[h]["rel_firm"], ctrl[h]["rel_firm"]),
            "dm_firm_masked": mask[h]["dm_firm"],
            "p_firm_masked_descriptive": mask[h]["p_firm"],
            "g1_official_exact_match_rate": g1.get("exact_match_rate"),
            "g1_official_max_abs_diff": g1.get("max_abs_diff"),
            "box_ctrl_dir": str(box_ctrl_dir), "masked_dir": str(masked_dir),
        })
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["channel", "h"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    def f(x, spec="+.3f"):
        return format(x, spec) if pd.notna(x) else "n/a"

    md = [
        f"**{DECLARATION}**",
        "",
        "# Same-lineage B2 ctrl/masked contrast (exploratory annex)",
        "",
        "The registered B2 anon arms exited at G1 — the box control fails "
        "the committed anchor (env unreconstructible + text-store lineage "
        "drift; see anon_arm.{csv,md} b2 disclosures). This annex reports "
        "the box's OWN ctrl/masked pair: both legs share one venv and one "
        "text cache, so the contrast is internally self-consistent, but "
        "neither leg ties to the committed estimand — hence exploratory, "
        "descriptive p only (no Holm), no branch mapping.",
        "",
        "| channel | h | vs HAR: ctrl rel% | masked rel% | share | DM | "
        "descr. p | vs HAR+firmID: ctrl rel% | masked rel% | share | DM | "
        "descr. p |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {r.channel} | {int(r.h)} | {f(r.rel_har_ctrl)} | "
            f"{f(r.rel_har_masked)} | {f(r.share_har, '+.2f')} | "
            f"{f(r.dm_har_masked, '+.2f')} | "
            f"{f(r.p_har_masked_descriptive, '.4g')} | "
            f"{f(r.rel_firm_ctrl)} | {f(r.rel_firm_masked)} | "
            f"{f(r.share_firm, '+.2f')} | {f(r.dm_firm_masked, '+.2f')} | "
            f"{f(r.p_firm_masked_descriptive, '.4g')} |")
    md += [
        "",
        "## Disclosures",
        "",
        "- SAME-LINEAGE, SELF-CONSISTENT: ctrl and masked were produced in "
        "the SAME box venv on the SAME box text cache, so the ctrl/masked "
        "contrast is internally coherent (identical vectoriser lineage on "
        "both legs).",
        "- BUT ANCHOR-FAILED: the same box ctrl does NOT reproduce the "
        "committed June predictions (official g1_control_b2*_boxvenv.json: "
        "exact-match 0; per-channel numbers in the csv's g1_official_* "
        "columns). Cause per the closed diagnosis: the committed (env x "
        "cache) pair is unreconstructible (env.json without package "
        "versions) and the text-store lineage drifted (different fitted "
        "vocabulary sets, idf max|diff| 7.5). The ctrl leg is therefore NOT "
        "the committed unmasked increment, and shares here are NOT the "
        "registered identity-share estimand.",
        "- Statistics are DESCRIPTIVE: day-clustered DM p values are "
        "reported raw, deliberately outside any Holm family; nothing here "
        "enters branch adjudication or the registered tables.",
        "- File hygiene: regenerated on each run (channels are merged by "
        "row replacement); no single-shot discipline applies to this annex.",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} "
        f"by scripts/analysis/anon_annex_samelineage.py.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[annex] wrote {OUT_CSV} ({len(df)} rows) and {OUT_MD}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--channel", choices=["ed", "lf"], default="ed")
    ap.add_argument("--box-ctrl-dir", required=True,
                    help="box control run dir (predictions.parquet inside); "
                         "staged location, deliberately not hardcoded")
    ap.add_argument("--masked-dir", default=None,
                    help="box masked run dir (default results/runs/"
                         "B2_tfidf_ridge_anonmask_full_<disc>_seed2026)")
    args = ap.parse_args()

    masked_dir = Path(args.masked_dir) if args.masked_dir else (
        REPO / "results/runs" /
        f"B2_tfidf_ridge_anonmask_full_{DISC[args.channel]}_seed2026")
    new = compute_channel(args.channel, Path(args.box_ctrl_dir), masked_dir)

    if OUT_CSV.exists():  # keep other channels' rows; replace this channel's
        prev = pd.read_csv(OUT_CSV)
        prev = prev[prev.channel != args.channel]
        new = pd.concat([prev, new], ignore_index=True)
    write_outputs(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
