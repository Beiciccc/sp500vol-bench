"""Row-3 (round-3 remediation) — TUNED-CHALLENGER arm vs the archived fixed recipe.

Answers DA CRITICAL #1 (results/REVIEW_ROUND3_FRESH_PANEL.md): were the challenger
nulls an artifact of an under-tuned FIXED training recipe (lr=8e-5, 15 epochs,
patience 3 for every neural model)?  Round 3 ran a 3-point lr grid
{5e-6, 1e-5, 2e-5} with max_epochs=5 / es_patience=1 per arm, selected the config
with minimum pooled validation QLIKE (vol-unit), and re-ran the winner end to end:

    C2t_finbert_s1_full_long_form_seed2026      (selected lr=2e-5)
    C2t_finbert_s1_full_event_driven_seed2026   (selected lr=5e-6)
    D2t_gated_fusion_full_long_form_seed2026    (selected lr=5e-6)

Grid audit: results/tables/row3_tuning_grid.csv (read here; selected rows quoted).

Per run x horizon this script reports, all in VOL units with DAY-CLUSTERED DM
(scripts/analysis/clustered_dm.py) and combiner weights fit on VALIDATION, frozen
on TEST (scripts/analysis/forecast_combination.py machinery):
  1. standalone test QLIKE / OOS R2, tuned vs the archived fixed-recipe counterpart
     (C2_finbert_s1 / D2_gated_fusion, seed2026), + pairwise clustered DM on QLIKE
     (does tuning significantly improve the challenger itself?);
  2. M1 log-space increment vs the single recalibrated HAR reference;
  3. M1 log-space increment vs the firm-identity-augmented reference
     (val-window firm mean of realised vol, spec identical to crossfamily_llm.py);
  4. label-shuffle placebo (seeds 1000-1004, clustered DM) for BOTH references and
     BOTH arms — "genuine" requires dm<0, Holm<.05 AND |placebo DM|<2.

Holm (PRE-DECLARED): within each comparison family over the 9 tuned cells
(3 arms x 3 horizons): family S (standalone tuned-vs-archived), family H (M1 vs
single HAR), family F (M1 vs HAR+firmID).  The archived arm is Holm-corrected on
the SAME 9-cell family for a symmetric verdict flip count; the committed 69-cell
verdict (m1_clustered.csv genuine_clust) is also quoted.

SANITY GATES (hard; failing gate = stop):
  G1  the archived C2/D2 M1-vs-HAR cells recomputed here reproduce the committed
      results/tables/m1_clustered.csv rows to machine precision.  BASIS:
      m1_clustered.csv is the SEED-2026 SINGLE-SEED table — the same basis as the
      archived counterpart runs and the tuned runs (all seed2026).
      m1_ensemble_primary.csv is NOT usable as the primary basis here because its
      C2/D2 rows are 3-seed ensembles (n_seeds=3); its s26_* columns ARE the
      seed-2026 basis and are cross-asserted against m1_clustered.csv below.
  G2  the firm-identity-reference machinery reproduces the committed
      results/tables/crossfamily_llm.csv qwen3_32b rows to machine precision.
  G3  standalone OOS R2 computed here reproduces each tuned run's committed
      metrics.json (test split) to machine precision.

Outputs: results/tables/row3_tuned_m1.{csv,md}
Run from repo root:  .venv/bin/python scripts/analysis/row3_tuned_m1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc
from clustered_dm import dm_test_clustered

KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
HORIZONS = (5, 10, 20)
EPS = 1e-8
HAR = "A2_har_rv"
GRID_CSV = "results/tables/row3_tuning_grid.csv"
M1_CLUSTERED = "results/tables/m1_clustered.csv"
M1_ENSEMBLE = "results/tables/m1_ensemble_primary.csv"
CROSSFAMILY = "results/tables/crossfamily_llm.csv"

ARMS = [  # (archived fixed-recipe model, tuned model, disclosure)
    ("C2_finbert_s1", "C2t_finbert_s1", "long_form"),
    ("C2_finbert_s1", "C2t_finbert_s1", "event_driven"),
    ("D2_gated_fusion", "D2t_gated_fusion", "long_form"),
]
FIXED_RECIPE = "lr=8e-5, max_epochs=15, es_patience=3"


def L(x):
    return np.log(np.clip(x, EPS, None))


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return b


def day_key(df):
    d = df["effective_trading_day"]
    if d.isna().any():
        fb = pd.to_datetime(df["filing_time_utc"], utc=True).dt.tz_localize(None)
        d = d.fillna(fb)
    return d.to_numpy()


def firm_combo(v, te, ftv, ftt):
    """Firm-identity-augmented log-space combo, spec IDENTICAL to crossfamily_llm.py:
    reference = [1, log fHAR, log firm-mean-val-RV]; augmented adds log fText.
    Returns (fR, fU) on test."""
    fm = v.groupby("ticker").label_realised_vol.mean()
    gm = v.label_realised_vol.mean()
    fid_v = v.ticker.map(fm).fillna(gm).values
    fid_t = te.ticker.map(fm).fillna(gm).values
    ly = L(v.label_realised_vol.values)
    bR = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v)]))
    bU = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v), L(ftv)]))
    fR = np.exp(bR[0] + bR[1] * L(te.fh.values) + bR[2] * L(fid_t))
    fU = np.exp(bU[0] + bU[1] * L(te.fh.values) + bU[2] * L(fid_t) + bU[3] * L(ftt))
    return fR, fU


def m1_one(model, disc):
    """Per horizon: standalone QLIKE/R2 + M1 vs single HAR and vs HAR+firmID,
    clustered DM + clustered label-shuffle placebo for both references."""
    har = fc.load(HAR, disc)[["split"] + KEY + [
        "prediction_realised_vol", "label_realised_vol", "filing_time_utc",
        "effective_trading_day"]].rename(columns={"prediction_realised_vol": "fh"})
    txt = fc.load(model, disc)[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "ft"})
    d = har.merge(txt, on=KEY)
    out = {}
    for h in HORIZONS:
        v = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
        te = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
        yv, fhv, ftv = v.label_realised_vol.to_numpy(), v.fh.to_numpy(), v.ft.to_numpy()
        yt, fhr, ftt = te.label_realised_vol.to_numpy(), te.fh.to_numpy(), te.ft.to_numpy()
        days = day_key(te)

        alone_q = float(fc.qlike(yt, ftt).mean())
        alone_r2 = float(1.0 - ((yt - ftt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())

        # M1 vs single recalibrated HAR (identical machinery to m1_clustered.py)
        fR, fU, g = fc.log_combo(yv, fhv, ftv, fhr, ftt)
        lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
        rel = 100.0 * float(lR.mean() - lU.mean()) / float(lR.mean())
        dm_h, p_h, n_days = dm_test_clustered(lU, lR, days, h)

        # M1 vs firm-identity-augmented reference (crossfamily_llm.py spec)
        fRf, fUf = firm_combo(v, te, ftv, ftt)
        lRf, lUf = fc.qlike(yt, fRf), fc.qlike(yt, fUf)
        relf = 100.0 * float(lRf.mean() - lUf.mean()) / float(lRf.mean())
        dm_f, p_f, _ = dm_test_clustered(lUf, lRf, days, h)

        # clustered placebo, both references, same seed stream as m1_clustered.py
        ph, pf = [], []
        for s in fc.PLACEBO_SEEDS:
            rng = np.random.default_rng(s)
            pv, pt = rng.permutation(ftv), rng.permutation(ftt)
            pR, pU, _ = fc.log_combo(yv, fhv, pv, fhr, pt)
            st, _, _ = dm_test_clustered(fc.qlike(yt, pU), fc.qlike(yt, pR), days, h)
            ph.append(st)
            pRf, pUf = firm_combo(v, te, pv, pt)
            stf, _, _ = dm_test_clustered(fc.qlike(yt, pUf), fc.qlike(yt, pRf), days, h)
            pf.append(stf)

        out[h] = dict(n_test=len(te), n_days=n_days,
                      alone_qlike=alone_q, alone_r2=alone_r2,
                      qlike_R=float(lR.mean()), qlike_U=float(lU.mean()),
                      rel_har=rel, g_log=float(g), dm_har=float(dm_h), p_har=float(p_h),
                      placebo_har=float(np.mean(ph)),
                      qlike_Rf=float(lRf.mean()), qlike_Uf=float(lUf.mean()),
                      rel_fid=relf, dm_fid=float(dm_f), p_fid=float(p_f),
                      placebo_fid=float(np.mean(pf)))
    return out


def pairwise_tuned_vs_arch(base, tuned, disc):
    """Clustered DM on standalone QLIKE, tuned (loss A) vs archived (loss B).
    Negative = tuned better."""
    har = fc.load(HAR, disc)[["split"] + KEY + ["label_realised_vol",
                                                "filing_time_utc", "effective_trading_day"]]
    b = fc.load(base, disc)[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "fb"})
    t = fc.load(tuned, disc)[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "ftu"})
    d = har.merge(b, on=KEY).merge(t, on=KEY)
    out = {}
    for h in HORIZONS:
        te = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
        yt = te.label_realised_vol.to_numpy()
        days = day_key(te)
        dm, p, nd = dm_test_clustered(fc.qlike(yt, te.ftu.to_numpy()),
                                      fc.qlike(yt, te.fb.to_numpy()), days, h)
        out[h] = dict(dm_alone=float(dm), p_alone=float(p), n_pair=len(te))
    return out


# ---------------------------------------------------------------------------
# SANITY GATES
# ---------------------------------------------------------------------------
def gate1_m1_clustered(arch):
    """Archived M1-vs-HAR cells must reproduce m1_clustered.csv to machine precision."""
    mc = pd.read_csv(M1_CLUSTERED)
    me = pd.read_csv(M1_ENSEMBLE)
    diffs = {}
    for (base, _tuned, disc), cells in arch.items():
        for h, c in cells.items():
            r = mc[(mc.disc == disc) & (mc.model == base) & (mc.h == h)]
            if len(r) != 1:
                raise AssertionError(f"G1: missing m1_clustered row {disc}/{base}/h{h}")
            r = r.iloc[0]
            got = [c["n_test"], c["n_days"], c["qlike_R"], c["qlike_U"], c["rel_har"],
                   c["g_log"], c["dm_har"], c["p_har"], c["placebo_har"]]
            ref = [r.n_obs, r.n_days, r.qlike_R, r.qlike_U, r.rel_impr_pct,
                   r.g_log, r.dm_q_clust, r.p_q_clust, r.placebo_dm_clust]
            dmax = float(np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))))
            diffs[f"{disc}/{base}/h{h}"] = dmax
            if not np.allclose(got, ref, rtol=1e-9, atol=1e-12):
                raise AssertionError(
                    f"G1 SANITY FAIL {disc}/{base}/h{h}: max|diff|={dmax:.3e}")
            # cross-assert basis: m1_ensemble_primary s26_* columns == m1_clustered
            e = me[(me.disc == disc) & (me.model == base) & (me.h == h)].iloc[0]
            if not np.allclose(
                    [e.s26_qlike_R, e.s26_qlike_U, e.s26_g_log, e.s26_dm_q_clu, e.s26_p_q_clu],
                    [r.qlike_R, r.qlike_U, r.g_log, r.dm_q_clust, r.p_q_clust],
                    rtol=1e-9, atol=1e-12):
                raise AssertionError(f"G1b basis cross-check FAIL {disc}/{base}/h{h}")
    return diffs


def gate2_crossfamily():
    """Firm-identity machinery must reproduce crossfamily_llm.csv qwen3_32b rows."""
    cf = pd.read_csv(CROSSFAMILY)
    diffs = {}
    for disc in ("long_form", "event_driven"):
        a2 = fc.load(HAR, disc)[KEY + ["split", "label_realised_vol",
                                       "prediction_realised_vol", "effective_trading_day"]] \
            .rename(columns={"prediction_realised_vol": "fh"})
        t = fc.load("C6_llmtext", disc)[KEY + ["prediction_realised_vol"]] \
            .rename(columns={"prediction_realised_vol": "ft"})
        for h in HORIZONS:
            m = a2[a2.horizon_days == h].merge(t[t.horizon_days == h], on=KEY).dropna()
            v, te = m[m.split == "val"], m[m.split == "test"]
            y = te.label_realised_vol.values
            fR, fU, _ = fc.log_combo(v.label_realised_vol.values, v.fh.values,
                                     v.ft.values, te.fh.values, te.ft.values)
            qR, qU = fc.qlike(y, fR), fc.qlike(y, fU)
            rel = 100 * np.mean(qR - qU) / np.mean(qR)
            dm, p, _ = dm_test_clustered(qU, qR, te.effective_trading_day.values, h)
            fRf, fUf = firm_combo(v, te, v.ft.values, te.ft.values)
            qRf, qUf = fc.qlike(y, fRf), fc.qlike(y, fUf)
            relf = 100 * np.mean(qRf - qUf) / np.mean(qRf)
            dmf, pfv, _ = dm_test_clustered(qUf, qRf, te.effective_trading_day.values, h)
            r = cf[(cf.disc == disc) & (cf.family == "qwen3_32b") & (cf.h == h)].iloc[0]
            got = [rel, dm, p, relf, dmf, pfv]
            ref = [r.rel_har, r.dm_har, r.p_har, r.rel_firm, r.dm_firm, r.p_firm]
            dmax = float(np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))))
            diffs[f"{disc}/qwen3_32b/h{h}"] = dmax
            if not np.allclose(got, ref, rtol=1e-9, atol=1e-12):
                raise AssertionError(
                    f"G2 SANITY FAIL {disc}/h{h}: max|diff|={dmax:.3e}")
    return diffs


def gate3_metrics(tuned_cells):
    """Standalone OOS R2 must reproduce each tuned run's committed metrics.json."""
    diffs = {}
    for (base, tuned, disc), cells in tuned_cells.items():
        mj = json.load(open(f"results/runs/{tuned}_full_{disc}_seed2026/metrics.json"))
        ref = {r["horizon_days"]: r["r2"] for r in mj if r["split"] == "test"}
        for h, c in cells.items():
            dmax = abs(c["alone_r2"] - ref[h])
            diffs[f"{disc}/{tuned}/h{h}"] = dmax
            if not np.isclose(c["alone_r2"], ref[h], rtol=1e-9, atol=1e-12):
                raise AssertionError(
                    f"G3 SANITY FAIL {disc}/{tuned}/h{h}: r2 diff={dmax:.3e}")
    return diffs


# ---------------------------------------------------------------------------
def verdict(dm, holm, placebo):
    if dm < 0 and holm < 0.05:
        return "genuine" if abs(placebo) < 2.0 else "sig-better(placebo-FAIL)"
    if dm > 0 and holm < 0.05:
        return "sig-WORSE"
    return "null"


def main():
    grid = pd.read_csv(GRID_CSV)
    sel = grid[grid.selected.astype(bool)].set_index(["model_id", "disclosure"])

    arch, tuned_c, pair = {}, {}, {}
    for base, tuned, disc in ARMS:
        arch[(base, tuned, disc)] = m1_one(base, disc)
        tuned_c[(base, tuned, disc)] = m1_one(tuned, disc)
        pair[(base, tuned, disc)] = pairwise_tuned_vs_arch(base, tuned, disc)

    g1 = gate1_m1_clustered(arch)
    g2 = gate2_crossfamily()
    g3 = gate3_metrics(tuned_c)
    print(f"SANITY G1 (m1_clustered.csv, 9 archived cells): PASS max|diff|={max(g1.values()):.2e}")
    print(f"SANITY G2 (crossfamily_llm.csv qwen rows, firmID path): PASS max|diff|={max(g2.values()):.2e}")
    print(f"SANITY G3 (tuned metrics.json R2): PASS max|diff|={max(g3.values()):.2e}")

    # assemble 9-cell frame
    rows = []
    for (base, tuned, disc) in [tuple(a) for a in ARMS]:
        s = sel.loc[(base, disc)]
        for h in HORIZONS:
            a, t, p = arch[(base, tuned, disc)][h], tuned_c[(base, tuned, disc)][h], \
                pair[(base, tuned, disc)][h]
            rows.append({
                "disc": disc, "base_model": base, "tuned_model": tuned, "h": h,
                "selected_lr": s.lr, "selected_val_qlike": s.val_qlike,
                "tuned_run_id": s.tuned_run_id,
                "n_test": t["n_test"], "n_days": t["n_days"],
                "alone_qlike_arch": a["alone_qlike"], "alone_qlike_tuned": t["alone_qlike"],
                "alone_r2_arch": a["alone_r2"], "alone_r2_tuned": t["alone_r2"],
                "dm_alone_clu": p["dm_alone"], "p_alone_clu": p["p_alone"],
                "har_qlike_R": t["qlike_R"],
                "har_rel_arch": a["rel_har"], "har_dm_arch": a["dm_har"],
                "har_p_arch": a["p_har"], "har_placebo_arch": a["placebo_har"],
                "har_g_arch": a["g_log"],
                "har_rel_tuned": t["rel_har"], "har_dm_tuned": t["dm_har"],
                "har_p_tuned": t["p_har"], "har_placebo_tuned": t["placebo_har"],
                "har_g_tuned": t["g_log"],
                "fid_rel_arch": a["rel_fid"], "fid_dm_arch": a["dm_fid"],
                "fid_p_arch": a["p_fid"], "fid_placebo_arch": a["placebo_fid"],
                "fid_rel_tuned": t["rel_fid"], "fid_dm_tuned": t["dm_fid"],
                "fid_p_tuned": t["p_fid"], "fid_placebo_tuned": t["placebo_fid"],
            })
    df = pd.DataFrame(rows)

    # PRE-DECLARED Holm: within each 9-cell family; archived re-Holmed on the SAME
    # family for a symmetric flip count (committed 69-cell verdict quoted alongside).
    df["p_alone_holm"] = fc.holm(df.p_alone_clu.values)
    df["har_holm_tuned"] = fc.holm(df.har_p_tuned.values)
    df["har_holm_arch"] = fc.holm(df.har_p_arch.values)
    df["fid_holm_tuned"] = fc.holm(df.fid_p_tuned.values)
    df["fid_holm_arch"] = fc.holm(df.fid_p_arch.values)

    mc = pd.read_csv(M1_CLUSTERED)
    df["har_genuine69_arch"] = [
        bool(mc[(mc.disc == r.disc) & (mc.model == r.base_model) & (mc.h == r.h)]
             .genuine_clust.iloc[0]) for _, r in df.iterrows()]

    for ref in ("har", "fid"):
        df[f"{ref}_verdict_arch"] = [
            verdict(r[f"{ref}_dm_arch"], r[f"{ref}_holm_arch"], r[f"{ref}_placebo_arch"])
            for _, r in df.iterrows()]
        df[f"{ref}_verdict_tuned"] = [
            verdict(r[f"{ref}_dm_tuned"], r[f"{ref}_holm_tuned"], r[f"{ref}_placebo_tuned"])
            for _, r in df.iterrows()]
        df[f"{ref}_flip"] = df[f"{ref}_verdict_arch"] != df[f"{ref}_verdict_tuned"]

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/row3_tuned_m1.csv", index=False)

    # -------- headline counts --------
    n_alone_better = int(((df.dm_alone_clu < 0) & (df.p_alone_holm < 0.05)).sum())
    n_alone_worse = int(((df.dm_alone_clu > 0) & (df.p_alone_holm < 0.05)).sum())
    gen_har_a = int((df.har_verdict_arch == "genuine").sum())
    gen_har_t = int((df.har_verdict_tuned == "genuine").sum())
    gen_fid_a = int((df.fid_verdict_arch == "genuine").sum())
    gen_fid_t = int((df.fid_verdict_tuned == "genuine").sum())
    new_har = df[(df.har_verdict_tuned == "genuine") & (df.har_verdict_arch != "genuine")]
    lost_har = df[(df.har_verdict_arch == "genuine") & (df.har_verdict_tuned != "genuine")]
    new_fid = df[(df.fid_verdict_tuned == "genuine") & (df.fid_verdict_arch != "genuine")]
    lost_fid = df[(df.fid_verdict_arch == "genuine") & (df.fid_verdict_tuned != "genuine")]

    def cellname(r):
        return f"{r.disc} {r.base_model} h{r.h}"

    md = []
    md.append("# Row 3 — tuned challengers (val-selected lr) vs archived fixed recipe: does tuning rescue the null? (DA CRITICAL #1)\n")
    md.append("## RESTATED vs BEFORE\n")
    md.append("| quantity | BEFORE (archived fixed recipe: " + FIXED_RECIPE + ") | RESTATED (row-3 val-tuned) |")
    md.append("|---|---|---|")
    md.append(f"| standalone: tuned significantly better than archived (clustered DM, Holm-9) | — | **{n_alone_better}/9** better, {n_alone_worse}/9 WORSE |")
    md.append(f"| M1 genuine cells vs single recalibrated HAR (Holm-9 + placebo) | {gen_har_a}/9 | **{gen_har_t}/9** |")
    md.append(f"| M1 genuine cells vs HAR+firm-identity reference (Holm-9 + placebo) | {gen_fid_a}/9 | **{gen_fid_t}/9** |")
    md.append(f"| null cells OVERTURNED by tuning (vs HAR) | — | {len(new_har)}"
              + (f" ({', '.join(cellname(r) for _, r in new_har.iterrows())})" if len(new_har) else "") + " |")
    md.append(f"| genuine cells DESTROYED by tuning (vs HAR) | — | {len(lost_har)}"
              + (f" ({', '.join(cellname(r) for _, r in lost_har.iterrows())})" if len(lost_har) else "") + " |")
    md.append(f"| null cells OVERTURNED by tuning (vs firm-ID ref) | — | {len(new_fid)}"
              + (f" ({', '.join(cellname(r) for _, r in new_fid.iterrows())})" if len(new_fid) else "") + " |")
    md.append(f"| genuine cells DESTROYED by tuning (vs firm-ID ref) | — | {len(lost_fid)}"
              + (f" ({', '.join(cellname(r) for _, r in lost_fid.iterrows())})" if len(lost_fid) else "") + " |")
    md.append("")
    md.append("Verdicts here use the PRE-DECLARED Holm family = the 9 tuned cells per comparison "
              "(archived arm re-Holmed on the same 9-cell family for symmetry; the committed "
              "69-cell verdict from m1_clustered.csv is quoted in the grid below as `gen69`). "
              "'genuine' = clustered DM<0, Holm<.05, AND |label-shuffle placebo DM|<2 "
              "(seeds 1000-1004, both references). Vol-unit QLIKE; combiner weights "
              "val-fit, test-frozen (fc.log_combo); firm-identity reference = val-window "
              "firm mean spec of crossfamily_llm.py.\n")

    md.append("## Selected configs (grid audit: row3_tuning_grid.csv)\n")
    md.append("| arm | grid (3 lrs, max_epochs=5, es_patience=1) | SELECTED | pooled val QLIKE | archived fixed recipe |")
    md.append("|---|---|---|---|---|")
    for base, tuned, disc in ARMS:
        s = sel.loc[(base, disc)]
        cand = grid[(grid.model_id == base) & (grid.disclosure == disc)]
        lrs = ", ".join(f"{v:g}" for v in cand.lr)
        md.append(f"| {base} {disc} | lr ∈ {{{lrs}}} | **lr={s.lr:g}** → {s.tuned_run_id} | "
                  f"{s.val_qlike:.3f} (min of {', '.join(f'{v:.3f}' for v in cand.val_qlike)}) | {FIXED_RECIPE} |")
    md.append("")

    md.append("## 1. Standalone: tuned vs archived (test, vol-unit QLIKE / OOS R2; clustered DM tuned−archived, negative = tuned better)\n")
    md.append("| arm | h | QLIKE arch | QLIKE tuned | R2 arch | R2 tuned | DM(clu) | p | Holm-9 | verdict |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    for _, r in df.iterrows():
        v = ("tuned BETTER" if (r.dm_alone_clu < 0 and r.p_alone_holm < .05) else
             ("tuned WORSE" if (r.dm_alone_clu > 0 and r.p_alone_holm < .05) else "ns"))
        md.append(f"| {r.disc} {r.base_model} | {r.h} | {r.alone_qlike_arch:.4f} | {r.alone_qlike_tuned:.4f} | "
                  f"{r.alone_r2_arch:+.3f} | {r.alone_r2_tuned:+.3f} | {r.dm_alone_clu:+.2f} | "
                  f"{r.p_alone_clu:.1e} | {r.p_alone_holm:.4f} | {v} |")
    md.append("")

    md.append("## 2. M1 increment vs single recalibrated HAR (clustered DM<0 = text helps)\n")
    md.append("| arm | h | rel% arch | DM arch | Holm-9 arch | gen69 | rel% tuned | DM tuned | p tuned | Holm-9 tuned | placebo tuned | verdict arch → tuned |")
    md.append("|---|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|---|")
    for _, r in df.iterrows():
        md.append(f"| {r.disc} {r.base_model} | {r.h} | {r.har_rel_arch:+.2f} | {r.har_dm_arch:+.2f} | "
                  f"{r.har_holm_arch:.4f} | {'Y' if r.har_genuine69_arch else 'n'} | {r.har_rel_tuned:+.2f} | "
                  f"{r.har_dm_tuned:+.2f} | {r.har_p_tuned:.1e} | {r.har_holm_tuned:.4f} | "
                  f"{r.har_placebo_tuned:+.2f} | {r.har_verdict_arch} → **{r.har_verdict_tuned}** |")
    md.append("")

    md.append("## 3. M1 increment vs firm-identity-augmented reference\n")
    md.append("| arm | h | rel% arch | DM arch | Holm-9 arch | rel% tuned | DM tuned | p tuned | Holm-9 tuned | placebo tuned | verdict arch → tuned |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    for _, r in df.iterrows():
        md.append(f"| {r.disc} {r.base_model} | {r.h} | {r.fid_rel_arch:+.2f} | {r.fid_dm_arch:+.2f} | "
                  f"{r.fid_holm_arch:.4f} | {r.fid_rel_tuned:+.2f} | {r.fid_dm_tuned:+.2f} | "
                  f"{r.fid_p_tuned:.1e} | {r.fid_holm_tuned:.4f} | {r.fid_placebo_tuned:+.2f} | "
                  f"{r.fid_verdict_arch} → **{r.fid_verdict_tuned}** |")
    md.append("")

    md.append("## SANITY\n")
    md.append(f"- **G1 PASS** — the 9 archived C2/D2 M1-vs-HAR cells recomputed here reproduce the "
              f"committed `m1_clustered.csv` (n_obs, n_days, qlike_R/U, rel%, g_log, clustered DM, p, "
              f"placebo) to machine precision; max |diff| = {max(g1.values()):.2e}. "
              f"BASIS: `m1_clustered.csv` is the seed-2026 SINGLE-SEED table — the same basis as the "
              f"archived counterparts and the tuned runs (all seed2026). `m1_ensemble_primary.csv` is "
              f"3-seed-ensemble for C2/D2 (n_seeds=3) and is therefore NOT the comparison basis; its "
              f"seed-2026 columns (s26_*) were cross-asserted equal to `m1_clustered.csv` for these rows.")
    md.append(f"- **G2 PASS** — the firm-identity-reference machinery reproduces the committed "
              f"`crossfamily_llm.csv` qwen3_32b rows (rel/DM/p vs both references) to machine "
              f"precision; max |diff| = {max(g2.values()):.2e}.")
    md.append(f"- **G3 PASS** — standalone OOS R2 computed here reproduces each tuned run's committed "
              f"`metrics.json` test rows to machine precision; max |diff| = {max(g3.values()):.2e}. "
              f"(metrics.json QLIKE is variance-unit; this table is vol-unit by convention.)")
    md.append("")

    md.append("## HEADLINE (honest)\n")
    md.append(f"**Validation tuning does NOT rescue the challengers — the null survives tuning.** "
              f"(1) The val-QLIKE-selected configs do not even reliably improve the challengers "
              f"themselves: tuned is significantly WORSE than the archived fixed recipe in "
              f"{n_alone_worse}/9 standalone cells and better in {n_alone_better}/9 (Holm-9) — "
              f"validation selection transfers poorly to test. "
              f"(2) Vs the single recalibrated HAR, tuning yields {len(new_har)} newly-genuine cell(s) "
              f"and DESTROYS {len(lost_har)} previously genuine cell(s) "
              f"({', '.join(cellname(r) for _, r in lost_har.iterrows()) if len(lost_har) else '—'}); "
              f"genuine count {gen_har_a}/9 → {gen_har_t}/9. "
              f"(3) Vs the firm-identity reference the only movement is "
              f"{', '.join(cellname(r) for _, r in new_fid.iterrows()) if len(new_fid) else 'none'}"
              f" ({gen_fid_a}/9 → {gen_fid_t}/9) — an isolated cell, not a systematic rescue. "
              f"DA CRITICAL #1 is answered: the fixed-recipe nulls are not an artifact of "
              f"under-tuning; giving the challengers a validation-tuned arm reshuffles isolated "
              f"cells but produces no consistent text increment, and the previously-reported "
              f"increments are themselves fragile to the training recipe.")

    with open("results/tables/row3_tuned_m1.md", "w") as fh:
        fh.write("\n".join(md) + "\n")

    print("=== row3 tuned-challenger analysis done ===")
    print(f"standalone (Holm-9): tuned better {n_alone_better}/9, worse {n_alone_worse}/9")
    print(f"M1 vs HAR genuine: arch {gen_har_a}/9 -> tuned {gen_har_t}/9 "
          f"(new={len(new_har)}, lost={len(lost_har)})")
    print(f"M1 vs firmID genuine: arch {gen_fid_a}/9 -> tuned {gen_fid_t}/9 "
          f"(new={len(new_fid)}, lost={len(lost_fid)})")
    print("wrote results/tables/row3_tuned_m1.{csv,md}")


if __name__ == "__main__":
    main()
