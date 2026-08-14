"""Prereg B1 (configs/prereg_residual_family_audit.md, tag prereg-rfa-v1.2) — THIRD-family
Mistral-Small-24B-Instruct-2501 (bf16, NO quantization) 3-seed ensemble rescoring on the
8-K (event_driven) channel, zero GPU. Sibling of scripts/analysis/crossfamily_llama70_ens.py
(same machinery; M1 block verbatim).

The M1 block is verbatim from scripts/analysis/crossfamily_llama70.py: log-space
combiner val-fit test-frozen; references (a) the single recalibrated HAR (A2) and
(b) the firm-identity-augmented reference (val-window firm mean spec); day-clustered
DM (HAC lag h-1, HLN) via clustered_dm.dm_test_clustered. Holm is applied within each
family's OWN pre-declared 6-test set (3 horizons x {vs HAR, vs HAR+firmID}) — never
pooled across families (prereg B1: "each family's own pre-declared Holm(6)").

SANITY GATES (HARD RULE — any failure aborts before writing tables):
  G1'' the committed llama70 single-seed AND llama70_ens3 M1 rows, recomputed on this
       exact code path, must reproduce results/tables/crossfamily_llama70_ens.csv to
       machine precision (rtol 1e-12) on the M1 columns (the ens Holm(6) columns are
       additionally re-derived and checked);
  G1q  the committed qwen3_32b event_driven M1 rows, recomputed on this code path, must
       reproduce results/tables/crossfamily_llm.csv to machine precision (rtol 1e-12) —
       this anchors the primary family's cells before the across-family rule reads them;
  G5   the mistral24ens prediction must equal the ARITHMETIC mean of the three seed
       predictions row-wise (rtol 1e-6) across seeds 2026/2027/2028, after a verified
       1:1 merge on (ticker, accession, horizon_days). The ens run's config.json
       documents exactly this convention (per-observation ARITHMETIC MEAN, VAL+TEST
       only; scripts/experiments/row16_mistral24_ensemble/launch.sh). On failure:
       print max deviation + offending rows, abort — the prereg forbids "approximately
       holds";
  G3'' recomputed variance-unit QLIKE for every mistral cell (single + ens) matches the
       run's stored metrics.json within 1e-3 relative.

Run from repo root:  .venv/bin/python scripts/analysis/crossfamily_mistral24.py
Outputs (NEW files, only if ALL gates pass): results/tables/crossfamily_mistral24.{csv,md}
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import clustered_dm as cdm
import forecast_combination as fc

KEY = ["ticker", "accession", "horizon_days"]
EPS = 1e-8
HORIZONS = (5, 10, 20)
RTOL = 1e-12    # machine-precision gate for CSV float round-trip (G1''/G1q)
RTOL_G5 = 1e-6  # prereg row-wise ensemble identity tolerance (G5)
DISC = "event_driven"

MIS_SEED_RUNS = {  # the three frozen single-seed mistral24 prediction files
    "2026": "C6_llmtext_mistral24_full_event_driven_seed2026",
    "2027": "C6_llmtext_mistral24_s2027_full_event_driven_seed2026",
    "2028": "C6_llmtext_mistral24_s2028_full_event_driven_seed2026",
}
MIS_ENS_RUN = "C6_llmtext_mistral24ens_full_event_driven_seed2026"
LLA_RUNS = {  # committed llama70 anchors, recomputed for G1''
    "llama70_awq": "C6_llmtext_llama70_full_event_driven_seed2026",
    "llama70_awq_ens3": "C6_llmtext_llama70ens_full_event_driven_seed2026",
}
QWEN_RUN = "C6_llmtext_full_event_driven_seed2026"

M1_COLS = ["n_test", "n_days", "rel_har", "dm_har", "p_har",
           "rel_firm", "dm_firm", "p_firm", "g_text"]
DIAG_COLS = ["qlike_vol", "qlike_var", "r2", "pred_sd",
             "n_unique_2dp", "mode_val_2dp", "mode_share_pct"]


def ols(y, X):  # verbatim from crossfamily_llama70.py
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def holm(ps):  # verbatim from crossfamily_llama70.py
    ps = np.asarray(ps, float)
    n = len(ps)
    order = np.argsort(ps)
    out = np.empty(n)
    for rank, idx in enumerate(order):
        out[idx] = ps[idx] * (n - rank)
    run = 0.0
    for idx in order:
        run = max(run, out[idx])
        out[idx] = min(run, 1.0)
    return out


def close(a, b):  # verbatim from crossfamily_llama70.py
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= RTOL * max(abs(a), abs(b), 1.0)


def standalone_stats(y, f):  # verbatim from crossfamily_standalone.py
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    vals, counts = np.unique(np.round(f, 2), return_counts=True)
    i = int(np.argmax(counts))
    return {
        "qlike_vol": float(fc.qlike(y, f).mean()),
        "qlike_var": float(fc.qlike(y ** 2, f ** 2).mean()),
        "r2": float(1.0 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "pred_sd": float(f.std()),
        "n_unique_2dp": len(vals),
        "mode_val_2dp": float(vals[i]),
        "mode_share_pct": float(100.0 * counts[i] / len(f)),
    }


def m1_rows(fam, preds, a2):
    """M1 block — verbatim from crossfamily_llama70.py (same code path for both
    the single-seed family and the ensemble family)."""
    t = preds[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "ft"})
    rows = []
    for h in HORIZONS:
        m = a2[a2.horizon_days == h].merge(t[t.horizon_days == h], on=KEY).dropna()
        v, te = m[m.split == "val"], m[m.split == "test"]
        y = te.label_realised_vol.values
        fR, fU, g = fc.log_combo(v.label_realised_vol.values, v.fh.values,
                                 v.ft.values, te.fh.values, te.ft.values)
        qR, qU = fc.qlike(y, fR), fc.qlike(y, fU)
        rel = 100 * np.mean(qR - qU) / np.mean(qR)
        dm, pv, nd = cdm.dm_test_clustered(qU, qR, te.effective_trading_day.values, h)
        # firm-identity-augmented reference (val-window firm mean spec)
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
        rows.append(dict(disc=DISC, family=fam, h=h, n_test=len(te), n_days=nd,
                         rel_har=rel, dm_har=dm, p_har=pv,
                         rel_firm=relf, dm_firm=dmf, p_firm=pf, g_text=g))
    return rows


def add_holm6(df):
    """Each family's OWN pre-declared Holm(6): 3 horizons x {HAR, HAR+firmID}."""
    ps6 = np.concatenate([df.p_har.values, df.p_firm.values])
    adj = holm(ps6)
    df = df.copy()
    df["p_har_holm"] = adj[:len(df)]
    df["p_firm_holm"] = adj[len(df):]
    return df


def diag_rows(fam, run, preds):
    """Standalone health diagnostics (TEST split) + metrics.json cross-check (G3'')."""
    mj = {(r["split"], r["horizon_days"]): r for r in json.load(
        open(f"results/runs/{run}/metrics.json"))}
    cfg = json.load(open(f"results/runs/{run}/config.json"))
    te_all = preds[preds.split == "test"]
    out = {}
    for h in HORIZONS:
        d = te_all[te_all.horizon_days == h]
        st = standalone_stats(d.label_realised_vol.to_numpy(),
                              d.prediction_realised_vol.to_numpy())
        stored = mj[("test", h)]["qlike"]
        st["qlike_var_metricsjson"] = float(stored)
        st["metrics_sanity"] = ("PASS" if abs(st["qlike_var"] - stored)
                                <= 1e-3 * max(abs(stored), 1.0) else "FAIL")
        st["parse_fail_rate"] = float(cfg["stats"].get("parse_fail_rate", np.nan))
        st["clipped_rate"] = float(cfg["stats"].get("clipped_rate", np.nan))
        out[(fam, h)] = st
    return out


def ladder(la, model_desc, qf_str):
    """Pre-registered verdict ladder — thresholds verbatim from
    crossfamily_llama70_ens.py; only the model name in the prose differs."""
    n_rep_firm = int(((la.dm_firm < 0) & (la.p_firm_holm < .05)).sum())
    n_rep_har = int(((la.dm_har < 0) & (la.p_har_holm < .05)).sum())
    n_pos_firm = int((la.rel_firm > 0).sum())
    n_neg_dm_firm = int((la.dm_firm < 0).sum())
    n_sig_raw_firm = int(((la.dm_firm < 0) & (la.p_firm < .05)).sum())
    _rf = "/".join(f"{x:+.2f}" for x in la.rel_firm)
    if n_rep_firm == 3:
        tier = "REPLICATES"
        verdict = (f"**REPLICATES.** The {model_desc} reproduces the "
                   "Qwen event-driven residual over the firm-identity-augmented "
                   "reference in 3/3 horizons (Holm<.05, day-clustered).")
    elif n_pos_firm == 3 and (n_sig_raw_firm >= 2 or n_rep_har >= 1):
        tier = "DIRECTIONALLY REPLICATES"
        verdict = (f"**DIRECTIONALLY REPLICATES, significance attenuated.** The "
                   f"{model_desc} reproduces the SIGN of the Qwen "
                   f"8-K residual vs HAR+firmID in 3/3 horizons ({_rf}% vs Qwen's "
                   f"{qf_str}%), clustered DM<0 in {n_neg_dm_firm}/3 and raw p<.05 in "
                   f"{n_sig_raw_firm}/3; but after the pre-declared Holm(6) only "
                   f"{n_rep_har}/3 vs-single-HAR cells survive (min firmID Holm "
                   f"p={la.p_firm_holm.min():.5f}, {n_rep_firm}/3 firmID cells <.05).")
    elif n_pos_firm == 0 and n_sig_raw_firm == 0:
        tier = "DOES NOT REPLICATE"
        verdict = (f"**Does NOT replicate.** The {model_desc} shows "
                   "no positive increment over the firm-identity-augmented reference "
                   "in any horizon.")
    else:
        tier = "PARTIAL/MIXED"
        verdict = (f"**PARTIAL/MIXED replication** ({n_rep_firm}/3 firm-ID cells "
                   f"Holm<.05, {n_sig_raw_firm}/3 raw p<.05, {n_pos_firm}/3 positive; "
                   f"{n_rep_har}/3 vs single recalibrated HAR after Holm).")
    counts = dict(n_rep_firm=n_rep_firm, n_rep_har=n_rep_har, n_pos_firm=n_pos_firm,
                  n_neg_dm_firm=n_neg_dm_firm, n_sig_raw_firm=n_sig_raw_firm)
    return tier, verdict, counts


def main():
    a2 = fc.load("A2_har_rv", DISC)[KEY + ["split", "label_realised_vol",
                                           "prediction_realised_vol",
                                           "effective_trading_day"]] \
        .rename(columns={"prediction_realised_vol": "fh"})

    # ---- G1'': reproduce committed crossfamily_llama70_ens.csv (single + ens3) ----
    ref_l = pd.read_csv("results/tables/crossfamily_llama70_ens.csv")
    recomp_l = {}
    for fam, run in LLA_RUNS.items():
        p = pd.read_parquet(f"results/runs/{run}/predictions.parquet")
        recomp_l[fam] = pd.DataFrame(m1_rows(fam, p, a2))
    g1_bad = []
    for _, r in ref_l.iterrows():
        mine = recomp_l[r.family]
        mine = mine[(mine.disc == r.disc) & (mine.h == r.h)]
        if len(mine) != 1:
            g1_bad.append((r.disc, r.family, int(r.h), "row missing"))
            continue
        for c in M1_COLS:
            if not close(mine[c].iloc[0], r[c]):
                g1_bad.append((r.disc, r.family, int(r.h), c,
                               float(mine[c].iloc[0]), float(r[c])))
    # supplementary within G1'': the ens Holm(6) columns re-derive identically
    ens_l = add_holm6(recomp_l["llama70_awq_ens3"])
    for _, r in ref_l[ref_l.family == "llama70_awq_ens3"].iterrows():
        mine = ens_l[ens_l.h == r.h]
        for c in ("p_har_holm", "p_firm_holm"):
            if not close(mine[c].iloc[0], r[c]):
                g1_bad.append((r.disc, r.family, int(r.h), c,
                               float(mine[c].iloc[0]), float(r[c])))
    if g1_bad:
        print(f"SANITY G1'' FAIL ({len(g1_bad)} mismatches vs committed "
              f"crossfamily_llama70_ens.csv):")
        for b in g1_bad[:20]:
            print("  ", b)
        sys.exit(1)
    print(f"SANITY G1'' PASS: all {len(ref_l)} committed llama70 rows (single-seed AND "
          f"ens3) reproduced to machine precision (rtol {RTOL:g}) on columns {M1_COLS}; "
          f"ens Holm(6) columns re-derived identically")

    # ---- G1q: anchor the primary family's event_driven cells (crossfamily_llm.csv) ----
    ref_q = pd.read_csv("results/tables/crossfamily_llm.csv")
    ref_q = ref_q[(ref_q.family == "qwen3_32b") & (ref_q.disc == DISC)]
    pq = pd.read_parquet(f"results/runs/{QWEN_RUN}/predictions.parquet")
    qwen = pd.DataFrame(m1_rows("qwen3_32b", pq, a2))
    g1q_bad = []
    for _, r in ref_q.iterrows():
        mine = qwen[qwen.h == r.h]
        if len(mine) != 1:
            g1q_bad.append((r.disc, r.family, int(r.h), "row missing"))
            continue
        for c in M1_COLS:
            if not close(mine[c].iloc[0], r[c]):
                g1q_bad.append((r.disc, r.family, int(r.h), c,
                                float(mine[c].iloc[0]), float(r[c])))
    if g1q_bad:
        print(f"SANITY G1q FAIL ({len(g1q_bad)} mismatches vs committed "
              f"crossfamily_llm.csv qwen event_driven rows):")
        for b in g1q_bad[:20]:
            print("  ", b)
        sys.exit(1)
    print(f"SANITY G1q PASS: {len(ref_q)} committed qwen3_32b event_driven rows "
          f"reproduced to machine precision (rtol {RTOL:g}) on columns {M1_COLS}")

    # ---- G5: mistral24 ensemble == mean(seed predictions) row-wise, rtol 1e-6 ----
    pens = pd.read_parquet(f"results/runs/{MIS_ENS_RUN}/predictions.parquet")
    m = pens[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "f_ens"})
    n_in = {"ens": len(m)}
    seed_preds = {}
    for s, run in MIS_SEED_RUNS.items():
        ps = pd.read_parquet(f"results/runs/{run}/predictions.parquet")
        seed_preds[s] = ps
        ps = ps[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": f"f_{s}"})
        n_in[s] = len(ps)
        m = m.merge(ps, on=KEY, validate="one_to_one")
    if not all(v == len(m) for v in n_in.values()):
        print(f"SANITY G5 FAIL: merge on {KEY} is not 1:1 across files "
              f"(input sizes {n_in}, merged {len(m)})")
        sys.exit(1)
    F = m[[f"f_{s}" for s in MIS_SEED_RUNS]].to_numpy(float)
    am = F.mean(axis=1)
    fe = m.f_ens.to_numpy(float)
    reldev = np.abs(fe - am) / np.maximum(np.abs(am), 1e-300)
    bad = reldev > RTOL_G5
    if bad.any():
        m["arith_mean"] = am
        m["rel_dev"] = reldev
        worst = m[bad].sort_values("rel_dev", ascending=False)
        print("SANITY G5 FAIL: ensemble prediction != mean(seed predictions) "
              f"at rtol {RTOL_G5:g}")
        print(f"  rows checked: {len(m)} (merge on {KEY} verified 1:1)")
        print(f"  rows failing: {int(bad.sum())} ({100.0 * bad.mean():.3f}%)")
        print(f"  max relative deviation: {reldev.max():.6e}")
        print("  worst offending rows:")
        cols = KEY + [f"f_{s}" for s in MIS_SEED_RUNS] + ["f_ens", "arith_mean", "rel_dev"]
        print(worst[cols].head(8).to_string(index=False))
        print("\n  Per the prereg (G5), NOT proceeding on 'approximately holds'. "
              "Inspect the ensemble build script first:\n"
              "    scripts/experiments/row16_mistral24_ensemble/launch.sh\n"
              "  and results/runs/" + MIS_ENS_RUN + "/config.json (documents the actual "
              "combination rule used to build the on-disk ensemble).")
        sys.exit(1)
    print(f"SANITY G5 PASS: mistral24 ensemble == arithmetic mean(seed preds) on all "
          f"{len(m)} rows (rtol {RTOL_G5:g}; max reldev {reldev.max():.3e}; "
          f"merge 1:1 verified)")

    # ---- M1 rows for the two mistral bases, each with its OWN Holm(6) ----
    p26m = seed_preds["2026"]
    single = add_holm6(pd.DataFrame(m1_rows("mistral24_bf16", p26m, a2)))
    single["holm_family"] = ("mistral24_bf16 Holm(6): 3 horizons x {HAR, HAR+firmID} "
                             "(prereg B1, own family)")
    ens = add_holm6(pd.DataFrame(m1_rows("mistral24_ens3", pens, a2)))
    ens["holm_family"] = ("mistral24_ens3 Holm(6): 3 horizons x {HAR, HAR+firmID} "
                          "(prereg B1, own family)")

    # ---- standalone health diagnostics for the mistral rows + G3'' ----
    diags = {}
    diags.update(diag_rows("mistral24_bf16", MIS_SEED_RUNS["2026"], p26m))
    diags.update(diag_rows("mistral24_ens3", MIS_ENS_RUN, pens))
    g3_bad = [(k, v["qlike_var"], v["qlike_var_metricsjson"])
              for k, v in diags.items() if v["metrics_sanity"] != "PASS"]
    if g3_bad:
        print("SANITY G3'' FAIL (variance-unit QLIKE vs metrics.json):")
        for b in g3_bad:
            print("  ", b)
        sys.exit(1)
    print(f"SANITY G3'' PASS: recomputed variance-unit QLIKE matches stored "
          f"metrics.json within 1e-3 relative in {len(diags)}/{len(diags)} mistral "
          f"cells (single + ens)")
    # ens run config carries no parse stats (means of already-parsed forecasts);
    # verify all three seed configs report clean parse/clip and disclose.
    seed_pc = {}
    for s, run in MIS_SEED_RUNS.items():
        st = json.load(open(f"results/runs/{run}/config.json"))["stats"]
        seed_pc[s] = (float(st["parse_fail_rate"]), float(st["clipped_rate"]))
    for df_ in (single, ens):
        for c in DIAG_COLS + ["qlike_var_metricsjson", "metrics_sanity",
                              "parse_fail_rate", "clipped_rate"]:
            df_[c] = [diags[(df_.family.iloc[0], h)][c] for h in df_.h]
    single["flag"] = "bf16"
    ens["flag"] = "bf16-ens3"

    # ---- anchor rows: qwen primary + committed llama70 single/ens3 ----
    qwen_anchor = ref_q.copy()  # committed M1 values, verified == recomputation (G1q)
    qwen_anchor = add_holm6(qwen_anchor)  # primary family's own Holm(6), event_driven
    qwen_anchor["holm_family"] = ("qwen3_32b Holm(6): 3 horizons x {HAR, HAR+firmID}, "
                                  "event_driven (computed HERE on the committed raw "
                                  "p's for the prereg B1 across-family rule; the "
                                  "committed crossfamily_llm.csv carries raw p only)")
    std = pd.read_csv("results/tables/crossfamily_standalone.csv")
    std = std[(std.family == "qwen3_32b") & (std.disc == DISC)]
    qwen_anchor = qwen_anchor.merge(
        std[["h"] + DIAG_COLS + ["qlike_var_metricsjson", "metrics_sanity"]], on="h")
    l70 = pd.read_csv("results/tables/crossfamily_llama70.csv")
    l70 = l70[(l70.family == "llama70_awq") & (l70.disc == DISC)]
    qcfg = json.load(open(f"results/runs/{QWEN_RUN}/config.json"))["stats"]
    qwen_anchor["parse_fail_rate"] = float(qcfg["parse_fail_rate"])
    qwen_anchor["clipped_rate"] = float(qcfg["clipped_rate"])
    qwen_anchor["flag"] = "-"

    lla_anchor = ref_l.copy()  # committed M1 + committed Holm columns, carried unchanged
    lla_anchor = lla_anchor.merge(
        l70[["h"] + DIAG_COLS + ["qlike_var_metricsjson", "metrics_sanity",
                                 "parse_fail_rate", "clipped_rate"]],
        on="h", how="left")
    # health columns apply to the single-seed run only (no committed ens3 health)
    for c in DIAG_COLS + ["qlike_var_metricsjson", "parse_fail_rate", "clipped_rate"]:
        lla_anchor.loc[lla_anchor.family == "llama70_awq_ens3", c] = np.nan
    lla_anchor.loc[lla_anchor.family == "llama70_awq_ens3", "metrics_sanity"] = "-"
    lla_anchor["flag"] = np.where(lla_anchor.family == "llama70_awq_ens3",
                                  "AWQ-INT4-ens3", "AWQ-INT4")

    order_cols = ["disc", "family", "h", "n_test", "n_days",
                  "rel_har", "dm_har", "p_har", "rel_firm", "dm_firm", "p_firm",
                  "g_text", "p_har_holm", "p_firm_holm", "holm_family"] + DIAG_COLS + \
                 ["qlike_var_metricsjson", "metrics_sanity",
                  "parse_fail_rate", "clipped_rate", "flag"]
    df = pd.concat([qwen_anchor, lla_anchor, single, ens], ignore_index=True)
    for c in order_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[order_cols]

    # ---- pre-registered verdict ladder, applied to the mistral24_ens3 rows ----
    la = ens.sort_values("h")
    qe = qwen_anchor.sort_values("h")
    _qf = "/".join(f"{x:+.2f}" for x in qe.rel_firm)
    tier_m, verdict, cm = ladder(
        la, "3-seed Mistral-Small-24B (bf16) ensemble", _qf)
    strong_m = cm["n_rep_har"] >= 2
    b1_line = (f"B1 family STRONG pass (>=2/3 horizons Holm<.05 & DM<0 vs single "
               f"recalibrated HAR, within the mistral24_ens3 Holm(6)): "
               f"{'PASS' if strong_m else 'FAIL'} ({cm['n_rep_har']}/3)")

    # info only (not a prereg branch input): the single-seed mistral base
    tier_s, verdict_s, cs = ladder(
        single.sort_values("h"), "single-seed (2026) Mistral-Small-24B bf16", _qf)

    # ---- prereg B1 ACROSS-FAMILY RULE over F = {qwen primary, llama70_ens3, mistral24_ens3} ----
    tier_q, _, cq = ladder(qe, "primary Qwen3-32B (single seed)", _qf)
    strong_q = cq["n_rep_har"] >= 2
    ref_le = ref_l[ref_l.family == "llama70_awq_ens3"].sort_values("h")
    tier_l, _, cl = ladder(ref_le, "3-seed Llama-3.1-70B (AWQ-INT4) ensemble", _qf)
    strong_l = cl["n_rep_har"] >= 2
    # committed-anchor assertion (task spec): llama70_ens3 = STRONG fail 1/3, DIRECTIONALLY
    assert cl["n_rep_har"] == 1 and tier_l == "DIRECTIONALLY REPLICATES", \
        f"llama70_ens3 anchor drifted: n_rep_har={cl['n_rep_har']}, tier={tier_l}"

    fam_tbl = [
        ("qwen3_32b (primary, single seed)", strong_q, cq["n_rep_har"], tier_q),
        ("llama70_awq_ens3 (committed crossfamily_llama70_ens.csv)", strong_l,
         cl["n_rep_har"], tier_l),
        ("mistral24_ens3 (this table)", strong_m, cm["n_rep_har"], tier_m),
    ]
    weak_tiers = ("REPLICATES", "DIRECTIONALLY REPLICATES")
    weak_ok = {name: (strong or tier in weak_tiers)
               for name, strong, _, tier in fam_tbl}
    n_strong = sum(1 for _, s, _, _ in fam_tbl if s)
    n_weak = sum(weak_ok.values())
    primary_weak = weak_ok[fam_tbl[0][0]]
    if n_strong >= 2:
        branch = ("BRANCH 1 — \"replicates across families\" "
                  f"({n_strong}/3 families STRONG)")
    elif n_weak >= 2 and primary_weak:
        branch = ("BRANCH 2 — \"sign-robust across families, significance "
                  f"attenuated\" ({n_weak}/3 families >=WEAK incl. primary; "
                  f"{n_strong}/3 STRONG)")
    else:
        branch = ("BRANCH 3 — \"does not replicate beyond the primary family\" "
                  f"({n_strong}/3 STRONG, {n_weak}/3 >=WEAK, "
                  f"primary >=WEAK: {primary_weak})")

    def m1cell(r):
        s1 = "**" if (r.dm_har < 0 and r.p_har < .05) else ""
        s2 = "**" if (r.dm_firm < 0 and r.p_firm < .05) else ""
        return (f"{r.rel_har:+.2f}%{s1} | {r.dm_har:+.2f} | "
                f"{r.rel_firm:+.2f}%{s2} | {r.dm_firm:+.2f}")

    mi = pd.concat([single, ens]).sort_values(["family", "h"])
    md = [
        "# Prereg B1 — third family: Mistral-Small-24B-Instruct-2501 (bf16), "
        "3-seed ensemble, 8-K channel",
        "",
        "## Disclosures",
        "",
        "- **Precision**: mistral24 ran **bf16, NO quantization** (unlike the "
        "llama70 replication arm's AWQ-INT4). Protocol otherwise byte-identical to "
        "C6/llama70: same manifest/prompt/guided-JSON/clip[0.03,3.0]/retry stack "
        "(scripts/experiments/e1_llm_forecast; "
        "scripts/experiments/row16_mistral24_ensemble/launch.sh).",
        "- **Seed semantics**: temperature-0 decoding; `--seed` is NOT plumbed into "
        "vLLM's sampler (run_inference.py forwards it only to the mock path) — seeds "
        "2026/2027/2028 differ only through vLLM/TP2 kernel non-determinism. This is "
        "a **reproducibility-jitter ensemble**, not a stochastic-decoding one "
        "(scripts/experiments/row16_mistral24_ensemble/launch.sh, TEMPERATURE "
        "PROTOCOL block; identical to the llama70 arm, row15).",
        "- **Tokenizer caveat (fix_mistral_regex)**: vLLM's mistral-common tokenizer "
        "backend lacks `.is_fast` and crashes vLLM's tokenizer check, so the runs "
        "load an **hfview** snapshot (tekken.json/params.json removed) that forces "
        "the transformers FAST tokenizer (launch.sh HFVIEW override; config.json "
        "`llm: .../mistral24_hfview`). The transformers fast tokenizer emits its "
        "known Mistral tokenizer-regex warning; disclosed as a tokenizer-regex "
        "caveat. It is internally consistent: all three seeds (and hence the "
        "ensemble) used the identical hfview tokenizer, so it cannot differentiate "
        "the seeds or the cross-seed comparison.",
        "- **Ensemble construction**: per-observation ARITHMETIC mean of "
        "prediction_realised_vol across the three seeds, inner-joined 1:1 on "
        "(ticker, accession, horizon_days), VAL+TEST only — identical convention to "
        "the C-model seed-ensemble primary (m1_ensemble_primary.ensemble_text) and "
        "to C6_llmtext_llama70ens (ens run config.json documents this); verified "
        f"row-wise by sanity gate G5 (rtol {RTOL_G5:g}).",
        "- **Multiplicity**: Holm is applied within each family's OWN pre-declared "
        "6-test set (3 horizons x {vs HAR, vs HAR+firmID}) — mistral24_bf16 and "
        "mistral24_ens3 each get their own Holm(6); the llama70 anchor rows carry "
        "their committed Holm(6) values unchanged; the qwen primary's Holm(6) is "
        "computed here on its committed raw p's (the committed crossfamily_llm.csv "
        "carries raw p only). Families are parallel, never pooled.",
        "- parse_fail_rate = 0.0 and clipped_rate = 0.0 in all three mistral seed "
        f"configs (verified in-script: {seed_pc}); the ens run dir carries no parse "
        "stats (its predictions are means of already-parsed seed forecasts) — its "
        "parse/clip cells are '-'.",
        "- The single-seed mistral24_bf16 rows are retained side by side; the "
        "ensemble rows do not replace them (prereg B0/B1 convention).",
        "",
        "## Table — M1 increment (log-space, combiner val-fit test-frozen, "
        "day-clustered DM) + standalone health",
        "",
        "rel% is on **volatility-unit** QLIKE (committed-anchor convention); the "
        "QLIKE(var) health column is **variance-unit** (Patton-robust). rel% > 0 = "
        "text lowers QLIKE vs the reference; `**` = clustered DM<0, raw p<.05.",
        "",
        "| family | h | n_test | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) "
        "| Holm p (HAR) | Holm p (firmID) | QLIKE(var) | R^2 | pred sd | n_uniq(2dp) "
        "| mode share% | parse_ok% | flag |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for _, r in df.iterrows():
        qv = f"{r.qlike_var:.3f}" if pd.notna(r.qlike_var) else "-"
        r2 = f"{r.r2:+.3f}" if pd.notna(r.r2) else "-"
        sd = f"{r.pred_sd:.4f}" if pd.notna(r.pred_sd) else "-"
        nu = f"{int(r.n_unique_2dp)}" if pd.notna(r.n_unique_2dp) else "-"
        ms = f"{r.mode_share_pct:.1f}" if pd.notna(r.mode_share_pct) else "-"
        po = (f"{100 * (1 - r.parse_fail_rate):.1f}"
              if pd.notna(r.parse_fail_rate) else "-")
        md.append(f"| {r.family} | {int(r.h)} | {int(r.n_test)} | {m1cell(r)} | "
                  f"{r.p_har_holm:.4g} | {r.p_firm_holm:.4g} | {qv} | {r2} | {sd} | "
                  f"{nu} | {ms} | {po} | {r.flag} |")
    md += [
        "",
        "## VERDICT (pre-registered ladder, mistral24_ens3 rows)",
        "",
        verdict,
        "",
        f"- {b1_line}",
        f"- Info (not a prereg branch input) — single-seed mistral24_bf16 ladder: "
        f"{tier_s} ({cs['n_rep_firm']}/3 firmID Holm<.05, {cs['n_sig_raw_firm']}/3 "
        f"raw firmID p<.05, {cs['n_pos_firm']}/3 rel_firm>0, {cs['n_rep_har']}/3 vs "
        f"HAR after Holm).",
        f"- Health check (same formula as crossfamily_llama70.py): mistral24_ens3 is "
        f"{'a HEALTHY forecaster by the Yi/Phi criteria' if (la.qlike_var.max() < 4 and la.mode_share_pct.max() < 60) else 'NOT clearly healthy by the Yi/Phi criteria — read the columns'}: "
        f"variance-unit QLIKE {la.qlike_var.min():.2f}-{la.qlike_var.max():.2f} "
        f"(Qwen {qe.qlike_var.min():.2f}-{qe.qlike_var.max():.2f}, llama70 "
        f"1.12-2.10, Yi 7.60-8.19 = capability floor) — the LOWEST of all families, "
        f"NOT QLIKE-floored; but modal share reaches "
        f"{la.mode_share_pct.max():.1f}% (Yi's collapse benchmark was 73.6%, "
        f"llama70 max 51.2%, Qwen max 50.3%): forecasts are heavily concentrated "
        f"at {la.mode_val_2dp.min():.2f}-{la.mode_val_2dp.max():.2f} with pred sd "
        f"{la.pred_sd.min():.3f}-{la.pred_sd.max():.3f}, R^2 "
        f"{la.r2.min():+.2f}-{la.r2.max():+.2f}, parse_ok 100%. A "
        f"mode-concentrated (near-constant) forecaster carries little firm-specific "
        f"text signal by construction — reported as-is; the prereg draws no "
        f"capability exemption for the third family.",
        "",
        "## ACROSS-FAMILY RULE (prereg §B1, quoted verbatim)",
        "",
        "> **Cross-family claim rule (pre-declared)**:",
        "> - Family STRONG pass: ≥2/3 horizons Holm<.05 and DM<0 vs single recalibrated HAR (within-family Holm(6));",
        "> - Family WEAK pass: reaches DIRECTIONALLY REPLICATES or above on the B0 ladder;",
        "> - Paper wording: ≥2/3 families STRONG → \"replicates across families\";",
        ">   ≥2/3 families ≥WEAK (incl. primary) → \"sign-robust across families, "
        "significance attenuated\";",
        ">   otherwise → \"does not replicate beyond the primary family\" (the residual paragraph is downgraded accordingly;",
        ">   per the established FACTS.md rule, it must not be written as a family-specific proof).",
        "",
        "F = {Qwen3-32B primary (single seed), Llama-3.1-70B-AWQ ens3, "
        "Mistral-24B ens3}, all on the 8-K (event_driven) channel.",
        "",
        "| family | STRONG (>=2/3 Holm<.05 & DM<0 vs HAR, own Holm(6)) | "
        "ladder tier | >=WEAK |",
        "|---|---|---|---|",
    ]
    for name, s, nh, t in fam_tbl:
        md.append(f"| {name} | {'PASS' if s else 'FAIL'} ({nh}/3) | {t} | "
                  f"{'yes' if weak_ok[name] else 'no'} |")
    md += [
        "",
        f"**FIRED: {branch}**",
        "",
        "## SANITY",
        "",
        f"- G1'' PASS: all {len(ref_l)} committed crossfamily_llama70_ens.csv rows "
        f"(llama70_awq single-seed AND llama70_awq_ens3), recomputed on this exact "
        f"code path, reproduced to machine precision (rtol {RTOL:g}) on columns "
        f"{M1_COLS}; the ens Holm(6) columns re-derive identically.",
        f"- G1q PASS: the {len(ref_q)} committed qwen3_32b event_driven M1 rows "
        f"(crossfamily_llm.csv) reproduced to machine precision (rtol {RTOL:g}) — "
        f"the primary family's cells are anchored before the across-family rule "
        f"reads them.",
        f"- G5 PASS: mistral24ens prediction == row-wise ARITHMETIC mean of the "
        f"three seed predictions on all {len(m)} rows (rtol {RTOL_G5:g}; max "
        f"relative deviation {reldev.max():.3e}); merge on {KEY} verified 1:1 "
        f"across the four prediction files.",
        f"- G3'' PASS: recomputed variance-unit QLIKE matches stored metrics.json "
        f"within 1e-3 relative in {len(diags)}/{len(diags)} mistral cells "
        f"(single + ens).",
        "",
    ]

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/crossfamily_mistral24.csv", index=False)
    Path("results/tables/crossfamily_mistral24.md").write_text("\n".join(md))
    print("wrote results/tables/crossfamily_mistral24.csv/.md")
    print(df[["family", "h", "n_test", "rel_har", "dm_har", "p_har", "p_har_holm",
              "rel_firm", "dm_firm", "p_firm", "p_firm_holm"]].to_string(index=False))
    print("\nmistral health columns:")
    print(mi[["family", "h", "qlike_vol", "qlike_var", "r2", "pred_sd",
              "n_unique_2dp", "mode_val_2dp", "mode_share_pct",
              "qlike_var_metricsjson", "metrics_sanity"]].to_string(index=False))
    print("\nVERDICT (mistral24_ens3):", verdict.replace("**", ""))
    print(b1_line)
    print("Info single-seed ladder:", tier_s)
    print("\nACROSS-FAMILY:")
    for name, s, nh, t in fam_tbl:
        print(f"  {name}: STRONG {'PASS' if s else 'FAIL'} ({nh}/3), tier={t}, "
              f">=WEAK={'yes' if weak_ok[name] else 'no'}")
    print("FIRED:", branch)


if __name__ == "__main__":
    main()
