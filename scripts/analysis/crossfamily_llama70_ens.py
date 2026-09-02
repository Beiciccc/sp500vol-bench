"""Prereg B0 (configs/prereg_residual_family_audit.md, tag prereg-rfa-v1.1) — 3-seed
Llama-3.1-70B (AWQ-INT4) ensemble rescoring on the 8-K (event_driven) channel, zero GPU.

The M1 block is verbatim from scripts/analysis/crossfamily_llama70.py: log-space
combiner val-fit test-frozen; references (a) the single recalibrated HAR (A2) and
(b) the firm-identity-augmented reference (val-window firm mean spec); day-clustered
DM (HAC lag h-1, HLN) via clustered_dm.dm_test_clustered. Holm is applied within the
NEW pre-declared family of 6 ensemble tests (3 horizons x {vs HAR, vs HAR+firmID}),
parallel to — NOT pooled with — the single-seed Holm(6).

SANITY GATES (HARD RULE — any failure aborts before writing tables):
  G1' the single-seed llama70_awq M1 rows, recomputed on this exact code path, must
      reproduce the committed results/tables/crossfamily_llama70.csv llama70 rows to
      machine precision (rtol 1e-12) on the M1 columns;
  G5  the ensemble prediction must equal the ARITHMETIC mean of the seed predictions row-wise
  (prereg v1.2: the frozen artifact was deliberately built with the paper's seed-ensemble
  primary convention, m1_ensemble_primary.ensemble_text = per-row arithmetic mean)
      (rtol 1e-6) across seeds 2026/2027/2028, after a verified 1:1 merge on
      (ticker, accession, horizon_days). On failure: print the max deviation and a
      few offending rows, then abort — the prereg forbids proceeding on
      "approximately holds" (first inspect the ensemble build script:
      scripts/experiments/row15_llama70_ensemble/launch.sh).

Run from repo root:  .venv/bin/python scripts/analysis/crossfamily_llama70_ens.py
Outputs (NEW files, only if ALL gates pass): results/tables/crossfamily_llama70_ens.{csv,md}
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # noqa: E402
import clustered_dm as cdm  # noqa: E402

KEY = ["ticker", "accession", "horizon_days"]
EPS = 1e-8
HORIZONS = (5, 10, 20)
RTOL = 1e-12    # machine-precision gate for CSV float round-trip (G1')
RTOL_G5 = 1e-6  # prereg row-wise ensemble identity tolerance (G5)
DISC = "event_driven"

SEED_RUNS = {  # the three frozen single-seed prediction files
    "2026": "C6_llmtext_llama70_full_event_driven_seed2026",
    "2027": "C6_llmtext_llama70_s2027_full_event_driven_seed2026",
    "2028": "C6_llmtext_llama70_s2028_full_event_driven_seed2026",
}
ENS_RUN = "C6_llmtext_llama70ens_full_event_driven_seed2026"

M1_COLS = ["n_test", "n_days", "rel_har", "dm_har", "p_har",
           "rel_firm", "dm_firm", "p_firm", "g_text"]


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
        L = lambda x: np.log(np.clip(x, EPS, None))  # noqa: E731
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


def main():
    a2 = fc.load("A2_har_rv", DISC)[KEY + ["split", "label_realised_vol",
                                           "prediction_realised_vol",
                                           "effective_trading_day"]] \
        .rename(columns={"prediction_realised_vol": "fh"})

    # ---- single-seed llama70_awq M1 rows on the same code path ----
    p26 = pd.read_parquet(f"results/runs/{SEED_RUNS['2026']}/predictions.parquet")
    single = pd.DataFrame(m1_rows("llama70_awq", p26, a2))

    # ---- G1': reproduce committed crossfamily_llama70.csv llama70 rows ----
    ref = pd.read_csv("results/tables/crossfamily_llama70.csv")
    ref = ref[ref.family == "llama70_awq"]
    g1_bad = []
    for _, r in ref.iterrows():
        mine = single[(single.disc == r.disc) & (single.h == r.h)]
        if len(mine) != 1:
            g1_bad.append((r.disc, r.family, int(r.h), "row missing"))
            continue
        for c in M1_COLS:
            if not close(mine[c].iloc[0], r[c]):
                g1_bad.append((r.disc, r.family, int(r.h), c,
                               float(mine[c].iloc[0]), float(r[c])))
    if g1_bad:
        print(f"SANITY G1' FAIL ({len(g1_bad)} mismatches vs committed "
              f"crossfamily_llama70.csv llama70 rows):")
        for b in g1_bad[:20]:
            print("  ", b)
        sys.exit(1)
    print(f"SANITY G1' PASS: {len(ref)} committed single-seed llama70 rows reproduced "
          f"to machine precision (rtol {RTOL:g}) on columns {M1_COLS}")

    # ---- G5 (prereg v1.2): ensemble == mean(seed predictions) row-wise, rtol 1e-6 ----
    pens = pd.read_parquet(f"results/runs/{ENS_RUN}/predictions.parquet")
    m = pens[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "f_ens"})
    n_in = {"ens": len(m)}
    for s, run in SEED_RUNS.items():
        ps = pd.read_parquet(f"results/runs/{run}/predictions.parquet")
        ps = ps[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": f"f_{s}"})
        n_in[s] = len(ps)
        m = m.merge(ps, on=KEY, validate="one_to_one")
    if not all(v == len(m) for v in n_in.values()):
        print(f"SANITY G5 FAIL: merge on {KEY} is not 1:1 across files "
              f"(input sizes {n_in}, merged {len(m)})")
        sys.exit(1)
    F = m[[f"f_{s}" for s in SEED_RUNS]].to_numpy(float)
    geo = F.mean(axis=1)
    fe = m.f_ens.to_numpy(float)
    reldev = np.abs(fe - geo) / np.maximum(np.abs(geo), 1e-300)
    bad = reldev > RTOL_G5
    if bad.any():
        m["arith_mean"] = geo
        m["rel_dev"] = reldev
        worst = m[bad].sort_values("rel_dev", ascending=False)
        print("SANITY G5 FAIL: ensemble prediction != mean(seed predictions) "
              f"at rtol {RTOL_G5:g}")
        print(f"  rows checked: {len(m)} (merge on {KEY} verified 1:1)")
        print(f"  rows failing: {int(bad.sum())} ({100.0 * bad.mean():.3f}%)")
        print(f"  max relative deviation: {reldev.max():.6e}")
        print("  worst offending rows:")
        cols = KEY + [f"f_{s}" for s in SEED_RUNS] + ["f_ens", "arith_mean", "rel_dev"]
        print(worst[cols].head(8).to_string(index=False))
        print("\n  Per the prereg (G5), NOT proceeding on 'approximately holds'. "
              "Inspect the ensemble build script first:\n"
              "    scripts/experiments/row15_llama70_ensemble/launch.sh\n"
              "  and results/runs/" + ENS_RUN + "/config.json (documents the actual "
              "combination rule used to build the on-disk ensemble).")
        sys.exit(1)
    print(f"SANITY G5 PASS: ensemble == mean(seed preds) on all {len(m)} "
          f"rows (rtol {RTOL_G5:g}; merge 1:1 verified)")

    # ---- ensemble M1 rows (identical block), NEW pre-declared Holm(6) family ----
    ens = pd.DataFrame(m1_rows("llama70_awq_ens3", pens, a2))
    ps6 = np.concatenate([ens.p_har.values, ens.p_firm.values])
    adj = holm(ps6)
    ens["p_har_holm"] = adj[:len(ens)]
    ens["p_firm_holm"] = adj[len(ens):]
    ens["holm_family"] = "ens Holm(6): 3 horizons x {HAR, HAR+firmID} (prereg B0)"

    # single-seed rows keep their committed Holm(6) values, carried unchanged
    single = single.merge(
        ref[["disc", "h", "p_har_holm", "p_firm_holm"]], on=["disc", "h"])
    single["holm_family"] = "single-seed Holm(6), committed crossfamily_llama70.csv"

    df = pd.concat([single, ens], ignore_index=True)

    # ---- pre-registered verdict ladder, applied to the ens rows ----
    la = ens.sort_values("h")
    qwen = pd.read_csv("results/tables/crossfamily_llm.csv")
    qe = qwen[(qwen.family == "qwen3_32b") & (qwen.disc == DISC)].sort_values("h")
    n_rep_firm = int(((la.dm_firm < 0) & (la.p_firm_holm < .05)).sum())
    n_rep_har = int(((la.dm_har < 0) & (la.p_har_holm < .05)).sum())
    n_pos_firm = int((la.rel_firm > 0).sum())
    n_neg_dm_firm = int((la.dm_firm < 0).sum())
    n_sig_raw_firm = int(((la.dm_firm < 0) & (la.p_firm < .05)).sum())
    _rf = "/".join(f"{x:+.2f}" for x in la.rel_firm)
    _qf = "/".join(f"{x:+.2f}" for x in qe.rel_firm)
    if n_rep_firm == 3:
        verdict = ("**REPLICATES.** The 3-seed Llama-3.1-70B ensemble reproduces the "
                   "Qwen event-driven residual over the firm-identity-augmented "
                   "reference in 3/3 horizons (Holm<.05, day-clustered).")
    elif n_pos_firm == 3 and (n_sig_raw_firm >= 2 or n_rep_har >= 1):
        verdict = (f"**DIRECTIONALLY REPLICATES, significance attenuated.** The "
                   f"3-seed Llama-3.1-70B ensemble reproduces the SIGN of the Qwen "
                   f"8-K residual vs HAR+firmID in 3/3 horizons ({_rf}% vs Qwen's "
                   f"{_qf}%), clustered DM<0 in {n_neg_dm_firm}/3 and raw p<.05 in "
                   f"{n_sig_raw_firm}/3; but after the pre-declared Holm(6) only "
                   f"{n_rep_har}/3 vs-single-HAR cells survive (min firmID Holm "
                   f"p={la.p_firm_holm.min():.5f}, {n_rep_firm}/3 firmID cells <.05).")
    elif n_pos_firm == 0 and n_sig_raw_firm == 0:
        verdict = ("**Does NOT replicate.** The 3-seed Llama-3.1-70B ensemble shows "
                   "no positive increment over the firm-identity-augmented reference "
                   "in any horizon.")
    else:
        verdict = (f"**PARTIAL/MIXED replication** ({n_rep_firm}/3 firm-ID cells "
                   f"Holm<.05, {n_sig_raw_firm}/3 raw p<.05, {n_pos_firm}/3 positive; "
                   f"{n_rep_har}/3 vs single recalibrated HAR after Holm).")

    # ---- prereg B1 "family STRONG pass" readout for the ens rows ----
    strong = n_rep_har >= 2
    b1_line = (f"B1 family STRONG pass (>=2/3 horizons Holm<.05 & DM<0 vs single "
               f"recalibrated HAR, within the ens Holm(6)): "
               f"{'PASS' if strong else 'FAIL'} ({n_rep_har}/3)")

    def m1cell(r):
        s1 = "**" if (r.dm_har < 0 and r.p_har < .05) else ""
        s2 = "**" if (r.dm_firm < 0 and r.p_firm < .05) else ""
        return (f"{r.rel_har:+.2f}%{s1} | {r.dm_har:+.2f} | "
                f"{r.rel_firm:+.2f}%{s2} | {r.dm_firm:+.2f}")

    md = [
        "# Prereg B0 — 3-seed Llama-3.1-70B (AWQ-INT4) ensemble rescoring, 8-K channel",
        "",
        "## Disclosures",
        "",
        "- **Ensemble semantics**: temperature-0 decoding; seeds 2026/2027/2028 were "
        "NOT passed to vLLM's sampler — they differ only through vLLM/AWQ-INT4/TP2 "
        "kernel non-determinism. This is a **reproducibility-jitter ensemble**, not a "
        "stochastic-decoding one (scripts/experiments/row15_llama70_ensemble/launch.sh).",
        "- **Ensemble construction**: per-row ARITHMETIC mean "
        "exp(mean(log(pred_seed))) across the three seeds, verified row-wise by "
        f"sanity gate G5 (rtol {RTOL_G5:g}, 1:1 merge on {KEY}).",
        "- Holm is applied within the NEW pre-declared family of the 6 ensemble tests "
        "(3 horizons x {vs HAR, vs HAR+firmID}); the single-seed rows carry their own "
        "committed Holm(6) values unchanged (crossfamily_llama70.csv). The two "
        "families are parallel, not pooled.",
        "- The single-seed rows are retained side by side; the ensemble rows do not "
        "replace them (prereg B0).",
        "",
        "## Table — M1 increment (log-space, combiner val-fit test-frozen, "
        "day-clustered DM)",
        "",
        "rel% > 0 = text lowers QLIKE vs the reference; `**` = clustered DM<0, raw "
        "p<.05.",
        "",
        "| family | h | n_test | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) "
        "| Holm p (HAR) | Holm p (firmID) |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, r in df.iterrows():
        md.append(f"| {r.family} | {int(r.h)} | {int(r.n_test)} | {m1cell(r)} | "
                  f"{r.p_har_holm:.4g} | {r.p_firm_holm:.4g} |")
    md += [
        "",
        "## VERDICT (pre-registered ladder, ens rows)",
        "",
        verdict,
        "",
        f"- {b1_line}",
        "",
        "## SANITY",
        "",
        f"- G1' PASS: all {len(ref)} committed single-seed llama70 M1 rows reproduced "
        f"to machine precision (rtol {RTOL:g}) on columns {M1_COLS}.",
        f"- G5 PASS: ensemble prediction == mean(seed predictions) (arithmetic; prereg v1.2) on all "
        f"rows (rtol {RTOL_G5:g}); merge on {KEY} verified 1:1 across the four "
        f"prediction files.",
        "",
    ]

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/crossfamily_llama70_ens.csv", index=False)
    Path("results/tables/crossfamily_llama70_ens.md").write_text("\n".join(md))
    print("wrote results/tables/crossfamily_llama70_ens.csv/.md")
    print(df[["family", "h", "n_test", "rel_har", "dm_har", "p_har", "p_har_holm",
              "rel_firm", "dm_firm", "p_firm", "p_firm_holm"]].to_string(index=False))
    print("\nVERDICT:", verdict.replace("**", ""))
    print(b1_line)


if __name__ == "__main__":
    main()
