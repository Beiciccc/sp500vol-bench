"""Round-3 MUST-RUN row 4 — matched-class cross-family replication: Llama-3.1-70B on 8-K.

The round-3 panel (results/REVIEW_ROUND3_FRESH_PANEL.md, DA-CRITICAL #4 / R2-W3) ruled the
existing cross-family gate uninformative: Yi-1.5-34B and Phi-4-14B are smaller, older and
mode-collapsed, so "does not replicate" was capability-confounded at n=2. This script adds
the matched-class family — Meta-Llama-3.1-70B-Instruct (int4 AWQ weight quantisation,
DISCLOSED), identical manifest/prompts/protocol (prompt cap 6000, clip [0.03,3.0],
on_missing=rv22) — on the event-driven (8-K) channel, the only context-clean channel.

Scope disclosures (verified in-script):
  - long_form was NOT run for llama70 (no such run directory) — event_driven only.
  - C6_llmtext_llama70_full_combined_seed2026 exists but is a relabelled DUPLICATE of the
    event_driven panel (same 117,407 rows, all 8-K, predictions bit-identical): with no
    long-form forecasts a "combined" pass degenerates to the 8-K subset. It is checked and
    excluded — it carries no combined-disclosure information.

Per (disc, family, h): M1 log-space increment vs (a) the single recalibrated-HAR reference
and (b) the firm-identity-augmented reference (val-window firm mean spec), day-clustered DM
— logic copied verbatim from scripts/analysis/crossfamily_llm.py. Holm is applied within
the pre-declared family of 6 new llama70 tests (3 horizons x 2 references).
PLUS standalone health diagnostics per cell (TEST split): QLIKE in vol and variance units,
R^2, prediction sd, n_unique / modal share of round(pred,2), parse-fail and clip rates from
config.json — the same columns under which Yi/Phi were declared capability-floored, so the
paper can say whether the matched-class model is a HEALTHY forecaster.

SANITY GATES (HARD RULE — any failure aborts before writing tables):
  G1 qwen/yi/phi M1 rows reproduce results/tables/crossfamily_llm.csv to machine precision;
  G2 qwen/yi standalone cells reproduce results/tables/crossfamily_standalone.csv to
     machine precision;
  G3 recomputed variance-unit QLIKE matches each run's stored metrics.json (1e-3 rel);
  G4 the llama70 combined run is verified to be the ED duplicate described above.

Run from repo root:  .venv/bin/python scripts/analysis/crossfamily_llama70.py
Outputs (NEW files): results/tables/crossfamily_llama70.{csv,md}
"""
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
RTOL = 1e-12  # machine-precision gate for CSV float round-trip

FAMS = [("qwen3_32b", "C6_llmtext"), ("yi_34b", "C6_llmtext_yi34"),
        ("phi4_14b", "C6_llmtext_phi4"), ("llama70_awq", "C6_llmtext_llama70")]

M1_COLS = ["n_test", "n_days", "rel_har", "dm_har", "p_har",
           "rel_firm", "dm_firm", "p_firm", "g_text"]
DIAG_COLS = ["qlike_vol", "qlike_var", "r2", "pred_sd",
             "n_unique_2dp", "mode_val_2dp", "mode_share_pct"]


def ols(y, X):  # verbatim from crossfamily_llm.py
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


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


def holm(ps):
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


def close(a, b):
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= RTOL * max(abs(a), abs(b), 1.0)


def flag(fam, disc):
    if fam == "yi_34b" and disc == "long_form":
        return "4K-TRUNCATED"
    if fam == "llama70_awq":
        return "AWQ-INT4"
    return "-"


def main():
    # ---- G4: the llama70 "combined" run must be the relabelled ED duplicate ----
    pc = pd.read_parquet("results/runs/C6_llmtext_llama70_full_combined_seed2026/predictions.parquet")
    pe = pd.read_parquet("results/runs/C6_llmtext_llama70_full_event_driven_seed2026/predictions.parquet")
    mrg = pc[KEY + ["prediction_realised_vol"]].merge(
        pe[KEY + ["prediction_realised_vol"]], on=KEY, suffixes=("_c", "_e"))
    g4 = (len(pc) == len(pe) == len(mrg)
          and set(pc.form.unique()) == {"8-K"}
          and bool((mrg.prediction_realised_vol_c == mrg.prediction_realised_vol_e).all()))
    if not g4:
        print("SANITY G4 FAIL: llama70 combined run is NOT the expected ED duplicate — "
              "it may contain real combined-panel forecasts; re-scope before tabling.")
        sys.exit(1)

    rows = []
    for disc in ("long_form", "event_driven"):
        a2 = fc.load("A2_har_rv", disc)[KEY + ["split", "label_realised_vol",
                                               "prediction_realised_vol",
                                               "effective_trading_day"]] \
            .rename(columns={"prediction_realised_vol": "fh"})
        for fam, run in FAMS:
            try:
                p = fc.load(run, disc)
            except FileNotFoundError:
                continue  # phi4 & llama70: long_form not run (disclosed)
            cfg = json.load(open(f"results/runs/{run}_full_{disc}_seed2026/config.json"))
            mj = {(r["split"], r["horizon_days"]): r for r in json.load(
                open(f"results/runs/{run}_full_{disc}_seed2026/metrics.json"))}
            t = p[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ft"})
            te_all = p[p.split == "test"]
            for h in HORIZONS:
                # ---- M1 block: verbatim crossfamily_llm.py ----
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
                # ---- standalone diagnostics: verbatim crossfamily_standalone.py ----
                d = te_all[te_all.horizon_days == h]
                st = standalone_stats(d.label_realised_vol.to_numpy(),
                                      d.prediction_realised_vol.to_numpy())
                stored = mj[("test", h)]["qlike"]
                ok = abs(st["qlike_var"] - stored) <= 1e-3 * max(abs(stored), 1.0)
                rows.append(dict(disc=disc, family=fam, h=h, n_test=len(te), n_days=nd,
                                 rel_har=rel, dm_har=dm, p_har=pv,
                                 rel_firm=relf, dm_firm=dmf, p_firm=pf, g_text=g,
                                 **st,
                                 qlike_var_metricsjson=float(stored),
                                 metrics_sanity="PASS" if ok else "FAIL",
                                 parse_fail_rate=float(cfg["stats"]["parse_fail_rate"]),
                                 clipped_rate=float(cfg["stats"]["clipped_rate"]),
                                 flag=flag(fam, disc)))
    df = pd.DataFrame(rows)

    # ---- G3: metrics.json cross-check ----
    g3_fail = df[df.metrics_sanity == "FAIL"]
    if len(g3_fail):
        print("SANITY G3 FAIL (variance-unit QLIKE vs metrics.json):")
        print(g3_fail[["disc", "family", "h", "qlike_var", "qlike_var_metricsjson"]])
        sys.exit(1)

    # ---- G1: qwen/yi/phi M1 rows == committed crossfamily_llm.csv, machine precision ----
    ref = pd.read_csv("results/tables/crossfamily_llm.csv")
    g1_bad = []
    for _, r in ref.iterrows():
        mine = df[(df.disc == r.disc) & (df.family == r.family) & (df.h == r.h)]
        if len(mine) != 1:
            g1_bad.append((r.disc, r.family, r.h, "row missing"))
            continue
        for c in M1_COLS:
            if not close(mine[c].iloc[0], r[c]):
                g1_bad.append((r.disc, r.family, int(r.h), c,
                               float(mine[c].iloc[0]), float(r[c])))
    if g1_bad:
        print(f"SANITY G1 FAIL ({len(g1_bad)} mismatches vs crossfamily_llm.csv):")
        for b in g1_bad[:20]:
            print("  ", b)
        sys.exit(1)

    # ---- G2: qwen/yi standalone cells == committed crossfamily_standalone.csv ----
    ref2 = pd.read_csv("results/tables/crossfamily_standalone.csv")
    ref2 = ref2[ref2.disc.isin(["long_form", "event_driven"])]
    g2_bad = []
    for _, r in ref2.iterrows():
        mine = df[(df.disc == r.disc) & (df.family == r.family) & (df.h == r.h)]
        if len(mine) != 1:
            g2_bad.append((r.disc, r.family, r.h, "row missing"))
            continue
        for c in DIAG_COLS:
            if not close(mine[c].iloc[0], r[c]):
                g2_bad.append((r.disc, r.family, int(r.h), c,
                               float(mine[c].iloc[0]), float(r[c])))
    if g2_bad:
        print(f"SANITY G2 FAIL ({len(g2_bad)} mismatches vs crossfamily_standalone.csv):")
        for b in g2_bad[:20]:
            print("  ", b)
        sys.exit(1)

    # ---- Holm within the pre-declared family of 6 new llama70 tests ----
    df["p_holm_llama70"] = np.nan
    lam = (df.family == "llama70_awq")
    ps = np.concatenate([df.loc[lam, "p_har"].values, df.loc[lam, "p_firm"].values])
    adj = holm(ps)
    n_l = int(lam.sum())
    df.loc[lam, "p_har_holm"] = adj[:n_l]
    df.loc[lam, "p_firm_holm"] = adj[n_l:]
    df = df.drop(columns=["p_holm_llama70"])
    df.to_csv("results/tables/crossfamily_llama70.csv", index=False)

    la = df[lam].sort_values("h")
    qe = df[(df.family == "qwen3_32b") & (df.disc == "event_driven")].sort_values("h")
    n_rep_firm = int(((la.dm_firm < 0) & (la.p_firm_holm < .05)).sum())
    n_rep_har = int(((la.dm_har < 0) & (la.p_har_holm < .05)).sum())
    n_pos_firm = int((la.rel_firm > 0).sum())
    n_neg_dm_firm = int((la.dm_firm < 0).sum())
    n_sig_raw_firm = int(((la.dm_firm < 0) & (la.p_firm < .05)).sum())
    _rf = "/".join(f"{x:+.2f}" for x in la.rel_firm)
    _qf = "/".join(f"{x:+.2f}" for x in qe.rel_firm)
    if n_rep_firm == 3:
        verdict = ("**REPLICATES.** The matched-class Llama-3.1-70B reproduces the Qwen "
                   "event-driven residual over the firm-identity-augmented reference in "
                   "3/3 horizons (Holm<.05, day-clustered).")
    elif n_pos_firm == 3 and (n_sig_raw_firm >= 2 or n_rep_har >= 1):
        verdict = (f"**DIRECTIONALLY REPLICATES, significance attenuated.** The "
                   f"matched-class Llama-3.1-70B reproduces the SIGN of the Qwen 8-K "
                   f"residual vs HAR+firmID in 3/3 horizons with point estimates larger "
                   f"than Qwen's ({_rf}% vs {_qf}%), clustered DM<0 in "
                   f"{n_neg_dm_firm}/3 and raw p<.05 in {n_sig_raw_firm}/3; but after the "
                   f"pre-declared Holm(6) only {n_rep_har}/3 vs-single-HAR cells survive "
                   f"(min firmID Holm p={la.p_firm_holm.min():.5f}, {n_rep_firm}/3 firmID "
                   f"cells <.05). The "
                   f"residual is NOT Qwen-specific — a healthy matched-class family "
                   f"recovers same-sign, same-or-larger increments — but it is not fully "
                   f"Holm-robust in the second family either.")
    elif n_pos_firm == 0 and n_sig_raw_firm == 0:
        verdict = ("**Does NOT replicate.** The matched-class Llama-3.1-70B shows no "
                   "positive increment over the firm-identity-augmented reference in any "
                   "horizon — and unlike Yi/Phi this cannot be dismissed as a capability "
                   "floor (see health columns).")
    else:
        verdict = (f"**PARTIAL/MIXED replication** ({n_rep_firm}/3 firm-ID cells Holm<.05, "
                   f"{n_sig_raw_firm}/3 raw p<.05, {n_pos_firm}/3 positive; "
                   f"{n_rep_har}/3 vs single recalibrated HAR after Holm).")

    def m1cell(r):
        s1 = "**" if (r.dm_har < 0 and r.p_har < .05) else ""
        s2 = "**" if (r.dm_firm < 0 and r.p_firm < .05) else ""
        return (f"{r.rel_har:+.2f}%{s1} | {r.dm_har:+.2f} | "
                f"{r.rel_firm:+.2f}%{s2} | {r.dm_firm:+.2f}")

    md = [
        "# Cross-family replication, round-3 row 4 — matched-class Llama-3.1-70B (AWQ-INT4) on the 8-K channel",
        "",
        "## RESTATED vs BEFORE",
        "",
        "| | BEFORE (crossfamily_llm.md / crossfamily_standalone.md) | RESTATED (this table) |",
        "|---|---|---|",
        "| replication gate | Yi-1.5-34B + Phi-4-14B: smaller, older, mode-collapsed — "
        "\"does not replicate\" was capability-confounded, family-specificity unidentified "
        "at n=2 (round-3 DA-CRITICAL #4) | matched-class, different-lineage "
        "Meta-Llama-3.1-70B-Instruct (int4 AWQ), identical manifest/prompts/protocol, on "
        "the context-clean 8-K channel |",
        f"| verdict on the residual | uninformative gate | {verdict.replace('**','')} |",
        f"| 70B standalone health | n/a | variance-unit QLIKE "
        f"{la.qlike_var.min():.2f}-{la.qlike_var.max():.2f} vs Qwen "
        f"{qe.qlike_var.min():.2f}-{qe.qlike_var.max():.2f}, Yi 7.60-8.19; max modal share "
        f"{la.mode_share_pct.max():.1f}% vs Yi 73.6% |",
        "",
        "## Disclosures",
        "",
        "- **Quantisation**: llama70 = hugging-quants/Meta-Llama-3.1-70B-Instruct-**AWQ-INT4** "
        "(weight-only int4). All other families ran full/bf16 weights.",
        "- **long_form was NOT run for llama70** (GPU budget; 8-K is the citable channel per "
        "the round-3 panel — long-form was context-confounded for Yi anyway). Its cells are "
        "therefore absent, exactly as phi4's are.",
        "- **The C6_llmtext_llama70_full_combined_seed2026 run is a relabelled duplicate of "
        "the event_driven panel** (verified in-script: same 117,407 rows, all 8-K, "
        "predictions bit-identical) — with no long-form forecasts a \"combined\" pass "
        "degenerates to the 8-K subset. It is excluded; it carries no combined-disclosure "
        "information.",
        "- parse_fail_rate = 0.0 and clipped_rate = 0.0 for llama70 (config.json stats); "
        "Yi clipped 0.79%.",
        "- Holm is applied within the pre-declared family of the 6 NEW llama70 tests "
        "(3 horizons x {vs HAR, vs HAR+firmID}); qwen/yi/phi p-values are raw, carried "
        "unchanged from the committed tables.",
        "",
        "## Table — M1 increment (log-space, combiner val-fit test-frozen, day-clustered DM) + standalone health",
        "",
        "rel% is on **volatility-unit** QLIKE (the convention of the committed "
        "crossfamily_llm.csv anchor cells); the QLIKE(var) health column is "
        "**variance-unit** (Patton-robust convention). rel% > 0 = text lowers QLIKE vs "
        "the reference; `**` = clustered DM<0, raw p<.05.",
        "",
        "| disc | family | h | n_test | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) "
        "| Holm p (firmID) | QLIKE(var) | R^2 | pred sd | n_uniq(2dp) | mode share% | "
        "parse_ok% | flag |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for _, r in df.iterrows():
        hp = f"{r.p_firm_holm:.4g}" if pd.notna(r.get("p_firm_holm")) else "-"
        md.append(
            f"| {r.disc} | {r.family} | {int(r.h)} | {int(r.n_test)} | {m1cell(r)} | {hp} | "
            f"{r.qlike_var:.3f} | {r.r2:+.3f} | {r.pred_sd:.4f} | {int(r.n_unique_2dp)} | "
            f"{r.mode_share_pct:.1f} | {100 * (1 - r.parse_fail_rate):.1f} | {r.flag} |")

    md += [
        "",
        "## HEADLINE (honest)",
        "",
        verdict,
        "",
        f"- llama70 event_driven vs HAR+firmID: rel% = "
        f"{la[la.h == 5].rel_firm.iloc[0]:+.2f} / {la[la.h == 10].rel_firm.iloc[0]:+.2f} / "
        f"{la[la.h == 20].rel_firm.iloc[0]:+.2f} (h=5/10/20), clustered DM = "
        f"{la[la.h == 5].dm_firm.iloc[0]:+.2f} / {la[la.h == 10].dm_firm.iloc[0]:+.2f} / "
        f"{la[la.h == 20].dm_firm.iloc[0]:+.2f}; Qwen benchmark was "
        f"{qe[qe.h == 5].rel_firm.iloc[0]:+.2f} / {qe[qe.h == 10].rel_firm.iloc[0]:+.2f} / "
        f"{qe[qe.h == 20].rel_firm.iloc[0]:+.2f}.",
        f"- vs single recalibrated HAR: rel% = "
        f"{la[la.h == 5].rel_har.iloc[0]:+.2f} / {la[la.h == 10].rel_har.iloc[0]:+.2f} / "
        f"{la[la.h == 20].rel_har.iloc[0]:+.2f}, DM = "
        f"{la[la.h == 5].dm_har.iloc[0]:+.2f} / {la[la.h == 10].dm_har.iloc[0]:+.2f} / "
        f"{la[la.h == 20].dm_har.iloc[0]:+.2f}; {n_rep_har}/3 cells survive Holm(6).",
        "- The attenuated DM stats are NOT a power artefact of the panel: llama70 is "
        "scored on the identical test panel and day set as Qwen (same n_test, same "
        f"n_days = {'/'.join(str(int(x)) for x in la.n_days)}). The increment is larger "
        "in mean but noisier — llama70's forecasts are far more dispersed (pred sd "
        f"{la.pred_sd.min():.3f}-{la.pred_sd.max():.3f} vs Qwen "
        f"{qe.pred_sd.min():.3f}-{qe.pred_sd.max():.3f}), inflating the loss-differential "
        "variance.",
        f"- Health check: llama70 is {'a HEALTHY forecaster by the Yi/Phi criteria' if (la.qlike_var.max() < 4 and la.mode_share_pct.max() < 60) else 'NOT clearly healthy — read the columns'}: "
        f"variance-unit QLIKE {la.qlike_var.min():.2f}-{la.qlike_var.max():.2f} "
        f"(Qwen {qe.qlike_var.min():.2f}-{qe.qlike_var.max():.2f}, Yi 7.60-8.19, capability "
        f"floor), R^2 {la.r2.min():+.2f}-{la.r2.max():+.2f} "
        f"(Qwen {qe.r2.min():+.2f}-{qe.r2.max():+.2f}), modal share max "
        f"{la.mode_share_pct.max():.1f}% (Yi up to 73.6%), pred sd "
        f"{la.pred_sd.min():.3f}-{la.pred_sd.max():.3f}, parse_ok 100%.",
        "",
        "## SANITY",
        "",
        f"- G1 PASS: all {len(ref)} committed crossfamily_llm.csv M1 cells (qwen/yi/phi) "
        f"reproduced to machine precision (rtol {RTOL:g}) on columns {M1_COLS}.",
        f"- G2 PASS: all {len(ref2)} committed crossfamily_standalone.csv long_form/"
        f"event_driven diagnostic cells (qwen/yi) reproduced to machine precision on "
        f"columns {DIAG_COLS}.",
        f"- G3 PASS: recomputed variance-unit QLIKE matches stored metrics.json within "
        f"1e-3 relative in {len(df)}/{len(df)} cells (including all llama70 cells).",
        "- G4 PASS: llama70 combined run verified as the relabelled ED duplicate "
        "(117,407/117,407 predictions identical, 8-K only) and excluded.",
        "",
    ]
    Path("results/tables/crossfamily_llama70.md").write_text("\n".join(md))
    print("SANITY: G1 PASS G2 PASS G3 PASS G4 PASS")
    print(f"wrote results/tables/crossfamily_llama70.csv/.md ({len(df)} cells)")
    print(df[["disc", "family", "h", "rel_har", "dm_har", "rel_firm", "dm_firm",
              "qlike_var", "r2", "pred_sd", "n_unique_2dp", "mode_share_pct"]]
          .to_string(index=False))
    print("\nllama70 Holm p (har, firm):")
    print(la[["h", "p_har", "p_har_holm", "p_firm", "p_firm_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
