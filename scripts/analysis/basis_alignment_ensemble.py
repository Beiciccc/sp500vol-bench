"""P0-2 — BASIS ALIGNMENT: the killer controls rerun on the SEED-ENSEMBLE primary basis.

Round-2 gap (results/REVIEW_ROUND2_GAPS.md, P0-2): the committed control tables
(maximal_reference, firm_identity_control, control_intersection) evaluate TEXT =
seed2026 forecasts, while the DECLARED M1 primary (m1_ensemble_primary.md, 38/69)
is the per-observation 3-seed ensemble. The controls therefore kill a different
forecast object than the declared primary — "the intersection headline is void".

Fix implemented here: rerun BOTH reference-augmentation controls and the control
intersection with text = the SAME seed-ensemble forecast the primary uses
(m1_ensemble_primary.ensemble_text: per-observation mean of prediction_realised_vol
across seeds 2026/2027/2028, inner join on [ticker, accession, horizon_days];
A/B, C6_llmtext, D4_llmfused remain single-run). Everything else is byte-identical
to the committed seed2026 spec (maximal_reference_firm_control.py):
  * price panel   = inner join of [A2_har_rv, A6_shar, A3_garch, A4_egarch, A5_arima];
  * f_R*   = exp OLS[1, log f_A2..log f_ARIMA]        (val-fit, frozen to test);
  * f_U*   = f_R* design + g*log f_text;
  * f_Rfirm= exp OLS[1, log f_A2, log firm_mean_val_RV] (firm mean over the firm's
             own val rows, global val mean fallback; coverage reported);
  * f_Ufirm= f_Rfirm design + g*log f_text;
  * zero-text check: f_Rfirm (no text) vs plain f_R (A2-only recal) — quantifies how
    much of the increment firm identity ALONE reproduces (text-free, so it can only
    move through the join sample);
  * inference: day-clustered DM (clustered_dm.py) on effective_trading_day
    (filing_time_utc fallback), HAC lag = h-1 in DAYS, day-block bootstrap CI;
  * multiplicity: BOTH raw p<.05 and Holm<.05 within each 69-cell family.

SANITY GATE (hard): single-seed rows (A/B-anchored text models, C6_llmtext,
D4_llmfused — 33/69 cells) run the identical code path on identical inputs and must
reproduce the committed seed2026 tables to <1e-9. The script FAILS loudly otherwise.

Outputs (NEW files only; the committed seed2026 tables are untouched and become the
appendix versions):
  results/tables/maximal_reference_ensemble.{csv,md}
  results/tables/firm_identity_ensemble.{csv,md}
  results/tables/control_intersection_ensemble.{csv,md}

Run from repo root:  .venv/bin/python scripts/analysis/basis_alignment_ensemble.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")
import forecast_combination as fc
import m1_ensemble_primary as mep
from clustered_dm import dm_test_clustered, mbb_ci_daily
from maximal_reference_firm_control import (
    PRICE,
    build_price_panel,
    firm_mean_val,
    log_ols_frozen,
)

T = Path("results/tables")
KEY = fc.KEY
SORT = fc.SORT
HORIZONS = fc.HORIZONS
GRIDKEY = ["disc", "model", "h"]
TOL = 1e-9
HEAD = ("long_form", "C2_finbert_s1", 10)  # the paper's original headline cell


# --------------------------------------------------------------------------- grids
def run_grids():
    """69-cell maximal-reference and firm-identity grids with ENSEMBLE text."""
    max_rows, firm_rows = [], []
    for disc, models in fc.SETS.items():
        panel = build_price_panel(disc)
        fmap, gmean, fcov, ocov = firm_mean_val(panel)
        panel["firm_mean_val"] = panel.ticker.map(fmap).fillna(gmean).astype(float)
        for m in models:
            ens, used = mep.ensemble_text(m, disc)  # KEY + ftext (seed-mean)
            d = panel.merge(ens, on=KEY, how="inner")
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv = dv.label_realised_vol.to_numpy()
                yt = dt.label_realised_vol.to_numpy()
                ftv, ftt = dv.ftext.to_numpy(), dt.ftext.to_numpy()
                fhv, fhr = dv.A2_har_rv.to_numpy(), dt.A2_har_rv.to_numpy()
                days_t = (dt.effective_trading_day.fillna(dt.filing_time_utc)).to_numpy()
                base = {"disc": disc, "model": m, "h": h, "n_test": len(dt),
                        "n_seeds": len(used), "seeds": "+".join(map(str, used))}

                # A2-only recalibrated reference on the SAME rows (info + zero-text anchor)
                fR_a2, fU_a2, _ = fc.log_combo(yv, fhv, ftv, fhr, ftt)
                lR_a2 = fc.qlike(yt, fR_a2)
                qR_a2 = float(lR_a2.mean())
                qU_a2 = float(fc.qlike(yt, fU_a2).mean())
                rel_a2 = 100.0 * (qR_a2 - qU_a2) / qR_a2 if qR_a2 > 0 else np.nan

                # ---------------- (a) MAXIMAL 5-price-pool reference ----------------
                pv = [dv[c].to_numpy() for c in PRICE]
                pt = [dt[c].to_numpy() for c in PRICE]
                fRs, _ = log_ols_frozen(yv, pv, pt)
                fUs, bU = log_ols_frozen(yv, pv + [ftv], pt + [ftt])
                lRs, lUs = fc.qlike(yt, fRs), fc.qlike(yt, fUs)
                qRs, qUs = float(lRs.mean()), float(lUs.mean())
                rel_s = 100.0 * (qRs - qUs) / qRs if qRs > 0 else np.nan
                dm_s, p_s, n_days = dm_test_clustered(lUs, lRs, days_t, h)
                _, lo_s, hi_s = mbb_ci_daily(lUs - lRs, days_t, h)
                max_rows.append({**base, "n_days": n_days,
                                 "qlike_Rstar": qRs, "qlike_Ustar": qUs,
                                 "rel_impr_pct_maximal": rel_s,
                                 "g_text_star": float(bU[-1]),
                                 "dm_q_clustered": dm_s, "p_q_clustered": p_s,
                                 "boot_lo": lo_s, "boot_hi": hi_s,
                                 "rel_impr_pct_A2only_sameRows": rel_a2})

                # ---------------- (b) FIRM-identity-augmented reference -------------
                fmv, fmt = dv.firm_mean_val.to_numpy(), dt.firm_mean_val.to_numpy()
                fRf, _ = log_ols_frozen(yv, [fhv, fmv], [fhr, fmt])
                fUf, bUf = log_ols_frozen(yv, [fhv, fmv, ftv], [fhr, fmt, ftt])
                lRf, lUf = fc.qlike(yt, fRf), fc.qlike(yt, fUf)
                qRf, qUf = float(lRf.mean()), float(lUf.mean())
                rel_f = 100.0 * (qRf - qUf) / qRf if qRf > 0 else np.nan
                dm_f, p_f, _ = dm_test_clustered(lUf, lRf, days_t, h)
                _, lo_f, hi_f = mbb_ci_daily(lUf - lRf, days_t, h)
                # zero-text: firm-mean-only reference vs plain f_R (NO text in either)
                dm_fo, p_fo, _ = dm_test_clustered(lRf, lR_a2, days_t, h)
                rel_fo = 100.0 * (qR_a2 - qRf) / qR_a2 if qR_a2 > 0 else np.nan
                firm_rows.append({**base, "n_days": n_days,
                                  "firm_val_coverage": fcov,
                                  "firm_val_coverage_test_obs": ocov,
                                  "qlike_Rfirm": qRf, "qlike_Ufirm": qUf,
                                  "rel_impr_pct_firm": rel_f,
                                  "g_text_firm": float(bUf[-1]),
                                  "dm_q_clustered": dm_f, "p_q_clustered": p_f,
                                  "boot_lo": lo_f, "boot_hi": hi_f,
                                  "rel_impr_pct_A2only_sameRows": rel_a2,
                                  "rel_impr_firmMeanOnly_vs_fR": rel_fo,
                                  "dm_firmMeanOnly_vs_fR": dm_fo,
                                  "p_firmMeanOnly_vs_fR": p_fo,
                                  "firm_beats_fR": bool(dm_fo < 0 and p_fo < 0.05)})
    return pd.DataFrame(max_rows), pd.DataFrame(firm_rows)


def add_flags(df, dm_col="dm_q_clustered", p_col="p_q_clustered"):
    """Holm within the 69-cell family + raw/Holm add/hurt flags + verdicts."""
    df = df.copy()
    df["holm_p"] = fc.holm(df[p_col].fillna(1.0).values)
    neg = df[dm_col] < 0
    df["adds_raw"] = neg & (df[p_col] < 0.05)
    df["adds_holm"] = neg & (df["holm_p"] < 0.05)
    df["hurts_raw"] = (~neg) & (df[p_col] < 0.05)
    df["hurts_holm"] = (~neg) & (df["holm_p"] < 0.05)
    df["verdict"] = np.select([df.adds_holm, df.hurts_holm],
                              ["text adds", "text HURTS"], default="null")
    return df


def merge_before(df, before, cols, suf="_s26"):
    """Attach the committed seed2026 columns for the same cells."""
    b = before[GRIDKEY + cols].rename(columns={c: c + suf for c in cols})
    out = df.merge(b, on=GRIDKEY, how="left", validate="1:1")
    assert len(out) == len(df)
    return out


def cell_str(r):
    return f"{r['disc']}/{r['model']}/h{r['h']}"


def fmt(x, p="+.2f"):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, p)


# --------------------------------------------------------------------------- main
def main():
    # BEFORE = committed seed2026 control tables (the reconciliation targets)
    mx0 = pd.read_csv(T / "maximal_reference.csv")
    fi0 = pd.read_csv(T / "firm_identity_control.csv")
    ci0 = pd.read_csv(T / "control_intersection.csv")
    en = pd.read_csv(T / "m1_ensemble_primary.csv")

    maxdf, firmdf = run_grids()
    maxdf, firmdf = add_flags(maxdf), add_flags(firmdf)

    # ---- attach committed seed2026 columns side by side
    maxdf = merge_before(maxdf, mx0, ["n_test", "rel_impr_pct_maximal", "dm_q_clustered",
                                      "p_q_clustered", "holm_p", "text_adds_maximal",
                                      "text_adds_maximal_holm"])
    firmdf = merge_before(firmdf, fi0, ["n_test", "rel_impr_pct_firm", "dm_q_clustered",
                                        "p_q_clustered", "holm_p", "text_survives_firm",
                                        "text_survives_firm_holm",
                                        "rel_impr_firmMeanOnly_vs_fR", "firm_beats_fR"])

    # ---- SANITY GATE: single-seed cells must reproduce the committed tables exactly
    ms, fs = maxdf[maxdf.n_seeds == 1], firmdf[firmdf.n_seeds == 1]
    sanity = {
        "n_cells": len(maxdf),
        "n_single_seed_cells": len(ms),
        "max_dm_diff_maximal_single": float((ms.dm_q_clustered - ms.dm_q_clustered_s26).abs().max()),
        "max_rel_diff_maximal_single": float((ms.rel_impr_pct_maximal - ms.rel_impr_pct_maximal_s26).abs().max()),
        "max_dm_diff_firm_single": float((fs.dm_q_clustered - fs.dm_q_clustered_s26).abs().max()),
        "max_rel_diff_firm_single": float((fs.rel_impr_pct_firm - fs.rel_impr_pct_firm_s26).abs().max()),
        "n_test_mismatch_single": int((ms.n_test != ms.n_test_s26).sum()),
        "n_test_mismatch_ensemble": int((maxdf.n_test != maxdf.n_test_s26).sum()),
        "max_n_test_shrink_ensemble": int((maxdf.n_test_s26 - maxdf.n_test).max()),
    }
    sanity["pass"] = bool(sanity["max_dm_diff_maximal_single"] < TOL
                          and sanity["max_rel_diff_maximal_single"] < TOL
                          and sanity["max_dm_diff_firm_single"] < TOL
                          and sanity["max_rel_diff_firm_single"] < TOL
                          and sanity["n_test_mismatch_single"] == 0)
    if not sanity["pass"]:
        print(json.dumps(sanity, indent=2))
        raise SystemExit("SANITY FAIL: single-seed rows do not reproduce the committed "
                         "seed2026 control tables — implementation drift, aborting.")

    maxdf.to_csv(T / "maximal_reference_ensemble.csv", index=False)
    firmdf.to_csv(T / "firm_identity_ensemble.csv", index=False)

    # ================= INTERSECTION on the ensemble basis =================
    # primary flags = the DECLARED primary (m1_ensemble_primary, same text object)
    prim = en[GRIDKEY + ["n_seeds", "vol_rel_impr_pct", "vol_dm_q_clu", "vol_p_q_clu",
                         "vol_dmq_holm_clu", "vol_placebo_dm_clu", "genuine_ens_vol"]].copy()
    prim["primary_raw"] = (prim.vol_dm_q_clu < 0) & (prim.vol_p_q_clu < 0.05)
    prim["primary_holm"] = (prim.vol_dm_q_clu < 0) & (prim.vol_dmq_holm_clu < 0.05)
    prim["primary_genuine"] = prim.genuine_ens_vol.astype(bool)  # + placebo gate

    ix = prim.merge(
        maxdf[GRIDKEY + ["adds_raw", "adds_holm", "rel_impr_pct_maximal"]].rename(
            columns={"adds_raw": "maximal_raw", "adds_holm": "maximal_holm"}),
        on=GRIDKEY, validate="1:1").merge(
        firmdf[GRIDKEY + ["adds_raw", "adds_holm", "rel_impr_pct_firm"]].rename(
            columns={"adds_raw": "firm_raw", "adds_holm": "firm_holm"}),
        on=GRIDKEY, validate="1:1")
    assert len(ix) == len(maxdf) == 69, "intersection grid must be the full 69 cells"

    # extra info column, EXPLICITLY seed2026-basis (kept OUT of every AND):
    wd = pd.read_csv(T / "withindate_placebo.csv")[GRIDKEY + ["verdict"]].rename(
        columns={"verdict": "withindate_verdict_s26basis"})
    ix = ix.merge(wd, on=GRIDKEY, how="left", validate="1:1")

    for b in ("raw", "holm"):
        ix[f"AND_maximal_firm_{b}"] = ix[f"maximal_{b}"] & ix[f"firm_{b}"]
        ix[f"AND_full_{b}"] = ix[f"primary_{b}"] & ix[f"maximal_{b}"] & ix[f"firm_{b}"]
    # strictest: placebo-gated declared-primary genuine AND both Holm controls
    ix["AND_genuine_holm"] = ix.primary_genuine & ix.maximal_holm & ix.firm_holm
    ix.to_csv(T / "control_intersection_ensemble.csv", index=False)

    # ---- survivor sets / disjointness (Holm basis)
    mx_set = {cell_str(r) for _, r in ix[ix.maximal_holm].iterrows()}
    fi_set = {cell_str(r) for _, r in ix[ix.firm_holm].iterrows()}
    overlap = sorted(mx_set & fi_set)

    # BEFORE counts from the committed intersection (seed2026 basis)
    b4 = {
        "primary_raw": int(ci0.clu_genuine_raw.sum()), "primary_holm": int(ci0.clu_genuine_holm.sum()),
        "maximal_raw": int(ci0.maximal_raw.sum()), "maximal_holm": int(ci0.maximal_holm.sum()),
        "firm_raw": int(ci0.firm_raw.sum()), "firm_holm": int(ci0.firm_holm.sum()),
        "AND_mf_raw": int(ci0.AND_maximal_firm_raw.sum()), "AND_mf_holm": int(ci0.AND_maximal_firm_holm.sum()),
        "AND_full_raw": int(ci0.AND_full_raw.sum()), "AND_full_holm": int(ci0.AND_full_holm.sum()),
    }
    mx_set0 = {f"{r.disc}/{r.model}/h{r.h}" for _, r in ci0[ci0.maximal_holm].iterrows()}
    fi_set0 = {f"{r.disc}/{r.model}/h{r.h}" for _, r in ci0[ci0.firm_holm].iterrows()}
    overlap0 = sorted(mx_set0 & fi_set0)

    now = {
        "primary_raw": int(ix.primary_raw.sum()), "primary_holm": int(ix.primary_holm.sum()),
        "primary_genuine": int(ix.primary_genuine.sum()),
        "maximal_raw": int(ix.maximal_raw.sum()), "maximal_holm": int(ix.maximal_holm.sum()),
        "firm_raw": int(ix.firm_raw.sum()), "firm_holm": int(ix.firm_holm.sum()),
        "AND_mf_raw": int(ix.AND_maximal_firm_raw.sum()), "AND_mf_holm": int(ix.AND_maximal_firm_holm.sum()),
        "AND_full_raw": int(ix.AND_full_raw.sum()), "AND_full_holm": int(ix.AND_full_holm.sum()),
        "AND_genuine_holm": int(ix.AND_genuine_holm.sum()),
    }

    write_maximal_md(maxdf, sanity)
    write_firm_md(firmdf, fi0, sanity)
    write_intersection_md(ix, b4, now, mx_set, fi_set, overlap, overlap0, sanity)

    summary = {"sanity": sanity, "before_seed2026": b4, "restated_ensemble": now,
               "maximal_surv_ens_holm": sorted(mx_set), "firm_surv_ens_holm": sorted(fi_set),
               "overlap_ens_holm": overlap, "overlap_s26_holm": overlap0,
               "lf_neg_firm_ens": int((firmdf[firmdf.disc == "long_form"].rel_impr_pct_firm < 0).sum()),
               "lf_neg_firm_s26": int((firmdf[firmdf.disc == "long_form"].rel_impr_pct_firm_s26 < 0).sum()),
               "firm_beats_fR_ens": int(firmdf.firm_beats_fR.sum()),
               "firm_beats_fR_s26": int(firmdf.firm_beats_fR_s26.sum()),
               "AND_full_raw_cells": [cell_str(r) for _, r in ix[ix.AND_full_raw].iterrows()],
               "AND_mf_raw_cells": [cell_str(r) for _, r in ix[ix.AND_maximal_firm_raw].iterrows()]}
    print(json.dumps(summary, indent=2, default=str))


# --------------------------------------------------------------------------- md
def _grid_md(md, df, disc, relcol, s26relcol):
    md.append(f"\n## {disc} — 69-cell grid slice (ensemble vs seed2026 side by side)\n"
              "| model | h | seeds | n_test | n_days | rel% s26 | cluDM s26 | Holm s26 | "
              "rel% ENS | cluDM ENS | p ENS | Holm ENS | daily-dQ 95% CI | verdict ENS |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df[df.disc == disc].sort_values(["model", "h"]).iterrows():
        md.append(
            f"| {r.model} | {r.h} | {r.seeds} | {int(r.n_test)} | {int(r.n_days)} | "
            f"{fmt(r[s26relcol])} | {fmt(r.dm_q_clustered_s26)} | {fmt(r.holm_p_s26, '.3f')} | "
            f"**{fmt(r[relcol])}** | {fmt(r.dm_q_clustered)} | {fmt(r.p_q_clustered, '.4f')} | "
            f"{fmt(r.holm_p, '.3f')} | [{fmt(r.boot_lo, '+.5f')},{fmt(r.boot_hi, '+.5f')}] | "
            f"**{r.verdict}** |")


def write_maximal_md(maxdf, sanity):
    n = len(maxdf)
    head = maxdf[(maxdf.disc == HEAD[0]) & (maxdf.model == HEAD[1]) & (maxdf.h == HEAD[2])].iloc[0]
    surv = maxdf[maxdf.adds_holm]
    md = ["# P0-2 — Maximal 5-price-pool reference on the SEED-ENSEMBLE basis "
          "(basis-aligned with the declared primary)\n",
          "## RESTATED vs BEFORE\n",
          "BEFORE = committed maximal_reference.{csv,md} (text = seed2026 forecasts). "
          "RESTATED = identical spec (5-model log pool, val-fit frozen, day-clustered DM, "
          "Holm within 69 cells), text = per-observation 3-seed ensemble — the SAME object "
          "the declared primary (m1_ensemble_primary, 38/69) tests. A/B, C6_llmtext, "
          "D4_llmfused are single-run and identical in both.\n",
          "| quantity | BEFORE (seed2026 text) | RESTATED (seed-ensemble text) |",
          "|---|---|---|",
          f"| cells where text adds, raw p<.05 | {int(maxdf.text_adds_maximal_s26.sum())}/{n} | "
          f"**{int(maxdf.adds_raw.sum())}/{n}** |",
          f"| cells where text adds, Holm<.05 | {int(maxdf.text_adds_maximal_holm_s26.sum())}/{n} | "
          f"**{int(maxdf.adds_holm.sum())}/{n}** |",
          f"| cells where text HURTS (Holm) | "
          f"{int(((maxdf.dm_q_clustered_s26 > 0) & (maxdf.holm_p_s26 < 0.05)).sum())}/{n} | "
          f"{int(maxdf.hurts_holm.sum())}/{n} |",
          f"| mean rel% (69 cells) | {maxdf.rel_impr_pct_maximal_s26.mean():+.2f}% | "
          f"{maxdf.rel_impr_pct_maximal.mean():+.2f}% |",
          f"| headline cell C2_finbert_s1/long_form/h10 | "
          f"{head.rel_impr_pct_maximal_s26:+.2f}% (cluDM {head.dm_q_clustered_s26:+.2f}, "
          f"Holm {head.holm_p_s26:.3f}) | **{head.rel_impr_pct_maximal:+.2f}%** (cluDM "
          f"{head.dm_q_clustered:+.2f}, p {head.p_q_clustered:.4f}, Holm {head.holm_p:.3f}) |",
          f"\n**Sanity gate: PASS** — the {sanity['n_single_seed_cells']}/{n} single-seed cells "
          f"reproduce the committed seed2026 table exactly (max |dDM| = "
          f"{sanity['max_dm_diff_maximal_single']:.2e}, max |drel| = "
          f"{sanity['max_rel_diff_maximal_single']:.2e}); {sanity['n_test_mismatch_ensemble']} of "
          f"{n} cells lose rows to the 3-seed inner join (max shrink "
          f"{sanity['max_n_test_shrink_ensemble']} obs).\n"]
    for disc in fc.SETS:
        _grid_md(md, maxdf, disc, "rel_impr_pct_maximal", "rel_impr_pct_maximal_s26")
    flips_gain = maxdf[maxdf.adds_holm & ~maxdf.text_adds_maximal_holm_s26]
    flips_lose = maxdf[~maxdf.adds_holm & maxdf.text_adds_maximal_holm_s26.astype(bool)]
    md.append("\n## Holm-survivor flips (seed2026 -> ensemble)\n")
    md.append("GAINED: " + ("; ".join(cell_str(r) for _, r in flips_gain.iterrows()) or "none"))
    md.append("\nLOST: " + ("; ".join(cell_str(r) for _, r in flips_lose.iterrows()) or "none"))
    md.append("\n\n## HONEST bottom line\n"
              f"- On the basis of the DECLARED primary (seed-ensemble text), the maximal "
              f"5-price-pool reference leaves **{int(maxdf.adds_holm.sum())}/{n}** Holm survivors "
              f"(seed2026 basis: {int(maxdf.text_adds_maximal_holm_s26.sum())}/{n}).")
    if len(surv):
        md.append("- Ensemble-basis survivors: " + "; ".join(
            f"{cell_str(r)} {r.rel_impr_pct_maximal:+.2f}% (cluDM {r.dm_q_clustered:+.2f})"
            for _, r in surv.iterrows()) + ".")
    (T / "maximal_reference_ensemble.md").write_text("\n".join(md))


def write_firm_md(firmdf, fi0, sanity):
    n = len(firmdf)
    lf = firmdf[firmdf.disc == "long_form"]
    head = firmdf[(firmdf.disc == HEAD[0]) & (firmdf.model == HEAD[1]) & (firmdf.h == HEAD[2])].iloc[0]
    surv = firmdf[firmdf.adds_holm]
    md = ["# P0-2 — Firm-identity-augmented reference on the SEED-ENSEMBLE basis "
          "(basis-aligned with the declared primary)\n",
          "## RESTATED vs BEFORE\n",
          "BEFORE = committed firm_identity_control.{csv,md} (text = seed2026). RESTATED = "
          "identical spec (reference = [1, log HAR, log firm-mean-val-RV], val-fit frozen, "
          "day-clustered DM, Holm within 69 cells), text = 3-seed ensemble = the declared "
          "primary's forecast object.\n",
          "| quantity | BEFORE (seed2026 text) | RESTATED (seed-ensemble text) |",
          "|---|---|---|",
          f"| cells where text survives, raw p<.05 | {int(firmdf.text_survives_firm_s26.sum())}/{n} | "
          f"**{int(firmdf.adds_raw.sum())}/{n}** |",
          f"| cells where text survives, Holm<.05 | {int(firmdf.text_survives_firm_holm_s26.sum())}/{n} | "
          f"**{int(firmdf.adds_holm.sum())}/{n}** |",
          f"| cells where text HURTS (Holm) | "
          f"{int(((firmdf.dm_q_clustered_s26 > 0) & (firmdf.holm_p_s26 < 0.05)).sum())}/{n} | "
          f"{int(firmdf.hurts_holm.sum())}/{n} |",
          f"| long_form cells with NEGATIVE rel% | {int((lf.rel_impr_pct_firm_s26 < 0).sum())}/{len(lf)} | "
          f"**{int((lf.rel_impr_pct_firm < 0).sum())}/{len(lf)}** |",
          f"| zero-text firm-mean beats plain f_R | {int(firmdf.firm_beats_fR_s26.sum())}/{n} "
          f"(mean {fi0.rel_impr_firmMeanOnly_vs_fR.mean():+.2f}%) | "
          f"**{int(firmdf.firm_beats_fR.sum())}/{n}** "
          f"(mean {firmdf.rel_impr_firmMeanOnly_vs_fR.mean():+.2f}%) |",
          f"| headline cell C2_finbert_s1/long_form/h10 | {head.rel_impr_pct_firm_s26:+.2f}% "
          f"(cluDM {head.dm_q_clustered_s26:+.2f}, Holm {head.holm_p_s26:.3f}) | "
          f"**{head.rel_impr_pct_firm:+.2f}%** (cluDM {head.dm_q_clustered:+.2f}, "
          f"p {head.p_q_clustered:.4f}, Holm {head.holm_p:.3f}) |",
          f"\n**Sanity gate: PASS** — the {sanity['n_single_seed_cells']}/{n} single-seed cells "
          f"reproduce the committed seed2026 table exactly (max |dDM| = "
          f"{sanity['max_dm_diff_firm_single']:.2e}, max |drel| = "
          f"{sanity['max_rel_diff_firm_single']:.2e}). The zero-text check involves NO text "
          f"forecast; it moves only through the (slightly smaller) 3-seed join sample.\n"]
    for disc in fc.SETS:
        _grid_md(md, firmdf, disc, "rel_impr_pct_firm", "rel_impr_pct_firm_s26")
    flips_gain = firmdf[firmdf.adds_holm & ~firmdf.text_survives_firm_holm_s26.astype(bool)]
    flips_lose = firmdf[~firmdf.adds_holm & firmdf.text_survives_firm_holm_s26.astype(bool)]
    md.append("\n## Holm-survivor flips (seed2026 -> ensemble)\n")
    md.append("GAINED: " + ("; ".join(cell_str(r) for _, r in flips_gain.iterrows()) or "none"))
    md.append("\nLOST: " + ("; ".join(cell_str(r) for _, r in flips_lose.iterrows()) or "none"))
    md.append("\n\n## HONEST bottom line\n"
              f"- Under the DECLARED primary basis, the firm-identity control leaves "
              f"**{int(firmdf.adds_holm.sum())}/{n}** Holm survivors (seed2026: "
              f"{int(firmdf.text_survives_firm_holm_s26.sum())}/{n}); long_form flips negative in "
              f"**{int((lf.rel_impr_pct_firm < 0).sum())}/{len(lf)}** cells; the zero-text "
              f"firm-mean reference reproduces the gain in "
              f"**{int(firmdf.firm_beats_fR.sum())}/{n}** cells.")
    if len(surv):
        md.append("- Ensemble-basis survivors: " + "; ".join(
            f"{cell_str(r)} {r.rel_impr_pct_firm:+.2f}% (cluDM {r.dm_q_clustered:+.2f})"
            for _, r in surv.iterrows()) + ".")
    (T / "firm_identity_ensemble.md").write_text("\n".join(md))


def write_intersection_md(ix, b4, now, mx_set, fi_set, overlap, overlap0, sanity):
    n = len(ix)
    raw_full = ix[ix.AND_full_raw]
    md = ["# P0-2 — Control intersection on the SEED-ENSEMBLE basis "
          "(controls now kill the DECLARED primary)\n",
          "## RESTATED vs BEFORE\n",
          "BEFORE = committed control_intersection.{csv,md}: primary marginal from "
          "m1_clustered (seed2026, 29/69) and controls on seed2026 text — an internal basis "
          "mismatch with the declared primary (m1_ensemble_primary, 38/69). RESTATED: every "
          "ingredient of the AND is now the SAME seed-ensemble forecast object: primary "
          "marginal = m1_ensemble_primary flags; maximal + firm controls = "
          "{maximal_reference,firm_identity}_ensemble. Raw p<.05 and Holm<.05 both reported; "
          "the within-date column is retained for information ONLY (seed2026 basis, excluded "
          "from every AND).\n",
          "| quantity | BEFORE (seed2026 basis) | RESTATED (seed-ensemble basis) |",
          "|---|---|---|",
          f"| primary marginal, raw / Holm | {b4['primary_raw']} / {b4['primary_holm']} "
          f"(m1_clustered) | **{now['primary_raw']} / {now['primary_holm']}** "
          f"(declared primary; placebo-gated genuine = {now['primary_genuine']}) |",
          f"| maximal-pool survivors, raw / Holm | {b4['maximal_raw']} / {b4['maximal_holm']} | "
          f"**{now['maximal_raw']} / {now['maximal_holm']}** |",
          f"| firm-identity survivors, raw / Holm | {b4['firm_raw']} / {b4['firm_holm']} | "
          f"**{now['firm_raw']} / {now['firm_holm']}** |",
          f"| AND maximal & firm, raw / Holm | {b4['AND_mf_raw']} / {b4['AND_mf_holm']} | "
          f"**{now['AND_mf_raw']} / {now['AND_mf_holm']}** |",
          f"| FULL AND (primary & maximal & firm), raw / Holm | {b4['AND_full_raw']} / "
          f"{b4['AND_full_holm']} | **{now['AND_full_raw']} / {now['AND_full_holm']}** |",
          f"| strictest: placebo-gated genuine & both Holm controls | — | "
          f"**{now['AND_genuine_holm']}** |",
          f"| maximal vs firm Holm-survivor overlap | {len(overlap0)} "
          f"({'disjoint' if not overlap0 else '; '.join(overlap0)}) | "
          f"**{len(overlap)} ({'disjoint' if not overlap else '; '.join(overlap)})** |",
          "\n## Does the headline hold on the declared-primary basis?\n",
          f"1. **Holm AND = 0**: {'YES — holds' if now['AND_full_holm'] == 0 else 'NO — BROKEN'} "
          f"(full AND = {now['AND_full_holm']}/{n} under Holm; maximal&firm AND = "
          f"{now['AND_mf_holm']}/{n}; strictest placebo-gated version = "
          f"{now['AND_genuine_holm']}/{n}).",
          f"2. **Survivor sets disjoint**: {'YES — holds' if not overlap else 'NO — BROKEN'} "
          f"(overlap = {overlap if overlap else 'empty'}).",
          f"3. Raw-p full AND = {now['AND_full_raw']}/{n}"
          + (": " + "; ".join(cell_str(r) for _, r in raw_full.iterrows())
             if len(raw_full) else " (none)") + ".\n"]
    md.append("\n## Per-cell flags (ensemble basis)\n"
              "| disc | model | h | seeds | primary raw/Holm/genuine | maximal raw/Holm | "
              "firm raw/Holm | AND mf Holm | AND full Holm | within-date (s26, info) |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
    def yn(v):
        return "Y" if v else "."
    for _, r in ix.sort_values(GRIDKEY).iterrows():
        md.append(f"| {r.disc} | {r.model} | {r.h} | {int(r.n_seeds)} | "
                  f"{yn(r.primary_raw)}/{yn(r.primary_holm)}/{yn(r.primary_genuine)} | "
                  f"{yn(r.maximal_raw)}/{yn(r.maximal_holm)} | {yn(r.firm_raw)}/{yn(r.firm_holm)} | "
                  f"{yn(r.AND_maximal_firm_holm)} | {yn(r.AND_full_holm)} | "
                  f"{r.withindate_verdict_s26basis} |")
    md.append("\n## Survivor sets (Holm, ensemble basis)\n")
    md.append(f"**Maximal (n={len(mx_set)}):** " + (", ".join(sorted(mx_set)) or "none"))
    md.append(f"\n**Firm-identity (n={len(fi_set)}):** " + (", ".join(sorted(fi_set)) or "none"))
    md.append("\n**Overlap:** " + (", ".join(overlap) if overlap
                                    else "EMPTY — the two survivor sets are disjoint"))
    md.append("\n\n## Sanity\n"
              f"- Single-seed cells ({sanity['n_single_seed_cells']}/{n}) bit-reproduce the "
              "committed seed2026 control tables (gate enforced in "
              "scripts/analysis/basis_alignment_ensemble.py; run aborts on drift).\n"
              f"- {sanity['n_test_mismatch_ensemble']}/{n} cells lose observations to the "
              f"3-seed inner join (max {sanity['max_n_test_shrink_ensemble']} obs).\n"
              "- Primary marginals reconcile with m1_ensemble_primary.md by construction "
              "(same CSV, same flags).")
    (T / "control_intersection_ensemble.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
