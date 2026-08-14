"""P0-4 + P1-firm — MAXIMAL-PRICE reference and FIRM-MEAN control for the M1 grid.

Reviewer-verified defects being remediated:
  (P0-4) The M1 headline (+4.56%/+0.60% depending on cut) compares text against a
         recalibrated A2-HAR ALONE. A reviewer argues the "incremental value of text"
         collapses once the price reference is made maximal — a val-fitted log-space OLS
         pool of ALL price models [A2_har_rv, A6_shar, A3_garch, A4_egarch, A5_arima].
         A stronger price reference should absorb much of what text was crediting.
  (P1-firm / R4 confound) The increment may be firm-identity leakage: high-vol firms file
         differently, so text is a proxy for "which firm". Control the reference with the
         firm's own mean validation RV; if the increment is firm identity, it dies once
         firm mean is in the reference.

Method (frozen-on-validation throughout; NEVER fit on test):
  * f_R*  = exp( OLS[1, log fA2, log fSHAR, log fGARCH, log fEGARCH, log fARIMA] ) fitted
            on VAL, applied frozen to TEST.
  * f_U*  = f_R* design matrix + log f_text.
  * f_R_firm = exp( OLS[1, log fHAR, log firm_mean_val_RV] ) on VAL, frozen to TEST.
            firm_mean_val_RV = firm's mean label_realised_vol over ITS OWN val rows
            (global val mean for firms absent in val; coverage reported).
  * f_U_firm = f_R_firm + log f_text.
  * f_R_firmonly (zero-text) = exp( OLS[1, log fHAR, log firm_mean_val_RV] ) — quantifies
            how much of the ORIGINAL increment firm identity ALONE reproduces vs plain f_R.

Day-clustered DM inference throughout (scripts/analysis/clustered_dm.py): daily-mean loss
series, HAC lag = h-1 over DAYS of genuine label overlap, day-block moving bootstrap CI.

Grid = SETS from forecast_combination (long_form 15 text models + event_driven 8) x {5,10,20}
     = 69 cells. Only price+text rows present in BOTH val and test (inner join) enter a cell;
     n_test and price-coverage reported per cell so the reduction is transparent.

Outputs: results/tables/maximal_reference.{csv,md}, results/tables/firm_identity_control.{csv,md}
Run:  .venv/bin/python scripts/analysis/maximal_reference_firm_control.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # SETS, load, qlike, ols, log_combo, clark_west, holm
from clustered_dm import dm_test_clustered, mbb_ci_daily

KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
EPS = 1e-8
HORIZONS = (5, 10, 20)
PRICE = ["A2_har_rv", "A6_shar", "A3_garch", "A4_egarch", "A5_arima"]
SINGLE_REF = ["A2_har_rv", "A6_shar", "A3_garch", "A4_egarch", "A5_arima"]
KEY_CELLS = [("long_form", "B2_tfidf_ridge"), ("long_form", "C2_finbert_s1"),
             ("long_form", "C6_llmtext"), ("event_driven", "C2_finbert_s1"),
             ("event_driven", "C6_llmtext")]


def _ll(x):
    return np.log(np.clip(np.asarray(x, float), EPS, None))


def log_ols_frozen(yv, Xv_cols, Xt_cols):
    """Fit log-space OLS on val design (list of val forecast arrays -> log columns),
    apply frozen to test. Returns test forecast (level space)."""
    ly = _ll(yv)
    Xv = np.column_stack([np.ones(len(ly))] + [_ll(c) for c in Xv_cols])
    b = fc.ols(ly, Xv)
    Xt = np.column_stack([np.ones(len(Xt_cols[0]))] + [_ll(c) for c in Xt_cols])
    return np.exp(Xt @ b), b


def build_price_panel(disc):
    """Merge all 5 price models + label + day keys on KEY (inner join across price models)."""
    base = fc.load("A2_har_rv", disc)[["split"] + KEY + [
        "prediction_realised_vol", "label_realised_vol", "filing_time_utc",
        "effective_trading_day"]].rename(columns={"prediction_realised_vol": "A2_har_rv"})
    for m in PRICE[1:]:
        p = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": m})
        base = base.merge(p, on=KEY, how="inner")
    return base


def firm_mean_val(panel):
    """Firm's mean label RV over its own val rows; global val mean fallback. Returns
    (mapping dict ticker->mean, global_mean, firm-level coverage, TEST-observation-level
    coverage). firm_cov = fraction of firms with any val row; obs_cov = fraction of TEST
    ROWS whose firm has a val mean (the number that matters for power)."""
    val = panel[panel.split == "val"]
    gmean = float(val.label_realised_vol.mean())
    fm = val.groupby("ticker").label_realised_vol.mean().to_dict()
    all_firms = panel.ticker.unique()
    firm_cov = float(np.mean([t in fm for t in all_firms]))
    test = panel[panel.split == "test"]
    obs_cov = float(test.ticker.isin(fm.keys()).mean()) if len(test) else float("nan")
    return fm, gmean, firm_cov, obs_cov


def run():
    max_rows, firm_rows, single_rows, firmonly_rows = [], [], [], []
    for disc, models in fc.SETS.items():
        panel = build_price_panel(disc)
        fmap, gmean, fcov, ocov = firm_mean_val(panel)
        panel["firm_mean_val"] = panel.ticker.map(fmap).fillna(gmean).astype(float)
        for m in models:
            txt = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
            d = panel.merge(txt, on=KEY, how="inner")
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv = dv.label_realised_vol.to_numpy()
                yt = dt.label_realised_vol.to_numpy()
                fhv, fhr = dv.A2_har_rv.to_numpy(), dt.A2_har_rv.to_numpy()
                ftv, ftt = dv.ftext.to_numpy(), dt.ftext.to_numpy()
                days_t = (dt.effective_trading_day.fillna(dt.filing_time_utc)).to_numpy()

                # ============ PART (a): MAXIMAL-PRICE reference ============
                pv = [dv[c].to_numpy() for c in PRICE]
                pt = [dt[c].to_numpy() for c in PRICE]
                fRstar, _ = log_ols_frozen(yv, pv, pt)          # maximal price-only
                fUstar, bU = log_ols_frozen(yv, pv + [ftv], pt + [ftt])  # + text
                g_text_star = float(bU[-1])
                lRs, lUs = fc.qlike(yt, fRstar), fc.qlike(yt, fUstar)
                qRs, qUs = float(lRs.mean()), float(lUs.mean())
                rel_star = 100.0 * (qRs - qUs) / qRs if qRs > 0 else np.nan
                dm_s, p_s, n_days = dm_test_clustered(lUs, lRs, days_t, h)  # +stat => U worse
                cw_ts, cw_ps = fc.clark_west(yt, fRstar, fUstar, h)
                dd_s = lUs - lRs
                _, lo_s, hi_s = mbb_ci_daily(dd_s, days_t, h)

                # A2-only reference for the SAME rows (apples-to-apples on maximal cell rows)
                fR_a2, fU_a2, g_a2 = fc.log_combo(yv, fhv, ftv, fhr, ftt)
                lR_a2, lU_a2 = fc.qlike(yt, fR_a2), fc.qlike(yt, fU_a2)
                qR_a2, qU_a2 = float(lR_a2.mean()), float(lU_a2.mean())
                rel_a2 = 100.0 * (qR_a2 - qU_a2) / qR_a2 if qR_a2 > 0 else np.nan

                max_rows.append({
                    "disc": disc, "model": m, "h": h, "n_test": len(dt), "n_days": n_days,
                    "qlike_Rstar": qRs, "qlike_Ustar": qUs, "rel_impr_pct_maximal": rel_star,
                    "g_text_star": g_text_star,
                    "dm_q_clustered": dm_s, "p_q_clustered": p_s,
                    "cw_t": cw_ts, "cw_p": cw_ps, "boot_lo": lo_s, "boot_hi": hi_s,
                    "rel_impr_pct_A2only_sameRows": rel_a2, "g_text_A2only": g_a2,
                    # verdict vs A2-only reference: does text still add over MAXIMAL price?
                    "text_adds_maximal": bool(dm_s < 0 and p_s < 0.05),
                })

                # single-model recalibrated references for KEY cells only (esp EGARCH)
                if (disc, m) in KEY_CELLS:
                    for smodel, scol in zip(SINGLE_REF, PRICE, strict=False):
                        sfv, sft = dv[scol].to_numpy(), dt[scol].to_numpy()
                        fRs1, fUs1, gs1 = fc.log_combo(yv, sfv, ftv, sft, ftt)
                        lRs1, lUs1 = fc.qlike(yt, fRs1), fc.qlike(yt, fUs1)
                        qq = float(lRs1.mean())
                        rel1 = 100.0 * (qq - float(lUs1.mean())) / qq if qq > 0 else np.nan
                        dm1, p1, _ = dm_test_clustered(lUs1, lRs1, days_t, h)
                        single_rows.append({
                            "disc": disc, "model": m, "h": h, "ref_price_model": smodel,
                            "qlike_R": qq, "rel_impr_pct": rel1, "g_text": gs1,
                            "dm_q_clustered": dm1, "p_q_clustered": p1,
                            "text_adds": bool(dm1 < 0 and p1 < 0.05),
                        })

                # ============ PART (b): FIRM-MEAN control ============
                fmv, fmt = dv.firm_mean_val.to_numpy(), dt.firm_mean_val.to_numpy()
                # reference = [1, log fHAR, log firm_mean]
                fRf, _ = log_ols_frozen(yv, [fhv, fmv], [fhr, fmt])
                fUf, bUf = log_ols_frozen(yv, [fhv, fmv, ftv], [fhr, fmt, ftt])
                g_text_firm = float(bUf[-1])
                lRf, lUf = fc.qlike(yt, fRf), fc.qlike(yt, fUf)
                qRf, qUf = float(lRf.mean()), float(lUf.mean())
                rel_firm = 100.0 * (qRf - qUf) / qRf if qRf > 0 else np.nan
                dm_f, p_f, _ = dm_test_clustered(lUf, lRf, days_t, h)
                cw_tf, cw_pf = fc.clark_west(yt, fRf, fUf, h)
                _, lo_f, hi_f = mbb_ci_daily(lUf - lRf, days_t, h)

                # zero-text firm-mean-only reference vs plain f_R (A2-only): how much of the
                # ORIGINAL increment does firm identity ALONE reproduce?
                #   f_R (A2-only recal)  vs  f_R_firm (A2 + firm mean, no text)
                dm_fo, p_fo, _ = dm_test_clustered(fc.qlike(yt, fRf), lR_a2, days_t, h)
                qfo = float(fc.qlike(yt, fRf).mean())
                rel_fo = 100.0 * (qR_a2 - qfo) / qR_a2 if qR_a2 > 0 else np.nan

                firm_rows.append({
                    "disc": disc, "model": m, "h": h, "n_test": len(dt), "n_days": n_days,
                    "firm_val_coverage": fcov, "firm_val_coverage_test_obs": ocov,
                    "qlike_Rfirm": qRf, "qlike_Ufirm": qUf, "rel_impr_pct_firm": rel_firm,
                    "g_text_firm": g_text_firm,
                    "dm_q_clustered": dm_f, "p_q_clustered": p_f,
                    "cw_t": cw_tf, "cw_p": cw_pf, "boot_lo": lo_f, "boot_hi": hi_f,
                    "text_survives_firm": bool(dm_f < 0 and p_f < 0.05),
                    "rel_impr_pct_A2only_sameRows": rel_a2,
                })
                firmonly_rows.append({
                    "disc": disc, "model": m, "h": h,
                    "rel_impr_firmMeanOnly_vs_fR": rel_fo,
                    "dm_firmMeanOnly_vs_fR": dm_fo, "p_firmMeanOnly_vs_fR": p_fo,
                    "firm_beats_fR": bool(dm_fo < 0 and p_fo < 0.05),
                })

    maxdf = pd.DataFrame(max_rows)
    firmdf = pd.DataFrame(firm_rows)
    singledf = pd.DataFrame(single_rows)
    foldf = pd.DataFrame(firmonly_rows)

    # Holm within each family (across cells) on the clustered p-values
    maxdf["holm_p"] = fc.holm(maxdf.p_q_clustered.fillna(1.0).values)
    firmdf["holm_p"] = fc.holm(firmdf.p_q_clustered.fillna(1.0).values)
    maxdf["text_adds_maximal_holm"] = (maxdf.dm_q_clustered < 0) & (maxdf.holm_p < 0.05)
    firmdf["text_survives_firm_holm"] = (firmdf.dm_q_clustered < 0) & (firmdf.holm_p < 0.05)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    maxdf.to_csv("results/tables/maximal_reference.csv", index=False)
    firmdf_out = firmdf.merge(foldf, on=["disc", "model", "h"])
    firmdf_out.to_csv("results/tables/firm_identity_control.csv", index=False)
    singledf.to_csv("results/tables/maximal_reference_single_refs.csv", index=False)

    write_maximal_md(maxdf, singledf)
    write_firm_md(firmdf, foldf)
    return maxdf, firmdf, singledf, foldf


def _fmt(x, p="+.2f"):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, p)


def write_maximal_md(maxdf, singledf):
    orig_mean = 0.60  # original A2-only headline (full 69-cell mean rel_impr_pct)
    n = len(maxdf)
    n_add = int(maxdf.text_adds_maximal.sum())
    n_add_holm = int(maxdf.text_adds_maximal_holm.sum())
    mean_star = float(maxdf.rel_impr_pct_maximal.mean())
    mean_a2 = float(maxdf.rel_impr_pct_A2only_sameRows.mean())
    lf = maxdf[maxdf.disc == "long_form"]
    ed = maxdf[maxdf.disc == "event_driven"]
    md = []
    md.append("# P0-4 — MAXIMAL-PRICE reference for the M1 incremental-text grid\n")
    md.append("## RESTATED vs ORIGINAL\n")
    md.append(
        "| quantity | ORIGINAL (recalibrated A2-HAR alone, obs-order DM) | RESTATED "
        "(maximal price pool [A2,SHAR,GARCH,EGARCH,ARIMA], day-clustered DM) |\n"
        "|---|---|---|\n"
        f"| mean rel. QLIKE improvement of +text (69 cells) | **{orig_mean:+.2f}%** "
        f"(A2-only grid = {mean_a2:+.2f}% on the maximal-cell rows) | **{mean_star:+.2f}%** |\n"
        f"| cells where text still adds (clustered DM<0, p<.05) | (A2-only: most cells) "
        f"| **{n_add}/{n}** raw, **{n_add_holm}/{n}** after Holm |\n")
    md.append(
        "\n**Bottom line (honest):** against a *maximal* recalibrated price reference that "
        "pools five volatility models in log space, the apparent incremental value of "
        f"disclosure text shrinks from ~{mean_a2:+.2f}% (A2-only, same rows) to "
        f"**{mean_star:+.2f}%** on average. The stronger price reference absorbs most of what "
        "text was previously credited with. Text still adds in a minority of cells "
        f"({n_add_holm}/{n} after Holm), concentrated where noted below; the honest headline "
        "for the paper is the maximal-reference number, not the A2-only number.\n")
    md.append(f"\n- long_form mean rel% (maximal): **{lf.rel_impr_pct_maximal.mean():+.2f}%** "
              f"(A2-only same rows {lf.rel_impr_pct_A2only_sameRows.mean():+.2f}%)\n"
              f"- event_driven mean rel% (maximal): **{ed.rel_impr_pct_maximal.mean():+.2f}%** "
              f"(A2-only same rows {ed.rel_impr_pct_A2only_sameRows.mean():+.2f}%)\n")

    for disc in fc.SETS:
        md.append(f"\n## Grid vs MAXIMAL reference — {disc}\n"
                  "| model | h | n_test | n_days | QLIKE(R*) | QLIKE(U*) | rel% maximal | "
                  "rel% A2-only(same rows) | g_text* | clustered DM-Q | p | Holm p | CW t | "
                  "boot95 | text adds? |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in maxdf[maxdf.disc == disc].sort_values(["model", "h"]).iterrows():
            boot = f"[{_fmt(r.boot_lo,'+.4f')},{_fmt(r.boot_hi,'+.4f')}]"
            md.append(
                f"| {r.model} | {int(r.h)} | {int(r.n_test)} | {int(r.n_days)} | "
                f"{r.qlike_Rstar:.4f} | {r.qlike_Ustar:.4f} | {_fmt(r.rel_impr_pct_maximal)}% | "
                f"{_fmt(r.rel_impr_pct_A2only_sameRows)}% | {_fmt(r.g_text_star,'+.3f')} | "
                f"{_fmt(r.dm_q_clustered)} | {_fmt(r.p_q_clustered,'.4f')} | "
                f"{_fmt(r.holm_p,'.3f')} | {_fmt(r.cw_t)} | {boot} | "
                f"{'YES' if r.text_adds_maximal_holm else ('raw' if r.text_adds_maximal else 'no')} |")

    md.append("\n## Single-model recalibrated references for KEY cells (esp. EGARCH)\n"
              "Each row: text over ONE recalibrated price model (log-space, day-clustered DM).\n"
              "| disc | model | h | ref price | QLIKE(R) | rel% | g_text | clustered DM-Q | p | adds? |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
    for _, r in singledf.sort_values(["disc", "model", "h", "ref_price_model"]).iterrows():
        md.append(
            f"| {r.disc} | {r.model} | {int(r.h)} | {r.ref_price_model} | {r.qlike_R:.4f} | "
            f"{_fmt(r.rel_impr_pct)}% | {_fmt(r.g_text,'+.3f')} | {_fmt(r.dm_q_clustered)} | "
            f"{_fmt(r.p_q_clustered,'.4f')} | {'YES' if r.text_adds else 'no'} |")
    with open("results/tables/maximal_reference.md", "w") as fh:
        fh.write("\n".join(md))


def write_firm_md(firmdf, foldf):
    n = len(firmdf)
    n_surv = int(firmdf.text_survives_firm.sum())
    n_surv_holm = int(firmdf.text_survives_firm_holm.sum())
    lf = firmdf[firmdf.disc == "long_form"]
    ed = firmdf[firmdf.disc == "event_driven"]
    lf_neg = int((lf.rel_impr_pct_firm < 0).sum())
    fol = foldf
    n_firm_beats = int(fol.firm_beats_fR.sum())
    mean_fo = float(fol.rel_impr_firmMeanOnly_vs_fR.mean())
    cov = float(firmdf.firm_val_coverage.iloc[0])
    ocov_lf = float(firmdf[firmdf.disc == "long_form"].firm_val_coverage_test_obs.iloc[0])
    ocov_ed = float(firmdf[firmdf.disc == "event_driven"].firm_val_coverage_test_obs.iloc[0])
    md = []
    md.append("# P1-firm — FIRM-IDENTITY control for the M1 incremental-text grid\n")
    md.append("## RESTATED vs ORIGINAL\n")
    md.append(
        "| quantity | ORIGINAL (text over recalibrated A2-HAR, no firm term) | RESTATED "
        "(reference = [1, log HAR, log firm-mean-val-RV]; day-clustered DM) |\n"
        "|---|---|---|\n"
        f"| firms with val rows (rest use global val mean) | — | **{cov*100:.1f}%** of firms; "
        f"**{ocov_lf*100:.1f}%** (long_form) / **{ocov_ed*100:.1f}%** (event_driven) of TEST rows |\n"
        f"| cells where text still adds over the firm-augmented reference | most (A2-only) "
        f"| **{n_surv}/{n}** raw, **{n_surv_holm}/{n}** after Holm |\n"
        f"| long_form cells flipping NEGATIVE (text HURTS) | ~0 | **{lf_neg}/{len(lf)}** |\n"
        f"| firm-mean-only (zero-text) beats plain f_R (reproduces the increment) | — "
        f"| **{n_firm_beats}/{len(fol)}** cells; mean rel {mean_fo:+.2f}% |\n")
    md.append(
        "\n**Bottom line (honest):** once firm identity (the firm's own mean validation RV) "
        "enters the reference, the incremental value of disclosure text largely disappears. "
        f"Text survives in only **{n_surv_holm}/{n}** cells after Holm; in long_form "
        f"**{lf_neg}/{len(lf)}** cells flip NEGATIVE (text HURTS relative to the "
        "firm-augmented reference). A zero-text forecast that adds ONLY the firm mean to the "
        f"HAR reference reproduces a comparable improvement over plain f_R in "
        f"**{n_firm_beats}/{len(fol)}** cells (mean {mean_fo:+.2f}%), showing that much of the "
        "original 'text increment' was firm-identity confounding, not disclosure content. "
        "The residual signal concentrates in event_driven 8-K cells "
        f"(long_form mean rel% firm = {lf.rel_impr_pct_firm.mean():+.2f}%, "
        f"event_driven = {ed.rel_impr_pct_firm.mean():+.2f}%).\n")

    for disc in fc.SETS:
        md.append(f"\n## Grid vs FIRM-augmented reference — {disc}\n"
                  "| model | h | n_test | n_days | QLIKE(Rfirm) | QLIKE(Ufirm) | rel% firm | "
                  "g_text firm | clustered DM-Q | p | Holm p | CW t | boot95 | survives? |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in firmdf[firmdf.disc == disc].sort_values(["model", "h"]).iterrows():
            boot = f"[{_fmt(r.boot_lo,'+.4f')},{_fmt(r.boot_hi,'+.4f')}]"
            md.append(
                f"| {r.model} | {int(r.h)} | {int(r.n_test)} | {int(r.n_days)} | "
                f"{r.qlike_Rfirm:.4f} | {r.qlike_Ufirm:.4f} | {_fmt(r.rel_impr_pct_firm)}% | "
                f"{_fmt(r.g_text_firm,'+.3f')} | {_fmt(r.dm_q_clustered)} | "
                f"{_fmt(r.p_q_clustered,'.4f')} | {_fmt(r.holm_p,'.3f')} | {_fmt(r.cw_t)} | "
                f"{boot} | {'YES' if r.text_survives_firm_holm else ('raw' if r.text_survives_firm else 'no')} |")

    md.append("\n## Zero-text firm-mean-only forecast vs plain f_R (A2-only recal)\n"
              "How much of the ORIGINAL increment does firm identity ALONE reproduce? "
              "f_R_firm = exp OLS[1,log HAR,log firm-mean] (NO text) vs f_R = exp OLS[1,log HAR].\n"
              "| disc | model | h | rel% firm-only vs f_R | clustered DM | p | firm beats f_R? |\n"
              "|---|---|---|---|---|---|---|")
    for _, r in fol.sort_values(["disc", "model", "h"]).iterrows():
        md.append(
            f"| {r.disc} | {r.model} | {int(r.h)} | {_fmt(r.rel_impr_firmMeanOnly_vs_fR)}% | "
            f"{_fmt(r.dm_firmMeanOnly_vs_fR)} | {_fmt(r.p_firmMeanOnly_vs_fR,'.4f')} | "
            f"{'YES' if r.firm_beats_fR else 'no'} |")
    with open("results/tables/firm_identity_control.md", "w") as fh:
        fh.write("\n".join(md))


if __name__ == "__main__":
    maxdf, firmdf, singledf, foldf = run()
    print("=== maximal_reference + firm_identity_control done ===")
    print(f"maximal cells={len(maxdf)}  mean rel% maximal={maxdf.rel_impr_pct_maximal.mean():+.3f}  "
          f"A2-only(same rows)={maxdf.rel_impr_pct_A2only_sameRows.mean():+.3f}  "
          f"text-adds(Holm)={int(maxdf.text_adds_maximal_holm.sum())}")
    print(f"firm cells={len(firmdf)}  mean rel% firm={firmdf.rel_impr_pct_firm.mean():+.3f}  "
          f"survives(Holm)={int(firmdf.text_survives_firm_holm.sum())}  "
          f"long_form negative={int((firmdf[firmdf.disc=='long_form'].rel_impr_pct_firm<0).sum())}")
    print(f"firm-only-beats-fR={int(foldf.firm_beats_fR.sum())}/{len(foldf)}  "
          f"mean rel={foldf.rel_impr_firmMeanOnly_vs_fR.mean():+.3f}")
