"""P0-4 (a) — MAXIMAL PRICE REFERENCE for the M1 incremental-text test.

Reviewer-verified defect: the M1 headline (+4.56% QLIKE for C2_finbert_s1 long_form
h=10) is measured against a recalibrated A2 HAR-RV ONLY. A text model can look
"incremental" simply by proxying price information that OTHER price models (SHAR,
GARCH, EGARCH, ARIMA) already carry. Fix: rebuild the reference as the val-fitted
log-space OLS combination of ALL price models:

    f_R* = exp( a + b2*log fA2 + b6*log fA6 + b3*log fA3 + b4*log fA4 + b5*log fA5 )
    f_U* = f_R* design + g*log f_text

(all coefficients fit on split=="val" ONLY, applied frozen to test), and rerun the
full 69-cell M1 grid. Inference uses the DAY-CLUSTERED DM test (clustered_dm.py):
daily-mean loss differentials on calendar days of effective_trading_day, HAC lag
= h-1 in DAYS, n = number of test days. Also reports, for the key cells
(B2_tfidf_ridge, C2_finbert_s1, C6_llmtext — long_form), every SINGLE-model
recalibrated reference (esp. EGARCH).

Run from repo root: .venv/bin/python scripts/analysis/maximal_reference.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc
from clustered_dm import dm_test_clustered, mbb_ci_daily

EPS = fc.EPS
KEY = fc.KEY
SORT = fc.SORT
HORIZONS = fc.HORIZONS
PRICE_MODELS = ["A2_har_rv", "A6_shar", "A3_garch", "A4_egarch", "A5_arima"]
KEY_CELLS = [("long_form", "B2_tfidf_ridge"), ("long_form", "C2_finbert_s1"),
             ("long_form", "C6_llmtext")]


def fit_apply_log(yv, Xv_list, Xt_list):
    """Val-fit log-space OLS of log y on [1, log f1, ..., log fk]; apply frozen to test."""
    ly = np.log(np.clip(np.asarray(yv, float), EPS, None))
    Lv = [np.log(np.clip(np.asarray(x, float), EPS, None)) for x in Xv_list]
    Lt = [np.log(np.clip(np.asarray(x, float), EPS, None)) for x in Xt_list]
    Xv = np.column_stack([np.ones(len(ly))] + Lv)
    beta = fc.ols(ly, Xv)
    Xt = np.column_stack([np.ones(len(Lt[0]))] + Lt)
    return np.exp(Xt @ beta), beta


def load_price_panel(disc):
    """A2 base (labels, days) merged with all other price-model forecasts on KEY."""
    base = fc.load("A2_har_rv", disc)[
        ["split"] + KEY + ["label_realised_vol", "filing_time_utc",
                           "effective_trading_day", "prediction_realised_vol"]
    ].rename(columns={"prediction_realised_vol": "f_A2_har_rv"})
    for pm in PRICE_MODELS[1:]:
        f = fc.load(pm, disc)[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": f"f_{pm}"})
        base = base.merge(f, on=KEY, how="inner")
    return base


def main():
    rows, key_rows = [], []
    for disc, models in fc.SETS.items():
        panel = load_price_panel(disc)
        for m in models:
            txt = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
            d = panel.merge(txt, on=KEY, how="inner")
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv = dv.label_realised_vol.to_numpy(); yt = dt.label_realised_vol.to_numpy()
                ftv = dv.ftext.to_numpy(); ftt = dt.ftext.to_numpy()
                days_t = dt.effective_trading_day.to_numpy()
                pv = [dv[f"f_{pm}"].to_numpy() for pm in PRICE_MODELS]
                pt = [dt[f"f_{pm}"].to_numpy() for pm in PRICE_MODELS]

                # ---- ORIGINAL reference: A2-only (recomputed, now with clustered DM)
                fR0, fU0, g0 = fc.log_combo(yv, pv[0], ftv, pt[0], ftt)
                lR0, lU0 = fc.qlike(yt, fR0), fc.qlike(yt, fU0)
                rel0 = 100.0 * (lR0.mean() - lU0.mean()) / lR0.mean()
                dm0, p0, nd0 = dm_test_clustered(lU0, lR0, days_t, h)

                # ---- MAXIMAL reference: all 5 price models
                fRs, bR = fit_apply_log(yv, pv, pt)
                fUs, bU = fit_apply_log(yv, pv + [ftv], pt + [ftt])
                lRs, lUs = fc.qlike(yt, fRs), fc.qlike(yt, fUs)
                rels = 100.0 * (lRs.mean() - lUs.mean()) / lRs.mean()
                dms, ps, nds = dm_test_clustered(lUs, lRs, days_t, h)
                mean_d, lo, hi = mbb_ci_daily(lUs - lRs, days_t, h)
                # does the maximal price set itself beat the A2-only reference?
                dmRR, pRR, _ = dm_test_clustered(lRs, lR0, days_t, h)

                rows.append({
                    "disc": disc, "model": m, "h": h,
                    "n_test": len(dt), "n_days": nds,
                    "qlike_R_a2": float(lR0.mean()), "qlike_U_a2": float(lU0.mean()),
                    "rel_a2_pct": float(rel0), "dmclu_a2": dm0, "pclu_a2": p0,
                    "qlike_Rstar": float(lRs.mean()), "qlike_Ustar": float(lUs.mean()),
                    "rel_star_pct": float(rels), "dmclu_star": dms, "pclu_star": ps,
                    "boot_lo": lo, "boot_hi": hi,
                    "g_text_star": float(bU[-1]),
                    "dm_Rstar_vs_Ra2": dmRR, "p_Rstar_vs_Ra2": pRR,
                })

                # ---- key cells: every single-price-model recalibrated reference
                if (disc, m) in KEY_CELLS:
                    for j, pm in enumerate(PRICE_MODELS):
                        fRj, fUj, gj = fc.log_combo(yv, pv[j], ftv, pt[j], ftt)
                        lRj, lUj = fc.qlike(yt, fRj), fc.qlike(yt, fUj)
                        relj = 100.0 * (lRj.mean() - lUj.mean()) / lRj.mean()
                        dmj, pj, ndj = dm_test_clustered(lUj, lRj, days_t, h)
                        key_rows.append({
                            "disc": disc, "model": m, "h": h, "reference": pm,
                            "qlike_R": float(lRj.mean()), "qlike_U": float(lUj.mean()),
                            "rel_pct": float(relj), "g_text": gj,
                            "dm_clustered": dmj, "p_clustered": pj, "n_days": ndj,
                        })

    df = pd.DataFrame(rows)
    df["pclu_a2_holm"] = fc.holm(df.pclu_a2.fillna(1.0).values)
    df["pclu_star_holm"] = fc.holm(df.pclu_star.fillna(1.0).values)

    def verdict(dm, holm_p):
        if holm_p < 0.05:
            return "text adds" if dm < 0 else "text HURTS"
        return "null"

    df["verdict_a2"] = [verdict(a, b) for a, b in zip(df.dmclu_a2, df.pclu_a2_holm)]
    df["verdict_star"] = [verdict(a, b) for a, b in zip(df.dmclu_star, df.pclu_star_holm)]
    kdf = pd.DataFrame(key_rows)
    kdf["p_holm"] = fc.holm(kdf.p_clustered.fillna(1.0).values)

    out = Path("results/tables")
    df.to_csv(out / "maximal_reference.csv", index=False)
    kdf.to_csv(out / "maximal_reference_key_cells.csv", index=False)

    # ---------- markdown ----------
    n = len(df)
    adds_a2 = int((df.verdict_a2 == "text adds").sum()); hurts_a2 = int((df.verdict_a2 == "text HURTS").sum())
    adds_s = int((df.verdict_star == "text adds").sum()); hurts_s = int((df.verdict_star == "text HURTS").sum())
    head = df[(df.model == "C2_finbert_s1") & (df.disc == "long_form") & (df.h == 10)].iloc[0]
    absorbed = int(((df.verdict_a2 == "text adds") & (df.verdict_star != "text adds")).sum())

    md = ["# P0-4a — Maximal price reference (all 5 price models) vs A2-only, day-clustered DM\n",
          "## RESTATED vs ORIGINAL\n",
          "| quantity | ORIGINAL (A2-only ref, obs-order HAC DM) | RESTATED (maximal 5-model ref, day-clustered DM) |",
          "|---|---|---|",
          f"| headline cell (C2_finbert_s1, long_form, h=10) | +4.56% QLIKE, DM=-12.77 | "
          f"{head.rel_star_pct:+.2f}% QLIKE, clustered DM={head.dmclu_star:+.2f} (p={head.pclu_star:.4f}, "
          f"Holm={head.pclu_star_holm:.4f}, n_days={int(head.n_days)}) |",
          f"| cells where text adds (Holm<.05) | 38/69 (placebo-confirmed, old inference) | "
          f"**{adds_s}/{n}** vs maximal ref ({adds_a2}/{n} vs A2-only ref under the SAME clustered DM) |",
          f"| cells where text HURTS (Holm<.05) | — | {hurts_s}/{n} vs maximal ref ({hurts_a2}/{n} vs A2-only) |",
          f"| A2-adds cells absorbed by the maximal price set | — | {absorbed} |",
          "\nReference: f_R* = exp(a + Σ b_j log f_j) over [A2_har_rv, A6_shar, A3_garch, A4_egarch, A5_arima], "
          "val-fit, frozen to test. f_U* adds g·log f_text. Inference: per-observation QLIKE differentials "
          "averaged within calendar day of effective_trading_day; DM with HAC lag=h-1 in DAYS; n = test days; "
          "95% CI = day-block moving bootstrap (blocks of h days). Holm within each 69-cell family.\n"]

    for disc in fc.SETS:
        md.append(f"\n## {disc} — 69-cell grid slice\n"
                  "| model | h | n_days | rel% (A2 ref) | cluDM (A2) | Holm | verdict(A2) | "
                  "rel% (max ref) | cluDM (max) | Holm | daily-dQ 95% CI | verdict(max) |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df[df.disc == disc].sort_values(["model", "h"]).iterrows():
            md.append(f"| {r.model} | {r.h} | {int(r.n_days)} | {r.rel_a2_pct:+.2f} | {r.dmclu_a2:+.2f} | "
                      f"{r.pclu_a2_holm:.3f} | {r.verdict_a2} | {r.rel_star_pct:+.2f} | {r.dmclu_star:+.2f} | "
                      f"{r.pclu_star_holm:.3f} | [{r.boot_lo:+.5f},{r.boot_hi:+.5f}] | **{r.verdict_star}** |")

    md.append("\n## Key cells — every SINGLE-model recalibrated reference (day-clustered DM)\n"
              "A text 'increment' that shrinks/dies against a single recalibrated EGARCH shows the text was "
              "proxying price dynamics the A2-only reference missed.\n")
    for (disc, m) in KEY_CELLS:
        md.append(f"\n### {m} ({disc})\n"
                  "| h | reference | QLIKE(R) | QLIKE(U) | rel% | g_text | cluDM | p | Holm | n_days |\n"
                  "|---|---|---|---|---|---|---|---|---|---|")
        sub = kdf[(kdf.disc == disc) & (kdf.model == m)].sort_values(["h", "reference"])
        for _, r in sub.iterrows():
            md.append(f"| {r.h} | {r.reference} | {r.qlike_R:.4f} | {r.qlike_U:.4f} | {r.rel_pct:+.2f} | "
                      f"{r.g_text:+.3f} | {r.dm_clustered:+.2f} | {r.p_clustered:.4f} | {r.p_holm:.3f} | "
                      f"{int(r.n_days)} |")

    # bottom line
    surv = df[df.verdict_star == "text adds"]
    md.append("\n## HONEST bottom line\n")
    md.append(f"- Against the maximal 5-model price reference with day-clustered inference, the text increment "
              f"survives in **{adds_s}/{n}** cells (was 38/69 'genuine' under A2-only + obs-order HAC) and "
              f"actively hurts in {hurts_s}.")
    if len(surv):
        md.append(f"- Surviving effect sizes: rel QLIKE {surv.rel_star_pct.min():+.2f}% to "
                  f"{surv.rel_star_pct.max():+.2f}% (median {surv.rel_star_pct.median():+.2f}%). "
                  f"Surviving cells: " + "; ".join(f"{r.disc}/{r.model}/h{r.h} {r.rel_star_pct:+.2f}%"
                                                   for _, r in surv.iterrows()) + ".")
    md.append(f"- Headline restatement: C2_finbert_s1 long_form h=10 goes +4.56% (A2-only) -> "
              f"{head.rel_a2_pct:+.2f}% under clustered DM (same ref) -> **{head.rel_star_pct:+.2f}%** against "
              f"the maximal price set (clustered DM {head.dmclu_star:+.2f}, Holm {head.pclu_star_holm:.4f}).")

    with open(out / "maximal_reference.md", "w") as fh:
        fh.write("\n".join(md))
    print(f"cells={n} adds_star={adds_s} hurts_star={hurts_s} adds_a2={adds_a2} absorbed={absorbed}")
    print(f"headline: A2ref {head.rel_a2_pct:+.3f}% (cluDM {head.dmclu_a2:+.2f}) | "
          f"maxref {head.rel_star_pct:+.3f}% (cluDM {head.dmclu_star:+.2f}, p={head.pclu_star:.4g})")
    print("wrote results/tables/maximal_reference.{csv,md} + maximal_reference_key_cells.csv")


if __name__ == "__main__":
    main()
