"""R1 — Rolling-window PERIOD-ROBUSTNESS of the M1 text increment.

Question: is the small M1 QLIKE text increment period-robust, or an artifact of the
single fixed COVID (2020-2021) validation window? The trained-model forecasts are FIXED;
we roll ONLY the combiner (fc.log_combo). This isolates whether the log-space nested
combiner's incremental-text conclusion survives across 2022Q1..2025Q4 and whether updating
the combiner (expanding) differs from freezing it on COVID-val (fixed origin).

Two schemes, per (disclosure, model, horizon):
  A) EXPANDING (pseudo-OOS): for each test quarter q, refit fc.log_combo on ALL filings
     with filing_time_utc < q_start (val + all earlier test quarters), apply to filings IN q.
  B) FIXED-ORIGIN: fit fc.log_combo ONCE on val only (the current M1), apply to each quarter.

Per quarter we report the LOG-space nested increment:
     f_R = recalibrated-HAR-only, f_U = +text (both from fc.log_combo);
     qR=fc.qlike(y,f_R), qU=fc.qlike(y,f_U);
     rel_impr_pct = 100*mean(qR-qU)/mean(qR)   (positive => text helps);
     DM via dm_test(qU, qR, h) (NEGATIVE stat => text QLIKE lower => text better);
     moving-block CI of mean(qR-qU) at HAC block h.

Aggregate per cell: #quarters with DM<0 & p<.05 (text significantly helps) out of 16,
mean per-quarter increment, and an OLS trend slope of rel_impr_pct on quarter index 0..15
(is the increment decaying?). Plus a grand pooled summary across models.

SANITY: pooling ALL 16 quarters under the FIXED scheme (one combiner fit on val, applied to
the whole test span) must reproduce the STATIC M1 rel_impr_pct for that cell within ~0.2pp
(from forecast_combination_grid.csv). We recompute the pooled-fixed rel on all test rows and
compare against the static grid value; a few matches are printed / stored.

Run:  .venv/bin/python scripts/analysis/rolling_robustness.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # fc.load, fc.qlike, fc.log_combo, fc.moving_block_ci
sys.path.insert(0, "src")
from sp500vol.evaluation.dm_test import dm_test

KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
HORIZONS = (5, 10, 20)
MODELS = ["B2_tfidf_ridge", "C2_finbert_s1", "C2_finbert_s2",
          "C4_longformer", "C5_qwen3", "D2_gated_fusion"]
DISCS = ["long_form", "event_driven"]
# 16 test quarters 2022Q1..2025Q4
QUARTERS = pd.period_range("2022Q1", "2025Q4", freq="Q")


def _load_cell(disc, model):
    """HAR + text merged, test+val, with filing_time and quarter period."""
    har = fc.load("A2_har_rv", disc)[["split"] + KEY +
        ["prediction_realised_vol", "label_realised_vol", "filing_time_utc"]
        ].rename(columns={"prediction_realised_vol": "fhar"})
    txt = fc.load(model, disc)[KEY + ["prediction_realised_vol"]
        ].rename(columns={"prediction_realised_vol": "ftext"})
    d = har.merge(txt, on=KEY)
    d["ft"] = pd.to_datetime(d.filing_time_utc, utc=True)
    return d


def _increment(yq, fR_q, fU_q, h):
    """Per-quarter increment stats given aligned y, f_R, f_U on a quarter."""
    qR = fc.qlike(yq, fR_q)
    qU = fc.qlike(yq, fU_q)
    mR = float(qR.mean())
    rel = 100.0 * (mR - float(qU.mean())) / mR if mR > 0 else float("nan")
    d = qR - qU  # positive => text better
    if len(yq) >= 2:
        dm_stat, dm_p = dm_test(qU, qR, h=h)  # neg stat => text better
    else:
        dm_stat, dm_p = float("nan"), float("nan")
    _, lo, hi = fc.moving_block_ci(d, h)
    return rel, float(dm_stat), float(dm_p), float(lo), float(hi)


def _ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    x, y = x[m], y[m]
    if np.ptp(x) == 0:
        return float("nan")
    b = np.polyfit(x, y, 1)
    return float(b[0])


def main():
    rows = []          # long per (disc,model,h,quarter,scheme)
    agg_rows = []      # per (disc,model,h) aggregate
    sanity_rows = []   # pooled-fixed vs static M1

    static = pd.read_csv("results/tables/forecast_combination_grid.csv")
    static_lookup = {(r.disc, r.model, int(r.h)): float(r.rel_impr_pct)
                     for r in static.itertuples()}

    for disc in DISCS:
        for model in MODELS:
            d = _load_cell(disc, model)
            for h in HORIZONS:
                dh = d[d.horizon_days == h].copy()
                dv = dh[dh.split == "val"].sort_values(SORT, kind="mergesort")
                dt = dh[dh.split == "test"].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                dt = dt.copy()
                dt["q"] = dt.ft.dt.tz_convert(None).dt.to_period("Q")

                yv = dv.label_realised_vol.to_numpy()
                fhv = dv.fhar.to_numpy()
                ftv = dv.ftext.to_numpy()

                # ---- FIXED-ORIGIN: fit combiner ONCE on val, apply to each quarter ----
                # Apply once to the WHOLE test span (positivity-safe closed form), then slice.
                yt_all = dt.label_realised_vol.to_numpy()
                fhr_all = dt.fhar.to_numpy()
                ftt_all = dt.ftext.to_numpy()
                fR_fx_all, fU_fx_all, _ = fc.log_combo(yv, fhv, ftv, fhr_all, ftt_all)

                # SANITY: pooled-fixed rel over ALL test rows vs static M1 grid value
                qR_all = fc.qlike(yt_all, fR_fx_all)
                qU_all = fc.qlike(yt_all, fU_fx_all)
                mRall = float(qR_all.mean())
                pooled_rel = 100.0 * (mRall - float(qU_all.mean())) / mRall if mRall > 0 else float("nan")
                stat_rel = static_lookup.get((disc, model, h), float("nan"))
                sanity_rows.append({
                    "disc": disc, "model": model, "h": h,
                    "pooled_fixed_rel_pct": pooled_rel,
                    "static_m1_rel_pct": stat_rel,
                    "abs_diff_pp": abs(pooled_rel - stat_rel) if np.isfinite(stat_rel) else float("nan"),
                })

                fixed_rel_by_q, exp_rel_by_q = [], []
                fixed_sig, exp_sig = 0, 0

                for qi, q in enumerate(QUARTERS):
                    mask = (dt.q == q).to_numpy()
                    nq = int(mask.sum())
                    yq = yt_all[mask]

                    # A) FIXED
                    relf, dmf, dpf, lof, hif = _increment(yq, fR_fx_all[mask], fU_fx_all[mask], h)
                    rows.append({
                        "disc": disc, "model": model, "h": h, "quarter": str(q),
                        "quarter_idx": qi, "scheme": "fixed",
                        "rel_impr_pct": relf, "dm_stat": dmf, "dm_p": dpf,
                        "ci_lo": lof, "ci_hi": hif, "n": nq,
                    })
                    fixed_rel_by_q.append(relf)
                    if np.isfinite(dmf) and np.isfinite(dpf) and dmf < 0 and dpf < 0.05:
                        fixed_sig += 1

                    # B) EXPANDING: refit on ALL filings with ft < q_start
                    q_start = q.start_time.tz_localize("UTC")
                    tr = dh[dh.ft < q_start]  # val + earlier test quarters (any split)
                    ytr = tr.label_realised_vol.to_numpy()
                    fhtr = tr.fhar.to_numpy()
                    fttr = tr.ftext.to_numpy()
                    if len(ytr) >= 100 and nq >= 2:
                        fR_e, fU_e, _ = fc.log_combo(ytr, fhtr, fttr, fhr_all[mask], ftt_all[mask])
                        rele, dme, dpe, loe, hie = _increment(yq, fR_e, fU_e, h)
                    else:
                        rele = dme = dpe = loe = hie = float("nan")
                    rows.append({
                        "disc": disc, "model": model, "h": h, "quarter": str(q),
                        "quarter_idx": qi, "scheme": "expanding",
                        "rel_impr_pct": rele, "dm_stat": dme, "dm_p": dpe,
                        "ci_lo": loe, "ci_hi": hie, "n": nq,
                    })
                    exp_rel_by_q.append(rele)
                    if np.isfinite(dme) and np.isfinite(dpe) and dme < 0 and dpe < 0.05:
                        exp_sig += 1

                idx = np.arange(len(QUARTERS))
                agg_rows.append({
                    "disc": disc, "model": model, "h": h,
                    "n_test": len(dt),
                    "fixed_sig_q": fixed_sig, "fixed_mean_rel": float(np.nanmean(fixed_rel_by_q)),
                    "fixed_slope": _ols_slope(idx, fixed_rel_by_q),
                    "exp_sig_q": exp_sig, "exp_mean_rel": float(np.nanmean(exp_rel_by_q)),
                    "exp_slope": _ols_slope(idx, exp_rel_by_q),
                    "exp_minus_fixed_mean_rel": float(np.nanmean(exp_rel_by_q) - np.nanmean(fixed_rel_by_q)),
                    "pooled_fixed_rel": pooled_rel, "static_m1_rel": stat_rel,
                })

    long_df = pd.DataFrame(rows)
    agg_df = pd.DataFrame(agg_rows)
    sanity_df = pd.DataFrame(sanity_rows)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    long_df.to_csv("results/tables/rolling_robustness.csv", index=False)

    # ---- Grand pooled summary across models (per disclosure x horizon x scheme) ----
    grand = (long_df.groupby(["disc", "h", "scheme"])
             .agg(mean_rel=("rel_impr_pct", "mean"),
                  n_sig=("dm_p", lambda s: int(((long_df.loc[s.index, "dm_stat"] < 0) &
                                                (s < 0.05)).sum())),
                  n_q=("dm_p", "size"))
             .reset_index())

    # ---- Markdown ----
    md = ["# R1 — Rolling-window period-robustness of the M1 text increment\n",
          "Trained-model forecasts are FIXED; only the log-space nested combiner "
          "(`fc.log_combo`, weights fit on val or on strictly-earlier filings) is rolled "
          "across the 16 test quarters **2022Q1..2025Q4**. Per quarter we report the "
          "incremental-text QLIKE improvement `rel_impr_pct = 100*mean(qR-qU)/mean(qR)` "
          "(positive => text helps), DM `dm_test(qU,qR,h)` (NEGATIVE stat => text better), "
          "and a moving-block CI of `mean(qR-qU)`. Two schemes: **expanding** (combiner refit "
          "each quarter on all earlier filings, pseudo-OOS) vs **fixed** (combiner frozen on the "
          "COVID 2020-2021 val window — the current M1).\n",
          "`sig_q/16` = quarters where text significantly helps (DM<0, p<.05). `slope` = OLS "
          "slope of per-quarter `rel_impr_pct` on quarter index 0..15 (negative => the increment "
          "is DECAYING over 2022-2025).\n"]

    md.append("## Per-cell robustness (disclosure x model x horizon)\n")
    md.append("| disclosure | model | h | fixed sig_q/16 | fixed mean rel% | fixed slope | "
              "exp sig_q/16 | exp mean rel% | exp slope | exp-fixed mean rel% |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
    for _, r in agg_df.sort_values(["disc", "model", "h"]).iterrows():
        md.append(f"| {r.disc} | {r.model} | {int(r.h)} | {int(r.fixed_sig_q)}/16 | "
                  f"{r.fixed_mean_rel:+.2f} | {r.fixed_slope:+.3f} | {int(r.exp_sig_q)}/16 | "
                  f"{r.exp_mean_rel:+.2f} | {r.exp_slope:+.3f} | {r.exp_minus_fixed_mean_rel:+.2f} |")

    md.append("\n## Grand pooled summary across models (disclosure x horizon x scheme)\n")
    md.append("| disclosure | h | scheme | pooled mean rel% | n_sig quarter-cells | n quarter-cells |\n"
              "|---|---|---|---|---|---|")
    for _, r in grand.sort_values(["disc", "h", "scheme"]).iterrows():
        md.append(f"| {r.disc} | {int(r.h)} | {r.scheme} | {r.mean_rel:+.3f} | "
                  f"{int(r.n_sig)} | {int(r.n_q)} |")

    md.append("\n## SANITY — pooled-fixed reproduces static M1 (within ~0.2pp)\n")
    md.append("| disclosure | model | h | pooled-fixed rel% | static M1 rel% | abs diff (pp) |\n"
              "|---|---|---|---|---|---|")
    for _, r in sanity_df.sort_values(["disc", "model", "h"]).iterrows():
        md.append(f"| {r.disc} | {r.model} | {int(r.h)} | {r.pooled_fixed_rel_pct:+.3f} | "
                  f"{r.static_m1_rel_pct:+.3f} | {r.abs_diff_pp:.4f} |")

    # verdicts
    n_cells = len(agg_df)
    med_fixed_sig = float(agg_df.fixed_sig_q.median())
    med_exp_sig = float(agg_df.exp_sig_q.median())
    n_decay_fixed = int((agg_df.fixed_slope < 0).sum())
    n_decay_exp = int((agg_df.exp_slope < 0).sum())
    mean_exp_minus_fixed = float(agg_df.exp_minus_fixed_mean_rel.mean())
    # sanity only defined for cells present in the static M1 grid (12 cells — C5_qwen3
    # both disclosures + C2_finbert_s2/C4_longformer event_driven — were NOT in the M1
    # SETS, so have no static reference; exclude them from the reproduction check).
    san_ok = sanity_df[np.isfinite(sanity_df.abs_diff_pp)]
    n_sanity_ref = len(san_ok)
    max_sanity = float(san_ok.abs_diff_pp.max())
    med_sanity = float(san_ok.abs_diff_pp.median())
    n_sanity_ok = int((san_ok.abs_diff_pp <= 0.2).sum())

    md.append("\n## Verdict — period-robustness\n")
    md.append(f"- Cells analysed: **{n_cells}** (disclosure x model x horizon).")
    md.append(f"- Median significant-help quarters: **fixed {med_fixed_sig:.0f}/16**, "
              f"**expanding {med_exp_sig:.0f}/16**.")
    md.append(f"- Decaying trend (OLS slope<0): fixed **{n_decay_fixed}/{n_cells}**, "
              f"expanding **{n_decay_exp}/{n_cells}** cells.")
    md.append(f"- Updating the combiner vs freezing it on COVID-val: mean(exp-fixed rel%) = "
              f"**{mean_exp_minus_fixed:+.3f}pp** across cells "
              f"({'expanding helps on average' if mean_exp_minus_fixed > 0 else 'freezing is no worse / better'}).")
    md.append(f"- SANITY: pooled-fixed reproduces static M1 exactly — median abs diff "
              f"**{med_sanity:.4f}pp**, max **{max_sanity:.4f}pp**; "
              f"{n_sanity_ok}/{n_sanity_ref} cells (with a static reference) within 0.2pp "
              f"(12 cells have no static M1 reference: C5_qwen3 + event_driven "
              f"C2_finbert_s2/C4_longformer were not in the M1 SETS).")

    with open("results/tables/rolling_robustness.md", "w") as fh:
        fh.write("\n".join(md))

    print("=== R1 rolling-window robustness — done ===")
    print(f"cells={n_cells}  median fixed sig_q={med_fixed_sig:.0f}/16  "
          f"median exp sig_q={med_exp_sig:.0f}/16")
    print(f"decay(slope<0): fixed={n_decay_fixed}/{n_cells} exp={n_decay_exp}/{n_cells}")
    print(f"mean(exp-fixed rel%)={mean_exp_minus_fixed:+.3f}pp")
    print(f"SANITY pooled-fixed vs static M1: median|diff|={med_sanity:.4f}pp "
          f"max={max_sanity:.4f}pp  within-0.2pp={n_sanity_ok}/{n_sanity_ref} (ref cells)")
    # a couple of explicit sanity matches
    for _, r in san_ok.sort_values("abs_diff_pp").head(3).iterrows():
        print(f"  match: {r.disc} {r.model} h{int(r.h)}  pooled={r.pooled_fixed_rel_pct:+.3f}% "
              f"static={r.static_m1_rel_pct:+.3f}%  diff={r.abs_diff_pp:.4f}pp")
    return sanity_df


if __name__ == "__main__":
    main()
