"""P1-firm — FIRM-MEAN CONTROL for the M1 incremental-text test (the R4 confound).

Reviewer-verified confound: text models can encode FIRM IDENTITY (some firms are
persistently volatile) rather than disclosure content; the recalibrated-HAR
reference has no firm effect, so a firm-identity proxy masquerades as "text adds".

Control: augment the reference with the firm's own validation-period mean RV:

    f_R_firm = exp( a + b*log fHAR + c*log firm_mean_val_RV )   (val-fit, frozen)
    f_U_firm = f_R_firm design + g*log f_text

firm_mean_val_RV = the firm's mean label_realised_vol over ITS OWN val-split rows
(same disclosure set + horizon); firms absent from val get the global val mean
(coverage reported). Also runs the ZERO-TEXT firm-mean-only forecast (f_R_firm)
against plain f_R to quantify how much of the original "text increment" firm
identity ALONE reproduces. Inference: day-clustered DM (clustered_dm.py), Holm
within the 69-cell grid.

Run from repo root: .venv/bin/python scripts/analysis/firm_identity_control.py
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


def fit_apply_log(yv, Xv_list, Xt_list):
    ly = np.log(np.clip(np.asarray(yv, float), EPS, None))
    Lv = [np.log(np.clip(np.asarray(x, float), EPS, None)) for x in Xv_list]
    Lt = [np.log(np.clip(np.asarray(x, float), EPS, None)) for x in Xt_list]
    beta = fc.ols(ly, np.column_stack([np.ones(len(ly))] + Lv))
    return np.exp(np.column_stack([np.ones(len(Lt[0]))] + Lt) @ beta), beta


def firm_means(dv, dt):
    """Firm mean val RV mapped to val and test rows; returns (fm_val, fm_test, coverage)."""
    fmap = dv.groupby("ticker")["label_realised_vol"].mean()
    gmean = float(dv["label_realised_vol"].mean())
    fm_v = dv["ticker"].map(fmap).to_numpy(dtype=float)
    fm_t = dt["ticker"].map(fmap)
    coverage = float(fm_t.notna().mean())
    return fm_v, fm_t.fillna(gmean).to_numpy(dtype=float), coverage


def main():
    rows, zero_rows = [], []
    for disc, models in fc.SETS.items():
        har = fc.load("A2_har_rv", disc)[
            ["split"] + KEY + ["label_realised_vol", "filing_time_utc",
                               "effective_trading_day", "prediction_realised_vol"]
        ].rename(columns={"prediction_realised_vol": "fhar"})

        # ---- ZERO-TEXT firm-mean-only vs plain f_R (per disc x h, full A2 panel)
        for h in fc.HORIZONS:
            dv = har[(har.horizon_days == h) & (har.split == "val")].sort_values(SORT, kind="mergesort")
            dt = har[(har.horizon_days == h) & (har.split == "test")].sort_values(SORT, kind="mergesort")
            yv, yt = dv.label_realised_vol.to_numpy(), dt.label_realised_vol.to_numpy()
            fhv, fht = dv.fhar.to_numpy(), dt.fhar.to_numpy()
            fm_v, fm_t, cov = firm_means(dv, dt)
            days_t = dt.effective_trading_day.to_numpy()
            fR, _ = fit_apply_log(yv, [fhv], [fht])
            fRf, bF = fit_apply_log(yv, [fhv, fm_v], [fht, fm_t])
            lR, lRf = fc.qlike(yt, fR), fc.qlike(yt, fRf)
            rel = 100.0 * (lR.mean() - lRf.mean()) / lR.mean()
            dm, p, nd = dm_test_clustered(lRf, lR, days_t, h)
            zero_rows.append({"disc": disc, "h": h, "n_test": len(dt), "n_days": nd,
                              "coverage": cov, "qlike_R": float(lR.mean()),
                              "qlike_R_firm": float(lRf.mean()), "rel_pct": float(rel),
                              "c_firm": float(bF[2]), "dm_clustered": dm, "p_clustered": p})

        for m in models:
            txt = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
            d = har.merge(txt, on=KEY, how="inner")
            for h in fc.HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv, yt = dv.label_realised_vol.to_numpy(), dt.label_realised_vol.to_numpy()
                fhv, fht = dv.fhar.to_numpy(), dt.fhar.to_numpy()
                ftv, ftt = dv.ftext.to_numpy(), dt.ftext.to_numpy()
                fm_v, fm_t, cov = firm_means(dv, dt)
                days_t = dt.effective_trading_day.to_numpy()

                # ORIGINAL A2-only reference (clustered DM restated)
                fR0, fU0, _ = fc.log_combo(yv, fhv, ftv, fht, ftt)
                lR0, lU0 = fc.qlike(yt, fR0), fc.qlike(yt, fU0)
                rel0 = 100.0 * (lR0.mean() - lU0.mean()) / lR0.mean()
                dm0, p0, _ = dm_test_clustered(lU0, lR0, days_t, h)

                # FIRM-augmented reference
                fRf, _ = fit_apply_log(yv, [fhv, fm_v], [fht, fm_t])
                fUf, bU = fit_apply_log(yv, [fhv, fm_v, ftv], [fht, fm_t, ftt])
                lRf, lUf = fc.qlike(yt, fRf), fc.qlike(yt, fUf)
                relf = 100.0 * (lRf.mean() - lUf.mean()) / lRf.mean()
                dmf, pf, ndf = dm_test_clustered(lUf, lRf, days_t, h)
                _, lo, hi = mbb_ci_daily(lUf - lRf, days_t, h)

                rows.append({
                    "disc": disc, "model": m, "h": h, "n_test": len(dt), "n_days": ndf,
                    "coverage": cov,
                    "rel_a2_pct": float(rel0), "dmclu_a2": dm0, "pclu_a2": p0,
                    "qlike_R_firm": float(lRf.mean()), "qlike_U_firm": float(lUf.mean()),
                    "rel_firm_pct": float(relf), "dmclu_firm": dmf, "pclu_firm": pf,
                    "boot_lo": lo, "boot_hi": hi, "g_text_firm": float(bU[3]),
                })

    df = pd.DataFrame(rows)
    df["pclu_a2_holm"] = fc.holm(df.pclu_a2.fillna(1.0).values)
    df["pclu_firm_holm"] = fc.holm(df.pclu_firm.fillna(1.0).values)

    def verdict(dm, hp):
        if hp < 0.05:
            return "text adds" if dm < 0 else "text HURTS"
        return "null"

    df["verdict_a2"] = [verdict(a, b) for a, b in zip(df.dmclu_a2, df.pclu_a2_holm, strict=False)]
    df["verdict_firm"] = [verdict(a, b) for a, b in zip(df.dmclu_firm, df.pclu_firm_holm, strict=False)]
    zdf = pd.DataFrame(zero_rows)
    zdf["p_holm"] = fc.holm(zdf.p_clustered.fillna(1.0).values)

    out = Path("results/tables")
    df.to_csv(out / "firm_identity_control.csv", index=False)
    zdf.to_csv(out / "firm_identity_control_zerotext.csv", index=False)

    n = len(df)
    adds_a2 = int((df.verdict_a2 == "text adds").sum())
    adds_f = int((df.verdict_firm == "text adds").sum())
    hurts_f = int((df.verdict_firm == "text HURTS").sum())
    lf = df[df.disc == "long_form"]; ed = df[df.disc == "event_driven"]
    lf_neg = int((lf.rel_firm_pct < 0).sum())
    surv = df[df.verdict_firm == "text adds"]
    head = df[(df.model == "C2_finbert_s1") & (df.disc == "long_form") & (df.h == 10)].iloc[0]

    md = ["# P1-firm — Firm-identity (firm-mean-RV) control on the M1 text increment, day-clustered DM\n",
          "## RESTATED vs ORIGINAL\n",
          "| quantity | ORIGINAL (recalibrated-HAR ref, no firm control) | RESTATED (HAR + firm-mean-val-RV ref) |",
          "|---|---|---|",
          f"| headline cell (C2_finbert_s1, long_form, h=10) | +4.56% (obs-order DM -12.77); "
          f"{head.rel_a2_pct:+.2f}% under clustered DM | **{head.rel_firm_pct:+.2f}%**, clustered DM "
          f"{head.dmclu_firm:+.2f}, Holm {head.pclu_firm_holm:.3f} |",
          f"| cells where text adds (Holm<.05) | 38/69 (original inference); {adds_a2}/{n} clustered-DM A2-only | "
          f"**{adds_f}/{n}** once firm identity is in the reference |",
          f"| long_form cells with NEGATIVE rel% | — | {lf_neg}/{len(lf)} |",
          f"| cells where text HURTS (Holm<.05) | — | {hurts_f}/{n} |",
          "\nReference: f_R_firm = exp(a + b·log fHAR + c·log firm_mean_val_RV), val-fit, frozen to test; "
          "firm_mean_val_RV = firm's mean label RV over its own val rows (missing firms -> global val mean; "
          "coverage reported per cell). f_U_firm adds g·log f_text. Day-clustered DM, Holm within the grid.\n"]

    md.append("\n## ZERO-TEXT check — how much does firm identity ALONE reproduce?\n"
              "f_R_firm (no text at all) vs plain recalibrated-HAR f_R. Positive rel% = firm identity alone "
              "improves the reference by that much — increment previously attributed to 'text'.\n"
              "| disc | h | n_days | coverage | QLIKE(f_R) | QLIKE(f_R_firm) | rel% | c_firm | cluDM | p | Holm |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in zdf.iterrows():
        md.append(f"| {r.disc} | {r.h} | {int(r.n_days)} | {r.coverage:.3f} | {r.qlike_R:.4f} | "
                  f"{r.qlike_R_firm:.4f} | {r.rel_pct:+.2f} | {r.c_firm:+.3f} | {r.dm_clustered:+.2f} | "
                  f"{r.p_clustered:.4f} | {r.p_holm:.3f} |")

    for disc in fc.SETS:
        md.append(f"\n## {disc} — 69-cell grid slice\n"
                  "| model | h | n_days | cov | rel% (A2 ref) | cluDM | Holm | verdict(A2) | "
                  "rel% (firm ref) | cluDM | Holm | daily-dQ 95% CI | verdict(firm) |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df[df.disc == disc].sort_values(["model", "h"]).iterrows():
            md.append(f"| {r.model} | {r.h} | {int(r.n_days)} | {r.coverage:.2f} | {r.rel_a2_pct:+.2f} | "
                      f"{r.dmclu_a2:+.2f} | {r.pclu_a2_holm:.3f} | {r.verdict_a2} | {r.rel_firm_pct:+.2f} | "
                      f"{r.dmclu_firm:+.2f} | {r.pclu_firm_holm:.3f} | "
                      f"[{r.boot_lo:+.5f},{r.boot_hi:+.5f}] | **{r.verdict_firm}** |")

    md.append("\n## HONEST bottom line\n")
    md.append(f"- Once the reference knows each firm's validation-period mean RV, the text increment survives "
              f"Holm<.05 in **{adds_f}/{n}** cells (vs {adds_a2}/{n} A2-only clustered, 38/69 originally) and "
              f"turns significantly NEGATIVE in {hurts_f}.")
    md.append(f"- long_form: {lf_neg}/{len(lf)} cells have negative point rel% under the firm control — "
              f"the 10-K/'long form' increment is largely FIRM IDENTITY, not disclosure content.")
    if len(surv):
        md.append("- Surviving cells: " + "; ".join(
            f"{r.disc}/{r.model}/h{r.h} {r.rel_firm_pct:+.2f}% (cluDM {r.dmclu_firm:+.2f})"
            for _, r in surv.iterrows()) + ".")
    else:
        md.append("- NO cell survives the firm-identity control at Holm<.05.")
    md.append(f"- Zero-text check: firm identity alone improves the plain recalibrated-HAR reference by "
              f"{zdf.rel_pct.min():+.2f}% to {zdf.rel_pct.max():+.2f}% "
              f"(long_form h=10: {zdf[(zdf.disc=='long_form')&(zdf.h==10)].rel_pct.iloc[0]:+.2f}%) — "
              f"a large share of what was previously booked as 'text'.")

    with open(out / "firm_identity_control.md", "w") as fh:
        fh.write("\n".join(md))
    print(f"cells={n} adds_firm={adds_f} hurts_firm={hurts_f} adds_a2clu={adds_a2} lf_neg={lf_neg}/{len(lf)}")
    print(zdf[["disc", "h", "rel_pct", "dm_clustered", "p_clustered", "coverage"]].to_string(index=False))
    print(f"headline: firm-ref {head.rel_firm_pct:+.3f}% (cluDM {head.dmclu_firm:+.2f}, "
          f"Holm {head.pclu_firm_holm:.4f})")
    print("wrote results/tables/firm_identity_control.{csv,md} + firm_identity_control_zerotext.csv")


if __name__ == "__main__":
    main()
