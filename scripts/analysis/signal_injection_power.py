"""ROW 1 (REVIEW_ROUND3_FRESH_PANEL) — Signal-injection POWER calibration + per-cell MDE.

Answers the corroborated Devil's-Advocate CRITICAL #4 / methodology CRITICAL /
perspective CRITICAL: "the 0/69 conjunction headline has no power analysis".

!!!!! ORACLE INJECTION — THIS IS A POWER CALIBRATION, NOT A FORECAST !!!!!
The synthetic text forecast uses TEST labels BY DESIGN (the one declared exception
to the no-look-ahead rule). Nothing here may ever be cited as a forecasting result.

Design (skeleton from the row-1 brief, followed literally):
 (a) For each of the 69 M1 cells (grid + seed-ensemble text object identical to
     m1_ensemble_primary.py), build a synthetic text forecast carrying a KNOWN,
     firm-identity-ORTHOGONAL signal. In log space, on the TEST split:
         s_i = (log y_i - log f_R,i) demeaned WITHIN FIRM (ticker) on the test split,
     where f_R = the single recalibrated-HAR reference (val-fit, frozen). Because s
     is within-firm demeaned, a firm-level regressor (the firm-identity reference)
     cannot mechanically absorb it. Then
         f_synth,i = exp( log f_text,i + delta * s_i )        (test rows only;
     validation text stays REAL, so every combiner weight remains a genuine
     validation-only fit — the oracle content enters ONLY through the test-side
     text array).
 (b) Per cell, delta is calibrated by bisection so the realised TEST rel-QLIKE
     improvement of the combined f_U over f_R hits targets {0.3, 0.5, 1.0}% within
     0.02pp. Since the stage-1 combined forecast is log f_U(delta) = log f_U(0)
     + g1*delta*s, the bisection runs on kappa = g1*delta (monotone on the
     relevant branch; QLIKE is convex in kappa), then delta = kappa/g1.
     Cells whose REAL improvement already exceeds the target get delta < 0 (signal
     REMOVED down to the target): the design equalises the realised effect at
     exactly X% in every cell, which is what makes the recovery rate a power curve.
 (c) For each target, the SAME f_synth is pushed through the full cascade with each
     stage's own validation-fit (real-text) weights:
       stage HAR  : f_R  = exp OLS[1, log fHAR]                      (A2 panel)
       stage FIRM : f_Rf = exp OLS[1, log fHAR, log firm_mean_val_RV] (5-price panel)
       stage POOL : f_R* = exp OLS[1, log fA2..log fARIMA]            (5-price panel)
     Detection = clustered DM < 0 AND Holm < .05 within the pre-declared 69-cell
     family of that (stage, target). Transmission into stage FIRM/POOL uses that
     stage's real val-fit text coefficient (kappa_stage = g_stage * delta), i.e.
     the cascade is tested AS DEPLOYED; per-stage effective strengths are reported.
 (d) Per-cell MDE from the REAL (delta=0) daily loss-differential series:
       MDE_rel% = (1.96 + 0.84) * SE_daily / mean(QLIKE_R) * 100
     with SE_daily = sqrt( HAC(lag=h-1) variance of the daily-mean differential
     / n_days ) — 80% power at 5% two-sided size (approx).

SANITY GATES (hard; the run ABORTS on failure):
  A. delta=0 stage-HAR must reproduce results/tables/m1_ensemble_primary.csv
     columns vol_qlike_R and vol_rel_impr_pct to machine precision on all 69 cells.
  B. delta=0 stage-FIRM must reproduce results/tables/firm_identity_ensemble.csv
     (qlike_Rfirm, rel_impr_pct_firm, dm_q_clustered) likewise.
  C. delta=0 stage-POOL must reproduce results/tables/maximal_reference_ensemble.csv
     (qlike_Rstar, rel_impr_pct_maximal, dm_q_clustered) likewise.
  D. delta=0 Holm detection counts must equal control_intersection_ensemble.csv
     (38 / 8 / 9); injected s must be within-firm mean-zero to <1e-12.

Outputs: results/tables/signal_injection_power.{csv,md}
Run from repo root:  .venv/bin/python scripts/analysis/signal_injection_power.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")
import forecast_combination as fc
import m1_ensemble_primary as mep
from clustered_dm import daily_mean, dm_test_clustered
from maximal_reference_firm_control import (
    PRICE,
    build_price_panel,
    firm_mean_val,
    log_ols_frozen,
)

from sp500vol.evaluation.dm_test import _hac_variance

T = Path("results/tables")
KEY = fc.KEY
SORT = fc.SORT
HORIZONS = fc.HORIZONS
GRIDKEY = ["disc", "model", "h"]
TARGETS = (0.3, 0.5, 1.0)   # target realised rel-QLIKE improvement of f_U over f_R, %
TOL_PP = 0.02               # bisection tolerance, percentage points
KAPPA_CAP = 8.0             # |kappa| search cap (never binds for these targets)
Z_POWER = 1.96 + 0.84       # 5% two-sided size + 80% power
GATE_TOL = 1e-10            # "machine precision" gate (CSV round-trip is exact)
EPS = 1e-8


# --------------------------------------------------------------------- calibration
def calibrate_kappa(rel_fn, target):
    """Bisect kappa so rel_fn(kappa) hits target within TOL_PP. rel_fn is monotone
    increasing on the searched branch (QLIKE convex in kappa; targets are far below
    the optimum). Returns (kappa, achieved_rel, converged)."""
    r0 = rel_fn(0.0)
    if abs(r0 - target) <= TOL_PP:
        return 0.0, r0, True
    if r0 < target:                      # inject signal: kappa > 0
        lo, hi = 0.0, 0.05
        while rel_fn(hi) < target and hi < KAPPA_CAP:
            lo, hi = hi, hi * 2.0
        if rel_fn(hi) < target:
            return hi, rel_fn(hi), False
    else:                                # remove signal down to target: kappa < 0
        lo, hi = -0.05, 0.0
        while rel_fn(lo) > target and lo > -KAPPA_CAP:
            hi, lo = lo, lo * 2.0
        if rel_fn(lo) > target:
            return lo, rel_fn(lo), False
    mid, r = 0.5 * (lo + hi), r0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        r = rel_fn(mid)
        if abs(r - target) <= TOL_PP:
            return mid, r, True
        if r < target:
            lo = mid
        else:
            hi = mid
    return mid, r, False


def rel_pct(qR, lU):
    return 100.0 * (qR - float(lU.mean())) / qR if qR > 0 else float("nan")


# --------------------------------------------------------------------- cell prep
def prep_cells():
    """Precompute, per 69-cell, everything delta-independent for all three stages."""
    cells = []
    for disc, models in fc.SETS.items():
        # stage-HAR panel: byte-identical to m1_ensemble_primary.py
        har = pd.read_parquet(mep.run_dir("A2_har_rv", disc, 2026))[
            ["split"] + KEY + ["prediction_realised_vol", "label_realised_vol",
                               "filing_time_utc", "effective_trading_day"]
        ].rename(columns={"prediction_realised_vol": "fhar"})
        # stage-FIRM/POOL panel: byte-identical to basis_alignment_ensemble.py
        panel = build_price_panel(disc)
        fmap, gmean, _fcov, _ocov = firm_mean_val(panel)
        panel["firm_mean_val"] = panel.ticker.map(fmap).fillna(gmean).astype(float)
        for m in models:
            ens, used = mep.ensemble_text(m, disc)
            d1 = har.merge(ens, on=KEY)
            d23 = panel.merge(ens, on=KEY, how="inner")
            for h in HORIZONS:
                dv = d1[(d1.horizon_days == h) & (d1.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d1[(d1.horizon_days == h) & (d1.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                c = {"disc": disc, "model": m, "h": h,
                     "n_seeds": len(used), "n_test": len(dt)}

                yv, fhv, ftv = (dv.label_realised_vol.to_numpy(),
                                dv.fhar.to_numpy(), dv.ftext.to_numpy())
                yt, fhr, ftt = (dt.label_realised_vol.to_numpy(),
                                dt.fhar.to_numpy(), dt.ftext.to_numpy())
                days1 = dt.effective_trading_day.to_numpy()

                # ---- stage HAR (weights = REAL validation fit; frozen) ----
                fR, fU0, g1 = fc.log_combo(yv, fhv, ftv, fhr, ftt)
                lR = fc.qlike(yt, fR)
                qR = float(lR.mean())
                c.update(yt=yt, days1=days1, lR1=lR, qR1=qR, fU10=fU0,
                         luU10=np.log(fU0), g1=float(g1))

                # ---- injected signal: within-firm demeaned log residual (TEST) ----
                lres = pd.Series(np.log(np.clip(yt, EPS, None))
                                 - np.log(np.clip(fR, EPS, None)))
                firm = pd.Series(dt.ticker.to_numpy())
                s1 = (lres - lres.groupby(firm).transform("mean")).to_numpy()
                c["s1"] = s1
                c["s_within_firm_max_absmean"] = float(
                    pd.Series(s1).groupby(firm).mean().abs().max())

                # ---- MDE from the REAL (delta=0) daily loss differential ----
                lU0 = fc.qlike(yt, fU0)
                dd, _ = daily_mean(lU0 - lR, days1)
                v = _hac_variance(dd, lag=max(h - 1, 0))
                se_daily = float(np.sqrt(v / len(dd))) if v > 0 else float("nan")
                c["n_days"] = len(dd)
                c["se_daily"] = se_daily
                c["mde_rel_pct"] = Z_POWER * se_daily / qR * 100.0

                # ---- stage FIRM + POOL on the 5-price panel ----
                dv2 = d23[(d23.horizon_days == h) & (d23.split == "val")].sort_values(SORT, kind="mergesort")
                dt2 = d23[(d23.horizon_days == h) & (d23.split == "test")].sort_values(SORT, kind="mergesort")
                yv2, yt2 = dv2.label_realised_vol.to_numpy(), dt2.label_realised_vol.to_numpy()
                fhv2, fhr2 = dv2.A2_har_rv.to_numpy(), dt2.A2_har_rv.to_numpy()
                ftv2, ftt2 = dv2.ftext.to_numpy(), dt2.ftext.to_numpy()
                fmv, fmt = dv2.firm_mean_val.to_numpy(), dt2.firm_mean_val.to_numpy()
                days2 = (dt2.effective_trading_day.fillna(dt2.filing_time_utc)).to_numpy()

                fRf, _ = log_ols_frozen(yv2, [fhv2, fmv], [fhr2, fmt])
                fUf0, bUf = log_ols_frozen(yv2, [fhv2, fmv, ftv2], [fhr2, fmt, ftt2])
                pv = [dv2[col].to_numpy() for col in PRICE]
                pt = [dt2[col].to_numpy() for col in PRICE]
                fRs, _ = log_ols_frozen(yv2, pv, pt)
                fUs0, bUs = log_ols_frozen(yv2, pv + [ftv2], pt + [ftt2])

                # map s onto the (subset) 5-price-panel test rows
                idx1 = pd.MultiIndex.from_arrays([dt.ticker, dt.accession])
                assert not idx1.duplicated().any(), "non-unique (ticker,accession) in cell"
                s_map = pd.Series(s1, index=idx1)
                idx2 = pd.MultiIndex.from_arrays([dt2.ticker, dt2.accession])
                s2 = s_map.reindex(idx2).to_numpy()
                assert not np.isnan(s2).any(), "5-price-panel test rows not a subset of A2 rows"

                c.update(yt2=yt2, days2=days2, n_test2=len(dt2), s2=s2,
                         lRf=fc.qlike(yt2, fRf), fUf0=fUf0, luUf0=np.log(fUf0),
                         g_firm=float(bUf[-1]),
                         lRs=fc.qlike(yt2, fRs), fUs0=fUs0, luUs0=np.log(fUs0),
                         g_star=float(bUs[-1]))
                c["qRf"] = float(c["lRf"].mean())
                c["qRs"] = float(c["lRs"].mean())
                cells.append(c)
    return cells


def stage_eval(c, stage, kappa_1):
    """Evaluate one stage at the calibrated injection. kappa_1 = g1*delta (stage-HAR
    effective strength); FIRM/POOL transmit with their OWN real val-fit g."""
    delta = kappa_1 / c["g1"]
    if stage == "har":
        yt, days, lR, qR, s = c["yt"], c["days1"], c["lR1"], c["qR1"], c["s1"]
        fU = c["fU10"] if kappa_1 == 0.0 else np.exp(c["luU10"] + kappa_1 * s)
        kap = kappa_1
    elif stage == "firm":
        yt, days, lR, qR, s = c["yt2"], c["days2"], c["lRf"], c["qRf"], c["s2"]
        kap = c["g_firm"] * delta
        fU = c["fUf0"] if kappa_1 == 0.0 else np.exp(c["luUf0"] + kap * s)
    else:
        yt, days, lR, qR, s = c["yt2"], c["days2"], c["lRs"], c["qRs"], c["s2"]
        kap = c["g_star"] * delta
        fU = c["fUs0"] if kappa_1 == 0.0 else np.exp(c["luUs0"] + kap * s)
    lU = fc.qlike(yt, fU)
    dm, p, _ = dm_test_clustered(lU, lR, days, c["h"])
    return {"rel": rel_pct(qR, lU), "dm": dm, "p": p, "kappa": kap}


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    cells = prep_cells()
    assert len(cells) == 69, f"expected the 69-cell M1 grid, got {len(cells)}"
    print(f"[prep] 69 cells ready in {time.time()-t0:.1f}s")

    # ---------------- delta = 0 (real data) pass: gates + baseline ----------------
    base_rows = []
    for c in cells:
        row = {k: c[k] for k in ("disc", "model", "h", "n_seeds", "n_test",
                                 "n_test2", "n_days", "qR1", "se_daily",
                                 "mde_rel_pct", "g1", "g_firm", "g_star",
                                 "s_within_firm_max_absmean")}
        for stage in ("har", "firm", "pool"):
            r = stage_eval(c, stage, 0.0)
            row.update({f"{stage}0_rel": r["rel"], f"{stage}0_dm": r["dm"],
                        f"{stage}0_p": r["p"]})
        base_rows.append(row)
    base = pd.DataFrame(base_rows)
    for stage in ("har", "firm", "pool"):
        base[f"{stage}0_holm"] = fc.holm(base[f"{stage}0_p"].fillna(1.0).values)
        base[f"{stage}0_detect"] = (base[f"{stage}0_dm"] < 0) & (base[f"{stage}0_holm"] < 0.05)

    # ---------------- SANITY GATES (abort on failure) ----------------
    en = pd.read_csv(T / "m1_ensemble_primary.csv")
    fi = pd.read_csv(T / "firm_identity_ensemble.csv")
    mx = pd.read_csv(T / "maximal_reference_ensemble.csv")
    ci = pd.read_csv(T / "control_intersection_ensemble.csv")
    g = base.merge(en[GRIDKEY + ["vol_qlike_R", "vol_rel_impr_pct"]], on=GRIDKEY,
                   validate="1:1")
    g = g.merge(fi[GRIDKEY + ["qlike_Rfirm", "rel_impr_pct_firm", "dm_q_clustered"]]
                .rename(columns={"dm_q_clustered": "dm_firm_tab"}), on=GRIDKEY, validate="1:1")
    g = g.merge(mx[GRIDKEY + ["qlike_Rstar", "rel_impr_pct_maximal", "dm_q_clustered"]]
                .rename(columns={"dm_q_clustered": "dm_star_tab"}), on=GRIDKEY, validate="1:1")
    sanity = {
        "A_max_diff_qlike_R_vs_m1_ensemble_primary": float((g.qR1 - g.vol_qlike_R).abs().max()),
        "A_max_diff_rel_pct_vs_m1_ensemble_primary": float((g.har0_rel - g.vol_rel_impr_pct).abs().max()),
        "B_max_diff_qlike_Rfirm_vs_firm_identity_ensemble": float((g.firm0_rel * 0 + (base.merge(fi[GRIDKEY + ['qlike_Rfirm']], on=GRIDKEY).qlike_Rfirm - base.merge(fi[GRIDKEY + ['qlike_Rfirm']], on=GRIDKEY).qlike_Rfirm)).abs().max()),
        "B_max_diff_rel_firm_vs_firm_identity_ensemble": float((g.firm0_rel - g.rel_impr_pct_firm).abs().max()),
        "B_max_diff_dm_firm": float((g.firm0_dm - g.dm_firm_tab).abs().max()),
        "C_max_diff_rel_pool_vs_maximal_reference_ensemble": float((g.pool0_rel - g.rel_impr_pct_maximal).abs().max()),
        "C_max_diff_dm_pool": float((g.pool0_dm - g.dm_star_tab).abs().max()),
        "D_delta0_holm_counts_this_run": [int(base.har0_detect.sum()),
                                          int(base.firm0_detect.sum()),
                                          int(base.pool0_detect.sum())],
        "D_delta0_holm_counts_committed_ci": [int(ci.primary_holm.sum()),
                                              int(ci.firm_holm.sum()),
                                              int(ci.maximal_holm.sum())],
        "D_s_within_firm_max_absmean": float(base.s_within_firm_max_absmean.max()),
    }
    # recompute B qlike_Rfirm diff properly (typo-proof, explicit)
    bq = base.merge(fi[GRIDKEY + ["qlike_Rfirm"]], on=GRIDKEY, validate="1:1")
    fq = []
    for c in cells:
        fq.append({"disc": c["disc"], "model": c["model"], "h": c["h"], "qRf": c["qRf"],
                   "qRs": c["qRs"]})
    fq = pd.DataFrame(fq).merge(fi[GRIDKEY + ["qlike_Rfirm"]], on=GRIDKEY, validate="1:1") \
                         .merge(mx[GRIDKEY + ["qlike_Rstar"]], on=GRIDKEY, validate="1:1")
    sanity["B_max_diff_qlike_Rfirm_vs_firm_identity_ensemble"] = float((fq.qRf - fq.qlike_Rfirm).abs().max())
    sanity["C_max_diff_qlike_Rstar_vs_maximal_reference_ensemble"] = float((fq.qRs - fq.qlike_Rstar).abs().max())

    gate_pass = (
        sanity["A_max_diff_qlike_R_vs_m1_ensemble_primary"] < GATE_TOL
        and sanity["A_max_diff_rel_pct_vs_m1_ensemble_primary"] < GATE_TOL
        and sanity["B_max_diff_qlike_Rfirm_vs_firm_identity_ensemble"] < GATE_TOL
        and sanity["B_max_diff_rel_firm_vs_firm_identity_ensemble"] < GATE_TOL
        and sanity["B_max_diff_dm_firm"] < GATE_TOL
        and sanity["C_max_diff_qlike_Rstar_vs_maximal_reference_ensemble"] < GATE_TOL
        and sanity["C_max_diff_rel_pool_vs_maximal_reference_ensemble"] < GATE_TOL
        and sanity["C_max_diff_dm_pool"] < GATE_TOL
        and sanity["D_delta0_holm_counts_this_run"] == sanity["D_delta0_holm_counts_committed_ci"]
        and sanity["D_s_within_firm_max_absmean"] < 1e-12
    )
    sanity["pass"] = bool(gate_pass)
    print("[sanity]", json.dumps(sanity, indent=2))
    if not gate_pass:
        raise SystemExit("SANITY GATE FAILED — delta=0 pipeline does not reproduce the "
                         "committed tables (m1_ensemble_primary / firm_identity_ensemble / "
                         "maximal_reference_ensemble / control_intersection_ensemble). "
                         "NO numbers shipped.")
    print(f"[sanity] ALL GATES PASS ({time.time()-t0:.1f}s)")

    # ---------------- calibration + cascade at each target ----------------
    inj_rows = []
    for i, c in enumerate(cells):
        def rel_fn(kappa, c=c):
            fU = c["fU10"] if kappa == 0.0 else np.exp(c["luU10"] + kappa * c["s1"])
            return rel_pct(c["qR1"], fc.qlike(c["yt"], fU))
        for tgt in TARGETS:
            kap1, achieved, ok = calibrate_kappa(rel_fn, tgt)
            row = {"disc": c["disc"], "model": c["model"], "h": c["h"],
                   "target_pct": tgt, "kappa1": kap1, "delta": kap1 / c["g1"],
                   "delta_negative": bool(kap1 / c["g1"] < 0),
                   "converged": ok, "rel1_achieved": achieved,
                   "g1": c["g1"], "g_firm": c["g_firm"], "g_star": c["g_star"],
                   "n_test": c["n_test"], "n_test2": c["n_test2"],
                   "n_days": c["n_days"], "mde_rel_pct": c["mde_rel_pct"]}
            for stage in ("har", "firm", "pool"):
                r = stage_eval(c, stage, kap1)
                row.update({f"{stage}_rel": r["rel"], f"{stage}_dm": r["dm"],
                            f"{stage}_p": r["p"], f"{stage}_kappa": r["kappa"]})
            inj_rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"[inject] {i+1}/69 cells done ({time.time()-t0:.1f}s)")
    inj = pd.DataFrame(inj_rows)

    # PRE-DECLARED Holm families: one 69-cell family per (stage, target) = 9 families
    for tgt in TARGETS:
        m = inj.target_pct == tgt
        for stage in ("har", "firm", "pool"):
            inj.loc[m, f"{stage}_holm"] = fc.holm(inj.loc[m, f"{stage}_p"].fillna(1.0).values)
    for stage in ("har", "firm", "pool"):
        inj[f"{stage}_detect"] = (inj[f"{stage}_dm"] < 0) & (inj[f"{stage}_holm"] < 0.05)
    inj["all3_detect"] = inj.har_detect & inj.firm_detect & inj.pool_detect

    # ---------------- outputs ----------------
    out = inj.merge(base[GRIDKEY + ["n_seeds", "qR1", "se_daily",
                                    "har0_rel", "har0_dm", "har0_holm", "har0_detect",
                                    "firm0_rel", "firm0_dm", "firm0_holm", "firm0_detect",
                                    "pool0_rel", "pool0_dm", "pool0_holm", "pool0_detect"]],
                    on=GRIDKEY, validate="m:1")
    out.to_csv(T / "signal_injection_power.csv", index=False)

    write_md(base, inj, sanity, ci)

    n_conv = int(inj.converged.sum())
    print(f"\n=== signal_injection_power done in {time.time()-t0:.1f}s ===")
    print(f"calibration converged {n_conv}/{len(inj)} (max miss "
          f"{(inj.rel1_achieved - inj.target_pct).abs().max():.4f}pp)")
    for tgt in TARGETS:
        m = inj[inj.target_pct == tgt]
        print(f"target {tgt:.1f}%: recover HAR {int(m.har_detect.sum())}/69  "
              f"FIRM {int(m.firm_detect.sum())}/69  POOL {int(m.pool_detect.sum())}/69  "
              f"ALL3 {int(m.all3_detect.sum())}/69  (delta<0 in {int(m.delta_negative.sum())} cells)")
    print(f"MDE median {base.mde_rel_pct.median():.2f}%  IQR "
          f"[{base.mde_rel_pct.quantile(.25):.2f}, {base.mde_rel_pct.quantile(.75):.2f}]%")
    print("wrote results/tables/signal_injection_power.{csv,md}")


# ----------------------------------------------------------------------------- md
def fmt(x, p="+.2f"):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, p)


def write_md(base, inj, sanity, ci):
    n = len(base)
    med = base.mde_rel_pct.median()
    q25, q75 = base.mde_rel_pct.quantile(.25), base.mde_rel_pct.quantile(.75)
    md = []
    md.append("# ROW 1 — Signal-injection power calibration + per-cell MDE for the "
              "0/69 cascade headline\n")
    md.append(
        "> **ORACLE INJECTION — POWER CALIBRATION, NOT A FORECAST.** The synthetic text "
        "forecast `f_synth = exp(log f_text + delta*s)` uses **test labels by design** "
        "(s is the within-firm-demeaned test log-residual of the recalibrated HAR "
        "reference). This is the ONE declared exception to the no-look-ahead rule in the "
        "round-3 remediation plan; it calibrates the cascade's detection power and may "
        "never be cited as forecasting performance. All combiner/reference weights remain "
        "genuine validation-only fits on REAL text (the oracle content enters only "
        "through the test-side text array).\n")
    md.append("## RESTATED vs BEFORE\n")
    md.append(
        "| | BEFORE (committed) | RESTATED (this table) |\n|---|---|---|\n"
        "| power of the cascade | never calibrated — panel CRITICAL: \"0/69 is not "
        "interpretable as evidence of absence\" | recovery rates measured at known "
        "injected firm-orthogonal signal of 0.3/0.5/1.0% rel-QLIKE + per-cell MDE |\n"
        f"| delta=0 baseline (real data) | HAR {int(ci.primary_holm.sum())}/69, firm "
        f"{int(ci.firm_holm.sum())}/69, pool {int(ci.maximal_holm.sum())}/69, full AND "
        f"{int(ci.AND_full_holm.sum())}/69 (control_intersection_ensemble) | reproduced "
        f"exactly: {sanity['D_delta0_holm_counts_this_run'][0]}/"
        f"{sanity['D_delta0_holm_counts_this_run'][1]}/"
        f"{sanity['D_delta0_holm_counts_this_run'][2]}, see SANITY |\n")
    md.append(
        "## Design\n\n"
        "Grid and text object = the declared primary (`m1_ensemble_primary.py`: 69 cells, "
        "per-observation 3-seed-ensemble text; A/B, C6, D4 single-run). Injected signal "
        "`s` = test-split log-residual of the single recalibrated-HAR reference, demeaned "
        "WITHIN FIRM on the test split — so a firm-level identity regressor cannot "
        "mechanically absorb it (verified: max within-firm |mean s| < 1e-12). Per cell, "
        "`delta` is bisected (on kappa = g1*delta, tolerance 0.02pp) so the realised test "
        "rel-QLIKE improvement of the stage-HAR combined forecast hits the target "
        "exactly; cells whose REAL improvement already exceeds the target receive "
        "delta<0 (signal REMOVED down to the target) — the design equalises the realised "
        "effect at exactly X% in every cell, making each recovery rate a power estimate "
        "at that effect size. The same `f_synth` then runs through the firm-identity "
        "reference (val-window firm-mean spec) and the maximal 5-price pool with their "
        "own REAL validation-fit text loadings (`kappa_stage = g_stage*delta` reported "
        "per cell): the cascade is stress-tested exactly as deployed. Stage FIRM/POOL "
        "run on the 5-price inner-join panel (test rows a subset of the HAR-stage "
        "panel; n_test2 reported per cell). No subsampling anywhere. **Unit convention:** "
        "every QLIKE in this table is in VOLATILITY units, q(y, f) on realised vol — the "
        "same convention as the committed cascade tables it gates against "
        "(`vol_qlike_R` / `qlike_Rfirm` / `qlike_Rstar`); the variance-unit convention "
        "q(y^2, f^2) is treated separately in the variance-unit remediation line.\n")
    md.append(
        "## PRE-DECLARED Holm families\n\n"
        "Nine families, declared before any result was inspected: for each injection "
        "target level in {0.3%, 0.5%, 1.0%} and each cascade stage in {single "
        "recalibrated HAR, firm-identity-augmented reference, maximal 5-price pool}, one "
        "family = the 69 day-clustered DM p-values of that (stage, level) grid. "
        "Detection = clustered DM < 0 AND Holm-adjusted p < .05 within the family "
        "(same 'detected' criterion as the committed cascade tables; the placebo gate is "
        "not part of the detection criterion here, matching the row-1 brief).\n")
    # ---------------- SANITY ----------------
    md.append("## SANITY\n")
    md.append(
        f"All gates enforced in-script; the run aborts before writing any table if one "
        f"fails. Status: **{'PASS' if sanity['pass'] else 'FAIL'}**.\n\n"
        "| gate | committed table reproduced at delta=0 | max abs diff | verdict |\n"
        "|---|---|---|---|\n"
        f"| A | m1_ensemble_primary.csv `vol_qlike_R` | "
        f"{sanity['A_max_diff_qlike_R_vs_m1_ensemble_primary']:.2e} | "
        f"{'PASS' if sanity['A_max_diff_qlike_R_vs_m1_ensemble_primary'] < GATE_TOL else 'FAIL'} |\n"
        f"| A | m1_ensemble_primary.csv `vol_rel_impr_pct` | "
        f"{sanity['A_max_diff_rel_pct_vs_m1_ensemble_primary']:.2e} | "
        f"{'PASS' if sanity['A_max_diff_rel_pct_vs_m1_ensemble_primary'] < GATE_TOL else 'FAIL'} |\n"
        f"| B | firm_identity_ensemble.csv `qlike_Rfirm` / `rel_impr_pct_firm` / `dm` | "
        f"{max(sanity['B_max_diff_qlike_Rfirm_vs_firm_identity_ensemble'], sanity['B_max_diff_rel_firm_vs_firm_identity_ensemble'], sanity['B_max_diff_dm_firm']):.2e} | "
        f"{'PASS' if max(sanity['B_max_diff_qlike_Rfirm_vs_firm_identity_ensemble'], sanity['B_max_diff_rel_firm_vs_firm_identity_ensemble'], sanity['B_max_diff_dm_firm']) < GATE_TOL else 'FAIL'} |\n"
        f"| C | maximal_reference_ensemble.csv `qlike_Rstar` / `rel_impr_pct_maximal` / `dm` | "
        f"{max(sanity['C_max_diff_qlike_Rstar_vs_maximal_reference_ensemble'], sanity['C_max_diff_rel_pool_vs_maximal_reference_ensemble'], sanity['C_max_diff_dm_pool']):.2e} | "
        f"{'PASS' if max(sanity['C_max_diff_qlike_Rstar_vs_maximal_reference_ensemble'], sanity['C_max_diff_rel_pool_vs_maximal_reference_ensemble'], sanity['C_max_diff_dm_pool']) < GATE_TOL else 'FAIL'} |\n"
        f"| D | control_intersection_ensemble Holm counts at delta=0 "
        f"(HAR/firm/pool) | {sanity['D_delta0_holm_counts_this_run']} == "
        f"{sanity['D_delta0_holm_counts_committed_ci']} | "
        f"{'PASS' if sanity['D_delta0_holm_counts_this_run'] == sanity['D_delta0_holm_counts_committed_ci'] else 'FAIL'} |\n"
        f"| D | injected s within-firm mean-zero | "
        f"{sanity['D_s_within_firm_max_absmean']:.2e} | "
        f"{'PASS' if sanity['D_s_within_firm_max_absmean'] < 1e-12 else 'FAIL'} |\n")
    n_conv = int(inj.converged.sum())
    md.append(
        f"Calibration: {n_conv}/{len(inj)} (cell, level) pairs converged within 0.02pp "
        f"(max |achieved - target| = {(inj.rel1_achieved - inj.target_pct).abs().max():.4f}pp).\n")

    # ---------------- headline ----------------
    md.append("## HEADLINE — cascade recovery of a known firm-orthogonal signal\n")
    md.append("| injected level (realised rel-QLIKE of f_U over f_R) | HAR stage "
              "(DM<0 & Holm<.05) | firm-identity stage | maximal-pool stage | "
              "full conjunction (all 3) | cells with delta<0 (signal removed to target) |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| 0 (real data) | {int(base.har0_detect.sum())}/69 | "
              f"{int(base.firm0_detect.sum())}/69 | {int(base.pool0_detect.sum())}/69 | "
              f"{int((base.har0_detect & base.firm0_detect & base.pool0_detect).sum())}/69 | — |")
    for tgt in TARGETS:
        m = inj[inj.target_pct == tgt]
        md.append(f"| {tgt:.1f}% | **{int(m.har_detect.sum())}/69** | "
                  f"**{int(m.firm_detect.sum())}/69** | **{int(m.pool_detect.sum())}/69** | "
                  f"**{int(m.all3_detect.sum())}/69** | {int(m.delta_negative.sum())} |")
    heads = []
    for tgt in TARGETS:
        m = inj[inj.target_pct == tgt]
        heads.append(f"at {tgt:.1f}% injected signal the cascade recovers "
                     f"{int(m.har_detect.sum())}/69 (HAR) / {int(m.firm_detect.sum())}/69 "
                     f"(firm) / {int(m.pool_detect.sum())}/69 (pool), full conjunction "
                     f"{int(m.all3_detect.sum())}/69")
    md.append("\n**Headline:** " + "; ".join(heads) + ".\n")

    # ---------------- MDE ----------------
    md.append("## Per-cell MDE (80% power, 5% two-sided size)\n")
    md.append("`MDE_rel% = (1.96+0.84) * SE_daily / mean(QLIKE_R) * 100`, with SE_daily "
              "the day-clustered (HAC lag = h-1 days) standard error of the mean daily "
              "loss differential of the REAL (delta=0) stage-HAR comparison; "
              "denominator = per-observation mean QLIKE of f_R (= `vol_qlike_R`).\n")
    md.append("| disclosure | h | n cells | median MDE_rel% | IQR | min | max |")
    md.append("|---|---|---|---|---|---|---|")
    for disc in fc.SETS:
        for h in HORIZONS:
            gsl = base[(base.disc == disc) & (base.h == h)].mde_rel_pct
            md.append(f"| {disc} | {h} | {len(gsl)} | {gsl.median():.2f} | "
                      f"[{gsl.quantile(.25):.2f}, {gsl.quantile(.75):.2f}] | "
                      f"{gsl.min():.2f} | {gsl.max():.2f} |")
    md.append(f"| **all** | — | {n} | **{med:.2f}** | [{q25:.2f}, {q75:.2f}] | "
              f"{base.mde_rel_pct.min():.2f} | {base.mde_rel_pct.max():.2f} |\n")

    obs_ge = base[base.har0_rel >= base.mde_rel_pct]
    obs_lt = base[base.har0_rel < base.mde_rel_pct]
    det = base[base.har0_detect]
    det_below = det[det.har0_rel < det.mde_rel_pct]
    md.append(
        f"**Observed effects vs detectability.** The real (delta=0) stage-HAR effects "
        f"span {base.har0_rel.min():+.2f}% to {base.har0_rel.max():+.2f}% "
        f"(the {int(base.har0_detect.sum())} Holm-detected cells span "
        f"{det.har0_rel.min():+.2f}% to {det.har0_rel.max():+.2f}%). "
        f"**{len(obs_ge)}/{n}** cells have an observed effect at or above their own MDE "
        f"(above detectability); **{len(obs_lt)}/{n}** sit below it. Of the "
        f"{int(base.har0_detect.sum())} detected cells, {len(det_below)} lie below their "
        f"prospective MDE (detected despite <80% ex-ante power — expected, since MDE is "
        f"an 80%-power threshold, not a significance bound). Cells powered (MDE <= "
        f"target) for each injected level: 0.3% -> "
        f"{int((base.mde_rel_pct <= 0.3).sum())}/69, 0.5% -> "
        f"{int((base.mde_rel_pct <= 0.5).sum())}/69, 1.0% -> "
        f"{int((base.mde_rel_pct <= 1.0).sum())}/69 — compare these analytic counts with "
        f"the empirical HAR-stage recovery rates above.\n")

    # ---------------- per-cell tables ----------------
    md.append("## Per-cell detail — real data (delta=0): observed effect vs MDE\n")
    md.append("| disc | model | h | n_days | rel%(real) | MDE_rel% | above MDE? | "
              "Holm-detected (HAR/firm/pool) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in base.sort_values(GRIDKEY).iterrows():
        md.append(f"| {r.disc} | {r.model} | {r.h} | {int(r.n_days)} | "
                  f"{fmt(r.har0_rel)} | {r.mde_rel_pct:.2f} | "
                  f"{'YES' if r.har0_rel >= r.mde_rel_pct else 'no'} | "
                  f"{'Y' if r.har0_detect else '.'}/{'Y' if r.firm0_detect else '.'}/"
                  f"{'Y' if r.pool0_detect else '.'} |")
    for tgt in TARGETS:
        m = inj[inj.target_pct == tgt]
        md.append(f"\n## Per-cell detail — injected level {tgt:.1f}%\n")
        md.append("| disc | model | h | delta | kappa(HAR) | kappa(firm) | kappa(pool) | "
                  "rel%(HAR) | DM(HAR) | Holm | rel%(firm) | DM(firm) | Holm | "
                  "rel%(pool) | DM(pool) | Holm | detect H/F/P |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in m.sort_values(GRIDKEY).iterrows():
            md.append(
                f"| {r.disc} | {r.model} | {r.h} | {fmt(r.delta, '+.3f')} | "
                f"{fmt(r.kappa1, '+.4f')} | {fmt(r.firm_kappa, '+.4f')} | "
                f"{fmt(r.pool_kappa, '+.4f')} | {fmt(r.har_rel)} | {fmt(r.har_dm)} | "
                f"{fmt(r.har_holm, '.3f')} | {fmt(r.firm_rel)} | {fmt(r.firm_dm)} | "
                f"{fmt(r.firm_holm, '.3f')} | {fmt(r.pool_rel)} | {fmt(r.pool_dm)} | "
                f"{fmt(r.pool_holm, '.3f')} | "
                f"{'Y' if r.har_detect else '.'}/{'Y' if r.firm_detect else '.'}/"
                f"{'Y' if r.pool_detect else '.'} |")

    # ---------------- caveats + bottom line ----------------
    md.append(
        "\n## Caveats (read before citing)\n\n"
        "1. **Oracle**: s is built from test labels; the calibrated deltas quantify "
        "detection power only, never achievable forecast gains.\n"
        "2. **Transmission**: one f_synth per (cell, level), calibrated on the HAR "
        "stage; the firm/pool stages receive it through their own REAL validation-fit "
        "text loadings (kappa_stage = g_stage*delta, tabulated). A stage can therefore "
        "miss an injected signal either through statistical noise or through a small/"
        "opposite-signed deployed loading — both are properties of the cascade as "
        "actually run, which is what this calibration measures.\n"
        "3. **delta<0 cells**: where the real effect already exceeds the target, signal "
        "is removed down to the target so every cell realises exactly X%; the recovery "
        "rate is then a clean power estimate at that effect size.\n"
        "4. Stage FIRM/POOL evaluate on the 5-price inner-join panel (n_test2 <= n_test); "
        "the within-firm demeaning is exact on the HAR-stage test panel and carries over "
        "to the subset up to dropped rows (the firm reference is a single global "
        "loading on a firm-level regressor, so exact per-firm zero mean is not required "
        "for non-absorption).\n"
        "5. MDE uses the normal-approximation (1.96+0.84) factor on the HAC daily SE; "
        "it is an 80%-power planning quantity, not a test.\n")
    m03, m05, m10 = (inj[inj.target_pct == t] for t in TARGETS)
    md.append(
        "## Bottom line\n\n"
        f"- {heads[0]}; {heads[1]}; {heads[2]}.\n"
        f"- Median per-cell MDE is {med:.2f}% rel-QLIKE (IQR [{q25:.2f}, {q75:.2f}]); "
        f"{int((base.mde_rel_pct <= 0.3).sum())}/69 cells are 80%-powered at 0.3%, "
        f"{int((base.mde_rel_pct <= 0.5).sum())}/69 at 0.5%, "
        f"{int((base.mde_rel_pct <= 1.0).sum())}/69 at 1.0%.\n"
        f"- Interpretation for the 0/69 headline: a genuinely firm-orthogonal signal of "
        f"1.0% would have been flagged by the HAR stage in {int(m10.har_detect.sum())}/69 "
        f"cells and survived the full conjunction in {int(m10.all3_detect.sum())}/69; at "
        f"0.3% the corresponding counts are {int(m03.har_detect.sum())}/69 and "
        f"{int(m03.all3_detect.sum())}/69. Observed real effects "
        f"({base.har0_rel.min():+.2f}% to {base.har0_rel.max():+.2f}%) must be read "
        f"against the per-cell MDE table above: effects below their cell's MDE were "
        f"never detectable at 80% power, and the conjunction's specificity (placebo-"
        f"validated) is now complemented by measured sensitivity.\n")
    (T / "signal_injection_power.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
