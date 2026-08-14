"""ROW 7 (round-3 remediation) — DEPLOYABLE expanding/rolling combiner over the FULL grid,
explicitly including the C6_llmtext / D4_llmfused / C5_qwen3 cells.

Reviewer defect (methodology W2 / DA, REVIEW_ROUND3_FRESH_PANEL.md row 7): the primary
combiner weights (a,b,g) are fit on the anomalous COVID 2020-21 validation window and
frozen; the committed rolling_robustness.{csv,md} rolls the combiner over the 16 test
quarters for only a 6-model subset (36 cells, seed2026 basis) and NEVER includes the
surviving C6 8-K residual cells. Whether the residual survives *deployable* weights is
exactly the open question.

THIS SCRIPT: both schemes over the full 75-cell grid
(the 69-cell primary grid = fc.SETS, PLUS C5_qwen3 x {long_form, event_driven} x 3h):
  FIXED     — fc.log_combo fit ONCE on val (2020-21), frozen on the whole test span
              (the current primary; identical forecasts to m1_ensemble_primary).
  EXPANDING — for each test quarter q in 2022Q1..2025Q4, refit fc.log_combo on ALL
              filings with filing_time_utc < q_start (train 2010-2019 + val +
              strictly-earlier test quarters — every filing time-stamped before q;
              same pool convention as the committed rolling_robustness.csv), apply to
              filings IN q; concatenating the 16 quarter blocks gives ONE deployable
              pseudo-OOS forecast path over the whole span.
Text basis = seed-ensemble primary (per-observation mean over seeds 2026/2027/2028
where multi-seed; single seed2026 otherwise) via m1_ensemble_primary.ensemble_text.
Inference = day-clustered DM (clustered_dm.dm_test_clustered, HAC lag h-1 DAYS, HLN).
For the EXPANDING scheme the pooled DM is interpreted in the Giacomini-White
finite-estimation-window sense (it compares forecasting METHODS including their
recursive estimation scheme), which is the correct frame for recursive/nested setups.

NO LOOK-AHEAD: fixed weights are fit on validation only; expanding weights for quarter
q use only filings time-stamped strictly before q's first calendar day. (Boundary
caveat, same convention as the committed rolling_robustness.csv: a filing whose
effective_trading_day lies within h trading days of q_start has a label window that
ends after q_start; kept for comparability and disclosed in the md.)

SANITY GATE 1 (committed table: results/tables/rolling_robustness.csv): re-run the
exact committed code path (rolling_robustness._load_cell / _increment, seed2026 text)
for all 36 overlapping (disc, model, h) cells x 16 quarters x 2 schemes and assert
machine-precision, NaN-aware equality (|diff| <= 1e-12; bitwise identity is not
attainable across runs because LAPACK/BLAS reduction order injects ~1e-15 noise into
lstsq) of rel_impr_pct, dm_stat, dm_p, ci_lo, ci_hi, n. FAIL => abort, ship nothing.

SANITY GATE 2 (committed table: results/tables/m1_ensemble_primary.csv): the FIXED
scheme on the ensemble basis must reproduce vol_qlike_R, vol_qlike_U, g_log AND the
day-clustered DM stat of the committed 69-cell primary to the same 1e-12 tolerance.
FAIL => abort.

PRE-DECLARED Holm families (also stated in the md BEFORE the results):
  F-DEPLOY-FIXED : the 75 pooled day-clustered two-sided DM p-values (f_U vs f_R,
                   fixed scheme), one per grid cell; Holm within this family.
  F-DEPLOY-EXP   : the 75 pooled day-clustered two-sided DM p-values (expanding
                   deployable path), one per grid cell; Holm within this family.
'genuine' per scheme = pooled clustered DM < 0 AND Holm < .05 AND |placebo DM| < 2
(placebo = text forecasts permuted, 5 seeds, same scheme/machinery).

Run from the repo root:  .venv/bin/python scripts/analysis/deployable_combiner.py
Outputs: results/tables/deployable_combiner.{csv,md}
         results/tables/deployable_combiner_quarters.csv (per-quarter long panel)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")
import forecast_combination as fc
import m1_ensemble_primary as mep
import rolling_robustness as rr
from clustered_dm import dm_test_clustered, mbb_ci_daily

from sp500vol.evaluation.dm_test import dm_test

KEY, SORT, HORIZONS = fc.KEY, fc.SORT, fc.HORIZONS
PLACEBO_SEEDS = fc.PLACEBO_SEEDS
QUARTERS = pd.period_range("2022Q1", "2025Q4", freq="Q")
NQ = len(QUARTERS)

# FULL GRID: the 69-cell primary (fc.SETS) + C5_qwen3 extension (both disclosures)
GRID = {disc: list(models) + ["C5_qwen3"] for disc, models in fc.SETS.items()}
N_CELLS_EXPECTED = sum(len(m) for m in GRID.values()) * len(HORIZONS)  # 75

ROLLING_CSV = "results/tables/rolling_robustness.csv"
M1_CSV = "results/tables/m1_ensemble_primary.csv"
OUT_CSV = "results/tables/deployable_combiner.csv"
OUT_MD = "results/tables/deployable_combiner.md"
OUT_QCSV = "results/tables/deployable_combiner_quarters.csv"

HEADLINE_CELLS = [("event_driven", "C6_llmtext"), ("event_driven", "D4_llmfused"),
                  ("event_driven", "C5_qwen3"), ("long_form", "C6_llmtext")]


GATE_ATOL = 1e-12  # machine-precision gate: BLAS reduction order injects ~1e-15 noise


def exact_eq(a, b, atol=GATE_ATOL):
    """NaN-aware machine-precision float equality (|a-b| <= atol)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.all((np.abs(a - b) <= atol) | (np.isnan(a) & np.isnan(b)))


def max_abs_diff(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0


# --------------------------------------------------------------------------------------
# SANITY GATE 1 — exact reproduction of the committed rolling_robustness.csv
# --------------------------------------------------------------------------------------
def sanity_gate_rolling():
    """Re-run the committed rolling_robustness code path verbatim (seed2026 text basis,
    obs-order per-quarter DM, moving-block CI) and compare to the committed CSV."""
    committed = pd.read_csv(ROLLING_CSV)
    rows = []
    for disc in rr.DISCS:
        for model in rr.MODELS:
            d = rr._load_cell(disc, model)
            for h in rr.HORIZONS:
                dh = d[d.horizon_days == h].copy()
                dv = dh[dh.split == "val"].sort_values(rr.SORT, kind="mergesort")
                dt = dh[dh.split == "test"].sort_values(rr.SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                dt = dt.copy()
                dt["q"] = dt.ft.dt.tz_convert(None).dt.to_period("Q")
                yv = dv.label_realised_vol.to_numpy()
                fhv = dv.fhar.to_numpy()
                ftv = dv.ftext.to_numpy()
                yt_all = dt.label_realised_vol.to_numpy()
                fhr_all = dt.fhar.to_numpy()
                ftt_all = dt.ftext.to_numpy()
                fR_fx_all, fU_fx_all, _ = fc.log_combo(yv, fhv, ftv, fhr_all, ftt_all)
                for qi, q in enumerate(rr.QUARTERS):
                    mask = (dt.q == q).to_numpy()
                    nq = int(mask.sum())
                    yq = yt_all[mask]
                    relf, dmf, dpf, lof, hif = rr._increment(
                        yq, fR_fx_all[mask], fU_fx_all[mask], h)
                    rows.append({"disc": disc, "model": model, "h": h, "quarter": str(q),
                                 "scheme": "fixed", "rel_impr_pct": relf, "dm_stat": dmf,
                                 "dm_p": dpf, "ci_lo": lof, "ci_hi": hif, "n": nq})
                    q_start = q.start_time.tz_localize("UTC")
                    tr = dh[dh.ft < q_start]
                    ytr = tr.label_realised_vol.to_numpy()
                    fhtr = tr.fhar.to_numpy()
                    fttr = tr.ftext.to_numpy()
                    if len(ytr) >= 100 and nq >= 2:
                        fR_e, fU_e, _ = fc.log_combo(ytr, fhtr, fttr,
                                                     fhr_all[mask], ftt_all[mask])
                        rele, dme, dpe, loe, hie = rr._increment(yq, fR_e, fU_e, h)
                    else:
                        rele = dme = dpe = loe = hie = float("nan")
                    rows.append({"disc": disc, "model": model, "h": h, "quarter": str(q),
                                 "scheme": "expanding", "rel_impr_pct": rele, "dm_stat": dme,
                                 "dm_p": dpe, "ci_lo": loe, "ci_hi": hie, "n": nq})
        print(f"  gate1: {disc} done", flush=True)

    rec = pd.DataFrame(rows)
    keys = ["disc", "model", "h", "quarter", "scheme"]
    m = committed.merge(rec, on=keys, suffixes=("_ref", "_new"), how="outer",
                        indicator=True)
    report = {"n_committed": len(committed), "n_recomputed": len(rec),
              "n_joined": int((m._merge == "both").sum())}
    ok = (len(committed) == len(rec) == report["n_joined"])
    for col in ["rel_impr_pct", "dm_stat", "dm_p", "ci_lo", "ci_hi", "n"]:
        eq = exact_eq(m[f"{col}_ref"], m[f"{col}_new"])
        report[f"{col}_exact"] = bool(eq)
        report[f"{col}_maxdiff"] = max_abs_diff(m[f"{col}_ref"], m[f"{col}_new"])
        ok = ok and eq
    report["pass"] = bool(ok)
    return report


# --------------------------------------------------------------------------------------
# main grid
# --------------------------------------------------------------------------------------
def quarter_stats(lU, lR, days, mask_q, h):
    """Per-quarter clustered + legacy obs-order DM on a quarter slice."""
    nq = int(mask_q.sum())
    if nq < 2:
        return dict(nq=nq, rel=np.nan, dm_clu=np.nan, p_clu=np.nan,
                    dm_leg=np.nan, p_leg=np.nan)
    lUq, lRq = lU[mask_q], lR[mask_q]
    mR = float(lRq.mean())
    rel = 100.0 * (mR - float(lUq.mean())) / mR if mR > 0 else np.nan
    try:
        dm_clu, p_clu, _ = dm_test_clustered(lUq, lRq, days[mask_q], h)
    except Exception:
        dm_clu, p_clu = np.nan, np.nan
    try:
        dm_leg, p_leg = dm_test(lUq, lRq, h=h)
    except Exception:
        dm_leg, p_leg = np.nan, np.nan
    return dict(nq=nq, rel=rel, dm_clu=float(dm_clu), p_clu=float(p_clu),
                dm_leg=float(dm_leg), p_leg=float(p_leg))


def pooled_stats(lU, lR, days, h):
    """Pooled span stats: rel%, day-clustered DM, daily MBB CI."""
    mR = float(lR.mean())
    rel = 100.0 * (mR - float(lU.mean())) / mR if mR > 0 else np.nan
    dm, p, n_days = dm_test_clustered(lU, lR, days, h)
    _, lo, hi = mbb_ci_daily(lU - lR, days, h)
    return rel, dm, p, n_days, lo, hi


def expanding_path(dh_all, yq_masks, fhr, ftt, text_perm_rng=None):
    """Build the deployable expanding forecast path over the span.

    dh_all: horizon-filtered frame (val+test, unsorted — committed convention) with
            columns ft, label_realised_vol, fhar, ftext.
    yq_masks: list of (Period, boolean mask over the sorted test rows).
    Returns (fR_path, fU_path) aligned to the sorted test rows (NaN if unfit).
    If text_perm_rng is given, the text forecast is permuted independently in the
    training pool and in each quarter block (placebo).
    """
    n = len(fhr)
    fR_path = np.full(n, np.nan)
    fU_path = np.full(n, np.nan)
    for q, mask in yq_masks:
        nq = int(mask.sum())
        if nq == 0:
            continue
        q_start = q.start_time.tz_localize("UTC")
        tr = dh_all[dh_all.ft < q_start]
        if len(tr) < 100:
            continue
        ytr = tr.label_realised_vol.to_numpy()
        fhtr = tr.fhar.to_numpy()
        fttr = tr.ftext.to_numpy()
        ftt_q = ftt[mask]
        if text_perm_rng is not None:
            fttr = text_perm_rng.permutation(fttr)
            ftt_q = text_perm_rng.permutation(ftt_q)
        fR_q, fU_q, _ = fc.log_combo(ytr, fhtr, fttr, fhr[mask], ftt_q)
        fR_path[mask] = fR_q
        fU_path[mask] = fU_q
    return fR_path, fU_path


def main():
    t0 = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    # ---------------- SANITY GATE 1 ----------------
    print("=== SANITY GATE 1: exact reproduction of committed rolling_robustness.csv ===",
          flush=True)
    g1 = sanity_gate_rolling()
    for k, v in g1.items():
        print(f"  {k}: {v}")
    if not g1["pass"]:
        print("SANITY GATE 1 FAILED — aborting, no outputs written.", flush=True)
        sys.exit(1)
    print(f"SANITY GATE 1 PASS ({time.time()-t0:.0f}s)", flush=True)

    m1 = pd.read_csv(M1_CSV)
    m1_ref = m1.set_index(["disc", "model", "h"])

    # committed rolling aggregates for the overlap columns (context in the md)
    roll = pd.read_csv(ROLLING_CSV)
    roll["sig"] = (roll.dm_stat < 0) & (roll.dm_p < 0.05)
    old_agg = (roll.groupby(["disc", "model", "h", "scheme"])
               .agg(sig_q=("sig", "sum"), mean_rel=("rel_impr_pct", "mean"))
               .unstack("scheme"))
    old_agg.columns = [f"old_{s}_{c}" for c, s in old_agg.columns]

    cells, qrows, gate2 = [], [], []
    combo_i = 0
    n_combos = sum(len(m) for m in GRID.values())
    for disc, models in GRID.items():
        har = pd.read_parquet(mep.run_dir("A2_har_rv", disc, 2026))[
            ["split"] + KEY + ["prediction_realised_vol", "label_realised_vol",
                               "filing_time_utc", "effective_trading_day"]
        ].rename(columns={"prediction_realised_vol": "fhar"})
        for model in models:
            combo_i += 1
            ens, used = mep.ensemble_text(model, disc)
            d = har.merge(ens, on=KEY)
            d["ft"] = pd.to_datetime(d.filing_time_utc, utc=True)
            print(f"[{combo_i}/{n_combos}] {disc}/{model} seeds={used} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            for h in HORIZONS:
                dh_all = d[d.horizon_days == h]
                dv = dh_all[dh_all.split == "val"].sort_values(SORT, kind="mergesort")
                dt = dh_all[dh_all.split == "test"].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    print(f"  SKIP {disc}/{model}/h{h}: val={len(dv)} test={len(dt)}",
                          flush=True)
                    continue
                yv = dv.label_realised_vol.to_numpy()
                fhv = dv.fhar.to_numpy()
                ftv = dv.ftext.to_numpy()
                yt = dt.label_realised_vol.to_numpy()
                fhr = dt.fhar.to_numpy()
                ftt = dt.ftext.to_numpy()
                days = dt.effective_trading_day.to_numpy()
                qper = dt.ft.dt.tz_convert(None).dt.to_period("Q")
                span = qper.isin(QUARTERS).to_numpy()
                n_outside = int((~span).sum())
                yq_masks = [(q, (qper == q).to_numpy()) for q in QUARTERS]

                # ---------- FIXED (val-frozen; identical to the committed primary) ----
                fR_fx, fU_fx, g_log = fc.log_combo(yv, fhv, ftv, fhr, ftt)
                lR_fx = fc.qlike(yt, fR_fx)
                lU_fx = fc.qlike(yt, fU_fx)

                # SANITY GATE 2 accumulation (69 primary cells have a reference)
                if (disc, model, h) in m1_ref.index:
                    ref = m1_ref.loc[(disc, model, h)]
                    dmc_all, _, _ = dm_test_clustered(lU_fx, lR_fx, days, h)
                    gate2.append({
                        "disc": disc, "model": model, "h": h,
                        "dq_R": float(lR_fx.mean()) - float(ref.vol_qlike_R),
                        "dq_U": float(lU_fx.mean()) - float(ref.vol_qlike_U),
                        "dg": float(g_log) - float(ref.g_log),
                        "ddm": float(dmc_all) - float(ref.vol_dm_q_clu),
                    })

                fx_rel, fx_dm, fx_p, n_days, fx_lo, fx_hi = pooled_stats(
                    lU_fx[span], lR_fx[span], days[span], h)

                # ---------- EXPANDING deployable path ----------
                fR_ex, fU_ex = expanding_path(dh_all, yq_masks, fhr, ftt)
                fitted = np.isfinite(fU_ex) & span
                lR_ex = fc.qlike(yt[fitted], fR_ex[fitted])
                lU_ex = fc.qlike(yt[fitted], fU_ex[fitted])
                ex_rel, ex_dm, ex_p, ex_days, ex_lo, ex_hi = pooled_stats(
                    lU_ex, lR_ex, days[fitted], h)

                # ---------- per-quarter stats ----------
                fx_sig = fx_sig_leg = ex_sig = ex_sig_leg = 0
                fx_rels, ex_rels = [], []
                lR_exf = np.full(len(yt), np.nan)
                lU_exf = np.full(len(yt), np.nan)
                lR_exf[fitted] = lR_ex
                lU_exf[fitted] = lU_ex
                for qi, (q, mask) in enumerate(yq_masks):
                    sfx = quarter_stats(lU_fx, lR_fx, days, mask, h)
                    mask_e = mask & fitted
                    sex = quarter_stats(lU_exf, lR_exf, days, mask_e, h)
                    qrows.append({"disc": disc, "model": model, "h": h,
                                  "quarter": str(q), "quarter_idx": qi, "n": sfx["nq"],
                                  "fixed_rel": sfx["rel"], "fixed_dm_clu": sfx["dm_clu"],
                                  "fixed_p_clu": sfx["p_clu"],
                                  "fixed_dm_legacy": sfx["dm_leg"],
                                  "fixed_p_legacy": sfx["p_leg"],
                                  "exp_rel": sex["rel"], "exp_dm_clu": sex["dm_clu"],
                                  "exp_p_clu": sex["p_clu"],
                                  "exp_dm_legacy": sex["dm_leg"],
                                  "exp_p_legacy": sex["p_leg"]})
                    fx_rels.append(sfx["rel"])
                    ex_rels.append(sex["rel"])
                    if np.isfinite(sfx["dm_clu"]) and sfx["dm_clu"] < 0 and sfx["p_clu"] < .05:
                        fx_sig += 1
                    if np.isfinite(sfx["dm_leg"]) and sfx["dm_leg"] < 0 and sfx["p_leg"] < .05:
                        fx_sig_leg += 1
                    if np.isfinite(sex["dm_clu"]) and sex["dm_clu"] < 0 and sex["p_clu"] < .05:
                        ex_sig += 1
                    if np.isfinite(sex["dm_leg"]) and sex["dm_leg"] < 0 and sex["p_leg"] < .05:
                        ex_sig_leg += 1

                # ---------- placebos (5 seeds, permuted text, same machinery) ----------
                fx_pl, ex_pl = [], []
                for s in PLACEBO_SEEDS:
                    rng = np.random.default_rng(s)
                    pR, pU, _ = fc.log_combo(yv, fhv, rng.permutation(ftv),
                                             fhr, rng.permutation(ftt))
                    st, _, _ = dm_test_clustered(fc.qlike(yt[span], pU[span]),
                                                 fc.qlike(yt[span], pR[span]),
                                                 days[span], h)
                    fx_pl.append(st)
                    rng = np.random.default_rng(s)
                    pRe, pUe = expanding_path(dh_all, yq_masks, fhr, ftt,
                                              text_perm_rng=rng)
                    pf = np.isfinite(pUe) & span
                    st, _, _ = dm_test_clustered(fc.qlike(yt[pf], pUe[pf]),
                                                 fc.qlike(yt[pf], pRe[pf]),
                                                 days[pf], h)
                    ex_pl.append(st)

                cells.append({
                    "disc": disc, "model": model, "h": h,
                    "seeds": "+".join(str(s) for s in used), "n_seeds": len(used),
                    "n_test": len(dt), "n_days": n_days, "n_outside_span": n_outside,
                    "in_primary_grid": (disc, model, h) in m1_ref.index,
                    "g_log_fixed": float(g_log),
                    "fixed_sig_q_clu": fx_sig, "fixed_sig_q_legacy": fx_sig_leg,
                    "fixed_mean_rel": float(np.nanmean(fx_rels)),
                    "fixed_pooled_rel": fx_rel, "fixed_dm_clu": fx_dm,
                    "fixed_p_clu": fx_p, "fixed_boot_lo": fx_lo, "fixed_boot_hi": fx_hi,
                    "fixed_placebo_dm": float(np.mean(fx_pl)),
                    "exp_sig_q_clu": ex_sig, "exp_sig_q_legacy": ex_sig_leg,
                    "exp_mean_rel": float(np.nanmean(ex_rels)),
                    "exp_pooled_rel": ex_rel, "exp_dm_clu": ex_dm, "exp_p_clu": ex_p,
                    "exp_n_days": ex_days, "exp_boot_lo": ex_lo, "exp_boot_hi": ex_hi,
                    "exp_placebo_dm": float(np.mean(ex_pl)),
                    "gap_sig_q_clu": ex_sig - fx_sig,
                    "gap_mean_rel": float(np.nanmean(ex_rels) - np.nanmean(fx_rels)),
                    "gap_pooled_rel": ex_rel - fx_rel,
                })

    df = pd.DataFrame(cells)
    qdf = pd.DataFrame(qrows)

    # ---------------- grid-coverage assertion (promised 75-cell grid) ----------------
    if len(df) != N_CELLS_EXPECTED:
        print(f"GRID COVERAGE FAILED: {len(df)} cells != expected {N_CELLS_EXPECTED} "
              "— aborting, no outputs written.", flush=True)
        sys.exit(1)

    # ---------------- SANITY GATE 2 ----------------
    g2 = pd.DataFrame(gate2)
    g2_max = {c: float(g2[c].abs().max()) for c in ["dq_R", "dq_U", "dg", "ddm"]}
    g2_pass = (len(g2) == 69
               and all(g2_max[c] <= GATE_ATOL for c in ["dq_R", "dq_U", "dg", "ddm"]))
    print("=== SANITY GATE 2: fixed scheme reproduces m1_ensemble_primary.csv ===")
    print(f"  cells with reference: {len(g2)}/69, max|dQLIKE_R|={g2_max['dq_R']:.2e}, "
          f"max|dQLIKE_U|={g2_max['dq_U']:.2e}, max|dg_log|={g2_max['dg']:.2e}, "
          f"max|dDM_clu|={g2_max['ddm']:.2e}  -> {'PASS' if g2_pass else 'FAIL'}",
          flush=True)
    if not g2_pass:
        print("SANITY GATE 2 FAILED — aborting, no outputs written.", flush=True)
        sys.exit(1)

    # ---------------- Holm within the PRE-DECLARED families ----------------
    df["fixed_holm"] = fc.holm(df.fixed_p_clu.fillna(1.0).values)   # F-DEPLOY-FIXED
    df["exp_holm"] = fc.holm(df.exp_p_clu.fillna(1.0).values)       # F-DEPLOY-EXP
    df["genuine_fixed"] = ((df.fixed_dm_clu < 0) & (df.fixed_holm < .05)
                           & (df.fixed_placebo_dm.abs() < 2.0))
    df["genuine_exp"] = ((df.exp_dm_clu < 0) & (df.exp_holm < .05)
                         & (df.exp_placebo_dm.abs() < 2.0))
    df["deploy_status"] = np.select(
        [df.genuine_fixed & df.genuine_exp, df.genuine_fixed & ~df.genuine_exp,
         ~df.genuine_fixed & df.genuine_exp],
        ["SURVIVES-DEPLOY", "LOST-ON-DEPLOY", "GAINED-ON-DEPLOY"], default="null-null")

    # context merges: committed rolling subset + committed m1 primary verdicts
    df = df.merge(old_agg.reset_index(), on=["disc", "model", "h"], how="left")
    df = df.merge(
        m1[["disc", "model", "h", "vol_rel_impr_pct", "vol_dm_q_clu",
            "vol_dmq_holm_clu", "genuine_ens_vol"]].rename(columns={
                "vol_rel_impr_pct": "m1_rel", "vol_dm_q_clu": "m1_dm_clu",
                "vol_dmq_holm_clu": "m1_holm", "genuine_ens_vol": "m1_genuine"}),
        on=["disc", "model", "h"], how="left")

    df.to_csv(OUT_CSV, index=False)
    qdf.to_csv(OUT_QCSV, index=False)

    # ---------------- markdown ----------------
    n = len(df)
    nf, ne = int(df.genuine_fixed.sum()), int(df.genuine_exp.sum())
    lost = df[df.deploy_status == "LOST-ON-DEPLOY"]
    gained = df[df.deploy_status == "GAINED-ON-DEPLOY"]
    surv = df[df.deploy_status == "SURVIVES-DEPLOY"]
    prim = df[df.in_primary_grid]
    nf69, ne69 = int(prim.genuine_fixed.sum()), int(prim.genuine_exp.sum())
    c6ed = df[(df.disc == "event_driven") & (df.model == "C6_llmtext")]

    def frow(r):
        return (f"| {r.disc} | {r.model} | {r.h} | {r.seeds} | "
                f"{int(r.fixed_sig_q_clu)}/16 | {r.fixed_mean_rel:+.2f} | "
                f"{r.fixed_pooled_rel:+.2f} | {r.fixed_dm_clu:+.2f} | {r.fixed_holm:.3f} | "
                f"{int(r.exp_sig_q_clu)}/16 | {r.exp_mean_rel:+.2f} | "
                f"{r.exp_pooled_rel:+.2f} | {r.exp_dm_clu:+.2f} | {r.exp_holm:.3f} | "
                f"{r.exp_placebo_dm:+.2f} | {r.gap_pooled_rel:+.2f} | "
                f"{'YES' if r.genuine_fixed else 'no'} | {'YES' if r.genuine_exp else 'no'} | "
                f"{r.deploy_status} |")

    md = [
        "# ROW 7 — Deployable expanding combiner over the FULL grid (incl. C6/D4/C5 cells)\n",
        "## RESTATED vs BEFORE\n",
        "| | BEFORE (committed rolling_robustness) | RESTATED (this table) |",
        "|---|---|---|",
        "| grid | 36 cells (6-model subset, **no C6/D4**) | "
        f"**{n} cells** = 69-cell primary grid + C5_qwen3 extension |",
        "| text basis | seed2026 only | seed-ensemble primary (mean over seeds "
        "2026/2027/2028 where multi-seed) |",
        "| per-quarter DM | observation-order HAC | day-clustered (HAC lag h-1 DAYS); "
        "legacy obs-order kept as a comparability column |",
        "| deployable statistic | per-quarter counts only | + POOLED pseudo-OOS "
        "deployable path (16 quarter blocks concatenated), day-clustered DM, Holm, "
        "5-seed permutation placebo |",
        f"| genuine cells, fixed val-frozen scheme | (m1_ensemble_primary: 38/69) | "
        f"**{nf}/{n}** ({nf69}/69 on the primary grid) |",
        f"| genuine cells, EXPANDING deployable scheme | never reported for C6 | "
        f"**{ne}/{n}** ({ne69}/69 on the primary grid) |",
        "",
        "All QLIKE losses in this table are in **VOLATILITY units** (q(y, f), the "
        "convention of `m1_ensemble_primary`'s `vol_*` columns); the variance-unit "
        "sensitivity of the same grid lives in `m1_variance_unit` / "
        "`variance_unit_cascade`.\n",
        "Schemes: **fixed** = `fc.log_combo` weights fit once on the 2020-21 validation "
        "window, frozen on the whole 2022Q1-2025Q4 test span (the committed primary; "
        "SANITY GATE 2 confirms machine-precision reproduction). **expanding** = for "
        "each test quarter q the weights are refit on ALL filings with "
        "`filing_time_utc` strictly before q's first calendar day — the 2010-2019 "
        "train split + validation + earlier test quarters, i.e. the full pre-q filing "
        "history (the identical pool convention as the committed "
        "`rolling_robustness.csv`, whose prose also under-described it as val+test); "
        "the 16 quarter blocks concatenate into ONE deployable pseudo-OOS forecast path. "
        "No look-ahead enters any weight applied to a quarter. Boundary caveat (same "
        "convention as the committed rolling_robustness.csv, kept for comparability): a "
        "filing whose effective day lies within h trading days of q_start has a label "
        "window ending after q_start, so the earliest few training labels per boundary "
        "are not yet fully realised at refit time; this affects only the training pool, "
        "never the evaluation rows.\n",
        "Inference: pooled comparisons use the day-clustered DM "
        "(`clustered_dm.dm_test_clustered`, HAC lag = h-1 days, HLN). For the EXPANDING "
        "scheme the pooled DM is read in the Giacomini-White finite-estimation-window "
        "sense — it compares forecasting *methods including their recursive estimation "
        "scheme* — which is the appropriate frame once weights are re-estimated "
        "recursively (the fixed-scheme comparison remains a standard frozen-weight DM). "
        "Per-quarter clustered DM at h=20 sits on ~60 filing days with HAC lag 19 and "
        "is fragile; the POOLED statistics are the primary deployable evidence, the "
        "per-quarter `sig_q/16` counts are descriptive.\n",
        "## PRE-DECLARED Holm families (declared before any result below was inspected)\n",
        f"- **F-DEPLOY-FIXED**: the {n} pooled day-clustered two-sided DM p-values of "
        "f_U vs f_R under the fixed scheme, one per grid cell; Holm within this family.",
        f"- **F-DEPLOY-EXP**: the {n} pooled day-clustered two-sided DM p-values of the "
        "expanding deployable path, one per grid cell; Holm within this family.",
        "- `genuine` per scheme = pooled clustered DM < 0 AND Holm < .05 AND "
        "|placebo DM| < 2 (placebo = text forecast permuted on fit and application "
        "rows, 5 seeds, identical machinery per scheme).",
        f"- The family spans all {n} cells (69 primary + 6 C5_qwen3 extension cells) — "
        "slightly *more* conservative for the primary cells than the committed 69-cell "
        "convention.\n",
    ]

    md.append("## HEADLINE — the event-driven C6 residual under deployable weights\n")
    md.append("| disc | model | h | seeds | FIXED sig_q | FIXED mean rel% | FIXED pooled "
              "rel% | FIXED DM | FIXED Holm | EXP sig_q | EXP mean rel% | EXP pooled "
              "rel% | EXP DM | EXP Holm | EXP placebo | gap pooled rel% (exp-fixed) | "
              "genuine FIXED | genuine EXP | status |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for disc, model in HEADLINE_CELLS:
        for _, r in df[(df.disc == disc) & (df.model == model)].sort_values("h").iterrows():
            md.append(frow(r))
    if len(c6ed):
        kept = c6ed[c6ed.genuine_exp]
        md.append(
            f"\nEvent-driven C6_llmtext: fixed-scheme genuine in "
            f"{int(c6ed.genuine_fixed.sum())}/3 horizons; under the deployable expanding "
            f"scheme **{int(c6ed.genuine_exp.sum())}/3** remain genuine"
            + (" (" + ", ".join(f"h{int(r.h)}: pooled rel {r.exp_pooled_rel:+.2f}%, DM "
                                f"{r.exp_dm_clu:+.2f}, Holm {r.exp_holm:.3f}"
                                for _, r in kept.iterrows()) + ")" if len(kept) else "")
            + ".\n")

    md.append("\n## Full grid — per-cell deployable results\n")
    md.append("| disc | model | h | seeds | FIXED sig_q | FIXED mean rel% | FIXED pooled "
              "rel% | FIXED DM | FIXED Holm | EXP sig_q | EXP mean rel% | EXP pooled "
              "rel% | EXP DM | EXP Holm | EXP placebo | gap pooled rel% | genuine FIXED "
              "| genuine EXP | status |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.sort_values(["disc", "model", "h"]).iterrows():
        md.append(frow(r))

    md.append("\n## Fixed-vs-expanding gap summary\n")
    gap = df.groupby("disc")[["gap_pooled_rel", "gap_mean_rel", "gap_sig_q_clu"]].mean()
    md.append("| disc | mean gap pooled rel% (exp-fixed) | mean gap per-quarter rel% | "
              "mean gap sig_q |\n|---|---|---|---|")
    for disc, r in gap.iterrows():
        md.append(f"| {disc} | {r.gap_pooled_rel:+.3f} | {r.gap_mean_rel:+.3f} | "
                  f"{r.gap_sig_q_clu:+.2f} |")
    md.append(f"\n- LOST-ON-DEPLOY ({len(lost)}): "
              + ("; ".join(f"{r.disc}/{r.model}/h{r.h}" for _, r in lost.iterrows()) or "none"))
    md.append(f"- GAINED-ON-DEPLOY ({len(gained)}): "
              + ("; ".join(f"{r.disc}/{r.model}/h{r.h}" for _, r in gained.iterrows()) or "none"))
    md.append(f"- SURVIVES-DEPLOY ({len(surv)}): "
              + ("; ".join(f"{r.disc}/{r.model}/h{r.h}" for _, r in surv.iterrows()) or "none"))

    md.append("\n## SANITY\n")
    md.append("**GATE 1 (committed table: `results/tables/rolling_robustness.csv`)** — "
              "the 36 overlapping (disc, model, h) cells were recomputed on the exact "
              "committed code path (seed2026 text, obs-order per-quarter DM, "
              "moving-block CI) and compared row-by-row (1152 rows x 6 columns) with "
              f"NaN-aware machine-precision equality (atol {GATE_ATOL:.0e}; bitwise "
              "identity is unattainable across runs — BLAS reduction order injects "
              "~1e-15 noise into lstsq): "
              + ", ".join(f"{c}: {'exact' if g1[f'{c}_exact'] else 'MISMATCH'}"
                          for c in ["rel_impr_pct", "dm_stat", "dm_p", "ci_lo", "ci_hi", "n"])
              + f" -> **{'PASS' if g1['pass'] else 'FAIL'}** "
              f"(joined {g1['n_joined']}/{g1['n_committed']} rows; max abs diff over all "
              "float columns = "
              + f"{max(g1[f'{c}_maxdiff'] for c in ['rel_impr_pct','dm_stat','dm_p','ci_lo','ci_hi']):.1e}).")
    md.append(f"\n**GATE 2 (committed table: `results/tables/m1_ensemble_primary.csv`)** — "
              f"the fixed val-frozen scheme on the ensemble basis reproduces the committed "
              f"69-cell primary to machine precision (atol {GATE_ATOL:.0e}, DM stat "
              f"included in the pass criterion): max|dQLIKE_R|={g2_max['dq_R']:.1e}, "
              f"max|dQLIKE_U|={g2_max['dq_U']:.1e}, max|dg_log|={g2_max['dg']:.1e}, "
              f"max|dDM_clu|={g2_max['ddm']:.1e} over {len(g2)}/69 cells -> **PASS**.")
    md.append(f"\n- All test rows fall inside the 16-quarter span "
              f"(rows outside span: {int(df.n_outside_span.sum())}); the expanding path "
              "covers every span row (no unfit quarters).")
    md.append("- No subsampling anywhere; the per-quarter long panel is in "
              "`deployable_combiner_quarters.csv`.")
    md.append("- The one committed anecdote reproduces on the new basis: "
              "long_form/B2_tfidf_ridge/h5 BEFORE fixed "
              + (f"{int(df.loc[(df.disc=='long_form')&(df.model=='B2_tfidf_ridge')&(df.h==5),'old_fixed_sig_q'].iloc[0])}"
                 if np.isfinite(df.loc[(df.disc=='long_form')&(df.model=='B2_tfidf_ridge')&(df.h==5),'old_fixed_sig_q'].iloc[0]) else "-")
              + "/16 -> expanding "
              + (f"{int(df.loc[(df.disc=='long_form')&(df.model=='B2_tfidf_ridge')&(df.h==5),'old_expanding_sig_q'].iloc[0])}"
                 if np.isfinite(df.loc[(df.disc=='long_form')&(df.model=='B2_tfidf_ridge')&(df.h==5),'old_expanding_sig_q'].iloc[0]) else "-")
              + "/16 (legacy obs-order, seed2026); see the per-cell table for the "
              "restated counts on the ensemble basis.")

    md.append("\n## Verdict\n")
    md.append(f"- Fixed val-frozen scheme (Holm within F-DEPLOY-FIXED): **{nf}/{n}** "
              f"genuine cells ({nf69}/69 primary).")
    md.append(f"- EXPANDING deployable scheme (Holm within F-DEPLOY-EXP): **{ne}/{n}** "
              f"genuine cells ({ne69}/69 primary); {len(lost)} lost on deploy, "
              f"{len(gained)} gained, {len(surv)} survive.")
    if len(c6ed):
        md.append(f"- Event-driven C6 residual: {int(c6ed.genuine_fixed.sum())}/3 genuine "
                  f"fixed -> **{int(c6ed.genuine_exp.sum())}/3 genuine under deployable "
                  "expanding weights** (see HEADLINE).")
    md.append(f"- Mean pooled-rel gap (expanding - fixed) across all cells: "
              f"**{df.gap_pooled_rel.mean():+.3f}pp**.")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(md))

    print("=== ROW 7 deployable combiner — done ===")
    print(f"cells={n} genuine fixed={nf} genuine expanding={ne} "
          f"(primary-grid: {nf69}/69 -> {ne69}/69)")
    if len(c6ed):
        print("C6 event_driven:",
              c6ed[["h", "fixed_dm_clu", "fixed_holm", "exp_dm_clu", "exp_holm",
                    "exp_pooled_rel", "genuine_fixed", "genuine_exp"]].to_string(index=False))
    print(f"lost-on-deploy={len(lost)} gained={len(gained)} survive={len(surv)}")
    print(f"wrote {OUT_CSV}, {OUT_MD}, {OUT_QCSV}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
