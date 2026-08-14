#!/usr/bin/env python
"""Second-domain (Yelp) protocol port — combiner, identity control, clustered DM,
placebos, signal-injection MDE (SECOND_DOMAIN_PLAN.md §2; my half of the pipeline).

Ports the finance protocol pieces to the Yelp business-month rating panel:

  (a) LOG-SPACE NESTED COMBINER (port of scripts/analysis/forecast_combination.py):
          f_R  = exp(a + b·log f_AR)                       recalibrated AR reference
          f_U  = exp(a + b·log f_AR + g·log f_text)        + text
      weights fit on VALIDATION only, frozen on test. Stars lie in [1, 5], strictly
      positive, so log space is safe. Loss = squared error on stars; MSE reported.
  (b) ENTITY-IDENTITY CONTROL (port of scripts/analysis/firm_identity_control.py,
      ticker -> business_id): reference exp OLS[1, log f_AR, log entity_mean], where
      entity_mean = the business's mean monthly stars over its own TRAIN+VAL panel
      months (strictly pre-test; missing businesses -> global train+val mean,
      coverage reported); the zero-text entity-mean-only comparison; identity share.
  (c) MONTH-CLUSTERED DM (port of scripts/analysis/clustered_dm.py, day -> calendar
      month): loss differentials averaged per test month, HAC lag = h-1 months of
      genuine outcome-window overlap, HLN small-sample correction, t(n_months-1);
      plus the business x month TWO-WAY (CGM) variant (port of twoway_dm.py).
  (d) PLACEBOS: label-shuffle (text forecasts permuted across all rows, 5 seeds) and
      within-month text-swap (permuted across businesses inside each calendar month,
      5 seeds); both refit the combiner weights on the permuted validation text, as
      in forecast_combination.py.
  (e) SIGNAL-INJECTION MDE per the row-1 methodology (signal_injection_power.py).
      !!! ORACLE INJECTION — POWER CALIBRATION, NOT A FORECAST !!!  The synthetic
      text forecast exp(log f_text + kappa·s) uses TEST labels by design: s is the
      within-ENTITY demeaned test log-residual of f_R, so an entity-level regressor
      cannot mechanically absorb it. kappa is bisected so the realised test rel-MSE
      improvement hits each target; detection = month-clustered DM < 0 & p < .05 at
      the AR stage and at the entity stage (transmitted through the stage's own real
      validation-fit text loading). Analytic per-cell MDE:
          MDE_rel% = (1.96 + 0.84) · SE_month / MSE_ref · 100.

Inputs (produced by yelp_build_panel.py + yelp_baseline_text.py; swapping synthetic
for real data is upstream's --data-root flag, nothing else):
    --panel      canonical panel parquet (entity means, gate counts)
    --preds-dir  directory with preds_ar_ridge.parquet, preds_tfidf_chrono.parquet
                 and baseline_metrics.json (cross-checked against this script's own
                 recalibration — the two halves must agree to float precision)
    --truth      OPTIONAL truth_months.parquet from make_synthetic.py: computes the
                 ORACLE injected-text benchmark dMSE (synthetic recovery gate only)

Output: --out (default results/second_domain/protocol_results.json), consumed by
yelp_cascade_table.py.

No look-ahead: combiner weights val-fit test-frozen; entity means train+val only;
hard assertions on split boundaries. Boundary note: val events within h-1 months of
the test start have outcome windows overlapping test-period calendar months (same
property as the finance protocol); the count is reported and --embargo-val drops
those rows from the weight fit as a robustness lever.

Run from repo root:
    .venv/bin/python scripts/experiments/second_domain/yelp_protocol.py \
        --panel results/second_domain/yelp_panel.parquet \
        --preds-dir results/second_domain/preds [--tag SYNTHETIC --truth ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
EPS = 1e-8
CLIP_LO, CLIP_HI = 1.0, 5.0
KEY = ["entity_id", "event_time", "horizon_months"]
PLACEBO_SEEDS = tuple(range(1000, 1020))  # 20 seeds: G4 borderline (5-seed mean |DM|=2.04) needs a converged noise floor
SWAP_SEEDS = (2000, 2001, 2002, 2003, 2004)
INJECTION_TARGETS = (0.5, 1.0, 2.0)   # % rel-MSE; an adaptive max(2, 1.5*MDE) is added
TOL_PP = 0.02                          # bisection tolerance, percentage points
KAPPA_CAP = 8.0
Z_POWER = 1.96 + 0.84                  # 80% power at 5% two-sided size
MIN_VAL, MIN_TEST = 100, 30


# ----------------------------------------------------------------- losses / algebra
def se_loss(y, f):
    return (np.asarray(f, float) - np.asarray(y, float)) ** 2


def clip_stars(f):
    return np.clip(np.asarray(f, float), CLIP_LO, CLIP_HI)


def log_ols_frozen(yv, Xv_list, Xt_list):
    """Log-space OLS fit on VALIDATION, applied frozen to test (port of
    firm_identity_control.fit_apply_log). Returns (clipped test forecast, beta)."""
    ly = np.log(np.clip(np.asarray(yv, float), EPS, None))
    Lv = [np.log(np.clip(np.asarray(x, float), EPS, None)) for x in Xv_list]
    Lt = [np.log(np.clip(np.asarray(x, float), EPS, None)) for x in Xt_list]
    A = np.column_stack([np.ones(len(ly))] + Lv)
    beta, *_ = np.linalg.lstsq(A, ly, rcond=None)
    f = np.exp(np.column_stack([np.ones(len(Lt[0]))] + Lt) @ beta)
    return clip_stars(f), beta


# --------------------------------------------------- DM machinery (month-clustered)
def _hac_variance(x, *, lag):
    """Newey-West HAC long-run variance (port of sp500vol.evaluation.dm_test)."""
    n = len(x)
    xc = x - np.mean(x)
    gamma0 = float(np.dot(xc, xc) / n)
    if lag == 0:
        return gamma0
    out = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        out += 2 * w * float(np.dot(xc[k:], xc[:-k]) / n)
    return out


def dm_test(loss_a, loss_b, *, h=1):
    """Diebold-Mariano with HLN small-sample correction and t(n-1) reference
    (port of sp500vol.evaluation.dm_test.dm_test). Positive stat = A worse."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    assert n >= 2, "need at least 2 observations for the DM test"
    mean_d = float(np.mean(d))
    var_d = _hac_variance(d, lag=max(h - 1, 0))
    if var_d <= 0:
        return (0.0, 1.0) if np.isclose(mean_d, 0.0) else (float("nan"), float("nan"))
    hln = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if hln <= 0:
        return float("nan"), float("nan")
    stat = (mean_d / np.sqrt(var_d / n)) * hln ** 0.5
    return float(stat), 2 * float(stats.t.sf(abs(stat), df=n - 1))


def month_key(event_times):
    """Normalise event times to calendar-month Periods (day -> month port of
    clustered_dm._day_index)."""
    return pd.PeriodIndex(pd.DatetimeIndex(np.asarray(event_times)), freq="M")


def monthly_mean(x, months):
    """Group x by calendar month; assert the month grid is contiguous so the HAC
    lag counts genuine months of outcome-window overlap."""
    g = pd.DataFrame({"m": np.asarray(months), "x": np.asarray(x, float)}) \
        .groupby("m", sort=True)["x"].mean()
    idx = pd.PeriodIndex(g.index, freq="M")
    if len(idx) > 1:
        steps = idx.asi8[1:] - idx.asi8[:-1]
        assert (steps == 1).all(), (
            "test months are not contiguous — the HAC lag would misalign; inspect "
            "the panel (or relax this assertion deliberately)")
    return g.to_numpy(), idx


def dm_test_month(loss_a, loss_b, months, h):
    """Month-clustered DM (port of clustered_dm.dm_test_clustered): equal weight per
    month, HAC lag = h-1 MONTHS. Returns (stat, p, n_months)."""
    assert len(loss_a) == len(loss_b) == len(months)
    ma, _ = monthly_mean(loss_a, months)
    mb, _ = monthly_mean(loss_b, months)
    stat, p = dm_test(ma, mb, h=int(h))
    return float(stat), float(p), len(ma)


def dm_test_2way(d, entities, months, h):
    """Business x month two-way CGM variance (port of twoway_dm.dm_test_2way with
    firm -> business_id, day -> calendar month). Positive stat = first loss worse."""
    d = np.asarray(d, float)
    ent = np.asarray(entities)
    mon = month_key(months)
    assert len(d) == len(ent) == len(mon)
    df = pd.DataFrame({"d": d, "ent": ent, "mon": np.asarray(mon)})
    mon_means = df.groupby("mon", sort=True)["d"].mean()
    T, G = len(mon_means), int(df.ent.nunique())
    dbar = float(mon_means.mean())
    n_t = df.groupby("mon")["d"].transform("size").to_numpy(float)
    df["u"] = (d - dbar) / (T * n_t)
    V_ent = float((df.groupby("ent")["u"].sum() ** 2).sum())
    V_cell = float((df.groupby(["ent", "mon"])["u"].sum() ** 2).sum())
    V_mon = float(_hac_variance(mon_means.to_numpy(), lag=max(int(h) - 1, 0)) / T)
    V2 = V_ent + V_mon - V_cell
    guard = bool(V2 <= 0.0)
    V2 = max(V2, 1e-30)
    stat = dbar / np.sqrt(V2)
    dof = min(G, T) - 1
    p = 2.0 * float(stats.t.sf(abs(stat), df=dof)) if dof > 0 else float("nan")
    return {"dm_2way": float(stat), "p_2way": float(p), "n_entities": G,
            "n_months": T, "guard_hit": guard}


# ------------------------------------------------------------------------- placebos
def permute_within(x, groups, rng):
    """Permute x within each group (the within-month text-swap placebo)."""
    x = np.asarray(x, float).copy()
    for pos in pd.Series(np.arange(len(x))).groupby(np.asarray(groups)).indices.values():
        x[pos] = x[rng.permutation(pos)]
    return x


def run_placebo(yv, refs_v, refs_t, fxv, fxt, yt, lref, months_t, h, seeds,
                groups_v=None, groups_t=None):
    """Refit the stage combiner on permuted text (val AND test permuted, weights
    re-fit on val — exactly the forecast_combination.py placebo). groups_* given
    -> within-month swap; otherwise global label-shuffle."""
    dms, ps, dqs = [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        pv = (rng.permutation(fxv) if groups_v is None
              else permute_within(fxv, groups_v, rng))
        pt = (rng.permutation(fxt) if groups_t is None
              else permute_within(fxt, groups_t, rng))
        fU, _ = log_ols_frozen(yv, refs_v + [pv], refs_t + [pt])
        lU = se_loss(yt, fU)
        stat, p, _ = dm_test_month(lU, lref, months_t, h)
        dms.append(stat)
        ps.append(p)
        dqs.append(float(lU.mean() - lref.mean()))
    return {"n_seeds": len(seeds), "mean_dmse": float(np.mean(dqs)),
            "mean_dm": float(np.mean(dms)), "max_abs_dm": float(np.max(np.abs(dms))),
            "mean_p": float(np.mean(ps))}


# ----------------------------------------------------------- signal injection / MDE
def calibrate_kappa(rel_fn, target):
    """Bisect kappa so rel_fn(kappa) hits target within TOL_PP (port of
    signal_injection_power.calibrate_kappa; monotone on the searched branch)."""
    r0 = rel_fn(0.0)
    if abs(r0 - target) <= TOL_PP:
        return 0.0, r0, True
    if r0 < target:
        lo, hi = 0.0, 0.05
        while rel_fn(hi) < target and hi < KAPPA_CAP:
            lo, hi = hi, hi * 2.0
        if rel_fn(hi) < target:
            return hi, rel_fn(hi), False
    else:
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


def mde_rel_pct(l_new, l_ref, months, h):
    """Analytic MDE (row-1 methodology): (1.96+0.84)·SE_month/MSE_ref·100, SE_month
    from the HAC(lag=h-1) variance of the monthly-mean REAL loss differential."""
    dd, _ = monthly_mean(np.asarray(l_new, float) - np.asarray(l_ref, float), months)
    v = _hac_variance(dd, lag=max(int(h) - 1, 0))
    se_m = float(np.sqrt(v / len(dd))) if v > 0 else float("nan")
    mse_ref = float(np.mean(l_ref))
    return Z_POWER * se_m / mse_ref * 100.0, len(dd)


def run_injection(yt, months_t, ents_t, fR, lR, fU0, g_ar, lRe, fUe0, g_ent,
                  h, mde_ar, real_rel):
    """Oracle entity-orthogonal injection (row-1 port). s = within-entity demeaned
    test log-residual of f_R; f_synth enters through exp(log f_U + kappa·s) with the
    REAL validation-fit weights frozen; entity stage receives kappa_ent = g_ent·delta.
    Fixed targets below the real effect REMOVE signal down to the target (row-1
    design: the realised effect is equalised at exactly X%); the ADAPTIVE target
    max(2, 1.5·MDE, real+1) always lies above the real effect, so it is a genuine
    injection — the machinery-validation detection gate."""
    lres = pd.Series(np.log(np.clip(yt, EPS, None)) - np.log(np.clip(fR, EPS, None)))
    ent = pd.Series(np.asarray(ents_t))
    s = (lres - lres.groupby(ent).transform("mean")).to_numpy()
    s_absmean = float(pd.Series(s).groupby(ent).mean().abs().max())
    assert s_absmean < 1e-9, "injected signal is not within-entity mean-zero"

    mseR, mseRe = float(lR.mean()), float(lRe.mean())
    luU0, luUe0 = np.log(fU0), np.log(fUe0)

    def rel_fn(k):
        fu = fU0 if k == 0.0 else clip_stars(np.exp(luU0 + k * s))
        return 100.0 * (mseR - float(se_loss(yt, fu).mean())) / mseR

    targets = list(INJECTION_TARGETS) + [
        max(2.0, round(1.5 * mde_ar, 2), round(real_rel + 1.0, 2))]
    out = []
    for i, tgt in enumerate(targets):
        kap, achieved, ok = calibrate_kappa(rel_fn, tgt)
        fu = clip_stars(np.exp(luU0 + kap * s))
        dm_a, p_a, _ = dm_test_month(se_loss(yt, fu), lR, months_t, h)
        # transmit into the entity stage through ITS OWN real val-fit text loading
        delta = kap / g_ar if abs(g_ar) > 1e-8 else float("nan")
        if np.isfinite(delta):
            kap_e = g_ent * delta
            fue = clip_stars(np.exp(luUe0 + kap_e * s))
            lue = se_loss(yt, fue)
            rel_e = 100.0 * (mseRe - float(lue.mean())) / mseRe
            dm_e, p_e, _ = dm_test_month(lue, lRe, months_t, h)
        else:
            kap_e = rel_e = dm_e = p_e = float("nan")
        out.append({
            "target_pct": float(tgt), "adaptive": bool(i == len(targets) - 1),
            "kappa": float(kap), "delta": float(delta), "converged": bool(ok),
            "achieved_rel_pct": float(achieved),
            "ar": {"dm": dm_a, "p": p_a, "detect": bool(dm_a < 0 and p_a < 0.05)},
            "entity": {"kappa": float(kap_e), "rel_pct": float(rel_e), "dm": dm_e,
                       "p": p_e,
                       "detect": bool(np.isfinite(dm_e) and dm_e < 0 and p_e < 0.05)},
        })
    return {"s_within_entity_max_absmean": s_absmean, "targets": out,
            "disclosure": "ORACLE injection — s uses test labels BY DESIGN; power "
                          "calibration only, never citable as forecast performance"}


# ------------------------------------------------------------------------- horizon
def run_horizon(d, h, ent_map, g_tv_mean, truth_delta, embargo_val):
    dv = d[d.split == "val"].sort_values(["event_time", "entity_id"], kind="mergesort")
    dt = d[d.split == "test"].sort_values(["event_time", "entity_id"], kind="mergesort")
    assert len(dv) >= MIN_VAL and len(dt) >= MIN_TEST, "val/test row floors violated"
    assert dv.event_time.max() < dt.event_time.min(), "val/test overlap in event time"

    # boundary overlap: val outcome windows crossing into the test-label period
    pv, pt = month_key(dv.event_time), month_key(dt.event_time)
    overlap = pv.asi8 + h >= pt.asi8.min() + 1
    n_overlap = int(overlap.sum())
    if embargo_val and n_overlap:
        dv = dv[~overlap]
        assert len(dv) >= MIN_VAL, "embargo left too few validation rows"

    yv, fav, fxv = (dv[c].to_numpy(float) for c in ("label", "f_ar", "f_text"))
    yt, fat, fxt = (dt[c].to_numpy(float) for c in ("label", "f_ar", "f_text"))
    months_t = month_key(dt.event_time)
    months_v = month_key(dv.event_time)
    ents_t = dt.entity_id.to_numpy()

    em_v = dv.entity_id.map(ent_map)
    em_t = dt.entity_id.map(ent_map)
    coverage = float(em_t.notna().mean())
    em_v = em_v.fillna(g_tv_mean).to_numpy(float)
    em_t = em_t.fillna(g_tv_mean).to_numpy(float)

    # ---- references and combined forecasts (all weights val-fit, test-frozen) ----
    fR, bR = log_ols_frozen(yv, [fav], [fat])                       # recalibrated AR
    fT, _ = log_ols_frozen(yv, [fxv], [fxt])                        # recal text-alone
    fU, bU = log_ols_frozen(yv, [fav, fxv], [fat, fxt])             # AR + text
    fRe, bRe = log_ols_frozen(yv, [fav, em_v], [fat, em_t])         # AR + entity mean
    fUe, bUe = log_ols_frozen(yv, [fav, em_v, fxv], [fat, em_t, fxt])
    g_ar, c_ent, g_ent = float(bU[-1]), float(bRe[-1]), float(bUe[-1])

    lraw = se_loss(yt, fat)
    lR, lT, lU = se_loss(yt, fR), se_loss(yt, fT), se_loss(yt, fU)
    lRe, lUe = se_loss(yt, fRe), se_loss(yt, fUe)
    mse = {k: float(v.mean()) for k, v in
           [("ar_raw", lraw), ("R", lR), ("T", lT), ("U", lU), ("Re", lRe), ("Ue", lUe)]}

    def rel(m_ref, m_new):
        return 100.0 * (m_ref - m_new) / m_ref

    # ---- rows 2-5 inference ----
    dm2, p2, n_months = dm_test_month(lT, lR, months_t, h)
    dm3, p3, _ = dm_test_month(lU, lR, months_t, h)
    tw3 = dm_test_2way(lU - lR, ents_t, dt.event_time, h)
    dm4, p4, _ = dm_test_month(lRe, lR, months_t, h)
    dm5, p5, _ = dm_test_month(lUe, lRe, months_t, h)
    tw5 = dm_test_2way(lUe - lRe, ents_t, dt.event_time, h)

    d3, d4, d5 = mse["R"] - mse["U"], mse["R"] - mse["Re"], mse["Re"] - mse["Ue"]
    share_chrono = 100.0 * d4 / d3 if d3 > 0 else float("nan")

    # ORACLE-GIVEN-TEXT benchmark (LEAKY test-fit projection, machinery yardstick
    # only): the residual text increment an optimal TEST-fit combiner would extract
    # from f_text over the same reference. The deployed val-fit combiner should
    # recover a substantial fraction of this — the recovery gate compares d5 to it.
    fRe_o, _ = log_ols_frozen(yt, [fat, em_t], [fat, em_t])
    fUe_o, _ = log_ols_frozen(yt, [fat, em_t, fxt], [fat, em_t, fxt])
    d5_oracle = float(se_loss(yt, fRe_o).mean() - se_loss(yt, fUe_o).mean())

    # ---- placebos (row-3 and row-5 stages) ----
    plac = {
        "row3_shuffle": run_placebo(yv, [fav], [fat], fxv, fxt, yt, lR,
                                    months_t, h, PLACEBO_SEEDS),
        "row5_shuffle": run_placebo(yv, [fav, em_v], [fat, em_t], fxv, fxt, yt, lRe,
                                    months_t, h, PLACEBO_SEEDS),
        "row3_swap": run_placebo(yv, [fav], [fat], fxv, fxt, yt, lR, months_t, h,
                                 SWAP_SEEDS, groups_v=np.asarray(months_v),
                                 groups_t=np.asarray(months_t)),
        "row5_swap": run_placebo(yv, [fav, em_v], [fat, em_t], fxv, fxt, yt, lRe,
                                 months_t, h, SWAP_SEEDS,
                                 groups_v=np.asarray(months_v),
                                 groups_t=np.asarray(months_t)),
    }

    # ---- MDE + oracle injection ----
    mde_ar, _ = mde_rel_pct(lU, lR, months_t, h)
    mde_ent, _ = mde_rel_pct(lUe, lRe, months_t, h)
    inj = run_injection(yt, months_t, ents_t, fR, lR, fU, g_ar, lRe, fUe, g_ent,
                        h, mde_ar, rel(mse["R"], mse["U"]))

    # ---- oracle injected-text benchmark from ground truth (synthetic only) ----
    oracle_dmse = None
    if truth_delta is not None:
        # truth month_idx convention (make_synthetic.py): year*12 + month - 1
        midx = np.asarray(months_t.year) * 12 + np.asarray(months_t.month) - 1
        key = pd.MultiIndex.from_arrays([ents_t, midx])
        delta = truth_delta.reindex(key).to_numpy(float)
        assert not np.isnan(delta).any(), "truth deltas missing for some test events"
        resid = yt - fRe
        cov = float(np.cov(resid, delta, ddof=0)[0, 1])
        oracle_dmse = cov ** 2 / float(np.var(delta))

    return {
        "n_val": len(dv), "n_test": len(dt), "n_test_months": int(n_months),
        "n_entities_test": int(pd.Series(ents_t).nunique()),
        "boundary_overlap_val_rows": n_overlap, "embargo_val": bool(embargo_val),
        "recal_b": float(bR[1]), "coverage_entity_mean": coverage,
        "mse": mse,
        "text_alone": {"delta_rel_pct": rel(mse["R"], mse["T"]), "dm": dm2, "p": p2},
        "ar_text": {"delta_rel_pct": rel(mse["R"], mse["U"]), "dm": dm3, "p": p3,
                    "g_text": g_ar, **tw3},
        "entity": {"delta_rel_pct": rel(mse["R"], mse["Re"]), "dm": dm4, "p": p4,
                   "c_entity": c_ent, "identity_share_chrono_pct": share_chrono},
        "residual": {"delta_rel_pct": rel(mse["Re"], mse["Ue"]), "dmse_abs": d5,
                     "dm": dm5, "p": p5, "g_text": g_ent,
                     "oracle_given_text_dmse": d5_oracle,
                     "oracle_given_text_rel_pct": 100.0 * d5_oracle / mse["Re"],
                     "recovered_frac_of_text_oracle": (d5 / d5_oracle
                                                       if d5_oracle != 0 else float("nan")),
                     **tw5},
        "placebo": plac,
        "mde": {"ar_stage_rel_pct": mde_ar, "entity_stage_rel_pct": mde_ent,
                "n_months": int(n_months)},
        "injection": inj,
        "oracle_injected_dmse": oracle_dmse,
    }


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(REPO / "results/second_domain/yelp_panel.parquet"))
    ap.add_argument("--preds-dir", default=str(REPO / "results/second_domain/preds"))
    ap.add_argument("--out", default=str(REPO / "results/second_domain/protocol_results.json"))
    ap.add_argument("--truth", default=None,
                    help="truth_months.parquet from make_synthetic.py (synthetic only)")
    ap.add_argument("--tag", default="REAL")
    ap.add_argument("--embargo-val", action="store_true",
                    help="drop val rows whose outcome window crosses into the test period")
    ap.add_argument("--text-arm", default="tfidf_chrono",
                    help="text arm to run the protocol on (preds_<arm>.parquet); "
                         "subsample arms intersect with the AR arm via the inner merge")
    args = ap.parse_args()
    t0 = time.time()

    preds = Path(args.preds_dir)
    ar = pd.read_parquet(preds / "preds_ar_ridge.parquet")
    tx = pd.read_parquet(preds / f"preds_{args.text_arm}.parquet")
    d = (ar.rename(columns={"prediction": "f_ar"})[KEY + ["split", "label", "f_ar"]]
         .merge(tx.rename(columns={"prediction": "f_text"})[KEY + ["label", "f_text"]],
                on=KEY, suffixes=("", "_tx"), validate="1:1"))
    assert np.allclose(d.label, d.label_tx), "label mismatch between AR and text arms"
    d = d.drop(columns=["label_tx"])
    assert not d[["label", "f_ar", "f_text"]].isna().any().any()
    assert d.label.between(CLIP_LO, CLIP_HI).all()
    assert (d.f_ar > 0).all() and (d.f_text > 0).all()
    assert not d.duplicated(KEY).any()

    # entity means: TRAIN+VAL observed monthly stars from the panel (strictly pre-test)
    panel = pd.read_parquet(args.panel,
                            columns=["entity_id", "event_time", "split", "ar_last_mean"])
    tv = panel[panel.split.isin(("train", "val"))].drop_duplicates(
        ["entity_id", "event_time"])
    te_min = panel.loc[panel.split == "test", "event_time"].min()
    assert tv.event_time.max() < te_min, "entity-mean window touches the test period"
    ent_map = tv.groupby("entity_id")["ar_last_mean"].mean()
    g_tv_mean = float(tv.ar_last_mean.mean())

    truth_delta = None
    if args.truth:
        tr = pd.read_parquet(args.truth, columns=["business_id", "month_idx", "delta"])
        truth_delta = tr.set_index(
            pd.MultiIndex.from_arrays([tr.business_id, tr.month_idx]))["delta"]

    horizons = sorted(d.horizon_months.unique())
    res = {}
    for h in horizons:
        res[str(int(h))] = run_horizon(d[d.horizon_months == h], int(h), ent_map,
                                       g_tv_mean, truth_delta, args.embargo_val)

    # ---- cross-check against the baseline half (same recalibration, same MSE);
    # ---- only valid on the full validation set, so skipped under --embargo-val ----
    metrics_path = preds / "baseline_metrics.json"
    crosscheck = {}
    if metrics_path.exists() and not args.embargo_val and args.text_arm == "tfidf_chrono":
        met = {m["horizon_months"]: m for m in
               json.loads(metrics_path.read_text())["horizons"]}
        for h in horizons:
            mine, theirs = res[str(int(h))], met[int(h)]
            db = abs(mine["recal_b"] - theirs["recal_b"])
            dm_ = abs(mine["mse"]["R"] - theirs["mse_test_ar_recal"])
            assert db < 5e-4, f"h={h}: recal_b disagrees with baseline half ({db:.2e})"
            assert dm_ < 1e-8, f"h={h}: MSE(f_R) disagrees with baseline half ({dm_:.2e})"
            crosscheck[str(int(h))] = {"recal_b_absdiff": db, "mse_R_absdiff": dm_}

    out = {
        "tag": args.tag, "text_arm": args.text_arm, "synthetic": bool(args.truth),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "panel": str(args.panel), "preds_dir": str(args.preds_dir),
        "embargo_val": bool(args.embargo_val),
        "placebo_seeds": list(PLACEBO_SEEDS), "swap_seeds": list(SWAP_SEEDS),
        "crosscheck_vs_baseline_half": crosscheck,
        "horizons": res,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    # ---- console summary ----
    print(f"\n=== [{args.tag}] yelp_protocol — done in {time.time() - t0:.1f}s ===")
    for h in horizons:
        r = res[str(int(h))]
        pl = r["placebo"]["row5_shuffle"]
        adaptive = [t for t in r["injection"]["targets"] if t["adaptive"]][0]
        print(f"h={h}m  n_test={r['n_test']:,} over {r['n_test_months']} months  "
              f"recal_b={r['recal_b']:.3f}  cov(ent-mean)={r['coverage_entity_mean']:.3f}")
        print(f"  MSE: raw AR={r['mse']['ar_raw']:.4f}  f_R={r['mse']['R']:.4f}  "
              f"text-alone={r['mse']['T']:.4f}  AR+text={r['mse']['U']:.4f}  "
              f"AR+ent={r['mse']['Re']:.4f}  AR+ent+text={r['mse']['Ue']:.4f}")
        print(f"  row3 AR+text     : {r['ar_text']['delta_rel_pct']:+.2f}%  "
              f"DM {r['ar_text']['dm']:+.2f} (p={r['ar_text']['p']:.4f}; "
              f"2way p={r['ar_text']['p_2way']:.4f})")
        print(f"  row4 AR+entity   : {r['entity']['delta_rel_pct']:+.2f}%  "
              f"DM {r['entity']['dm']:+.2f} (p={r['entity']['p']:.4f}); "
              f"identity share (chrono) {r['entity']['identity_share_chrono_pct']:.0f}%")
        print(f"  row5 residual    : {r['residual']['delta_rel_pct']:+.2f}%  "
              f"DM {r['residual']['dm']:+.2f} (p={r['residual']['p']:.4f}; "
              f"2way p={r['residual']['p_2way']:.4f}); text-oracle "
              f"{r['residual']['oracle_given_text_rel_pct']:+.2f}% "
              f"(recovered {100 * r['residual']['recovered_frac_of_text_oracle']:.0f}%)")
        print(f"  placebo (row5)   : shuffle mean DM {pl['mean_dm']:+.2f} "
              f"(mean p={pl['mean_p']:.3f}); swap mean DM "
              f"{r['placebo']['row5_swap']['mean_dm']:+.2f}")
        print(f"  MDE 80% power    : AR stage {r['mde']['ar_stage_rel_pct']:.2f}%  "
              f"entity stage {r['mde']['entity_stage_rel_pct']:.2f}%")
        print(f"  injection        : adaptive target {adaptive['target_pct']:.2f}% -> "
              f"detect AR={adaptive['ar']['detect']} entity={adaptive['entity']['detect']} "
              f"(converged={adaptive['converged']})")
        if r["oracle_injected_dmse"] is not None:
            print(f"  DGP truth        : injected-text dMSE={r['oracle_injected_dmse']:.5f}; "
                  f"the TF-IDF arm extracted "
                  f"{100 * r['residual']['oracle_given_text_dmse'] / r['oracle_injected_dmse']:.0f}% "
                  f"of it into f_text (fixture-arm diagnostic, not a protocol property)")
    if crosscheck:
        print("crosscheck vs baseline half: recal_b and MSE(f_R) agree — PASS")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
