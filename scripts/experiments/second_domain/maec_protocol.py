#!/usr/bin/env python
"""MAEC audit — protocol port (prereg configs/prereg_maec_audit.md §5/§6, tag
prereg-maec-v1.0). Port of yelp_protocol.py with entity=permno, cluster=call
DATE (non-contiguous date grid), combiner = v-space OLS.

Ports, cell by cell:
  (a) COMBINER: labels are already v = log daily vol (§2.1), so the Yelp
      log-space combiner exp(a + b*log f) degenerates to LINEAR OLS ON v —
      an algebraically identical port of `log_ols_frozen` without the exp/log
      wrappers. Weights fit on VALIDATION only, frozen on test; forecasts
      clipped to v in [ln 1e-4, 0] (sigma_daily in [1e-4, 1.0], §5).
  (b) TWO REFERENCES (§5, mirror prereg-rfa dual-reference): R-AR =
      OLS[1, V_past^(n)] (headline, OPEN-3) and R-HAR = OLS[1, V_past^(5),
      V_past^(22), V_past^(66)]; the full arm ladder is run against BOTH.
  (c) STPEV entity-mean control (§5, OPEN-4): point-in-time EXPANDING column
      from the panel is primary; the Yelp-port train+val FIXED mean is the
      robustness block. Zero-text STPEV-only row reported (descriptive).
  (d) CALL-DATE-CLUSTERED DM (§6.1): loss differentials averaged per call
      date; the date grid is NON-CONTIGUOUS (earnings seasons), so the Yelp
      `monthly_mean` contiguity assert is relaxed to ordered+unique and the
      HAC lag is L_n = max over test call dates of the number of LATER distinct
      test call dates within n-1 TRADING days (computed from the frozen test
      date grid + the union trading calendar; metadata only, printed via
      --print-lags). Co-primary: date x permno two-way CGM (dm_test_2way port).
  (e) PLACEBOS (§6.3): label-shuffle 20 seeds (1000-1019, G4) and within-DATE
      text-swap 5 seeds (2000-2004, G4b; single-call dates cannot swap — the
      effective swap fraction is disclosed).
  (f) MDE + ORACLE INJECTION (§6.4): analytic MDE = (1.96+0.84)*SE_date /
      MSE_ref * 100 with HAC(L_n) SE over date-mean loss differentials;
      injection s = within-PERMNO demeaned test v-residual of f_R (v-space
      addition replaces the Yelp log-space multiplication), targets
      {0.5, 1, 2}% + adaptive max(2, 1.5*MDE, real+1).
      !!! ORACLE injection — power calibration only, never citable as
      forecast performance !!!
  (g) DATE-BLOCK MOVING BOOTSTRAP CI (§6.4): block = 5 call dates, 2,000
      draws, seed 2026, on the date-equal-weighted rel-MSE of the combined
      (row-3) and residual (row-5) stages.
  (h) HOLM(8) FAMILIES (§6.2): F1 "combined increment" and F2
      "identity-controlled residual", each 4 horizons x 2 references, per
      run (= arm x alignment). probe / text-alone / STPEV-only rows are
      descriptive and never enter Holm.
  (i) SINGLE-SHOT GUARD (§6.5): refuses to overwrite an existing output
      unless --force-rerun --reason ... (reason is logged into the output).

Inputs: --panel (maec_build_panel.py output; --alignment selects the §2.3
branch), --preds-dir with preds_<arm>.parquet in the schema
[permno, call_date, horizon, split, label, prediction, arm]. Text arms are fit
once under the PRIMARY alignment (§2.3: the sensitivity arm re-derives labels
and combiner fits only), so under --alignment shifted the merge is the
intersection and the dropped-row count is printed loudly.

Self-test (synthetic in-memory panel; the ONLY mode this task executes besides
--print-lags):    .venv/bin/python scripts/experiments/second_domain/maec_protocol.py --selftest

Bug-fix log (prereg §6.5: reruns are bug-fixes only; every code fix is dated
here and echoed in the prereg revision record):
  2026-07-15  crosscheck_vs_baseline_half REWRITTEN before any real
              (arm x alignment) scoring run. The original block read
              mse_test_<ref>_recal keys from maec_baseline_metrics.json —
              keys that were REMOVED from the fit stage under its
              no-test-metric discipline — so the cross-check had silently
              degenerated into a no-op. It is now a PREDICTION-LEVEL
              consistency gate: the protocol refits its own references
              (v_ols_frozen on the alignment panel's val rows) and compares
              the resulting val+test predictions ROW-LEVEL against the stored
              fit-stage halves preds_r_{ar,har}_{alignment}.parquet, asserting
              max |diff| < 1e-8. No metric of any kind is computed — the gate
              touches predictions only.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
CLIP_LO, CLIP_HI = float(np.log(1e-4)), 0.0        # v-units (§5)
KEY = ["permno", "call_date", "horizon"]
HORIZONS = (3, 7, 15, 30)
REFS = ("r_ar", "r_har")                            # OPEN-3: r_ar is headline
PLACEBO_SEEDS = tuple(range(1000, 1020))            # §6.3 G4, Yelp precedent
SWAP_SEEDS = (2000, 2001, 2002, 2003, 2004)         # §6.3 G4b
INJECTION_TARGETS = (0.5, 1.0, 2.0)                 # §6.4 (% rel-MSE)
TOL_PP = 0.02
KAPPA_CAP = 8.0
Z_POWER = 1.96 + 0.84                               # 80% power, 5% two-sided
MIN_VAL, MIN_TEST = 100, 30                         # §4
BOOT_BLOCK, BOOT_DRAWS, BOOT_SEED = 5, 2000, 2026   # §6.4 date-block bootstrap
DEFAULT_PANEL = "/Volumes/Z/second-domain/earnings_calls/maec_panel.parquet"
DEFAULT_CAL = "/Volumes/Z/second-domain/earnings_calls/maec_calendar.parquet"


# ----------------------------------------------------------------- combiner
def se_loss(y, f):
    return (np.asarray(f, float) - np.asarray(y, float)) ** 2


def clip_v(f):
    return np.clip(np.asarray(f, float), CLIP_LO, CLIP_HI)


def v_ols_frozen(yv, Xv_list, Xt_list):
    """v-space OLS fit on VALIDATION, applied frozen to test — the identity
    port of yelp_protocol.log_ols_frozen (labels are already logs, §2.1).
    Returns (clipped test forecast, beta)."""
    A = np.column_stack([np.ones(len(np.asarray(yv)))]
                        + [np.asarray(x, float) for x in Xv_list])
    beta, *_ = np.linalg.lstsq(A, np.asarray(yv, float), rcond=None)
    T = np.column_stack([np.ones(len(np.asarray(Xt_list[0])))]
                        + [np.asarray(x, float) for x in Xt_list])
    return clip_v(T @ beta), beta


# ------------------------------------------------- DM machinery (date grid)
def _hac_variance(x, *, lag):
    n = len(x)
    xc = x - np.mean(x)
    gamma0 = float(np.dot(xc, xc) / n)
    out = gamma0
    for k in range(1, min(int(lag), n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        out += 2 * w * float(np.dot(xc[k:], xc[:-k]) / n)
    return out


def dm_test(loss_a, loss_b, *, lag):
    """DM with HLN correction at horizon h = lag+1 and t(n-1) reference
    (sp500vol.evaluation.dm_test port, lag made explicit for the L_n grid).
    Positive stat = A worse."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    assert n >= 2, "need at least 2 observations for the DM test"
    mean_d = float(np.mean(d))
    var_d = _hac_variance(d, lag=max(int(lag), 0))
    if var_d <= 0:
        return (0.0, 1.0) if np.isclose(mean_d, 0.0) else (float("nan"), float("nan"))
    h = int(lag) + 1
    hln = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if hln <= 0:
        return float("nan"), float("nan")
    stat = (mean_d / np.sqrt(var_d / n)) * hln ** 0.5
    return float(stat), 2 * float(stats.t.sf(abs(stat), df=n - 1))


def date_mean(x, dates):
    """Group x by call date. §6.1: the date grid is NON-contiguous (earnings
    seasons) — the Yelp contiguity assert is relaxed to ordered + unique."""
    g = pd.DataFrame({"d": pd.DatetimeIndex(np.asarray(dates)),
                      "x": np.asarray(x, float)}).groupby("d", sort=True)["x"].mean()
    idx = pd.DatetimeIndex(g.index)
    assert idx.is_monotonic_increasing and idx.is_unique, "date grid not ordered/unique"
    return g.to_numpy(), idx


def hac_lag_L(test_dates, calendar, n) -> int:
    """§6.1 frozen lag: L_n = max over test call dates of the number of LATER
    distinct test call dates within n-1 TRADING days (trading-day distance on
    the union calendar; metadata only — depends on dates, never on labels)."""
    grid = np.sort(pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(test_dates))).values)
    cal = np.sort(pd.DatetimeIndex(calendar).values)
    pos = np.searchsorted(cal, grid, side="right") - 1   # last trading day <= d
    L = 0
    for i in range(len(grid)):
        later = pos[i + 1:] - pos[i]
        L = max(L, int((later <= n - 1).sum()))
    return L


def dm_test_date(loss_a, loss_b, dates, lag):
    """Call-date-clustered DM: equal weight per call date, HAC lag = L_n."""
    assert len(loss_a) == len(loss_b) == len(dates)
    ma, _ = date_mean(loss_a, dates)
    mb, _ = date_mean(loss_b, dates)
    stat, p = dm_test(ma, mb, lag=lag)
    return float(stat), float(p), int(len(ma))


def dm_test_2way(d, entities, dates, lag):
    """Permno x call-date two-way CGM (yelp dm_test_2way port; HAC lag = L_n
    on the date dimension). Positive stat = first loss worse."""
    d = np.asarray(d, float)
    ent = np.asarray(entities)
    dat = pd.DatetimeIndex(np.asarray(dates))
    assert len(d) == len(ent) == len(dat)
    df = pd.DataFrame({"d": d, "ent": ent, "dat": np.asarray(dat)})
    dat_means = df.groupby("dat", sort=True)["d"].mean()
    T, G = int(len(dat_means)), int(df.ent.nunique())
    dbar = float(dat_means.mean())
    n_t = df.groupby("dat")["d"].transform("size").to_numpy(float)
    df["u"] = (d - dbar) / (T * n_t)
    V_ent = float((df.groupby("ent")["u"].sum() ** 2).sum())
    V_cell = float((df.groupby(["ent", "dat"])["u"].sum() ** 2).sum())
    V_dat = float(_hac_variance(dat_means.to_numpy(), lag=max(int(lag), 0)) / T)
    V2 = V_ent + V_dat - V_cell
    guard = bool(V2 <= 0.0)
    V2 = max(V2, 1e-30)
    stat = dbar / np.sqrt(V2)
    dof = min(G, T) - 1
    p = 2.0 * float(stats.t.sf(abs(stat), df=dof)) if dof > 0 else float("nan")
    return {"dm_2way": float(stat), "p_2way": float(p), "n_entities": G,
            "n_dates": T, "guard_hit": guard}


def holm_adjust(pvals):
    """Holm step-down adjusted p-values (§6.2 Holm(8) families)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------- placebos
def permute_within(x, groups, rng):
    x = np.asarray(x, float).copy()
    for pos in pd.Series(np.arange(len(x))).groupby(np.asarray(groups)).indices.values():
        x[pos] = x[rng.permutation(pos)]
    return x


def run_placebo(yv, refs_v, refs_t, fxv, fxt, yt, lref, dates_t, lag, seeds,
                groups_v=None, groups_t=None):
    """Refit the stage combiner on permuted text (val AND test permuted, weights
    re-fit on val — the forecast_combination.py placebo). groups_* given ->
    within-call-date swap; otherwise global label-shuffle."""
    dms, ps, dqs = [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        pv = (rng.permutation(fxv) if groups_v is None
              else permute_within(fxv, groups_v, rng))
        pt = (rng.permutation(fxt) if groups_t is None
              else permute_within(fxt, groups_t, rng))
        fU, _ = v_ols_frozen(yv, refs_v + [pv], refs_t + [pt])
        lU = se_loss(yt, fU)
        stat, p, _ = dm_test_date(lU, lref, dates_t, lag)
        dms.append(stat)
        ps.append(p)
        dqs.append(float(lU.mean() - lref.mean()))
    return {"n_seeds": len(seeds), "mean_dmse": float(np.mean(dqs)),
            "mean_dm": float(np.mean(dms)), "max_abs_dm": float(np.max(np.abs(dms))),
            "mean_p": float(np.mean(ps))}


# ---------------------------------------------------------- MDE / injection / CI
def calibrate_kappa(rel_fn, target):
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


def mde_rel_pct(l_new, l_ref, dates, lag):
    """Analytic MDE: (1.96+0.84)*SE_date/MSE_ref*100, SE_date from the HAC(L_n)
    variance of the date-mean REAL loss differential (§6.4)."""
    dd, _ = date_mean(np.asarray(l_new, float) - np.asarray(l_ref, float), dates)
    v = _hac_variance(dd, lag=max(int(lag), 0))
    se_d = float(np.sqrt(v / len(dd))) if v > 0 else float("nan")
    mse_ref = float(np.mean(l_ref))
    return Z_POWER * se_d / mse_ref * 100.0, int(len(dd))


def run_injection(yt, dates_t, ents_t, fR, lR, fU0, g_ar, lRe, fUe0, g_ent,
                  lag, mde_ar, real_rel):
    """Oracle permno-orthogonal injection (yelp run_injection port; v-space
    addition replaces log-space multiplication). s = within-permno demeaned
    test v-residual of f_R; the entity stage receives kappa_ent = g_ent*delta
    through ITS OWN real val-fit text loading."""
    res = pd.Series(np.asarray(yt, float) - np.asarray(fR, float))
    ent = pd.Series(np.asarray(ents_t))
    s = (res - res.groupby(ent).transform("mean")).to_numpy()
    s_absmean = float(pd.Series(s).groupby(ent).mean().abs().max())
    assert s_absmean < 1e-9, "injected signal is not within-entity mean-zero"  # G3

    mseR, mseRe = float(lR.mean()), float(lRe.mean())

    def rel_fn(k):
        fu = fU0 if k == 0.0 else clip_v(fU0 + k * s)
        return 100.0 * (mseR - float(se_loss(yt, fu).mean())) / mseR

    targets = list(INJECTION_TARGETS) + [
        max(2.0, round(1.5 * mde_ar, 2), round(real_rel + 1.0, 2))]
    out = []
    for i, tgt in enumerate(targets):
        kap, achieved, ok = calibrate_kappa(rel_fn, tgt)
        fu = clip_v(fU0 + kap * s)
        dm_a, p_a, _ = dm_test_date(se_loss(yt, fu), lR, dates_t, lag)
        delta = kap / g_ar if abs(g_ar) > 1e-8 else float("nan")
        if np.isfinite(delta):
            kap_e = g_ent * delta
            fue = clip_v(fUe0 + kap_e * s)
            lue = se_loss(yt, fue)
            rel_e = 100.0 * (mseRe - float(lue.mean())) / mseRe
            dm_e, p_e, _ = dm_test_date(lue, lRe, dates_t, lag)
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


def block_bootstrap_ci(l_ref, l_new, dates, block=BOOT_BLOCK, draws=BOOT_DRAWS,
                       seed=BOOT_SEED):
    """Date-block MOVING bootstrap CI (§6.4) on the date-equal-weighted rel-MSE
    100*(MSE_ref - MSE_new)/MSE_ref. Blocks = `block` consecutive call dates."""
    mr, _ = date_mean(l_ref, dates)
    mn, _ = date_mean(l_new, dates)
    D = len(mr)
    if D <= block:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "n_dates": D}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(D / block))
    starts = rng.integers(0, D - block + 1, size=(draws, n_blocks))
    rels = np.empty(draws)
    for b in range(draws):
        idx = (starts[b][:, None] + np.arange(block)[None, :]).ravel()[:D]
        rr, nn = float(mr[idx].mean()), float(mn[idx].mean())
        rels[b] = 100.0 * (rr - nn) / rr if rr != 0 else np.nan
    lo, hi = np.nanpercentile(rels, [2.5, 97.5])
    return {"ci_lo": float(lo), "ci_hi": float(hi), "n_dates": D,
            "block_dates": block, "draws": draws, "seed": seed}


# ---------------------------------------------------------------- arm ladder
def ref_features(ref):
    return ["v_past_match"] if ref == "r_ar" else ["v_past_5", "v_past_22", "v_past_66"]


def run_ref_ladder(ref, dv, dt, yv, yt, fxv, fxt, stpev_col, dates_t, ents_t,
                   lag, groups_v, groups_t, light=False):
    """The §5 arm ladder against ONE reference X in {R-AR, R-HAR}:
    text-alone, combined increment (row-3), STPEV control (row-4),
    identity-controlled residual (row-5), STPEV-only, identity share."""
    cols = ref_features(ref)
    Xv = [dv[c].to_numpy(float) for c in cols]
    Xt = [dt[c].to_numpy(float) for c in cols]
    sv = dv[stpev_col].to_numpy(float)
    st = dt[stpev_col].to_numpy(float)

    fR, bR = v_ols_frozen(yv, Xv, Xt)                       # recalibrated reference
    fT, _ = v_ols_frozen(yv, [fxv], [fxt])                  # recal text-alone
    fU, bU = v_ols_frozen(yv, Xv + [fxv], Xt + [fxt])       # ref + text
    fRe, bRe = v_ols_frozen(yv, Xv + [sv], Xt + [st])       # ref + STPEV
    fUe, bUe = v_ols_frozen(yv, Xv + [sv, fxv], Xt + [st, fxt])
    fS, _ = v_ols_frozen(yv, [sv], [st])                    # STPEV-only (descriptive)
    g_ar, c_ent, g_ent = float(bU[-1]), float(bRe[-1]), float(bUe[-1])

    # published-convention raw reading (§4): raw V_past^(n) as the forecast
    lraw = se_loss(yt, clip_v(dt["v_past_match"].to_numpy(float)))
    lR, lT, lU = se_loss(yt, fR), se_loss(yt, fT), se_loss(yt, fU)
    lRe, lUe, lS = se_loss(yt, fRe), se_loss(yt, fUe), se_loss(yt, fS)
    mse = {k: float(v.mean()) for k, v in
           [("raw_vpast_match", lraw), ("R", lR), ("T", lT), ("U", lU),
            ("Re", lRe), ("Ue", lUe), ("stpev_only", lS)]}

    def rel(m_ref, m_new):
        return 100.0 * (m_ref - m_new) / m_ref

    dm2, p2, n_dates = dm_test_date(lT, lR, dates_t, lag)
    dm3, p3, _ = dm_test_date(lU, lR, dates_t, lag)
    tw3 = dm_test_2way(lU - lR, ents_t, dates_t, lag)
    dm4, p4, _ = dm_test_date(lRe, lR, dates_t, lag)
    dm5, p5, _ = dm_test_date(lUe, lRe, dates_t, lag)
    tw5 = dm_test_2way(lUe - lRe, ents_t, dates_t, lag)
    dmS, pS, _ = dm_test_date(lS, lR, dates_t, lag)

    d3, d4, d5 = mse["R"] - mse["U"], mse["R"] - mse["Re"], mse["Re"] - mse["Ue"]
    share = 100.0 * d4 / d3 if d3 > 0 else float("nan")

    # oracle-given-text yardstick (LEAKY test-fit projection, machinery only)
    fRe_o, _ = v_ols_frozen(yt, Xt + [st], Xt + [st])
    fUe_o, _ = v_ols_frozen(yt, Xt + [st, fxt], Xt + [st, fxt])
    d5_oracle = float(se_loss(yt, fRe_o).mean() - se_loss(yt, fUe_o).mean())

    out = {
        "mse": mse, "n_dates": n_dates,
        "recal_beta_R": [float(b) for b in bR],
        "text_alone": {"delta_rel_pct": rel(mse["R"], mse["T"]), "dm": dm2, "p": p2},
        "combined": {"delta_rel_pct": rel(mse["R"], mse["U"]), "dm": dm3, "p": p3,
                     "g_text": g_ar, **tw3,
                     "bootstrap_ci_rel_pct": block_bootstrap_ci(lR, lU, dates_t)},
        "entity": {"delta_rel_pct": rel(mse["R"], mse["Re"]), "dm": dm4, "p": p4,
                   "c_entity": c_ent, "identity_share_pct": share},
        "stpev_only": {"delta_rel_pct": rel(mse["R"], mse["stpev_only"]),
                       "dm": dmS, "p": pS},
        "residual": {"delta_rel_pct": rel(mse["Re"], mse["Ue"]), "dmse_abs": d5,
                     "dm": dm5, "p": p5, "g_text": g_ent,
                     "oracle_given_text_dmse": d5_oracle,
                     "oracle_given_text_rel_pct": 100.0 * d5_oracle / mse["Re"],
                     "recovered_frac_of_text_oracle": (
                         d5 / d5_oracle if d5_oracle != 0 else float("nan")),
                     **tw5,
                     "bootstrap_ci_rel_pct": block_bootstrap_ci(lRe, lUe, dates_t)},
    }
    if light:
        return out

    out["placebo"] = {
        "row3_shuffle": run_placebo(yv, Xv, Xt, fxv, fxt, yt, lR,
                                    dates_t, lag, PLACEBO_SEEDS),
        "row5_shuffle": run_placebo(yv, Xv + [sv], Xt + [st], fxv, fxt, yt, lRe,
                                    dates_t, lag, PLACEBO_SEEDS),
        "row3_swap": run_placebo(yv, Xv, Xt, fxv, fxt, yt, lR, dates_t, lag,
                                 SWAP_SEEDS, groups_v=groups_v, groups_t=groups_t),
        "row5_swap": run_placebo(yv, Xv + [sv], Xt + [st], fxv, fxt, yt, lRe,
                                 dates_t, lag, SWAP_SEEDS,
                                 groups_v=groups_v, groups_t=groups_t),
    }
    swap_frac = float(pd.Series(groups_t).groupby(pd.Series(groups_t))
                      .transform("size").gt(1).mean())
    out["placebo"]["swap_effective_row_frac_test"] = swap_frac

    mde_ar, _ = mde_rel_pct(lU, lR, dates_t, lag)
    mde_ent, _ = mde_rel_pct(lUe, lRe, dates_t, lag)
    out["mde"] = {"ar_stage_rel_pct": mde_ar, "entity_stage_rel_pct": mde_ent,
                  "n_dates": n_dates}
    out["injection"] = run_injection(
        yt, dates_t, ents_t, fR, lR, fU, g_ar, lRe, fUe, g_ent, lag,
        mde_ar, rel(mse["R"], mse["U"]))
    return out


def run_horizon(d, h, lag, embargo_val, test_start, stpev_primary="stpev_expanding",
                light=False):
    dv = d[d.split == "val"].sort_values(["call_date", "permno"], kind="mergesort")
    dt = d[d.split == "test"].sort_values(["call_date", "permno"], kind="mergesort")
    assert len(dv) >= MIN_VAL and len(dt) >= MIN_TEST, "val/test row floors violated"
    assert dv.call_date.max() < dt.call_date.min(), "val/test overlap in call date"

    n_overlap = int((dv["label_win_end"] >= test_start).sum())
    if embargo_val and n_overlap:
        dv = dv[dv["label_win_end"] < test_start]
        assert len(dv) >= MIN_VAL, "embargo left too few validation rows"

    yv, fxv = dv["label"].to_numpy(float), dv["f_text"].to_numpy(float)
    yt, fxt = dt["label"].to_numpy(float), dt["f_text"].to_numpy(float)
    dates_t = dt["call_date"].to_numpy()
    ents_t = dt["permno"].to_numpy()
    groups_v, groups_t = dv["call_date"].to_numpy(), dt["call_date"].to_numpy()

    res = {"n_val": int(len(dv)), "n_test": int(len(dt)),
           "n_test_dates": int(pd.Series(dates_t).nunique()),
           "n_entities_test": int(pd.Series(ents_t).nunique()),
           "hac_lag_L": int(lag),
           "boundary_overlap_val_rows": n_overlap, "embargo_val": bool(embargo_val),
           "stpev_variant_primary": stpev_primary,
           "coverage_stpev_prior_test": (
               float((dt["stpev_n_prior"] > 0).mean())
               if "stpev_n_prior" in dt.columns else float("nan"))}
    for ref in REFS:
        res[ref] = run_ref_ladder(ref, dv, dt, yv, yt, fxv, fxt, stpev_primary,
                                  dates_t, ents_t, lag, groups_v, groups_t,
                                  light=light)
    # STPEV robustness block (OPEN-4: Yelp-port fixed mean), rows 4-5 readouts only
    if "stpev_fixed" in d.columns and not light:
        res["stpev_fixed_robustness"] = {
            ref: {k: run_ref_ladder(ref, dv, dt, yv, yt, fxv, fxt, "stpev_fixed",
                                    dates_t, ents_t, lag, groups_v, groups_t,
                                    light=True)[k]
                  for k in ("entity", "residual", "stpev_only")}
            for ref in REFS}
    return res


# -------------------------------------------------- reference cross-check
def crosscheck_reference_predictions(panel, preds_dir, alignment):
    """Bug-fix 2026-07-15 (prereg §6.5, see the header log): row-level
    PREDICTION consistency gate against the fit-stage reference halves
    preds_r_{ar,har}_<alignment>.parquet (val+test rows). The references are
    refit HERE from the alignment panel through the protocol's own combiner
    (v_ols_frozen: val-fit, test-frozen; val rows in-sample by construction on
    both halves) and compared row-by-row. Pure consistency gate — NO metric is
    computed anywhere in this function."""
    out = {}
    for ref in REFS:
        fp = Path(preds_dir) / f"preds_{ref}_{alignment}.parquet"
        if not fp.exists():
            continue
        stored = pd.read_parquet(fp)
        cols = ref_features(ref)
        for h in HORIZONS:
            dh = panel[panel["horizon"] == h]
            dv = dh[dh["split"] == "val"]
            dt = dh[dh["split"] == "test"]
            yv = dv["label"].to_numpy(float)
            Xv = [dv[c].to_numpy(float) for c in cols]
            Xt = [dt[c].to_numpy(float) for c in cols]
            pv, _ = v_ols_frozen(yv, Xv, Xv)     # val rows: in-sample val-fit
            pt, _ = v_ols_frozen(yv, Xv, Xt)     # test rows: frozen
            mine = pd.concat([dv[KEY].assign(pred_protocol=pv),
                              dt[KEY].assign(pred_protocol=pt)],
                             ignore_index=True)
            sh = stored[stored["horizon"] == h]
            m = mine.merge(sh[KEY + ["prediction"]], on=KEY, how="outer",
                           indicator=True, validate="1:1")
            n_unmatched = int((m["_merge"] != "both").sum())
            assert n_unmatched == 0, (
                f"{fp.name} h={h}: row-set mismatch vs the {alignment} panel "
                f"val+test rows ({n_unmatched} unmatched)")
            maxdiff = float(np.max(np.abs(
                m["pred_protocol"].to_numpy(float)
                - m["prediction"].to_numpy(float))))
            assert maxdiff < 1e-8, (
                f"{fp.name} h={h}: reference PREDICTIONS disagree with the "
                f"fit-stage half (max |diff| = {maxdiff:.3e})")
            out[f"{h}_{ref}"] = {"n_rows": int(len(m)),
                                 "max_pred_absdiff": maxdiff}
    return out


# --------------------------------------------------------------------- lags
def print_lags(panel_path, cal_path, alignment):
    cols = ["alignment", "split", "horizon", "call_date"]
    panel = pd.read_parquet(panel_path, columns=cols)   # metadata only: no labels
    cal = pd.read_parquet(cal_path)["date"]
    out = {}
    for align in (["primary", "shifted"] if alignment == "both" else [alignment]):
        pa = panel[(panel["alignment"] == align) & (panel["split"] == "test")]
        out[align] = {}
        for h in HORIZONS:
            dts = pa.loc[pa["horizon"] == h, "call_date"]
            L = hac_lag_L(dts, cal, h)
            D = int(dts.nunique())
            out[align][f"h{h}"] = {"L": L, "n_test_dates": D,
                                   "L_over_dates": round(L / D, 4)}
            print(f"[{align}] h={h:>2}: L_n={L:>3}  n_test_dates={D}  "
                  f"L/dates={L / D:.3f}")
    return out


# ------------------------------------------------------------------ selftest
def make_synthetic(rng, gamma=0.25, n_entities=250):
    """In-memory synthetic panel + text preds with a PLANTED, within-entity
    text signal (gamma=0 -> pure-noise arm for the size check)."""
    cal = pd.bdate_range("2015-01-02", "2018-12-31")
    rows = []
    for e in range(n_entities):
        mu = -4.6 + 0.35 * rng.standard_normal()
        base = int(rng.integers(0, 9))
        for q in range(70, len(cal) - 40, 63):
            pos = q + base + int(rng.integers(0, 3))
            if pos >= len(cal) - 35:
                break
            rows.append((10000 + e, cal[pos], mu))
    df = pd.DataFrame(rows, columns=["permno", "call_date", "mu"])
    q70, q80 = df.call_date.quantile([0.70, 0.80])
    df["split"] = np.where(df.call_date <= q70, "train",
                           np.where(df.call_date <= q80, "val", "test"))
    panels, preds = [], []
    for h in HORIZONS:
        d = df.copy()
        n = len(d)
        d["horizon"] = h
        vpm = d["mu"] + 0.5 * rng.standard_normal(n)
        d["v_past_match"] = vpm
        d["v_past_5"] = vpm + 0.2 * rng.standard_normal(n)
        d["v_past_22"] = vpm + 0.2 * rng.standard_normal(n)
        d["v_past_66"] = vpm + 0.2 * rng.standard_normal(n)
        s = rng.standard_normal(n)                       # within-entity text signal
        d["label"] = (0.55 * vpm + 0.45 * d["mu"] + gamma * s
                      + 0.35 * rng.standard_normal(n))
        d["label_win_end"] = d["call_date"] + pd.Timedelta(days=int(h * 1.5))
        # STPEV columns emulating the build (expanding prior-label mean)
        d = d.sort_values(["permno", "call_date"], kind="mergesort")
        g_tv = float(d.loc[d.split.isin(("train", "val")), "label"].mean())
        exp_mean = (d.groupby("permno")["label"]
                    .transform(lambda x: x.shift(1).expanding().mean()))
        d["stpev_expanding"] = exp_mean.fillna(g_tv)
        d["stpev_n_prior"] = d.groupby("permno").cumcount()
        fixed = (d[d.split.isin(("train", "val"))]
                 .groupby("permno")["label"].mean())
        d["stpev_fixed"] = d["permno"].map(fixed).fillna(g_tv)
        panels.append(d.drop(columns=["mu"]))
        pr = d[["permno", "call_date", "horizon", "split", "label"]].copy()
        pr["prediction"] = 1.1 * s + 0.5 * rng.standard_normal(n) - 4.6
        pr["arm"] = "synthetic_text"
        preds.append(pr)
    return pd.concat(panels, ignore_index=True), pd.concat(preds, ignore_index=True), cal


def selftest():
    print("=== maec_protocol --selftest (synthetic in-memory panel) ===")
    t0 = time.time()
    rng = np.random.default_rng(7)
    panel, preds, cal = make_synthetic(rng, gamma=0.25)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    d = panel.merge(preds.rename(columns={"prediction": "f_text"})
                    [KEY + ["f_text"]], on=KEY, validate="1:1")
    test_start = d.loc[d.split == "test", "call_date"].min()

    lags, res = {}, {}
    for h in HORIZONS:
        dh = d[d.horizon == h]
        L = hac_lag_L(dh.loc[dh.split == "test", "call_date"], cal, h)
        lags[h] = L
        res[h] = run_horizon(dh, h, L, False, test_start)

    check("L_n monotone nondecreasing in n",
          all(lags[a] <= lags[b] for a, b in zip(HORIZONS, HORIZONS[1:])),
          f"lags={lags}")
    check("L_n <= n-1 on an all-trading-day grid",
          all(lags[h] <= h - 1 for h in HORIZONS), f"lags={lags}")

    rec3 = [res[h][r]["combined"] for h in HORIZONS for r in REFS]
    check("planted signal recovered (row3 DM<0, p<.05, all 8 cells)",
          all(c["dm"] < 0 and c["p"] < 0.05 for c in rec3),
          f"p={[round(c['p'], 5) for c in rec3]}")
    rec5 = [res[h][r]["residual"] for h in HORIZONS for r in REFS]
    check("within-entity signal survives STPEV control (row5, >=7/8 cells)",
          sum(c["dm"] < 0 and c["p"] < 0.05 for c in rec5) >= 7)

    p3 = [res[h][r]["combined"]["p"] for h in HORIZONS for r in REFS]
    adj = holm_adjust(p3)
    check("Holm(8) adjusted >= raw and monotone",
          bool(np.all(adj >= np.asarray(p3) - 1e-15) and adj.max() <= 1.0))

    pl = [res[h][r]["placebo"]["row3_shuffle"] for h in HORIZONS for r in REFS]
    check("label-shuffle placebo null (|mean DM|<1.5, mean p>0.15, all cells)",
          all(abs(x["mean_dm"]) < 1.5 and x["mean_p"] > 0.15 for x in pl),
          f"mean_dm={[round(x['mean_dm'], 2) for x in pl]}")
    fr = res[HORIZONS[0]][REFS[0]]["placebo"]["swap_effective_row_frac_test"]
    check("within-date swap executed (effective row fraction in (0,1])",
          0.0 < fr <= 1.0, f"frac={fr:.3f}")

    inj = [t for h in HORIZONS
           for t in res[h]["r_ar"]["injection"]["targets"] if t["adaptive"]]
    check("oracle injection: adaptive target detected at AR stage (all horizons)",
          all(t["ar"]["detect"] for t in inj),
          f"targets={[t['target_pct'] for t in inj]}")
    check("injection s within-entity mean-zero assert executed",
          all(res[h]["r_ar"]["injection"]["s_within_entity_max_absmean"] < 1e-9
              for h in HORIZONS))

    ci = res[HORIZONS[1]]["r_ar"]["combined"]["bootstrap_ci_rel_pct"]
    check("date-block bootstrap CI brackets the point rel% (h=7, r_ar)",
          ci["ci_lo"] <= res[HORIZONS[1]]["r_ar"]["combined"]["delta_rel_pct"]
          <= ci["ci_hi"], f"CI=[{ci['ci_lo']:.2f},{ci['ci_hi']:.2f}]")

    # size check: pure-noise text arm must NOT be detected (light ladder)
    rng2 = np.random.default_rng(11)
    panel0, preds0, cal0 = make_synthetic(rng2, gamma=0.0)
    d0 = panel0.merge(preds0.rename(columns={"prediction": "f_text"})
                      [KEY + ["f_text"]], on=KEY, validate="1:1")
    ts0 = d0.loc[d0.split == "test", "call_date"].min()
    null_p = []
    for h in HORIZONS:
        dh = d0[d0.horizon == h]
        L = hac_lag_L(dh.loc[dh.split == "test", "call_date"], cal0, h)
        r = run_horizon(dh, h, L, False, ts0, light=True)
        null_p += [r[ref]["combined"]["p"] for ref in REFS]
    check("pure-noise arm not detected (row3 p>.05 in >=7/8 cells)",
          sum(p > 0.05 for p in null_p) >= 7,
          f"p={[round(p, 3) for p in null_p]}")

    n_pass = sum(ok for _, ok in checks)
    print(f"\nselftest: {n_pass}/{len(checks)} checks passed "
          f"in {time.time() - t0:.1f}s")
    if n_pass != len(checks):
        sys.exit(1)
    print("SELFTEST PASS")


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--calendar", default=DEFAULT_CAL)
    ap.add_argument("--preds-dir", default=str(REPO / "results/second_domain/maec/preds"))
    ap.add_argument("--arm", default="tfidf",
                    help="text arm (preds_<arm>.parquet)")
    ap.add_argument("--alignment", choices=["primary", "shifted"], default="primary")
    ap.add_argument("--out", default=None,
                    help="default results/second_domain/maec/protocol_<arm>_<alignment>.json")
    ap.add_argument("--tag", default="REAL")
    ap.add_argument("--embargo-val", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic machinery self-test and exit")
    ap.add_argument("--print-lags", action="store_true",
                    help="print the §6.1 HAC lags L_n from the frozen test date "
                         "grid (metadata only: reads dates, never labels) and exit")
    ap.add_argument("--lags-alignment", default="both",
                    choices=["primary", "shifted", "both"])
    ap.add_argument("--force-rerun", action="store_true",
                    help="§6.5 single-shot override; requires --reason")
    ap.add_argument("--reason", default=None,
                    help="revision-log reason for a --force-rerun")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.print_lags:
        print_lags(args.panel, args.calendar, args.lags_alignment)
        return

    out_path = Path(args.out) if args.out else (
        REPO / "results/second_domain/maec" /
        f"protocol_{args.arm}_{args.alignment}.json")
    # §6.5 single-shot discipline
    if out_path.exists():
        if not (args.force_rerun and args.reason):
            sys.exit(f"REFUSED (§6.5 single-shot): {out_path} exists. Reruns are "
                     f"bug-fixes only — pass --force-rerun --reason '...' and log "
                     f"the diff in the prereg revision record.")
        print(f"[§6.5] force-rerun of {out_path}; reason: {args.reason}")

    t0 = time.time()
    panel = pd.read_parquet(args.panel)
    panel = panel[panel["alignment"] == args.alignment].copy()
    cal = pd.read_parquet(args.calendar)["date"]
    tx = pd.read_parquet(Path(args.preds_dir) / f"preds_{args.arm}.parquet")

    d = panel.merge(tx.rename(columns={"prediction": "f_text"})
                    [KEY + ["label", "f_text"]], on=KEY, how="inner",
                    suffixes=("", "_tx"), validate="1:1")
    n_drop = len(panel) - len(d)
    if args.alignment == "primary":
        assert n_drop == 0, (f"G5 FAIL: {n_drop} panel rows missing from the arm "
                             f"predictions under the PRIMARY alignment")
        assert np.allclose(d["label"], d["label_tx"]), \
            "G5 FAIL: label mismatch between panel and arm predictions"
    elif n_drop:
        print(f"[shifted alignment] inner merge dropped {n_drop} rows "
              f"(text arms are fit once under primary, §2.3) — DISCLOSED")
    d = d.drop(columns=["label_tx"])
    assert not d[["label", "f_text", "v_past_match", "stpev_expanding"]].isna().any().any()
    assert not d.duplicated(KEY).any()

    test_start = d.loc[d["split"] == "test", "call_date"].min()
    res, lag_meta = {}, {}
    for h in HORIZONS:
        dh = d[d["horizon"] == h]
        L = hac_lag_L(dh.loc[dh["split"] == "test", "call_date"], cal, h)
        lag_meta[str(h)] = L
        print(f"h={h}: HAC lag L_n = {L}")
        res[str(h)] = run_horizon(dh, h, L, args.embargo_val, test_start)

    # §6.2 Holm(8) families per run: F1 combined, F2 residual (primary STPEV)
    cells = [(str(h), ref) for h in HORIZONS for ref in REFS]
    for fam, row in (("F1_combined", "combined"), ("F2_residual", "residual")):
        ps = [res[h][ref][row]["p"] for h, ref in cells]
        adj = holm_adjust(ps)
        for (h, ref), a in zip(cells, adj):
            res[h][ref][row]["p_holm8"] = float(a)

    # cross-check vs the baseline half — bug-fix 2026-07-15 (§6.5, header log):
    # row-level PREDICTION consistency vs preds_r_{ar,har}_<alignment>.parquet
    # (val+test), refit from the FULL alignment panel (pre-merge, text-free);
    # skipped under --embargo-val (the halves were fit on the full val set).
    crosscheck = {}
    if not args.embargo_val:
        crosscheck = crosscheck_reference_predictions(
            panel, Path(args.preds_dir), args.alignment)
        if crosscheck:
            worst = max(v["max_pred_absdiff"] for v in crosscheck.values())
            print(f"crosscheck vs fit-stage reference halves: "
                  f"{len(crosscheck)} cells, max |pred diff| = {worst:.2e} (<1e-8)")

    out = {
        "tag": args.tag, "arm": args.arm, "alignment": args.alignment,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "panel": str(args.panel), "preds_dir": str(args.preds_dir),
        "embargo_val": bool(args.embargo_val),
        "merge_dropped_rows": int(n_drop),
        "placebo_seeds": list(PLACEBO_SEEDS), "swap_seeds": list(SWAP_SEEDS),
        "hac_lags_L": lag_meta,
        "force_rerun_reason": args.reason,
        "crosscheck_vs_baseline_half": crosscheck,
        "horizons": res,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    print(f"\n=== [{args.tag}] maec_protocol {args.arm}/{args.alignment} — "
          f"done in {time.time() - t0:.1f}s ===")
    for h in HORIZONS:
        r = res[str(h)]
        for ref in REFS:
            c, rr = r[ref]["combined"], r[ref]["residual"]
            print(f"h={h:>2} [{ref:5}] row3 {c['delta_rel_pct']:+.2f}% "
                  f"(p={c['p']:.4f}/holm {c['p_holm8']:.4f})  "
                  f"row5 {rr['delta_rel_pct']:+.2f}% "
                  f"(p={rr['p']:.4f}/holm {rr['p_holm8']:.4f})  "
                  f"share={r[ref]['entity']['identity_share_pct']:.0f}%  "
                  f"MDE(ent)={r[ref]['mde']['entity_stage_rel_pct']:.2f}%")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
