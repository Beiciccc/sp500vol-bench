"""PREREG D (prereg-cd-v1.0, configs/prereg_mechanism_and_labels.md §D) — TASK 2.

Full 69-cell cascade (primary -> firm-identity -> maximal pool -> conjunction) rerun
under RANGE-BASED labels: Parkinson (PRIMARY) with Garman-Klass as the same-table
restatement. Day-clustered DM, per-family Holm, placebo gates, per-cell MDE and
signal-injection recovery — every code path reused from the committed cascade:

  m1_ensemble_primary.py           (ensemble text object, cell stats, placebo)
  basis_alignment_ensemble.py /
  maximal_reference_firm_control.py (firm + maximal-pool references, day keys)
  clustered_dm.py                   (day-clustered DM, Holm via fc.holm)
  signal_injection_power.py         (calibrate_kappa, stage_eval, MDE, 9 Holm families)

KEY DESIGN (prereg §D, binding):
  * TEXT ARMS ARE FROZEN — predictions reused from the committed runs (3-seed
    per-observation ensemble, identical to the declared primary). They were optimised
    for the close-to-close target: FIRST LIMITATION, readings conservative for text.
  * Log-space recalibration + combiner weights REFIT on validation with the NEW labels.
  * A-block feature-side consistency: the label-consuming HAR-family references are
    REFIT on range-based features + labels via the committed fitting code:
      A2_har_rv  — src/sp500vol/models/price/har_rv.HARRV (train split, per horizon,
                   log OLS + Duan smearing), features rv_1d/5d/22d -> same estimator;
      A6_shar    — scripts/experiments/stronger_baselines conventions (fit_log_ols +
                   BPQ insanity filter); rv_5/rv_22 -> same estimator; the signed daily
                   semivols RS-/RS+ stay return-based (no range-based analogue exists —
                   disclosed).
    A3_garch / A4_egarch / A5_arima are LABEL-FREE return-based forecasters with no
    aligned-panel RV features: they enter FROZEN (recalibrated on val to the new label),
    exactly the committed combination-time treatment.
  * firm-mean reference term recomputed from the firm's own VAL rows of the NEW label.

MODES:
  --mode selftest  light local checks: inputs present + A2 refit machinery reproduces
                   the stored run (one disclosure). No cascade. Writes nothing.
  --mode g1        SANITY GATE G1: the full cascade code path on the ORIGINAL labels
                   (stored predictions) must reproduce the committed
                   38/69 (primary) / 8/69 (firm) / 9/69 (pool) / 0/69 (conjunction),
                   per-cell stats to machine precision, committed MDEs, committed
                   injection recovery counts, and the A2/A6 refit machinery must
                   reproduce the stored prediction parquets. Writes the sentinel
                   results/tables/_rangebased_g1_pass.json on PASS; hard-fails otherwise.
  --mode final     SINGLE SHOT. Requires the G1 sentinel (pass) + rangebased labels
                   parquet (G2 gate passed inside its builder). REFUSES if
                   results/tables/rangebased_cascade.csv already exists. Runs the
                   Parkinson cascade (full detail) + GK (same-table restatement),
                   writes results/tables/rangebased_cascade.{csv,md}.

Run from repo root:
  RB_THREADS=5 .venv/bin/python scripts/analysis/rangebased_cascade.py --mode g1
"""
from __future__ import annotations

import os

_THREADS = os.environ.get("RB_THREADS", "5")  # env caps BEFORE numpy import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "scripts/experiments")
sys.path.insert(0, "src")

import forecast_combination as fc  # noqa: E402
import m1_ensemble_primary as mep  # noqa: E402
import stronger_baselines as sb  # noqa: E402
from clustered_dm import daily_mean, dm_test_clustered  # noqa: E402
from maximal_reference_firm_control import (  # noqa: E402
    PRICE, build_price_panel, firm_mean_val, log_ols_frozen)
from signal_injection_power import (  # noqa: E402
    TARGETS, Z_POWER, calibrate_kappa, rel_pct, stage_eval)
from sp500vol.evaluation.dm_test import _hac_variance  # noqa: E402
from sp500vol.models.price.har_rv import HARRV  # noqa: E402

DATA_ROOT = Path(os.environ.get("SP500VOL_DATA_ROOT", "/path/to/data-root/sp500vol-data"))
# stronger_baselines hardcodes /path/to/data-root paths at module level — re-root for the box.
sb.RETURNS_PATH = str(DATA_ROOT / "market" / "crsp" / "market_returns.parquet")
sb.ALIGNED_PATH = str(DATA_ROOT / "processed" / "full" / "aligned_filings.parquet")

RB_PARQUET = DATA_ROOT / "processed" / "full" / "rangebased_labels.parquet"
RB_META = DATA_ROOT / "processed" / "full" / "rangebased_labels_meta.json"

T = Path("results/tables")
SENTINEL = T / "_rangebased_g1_pass.json"
FINAL_CSV = T / "rangebased_cascade.csv"
FINAL_MD = T / "rangebased_cascade.md"

KEY = fc.KEY
SORT = fc.SORT
HORIZONS = fc.HORIZONS
GRIDKEY = ["disc", "model", "h"]
PLACEBO_SEEDS = fc.PLACEBO_SEEDS
# G1 numeric tolerance: 1e-9 = same-machine CSV round-trip "machine precision".
# On a different machine/BLAS, lstsq can drift at ~1e-11; counts must STILL match
# exactly (hard gate) — the stats tolerance alone may be relaxed via RB_G1_TOL with
# the actual diffs recorded in the sentinel for the main session to judge.
GATE_TOL = float(os.environ.get("RB_G1_TOL", "1e-9"))
REFIT_TOL = float(os.environ.get("RB_REFIT_TOL", "1e-6"))  # refit vs stored parquets
EPS = 1e-8


# ============================================================ A-block refits
def _join_rb(df: pd.DataFrame, rb: pd.DataFrame, est: str) -> pd.DataFrame:
    """Attach range-based label + features for estimator `est` ('pk'|'gk')."""
    cols = ["accession", "horizon_days", "ticker", f"label_{est}",
            f"{est}_1d", f"{est}_5d", f"{est}_22d"]
    r = rb[cols].rename(columns={"ticker": "ticker_rb"})
    out = df.merge(r, on=["accession", "horizon_days"], how="left", validate="m:1")
    assert (out["ticker"] == out["ticker_rb"]).all(), "ticker mismatch on accession join"
    return out.drop(columns=["ticker_rb"])


def refit_a2(disc: str, rb: pd.DataFrame | None, est: str | None):
    """Refit A2 HAR via the committed HARRV class. est=None -> reproduction check on the
    stored features/labels (must match the stored prediction parquet). Returns
    (df with columns KEY+split+times+fhar_new+label_new, n_rows_lost, max_abs_repro_diff)."""
    a2 = pd.read_parquet(mep.run_dir("A2_har_rv", disc, 2026))
    n0 = len(a2)
    if est is None:
        feat = a2[["feature_rv_1d", "feature_rv_5d", "feature_rv_22d"]].copy()
        y = a2["label_realised_vol"].to_numpy(dtype=float)
        df = a2
        n_lost = 0
    else:
        df = _join_rb(a2, rb, est)
        keep = (np.isfinite(df[f"label_{est}"]) & np.isfinite(df[f"{est}_1d"])
                & np.isfinite(df[f"{est}_5d"]) & np.isfinite(df[f"{est}_22d"]))
        n_lost = int((~keep).sum())
        df = df[keep].reset_index(drop=True)
        feat = df[[f"{est}_1d", f"{est}_5d", f"{est}_22d"]].rename(columns={
            f"{est}_1d": "feature_rv_1d", f"{est}_5d": "feature_rv_5d",
            f"{est}_22d": "feature_rv_22d"})
        y = df[f"label_{est}"].to_numpy(dtype=float)
    fitdf = feat.copy()
    fitdf["horizon_days"] = df["horizon_days"].to_numpy()
    model = HARRV(log_transform=True, smearing=True)  # committed A2 config
    tr = (df["split"] == "train").to_numpy()
    model.fit(fitdf[tr], y[tr])
    pred = model.predict(fitdf)
    repro = float(np.max(np.abs(pred - a2["prediction_realised_vol"].to_numpy())))\
        if est is None else float("nan")
    out = df[["split"] + KEY + ["filing_time_utc", "effective_trading_day"]].copy()
    out["fhar_new"] = pred
    out["label_new"] = y
    return out, n_lost, repro, n0


def refit_shar(disc: str, rb: pd.DataFrame | None, feats: pd.DataFrame, est: str | None):
    """Refit A6_shar via the committed stronger_baselines conventions.
    est=None -> reproduction check against the stored A6_shar parquet."""
    a6 = pd.read_parquet(mep.run_dir("A6_shar", disc, 2026))
    n0 = len(a6)
    df = a6.merge(feats[KEY + ["rs_neg_1", "rs_pos_1"]], on=KEY, how="left",
                  validate="1:1")
    assert np.isfinite(df["rs_neg_1"]).all() and np.isfinite(df["rs_pos_1"]).all(), \
        "stored A6_shar rows must all have rebuilt RS features (feat_ok rows)"
    if est is None:
        rv5 = df["feature_rv_5d"].to_numpy(dtype=float)
        rv22 = df["feature_rv_22d"].to_numpy(dtype=float)
        y = df["label_realised_vol"].to_numpy(dtype=float)
        n_lost = 0
    else:
        df = _join_rb(df, rb, est)
        keep = (np.isfinite(df[f"label_{est}"]) & np.isfinite(df[f"{est}_5d"])
                & np.isfinite(df[f"{est}_22d"]))
        n_lost = int((~keep).sum())
        df = df[keep].reset_index(drop=True)
        rv5 = df[f"{est}_5d"].to_numpy(dtype=float)
        rv22 = df[f"{est}_22d"].to_numpy(dtype=float)
        y = df[f"label_{est}"].to_numpy(dtype=float)
    pred = np.empty(len(df))
    for h in HORIZONS:
        hm = (df["horizon_days"] == h).to_numpy()
        dh = df[hm]
        yh = y[hm]
        tr = (dh["split"] == "train").to_numpy()
        X = np.column_stack([sb.log_feat(dh["rs_neg_1"].to_numpy(float)),
                             sb.log_feat(dh["rs_pos_1"].to_numpy(float)),
                             sb.log_feat(rv5[hm]), sb.log_feat(rv22[hm])])
        params, smear = sb.fit_log_ols(X[tr], yh[tr])
        raw = sb.predict_log_ols(X, params, smear)
        lo, hi, mean_tr = float(yh[tr].min()), float(yh[tr].max()), float(yh[tr].mean())
        pred[hm] = np.where((raw < lo) | (raw > hi), mean_tr, raw)  # BPQ insanity filter
    repro = float(np.max(np.abs(pred - a6["prediction_realised_vol"].to_numpy())))\
        if est is None else float("nan")
    out = df[KEY].copy()
    out["fshar_new"] = pred
    return out, n_lost, repro, n0


# ============================================================ cascade engine
def prep_cells(est: str | None, rb: pd.DataFrame | None, feats: pd.DataFrame | None,
               log=print):
    """69-cell prep — verbatim structure of signal_injection_power.prep_cells, with the
    label/reference substitution for est in {'pk','gk'} and per-cell primary-stage
    placebo added (m1_ensemble_primary.cell_stats convention)."""
    cells, losses, rankings = [], [], []
    for disc, models in fc.SETS.items():
        # ---- stage-HAR panel ----
        if est is None:
            har = pd.read_parquet(mep.run_dir("A2_har_rv", disc, 2026))[
                ["split"] + KEY + ["prediction_realised_vol", "label_realised_vol",
                                   "filing_time_utc", "effective_trading_day"]
            ].rename(columns={"prediction_realised_vol": "fhar"})
        else:
            a2_new, a2_lost, _, a2_n0 = refit_a2(disc, rb, est)
            har = a2_new.rename(columns={"fhar_new": "fhar", "label_new": "label_realised_vol"})
            losses.append({"disc": disc, "what": "A2 refit rows lost", "n0": a2_n0,
                           "lost": a2_lost})
        # ---- 5-price panel (firm + pool stages) ----
        panel = build_price_panel(disc)
        n_panel0 = len(panel)
        if est is not None:
            shar_new, sh_lost, _, sh_n0 = refit_shar(disc, rb, feats, est)
            losses.append({"disc": disc, "what": "A6_shar refit rows lost", "n0": sh_n0,
                           "lost": sh_lost})
            panel = panel.merge(
                har[KEY + ["fhar"]].rename(columns={"fhar": "_fhar_new"}),
                on=KEY, how="inner", validate="1:1")
            panel = panel.merge(shar_new, on=KEY, how="inner", validate="1:1")
            panel = _join_rb(panel, rb, est)
            keep = np.isfinite(panel[f"label_{est}"])
            panel = panel[keep].reset_index(drop=True)
            panel["A2_har_rv"] = panel["_fhar_new"]
            panel["A6_shar"] = panel["fshar_new"]
            panel["label_realised_vol"] = panel[f"label_{est}"]
            losses.append({"disc": disc, "what": "5-price panel rows lost", "n0": n_panel0,
                           "lost": n_panel0 - len(panel)})
        fmap, gmean, _fc_, _oc_ = firm_mean_val(panel)   # firm mean of the CURRENT label
        panel["firm_mean_val"] = panel.ticker.map(fmap).fillna(gmean).astype(float)

        # ---- single-reference ranking (branch-(d) diagnostic) ----
        for h in HORIZONS:
            dv = panel[(panel.horizon_days == h) & (panel.split == "val")].sort_values(
                SORT, kind="mergesort")
            dt = panel[(panel.horizon_days == h) & (panel.split == "test")].sort_values(
                SORT, kind="mergesort")
            yv, yt = dv.label_realised_vol.to_numpy(), dt.label_realised_vol.to_numpy()
            qs = {}
            for pcol in PRICE:
                fR1, _ = log_ols_frozen(yv, [dv[pcol].to_numpy()], [dt[pcol].to_numpy()])
                qs[pcol] = float(fc.qlike(yt, fR1).mean())
            order = sorted(qs, key=qs.get)
            rankings.append({"disc": disc, "h": h, "a2_rank": order.index("A2_har_rv") + 1,
                             "best": order[0],
                             **{f"qlike_{k}": v for k, v in qs.items()}})

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

                # ---- primary-stage placebo (m1_ensemble_primary convention) ----
                pdm = []
                for s in PLACEBO_SEEDS:
                    rng = np.random.default_rng(s)
                    pR, pU, _ = fc.log_combo(yv, fhv, rng.permutation(ftv),
                                             fhr, rng.permutation(ftt))
                    stat, _p, _n = dm_test_clustered(fc.qlike(yt, pU), fc.qlike(yt, pR),
                                                     days1, h)
                    pdm.append(stat)
                c["placebo_dm_clu"] = float(np.mean(pdm))

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
        log(f"[prep:{est or 'orig'}] {disc} done ({len(cells)} cells so far)")
    return cells, pd.DataFrame(losses), pd.DataFrame(rankings)


def base_table(cells):
    """delta=0 pass: per-cell stats for all three stages + Holm + detect + genuine."""
    rows = []
    for c in cells:
        row = {k: c[k] for k in ("disc", "model", "h", "n_seeds", "n_test", "n_test2",
                                 "n_days", "qR1", "qRf", "qRs", "se_daily", "mde_rel_pct",
                                 "g1", "g_firm", "g_star", "placebo_dm_clu",
                                 "s_within_firm_max_absmean")}
        for stage in ("har", "firm", "pool"):
            r = stage_eval(c, stage, 0.0)
            row.update({f"{stage}0_rel": r["rel"], f"{stage}0_dm": r["dm"],
                        f"{stage}0_p": r["p"]})
        rows.append(row)
    base = pd.DataFrame(rows)
    for stage in ("har", "firm", "pool"):
        base[f"{stage}0_holm"] = fc.holm(base[f"{stage}0_p"].fillna(1.0).values)
        base[f"{stage}0_detect"] = (base[f"{stage}0_dm"] < 0) & (base[f"{stage}0_holm"] < 0.05)
    base["genuine"] = (base.har0_detect & (base.placebo_dm_clu.abs() < 2.0))
    base["conj_detect"] = base.har0_detect & base.firm0_detect & base.pool0_detect
    return base


def run_injection(cells, log=print):
    """Verbatim signal_injection_power calibration + cascade at TARGETS."""
    t0 = time.time()
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
                   "converged": ok, "rel1_achieved": achieved}
            for stage in ("har", "firm", "pool"):
                r = stage_eval(c, stage, kap1)
                row.update({f"{stage}_rel": r["rel"], f"{stage}_dm": r["dm"],
                            f"{stage}_p": r["p"], f"{stage}_kappa": r["kappa"]})
            inj_rows.append(row)
        if (i + 1) % 20 == 0:
            log(f"[inject] {i + 1}/{len(cells)} cells ({time.time() - t0:.1f}s)")
    inj = pd.DataFrame(inj_rows)
    for tgt in TARGETS:  # PRE-DECLARED 9 Holm families: one 69-cell family per (stage, target)
        m = inj.target_pct == tgt
        for stage in ("har", "firm", "pool"):
            inj.loc[m, f"{stage}_holm"] = fc.holm(inj.loc[m, f"{stage}_p"].fillna(1.0).values)
    for stage in ("har", "firm", "pool"):
        inj[f"{stage}_detect"] = (inj[f"{stage}_dm"] < 0) & (inj[f"{stage}_holm"] < 0.05)
    inj["all3_detect"] = inj.har_detect & inj.firm_detect & inj.pool_detect
    return inj


def recovery_counts(inj):
    out = {}
    for tgt in TARGETS:
        m = inj[inj.target_pct == tgt]
        out[tgt] = {"har": int(m.har_detect.sum()), "firm": int(m.firm_detect.sum()),
                    "pool": int(m.pool_detect.sum()), "all3": int(m.all3_detect.sum()),
                    "delta_negative": int(m.delta_negative.sum()),
                    "converged": int(m.converged.sum())}
    return out


# ============================================================ G1 mode
def run_g1() -> int:
    t0 = time.time()
    print(f"[g1] cascade code path on ORIGINAL labels (threads={_THREADS})")
    feats = sb.build_return_features()

    # ---- refit machinery reproduction gates ----
    refit = {}
    for disc in fc.SETS:
        _, _, d_a2, _ = refit_a2(disc, None, None)
        _, _, d_a6, _ = refit_shar(disc, None, feats, None)
        refit[f"A2_refit_max_abs_diff_{disc}"] = d_a2
        refit[f"A6_shar_refit_max_abs_diff_{disc}"] = d_a6
        print(f"[g1:refit] {disc}: A2 max|diff|={d_a2:.3e}  A6_shar max|diff|={d_a6:.3e}")
    refit_pass = all(v < REFIT_TOL for v in refit.values())

    # ---- cascade on original labels ----
    cells, _, rankings = prep_cells(None, None, None)
    assert len(cells) == 69, f"expected 69 cells, got {len(cells)}"
    base = base_table(cells)

    en = pd.read_csv(T / "m1_ensemble_primary.csv")
    fi = pd.read_csv(T / "firm_identity_ensemble.csv")
    mx = pd.read_csv(T / "maximal_reference_ensemble.csv")
    ci = pd.read_csv(T / "control_intersection_ensemble.csv")
    sip = pd.read_csv(T / "signal_injection_power.csv")

    g = base.merge(en[GRIDKEY + ["vol_qlike_R", "vol_rel_impr_pct", "vol_dm_q_clu",
                                 "vol_placebo_dm_clu"]], on=GRIDKEY, validate="1:1")
    g = g.merge(fi[GRIDKEY + ["qlike_Rfirm", "rel_impr_pct_firm", "dm_q_clustered"]]
                .rename(columns={"dm_q_clustered": "dm_firm_tab"}), on=GRIDKEY, validate="1:1")
    g = g.merge(mx[GRIDKEY + ["qlike_Rstar", "rel_impr_pct_maximal", "dm_q_clustered"]]
                .rename(columns={"dm_q_clustered": "dm_star_tab"}), on=GRIDKEY, validate="1:1")
    sip0 = sip[sip.target_pct == TARGETS[0]][GRIDKEY + ["mde_rel_pct"]].rename(
        columns={"mde_rel_pct": "mde_committed"})
    g = g.merge(sip0, on=GRIDKEY, validate="1:1")

    diffs = {
        "primary_qlike_R": float((g.qR1 - g.vol_qlike_R).abs().max()),
        "primary_rel": float((g.har0_rel - g.vol_rel_impr_pct).abs().max()),
        "primary_dm": float((g.har0_dm - g.vol_dm_q_clu).abs().max()),
        "primary_placebo": float((g.placebo_dm_clu - g.vol_placebo_dm_clu).abs().max()),
        "firm_qlike_R": float((g.qRf - g.qlike_Rfirm).abs().max()),
        "firm_rel": float((g.firm0_rel - g.rel_impr_pct_firm).abs().max()),
        "firm_dm": float((g.firm0_dm - g.dm_firm_tab).abs().max()),
        "pool_qlike_R": float((g.qRs - g.qlike_Rstar).abs().max()),
        "pool_rel": float((g.pool0_rel - g.rel_impr_pct_maximal).abs().max()),
        "pool_dm": float((g.pool0_dm - g.dm_star_tab).abs().max()),
        "mde": float((g.mde_rel_pct - g.mde_committed).abs().max()),
    }
    counts = {
        "primary_holm": [int(base.har0_detect.sum()), int(ci.primary_holm.sum())],
        "firm_holm": [int(base.firm0_detect.sum()), int(ci.firm_holm.sum())],
        "pool_holm": [int(base.pool0_detect.sum()), int(ci.maximal_holm.sum())],
        "conjunction": [int(base.conj_detect.sum()), int(ci.AND_full_holm.sum())],
        "genuine": [int(base.genuine.sum()), int(en.genuine_ens_vol.sum())],
    }
    print("[g1] max-abs diffs vs committed:", json.dumps(diffs, indent=2))
    print("[g1] counts [this run, committed]:", json.dumps(counts))

    # ---- injection reproduction ----
    inj = run_injection(cells)
    rec = recovery_counts(inj)
    rec_committed = {}
    for tgt in TARGETS:
        m = sip[sip.target_pct == tgt]
        rec_committed[tgt] = {"har": int(m.har_detect.sum()), "firm": int(m.firm_detect.sum()),
                              "pool": int(m.pool_detect.sum()), "all3": int(m.all3_detect.sum())}
    inj_match = all(all(rec[t][k] == rec_committed[t][k] for k in ("har", "firm", "pool", "all3"))
                    for t in TARGETS)
    print("[g1] injection recovery this run:", json.dumps({str(k): v for k, v in rec.items()}))
    print("[g1] injection recovery committed:", json.dumps({str(k): v for k, v in rec_committed.items()}))

    stats_pass = all(v < GATE_TOL for v in diffs.values())
    counts_pass = all(a == b for a, b in counts.values())
    g1 = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "threads": _THREADS, "gate_tol": GATE_TOL, "refit_tol": REFIT_TOL,
        "max_abs_diffs_vs_committed": diffs,
        "counts_this_run_vs_committed": counts,
        "refit_reproduction": refit,
        "injection_recovery_this_run": {str(k): v for k, v in rec.items()},
        "injection_recovery_committed": {str(k): v for k, v in rec_committed.items()},
        "injection_counts_match": inj_match,
        "mde_median_this_run": float(base.mde_rel_pct.median()),
        "single_ref_a2_rank_orig": rankings.to_dict("records"),
        "pass": bool(stats_pass and counts_pass and refit_pass and inj_match),
    }
    T.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(json.dumps(g1, indent=2))
    if not g1["pass"]:
        print(f"G1 FAIL (stats_pass={stats_pass} counts_pass={counts_pass} "
              f"refit_pass={refit_pass} inj_match={inj_match}) — sentinel records the "
              "failure; FINAL MODE WILL REFUSE TO RUN.")
        return 1
    print(f"[g1] ALL GATES PASS in {time.time() - t0:.1f}s — sentinel written: {SENTINEL}")
    return 0


# ============================================================ FINAL mode
def fired_branch(conj_pk: int, mde_med_pk: float, mde_med_committed: float):
    shrink = 100.0 * (mde_med_committed - mde_med_pk) / mde_med_committed
    if conj_pk > 0:
        return "(c)", shrink, ("conjunction > 0 under Parkinson: text survives primary AND "
                               "firm-identity AND maximal-pool jointly under Holm — honest "
                               "reversal; residual chapter must be rewritten (prereg (c)).")
    if shrink >= 30.0:
        return "(a)", shrink, ("null holds (conjunction 0/69) AND median MDE shrinks "
                               f"{shrink:.1f}% >= 30% — wording upgrade per prereg (a): "
                               "evidence of absence at the observed effect sizes; "
                               "rank-1 noise-proxy objection retired.")
    return "(b)", shrink, (f"null holds (conjunction 0/69) but median MDE shrinks only "
                           f"{shrink:.1f}% < 30% — report as-is, qualifiers unchanged "
                           "(prereg (b)).")


def run_final() -> int:
    t0 = time.time()
    # ---------------- single-shot + prerequisite gates ----------------
    if FINAL_CSV.exists():
        print(f"REFUSED: {FINAL_CSV} already exists — single-shot discipline "
              "(prereg-cd-v1.0). Nothing recomputed, nothing overwritten.")
        return 4
    if not SENTINEL.exists():
        print("REFUSED: G1 sentinel missing — run --mode g1 first.")
        return 5
    g1 = json.loads(SENTINEL.read_text())
    if not g1.get("pass"):
        print("REFUSED: G1 sentinel records a FAILURE — fix and rerun --mode g1.")
        return 5
    if not RB_PARQUET.exists() or not RB_META.exists():
        print("REFUSED: rangebased labels parquet/meta missing — run "
              "scripts/analysis/rangebased_labels.py first.")
        return 5
    meta = json.loads(RB_META.read_text())
    if not meta.get("G2_pk_pass"):
        print("REFUSED: G2 (Parkinson rank correlation) did not pass in the labels build.")
        return 5

    rb = pd.read_parquet(RB_PARQUET)
    feats = sb.build_return_features()

    results = {}
    for est in ("pk", "gk"):
        print(f"\n================ {est.upper()} cascade ================")
        cells, losses, rankings = prep_cells(est, rb, feats)
        assert len(cells) == 69, (f"{est}: expected 69 cells, got {len(cells)} — "
                                  "coverage loss broke the grid; see losses table")
        base = base_table(cells)
        inj = run_injection(cells)
        results[est] = {"base": base, "inj": inj, "losses": losses,
                        "rankings": rankings, "rec": recovery_counts(inj)}
        print(f"[{est}] primary {int(base.har0_detect.sum())}/69  "
              f"firm {int(base.firm0_detect.sum())}/69  "
              f"pool {int(base.pool0_detect.sum())}/69  "
              f"conjunction {int(base.conj_detect.sum())}/69  "
              f"genuine(placebo-gated) {int(base.genuine.sum())}/69  "
              f"MDE median {base.mde_rel_pct.median():.3f}%")

    # ---------------- committed (old) columns ----------------
    en = pd.read_csv(T / "m1_ensemble_primary.csv")
    fi = pd.read_csv(T / "firm_identity_ensemble.csv")
    mx = pd.read_csv(T / "maximal_reference_ensemble.csv")
    ci = pd.read_csv(T / "control_intersection_ensemble.csv")
    sip = pd.read_csv(T / "signal_injection_power.csv")
    old = en[GRIDKEY + ["n_test", "vol_qlike_R", "vol_rel_impr_pct", "vol_dm_q_clu",
                        "vol_dmq_holm_clu", "vol_placebo_dm_clu", "genuine_ens_vol"]].rename(
        columns={"n_test": "old_n_test", "vol_qlike_R": "old_qlike_R",
                 "vol_rel_impr_pct": "old_rel", "vol_dm_q_clu": "old_dm",
                 "vol_dmq_holm_clu": "old_holm", "vol_placebo_dm_clu": "old_placebo_dm",
                 "genuine_ens_vol": "old_genuine"})
    old = old.merge(ci[GRIDKEY + ["primary_holm", "firm_holm", "maximal_holm",
                                  "AND_full_holm"]].rename(
        columns={"primary_holm": "old_primary_detect", "firm_holm": "old_firm_detect",
                 "maximal_holm": "old_pool_detect", "AND_full_holm": "old_conj"}),
        on=GRIDKEY, validate="1:1")
    old = old.merge(fi[GRIDKEY + ["rel_impr_pct_firm", "dm_q_clustered"]].rename(
        columns={"rel_impr_pct_firm": "old_firm_rel", "dm_q_clustered": "old_firm_dm"}),
        on=GRIDKEY, validate="1:1")
    old = old.merge(mx[GRIDKEY + ["rel_impr_pct_maximal", "dm_q_clustered"]].rename(
        columns={"rel_impr_pct_maximal": "old_pool_rel", "dm_q_clustered": "old_pool_dm"}),
        on=GRIDKEY, validate="1:1")
    sip0 = sip[sip.target_pct == TARGETS[0]][GRIDKEY + ["mde_rel_pct"]].rename(
        columns={"mde_rel_pct": "old_mde"})
    old = old.merge(sip0, on=GRIDKEY, validate="1:1")
    mde_med_old = float(old.old_mde.median())

    # ---------------- final per-cell CSV ----------------
    def stage_cols(base, prefix):
        b = base.copy()
        ren = {"n_test": f"{prefix}_n_test", "n_test2": f"{prefix}_n_test2",
               "n_days": f"{prefix}_n_days", "qR1": f"{prefix}_qlike_R",
               "mde_rel_pct": f"{prefix}_mde", "placebo_dm_clu": f"{prefix}_placebo_dm",
               "genuine": f"{prefix}_genuine", "conj_detect": f"{prefix}_conj"}
        for stage, tag in (("har", "primary"), ("firm", "firm"), ("pool", "pool")):
            ren.update({f"{stage}0_rel": f"{prefix}_{tag}_rel",
                        f"{stage}0_dm": f"{prefix}_{tag}_dm",
                        f"{stage}0_p": f"{prefix}_{tag}_p",
                        f"{stage}0_holm": f"{prefix}_{tag}_holm",
                        f"{stage}0_detect": f"{prefix}_{tag}_detect"})
        keep = GRIDKEY + ["n_seeds"] + list(ren)
        return b[keep].rename(columns=ren)

    out = stage_cols(results["pk"]["base"], "pk").merge(
        stage_cols(results["gk"]["base"], "gk").drop(columns=["n_seeds"]),
        on=GRIDKEY, validate="1:1").merge(old, on=GRIDKEY, validate="1:1")
    # injection per-cell detect flags (pk primary object)
    for est in ("pk", "gk"):
        inj = results[est]["inj"]
        for tgt in TARGETS:
            tag = f"{est}_i{str(tgt).replace('.', '')}"
            sl = inj[inj.target_pct == tgt][GRIDKEY + ["har_detect", "firm_detect",
                                                       "pool_detect", "all3_detect"]]
            sl = sl.rename(columns={c: f"{tag}_{c}" for c in
                                    ("har_detect", "firm_detect", "pool_detect", "all3_detect")})
            out = out.merge(sl, on=GRIDKEY, validate="1:1")
    out = out.sort_values(GRIDKEY).reset_index(drop=True)

    # ---------------- summary + branch ----------------
    def stage_counts(base):
        return {"primary": int(base.har0_detect.sum()), "firm": int(base.firm0_detect.sum()),
                "pool": int(base.pool0_detect.sum()), "conj": int(base.conj_detect.sum()),
                "genuine": int(base.genuine.sum()),
                "mde_median": float(base.mde_rel_pct.median()),
                "mde_q25": float(base.mde_rel_pct.quantile(.25)),
                "mde_q75": float(base.mde_rel_pct.quantile(.75))}

    cnt = {"old": {"primary": int(ci.primary_holm.sum()), "firm": int(ci.firm_holm.sum()),
                   "pool": int(ci.maximal_holm.sum()), "conj": int(ci.AND_full_holm.sum()),
                   "genuine": int(en.genuine_ens_vol.sum()), "mde_median": mde_med_old,
                   "mde_q25": float(old.old_mde.quantile(.25)),
                   "mde_q75": float(old.old_mde.quantile(.75))},
           "pk": stage_counts(results["pk"]["base"]),
           "gk": stage_counts(results["gk"]["base"])}
    branch, shrink, branch_note = fired_branch(cnt["pk"]["conj"], cnt["pk"]["mde_median"],
                                               mde_med_old)

    rec_old = {str(t): {"har": int(sip[sip.target_pct == t].har_detect.sum()),
                        "firm": int(sip[sip.target_pct == t].firm_detect.sum()),
                        "pool": int(sip[sip.target_pct == t].pool_detect.sum()),
                        "all3": int(sip[sip.target_pct == t].all3_detect.sum())}
               for t in TARGETS}

    # single-shot re-check immediately before write
    if FINAL_CSV.exists():
        print(f"REFUSED AT WRITE TIME: {FINAL_CSV} appeared during the run.")
        return 4
    out.to_csv(FINAL_CSV, index=False)
    write_md(cnt, results, old, branch, shrink, branch_note, rec_old, g1, meta, out)
    print(f"\n[final] wrote {FINAL_CSV} + {FINAL_MD} in {time.time() - t0:.1f}s")
    print(json.dumps({"counts": cnt, "branch": branch, "mde_shrink_pct": round(shrink, 2),
                      "recovery_pk": {str(k): v for k, v in results['pk']['rec'].items()}},
                     indent=2))
    return 0


def _fmt(x, p="+.2f"):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, p)


def write_md(cnt, results, old, branch, shrink, branch_note, rec_old, g1, meta, out):
    pk, gk = results["pk"], results["gk"]
    md = ["# PREREG D — range-based (Parkinson primary / GK restated) 69-cell cascade\n",
          "Pre-registered in configs/prereg_mechanism_and_labels.md §D (tag prereg-cd-v1.0), "
          "single-shot. Text-arm predictions FROZEN (3-seed ensemble, the declared primary "
          "object); combiner/recalibration + firm-mean refit on validation with the NEW "
          "labels; A2 + A6_shar refit on range-based features+labels via the committed "
          "fitting code; A3/A4/A5 frozen label-free return-based (recalibrated). "
          "Day-clustered DM, Holm within each pre-declared 69-cell family, placebo gate, "
          "per-cell MDE + injection recovery — machinery verbatim from the committed "
          "cascade (G1-gated below).\n"]

    md.append("## OLD vs NEW cascade (Holm<.05 detections per stage, 69 cells)\n")
    md.append("| stage | committed (close-to-close) | **Parkinson (primary)** | GK (restated) |")
    md.append("|---|---|---|---|")
    for tag, label in (("primary", "primary: text over recalibrated HAR"),
                       ("firm", "firm-identity-augmented reference"),
                       ("pool", "maximal 5-price pool"),
                       ("conj", "**full conjunction (primary AND firm AND pool)**"),
                       ("genuine", "placebo-gated genuine (primary stage)")):
        md.append(f"| {label} | {cnt['old'][tag]}/69 | **{cnt['pk'][tag]}/69** | "
                  f"{cnt['gk'][tag]}/69 |")
    md.append("")

    md.append("## MDE — old vs new (per-cell, 80% power, 5% two-sided; prereg branch (a) "
              "needs median shrink >= 30%)\n")
    md.append("| | committed | **Parkinson** | GK |")
    md.append("|---|---|---|---|")
    md.append(f"| median MDE_rel% | {cnt['old']['mde_median']:.3f} | "
              f"**{cnt['pk']['mde_median']:.3f}** | {cnt['gk']['mde_median']:.3f} |")
    md.append(f"| IQR | [{cnt['old']['mde_q25']:.2f}, {cnt['old']['mde_q75']:.2f}] | "
              f"[{cnt['pk']['mde_q25']:.2f}, {cnt['pk']['mde_q75']:.2f}] | "
              f"[{cnt['gk']['mde_q25']:.2f}, {cnt['gk']['mde_q75']:.2f}] |")
    md.append(f"| median shrink vs committed | — | **{shrink:.1f}%** | "
              f"{100.0 * (cnt['old']['mde_median'] - cnt['gk']['mde_median']) / cnt['old']['mde_median']:.1f}% |")
    md.append("")

    md.append("## Injection recovery — old vs new (Holm-detected /69 per pre-declared "
              "(stage, level) family)\n")
    md.append("| level | committed HAR/firm/pool/all3 | **Parkinson** HAR/firm/pool/all3 | "
              "GK HAR/firm/pool/all3 |")
    md.append("|---|---|---|---|")
    for t in TARGETS:
        ro, rp, rg = rec_old[str(t)], pk["rec"][t], gk["rec"][t]
        md.append(f"| {t:.1f}% | {ro['har']}/{ro['firm']}/{ro['pool']}/{ro['all3']} | "
                  f"**{rp['har']}/{rp['firm']}/{rp['pool']}/{rp['all3']}** | "
                  f"{rg['har']}/{rg['firm']}/{rg['pool']}/{rg['all3']} |")
    md.append("")

    md.append(f"## FIRED BRANCH: **{branch}**\n")
    md.append(branch_note + "\n")
    rk_pk = pd.DataFrame(pk["rankings"])
    rk_or = pd.DataFrame(g1["single_ref_a2_rank_orig"])
    md.append("Branch-(d) reference-ordering check (A2 rank among the 5 single "
              "recalibrated price references, per disc x h): "
              f"original labels mean rank {rk_or.a2_rank.mean():.2f} "
              f"(rank-1 in {int((rk_or.a2_rank == 1).sum())}/6), Parkinson mean rank "
              f"{rk_pk.a2_rank.mean():.2f} (rank-1 in {int((rk_pk.a2_rank == 1).sum())}/6). "
              + ("HAR remains the strong reference — (d) does not fire.\n"
                 if rk_pk.a2_rank.mean() <= rk_or.a2_rank.mean() + 1 else
                 "HAR ordering degraded materially — treat (d) alongside the branch above "
                 "(G2 label gate already passed).\n"))

    md.append("## Per-cell detail — Parkinson (primary object)\n")
    md.append("| disc | model | h | n_test | rel% old->PK (primary) | DM old->PK | "
              "detect old H/F/P | detect PK H/F/P | placebo PK | MDE old->PK |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        md.append(
            f"| {r.disc} | {r.model} | {r.h} | {int(r.pk_n_test)} | "
            f"{_fmt(r.old_rel)}->{_fmt(r.pk_primary_rel)} | "
            f"{_fmt(r.old_dm)}->{_fmt(r.pk_primary_dm)} | "
            f"{'Y' if r.old_primary_detect else '.'}/"
            f"{'Y' if r.old_firm_detect else '.'}/{'Y' if r.old_pool_detect else '.'} | "
            f"{'Y' if r.pk_primary_detect else '.'}/"
            f"{'Y' if r.pk_firm_detect else '.'}/{'Y' if r.pk_pool_detect else '.'} | "
            f"{_fmt(r.pk_placebo_dm)} | {r.old_mde:.2f}->{r.pk_mde:.2f} |")
    md.append("")

    md.append("## Disclosures\n")
    md.append(
        "1. **Frozen text predictions (FIRST LIMITATION-FEEDER).** Every text arm was "
        "trained and tuned against the close-to-close RV target; its predictions are "
        "reused frozen and only the log-space recalibration/combiner weights are refit "
        "on validation under the new labels. The readout is therefore CONSERVATIVE for "
        "the text side: a text arm optimised directly for range-based targets could do "
        "better. All range-based null readings must carry this caveat.\n"
        "2. **Estimator formulas.** Parkinson: pk_i = ln(H_i/L_i)^2/(4 ln 2); "
        "Garman-Klass (standard): gk_i = 0.5 ln(H_i/L_i)^2 - (2 ln 2 - 1) ln(C_i/O_i)^2. "
        "Labels sqrt(252/H * sum est_i) over the SAME H-trading-day windows as the "
        "committed labels; features sqrt(252/w * trailing-w-valid-day sums) ending at "
        "feature_window_end — the exact alignment.py conventions (verified to <1e-8, "
        "see SANITY). GK day values can be negative; window sums are clipped at 0 "
        f"(label clip count: {meta['coverage']['label_gk_clipped_to_zero']}; feature "
        f"clips: {meta['coverage']['gk_feature_clip_counts']}).\n"
        "3. **Feature-side consistency.** A2_har_rv refit (HARRV class: train split, per "
        "horizon, log OLS + Duan smearing) on range-based rv_1d/5d/22d + range-based "
        "label; A6_shar refit (stronger_baselines conventions incl. the BPQ insanity "
        "filter) with rv_5/rv_22 converted — its signed daily semivols RS-/RS+ remain "
        "return-based (a range estimator has no sign decomposition; disclosed, not a "
        "free choice). A3_garch/A4_egarch/A5_arima are label-free return-based "
        "forecasters with no aligned-panel RV features: frozen, recalibrated on val — "
        "identical to the committed combination-time treatment. The firm-identity "
        "reference term is the firm's own VAL-split mean of the NEW label.\n"
        "4. **Refit machinery validated**: on the ORIGINAL features/labels the refit "
        "code reproduces the stored A2/A6_shar prediction parquets (max abs diffs in "
        "SANITY). Same seeds, same placebo permutations, same Holm families as the "
        "committed cascade. Single-shot: this table was written once; the script "
        "refuses to overwrite it.\n")

    md.append("## SANITY\n")
    md.append("| gate | result |")
    md.append("|---|---|")
    md.append(f"| G1 cascade path reproduces committed tables (original labels) | "
              f"**{'PASS' if g1['pass'] else 'FAIL'}** — max abs diff "
              f"{max(g1['max_abs_diffs_vs_committed'].values()):.2e} over "
              f"primary/firm/pool/MDE/placebo; counts "
              f"{g1['counts_this_run_vs_committed']}; injection counts match: "
              f"{g1['injection_counts_match']} |")
    md.append(f"| G1 refit machinery reproduces stored A2/A6_shar runs | "
              f"max abs diff {max(g1['refit_reproduction'].values()):.2e} (tol {REFIT_TOL:.0e}) |")
    g2rows = ", ".join(f"h{h}: PK {meta['G2'][f'h{h}']['spearman_pk_vs_old']:.4f} / "
                       f"GK {meta['G2'][f'h{h}']['spearman_gk_vs_old']:.4f}"
                       for h in (5, 10, 20))
    md.append(f"| G2 rank correlation new-vs-old labels (gate > 0.8) | {g2rows} — "
              f"PK {'PASS' if meta['G2_pk_pass'] else 'FAIL'}, "
              f"GK {'pass' if meta['G2_gk_pass'] else 'BELOW GATE (disclosed)'} |")
    md.append(f"| G3 leakage asserts | {meta['G3_leakage_asserts']}; combiner/reference "
              "weights val-only by construction (log_ols_frozen / log_combo) |")
    n_pl_pk = int((results['pk']['base'].placebo_dm_clu.abs() < 2.0).sum())
    md.append(f"| G4 placebo (primary stage, 5 permutation seeds) | |placebo DM|<2 in "
              f"{n_pl_pk}/69 Parkinson cells (committed convention) |")
    cov = meta["coverage"]
    md.append(f"| Coverage (labels parquet, {cov['panel_rows']} panel rows) | "
              f"PK label lost {cov['label_pk_lost']}, GK label lost {cov['label_gk_lost']}; "
              f"usable rows old/PK/GK = {cov['rows_usable_old(label+feat)']}/"
              f"{cov['rows_usable_pk(label+feat)']}/{cov['rows_usable_gk(label+feat)']} |")
    for est in ("pk", "gk"):
        ls = results[est]["losses"]
        items = "; ".join(f"{r.disc} {r.what}: {int(r.lost)}/{int(r.n0)}"
                          for _, r in ls.iterrows())
        md.append(f"| Cascade row losses ({est.upper()}) | {items} |")
    md.append(f"| Per-cell n_test drift (PK vs committed) | max "
              f"{int((out.old_n_test - out.pk_n_test).max())} obs, total "
              f"{int((out.old_n_test - out.pk_n_test).sum())} obs across 69 cells |")
    md.append("")
    FINAL_MD.write_text("\n".join(md))


# ============================================================ selftest
def run_selftest() -> int:
    ok = True
    need = [T / f for f in ("m1_ensemble_primary.csv", "firm_identity_ensemble.csv",
                            "maximal_reference_ensemble.csv",
                            "control_intersection_ensemble.csv",
                            "signal_injection_power.csv", "_realized_returns.parquet")]
    for p in need:
        print(f"[selftest] {'OK ' if p.exists() else 'MISSING'} {p}")
        ok &= p.exists()
    n_runs = 0
    for disc, models in fc.SETS.items():
        for m in ["A2_har_rv", "A6_shar", "A3_garch", "A4_egarch", "A5_arima"] + models:
            if mep.run_dir(m, disc, 2026).exists():
                n_runs += 1
            else:
                print(f"[selftest] MISSING run {m}/{disc}/2026")
                ok = False
    print(f"[selftest] {n_runs} seed2026 run parquets present")
    print(f"[selftest] rangebased labels parquet {'present' if RB_PARQUET.exists() else 'ABSENT (expected before the box run)'}")
    _, _, d_a2, n0 = refit_a2("long_form", None, None)
    print(f"[selftest] A2 refit reproduction (long_form, {n0} rows): max|diff|={d_a2:.3e} "
          f"-> {'PASS' if d_a2 < REFIT_TOL else 'FAIL'}")
    ok &= d_a2 < REFIT_TOL
    print(f"[selftest] {'ALL OK' if ok else 'PROBLEMS FOUND'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "g1", "final"], required=True)
    args = ap.parse_args()
    if args.mode == "selftest":
        return run_selftest()
    if args.mode == "g1":
        return run_g1()
    return run_final()


if __name__ == "__main__":
    sys.exit(main())
