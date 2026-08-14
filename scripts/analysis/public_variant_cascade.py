"""PREREG H (prereg-h-v1.0, configs/prereg_public_variant.md) — TASK 2: the cascade.

Three-panel (label_parity Stage-5, cascade edition) full 69-cell cascade + all-arm
standalone leaderboard under the licence-free public-price variant:

  A = full panel + CRSP labels        (ANCHOR: G1 reproduction gate, nothing new)
  B = covered rows (clean public label exists) INTERSECT panel + CRSP labels
      (isolates the survivorship/panel-selection effect; all forecasts frozen,
      combiner/reference weights refit on the filtered validation rows)
  C = covered rows + PUBLIC labels & features (= the shippable variant itself):
      A2_har_rv and A6_shar REFIT on public features + public labels via the
      committed fitting code (HARRV / stronger_baselines conventions incl. the BPQ
      insanity filter; A6's RS-/RS+ rebuilt from public returns); A3/A4/A5 are
      label-free return-based forecasters — FROZEN, recalibrated on val inside the
      pool/combination machinery (the range-based precedent); the firm-mean
      reference term is the firm's own VAL rows of the panel's OWN label.

MACHINERY: est-keyed rangebased_cascade extended with a ROW-FILTER hook — the cell
loop is a verbatim copy of rangebased_cascade.prep_cells (validated by G1 below);
base_table / run_injection / recovery_counts / refit_a2 are IMPORTED from it.
Leaderboard = the committed dm machinery (m1_clustered.build_joined universe of
dm_pairwise_clustered.csv: 20 seed-ensembled challengers x 3 disclosures x 3
horizons = 180 comparisons), day-clustered VARIANCE-UNIT QLIKE DM vs A2, Holm
within each (disclosure, horizon) group — the exact convention of the committed
variance_unit_standalone180.csv, which the G1 mode must reproduce to machine
precision.

MODES:
  --mode g1     SANITY GATE G1 (panel A anchor): (i) cascade code path on the
                ORIGINAL labels must reproduce the committed per-cell stats
                (machine precision) and counts 38/8/9/0 (+genuine 38), committed
                MDEs and committed injection-recovery counts; (ii) A2/A6 refit
                machinery must reproduce the stored prediction parquets;
                (iii) the leaderboard on panel A must reproduce
                variance_unit_standalone180.csv (var-QLIKE columns) exactly.
                Writes results/tables/_public_variant_g1_pass.json; hard-fails
                otherwise.
  --mode final  SINGLE SHOT. Requires the G1 sentinel (pass) + the labels parquet
                (its own gates passed inside scripts/analysis/public_variant_labels.py).
                REFUSES if results/tables/public_variant_cascade.csv already exists.
                Runs panels B and C (cascade + injection + leaderboard), writes
                results/tables/public_variant_cascade.{csv,md} +
                results/tables/public_variant_leaderboard.csv (extra output,
                disclosed in the md).

Run from repo root:
  PV_THREADS=2 .venv/bin/python scripts/analysis/public_variant_cascade.py --mode g1
"""
from __future__ import annotations

import os

_THREADS = os.environ.get("PV_THREADS", "2")  # env caps BEFORE numpy import (shared box)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
sys.path.insert(0, str(REPO / "src"))

import forecast_combination as fc
import m1_clustered as mc
import m1_ensemble_primary as mep
import rangebased_cascade as rbc
import stronger_baselines as sb
from clustered_dm import daily_mean, dm_test_clustered
from maximal_reference_firm_control import (
    PRICE,
    build_price_panel,
    firm_mean_val,
    log_ols_frozen,
)
from signal_injection_power import TARGETS, Z_POWER
from variance_unit_cascade import qlike_var

from sp500vol.evaluation.dm_test import _hac_variance

DATA_ROOT = Path(os.environ.get("SP500VOL_DATA_ROOT", "/Volumes/Z/sp500vol-data"))
PUB_PARQUET = DATA_ROOT / "processed" / "full" / "public_variant_labels.parquet"
PUB_META = DATA_ROOT / "processed" / "full" / "public_variant_labels_meta.json"

T = Path("results/tables")
SENTINEL = T / "_public_variant_g1_pass.json"
FINAL_CSV = T / "public_variant_cascade.csv"
FINAL_MD = T / "public_variant_cascade.md"
LB_CSV = T / "public_variant_leaderboard.csv"
STAND180 = T / "variance_unit_standalone180.csv"

KEY = fc.KEY
SORT = fc.SORT
HORIZONS = fc.HORIZONS
GRIDKEY = ["disc", "model", "h"]
PLACEBO_SEEDS = fc.PLACEBO_SEEDS
GATE_TOL = float(os.environ.get("PV_G1_TOL", "1e-9"))
REFIT_TOL = float(os.environ.get("PV_REFIT_TOL", "1e-6"))
EPS = 1e-8
PRICE_CH = {"A3_garch", "A4_egarch", "A5_arima"}  # leaderboard price-arm challengers

_JOIN_CACHE: dict = {}


# ============================================================ pub A6 refit
def refit_shar_pub(disc: str, rb: pd.DataFrame):
    """A6_shar refit on PUBLIC features + labels (stronger_baselines conventions,
    incl. BPQ insanity filter). RS-/RS+ rebuilt from public returns (prereg)."""
    a6 = pd.read_parquet(mep.run_dir("A6_shar", disc, 2026))
    n0 = len(a6)
    cols = ["accession", "horizon_days", "ticker", "label_pub", "pub_5d", "pub_22d",
            "rs_neg_pub", "rs_pos_pub"]
    r = rb[cols].rename(columns={"ticker": "ticker_rb"})
    df = a6.merge(r, on=["accession", "horizon_days"], how="left", validate="m:1")
    assert (df["ticker"] == df["ticker_rb"]).all(), "ticker mismatch on accession join"
    df = df.drop(columns=["ticker_rb"])
    keep = (np.isfinite(df["label_pub"]) & np.isfinite(df["pub_5d"])
            & np.isfinite(df["pub_22d"]) & np.isfinite(df["rs_neg_pub"])
            & np.isfinite(df["rs_pos_pub"]))
    n_lost = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    y = df["label_pub"].to_numpy(dtype=float)
    pred = np.empty(len(df))
    for h in HORIZONS:
        hm = (df["horizon_days"] == h).to_numpy()
        dh = df[hm]
        yh = y[hm]
        tr = (dh["split"] == "train").to_numpy()
        X = np.column_stack([sb.log_feat(dh["rs_neg_pub"].to_numpy(float)),
                             sb.log_feat(dh["rs_pos_pub"].to_numpy(float)),
                             sb.log_feat(dh["pub_5d"].to_numpy(float)),
                             sb.log_feat(dh["pub_22d"].to_numpy(float))])
        params, smear = sb.fit_log_ols(X[tr], yh[tr])
        raw = sb.predict_log_ols(X, params, smear)
        lo, hi, mean_tr = float(yh[tr].min()), float(yh[tr].max()), float(yh[tr].mean())
        pred[hm] = np.where((raw < lo) | (raw > hi), mean_tr, raw)  # BPQ insanity filter
    out = df[KEY].copy()
    out["fshar_new"] = pred
    return out, n_lost, n0


# ============================================================ cascade engine (hooked)
def _filt(df: pd.DataFrame, cov_idx: pd.MultiIndex) -> pd.DataFrame:
    """ROW-FILTER hook: keep rows whose (accession, horizon_days) has a clean
    public label (the 'covered' set)."""
    m = pd.MultiIndex.from_arrays([df["accession"], df["horizon_days"]]).isin(cov_idx)
    return df[m].reset_index(drop=True)


def prep_cells_h(panel_tag: str, rb: pd.DataFrame | None, cov_idx: pd.MultiIndex | None,
                 log=print):
    """69-cell prep — VERBATIM cell loop of rangebased_cascade.prep_cells, with the
    prereg-H panel hooks at the top:
       'A' : stored labels/forecasts, no filter (G1 anchor);
       'B' : stored labels/forecasts, covered-row filter;
       'C' : public labels/features — A2/A6 refit ('pub' estimator), covered rows.
    """
    cells, losses, rankings = [], [], []
    for disc, models in fc.SETS.items():
        # ---- stage-HAR panel ----
        if panel_tag in ("A", "B"):
            har = pd.read_parquet(mep.run_dir("A2_har_rv", disc, 2026))[
                ["split"] + KEY + ["prediction_realised_vol", "label_realised_vol",
                                   "filing_time_utc", "effective_trading_day"]
            ].rename(columns={"prediction_realised_vol": "fhar"})
            n_har0 = len(har)
            if panel_tag == "B":
                har = _filt(har, cov_idx)
                losses.append({"disc": disc, "what": "A2 panel rows filtered (not covered)",
                               "n0": n_har0, "lost": n_har0 - len(har)})
        else:
            a2_new, a2_lost, _, a2_n0 = rbc.refit_a2(disc, rb, "pub")
            har = a2_new.rename(columns={"fhar_new": "fhar",
                                         "label_new": "label_realised_vol"})
            losses.append({"disc": disc, "what": "A2 refit rows lost (label/feat)",
                           "n0": a2_n0, "lost": a2_lost})
        # ---- 5-price panel (firm + pool stages) ----
        panel = build_price_panel(disc)
        n_panel0 = len(panel)
        if panel_tag == "B":
            panel = _filt(panel, cov_idx)
            losses.append({"disc": disc, "what": "5-price panel rows filtered (not covered)",
                           "n0": n_panel0, "lost": n_panel0 - len(panel)})
        elif panel_tag == "C":
            shar_new, sh_lost, sh_n0 = refit_shar_pub(disc, rb)
            losses.append({"disc": disc, "what": "A6_shar refit rows lost (label/feat/RS)",
                           "n0": sh_n0, "lost": sh_lost})
            panel = panel.merge(
                har[KEY + ["fhar"]].rename(columns={"fhar": "_fhar_new"}),
                on=KEY, how="inner", validate="1:1")
            panel = panel.merge(shar_new, on=KEY, how="inner", validate="1:1")
            panel = rbc._join_rb(panel, rb, "pub")
            keep = np.isfinite(panel["label_pub"])
            panel = panel[keep].reset_index(drop=True)
            panel["A2_har_rv"] = panel["_fhar_new"]
            panel["A6_shar"] = panel["fshar_new"]
            panel["label_realised_vol"] = panel["label_pub"]
            losses.append({"disc": disc, "what": "5-price panel rows lost", "n0": n_panel0,
                           "lost": n_panel0 - len(panel)})
        fmap, gmean, _fc_, _oc_ = firm_mean_val(panel)   # firm mean of the CURRENT label
        panel["firm_mean_val"] = panel.ticker.map(fmap).fillna(gmean).astype(float)

        # ---- single-reference ranking (diagnostic; rangebased convention) ----
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
        log(f"[prep:{panel_tag}] {disc} done ({len(cells)} cells so far)")
    return cells, pd.DataFrame(losses), pd.DataFrame(rankings)


# ============================================================ leaderboard
def joined(disc):
    if disc not in _JOIN_CACHE:
        _JOIN_CACHE[disc] = mc.build_joined(disc)
    return _JOIN_CACHE[disc]


def leaderboard(panel_tag: str, pub_map: pd.Series | None, log=print) -> pd.DataFrame:
    """All-arm standalone verdicts vs A2 — day-clustered VARIANCE-UNIT QLIKE DM,
    Holm within each (disclosure, horizon) group over the vs-A2 challenger set
    (the exact convention of the committed variance_unit_standalone180.csv)."""
    rows = []
    for disc in mc.DISCLOSURES:
        merged, present = joined(disc)
        if merged is None or mc.HAR not in present:
            continue
        m = merged.join(pub_map, on=KEY) if pub_map is not None else merged
        for h in HORIZONS:
            g = m[m.horizon_days == h].sort_values(SORT, kind="mergesort")
            if panel_tag in ("B", "C"):
                g = g[g.label_pub.notna()]
            if len(g) < 30:
                continue
            y = (g.label_pub if panel_tag == "C" else g.label_realised_vol).to_numpy()
            days = mc.day_key(g)
            qz_har = qlike_var(y, g[f"pred__{mc.HAR}"].to_numpy())
            grp = []
            for ch in present:
                if ch == mc.HAR:
                    continue
                f = g[f"pred__{ch}"].to_numpy()
                dm, p, n_days = dm_test_clustered(qlike_var(y, f), qz_har, days, h)
                grp.append({"panel": panel_tag, "disclosure": disc, "horizon": h,
                            "challenger": ch, "n_obs": len(g), "n_days": n_days,
                            "dm_qlike_var_clu": dm, "p_qlike_var_clu": p})
            gdf = pd.DataFrame(grp)
            gdf["holm_qlike_var_clu"] = fc.holm(gdf.p_qlike_var_clu.fillna(1.0).to_numpy())
            rows.append(gdf)
        log(f"[leaderboard:{panel_tag}] {disc} done")
    full = pd.concat(rows, ignore_index=True)
    full["better_holm"] = (full.dm_qlike_var_clu < 0) & (full.holm_qlike_var_clu < 0.05)
    full["better_raw"] = (full.dm_qlike_var_clu < 0) & (full.p_qlike_var_clu < 0.05)
    full["worse_holm"] = (full.dm_qlike_var_clu > 0) & (full.holm_qlike_var_clu < 0.05)
    full["is_price_arm"] = full.challenger.isin(PRICE_CH)
    return full


def lb_summary(lb: pd.DataFrame) -> dict:
    text = lb[~lb.is_price_arm]
    price = lb[lb.is_price_arm]
    return {"n": len(lb),
            "better_holm": int(lb.better_holm.sum()),
            "better_raw": int(lb.better_raw.sum()),
            "worse_holm": int(lb.worse_holm.sum()),
            "text_better_holm": int(text.better_holm.sum()),
            "text_better_raw": int(text.better_raw.sum()),
            "text_n": len(text),
            "price_better_holm": int(price.better_holm.sum()),
            "winners": sorted((lb[lb.better_holm].disclosure + "/h"
                               + lb[lb.better_holm].horizon.astype(str) + "/"
                               + lb[lb.better_holm].challenger).tolist())}


# ============================================================ G1 mode
def run_g1() -> int:
    t0 = time.time()
    print(f"[g1] PREREG-H anchor: cascade + leaderboard on ORIGINAL labels "
          f"(threads={_THREADS})")
    feats = sb.build_return_features()

    # ---- refit machinery reproduction gates (rangebased code, est=None) ----
    refit = {}
    for disc in fc.SETS:
        _, _, d_a2, _ = rbc.refit_a2(disc, None, None)
        _, _, d_a6, _ = rbc.refit_shar(disc, None, feats, None)
        refit[f"A2_refit_max_abs_diff_{disc}"] = d_a2
        refit[f"A6_shar_refit_max_abs_diff_{disc}"] = d_a6
        print(f"[g1:refit] {disc}: A2 max|diff|={d_a2:.3e}  A6_shar max|diff|={d_a6:.3e}")
    refit_pass = all(v < REFIT_TOL for v in refit.values())

    # ---- cascade on original labels through THIS script's hooked prep ----
    cells, _, rankings = prep_cells_h("A", None, None)
    assert len(cells) == 69, f"expected 69 cells, got {len(cells)}"
    base = rbc.base_table(cells)

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
    inj = rbc.run_injection(cells)
    rec = rbc.recovery_counts(inj)
    rec_committed = {}
    for tgt in TARGETS:
        m = sip[sip.target_pct == tgt]
        rec_committed[tgt] = {"har": int(m.har_detect.sum()), "firm": int(m.firm_detect.sum()),
                              "pool": int(m.pool_detect.sum()), "all3": int(m.all3_detect.sum())}
    inj_match = all(all(rec[t][k] == rec_committed[t][k] for k in ("har", "firm", "pool", "all3"))
                    for t in TARGETS)
    print("[g1] injection recovery this run:", json.dumps({str(k): v for k, v in rec.items()}))
    print("[g1] injection recovery committed:", json.dumps({str(k): v for k, v in rec_committed.items()}))
    del cells

    # ---- leaderboard reproduction on panel A ----
    lbA = leaderboard("A", None)
    ref = pd.read_csv(STAND180)
    j = lbA.merge(ref[["disclosure", "horizon", "challenger", "n_obs", "n_days",
                       "dm_qlike_var_clu", "p_qlike_var_clu", "holm_qlike_var_clu",
                       "better_qlike_var_holm"]],
                  on=["disclosure", "horizon", "challenger"], how="inner",
                  validate="1:1", suffixes=("", "_c"))
    assert len(j) == len(ref) == 180, f"leaderboard universe mismatch: {len(j)} vs {len(ref)}"
    lb_diffs = {"dm": float((j.dm_qlike_var_clu - j.dm_qlike_var_clu_c).abs().max()),
                "p": float((j.p_qlike_var_clu - j.p_qlike_var_clu_c).abs().max()),
                "holm": float((j.holm_qlike_var_clu - j.holm_qlike_var_clu_c).abs().max()),
                "n_obs_mismatch": int((j.n_obs != j.n_obs_c).sum()),
                "n_days_mismatch": int((j.n_days != j.n_days_c).sum()),
                "verdict_mismatch": int((j.better_holm != j.better_qlike_var_holm).sum())}
    lb_pass = (lb_diffs["dm"] < GATE_TOL and lb_diffs["p"] < GATE_TOL
               and lb_diffs["holm"] < GATE_TOL and lb_diffs["n_obs_mismatch"] == 0
               and lb_diffs["n_days_mismatch"] == 0 and lb_diffs["verdict_mismatch"] == 0)
    print("[g1] leaderboard-vs-variance_unit_standalone180 diffs:", json.dumps(lb_diffs))

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
        "leaderboard_diffs_vs_standalone180": lb_diffs,
        "leaderboard_pass": bool(lb_pass),
        "single_ref_a2_rank_orig": rankings.to_dict("records"),
        "pass": bool(stats_pass and counts_pass and refit_pass and inj_match and lb_pass),
    }
    T.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(json.dumps(g1, indent=2))
    if not g1["pass"]:
        print(f"G1 FAIL (stats_pass={stats_pass} counts_pass={counts_pass} "
              f"refit_pass={refit_pass} inj_match={inj_match} lb_pass={lb_pass}) — "
              "sentinel records the failure; FINAL MODE WILL REFUSE TO RUN.")
        return 1
    print(f"[g1] ALL GATES PASS in {time.time() - t0:.1f}s — sentinel written: {SENTINEL}")
    return 0


# ============================================================ FINAL mode
def fired_branch(cnt, lb_flips, text_winner_cells):
    """Prereg-H branch commitments, operationalised BEFORE any panel-B/C statistic
    was read (see md 'Branch operationalisation'):
      (c) fires if the conjunction is > 0 on B or C, OR any text/fusion arm becomes a
          Holm standalone winner on B or C (the paper's verdict object flips);
      (a) fires if conjunction stays 0/69 on B AND C and every one of the 180
          per-comparison standalone Holm verdicts on B and on C equals panel A's
          committed verdict;
      (b) otherwise (verdicts hold at the object level, composition moves)."""
    conj_pos = cnt["B"]["conj"] > 0 or cnt["C"]["conj"] > 0
    if conj_pos or text_winner_cells:
        why = []
        if conj_pos:
            why.append(f"conjunction > 0 (B={cnt['B']['conj']}, C={cnt['C']['conj']})")
        if text_winner_cells:
            why.append("text/fusion standalone winner(s): " + "; ".join(text_winner_cells))
        return "(c)", ("VERDICT FLIP — " + " AND ".join(why) + ". Honest report; the "
                       "variant still ships; the flip itself becomes the finding, "
                       "located by the A-vs-B-vs-C decomposition (prereg (c)).")
    if lb_flips["B"] == 0 and lb_flips["C"] == 0:
        return "(a)", ("VERDICT PRESERVED — B and C standalone Holm verdicts match "
                       "panel A on all 180 comparisons each, and the conjunction stays "
                       "0/69 on both panels: the licence-free variant is promoted to a "
                       "formal release artifact; 12_reproducibility rewrites 'withheld' "
                       "as 'shipped variant + quantified survivorship cost' (prereg (a)).")
    return "(b)", (f"COMPOSITION MOVED, VERDICT HELD — conjunction 0/69 on B and C and "
                   f"no text/fusion standalone winner, but {lb_flips['B']} (B) / "
                   f"{lb_flips['C']} (C) of 180 per-comparison standalone Holm verdicts "
                   "differ from panel A (composition shifts among price arms and/or "
                   "significance-boundary cells). Same upgrade as (a); the composition "
                   "differences are tabulated honestly (prereg (b)).")


def stage_counts(base):
    return {"primary": int(base.har0_detect.sum()), "firm": int(base.firm0_detect.sum()),
            "pool": int(base.pool0_detect.sum()), "conj": int(base.conj_detect.sum()),
            "genuine": int(base.genuine.sum()),
            "mde_median": float(base.mde_rel_pct.median()),
            "mde_q25": float(base.mde_rel_pct.quantile(.25)),
            "mde_q75": float(base.mde_rel_pct.quantile(.75)),
            "placebo_ok": int((base.placebo_dm_clu.abs() < 2.0).sum())}


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


def run_final() -> int:
    t0 = time.time()
    # ---------------- single-shot + prerequisite gates ----------------
    if FINAL_CSV.exists() or LB_CSV.exists():
        print(f"REFUSED: {FINAL_CSV} (or the leaderboard csv) already exists — "
              "single-shot discipline (prereg-h-v1.0). Nothing recomputed, "
              "nothing overwritten.")
        return 4
    if not SENTINEL.exists():
        print("REFUSED: G1 sentinel missing — run --mode g1 first.")
        return 5
    g1 = json.loads(SENTINEL.read_text())
    if not g1.get("pass"):
        print("REFUSED: G1 sentinel records a FAILURE — fix and rerun --mode g1.")
        return 5
    if not PUB_PARQUET.exists() or not PUB_META.exists():
        print("REFUSED: public labels parquet/meta missing — run "
              "scripts/analysis/public_variant_labels.py first.")
        return 5
    meta = json.loads(PUB_META.read_text())
    if not meta["gates"]["G2_parity"].get("pass"):
        print("REFUSED: G2 (public-label parity) did not pass in the labels build.")
        return 5

    rb = pd.read_parquet(PUB_PARQUET)
    cov = rb[np.isfinite(rb.label_pub)]
    cov_idx = pd.MultiIndex.from_arrays([cov.accession, cov.horizon_days])
    pub_map = rb.set_index(KEY)["label_pub"]
    n_cov = len(cov)
    assert n_cov == meta["coverage"]["rows_covered_label"], \
        "covered-row reconciliation vs labels meta failed"
    print(f"[final] covered rows: {n_cov:,}/{len(rb):,} "
          f"({100 * n_cov / len(rb):.2f}% — reconciled to the labels coverage table)")

    results = {}
    for tag in ("B", "C"):
        print(f"\n================ panel {tag} cascade ================")
        cells, losses, rankings = prep_cells_h(tag, rb, cov_idx)
        assert len(cells) == 69, (f"panel {tag}: expected 69 cells, got {len(cells)} — "
                                  "coverage loss broke the grid; see losses table")
        base = rbc.base_table(cells)
        inj = rbc.run_injection(cells)
        results[tag] = {"base": base, "inj": inj, "losses": losses,
                        "rankings": rankings, "rec": rbc.recovery_counts(inj)}
        del cells
        print(f"[{tag}] primary {int(base.har0_detect.sum())}/69  "
              f"firm {int(base.firm0_detect.sum())}/69  "
              f"pool {int(base.pool0_detect.sum())}/69  "
              f"conjunction {int(base.conj_detect.sum())}/69  "
              f"genuine {int(base.genuine.sum())}/69  "
              f"MDE median {base.mde_rel_pct.median():.3f}%")

    # ---------------- leaderboards ----------------
    print("\n================ leaderboards (B, C) ================")
    lbs = {tag: leaderboard(tag, pub_map) for tag in ("B", "C")}
    refA = pd.read_csv(STAND180)
    refA_flags = refA[["disclosure", "horizon", "challenger",
                       "better_qlike_var_holm", "better_qlike_var_raw",
                       "worse_qlike_var_holm", "dm_qlike_var_clu", "n_obs", "n_days"]]\
        .rename(columns={"better_qlike_var_holm": "A_better_holm",
                         "better_qlike_var_raw": "A_better_raw",
                         "worse_qlike_var_holm": "A_worse_holm",
                         "dm_qlike_var_clu": "A_dm", "n_obs": "A_n_obs",
                         "n_days": "A_n_days"})
    lb_all = []
    lb_flips = {}
    for tag in ("B", "C"):
        lbj = lbs[tag].merge(refA_flags, on=["disclosure", "horizon", "challenger"],
                             how="left", validate="1:1")
        lbj["verdict_flip_vs_A"] = lbj.better_holm != lbj.A_better_holm
        lb_flips[tag] = int(lbj.verdict_flip_vs_A.sum())
        lb_all.append(lbj)
    lb_all = pd.concat(lb_all, ignore_index=True)
    lb_sums = {tag: lb_summary(lbs[tag]) for tag in ("B", "C")}
    sumsA = {"n": 180,
             "better_holm": int(refA.better_qlike_var_holm.sum()),
             "better_raw": int(refA.better_qlike_var_raw.sum()),
             "worse_holm": int(refA.worse_qlike_var_holm.sum()),
             "text_better_holm": int(refA[~refA.challenger.isin(PRICE_CH)]
                                     .better_qlike_var_holm.sum()),
             "text_n": int((~refA.challenger.isin(PRICE_CH)).sum()),
             "price_better_holm": int(refA[refA.challenger.isin(PRICE_CH)]
                                      .better_qlike_var_holm.sum()),
             "winners": sorted((refA[refA.better_qlike_var_holm].disclosure + "/h"
                                + refA[refA.better_qlike_var_holm].horizon.astype(str)
                                + "/" + refA[refA.better_qlike_var_holm].challenger)
                               .tolist())}
    text_winner_cells = sorted(
        (lb_all[lb_all.better_holm & ~lb_all.is_price_arm].panel + ":"
         + lb_all[lb_all.better_holm & ~lb_all.is_price_arm].disclosure + "/h"
         + lb_all[lb_all.better_holm & ~lb_all.is_price_arm].horizon.astype(str) + "/"
         + lb_all[lb_all.better_holm & ~lb_all.is_price_arm].challenger).tolist())

    # ---------------- committed (panel A) cascade columns ----------------
    en = pd.read_csv(T / "m1_ensemble_primary.csv")
    fi = pd.read_csv(T / "firm_identity_ensemble.csv")
    mx = pd.read_csv(T / "maximal_reference_ensemble.csv")
    ci = pd.read_csv(T / "control_intersection_ensemble.csv")
    sip = pd.read_csv(T / "signal_injection_power.csv")
    old = en[GRIDKEY + ["n_test", "vol_qlike_R", "vol_rel_impr_pct", "vol_dm_q_clu",
                        "vol_dmq_holm_clu", "vol_placebo_dm_clu", "genuine_ens_vol"]].rename(
        columns={"n_test": "a_n_test", "vol_qlike_R": "a_qlike_R",
                 "vol_rel_impr_pct": "a_rel", "vol_dm_q_clu": "a_dm",
                 "vol_dmq_holm_clu": "a_holm", "vol_placebo_dm_clu": "a_placebo_dm",
                 "genuine_ens_vol": "a_genuine"})
    old = old.merge(ci[GRIDKEY + ["primary_holm", "firm_holm", "maximal_holm",
                                  "AND_full_holm"]].rename(
        columns={"primary_holm": "a_primary_detect", "firm_holm": "a_firm_detect",
                 "maximal_holm": "a_pool_detect", "AND_full_holm": "a_conj"}),
        on=GRIDKEY, validate="1:1")
    old = old.merge(fi[GRIDKEY + ["rel_impr_pct_firm", "dm_q_clustered"]].rename(
        columns={"rel_impr_pct_firm": "a_firm_rel", "dm_q_clustered": "a_firm_dm"}),
        on=GRIDKEY, validate="1:1")
    old = old.merge(mx[GRIDKEY + ["rel_impr_pct_maximal", "dm_q_clustered"]].rename(
        columns={"rel_impr_pct_maximal": "a_pool_rel", "dm_q_clustered": "a_pool_dm"}),
        on=GRIDKEY, validate="1:1")
    sip0 = sip[sip.target_pct == TARGETS[0]][GRIDKEY + ["mde_rel_pct"]].rename(
        columns={"mde_rel_pct": "a_mde"})
    old = old.merge(sip0, on=GRIDKEY, validate="1:1")

    cnt = {"A": {"primary": int(ci.primary_holm.sum()), "firm": int(ci.firm_holm.sum()),
                 "pool": int(ci.maximal_holm.sum()), "conj": int(ci.AND_full_holm.sum()),
                 "genuine": int(en.genuine_ens_vol.sum()),
                 "mde_median": float(old.a_mde.median()),
                 "mde_q25": float(old.a_mde.quantile(.25)),
                 "mde_q75": float(old.a_mde.quantile(.75)),
                 "placebo_ok": int((en.vol_placebo_dm_clu.abs() < 2.0).sum())},
           "B": stage_counts(results["B"]["base"]),
           "C": stage_counts(results["C"]["base"])}

    rec_old = {str(t): {"har": int(sip[sip.target_pct == t].har_detect.sum()),
                        "firm": int(sip[sip.target_pct == t].firm_detect.sum()),
                        "pool": int(sip[sip.target_pct == t].pool_detect.sum()),
                        "all3": int(sip[sip.target_pct == t].all3_detect.sum())}
               for t in TARGETS}

    branch, branch_note = fired_branch(cnt, lb_flips, text_winner_cells)

    # ---------------- final per-cell CSV ----------------
    out = stage_cols(results["B"]["base"], "b").merge(
        stage_cols(results["C"]["base"], "c").drop(columns=["n_seeds"]),
        on=GRIDKEY, validate="1:1").merge(old, on=GRIDKEY, validate="1:1")
    for tag in ("B", "C"):
        inj = results[tag]["inj"]
        for tgt in TARGETS:
            lab = f"{tag.lower()}_i{str(tgt).replace('.', '')}"
            sl = inj[inj.target_pct == tgt][GRIDKEY + ["har_detect", "firm_detect",
                                                       "pool_detect", "all3_detect"]]
            sl = sl.rename(columns={c: f"{lab}_{c}" for c in
                                    ("har_detect", "firm_detect", "pool_detect",
                                     "all3_detect")})
            out = out.merge(sl, on=GRIDKEY, validate="1:1")
    out = out.sort_values(GRIDKEY).reset_index(drop=True)

    # single-shot re-check immediately before write
    if FINAL_CSV.exists() or LB_CSV.exists():
        print("REFUSED AT WRITE TIME: final table appeared during the run.")
        return 4
    out.to_csv(FINAL_CSV, index=False)
    lb_all.to_csv(LB_CSV, index=False)
    write_md(cnt, results, old, out, branch, branch_note, rec_old, g1, meta,
             lb_sums, sumsA, lb_all, lb_flips, text_winner_cells, n_cov, len(rb))
    print(f"\n[final] wrote {FINAL_CSV} + {FINAL_MD} + {LB_CSV} in {time.time() - t0:.1f}s")
    print(json.dumps({"counts": cnt, "leaderboard": {"A": sumsA, **lb_sums},
                      "lb_verdict_flips_vs_A": lb_flips, "branch": branch}, indent=2,
                     default=str))
    return 0


def _fmt(x, p="+.2f"):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, p)


def write_md(cnt, results, old, out, branch, branch_note, rec_old, g1, meta,
             lb_sums, sumsA, lb_all, lb_flips, text_winner_cells, n_cov, n_rows):
    drift = meta["refetch_drift_vs_committed_label_parity"]
    covm = meta["coverage"]
    sc = covm["per_split"]
    trade = (f"coverage: train {100 * sc['train']['coverage_clean']:.1f}% / "
             f"val {100 * sc['val']['coverage_clean']:.1f}% / "
             f"test {100 * sc['test']['coverage_clean']:.1f}% of modelled rows; "
             f"benchmark-row clean coverage {100 * n_cov / n_rows:.2f}%; exit-firm rows "
             f"{100 * covm['exit_firm_row_coverage']:.1f}% vs active "
             f"{100 * covm['active_firm_row_coverage']:.1f}%")
    md = ["# PREREG H — licence-free public-price variant: three-panel full cascade "
          "+ standalone leaderboard\n",
          "Pre-registered in configs/prereg_public_variant.md (tag prereg-h-v1.0), "
          "single-shot. Panels: **A** = full panel + CRSP labels (committed anchor, "
          "G1-reproduced); **B** = public-coverage rows + CRSP labels (survivorship "
          "isolated); **C** = public-coverage rows + PUBLIC labels/features (the "
          "shippable variant). Text-arm predictions FROZEN (3-seed ensemble, the "
          "declared primary object); combiner/recalibration + firm-mean refit on "
          "validation with each panel's own rows/labels; panel C refits A2 + A6_shar "
          "on public features+labels via the committed fitting code, A3/A4/A5 frozen "
          "label-free (val-recalibrated). Day-clustered DM, Holm within each "
          "pre-declared family, placebo gate, per-cell MDE + injection recovery — "
          "machinery verbatim from the committed cascade (G1-gated below).\n",
          f"**THE TRADE, priced openly (recomputed from this refetch):** {trade}.\n"]
    if drift["FLAGGED"]:
        md.append("> **REFETCH-DRIFT FLAG (prereg threshold exceeded):** clean coverage "
                  f"drift {drift['coverage_clean_drift_pp']:+.3f}pp (threshold 0.5pp) "
                  f"and/or Pearson drift {drift['pearson_drift']:+.6f} (threshold "
                  "0.001) vs the committed label_parity table — every number below "
                  "carries this caveat.\n")

    md.append("## FIRED BRANCH: **" + branch + "**\n")
    md.append(branch_note + "\n")
    md.append("**Branch operationalisation (fixed before any B/C statistic was read):** "
              "(c) fires iff conjunction>0 on B or C OR any text/fusion arm (any "
              "challenger outside {A3_garch, A4_egarch, A5_arima}) becomes a Holm "
              "standalone winner on B or C; (a) fires iff conjunction stays 0/69 on both "
              "panels AND all 180 per-comparison standalone Holm verdicts match panel A "
              "on B and on C; (b) otherwise. Panel A's committed standalone verdicts "
              "already contain price-arm winners "
              f"({sumsA['better_holm']}/180, all GARCH-family), so 'a standalone winner "
              "appears' is read against the paper's verdict object: the TEXT standalone "
              "null.\n")

    md.append("## A-vs-B-vs-C decomposition (A-B = survivorship; B-C = label source)\n")
    md.append(f"Cascade cells: 69 per panel. {trade}.\n")
    md.append("| object | A (committed anchor) | **B (covered, CRSP labels)** | "
              "**C (covered, public labels)** | A-B (survivorship) | B-C (label source) |")
    md.append("|---|---|---|---|---|---|")
    for tag, label in (("primary", "primary: text over recalibrated HAR (Holm)"),
                       ("firm", "firm-identity-augmented reference (Holm)"),
                       ("pool", "maximal 5-price pool (Holm)"),
                       ("conj", "**full conjunction (primary AND firm AND pool)**"),
                       ("genuine", "placebo-gated genuine (primary stage)")):
        a, b, c = cnt["A"][tag], cnt["B"][tag], cnt["C"][tag]
        md.append(f"| {label} | {a}/69 | **{b}/69** | **{c}/69** | {a - b:+d} | {b - c:+d} |")
    md.append(f"| median MDE_rel% | {cnt['A']['mde_median']:.3f} | "
              f"**{cnt['B']['mde_median']:.3f}** | **{cnt['C']['mde_median']:.3f}** | "
              f"{cnt['A']['mde_median'] - cnt['B']['mde_median']:+.3f} | "
              f"{cnt['B']['mde_median'] - cnt['C']['mde_median']:+.3f} |")
    md.append(f"| standalone better-than-A2 (Holm, of 180) | {sumsA['better_holm']} | "
              f"**{lb_sums['B']['better_holm']}** | **{lb_sums['C']['better_holm']}** | "
              f"{sumsA['better_holm'] - lb_sums['B']['better_holm']:+d} | "
              f"{lb_sums['B']['better_holm'] - lb_sums['C']['better_holm']:+d} |")
    md.append(f"| ... of which TEXT/fusion arms | {sumsA['text_better_holm']}/"
              f"{sumsA['text_n']} | **{lb_sums['B']['text_better_holm']}/"
              f"{lb_sums['B']['text_n']}** | **{lb_sums['C']['text_better_holm']}/"
              f"{lb_sums['C']['text_n']}** |  |  |")
    md.append(f"| standalone significantly WORSE (Holm) | {sumsA['worse_holm']} | "
              f"{lb_sums['B']['worse_holm']} | {lb_sums['C']['worse_holm']} |  |  |")
    md.append("")

    md.append("## Standalone leaderboard — day-clustered variance-unit QLIKE DM vs A2 "
              "(committed convention; Holm within each (disclosure, horizon) group)\n")
    md.append(f"Universe: the dm_pairwise_clustered.csv 180 comparisons (20 "
              f"seed-ensembled challengers x 3 disclosures x 3 horizons). {trade}.\n")
    md.append("| panel | better raw | better Holm | text/fusion better Holm | "
              "worse Holm | Holm winners |")
    md.append("|---|---|---|---|---|---|")
    for tag, s in (("A (committed)", sumsA), ("B", lb_sums["B"]), ("C", lb_sums["C"])):
        md.append(f"| {tag} | {s['better_raw']} | {s['better_holm']} | "
                  f"{s['text_better_holm']}/{s['text_n']} | {s['worse_holm']} | "
                  f"{'; '.join(s['winners']) if s['winners'] else 'none'} |")
    md.append("")
    md.append(f"Per-comparison Holm-verdict flips vs panel A: **B {lb_flips['B']}/180, "
              f"C {lb_flips['C']}/180**." +
              ("" if not (lb_flips["B"] or lb_flips["C"]) else " Flipped comparisons:"))
    fl = lb_all[lb_all.verdict_flip_vs_A]
    if len(fl):
        md.append("\n| panel | disclosure | h | challenger | A verdict (DM) | "
                  "panel verdict (DM, Holm) |")
        md.append("|---|---|---|---|---|---|")
        for _, r in fl.iterrows():
            md.append(f"| {r.panel} | {r.disclosure} | {r.horizon} | {r.challenger} | "
                      f"{'BETTER' if r.A_better_holm else 'not-better'} "
                      f"({_fmt(r.A_dm)}) | {'BETTER' if r.better_holm else 'not-better'} "
                      f"({_fmt(r.dm_qlike_var_clu)}, Holm "
                      f"{r.holm_qlike_var_clu:.4f}) |")
    md.append("")

    md.append("## Injection recovery — Holm-detected /69 per pre-declared "
              "(stage, level) family\n")
    md.append("| level | A (committed) HAR/firm/pool/all3 | **B** HAR/firm/pool/all3 | "
              "**C** HAR/firm/pool/all3 |")
    md.append("|---|---|---|---|")
    for t in TARGETS:
        ro, rb_, rc = rec_old[str(t)], results["B"]["rec"][t], results["C"]["rec"][t]
        md.append(f"| {t:.1f}% | {ro['har']}/{ro['firm']}/{ro['pool']}/{ro['all3']} | "
                  f"**{rb_['har']}/{rb_['firm']}/{rb_['pool']}/{rb_['all3']}** | "
                  f"**{rc['har']}/{rc['firm']}/{rc['pool']}/{rc['all3']}** |")
    md.append("")

    md.append("## MDE — per panel (80% power, 5% two-sided)\n")
    md.append("| | A (committed) | B | C |")
    md.append("|---|---|---|---|")
    md.append(f"| median MDE_rel% | {cnt['A']['mde_median']:.3f} | "
              f"{cnt['B']['mde_median']:.3f} | {cnt['C']['mde_median']:.3f} |")
    md.append(f"| IQR | [{cnt['A']['mde_q25']:.2f}, {cnt['A']['mde_q75']:.2f}] | "
              f"[{cnt['B']['mde_q25']:.2f}, {cnt['B']['mde_q75']:.2f}] | "
              f"[{cnt['C']['mde_q25']:.2f}, {cnt['C']['mde_q75']:.2f}] |")
    md.append("")

    rk_or = pd.DataFrame(g1["single_ref_a2_rank_orig"])
    md.append("Reference-ordering diagnostic (A2 rank among the 5 single recalibrated "
              "price references, per disc x h): A mean rank "
              f"{rk_or.a2_rank.mean():.2f} (rank-1 in "
              f"{int((rk_or.a2_rank == 1).sum())}/6); "
              + "; ".join(f"{tag} mean rank "
                          f"{pd.DataFrame(results[tag]['rankings']).a2_rank.mean():.2f} "
                          f"(rank-1 in "
                          f"{int((pd.DataFrame(results[tag]['rankings']).a2_rank == 1).sum())}/6)"
                          for tag in ("B", "C")) + ".\n")

    md.append("## Per-cell detail — panels B and C vs the committed panel A\n")
    md.append("| disc | model | h | n_test A->B->C | rel% A->B->C (primary) | "
              "DM A->B->C | detect A H/F/P | detect B H/F/P | detect C H/F/P | "
              "placebo B/C | MDE A->B->C |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        md.append(
            f"| {r.disc} | {r.model} | {r.h} | {int(r.a_n_test)}->{int(r.b_n_test)}->"
            f"{int(r.c_n_test)} | {_fmt(r.a_rel)}->{_fmt(r.b_primary_rel)}->"
            f"{_fmt(r.c_primary_rel)} | {_fmt(r.a_dm)}->{_fmt(r.b_primary_dm)}->"
            f"{_fmt(r.c_primary_dm)} | "
            f"{'Y' if r.a_primary_detect else '.'}/{'Y' if r.a_firm_detect else '.'}/"
            f"{'Y' if r.a_pool_detect else '.'} | "
            f"{'Y' if r.b_primary_detect else '.'}/{'Y' if r.b_firm_detect else '.'}/"
            f"{'Y' if r.b_pool_detect else '.'} | "
            f"{'Y' if r.c_primary_detect else '.'}/{'Y' if r.c_firm_detect else '.'}/"
            f"{'Y' if r.c_pool_detect else '.'} | "
            f"{_fmt(r.b_placebo_dm)}/{_fmt(r.c_placebo_dm)} | "
            f"{r.a_mde:.2f}->{r.b_mde:.2f}->{r.c_mde:.2f} |")
    md.append("")
    flips_ab = out[out.a_primary_detect.astype(bool) != out.b_primary_detect.astype(bool)]
    flips_bc = out[out.b_primary_detect.astype(bool) != out.c_primary_detect.astype(bool)]
    md.append(f"Primary-stage per-cell detection flips: A->B {len(flips_ab)}/69, "
              f"B->C {len(flips_bc)}/69 (composition detail above; verdict objects in "
              "the decomposition table).\n")

    md.append("## Refetch-drift disclosure (prereg G3; committed label_parity vs this "
              "refetch)\n")
    md.append("| quantity | committed (2026-07-09 fetch) | this refetch | drift | "
              "flag threshold |")
    md.append("|---|---|---|---|---|")
    md.append(f"| clean row coverage | {100 * drift['coverage_clean_committed']:.2f}% | "
              f"{100 * drift['coverage_clean_now']:.2f}% | "
              f"{drift['coverage_clean_drift_pp']:+.3f}pp | 0.5pp "
              f"{'**EXCEEDED**' if drift['flag_coverage'] else '(ok)'} |")
    md.append(f"| Pearson log-RV (covered modelled rows) | "
              f"{drift['pearson_committed']:.6f} | {drift['pearson_now']:.6f} | "
              f"{drift['pearson_drift']:+.6f} | 0.001 "
              f"{'**EXCEEDED**' if drift['flag_correlation'] else '(ok)'} |")
    md.append(f"| tickers with Yahoo data | {drift['n_yahoo_ok_committed']} | "
              f"{drift['n_yahoo_ok_now']} | "
              f"{drift['n_yahoo_ok_now'] - drift['n_yahoo_ok_committed']:+d} | — |")
    md.append(f"| symbol-mismatch screened | {drift['n_mismatch_committed']} | "
              f"{drift['n_mismatch_now']} | added: "
              f"{', '.join(drift['mismatch_added_vs_committed']) or 'none'}; dropped: "
              f"{', '.join(drift['mismatch_dropped_vs_committed']) or 'none'} | — |")
    md.append(f"| parity n (covered modelled rows) | {drift['parity_n_committed']:,} | "
              f"{drift['parity_n_now']:,} | "
              f"{drift['parity_n_now'] - drift['parity_n_committed']:+,d} | — |")
    md.append("")

    md.append("## Disclosures\n")
    losses_txt = {}
    for tag in ("B", "C"):
        ls = results[tag]["losses"]
        losses_txt[tag] = "; ".join(f"{r.disc} {r.what}: {int(r.lost)}/{int(r.n0)}"
                                    for _, r in ls.iterrows())
    md.append(
        "1. **Frozen text predictions (LIMITATION-FEEDER).** Every text arm was trained "
        "and tuned against the CRSP close-to-close RV target on the FULL panel; "
        "predictions are reused frozen and only the log-space recalibration/combiner "
        "weights are refit per panel. Panel-C readings are therefore conservative for "
        "the text side.\n"
        "2. **Yahoo terms bar redistribution.** The release artifact is the rebuild "
        "pipeline + fetch script (scripts/analysis/public_variant_labels.py), never the "
        "data; the price cache lives only in the session scratchpad. The "
        "symbol-mismatch screen itself needs CRSP — a licence-free builder cannot run "
        "it (inherited label_parity caveat; part of the variant's honest labelling).\n"
        "3. **A-block treatment (prereg).** Panel C: A2_har_rv refit (committed HARRV "
        "class: train split, per horizon, log OLS + Duan smearing) on public "
        "pub_1d/5d/22d + public label; A6_shar refit (stronger_baselines conventions "
        "incl. the BPQ insanity filter) with RS-/RS+ rebuilt from public signed daily "
        "returns at the feature-window end; A3/A4/A5 are label-free return-based "
        "forecasters — frozen, recalibrated on val inside the committed combination "
        "machinery. Panel B freezes ALL stored forecasts and only filters rows. The "
        "firm-identity reference term is the firm's own VAL-split mean of the panel's "
        "OWN label.\n"
        f"4. **Row losses (counted, reconciled to the coverage table).** Panel B: "
        f"{losses_txt['B']}. Panel C: {losses_txt['C']}. The label-verification gate is "
        "covered-rows-only on the public side: rows without a clean public label are "
        "counted (never scored); the CRSP-side reconstruction is machine-precision on "
        "ALL rows (labels build, gate L1).\n"
        "5. **Extra output file disclosed:** results/tables/public_variant_leaderboard.csv "
        "(per-comparison leaderboard detail for the three panels' 180-comparison "
        "universe with flip flags).\n"
        "6. **Single-shot:** this table was written once; the script refuses to "
        "overwrite it. Same seeds, same placebo permutations, same Holm families as "
        "the committed cascade.\n")

    md.append("## SANITY\n")
    md.append("| gate | result |")
    md.append("|---|---|")
    md.append(f"| G1 cascade path reproduces committed tables (panel A, original "
              f"labels) | **{'PASS' if g1['pass'] else 'FAIL'}** — max abs diff "
              f"{max(g1['max_abs_diffs_vs_committed'].values()):.2e} over "
              f"primary/firm/pool/MDE/placebo; counts "
              f"{g1['counts_this_run_vs_committed']}; injection counts match: "
              f"{g1['injection_counts_match']} |")
    md.append(f"| G1 refit machinery reproduces stored A2/A6_shar runs | max abs diff "
              f"{max(g1['refit_reproduction'].values()):.2e} (tol {REFIT_TOL:.0e}) |")
    lbd = g1["leaderboard_diffs_vs_standalone180"]
    md.append(f"| G1 leaderboard reproduces variance_unit_standalone180.csv (panel A) | "
              f"max abs diff {max(lbd['dm'], lbd['p'], lbd['holm']):.2e}; verdict "
              f"mismatches {lbd['verdict_mismatch']}/180 — "
              f"**{'PASS' if g1['leaderboard_pass'] else 'FAIL'}** |")
    gl = meta["gates"]
    md.append(f"| L1 CRSP label reconstruction (labels build) | n="
              f"{gl['L1_crsp_label_reconstruction']['n_rows']:.0f}, unreconstructed="
              f"{gl['L1_crsp_label_reconstruction']['n_unreconstructed']:.0f}, max abs "
              f"diff {gl['L1_crsp_label_reconstruction']['max_abs_diff']:.1e} — "
              f"**{'PASS' if gl['L1_crsp_label_reconstruction']['passed'] else 'FAIL'}** |")
    md.append(f"| L1b feature windows (CRSP side, TickerSeries machinery) | max abs "
              f"diff {max(gl['L1b_feature_windows']['max_abs_dr1'], gl['L1b_feature_windows']['max_abs_drv5'], gl['L1b_feature_windows']['max_abs_drv22']):.1e} "
              f"(tol 1e-8) — **PASS** |")
    md.append(f"| L1c one-mask consistency + L2 A2-QLIKE anchor | {gl['L1c_mask_consistency']}; "
              f"{gl['L2_a2_qlike_anchor']} |")
    md.append(f"| G2 public-label parity (gate >= 0.99) | Pearson(logRV)="
              f"{gl['G2_parity']['pearson_logRV']:.6f} on "
              f"{gl['G2_parity']['n']:,} covered modelled rows — **PASS** |")
    md.append(f"| G3 coverage reconciliation | parquet covered rows == coverage-table "
              f"covered rows (assert); drift table above; FLAGGED={drift['FLAGGED']} |")
    for tag in ("B", "C"):
        base = results[tag]["base"]
        md.append(f"| G4 placebo (panel {tag}) | |placebo DM|<2 in "
                  f"{int((base.placebo_dm_clu.abs() < 2.0).sum())}/69 cells "
                  f"(committed convention) |")
    md.append(f"| Per-cell n_test totals | A {int(out.a_n_test.sum()):,} -> B "
              f"{int(out.b_n_test.sum()):,} -> C {int(out.c_n_test.sum()):,} obs "
              f"across 69 cells (max per-cell loss A->B "
              f"{int((out.a_n_test - out.b_n_test).max())}, B->C "
              f"{int((out.b_n_test - out.c_n_test).max())}) |")
    md.append("")
    FINAL_MD.write_text("\n".join(md))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["g1", "final"], required=True)
    args = ap.parse_args()
    if args.mode == "g1":
        return run_g1()
    return run_final()


if __name__ == "__main__":
    sys.exit(main())
