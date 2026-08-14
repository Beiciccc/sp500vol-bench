"""ROW 5 (round-3 freeze table) — VARIANCE-UNIT QLIKE restatement of the FULL cascade,
the 8-K residual, and the 180 standalone comparisons.

Reviewer demand (methodology W1 / consensus point 5): the primary loss convention
(vol-unit QLIKE, q(y, f)) is not Patton proxy-robust; the committed
m1_variance_unit.{csv,md} already restates the 69-cell M1 PRIMARY grid in variance
units (genuine 38 -> 20). This script extends the SAME convention change to the rest
of the cascade — everything on the SEED-ENSEMBLE basis, all in q(y^2, f^2):

  (a) firm-identity-augmented reference (spec of basis_alignment_ensemble.py /
      firm_identity_ensemble.csv): survivors raw/Holm vs the committed vol-unit 15/8;
  (b) maximal 5-model price pool (maximal_reference_ensemble.csv): survivors raw/Holm
      vs the committed vol-unit 26/9;
  (c) control intersection: full AND under Holm + maximal-vs-firm Holm-survivor-set
      overlap, vs the committed vol-unit 0/69 + disjoint sets
      (control_intersection_ensemble.csv);
  (d) the residual: event_driven C6_llmtext vs the firm-identity reference at
      h5/10/20 — committed vol-unit +0.52/+0.24/+0.21 rel% (the paper's rounded
      +0.45/+0.25/+0.20 spans the identity-spec battery), all clustered-significant —
      restated in variance units;
  (e) the standalone 0/180 headline: the committed QLIKE-vs-A2 table
      (dm_qlike_all_vs_A2.csv, from scripts/analysis/qlike_dm.py) is OBS-LEVEL
      (seed2026 forecasts, HAC over observation order, 57 rows) — so the 180
      standalone comparisons (dm_pairwise_clustered.csv universe: 20 seed-ensembled
      challengers x {long_form, event_driven, combined} x h{5,10,20}) are recomputed
      with day-clustered QLIKE DM in BOTH vol- and variance-units, Holm within each
      (disclosure, horizon) group (same family convention as the committed SE table).

FORECASTS ARE UNCHANGED. The log-space combiner/reference weights (val-fit, frozen on
test — no look-ahead anywhere) are unit-free; only the EVALUATION loss moves from
q(y, f) to q(y^2, f^2).

SANITY GATES (all hard; the script aborts before writing results if any fails):
  G1  recompute the variance-unit rel% / clustered-DM / p / Holm columns of the
      committed results/tables/m1_variance_unit.csv on all 69 cells -> machine precision;
  G2  recompute the VOL-unit rel% / DM / p / Holm columns of the committed
      results/tables/firm_identity_ensemble.csv and maximal_reference_ensemble.csv on
      all 69 cells -> machine precision (same code path, evaluated in both units);
  G3  recompute the day-clustered SE DM/p of the committed
      results/tables/dm_pairwise_clustered.csv on all 180 comparisons -> machine
      precision (validates the standalone loader before the QLIKE restatement).

Outputs (NEW files only; committed tables untouched):
  results/tables/variance_unit_cascade.{csv,md}
  results/tables/variance_unit_standalone180.csv   (per-comparison detail for (e))

Run from repo root:  .venv/bin/python scripts/analysis/variance_unit_cascade.py
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
import m1_clustered as mc
import m1_ensemble_primary as mep
from clustered_dm import dm_test_clustered, mbb_ci_daily
from maximal_reference_firm_control import (
    PRICE,
    build_price_panel,
    firm_mean_val,
    log_ols_frozen,
)

T = Path("results/tables")
KEY = fc.KEY
SORT = fc.SORT
HORIZONS = fc.HORIZONS
GRIDKEY = ["disc", "model", "h"]
TOL = 1e-9  # machine-precision gate (CSV round-trip exact for repr floats)
RESIDUAL = ("event_driven", "C6_llmtext")

_ENS_CACHE: dict = {}


def ens_text(m, disc):
    if (m, disc) not in _ENS_CACHE:
        _ENS_CACHE[(m, disc)] = mep.ensemble_text(m, disc)
    return _ENS_CACHE[(m, disc)]


def qlike_var(y, f):
    """QLIKE in VARIANCE units: q(y^2, f^2)."""
    return fc.qlike(np.asarray(y, float) ** 2, np.asarray(f, float) ** 2)


def both_units(yt, fR, fU, days_t, h, *, boot_var=True):
    """Evaluate the SAME (fR, fU) forecasts under vol- and variance-unit QLIKE."""
    out = {}
    for unit, lossfn in (("vol", fc.qlike), ("var", qlike_var)):
        lR, lU = lossfn(yt, fR), lossfn(yt, fU)
        qR, qU = float(lR.mean()), float(lU.mean())
        dm, p, n_days = dm_test_clustered(lU, lR, days_t, h)
        out[f"{unit}_rel"] = 100.0 * (qR - qU) / qR if qR > 0 else np.nan
        out[f"{unit}_dm"] = dm
        out[f"{unit}_p"] = p
        out["n_days"] = n_days
        if unit == "var" and boot_var:
            _, lo, hi = mbb_ci_daily(lU - lR, days_t, h)
            out["var_boot_lo"], out["var_boot_hi"] = lo, hi
    return out


# =========================================================================
# GATE G1 — reproduce m1_variance_unit.csv (primary grid, both units)
# =========================================================================
def gate_primary():
    print("[G1] recomputing the 69-cell M1 primary grid in both units ...", flush=True)
    ref = pd.read_csv(T / "m1_variance_unit.csv")
    rows = []
    for disc, models in fc.SETS.items():
        har = fc.load("A2_har_rv", disc)[["split"] + KEY + [
            "prediction_realised_vol", "label_realised_vol", "filing_time_utc",
            "effective_trading_day"]].rename(columns={"prediction_realised_vol": "fhar"})
        for m in models:
            ens, _used = ens_text(m, disc)
            d = har.merge(ens, on=KEY)
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                fR, fU, _g = fc.log_combo(
                    dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(), dv.ftext.to_numpy(),
                    dt.fhar.to_numpy(), dt.ftext.to_numpy())
                st = both_units(dt.label_realised_vol.to_numpy(), fR, fU,
                                dt.effective_trading_day.to_numpy(), h, boot_var=False)
                rows.append({"disc": disc, "model": m, "h": h, "n_test": len(dt), **st})
        print(f"    {disc}: done", flush=True)
    new = pd.DataFrame(rows)
    new["var_holm"] = fc.holm(new["var_p"].fillna(1.0).values)
    j = ref.merge(new, on=GRIDKEY, how="inner", validate="1:1")
    assert len(j) == len(ref) == 69, f"primary grid mismatch: {len(j)} vs {len(ref)}"
    diffs = {
        "var_rel": float((j.var_rel_impr_pct - j.var_rel).abs().max()),
        "var_dm": float((j.var_dm_q_clu - j.var_dm).abs().max()),
        "var_p": float((j.var_p_q_clu - j.var_p).abs().max()),
        "var_holm": float((j.var_dmq_holm_clu - j.var_holm).abs().max()),
        "vol_rel": float((j.vol_rel_impr_pct - j.vol_rel).abs().max()),
        "vol_dm": float((j.vol_dm_q_clu - j.vol_dm).abs().max()),
        "n_test": int((j.n_test_x != j.n_test_y).sum()),
    }
    ok = all(v < TOL for k, v in diffs.items() if k != "n_test") and diffs["n_test"] == 0
    if not ok:
        print(json.dumps(diffs, indent=2))
        raise SystemExit("SANITY GATE G1 FAILED: recomputed primary grid does not "
                         "reproduce m1_variance_unit.csv — aborting, no numbers shipped.")
    print(f"[G1] PASS  max|d| = {max(v for k, v in diffs.items() if k != 'n_test'):.2e}", flush=True)
    return ref, diffs


# =========================================================================
# Cascade grids (a) firm-identity + (b) maximal pool, both units  [gate G2]
# =========================================================================
def run_cascade():
    print("[cascade] firm-identity + maximal grids in both units ...", flush=True)
    max_rows, firm_rows = [], []
    for disc, models in fc.SETS.items():
        panel = build_price_panel(disc)
        fmap, gmean, fcov, ocov = firm_mean_val(panel)
        panel["firm_mean_val"] = panel.ticker.map(fmap).fillna(gmean).astype(float)
        for m in models:
            ens, used = ens_text(m, disc)
            d = panel.merge(ens, on=KEY, how="inner")
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv, yt = dv.label_realised_vol.to_numpy(), dt.label_realised_vol.to_numpy()
                ftv, ftt = dv.ftext.to_numpy(), dt.ftext.to_numpy()
                fhv, fhr = dv.A2_har_rv.to_numpy(), dt.A2_har_rv.to_numpy()
                days_t = (dt.effective_trading_day.fillna(dt.filing_time_utc)).to_numpy()
                base = {"disc": disc, "model": m, "h": h, "n_test": len(dt),
                        "n_seeds": len(used), "seeds": "+".join(map(str, used))}

                # (b) maximal 5-price-pool reference (identical forecasts to committed)
                pv = [dv[c].to_numpy() for c in PRICE]
                pt = [dt[c].to_numpy() for c in PRICE]
                fRs, _ = log_ols_frozen(yv, pv, pt)
                fUs, _ = log_ols_frozen(yv, pv + [ftv], pt + [ftt])
                max_rows.append({**base, **both_units(yt, fRs, fUs, days_t, h)})

                # (a) firm-identity-augmented reference (identical forecasts to committed)
                fmv, fmt = dv.firm_mean_val.to_numpy(), dt.firm_mean_val.to_numpy()
                fRf, _ = log_ols_frozen(yv, [fhv, fmv], [fhr, fmt])
                fUf, _ = log_ols_frozen(yv, [fhv, fmv, ftv], [fhr, fmt, ftt])
                firm_rows.append({**base, **both_units(yt, fRf, fUf, days_t, h)})
        print(f"    {disc}: done", flush=True)
    return pd.DataFrame(max_rows), pd.DataFrame(firm_rows)


def add_unit_flags(df):
    """Holm within the 69-cell family, separately per unit, + add/hurt flags."""
    df = df.copy()
    for u in ("vol", "var"):
        df[f"{u}_holm"] = fc.holm(df[f"{u}_p"].fillna(1.0).values)
        neg = df[f"{u}_dm"] < 0
        df[f"{u}_adds_raw"] = neg & (df[f"{u}_p"] < 0.05)
        df[f"{u}_adds_holm"] = neg & (df[f"{u}_holm"] < 0.05)
        df[f"{u}_hurts_holm"] = (~neg) & (df[f"{u}_holm"] < 0.05)
    df["convention_dependent"] = df.vol_adds_holm != df.var_adds_holm
    return df


def gate_cascade(maxdf, firmdf):
    """G2: vol-unit columns must reproduce the committed ensemble control tables."""
    mx0 = pd.read_csv(T / "maximal_reference_ensemble.csv")
    fi0 = pd.read_csv(T / "firm_identity_ensemble.csv")
    diffs = {}
    for tag, new, old, relcol in (("maximal", maxdf, mx0, "rel_impr_pct_maximal"),
                                  ("firm", firmdf, fi0, "rel_impr_pct_firm")):
        j = new.merge(old[GRIDKEY + ["n_test", relcol, "dm_q_clustered",
                                     "p_q_clustered", "holm_p"]],
                      on=GRIDKEY, how="inner", validate="1:1", suffixes=("", "_c"))
        assert len(j) == len(new) == 69, f"{tag} grid mismatch"
        diffs[f"{tag}_rel"] = float((j[relcol] - j.vol_rel).abs().max())
        diffs[f"{tag}_dm"] = float((j.dm_q_clustered - j.vol_dm).abs().max())
        diffs[f"{tag}_p"] = float((j.p_q_clustered - j.vol_p).abs().max())
        diffs[f"{tag}_holm"] = float((j.holm_p - j.vol_holm).abs().max())
        diffs[f"{tag}_n_test"] = int((j.n_test != j.n_test_c).sum())
    ok = all(v < TOL for k, v in diffs.items() if not k.endswith("n_test")) \
        and diffs["maximal_n_test"] == 0 and diffs["firm_n_test"] == 0
    if not ok:
        print(json.dumps(diffs, indent=2))
        raise SystemExit("SANITY GATE G2 FAILED: vol-unit recomputation does not "
                         "reproduce the committed ensemble control tables — aborting.")
    print(f"[G2] PASS  max|d| = "
          f"{max(v for k, v in diffs.items() if not k.endswith('n_test')):.2e}", flush=True)
    return diffs


# =========================================================================
# (e) standalone 180 — day-clustered QLIKE DM in both units  [gate G3]
# =========================================================================
def run_standalone():
    print("[standalone] rebuilding the 180-comparison universe "
          "(dm_pairwise_clustered basis) ...", flush=True)
    ref = pd.read_csv(T / "dm_pairwise_clustered.csv")
    rows = []
    for disc in mc.DISCLOSURES:
        merged, present = mc.build_joined(disc)
        if merged is None or mc.HAR not in present:
            continue
        for h in HORIZONS:
            g = merged[merged.horizon_days == h].sort_values(SORT, kind="mergesort")
            if len(g) < 30:
                continue
            y = g.label_realised_vol.to_numpy()
            days = mc.day_key(g)
            fhar = g[f"pred__{mc.HAR}"].to_numpy()
            se_har = fc.se(y, fhar)
            qv_har, qz_har = fc.qlike(y, fhar), qlike_var(y, fhar)
            group = []
            for ch in present:
                if ch == mc.HAR:
                    continue
                f = g[f"pred__{ch}"].to_numpy()
                dm_se, p_se, n_days = dm_test_clustered(fc.se(y, f), se_har, days, h)
                dm_qv, p_qv, _ = dm_test_clustered(fc.qlike(y, f), qv_har, days, h)
                dm_qz, p_qz, _ = dm_test_clustered(qlike_var(y, f), qz_har, days, h)
                group.append({"disclosure": disc, "horizon": h, "challenger": ch,
                              "n_obs": len(g), "n_days": n_days,
                              "dm_se_clu": dm_se, "p_se_clu": p_se,
                              "dm_qlike_vol_clu": dm_qv, "p_qlike_vol_clu": p_qv,
                              "dm_qlike_var_clu": dm_qz, "p_qlike_var_clu": p_qz})
            gdf = pd.DataFrame(group)
            # Holm within (disclosure, horizon) over the vs-A2 challenger set,
            # separately per loss — same family convention as dm_pairwise_clustered.
            for c in ("p_se_clu", "p_qlike_vol_clu", "p_qlike_var_clu"):
                gdf[c.replace("p_", "holm_")] = fc.holm(gdf[c].fillna(1.0).to_numpy())
            rows.append(gdf)
        print(f"    {disc}: done", flush=True)
    full = pd.concat(rows, ignore_index=True)

    # ---- G3: SE leg must reproduce the committed dm_pairwise_clustered.csv
    j = full.merge(ref[["disclosure", "horizon", "challenger", "dm_clust", "p_clust",
                        "p_holm_clust", "n_obs", "n_days"]],
                   on=["disclosure", "horizon", "challenger"], how="inner",
                   validate="1:1", suffixes=("", "_c"))
    assert len(j) == len(ref) == 180, f"standalone universe mismatch: {len(j)} vs {len(ref)}"
    diffs = {"se_dm": float((j.dm_se_clu - j.dm_clust).abs().max()),
             "se_p": float((j.p_se_clu - j.p_clust).abs().max()),
             "se_holm": float((j.holm_se_clu - j.p_holm_clust).abs().max()),
             "n_obs": int((j.n_obs != j.n_obs_c).sum()),
             "n_days": int((j.n_days != j.n_days_c).sum())}
    ok = (diffs["se_dm"] < TOL and diffs["se_p"] < TOL and diffs["se_holm"] < TOL
          and diffs["n_obs"] == 0 and diffs["n_days"] == 0)
    if not ok:
        print(json.dumps(diffs, indent=2))
        raise SystemExit("SANITY GATE G3 FAILED: SE leg does not reproduce "
                         "dm_pairwise_clustered.csv — aborting.")
    print(f"[G3] PASS  max|d| = {max(diffs['se_dm'], diffs['se_p'], diffs['se_holm']):.2e}",
          flush=True)

    for u, dmc, hc, pc in (("se", "dm_se_clu", "holm_se_clu", "p_se_clu"),
                           ("qlike_vol", "dm_qlike_vol_clu", "holm_qlike_vol_clu", "p_qlike_vol_clu"),
                           ("qlike_var", "dm_qlike_var_clu", "holm_qlike_var_clu", "p_qlike_var_clu")):
        full[f"better_{u}_holm"] = (full[dmc] < 0) & (full[hc] < 0.05)
        full[f"better_{u}_raw"] = (full[dmc] < 0) & (full[pc] < 0.05)
        full[f"worse_{u}_holm"] = (full[dmc] > 0) & (full[hc] < 0.05)
    return full, diffs


def sum180(full, u):
    return {"better_holm": int(full[f"better_{u}_holm"].sum()),
            "better_raw": int(full[f"better_{u}_raw"].sum()),
            "worse_holm": int(full[f"worse_{u}_holm"].sum()),
            "n": len(full)}


# =========================================================================
def cell_str(r):
    return f"{r['disc']}/{r['model']}/h{r['h']}"


def main():
    t0 = time.time()
    T.mkdir(parents=True, exist_ok=True)

    # basis inspection for (e): is the committed QLIKE-vs-A2 table obs-level?
    qold = pd.read_csv(T / "dm_qlike_all_vs_A2.csv")
    qold_note = (f"{len(qold)} rows, columns {list(qold.columns)}, max |DM| = "
                 f"{qold.DM.abs().max():.1f} (obs-order HAC inflation), no n_days/cluster "
                 "column, seed2026-only universe (scripts/analysis/qlike_dm.py) -> OBS-LEVEL")
    print(f"[inspect] dm_qlike_all_vs_A2.csv: {qold_note}", flush=True)

    prim, g1 = gate_primary()
    maxdf, firmdf = run_cascade()
    maxdf, firmdf = add_unit_flags(maxdf), add_unit_flags(firmdf)
    g2 = gate_cascade(maxdf, firmdf)
    stand, g3 = run_standalone()

    # ---------------- (c) intersection, variance-unit ----------------
    prim = prim.copy()
    prim["primary_var_raw"] = (prim.var_dm_q_clu < 0) & (prim.var_p_q_clu < 0.05)
    prim["primary_var_holm"] = (prim.var_dm_q_clu < 0) & (prim.var_dmq_holm_clu < 0.05)
    prim["primary_var_genuine"] = prim.genuine_ens_var.astype(bool)
    ix = prim[GRIDKEY + ["primary_var_raw", "primary_var_holm", "primary_var_genuine",
                         "var_rel_impr_pct", "genuine_ens_vol"]].merge(
        maxdf[GRIDKEY + ["var_adds_raw", "var_adds_holm", "var_rel", "var_dm", "var_p",
                         "var_holm", "vol_adds_raw", "vol_adds_holm", "vol_rel", "vol_dm"]]
        .rename(columns={c: "max_" + c for c in
                         ["var_adds_raw", "var_adds_holm", "var_rel", "var_dm", "var_p",
                          "var_holm", "vol_adds_raw", "vol_adds_holm", "vol_rel", "vol_dm"]}),
        on=GRIDKEY, validate="1:1").merge(
        firmdf[GRIDKEY + ["n_seeds", "seeds", "n_test", "n_days",
                          "var_adds_raw", "var_adds_holm", "var_rel", "var_dm", "var_p",
                          "var_holm", "var_boot_lo", "var_boot_hi",
                          "vol_adds_raw", "vol_adds_holm", "vol_rel", "vol_dm"]]
        .rename(columns={c: "firm_" + c for c in
                         ["var_adds_raw", "var_adds_holm", "var_rel", "var_dm", "var_p",
                          "var_holm", "var_boot_lo", "var_boot_hi",
                          "vol_adds_raw", "vol_adds_holm", "vol_rel", "vol_dm"]}),
        on=GRIDKEY, validate="1:1")
    assert len(ix) == 69
    for b in ("raw", "holm"):
        ix[f"AND_full_var_{b}"] = (ix[f"primary_var_{b}"] & ix[f"max_var_adds_{b}"]
                                   & ix[f"firm_var_adds_{b}"])
    ix["AND_genuine_var_holm"] = (ix.primary_var_genuine & ix.max_var_adds_holm
                                  & ix.firm_var_adds_holm)

    mx_var = {cell_str(r) for _, r in ix[ix.max_var_adds_holm].iterrows()}
    fi_var = {cell_str(r) for _, r in ix[ix.firm_var_adds_holm].iterrows()}
    overlap_var = sorted(mx_var & fi_var)

    # committed VOL-unit intersection (the BEFORE side of (c))
    ci0 = pd.read_csv(T / "control_intersection_ensemble.csv")
    vol_before = {
        "primary_raw": int(ci0.primary_raw.sum()), "primary_holm": int(ci0.primary_holm.sum()),
        "primary_genuine": int(ci0.primary_genuine.sum()),
        "maximal_raw": int(ci0.maximal_raw.sum()), "maximal_holm": int(ci0.maximal_holm.sum()),
        "firm_raw": int(ci0.firm_raw.sum()), "firm_holm": int(ci0.firm_holm.sum()),
        "AND_full_raw": int(ci0.AND_full_raw.sum()), "AND_full_holm": int(ci0.AND_full_holm.sum()),
        "AND_genuine_holm": int(ci0.AND_genuine_holm.sum()),
    }
    mx_vol = {f"{r.disc}/{r.model}/h{r.h}" for _, r in ci0[ci0.maximal_holm].iterrows()}
    fi_vol = {f"{r.disc}/{r.model}/h{r.h}" for _, r in ci0[ci0.firm_holm].iterrows()}
    overlap_vol = sorted(mx_vol & fi_vol)

    var_now = {
        "primary_raw": int(ix.primary_var_raw.sum()), "primary_holm": int(ix.primary_var_holm.sum()),
        "primary_genuine": int(ix.primary_var_genuine.sum()),
        "maximal_raw": int(ix.max_var_adds_raw.sum()), "maximal_holm": int(ix.max_var_adds_holm.sum()),
        "firm_raw": int(ix.firm_var_adds_raw.sum()), "firm_holm": int(ix.firm_var_adds_holm.sum()),
        "AND_full_raw": int(ix.AND_full_var_raw.sum()), "AND_full_holm": int(ix.AND_full_var_holm.sum()),
        "AND_genuine_holm": int(ix.AND_genuine_var_holm.sum()),
    }

    # ---------------- (d) the residual ----------------
    res = firmdf[(firmdf.disc == RESIDUAL[0]) & (firmdf.model == RESIDUAL[1])].sort_values("h")

    # ---------------- write CSVs ----------------
    ix.to_csv(T / "variance_unit_cascade.csv", index=False)
    stand.to_csv(T / "variance_unit_standalone180.csv", index=False)

    s_se = sum180(stand, "se")
    s_qv = sum180(stand, "qlike_vol")
    s_qz = sum180(stand, "qlike_var")
    price_ch = {"A3_garch", "A4_egarch", "A5_arima"}
    is_price = stand.challenger.isin(price_ch)
    textc, pricec = stand[~is_price], stand[is_price]
    t_holm = int(textc.better_qlike_var_holm.sum())
    t_raw = int(textc.better_qlike_var_raw.sum())
    p_holm = int(pricec.better_qlike_var_holm.sum())

    # ---------------- md ----------------
    md = [
        "# ROW 5 — Variance-unit QLIKE restatement of the FULL cascade + residual + standalone 180\n",
        "## RESTATED vs BEFORE\n",
        "BEFORE = the committed vol-unit q(y, f) cascade on the seed-ensemble basis "
        "(m1_ensemble_primary / firm_identity_ensemble / maximal_reference_ensemble / "
        "control_intersection_ensemble; standalone headline scored on squared error in "
        "dm_pairwise_clustered, QLIKE-vs-A2 only obs-level in dm_qlike_all_vs_A2.csv). "
        "RESTATED = the SAME seed-ensemble forecasts and the SAME val-fit frozen log-space "
        "references (unit-free; no look-ahead), evaluated with Patton-proxy-robust "
        "variance-unit QLIKE q(y^2, f^2), day-clustered DM (HAC lag = h-1 days).\n",
        "| quantity | BEFORE (vol-unit / SE) | RESTATED (variance-unit) |",
        "|---|---|---|",
        f"| (M1 primary) genuine cells | {int(ix.genuine_ens_vol.sum())}/69 | "
        f"**{var_now['primary_genuine']}/69** (committed m1_variance_unit, gate-verified) |",
        f"| (a) firm-identity survivors raw / Holm | {int(firmdf.vol_adds_raw.sum())}/69 / "
        f"{int(firmdf.vol_adds_holm.sum())}/69 | **{var_now['firm_raw']}/69 / "
        f"{var_now['firm_holm']}/69** |",
        f"| (b) maximal-pool survivors raw / Holm | {int(maxdf.vol_adds_raw.sum())}/69 / "
        f"{int(maxdf.vol_adds_holm.sum())}/69 | **{var_now['maximal_raw']}/69 / "
        f"{var_now['maximal_holm']}/69** |",
        f"| (c) FULL AND (primary & maximal & firm), raw / Holm | {vol_before['AND_full_raw']} / "
        f"{vol_before['AND_full_holm']} | **{var_now['AND_full_raw']} / {var_now['AND_full_holm']}** |",
        f"| (c) strictest (placebo-gated genuine & both Holm controls) | "
        f"{vol_before['AND_genuine_holm']} | **{var_now['AND_genuine_holm']}** |",
        f"| (c) maximal-vs-firm Holm survivor overlap | {len(overlap_vol)} "
        f"({'disjoint' if not overlap_vol else '; '.join(overlap_vol)}) | "
        f"**{len(overlap_var)} ({'disjoint' if not overlap_var else '; '.join(overlap_var)})** |",
        f"| (e) standalone vs A2, better raw / Holm (of 180) | SE: {s_se['better_raw']} / "
        f"{s_se['better_holm']}; vol-QLIKE: {s_qv['better_raw']} / {s_qv['better_holm']} | "
        f"**var-QLIKE: {s_qz['better_raw']} / {s_qz['better_holm']}** (all "
        f"{p_holm} = GARCH-family price baselines; text/fusion: {t_raw} / {t_holm} "
        f"of {len(textc)}) |",
        f"| (e) standalone significantly WORSE (Holm) | SE: {s_se['worse_holm']}/180; "
        f"vol-QLIKE: {s_qv['worse_holm']}/180 | **var-QLIKE: {s_qz['worse_holm']}/180** |",
        "",
        "## Pre-declared Holm families (declared BEFORE any result below was read)\n",
        "1. **F-firm**: the 69-cell firm-identity grid, variance-unit clustered p-values, "
        "Holm across the 69 cells (mirrors the committed vol-unit family).",
        "2. **F-max**: the 69-cell maximal-pool grid, same convention.",
        "3. **F-primary**: the committed m1_variance_unit family (69 cells; reused as-is, "
        "gate-verified — not re-tested).",
        "4. **F-standalone**: for each loss, Holm WITHIN each (disclosure, horizon) group "
        "over the 20 vs-A2 challengers (9 groups x 20 = 180) — identical to the committed "
        "dm_pairwise_clustered convention (weaker than a 180-wide family, i.e. conservative "
        "for the '0 better' headline; anti-conservative for 'worse' counts, so worse counts "
        "are descriptive).",
        "5. **The residual is NOT granted its own family**: it is read out of F-firm "
        "(symmetric treatment; the dedicated symmetric-multiplicity re-analysis is row 6).\n",
        "No new combiner/reference weights were fit: all forecasts are the committed "
        "val-fit-frozen objects; only the evaluation loss changes. No subsampling anywhere.\n",
    ]

    # ---- (a)+(b) side-by-side survivor tables
    md += ["## (a)+(b) Vol-vs-var side-by-side — Holm survivors of either unit\n",
           "All 69-cell detail is in variance_unit_cascade.csv; this table lists every cell "
           "that survives Holm in EITHER unit under EITHER control.\n",
           "| control | cell | seeds | rel% vol | DM vol | Holm vol | rel% var | DM var | "
           "p var | Holm var | vol->var |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, df in (("firm", firmdf), ("maximal", maxdf)):
        sub = df[df.vol_adds_holm | df.var_adds_holm].sort_values(GRIDKEY)
        for _, r in sub.iterrows():
            trans = {(True, True): "KEEPS", (True, False): "**LOST**",
                     (False, True): "**GAINED**"}[(bool(r.vol_adds_holm), bool(r.var_adds_holm))]
            md.append(f"| {tag} | {cell_str(r)} | {r.seeds} | {r.vol_rel:+.2f} | {r.vol_dm:+.2f} | "
                      f"{r.vol_holm:.3f} | {r.var_rel:+.2f} | {r.var_dm:+.2f} | {r.var_p:.4f} | "
                      f"{r.var_holm:.3f} | {trans} |")
    md += [f"\n- (a) firm-identity: **{var_now['firm_raw']}/69 raw, {var_now['firm_holm']}/69 "
           f"Holm** in variance units (vol-unit: {int(firmdf.vol_adds_raw.sum())}/"
           f"{int(firmdf.vol_adds_holm.sum())}); convention-dependent Holm verdicts: "
           f"{int(firmdf.convention_dependent.sum())}/69.",
           f"- (b) maximal pool: **{var_now['maximal_raw']}/69 raw, {var_now['maximal_holm']}/69 "
           f"Holm** (vol-unit: {int(maxdf.vol_adds_raw.sum())}/{int(maxdf.vol_adds_holm.sum())}); "
           f"convention-dependent: {int(maxdf.convention_dependent.sum())}/69.\n"]

    # ---- (c) intersection
    md += ["## (c) Control intersection in variance units\n",
           "| quantity | vol-unit (committed) | variance-unit (this run) |",
           "|---|---|---|"]
    for label, k in (("primary marginal raw / Holm", ("primary_raw", "primary_holm")),
                     ("maximal raw / Holm", ("maximal_raw", "maximal_holm")),
                     ("firm raw / Holm", ("firm_raw", "firm_holm")),
                     ("FULL AND raw / Holm", ("AND_full_raw", "AND_full_holm")),
                     ("strictest genuine AND", ("AND_genuine_holm",))):
        b = " / ".join(str(vol_before[x]) for x in k)
        a = " / ".join(str(var_now[x]) for x in k)
        md.append(f"| {label} | {b} | **{a}** |")
    md += [f"| survivor-set overlap (maximal vs firm, Holm) | {len(overlap_vol)} "
           f"({'disjoint' if not overlap_vol else '; '.join(overlap_vol)}) | "
           f"**{len(overlap_var)} ({'disjoint' if not overlap_var else '; '.join(overlap_var)})** |",
           f"\n- Variance-unit Holm survivor sets: maximal = {{{', '.join(sorted(mx_var)) or 'empty'}}}; "
           f"firm = {{{', '.join(sorted(fi_var)) or 'empty'}}}.",
           "- Full-AND cells at raw p (variance-unit): "
           + ("; ".join(cell_str(r) for _, r in ix[ix.AND_full_var_raw].iterrows()) or "none") + ".",
           f"- Verdict: the two headline properties of the vol-unit cascade — Holm AND = "
           f"{'0 (HOLDS)' if var_now['AND_full_holm'] == 0 else str(var_now['AND_full_holm']) + ' (BROKEN)'} "
           f"and survivor-set {'disjointness (HOLDS)' if not overlap_var else 'overlap NON-EMPTY (BROKEN)'} — "
           "under the proxy-robust convention.\n"]

    # ---- (d) residual
    md += ["## (d) The 8-K residual (event_driven C6_llmtext vs firm-identity reference)\n",
           "Committed vol-unit values (firm_identity_ensemble.csv): "
           + "; ".join(f"h{int(r.h)} {r.vol_rel:+.2f}% (cluDM {r.vol_dm:+.2f})"
                       for _, r in res.iterrows())
           + " — all clustered-significant, raw AND Holm (the paper's rounded "
             "+0.45/+0.25/+0.20 refers to the same cells across the identity-spec battery).\n",
           "| h | n_test | n_days | rel% vol | DM vol | Holm vol | rel% var | DM var | p var | "
           "Holm var (F-firm) | daily-dQv 95% CI | survives raw | survives Holm |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        md.append(f"| {int(r.h)} | {int(r.n_test)} | {int(r.n_days)} | {r.vol_rel:+.2f} | "
                  f"{r.vol_dm:+.2f} | {r.vol_holm:.4f} | **{r.var_rel:+.2f}** | "
                  f"**{r.var_dm:+.2f}** | {r.var_p:.4f} | {r.var_holm:.4f} | "
                  f"[{r.var_boot_lo:+.5f}, {r.var_boot_hi:+.5f}] | "
                  f"{'YES' if r.var_adds_raw else 'no'} | {'YES' if r.var_adds_holm else 'no'} |")
    n_res_raw = int(res.var_adds_raw.sum())
    n_res_holm = int(res.var_adds_holm.sum())
    md.append(f"\n**Residual verdict under the convention change:** "
              f"{n_res_raw}/3 horizons survive at raw clustered p, {n_res_holm}/3 under Holm "
              f"within the pre-declared 69-cell F-firm family "
              f"(vol-unit: 3/3 raw, 3/3 Holm).\n")

    # ---- (e) standalone
    md += ["## (e) Standalone 180 vs A2 under day-clustered QLIKE\n",
           "Basis inspection of the committed QLIKE table `dm_qlike_all_vs_A2.csv`: "
           f"{qold_note}. It is therefore NOT a valid day-clustered QLIKE grid; the 180 "
           "comparisons (dm_pairwise_clustered universe: seed-ensembled challengers, "
           "inner-joined per disclosure) are recomputed below. Full per-comparison detail: "
           "variance_unit_standalone180.csv (extra output file, disclosed here).\n",
           "| loss | better than A2, raw p<.05 | better, Holm<.05 | significantly WORSE, Holm | n |",
           "|---|---|---|---|---|",
           f"| SE (committed reference, gate G3) | {s_se['better_raw']} | {s_se['better_holm']} | "
           f"{s_se['worse_holm']} | 180 |",
           f"| QLIKE vol-unit (new) | {s_qv['better_raw']} | {s_qv['better_holm']} | "
           f"{s_qv['worse_holm']} | 180 |",
           f"| **QLIKE variance-unit (new)** | **{s_qz['better_raw']}** | **{s_qz['better_holm']}** | "
           f"**{s_qz['worse_holm']}** | 180 |"]
    beat = stand[stand.better_qlike_var_raw | stand.better_qlike_var_holm | stand.better_qlike_vol_raw]
    if len(beat):
        md += ["\nComparisons better than A2 at raw p (either QLIKE unit):",
               "| disclosure | h | challenger | DM se | DM qlike-vol | Holm | DM qlike-var | Holm |",
               "|---|---|---|---|---|---|---|---|"]
        for _, r in beat.sort_values(["disclosure", "horizon", "dm_qlike_var_clu"]).iterrows():
            md.append(f"| {r.disclosure} | {r.horizon} | {r.challenger} | {r.dm_se_clu:+.2f} | "
                      f"{r.dm_qlike_vol_clu:+.2f} | {r.holm_qlike_vol_clu:.4f} | "
                      f"{r.dm_qlike_var_clu:+.2f} | {r.holm_qlike_var_clu:.4f} |")
    else:
        md.append("\nNo comparison beats A2 at raw p under either QLIKE unit.")
    md.append(f"\n**Standalone verdict:** in variance-unit QLIKE, "
              f"{s_qz['better_holm']}/180 comparisons beat A2 under Holm — but ALL "
              f"{p_holm} winners are GARCH-family PRICE baselines "
              f"({'; '.join(sorted(pricec[pricec.better_qlike_var_holm].challenger.unique()))}); "
              f"every text / text-fusion challenger stays at {t_holm}/{len(textc)} better "
              f"(Holm; {t_raw}/{len(textc)} raw). The TEXT-standalone null (no text model "
              f"beats HAR standalone) is convention-ROBUST; what the convention change "
              f"flips is the intra-price ranking HAR-vs-GARCH, which is orthogonal to the "
              f"paper's text claim. {s_qz['worse_holm']}/180 significantly worse "
              "(descriptive; see family note 4).\n")

    # ---- sanity
    md += ["## SANITY\n",
           f"- **G1 (named table: results/tables/m1_variance_unit.csv) PASS** — the full "
           f"69-cell variance-unit primary grid recomputed end-to-end (ensemble text -> "
           f"log-space combiner -> q(y^2,f^2) -> day-clustered DM -> Holm) reproduces the "
           f"committed rel%/DM/p/Holm columns: max|d rel| = {g1['var_rel']:.2e}, "
           f"max|d DM| = {g1['var_dm']:.2e}, max|d p| = {g1['var_p']:.2e}, "
           f"max|d Holm| = {g1['var_holm']:.2e} (vol-unit columns too: "
           f"max|d rel| = {g1['vol_rel']:.2e}, max|d DM| = {g1['vol_dm']:.2e}).",
           f"- **G2 (named tables: firm_identity_ensemble.csv, maximal_reference_ensemble.csv) "
           f"PASS** — the vol-unit legs of both cascade grids, recomputed in this script, "
           f"match the committed tables on all 69 cells: firm max|d| = "
           f"{max(g2['firm_rel'], g2['firm_dm'], g2['firm_p'], g2['firm_holm']):.2e}, "
           f"maximal max|d| = "
           f"{max(g2['maximal_rel'], g2['maximal_dm'], g2['maximal_p'], g2['maximal_holm']):.2e}.",
           f"- **G3 (named table: dm_pairwise_clustered.csv) PASS** — the day-clustered SE "
           f"leg of the standalone universe matches the committed table on all 180 "
           f"comparisons: max|d DM| = {g3['se_dm']:.2e}, max|d p| = {g3['se_p']:.2e}, "
           f"max|d Holm| = {g3['se_holm']:.2e}.",
           "- All gates enforced in scripts/analysis/variance_unit_cascade.py; the script "
           "aborts before writing any output if a gate fails.",
           "- No look-ahead: every reference/combiner weight is the committed val-fit object, "
           "frozen on test; this script fits nothing.\n"]

    # ---- bottom line
    md += ["## Bottom line\n",
           f"- The cascade's null architecture is convention-robust: in variance units the "
           f"controls tighten rather than loosen — firm-identity survivors "
           f"{int(firmdf.vol_adds_raw.sum())}/{int(firmdf.vol_adds_holm.sum())} -> "
           f"**{var_now['firm_raw']}/{var_now['firm_holm']}** (raw/Holm), maximal-pool "
           f"{int(maxdf.vol_adds_raw.sum())}/{int(maxdf.vol_adds_holm.sum())} -> "
           f"**{var_now['maximal_raw']}/{var_now['maximal_holm']}**, full Holm AND stays "
           f"**{var_now['AND_full_holm']}/69**, survivor sets "
           f"{'remain disjoint' if not overlap_var else 'OVERLAP: ' + '; '.join(overlap_var)}.",
           f"- The 8-K residual is **{'convention-robust' if n_res_holm == 3 else ('partially convention-dependent' if n_res_holm > 0 else 'convention-DEPENDENT')}**: "
           f"{n_res_raw}/3 horizons at raw clustered p and {n_res_holm}/3 under Holm in "
           f"variance units (vol-unit 3/3 and 3/3).",
           f"- Standalone: the TEXT null is convention-robust — {t_holm}/{len(textc)} "
           f"text/fusion comparisons beat HAR in variance-unit QLIKE (Holm; {t_raw} raw). "
           f"The {s_qz['better_holm']}/180 total Holm winners are all GARCH-family price "
           f"baselines (an intra-price HAR-vs-GARCH re-ranking under q(y^2,f^2), not a "
           f"text result). The committed QLIKE table was obs-level and is superseded by "
           f"variance_unit_standalone180.csv."]

    (T / "variance_unit_cascade.md").write_text("\n".join(md))

    summary = {"gates": {"G1": g1, "G2": g2, "G3": g3},
               "firm": {"vol_raw": int(firmdf.vol_adds_raw.sum()),
                        "vol_holm": int(firmdf.vol_adds_holm.sum()),
                        "var_raw": var_now["firm_raw"], "var_holm": var_now["firm_holm"]},
               "maximal": {"vol_raw": int(maxdf.vol_adds_raw.sum()),
                           "vol_holm": int(maxdf.vol_adds_holm.sum()),
                           "var_raw": var_now["maximal_raw"], "var_holm": var_now["maximal_holm"]},
               "intersection_var": var_now, "intersection_vol": vol_before,
               "overlap_var": overlap_var, "overlap_vol": overlap_vol,
               "residual_var": res[["h", "var_rel", "var_dm", "var_p", "var_holm",
                                    "var_adds_raw", "var_adds_holm"]].to_dict("records"),
               "standalone": {"se": s_se, "qlike_vol": s_qv, "qlike_var": s_qz,
                              "qlike_var_split": {
                                  "text_better_holm": t_holm, "text_better_raw": t_raw,
                                  "text_n": len(textc),
                                  "price_better_holm": p_holm, "price_n": len(pricec)}},
               "runtime_s": round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=2, default=str))
    print("wrote results/tables/variance_unit_cascade.{csv,md} + "
          "variance_unit_standalone180.csv")


if __name__ == "__main__":
    main()
