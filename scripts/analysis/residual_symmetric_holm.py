"""ROW 6 (round-3 remediation) — SYMMETRIC MULTIPLICITY for the 8-K residual +
TWO-WAY (firm x day) CGM clustering on the SEED-ENSEMBLE basis.

Reviewer complaints being remediated (REVIEW_ROUND3_FRESH_PANEL.md, freeze row 6;
methodology CRITICAL #4 / MAJOR "symmetric multiplicity"):
  (a) every NULL claim is Holm-gated, but the one surviving POSITIVE finding (the
      event-driven C6 8-K residual vs the firm-identity reference) was cited at raw
      clustered significance — it was never Holm-corrected inside a PRE-DECLARED
      family symmetric with the null claims;
  (b) the two-way (firm x day) clustering robustness (twoway_cluster.csv) was run
      on the seed2026 basis only, while the DECLARED primary is the 3-seed
      per-observation ensemble (m1_ensemble_primary.md, 38/69 day-clustered).

PRE-DECLARED HOLM FAMILIES for the residual (declared here, before any result is
computed; both use day-clustered DM p-values, the committed primary inference):
  FAMILY-1 "C6 symmetric" (12 cells): C6_llmtext x {long_form, event_driven}
      x h in {5,10,20} x reference in {single recalibrated HAR, firm-identity
      (HAR + firm-mean-val-RV)}. Holm over the 12 p-values. This is the family a
      symmetric protocol would have declared for the C6 residual claim: the same
      model, both disclosure channels, all horizons, BOTH ends of the reference
      interval. C6 is single-seed (seed2026 only), so this family is basis-invariant
      (asserted).
  FAMILY-2 "69-cell firm-identity grid" (69 cells): every text model x disclosure
      x horizon vs the firm-identity reference — the SAME family the null claims
      use (firm_identity_control.csv holm_p). Reported on both bases (seed2026 =
      committed; seed-ensemble = declared primary), plus a two-way-p variant.
  The residual claim is "genuine" in a family only if clustered DM < 0 AND
  Holm(p) < .05 within that family (placebo gate |DM|<2 where placebos exist).

TWO-WAY MECHANICS: identical to scripts/analysis/twoway_dm.py (CGM
V_2way = V_firm + V_day(HAC, lag=h-1) - V_firm∩day; non-PSD guard; df =
min(#firms,#days)-1) — only the TEXT forecast basis changes (m1_ensemble_primary.
ensemble_text: per-observation mean over seeds 2026/2027/2028 for 3-seed C/D
models; A/B, C6_llmtext, D4_llmfused stay single-seed).

NO LOOK-AHEAD: all combiner/reference weights are OLS-fit on split=="val" only and
applied frozen to split=="test" (fc.log_combo / mrf.log_ols_frozen, unchanged).

SANITY GATES (hard assertions, machine precision; script stops on failure):
  G1  seed2026 panel A (single-HAR ref, 69 cells): dm_day/p_day/holm_day/dm_2way/
      p_2way/holm_2way/placebo_dm_2way reproduce results/tables/twoway_cluster.csv
      (panel a_m1_grid) exactly.
  G2  seed2026 panel B (firm-identity ref, 69 cells): same columns reproduce
      twoway_cluster.csv (panel b_firm_ref) exactly.
  G3  seed2026 day placebo + genuine flags reproduce results/tables/m1_clustered.csv
      (placebo_dm_clust, genuine_clust).
  G4  ENSEMBLE panel A day-clustered columns reproduce
      results/tables/m1_ensemble_primary.csv (vol_dm_q_clu, vol_p_q_clu,
      vol_dmq_holm_clu, vol_placebo_dm_clu, genuine_ens_vol).
  G5  ENSEMBLE panel B day-clustered columns reproduce
      results/tables/firm_identity_ensemble.csv (dm_q_clustered, p_q_clustered,
      holm_p).
  G6  single-seed cells (n_seeds==1, incl. every C6 cell) are IDENTICAL between the
      seed2026 and ensemble bases (both panels, all stats).

Outputs (NEW files only):
  results/tables/residual_symmetric_holm.csv
  results/tables/residual_symmetric_holm.md
Run from repo root:  .venv/bin/python scripts/analysis/residual_symmetric_holm.py
"""
from __future__ import annotations

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
import maximal_reference_firm_control as mrf
from clustered_dm import dm_test_clustered
from twoway_dm import dm_test_2way

KEY = fc.KEY
SORT = fc.SORT
HORIZONS = fc.HORIZONS
HAR = "A2_har_rv"
BASES = ("s26", "ens")
TOL = dict(rtol=1e-9, atol=1e-12, equal_nan=True)
RESIDUAL_CELLS = [("event_driven", 5), ("event_driven", 10), ("event_driven", 20)]
T = Path("results/tables")


def text_forecast(m, disc, basis):
    """Text forecast on the requested basis. Returns (df[KEY+ftext], n_seeds, seeds)."""
    if basis == "s26":
        df = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": "ftext"})
        return df, 1, "2026"
    ens, used = mep.ensemble_text(m, disc)  # per-obs mean over seeds 2026/2027/2028
    return ens, len(used), "+".join(map(str, used))


def tw_fields(tw):
    return {"dm_2way": tw.stat, "p_2way": tw.p, "n_firms": tw.n_firms,
            "n_days": tw.n_days, "guard_hit": tw.guard_hit,
            "se_infl_2way_vs_day": float(np.sqrt(tw.V_2way / tw.V_day))
            if tw.V_day > 0 else np.nan}


# ---------------------------------------------------------------------------
# Cell computations (one function per panel; called for both bases)
# ---------------------------------------------------------------------------
def panel_a_cell(dv, dt, h):
    """Single recalibrated-HAR reference: fc.log_combo (val-fit, frozen to test),
    vol-unit QLIKE, day-clustered + two-way DM, placebo under BOTH statistics."""
    yv, fhv, ftv = dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(), dv.ftext.to_numpy()
    yt, fhr, ftt = dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(), dt.ftext.to_numpy()
    days = mc.day_key(dt)
    firms = dt.ticker.to_numpy()

    fR, fU, _ = fc.log_combo(yv, fhv, ftv, fhr, ftt)
    lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
    dm_day, p_day, _ = dm_test_clustered(lU, lR, days, h)
    tw = dm_test_2way(lU - lR, firms, days, h)

    p_day_pl, p_2w_pl = [], []
    for s in fc.PLACEBO_SEEDS:
        rng = np.random.default_rng(s)
        pR, pU, _ = fc.log_combo(yv, fhv, rng.permutation(ftv), fhr, rng.permutation(ftt))
        plR, plU = fc.qlike(yt, pR), fc.qlike(yt, pU)
        p_day_pl.append(dm_test_clustered(plU, plR, days, h)[0])
        p_2w_pl.append(dm_test_2way(plU - plR, firms, days, h).stat)

    row = {"n_test": len(dt), "dm_day": dm_day, "p_day": p_day,
           "placebo_dm_day": float(np.mean(p_day_pl)),
           "placebo_dm_2way": float(np.mean(p_2w_pl))}
    row.update(tw_fields(tw))
    return row


def panel_b_cell(dv, dt, h):
    """Firm-identity reference: exp OLS[1, log fHAR, log firm_mean_val_RV] (val-fit,
    frozen); +text adds g*log ftext. Day-clustered + two-way DM (no placebo, matching
    the committed twoway_cluster panel-b convention)."""
    yv, yt = dv.label_realised_vol.to_numpy(), dt.label_realised_vol.to_numpy()
    fhv, fhr = dv.A2_har_rv.to_numpy(), dt.A2_har_rv.to_numpy()
    ftv, ftt = dv.ftext.to_numpy(), dt.ftext.to_numpy()
    fmv, fmt = dv.firm_mean_val.to_numpy(), dt.firm_mean_val.to_numpy()
    days = (dt.effective_trading_day.fillna(dt.filing_time_utc)).to_numpy()
    firms = dt.ticker.to_numpy()

    fRf, _ = mrf.log_ols_frozen(yv, [fhv, fmv], [fhr, fmt])
    fUf, _ = mrf.log_ols_frozen(yv, [fhv, fmv, ftv], [fhr, fmt, ftt])
    lRf, lUf = fc.qlike(yt, fRf), fc.qlike(yt, fUf)
    qRf = float(lRf.mean())
    dm_day, p_day, _ = dm_test_clustered(lUf, lRf, days, h)
    tw = dm_test_2way(lUf - lRf, firms, days, h)

    row = {"n_test": len(dt), "dm_day": dm_day, "p_day": p_day,
           "rel_impr_pct_firm": 100.0 * (qRf - float(lUf.mean())) / qRf if qRf > 0 else np.nan}
    row.update(tw_fields(tw))
    return row


def run_grids():
    """Both panels x both bases in one pass over the data."""
    rows = []
    t0 = time.time()
    for disc, models in fc.SETS.items():
        har = fc.load(HAR, disc)[["split"] + KEY + [
            "prediction_realised_vol", "label_realised_vol", "filing_time_utc",
            "effective_trading_day"]].rename(columns={"prediction_realised_vol": "fhar"})
        price = mrf.build_price_panel(disc)
        fmap, gmean, _fc_, _oc_ = mrf.firm_mean_val(price)
        price["firm_mean_val"] = price.ticker.map(fmap).fillna(gmean).astype(float)
        for m in models:
            for basis in BASES:
                txt, n_seeds, seeds = text_forecast(m, disc, basis)
                dA = har.merge(txt, on=KEY)                     # panel A sample
                dB = price.merge(txt, on=KEY, how="inner")      # panel B sample
                for h in HORIZONS:
                    base = {"basis": basis, "disc": disc, "model": m, "h": h,
                            "n_seeds": n_seeds, "seeds": seeds}
                    for pname, d, cellfn in (("A_singleHAR", dA, panel_a_cell),
                                             ("B_firmref", dB, panel_b_cell)):
                        dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(
                            SORT, kind="mergesort")
                        dt_ = d[(d.horizon_days == h) & (d.split == "test")].sort_values(
                            SORT, kind="mergesort")
                        if len(dv) < 100 or len(dt_) < 30:
                            continue
                        rows.append({**base, "panel": pname, **cellfn(dv, dt_, h)})
            print(f"  [{time.time()-t0:7.1f}s] {disc}/{m} done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def holm_within(df, pcol):
    """Holm within each (panel, basis) 69-cell grid — the committed convention."""
    return df.groupby(["panel", "basis"])[pcol].transform(
        lambda s: fc.holm(s.fillna(1.0).to_numpy()))


def sanity(df):
    """All six gates. Raises AssertionError on any failure."""
    rep = []

    def gate(name, got, want, cols):
        for c_got, c_want in cols:
            a = np.asarray(got[c_got], float)
            b = np.asarray(want[c_want], float)
            d = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
            if not np.allclose(a, b, **TOL):
                raise AssertionError(f"SANITY FAIL {name}: {c_got} vs {c_want} max|diff|={d:.3e}")
            rep.append(f"{name} {c_got}: max|diff|={d:.2e}")

    twc = pd.read_csv(T / "twoway_cluster.csv")
    m1c = pd.read_csv(T / "m1_clustered.csv")
    mens = pd.read_csv(T / "m1_ensemble_primary.csv")
    fie = pd.read_csv(T / "firm_identity_ensemble.csv")
    on = ["disc", "model", "h"]

    # G1 — s26 panel A vs twoway_cluster panel a
    a26 = df[(df.panel == "A_singleHAR") & (df.basis == "s26")]
    ta = twc[twc.panel == "a_m1_grid"]
    m = a26.merge(ta, on=on, suffixes=("", "_ref"))
    assert len(m) == 69, f"G1 join: {len(m)}"
    gate("G1(twoway_cluster.csv panel a)", m, m,
         [("dm_day", "dm_day_ref"), ("p_day", "p_day_ref"), ("holm_day", "holm_day_ref"),
          ("dm_2way", "dm_2way_ref"), ("p_2way", "p_2way_ref"), ("holm_2way", "holm_2way_ref"),
          ("placebo_dm_2way", "placebo_dm_2way_ref")])

    # G2 — s26 panel B vs twoway_cluster panel b
    b26 = df[(df.panel == "B_firmref") & (df.basis == "s26")]
    tb = twc[twc.panel == "b_firm_ref"]
    m = b26.merge(tb, on=on, suffixes=("", "_ref"))
    assert len(m) == 69, f"G2 join: {len(m)}"
    gate("G2(twoway_cluster.csv panel b)", m, m,
         [("dm_day", "dm_day_ref"), ("p_day", "p_day_ref"), ("holm_day", "holm_day_ref"),
          ("dm_2way", "dm_2way_ref"), ("p_2way", "p_2way_ref"), ("holm_2way", "holm_2way_ref")])

    # G3 — s26 day placebo + genuine flag vs m1_clustered.csv
    m = a26.merge(m1c[on + ["placebo_dm_clust", "genuine_clust"]], on=on)
    assert len(m) == 69, f"G3 join: {len(m)}"
    gate("G3(m1_clustered.csv)", m, m, [("placebo_dm_day", "placebo_dm_clust")])
    if not (m.genuine_day.astype(bool) == m.genuine_clust.astype(bool)).all():
        raise AssertionError("SANITY FAIL G3: genuine_day != m1_clustered.genuine_clust")
    rep.append("G3 genuine flags: identical")

    # G4 — ens panel A day columns vs m1_ensemble_primary.csv
    aen = df[(df.panel == "A_singleHAR") & (df.basis == "ens")]
    m = aen.merge(mens[on + ["vol_dm_q_clu", "vol_p_q_clu", "vol_dmq_holm_clu",
                             "vol_placebo_dm_clu", "genuine_ens_vol"]], on=on)
    assert len(m) == 69, f"G4 join: {len(m)}"
    gate("G4(m1_ensemble_primary.csv)", m, m,
         [("dm_day", "vol_dm_q_clu"), ("p_day", "vol_p_q_clu"),
          ("holm_day", "vol_dmq_holm_clu"), ("placebo_dm_day", "vol_placebo_dm_clu")])
    if not (m.genuine_day.astype(bool) == m.genuine_ens_vol.astype(bool)).all():
        raise AssertionError("SANITY FAIL G4: genuine_day(ens) != genuine_ens_vol")
    rep.append("G4 genuine flags: identical")

    # G5 — ens panel B day columns vs firm_identity_ensemble.csv
    ben = df[(df.panel == "B_firmref") & (df.basis == "ens")]
    m = ben.merge(fie[on + ["dm_q_clustered", "p_q_clustered", "holm_p"]], on=on)
    assert len(m) == 69, f"G5 join: {len(m)}"
    gate("G5(firm_identity_ensemble.csv)", m, m,
         [("dm_day", "dm_q_clustered"), ("p_day", "p_q_clustered"), ("holm_day", "holm_p")])

    # G6 — single-seed cells identical between bases. "Single-seed" is a property
    # of the MODEL (ensemble basis uses 1 seed), not of the basis tag: s26-basis
    # rows always carry n_seeds=1, so select the cells via the ENSEMBLE rows.
    single_keys = df[(df.basis == "ens") & (df.n_seeds == 1)][["panel"] + on]
    sub = df.merge(single_keys, on=["panel"] + on)
    piv = sub.pivot_table(index=["panel"] + on, columns="basis",
                          values=["dm_day", "p_day", "dm_2way", "p_2way"],
                          aggfunc="first")
    assert len(piv) == len(single_keys), "G6 pivot lost cells"
    for v in ("dm_day", "p_day", "dm_2way", "p_2way"):
        a, b = piv[(v, "s26")].to_numpy(), piv[(v, "ens")].to_numpy()
        if np.isnan(a).sum() != np.isnan(b).sum():
            raise AssertionError(f"SANITY FAIL G6: NaN pattern differs for {v}")
        d = float(np.nanmax(np.abs(a - b)))
        if not np.allclose(a, b, **TOL):
            raise AssertionError(f"SANITY FAIL G6: single-seed {v} differs across bases "
                                 f"max|diff|={d:.3e}")
        rep.append(f"G6 single-seed {v}: max|diff|={d:.2e} over {len(a)} cells")
    return rep


# ---------------------------------------------------------------------------
def main():
    T.mkdir(parents=True, exist_ok=True)
    print("building 69-cell x {A_singleHAR, B_firmref} x {s26, ens} grids ...", flush=True)
    df = run_grids()
    n_a = (df.panel == "A_singleHAR").sum()
    n_b = (df.panel == "B_firmref").sum()
    assert n_a == 138 and n_b == 138, f"cell counts: A={n_a} B={n_b} (want 138 each)"

    # grid-family Holm (committed convention: within each 69-cell panel x basis)
    df["holm_day"] = holm_within(df, "p_day")
    df["holm_2way"] = holm_within(df, "p_2way")

    # verdicts
    is_a = df.panel == "A_singleHAR"
    df["genuine_day"] = np.where(
        is_a, (df.dm_day < 0) & (df.holm_day < 0.05) & (df.placebo_dm_day.abs() < 2.0),
        (df.dm_day < 0) & (df.holm_day < 0.05))
    df["genuine_2way"] = np.where(
        is_a, (df.dm_2way < 0) & (df.holm_2way < 0.05) & (df.placebo_dm_2way.abs() < 2.0),
        (df.dm_2way < 0) & (df.holm_2way < 0.05))
    df["hurts_day"] = (df.dm_day > 0) & (df.holm_day < 0.05)
    df["hurts_2way"] = (df.dm_2way > 0) & (df.holm_2way < 0.05)

    print("running sanity gates ...", flush=True)
    rep = sanity(df)
    for line in rep:
        print("  " + line)
    print("ALL SANITY GATES PASS", flush=True)

    # ---------------- (a) PRE-DECLARED symmetric Holm families ----------------
    # FAMILY-1: 12 C6 cells (2 disc x 3 h x 2 references), day-clustered p.
    # C6 is single-seed => basis-invariant (G6 already asserts); use s26 rows.
    f1 = df[(df.model == "C6_llmtext") & (df.basis == "s26")].copy()
    assert len(f1) == 12, f"FAMILY-1 size {len(f1)} != 12"
    f1["fam12_holm_day"] = fc.holm(f1.p_day.fillna(1.0).to_numpy())
    f1["fam12_holm_2way"] = fc.holm(f1.p_2way.fillna(1.0).to_numpy())
    f1["fam12_survives_day"] = (f1.dm_day < 0) & (f1.fam12_holm_day < 0.05)
    f1["fam12_survives_2way"] = (f1.dm_2way < 0) & (f1.fam12_holm_2way < 0.05)
    df = df.merge(f1[["panel", "disc", "model", "h", "fam12_holm_day", "fam12_holm_2way",
                      "fam12_survives_day", "fam12_survives_2way"]],
                  on=["panel", "disc", "model", "h"], how="left")

    # FAMILY-2: the 69-cell firm-identity grid == panel B holm_day per basis
    # (identical to committed firm_identity_control.holm_p / firm_identity_ensemble.holm_p
    #  by gates G2/G5) — plus its two-way variant holm_2way.

    def resid(panel, basis):
        return df[(df.panel == panel) & (df.basis == basis) & (df.model == "C6_llmtext")
                  & (df.disc == "event_driven")].set_index("h").loc[[5, 10, 20]]

    rB26, rBen = resid("B_firmref", "s26"), resid("B_firmref", "ens")
    rA26 = resid("A_singleHAR", "s26")
    f1B = f1[(f1.panel == "B_firmref") & (f1.disc == "event_driven")].set_index("h").loc[[5, 10, 20]]

    # ---------------- (b) ensemble-basis headline counts ----------------
    def counts(panel, basis):
        s = df[(df.panel == panel) & (df.basis == basis)]
        return (int(s.genuine_day.sum()), int(s.genuine_2way.sum()),
                int(s.hurts_day.sum()), int(s.hurts_2way.sum()),
                int((s.genuine_day != s.genuine_2way).sum()), int(s.guard_hit.sum()))

    aG26 = counts("A_singleHAR", "s26")
    aGen = counts("A_singleHAR", "ens")
    bG26 = counts("B_firmref", "s26")
    bGen = counts("B_firmref", "ens")

    cols = ["basis", "panel", "disc", "model", "h", "n_seeds", "seeds", "n_test",
            "n_firms", "n_days", "dm_day", "p_day", "holm_day", "placebo_dm_day",
            "genuine_day", "dm_2way", "p_2way", "holm_2way", "placebo_dm_2way",
            "genuine_2way", "hurts_day", "hurts_2way", "guard_hit",
            "se_infl_2way_vs_day", "rel_impr_pct_firm",
            "fam12_holm_day", "fam12_holm_2way", "fam12_survives_day",
            "fam12_survives_2way"]
    df.reindex(columns=cols).to_csv(T / "residual_symmetric_holm.csv", index=False)

    # ---------------- markdown ----------------
    def yn(x):
        return "YES" if bool(x) else "no"

    md = [
        "# ROW 6 — Symmetric Holm for the 8-K residual + two-way (firm x day) "
        "clustering on the SEED-ENSEMBLE basis\n",
        "UNITS: every DM statistic in this table (dm_day/dm_2way and all Holm "
        "columns) is on **VOL-unit QLIKE** losses — the committed primary "
        "(m1_ensemble_primary `vol_*` columns, firm_identity_* `_q_` columns). "
        "No variance-unit (RV^2) QLIKE appears here; see variance_unit_cascade for "
        "that restatement.\n",
        "## RESTATED vs BEFORE\n",
        "| quantity | BEFORE (committed) | RESTATED (this table) |",
        "|---|---|---|",
        "| residual multiplicity | event-driven C6 cells vs firm-identity quoted at raw "
        "clustered p (firm_identity_control.csv); never Holm-corrected in a pre-declared "
        "family symmetric with the null claims | Holm in TWO pre-declared families: "
        "12-cell C6-symmetric family and the 69-cell firm-identity grid family, both "
        "day-clustered primary + two-way variant |",
        f"| two-way clustering basis | seed2026 only (twoway_cluster.csv) | seed-ensemble "
        f"primary: M1 grid genuine {aGen[0]}/69 (day) -> **{aGen[1]}/69** (two-way); "
        f"firm-identity survivors {bGen[0]}/69 (day) -> **{bGen[1]}/69** (two-way) |",
        f"| M1 grid genuine, seed2026 (context) | {aG26[0]}/69 day -> {aG26[1]}/69 two-way "
        f"(twoway_cluster.csv) | unchanged (gate G1) |",
        "",
        "## PRE-DECLARED HOLM FAMILIES (declared in the script header BEFORE any "
        "result was computed)\n",
        "* **FAMILY-1 'C6 symmetric' (12 cells):** C6_llmtext x {long_form, "
        "event_driven} x h in {5,10,20} x reference in {single recalibrated HAR, "
        "firm-identity (HAR + firm-mean-val-RV)}. Day-clustered DM p, Holm over 12; "
        "two-way variant reported alongside. C6 is single-seed, so this family is "
        "basis-invariant (asserted, gate G6).",
        "* **FAMILY-2 '69-cell firm-identity grid':** all 69 text-model x disclosure "
        "x horizon cells vs the firm-identity reference — the SAME family the null "
        "claims use (committed holm_p of firm_identity_control.csv /"
        " firm_identity_ensemble.csv; reproduced by gates G2/G5). Reported on both "
        "bases + two-way variant.",
        "* Survival rule: clustered DM < 0 AND Holm(p) < .05 within the family "
        "(panel-A cells additionally need |placebo DM| < 2; the firm-ref panel has no "
        "placebo, matching the committed convention).",
        "",
        "## (a) Does the event-driven C6 residual survive symmetric Holm?\n",
        "| residual cell | raw clustered p (day) | FAMILY-1 Holm (12) | survives? | "
        "FAMILY-2 Holm (69, s26) | survives? | FAMILY-2 Holm (69, ensemble) | survives? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in (5, 10, 20):
        r1, r26, ren = f1B.loc[h], rB26.loc[h], rBen.loc[h]
        md.append(
            f"| event_driven C6 h{h} vs firm-identity | {r26.p_day:.2e} | "
            f"{r1.fam12_holm_day:.4f} | {yn(r1.fam12_survives_day)} | "
            f"{r26.holm_day:.4f} | {yn(r26.genuine_day)} | "
            f"{ren.holm_day:.4f} | {yn(ren.genuine_day)} |")
    md += [
        "",
        "Two-way variant of the same families:",
        "| residual cell | p_2way | FAMILY-1 Holm-2way (12) | survives? | "
        "FAMILY-2 Holm-2way (69, s26) | survives? | FAMILY-2 Holm-2way (69, ensemble) "
        "| survives? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in (5, 10, 20):
        r1, r26, ren = f1B.loc[h], rB26.loc[h], rBen.loc[h]
        md.append(
            f"| event_driven C6 h{h} vs firm-identity | {r26.p_2way:.2e} | "
            f"{r1.fam12_holm_2way:.4f} | {yn(r1.fam12_survives_2way)} | "
            f"{r26.holm_2way:.4f} | {yn(r26.genuine_2way)} | "
            f"{ren.holm_2way:.4f} | {yn(ren.genuine_2way)} |")

    md += ["", "### FAMILY-1 full 12-cell table (basis-invariant; single-seed C6)\n",
           "| disc | h | reference | dm_day | p_day | fam12 Holm | survives | dm_2way | "
           "p_2way | fam12 Holm-2way | survives |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    refname = {"A_singleHAR": "single-HAR", "B_firmref": "firm-identity"}
    for _, r in f1.sort_values(["disc", "h", "panel"]).iterrows():
        md.append(f"| {r.disc} | {r.h} | {refname[r.panel]} | {r.dm_day:+.2f} | "
                  f"{r.p_day:.2e} | {r.fam12_holm_day:.4f} | {yn(r.fam12_survives_day)} | "
                  f"{r.dm_2way:+.2f} | {r.p_2way:.2e} | {r.fam12_holm_2way:.4f} | "
                  f"{yn(r.fam12_survives_2way)} |")

    md += [
        "",
        "## (b) Two-way (firm x day) CGM clustering on the SEED-ENSEMBLE basis\n",
        "Identical CGM machinery as the committed twoway_cluster.csv "
        "(scripts/analysis/twoway_dm.py); only the text-forecast basis changes to the "
        "declared primary (per-observation 3-seed mean; single-seed models unchanged).\n",
        "| grid | basis | genuine/survives (day, Holm<.05) | (two-way, Holm<.05) | "
        "hurts day->2way | day/2way verdict flips | guard hits |",
        "|---|---|---|---|---|---|---|",
        f"| (i) M1 69-cell, single-HAR ref | seed2026 (twoway_cluster.csv) | {aG26[0]}/69 | "
        f"{aG26[1]}/69 | {aG26[2]}->{aG26[3]} | {aG26[4]} | {aG26[5]} |",
        f"| (i) M1 69-cell, single-HAR ref | **seed-ensemble (primary)** | **{aGen[0]}/69** | "
        f"**{aGen[1]}/69** | {aGen[2]}->{aGen[3]} | {aGen[4]} | {aGen[5]} |",
        f"| (ii) firm-identity ref | seed2026 (twoway_cluster.csv) | {bG26[0]}/69 | "
        f"{bG26[1]}/69 | {bG26[2]}->{bG26[3]} | {bG26[4]} | {bG26[5]} |",
        f"| (ii) firm-identity ref | **seed-ensemble (primary)** | **{bGen[0]}/69** | "
        f"**{bGen[1]}/69** | {bGen[2]}->{bGen[3]} | {bGen[4]} | {bGen[5]} |",
        "",
        "### (iii) Residual cells, two-way verdicts (ensemble basis = seed2026 basis "
        "for C6, gate G6)\n",
        "| cell | reference | dm_day | Holm(day,69) | dm_2way | p_2way | Holm(2way,69) | "
        "verdict day -> two-way |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in (5, 10, 20):
        r = rBen.loc[h]
        v1 = "survives" if r.genuine_day else "ns"
        v2 = "survives" if r.genuine_2way else "ns"
        md.append(f"| event_driven C6 h{h} | firm-identity | {r.dm_day:+.2f} | "
                  f"{r.holm_day:.4f} | {r.dm_2way:+.2f} | {r.p_2way:.4f} | "
                  f"{r.holm_2way:.4f} | {v1} -> {v2} |")
    for h in (5, 10, 20):
        r = resid("A_singleHAR", "ens").loc[h]
        v1 = "genuine" if r.genuine_day else "ns"
        v2 = "genuine" if r.genuine_2way else "ns"
        md.append(f"| event_driven C6 h{h} | single-HAR | {r.dm_day:+.2f} | "
                  f"{r.holm_day:.4f} | {r.dm_2way:+.2f} | {r.p_2way:.4f} | "
                  f"{r.holm_2way:.4f} | {v1} -> {v2} |")

    # ensemble flips detail
    flips = df[(df.basis == "ens") & (df.genuine_day != df.genuine_2way)]
    md += ["", "### Ensemble-basis verdict flips (day -> two-way)\n"]
    if len(flips):
        md += ["| panel | disc | model | h | dm_day | Holm(day) | dm_2way | p_2way | "
               "Holm(2way) | flip |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in flips.sort_values(["panel", "disc", "model", "h"]).iterrows():
            md.append(f"| {refname[r.panel]} | {r.disc} | {r.model} | {r.h} | "
                      f"{r.dm_day:+.2f} | {r.holm_day:.4f} | {r.dm_2way:+.2f} | "
                      f"{r.p_2way:.4f} | {r.holm_2way:.4f} | "
                      f"{yn(r.genuine_day)} -> {yn(r.genuine_2way)} |")
    else:
        md.append("None.")

    md += [
        "",
        "## SANITY\n",
        "All gates are HARD assertions at machine precision "
        "(np.allclose rtol=1e-9, atol=1e-12); the script aborts before writing any "
        "output if one fails. All PASS:\n",
    ] + [f"* {line}" for line in rep] + [
        "",
        "Gate targets (committed tables): `twoway_cluster.csv` (G1 panel a / G2 panel b: "
        "dm_day, p_day, holm_day, dm_2way, p_2way, holm_2way, placebo_dm_2way), "
        "`m1_clustered.csv` (G3: placebo_dm_clust, genuine_clust), "
        "`m1_ensemble_primary.csv` (G4: vol_dm_q_clu, vol_p_q_clu, vol_dmq_holm_clu, "
        "vol_placebo_dm_clu, genuine_ens_vol), `firm_identity_ensemble.csv` "
        "(G5: dm_q_clustered, p_q_clustered, holm_p), plus G6: every single-seed cell "
        "(incl. all C6 cells) identical between the seed2026 and ensemble bases.",
        "",
        "No look-ahead: all reference/combiner weights are fit on the validation split "
        "only and applied frozen to test (fc.log_combo / mrf.log_ols_frozen).",
        "",
        "## Bottom line\n",
        f"* FAMILY-1 (12-cell symmetric C6 family): the event-driven residual survives "
        f"Holm at h5/h10/h20 = {yn(f1B.loc[5].fam12_survives_day)}/"
        f"{yn(f1B.loc[10].fam12_survives_day)}/{yn(f1B.loc[20].fam12_survives_day)} "
        f"(two-way variant: {yn(f1B.loc[5].fam12_survives_2way)}/"
        f"{yn(f1B.loc[10].fam12_survives_2way)}/{yn(f1B.loc[20].fam12_survives_2way)}).",
        f"* FAMILY-2 (69-cell firm-identity grid, ensemble basis): survives at h5/h10/h20 = "
        f"{yn(rBen.loc[5].genuine_day)}/{yn(rBen.loc[10].genuine_day)}/"
        f"{yn(rBen.loc[20].genuine_day)} day-clustered; "
        f"{yn(rBen.loc[5].genuine_2way)}/{yn(rBen.loc[10].genuine_2way)}/"
        f"{yn(rBen.loc[20].genuine_2way)} two-way.",
        f"* Ensemble-basis two-way headline: M1 grid {aGen[0]}/69 (day) -> {aGen[1]}/69 "
        f"(two-way); firm-identity survivors {bGen[0]}/69 -> {bGen[1]}/69. "
        "Movement is toward the null, consistent with the seed2026 two-way robustness "
        "(wider SEs by construction).",
    ]

    with open(T / "residual_symmetric_holm.md", "w") as fh:
        fh.write("\n".join(md))

    print("\n=== ROW 6 residual symmetric Holm + ensemble two-way — done ===")
    print(f"FAMILY-1 (12): ed C6 h5/10/20 survive day-Holm: "
          f"{[bool(f1B.loc[h].fam12_survives_day) for h in (5, 10, 20)]}, "
          f"2way: {[bool(f1B.loc[h].fam12_survives_2way) for h in (5, 10, 20)]}")
    print(f"FAMILY-2 (69, ens): survive day-Holm: "
          f"{[bool(rBen.loc[h].genuine_day) for h in (5, 10, 20)]}, "
          f"2way: {[bool(rBen.loc[h].genuine_2way) for h in (5, 10, 20)]}")
    print(f"(b) ensemble two-way: M1 grid {aGen[0]}/69 day -> {aGen[1]}/69 2way; "
          f"firm-ref {bGen[0]}/69 day -> {bGen[1]}/69 2way")
    print("wrote results/tables/residual_symmetric_holm.{csv,md}")


if __name__ == "__main__":
    main()
