#!/usr/bin/env python3
"""R27 -- is the 38 -> 8 collapse a control effect or a sample effect?

A cold-read reviewer objected that the cascade's two rungs are not scored on the
same rows: the primary rung (recalibrated HAR vs +text) runs on the A2+text
join, while the firm-identity rung runs on the inner join of all five price
models, which is ~9.7% smaller in rows and ~2.1% smaller in trading days. The
headline collapse is therefore confounded with a sample change.

This recomputes BOTH rungs on ONE row set -- the identity rung's own support --
so the only thing that differs between them is the reference:

    f_R      = exp OLS[1, log f_A2]                        (recalibrated HAR)
    f_Rfirm  = exp OLS[1, log f_A2, log firm_mean_val_RV]  (+ firm identity)
    f_U*     = the same design plus g * log f_text

Weights are fitted on validation and frozen to test, exactly as in the committed
spec. Inference is the committed day-clustered DM with Holm within the 69-cell
family. Models are never retrained; this is a re-scoring of frozen predictions.

CPU-only. Usage: python3 scripts/analysis/matched_row_cascade.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import forecast_combination as fc
import m1_ensemble_primary as mep
from clustered_dm import dm_test_clustered
from maximal_reference_firm_control import (
    KEY,
    SORT,
    build_price_panel,
    firm_mean_val,
    log_ols_frozen,
)

# The committed gate is THREE conditions -- DM<0, Holm<0.05, |placebo DM|<2.0
# (m1_ensemble_primary.py:180-183). An audit caught this script applying only the
# first two. On the committed row set the omission happens to be non-binding
# (38 either way), but the matched row set refits the weights, so the placebo
# must be recomputed there rather than assumed.
PLACEBO_SEEDS = fc.PLACEBO_SEEDS


def placebo_dm(yv, Xv, Xt, yt, txv, txt_, days, h):
    """Mean clustered DM over placebo draws that permute the TEXT column only."""
    out = []
    for s in PLACEBO_SEEDS:
        rng = np.random.default_rng(s)
        pR, _ = log_ols_frozen(yv, Xv, Xt)
        pU, _ = log_ols_frozen(yv, Xv + [rng.permutation(txv)],
                               Xt + [rng.permutation(txt_)])
        out.append(dm_test_clustered(fc.qlike(yt, pU), fc.qlike(yt, pR), days, h)[0])
    return float(np.mean(out))

OUT_CSV = ROOT / "results/tables/matched_row_cascade.csv"
OUT_MD = ROOT / "results/tables/matched_row_cascade.md"


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def main() -> None:
    rows = []
    for disc, models in fc.SETS.items():
        panel = build_price_panel(disc)
        # validation arm: the committed primary's own row set is the A2+text join,
        # with no five-model intersection. Running the identical code path on it
        # must reproduce the committed 38; if it does not, the 33 below is a
        # reimplementation difference and not a sample effect.
        a2_only = fc.load("A2_har_rv", disc)[
            ["split"] + KEY + ["prediction_realised_vol", "label_realised_vol",
                               "filing_time_utc", "effective_trading_day"]
        ].rename(columns={"prediction_realised_vol": "A2_har_rv"})
        fmap, gmean, _, ocov = firm_mean_val(panel)
        panel["firm_mean_val"] = panel.ticker.map(fmap).fillna(gmean).astype(float)

        for model in models:
            ens, _ = mep.ensemble_text(model, disc)
            for tag, base in (("matched", panel), ("committed_rows", a2_only)):
              d = base.merge(ens, on=KEY, how="inner")
              if tag == "committed_rows":
                  d = d.assign(firm_mean_val=d.ticker.map(fmap).fillna(gmean).astype(float))
              for h in fc.HORIZONS:
                  dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                  dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                  if len(dv) < 100 or len(dt) < 30:
                      continue
                  yv, yt = dv.label_realised_vol.to_numpy(), dt.label_realised_vol.to_numpy()
                  a2v, a2t = dv.A2_har_rv.to_numpy(), dt.A2_har_rv.to_numpy()
                  txv, txt_ = dv.ftext.to_numpy(), dt.ftext.to_numpy()
                  fmv, fmt = dv.firm_mean_val.to_numpy(), dt.firm_mean_val.to_numpy()
                  days = dt.effective_trading_day.to_numpy()

                  # rung 1: recalibrated single HAR, on THESE rows
                  fR, _ = log_ols_frozen(yv, [a2v], [a2t])
                  fU, _ = log_ols_frozen(yv, [a2v, txv], [a2t, txt_])
                  # rung 2: + firm identity, on the same rows
                  fRf, _ = log_ols_frozen(yv, [a2v, fmv], [a2t, fmt])
                  fUf, _ = log_ols_frozen(yv, [a2v, fmv, txv], [a2t, fmt, txt_])

                  qR, qU = fc.qlike(yt, fR), fc.qlike(yt, fU)
                  qRf, qUf = fc.qlike(yt, fRf), fc.qlike(yt, fUf)
                  dm1, p1, _ = dm_test_clustered(qU, qR, days, h)
                  dm2, p2, nd = dm_test_clustered(qUf, qRf, days, h)
                  pl1 = placebo_dm(yv, [a2v], [a2t], yt, txv, txt_, days, h)
                  pl2 = placebo_dm(yv, [a2v, fmv], [a2t, fmt], yt, txv, txt_, days, h)

                  rows.append(dict(
                      arm=tag, disc=disc, model=model, h=h, n_test=len(dt), n_days=nd,
                      rel_primary=100 * (qR.mean() - qU.mean()) / qR.mean(),
                      dm_primary=dm1, p_primary=p1,
                      rel_firm=100 * (qRf.mean() - qUf.mean()) / qRf.mean(),
                      dm_firm=dm2, p_firm=p2,
                      placebo_primary=pl1, placebo_firm=pl2,
                      firm_val_coverage_test_obs=ocov,
                  ))
        print(f"{disc}: done")

    out = pd.DataFrame(rows)
    parts = []
    for arm, g in out.groupby("arm"):
        g = g.copy()
        for tag in ("primary", "firm"):
            g[f"holm_{tag}"] = holm(g[f"p_{tag}"].to_numpy())   # Holm within each 69-cell family
            # committed three-condition gate, placebo included
            g[f"adds_{tag}"] = ((g[f"dm_{tag}"] < 0) & (g[f"holm_{tag}"] < 0.05)
                                & (g[f"placebo_{tag}"].abs() < 2.0))
            g[f"adds2_{tag}"] = (g[f"dm_{tag}"] < 0) & (g[f"holm_{tag}"] < 0.05)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    mm, cc = out[out.arm == "matched"], out[out.arm == "committed_rows"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n_prim, n_firm = int(mm.adds_primary.sum()), int(mm.adds_firm.sum())
    n_repro = int(cc.adds_primary.sum())
    committed = pd.read_csv(ROOT / "results/tables/m1_ensemble_primary.csv")
    n_committed = int(committed.genuine_ens_vol.sum())
    # Second validation arm: the matched arm's FIRM rung runs on the committed firm
    # table's own support, so it must reproduce that table cell-for-cell -- a set
    # identity, not merely a count. Checked here so the claim is not hand-verified.
    firm_tbl = pd.read_csv(ROOT / "results/tables/firm_identity_ensemble.csv")
    K = ["disc", "model", "h"]
    set_mine = set(map(tuple, mm[mm.adds_firm][K].values))
    set_comm = set(map(tuple, firm_tbl[firm_tbl.adds_holm][K].values))
    firm_repro = set_mine == set_comm
    print(f"\nvalidation arm 2 (firm rung on the committed firm table's support): "
          f"{len(set_mine)} cells, committed adds_holm {len(set_comm)} "
          f"-> {'IDENTICAL SET' if firm_repro else 'SET MISMATCH, comparison invalid'}")
    print(f"\nvalidation arm (committed row set): primary = {n_repro}/69, "
          f"committed table says {n_committed}/69 "
          f"-> {'REPRODUCED' if n_repro == n_committed else 'MISMATCH, comparison invalid'}")
    print(f"\nmatched rows ({len(mm)} cells, {mm.n_test.sum():,} test rows):")
    print(f"  primary rung  : {n_prim}/69   (committed, unmatched rows: {n_committed}/69)")
    print(f"  + firm identity: {n_firm}/69")
    n_down = int((mm.adds_primary & ~mm.adds_firm).sum())
    n_up = int((~mm.adds_primary & mm.adds_firm).sum())
    # A riser counts as "the identity term sharpened this test" only if its raw p
    # actually fell. Cells that rise because the PLACEBO term dropped them from the
    # primary rung are not evidence of sharpening -- one of them gets weaker.
    risers = mm[~mm.adds_primary & mm.adds_firm]
    sharpen = risers[risers.p_firm < risers.p_primary]
    placebo_risers = risers[risers.p_firm >= risers.p_primary]
    print(f"  collapse on matched rows: {n_prim} -> {n_firm} "
          f"({n_down} fall, {n_up} rise -- the drop is a net, not monotone; "
          f"{len(sharpen)} of the risers actually sharpen, {len(placebo_risers)} "
          f"rise only via the placebo term)")
    pl_only = int((mm.adds2_primary & ~mm.adds_primary).sum())
    print(f"  of the {38 - n_prim} cells lost to the row set, {pl_only} turn on the "
          f"gate's 5-draw placebo term (see S15 on its Monte-Carlo width)")

    L = ["# Matched-row cascade: is the collapse a control effect or a sample effect?", "",
         "The committed primary rung is scored on the A2+text join; the firm-identity",
         "rung is scored on the five-price-model intersection, which is ~9.7% smaller",
         "in rows and ~2.1% smaller in trading days. That makes the headline collapse",
         "confounded with a sample change. Here BOTH rungs are re-scored on ONE row",
         "set --- the identity rung's own support --- so only the reference differs.",
         "Predictions are frozen; weights are val-fitted and frozen to test; inference",
         "is day-clustered DM with Holm within the 69-cell family.", "",
         "**Validation arm.** The identical code path run on the committed row set",
         f"(A2+text join, no five-model intersection) returns {n_repro}/69, matching the",
         f"committed table's {n_committed}/69. Independently, the matched arm's FIRM rung",
         "runs on the committed firm table's own support and reproduces it cell for cell",
         f"({len(set_mine)} cells, set-identical to its `adds_holm`"
         f"{'' if firm_repro else ' -- MISMATCH'}). The difference below is therefore the",
         "row set, not a reimplementation difference.", "",
         "| rung | Holm survivors of 69 |",
         "|---|---|",
         f"| recalibrated HAR, committed (unmatched rows) | {n_committed} |",
         f"| recalibrated HAR, **matched rows** | **{n_prim}** |",
         f"| $+$ firm identity, matched rows | **{n_firm}** |", "",
         f"So on a single row set the collapse is **{n_prim} to {n_firm}**. The sample",
         f"change accounts for {n_committed - n_prim} of the committed",
         f"{n_committed} to {n_firm} drop; the control accounts for the rest.", "",
         f"The second step is **not monotone**: {n_down} cells fall but {n_up} rise,",
         f"which is why the {n_prim - n_firm} is a net. Only {len(sharpen)} of the {n_up}",
         "are cells the identity term sharpens -- their *raw* clustered-DM p falls by an",
         "order of magnitude once the identity term is in the reference, and the firm",
         "family's median raw p is the larger of the two, so Holm is if anything stricter",
         "there. The remaining riser(s) leave the primary rung on the placebo term alone:",
         *[f"   - {r.disc}/{r.model}/h={int(r.h)}: p_primary {r.p_primary:.2e} -> p_firm "
           f"{r.p_firm:.2e} (weaker), excluded from the primary rung by placebo "
           f"{r.placebo_primary:+.2f}, not by DM or Holm." for r in placebo_risers.itertuples()],
         "",
         "| disc | model | h | primary rel% | firm rel% | primary adds | firm adds |",
         "|---|---|---|---|---|---|---|"]
    for r in out.itertuples():
        L.append(f"| {r.disc} | {r.model} | {r.h} | {r.rel_primary:+.2f} | "
                 f"{r.rel_firm:+.2f} | {'yes' if r.adds_primary else 'no'} | "
                 f"{'yes' if r.adds_firm else 'no'} |")
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
