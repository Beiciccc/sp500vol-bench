"""B4 — PRICE-FRONTIER COMPLETENESS AUDIT: does the maximal pool leave price models out?

Reviewer charge (R6 quant_finance #8, and it connects to their rank-1 about the daily
RV proxy): the "maximal" price pool has five members (HAR, SHAR, GARCH, EGARCH, ARIMA)
while HARQ and HAR-X(VIX) are archived, computed, and absent. HARQ especially, since its
whole purpose is to correct RV *measurement error* — exactly the weakness a daily
close-to-close proxy has. The paper dismisses HARQ in one parenthetical
("log-space-unstable, disclosed").

This is not a mechanical fix, because the direction of the effect is not obvious:
  - adding a GOOD member makes the pool stronger -> fewer text survivors -> the null
    gets STRONGER (conservative, safe);
  - adding an UNSTABLE member can make a val-fitted pool WORSE out of sample -> more
    text survivors -> that would be reverse reference-shopping, i.e. the paper would be
    crediting text for the pool's own breakage.
So we measure it instead of asserting it. Either outcome answers the charge: either the
frontier is completed, or the one-line dismissal is replaced by evidence.

Reports, per (disclosure x horizon) panel, all weights val-fit and frozen to test:
  * test QLIKE of the 5-member pool vs 6-member (+HARQ) vs 6-member (+HAR-X) vs 7-member
  * the fitted log weights (an unstable member shows up as an exploded coefficient)
  * whether each larger pool beats the 5-member pool (day-clustered DM)

Run:  .venv/bin/python scripts/analysis/pool_frontier_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc
from clustered_dm import dm_test_clustered
from maximal_reference import fit_apply_log

KEY, SORT, HORIZONS = fc.KEY, fc.SORT, fc.HORIZONS
BASE5 = ["A2_har_rv", "A6_shar", "A3_garch", "A4_egarch", "A5_arima"]
CANDIDATES = {"harq": "A6_harq", "harx_vix": "A7_harx_vix"}
POOLS = {
    "pool5 (paper)": BASE5,
    "pool6 +HARQ": BASE5 + ["A6_harq"],
    "pool6 +HARX": BASE5 + ["A7_harx_vix"],
    "pool7 (all)": BASE5 + ["A6_harq", "A7_harx_vix"],
}


def load_panel(disc, models):
    base = fc.load("A2_har_rv", disc)[
        ["split"] + KEY + ["label_realised_vol", "filing_time_utc",
                           "effective_trading_day", "prediction_realised_vol"]
    ].rename(columns={"prediction_realised_vol": "f_A2_har_rv"})
    for pm in models:
        if pm == "A2_har_rv":
            continue
        f = fc.load(pm, disc)[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": f"f_{pm}"})
        base = base.merge(f, on=KEY, how="inner")
    return base


def main():
    all_models = sorted(set(BASE5 + list(CANDIDATES.values())))
    rows = []
    for disc in fc.SETS:
        panel = load_panel(disc, all_models)
        for h in HORIZONS:
            dv = panel[(panel.horizon_days == h) & (panel.split == "val")].sort_values(
                SORT, kind="mergesort")
            dt = panel[(panel.horizon_days == h) & (panel.split == "test")].sort_values(
                SORT, kind="mergesort")
            if len(dv) < 100 or len(dt) < 30:
                continue
            yv, yt = dv.label_realised_vol.to_numpy(), dt.label_realised_vol.to_numpy()
            days_t = dt.effective_trading_day.to_numpy()
            losses = {}
            for name, members in POOLS.items():
                fR, beta = fit_apply_log(yv, [dv[f"f_{m}"].to_numpy() for m in members],
                                         [dt[f"f_{m}"].to_numpy() for m in members])
                losses[name] = fc.qlike(yt, fR)
                rows.append({
                    "disc": disc, "h": h, "pool": name, "n_members": len(members),
                    "qlike_test": float(losses[name].mean()),
                    "max_abs_weight": float(np.abs(beta[1:]).max()),
                    "weights": ";".join(f"{m}={b:+.2f}" for m, b in zip(members, beta[1:], strict=False)),
                })
            # each larger pool vs the paper's pool5, day-clustered
            for name in POOLS:
                if name == "pool5 (paper)":
                    continue
                dm, p, _ = dm_test_clustered(losses[name], losses["pool5 (paper)"], days_t, h)
                i = next(k for k, r in enumerate(rows)
                         if r["disc"] == disc and r["h"] == h and r["pool"] == name)
                rows[i].update(dm_vs_pool5=dm, p_vs_pool5=p,
                               better_than_pool5=bool(dm < 0 and p < 0.05))
    df = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/pool_frontier_audit.csv", index=False)

    print(df[["disc", "h", "pool", "qlike_test", "max_abs_weight",
              "dm_vs_pool5", "p_vs_pool5"]].to_string(index=False))
    print("\n--- per-pool mean test QLIKE across panels ---")
    print(df.groupby("pool").qlike_test.mean().sort_values().to_string())
    print("\n--- weight explosion check (an unstable member shows as a large |weight|) ---")
    print(df.groupby("pool").max_abs_weight.max().to_string())
    wins = df[df.pool != "pool5 (paper)"].groupby("pool").better_than_pool5.sum()
    print("\n--- panels where the larger pool is clustered-significantly better than pool5 "
          f"(of {df.disc.nunique() * len(HORIZONS)}) ---")
    print(wins.to_string())
    print("\nwrote results/tables/pool_frontier_audit.csv")


if __name__ == "__main__":
    main()


def cascade_vs_pool(pool_name, members):
    """69-cell text-increment grid against a given pool reference (val-fit, test-frozen,
    day-clustered DM, Holm within the 69-cell family) — the number the paper reports."""
    rows = []
    for disc, models in fc.SETS.items():
        panel = load_panel(disc, sorted(set(BASE5 + list(CANDIDATES.values()))))
        for m in models:
            txt = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
            d = panel.merge(txt, on=KEY, how="inner")
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv, yt = dv.label_realised_vol.to_numpy(), dt.label_realised_vol.to_numpy()
                days_t = dt.effective_trading_day.to_numpy()
                Xv = [dv[f"f_{p}"].to_numpy() for p in members]
                Xt = [dt[f"f_{p}"].to_numpy() for p in members]
                fR, _ = fit_apply_log(yv, Xv, Xt)
                fU, _ = fit_apply_log(yv, Xv + [dv.ftext.to_numpy()], Xt + [dt.ftext.to_numpy()])
                lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
                dm, p, _ = dm_test_clustered(lU, lR, days_t, h)
                rows.append({"disc": disc, "model": m, "h": h, "pool": pool_name,
                             "rel_pct": 100 * (lR.mean() - lU.mean()) / lR.mean(),
                             "dm": dm, "p": p})
    df = pd.DataFrame(rows)
    # Holm within the family
    df = df.sort_values("p").reset_index(drop=True)
    n = len(df)
    df["holm_p"] = np.minimum.accumulate(
        np.minimum(1.0, (n - np.arange(n)) * df.p.to_numpy())[::-1])[::-1]
    df["genuine_raw"] = (df.dm < 0) & (df.p < 0.05)
    df["genuine_holm"] = (df.dm < 0) & (df.holm_p < 0.05)
    return df


if __name__ != "__main__":
    pass
