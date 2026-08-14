"""MATCHED-FIRM TEXT SWAP — sharpening the identity bound toward identification.

Reviewer context (R5, AC concern #4): the firm-identity reference BOUNDS the
shortcut share but cannot identify it, because a firm-mean control absorbs genuine
firm-stable text along with the shortcut. A matched-firm text swap separates the two
channels directly:

  Within each calendar day, pair firms whose validation-period mean RV is closest
  (pre-test information only), and SWAP their text forecasts. The swap PRESERVES the
  identity/level channel (each document now speaks for a firm of near-identical
  typical volatility) while DESTROYING the document-firm correspondence (content).

  - If a cell's increment is an identity shortcut, the swapped increment survives
    (retention ~1): the "text" only needed to say which volatility bucket it came from.
  - If the increment is genuine document content, the swap kills it (retention ~0).

This is sharper than the within-date RANDOM permutation (withindate_placebo.py),
which destroys identity and content together. The three points (real, matched-swap,
random-permutation) triangulate the shortcut share the reference interval only
bounds.

Pairing: firm-level greedy nearest-neighbour on firm mean val RV within each day
(distinct firms only; odd firm out unswapped; a firm's k-th filing swaps with its
partner's k-th, unmatched extras unswapped). Swaps are applied on val AND test; the
combiner is re-fit on the swapped val exactly as the placebo convention. DM inference
is day-clustered. Also runs the firm-identity-reference version for the event-driven
C6 cells (the paper's bounded 8-K residual).

Run from repo root:  .venv/bin/python scripts/analysis/matched_firm_swap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc
from clustered_dm import dm_test_clustered
from firm_identity_control import firm_means, fit_apply_log
from withindate_placebo import day_key, rel_pct

KEY, SORT, HORIZONS = fc.KEY, fc.SORT, fc.HORIZONS
GRID_PATH = "results/tables/forecast_combination_grid.csv"
RESIDUAL_CELLS = {("event_driven", "C6_llmtext")}  # firm-ref version also run here


def matched_swap(ftext, tickers, days, rv_map, g_mean):
    """Swap text forecasts between val-RV-matched firm pairs within each day.

    Returns (swapped array, swapped-row fraction). Deterministic: no RNG anywhere.
    """
    ft = np.asarray(ftext, dtype=np.float64)
    out = ft.copy()
    df = pd.DataFrame({"ticker": np.asarray(tickers), "day": np.asarray(days)})
    df["rv"] = df.ticker.map(rv_map).fillna(g_mean)
    n_swapped = 0
    for _, g in df.groupby("day"):
        firms = (g.drop_duplicates("ticker").sort_values(["rv", "ticker"])
                 .ticker.tolist())
        for a, b in zip(firms[0::2], firms[1::2], strict=False):
            ia = g.index[g.ticker == a].to_numpy()
            ib = g.index[g.ticker == b].to_numpy()
            k = min(len(ia), len(ib))
            out[ia[:k]], out[ib[:k]] = ft[ib[:k]], ft[ia[:k]]
            n_swapped += 2 * k
    return out, n_swapped / len(ft)


def main():
    grid = pd.read_csv(GRID_PATH).set_index(["disc", "model", "h"])
    wd = pd.read_csv("results/tables/withindate_placebo.csv").set_index(
        ["disc", "model", "h"])
    rows = []
    for disc, models in fc.SETS.items():
        har = fc.load("A2_har_rv", disc)[
            ["split"] + KEY + ["prediction_realised_vol", "label_realised_vol",
                               "filing_time_utc", "effective_trading_day"]
        ].rename(columns={"prediction_realised_vol": "fhar"})
        for m in models:
            txt = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
            d = har.merge(txt, on=KEY)
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(
                    SORT, kind="mergesort").reset_index(drop=True)
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(
                    SORT, kind="mergesort").reset_index(drop=True)
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv, fhv, ftv = (dv.label_realised_vol.to_numpy(),
                                dv.fhar.to_numpy(), dv.ftext.to_numpy())
                yt, fht, ftt = (dt.label_realised_vol.to_numpy(),
                                dt.fhar.to_numpy(), dt.ftext.to_numpy())
                days_v, days_t = day_key(dv), day_key(dt)
                rv_map = dv.groupby("ticker")["label_realised_vol"].mean()
                g_mean = float(dv.label_realised_vol.mean())

                # ---- REAL (sanity vs the committed grid) ----
                fR, fU, _ = fc.log_combo(yv, fhv, ftv, fht, ftt)
                lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
                real_rel = rel_pct(lR, lU)
                gref = float(grid.loc[(disc, m, h), "rel_impr_pct"])
                assert np.isclose(real_rel, gref, atol=1e-8), \
                    f"sanity fail {disc}/{m}/h{h}: {real_rel} vs grid {gref}"

                # ---- MATCHED-FIRM SWAP vs single recalibrated HAR ----
                ftv_s, frac_v = matched_swap(ftv, dv.ticker, days_v, rv_map, g_mean)
                ftt_s, frac_t = matched_swap(ftt, dt.ticker, days_t, rv_map, g_mean)
                sR, sU, _ = fc.log_combo(yv, fhv, ftv_s, fht, ftt_s)
                slR, slU = fc.qlike(yt, sR), fc.qlike(yt, sU)
                swap_rel = rel_pct(slR, slU)
                swap_dm, swap_p, n_days = dm_test_clustered(slU, slR, days_t, h)
                retention = swap_rel / real_rel if real_rel > 0 else float("nan")

                genuine = bool(grid.loc[(disc, m, h), "genuine"])
                wd_rel = float(wd.loc[(disc, m, h), "wd_placebo_rel_pct_mean"])

                row = {"disc": disc, "model": m, "h": h, "n_test": len(dt),
                       "n_days_test": n_days, "genuine": genuine,
                       "swap_frac_test": frac_t,
                       "real_rel_pct": real_rel, "swap_rel_pct": swap_rel,
                       "swap_dm_clustered": swap_dm, "swap_p_clustered": swap_p,
                       "retention_vs_real": retention,
                       "wd_random_rel_pct": wd_rel}

                # ---- firm-identity-reference version (the 8-K residual cells) ----
                if (disc, m) in RESIDUAL_CELLS:
                    fm_v, fm_t, _ = firm_means(dv, dt)
                    fRf, _ = fit_apply_log(yv, [fhv, fm_v], [fht, fm_t])
                    fUf, _ = fit_apply_log(yv, [fhv, fm_v, ftv], [fht, fm_t, ftt])
                    relf = rel_pct(fc.qlike(yt, fRf), fc.qlike(yt, fUf))
                    fUs, _ = fit_apply_log(yv, [fhv, fm_v, ftv_s], [fht, fm_t, ftt_s])
                    lRf, lUs = fc.qlike(yt, fRf), fc.qlike(yt, fUs)
                    relf_s = rel_pct(lRf, lUs)
                    dmf, pf, _ = dm_test_clustered(lUs, lRf, days_t, h)
                    row.update(firmref_real_rel_pct=relf, firmref_swap_rel_pct=relf_s,
                               firmref_swap_dm=dmf, firmref_swap_p=pf,
                               firmref_retention=(relf_s / relf if relf > 0
                                                  else float("nan")))
                rows.append(row)
                print(f"{disc:12s} {m:16s} h={h:2d}  real={real_rel:+.2f}%  "
                      f"swap={swap_rel:+.2f}%  retention={retention:+.2f}  "
                      f"(random wd={wd_rel:+.2f}%)")

    df = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/matched_firm_swap.csv", index=False)

    gen = df[df.genuine & (df.real_rel_pct > 0)]
    med_ret = float(gen.retention_vs_real.median()) if len(gen) else float("nan")
    lf = gen[gen.disc == "long_form"]; ed = gen[gen.disc == "event_driven"]
    res = df[[c in RESIDUAL_CELLS for c in zip(df.disc, df.model, strict=False)]]

    md = ["# Matched-firm text swap — separating identity from content\n",
          "Within each day, firms are paired by nearest validation-period mean RV and "
          "their text forecasts swapped (identity/level channel preserved, "
          "document-firm correspondence destroyed). Retention = swapped/real "
          "increment. Identity-shortcut cells retain (~1); content cells die (~0). "
          "Random within-date permutation shown for triangulation.\n",
          f"**Genuine cells (n={len(gen)})**: median retention "
          f"**{med_ret:.2f}** (long-form {float(lf.retention_vs_real.median()):.2f} "
          f"over {len(lf)}; event-driven {float(ed.retention_vs_real.median()):.2f} "
          f"over {len(ed)}).\n",
          "## The bounded 8-K residual under the firm-identity reference\n",
          "| h | real rel% | swap rel% | retention | swap DM | p |",
          "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        md.append(f"| {r.h} | {r.firmref_real_rel_pct:+.3f} | "
                  f"{r.firmref_swap_rel_pct:+.3f} | {r.firmref_retention:+.2f} | "
                  f"{r.firmref_swap_dm:+.2f} | {r.firmref_swap_p:.4f} |")
    md.append("\nFull grid: matched_firm_swap.csv (swap coverage, clustered DM, "
              "random-permutation comparison per cell).")
    Path("results/tables/matched_firm_swap.md").write_text("\n".join(md) + "\n")
    print("\nwrote results/tables/matched_firm_swap.{csv,md}")
    print(f"HEADLINE: median retention among genuine cells = {med_ret:.2f}")


if __name__ == "__main__":
    main()
