"""P0-3 — WITHIN-DATE permutation placebo + DATE-MEAN-TEXT control (all 69 cells).

Reviewer context: the original placebo in forecast_combination.py permutes the text
forecast over the WHOLE sample, which destroys BOTH the which-firm (cross-sectional)
and the when (regime-timing) information at once. A signal that survives *because it
merely tracks the aggregate vol regime* (VIX-like, the FinText critique) would still
pass that placebo test's logic in reverse. Two sharper controls separate the channels:

(a) WITHIN-DATE placebo: permute the text forecast ONLY among filings that share the
    same calendar day (effective_trading_day), on val AND test, seeds 1000..1004.
    This destroys which-firm information while preserving ALL date/regime information.
    A real cross-sectional signal DIES under within-date permutation; a pure
    regime-timing signal SURVIVES it.

(b) DATE-MEAN-TEXT combiner: replace each filing's text forecast with the DAY-MEAN of
    all text forecasts that day (per model/horizon cell) — a pure "when" signal with
    zero cross-sectional content — and push it through the same val-fit/test-apply
    log-space combiner. If the date-mean carries most of the increment, the signal is
    regime-timing (bad — VIX-like); if it carries little, it is cross-sectional
    (good — differentiates from FinText).

All DM inference is DAY-CLUSTERED (scripts/analysis/clustered_dm.py): loss
differentials are averaged within calendar day before HAC(lag=h-1) over DAYS.

Leakage: combiner weights are fit on split=="val" only and applied frozen to test,
identically to forecast_combination.py. SANITY: the real (unpermuted) rel% per cell
must reproduce forecast_combination_grid.csv:rel_impr_pct exactly.

Run from the repo root:  .venv/bin/python scripts/analysis/withindate_placebo.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc
from clustered_dm import dm_test_clustered

KEY, SORT, HORIZONS = fc.KEY, fc.SORT, fc.HORIZONS
SEEDS = fc.PLACEBO_SEEDS  # (1000..1004)
GRID_PATH = "results/tables/forecast_combination_grid.csv"


def day_key(df):
    """Calendar-day key: effective_trading_day, fallback filing_time_utc date."""
    d = pd.to_datetime(df["effective_trading_day"], errors="coerce")
    if d.isna().any():
        fb = pd.to_datetime(df["filing_time_utc"], utc=True).dt.tz_localize(None)
        d = d.fillna(fb)
    return d.dt.normalize().to_numpy()


def permute_within_day(vals, days, rng):
    """Permute vals ONLY within groups sharing the same calendar day."""
    vals = np.asarray(vals, dtype=np.float64)
    out = vals.copy()
    idx_map = pd.DataFrame({"day": days}).groupby("day").indices
    for k in sorted(idx_map):  # deterministic group order
        idx = idx_map[k]
        if len(idx) > 1:
            out[idx] = vals[rng.permutation(idx)]
    return out


def day_mean_of(vals, days):
    """Replace each value by the mean of all values sharing its calendar day."""
    s = pd.Series(np.asarray(vals, dtype=np.float64))
    return s.groupby(pd.Series(days).to_numpy()).transform("mean").to_numpy()


def rel_pct(lR, lU):
    qR, qU = float(np.mean(lR)), float(np.mean(lU))
    return 100.0 * (qR - qU) / qR if qR > 0 else float("nan")


def main():
    grid = pd.read_csv(GRID_PATH).set_index(["disc", "model", "h"])
    rows, sanity_bad = [], []
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
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv, fhv, ftv = dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(), dv.ftext.to_numpy()
                yt, fhr, ftt = dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(), dt.ftext.to_numpy()
                days_v, days_t = day_key(dv), day_key(dt)

                # ---- REAL cell (must reproduce forecast_combination_grid.csv) ----
                fR, fU, g_log = fc.log_combo(yv, fhv, ftv, fhr, ftt)
                lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
                real_rel = rel_pct(lR, lU)
                dm_c, p_c, n_days = dm_test_clustered(lU, lR, days_t, h)

                gref = float(grid.loc[(disc, m, h), "rel_impr_pct"])
                if not np.isclose(real_rel, gref, atol=1e-8):
                    sanity_bad.append((disc, m, h, real_rel, gref))

                # ---- (a) WITHIN-DATE permutation placebo, 5 seeds ----
                wd_rel, wd_dm, wd_p = [], [], []
                for s in SEEDS:
                    rng = np.random.default_rng(s)
                    ftv_p = permute_within_day(ftv, days_v, rng)
                    ftt_p = permute_within_day(ftt, days_t, rng)
                    pR, pU, _ = fc.log_combo(yv, fhv, ftv_p, fhr, ftt_p)
                    plR, plU = fc.qlike(yt, pR), fc.qlike(yt, pU)
                    wd_rel.append(rel_pct(plR, plU))
                    st, pv, _ = dm_test_clustered(plU, plR, days_t, h)
                    wd_dm.append(st); wd_p.append(pv)
                wd_rel_m, wd_dm_m = float(np.mean(wd_rel)), float(np.mean(wd_dm))

                # ---- (b) DATE-MEAN-TEXT combiner (pure "when" signal) ----
                ftv_d = day_mean_of(ftv, days_v)
                ftt_d = day_mean_of(ftt, days_t)
                gR, gU, g_dm = fc.log_combo(yv, fhv, ftv_d, fhr, ftt_d)
                glR, glU = fc.qlike(yt, gR), fc.qlike(yt, gU)
                dmean_rel = rel_pct(glR, glU)
                dmean_dm, dmean_p, _ = dm_test_clustered(glU, glR, days_t, h)

                # ---- decomposition shares (only meaningful when the real increment > 0) ----
                if real_rel > 0:
                    frac_regime = float(np.clip(dmean_rel / real_rel, 0.0, 1.0))
                    frac_cross = 1.0 - frac_regime
                    wd_survival = float(np.clip(wd_rel_m / real_rel, 0.0, 1.0))
                else:
                    frac_regime = frac_cross = wd_survival = float("nan")

                genuine = bool(grid.loc[(disc, m, h), "genuine"])
                if not genuine:
                    verdict = "n/a (not genuine)"
                elif frac_regime >= 0.5 or wd_survival >= 0.5:
                    verdict = "regime-timing"
                elif frac_regime <= 0.25 and wd_survival <= 0.25:
                    verdict = "cross-sectional"
                else:
                    verdict = "mixed"

                rows.append({
                    "disc": disc, "model": m, "h": h, "n_test": len(dt), "n_days_test": n_days,
                    "genuine": genuine,
                    "real_rel_pct": real_rel, "real_dm_clustered": dm_c, "real_p_clustered": p_c,
                    "grid_rel_pct": gref, "sanity_match": bool(np.isclose(real_rel, gref, atol=1e-8)),
                    "wd_placebo_rel_pct_mean": wd_rel_m, "wd_placebo_dm_clustered_mean": wd_dm_m,
                    "wd_placebo_p_clustered_mean": float(np.mean(wd_p)),
                    "datemean_rel_pct": dmean_rel, "datemean_dm_clustered": dmean_dm,
                    "datemean_p_clustered": dmean_p,
                    "frac_regime_timing": frac_regime, "frac_cross_sectional": frac_cross,
                    "wd_placebo_survival": wd_survival, "verdict": verdict,
                })
                print(f"{disc:12s} {m:16s} h={h:2d}  real={real_rel:+.2f}%  "
                      f"wd={wd_rel_m:+.2f}%  datemean={dmean_rel:+.2f}%  {verdict}")

    df = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/withindate_placebo.csv", index=False)

    gen = df[df.genuine]
    med_regime = float(gen.frac_regime_timing.median()) if len(gen) else float("nan")
    med_cross = float(gen.frac_cross_sectional.median()) if len(gen) else float("nan")
    med_surv = float(gen.wd_placebo_survival.median()) if len(gen) else float("nan")
    vc = gen.verdict.value_counts().to_dict()

    md = ["# P0-3 — Within-date permutation placebo + date-mean-text control "
          "(cross-sectional vs regime-timing decomposition)\n",
          "## RESTATED vs ORIGINAL\n",
          "| | ORIGINAL (forecast_combination_grid.csv) | RESTATED (this table) |",
          "|---|---|---|",
          "| Placebo design | whole-sample permutation of the text forecast — destroys "
          "which-firm AND when information simultaneously | (a) WITHIN-DATE permutation "
          "(keeps all date/regime info, kills which-firm); (b) DATE-MEAN-TEXT combiner "
          "(pure when-signal, zero cross-sectional content) |",
          "| Inference | observation-order HAC DM (reviewer-verified ~2x inflated) | "
          "day-clustered DM: daily-mean loss differentials, HAC lag=h-1 over DAYS |",
          f"| What it can say | 'the increment is not a procedural artifact' | of each genuine "
          f"increment, what fraction is cross-sectional (which-firm) vs regime-timing (when): "
          f"median regime-timing share = {med_regime:.2f}, median cross-sectional share = "
          f"{med_cross:.2f}, median within-date-placebo survival = {med_surv:.2f} |\n",
          f"**Cells:** {len(df)} total, {int(df.genuine.sum())} genuine (per the original grid's "
          f"placebo-confirmed flag). SANITY: real rel% reproduces the grid in "
          f"{int(df.sanity_match.sum())}/{len(df)} cells.\n",
          f"**Verdicts over the {len(gen)} genuine cells:** " +
          ", ".join(f"{k} = {v}" for k, v in sorted(vc.items())) + ".\n",
          "Reading guide: `wd_placebo_*` = mean over seeds 1000-1004 of the within-date "
          "permutation (a); if the increment DIES here (rel%~0, |DM|<2) the signal is "
          "cross-sectional. `datemean_*` = the day-mean-text combiner (b); the fraction "
          "`frac_regime_timing = datemean_rel / real_rel` (clipped to [0,1]) is the share of the "
          "increment a pure when-signal already delivers. All DM stats are day-clustered "
          "(negative = text-augmented combiner better).\n"]

    for disc in fc.SETS:
        md.append(f"\n## {disc}\n"
                  "| model | h | n_days | real rel% | real DM(cl) | wd-placebo rel% | wd-placebo DM(cl) | "
                  "date-mean rel% | date-mean DM(cl) | frac regime | frac cross-sec | verdict |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df[df.disc == disc].sort_values(["model", "h"]).iterrows():
            md.append(
                f"| {r.model} | {r.h} | {r.n_days_test} | {r.real_rel_pct:+.2f} | "
                f"{r.real_dm_clustered:+.2f} | {r.wd_placebo_rel_pct_mean:+.2f} | "
                f"{r.wd_placebo_dm_clustered_mean:+.2f} | {r.datemean_rel_pct:+.2f} | "
                f"{r.datemean_dm_clustered:+.2f} | "
                f"{'' if np.isnan(r.frac_regime_timing) else format(r.frac_regime_timing, '.2f')} | "
                f"{'' if np.isnan(r.frac_cross_sectional) else format(r.frac_cross_sectional, '.2f')} | "
                f"{r.verdict} |")

    n_cs = vc.get("cross-sectional", 0); n_rt = vc.get("regime-timing", 0); n_mx = vc.get("mixed", 0)
    md.append(f"\n## Verdict\n"
              f"- Of the {len(gen)} genuine cells: **{n_cs} cross-sectional, {n_rt} regime-timing, "
              f"{n_mx} mixed** (rule: regime if date-mean share >= 0.5 or within-date placebo "
              f"survival >= 0.5; cross-sectional if both <= 0.25).\n"
              f"- Median decomposition of a genuine increment: **{med_cross:.0%} cross-sectional "
              f"(which-firm) vs {med_regime:.0%} regime-timing (when)**; the within-date placebo "
              f"retains a median {med_surv:.0%} of the real increment.\n"
              f"- Interpretation: a LOW regime share differentiates the finding from FinText/VIX-style "
              f"aggregate-regime tracking; a HIGH regime share would mean the 'text signal' is mostly "
              f"a calendar effect.")

    with open("results/tables/withindate_placebo.md", "w") as fh:
        fh.write("\n".join(md))

    print(f"\ncells={len(df)} genuine={len(gen)} sanity_match={int(df.sanity_match.sum())}/{len(df)}")
    print(f"median frac_regime={med_regime:.3f} frac_cross={med_cross:.3f} wd_survival={med_surv:.3f}")
    print(f"verdicts (genuine): {vc}")
    if sanity_bad:
        print("SANITY FAILURES:", sanity_bad[:10])
    print("wrote results/tables/withindate_placebo.{csv,md}")


if __name__ == "__main__":
    main()
