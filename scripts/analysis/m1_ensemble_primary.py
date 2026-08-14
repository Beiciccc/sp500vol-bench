"""P1-a remediation — SEED-ENSEMBLE PRIMARY + VARIANCE-UNIT PRIMARY for the M1 grid.

Fixes two reviewer-verified defects of results/tables/forecast_combination_grid.csv:
  1. Hardcoded seed2026 primary: C/D models with 3 seeds (2026/2027/2028) are now
     ensembled by averaging the per-observation prediction across seeds (join on
     ticker, accession, horizon_days) BEFORE the M1 combiner. A/B (seed-invariant)
     and C6_llmtext / D4_llmfused (seed2026 only) stay single-seed.
  2. Observation-order HAC DM: all DM inference now uses the day-clustered DM
     (scripts/analysis/clustered_dm.py): daily-mean loss differentials over calendar
     days of effective_trading_day, HAC lag = h-1 in DAYS, n = number of days,
     day-block moving bootstrap.

Additionally reruns the same seed-ensemble grid with the EVALUATION loss in
VARIANCE units, q(y^2, f^2) — the combiner is untouched (log-space OLS is
unit-free); only the loss convention for evaluation/DM changes — and flags
convention-dependent cells.

Leakage discipline unchanged: combiner weights fit on split=="val" only, applied
frozen to split=="test".

Run from the repo root:  .venv/bin/python scripts/analysis/m1_ensemble_primary.py

Outputs (NEW files; originals untouched for before/after):
  results/tables/m1_ensemble_primary.{csv,md}
  results/tables/m1_variance_unit.{csv,md}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")
import forecast_combination as fc  # noqa: E402
from clustered_dm import daily_mean, dm_test_clustered, mbb_ci_daily  # noqa: E402
from sp500vol.evaluation.dm_test import _hac_variance  # noqa: E402

KEY = fc.KEY
SORT = fc.SORT
HORIZONS = fc.HORIZONS
PLACEBO_SEEDS = fc.PLACEBO_SEEDS
SETS = fc.SETS
SEEDS = (2026, 2027, 2028)
ORIG_GRID = "results/tables/m1_original_grid_snapshot.csv"  # not written; read live below


def run_dir(run, disc, seed):
    return Path(f"results/runs/{run}_full_{disc}_seed{seed}/predictions.parquet")


def ensemble_text(run, disc):
    """Average prediction_realised_vol across available seeds (inner join on KEY)."""
    frames, used = [], []
    for s in SEEDS:
        p = run_dir(run, disc, s)
        if p.exists():
            frames.append(
                pd.read_parquet(p)[KEY + ["prediction_realised_vol"]].rename(
                    columns={"prediction_realised_vol": f"f{s}"}
                )
            )
            used.append(s)
    if not frames:
        raise FileNotFoundError(f"no seed runs for {run}/{disc}")
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=KEY, how="inner")
    out["ftext"] = out[[f"f{s}" for s in used]].mean(axis=1)
    return out[KEY + ["ftext"]], used


def qlike_var(y, f):
    """QLIKE in VARIANCE units: q(y^2, f^2)."""
    return fc.qlike(np.asarray(y, float) ** 2, np.asarray(f, float) ** 2)


def clark_west_clustered(y, f_small, f_big, days, h):
    """Clark-West with the adjusted-MSPE series aggregated to daily means first."""
    y = np.asarray(y, float)
    fhat = (y - f_small) ** 2 - (y - f_big) ** 2 + (
        np.asarray(f_small, float) - np.asarray(f_big, float)
    ) ** 2
    dd, _ = daily_mean(fhat, days)
    n = len(dd)
    m = float(dd.mean())
    v = _hac_variance(dd, lag=max(h - 1, 0))
    if v <= 0:
        return (0.0, 1.0) if np.isclose(m, 0.0) else (float("nan"), float("nan"))
    t = m / np.sqrt(v / n)
    return float(t), float(stats.t.sf(t, df=n - 1))


def cell_stats(yv, fhv, ftv, yt, fhr, ftt, days_t, h):
    """One grid cell: log-space combiner (val-fit) + clustered inference on test.

    Returns dict with vol-unit and variance-unit evaluation of the SAME forecasts.
    """
    fR, fU, g_log = fc.log_combo(yv, fhv, ftv, fhr, ftt)
    out = {"g_log": float(g_log)}
    # placebo forecasts computed once, evaluated in both units
    placebo = []
    for s in PLACEBO_SEEDS:
        rng = np.random.default_rng(s)
        pR, pU, _ = fc.log_combo(yv, fhv, rng.permutation(ftv), fhr, rng.permutation(ftt))
        placebo.append((pR, pU))
    for unit, lossfn in (("vol", fc.qlike), ("var", qlike_var)):
        lraw = lossfn(yt, fhr)
        lR, lU = lossfn(yt, fR), lossfn(yt, fU)
        dmq, pq, n_days = dm_test_clustered(lU, lR, days_t, h)
        mean_d, lo, hi = mbb_ci_daily(lU - lR, days_t, h)
        qR, qU = float(lR.mean()), float(lU.mean())
        pdm = [dm_test_clustered(lossfn(yt, pU), lossfn(yt, pR), days_t, h)[0]
               for pR, pU in placebo]
        out.update({
            f"{unit}_qlike_raw": float(lraw.mean()),
            f"{unit}_qlike_R": qR, f"{unit}_qlike_U": qU,
            f"{unit}_rel_impr_pct": 100.0 * (qR - qU) / qR if qR > 0 else float("nan"),
            f"{unit}_dm_q_clu": dmq, f"{unit}_p_q_clu": pq,
            f"{unit}_boot_lo_daily": lo, f"{unit}_boot_hi_daily": hi,
            f"{unit}_placebo_dm_clu": float(np.mean(pdm)),
        })
        if unit == "vol":
            out["n_days"] = n_days
            cw_t, cw_p = clark_west_clustered(yt, fR, fU, days_t, h)
            out["cw_t_clu"], out["cw_p_clu"] = cw_t, cw_p
    return out


def main():
    orig = pd.read_csv("results/tables/forecast_combination_grid.csv")

    rows = []
    for disc, models in SETS.items():
        har = pd.read_parquet(run_dir("A2_har_rv", disc, 2026))[
            ["split"] + KEY + ["prediction_realised_vol", "label_realised_vol",
                               "filing_time_utc", "effective_trading_day"]
        ].rename(columns={"prediction_realised_vol": "fhar"})
        for m in models:
            ens, used = ensemble_text(m, disc)
            s26 = pd.read_parquet(run_dir(m, disc, 2026))[
                KEY + ["prediction_realised_vol"]
            ].rename(columns={"prediction_realised_vol": "ftext"})
            d_ens = har.merge(ens, on=KEY)
            d_s26 = har.merge(s26, on=KEY)
            for h in HORIZONS:
                out = {"disc": disc, "model": m, "h": h,
                       "n_seeds": len(used), "seeds": "+".join(str(s) for s in used)}
                for tag, d in (("ens", d_ens), ("s26", d_s26)):
                    dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                    dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                    if len(dv) < 100 or len(dt) < 30:
                        continue
                    st = cell_stats(
                        dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(), dv.ftext.to_numpy(),
                        dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(), dt.ftext.to_numpy(),
                        dt.effective_trading_day.to_numpy(), h,
                    )
                    if tag == "ens":
                        out["n_test"] = len(dt)
                        out.update(st)
                    else:  # seed2026-only, clustered DM (attribution + sanity)
                        out.update({
                            "s26_qlike_R": st["vol_qlike_R"], "s26_qlike_U": st["vol_qlike_U"],
                            "s26_g_log": st["g_log"],
                            "s26_dm_q_clu": st["vol_dm_q_clu"], "s26_p_q_clu": st["vol_p_q_clu"],
                            "s26_placebo_dm_clu": st["vol_placebo_dm_clu"],
                        })
                rows.append(out)

    df = pd.DataFrame(rows)

    # Holm within each analysis grid (matches original: one family across all cells)
    df["vol_dmq_holm_clu"] = fc.holm(df.vol_p_q_clu.fillna(1.0).values)
    df["var_dmq_holm_clu"] = fc.holm(df.var_p_q_clu.fillna(1.0).values)
    df["s26_dmq_holm_clu"] = fc.holm(df.s26_p_q_clu.fillna(1.0).values)

    def genuine(dm, holm_p, placebo):
        return (dm < 0) & (holm_p < 0.05) & (placebo.abs() < 2.0)

    df["genuine_ens_vol"] = genuine(df.vol_dm_q_clu, df.vol_dmq_holm_clu, df.vol_placebo_dm_clu)
    df["genuine_ens_var"] = genuine(df.var_dm_q_clu, df.var_dmq_holm_clu, df.var_placebo_dm_clu)
    df["genuine_s26_clu"] = genuine(df.s26_dm_q_clu, df.s26_dmq_holm_clu, df.s26_placebo_dm_clu)

    # merge the ORIGINAL grid (seed2026, observation-order HAC)
    o = orig[["disc", "model", "h", "n_test", "qlike_R", "qlike_U", "rel_impr_pct",
              "g_log", "dm_q", "dmq_holm", "placebo_dm", "genuine"]].rename(
        columns={"n_test": "orig_n_test", "qlike_R": "orig_qlike_R",
                 "qlike_U": "orig_qlike_U", "rel_impr_pct": "orig_rel_impr_pct",
                 "g_log": "orig_g_log", "dm_q": "orig_dm_q", "dmq_holm": "orig_dmq_holm",
                 "placebo_dm": "orig_placebo_dm", "genuine": "orig_genuine"})
    df = df.merge(o, on=["disc", "model", "h"], how="left")

    df["status_vs_original"] = np.select(
        [df.orig_genuine & df.genuine_ens_vol,
         df.orig_genuine & ~df.genuine_ens_vol,
         ~df.orig_genuine & df.genuine_ens_vol],
        ["SURVIVES", "LOST", "GAINED"], default="null-null")
    df["convention_dependent"] = df.genuine_ens_vol != df.genuine_ens_var

    # ---- SANITY (a): single-seed rows must reproduce the original grid exactly ----
    single = df[df.n_seeds == 1]
    sanity = {
        "n_single_seed_cells": int(len(single)),
        "max_abs_diff_qlike_R": float((single.vol_qlike_R - single.orig_qlike_R).abs().max()),
        "max_abs_diff_qlike_U": float((single.vol_qlike_U - single.orig_qlike_U).abs().max()),
        "max_abs_diff_g_log": float((single.g_log - single.orig_g_log).abs().max()),
        "n_test_mismatch": int((df.n_test != df.orig_n_test).sum()),
    }
    sanity["pass"] = bool(sanity["max_abs_diff_qlike_R"] < 1e-9
                          and sanity["max_abs_diff_qlike_U"] < 1e-9
                          and sanity["max_abs_diff_g_log"] < 1e-9)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/m1_ensemble_primary.csv", index=False)

    n = len(df)
    n_orig_gen = int(df.orig_genuine.sum())
    n_ens_gen = int(df.genuine_ens_vol.sum())
    n_s26_gen = int(df.genuine_s26_clu.sum())
    n_surv = int((df.status_vs_original == "SURVIVES").sum())
    n_lost = int((df.status_vs_original == "LOST").sum())
    n_gain = int((df.status_vs_original == "GAINED").sum())
    n_var_gen = int(df.genuine_ens_var.sum())
    n_conv = int(df.convention_dependent.sum())

    # ================= m1_ensemble_primary.md =================
    md = ["# M1 PRIMARY (restated) — seed-ensemble predictions + day-clustered DM\n",
          "## RESTATED vs ORIGINAL\n",
          "| | ORIGINAL (seed2026 only, obs-order HAC DM) | RESTATED (3-seed ensemble, day-clustered DM) |",
          "|---|---|---|",
          f"| genuine text-increment cells | {n_orig_gen}/{n} | **{n_ens_gen}/{n}** |",
          f"| inference unit | n_obs (~10-25 same-day filings treated independent) | n_days (median {int(df.n_days.median())} days), HAC lag=h-1 in DAYS |",
          f"| seed handling | hardcoded seed2026 | per-observation mean across seeds {SEEDS} for 3-seed C/D models |",
          "",
          f"Attribution: seed2026-only + clustered DM (clustering alone) gives {n_s26_gen}/{n} genuine cells; "
          f"the ensemble step then moves this to {n_ens_gen}/{n}. Of the {n_orig_gen} originally-genuine cells: "
          f"**{n_surv} SURVIVE**, **{n_lost} are LOST**, and **{n_gain} cells are newly GAINED**.\n",
          "Genuine = clustered DM-QLIKE < 0, Holm(clustered p) < .05 across the "
          f"{n}-cell grid, |clustered placebo DM| < 2. Combiner weights fit on validation only, frozen on test "
          "(unchanged). This table replaces the hardcoded-seed2026 primary and neutralizes the "
          "seed2027-flip objection: the primary object is now the seed-averaged forecast.\n",
          f"**Sanity (a): single-seed rows (A/B-anchored text models, C6, D4) reproduce the original grid "
          f"QLIKE columns exactly — max|dQLIKE_R|={sanity['max_abs_diff_qlike_R']:.2e}, "
          f"max|dQLIKE_U|={sanity['max_abs_diff_qlike_U']:.2e}, max|dg_log|={sanity['max_abs_diff_g_log']:.2e} "
          f"over {sanity['n_single_seed_cells']} cells: {'PASS' if sanity['pass'] else 'FAIL'}.**\n"]

    for disc in SETS:
        md.append(f"\n## {disc} — ensemble primary (vol-unit QLIKE, day-clustered)\n"
                  "| model | h | seeds | n_test | n_days | QLIKE(R) | QLIKE(U) | rel% | g_log | "
                  "DM-Q(clu) | p(clu) | Holm | placebo DM(clu) | CW t(clu) | orig DM-Q | orig Holm | "
                  "orig genuine | NEW genuine | status |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df[df.disc == disc].sort_values(["model", "h"]).iterrows():
            md.append(
                f"| {r.model} | {r.h} | {r.seeds} | {int(r.n_test)} | {int(r.n_days)} | "
                f"{r.vol_qlike_R:.4f} | {r.vol_qlike_U:.4f} | {r.vol_rel_impr_pct:+.2f} | {r.g_log:+.3f} | "
                f"{r.vol_dm_q_clu:+.2f} | {r.vol_p_q_clu:.4f} | {r.vol_dmq_holm_clu:.3f} | "
                f"{r.vol_placebo_dm_clu:+.2f} | {r.cw_t_clu:+.2f} | {r.orig_dm_q:+.2f} | "
                f"{r.orig_dmq_holm:.3f} | {'YES' if r.orig_genuine else 'no'} | "
                f"{'YES' if r.genuine_ens_vol else 'no'} | {r.status_vs_original} |")

    lost = df[df.status_vs_original == "LOST"]
    gained = df[df.status_vs_original == "GAINED"]
    md.append("\n## Flips\n")
    md.append("**LOST** (genuine under seed2026+obs-order DM, NOT genuine under ensemble+clustered): "
              + ("; ".join(f"{r.disc}/{r.model}/h{r.h} (orig Holm={r.orig_dmq_holm:.3f} -> "
                           f"clu Holm={r.vol_dmq_holm_clu:.3f})" for _, r in lost.iterrows()) or "none") + "\n")
    md.append("**GAINED**: " + ("; ".join(f"{r.disc}/{r.model}/h{r.h}" for _, r in gained.iterrows()) or "none") + "\n")
    md.append(f"\n## Bottom line\n- **{n_ens_gen}/{n}** cells keep a genuine, placebo-confirmed text increment "
              f"under the honest primary (seed-ensemble + day-clustered DM), vs {n_orig_gen}/{n} originally; "
              f"{n_lost} lost, {n_gain} gained, {n_surv} survive.\n"
              f"- Clustering alone (seed2026): {n_s26_gen}/{n} — the drop from {n_orig_gen} is the "
              f"t-inflation the reviewer flagged; the ensemble step recovers/moves cells on top of that.\n")
    with open("results/tables/m1_ensemble_primary.md", "w") as fh:
        fh.write("\n".join(md))

    # ================= m1_variance_unit.md =================
    vcols = ["disc", "model", "h", "n_seeds", "n_test", "n_days",
             "vol_qlike_R", "vol_qlike_U", "vol_rel_impr_pct", "vol_dm_q_clu", "vol_dmq_holm_clu",
             "var_qlike_raw", "var_qlike_R", "var_qlike_U", "var_rel_impr_pct",
             "var_dm_q_clu", "var_p_q_clu", "var_dmq_holm_clu", "var_placebo_dm_clu",
             "var_boot_lo_daily", "var_boot_hi_daily",
             "genuine_ens_vol", "genuine_ens_var", "convention_dependent"]
    df[vcols].to_csv("results/tables/m1_variance_unit.csv", index=False)

    dep = df[df.convention_dependent]
    mdv = ["# M1 — VARIANCE-UNIT primary (same seed-ensemble forecasts, QLIKE on variances)\n",
           "## RESTATED vs ORIGINAL\n",
           "The original grid evaluated QLIKE in VOLATILITY units, q(y, f) — a non-standard convention "
           "(the QLIKE literature works on variances). Here the SAME seed-ensemble forecasts and the SAME "
           "log-space val-fit combiner (unit-free) are evaluated with q(y^2, f^2), day-clustered DM. "
           "Only the evaluation loss changes.\n",
           "| | vol-unit (restated primary) | variance-unit (this table) |",
           "|---|---|---|",
           f"| genuine cells | {n_ens_gen}/{n} | **{n_var_gen}/{n}** |",
           f"| convention-dependent cells (verdict differs) | — | **{n_conv}** |",
           ""]
    for disc in SETS:
        mdv.append(f"\n## {disc} — variance-unit evaluation (ensemble, day-clustered)\n"
                   "| model | h | QLIKEv(R) | QLIKEv(U) | rel% | DM-Qv(clu) | p | Holm | placebo DM | "
                   "genuine(var) | genuine(vol) | convention-dep |\n"
                   "|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df[df.disc == disc].sort_values(["model", "h"]).iterrows():
            mdv.append(
                f"| {r.model} | {r.h} | {r.var_qlike_R:.4f} | {r.var_qlike_U:.4f} | "
                f"{r.var_rel_impr_pct:+.2f} | {r.var_dm_q_clu:+.2f} | {r.var_p_q_clu:.4f} | "
                f"{r.var_dmq_holm_clu:.3f} | {r.var_placebo_dm_clu:+.2f} | "
                f"{'YES' if r.genuine_ens_var else 'no'} | {'YES' if r.genuine_ens_vol else 'no'} | "
                f"{'FLIP' if r.convention_dependent else '-'} |")
    mdv.append("\n## Convention-dependent cells\n")
    if len(dep):
        mdv.append("| disc | model | h | vol verdict | var verdict | vol Holm | var Holm |\n|---|---|---|---|---|---|---|")
        for _, r in dep.iterrows():
            mdv.append(f"| {r.disc} | {r.model} | {r.h} | {'genuine' if r.genuine_ens_vol else 'no'} | "
                       f"{'genuine' if r.genuine_ens_var else 'no'} | {r.vol_dmq_holm_clu:.3f} | {r.var_dmq_holm_clu:.3f} |")
    else:
        mdv.append("None — every verdict is invariant to the vol-vs-variance QLIKE convention.")
    mdv.append(f"\n## Bottom line\n- Variance-unit evaluation gives **{n_var_gen}/{n}** genuine cells vs "
               f"{n_ens_gen}/{n} in vol units; **{n_conv}** cells are convention-dependent"
               + (" (listed above); conclusions for those cells must not be cited as convention-robust."
                  if n_conv else "; the headline finding is convention-robust.") + "\n")
    with open("results/tables/m1_variance_unit.md", "w") as fh:
        fh.write("\n".join(mdv))

    summary = {"n_cells": n, "orig_genuine": n_orig_gen, "s26_clustered_genuine": n_s26_gen,
               "ensemble_clustered_genuine_vol": n_ens_gen, "ensemble_clustered_genuine_var": n_var_gen,
               "survives": n_surv, "lost": n_lost, "gained": n_gain,
               "convention_dependent": n_conv, "sanity": sanity,
               "lost_cells": lost[["disc", "model", "h"]].to_dict("records"),
               "gained_cells": gained[["disc", "model", "h"]].to_dict("records"),
               "convention_dependent_cells": dep[["disc", "model", "h"]].to_dict("records")}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
