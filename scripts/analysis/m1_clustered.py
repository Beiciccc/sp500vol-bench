"""P0-1 — CLUSTERED-DM RESTATEMENT of the M1 Family-1 grid + pairwise-DM headline.

Reviewer-verified defect: every DM stat in forecast_combination.py / dm_pairwise.py runs
HAC(lag=h-1) over OBSERVATION order, but ~10-25 same-day filings share the same market
shocks, so t-stats are inflated ~2x. The FORECASTS do not change — only the inference.

Fix (canonical spec, implemented once in scripts/analysis/clustered_dm.py):
  per-obs loss differential -> daily mean by calendar day of effective_trading_day
  (fallback filing_time_utc date) -> dm_test on the two daily-mean loss series with
  h = label horizon in TRADING DAYS (HAC lag = h-1 now counts DAYS of genuine label
  overlap); n = number of days. Day-block moving bootstrap: blocks of h consecutive DAYS.

This script:
  1. re-runs the ENTIRE 69-cell M1 Family-1 analysis (same fc.log_combo val-fit /
     test-apply forecasts) and reproduces the obs-level columns of
     forecast_combination_grid.csv EXACTLY (hard sanity assertion);
  2. adds clustered DM (QLIKE + SE), clustered Clark-West, day-block bootstrap CI,
     clustered placebo gate (same PLACEBO_SEEDS), Holm within family as before;
  3. restates the pairwise-DM headline (0/180 challengers beat A2 on SE) with
     clustered inference.

Outputs (NEW files only; originals untouched for before/after):
  results/tables/m1_clustered.{csv,md}
  results/tables/dm_pairwise_clustered.{csv,md}
Run from repo root:  .venv/bin/python scripts/analysis/m1_clustered.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # noqa: E402
from clustered_dm import daily_mean, dm_test_clustered, mbb_ci_daily  # noqa: E402

sys.path.insert(0, "src")
from sp500vol.evaluation.dm_test import dm_test, _hac_variance  # noqa: E402

KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
HORIZONS = (5, 10, 20)
HAR = "A2_har_rv"
GRID_CSV = "results/tables/forecast_combination_grid.csv"
PAIRWISE_CSV = "results/tables/dm_pairwise.csv"


def day_key(df):
    """Calendar-day key: effective_trading_day, fallback filing_time_utc date."""
    d = df["effective_trading_day"]
    if d.isna().any():
        fb = pd.to_datetime(df["filing_time_utc"], utc=True).dt.tz_localize(None)
        d = d.fillna(fb)
    return d.to_numpy()


def clark_west_clustered(y, f_small, f_big, days, h):
    """Clark-West MSPE-adjusted stat on the DAILY-MEAN adjusted differential."""
    y = np.asarray(y, float)
    fhat = (y - f_small) ** 2 - (y - f_big) ** 2 + (
        np.asarray(f_small, float) - np.asarray(f_big, float)) ** 2
    fd, _ = daily_mean(fhat, days)
    n = len(fd)
    m = float(fd.mean())
    v = _hac_variance(fd, lag=max(h - 1, 0))
    if v <= 0:
        return (0.0, 1.0) if np.isclose(m, 0.0) else (float("nan"), float("nan"))
    t = m / np.sqrt(v / n)
    return float(t), float(stats.t.sf(t, df=n - 1))


# ---------------------------------------------------------------------------
# Part 1 — M1 Family-1 grid, obs-level (sanity) + clustered restatement
# ---------------------------------------------------------------------------
def run_m1_grid():
    rows = []
    disc_days = {}
    for disc, models in fc.SETS.items():
        har = fc.load(HAR, disc)[["split"] + KEY + [
            "prediction_realised_vol", "label_realised_vol", "filing_time_utc",
            "effective_trading_day"]].rename(columns={"prediction_realised_vol": "fhar"})
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
                days = day_key(dt)

                # FAMILY 1 — LOG space (identical machinery; forecasts unchanged)
                fR, fU, g_log = fc.log_combo(yv, fhv, ftv, fhr, ftt)
                lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
                sR, sU = fc.se(yt, fR), fc.se(yt, fU)

                # --- ORIGINAL obs-level inference (must reproduce the grid) ---
                cw_t, cw_p = fc.clark_west(yt, fR, fU, h)
                dmq, pq = dm_test(lU, lR, h=h)
                dms, ps_ = dm_test(sU, sR, h=h)
                _, lo, hi = fc.moving_block_ci(lU - lR, h)

                # --- CLUSTERED inference (daily means, HAC lag = h-1 in DAYS) ---
                dmq_c, pq_c, n_days = dm_test_clustered(lU, lR, days, h)
                dms_c, ps_c, _ = dm_test_clustered(sU, sR, days, h)
                cw_t_c, cw_p_c = clark_west_clustered(yt, fR, fU, days, h)
                dmean_c, lo_c, hi_c = mbb_ci_daily(lU - lR, days, h)

                # --- PLACEBO — identical permutation stream as the original ---
                pdq, pdm, pdm_c = [], [], []
                for s in fc.PLACEBO_SEEDS:
                    rng = np.random.default_rng(s)
                    pR, pU, _ = fc.log_combo(yv, fhv, rng.permutation(ftv), fhr, rng.permutation(ftt))
                    plU, plR = fc.qlike(yt, pU), fc.qlike(yt, pR)
                    pdq.append(float(plU.mean() - plR.mean()))
                    st, _p = dm_test(plU, plR, h=h)
                    pdm.append(st)
                    st_c, _pc, _nd = dm_test_clustered(plU, plR, days, h)
                    pdm_c.append(st_c)
                placebo_dq = float(np.mean(pdq))
                placebo_dm = float(np.mean(pdm))
                placebo_dm_c = float(np.mean(pdm_c))

                qR, qU = float(lR.mean()), float(lU.mean())
                rel = 100.0 * (qR - qU) / qR if qR > 0 else float("nan")
                disc_days.setdefault((disc, h), n_days)

                rows.append({
                    "disc": disc, "model": m, "h": h,
                    "n_obs": len(dt), "n_days": n_days,
                    "qlike_R": qR, "qlike_U": qU, "rel_impr_pct": rel, "g_log": g_log,
                    # original obs-level
                    "dm_q": float(dmq), "p_q": float(pq),
                    "dm_se": float(dms), "p_se": float(ps_),
                    "cw_t": cw_t, "cw_p": cw_p,
                    "boot_lo": lo, "boot_hi": hi,
                    "placebo_dq": placebo_dq, "placebo_dm": placebo_dm,
                    # clustered
                    "dm_q_clust": dmq_c, "p_q_clust": pq_c,
                    "dm_se_clust": dms_c, "p_se_clust": ps_c,
                    "cw_t_clust": cw_t_c, "cw_p_clust": cw_p_c,
                    "boot_lo_daily": lo_c, "boot_hi_daily": hi_c,
                    "placebo_dm_clust": placebo_dm_c,
                })

    df = pd.DataFrame(rows)
    # Holm WITHIN family (Family 1, pooled over the 69 cells) — as before.
    df["dmq_holm"] = fc.holm(df.p_q.fillna(1.0).values)
    df["cw_holm"] = fc.holm(df.cw_p.fillna(1.0).values)
    df["dmq_holm_clust"] = fc.holm(df.p_q_clust.fillna(1.0).values)
    df["cw_holm_clust"] = fc.holm(df.cw_p_clust.fillna(1.0).values)
    df["genuine"] = (df.dm_q < 0) & (df.dmq_holm < 0.05) & (df.placebo_dm.abs() < 2.0)
    df["genuine_clust"] = (df.dm_q_clust < 0) & (df.dmq_holm_clust < 0.05) & (df.placebo_dm_clust.abs() < 2.0)
    return df, disc_days


def sanity_vs_grid(df):
    """Obs-level columns must reproduce forecast_combination_grid.csv EXACTLY."""
    g = pd.read_csv(GRID_CSV)
    if len(g) != len(df):
        raise AssertionError(f"cell count mismatch: grid {len(g)} vs rerun {len(df)}")
    new = df.rename(columns={"n_obs": "n_test"})
    keep = ["disc", "model", "h", "n_test", "qlike_R", "qlike_U", "rel_impr_pct",
            "g_log", "dm_q", "p_q", "dm_se", "p_se", "cw_t", "cw_p", "boot_lo",
            "boot_hi", "placebo_dq", "placebo_dm", "dmq_holm", "cw_holm", "genuine"]
    merged = g[keep].merge(new[keep], on=["disc", "model", "h"],
                           suffixes=("_grid", "_new"))
    if len(merged) != len(g):
        raise AssertionError("key mismatch merging grid")
    checks = {}
    for a in keep[3:]:
        va = np.asarray(merged[f"{a}_grid"], float)
        vb = np.asarray(merged[f"{a}_new"], float)
        diff = float(np.nanmax(np.abs(va - vb))) if len(va) else 0.0
        checks[a] = diff
        if not np.allclose(va, vb, rtol=1e-9, atol=1e-12, equal_nan=True):
            raise AssertionError(f"SANITY FAIL: column {a} max|diff|={diff:.3e}")
    return checks


# ---------------------------------------------------------------------------
# Part 2 — pairwise DM vs A2 on SE, clustered restatement
# ---------------------------------------------------------------------------
SEED_INVARIANT = {
    "A2_har_rv", "A3_garch", "A4_egarch", "A5_arima",
    "B1_bow_ridge", "B2_tfidf_ridge", "B3_lm_linear", "B4_lm_features",
}
MULTI_SEED = [
    "C1_bert_s1", "C1_bert_s2", "C2_finbert_s1", "C2_finbert_s2", "C2_finbert_s3",
    "C2_finbert_s4", "C3_roberta_s1", "C4_longformer", "C5_qwen3", "C5_gteqwen2",
    "C5_e5mistral", "D1_concat_mlp", "D2_gated_fusion", "D3_qwen3", "D3_gteqwen2",
    "D3_e5mistral",
]
BASELINE_MODELS = ["A2_har_rv", "A3_garch", "A4_egarch", "A5_arima", "B2_tfidf_ridge"]
MODEL_SET = BASELINE_MODELS + MULTI_SEED
SEEDS_3 = (2026, 2027, 2028)
DISCLOSURES = ("long_form", "event_driven", "combined")
COLS = KEY + ["label_realised_vol", "prediction_realised_vol", "filing_time_utc",
              "effective_trading_day"]


def _run_path(model, disc, seed):
    return Path(f"results/runs/{model}_full_{disc}_seed{seed}/predictions.parquet")


def load_point(model, disc):
    """Identical to dm_pairwise.load_point but also carries effective_trading_day."""
    if model in SEED_INVARIANT:
        p = _run_path(model, disc, 2026)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df = df[df.split == "test"]
        return df[COLS]
    frames = []
    for s in SEEDS_3:
        p = _run_path(model, disc, s)
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d = d[d.split == "test"]
        frames.append(d[COLS])
    if not frames:
        return None
    cat = pd.concat(frames, ignore_index=True)
    return (cat.groupby(KEY, as_index=False)
               .agg(label_realised_vol=("label_realised_vol", "first"),
                    prediction_realised_vol=("prediction_realised_vol", "mean"),
                    filing_time_utc=("filing_time_utc", "first"),
                    effective_trading_day=("effective_trading_day", "first")))


def build_joined(disc):
    present, merged = [], None
    for m in MODEL_SET:
        d = load_point(m, disc)
        if d is None:
            continue
        present.append(m)
        sub = d[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": f"pred__{m}"})
        if merged is None:
            base = d[KEY + ["label_realised_vol", "filing_time_utc",
                            "effective_trading_day"]].copy()
            merged = base.merge(sub, on=KEY)
        else:
            merged = merged.merge(sub, on=KEY)
    return merged, present


def run_pairwise_clustered():
    orig = pd.read_csv(PAIRWISE_CSV)
    orig_vsA2 = orig[(orig.baseline == HAR) & (orig.challenger != HAR)].rename(
        columns={"dm_stat": "dm_obs", "p_raw": "p_obs", "p_holm": "p_holm_obs",
                 "better": "better_obs"})
    rows = []
    for disc in DISCLOSURES:
        merged, present = build_joined(disc)
        if merged is None or not present or HAR not in present:
            continue
        for h in HORIZONS:
            g = merged[merged.horizon_days == h].sort_values(SORT, kind="mergesort")
            if len(g) < 30:
                continue
            y = g.label_realised_vol.to_numpy()
            days = day_key(g)
            se_har = fc.se(y, g[f"pred__{HAR}"].to_numpy())
            group = []
            for ch in present:
                if ch == HAR:
                    continue
                se_ch = fc.se(y, g[f"pred__{ch}"].to_numpy())
                stat_c, p_c, n_days = dm_test_clustered(se_ch, se_har, days, h)
                group.append([disc, h, ch, HAR, len(g), n_days, stat_c, p_c])
            gdf = pd.DataFrame(group, columns=["disclosure", "horizon", "challenger",
                                               "baseline", "n_obs", "n_days",
                                               "dm_clust", "p_clust"])
            # Holm within (disclosure, horizon) over the vs-A2 challenger set.
            # (WEAKER correction than the original 420-pair family -> conservative
            # for the "0 challengers beat A2" claim.)
            gdf["p_holm_clust"] = fc.holm(gdf.p_clust.fillna(1.0).to_numpy())
            rows.append(gdf)
    full = pd.concat(rows, ignore_index=True)
    full = full.merge(orig_vsA2[["disclosure", "horizon", "challenger",
                                 "dm_obs", "p_obs", "p_holm_obs", "better_obs"]],
                      on=["disclosure", "horizon", "challenger"], how="left")
    full["better_clust"] = (full.dm_clust < 0) & (full.p_holm_clust < 0.05)
    full["better_clust_raw"] = (full.dm_clust < 0) & (full.p_clust < 0.05)
    full["sig_worse_clust"] = (full.dm_clust > 0) & (full.p_holm_clust < 0.05)
    full["sig_worse_obs"] = (full.dm_obs > 0) & (full.p_holm_obs < 0.05)
    return full


# ---------------------------------------------------------------------------
def main():
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    df, disc_days = run_m1_grid()
    checks = sanity_vs_grid(df)
    print("SANITY vs forecast_combination_grid.csv: PASS "
          f"(max col diff {max(checks.values()):.2e})")

    g = pd.read_csv(GRID_CSV)
    n_gen_orig = int(g.genuine.sum())
    n_gen_clust = int(df.genuine_clust.sum())

    flips = df[(df.genuine) & (~df.genuine_clust)][
        ["disc", "model", "h", "n_obs", "n_days", "rel_impr_pct",
         "dm_q", "p_q", "dmq_holm", "dm_q_clust", "p_q_clust", "dmq_holm_clust",
         "placebo_dm_clust"]]
    gains = df[(~df.genuine) & (df.genuine_clust)]
    surv = df[df.genuine_clust]

    # placebo gate (clustered)
    n_placebo_fail_orig = int((df.placebo_dm.abs() >= 2.0).sum())
    n_placebo_fail_clust = int((df.placebo_dm_clust.abs() >= 2.0).sum())

    # flagged cells
    def cell(disc, model, h):
        r = df[(df.disc == disc) & (df.model == model) & (df.h == h)]
        return r.iloc[0] if len(r) else None

    c2 = cell("long_form", "C2_finbert_s1", 5)

    df.to_csv("results/tables/m1_clustered.csv", index=False)

    # ---- markdown ----
    shrink = (df.dm_q_clust.abs() / df.dm_q.abs()).replace([np.inf, -np.inf], np.nan)
    md = ["# M1 Family-1 — CLUSTERED-DM restatement (day-clustered inference)\n",
          "## RESTATED vs ORIGINAL\n",
          "| quantity | ORIGINAL (obs-level HAC) | RESTATED (day-clustered) |",
          "|---|---|---|",
          f"| genuine cells (DM-QLIKE<0, Holm<.05, placebo null) | **{n_gen_orig}/69** | **{n_gen_clust}/69** |",
          f"| DM-QLIKE helps (Holm<.05) | {int(((g.dm_q<0)&(g.dmq_holm<0.05)).sum())} | {int(((df.dm_q_clust<0)&(df.dmq_holm_clust<0.05)).sum())} |",
          f"| DM-QLIKE worse (Holm<.05) | {int(((g.dm_q>0)&(g.dmq_holm<0.05)).sum())} | {int(((df.dm_q_clust>0)&(df.dmq_holm_clust<0.05)).sum())} |",
          f"| Clark-West adds (Holm<.05) | {int(((g.cw_t>0)&(g.cw_holm<0.05)).sum())} | {int(((df.cw_t_clust>0)&(df.cw_holm_clust<0.05)).sum())} |",
          f"| placebo-gate failures (\\|DM\\|>=2) | {n_placebo_fail_orig} | {n_placebo_fail_clust} |",
          f"| genuine effect-size range (rel QLIKE %) | "
          f"{g.loc[g.genuine,'rel_impr_pct'].min():.2f}-{g.loc[g.genuine,'rel_impr_pct'].max():.2f} | "
          + (f"{surv.rel_impr_pct.min():.2f}-{surv.rel_impr_pct.max():.2f} |" if n_gen_clust else "n/a |"),
          f"| median \\|DM\\| shrink factor (clust/obs) | 1.00 | {shrink.median():.2f} |",
          "",
          "The forecasts are IDENTICAL (fc.log_combo, weights fit on validation, frozen "
          "on test); only the inference changes. Obs-level columns of this rerun "
          "reproduce `forecast_combination_grid.csv` exactly (hard assertion, max "
          f"column diff {max(checks.values()):.1e}). Clustering: per-obs loss "
          "differentials averaged within calendar day of `effective_trading_day`; DM "
          "run on the daily series with HAC lag=h-1 in DAYS; day-block (h-day) moving "
          "bootstrap CI; placebo permutations identical seeds (1000-1004).\n",
          "**n_obs vs n_days (test):**",
          "| disclosure | h | n_obs | n_days |", "|---|---|---|---|"]
    for (disc, h), nd in sorted(disc_days.items()):
        no = int(df[(df.disc == disc) & (df.h == h)].n_obs.iloc[0])
        md.append(f"| {disc} | {h} | {no} | {nd} |")

    md += ["", "## Reviewer-flagged cells\n",
           "| cell | dm_q obs | p obs | Holm obs | dm_q clust | p clust | Holm clust | genuine obs->clust |",
           "|---|---|---|---|---|---|---|---|"]
    flagged = [("long_form", "C2_finbert_s1", 5)]
    flagged += [(d_, m_, h_) for d_ in ("long_form", "event_driven")
                for m_ in ("B2_tfidf_ridge", "C6_llmtext") for h_ in HORIZONS]
    for d_, m_, h_ in flagged:
        r = cell(d_, m_, h_)
        if r is None:
            continue
        go = bool(g[(g.disc == d_) & (g.model == m_) & (g.h == h_)].genuine.iloc[0])
        md.append(f"| {d_} {m_} h{h_} | {r.dm_q:+.2f} | {r.p_q:.2e} | {r.dmq_holm:.3f} | "
                  f"{r.dm_q_clust:+.2f} | {r.p_q_clust:.2e} | {r.dmq_holm_clust:.3f} | "
                  f"{'YES' if go else 'no'} -> {'YES' if r.genuine_clust else 'no'} |")

    md += ["", f"## Cells that FLIP (genuine -> not genuine under clustering): {len(flips)}\n",
           "| disc | model | h | rel% | dm_q obs | Holm obs | dm_q clust | p clust | Holm clust |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in flips.iterrows():
        md.append(f"| {r.disc} | {r.model} | {r.h} | {r.rel_impr_pct:+.2f} | {r.dm_q:+.2f} | "
                  f"{r.dmq_holm:.4f} | {r.dm_q_clust:+.2f} | {r.p_q_clust:.4f} | {r.dmq_holm_clust:.4f} |")
    if len(gains):
        md += ["", f"Cells newly genuine under clustering: {len(gains)} "
               f"({', '.join(gains.disc + ' ' + gains.model + ' h' + gains.h.astype(str))})"]
    else:
        md += ["", "No cell becomes genuine under clustering that was not before."]

    md += ["", "## Surviving genuine cells (clustered)\n",
           "| disc | model | h | rel% | dm_q clust | Holm clust | daily-boot 95% CI (QLIKE diff) | placebo DM clust |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in surv.sort_values(["disc", "model", "h"]).iterrows():
        md.append(f"| {r.disc} | {r.model} | {r.h} | {r.rel_impr_pct:+.2f} | {r.dm_q_clust:+.2f} | "
                  f"{r.dmq_holm_clust:.4f} | [{r.boot_lo_daily:+.5f}, {r.boot_hi_daily:+.5f}] | "
                  f"{r.placebo_dm_clust:+.2f} |")

    md += ["", "## Full 69-cell grid (original vs clustered)\n",
           "| disc | model | h | n_obs | n_days | rel% | dm_q | p_q | Holm | dm_q_cl | p_cl | Holm_cl | dm_se | dm_se_cl | genuine | genuine_cl |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.disc} | {r.model} | {r.h} | {r.n_obs} | {r.n_days} | {r.rel_impr_pct:+.2f} | "
                  f"{r.dm_q:+.2f} | {r.p_q:.1e} | {r.dmq_holm:.3f} | {r.dm_q_clust:+.2f} | "
                  f"{r.p_q_clust:.1e} | {r.dmq_holm_clust:.3f} | {r.dm_se:+.2f} | {r.dm_se_clust:+.2f} | "
                  f"{'Y' if r.genuine else 'n'} | {'Y' if r.genuine_clust else 'n'} |")

    with open("results/tables/m1_clustered.md", "w") as fh:
        fh.write("\n".join(md))

    # ---------------- pairwise ----------------
    pw = run_pairwise_clustered()
    pw.to_csv("results/tables/dm_pairwise_clustered.csv", index=False)
    n_cells = len(pw)
    n_better_obs = int(pw.better_obs.fillna(False).sum())
    n_better_cl = int(pw.better_clust.sum())
    n_better_cl_raw = int(pw.better_clust_raw.sum())
    n_worse_obs = int(pw.sig_worse_obs.fillna(False).sum())
    n_worse_cl = int(pw.sig_worse_clust.sum())
    shrink_pw = (pw.dm_clust.abs() / pw.dm_obs.abs()).median()

    md2 = ["# Pairwise DM vs A2_har_rv on squared error — CLUSTERED restatement\n",
           "## RESTATED vs ORIGINAL\n",
           "| quantity | ORIGINAL (obs-level HAC) | RESTATED (day-clustered) |",
           "|---|---|---|",
           f"| challengers significantly BETTER than A2 (Holm<.05) | {n_better_obs}/{n_cells} | **{n_better_cl}/{n_cells}** |",
           f"| ... even at RAW p<.05 (no Holm) | - | {n_better_cl_raw}/{n_cells} |",
           f"| challengers significantly WORSE than A2 (Holm<.05) | {n_worse_obs}/{n_cells} | {n_worse_cl}/{n_cells} |",
           f"| median \\|DM\\| shrink factor | 1.00 | {shrink_pw:.2f} |",
           "",
           "Same seed-ensembled test-split forecasts and inner-joined sample as "
           "`dm_pairwise.csv`; only inference changes (daily-mean SE differential, HAC "
           "lag=h-1 in days). Holm here is applied WITHIN each (disclosure,horizon) "
           "group over the vs-A2 challenger set only — a WEAKER correction than the "
           "original 420-pair family, i.e. conservative for the '0 challengers beat "
           "A2' headline. Original obs-level columns are copied from dm_pairwise.csv "
           "for the before/after.\n",
           "| disclosure | h | challenger | n_obs | n_days | dm obs | p_holm obs | dm clust | p clust | p_holm clust | verdict clust |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in pw.sort_values(["disclosure", "horizon", "dm_clust"]).iterrows():
        v = ("BETTER*" if r.better_clust else
             ("sig worse" if r.sig_worse_clust else
              ("worse(ns)" if r.dm_clust > 0 else "better(ns)")))
        md2.append(f"| {r.disclosure} | {r.horizon} | {r.challenger} | {r.n_obs} | {r.n_days} | "
                   f"{r.dm_obs:+.2f} | {r.p_holm_obs:.4f} | {r.dm_clust:+.2f} | "
                   f"{r.p_clust:.4f} | {r.p_holm_clust:.4f} | {v} |")

    with open("results/tables/dm_pairwise_clustered.md", "w") as fh:
        fh.write("\n".join(md2))

    print("=== P0-1 clustered restatement done ===")
    print(f"M1 grid: genuine {n_gen_orig}/69 (obs) -> {n_gen_clust}/69 (clustered); "
          f"flips={len(flips)}; newly-genuine={len(gains)}")
    if n_gen_clust:
        print(f"surviving effect sizes: rel QLIKE {surv.rel_impr_pct.min():.2f}-"
              f"{surv.rel_impr_pct.max():.2f}%")
    print(f"placebo-gate failures: obs {n_placebo_fail_orig} -> clust {n_placebo_fail_clust}")
    if c2 is not None:
        print(f"C2_finbert_s1 lf h5: p {c2.p_q:.2e} -> {c2.p_q_clust:.2e} "
              f"(dm {c2.dm_q:+.2f} -> {c2.dm_q_clust:+.2f})")
    print(f"pairwise vs A2: better {n_better_obs}/{n_cells} (obs) -> {n_better_cl}/{n_cells} "
          f"(clustered, Holm) / {n_better_cl_raw} at raw p; sig-worse {n_worse_obs} -> {n_worse_cl}")
    print(f"median |DM| shrink: grid {shrink.median():.2f}, pairwise {shrink_pw:.2f}")


if __name__ == "__main__":
    main()
