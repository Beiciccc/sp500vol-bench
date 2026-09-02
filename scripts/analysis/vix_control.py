"""P1-b — VIX CONTROL for the M1 incremental-text finding (remediation).

Reviewer concern: the text increment over recalibrated HAR could be a market-level
"when" effect (calendar clustering of filings around volatile periods) rather than
firm-specific disclosure information. Control: add the market's own implied-vol
signal (VIX, point-in-time) to the RECALIBRATED reference.

(a) VIX-augmented reference, LOG space, weights fit on VALIDATION only:
        f_R_vix = exp(a + b*log fHAR + c*log VIX_pit)
        f_U_vix = exp(a + b*log fHAR + c*log VIX_pit + g*log fText)
    VIX_pit = last available VIX close STRICTLY BEFORE effective_trading_day
    (the label-window start) — point-in-time discipline, no same-day close.
    Cells: all "genuine" cells from forecast_combination_summary.json + all C6 cells.
    Inference: day-clustered DM (clustered_dm.dm_test_clustered) + day-block MBB CI
    + permuted-text placebo (5 seeds) under the VIX-augmented reference.

(b) HAR-X(VIX) standalone baseline: pooled (across firms) per-horizon log-OLS
        log RV ~ [1, log rv1, log rv5, log rv22, log VIX_pit]
    with Duan smearing, replicating src/sp500vol/models/price/har_rv.py conventions
    (epsilon=1e-12, log(x+eps), smear = mean(exp(train residuals))). Fit on TRAIN,
    predicted for all splits; run dirs results/runs/A7_harx_vix_full_<disc>_seed2026
    written per the standard B2 schema (metrics.json 9 rows, variance-unit QLIKE
    via sp500vol.evaluation.metrics.all_metrics).

DM sign convention everywhere: positive stat = FIRST loss series WORSE.
Run from the repo root:  .venv/bin/python scripts/analysis/vix_control.py
"""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")
from clustered_dm import dm_test_clustered, mbb_ci_daily  # noqa: E402
import forecast_combination as fc  # noqa: E402
from sp500vol.evaluation.metrics import all_metrics  # noqa: E402

ROOT = Path(".")
TABLES = ROOT / "results/tables"
RUNS = ROOT / "results/runs"
RAW_VIX = TABLES / "_vix_daily_raw.csv"
VIX_CSV = TABLES / "_vix_daily.csv"
HAR_EPS = 1e-12  # har_rv.py epsilon convention
DISCS_AB = ("long_form", "event_driven")
DISCS_ALL = ("long_form", "event_driven", "combined")


# ---------------------------------------------------------------- VIX data
def prepare_vix():
    if not RAW_VIX.exists():
        raise FileNotFoundError(
            f"{RAW_VIX} missing — download first: curl -sL "
            "'https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS' -o " + str(RAW_VIX))
    raw = pd.read_csv(RAW_VIX)
    raw.columns = ["date", "vix_close"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["vix_close"] = pd.to_numeric(raw["vix_close"], errors="coerce")  # FRED '.' -> NaN
    raw = raw.dropna(subset=["vix_close"]).sort_values("date").reset_index(drop=True)
    # keep Dec-2009 onward so the earliest effective_trading_day (2010-02-04) has a
    # strictly-earlier close available; save 2009-12-01..2025-12-31
    vix = raw[(raw.date >= "2009-12-01") & (raw.date <= "2025-12-31")].reset_index(drop=True)
    vix.to_csv(VIX_CSV, index=False)
    # validation
    n = len(vix)
    peak_2020_03 = vix.loc[(vix.date >= "2020-03-01") & (vix.date < "2020-04-01"), "vix_close"].max()
    low_2017 = vix.loc[(vix.date >= "2017-01-01") & (vix.date < "2018-01-01"), "vix_close"].min()
    assert 3800 <= n <= 4300, f"unexpected VIX row count {n}"
    assert peak_2020_03 > 80, f"2020-03 peak {peak_2020_03} not > 80"
    assert low_2017 < 10, f"2017 low {low_2017} not ~9"
    print(f"[vix] {n} rows {vix.date.min().date()}..{vix.date.max().date()} | "
          f"2020-03 peak={peak_2020_03:.2f} | 2017 low={low_2017:.2f}")
    return vix, {"n_rows": n, "peak_2020_03": float(peak_2020_03), "low_2017": float(low_2017),
                 "first": str(vix.date.min().date()), "last": str(vix.date.max().date())}


def vix_pit_lookup(vix, days):
    """Last VIX close STRICTLY BEFORE each day (PIT). Returns (vix values, vix dates)."""
    q = pd.DataFrame({"day": pd.to_datetime(pd.Series(np.asarray(days))).dt.normalize()})
    q["_row"] = np.arange(len(q))
    merged = pd.merge_asof(q.sort_values("day", kind="mergesort"),
                           vix.rename(columns={"date": "vix_date"}),
                           left_on="day", right_on="vix_date",
                           direction="backward", allow_exact_matches=False)
    merged = merged.sort_values("_row", kind="mergesort")
    if merged.vix_close.isna().any():
        bad = merged.loc[merged.vix_close.isna(), "day"].unique()[:5]
        raise ValueError(f"no PIT VIX for days {bad}")
    assert (merged.vix_date < merged.day).all(), "PIT violation: VIX date not strictly before day"
    return merged.vix_close.to_numpy(), merged.vix_date.to_numpy()


# ---------------------------------------------------------------- part (a)
def log_combo_vix(yv, fhv, vv, ftv, fhr, vt, ftt):
    """VIX-augmented log-space nested combination, val-fit / test-apply.

    Returns (f_R_vix, f_U_vix, g_text, c_vix_ref)."""
    E = fc.EPS
    ly = np.log(np.clip(yv, E, None))
    lh_v, lv_v, lt_v = (np.log(np.clip(a, E, None)) for a in (fhv, vv, ftv))
    lh_t, lv_t, lt_t = (np.log(np.clip(a, E, None)) for a in (fhr, vt, ftt))
    XR = np.column_stack([np.ones(len(ly)), lh_v, lv_v])
    XU = np.column_stack([np.ones(len(ly)), lh_v, lv_v, lt_v])
    bR = fc.ols(ly, XR)
    bU = fc.ols(ly, XU)
    fR = np.exp(bR[0] + bR[1] * lh_t + bR[2] * lv_t)
    fU = np.exp(bU[0] + bU[1] * lh_t + bU[2] * lv_t + bU[3] * lt_t)
    return fR, fU, float(bU[3]), float(bR[2])


def part_a(vix):
    summary = json.load(open(TABLES / "forecast_combination_summary.json"))
    grid = pd.read_csv(TABLES / "forecast_combination_grid.csv").set_index(["disc", "model", "h"])
    cells = [(c["disc"], c["model"], int(c["h"])) for c in summary["genuine_cells"]]
    for disc in DISCS_AB:                     # + ALL C6 cells (contamination-flagged model)
        for h in fc.HORIZONS:
            if ("%s" % disc, "C6_llmtext", h) not in cells:
                cells.append((disc, "C6_llmtext", h))
    cells = sorted(set(cells), key=lambda c: (c[0], c[1], c[2]))
    print(f"[a] {len(cells)} cells (38 genuine + C6 h20 x2)")

    har_cache = {d: fc.load("A2_har_rv", d)[["split"] + fc.KEY +
                 ["prediction_realised_vol", "label_realised_vol", "filing_time_utc",
                  "effective_trading_day"]].rename(columns={"prediction_realised_vol": "fhar"})
                 for d in DISCS_AB}
    txt_cache = {}
    rows = []
    for disc, model, h in cells:
        key = (disc, model)
        if key not in txt_cache:
            txt_cache[key] = fc.load(model, disc)[fc.KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
        d = har_cache[disc].merge(txt_cache[key], on=fc.KEY)
        dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(fc.SORT, kind="mergesort")
        dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(fc.SORT, kind="mergesort")
        if len(dv) < 100 or len(dt) < 30:
            continue
        yv, fhv, ftv = dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(), dv.ftext.to_numpy()
        yt, fhr, ftt = dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(), dt.ftext.to_numpy()
        vv, _ = vix_pit_lookup(vix, dv.effective_trading_day)
        vt, _ = vix_pit_lookup(vix, dt.effective_trading_day)
        days = dt.effective_trading_day.to_numpy()

        # 1) restated no-VIX cell with day-clustered DM (attribution: clustering alone)
        fR, fU, g_log = fc.log_combo(yv, fhv, ftv, fhr, ftt)
        lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
        dm_clu, p_clu, n_days = dm_test_clustered(lU, lR, days, h)

        # 2) VIX-augmented reference (the control) with day-clustered DM
        fRv, fUv, g_vix, c_vix = log_combo_vix(yv, fhv, vv, ftv, fhr, vt, ftt)
        lRv, lUv = fc.qlike(yt, fRv), fc.qlike(yt, fUv)
        dm_vix, p_vix, _ = dm_test_clustered(lUv, lRv, days, h)
        dmean, lo, hi = mbb_ci_daily(lUv - lRv, days, h)
        qRv, qUv = float(lRv.mean()), float(lUv.mean())
        rel_vix = 100.0 * (qRv - qUv) / qRv if qRv > 0 else float("nan")

        # 3) placebo under the VIX-augmented reference (permuted text, clustered DM)
        pstats = []
        for s in fc.PLACEBO_SEEDS:
            rng = np.random.default_rng(s)
            _, pU, _, _ = log_combo_vix(yv, fhv, vv, rng.permutation(ftv), fhr, vt,
                                        rng.permutation(ftt))
            st, _, _ = dm_test_clustered(fc.qlike(yt, pU), lRv, days, h)
            pstats.append(st)
        placebo_dm = float(np.mean(pstats))

        g = grid.loc[(disc, model, h)]
        rows.append({
            "disc": disc, "model": model, "h": h, "n_test": len(dt), "n_days": n_days,
            "orig_rel_impr_pct": float(g.rel_impr_pct), "orig_dm_q": float(g.dm_q),
            "orig_dmq_holm": float(g.dmq_holm), "orig_genuine": bool(g.genuine),
            "clu_dm_q": dm_clu, "clu_p_q": p_clu,
            "qlike_R_vix": qRv, "qlike_U_vix": qUv, "rel_impr_pct_vix": rel_vix,
            "g_text_vix": g_vix, "c_vix_ref": c_vix,
            "vix_dm_q": dm_vix, "vix_p_q": p_vix,
            "vix_boot_dmean": dmean, "vix_boot_lo": lo, "vix_boot_hi": hi,
            "vix_placebo_dm": placebo_dm,
        })
        print(f"  {disc:12s} {model:16s} h={h:2d} nd={n_days:4d} "
              f"orig={g.dm_q:+6.2f} clu={dm_clu:+6.2f} vix={dm_vix:+6.2f} (p={p_vix:.4f})")

    df = pd.DataFrame(rows)
    df["clu_holm"] = fc.holm(df.clu_p_q.fillna(1.0).values)
    df["vix_holm"] = fc.holm(df.vix_p_q.fillna(1.0).values)
    df["survives_clustering"] = (df.clu_dm_q < 0) & (df.clu_holm < 0.05)
    df["survives_vix"] = ((df.vix_dm_q < 0) & (df.vix_holm < 0.05)
                          & (df.vix_placebo_dm.abs() < 2.0))
    return df


# ---------------------------------------------------------------- part (b)
def fit_harx_vix(disc, vix):
    """HAR-X(VIX): pooled per-horizon log-OLS + Duan smearing (har_rv.py conventions)."""
    a2 = pd.read_parquet(RUNS / f"A2_har_rv_full_{disc}_seed2026/predictions.parquet")
    vpit, _ = vix_pit_lookup(vix, a2.effective_trading_day)
    feats = np.column_stack([
        np.log(a2.feature_rv_1d.to_numpy() + HAR_EPS),
        np.log(a2.feature_rv_5d.to_numpy() + HAR_EPS),
        np.log(a2.feature_rv_22d.to_numpy() + HAR_EPS),
        np.log(vpit),
    ])
    y = a2.label_realised_vol.to_numpy(dtype=float)
    ly = np.log(np.where(y >= 0.0, y, np.nan) + HAR_EPS)
    pred = np.full(len(a2), np.nan)
    coefs = {}
    max_log = np.log(np.finfo(float).max) - 1.0
    for h in fc.HORIZONS:
        hm = (a2.horizon_days == h).to_numpy()
        tm = hm & (a2.split == "train").to_numpy()
        valid = tm & np.isfinite(ly) & np.isfinite(feats).all(axis=1)
        X = np.column_stack([np.ones(valid.sum()), feats[valid]])
        beta, *_ = np.linalg.lstsq(X, ly[valid], rcond=None)
        smear = float(np.mean(np.exp(ly[valid] - X @ beta)))       # Duan smearing
        raw = np.clip(beta[0] + feats[hm] @ beta[1:], np.log(HAR_EPS), max_log)
        pred[hm] = np.maximum(np.exp(raw) * smear - HAR_EPS, HAR_EPS)
        coefs[h] = {"intercept": float(beta[0]),
                    "log_rv1": float(beta[1]), "log_rv5": float(beta[2]),
                    "log_rv22": float(beta[3]), "log_vix": float(beta[4]),
                    "smear": smear, "n_train": int(valid.sum())}
    assert np.isfinite(pred).all()
    run_id = f"A7_harx_vix_full_{disc}_seed2026"
    out = a2.copy()
    out["run_id"] = run_id
    out["model_id"] = "A7_harx_vix"
    out["prediction_realised_vol"] = pred
    rd = RUNS / run_id
    rd.mkdir(parents=True, exist_ok=True)
    out.to_parquet(rd / "predictions.parquet", index=False)
    metrics = []
    for split in ("test", "train", "val"):
        for h in fc.HORIZONS:
            m = out[(out.split == split) & (out.horizon_days == h)]
            metrics.append({"split": split, "disclosure_subset": disc, "horizon_days": int(h),
                            "n": int(len(m)),
                            **all_metrics(m.label_realised_vol.to_numpy(),
                                          m.prediction_realised_vol.to_numpy())})
    json.dump(metrics, open(rd / "metrics.json", "w"), indent=2)
    json.dump({"model": "A7_harx_vix", "dataset": "full", "disclosure": disc, "seed": 2026,
               "model_config": {
                   "model_id": "A7_harx_vix",
                   "description": "HAR-X(VIX): pooled per-horizon log-OLS "
                                  "[1, log rv1, log rv5, log rv22, log VIX_pit] with Duan "
                                  "smearing; har_rv.py conventions (eps=1e-12). VIX_pit = last "
                                  "close STRICTLY BEFORE effective_trading_day (PIT).",
                   "vix_source": "FRED VIXCLS (results/tables/_vix_daily.csv)",
                   "training": {"fit_split": "train", "fit_per_horizon": True,
                                "log_target": True, "smearing": True},
                   "coefficients": coefs},
               "created_by": "scripts/analysis/vix_control.py (P1-b remediation)"},
              open(rd / "config.json", "w"), indent=2)
    json.dump({"segments": [{"label": "training", "seconds": 1.0}], "total_seconds": 1.0,
               "total_gpu_hours": 0.0, "hourly_rate_usd": 0.0, "total_cost_usd": 0.0},
              open(rd / "cost.json", "w"), indent=2)
    json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "git_sha": None, "git_dirty": None,
               "python_version": platform.python_version(), "platform": platform.platform(),
               "gpu": [], "pip_freeze_hash": None, "cloud_provider": "local-cpu",
               "instance_type": "laptop"}, open(rd / "env.json", "w"), indent=2)
    return out, a2, coefs


def part_b(vix):
    rows = []
    coef_all = {}
    for disc in DISCS_ALL:
        a7, a2, coefs = fit_harx_vix(disc, vix)
        coef_all[disc] = coefs
        for h in fc.HORIZONS:
            m7 = a7[(a7.split == "test") & (a7.horizon_days == h)].sort_values(fc.SORT, kind="mergesort")
            m2 = a2[(a2.split == "test") & (a2.horizon_days == h)].sort_values(fc.SORT, kind="mergesort")
            y = m2.label_realised_vol.to_numpy()
            l7 = fc.qlike(y, m7.prediction_realised_vol.to_numpy())
            l2 = fc.qlike(y, m2.prediction_realised_vol.to_numpy())
            dm, p, n_days = dm_test_clustered(l7, l2, m2.effective_trading_day.to_numpy(), h)
            rows.append({"disc": disc, "h": h, "n_test": len(m2), "n_days": n_days,
                         "qlike_A2": float(l2.mean()), "qlike_A7_harx_vix": float(l7.mean()),
                         "rel_impr_pct": 100.0 * (l2.mean() - l7.mean()) / l2.mean(),
                         "dm_A7_vs_A2": dm, "p": p,
                         "log_vix_coef": coefs[h]["log_vix"], "smear": coefs[h]["smear"]})
            print(f"  [b] {disc:12s} h={h:2d} A2={l2.mean():.4f} A7={l7.mean():.4f} "
                  f"dm={dm:+.2f} p={p:.4f} vix_coef={coefs[h]['log_vix']:+.3f}")
    df = pd.DataFrame(rows)
    df["holm"] = fc.holm(df.p.fillna(1.0).values)
    return df, coef_all


# ---------------------------------------------------------------- sanity
def pit_spot_check(vix):
    d = pd.read_parquet(RUNS / "A2_har_rv_full_long_form_seed2026/predictions.parquet")
    d = d[(d.split == "test") & (d.horizon_days == 5)]
    picks = d.sort_values("effective_trading_day").iloc[[0, len(d) // 2, -1]]
    out = []
    for _, r in picks.iterrows():
        v, vd = vix_pit_lookup(vix, [r.effective_trading_day])
        vd = pd.Timestamp(vd[0])
        ok = vd < pd.Timestamp(r.effective_trading_day)
        out.append({"ticker": r.ticker, "accession": r.accession,
                    "filing_time_utc": str(r.filing_time_utc),
                    "effective_trading_day": str(pd.Timestamp(r.effective_trading_day).date()),
                    "vix_date_used": str(vd.date()), "vix_close": float(v[0]),
                    "pit_ok(vix<label_window_start)": bool(ok)})
        assert ok
    return out


# ---------------------------------------------------------------- report
def write_outputs(dfa, dfb, vix_meta, spot, summary):
    dfa.to_csv(TABLES / "vix_control.csv", index=False)
    dfb.to_csv(TABLES / "vix_control_harx.csv", index=False)

    n = len(dfa)
    n_orig = int(dfa.orig_genuine.sum())
    n_clu = int(dfa.survives_clustering.sum())
    n_vix = int(dfa.survives_vix.sum())
    n_both = int((dfa.survives_clustering & dfa.survives_vix).sum())
    gsub = dfa[dfa.orig_genuine]
    n_g_clu = int(gsub.survives_clustering.sum())
    n_g_vix = int(gsub.survives_vix.sum())

    md = ["# P1-b — VIX control on the M1 incremental-text finding\n",
          "## RESTATED vs ORIGINAL\n",
          "| | ORIGINAL (obs-level DM, no VIX) | RESTATED (day-clustered DM) | RESTATED (day-clustered DM + VIX-augmented reference) |",
          "|---|---|---|---|",
          f"| Significant text increment (Holm<.05, placebo-clean) | {summary['n_genuine_increment']}/{summary['n_cells']} cells "
          f"(of which {n_orig} re-tested here) | {n_g_clu}/{n_orig} of the original genuine cells survive | "
          f"{n_g_vix}/{n_orig} of the original genuine cells survive |",
          f"| All {n} re-tested cells (38 genuine + C6 h20 x2) | — | {n_clu}/{n} | {n_vix}/{n} (both filters: {n_both}) |",
          "",
          "**Reading:** ORIGINAL inference ran HAC(lag=h-1) over observation order — with ~10-25 same-day "
          "filings sharing market shocks, those t-stats are inflated. RESTATED collapses losses to daily "
          "means (equal weight per day), runs DM with HAC lag = h-1 DAYS, and additionally puts log VIX "
          "(point-in-time: last close STRICTLY BEFORE the label-window start, effective_trading_day) into "
          "the recalibrated reference, so the text forecast must add information BEYOND market-level "
          "implied volatility. The within-date placebo logic predicted the increment survives a pure "
          "'when' control; the table below shows cell-by-cell whether it does.\n",
          f"**VIX data:** FRED VIXCLS, {vix_meta['n_rows']} rows {vix_meta['first']}..{vix_meta['last']} "
          f"(results/tables/_vix_daily.csv). Validation: 2020-03 peak={vix_meta['peak_2020_03']:.2f} (>80 OK), "
          f"2017 low={vix_meta['low_2017']:.2f} (~9 OK).\n",
          "## (a) Text increment under the VIX-augmented recalibrated reference (log space, val-fit)\n",
          "DM sign: negative = text-augmented f_U_vix BETTER than reference f_R_vix. n_days = clustered "
          "sample size. `c_vix_ref` = log-VIX loading in the reference; `g_text_vix` = text elasticity "
          "given HAR+VIX. Holm within this table. `survives_vix` requires DM<0, Holm<.05 and |placebo|<2.\n",
          "| disc | model | h | n_test | n_days | orig DM | orig Holm | clu DM | clu Holm | VIX DM | VIX p | VIX Holm | rel%(vix) | g_text | c_vix | MBB 95% CI (daily dQLIKE) | placebo DM | orig genuine | surv. clu | surv. VIX |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in dfa.iterrows():
        md.append(f"| {r.disc} | {r.model} | {r.h} | {r.n_test} | {r.n_days} | {r.orig_dm_q:+.2f} | "
                  f"{r.orig_dmq_holm:.3g} | {r.clu_dm_q:+.2f} | {r.clu_holm:.3g} | {r.vix_dm_q:+.2f} | "
                  f"{r.vix_p_q:.4f} | {r.vix_holm:.3g} | {r.rel_impr_pct_vix:+.2f} | {r.g_text_vix:+.3f} | "
                  f"{r.c_vix_ref:+.3f} | [{r.vix_boot_lo:+.5f}, {r.vix_boot_hi:+.5f}] | "
                  f"{r.vix_placebo_dm:+.2f} | {'Y' if r.orig_genuine else '-'} | "
                  f"{'Y' if r.survives_clustering else 'no'} | {'**Y**' if r.survives_vix else 'no'} |")

    md += ["\n## (b) HAR-X(VIX) standalone baseline (A7) vs A2 HAR-RV — test QLIKE (vol-unit), clustered DM\n",
           "Pooled per-horizon log-OLS [1, log rv1, log rv5, log rv22, log VIX_pit] + Duan smearing, "
           "fit on train, har_rv.py conventions. Run dirs: results/runs/A7_harx_vix_full_<disc>_seed2026 "
           "(metrics.json uses the pipeline's variance-unit QLIKE; this table uses M1's vol-unit QLIKE). "
           "DM sign: negative = A7 better than A2.\n",
           "| disc | h | n_test | n_days | QLIKE A2 | QLIKE A7 | rel% | DM(A7 vs A2) | p | Holm | log-VIX coef | smear |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in dfb.iterrows():
        md.append(f"| {r.disc} | {r.h} | {r.n_test} | {r.n_days} | {r.qlike_A2:.4f} | "
                  f"{r.qlike_A7_harx_vix:.4f} | {r.rel_impr_pct:+.2f} | {r.dm_A7_vs_A2:+.2f} | "
                  f"{r.p:.4f} | {r.holm:.3g} | {r.log_vix_coef:+.3f} | {r.smear:.3f} |")

    md += ["\n## PIT sanity spot-check (3 test filings, long_form h=5)\n",
           "Discipline: VIX value used is the last close with date STRICTLY BEFORE "
           "effective_trading_day (= label-window start); no same-day information.\n",
           "| ticker | accession | filing_time_utc | effective_trading_day | VIX date used | VIX close | PIT ok |",
           "|---|---|---|---|---|---|---|"]
    for s in spot:
        md.append(f"| {s['ticker']} | {s['accession']} | {s['filing_time_utc']} | "
                  f"{s['effective_trading_day']} | {s['vix_date_used']} | {s['vix_close']:.2f} | "
                  f"{s['pit_ok(vix<label_window_start)']} |")

    md += ["\n## Bottom line\n",
           f"- Day-clustered DM alone: {n_g_clu}/{n_orig} originally-genuine cells stay significant "
           f"(Holm<.05 within this table).",
           f"- Adding the VIX control on top: {n_g_vix}/{n_orig} originally-genuine cells survive a "
           f"market-level implied-volatility control (placebo-clean).",
           "- HAR-X(VIX) shows how much of the field a pure market-vol regressor claims on its own "
           "(table b); the text increment in (a) is measured NET of that signal.",
           "- Weights fit on validation only, applied frozen to test; VIX strictly point-in-time."]

    (TABLES / "vix_control.md").write_text("\n".join(md))
    print("wrote results/tables/vix_control.{csv,md} + vix_control_harx.csv")
    return {"n_cells": n, "n_orig_genuine_retested": n_orig, "n_survive_clustered": n_clu,
            "n_survive_vix": n_vix, "n_genuine_survive_clustered": n_g_clu,
            "n_genuine_survive_vix": n_g_vix}


def main():
    vix, vix_meta = prepare_vix()
    spot = pit_spot_check(vix)
    for s in spot:
        print("[pit]", s)
    summary = json.load(open(TABLES / "forecast_combination_summary.json"))
    dfa = part_a(vix)
    dfb, _ = part_b(vix)
    res = write_outputs(dfa, dfb, vix_meta, spot, summary)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
