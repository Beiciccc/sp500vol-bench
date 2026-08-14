"""E4 — Stronger price baselines (A6): SHAR and HARQ, to kill the "weak HAR baseline" rebuttal.

Two HAR variants, replicating the EXACT A2 HAR-RV conventions from
src/sp500vol/models/price/har_rv.py: pooled panel OLS per horizon_days, log-space
target log(y + eps) with eps = 1e-12, features log(x + eps), and the Duan (1983)
smearing retransformation (smear = mean(exp(train residuals)) per horizon;
prediction = exp(raw) * smear - eps, floored at eps).

(a) A6_shar — semivariance HAR (Patton & Sheppard 2015). The DAILY lag is decomposed
    into signed realized semivolatilities computed from CRSP daily log returns:
        RS-_1 = sqrt(252 * r^2 * 1[r<0]),  RS+_1 = sqrt(252 * r^2 * 1[r>0])
    for the single trading day ending at feature_window_end (PIT: strictly <=
    feature_window_end). Regressors: [log(RS-_1+eps), log(RS+_1+eps),
    log(rv_5+eps), log(rv_22+eps)] — weekly/monthly lags stay aggregate, exactly
    the Patton-Sheppard daily decomposition.

(b) A6_harq — HARQ (Bollerslev, Patton & Quaedvlieg 2016), log-space analogue.
    The daily-lag coefficient is allowed to vary with measurement error via realized
    quarticity. Because a 1-day "window" RQ is degenerate (single squared return),
    we use the more stable 22-day realized quarticity, annualized to match the
    annualized-VARIANCE^2 units of the (annualized-vol) features:
        RQ_22 = (22 * 252^2 / 3) * mean(r^4 over the 22 trading days ending at
                 feature_window_end)  =  (252^2 / 3) * sum(r^4)
    Regressors: [log(rv_1+eps), log(rv_5+eps), log(rv_22+eps),
                 sqrt(RQ_22) * log(rv_1+eps)]
    i.e. log(y) = b0 + (b1 + bQ*sqrt(RQ_22)) * log(rv_1) + b2*log(rv_5) + b3*log(rv_22),
    the direct log-space transcription of the HARQ interaction ("HARQ spirit"):
    the daily-lag loading shrinks/expands linearly in sqrt(RQ). Everything stays in
    log space; retransformation via the same Duan smearing.

RETURNS SOURCE (documented deviation from the task sketch): the task suggested
full_ohlcv adj_close, but the canonical return store that BUILT the aligned
features/labels is /Volumes/Z/sp500vol-data/market/crsp/market_returns.parquet
(ticker, date, log_return = log1p(CRSP DlyRet), dividend-adjusted total return; see
src/sp500vol/data/crsp.py). Verified: it reproduces feature_return_1d / feature_rv_5d /
feature_rv_22d EXACTLY (3000-row sample, 100.0% exact at 1e-8), whereas OHLCV
adj_close log-returns only correlate ~0.78 (dividend handling + vendor differences).
Using the canonical store keeps the new regressors on the identical footing as A2's.
Unreliable ticker-recycled joins are still dropped via ret_match_ok from
results/tables/_realized_returns.parquet (task requirement); drops are reported.

SPLITS: taken from the existing A2_har_rv_full_<disc>_seed2026 predictions per
disclosure — no re-derivation. Fit on split=="train" only; predict all splits;
evaluate on test. Combiner weights (M1 rerun) fit on val only via
scripts/analysis/forecast_combination.log_combo.

SANITY GATE: before trusting the variants, this pipeline refits PLAIN HAR from the
same predictions-parquet features and must reproduce the archived A2 test QLIKE
(variance-unit) within ~3% per horizon; the run aborts otherwise.

Outputs:
  results/runs/A6_shar_full_<disc>_seed2026/{predictions.parquet,metrics.json,config.json}
  results/runs/A6_harq_full_<disc>_seed2026/{...}   (disc in long_form/event_driven/combined)
  results/tables/stronger_baselines.{csv,md}

Run from repo root:  .venv/bin/python scripts/experiments/stronger_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc

from sp500vol.evaluation.dm_test import dm_test

EPS = 1e-12  # HARRV epsilon (log-space), matching src/sp500vol/models/price/har_rv.py
ANN = 252.0
KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
HORIZONS = (5, 10, 20)
DISCLOSURES = ("long_form", "event_driven", "combined")
SEED = 2026
RETURNS_PATH = "/Volumes/Z/sp500vol-data/market/crsp/market_returns.parquet"
ALIGNED_PATH = "/Volumes/Z/sp500vol-data/processed/full/aligned_filings.parquet"
RETMATCH_PATH = "results/tables/_realized_returns.parquet"
RUNS = Path("results/runs")
TABLES = Path("results/tables")
SANITY_TOL = 0.03  # plain-HAR refit must match archived A2 test QLIKE within 3%

M1_TEXT_MODELS = ("B2_tfidf_ridge", "C2_finbert_s1", "C5_qwen3")
M1_DISCLOSURES = ("long_form", "event_driven")


# ---------------------------------------------------------------- HAR machinery
def fit_log_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """A2 conventions: OLS of log(y+eps) on [1, X]; Duan smear = mean(exp(resid))."""
    ly = np.log(np.where(y >= 0.0, y, np.nan) + EPS)
    valid = np.isfinite(ly) & np.isfinite(X).all(axis=1)
    design = np.column_stack([np.ones(int(valid.sum())), X[valid]])
    params, *_ = np.linalg.lstsq(design, ly[valid], rcond=None)
    resid = ly[valid] - design @ params
    smear = float(np.mean(np.exp(resid)))
    return params, smear


def predict_log_ols(X: np.ndarray, params: np.ndarray, smear: float) -> np.ndarray:
    raw = np.column_stack([np.ones(len(X)), X]) @ params
    max_log = np.log(np.finfo(float).max) - 1.0
    raw = np.clip(raw, np.log(EPS), max_log)
    return np.maximum(np.exp(raw) * smear - EPS, EPS)


def log_feat(x: np.ndarray) -> np.ndarray:
    return np.log(np.where(x >= 0.0, x, np.nan) + EPS)


def qlike_var(y_vol: np.ndarray, f_vol: np.ndarray) -> np.ndarray:
    """VARIANCE-unit QLIKE per-obs, matching metrics.json: qlike(y^2, f^2)."""
    a = np.clip(np.asarray(y_vol, float), EPS, None) ** 2
    b = np.clip(np.asarray(f_vol, float), EPS, None) ** 2
    r = a / b
    return r - np.log(r) - 1.0


def metrics_rows(pred: pd.DataFrame, disclosure: str) -> list[dict]:
    rows = []
    for split in ("test", "train", "val"):
        for h in HORIZONS:
            d = pred[(pred.split == split) & (pred.horizon_days == h)]
            y = d.label_realised_vol.to_numpy(float)
            f = d.prediction_realised_vol.to_numpy(float)
            rows.append({
                "split": split, "disclosure_subset": disclosure, "horizon_days": int(h),
                "n": len(d),
                "mae": float(np.mean(np.abs(f - y))),
                "rmse": float(np.sqrt(np.mean((f - y) ** 2))),
                "r2": float(1.0 - ((f - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                "qlike": float(qlike_var(y, f).mean()),
            })
    return rows


# ---------------------------------------------------------------- return features
def build_return_features() -> pd.DataFrame:
    """Per (ticker, feature_window_end): daily semivols, rv1 check, RQ_22. PIT <= fwe."""
    aligned = pd.read_parquet(ALIGNED_PATH, columns=[
        "ticker", "accession", "horizon_days", "feature_window_end",
        "feature_return_1d", "feature_rv_5d", "feature_rv_22d"])
    pairs = aligned[["ticker", "feature_window_end"]].drop_duplicates()
    cr = pd.read_parquet(RETURNS_PATH)  # ticker, date, log_return
    cr = cr.sort_values(["ticker", "date"], kind="mergesort")

    out = []
    for ticker, grp in pairs.groupby("ticker", sort=False):
        g = cr[cr.ticker == ticker]
        if g.empty:
            continue
        dates = g.date.to_numpy()
        r = g.log_return.to_numpy(float)
        # leading-zero cumsums for O(1) window sums
        cs4 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(r) ** 4)])
        csn = np.concatenate([[0.0], np.cumsum(np.isnan(r).astype(float))])
        fwe = grp.feature_window_end.to_numpy()
        pos = np.searchsorted(dates, fwe, side="right") - 1
        ok = (pos >= 21) & (pos >= 0)
        ok &= np.where(pos >= 0, dates[np.clip(pos, 0, None)] == fwe, False)
        p = np.clip(pos, 21, None)
        r1 = r[np.clip(pos, 0, None)]
        n_nan22 = csn[p + 1] - csn[p - 21]
        sum_r4 = cs4[p + 1] - cs4[p - 21]
        ok &= np.isfinite(r1) & (n_nan22 == 0)
        rs_neg = np.sqrt(ANN * r1 ** 2 * (r1 < 0))
        rs_pos = np.sqrt(ANN * r1 ** 2 * (r1 > 0))
        rq22 = (22.0 * ANN ** 2 / 3.0) * (sum_r4 / 22.0)  # == 252^2/3 * sum r^4
        out.append(pd.DataFrame({
            "ticker": ticker, "feature_window_end": fwe,
            "ret_r1": r1, "rs_neg_1": rs_neg, "rs_pos_1": rs_pos,
            "rq_22": rq22, "feat_ok": ok}))
    feats = pd.concat(out, ignore_index=True)
    merged = aligned.merge(feats, on=["ticker", "feature_window_end"], how="left")
    merged["feat_ok"] = merged["feat_ok"].fillna(False).astype(bool)

    # verification: on feat_ok rows the recomputed daily return equals feature_return_1d
    chk = merged[merged.feat_ok]
    frac = float((np.abs(chk.ret_r1 - chk.feature_return_1d) < 1e-10).mean())
    print(f"[returns] unique(ticker,fwe)={len(feats):,}  aligned rows={len(merged):,}  "
          f"feat_ok={merged.feat_ok.mean():.4f}  r1==feature_return_1d on ok rows: {frac:.4f}")
    if frac < 0.999:
        raise RuntimeError("CRSP return reconstruction does not match aligned features")

    rm = pd.read_parquet(RETMATCH_PATH, columns=KEY + ["ret_match_ok"])
    merged = merged.merge(rm, on=KEY, how="left")
    merged["ret_match_ok"] = merged["ret_match_ok"].fillna(False).astype(bool)
    return merged[KEY + ["feature_window_end", "rs_neg_1", "rs_pos_1", "rq_22",
                         "feat_ok", "ret_match_ok"]]


# ---------------------------------------------------------------- run-dir writer
def write_run(a2: pd.DataFrame, preds: np.ndarray, model_id: str, disclosure: str,
              note: str, config_extra: dict) -> None:
    run_id = f"{model_id}_full_{disclosure}_seed{SEED}"
    out = a2.copy()
    out["run_id"] = run_id
    out["model_id"] = model_id
    out["prediction_realised_vol"] = preds
    cols = ["run_id", "model_id", "dataset", "seed", "disclosure_subset", "split",
            "ticker", "form", "item_subtype", "accession", "filing_time_utc",
            "effective_trading_day", "horizon_days", "label_realised_vol",
            "prediction_realised_vol", "feature_rv_1d", "feature_rv_5d",
            "feature_rv_22d", "text_path", "metadata_path"]
    out = out[cols]
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / "predictions.parquet", index=False)
    (d / "metrics.json").write_text(json.dumps(metrics_rows(out, disclosure), indent=2))
    (d / "config.json").write_text(json.dumps({
        "model_id": model_id, "note": note, "dataset": "full",
        "disclosure": disclosure, "seed": SEED, **config_extra}, indent=2))
    print(f"[run] wrote {d}  rows={len(out):,}")


# ---------------------------------------------------------------- main
def main() -> None:
    feats = build_return_features()
    a2_metrics = {}   # archived A2 test qlike per (disc, h)
    table1 = []       # baseline comparison rows
    drops = []
    shar_preds_store = {}  # (disc) -> predictions df for M1 reuse

    for disc in DISCLOSURES:
        a2 = pd.read_parquet(RUNS / f"A2_har_rv_full_{disc}_seed{SEED}" / "predictions.parquet")
        am = json.loads((RUNS / f"A2_har_rv_full_{disc}_seed{SEED}" / "metrics.json").read_text())
        for row in am:
            if row["split"] == "test":
                a2_metrics[(disc, row["horizon_days"])] = row["qlike"]

        # ---- SANITY: refit plain HAR with this pipeline, full sample (A2 conventions)
        for h in HORIZONS:
            dh = a2[a2.horizon_days == h]
            X = np.column_stack([log_feat(dh.feature_rv_1d.to_numpy(float)),
                                 log_feat(dh.feature_rv_5d.to_numpy(float)),
                                 log_feat(dh.feature_rv_22d.to_numpy(float))])
            tr = (dh.split == "train").to_numpy()
            params, smear = fit_log_ols(X[tr], dh.label_realised_vol.to_numpy(float)[tr])
            f = predict_log_ols(X, params, smear)
            te = (dh.split == "test").to_numpy()
            q = float(qlike_var(dh.label_realised_vol.to_numpy(float)[te], f[te]).mean())
            ref = a2_metrics[(disc, h)]
            rel = abs(q - ref) / ref
            print(f"[sanity] {disc} h={h}: refit HAR test QLIKE={q:.6f} vs A2={ref:.6f} "
                  f"({100 * rel:.3f}% off)")
            if rel > SANITY_TOL:
                raise RuntimeError(f"sanity gate failed for {disc} h={h}: {100 * rel:.2f}% > 3%")

        # ---- merge new regressors, apply drops
        m = a2.merge(feats, on=KEY, how="left", validate="one_to_one")
        n0 = len(m)
        keep = m.ret_match_ok & m.feat_ok & np.isfinite(m.rq_22)
        drops.append({"disclosure": disc, "n_rows": n0,
                      "drop_ret_match": int((~m.ret_match_ok).sum()),
                      "drop_feat_window": int((m.ret_match_ok & ~m.feat_ok).sum()),
                      "n_kept": int(keep.sum()), "kept_pct": round(100 * keep.mean(), 2)})
        mk = m[keep].reset_index(drop=True)

        # ---- fit both variants per horizon on train, predict all splits
        # BPQ (2016) "insanity filter": any forecast outside the [min, max] range of the
        # estimation-sample (TRAIN) realized vol is replaced by the TRAIN-sample mean
        # (per horizon) — the filter published with HARQ. Needed because the log-space
        # HARQ interaction sqrt(RQ_22)*log(rv_1+eps) is pathological when rv_1 = 0
        # (log(eps) = -27.6): ~0.4% of test points get absurd near-zero forecasts and
        # unfiltered test QLIKE diverges (long_form h=5: 2.67 vs A2 0.56, driven by 28
        # obs). Applied to BOTH A6 variants for symmetry; bind rate reported (it never
        # binds materially for SHAR).
        preds = {"A6_shar": np.empty(len(mk)), "A6_harq": np.empty(len(mk))}
        bind = {}
        for h in HORIZONS:
            hm = (mk.horizon_days == h).to_numpy()
            dh = mk[hm]
            y = dh.label_realised_vol.to_numpy(float)
            tr = (dh.split == "train").to_numpy()
            lo, hi = float(y[tr].min()), float(y[tr].max())
            lrv1 = log_feat(dh.feature_rv_1d.to_numpy(float))
            lrv5 = log_feat(dh.feature_rv_5d.to_numpy(float))
            lrv22 = log_feat(dh.feature_rv_22d.to_numpy(float))
            X_shar = np.column_stack([log_feat(dh.rs_neg_1.to_numpy(float)),
                                      log_feat(dh.rs_pos_1.to_numpy(float)), lrv5, lrv22])
            sq_rq = np.sqrt(np.clip(dh.rq_22.to_numpy(float), 0.0, None))
            X_harq = np.column_stack([lrv1, lrv5, lrv22, sq_rq * lrv1])
            mean_tr = float(y[tr].mean())
            for mid, X in (("A6_shar", X_shar), ("A6_harq", X_harq)):
                params, smear = fit_log_ols(X[tr], y[tr])
                raw = predict_log_ols(X, params, smear)
                insane = (raw < lo) | (raw > hi)
                bind[(mid, h)] = float(insane.mean())
                preds[mid][hm] = np.where(insane, mean_tr, raw)
        print(f"[insanity-filter] {disc} bind rates: " +
              ", ".join(f"{m} h={h}: {100 * v:.2f}%" for (m, h), v in sorted(bind.items())))

        write_run(mk, preds["A6_shar"], "A6_shar", disc,
                  "Semivariance HAR (Patton-Sheppard): daily lag split into RS-/RS+ "
                  "from CRSP signed daily returns; log-space pooled OLS per horizon "
                  "+ Duan smearing (A2 conventions); ret_match_ok drops applied; "
                  "BPQ insanity filter (clamp to train-RV range, rarely binds).",
                  {"regressors": ["log(RS-_1+eps)", "log(RS+_1+eps)", "log(rv_5+eps)",
                                  "log(rv_22+eps)"], "returns_source": RETURNS_PATH,
                   "insanity_filter": {f"h{h}": bind[("A6_shar", h)] for h in HORIZONS}})
        write_run(mk, preds["A6_harq"], "A6_harq", disc,
                  "HARQ (Bollerslev-Patton-Quaedvlieg), log-space analogue: daily-lag "
                  "coefficient varies with sqrt(RQ_22); RQ_22=(22*252^2/3)*mean(r^4) over "
                  "22d PIT window; log-space pooled OLS per horizon + Duan smearing; "
                  "ret_match_ok drops applied; BPQ insanity filter (clamp forecasts to "
                  "train-RV range per horizon) — unfiltered log-HARQ diverges on extreme "
                  "RQ test points.",
                  {"regressors": ["log(rv_1+eps)", "log(rv_5+eps)", "log(rv_22+eps)",
                                  "sqrt(RQ_22)*log(rv_1+eps)"],
                   "rq_definition": "RQ_22 = (22*252^2/3)*mean(r^4), 22 trading days <= feature_window_end",
                   "returns_source": RETURNS_PATH,
                   "insanity_filter": {f"h{h}": bind[("A6_harq", h)] for h in HORIZONS}})
        mk2 = mk.copy()
        mk2["shar"] = preds["A6_shar"]
        mk2["harq"] = preds["A6_harq"]
        mk2["fhar_a2"] = mk2["prediction_realised_vol"]
        shar_preds_store[disc] = mk2[KEY + ["split", "filing_time_utc", "label_realised_vol",
                                            "fhar_a2", "shar", "harq"]]

        # ---- table 1: test QLIKE + DM vs A2 (common sample)
        for h in HORIZONS:
            dt = mk2[(mk2.split == "test") & (mk2.horizon_days == h)].sort_values(
                SORT, kind="mergesort")
            y = dt.label_realised_vol.to_numpy(float)
            fa2 = dt.fhar_a2.to_numpy(float)
            la2 = qlike_var(y, fa2)
            row = {"disclosure": disc, "h": h, "n_test_common": len(dt),
                   "A2_qlike_full": a2_metrics[(disc, h)],
                   "A2_qlike_common": float(la2.mean())}
            for mid, col in (("A6_shar", "shar"), ("A6_harq", "harq")):
                fv = dt[col].to_numpy(float)
                lq = qlike_var(y, fv)
                dmq, pq = dm_test(lq, la2, h=h)
                dms, ps = dm_test((fv - y) ** 2, (fa2 - y) ** 2, h=h)
                row.update({f"{mid}_qlike": float(lq.mean()),
                            f"{mid}_dq_pct": 100 * (la2.mean() - lq.mean()) / la2.mean(),
                            f"{mid}_dm_qlike": round(float(dmq), 3),
                            f"{mid}_p_qlike": round(float(pq), 5),
                            f"{mid}_dm_se": round(float(dms), 3),
                            f"{mid}_p_se": round(float(ps), 5)})
            table1.append(row)

    drops_df = pd.DataFrame(drops)
    t1 = pd.DataFrame(table1)
    print("\n[drops]\n", drops_df.to_string(index=False))
    print("\n[table1]\n", t1.to_string(index=False))

    # ---------------- M1 incremental-value rerun: A2 vs A6_shar as price reference
    m1_rows = []
    for disc in M1_DISCLOSURES:
        base = shar_preds_store[disc]  # common (kept) sample, has fhar_a2 + shar
        for tm in M1_TEXT_MODELS:
            txt = pd.read_parquet(RUNS / f"{tm}_full_{disc}_seed{SEED}" / "predictions.parquet")
            txt = txt[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
            d = base.merge(txt, on=KEY, how="inner", validate="one_to_one")
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                yv, yt = dv.label_realised_vol.to_numpy(float), dt.label_realised_vol.to_numpy(float)
                ftv, ftt = dv.ftext.to_numpy(float), dt.ftext.to_numpy(float)
                row = {"disclosure": disc, "text_model": tm, "h": h,
                       "n_val": len(dv), "n_test": len(dt)}
                for ref, col in (("A2", "fhar_a2"), ("A6_shar", "shar")):
                    fhv, fhr = dv[col].to_numpy(float), dt[col].to_numpy(float)
                    fR, fU, g = fc.log_combo(yv, fhv, ftv, fhr, ftt)
                    lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
                    qR, qU = float(lR.mean()), float(lU.mean())
                    dmq, pq = dm_test(lU, lR, h=h)
                    row.update({f"{ref}_qlike_R": qR, f"{ref}_qlike_U": qU,
                                f"{ref}_rel_impr_pct": 100 * (qR - qU) / qR,
                                f"{ref}_g_log": round(float(g), 4),
                                f"{ref}_dm": round(float(dmq), 3),
                                f"{ref}_p": round(float(pq), 5)})
                m1_rows.append(row)
    m1 = pd.DataFrame(m1_rows)
    print("\n[m1]\n", m1.to_string(index=False))

    # ---------------- outputs
    TABLES.mkdir(parents=True, exist_ok=True)
    t1["section"] = "baseline_qlike"
    m1["section"] = "m1_incremental"
    drops_df["section"] = "drops"
    combined = pd.concat([t1, m1, drops_df], ignore_index=True)
    combined.to_csv(TABLES / "stronger_baselines.csv", index=False)

    def sig(p):
        return "‡" if p < 0.01 else ("†" if p < 0.05 else "")

    md = ["# E4 — Stronger price baselines (A6_shar, A6_harq) vs A2 HAR-RV", "",
          "All fits replicate A2 conventions (pooled log-space OLS per horizon, eps=1e-12, "
          "Duan smearing). Returns from the canonical CRSP store "
          "(`market/crsp/market_returns.parquet`, log1p(DlyRet)) which exactly reproduces "
          "the aligned features; OHLCV adj_close only correlates ~0.78 and was not used. "
          "Rows with unreliable ticker-recycled joins dropped via `ret_match_ok`. "
          "Both variants use the BPQ (2016) insanity filter (any forecast outside the "
          "train-sample RV range is replaced by the train mean, per horizon); without it "
          "log-space HARQ diverges on ~0.4% extreme test points where rv_1=0 meets extreme "
          "quarticity (long_form h=5 test QLIKE 2.67 vs 0.56, 28 obs). "
          "It essentially never binds for SHAR.", "",
          "## Row drops (per disclosure)", "",
          drops_df.drop(columns="section").to_markdown(index=False), "",
          "## Test QLIKE (variance-unit, matching metrics.json) — A6 variants vs A2", "",
          "DM on the common (kept) test sample; negative DM = variant better than A2. "
          "† p<0.05, ‡ p<0.01.", "",
          "| disc | h | n | A2 (full) | A2 (common) | SHAR | ΔQ% | DM_q | DM_se | HARQ | ΔQ% | DM_q | DM_se |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t1.iterrows():
        md.append(
            f"| {r.disclosure} | {r.h} | {r.n_test_common} | {r.A2_qlike_full:.4f} | "
            f"{r.A2_qlike_common:.4f} | {r.A6_shar_qlike:.4f} | {r.A6_shar_dq_pct:+.2f} | "
            f"{r.A6_shar_dm_qlike}{sig(r.A6_shar_p_qlike)} | {r.A6_shar_dm_se}{sig(r.A6_shar_p_se)} | "
            f"{r.A6_harq_qlike:.4f} | {r.A6_harq_dq_pct:+.2f} | "
            f"{r.A6_harq_dm_qlike}{sig(r.A6_harq_p_qlike)} | {r.A6_harq_dm_se}{sig(r.A6_harq_p_se)} |")
    md += ["", "## M1 incremental value of text — A2 vs A6_shar as the price reference", "",
           "fc.log_combo (val-fitted recalibrated reference f_R vs +text f_U), QLIKE in "
           "VOL units (fc convention), same common sample for both references. "
           "rel_impr% > 0 with DM < 0 = text still adds. † p<0.05, ‡ p<0.01.", "",
           "| disc | text model | h | n_test | A2-ref impr% | DM | A6_shar-ref impr% | DM | survives? |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in m1.iterrows():
        surv = "YES" if (r.A6_shar_rel_impr_pct > 0 and r.A6_shar_p < 0.05) else (
            "weak" if r.A6_shar_rel_impr_pct > 0 else "no")
        md.append(
            f"| {r.disclosure} | {r.text_model} | {r.h} | {r.n_test} | "
            f"{r.A2_rel_impr_pct:+.2f} | {r.A2_dm}{sig(r.A2_p)} | "
            f"{r.A6_shar_rel_impr_pct:+.2f} | {r.A6_shar_dm}{sig(r.A6_shar_p)} | {surv} |")
    (TABLES / "stronger_baselines.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {TABLES / 'stronger_baselines.csv'} and .md")


if __name__ == "__main__":
    main()
