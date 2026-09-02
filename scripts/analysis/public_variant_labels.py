#!/usr/bin/env python
"""PREREG H (prereg-h-v1.0, configs/prereg_public_variant.md) — TASK 1: labels + features.

Builds the "pub" estimator inputs for the public-price full-cascade variant:
forward RV LABELS and A-block FEATURES (rv_1d/5d/22d analogues + SHAR's signed
RS-/RS+ daily semivols) recomputed from a genuinely licence-free source (Yahoo
Finance v8 chart API daily adjusted close), on the benchmark's EXACT per-row
label/feature windows.

MACHINERY (reused verbatim, per the prereg):
  * Stage-1 fetcher  : label_parity.fetch_one (Yahoo v8, same URL/params/retries),
                       6 network workers, cache re-created under the CURRENT session
                       scratchpad (the committed cache is gone; refetch drift vs the
                       committed results/tables/label_parity.csv is disclosed below).
  * Stage-2 labels   : label_parity.squared_series(mask_gaps=True on the public side)
                       + label_parity.compute_labels over the exact aligned windows
                       (GATE 3 calendar assert built in).
  * mismatch screen  : label_parity.mismatch_screen (public-vs-CRSP daily-return corr
                       < 0.80 on >= 60 overlap days => ticker NOT covered).
  * coverage decomp  : label_parity conventions (reason / exit-year / filing-year),
                       extended with the per-SPLIT trade numbers the cascade md needs.
  * features         : rangebased_labels.TickerSeries (at_day / tail_sum — the exact
                       alignment.py _return_at / _backward_realised_vol mirrors,
                       window-verified on the CRSP side to < 1e-8 before use), fed
                       with the gap-masked public return series:
                         pub_1d  = sqrt(252 * r_pub(fwe)^2)     (= sqrt(252)*|r|)
                         pub_5d  = sqrt(252/5  * sum last-5-valid r^2 <= fwe)
                         pub_22d = sqrt(252/22 * sum last-22-valid r^2 <= fwe)
                         rs_neg_pub/rs_pos_pub = sqrt(252 * r^2 * 1[r<0 / r>0]) at fwe
                       (stronger_baselines.build_return_features sign decomposition,
                       rebuilt from public adjusted closes per the prereg).

GATES (this script; the cascade has its own G1):
  L1 (label-verification, relaxed to covered-rows-only per the task brief): the CRSP
     labels recomputed through THIS pipeline's Stage-2 machinery must equal the stored
     aligned labels to < 1e-8 on every reconstructable row with ZERO unreconstructable
     rows (the CRSP side is complete, so "covered" = all rows there); predictions.parquet
     labels must equal aligned labels exactly on the modelled panel. On the PUBLIC side
     rows without a clean label are COUNTED (never scored) and reconciled to the
     coverage table — that is the covered-rows-only relaxation.
  L1b (feature-window verification): the TickerSeries feature machinery, fed CRSP
     returns, must reproduce stored feature_return_1d / feature_rv_5d / feature_rv_22d
     to < 1e-8 (NaN-pattern mismatches counted) — proves the feature windows used for
     the public features are the alignment.py windows.
  L1c (mask consistency): the gap-masked public return set used for FEATURES must
     regenerate byte-identical squared-series arrays to the label-side
     squared_series(mask_gaps=True) call — one mask, two consumers.
  L2 : A2 raw test QLIKE recomputed for the C2 cells == committed
       forecast_combination_grid.csv qlike_raw (machine precision) — anchors fc.load.
  G2 (prereg): Pearson(log RV) public-vs-CRSP on covered modelled rows >= 0.99
       (prior 0.998) — HARD ABORT below.
  G3 (prereg): coverage reconciliation vs the committed label_parity.csv
       (80.19% clean +- refetch drift), per-split / per-exit-status decomposition
       side-by-side; drift > 0.5pp coverage or > 0.001 correlation is FLAGGED
       prominently in every downstream product.

Outputs:
  $SP500VOL_DATA_ROOT/processed/full/public_variant_labels.parquet
      keyed accession x horizon_days:
      [accession, ticker, horizon_days, label_pub, pub_1d, pub_5d, pub_22d,
       rs_neg_pub, rs_pos_pub]
  $SP500VOL_DATA_ROOT/processed/full/public_variant_labels_meta.json
      (gates, coverage incl. per-split trade numbers, refetch-drift table, formulas)

Run from repo root:  PV_THREADS=2 .venv/bin/python scripts/analysis/public_variant_labels.py
"""
from __future__ import annotations

import os
import tempfile

_THREADS = os.environ.get("PV_THREADS", "2")  # env caps BEFORE numpy import (shared box)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import label_parity as lp  # noqa: E402  (fetch_one, squared_series, compute_labels, ...)
import forecast_combination as fc  # noqa: E402
from rangebased_labels import TickerSeries, _dt  # noqa: E402

DATA = Path(os.environ.get("SP500VOL_DATA_ROOT", "/path/to/data-root/sp500vol-data"))
SCRATCH = Path(os.environ.get("SP500VOL_SCRATCH", tempfile.gettempdir())) / "public_prices"
SCRATCH.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = DATA / "processed/full/public_variant_labels.parquet"
OUT_META = DATA / "processed/full/public_variant_labels_meta.json"
COMMITTED_PARITY = REPO / "results/tables/label_parity.csv"

KEY = ["ticker", "accession", "horizon_days"]
HORIZONS = (5, 10, 20)
ANN = 252.0
VERIFY_TOL = 1e-8
G2_MIN = 0.99
DRIFT_COV_PP = 0.5      # flag threshold, percentage points of clean coverage
DRIFT_CORR = 0.001      # flag threshold, Pearson log-RV
DISCS = ["long_form", "event_driven"]


# --------------------------------------------------------------------------- #
# Stage 1 — fetch (label_parity fetch_one VERBATIM; new cache dir; 6 workers)
# --------------------------------------------------------------------------- #
def fetch_public_prices(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_px = SCRATCH / "yahoo_adjclose.parquet"
    cache_st = SCRATCH / "yahoo_status.csv"
    if cache_px.exists() and cache_st.exists():
        st = pd.read_csv(cache_st)
        if set(tickers) <= set(st.ticker):
            print(f"[1] using cached Yahoo download: {cache_px}")
            return pd.read_parquet(cache_px), st
    print(f"[1] fetching Yahoo adj-close for {len(tickers)} tickers "
          f"(label_parity.fetch_one verbatim; 6 network workers) ...")
    frames, status = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(lp.fetch_one, t): t for t in tickers}
        for fut in as_completed(futs):
            tkr, st_, df = fut.result()
            status.append({"ticker": tkr, "status": st_,
                           "n_days": 0 if df is None else int(len(df)),
                           "first": None if df is None else str(df.date.min().date()),
                           "last": None if df is None else str(df.date.max().date())})
            if df is not None:
                frames.append(df)
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(tickers)} tickers fetched")
    px = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["ticker", "date", "adj_close"])
    st = pd.DataFrame(status).sort_values("ticker")
    px.to_parquet(cache_px, index=False)
    st.to_csv(cache_st, index=False)
    ok = int((st.status == "ok").sum())
    print(f"    done: {ok}/{len(tickers)} tickers returned data; cache -> {cache_px}")
    return px, st


# --------------------------------------------------------------------------- #
# gap mask — the ONE mask, mirroring label_parity.squared_series(mask_gaps=True)
# --------------------------------------------------------------------------- #
def gap_masked_returns(ret: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep only returns on calendar days whose previous observation is the
    immediately preceding trading day (first return per ticker dropped)."""
    pos = pd.Series(np.arange(len(cal)), index=cal)
    parts = []
    for tkr, g in ret.groupby("ticker", sort=False):
        g = g.sort_values("date")
        idx = pos.reindex(pd.DatetimeIndex(g.date).normalize()).to_numpy()
        keep = ~np.isnan(idx)
        g = g[keep]
        ii = idx[keep].astype(int)
        if len(g) == 0:
            continue
        prev_ok = np.zeros(len(g), dtype=bool)
        prev_ok[1:] = (ii[1:] - ii[:-1]) == 1
        parts.append(g[prev_ok])
    return pd.concat(parts, ignore_index=True)


def gate_l1c_mask_consistency(masked: pd.DataFrame, sq_pub: dict, cal) -> None:
    """The masked return set must regenerate the label-side squared arrays exactly."""
    pos = pd.Series(np.arange(len(cal)), index=cal)
    worst = 0.0
    n_pat = 0
    for tkr, g in masked.groupby("ticker", sort=False):
        arr = np.full(len(cal), np.nan)
        ii = pos.reindex(pd.DatetimeIndex(g.date).normalize()).to_numpy().astype(int)
        arr[ii] = np.square(g.log_return.to_numpy(dtype=np.float64))
        ref = sq_pub.get(tkr)
        if ref is None:
            raise AssertionError(f"L1c: ticker {tkr} missing from label-side series")
        pat = (np.isnan(arr) != np.isnan(ref)).sum()
        n_pat += int(pat)
        m = np.isfinite(arr) & np.isfinite(ref)
        if m.any():
            worst = max(worst, float(np.max(np.abs(arr[m] - ref[m]))))
    if n_pat != 0 or worst != 0.0:
        raise AssertionError(f"GATE L1c FAILED: mask divergence (pattern={n_pat}, "
                             f"max|diff|={worst}) between feature and label series")
    print(f"[L1c] mask consistency: feature-side masked set regenerates the label-side "
          f"squared series exactly (pattern mismatches=0, max|diff|=0.0) -> PASS")


# --------------------------------------------------------------------------- #
# features via TickerSeries (rangebased machinery), any return source
# --------------------------------------------------------------------------- #
def build_feature_series(ret: pd.DataFrame):
    """(ticker -> TickerSeries of r^2, ticker -> TickerSeries of signed r)."""
    ret = ret.sort_values(["ticker", "date"], kind="mergesort")
    r2s, rs = {}, {}
    for tkr, g in ret.groupby("ticker", sort=False):
        d = _dt(g.date.to_numpy())
        r = g.log_return.to_numpy(dtype=float)
        r2s[tkr] = TickerSeries(d, r ** 2)
        rs[tkr] = TickerSeries(d, r)
    return r2s, rs


def compute_features(af: pd.DataFrame, r2s: dict, rs: dict) -> dict[str, np.ndarray]:
    n = len(af)
    out = {k: np.full(n, np.nan) for k in ("f1d", "f5d", "f22d", "r1", "rsn", "rsp")}
    fwe = _dt(af.feature_window_end.to_numpy())
    for ticker, idx in af.groupby("ticker", sort=False).indices.items():
        t = str(ticker)
        f = fwe[idx]
        if t in r2s:
            ts = r2s[t]
            out["f1d"][idx] = np.sqrt(ANN * ts.at_day(f))
            out["f5d"][idx] = np.sqrt(ANN / 5.0 * ts.tail_sum(f, 5))
            out["f22d"][idx] = np.sqrt(ANN / 22.0 * ts.tail_sum(f, 22))
        if t in rs:
            r1 = rs[t].at_day(f)
            out["r1"][idx] = r1
            out["rsn"][idx] = np.sqrt(ANN * r1 ** 2 * (r1 < 0))
            out["rsp"][idx] = np.sqrt(ANN * r1 ** 2 * (r1 > 0))
    return out


# --------------------------------------------------------------------------- #
def committed_reference() -> dict:
    """Pull the committed label_parity.csv drift-comparison quantities."""
    c = pd.read_csv(COMMITTED_PARITY)
    cov = c[(c.section == "coverage") & (c.metric == "overall")].iloc[0]
    tk = c[(c.section == "ticker_status")].iloc[0]
    par = c[(c.section == "parity_corr") & (c.scope == "modelled_all_splits")].iloc[0]
    by_split = c[(c.section == "parity_corr") & (c.scope == "by_split_h")][
        ["split", "h", "n", "pearson_logRV"]].copy()
    by_exit = c[c.section == "coverage_by_exit_year"][
        ["exit_year", "n_firms", "n_rows", "coverage_clean"]].copy()
    by_fy = c[c.section == "coverage_by_filing_year"][
        ["year", "n_rows", "coverage_clean"]].copy()
    return {
        "coverage_raw": float(cov.coverage_raw), "coverage_clean": float(cov.coverage_clean),
        "rows_covered": int(cov.rows_covered),
        "n_yahoo_ok": int(tk.n_yahoo_ok), "n_no_data": int(tk.n_no_data),
        "n_mismatch": int(tk.n_mismatch),
        "mismatch_tickers": set(str(tk.mismatch_tickers).split(";")),
        "pearson": float(par.pearson_logRV), "spearman": float(par.spearman_logRV),
        "parity_n": int(par.n),
        "by_split_h": by_split, "by_exit": by_exit, "by_filing_year": by_fy,
    }


def main() -> int:
    t0 = time.time()
    print(f"[0] PREREG H labels build (threads={_THREADS}); scratch cache: {SCRATCH}")

    af = pd.read_parquet(
        DATA / "processed/full/aligned_filings.parquet",
        columns=KEY + ["form", "label_realised_vol", "label_window_start",
                       "label_window_end", "feature_window_end", "effective_trading_day",
                       "feature_return_1d", "feature_rv_5d", "feature_rv_22d"])
    a2 = {}
    for disc in DISCS:
        a2[disc] = fc.load("A2_har_rv", disc)[
            ["split", "form"] + KEY + ["prediction_realised_vol", "label_realised_vol",
                                       "filing_time_utc", "effective_trading_day"]
        ].rename(columns={"prediction_realised_vol": "fhar"})
    tickers = sorted(af.ticker.unique())
    print(f"    benchmark rows={len(af):,} tickers={len(tickers)}")

    # ---------------- Stage 1: fetch ----------------
    px, status = fetch_public_prices(tickers)
    pub_ret = lp.public_log_returns(px)
    crsp_ret = pd.read_parquet(DATA / "processed/full/market_returns.parquet")
    crsp_ret["date"] = pd.to_datetime(crsp_ret.date).dt.normalize()
    pub_ret["date"] = pd.to_datetime(pub_ret.date).dt.normalize()

    # ---------------- Stage 2: labels (label_parity machinery VERBATIM) ----------------
    print("[2] recomputing labels on the exact aligned windows (CRSP + public) ...")
    cal = lp.calendar_index()
    sq_crsp = lp.squared_series(crsp_ret, cal, mask_gaps=False)
    sq_pub = lp.squared_series(pub_ret, cal, mask_gaps=True)
    y_crsp = lp.compute_labels(af, sq_crsp, cal)   # GATE 3 calendar assert inside
    y_pub = lp.compute_labels(af, sq_pub, cal)
    print(f"    CRSP reconstructed {np.isfinite(y_crsp).sum():,}/{len(af):,}; "
          f"public label valid on {np.isfinite(y_pub).sum():,}/{len(af):,} rows")

    # ---------------- GATE L1 ----------------
    lp.ROWS.clear()
    g1 = lp.gate1_labels(af, y_crsp, a2)  # aborts on failure (machine precision, all rows)

    # ---------------- GATE L2 ----------------
    panels_model = {disc: lp.build_model_panel(disc, "C2_finbert_s1", a2[disc])
                    for disc in DISCS}
    lp.gate2_qlike(panels_model)          # aborts on failure

    # ---------------- mismatch screen ----------------
    print("[4] symbol-mismatch screen ...")
    mm = lp.mismatch_screen(crsp_ret, pub_ret)
    mm_t = set(mm[mm.mismatch].ticker)
    n_ok = int((status.status == "ok").sum())
    print(f"    tickers: ok={n_ok}, no-data={len(tickers) - n_ok}, mismatch={len(mm_t)}")

    af = af.assign(y_pub=y_pub)
    af["y_pub_clean"] = np.where(af.ticker.isin(mm_t), np.nan, af.y_pub)

    # ---------------- features: CRSP verification then public ----------------
    print("[5] feature-window machinery: CRSP-side verification (GATE L1b) ...")
    r2s_c, rs_c = build_feature_series(crsp_ret)
    chk = compute_features(af, r2s_c, rs_c)
    _no_fwe = af.feature_window_end.isna().to_numpy()
    for _k in chk:  # NaT feature_window_end rows carry no aligned features by design
        chk[_k] = np.where(_no_fwe, np.nan, chk[_k])
    r1_old = af.feature_return_1d.to_numpy(dtype=float)
    f5_old = af.feature_rv_5d.to_numpy(dtype=float)
    f22_old = af.feature_rv_22d.to_numpy(dtype=float)
    m1 = np.isfinite(chk["r1"]) & np.isfinite(r1_old)
    m5 = np.isfinite(chk["f5d"]) & np.isfinite(f5_old)
    m22 = np.isfinite(chk["f22d"]) & np.isfinite(f22_old)
    d_r1 = float(np.max(np.abs(chk["r1"][m1] - r1_old[m1])))
    d_f5 = float(np.max(np.abs(chk["f5d"][m5] - f5_old[m5])))
    d_f22 = float(np.max(np.abs(chk["f22d"][m22] - f22_old[m22])))
    pat5 = int((np.isfinite(chk["f5d"]) != np.isfinite(f5_old)).sum())
    pat22 = int((np.isfinite(chk["f22d"]) != np.isfinite(f22_old)).sum())
    print(f"    max|dr1|={d_r1:.3e}  max|drv5|={d_f5:.3e} (NaN-pattern {pat5})  "
          f"max|drv22|={d_f22:.3e} (NaN-pattern {pat22})")
    if not (d_r1 < VERIFY_TOL and d_f5 < VERIFY_TOL and d_f22 < VERIFY_TOL):
        print("GATE L1b FAILED — feature windows do not reproduce the aligned features. "
              "ABORTING, nothing written.")
        return 2
    del r2s_c, rs_c, chk

    print("[6] public features (gap-masked series; GATE L1c mask consistency) ...")
    masked = gap_masked_returns(pub_ret, cal)
    gate_l1c_mask_consistency(masked, sq_pub, cal)
    r2s_p, rs_p = build_feature_series(masked)
    pf = compute_features(af, r2s_p, rs_p)
    no_fwe = af.feature_window_end.isna().to_numpy()
    for k in pf:  # mismatch tickers are NOT covered — features masked like the label;
        # rows with NaT feature_window_end (12 aligned rows, outside the modelled
        # panel) carry no features either (searchsorted(NaT) would silently hit the
        # end of history — masked here, counted in the coverage table).
        pf[k] = np.where(af.ticker.isin(mm_t) | no_fwe, np.nan, pf[k])

    # ---------------- coverage decomposition + per-split trade numbers ----------------
    print("[7] coverage decomposition ...")
    last_day = crsp_ret.groupby("ticker").date.max()
    exit_year = last_day.apply(lambda d: "active" if d >= lp.ACTIVE_CUTOFF else str(d.year))
    af["exit_year"] = af.ticker.map(exit_year)
    st_map = status.set_index("ticker").status
    reason = np.where(af.ticker.map(st_map).ne("ok"), "no_public_data",
                      np.where(af.ticker.isin(mm_t), "symbol_mismatch",
                               np.where(af.y_pub.isna(), "window_incomplete", "covered")))
    af["pub_reason"] = np.where(af.y_pub_clean.notna(), "covered", reason)
    cov_raw = float(af.y_pub.notna().mean())
    cov_clean = float(af.y_pub_clean.notna().mean())
    reasons = af.pub_reason.value_counts().to_dict()
    by_exit = []
    for ey, g in af.groupby("exit_year"):
        by_exit.append({"exit_year": ey, "n_firms": int(g.ticker.nunique()),
                        "n_rows": int(len(g)),
                        "coverage_clean": float(g.y_pub_clean.notna().mean())})
    af["filing_year"] = pd.DatetimeIndex(af.effective_trading_day).year
    by_fy = [{"year": int(fy), "n_rows": int(len(g)),
              "coverage_clean": float(g.y_pub_clean.notna().mean())}
             for fy, g in af.groupby("filing_year")]
    ex_rows = af[af.exit_year != "active"]
    exit_cov = float(ex_rows.y_pub_clean.notna().mean())
    active_cov = float(af[af.exit_year == "active"].y_pub_clean.notna().mean())

    pub_map = af.set_index(KEY)[["y_pub", "y_pub_clean"]]
    mod = pd.concat([a2[disc].assign(disc=disc) for disc in DISCS], ignore_index=True)
    mod = mod.join(pub_map, on=KEY)
    split_cov = {}
    for split in ("train", "val", "test"):
        s = mod[mod.split == split]
        split_cov[split] = {"n_rows": int(len(s)),
                            "n_covered": int(s.y_pub_clean.notna().sum()),
                            "coverage_clean": float(s.y_pub_clean.notna().mean())}
    print("    per-split clean coverage (modelled panel): " +
          ", ".join(f"{k} {100 * v['coverage_clean']:.1f}%" for k, v in split_cov.items()))

    # ---------------- GATE G2: parity on covered modelled rows ----------------
    ok = mod.y_pub_clean.notna() & (mod.label_realised_vol > 0) & (mod.y_pub_clean > 0)
    d = mod[ok]
    lx, ly = np.log(d.label_realised_vol.to_numpy()), np.log(d.y_pub_clean.to_numpy())
    pearson = float(stats.pearsonr(lx, ly)[0])
    spearman = float(stats.spearmanr(lx, ly)[0])
    print(f"[G2] covered-row parity: n={len(d):,} Pearson(logRV)={pearson:.6f} "
          f"Spearman={spearman:.6f} (gate >= {G2_MIN})")
    if pearson < G2_MIN:
        print("GATE G2 FAILED — Pearson below 0.99: ABORT AND DEBUG THE FETCH "
              "(prereg instruction). Nothing written.")
        return 3
    by_split_h = []
    for split in ("train", "val", "test"):
        for h in HORIZONS:
            s = mod[(mod.split == split) & (mod.horizon_days == h)]
            oks = s.y_pub_clean.notna() & (s.label_realised_vol > 0) & (s.y_pub_clean > 0)
            ss = s[oks]
            pe = float(stats.pearsonr(np.log(ss.label_realised_vol), np.log(ss.y_pub_clean))[0]) \
                if len(ss) >= 30 else float("nan")
            by_split_h.append({"split": split, "h": h, "n": int(len(ss)), "pearson": pe})

    # ---------------- G3: refetch drift vs committed label_parity.csv ----------------
    ref = committed_reference()
    drift = {
        "coverage_clean_now": cov_clean, "coverage_clean_committed": ref["coverage_clean"],
        "coverage_clean_drift_pp": 100.0 * (cov_clean - ref["coverage_clean"]),
        "coverage_raw_now": cov_raw, "coverage_raw_committed": ref["coverage_raw"],
        "pearson_now": pearson, "pearson_committed": ref["pearson"],
        "pearson_drift": pearson - ref["pearson"],
        "spearman_now": spearman, "spearman_committed": ref["spearman"],
        "parity_n_now": int(len(d)), "parity_n_committed": ref["parity_n"],
        "n_yahoo_ok_now": n_ok, "n_yahoo_ok_committed": ref["n_yahoo_ok"],
        "n_mismatch_now": len(mm_t), "n_mismatch_committed": ref["n_mismatch"],
        "mismatch_added_vs_committed": sorted(mm_t - ref["mismatch_tickers"]),
        "mismatch_dropped_vs_committed": sorted(ref["mismatch_tickers"] - mm_t),
    }
    drift["flag_coverage"] = abs(drift["coverage_clean_drift_pp"]) > DRIFT_COV_PP
    drift["flag_correlation"] = abs(drift["pearson_drift"]) > DRIFT_CORR
    drift["FLAGGED"] = bool(drift["flag_coverage"] or drift["flag_correlation"])
    ref_split = ref["by_split_h"].set_index(["split", "h"])
    for r in by_split_h:
        key = (r["split"], float(r["h"]))
        if key in ref_split.index:
            r["n_committed"] = int(ref_split.loc[key, "n"])
            r["pearson_committed"] = float(ref_split.loc[key, "pearson_logRV"])
    print(f"[G3] refetch drift: clean coverage {100 * cov_clean:.2f}% vs committed "
          f"{100 * ref['coverage_clean']:.2f}% (d={drift['coverage_clean_drift_pp']:+.3f}pp); "
          f"Pearson {pearson:.6f} vs {ref['pearson']:.6f} "
          f"(d={drift['pearson_drift']:+.6f}); FLAGGED={drift['FLAGGED']}")

    # exit-year drift (committed side-by-side)
    ref_exit = ref["by_exit"].set_index("exit_year")
    for r in by_exit:
        if r["exit_year"] in ref_exit.index:
            r["coverage_clean_committed"] = float(
                ref_exit.loc[r["exit_year"], "coverage_clean"])
    ref_fy = ref["by_filing_year"].set_index("year")
    for r in by_fy:
        if float(r["year"]) in ref_fy.index:
            r["coverage_clean_committed"] = float(ref_fy.loc[float(r["year"]), "coverage_clean"])

    # ---------------- reconcile: parquet coverage == coverage table ----------------
    label_pub = af.y_pub_clean.to_numpy()
    n_cov_parquet = int(np.isfinite(label_pub).sum())
    assert n_cov_parquet == int(reasons.get("covered", 0)), \
        "coverage reconciliation failed: parquet covered != reason table covered"
    feat_ok = (np.isfinite(pf["f1d"]) & np.isfinite(pf["f5d"]) & np.isfinite(pf["f22d"]))
    rs_ok = np.isfinite(pf["rsn"]) & np.isfinite(pf["rsp"])
    cov_out = {
        "panel_rows": int(len(af)),
        "rows_covered_label": n_cov_parquet,
        "rows_label_missing": int(len(af) - n_cov_parquet),
        "reasons": {k: int(v) for k, v in reasons.items()},
        "rows_pubfeat_ok(1d,5d,22d)": int(feat_ok.sum()),
        "rows_rs_ok": int(rs_ok.sum()),
        "rows_usable_pub(label+feat)": int((np.isfinite(label_pub) & feat_ok).sum()),
        "rows_usable_pub(label+feat+rs)": int((np.isfinite(label_pub) & feat_ok & rs_ok).sum()),
        "exit_firm_row_coverage": exit_cov, "active_firm_row_coverage": active_cov,
        "per_split": split_cov,
    }
    print("[coverage] " + json.dumps({k: v for k, v in cov_out.items()
                                      if k != "per_split"}, indent=2))

    # ---------------- outputs ----------------
    out = pd.DataFrame({
        "accession": af.accession.to_numpy(), "ticker": af.ticker.to_numpy(),
        "horizon_days": af.horizon_days.to_numpy(),
        "label_pub": label_pub,
        "pub_1d": pf["f1d"], "pub_5d": pf["f5d"], "pub_22d": pf["f22d"],
        "rs_neg_pub": pf["rsn"], "rs_pos_pub": pf["rsp"],
    })
    assert not out.duplicated(["accession", "horizon_days"]).any(), \
        "output must be uniquely keyed accession x horizon"
    meta = {
        "prereg": "configs/prereg_public_variant.md (prereg-h-v1.0)",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "threads": _THREADS,
        "source": "Yahoo Finance v8 chart API adjusted close (split+dividend adjusted), "
                  "fetched via label_parity.fetch_one verbatim; cache (non-redistributable) "
                  f"at {SCRATCH}",
        "gates": {
            "L1_crsp_label_reconstruction": {k: (float(v) if isinstance(v, (int, float))
                                                 else bool(v)) for k, v in g1.items()},
            "L1b_feature_windows": {"max_abs_dr1": d_r1, "max_abs_drv5": d_f5,
                                    "max_abs_drv22": d_f22, "nan_pattern_rv5": pat5,
                                    "nan_pattern_rv22": pat22, "tol": VERIFY_TOL,
                                    "pass": True},
            "L1c_mask_consistency": "PASS (exact)",
            "L2_a2_qlike_anchor": "PASS (aborts otherwise; label_parity.gate2_qlike)",
            "G2_parity": {"n": int(len(d)), "pearson_logRV": pearson,
                          "spearman_logRV": spearman, "gate_min": G2_MIN, "pass": True},
            "G3_coverage_reconciliation": "covered rows in parquet == coverage table (assert)",
        },
        "coverage": cov_out,
        "parity_by_split_h": by_split_h,
        "coverage_by_exit_year": by_exit,
        "coverage_by_filing_year": by_fy,
        "refetch_drift_vs_committed_label_parity": drift,
        "formulas": {
            "label_pub": "sqrt(252/H * sum r_pub^2) over the exact aligned label windows; "
                         "gap-spanning returns masked; any missing/masked day in the window "
                         "=> NO label (coverage failure, never a wrong label); "
                         "mismatch-screened tickers not covered",
            "pub_1d": "sqrt(252 * r_pub(fwe)^2)",
            "pub_5d/22d": "sqrt(252/w * trailing-w-valid-day sum of r_pub^2 <= fwe) — "
                          "alignment.py _backward_realised_vol convention (rangebased "
                          "TickerSeries.tail_sum, CRSP-verified < 1e-8)",
            "rs_neg/pos_pub": "sqrt(252 * r^2 * 1[r<0 / r>0]) at fwe — "
                              "stronger_baselines.build_return_features decomposition "
                              "rebuilt from public adjusted closes",
        },
    }
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"[done] {time.time() - t0:.0f}s -> {OUT_PARQUET} ({len(out):,} rows) + meta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
