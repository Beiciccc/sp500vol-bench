"""PREREG D (prereg-cd-v1.0, configs/prereg_mechanism_and_labels.md §D) — TASK 1.

Range-based (Parkinson / Garman-Klass) forward LABELS over the EXACT label windows the
committed pipeline uses (src/sp500vol/data/alignment.py), plus the A-block past-RV
FEATURES (rv_1d / rv_5d / rv_22d) recomputed in the same estimator over the SAME
feature windows (mirroring alignment.py `_return_at` / `_backward_realised_vol`:
features end strictly before the filing; trailing-window of VALID own-estimator days).

Estimators (per trading day i; prereg: "Parkinson is primary, GK in the same table"):
  Parkinson      pk_i = ln(H_i/L_i)^2 / (4 ln 2)
  Garman-Klass   gk_i = 0.5 ln(H_i/L_i)^2 - (2 ln 2 - 1) ln(C_i/O_i)^2   (standard form;
                 can be negative on rare days -> window sums clipped at 0, counts disclosed)
Annualisation matches src/sp500vol/features/volatility.py exactly:
  label  = sqrt(252/H * sum_{window} est_i)      (vs sqrt(252/H * sum r_i^2))
  rv_1d  = sqrt(252 * est_at_fwe)                (vs sqrt(252) * |r_at_fwe|)
  rv_5d  = sqrt(252/5  * sum_{last 5 valid days <= fwe} est_i)
  rv_22d = sqrt(252/22 * sum_{last 22 valid days <= fwe} est_i)

WINDOW-MACHINERY VERIFICATION (hard gate): the ORIGINAL close-to-close label and the
ORIGINAL rv_5d/rv_22d features are recomputed from market_returns.parquet through THIS
script's window extraction and must reproduce the aligned panel's stored
label_realised_vol / feature_rv_5d / feature_rv_22d to <1e-8. This proves the label and
feature windows used here are the SAME windows alignment.py used.

GATES:
  G2  Spearman rank correlation new-vs-old labels per horizon: Parkinson must be > 0.8
      (HARD ABORT below, per prereg branch (d) gate); GK reported (warn if <= 0.8).
  G3  leakage asserts: feature_window_end < label_window_start for every row;
      label windows strictly after the effective trading day; feature days <= fwe.
  Coverage accounting: rows lost vs the current panel reported EXACTLY (verified OHLC
      window coverage is 100%, so losses must be ~0).

Output (full mode): $SP500VOL_DATA_ROOT/processed/full/rangebased_labels.parquet
  keyed accession x horizon_days: [accession, ticker, horizon_days,
   label_pk, label_gk, pk_1d, pk_5d, pk_22d, gk_1d, gk_5d, gk_22d, label_gk_clipped]
  + rangebased_labels_meta.json next to it (gates, coverage, formulas).

Smoke mode (--smoke N): random N filings (seed 2026), runs every check, writes NOTHING.

Run from repo root:
  RB_THREADS=5 .venv/bin/python scripts/analysis/rangebased_labels.py [--smoke 2000]
"""
from __future__ import annotations

import os

_THREADS = os.environ.get("RB_THREADS", "5")  # env caps BEFORE numpy import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

DATA_ROOT = Path(os.environ.get("SP500VOL_DATA_ROOT", "/path/to/data-root/sp500vol-data"))
ALIGNED = DATA_ROOT / "processed" / "full" / "aligned_filings.parquet"
OHLCV = DATA_ROOT / "market" / "full_ohlcv.parquet"
RETURNS = DATA_ROOT / "market" / "crsp" / "market_returns.parquet"
OUT_PARQUET = DATA_ROOT / "processed" / "full" / "rangebased_labels.parquet"
OUT_META = DATA_ROOT / "processed" / "full" / "rangebased_labels_meta.json"

ANN = 252.0
LN2_4 = 4.0 * np.log(2.0)
GK_B = 2.0 * np.log(2.0) - 1.0
VERIFY_TOL = 1e-8
G2_MIN = 0.8
SMOKE_SEED = 2026


def _dt(x) -> np.ndarray:
    return pd.to_datetime(pd.Series(np.asarray(x))).to_numpy(dtype="datetime64[ns]")


class TickerSeries:
    """Per-ticker daily estimator series with O(1) window sums + valid-day tails."""

    def __init__(self, dates: np.ndarray, values: np.ndarray):
        self.dates = dates                      # sorted datetime64[ns]
        self.values = values                    # float64, may contain NaN
        self.cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
        self.cn = np.concatenate([[0.0], np.cumsum(np.isnan(values).astype(float))])
        vmask = ~np.isnan(values)
        self.vdates = dates[vmask]              # valid days only
        self.vcs = np.concatenate([[0.0], np.cumsum(values[vmask])])

    def window_sum(self, start: np.ndarray, end: np.ndarray):
        """Sum over ticker days in [start, end]; returns (sum, n_days, n_nan)."""
        i0 = np.searchsorted(self.dates, start, side="left")
        i1 = np.searchsorted(self.dates, end, side="right")
        return self.cs[i1] - self.cs[i0], (i1 - i0), self.cn[i1] - self.cn[i0]

    def at_day(self, day: np.ndarray):
        """Value exactly AT `day` (NaN when the day is absent) — mirrors _return_at."""
        idx = np.searchsorted(self.dates, day, side="left")
        idx_c = np.clip(idx, 0, len(self.dates) - 1)
        hit = (len(self.dates) > 0) & (self.dates[idx_c] == day) & (idx < len(self.dates))
        out = np.where(hit, self.values[idx_c], np.nan)
        return out

    def tail_sum(self, end: np.ndarray, window: int):
        """Sum of the trailing `window` VALID days ending at the last valid day <= end
        (NaN when fewer than `window` valid days) — mirrors _backward_realised_vol."""
        pos = np.searchsorted(self.vdates, end, side="right")  # count of valid days <= end
        ok = pos >= window
        lo = np.clip(pos - window, 0, None)
        s = self.vcs[pos] - self.vcs[lo]
        return np.where(ok, s, np.nan)


def build_series(df: pd.DataFrame, value_col: str) -> dict[str, TickerSeries]:
    out = {}
    for t, g in df.groupby("ticker", sort=False):
        out[str(t)] = TickerSeries(_dt(g["date"].to_numpy()),
                                   g[value_col].to_numpy(dtype=float))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="run on a random subsample of N filings; write NOTHING")
    args = ap.parse_args()
    t0 = time.time()

    al = pd.read_parquet(ALIGNED, columns=[
        "accession", "ticker", "horizon_days", "form", "effective_trading_day",
        "label_window_start", "label_window_end", "feature_window_end",
        "label_realised_vol", "feature_return_1d", "feature_rv_5d", "feature_rv_22d"])
    n_panel_total = len(al)
    if args.smoke:
        acc = al["accession"].drop_duplicates()
        keep = acc.sample(n=min(args.smoke, len(acc)), random_state=SMOKE_SEED)
        al = al[al["accession"].isin(set(keep))].reset_index(drop=True)
        print(f"[smoke] {len(al)} rows from {len(keep)} filings (of {n_panel_total} rows)")

    # ---------------- G3 leakage asserts on the panel geometry ----------------
    fwe = _dt(al["feature_window_end"].to_numpy())
    lws = _dt(al["label_window_start"].to_numpy())
    lwe = _dt(al["label_window_end"].to_numpy())
    etd = _dt(al["effective_trading_day"].to_numpy())
    have_fwe = ~pd.isna(al["feature_window_end"]).to_numpy()
    assert (lws[have_fwe] > fwe[have_fwe]).all(), \
        "G3 FAIL: feature_window_end >= label_window_start"
    assert (lws > etd).all(), "G3 FAIL: label window not strictly after effective day"
    assert (lwe >= lws).all(), "G3 FAIL: inverted label window"
    print(f"[G3] leakage asserts PASS on {len(al)} rows "
          f"({int((~have_fwe).sum())} rows have NaT feature_window_end)")

    # ---------------- daily estimators from OHLC ----------------
    o = pd.read_parquet(OHLCV, columns=["ticker", "date", "open", "high", "low", "close"])
    o = o.sort_values(["ticker", "date"], kind="mergesort")
    valid_hl = (o["high"] > 0) & (o["low"] > 0) & (o["high"] >= o["low"])
    valid_oc = (o["open"] > 0) & (o["close"] > 0)
    n_bad_hl = int((~valid_hl.fillna(False)).sum())
    n_bad_oc = int((~valid_oc.fillna(False)).sum())
    lhl = np.where(valid_hl, np.log(o["high"] / o["low"]), np.nan)
    lco = np.where(valid_hl & valid_oc, np.log(o["close"] / o["open"]), np.nan)
    o["pk"] = lhl ** 2 / LN2_4
    o["gk"] = 0.5 * lhl ** 2 - GK_B * lco ** 2
    n_gk_negative_days = int((o["gk"] < 0).sum())
    print(f"[est] OHLC rows={len(o)}  invalid H/L rows={n_bad_hl}  invalid O/C rows={n_bad_oc}  "
          f"negative GK days={n_gk_negative_days} ({100 * n_gk_negative_days / len(o):.3f}%)")

    cal = np.unique(_dt(o["date"].to_numpy()))  # exchange calendar = union of ticker days
    pk_s = build_series(o, "pk")
    gk_s = build_series(o, "gk")

    r = pd.read_parquet(RETURNS)  # ticker, date, log_return
    r = r.sort_values(["ticker", "date"], kind="mergesort")
    r["r2"] = r["log_return"].astype(float) ** 2
    r2_s = build_series(r, "r2")
    r_s = build_series(r, "log_return")
    print(f"[load] series built in {time.time() - t0:.1f}s")

    # ---------------- per-row computation, grouped by ticker ----------------
    H = al["horizon_days"].to_numpy(dtype=float)
    n = len(al)
    cols = {k: np.full(n, np.nan) for k in (
        "label_pk", "label_gk", "pk_1d", "pk_5d", "pk_22d", "gk_1d", "gk_5d", "gk_22d",
        "rv_label_check", "rv_5d_check", "rv_22d_check", "r1_check")}
    flags = {k: np.zeros(n, dtype=bool) for k in (
        "win_ok_cal", "win_ok_tick", "pk_win_full", "gk_win_full", "label_gk_clipped",
        "gk_1d_clipped", "gk_5d_clipped", "gk_22d_clipped", "rv_win_full")}

    g0 = np.searchsorted(cal, lws, side="left")
    g1 = np.searchsorted(cal, lwe, side="right")
    flags["win_ok_cal"] = (g1 - g0) == H.astype(int)

    for ticker, idx in al.groupby("ticker", sort=False).indices.items():
        t = str(ticker)
        s, e, f = lws[idx], lwe[idx], fwe[idx]
        h = H[idx]
        if t in pk_s:
            ts = pk_s[t]
            sm, nd, nn = ts.window_sum(s, e)
            flags["win_ok_tick"][idx] = nd == h
            full = (nd == h) & (nn == 0)
            flags["pk_win_full"][idx] = full
            cols["label_pk"][idx] = np.where(full, np.sqrt(ANN / h * sm), np.nan)
            cols["pk_1d"][idx] = np.sqrt(ANN * ts.at_day(f))
            s5 = ts.tail_sum(f, 5)
            s22 = ts.tail_sum(f, 22)
            cols["pk_5d"][idx] = np.sqrt(ANN / 5.0 * s5)
            cols["pk_22d"][idx] = np.sqrt(ANN / 22.0 * s22)
        if t in gk_s:
            ts = gk_s[t]
            sm, nd, nn = ts.window_sum(s, e)
            full = (nd == h) & (nn == 0)
            flags["gk_win_full"][idx] = full
            clipped = full & (sm < 0)
            flags["label_gk_clipped"][idx] = clipped
            cols["label_gk"][idx] = np.where(full, np.sqrt(ANN / h * np.clip(sm, 0, None)),
                                             np.nan)
            g1v = ts.at_day(f)
            flags["gk_1d_clipped"][idx] = g1v < 0
            cols["gk_1d"][idx] = np.sqrt(ANN * np.clip(g1v, 0, None))
            s5 = ts.tail_sum(f, 5)
            s22 = ts.tail_sum(f, 22)
            flags["gk_5d_clipped"][idx] = s5 < 0
            flags["gk_22d_clipped"][idx] = s22 < 0
            cols["gk_5d"][idx] = np.sqrt(ANN / 5.0 * np.clip(s5, 0, None))
            cols["gk_22d"][idx] = np.sqrt(ANN / 22.0 * np.clip(s22, 0, None))
        if t in r2_s:
            ts = r2_s[t]
            sm, nd, nn = ts.window_sum(s, e)
            full = (nd == h) & (nn == 0)
            flags["rv_win_full"][idx] = full
            cols["rv_label_check"][idx] = np.where(full, np.sqrt(ANN / h * sm), np.nan)
            cols["rv_5d_check"][idx] = np.sqrt(ANN / 5.0 * ts.tail_sum(f, 5))
            cols["rv_22d_check"][idx] = np.sqrt(ANN / 22.0 * ts.tail_sum(f, 22))
            cols["r1_check"][idx] = r_s[t].at_day(f)
    print(f"[compute] per-row done in {time.time() - t0:.1f}s")

    # ---------------- WINDOW-MACHINERY VERIFICATION (hard gates) ----------------
    y_old = al["label_realised_vol"].to_numpy(dtype=float)
    chk = cols["rv_label_check"]
    m = np.isfinite(chk)
    lab_diff = float(np.nanmax(np.abs(chk[m] - y_old[m]))) if m.any() else np.nan
    n_lab_unverifiable = int((~m).sum())
    f5_old = al["feature_rv_5d"].to_numpy(dtype=float)
    f22_old = al["feature_rv_22d"].to_numpy(dtype=float)
    r1_old = al["feature_return_1d"].to_numpy(dtype=float)
    m5 = np.isfinite(cols["rv_5d_check"]) & np.isfinite(f5_old)
    m22 = np.isfinite(cols["rv_22d_check"]) & np.isfinite(f22_old)
    m1 = np.isfinite(cols["r1_check"]) & np.isfinite(r1_old)
    f5_diff = float(np.max(np.abs(cols["rv_5d_check"][m5] - f5_old[m5])))
    f22_diff = float(np.max(np.abs(cols["rv_22d_check"][m22] - f22_old[m22])))
    r1_diff = float(np.max(np.abs(cols["r1_check"][m1] - r1_old[m1])))
    nan_mismatch_5 = int((np.isfinite(cols["rv_5d_check"]) != np.isfinite(f5_old)).sum())
    nan_mismatch_22 = int((np.isfinite(cols["rv_22d_check"]) != np.isfinite(f22_old)).sum())
    print(f"[verify] label windows: max|recomputed RV - stored label| = {lab_diff:.3e} "
          f"on {int(m.sum())}/{n} rows ({n_lab_unverifiable} unverifiable)")
    print(f"[verify] feature windows: max|drv5|={f5_diff:.3e} (NaN-pattern mismatch "
          f"{nan_mismatch_5}), max|drv22|={f22_diff:.3e} (mismatch {nan_mismatch_22}), "
          f"max|dr1|={r1_diff:.3e}")
    verify_pass = (lab_diff < VERIFY_TOL and f5_diff < VERIFY_TOL
                   and f22_diff < VERIFY_TOL and r1_diff < VERIFY_TOL
                   and n_lab_unverifiable == 0)
    if not verify_pass:
        print("WINDOW-MACHINERY VERIFICATION FAILED — the windows here do NOT reproduce "
              "the committed labels/features. ABORTING (nothing written).")
        return 2

    # ---------------- coverage accounting ----------------
    feat_ok_old = np.isfinite(r1_old) & np.isfinite(f5_old) & np.isfinite(f22_old)
    feat_ok_pk = (np.isfinite(cols["pk_1d"]) & np.isfinite(cols["pk_5d"])
                  & np.isfinite(cols["pk_22d"]))
    feat_ok_gk = (np.isfinite(cols["gk_1d"]) & np.isfinite(cols["gk_5d"])
                  & np.isfinite(cols["gk_22d"]))
    cov = {
        "panel_rows": n,
        "label_pk_ok": int(np.isfinite(cols["label_pk"]).sum()),
        "label_pk_lost": int((~np.isfinite(cols["label_pk"])).sum()),
        "label_gk_ok": int(np.isfinite(cols["label_gk"]).sum()),
        "label_gk_lost": int((~np.isfinite(cols["label_gk"])).sum()),
        "label_gk_clipped_to_zero": int(flags["label_gk_clipped"].sum()),
        "feat_ok_old": int(feat_ok_old.sum()),
        "feat_ok_pk": int(feat_ok_pk.sum()),
        "feat_ok_gk": int(feat_ok_gk.sum()),
        "rows_usable_old(label+feat)": int(feat_ok_old.sum()),  # old label always present
        "rows_usable_pk(label+feat)": int((np.isfinite(cols["label_pk"]) & feat_ok_pk).sum()),
        "rows_usable_gk(label+feat)": int((np.isfinite(cols["label_gk"]) & feat_ok_gk).sum()),
        "gk_feature_clip_counts": {k: int(flags[k].sum()) for k in
                                   ("gk_1d_clipped", "gk_5d_clipped", "gk_22d_clipped")},
        "calendar_window_len_mismatch": int((~flags["win_ok_cal"]).sum()),
        "ticker_window_len_mismatch": int((~flags["win_ok_tick"]).sum()),
        "invalid_HL_ohlc_rows": n_bad_hl, "invalid_OC_ohlc_rows": n_bad_oc,
        "negative_gk_days_pct": round(100 * n_gk_negative_days / len(o), 4),
    }
    print("[coverage] " + json.dumps(cov, indent=2))

    # ---------------- G2 rank-correlation gate (per horizon) ----------------
    g2 = {}
    for h in (5, 10, 20):
        hm = (al["horizon_days"] == h).to_numpy()
        mp = hm & np.isfinite(cols["label_pk"])
        mg = hm & np.isfinite(cols["label_gk"])
        rho_p = float(stats.spearmanr(cols["label_pk"][mp], y_old[mp]).statistic)
        rho_g = float(stats.spearmanr(cols["label_gk"][mg], y_old[mg]).statistic)
        g2[f"h{h}"] = {"spearman_pk_vs_old": round(rho_p, 6),
                       "spearman_gk_vs_old": round(rho_g, 6),
                       "n_pk": int(mp.sum()), "n_gk": int(mg.sum())}
        print(f"[G2] h={h}: Spearman(PK, old)={rho_p:.4f}  Spearman(GK, old)={rho_g:.4f}")
    pk_pass = all(v["spearman_pk_vs_old"] > G2_MIN for v in g2.values())
    gk_pass = all(v["spearman_gk_vs_old"] > G2_MIN for v in g2.values())
    if not pk_pass:
        print(f"G2 FAIL: Parkinson-vs-old rank correlation <= {G2_MIN} for some horizon — "
              "prereg branch (d) gate: ABORT AND DEBUG LABEL CONSTRUCTION. Nothing written.")
        return 3
    if not gk_pass:
        print(f"G2 WARNING: GK rank correlation <= {G2_MIN} for some horizon — GK is the "
              "same-table secondary; value disclosed, cascade proceeds on Parkinson.")

    meta = {
        "prereg": "configs/prereg_mechanism_and_labels.md §D (prereg-cd-v1.0)",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "smoke": bool(args.smoke), "smoke_n_filings": int(args.smoke),
        "data_root": str(DATA_ROOT),
        "estimators": {
            "parkinson": "pk_i = ln(H/L)^2 / (4 ln 2); label = sqrt(252/H * sum pk_i)",
            "garman_klass": "gk_i = 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2; window sums "
                            "clipped at 0 (counts in coverage)",
            "annualisation": "sqrt(252/H * sum) — identical to volatility.py convention",
            "features": "rv_1d=sqrt(252*est@fwe); rv_5d/22d=sqrt(252/w * trailing-w-valid-"
                        "day sum <= fwe) — mirrors alignment.py _return_at / "
                        "_backward_realised_vol",
        },
        "window_verification": {
            "max_abs_diff_label": lab_diff, "max_abs_diff_rv5": f5_diff,
            "max_abs_diff_rv22": f22_diff, "max_abs_diff_r1": r1_diff,
            "nan_pattern_mismatch_rv5": nan_mismatch_5,
            "nan_pattern_mismatch_rv22": nan_mismatch_22,
            "tolerance": VERIFY_TOL, "pass": True,
        },
        "G2": g2, "G2_pk_pass": pk_pass, "G2_gk_pass": gk_pass,
        "G3_leakage_asserts": "PASS (fwe < label_window_start; label window strictly "
                              "after effective day; feature days <= fwe by construction)",
        "coverage": cov,
    }

    if args.smoke:
        print("[smoke] all gates evaluated; NOTHING written.")
        print(json.dumps({"G2": g2, "coverage_usable_pk": cov["rows_usable_pk(label+feat)"],
                          "panel_rows": n}, indent=2))
        return 0

    out = pd.DataFrame({
        "accession": al["accession"].to_numpy(),
        "ticker": al["ticker"].to_numpy(),
        "horizon_days": al["horizon_days"].to_numpy(),
        "label_pk": cols["label_pk"], "label_gk": cols["label_gk"],
        "pk_1d": cols["pk_1d"], "pk_5d": cols["pk_5d"], "pk_22d": cols["pk_22d"],
        "gk_1d": cols["gk_1d"], "gk_5d": cols["gk_5d"], "gk_22d": cols["gk_22d"],
        "label_gk_clipped": flags["label_gk_clipped"],
    })
    assert not out.duplicated(["accession", "horizon_days"]).any(), \
        "output must be uniquely keyed accession x horizon"
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {OUT_PARQUET} ({len(out)} rows) + meta in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
