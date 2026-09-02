#!/usr/bin/env python3
"""R3.1 -- how much of the RV label is estimator noise, and what does that do to
the reported MDE?

Reviewer R3.1: "h=5 RV is built from five daily returns; its sampling error is
large. The power calibration reports the MDE GIVEN this label; it cannot say
whether label measurement noise already drowns a real text signal."

We cannot answer with intraday data (none is licensed here). We can price the
INTRADAY half of the label, because two nearly-unbiased estimators of the same
intraday integrated variance are available from the same daily bars:

  OC  open-to-close realised variance
  GK  Garman--Klass range estimator (~7x more efficient)

Write log OC = m + u and log GK = m + v. Their errors are NOT independent --
both read one price path -- but the dependence is positive, which makes
Cov(log OC, log GK) overstate Var(m); the simulated bias probe below measures
Corr(u, v) = +0.26 and shows the identity returning 0.454 where the truth is
0.507. So

    noise share = 1 - Cov(log OC, log GK)/Var(log OC)

is a LOWER BOUND on the noise in the intraday component.

Scope, stated plainly: the paper's label is CLOSE-to-close, and roughly 55% of
its variance is the overnight return, which GK does not see. Scoring the label
directly against GK would charge every shift in the overnight/intraday mix to
"noise" and inflate the answer, so we do not do it (the mis-targeted number is
carried in the CSV as noise_share_cc_mismatched for comparison only). The
overnight half is estimated from a single return per day -- the noisiest
estimator available -- and this identification cannot price it, so the label's
total noise share is not pinned down here; what is pinned down is that its
intraday half is at least ~55% noise at h=5.

CPU-only; reads the frozen OHLCV panel and the committed MDE table.
Usage: python3 scripts/analysis/label_noise_budget.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
import os
# Set SP500VOL_DATA_ROOT to your regenerated data root; the default is the
# repo-relative layout the release package documents.
DATA = pathlib.Path(os.environ.get("SP500VOL_DATA_ROOT",
                                   str(ROOT / "data")))
OHLCV = DATA / "market/full_ohlcv.parquet"
RETURNS = DATA / "processed/full/market_returns.parquet"   # split-adjusted
OUT_CSV = ROOT / "results/tables/label_noise_budget.csv"
OUT_MD = ROOT / "results/tables/label_noise_budget.md"
HORIZONS = (5, 10, 20)
TEST_START = "2022-01-01"


def load() -> pd.DataFrame:
    d = pd.read_parquet(OHLCV, columns=["ticker", "date", "high", "low", "close", "open"])
    d = d[d["date"] >= "2021-01-01"].copy()          # test era + one year of lead-in
    d = d.sort_values(["ticker", "date"])
    d = d[(d[["high", "low", "close", "open"]] > 0).all(axis=1)]
    # The OHLCV `close` column is NOT split-adjusted (and `adj_close` is a copy
    # of it), so differencing raw closes manufactures a fake return of up to
    # 3.9 log points on every split date. Take the cross-day return from the
    # split-adjusted series instead; the same-day OC and GK terms are immune
    # because all four prices share one day's scale.
    g = d.groupby("ticker", sort=False)
    r = pd.read_parquet(RETURNS)
    r["date"] = pd.to_datetime(r["date"])
    d = d.merge(r.rename(columns={"log_return": "r"}), on=["ticker", "date"], how="left")
    d = d.sort_values(["ticker", "date"])
    g = d.groupby("ticker", sort=False)
    # Garman--Klass daily variance: 0.5*(ln H/L)^2 - (2 ln2 - 1)*(ln C/O)^2
    hl = np.log(d["high"] / d["low"])
    co = np.log(d["close"] / d["open"])
    d["gk"] = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    d["cc"] = d["r"] ** 2          # close-to-close: the label's own target
    d["oc"] = co**2                # open-to-close: what GK actually prices
    # overnight = adjusted close-to-close minus same-day open-to-close, which
    # keeps it on the split-adjusted scale
    d["on"] = (d["r"] - co) ** 2
    return d.dropna(subset=["r"])


def windows(d: pd.DataFrame, h: int) -> pd.DataFrame:
    """Non-overlapping forward windows of h trading days, per ticker."""
    out = []
    for tic, sub in d.groupby("ticker", sort=False):
        sub = sub.reset_index(drop=True)
        n = len(sub) // h * h
        if n < h:
            continue
        blk = np.arange(n) // h
        s = sub.iloc[:n].assign(blk=blk)
        agg = s.groupby("blk").agg(cc=("cc", "sum"), gk=("gk", "sum"),
                                   oc=("oc", "sum"), on=("on", "sum"),
                                   date=("date", "first"), k=("cc", "size"))
        agg = agg[agg["k"] == h]
        agg["ticker"] = tic
        out.append(agg)
    w = pd.concat(out, ignore_index=True)
    w = w[(w["cc"] > 0) & (w["gk"] > 0) & (w["oc"] > 0) & (w["on"] > 0)]
    return w[w["date"] >= TEST_START]


def budget(w: pd.DataFrame, target: str = "oc") -> dict:
    """Noise share of `target` identified against GK. Default is open-to-close,
    which is the quantity GK actually prices; passing "cc" scores the label
    itself against an estimator blind to its overnight half and is reported
    only to show how much that mismatch moves the answer."""
    lcc, lgk = np.log(w[target].to_numpy()), np.log(w["gk"].to_numpy())
    # de-mean by ticker so the decomposition is about within-firm variation over
    # time, which is what the forecasts are asked to track
    df = pd.DataFrame({"t": w["ticker"].to_numpy(), "lcc": lcc, "lgk": lgk})
    df["lcc"] -= df.groupby("t")["lcc"].transform("mean")
    df["lgk"] -= df.groupby("t")["lgk"].transform("mean")
    v_cc = df["lcc"].var(ddof=1)
    cov = df[["lcc", "lgk"]].cov().iloc[0, 1]
    signal = max(cov, 0.0)
    noise = max(v_cc - signal, 0.0)
    share = noise / v_cc if v_cc > 0 else np.nan
    return dict(n_windows=len(df), var_log_cc=v_cc, cov_cc_gk=cov,
                var_signal=signal, var_noise=noise, noise_share=share,
                r2_ceiling=1 - share,
                # a true proportional increment in the loss is attenuated by the
                # signal share when scored against the noisy label
                attenuation=signal / v_cc if v_cc > 0 else np.nan,
                theory_var_noise_2_over_h=np.nan)


def bias_probe(h: int = 5, n_win: int = 20000, steps: int = 78, seed: int = 7) -> dict:
    """Simulate a diffusion with KNOWN latent variance and measure how far the
    Cov identity lands from the truth. Both estimators read one price path, so
    their errors co-move; this quantifies the resulting bias and its sign."""
    rng = np.random.default_rng(seed)
    sig2 = np.exp(rng.normal(np.log(4e-4), 0.7, n_win))
    lcc, lgk, lm = [], [], []
    for s2 in sig2:
        path = np.cumsum(rng.normal(0, np.sqrt(s2 / steps), (h, steps)), axis=1)
        c = path[:, -1]
        hi, lo = path.max(axis=1), path.min(axis=1)
        cc = float((c ** 2).sum())
        gk = float((0.5 * (hi - lo) ** 2 - (2 * np.log(2) - 1) * c ** 2).sum())
        if cc <= 0 or gk <= 0:
            continue
        lcc.append(np.log(cc)); lgk.append(np.log(gk)); lm.append(np.log(s2 * h))
    lcc, lgk, lm = map(np.asarray, (lcc, lgk, lm))
    u, v = lcc - lm, lgk - lm
    true_share = u.var(ddof=1) / lcc.var(ddof=1)
    est_share = 1 - np.cov(lcc, lgk)[0, 1] / lcc.var(ddof=1)
    return dict(corr_uv=float(np.corrcoef(u, v)[0, 1]),
                true_share=float(true_share), est_share=float(est_share))


def main() -> None:
    d = load()
    rows = []
    for h in HORIZONS:
        w = windows(d, h)
        b = budget(w, "oc")
        b["h"] = h
        b["noise_share_cc_mismatched"] = budget(w, "cc")["noise_share"]
        b["overnight_share_of_label"] = float(
            w["on"].sum() / (w["on"].sum() + w["oc"].sum()))
        b["theory_var_noise_2_over_h"] = 2.0 / h
        rows.append(b)
        print(f"h={h:2d}  windows={b['n_windows']:6d}  noise share (open-to-close, "
              f"target-matched)={b['noise_share']:.3f}  R2 ceiling={b['r2_ceiling']:.3f}  "
              f"| overnight is {b['overnight_share_of_label']:.1%} of the label")

    out = pd.DataFrame(rows)[["h", "n_windows", "var_log_cc", "cov_cc_gk",
                              "var_signal", "var_noise", "noise_share",
                              "r2_ceiling", "attenuation",
                              "noise_share_cc_mismatched",
                              "overnight_share_of_label",
                              "theory_var_noise_2_over_h"]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    bp = bias_probe()
    print(f"\nbias probe (simulated diffusion, known latent variance): "
          f"Corr(u,v)={bp['corr_uv']:+.3f}, identity returns {bp['est_share']:.3f} "
          f"where the truth is {bp['true_share']:.3f} -> the tabulated shares are "
          f"LOWER BOUNDS")

    mde = pd.read_csv(ROOT / "results/tables/signal_injection_power.csv")
    med = mde.groupby("h")["mde_rel_pct"].median()

    L = ["# Label-noise budget for the close-to-close RV label", "",
         "Two nearly-unbiased estimators of the same latent integrated variance",
         "over the same window identify the label's estimator noise without any",
         "high-frequency data: close-to-close (the label) and Garman--Klass",
         "(same daily bars, disjoint information). Within-firm, test era",
         f"(windows starting {TEST_START} onward), non-overlapping windows.", "",
         "| h | windows | Var(log CC) | signal | noise | noise share | R2 ceiling | median MDE (%) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in out.itertuples():
        L.append(f"| {r.h} | {r.n_windows:,} | {r.var_log_cc:.3f} | {r.var_signal:.3f} | "
                 f"{r.var_noise:.3f} | {r.noise_share:.1%} | {r.r2_ceiling:.3f} | "
                 f"{med.get(r.h, float('nan')):.2f} |")
    L += ["",
          "`noise share` = 1 - Cov(log CC, log GK)/Var(log CC). `R2 ceiling` is the",
          "highest R^2 any forecast can attain against this label even with perfect",
          "knowledge of the latent variance. `attenuation` (in the CSV) is the factor",
          "by which a true proportional loss reduction shrinks when scored against",
          "the noisy label: an increment of size delta in true-RV units is measured",
          "as roughly `attenuation` x delta here.", "",
          "## Scope and direction of the bias", "",
          "Rows price the INTRADAY component (open-to-close vs GK, target-matched).",
          f"Roughly {out['overnight_share_of_label'].mean():.0%} of the close-to-close label's variance is the",
          "overnight return, which GK cannot see; scoring the label directly against",
          "GK charges mix shifts to noise and inflates the share (carried as",
          "`noise_share_cc_mismatched`). The overnight half is estimated from one",
          "return per day and is not identified here.", "",
          "The identity also assumes the two errors are uncorrelated. They",
          "are not: both read the same price path, so a large intraday move",
          "inflates both. A simulated diffusion with known latent variance gives",
          f"Corr(u, v) = {bp['corr_uv']:+.2f} at h=5, under which this identity returns a",
          f"{bp['est_share']:.3f} noise share where the truth is {bp['true_share']:.3f}. A positive",
          "Cov(u, v) makes the covariance overstate the signal, so the true noise",
          "share of the intraday component is **at least** what is tabulated.",
          "for a paper reporting a null. `bias_probe()` in this script reproduces",
          "the simulation.", ""]
    OUT_MD.write_text("\n".join(L))
    print(f"\nwrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
