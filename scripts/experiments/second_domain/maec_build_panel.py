#!/usr/bin/env python
"""MAEC audit — canonical panel builder (prereg configs/prereg_maec_audit.md,
tag prereg-maec-v1.0, FROZEN; this script implements §2-§4 + gates G3/G5/G6).

Builds the (permno, call_date, horizon) label/feature panel for the MAEC
earnings-call volatility audit, BOTH alignments in one file:

  alignment='primary'  label window {t_1..t_n}   (strict post-call, §2.3 PRIMARY)
  alignment='shifted'  label window {t_0..t_n-1} (day-0 inclusive sensitivity)
  past-vol windows end at t_0 (primary) / t_0-1 (shifted), §2.4.

Labels (§2.1, the audited literature's own Eq.-1 convention):
    v = ln( sqrt( (1/n) * sum_{t in window} (r_t - rbar)^2 ) ),  r_t = CRSP DlyRet
(simple total return, dividend+split inclusive). Zero-cost robustness column
label_logret uses r_t = log1p(DlyRet). sigma clipped to [1e-4, 1.0] before the
log for labels AND past-vol features (clip counts disclosed).

Price sources (§1) and the RETURN-CONSTRUCTION DISCLOSURE:
  * 700 non-S&P500 tickers: crsp_sp1500_daily_2014_2019.parquet, column `ret`
    = CRSP DlyRet directly, keyed (permno, date).
  * 513 S&P500-side tickers: prereg §1 names full_ohlcv.parquet, but that cache
    has NO return column and its adj_close is the RAW DlyClose (verified:
    scripts/ingest_wrds.py::build_returns sets adj_close=df["DlyClose"]), so
    pct_change(adj_close) would be a MATERIALLY DIFFERENT construction (no
    dividends, no split adjustment). DlyRet is instead recovered EXACTLY from
    the sibling artifact of the same ingest build,
    /Volumes/Z/sp500vol-data/market/crsp/market_returns.parquet, whose
    log_return = log1p(DlyRet)  =>  DlyRet = expm1(log_return).
    Both sources therefore use the IDENTICAL §2.1 estimand; the report
    quantifies how wrong pct_change(adj_close) would have been.
  * S&P500-side GAP-FILL (§11 v1.2(2)): market_returns.parquet covers only
    index-membership windows, so ~307-311 S&P500-side calls fell to the §3.2
    price gates as a cache-structure artifact. crsp_sp500side_gapfill_2014_2019
    .parquet (CRSP DlyRet extracted from the full-universe raw zip, same
    mechanism as the 700-ticker extract) is merged keyed (permno, date) with
    precedence ONLY where market_returns has no row for that firm-day; on
    overlapping days the two DlyRet constructions are asserted equal (<=1e-10)
    and the ORIGINAL market_returns value is used. Rows whose label/past
    windows consumed >=1 gap-filled day carry price_source
    'sp500_cache+gapfill'.

Ticker->PERMNO resolution (§3.3 + S&P500-side rule, all counts disclosed):
  * 700-side: ticker_permno_map.parquet; 694 unambiguous; dual-class GEF/HVT/
    WSO resolved ONCE by higher median daily dollar volume |close|*volume over
    the 2014-2019 extraction window (both medians disclosed); in-window reuse
    ENR/FLOW/TIVO resolved point-in-time by call_date in [name_start, name_end];
    zero or double window -> call dropped and counted.
  * 513-side: data/universe/sp500_membership.parquet point-in-time interval
    containing call_date; else, if the ticker maps to exactly one permno
    ticker-wide, that permno (flagged+counted); multi-permno tickers
    (CB/IR/JCI/NWSA) additionally have their return series masked to the
    resolved membership interval (ticker-keyed prices switch firm identity).

Sanity gates implemented HERE (G1/G2/G4 belong to the scoring stage):
  G3 split asserts: pinned boundaries (§4: train <= 2017-02-23, val
     2017-02-24..2017-05-09, test 2017-05-10..); pre-exclusion counts MUST be
     2436/333/674; max(train)<min(val)<min(test); MIN_VAL=100/MIN_TEST=30 per
     horizon; STPEV point-in-time assert (every contributing label window ends
     <= the current call's date).
  G5 keys: exactly one permno per call after disambiguation; (permno,
     call_date, horizon) unique within each alignment.
  G6 exclusion accounting: 3,443 - stubs(32) - ambiguity drops - price-gate
     drops == final rows, per alignment x horizon, hard-asserted.

Outputs:
  /Volumes/Z/second-domain/earnings_calls/maec_panel.parquet   (both alignments)
  /Volumes/Z/second-domain/earnings_calls/maec_calendar.parquet (union grid,
      consumed by maec_protocol.py for the L_n HAC lags)
  results/second_domain/maec/build_report.json

Run from repo root (threads capped BEFORE numpy import):
    .venv/bin/python scripts/experiments/second_domain/maec_build_panel.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
EC_ROOT = Path("/Volumes/Z/second-domain/earnings_calls")
SP500_ROOT = Path("/Volumes/Z/sp500vol-data")

MANIFEST = EC_ROOT / "maec_manifest.parquet"
SP1500_PRICES = EC_ROOT / "crsp_sp1500_daily_2014_2019.parquet"
TICKER_MAP = EC_ROOT / "ticker_permno_map.parquet"
COVERAGE_CSV = EC_ROOT / "maec_price_coverage_by_ticker.csv"
GAPFILL = EC_ROOT / "crsp_sp500side_gapfill_2014_2019.parquet"
FULL_OHLCV = SP500_ROOT / "market" / "full_ohlcv.parquet"
MARKET_RETURNS = SP500_ROOT / "market" / "crsp" / "market_returns.parquet"
MEMBERSHIP = REPO / "data" / "universe" / "sp500_membership.parquet"
NAMES_ZIP = SP500_ROOT / "raw" / "wrds" / "crsp_names_csv.zip"

OUT_PANEL = EC_ROOT / "maec_panel.parquet"
OUT_CAL = EC_ROOT / "maec_calendar.parquet"
OUT_REPORT = REPO / "results" / "second_domain" / "maec" / "build_report.json"

HORIZONS = (3, 7, 15, 30)
ALIGNMENTS = ("primary", "shifted")
HAR_WINDOWS = (5, 22, 66)
STUB_CHARS = 100                     # §3.1 (OPEN-9 ruled: 100)
SIGMA_LO, SIGMA_HI = 1e-4, 1.0       # §5 clip range, applied to labels+features
TRAIN_END = pd.Timestamp("2017-02-23")   # §4 pinned
VAL_START = pd.Timestamp("2017-02-24")
VAL_END = pd.Timestamp("2017-05-09")
TEST_START = pd.Timestamp("2017-05-10")
PRE_COUNTS = {"train": 2436, "val": 333, "test": 674}  # §4, asserted
MIN_VAL, MIN_TEST = 100, 30
CAL_LO, CAL_HI = pd.Timestamp("2014-01-01"), pd.Timestamp("2019-06-30")
DUAL_CLASS = ("GEF", "HVT", "WSO")   # §3.3
REUSE = ("ENR", "FLOW", "TIVO")      # §3.3
PAST_PRESENCE = 0.80                 # §3.2


def split_of(d: pd.Timestamp) -> str:
    if d <= TRAIN_END:
        return "train"
    if d <= VAL_END:
        return "val"
    return "test"


def v_of(returns: np.ndarray) -> tuple[float, float]:
    """§2.1 estimand: (ln(demeaned RMS of daily returns) with sigma clipped,
    unclipped sigma)."""
    r = np.asarray(returns, float)
    sig = float(np.sqrt(np.mean((r - r.mean()) ** 2)))
    return float(np.log(min(max(sig, SIGMA_LO), SIGMA_HI))), sig


# --------------------------------------------------------------- input loading
def load_manifest() -> pd.DataFrame:
    mf = pd.read_parquet(MANIFEST)
    assert len(mf) == 3443, f"manifest rows {len(mf)} != 3443"
    mf["call_date"] = pd.to_datetime(mf["call_date"]).astype("datetime64[ns]")
    assert not mf.duplicated(["ticker", "call_date"]).any(), \
        "(ticker, call_date) duplicates in manifest"
    mf["split"] = mf["call_date"].map(split_of)
    counts = mf["split"].value_counts().to_dict()
    for k, v in PRE_COUNTS.items():
        assert counts[k] == v, f"G3 FAIL: pre-exclusion {k} count {counts[k]} != {v}"
    return mf


def load_returns_700() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    df = pd.read_parquet(
        SP1500_PRICES, columns=["ticker", "permno", "date", "close", "ret", "volume"])
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    assert not df.duplicated(["permno", "date"]).any(), "(permno,date) dup in sp1500"
    cal = pd.DatetimeIndex(np.sort(df["date"].unique()))
    return df, cal


def load_returns_513(tickers: set) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    mr = pd.read_parquet(MARKET_RETURNS)
    mr["date"] = pd.to_datetime(mr["date"]).astype("datetime64[ns]")
    mr = mr[(mr["date"] >= CAL_LO) & (mr["date"] <= CAL_HI)
            & mr["ticker"].isin(tickers)].copy()
    assert not mr.duplicated(["ticker", "date"]).any(), "(ticker,date) dup in market_returns"
    # EXACT DlyRet recovery: log_return = log1p(DlyRet) per ingest_wrds.build_returns
    mr["ret"] = np.expm1(mr["log_return"].astype(float))
    cal = pd.DatetimeIndex(np.sort(mr["date"].unique()))
    return mr[["ticker", "date", "ret"]], cal


def load_gapfill() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """§11 v1.2(2) third price source: CRSP DlyRet for the S&P500-side permnos
    whose membership-window cache left incomplete/missing price windows."""
    gf = pd.read_parquet(GAPFILL, columns=["ticker", "permno", "date", "ret"])
    gf["date"] = pd.to_datetime(gf["date"]).astype("datetime64[ns]")
    assert not gf.duplicated(["permno", "date"]).any(), \
        "conflicting duplicate (permno,date) rows in gapfill"
    cal = pd.DatetimeIndex(np.sort(gf["date"].unique()))
    return gf, cal


def verify_gapfill_overlap(by_ticker: dict, by_gap: dict, pairs: dict,
                           cal: pd.DatetimeIndex) -> dict:
    """Where BOTH sources have a row for the same firm-day, the DlyRet-derived
    returns must agree to <=1e-10 (checked on ALL joint days, a superset of
    any sample); the original market_returns value takes precedence. For
    multi-permno tickers the ticker series only identifies the firm inside the
    membership interval, so the comparison is restricted there."""
    n_pairs, n_joint, n_fill, max_diff = 0, 0, 0, 0.0
    for (tk, p), interval in sorted(pairs.items()):
        a, b = by_ticker.get(tk), by_gap.get(p)
        if b is None:
            continue
        if a is None:
            n_fill += int(np.isfinite(b).sum())
            continue
        a = a.copy()
        if interval is not None:   # list of membership intervals for this pair
            allowed = np.zeros(len(cal), bool)
            for lo, hi in interval:
                allowed |= (cal >= lo) & (cal <= hi)
            a[~allowed] = np.nan
        joint = np.isfinite(a) & np.isfinite(b)
        n_pairs += 1
        n_fill += int((~np.isfinite(a) & np.isfinite(b)).sum())
        if joint.any():
            n_joint += int(joint.sum())
            max_diff = max(max_diff, float(np.max(np.abs(a[joint] - b[joint]))))
    assert max_diff <= 1e-10, \
        f"gap-fill DlyRet conflicts with market_returns: max |diff|={max_diff}"
    return {"n_pairs_checked": n_pairs, "n_joint_days": n_joint,
            "max_absret_diff": max_diff, "tolerance": 1e-10,
            "n_gapfill_only_days": n_fill,
            "verdict": ("PASS — identical DlyRet on all joint firm-days; "
                        "original market_returns value used there")}


def adj_close_comparison(mr: pd.DataFrame) -> dict:
    """Quantify how different pct_change(adj_close) (= raw DlyClose) is from
    DlyRet on the 513-side, for the return-construction disclosure."""
    oh = pd.read_parquet(FULL_OHLCV, columns=["ticker", "date", "adj_close"])
    oh["date"] = pd.to_datetime(oh["date"])
    oh = oh[(oh["date"] >= CAL_LO) & (oh["date"] <= CAL_HI)
            & oh["ticker"].isin(set(mr["ticker"]))]
    oh = oh.sort_values(["ticker", "date"])
    oh["px_ret"] = oh.groupby("ticker")["adj_close"].pct_change(fill_method=None)
    m = mr.merge(oh[["ticker", "date", "px_ret"]], on=["ticker", "date"], how="left")
    m = m.dropna(subset=["px_ret"])
    diff = (m["px_ret"] - m["ret"]).abs()
    worst = m.loc[diff.idxmax()]
    return {
        "n_days_compared": len(m),
        "frac_absdiff_gt_1e-6": float((diff > 1e-6).mean()),
        "frac_absdiff_gt_1bp": float((diff > 1e-4).mean()),
        "frac_absdiff_gt_100bp": float((diff > 1e-2).mean()),
        "max_absdiff": float(diff.max()),
        "worst_example": {"ticker": str(worst["ticker"]),
                          "date": str(pd.Timestamp(worst["date"]).date()),
                          "dlyret": float(worst["ret"]),
                          "pct_change_adj_close": float(worst["px_ret"])},
        "verdict": ("MATERIALLY DIFFERENT construction (raw DlyClose: no dividends,"
                    " no split adjustment) — REJECTED; DlyRet recovered exactly from"
                    " market_returns.parquet instead"),
    }


def load_names_for(permnos: set) -> pd.DataFrame:
    """CRSP IssuerNm history (point-in-time) for the S&P500-side permnos."""
    with zipfile.ZipFile(NAMES_ZIP) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            nm = pd.read_csv(
                io.TextIOWrapper(fh, encoding="utf-8", errors="replace"),
                usecols=["PERMNO", "SecInfoStartDt", "SecInfoEndDt", "IssuerNm"])
    nm = nm[nm["PERMNO"].isin(permnos)].copy()
    nm["SecInfoStartDt"] = pd.to_datetime(nm["SecInfoStartDt"])
    nm["SecInfoEndDt"] = pd.to_datetime(nm["SecInfoEndDt"])
    return nm.rename(columns={"PERMNO": "permno", "IssuerNm": "name"})


# --------------------------------------------------------- ticker -> permno
def resolve_700(mf700: pd.DataFrame, tmap: pd.DataFrame, px700: pd.DataFrame,
                report: dict) -> tuple[pd.Series, set]:
    """Returns (call_id -> permno) for the 700-side + dropped call_ids (§3.3)."""
    permno_of, dropped = {}, set()
    plain = tmap[~tmap["ambiguous"]]
    assert plain.groupby("ticker")["permno"].nunique().max() == 1
    plain_map = plain.set_index("ticker")["permno"].to_dict()

    # dual-class: ONE choice per ticker by median daily dollar volume (§3.3, OPEN-11)
    dual_info = {}
    for tk in DUAL_CLASS:
        rows = tmap[(tmap["ticker"] == tk) & tmap["ambiguous"]]
        med = {}
        for p in rows["permno"]:
            g = px700[px700["permno"] == p]
            dv = (g["close"].abs() * g["volume"]).dropna()
            med[int(p)] = float(dv.median())
        chosen = max(med, key=med.get)
        cls = rows.set_index("permno")["share_class"].to_dict()
        dual_info[tk] = {"chosen_permno": chosen,
                         "chosen_share_class": str(cls.get(chosen)),
                         "median_dollar_volume_by_permno": med}
        plain_map[tk] = chosen
    report["dual_class_resolutions"] = dual_info

    # in-window ticker reuse: point-in-time by [name_start, name_end] (§3.3)
    reuse_info = {}
    reuse_rows = {tk: tmap[(tmap["ticker"] == tk) & tmap["ambiguous"]] for tk in REUSE}
    for _, r in mf700.iterrows():
        tk, cd, cid = r["ticker"], r["call_date"], r["call_id"]
        if tk in REUSE:
            rr = reuse_rows[tk]
            end = rr["name_end"].fillna(pd.Timestamp("2099-01-01"))
            hits = rr[(rr["name_start"] <= cd) & (cd <= end)]
            info = reuse_info.setdefault(tk, {"assigned": {}, "dropped_calls": []})
            if len(hits) == 1:
                p = int(hits["permno"].iloc[0])
                permno_of[cid] = p
                info["assigned"][str(p)] = info["assigned"].get(str(p), 0) + 1
            else:  # zero or double window -> drop + count (§3.3)
                dropped.add(cid)
                info["dropped_calls"].append(cid)
        else:
            permno_of[cid] = int(plain_map[tk])
    report["reuse_resolutions"] = reuse_info
    return pd.Series(permno_of), dropped


def resolve_513(mf513: pd.DataFrame, memb: pd.DataFrame,
                report: dict) -> tuple[pd.Series, dict]:
    """S&P500-side point-in-time resolution via membership intervals."""
    memb = memb.copy()
    memb["member_to_f"] = memb["member_to"].fillna(pd.Timestamp("2099-01-01"))
    nuni = memb.groupby("ticker")["permno"].nunique()
    multi = set(nuni[nuni > 1].index)
    permno_of, how = {}, {"in_window": 0, "unique_fallback": 0, "unresolved": 0}
    masks = {}  # call_id -> (lo, hi) usable-date interval for multi-permno tickers
    for _, r in mf513.iterrows():
        tk, cd, cid = r["ticker"], r["call_date"], r["call_id"]
        rows = memb[memb["ticker"] == tk]
        hit = rows[(rows["member_from"] <= cd) & (cd <= rows["member_to_f"])]
        if len(hit) >= 1:
            assert hit["permno"].nunique() == 1, f"{cid}: >1 permno in-window"
            permno_of[cid] = int(hit["permno"].iloc[0])
            how["in_window"] += 1
            if tk in multi:  # ticker-keyed prices switch firm identity: mask
                masks[cid] = (hit["member_from"].min(), hit["member_to_f"].max())
        elif rows["permno"].nunique() == 1:
            permno_of[cid] = int(rows["permno"].iloc[0])
            how["unique_fallback"] += 1
        else:
            how["unresolved"] += 1
    assert how["unresolved"] == 0, "unresolved sp500-side calls — extend the rule"
    report["sp500_side_resolution"] = {
        **how,
        "multi_permno_tickers_interval_masked": sorted(
            mf513.loc[mf513["call_id"].isin(masks), "ticker"].unique().tolist()),
        "n_calls_interval_masked": len(masks),
        "rule": ("membership interval containing call_date; else unique-permno "
                 "fallback (disclosed); multi-permno tickers masked to interval"),
    }
    return pd.Series(permno_of), masks


# ----------------------------------------------------------------- panel build
def build_series(px700: pd.DataFrame, mr513: pd.DataFrame, gf513: pd.DataFrame,
                 cal: pd.DatetimeIndex) -> tuple[dict, dict, dict]:
    """Aligned return arrays on the union calendar: 700-side keyed by permno,
    513-side keyed by ticker, §11 v1.2(2) gap-fill keyed by permno."""
    pos = pd.Series(np.arange(len(cal)), index=cal)
    by_permno, by_ticker, by_gap = {}, {}, {}
    for p, g in px700.dropna(subset=["ret"]).groupby("permno"):
        a = np.full(len(cal), np.nan)
        a[pos.loc[g["date"]].to_numpy()] = g["ret"].to_numpy(float)
        by_permno[int(p)] = a
    for t, g in mr513.groupby("ticker"):
        a = np.full(len(cal), np.nan)
        a[pos.loc[g["date"]].to_numpy()] = g["ret"].to_numpy(float)
        by_ticker[str(t)] = a
    for p, g in gf513.dropna(subset=["ret"]).groupby("permno"):
        a = np.full(len(cal), np.nan)
        a[pos.loc[g["date"]].to_numpy()] = g["ret"].to_numpy(float)
        by_gap[int(p)] = a
    return by_permno, by_ticker, by_gap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PANEL))
    ap.add_argument("--report", default=str(OUT_REPORT))
    args = ap.parse_args()
    t0 = time.time()
    report: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prereg": ("configs/prereg_maec_audit.md @ prereg-maec-v1.0"
                               " + §11 v1.2 build amendments")}

    mf = load_manifest()
    tmap = pd.read_parquet(TICKER_MAP)
    map700_tickers = set(tmap["ticker"])
    mf700 = mf[mf["ticker"].isin(map700_tickers)]
    mf513 = mf[~mf["ticker"].isin(map700_tickers)]
    report["calls"] = {"total": len(mf), "side_700": len(mf700), "side_513": len(mf513)}

    px700, cal700 = load_returns_700()
    mr513, cal513 = load_returns_513(set(mf513["ticker"]))
    gf513, calgap = load_gapfill()
    cal = cal700.union(cal513).union(calgap)
    report["calendar"] = {
        "n_days_union": len(cal), "n_days_700": len(cal700),
        "n_days_513": len(cal513),
        "days_only_in_one_source": int(len(cal) - len(cal700.intersection(cal513))),
        "span": [str(cal[0].date()), str(cal[-1].date())]}

    # ---- return-construction disclosure (§2.1 vs the full_ohlcv reality) ----
    report["return_construction"] = {
        "side_700": "CRSP DlyRet from crsp_sp1500_daily_2014_2019.parquet `ret`, keyed (permno,date)",
        "side_513": ("DlyRet = expm1(log_return) from market/crsp/market_returns.parquet "
                     "(log_return = log1p(DlyRet) per scripts/ingest_wrds.py::build_returns) "
                     "— IDENTICAL estimand to the 700-side"),
        "prereg_note": ("§1 names full_ohlcv.parquet for the 513 tickers; it has no return "
                        "column and adj_close = raw DlyClose (ingest_wrds.build_returns), so "
                        "the §2.1 DlyRet estimand is taken from the sibling market_returns "
                        "artifact of the SAME ingest build"),
        "pct_change_adj_close_check": adj_close_comparison(mr513),
    }

    # ---- stub exclusion (§3.1) ----
    stubs = mf[mf["n_chars"] < STUB_CHARS]
    assert len(stubs) == 32, f"G6 FAIL: stub count {len(stubs)} != 32"
    report["stub_exclusion"] = {
        "rule": f"n_chars < {STUB_CHARS}", "n": len(stubs),
        "per_split": stubs["split"].value_counts().to_dict(),
        "kept_100_500_chars": int(((mf.n_chars >= 100) & (mf.n_chars < 500)).sum()),
    }
    stub_ids = set(stubs["call_id"])

    # ---- ticker -> permno (§3.3) ----
    memb = pd.read_parquet(MEMBERSHIP)
    p700, dropped_ambig = resolve_700(mf700, tmap, px700, report)
    p513, masks = resolve_513(mf513, memb, report)
    permno_of = pd.concat([p700, p513])
    dropped_ambig -= stub_ids            # no double-count in the G6 accounting
    # sp500-side tickers with data in full_ohlcv but NO market_returns rows
    # (previously fell to the §3.2 label gate; now gap-filled per §11 v1.2(2))
    no_ret = sorted(set(mf513["ticker"]) - set(mr513["ticker"]))
    report["sp500_side_tickers_without_return_rows"] = {
        "tickers": no_ret,
        "n_calls": int(mf513["ticker"].isin(no_ret).sum()),
        "note": ("membership-window cache artifact; covered by "
                 "crsp_sp500side_gapfill_2014_2019.parquet (§11 v1.2(2))")}
    mfk = mf[~mf["call_id"].isin(stub_ids) & ~mf["call_id"].isin(dropped_ambig)].copy()
    mfk["permno"] = mfk["call_id"].map(permno_of)
    assert mfk["permno"].notna().all(), "G5 FAIL: unresolved permno on kept call"
    mfk["permno"] = mfk["permno"].astype(int)
    assert mfk.groupby("call_id")["permno"].nunique().max() == 1, \
        "G5 FAIL: >1 permno per call"
    report["ambiguity_dropped_calls"] = sorted(dropped_ambig)

    # ---- company names (for maec_prompt.py §5-3) ----
    name700 = tmap.drop_duplicates(["ticker", "permno"]).set_index(
        ["ticker", "permno"])["name"].to_dict()
    nm513 = load_names_for(set(p513))
    def company_name(row) -> str:
        key = (row["ticker"], row["permno"])
        if key in name700:
            return str(name700[key])
        g = nm513[nm513["permno"] == row["permno"]]
        hit = g[(g["SecInfoStartDt"] <= row["call_date"])
                & (row["call_date"] <= g["SecInfoEndDt"])]
        if len(hit) == 0:
            hit = g[g["SecInfoStartDt"] <= row["call_date"]].sort_values("SecInfoStartDt")
            if len(hit) == 0:
                hit = g.sort_values("SecInfoStartDt")
        return str(hit["name"].iloc[-1]) if len(hit) else ""
    mfk["company_name"] = mfk.apply(company_name, axis=1)
    report["company_name_missing"] = int((mfk["company_name"] == "").sum())

    by_permno, by_ticker, by_gap = build_series(px700, mr513, gf513, cal)
    lowcov = set(pd.read_csv(COVERAGE_CSV).query("coverage < 0.9")["query_ticker"])

    # ---- §11 v1.2(2) gap-fill overlap verification (precedence: original) ----
    memb_f = memb.copy()
    memb_f["member_to_f"] = memb_f["member_to"].fillna(pd.Timestamp("2099-01-01"))
    nuni = memb_f.groupby("ticker")["permno"].nunique()
    multi = set(nuni[nuni > 1].index)
    pairs = {}
    for tk, p in (mfk.loc[mfk["ticker"].isin(set(mf513["ticker"])),
                          ["ticker", "permno"]].drop_duplicates().itertuples(
                              index=False)):
        if tk in multi:
            g = memb_f[(memb_f["ticker"] == tk) & (memb_f["permno"] == p)]
            pairs[(tk, int(p))] = list(zip(g["member_from"], g["member_to_f"], strict=False))
        else:
            pairs[(tk, int(p))] = None
    gapfill_check = verify_gapfill_overlap(by_ticker, by_gap, pairs, cal)

    # ---- per-call anchor positions ----
    cal_vals = cal.values
    t0_pos = np.searchsorted(cal_vals, mfk["call_date"].values, side="right") - 1
    assert (t0_pos >= 66) .all(), "past-window head-room violated"
    mfk = mfk.assign(t0_pos=t0_pos)

    # ---- row construction, both alignments ----
    rows, excl = [], {}
    clips = {"label_sigma_lo": 0, "feature_sigma_lo": 0, "sigma_hi": 0}
    for align in ALIGNMENTS:
        for h in HORIZONS:
            excl[(align, h)] = {"label_incomplete": 0,
                                "series_ends_in_label_window": 0,
                                "past_insufficient": 0,
                                "from_low_coverage_44": 0,
                                "dropped_sp1500_side": 0,
                                "dropped_sp500_side": 0}
    sp500_tickers = set(mf513["ticker"])
    for r in mfk.itertuples(index=False):
        fillpos = None
        if r.ticker in sp500_tickers:
            src = "sp500_cache"
            series = by_ticker.get(r.ticker)
            if series is not None and r.call_id in masks:
                lo, hi = masks[r.call_id]
                series = series.copy()
                series[(cal < lo) | (cal > hi)] = np.nan
            # §11 v1.2(2): gap-fill ONLY firm-days absent from the primary
            # source (post-mask: masked-out days are not this permno's rows)
            gap = by_gap.get(r.permno)
            if gap is not None:
                base = (series if series is not None
                        else np.full(len(cal), np.nan))
                fp = ~np.isfinite(base) & np.isfinite(gap)
                if fp.any():
                    series = base.copy()
                    series[fp] = gap[fp]
                    fillpos = fp
        else:
            src = "sp1500_extract"
            series = by_permno.get(r.permno)
        if series is None:
            series = np.full(len(cal), np.nan)
        finite = np.where(np.isfinite(series))[0]
        last_pos = int(finite[-1]) if len(finite) else -1
        tp = int(r.t0_pos)
        for align in ALIGNMENTS:
            lab_start = tp + 1 if align == "primary" else tp
            past_end = tp if align == "primary" else tp - 1
            for h in HORIZONS:
                ex = excl[(align, h)]
                lab_end = lab_start + h - 1
                if lab_end >= len(cal):
                    ex["label_incomplete"] += 1
                    continue
                side_key = ("dropped_sp500_side" if src == "sp500_cache"
                            else "dropped_sp1500_side")
                w = series[lab_start:lab_end + 1]
                if not np.isfinite(w).all():
                    if lab_start <= last_pos < lab_end:
                        ex["series_ends_in_label_window"] += 1
                    else:
                        ex["label_incomplete"] += 1
                    if r.ticker in lowcov:
                        ex["from_low_coverage_44"] += 1
                    ex[side_key] += 1
                    continue
                # past windows: match-n + HAR, presence >= 80% each (§3.2)
                feats, ok = {}, True
                for wn, nm in [(h, "v_past_match")] + \
                              [(k, f"v_past_{k}") for k in HAR_WINDOWS]:
                    pw = series[past_end - wn + 1:past_end + 1]
                    pres = np.isfinite(pw)
                    if pres.sum() < PAST_PRESENCE * wn:
                        ok = False
                        break
                    v, sig = v_of(pw[pres])
                    if sig < SIGMA_LO:
                        clips["feature_sigma_lo"] += 1
                    if sig > SIGMA_HI:
                        clips["sigma_hi"] += 1
                    feats[nm] = v
                if not ok:
                    ex["past_insufficient"] += 1
                    if r.ticker in lowcov:
                        ex["from_low_coverage_44"] += 1
                    ex[side_key] += 1
                    continue
                lab, sig = v_of(w)
                if sig < SIGMA_LO:
                    clips["label_sigma_lo"] += 1
                lab_lr, _ = v_of(np.log1p(w))
                # did any consumed day (past 66-window head .. label end) come
                # from the §11 v1.2(2) gap-fill source?
                row_src = src
                if fillpos is not None and \
                        fillpos[past_end - HAR_WINDOWS[-1] + 1:lab_end + 1].any():
                    row_src = src + "+gapfill"
                rows.append({
                    "alignment": align, "call_id": r.call_id, "ticker": r.ticker,
                    "permno": r.permno, "call_date": r.call_date, "split": r.split,
                    "horizon": h, "label": lab, "label_logret": lab_lr, **feats,
                    "t0": cal[tp], "t1": cal[tp + 1],
                    "label_win_start": cal[lab_start], "label_win_end": cal[lab_end],
                    "n_chars": r.n_chars, "text_path": r.path,
                    "company_name": r.company_name, "price_source": row_src,
                })
    panel = pd.DataFrame(rows)
    report["sigma_clips"] = clips

    # ---- §11 v1.2(2) gap-fill disclosure ----
    gap_rows_ah = {
        f"{a}_h{h}": int(((panel["alignment"] == a) & (panel["horizon"] == h)
                          & (panel["price_source"] == "sp500_cache+gapfill")
                          ).sum())
        for a in ALIGNMENTS for h in HORIZONS}
    report["sp500_side_gapfill"] = {
        "file": str(GAPFILL),
        "rule": ("prereg §11 v1.2(2): CRSP DlyRet from the full-universe raw "
                 "zip for S&P500-side permnos whose membership-window cache "
                 "left incomplete/missing price windows; precedence ONLY "
                 "where market_returns has no (firm, day) row"),
        "rows": len(gf513),
        "n_permnos": int(gf513["permno"].nunique()),
        "n_tickers": int(gf513["ticker"].nunique()),
        "span": [str(gf513["date"].min().date()),
                 str(gf513["date"].max().date())],
        "overlap_with_market_returns": gapfill_check,
        "panel_rows_using_gapfill": gap_rows_ah,
    }

    # ---- STPEV: point-in-time expanding (primary) + train+val fixed (§5) ----
    stpev_cov = {}
    parts = []
    for (align, h), d in panel.groupby(["alignment", "horizon"]):
        d = d.sort_values(["permno", "call_date"], kind="mergesort").copy()
        tv = d[d["split"].isin(("train", "val"))]
        g_tv = float(tv["label"].mean())
        fixed = tv.groupby("permno")["label"].mean()
        d["stpev_fixed"] = d["permno"].map(fixed)
        d["stpev_fixed_has_entity"] = d["stpev_fixed"].notna()
        d["stpev_fixed"] = d["stpev_fixed"].fillna(g_tv)
        exp_vals, n_prior = np.empty(len(d)), np.zeros(len(d), int)
        i0 = 0
        for _, g in d.groupby("permno", sort=False):
            ends = g["label_win_end"].to_numpy()
            labs = g["label"].to_numpy()
            cds = g["call_date"].to_numpy()
            for i in range(len(g)):
                mask = ends <= cds[i]          # §5: prior window fully realised
                if mask.any():
                    assert ends[mask].max() <= cds[i]   # G3 STPEV point-in-time
                    exp_vals[i0 + i] = labs[mask].mean()
                    n_prior[i0 + i] = int(mask.sum())
                else:
                    exp_vals[i0 + i] = g_tv
            i0 += len(g)
        d["stpev_expanding"] = exp_vals
        d["stpev_n_prior"] = n_prior
        te = d["split"] == "test"
        stpev_cov[f"{align}_h{h}"] = {
            "global_tv_mean": g_tv,
            "expanding_frac_with_prior_all": float((n_prior > 0).mean()),
            "expanding_frac_with_prior_test": float((n_prior[te.to_numpy()] > 0).mean()),
            "fixed_entity_coverage_test": float(d.loc[te, "stpev_fixed_has_entity"].mean()),
        }
        parts.append(d)
    panel = pd.concat(parts, ignore_index=True)
    report["stpev_coverage"] = stpev_cov

    # ---- G5 keys / G3 split asserts on the final panel ----
    for align in ALIGNMENTS:
        pa = panel[panel["alignment"] == align]
        assert not pa.duplicated(["permno", "call_date", "horizon"]).any(), \
            f"G5 FAIL: duplicate key in {align}"
        tr, va, te = (pa.loc[pa["split"] == s, "call_date"] for s in
                      ("train", "val", "test"))
        assert tr.max() < va.min() < te.min(), f"G3 FAIL: split order in {align}"
        assert tr.max() <= TRAIN_END and va.min() >= VAL_START and \
            va.max() <= VAL_END and te.min() >= TEST_START, "G3 FAIL: pinned dates"
        for h in HORIZONS:
            ph = pa[pa["horizon"] == h]
            nv = int((ph["split"] == "val").sum())
            nt = int((ph["split"] == "test").sum())
            assert nv >= MIN_VAL, f"G3 FAIL: {align} h={h} val rows {nv} < {MIN_VAL}"
            assert nt >= MIN_TEST, f"G3 FAIL: {align} h={h} test rows {nt} < {MIN_TEST}"

    # ---- G6 exclusion accounting, reconciled to 3,443 ----
    n_stub, n_amb = len(stub_ids), len(dropped_ambig)
    acct, counts = {}, {}
    for align in ALIGNMENTS:
        for h in HORIZONS:
            ph = panel[(panel["alignment"] == align) & (panel["horizon"] == h)]
            ex = excl[(align, h)]
            price_drops = (ex["label_incomplete"] + ex["series_ends_in_label_window"]
                           + ex["past_insufficient"])
            total = 3443 - n_stub - n_amb - price_drops
            assert total == len(ph), (
                f"G6 FAIL {align} h={h}: 3443-{n_stub}-{n_amb}-{price_drops}"
                f"={total} != {len(ph)}")
            acct[f"{align}_h{h}"] = {
                "stub": n_stub, "ambiguity_dropped": n_amb, **ex,
                "rows_final": len(ph),
                "reconciles_to_3443": True}
            counts[f"{align}_h{h}"] = ph["split"].value_counts().to_dict()
    report["exclusion_accounting"] = acct
    report["rows_per_split"] = counts

    # ---- boundary overlap disclosure (§4) ----
    bo = {}
    for align in ALIGNMENTS:
        for h in HORIZONS:
            ph = panel[(panel["alignment"] == align) & (panel["horizon"] == h)]
            n = int(((ph["split"] == "val")
                     & (ph["label_win_end"] >= TEST_START)).sum())
            bo[f"{align}_h{h}"] = n
    report["boundary_overlap_val_rows"] = bo
    report["gates"] = {"G3_split_asserts": "PASS", "G5_keys": "PASS",
                       "G6_accounting": "PASS"}

    # ---- write ----
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False, compression="zstd")
    pd.DataFrame({"date": cal}).to_parquet(OUT_CAL, index=False)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(
        report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    print(f"\n=== maec_build_panel done in {time.time() - t0:.1f}s ===")
    print(f"panel rows: {len(panel):,} -> {args.out}")
    for k, v in counts.items():
        print(f"  {k}: {v}  (rows_using_gapfill={gap_rows_ah[k]})")
    print("\nTEST rows per alignment x horizon:")
    for k, v in counts.items():
        print(f"  {k}: test={v.get('test', 0)}")
    print(f"gates: G3 PASS, G5 PASS, G6 PASS; report -> {args.report}")


if __name__ == "__main__":
    main()
