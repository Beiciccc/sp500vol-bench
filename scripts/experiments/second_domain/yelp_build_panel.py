#!/usr/bin/env python
"""Build the canonical Yelp business-month panel (second domain, SECOND_DOMAIN_PLAN.md §2).

Panel definition (spec-exact):
    entity   = business_id
    event    = business-month with >= --min-reviews (default 3) reviews in the month;
               the entity must have >= --min-months (default 20) qualifying months
    text     = concatenation of the event month's reviews, each truncated to its first
               256 words (implementation cap: at most --max-reviews-per-event reviews
               per event are concatenated; capped events are counted and reported)
    outcome  = forward mean stars over the horizon window (review-level mean =
               sum of stars / count), requiring >= 3 reviews in the outcome window
               and the window to end within the observed data range
    horizons = 1 month (label = month t+1) and 3 months (label = months t+1..t+3)
    splits   = train <= 2016-12 / val 2017 / test 2018-01..2021-12 (by EVENT month;
               post-2021 events are dropped)

AR features (all computed from months <= t; no look-ahead):
    ar_last_mean      mean stars in the event month t itself
    ar_roll3_mean     mean of the monthly mean-star series over months t-2..t (months
                      present in the data)
    ar_roll12_mean    same over months t-11..t
    ar_log_n_reviews  log(1 + review count in month t)

Canonical parquet schema (long format, one row per entity x month x horizon):
    [entity_id, event_time, split, horizon_months, label, text, n_reviews,
     ar_last_mean, ar_roll3_mean, ar_roll12_mean, ar_log_n_reviews]

Includes the Phase-0 audit printout and HARD gate-G1 assertions
(>= 100 entities each with >= 20 qualifying months; >= 10k events;
val >= 100 and test >= 30 rows per horizon).

Swapping synthetic -> real data is the --data-root flag, nothing else:
    python yelp_build_panel.py --data-root /path/to/data-root/second-domain/yelp \
                               --out results/second_domain/yelp_panel.parquet
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_END = 2016 * 12 + 11          # 2016-12
VAL_START, VAL_END = 2017 * 12, 2017 * 12 + 11
TEST_START, TEST_END = 2018 * 12, 2021 * 12 + 11
HORIZONS = (1, 3)
OUTCOME_MIN_REVIEWS = 3
TRUNC_WORDS = 256

REVIEW_FILE = "yelp_academic_dataset_review.json"
BUSINESS_FILE = "yelp_academic_dataset_business.json"


def parse_month(date_str: str) -> int:
    """'YYYY-MM-DD[ HH:MM:SS]' -> integer month index (year*12 + month-1)."""
    return int(date_str[:4]) * 12 + int(date_str[5:7]) - 1


def month_ts(m: int) -> pd.Timestamp:
    return pd.Timestamp(year=m // 12, month=m % 12 + 1, day=1)


def split_of(m: int) -> str:
    if m <= TRAIN_END:
        return "train"
    if m <= VAL_END:
        return "val"
    if m <= TEST_END:
        return "test"
    return "post"


def pass1_counts(path: Path):
    """Stream review.json once: per (business_id, month) review count and star sum."""
    cnt: dict = defaultdict(int)
    ssum: dict = defaultdict(float)
    t0, n = time.time(), 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["business_id"], parse_month(r["date"]))
            cnt[key] += 1
            ssum[key] += float(r["stars"])
            n += 1
            if n % 1_000_000 == 0:
                print(f"  pass 1: {n:,} reviews ({time.time() - t0:.0f}s)", flush=True)
    print(f"  pass 1 done: {n:,} reviews, {len(cnt):,} business-months "
          f"({time.time() - t0:.0f}s)", flush=True)
    return cnt, ssum, n


def pass2_text(path: Path, events: set, max_reviews: int):
    """Stream review.json again, keeping truncated text ONLY for qualifying events."""
    texts: dict = defaultdict(list)
    t0, n = time.time(), 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            key = (r["business_id"], parse_month(r["date"]))
            if key not in events:
                continue
            bucket = texts[key]
            if len(bucket) >= max_reviews:
                continue
            bucket.append(" ".join((r.get("text") or "").split()[:TRUNC_WORDS]))
            if n % 1_000_000 == 0:
                print(f"  pass 2: {n:,} reviews ({time.time() - t0:.0f}s)", flush=True)
    print(f"  pass 2 done ({time.time() - t0:.0f}s)", flush=True)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/path/to/data-root/second-domain/yelp",
                    help="directory holding the two Yelp JSON files")
    ap.add_argument("--out", default="results/second_domain/yelp_panel.parquet")
    ap.add_argument("--min-reviews", type=int, default=3,
                    help="reviews required in the event month")
    ap.add_argument("--min-months", type=int, default=20,
                    help="qualifying months required per entity")
    ap.add_argument("--max-reviews-per-event", type=int, default=100,
                    help="cap on reviews concatenated per event month")
    args = ap.parse_args()

    root = Path(args.data_root)
    review_path = root / REVIEW_FILE
    assert review_path.exists(), f"missing {review_path}"

    print(f"Building panel from {root} ...", flush=True)
    cnt, ssum, n_reviews_total = pass1_counts(review_path)
    max_month = max(m for _, m in cnt)
    print(f"  observed months: {month_ts(min(m for _, m in cnt)).date()} .. "
          f"{month_ts(max_month).date()}")

    # --- qualifying events and entity filter (gate-G1 population) -------------------
    qual_months: dict = defaultdict(list)
    for (bid, m), c in cnt.items():
        if c >= args.min_reviews and m <= TEST_END:
            qual_months[bid].append(m)
    keep = {bid for bid, ms in qual_months.items() if len(ms) >= args.min_months}
    events = {(bid, m) for bid in keep for m in qual_months[bid]}
    print(f"  entities with >= {args.min_months} qualifying months: {len(keep):,}; "
          f"base events: {len(events):,}")

    texts = pass2_text(review_path, events, args.max_reviews_per_event)
    n_capped = sum(1 for k in texts
                   if cnt[k] > args.max_reviews_per_event)

    # --- assemble the long-format panel ---------------------------------------------
    rows = []
    for bid, m in sorted(events):
        c_t = cnt[(bid, m)]
        last_mean = ssum[(bid, m)] / c_t

        def window_mean(lo: int) -> float:
            vals = [ssum[(bid, mm)] / cnt[(bid, mm)]
                    for mm in range(lo, m + 1) if (bid, mm) in cnt]
            return float(np.mean(vals))  # window always contains month t

        roll3 = window_mean(m - 2)
        roll12 = window_mean(m - 11)
        text = " ".join(texts[(bid, m)])
        split = split_of(m)
        assert split != "post", "post-2021 event leaked past the filter"

        for h in HORIZONS:
            if m + h > max_month:
                continue  # outcome window extends beyond observed data
            wc = sum(cnt.get((bid, mm), 0) for mm in range(m + 1, m + h + 1))
            if wc < OUTCOME_MIN_REVIEWS:
                continue
            ws = sum(ssum.get((bid, mm), 0.0) for mm in range(m + 1, m + h + 1))
            rows.append((bid, month_ts(m), split, h, ws / wc, text, c_t,
                         last_mean, roll3, roll12, math.log1p(c_t)))

    panel = pd.DataFrame(rows, columns=[
        "entity_id", "event_time", "split", "horizon_months", "label", "text",
        "n_reviews", "ar_last_mean", "ar_roll3_mean", "ar_roll12_mean",
        "ar_log_n_reviews"])
    panel = panel.sort_values(["entity_id", "event_time", "horizon_months"],
                              kind="mergesort").reset_index(drop=True)

    # --- hard assertions (no look-ahead by construction: AR features use months <= t,
    # --- labels use months > t only) --------------------------------------------------
    assert not panel.duplicated(["entity_id", "event_time", "horizon_months"]).any(), \
        "duplicate (entity, month, horizon) rows"
    assert panel.label.between(1.0, 5.0).all(), "labels outside [1, 5]"
    assert panel.ar_last_mean.between(1.0, 5.0).all(), "ar_last_mean outside [1, 5]"
    ar_cols = ["ar_last_mean", "ar_roll3_mean", "ar_roll12_mean", "ar_log_n_reviews"]
    assert np.isfinite(panel[ar_cols].to_numpy()).all(), "non-finite AR features"
    assert (panel.n_reviews >= args.min_reviews).all(), "event below min-reviews"
    assert (panel.text.str.len() > 0).all(), "empty event text"
    for sp, lo, hi in (("train", -10 ** 9, TRAIN_END), ("val", VAL_START, VAL_END),
                       ("test", TEST_START, TEST_END)):
        mm = panel.loc[panel.split == sp, "event_time"]
        if len(mm):
            idx = mm.dt.year * 12 + mm.dt.month - 1
            assert idx.between(lo, hi).all(), f"{sp} events outside window"

    # --- final entity re-filter: outcome-window filtering can drop an entity below the
    # --- qualifying-month floor; re-apply the floor on the FINAL panel (one pass is
    # --- sufficient: removing an entity does not reduce any other entity's events) ----
    _base0 = panel.drop_duplicates(["entity_id", "event_time"])
    _epe0 = _base0.groupby("entity_id").size()
    _keep = _epe0[_epe0 >= args.min_months].index
    n_dropped_refilter = panel.entity_id.nunique() - len(_keep)
    panel = panel[panel.entity_id.isin(_keep)].copy()
    if n_dropped_refilter:
        print(f"final re-filter: dropped {n_dropped_refilter} entities that fell below "
              f"{args.min_months} qualifying months after outcome-window filtering")

    # --- Phase-0 audit printout -------------------------------------------------------
    base = panel.drop_duplicates(["entity_id", "event_time"])
    ev_per_ent = base.groupby("entity_id").size()
    print("\n=== Phase-0 panel audit ===")
    print(f"reviews streamed: {n_reviews_total:,}")
    print(f"entities: {base.entity_id.nunique():,}  base events (entity-months): "
          f"{len(base):,}  median events/entity: {ev_per_ent.median():.0f}")
    print(f"events with text capped at {args.max_reviews_per_event} reviews: "
          f"{n_capped:,} ({100.0 * n_capped / max(len(events), 1):.1f}%)")
    print(f"text length (words) p50/p90: "
          f"{panel.text.str.split().str.len().quantile([0.5, 0.9]).round(0).tolist()}")
    for h in HORIZONS:
        sub = panel[panel.horizon_months == h]
        counts = sub.split.value_counts()
        stats = sub.groupby("split")["label"].agg(["mean", "std"]).round(3)
        print(f"h={h}m rows: train={counts.get('train', 0):,} "
              f"val={counts.get('val', 0):,} test={counts.get('test', 0):,} | "
              f"label mean/sd by split: "
              + "; ".join(f"{s}={stats.loc[s,'mean']:.3f}/{stats.loc[s,'std']:.3f}"
                          for s in ("train", "val", "test") if s in stats.index))

    # --- consistency sanity: panel entity means vs business.json lifetime stars ------
    biz_path = root / BUSINESS_FILE
    if biz_path.exists():
        life = {}
        with open(biz_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    b = json.loads(line)
                except json.JSONDecodeError:
                    continue
                life[b["business_id"]] = float(b.get("stars") or np.nan)
        ent_mean = base.groupby("entity_id")["ar_last_mean"].mean()
        pair = pd.DataFrame({"panel": ent_mean,
                             "lifetime": ent_mean.index.map(life)}).dropna()
        r = float(pair.panel.corr(pair.lifetime))
        print(f"sanity: corr(panel entity mean, business.json lifetime stars) = {r:.3f} "
              f"(n={len(pair)})")
        assert r > 0.5, "panel means inconsistent with business.json lifetime stars"

    # --- GATE G1 (hard) ---------------------------------------------------------------
    n_ent = base.entity_id.nunique()
    assert n_ent >= 100, f"GATE G1 FAIL: only {n_ent} entities (< 100)"
    assert (ev_per_ent >= args.min_months).all(), \
        "GATE G1 FAIL: an entity slipped below the qualifying-month floor"
    assert len(base) >= 10_000, f"GATE G1 FAIL: only {len(base):,} events (< 10k)"
    for h in HORIZONS:
        sub = panel[panel.horizon_months == h]
        nv = int((sub.split == "val").sum())
        nt = int((sub.split == "test").sum())
        assert nv >= 100, f"GATE G1 FAIL: h={h}m val rows {nv} < 100"
        assert nt >= 30, f"GATE G1 FAIL: h={h}m test rows {nt} < 30"
    print(f"\nGATE G1 PASS: {n_ent:,} entities (each >= {args.min_months} months), "
          f"{len(base):,} events, val/test row floors met for all horizons")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False, compression="zstd")
    print(f"wrote {out} ({len(panel):,} rows)")


if __name__ == "__main__":
    main()
