#!/usr/bin/env python
"""Phase-0 panel audit for the Yelp second-domain replication.

Streams the Yelp Open Dataset JSON files into parquet, then audits the
business-period panel over the grid
    {min_reviews_per_period in {3, 5}} x {monthly, quarterly}
reporting, per cell:
    - n_entities with >= MIN_PERIODS qualifying periods
    - total qualifying events (entity-periods) among those entities
    - median events / entity
    - per-split event counts (train <= 2016, val 2017, test 2018-2021)

GATE G1: >= 100 entities and >= 10k events in the monthly/min3 cell
(quarterly/min3 is the fallback cell).

Usage:
    python yelp_phase0_audit.py [--data-dir /path/to/data-root/second-domain/yelp]
                                [--skip-convert]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MIN_PERIODS = 20  # qualifying periods required for an entity to enter the panel
CHUNK_ROWS = 500_000

REVIEW_SCHEMA = pa.schema(
    [
        ("business_id", pa.string()),
        ("date", pa.string()),  # YYYY-MM-DD
        ("stars", pa.float32()),
        ("text_len", pa.int32()),
    ]
)


def convert_reviews(src: Path, dst: Path) -> None:
    t0 = time.time()
    writer = pq.ParquetWriter(dst, REVIEW_SCHEMA, compression="zstd")
    buf = {k: [] for k in ("business_id", "date", "stars", "text_len")}
    n = 0
    with open(src, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            buf["business_id"].append(r["business_id"])
            buf["date"].append(r["date"][:10])
            buf["stars"].append(float(r["stars"]))
            buf["text_len"].append(len(r.get("text") or ""))
            n += 1
            if n % CHUNK_ROWS == 0:
                writer.write_table(pa.table(buf, schema=REVIEW_SCHEMA))
                buf = {k: [] for k in buf}
                print(f"  reviews: {n:,} rows ({time.time() - t0:.0f}s)", flush=True)
    if buf["business_id"]:
        writer.write_table(pa.table(buf, schema=REVIEW_SCHEMA))
    writer.close()
    print(f"  reviews -> {dst}: {n:,} rows in {time.time() - t0:.0f}s", flush=True)


def convert_business(src: Path, dst: Path) -> None:
    rows = []
    keep = (
        "business_id",
        "name",
        "city",
        "state",
        "latitude",
        "longitude",
        "stars",
        "review_count",
        "is_open",
        "categories",
    )
    with open(src, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({k: b.get(k) for k in keep})
    df = pd.DataFrame(rows)
    df.to_parquet(dst, index=False)
    print(f"  business -> {dst}: {len(df):,} rows", flush=True)


def split_of(year: int) -> str:
    if year <= 2016:
        return "train(<=2016)"
    if year == 2017:
        return "val(2017)"
    if 2018 <= year <= 2021:
        return "test(2018-21)"
    return "post-2021"


def audit(reviews_pq: Path) -> None:
    df = pd.read_parquet(reviews_pq, columns=["business_id", "date"])
    print(f"\nLoaded {len(df):,} reviews, "
          f"{df['business_id'].nunique():,} businesses, "
          f"date range {df['date'].min()} .. {df['date'].max()}")

    ym = df["date"].str[:7]  # YYYY-MM
    yr = df["date"].str[:4]
    mm = df["date"].str[5:7].astype(int)
    yq = yr + "Q" + ((mm - 1) // 3 + 1).astype(str)

    results = []
    for bucket, key in (("monthly", ym), ("quarterly", yq)):
        counts = (
            pd.DataFrame({"business_id": df["business_id"], "period": key})
            .groupby(["business_id", "period"], sort=False)
            .size()
        )
        for min_rev in (3, 5):
            q = counts[counts >= min_rev]
            per_ent = q.groupby(level="business_id").size()
            ents = per_ent[per_ent >= MIN_PERIODS]
            ev = q[q.index.get_level_values("business_id").isin(ents.index)]
            years = ev.index.get_level_values("period").str[:4].astype(int)
            splits = pd.Series(years).map(split_of).value_counts()
            results.append(
                {
                    "bucket": bucket,
                    "min_rev": min_rev,
                    "n_entities": len(ents),
                    "total_events": len(ev),
                    "median_ev_per_ent": float(ents.median()) if len(ents) else 0.0,
                    "train(<=2016)": int(splits.get("train(<=2016)", 0)),
                    "val(2017)": int(splits.get("val(2017)", 0)),
                    "test(2018-21)": int(splits.get("test(2018-21)", 0)),
                    "post-2021": int(splits.get("post-2021", 0)),
                }
            )

    out = pd.DataFrame(results)
    print("\n=== Phase-0 panel audit (entities need >= "
          f"{MIN_PERIODS} qualifying periods) ===")
    print(out.to_string(index=False))

    g1 = out[(out["bucket"] == "monthly") & (out["min_rev"] == 3)].iloc[0]
    ok = g1["n_entities"] >= 100 and g1["total_events"] >= 10_000
    print(f"\nGATE G1 (monthly/min3: >=100 entities & >=10k events): "
          f"{'PASS' if ok else 'FAIL'} "
          f"({g1['n_entities']:,} entities, {g1['total_events']:,} events)")
    if not ok:
        gq = out[(out["bucket"] == "quarterly") & (out["min_rev"] == 3)].iloc[0]
        okq = gq["n_entities"] >= 100 and gq["total_events"] >= 10_000
        print(f"GATE G1 fallback (quarterly/min3): {'PASS' if okq else 'FAIL'} "
              f"({gq['n_entities']:,} entities, {gq['total_events']:,} events)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/path/to/data-root/second-domain/yelp")
    ap.add_argument("--skip-convert", action="store_true",
                    help="reuse existing parquet files")
    args = ap.parse_args()

    d = Path(args.data_dir)
    reviews_json = d / "yelp_academic_dataset_review.json"
    business_json = d / "yelp_academic_dataset_business.json"
    pq_dir = d / "parquet"
    pq_dir.mkdir(exist_ok=True)
    reviews_pq = pq_dir / "reviews.parquet"
    business_pq = pq_dir / "business.parquet"

    if not args.skip_convert or not reviews_pq.exists():
        for p in (reviews_json, business_json):
            if not p.exists():
                sys.exit(f"missing source file: {p}")
        print("Converting business.json ...", flush=True)
        convert_business(business_json, business_pq)
        print("Converting review.json (streaming) ...", flush=True)
        convert_reviews(reviews_json, reviews_pq)

    audit(reviews_pq)


if __name__ == "__main__":
    main()
