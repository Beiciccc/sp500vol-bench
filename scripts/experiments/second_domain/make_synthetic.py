#!/usr/bin/env python
"""Synthetic Yelp-shaped fixture for the second-domain pipeline (SECOND_DOMAIN_PLAN.md).

Emits line-delimited JSON files with EXACTLY the real Yelp Open Dataset filenames and
field layout, so `yelp_build_panel.py --data-root <out>` exercises the identical code
path that the real download will use:

    <out>/yelp_academic_dataset_review.json
    <out>/yelp_academic_dataset_business.json

Data-generating process (every effect is KNOWN, so the downstream machinery -- AR
baseline, recalibration, naive pooled-split arm, identity control, combiner -- can be
verified end-to-end today, before the real data lands):

    entity mean        mu_b ~ N(3.6, 0.45) clipped to [1.8, 4.8]      (STRONG identity)
    persistence        s_{b,t} = 0.8 * s_{b,t-1} + N(0, 0.15)         (AR channel)
    text shock         delta_{b,t} ~ N(0, 0.30)
    latent quality     q_{b,t} = mu_b + s_{b,t} + BETA_TEXT * delta_{b,t-1}
    review star        clip(round(q_{b,t} + N(0, 0.70)), 1, 5), integer
    reviews per month  Poisson(lambda_b), lambda_b ~ LogNormal(ln 6, 0.35)

Text channels per review:
    - sentiment words tied to the review's OWN star (contemporaneous, no forecast value
      beyond the AR features);
    - FORWARD-signal words whose inclusion odds follow delta_{b,t} -- the SMALL injected
      text effect: month-t text genuinely predicts the month-(t+1) mean via BETA_TEXT,
      and this information is in NO AR feature;
    - one per-business signature token (85% of reviews) -- the identity channel that a
      pooled random split lets a bag-of-words model exploit (gate G3 demonstration).

Note on horizon coverage: the task sketch says "~200 businesses x 60 months"; the panel
splits (train <= 2016 / val 2017 / test 2018-2021 + 3-month outcome windows) require a
longer calendar, so the default spans 2013-01..2022-03 (111 months). Businesses average
roughly 100 qualifying months each, comfortably clearing gate G1.

Ground truth is stored alongside (truth_entity.parquet, truth_months.parquet) for
optional diagnostics; the pipeline itself never reads it.

Usage:
    python make_synthetic.py --out <dir> [--n-businesses 200] [--seed 2026]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RHO, S_SD = 0.8, 0.15
MU_MEAN, MU_SD = 3.6, 0.45
MU_LO, MU_HI = 1.8, 4.8
DELTA_SD = 0.30
BETA_TEXT = 0.30
STAR_NOISE = 0.70
SIG_TOKEN_PROB = 0.85
FWD_SLOTS, FWD_SLOT_PROB, FWD_SHARPNESS = 2, 0.6, 2.5

FILLER = [
    "the", "place", "was", "we", "ordered", "service", "food", "came", "back",
    "really", "menu", "table", "staff", "time", "night", "lunch", "dinner",
    "drinks", "price", "portion", "atmosphere", "location", "parking", "wait",
    "again", "visit", "plate", "chips", "coffee", "dessert", "starter", "main",
]
POS = ["delicious", "friendly", "fresh", "excellent", "lovely", "brilliant",
       "tasty", "charming", "superb", "wonderful"]
NEG = ["stale", "rude", "dirty", "bland", "awful", "dreadful", "slow",
       "overpriced", "cold", "disappointing"]
FWD_POS = ["buzzing", "expansion", "renovated", "newchef", "promising"]
FWD_NEG = ["closingdown", "shortstaffed", "decline", "rundown", "understaffed"]


def month_list(start: str, end: str) -> list[int]:
    a = int(start[:4]) * 12 + int(start[5:7]) - 1
    b = int(end[:4]) * 12 + int(end[5:7]) - 1
    assert a <= b, "start month must not be after end month"
    return list(range(a, b + 1))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory for the fixture")
    ap.add_argument("--n-businesses", type=int, default=200)
    ap.add_argument("--start", default="2013-01")
    ap.add_argument("--end", default="2022-03")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    months = month_list(args.start, args.end)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    truth_ent, truth_mon, biz_rows = [], [], []
    n_total = 0
    rid = 0

    with open(out / "yelp_academic_dataset_review.json", "w", encoding="utf-8") as fh:
        for b in range(args.n_businesses):
            bid = f"synbiz{b:04d}"
            mu = float(np.clip(rng.normal(MU_MEAN, MU_SD), MU_LO, MU_HI))
            lam = float(np.exp(rng.normal(np.log(6.0), 0.35)))
            sig_tok = f"sig{b:04d}tok"
            truth_ent.append({"business_id": bid, "mu": mu, "lam": lam})

            s, prev_delta = 0.0, 0.0
            stars_sum, stars_n = 0.0, 0
            for m in months:
                s = RHO * s + float(rng.normal(0.0, S_SD))
                delta = float(rng.normal(0.0, DELTA_SD))
                q = mu + s + BETA_TEXT * prev_delta
                n = int(rng.poisson(lam))
                truth_mon.append({"business_id": bid, "month_idx": m,
                                  "s": s, "delta": delta, "q": q, "n": n})
                year, mo = m // 12, m % 12 + 1
                for _ in range(n):
                    star = int(np.clip(round(q + rng.normal(0.0, STAR_NOISE)), 1, 5))
                    words = [str(w) for w in rng.choice(FILLER, size=24)]
                    p_pos = (star - 1) / 4.0
                    for _ in range(4):
                        pool = POS if rng.random() < p_pos else NEG
                        words.append(str(rng.choice(pool)))
                    # forward-signal words: the injected text effect (predicts t+1)
                    for _ in range(FWD_SLOTS):
                        if rng.random() < FWD_SLOT_PROB:
                            pool = (FWD_POS if rng.random() < sigmoid(FWD_SHARPNESS * delta)
                                    else FWD_NEG)
                            words.append(str(rng.choice(pool)))
                    if rng.random() < SIG_TOKEN_PROB:
                        words.append(sig_tok)  # identity channel
                    rng.shuffle(words)
                    day = int(rng.integers(1, 28))
                    hh, mi, ss = (int(rng.integers(0, 24)), int(rng.integers(0, 60)),
                                  int(rng.integers(0, 60)))
                    rec = {
                        "review_id": f"rev{rid:08d}",
                        "user_id": f"user{int(rng.integers(0, 50_000)):06d}",
                        "business_id": bid,
                        "stars": float(star),
                        "useful": 0, "funny": 0, "cool": 0,
                        "text": " ".join(words),
                        "date": f"{year:04d}-{mo:02d}-{day:02d} {hh:02d}:{mi:02d}:{ss:02d}",
                    }
                    fh.write(json.dumps(rec) + "\n")
                    rid += 1
                    stars_sum += star
                    stars_n += 1
                prev_delta = delta
            n_total += stars_n
            lifetime = round(2.0 * stars_sum / max(stars_n, 1)) / 2.0
            biz_rows.append({
                "business_id": bid, "name": f"Synthetic Diner {b:04d}", "address": "",
                "city": "Leeds", "state": "XX", "postal_code": "", "latitude": 0.0,
                "longitude": 0.0, "stars": lifetime, "review_count": stars_n,
                "is_open": 1, "attributes": None, "categories": "Restaurants",
                "hours": None,
            })

    with open(out / "yelp_academic_dataset_business.json", "w", encoding="utf-8") as fh:
        for row in biz_rows:
            fh.write(json.dumps(row) + "\n")

    pd.DataFrame(truth_ent).to_parquet(out / "truth_entity.parquet", index=False)
    pd.DataFrame(truth_mon).to_parquet(out / "truth_months.parquet", index=False)

    assert rid == n_total and n_total > 0, "review bookkeeping mismatch"
    print(f"SYNTHETIC fixture written to {out}")
    print(f"  businesses={args.n_businesses}  months={len(months)} "
          f"({args.start}..{args.end})  reviews={n_total:,}")
    print(f"  injected effects: entity-mean sd={MU_SD}, AR rho={RHO}, "
          f"text effect BETA_TEXT={BETA_TEXT} x delta sd={DELTA_SD} "
          f"(next-month mean shift sd~{BETA_TEXT * DELTA_SD:.3f} stars)")


if __name__ == "__main__":
    main()
