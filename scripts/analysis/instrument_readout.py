"""T6-15 — What the surviving instrument actually emits.

The study's one surviving positive is the prompted-LLM (C6) increment on the
event-driven 8-K channel. The report states it as a percentage of reference
QLIKE and never says what the instrument DOES. This script commits the readout.

It is a DIAGNOSTIC OF FROZEN FORECASTS, not a new test. No model is run, no
weight is fitted, nothing is re-estimated: it reads the committed test
predictions of the C6 arm and of the A2 HAR-RV baseline and describes them.
There is no clustering, no Holm correction and no placebo here, so nothing it
prints is a survivor claim and none of it enters any pre-declared family.

What it measures, on the surviving cell (C6, event-driven, h=5):
  (1) how many distinct values the prompted arm emits at all;
  (2) the share carried by the two modal values;
  (3) whether that near-binary flag separates realised volatility WITHIN
      strata of the price forecast -- i.e. whether it adds ordering the price
      model does not already have. Rows are bucketed into deciles of the A2
      HAR-RV forecast, computed on the full test panel, and the two modal
      groups are compared inside each decile.

Run from repo root:  .venv/bin/python scripts/analysis/instrument_readout.py
Outputs (NEW files): results/tables/instrument_readout.{csv,md}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DISC = "event_driven"
H = 5
C6 = "results/runs/C6_llmtext_full_event_driven_seed2026/predictions.parquet"
A2 = "results/runs/A2_har_rv_full_event_driven_seed2026/predictions.parquet"
OUT = Path("results/tables")

# Gate values. These are what the report prints for this cell; if the committed
# predictions stop reproducing them the evidence has drifted and this script must
# not quietly emit a different table.
EXPECT_ROWS = 25109


def gate(cond, msg):
    if not cond:
        sys.exit(f"GATE FAILED: {msg}")


def main():
    c6 = pd.read_parquet(C6)
    a2 = pd.read_parquet(A2)
    te = c6[(c6.split == "test") & (c6.horizon_days == H)]
    gate(len(te) == EXPECT_ROWS,
         f"C6 {DISC} test h={H} has {len(te)} rows, expected {EXPECT_ROWS}")

    v = te.prediction_realised_vol
    vc = v.value_counts()
    n_distinct = int(v.nunique())
    hi, lo = float(vc.index[0]), float(vc.index[1])
    sh_hi, sh_lo = 100 * vc.iloc[0] / len(v), 100 * vc.iloc[1] / len(v)
    sh_two = sh_hi + sh_lo

    a2t = (a2[(a2.split == "test") & (a2.horizon_days == H)]
           [["accession", "prediction_realised_vol"]]
           .rename(columns={"prediction_realised_vol": "har"}))
    m = te.merge(a2t, on="accession", how="inner")
    gate(len(m) == EXPECT_ROWS,
         f"merge on accession gave {len(m)} rows, expected {EXPECT_ROWS}")

    # Deciles on the FULL test panel, so the strata are a property of the price
    # forecast and not of the subset the two modal values happen to occupy.
    m["dec"] = pd.qcut(m.har, 10, labels=False, duplicates="drop")
    sub = m[m.prediction_realised_vol.isin([hi, lo])].copy()
    sub["grp"] = np.where(sub.prediction_realised_vol == hi, "high", "low")

    g = sub.groupby(["dec", "grp"]).label_realised_vol.mean().unstack()
    n = sub.groupby(["dec", "grp"]).size().unstack()
    g["diff"] = g["high"] - g["low"]
    g["n_high"], g["n_low"] = n["high"], n["low"]
    n_pos = int((g["diff"] > 0).sum())
    strictly_monotone = bool(np.all(np.diff(g["diff"].values) > 0))

    OUT.mkdir(parents=True, exist_ok=True)
    g.reset_index().to_csv(OUT / "instrument_readout.csv", index=False)

    with open(OUT / "instrument_readout.md", "w") as f:
        f.write("# T6-15 — What the surviving instrument emits "
                f"(C6, {DISC}, h={H}, {len(te):,} test rows)\n\n")
        f.write("DIAGNOSTIC OF FROZEN FORECASTS. No clustering, no Holm, no placebo; "
                "this enters no pre-declared family and is not a survivor claim.\n\n")
        f.write(f"- distinct emitted values: **{n_distinct}**\n")
        f.write(f"- two modal values carry **{sh_two:.1f}%** of forecasts "
                f"({hi} at {sh_hi:.1f}%, {lo} at {sh_lo:.1f}%)\n")
        f.write(f"- the {hi}/{lo} flag separates mean realised volatility in "
                f"**{n_pos} of {len(g)}** deciles of the HAR-RV forecast\n")
        f.write(f"- gap widens from {g['diff'].iloc[0]:+.4f} in the lowest decile to "
                f"{g['diff'].iloc[-1]:+.4f} in the highest; "
                f"strictly increasing across deciles: **{strictly_monotone}**\n\n")
        f.write("| HAR-RV decile | n(high) | n(low) | mean RV, high | mean RV, low | difference |\n")
        f.write("|---|---|---|---|---|---|\n")
        for d, r in g.iterrows():
            f.write(f"| {int(d) + 1} | {int(r.n_high):,} | {int(r.n_low):,} | "
                    f"{r['high']:.4f} | {r['low']:.4f} | {r['diff']:+.4f} |\n")

    print(f"distinct={n_distinct}  modal share={sh_two:.1f}%  positive in {n_pos}/{len(g)} deciles  "
          f"strictly monotone={strictly_monotone}")
    print(f"wrote {OUT/'instrument_readout.md'} and .csv")


if __name__ == "__main__":
    main()
