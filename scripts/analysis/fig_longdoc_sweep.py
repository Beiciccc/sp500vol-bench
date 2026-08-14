"""Publication figure: long-doc strategy sweep (S1..S5) on long_form.

QLIKE (mean +- std over seeds) vs strategy for the FinBERT family, one series
per horizon, with BERT-base S1/S2 overlaid, and a horizontal HAR reference per
horizon. Source: results/tables/longdoc_sweep.csv.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = "."
SRC = os.path.join(REPO, "results/tables/longdoc_sweep.csv")
OUTDIR = os.path.join(REPO, "results/figures")
os.makedirs(OUTDIR, exist_ok=True)

STRAT_ORDER = ["S1_truncation", "S2_chunk_mean", "S3_chunk_attn",
               "S4_hierarchical", "S5_long_context"]
STRAT_LABEL = {"S1_truncation": "S1\nTruncation", "S2_chunk_mean": "S2\nChunk-mean",
               "S3_chunk_attn": "S3\nChunk-attn", "S4_hierarchical": "S4\nHierarch.",
               "S5_long_context": "S5\nLong-ctx"}
HORIZONS = [5, 10, 20]
# Okabe-Ito colorblind-safe palette
HCOLOR = {5: "#0072B2", 10: "#D55E00", 20: "#009E73"}

df = pd.read_csv(SRC)
lf = df[df.disclosure == "long_form"].copy()

# HAR reference (A2, seed-invariant) per horizon
har = {}
m = json.load(open(os.path.join(REPO, "results/runs/A2_har_rv_full_long_form_seed2026/metrics.json")))
for h in HORIZONS:
    r = [d for d in m if d["split"] == "test" and d["horizon_days"] == h
         and d["disclosure_subset"] == "long_form"]
    har[h] = r[0]["qlike"]

x = np.arange(len(STRAT_ORDER))

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                     "legend.fontsize": 10.5, "xtick.labelsize": 11, "ytick.labelsize": 11})
fig, ax = plt.subplots(figsize=(9.5, 6.2))

# HAR horizontal reference lines (per horizon)
for h in HORIZONS:
    ax.axhline(har[h], color=HCOLOR[h], ls=":", lw=1.3, alpha=0.75, zorder=1)
ax.text(x[-1] + 0.02, har[5], " HAR ref", va="bottom", ha="left",
        fontsize=9.5, color="#555555")

# FinBERT family: solid lines with error bars, one series per horizon
fb = lf[lf.family == "FinBERT"]
for h in HORIZONS:
    sub = fb[fb.horizon == h].set_index("strategy").reindex(STRAT_ORDER)
    ax.errorbar(x, sub.qlike_mean.values, yerr=sub.qlike_std.values,
                marker="o", ms=7, lw=2.1, capsize=4, color=HCOLOR[h],
                label=f"FinBERT  h={h}d", zorder=3)

# BERT-base overlay: S1/S2 only, dashed with square markers
bb = lf[lf.family == "BERT-base"]
for h in HORIZONS:
    sub = bb[bb.horizon == h].set_index("strategy").reindex(STRAT_ORDER)
    xb = x[:2]
    ax.errorbar(xb, sub.qlike_mean.values[:2], yerr=sub.qlike_std.values[:2],
                marker="s", ms=6, lw=1.6, ls="--", capsize=3,
                color=HCOLOR[h], alpha=0.55, markerfacecolor="white",
                label=f"BERT-base h={h}d (S1/S2)", zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([STRAT_LABEL[s] for s in STRAT_ORDER])
ax.set_xlabel("Long-document handling strategy")
ax.set_ylabel("QLIKE on test  (lower is better)")
ax.set_title("Long-document strategy sweep on long-form filings\n"
             "QLIKE (mean $\\pm$ std over 3 seeds); dotted = HAR baseline per horizon",
             fontsize=13)
ax.set_xlim(-0.35, len(STRAT_ORDER) - 0.35)
ax.grid(axis="y", ls="-", lw=0.5, alpha=0.35)
ax.legend(ncol=2, loc="upper right", framealpha=0.92)

fig.tight_layout()
png = os.path.join(OUTDIR, "fig_longdoc_sweep.png")
pdf = os.path.join(OUTDIR, "fig_longdoc_sweep.pdf")
fig.savefig(png, dpi=150)
fig.savefig(pdf)
plt.close(fig)

for p in (png, pdf):
    sz = os.path.getsize(p)
    print(f"WROTE {p}  ({sz} bytes)  exists={os.path.exists(p) and sz>0}")
print("HAR refs:", {h: round(har[h], 4) for h in HORIZONS})
