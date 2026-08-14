"""Publication figure: grouped-bar QLIKE leaderboard on long_form, faceted by horizon.
Source: results/tables/seed_aggregate.csv (+ A2_har_rv metrics.json for the HAR reference line).
Writes results/figures/fig_leaderboard.{png,pdf}.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HORIZONS = (5, 10, 20)

# ---- load seed-aggregated QLIKE (long_form) ----
agg = pd.read_csv(os.path.join(REPO, "results/tables/seed_aggregate.csv"))
lf = agg[agg.disclosure == "long_form"].copy()

# ---- HAR reference straight from metrics.json (not in aggregate table) ----
har = {}
hm = json.load(open(os.path.join(REPO, "results/runs/A2_har_rv_full_long_form_seed2026/metrics.json")))
for d in hm:
    if d["split"] == "test" and d["disclosure_subset"] == "long_form":
        har[int(d["horizon_days"])] = float(d["qlike"])

# ---- block assignment ----
def block_of(m):
    return m[0]  # A/B/C/D

BLOCK_COLORS = {  # Okabe-Ito colorblind-safe
    "A": "#999999",  # price/classical (grey)
    "B": "#0072B2",  # text-classical (blue)
    "C": "#009E73",  # neural text (green)
    "D": "#D55E00",  # fusion (vermillion)
}
BLOCK_LABEL = {
    "A": "A price/classical",
    "B": "B text-classical",
    "C": "C neural text",
    "D": "D text+price fusion",
}

lf["block"] = lf.model.apply(block_of)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
fig, axes = plt.subplots(3, 1, figsize=(13, 15), sharex=False)

for ax, h in zip(axes, HORIZONS):
    sub = lf[lf.horizon == h].copy()
    # sort within block by qlike ascending (best first), block order A,B,C,D
    sub["block_rank"] = sub.block.map({"A": 0, "B": 1, "C": 2, "D": 3})
    sub = sub.sort_values(["block_rank", "qlike_mean"]).reset_index(drop=True)

    # x positions with a gap between blocks
    x, xt, cols, gap = [], [], [], 0.0
    prev_block = None
    pos = 0.0
    for _, r in sub.iterrows():
        if prev_block is not None and r.block != prev_block:
            pos += 1.0  # gap between blocks
        x.append(pos)
        xt.append(r.model)
        cols.append(BLOCK_COLORS[r.block])
        prev_block = r.block
        pos += 1.0
    x = np.array(x)

    ax.bar(x, sub.qlike_mean.values, yerr=sub.qlike_std.values, color=cols,
           edgecolor="black", linewidth=0.4, capsize=2.5, error_kw={"elinewidth": 0.8})

    # HAR reference line
    hval = har[h]
    ax.axhline(hval, color="red", linestyle="--", linewidth=1.4, zorder=5)
    ax.text(x[-1], hval, f"  A2_har_rv = {hval:.3f}", color="red", va="bottom",
            ha="right", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xt, rotation=55, ha="right", fontsize=9)
    ax.set_ylabel("Test QLIKE")
    ax.set_title(f"Horizon = {h} days")
    ax.margins(x=0.01)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # annotate that all text models sit above the HAR line
    text_above = sub[(sub.block.isin(["B", "C", "D"])) & (sub.qlike_mean > hval)]
    n_text = int((sub.block.isin(["B", "C", "D"])).sum())
    ax.annotate(
        f"All {len(text_above)}/{n_text} text/fusion models sit ABOVE (worse than) HAR",
        xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
        fontsize=10, color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec="red", alpha=0.9))

# legend
handles = [plt.Rectangle((0, 0), 1, 1, color=BLOCK_COLORS[b], ec="black", lw=0.4)
           for b in ["A", "B", "C", "D"]]
handles.append(plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.4))
labels = [BLOCK_LABEL[b] for b in ["A", "B", "C", "D"]] + ["A2_har_rv reference"]
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True,
           bbox_to_anchor=(0.5, 1.005), fontsize=10)

fig.suptitle(
    "Leaderboard: test QLIKE by model on long-form disclosures (lower is better)\n"
    "grouped by block, sorted within block; error bars = seed std (C/D: 3 seeds)",
    y=1.035, fontsize=13, fontweight="bold")

fig.tight_layout(rect=[0, 0, 1, 0.98])

os.makedirs(os.path.join(REPO, "results/figures"), exist_ok=True)
png = os.path.join(REPO, "results/figures/fig_leaderboard.png")
pdf = os.path.join(REPO, "results/figures/fig_leaderboard.pdf")
fig.savefig(png, dpi=150, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

# ---- verify + report ----
for p in (png, pdf):
    sz = os.path.getsize(p)
    print(f"WROTE {p} ({sz} bytes)")
    assert sz > 0

for h in HORIZONS:
    sub = lf[lf.horizon == h]
    txt = sub[sub.block.isin(["B", "C", "D"])]
    above = (txt.qlike_mean > har[h]).sum()
    best = sub.loc[sub.qlike_mean.idxmin()]
    print(f"h={h}: HAR={har[h]:.4f} | text/fusion above HAR = {above}/{len(txt)} | "
          f"overall-best={best.model} ({best.qlike_mean:.4f})")
