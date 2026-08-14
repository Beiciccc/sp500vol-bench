"""Appendix figure FP4 — what the training compute bought.

  (a) every arm's best long-form test QLIKE against the GPU-hours its runs
      consumed.  The efficiency frontier is a single point at zero cost, and
      inside the arms that did spend GPU time the rank correlation between
      spending and loss is POSITIVE;
  (b) where the 590.4 GPU-hours went, arm by arm, with each arm's accuracy
      rank out of the 25 on the artefact.

Descriptive, not causal: the expensive arms are the fine-tuned encoders and
the cheap ones the frozen-embedding fusions, so (a) prices architectures, not
a compute-response curve.

Sources
-------
results/tables/cost_accuracy.csv   GPU-hours and per-horizon long-form QLIKE
results/tables/cost_accuracy.md    the block totals and the CPU-baseline note
"""
import os
import sys
import textwrap

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANALYSIS = "scripts/analysis"
sys.path.insert(0, ANALYSIS)

import diss_style as ds
import matplotlib.pyplot as plt
from supp_style import BLUE, GREY, INK, INK2, PURPLE, RULE, TAB, VERM, YELLOW, apply_style, gate

# ---------------------------------------------------------------- evidence
c = pd.read_csv(os.path.join(TAB, "cost_accuracy.csv"))
c = c.sort_values("best_qlike").reset_index(drop=True)
c["rank"] = np.arange(1, len(c) + 1)

paid = c[c.gpu_hours_total > 0]
rho, prho = spearmanr(paid.gpu_hours_total, paid.best_qlike)

tot_c = float(c[c.block == "C"].gpu_hours_total.sum())
tot_d = float(c[c.block == "D"].gpu_hours_total.sum())
total = tot_c + tot_d
c4 = float(c[c.model == "C4_longformer"].gpu_hours_total.iloc[0])
har = float(c[c.model == "A2_har_rv"].best_qlike.iloc[0])

top5 = paid.sort_values("gpu_hours_total", ascending=False).head(5)
top5_share = 100 * float(top5.gpu_hours_total.sum()) / total

# ------------------------------------------------------------------- gate
gate(
    {"n_arms": 25, "block_C_gpu_h": 566.7, "block_D_gpu_h": 23.6,
     "total_gpu_h": 590.4, "longformer_gpu_h": 254.7,
     "longformer_share_pct": 43.1, "har_best_qlike": 0.297,
     "n_pareto_arms": 1, "spearman_rho_paid": 0.76},
    {"n_arms": len(c), "block_C_gpu_h": round(tot_c, 1),
     "block_D_gpu_h": round(tot_d, 1), "total_gpu_h": round(total, 1),
     "longformer_gpu_h": round(c4, 1),
     "longformer_share_pct": round(100 * c4 / total, 1),
     "har_best_qlike": round(har, 3),
     "n_pareto_arms": int((c.on_pareto_frontier == "yes").sum()),
     "spearman_rho_paid": round(float(rho), 2)},
)

BLK = {"A": (BLUE, "o", "A  price baselines"),
       "B": (YELLOW, "D", "B  classical text pipelines"),
       "C": (VERM, "s", "C  fine-tuned and frozen text encoders"),
       "D": (PURPLE, "^", "D  price-and-text fusion")}

# ------------------------------------------------------------------ canvas
apply_style(9)
fig = plt.figure(figsize=ds.canvas(7.45))
gs = fig.add_gridspec(2, 1, height_ratios=[1.32, 1.00],
                      left=0.300, right=0.972, top=0.925, bottom=0.185,
                      hspace=0.42)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])

# ===================================================== (a) the cost frontier
for blk, (col, mk, lab) in BLK.items():
    sel = c[c.block == blk]
    axA.plot(sel.gpu_hours_total, sel.best_qlike, ls="none", marker=mk,
             ms=5.6, mfc=col, mec=col, alpha=0.9, label=lab, zorder=4)

axA.set_xscale("symlog", linthresh=0.15, linscale=0.55)
axA.set_xlim(-0.03, 700)
axA.set_xticks([0, 0.1, 1, 10, 100])
axA.set_xticklabels(["0", "0.1", "1", "10", "100"])
axA.set_ylim(0.15, 1.20)
axA.set_yticks([0.25, 0.50, 0.75, 1.00])
# The HAR line and its caption are apparatus, not one of the four model blocks:
# INK2 puts them a register below the data labels without touching geometry.
axA.axhline(har, color=INK2, ls="--", lw=0.8, zorder=2)
axA.text(620, har - 0.022, "A2 HAR-RV reference, 0.297", fontsize=8.9,
         color=INK2, ha="right", va="top")
axA.set_xlabel("GPU-hours consumed by that arm's runs (symmetric log; 0 is exact)")
axA.set_ylabel("best long-form test QLIKE,\nvariance units (lower is better)",
               fontsize=8.9)

_par = c[c.on_pareto_frontier == "yes"]
axA.plot(_par.gpu_hours_total, _par.best_qlike, ls="none", marker="o",
         ms=11.0, mfc="none", mec=GREY, mew=1.0, zorder=5)

LBL = [("A3_garch", 12, -13, "left"), ("C4_longformer", 0, 14, "center"),
       ("D3_gteqwen2", 9, -14, "left"), ("B1_bow_ridge", 10, 0, "left"),
       ("C3_roberta_s1", 9, 6, "left"), ("C2_finbert_s2", -9, 9, "right"),
       # -8, not -12: at -12 the label's descenders sat 0.75 pt off the dashed
       # HAR reference line, so the instrument crossed the type.  Moving it up
       # is inside the axes and therefore free of the tight bounding box.
       ("C5_gteqwen2", 7, -8, "left")]
PRETTY = {"A3_garch": "A3 GARCH — the entire frontier, at zero cost",
          "C4_longformer": "C4 Longformer",
          "D3_gteqwen2": "D3 price + gte-Qwen2",
          "B1_bow_ridge": "B1 bag-of-words ridge",
          "C3_roberta_s1": "C3 RoBERTa",
          "C2_finbert_s2": "C2 FinBERT (chunked)",
          "C5_gteqwen2": "C5 frozen 7–8B embedding arms"}
for mid, dx, dy, ha in LBL:
    r = c[c.model == mid].iloc[0]
    axA.annotate(PRETTY[mid], xy=(r.gpu_hours_total, r.best_qlike),
                 xytext=(dx, dy), textcoords="offset points", fontsize=8.9,
                 color=GREY, ha=ha, va="center")

axA.legend(loc="upper left", fontsize=8.9, handletextpad=0.4, borderpad=0.3,
           labelspacing=0.28, bbox_to_anchor=(0.125, 0.86))
# This block is the panel's finding, so it takes the primary ink the panel
# heading takes.  It used to be VERM_TXT, and vermillion is already spoken for
# in this very panel: it is block C, eleven squares and a legend swatch.  A hue
# that names a model block cannot also mean "read this sentence".  What separates
# the finding from the seven point labels is now a hairline, which costs no
# geometry, rather than a colour that contradicted the legend.
axA.text(0.988, 0.985,
         f"among the {len(paid)} arms that spent GPU time,\n"
         f"Spearman $\\rho$ = {rho:+.2f} ($p$ = {prho:.4f}) between\n"
         "hours spent and loss: more compute,\nhigher loss",
         transform=axA.transAxes, fontsize=8.9, color=INK, ha="right",
         va="top")
# Sited by measurement, and the first siting was wrong on the page: the legend's
# four rows occupy the bands 0.607-0.651, 0.664-0.708, 0.721-0.765 and
# 0.778-0.822 in axes fractions, so a rule anywhere below the block's own bottom
# (0.776) necessarily lands in one of them.  At x = 0.60 it started only 12.7 pt
# after "B  classical text pipelines" ended and rendered as that row's own
# underline, tying the legend to a sentence about all sixteen paid arms.  Starting
# at x = 0.78 -- clear of the whole legend, whose widest row ends at 0.743 -- puts
# 66 pt of white between them, and the rule reads as what it is: the lower edge
# of the finding block, right-aligned with it.  Nothing else lives in that span
# (the highest marker right of 0.78 is at y = 0.433).
# add_artist, not plot, so the hairline cannot touch the data limits.
axA.add_artist(plt.Line2D([0.78, 0.988], [0.740, 0.740],
                          transform=axA.transAxes, color=RULE, lw=0.5,
                          solid_capstyle="butt", zorder=1))

# ================================================= (b) where the hours went
srt = paid.sort_values("gpu_hours_total", ascending=True)
yy = np.arange(len(srt))
axB.barh(yy, srt.gpu_hours_total, height=0.66,
         color=[BLK[b][0] for b in srt.block], edgecolor="none", zorder=3)
NAME = {"C4_longformer": "C4 Longformer", "C2_finbert_s2": "C2 FinBERT (chunk)",
        "C2_finbert_s3": "C2 FinBERT (first+last)", "C2_finbert_s4": "C2 FinBERT (stride)",
        "C1_bert_s2": "C1 BERT (chunk)", "C1_bert_s1": "C1 BERT (trunc.)",
        "C3_roberta_s1": "C3 RoBERTa (trunc.)", "C2_finbert_s1": "C2 FinBERT (trunc.)",
        "D2_gated_fusion": "D2 gated fusion", "D1_concat_mlp": "D1 concat MLP",
        "C5_e5mistral": "C5 E5-Mistral", "C5_gteqwen2": "C5 gte-Qwen2",
        "C5_qwen3": "C5 Qwen3-emb.", "D3_gteqwen2": "D3 price + gte-Qwen2",
        "D3_e5mistral": "D3 price + E5-Mistral", "D3_qwen3": "D3 price + Qwen3"}
axB.set_yticks(yy)
axB.set_yticklabels([NAME[m] for m in srt.model], fontsize=8.9)
for y, (_, r) in zip(yy, srt.iterrows(), strict=False):
    # The share of the budget is printed because a logarithmic axis cannot
    # carry it: bar length here is log hours, so the 254.7 h arm draws about
    # 3.4 times the 12.7 h arm rather than 20 times, and the panel's claim is
    # a share-of-total claim.
    share = 100 * r.gpu_hours_total / total
    lab = (f"{r.gpu_hours_total:,.1f} h  ·  {share:.1f} %" if share >= 0.05
           else f"{r.gpu_hours_total:,.1f} h  ·  <0.1 %")
    xfrac = ((np.log10(r.gpu_hours_total) - np.log10(0.15))
             / (np.log10(5000) - np.log10(0.15)))
    if xfrac > 0.65:                       # only the 254.7 h arm reaches here
        axB.text(r.gpu_hours_total / 1.10, y, lab, va="center", ha="right",
                 fontsize=8.9, color="white", zorder=4)
    else:
        axB.text(r.gpu_hours_total * 1.12 + 0.02, y, lab, va="center",
                 ha="left", fontsize=8.9, color=GREY)
    axB.text(1.0, y, f"rank {int(r['rank'])}",
             transform=axB.get_yaxis_transform(),
             va="center", ha="right", fontsize=8.9, color=GREY)
axB.set_xscale("log")
axB.set_xlim(0.15, 5000)
axB.set_xticks([0.2, 1, 10, 100])
axB.set_xticklabels(["0.2", "1", "10", "100"])
axB.set_ylim(-0.8, len(srt) - 0.2)
axB.set_xlabel("GPU-hours, log scale")

# ---------------------------------------------------------- panel headings
fig.canvas.draw()


def heading(ax, text, dy=0.014):
    fig.text(0.004, ax.get_position().y1 + dy, text, fontsize=9.2, color=GREY,
             ha="left", va="bottom")


heading(axA, "(a)  Accuracy against cost: the efficiency frontier is one "
             "point, and it is free")
heading(axB, f"(b)  Where the {total:,.1f} GPU-hours went, with each arm's "
             f"accuracy rank out of 25")

NOTE = (
    "Cost is total GPU-hours over all runs of that arm (three seeds x three "
    "disclosure channels for blocks C and D); accuracy is the best test QLIKE "
    "over h = 5, 10 and 20 on the long-form panel, in variance units, seed-"
    "averaged before the minimum is taken. Block A and B arms ran on CPU and "
    "are plotted at exactly zero GPU-hours; their wall-clock seconds are "
    "recorded separately. Prompted-LLM inference is outside this accounting. "
    "The Longformer arm alone accounts for "
    f"{100 * c4 / total:.1f} % of the total."
)
# Apparatus block: recessive ink and a hairline above it, so a reader can tell
# at a glance that these six lines are the figure's basis statement and not more
# of its argument.  Every word is kept -- they carry the run count, the horizon
# set, the CPU-arm convention and the bound of the accounting.
# Not note(): note() assumes va="top" and hard-codes linespacing 1.32, and this
# block is anchored by its BOTTOM at y = 0.006 (which is the page's bottom edge
# under bbox_inches="tight") at its own 1.34.  Only the ink and the rule are
# borrowed.  The rule sits in the 9.1 pt gap between this block's top (0.117) and
# panel (b)'s x-label (0.134), centred in it at 0.1255 rather than at note()'s
# +0.012: at 0.129 it sat 1.7 pt under the x-label's own bottom and read as that
# label's underline.  It stops at x = 0.50, left of the label's x0 = 0.539.
# butt cap so its left end cannot push the page wider.
fig.lines.append(plt.Line2D([0.004, 0.500], [0.1255, 0.1255],
                            transform=fig.transFigure, color=RULE,
                            linewidth=0.5, solid_capstyle="butt", zorder=0.5))
fig.text(0.004, 0.006, textwrap.fill(NOTE, 92), fontsize=8.9, color=INK2,
         ha="left", va="bottom", linespacing=1.34)

ds.finish(fig, "FP4_compute_accuracy", max_render_pt=595.0,
          note="appendix figure: compute-accuracy frontier and the GPU-hour "
               "budget by arm")
