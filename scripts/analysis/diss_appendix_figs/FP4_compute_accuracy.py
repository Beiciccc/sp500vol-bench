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

from supp_style import (apply_style, gate, BLUE, SKY, VERM, VERM_TXT,  # noqa: E402
                        GREEN, YELLOW, PURPLE, GREY, LIGHT, TAB, REPO,
                        INK, INK2, RULE)
import diss_style as ds  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

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
    {"n_arms": int(len(c)), "block_C_gpu_h": round(tot_c, 1),
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
# Three of these seven strings name a pile, not a point, and the singular names
# they carried were read off the wrong marker.  The anchor points are unchanged;
# only the wording is, and every replacement is measured at 8.9 pt Helvetica
# against the ink already on the page (widths below in px of the 1190 px
# render, where the figure's own ink runs x = 4 to 1185), so the tight bounding
# box cannot move:
#   C3_roberta_s1  the mark at ~13 h is three overlapping squares -- C1 BERT,
#     C2 FinBERT and C3 RoBERTa, all S1 truncation -- so a lone "C3 RoBERTa"
#     made three arms with ranks 20, 24 and 23 read as scatter within one.
#     "C1/C2/C3 truncation" is 223 px wide from x = 889, ending at 1112 against
#     the figure's rightmost ink at 1185.
#   C2_finbert_s2  the cluster at 51.7-71.3 h is four squares (C1 chunk-mean,
#     C2 chunk-mean, chunk-attention and hierarchical), and the label's nearest
#     neighbour was not its own arm but C2's hierarchical one, so the singular
#     name mis-assigned cost and accuracy across two arms of the same encoder.
#     All four are chunked arms, so the family name is the true one; it is also
#     17.6 px NARROWER than the string it replaces and right-anchored, so its
#     left edge moves inward.
#   D3_gteqwen2  the stack at 0.2 h is all three price-plus-embedding fusions,
#     which is what the caption claims and panel (b) lists; the plural matches
#     the treatment the C5 cluster already gets.  It ends at x = 807, clear of
#     the grey HAR sentence that starts at 857.
#   A3_garch  the ring at x = 0 encloses a fused blob -- five block A arms
#     inside about 20 px, with A2's marker hidden under the dashed reference
#     line -- so the frontier point cannot be read off the mark.  Printing its
#     QLIKE in the label it already has makes it verifiable, and it sits beside
#     the A2 reference value in the same panel.  0.2686 -> 0.269 from
#     results/tables/cost_accuracy.csv (A3_garch, best_qlike).  "zero cost"
#     becomes "zero GPU cost" for the same reason the (a) heading does.  The
#     string is left-anchored and ran x = 422-913; eleven characters add about
#     130 px, ending near 1040, clear of the axes' right edge at 1173.
PRETTY = {"A3_garch": "A3 GARCH, 0.269 — the entire frontier, at zero GPU cost",
          "C4_longformer": "C4 Longformer",
          "D3_gteqwen2": "D3 price + embeddings",
          "B1_bow_ridge": "B1 bag-of-words ridge",
          "C3_roberta_s1": "C1/C2/C3 truncation",
          "C2_finbert_s2": "C1/C2 chunking arms",
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
# the finding from the seven point labels is its corner and its right alignment,
# neither of which costs geometry, rather than a colour that contradicted the
# legend.
axA.text(0.988, 0.985,
         f"among the {len(paid)} arms that spent GPU time,\n"
         f"Spearman $\\rho$ = {rho:+.2f} ($p$ = {prho:.4f}) between\n"
         "hours spent and loss: more compute,\nhigher loss",
         transform=axA.transAxes, fontsize=8.9, color=INK, ha="right",
         va="top")
# The hairline that used to close this block off underneath is gone.  Inside a
# data area it could not be read as typography: it was an unlabelled horizontal
# grey line spanning 170 px at a fixed height, in a panel that already carries a
# real horizontal reference at a fixed height (the dashed A2 line), so it
# invited reading as a second reference level.  The finding block is already set
# apart by its corner, its right alignment and a clear line of white above the
# nearest ink.  Removing ink cannot enlarge a tight bounding box.

# ================================================= (b) where the hours went
srt = paid.sort_values("gpu_hours_total", ascending=True)
yy = np.arange(len(srt))
# Dots, not bars.  A bar says "this much", and on a logarithmic axis its length
# is measured from an arbitrary origin -- here the left limit, 0.15 h -- so the
# 254.7 h arm, holding 43.1 % of the budget, drew only about 2.1 times the
# length of the 4.4 h arm holding 0.8 %: a 58-fold ratio in hours rendered as a
# 2.1-fold ratio in ink, in a panel whose whole claim ("where the 590.4 hours
# went") is part-to-whole.
# Position on a log axis is not: a dot at 254.7 h sits a decade and a half right
# of a dot at 4.4 h because that is the ratio.  The hairline behind each dot is
# a leader from the arm's name to its mark, in the same apparatus grey as the
# rule under the note block, not a quantity; and every row still prints its
# hours and its share of the budget in figures.  Strictly less ink, inside the
# axes, at the same sixteen row positions: limits, pitch and the tight bounding
# box are all unchanged.
axB.hlines(yy, 0.15, srt.gpu_hours_total, color=RULE, lw=0.6, zorder=2)
for _y, _blk, _h in zip(yy, srt.block, srt.gpu_hours_total):
    axB.plot([_h], [_y], ls="none", marker="o", ms=5.8, mfc=BLK[_blk][0],
             mec=BLK[_blk][0], zorder=3)
# The S3 and S4 rows used to read "first+last" and "stride".  Neither strategy
# was ever run: S2 is sliding-window chunking with mean pooling, S3 the same
# chunking with learned attention pooling, S4 the same chunking with a
# chunk-level transformer above it -- configs/models/C2_finbert_s{2,3,4}.yaml
# and src/sp500vol/models/neural_text/bert_s{2,3,4}.py, and Section 3.4 and
# Appendix D say the same in prose.  The names below are those.  "(chunk)"
# alone is also dropped for "(chunk mean)", because with three chunking rows on
# the panel the pooling rule is what separates them.  These are y tick labels
# growing leftwards from x = 345 of a 1190 px render; the longest of them now
# starts at 49, and the figure's leftmost ink is the note block at 4, so the
# tick column stays inside the page the note already sets.
NAME = {"C4_longformer": "C4 Longformer", "C2_finbert_s2": "C2 FinBERT (chunk mean)",
        "C2_finbert_s3": "C2 FinBERT (chunk attn.)", "C2_finbert_s4": "C2 FinBERT (hierarchical)",
        "C1_bert_s2": "C1 BERT (chunk mean)", "C1_bert_s1": "C1 BERT (trunc.)",
        "C3_roberta_s1": "C3 RoBERTa (trunc.)", "C2_finbert_s1": "C2 FinBERT (trunc.)",
        "D2_gated_fusion": "D2 gated fusion", "D1_concat_mlp": "D1 concat MLP",
        "C5_e5mistral": "C5 E5-Mistral", "C5_gteqwen2": "C5 gte-Qwen2",
        # "C5 Qwen3", not "C5 Qwen3-emb.": the two neighbouring encoder/fusion
        # pairs name the same object twice ("C5 gte-Qwen2"/"D3 price +
        # gte-Qwen2", "C5 E5-Mistral"/"D3 price + E5-Mistral"), so the lone
        # suffix against "D3 price + Qwen3" read as a different model.  The
        # block letter already says these three are frozen embedders, and this
        # is the leftward-growing tick column, so a shorter string is the safe
        # direction.
        "C5_qwen3": "C5 Qwen3", "D3_gteqwen2": "D3 price + gte-Qwen2",
        "D3_e5mistral": "D3 price + E5-Mistral", "D3_qwen3": "D3 price + Qwen3"}
axB.set_yticks(yy)
axB.set_yticklabels([NAME[m] for m in srt.model], fontsize=8.9)
for y, (_, r) in zip(yy, srt.iterrows()):
    # The share of the budget is printed because no axis on this panel carries
    # it: the marks encode hours logarithmically, and the panel's claim is a
    # share-of-total claim.  These figures, not the marks, are what a reader
    # adds up.
    share = 100 * r.gpu_hours_total / total
    lab = (f"{r.gpu_hours_total:,.1f} h  ·  {share:.1f} %" if share >= 0.05
           else f"{r.gpu_hours_total:,.1f} h  ·  <0.1 %")
    xfrac = ((np.log10(r.gpu_hours_total) - np.log10(0.15))
             / (np.log10(5000) - np.log10(0.15)))
    if xfrac > 0.65:                       # only the 254.7 h arm reaches here
        # This row's mark stands at x = 949 of a 1190 px render and the rank
        # column starts at 1104, so the label cannot follow the dot outward:
        # set to the right it would run to about 1148 and collide.  It goes to
        # the LEFT of the dot instead, in the same grey as the other fifteen
        # rows -- the white-on-bar setting died with the bar -- with a white
        # stroke beneath it so the leader hairline cannot run through the
        # digits.  Leftward, into the empty half of its own row, and inside the
        # axes: nothing moves outward and the page box does not change.
        axB.text(r.gpu_hours_total / 1.15, y, lab, va="center", ha="right",
                 fontsize=8.9, color=GREY, zorder=4,
                 path_effects=[pe.withStroke(linewidth=2.4,
                                             foreground="white")])
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


# "and it is free" claimed more than the panel measures: the x-axis is
# GPU-hours, and the note below records that the block A and B arms ran on CPU
# with their wall-clock seconds accounted separately.  "at zero GPU cost" is
# exactly what the axis says.  Two characters longer, on a heading whose ink
# ends at x = 837 of a 1190 px render, so the tight bounding box is untouched.
heading(axA, "(a)  Accuracy against cost: the efficiency frontier is one "
             "point, at zero GPU cost")
# The old (b) heading promised ranks "out of 25" over sixteen rows, so the nine
# ranks that never appear (1-5, the block A arms, and 19, 21, 22, 25, the block
# B ones) read as missing data rather than as arms that spent no GPU time.
# Naming the sixteen inside the heading is what the panel actually shows; the
# denominator 25 stays because the ranks are ranks among all 25.  Eleven
# characters longer, from x = 896 to about x = 1035, inside the 1185 px the
# figure's rightmost ink already sets.
heading(axB, f"(b)  Where the {total:,.1f} GPU-hours went: the {len(paid)} "
             f"arms with GPU time, accuracy rank out of 25")

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
