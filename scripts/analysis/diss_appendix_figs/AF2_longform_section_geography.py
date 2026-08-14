"""AF2 -- where inside a long-form filing the apparent increment lives.

Four TF-IDF+ridge variants trained on a single span of each 10-K/10-Q (or on the
complement of the three named spans), replicating the archived B2 recipe exactly,
against the same recalibrated-HAR reference as the primary combination grid.

Sources (every plotted number is read from this file in this run):
  results/tables/section_ablation.csv
      model_id, section, horizon_days, n_test, frac_nonempty_all,
      frac_nonempty_10K, frac_nonempty_10Q, qlike_test_var, qlike_vs_B2_pct,
      m1_rel_improve_pct, m1_g_text, m1_dm_stat, m1_dm_p

Dissertation sentences this must not contradict:
  chapters/04_results.tex:86   the grid's largest increment, +5.92% (TF-IDF
      ridge, long-form, h = 20)
  chapters/05_validation.tex:49  "+3.33/+3.48/+5.92 per cent" for the same arm
  appendices/C_full_results.tex  long-form supports 7,951 / 7,933 / 7,902

Basis this figure must declare on its face: long-form panel only; single seed
2026; observation-level DM, not the report's day-clustered primary; no Holm
family; the reference is the recalibrated HAR alone -- the FIRST rung of the
ladder, with no firm-identity term and no maximal price pool -- so every bar is
an APPARENT increment of exactly the kind the ladder goes on to dissolve.

CPU only; no model is refitted.
"""
import os
import sys

import numpy as np
import pandas as pd

ANALYSIS = "scripts/analysis"
sys.path.insert(0, ANALYSIS)
import diss_style as ds
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from supp_style import (
    BLUE,
    GREEN,
    GREY,
    INK2,
    LIGHT,
    RULE,
    SKY,
    TAB,
    VERM,
    VERM_TXT,
    annot,
    apply_style,
    gate,
)

# --------------------------------------------------------------- evidence
D = pd.read_csv(os.path.join(TAB, "section_ablation.csv"))
HS = (5, 10, 20)

FULL = "B2_tfidf_ridge"
SPANS = ["B2sec_item7", "B2sec_item1a", "B2sec_rest", "B2sec_item7a"]
LAB = {
    FULL: "full text",
    "B2sec_item7": "MD&A",
    "B2sec_item1a": "Risk Factors",
    "B2sec_rest": "remainder",
    "B2sec_item7a": "Item 7A",
}
COL = {FULL: GREY, "B2sec_item7": BLUE, "B2sec_item1a": SKY,
       "B2sec_rest": GREEN, "B2sec_item7a": VERM}
HATCH = {FULL: "", "B2sec_item7": "", "B2sec_item1a": "///",
         "B2sec_rest": "\\\\\\", "B2sec_item7a": "xxx"}
DISJOINT = ["B2sec_item1a", "B2sec_item7", "B2sec_rest"]


def col(mid, name):
    s = D[D.model_id == mid].sort_values("horizon_days")
    return s[name].to_numpy()


full_m1 = col(FULL, "m1_rel_improve_pct")
repro = col("B2sec_fullrepro", "qlike_vs_B2_pct")
cover = {m: float(col(m, "frac_nonempty_all")[0]) for m in SPANS}

# --------------------------------------------------------------- the gate
gate(
    {
        "n_rows": 18,
        "n_variants": 6,
        "n_test": [7951, 7933, 7902],
        "full_text_m1": [3.33, 3.48, 5.92],
        "sanity_repro_max_abs_pct": 0.01,
        "item7a_coverage": 0.2461,
        "item7a_coverage_10q": 0.0,
        "item7a_m1_negative_cells": 3,
        "spans_adding_all_horizons": 3,
        "rest_standalone_better_cells": 3,
    },
    {
        "n_rows": len(D),
        "n_variants": int(D.model_id.nunique()),
        "n_test": [int(v) for v in col(FULL, "n_test")],
        "full_text_m1": [round(float(v), 2) for v in full_m1],
        "sanity_repro_max_abs_pct": round(float(np.abs(repro).max()), 2),
        "item7a_coverage": round(cover["B2sec_item7a"], 4),
        "item7a_coverage_10q": round(
            float(col("B2sec_item7a", "frac_nonempty_10Q")[0]), 4),
        "item7a_m1_negative_cells": int(
            (col("B2sec_item7a", "m1_rel_improve_pct") < 0).sum()),
        "spans_adding_all_horizons": int(sum(
            bool((col(m, "m1_rel_improve_pct") > 0).all()
                 and (col(m, "m1_dm_stat") < 0).all()) for m in DISJOINT)),
        "rest_standalone_better_cells": int(
            (col("B2sec_rest", "qlike_vs_B2_pct") < 0).sum()),
    },
)

# --------------------------------------------------------------- geometry
apply_style(9)
W, H = 6.10, 6.44
fig = plt.figure(figsize=(W, H))
LINE = 0.152


def rect(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


A_L, A_BOT, A_HGT = 0.52, 3.88, 1.90
A_W = W - A_L - 0.16

B_L, B_BOT, B_HGT = 0.52, 1.74, 1.42
B_W = 2.28
C_L = 3.62
C_W = W - C_L - 0.16

# ------------------------------------------------ panel (a) the increments
axA = fig.add_axes(rect(A_L, A_BOT, A_W, A_HGT))
order = [FULL] + SPANS
nb = len(order)
bw = 0.80 / nb
axA.axhline(0, color=GREY, lw=0.7, zorder=4)
for j, mid in enumerate(order):
    v = col(mid, "m1_rel_improve_pct")
    x = np.arange(3) + (j - (nb - 1) / 2) * bw
    axA.bar(x, v, width=bw * 0.92, color=COL[mid], edgecolor=GREY, lw=0.5,
            hatch=HATCH[mid], zorder=3)
    for xi, vi in zip(x, v, strict=False):
        axA.text(xi, vi + (0.16 if vi >= 0 else -0.16), f"{vi:+.2f}",
                 ha="center", va="bottom" if vi >= 0 else "top", fontsize=8.6,
                 color=VERM_TXT if vi < 0 else GREY, rotation=90, zorder=6)
axA.set_xticks(np.arange(3))
axA.set_xticklabels([f"$h$ = {h}" for h in HS])
axA.tick_params(axis="x", length=0, pad=3)
axA.set_xlim(-0.62, 2.62)
axA.set_ylim(-5.6, 8.4)
axA.set_yticks([-4, -2, 0, 2, 4, 6])
axA.set_ylabel("QLIKE improvement (%)", fontsize=9, labelpad=2)
axA.yaxis.grid(True, color=LIGHT, lw=0.5, zorder=0)
axA.set_axisbelow(True)

fig.legend(handles=[Patch(facecolor=COL[m], edgecolor=GREY, lw=0.5,
                          hatch=HATCH[m], label=LAB[m]) for m in order],
           ncol=5, loc="lower center",
           bbox_to_anchor=(0.53, (A_BOT + A_HGT + 0.055) / H),
           handlelength=1.2, handletextpad=0.45, columnspacing=1.3,
           borderpad=0.0, fontsize=9)
fig.text(0.012, (A_BOT + A_HGT + 0.315) / H,
         # Not "every span": Item 7A is negative at all three horizons
         # (-1.42/-2.11/-3.63), which the generator's own gate records as
         # item7a_m1_negative_cells = 3.  src: results/tables/section_ablation.csv
         "(a)  Three spans add over the recalibrated HAR; none recovers the "
         "whole filing; Item 7A subtracts",
         ha="left", va="center", fontsize=9.5, color=GREY, weight="bold")

# --------------------------------------------- panel (b) the additivity check
axB = fig.add_axes(rect(B_L, B_BOT, B_W, B_HGT))
axB.axhline(0, color=GREY, lw=0.7, zorder=4)
for i, h in enumerate(HS):
    axB.bar(i - 0.20, full_m1[i], width=0.34, color=GREY, edgecolor=GREY,
            lw=0.5, zorder=3)
    base = 0.0
    for mid in DISJOINT:
        v = float(col(mid, "m1_rel_improve_pct")[i])
        axB.bar(i + 0.20, v, bottom=base, width=0.34, color=COL[mid],
                edgecolor=GREY, lw=0.5, hatch=HATCH[mid], zorder=3)
        base += v
    axB.text(i + 0.20, base + 0.22, f"{base:.2f}", ha="center", va="bottom",
             fontsize=8.6, color=GREY, zorder=6)
    axB.text(i - 0.20, full_m1[i] + 0.22, f"{full_m1[i]:.2f}", ha="center",
             va="bottom", fontsize=8.6, color=GREY, zorder=6)
axB.set_xticks(np.arange(3))
axB.set_xticklabels([f"{h}" for h in HS])
axB.set_xlabel("horizon $h$ (trading days)", fontsize=9, labelpad=2)
axB.tick_params(axis="x", length=0, pad=3)
axB.set_xlim(-0.62, 2.62)
axB.set_ylim(0, 9.6)
axB.set_yticks([0, 2, 4, 6, 8])
axB.set_ylabel("QLIKE improvement (%)", fontsize=9, labelpad=2)
axB.yaxis.grid(True, color=LIGHT, lw=0.5, zorder=0)
axB.set_axisbelow(True)
fig.text(0.012, (B_BOT + B_HGT + 0.315) / H,
         "(b)  The parts sum to more than the whole",
         ha="left", va="center", fontsize=9.5, color=GREY, weight="bold")
fig.text(0.012, (B_BOT + B_HGT + 0.105) / H,
         "left: full text;  right: the three spans stacked",
         ha="left", va="center", fontsize=8.6, color=INK2)

# ------------------------------------------ panel (c) standalone forecasters
axC = fig.add_axes(rect(C_L, B_BOT, C_W, B_HGT))
axC.axhline(0, color=GREY, lw=0.9, zorder=4)
nb2 = len(SPANS)
bw2 = 0.80 / nb2
for j, mid in enumerate(SPANS):
    v = col(mid, "qlike_vs_B2_pct")
    x = np.arange(3) + (j - (nb2 - 1) / 2) * bw2
    axC.bar(x, v, width=bw2 * 0.92, color=COL[mid], edgecolor=GREY, lw=0.5,
            hatch=HATCH[mid], zorder=3)
axC.set_xticks(np.arange(3))
axC.set_xticklabels([f"{h}" for h in HS])
axC.set_xlabel("horizon $h$ (trading days)", fontsize=9, labelpad=2)
axC.tick_params(axis="x", length=0, pad=3)
axC.set_xlim(-0.62, 2.62)
axC.set_ylim(-6.5, 17.5)
axC.set_yticks([-5, 0, 5, 10, 15])
axC.set_ylabel("test QLIKE vs full text (%)", fontsize=9, labelpad=2)
axC.yaxis.grid(True, color=LIGHT, lw=0.5, zorder=0)
axC.set_axisbelow(True)
# These two are axis-direction aids, not data: recessive ink, and moved out of
# the marks.  At x = 0.02 the lower one ran straight through the h = 5 remainder
# bar (-4.29, the deepest bar in the panel).  Both corners on the right are
# empty -- the tallest bar at h = 20 is +8.82 against a +17.5 top, and below
# -3.6 only the h = 5 remainder bar reaches, far to the left -- so right
# alignment clears the data at both ends and keeps the pair on one edge.  The
# halo is belt-and-braces; nothing is drawn under them now.
annot(axC, 0.98, 0.965, "worse than full text", size=8.6, color=INK2,
      transform=axC.transAxes, ha="right", va="top")
annot(axC, 0.98, 0.035, "better than full text", size=8.6, color=INK2,
      transform=axC.transAxes, ha="right", va="bottom")
fig.text(C_L / W - 0.012, (B_BOT + B_HGT + 0.315) / H,
         "(c)  Worse standalone forecasters",
         ha="left", va="center", fontsize=9.5, color=GREY, weight="bold")
fig.text(C_L / W - 0.012, (B_BOT + B_HGT + 0.105) / H,
         "variance-unit test QLIKE vs full text",
         ha="left", va="center", fontsize=8.6, color=INK2)

# --------------------------------------------------------------- footnote
foot = [
    "Basis. Long-form panel only, 7,951 / 7,933 / 7,902 test rows at h = "
    "5/10/20; single",
    "seed 2026; observation-level DM, not the report's day-clustered primary, "
    "and no Holm",
    "family --- an off-basis diagnostic. The reference is the recalibrated HAR "
    "alone, the",
    "FIRST rung of the ladder, so these are apparent increments of the kind the "
    "firm-",
    "identity and maximal-pool rungs dissolve. Non-empty coverage: MD&A "
    "{i7:.1%}, Risk".format(i7=cover["B2sec_item7"]),
    "Factors {i1a:.1%}, remainder 100%, Item 7A {i7a:.1%} (10-K only, 0% of "
    "10-Qs), whose negative".format(i1a=cover["B2sec_item1a"],
                                    i7a=cover["B2sec_item7a"]),
    "bars are an artefact of a three-quarters-empty text column rather than "
    "evidence about",
    "Item 7A content. The recipe reproduces the archived run to within 0.01% of "
    "test QLIKE.",
]
# The block was unframed prose in the same ink as the panel titles and the data
# labels, so nothing told a reader which lines are the argument and which are the
# basis statement.  It becomes recessive ink under a hairline instead.  Not
# note(): that helper hardcodes linespacing=1.32, and this block is set as one
# fig.text per line at LINE = 0.152 in (1.27 for 8.6 pt), which must not tighten.
#
# GEOMETRY.  finish() writes bbox_inches="tight" and the audit reports this
# figure binding on WIDTH at scale 1.0704, so width is the printed-size budget
# and height is nearly free (465 graphic pt against a ~595 pt cap).  Measured
# extents at 6.10 x 6.44 in: panel (b)'s x-label bottom is 1.404 in, the widest
# footnote line ends at 4.887 in, and the right-most content in the whole figure
# is the panel axes' right edge at 5.94 in.  So the block drops 0.10 in (bottom-
# most ink 0.159 -> 0.059 in, still inside the canvas; height grows 7 pt, printed
# size unchanged) to open the gap, and the rule spans 0.012 -> 5.94/W, i.e. out
# to existing content and not one point past it.  Widening would cost every
# glyph in the figure.
FOOT_TOP = 1.18
FOOT_RULE_Y = 1.33
fig.add_artist(Line2D([0.012, 5.94 / W], [FOOT_RULE_Y / H, FOOT_RULE_Y / H],
                      transform=fig.transFigure, color=RULE, linewidth=0.5,
                      zorder=0.5))
for k, t in enumerate(foot):
    fig.text(0.012, (FOOT_TOP - LINE * k) / H, t, ha="left", va="center",
             fontsize=8.6, color=INK2)

ds.finish(fig, "AF2_longform_section_geography",
          note="section spans of the long-form filing at the first ladder rung")
