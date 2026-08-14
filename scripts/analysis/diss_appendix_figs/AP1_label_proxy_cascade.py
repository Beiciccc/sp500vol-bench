"""Appendix figure AP1 — the label-proxy relabelling, whole.

Section 5.2 reports the range-based rerun as a count-weakening perturbation
(19 genuine against 38, conjunction 0 -> 1).  Three things the committed
evidence holds that the count alone cannot convey:

  (a) the rung-by-rung counts under all three label proxies side by side --
      the primary and identity rungs FALL while the maximal-pool rung RISES,
      so this is a re-ordering of the ladder and not a uniform weakening;
  (b) what happens to each of the 69 cells' primary increment, committed
      labels against Parkinson, with the lone conjunction survivor named --
      the exception to the study's central negative;
  (c) that survivor's own descent down the three rungs, against the cell's
      own minimum detectable effect: +2.51 % -> +1.13 % -> +0.016 %.

Panel (d) carries the branch-(d) reference-ordering check, the one leg of
this perturbation that STRENGTHENS the design: recalibrated HAR is rank-1
among the five single price references in 0 of 6 panels on the committed
labels and 3 of 6 under Parkinson.

Sources
-------
results/tables/rangebased_cascade.csv   69 cells x {old, pk, gk} x rungs
results/tables/rangebased_cascade.md    rung/MDE/injection summary blocks,
                                        branch-(d) rank check
"""
import os
import sys
import textwrap

import pandas as pd

ANALYSIS = "scripts/analysis"
sys.path.insert(0, ANALYSIS)

from supp_style import (apply_style, gate, BLUE, SKY, VERM, VERM_TXT,  # noqa: E402
                        GREEN, YELLOW, GREY, LIGHT, TAB, INK, INK2, RULE)
import diss_style as ds  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- evidence
d = pd.read_csv(os.path.join(TAB, "rangebased_cascade.csv"))

# Committed counts the report states; the build aborts if the table moved.
gate(
    {"old_genuine": 38, "pk_genuine": 19, "gk_genuine": 21,
     "old_conj": 0, "pk_conj": 1, "gk_conj": 2, "n_cells": 69},
    {"old_genuine": int(d.old_genuine.sum()),
     "pk_genuine": int(d.pk_genuine.sum()),
     "gk_genuine": int(d.gk_genuine.sum()),
     "old_conj": int(d.old_conj.sum()),
     "pk_conj": int(d.pk_conj.sum()),
     "gk_conj": int(d.gk_conj.sum()),
     "n_cells": len(d)},
)

def rung(pre, col):
    """Holm-detected cells at one rung under one label proxy."""
    return int(d[f"{pre}_{col}_detect"].sum())


# The committed primary rung is placebo-gated; the strengthened rungs are
# Holm-only, exactly as the cascade table reports them.
counts = {
    "Committed": [int(d.old_genuine.sum()), rung("old", "firm"),
                  rung("old", "pool"), int(d.old_conj.sum())],
    "Parkinson": [int(d.pk_genuine.sum()), rung("pk", "firm"),
                  rung("pk", "pool"), int(d.pk_conj.sum())],
    "Garman--Klass": [int(d.gk_genuine.sum()), rung("gk", "firm"),
                      rung("gk", "pool"), int(d.gk_conj.sum())],
}
# The .md's committed summary block, re-derived here, must agree.
gate({"pk_firm": 7, "pk_pool": 15, "gk_firm": 8, "gk_pool": 15,
      "old_firm": 8, "old_pool": 9},
     {"pk_firm": counts["Parkinson"][1], "pk_pool": counts["Parkinson"][2],
      "gk_firm": counts["Garman--Klass"][1],
      "gk_pool": counts["Garman--Klass"][2],
      "old_firm": counts["Committed"][1], "old_pool": counts["Committed"][2]})

surv = d[d.pk_conj].iloc[0]
gate({"survivor": "event_driven/C2_finbert_s1/5"},
     {"survivor": f"{surv.disc}/{surv.model}/{surv.h}"})

apply_style(9.4)  # drawn above 9 so the width-bound inclusion scale still prints >= 9 pt
fig = plt.figure(figsize=(5.92, 7.62))  # width pinned so the tight bbox stays <= 455.24pt (the 9pt floor is scale-sensitive)
gs = fig.add_gridspec(
    3, 2, height_ratios=[1.00, 1.16, 0.80], width_ratios=[1.34, 1.0],
    left=0.108, right=0.988, top=0.945, bottom=0.215, hspace=0.72, wspace=0.62)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, :])
axC = fig.add_subplot(gs[2, 0])
axD = fig.add_subplot(gs[2, 1])

for ax in (axA, axB, axC, axD):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ------------------------------------------------------ (a) rungs by proxy
RUNGS = ["Primary rung\n(placebo-gated)", "Firm-identity\nreference",
         "Maximal price\npool", "Full\nconjunction"]
SER = [("Committed", BLUE, ""), ("Parkinson", VERM, "///"),
       ("Garman--Klass", YELLOW, "...")]
bw = 0.26
for j, (name, col, hatch) in enumerate(SER):
    xs = [i + (j - 1) * bw for i in range(4)]
    axA.bar(xs, counts[name], width=bw, color=col, hatch=hatch,
            edgecolor="white", linewidth=0.6, label=name.replace("--", "–"),
            zorder=3)
    for x, v in zip(xs, counts[name]):
        axA.text(x, v + 0.9, str(v), ha="center", va="bottom", fontsize=9.7,
                 color=GREY, zorder=4)
axA.set_xticks(range(4))
axA.set_xticklabels(RUNGS, fontsize=9.7)
axA.set_ylabel("cells detected\nof 69", fontsize=9.9)
axA.set_ylim(0, 56)
axA.tick_params(axis="y", labelsize=9.7)
axA.grid(axis="y", color=LIGHT, linewidth=0.6, zorder=0)
axA.set_axisbelow(True)
axA.legend(frameon=False, fontsize=9.6, ncol=3, loc="upper right",
           handlelength=1.5, columnspacing=1.3, borderaxespad=0.1)
# The callout is about the rung as a whole, so it must NOT wear a series
# colour: vermillion is the Parkinson proxy in this panel and in (b) and (d),
# and a vermillion sentence here reads as if it belonged to those bars alone.
# Its arrow also used to land on the pool group's own "15" data label; it now
# meets the right flank of that bar, below the labels.
axA.annotate("the pool rung RISES\nwhile the others fall",
             xy=(2 + 1.5 * bw, 9.6),
             xytext=(2.30, 30), fontsize=9.7, color=INK, ha="left",
             arrowprops=dict(arrowstyle="->", color=INK, lw=0.8,
                             shrinkA=1, shrinkB=2))

# ------------------------------- (b) every cell's primary increment, o -> pk
dd = d.sort_values("old_rel").reset_index(drop=True)
xs = range(len(dd))
axB.axhline(0, color=GREY, linewidth=0.7, zorder=2)
for x, (_, r) in zip(xs, dd.iterrows()):
    axB.plot([x, x], [r.old_rel, r.pk_primary_rel], color=LIGHT,
             linewidth=0.9, zorder=1, solid_capstyle="butt")
axB.scatter(xs, dd.old_rel, s=13, color=BLUE, zorder=3, linewidths=0,
            label="committed labels")
axB.scatter(xs, dd.pk_primary_rel, s=13, color=VERM, marker="D", zorder=3,
            linewidths=0, label="Parkinson labels")
k = int(dd.index[dd.pk_conj][0])
axB.scatter([k], [dd.pk_primary_rel[k]], s=76, facecolors="none",
            edgecolors=GREEN, linewidths=1.5, zorder=5)
axB.annotate("FinBERT 8-K $h$=5 — the lone\nconjunction survivor (+2.51 %)",
             xy=(k, dd.pk_primary_rel[k]), xytext=(max(k - 21, 1.0), 4.55),
             fontsize=9.7, color=GREEN, ha="left",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9,
                             shrinkA=1, shrinkB=5))
axB.set_xlim(-1.5, len(dd) + 0.5)
axB.set_ylim(-6.9, 6.6)
axB.set_xlabel("the 69 combination cells, ordered by their committed increment",
               fontsize=9.8)
axB.set_ylabel("increment over the recalibrated\nHAR reference (%)", fontsize=9.9)
axB.set_xticks([])
axB.tick_params(axis="y", labelsize=9.7)
axB.legend(frameon=False, fontsize=9.6, loc="lower right", handletextpad=0.35,
           borderaxespad=0.15)

# ------------------------------------------ (c) the survivor down the rungs
labels = ["over recalibrated\nHAR", "over maximal\nprice pool",
          "over firm-identity\nreference"]
vals = [surv.pk_primary_rel, surv.pk_pool_rel, surv.pk_firm_rel]
ys = [2, 1, 0]
# All three bars are the same cell's increment under the SAME (Parkinson)
# labels, so they take the Parkinson series colour used in (a), (b) and (d).
# The third bar used to be green, which in this figure means "the lone
# conjunction survivor" -- the ring and callout in (b) -- so green on the
# rung where that survivor collapses to a near-zero below its own MDE read as
# the opposite of what the panel shows.
axC.barh(ys, vals, height=0.52, color=[VERM, VERM, VERM], zorder=3,
         edgecolor="white", linewidth=0.6)
axC.axvline(surv.pk_mde, color=GREY, linewidth=0.9, linestyle=(0, (3, 2)),
            zorder=4)
axC.text(surv.pk_mde + 0.08, 2.62, f"MDE {surv.pk_mde:.2f} %",
         fontsize=9.7, color=GREY, ha="left", va="center")
for y, v in zip(ys, vals):
    inside = v > 0.9
    axC.text(v - 0.09 if inside else v + 0.075, y, f"{v:+.3f} %",
             fontsize=9.7, va="center", ha="right" if inside else "left",
             color="white" if inside else INK, zorder=5)
axC.set_yticks(ys)
axC.set_yticklabels(labels, fontsize=9.7)
axC.set_xlim(0, 3.05)
axC.set_ylim(-0.62, 3.05)
axC.set_xlabel("increment (%)", fontsize=9.8)
axC.tick_params(axis="x", labelsize=9.7)
axC.grid(axis="x", color=LIGHT, linewidth=0.6, zorder=0)
axC.set_axisbelow(True)

# ------------------------------------- (d) branch-(d) reference ordering
# Committed summary, results/tables/rangebased_cascade.md branch-(d) block.
RANK = {"Committed": (3.67, 0), "Parkinson": (1.50, 3)}
axD.barh([1, 0], [RANK["Committed"][0], RANK["Parkinson"][0]], height=0.46,
         color=[BLUE, VERM], zorder=3, edgecolor="white", linewidth=0.6)
for y, k2 in zip([1, 0], ["Committed", "Parkinson"]):
    mean, first = RANK[k2]
    axD.text(mean + 0.10, y, f"{mean:.2f}", fontsize=9.7, va="center",
             color=GREY)
axD.set_yticks([1, 0])
axD.set_yticklabels([f"committed labels\nrank-1 in {RANK['Committed'][1]} of 6",
                     f"Parkinson labels\nrank-1 in {RANK['Parkinson'][1]} of 6"],
                    fontsize=9.7)
axD.set_xlim(0, 4.9)
axD.set_ylim(-0.62, 1.82)
axD.set_xticks([1, 2, 3, 4, 5])
axD.set_xlabel("mean rank among the 5 single\nprice references (1 = strongest)",
               fontsize=9.6)
axD.tick_params(axis="x", labelsize=9.7)
axD.grid(axis="x", color=LIGHT, linewidth=0.6, zorder=0)
axD.set_axisbelow(True)

fig.canvas.draw()


def heading(ax, text, dy, x=None):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.094 if x is None else x, pos.y1 + dy, text,
             fontsize=10.4, color=GREY, ha="left", va="bottom")


heading(axA, "(a)  The ladder under three label proxies, same 69 cells, same "
             "frozen text forecasts", 0.017, x=0.004)
heading(axB, "(b)  What relabelling does to each cell's primary increment",
        0.034, x=0.004)
heading(axC, "(c)  The Parkinson survivor, rung by rung", 0.016, x=0.004)
heading(axD, "(d)  Does relabelling unseat HAR?", 0.016, x=0.575)

NOTE = (
    "Price references, recalibration and combiner weights are refitted on the "
    "new labels; the text arms are NOT retrained -- an asymmetry conservative "
    "for text. In (a) the primary rung is placebo-gated and the two "
    "strengthened rungs are Holm-only. In (c) the MDE is that cell's own, at "
    "80 % power; its +0.016 % over the identity reference is detected at "
    "DM -5.90 and is a precisely estimated near-zero, not a large gain."
)
# The apparatus block: every word of it is kept, at the same size, but in the
# recessive ink and under a hairline, so the reader can see at a glance that
# these are the figure's basis statements and not its data labels. The rule is
# placed midway between the block and the lowest axis label already drawn, i.e.
# strictly inside the existing tight bbox -- nothing here widens the page.
note_txt = fig.text(0.004, 0.006, textwrap.fill(NOTE, 100), fontsize=9.7,
                    color=INK2, ha="left", va="bottom", linespacing=1.32)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
nb = note_txt.get_window_extent(rend).transformed(inv)
low = min(a.xaxis.get_label().get_window_extent(rend).transformed(inv).y0
          for a in (axC, axD))
y_rule = 0.5 * (nb.y1 + low)
# Butt cap and a hair of inset: the note's longest line is what sets the right
# edge of the tight bbox, and a projecting cap on this rule would push that edge
# out by a quarter point -- enough to lower the printed type size.
fig.lines.append(plt.Line2D([0.004, nb.x1 - 0.006], [y_rule, y_rule],
                            transform=fig.transFigure, color=RULE,
                            linewidth=0.5, solid_capstyle="butt", zorder=0.5))

ds.finish(fig, "AP1_label_proxy_cascade", max_render_pt=595.0,
          note="appendix figure: range-based relabelling -- rungs by proxy, "
               "per-cell increments, the lone conjunction survivor, and the "
               "branch-(d) reference-ordering check")
