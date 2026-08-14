"""Appendix figure — how much of a single-seed verdict is the seed.

Every fine-tuned text and fusion arm was trained three times (seeds 2026/2027/
2028) and the combination grid recomputed from scratch for each.  This figure
draws the resulting dispersion over all 144 disclosure-by-model-by-horizon
cells, and then zooms on the fourteen cells that seed 2026 alone would have
declared genuine.

(a) dispersion against effect: the across-seed standard deviation of the text
    increment plotted on its across-seed mean.  Points above the diagonal wedge
    are cells whose seed-to-seed spread exceeds the increment they report.
(b) sign instability by model block: the share of cells whose DM statistic
    changes sign across the three seeds.  The frozen-embedding arms, whose only
    trained component is a small head, are the stable ones.
(c) the fourteen seed-2026 'genuine' cells, seed by seed.

Source: results/tables/m1_multiseed.csv
Out:    writing/dissertation/figures/stability_seed_dispersion.pdf
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, "scripts/analysis")
import supp_style  # noqa: E402
from supp_style import (apply_style, finish, gate, BLUE, SKY, VERM, VERM_TXT,  # noqa: E402
                        GREEN, YELLOW, PURPLE, GREY, LIGHT, TAB, REPO,
                        INK2, RULE)

supp_style.OUTDIR = os.path.join(REPO, "writing", "dissertation", "figures")

SHORT = {
    "C1_bert_s1": "BERT-1", "C1_bert_s2": "BERT-2", "C2_finbert_s1": "FinBERT-1",
    "C2_finbert_s2": "FinBERT-2", "C2_finbert_s3": "FinBERT-3",
    "C2_finbert_s4": "FinBERT-4", "C3_roberta_s1": "RoBERTa",
    "C4_longformer": "Longformer", "C5_e5mistral": "E5-Mistral",
    "C5_gteqwen2": "GTE-Qwen2", "C5_qwen3": "Qwen3-emb",
    "D1_concat_mlp": "Concat", "D2_gated_fusion": "Gated",
    "D3_e5mistral": "Fuse-E5", "D3_gteqwen2": "Fuse-GTE", "D3_qwen3": "Fuse-Qwen3",
}
DISC = {"long_form": "10-K/Q", "event_driven": "8-K", "combined": "both"}
SEEDS = [2026, 2027, 2028]

m = pd.read_csv(os.path.join(TAB, "m1_multiseed.csv"))
m["block"] = m.model.str.split("_").str[0]
m["gen26"] = m.grid2026_genuine.astype(str).eq("True")
wedge = m.rel_std > m.rel_mean.abs()
g = m[m.gen26].copy()

gate({"cells": 144, "sign_flips": 37, "genuine_2026": 14,
      "hold_3of3": 5, "hold_2of3": 12, "wedge": 40, "seeds": 3},
     {"cells": int(len(m)), "sign_flips": int(m.sign_disagree.sum()),
      "genuine_2026": int(len(g)), "hold_3of3": int((g.n_sig_holm == 3).sum()),
      "hold_2of3": int((g.n_sig_holm >= 2).sum()), "wedge": int(wedge.sum()),
      "seeds": int(m.n_seeds.max())})

# ------------------------------------------------------------------ canvas
apply_style(9)
fig = plt.figure(figsize=(6.10, 7.30))
# (a) and (b) sit 0.020 lower than they used to. The band above them held only
# 5.5 pt of white between the apparatus note and the (a) title -- not enough for
# the hairline that separates the two -- while the band between (a)/(b) and (c)
# held 38 pt doing nothing. Moving the two top panels down borrows from the
# slack band and changes no bounding box: the page is still pinned at the top by
# the note at y = 0.992, at the bottom by the note at y = 0.006, on the left by
# the (c) row labels and on the right by the (b) tick labels.
ax_a = fig.add_axes([0.125, 0.605, 0.500, 0.268])
ax_b = fig.add_axes([0.755, 0.605, 0.230, 0.268])
ax_c = fig.add_axes([0.240, 0.132, 0.660, 0.370])

# ----------------------------------------------- (a) dispersion vs effect
xmax = 7.0
xs = np.linspace(0, xmax, 200)
ax_a.fill_between(xs, np.abs(xs), 13, color=LIGHT, alpha=0.55, lw=0, zorder=0)
ax_a.fill_between(-xs, np.abs(xs), 13, color=LIGHT, alpha=0.55, lw=0, zorder=0)
ax_a.plot([-xmax, 0, xmax], [xmax, 0, xmax], color=GREY, lw=0.7,
          ls=(0, (4, 3)), zorder=1)
# The zero rule stops below the legend block: at full height it ran through
# both legend rows, and no cell near x = 0 carries a spread above 4.5 pp.
ax_a.plot([0, 0], [0, 5.4], color=GREY, lw=0.6, zorder=1)

for flag, col, mk, ms, lab in ((False, GREY, "o", 11, "seeds agree in sign"),
                               (True, VERM, "D", 15, "seeds disagree in sign")):
    sub = m[m.sign_disagree == flag]
    ax_a.scatter(sub.rel_mean, sub.rel_std, s=ms, marker=mk, facecolors=col,
                 edgecolors=col, linewidths=0.6, alpha=0.85, zorder=3,
                 label=lab)

ax_a.set_xlim(-9.5, 6.5)
ax_a.set_ylim(0, 7.0)
ax_a.set_xlabel("mean increment across the three seeds (%)", fontsize=9)
ax_a.set_ylabel("s.d. across seeds (pp)", fontsize=9)
ax_a.set_title("(a) 144 cells: how the spread compares with the effect",
               fontsize=9, color=GREY, pad=4, loc="left")
# Three short lines in the empty left flank, below the wedge boundary: the old
# two-line block sat across the boundary, which struck out "shaded:" and "40 of
# 144", and its right edge ran into the cell at (-7.95, 6.08).
# INK2, not INK: this block describes the instrument (what the shading means and
# over what denominator), so it should not read at the same strength as the
# points and the axis labels it sits beside.
ax_a.text(-9.3, 4.05, f"shaded: spread\nexceeds effect\n"
                      f"({int(wedge.sum())} of 144)",
          fontsize=9, color=INK2, va="top", ha="left", linespacing=1.15)
# borderaxespad pulls the legend clear of the right wedge boundary, which used
# to cross the "sign" of the second entry.
ax_a.legend(loc="upper right", fontsize=9, handletextpad=0.3, borderpad=0.2,
            labelspacing=0.25, borderaxespad=0.9, scatterpoints=1)

# ------------------------------------------------ (b) flips by model block
BLK = m.groupby("block").agg(n=("h", "size"), flips=("sign_disagree", "sum"))
BLK["pct"] = 100 * BLK.flips / BLK.n
BLK = BLK.sort_values("pct")
ypos = np.arange(len(BLK))
cols = [SKY if b in ("C5", "D3") else VERM for b in BLK.index]
ax_b.barh(ypos, BLK.pct, height=0.66, color=cols, edgecolor="none")
for y, (b, r) in zip(ypos, BLK.iterrows()):
    ax_b.text(r.pct + 2.5, y, f"{int(r.flips)}/{int(r.n)}", fontsize=9,
              color=SKY if b in ("C5", "D3") else VERM_TXT, va="center")
ax_b.set_yticks(ypos)
ax_b.set_yticklabels(BLK.index, fontsize=9)
for lab, b in zip(ax_b.get_yticklabels(), BLK.index):
    lab.set_color(SKY if b in ("C5", "D3") else GREY)
ax_b.set_xlim(0, 100)
ax_b.set_xticks([0, 25, 50, 75])
ax_b.set_xticklabels(["0", "25", "50", "75%"])
ax_b.set_xlabel("cells flipping\nthe DM sign", fontsize=9, linespacing=1.15)
ax_b.set_title("(b) by block", fontsize=9, color=GREY, pad=4, loc="left")
ax_b.set_ylim(-0.7, len(BLK) - 0.3)
ax_b.tick_params(axis="y", length=0)
for s in ("left", "right", "top"):
    ax_b.spines[s].set_visible(False)

# --------------------------------- (c) the fourteen seed-2026 genuine cells
g = g.sort_values("rel_mean").reset_index(drop=True)
yy = np.arange(len(g))
# The zero rule stops below the count statement, the same way (a)'s stops below
# its legend: as a full-height axvline it struck through "12 in at least two",
# which is the one real collision in this panel. It still spans every data row.
# len(g) - 0.60 is 0.40 above the TOP ROW's centre, which is len(g) - 1 = 13, not
# above the view ceiling: rows are indexed 0..13, so len(g) + 0.60 would have left
# the rule inside the statement it was meant to clear.
ax_c.plot([0, 0], [-3.5, len(g) - 0.60], color=GREY, lw=0.6, zorder=2)
for i, r in g.iterrows():
    vals = [r[f"rel_impr_pct_{s}"] for s in SEEDS]
    ax_c.plot([min(vals), max(vals)], [i, i], color=LIGHT, lw=2.6,
              solid_capstyle="round", zorder=1)
    for s, mk in zip(SEEDS, ("o", "s", "^")):
        v = r[f"rel_impr_pct_{s}"]
        holm_ok = (r[f"p_holm_{s}"] < 0.05) and (r[f"dm_q_{s}"] < 0)
        ax_c.plot([v], [i], ls="none", marker=mk, ms=5.0,
                  mfc=BLUE if holm_ok else "white",
                  mec=BLUE if holm_ok else VERM, mew=0.9, zorder=3)

labels = [f"{SHORT[r.model]} {DISC[r.disc]} h={r.h}" for _, r in g.iterrows()]
ax_c.set_yticks(yy)
ax_c.set_yticklabels(labels, fontsize=9)
# Top of the view was len(g) + 0.35, which left 1.4 pt between the count
# statement below the title and the markers of the top row, so the statement read
# as glued to the data. Raising the ceiling inside the same axes rectangle opens
# that gap to about 7 pt and moves no boundary of the page.
ax_c.set_ylim(-3.5, len(g) + 0.95)
ax_c.set_xlim(-9.7, 7.6)
ax_c.set_xlabel("text increment over the recalibrated price reference (%)",
                fontsize=9)
ax_c.set_title("(c) the 14 cells seed 2026 alone would have called genuine",
               fontsize=9, color=GREY, pad=4, loc="left")
ax_c.tick_params(axis="y", length=0)
ax_c.grid(axis="y", color=LIGHT, lw=0.4, alpha=0.7)
ax_c.set_axisbelow(True)

for i, r in g.iterrows():
    ax_c.text(7.5, i, f"{int(r.n_sig_holm)}/3", fontsize=9, ha="right",
              va="center", color=BLUE if r.n_sig_holm == 3 else GREY)

hand = [Line2D([], [], ls="none", marker=mk, ms=4.6, mfc=BLUE, mec=BLUE,
               label=f"seed {s}") for s, mk in zip(SEEDS, ("o", "s", "^"))]
hand.append(Line2D([], [], ls="none", marker="o", ms=4.6, mfc="white",
                   mec=VERM, mew=0.9, label="not Holm-significant, that seed"))
ax_c.legend(handles=hand, loc="lower left", fontsize=9, handletextpad=0.35,
            frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
            labelspacing=0.26, borderpad=0.25, borderaxespad=0.35, ncol=2,
            columnspacing=1.0)

# A basis statement, in INK2, so the title above it keeps primacy: at one ink and
# one size the two lines read as a wrapped two-line title rather than as a title
# plus a count over the panel's own denominator.
ax_c.text(-9.5, len(g) + 0.88,
          f"{int((g.n_sig_holm == 3).sum())} of the 14 hold in all three seeds; "
          f"{int((g.n_sig_holm >= 2).sum())} in at least two",
          fontsize=9, color=INK2, va="top", ha="left")

# ------------------------------------------------------------------ notes
# Both blocks keep their wording, their y positions and their 0.022 line pitch
# (note() would have retightened it to linespacing 1.32), and change only ink and
# separation: INK2 plus one hairline on the figure side of each block. Before
# this, the apparatus prose, the panel titles and the data labels were all one
# ink at one size, so nothing told a reader which sentence was the argument and
# which was the basis statement. The hairlines sit in white that already existed,
# so the bounding box -- and with it the printed point size -- is untouched.
NOTE_X0, NOTE_X1 = 0.125, 0.935


def _hair(y):
    fig.lines.append(plt.Line2D([NOTE_X0, NOTE_X1], [y, y],
                                transform=fig.transFigure, color=RULE,
                                linewidth=0.5, zorder=0.5))


fig.text(0.125, 0.992,
         "Seed-ensemble averaging is the study's answer to what this figure "
         "shows. Panels (a)",
         fontsize=9, color=INK2, va="top", ha="left")
fig.text(0.125, 0.970,
         "and (b) cover all 144 single-seed cells (48 per disclosure panel, "
         "including the",
         fontsize=9, color=INK2, va="top", ha="left")
fig.text(0.125, 0.948,
         "combined panel); price and classical arms are seed-invariant and "
         "are excluded.",
         fontsize=9, color=INK2, va="top", ha="left")
_hair(0.914)
_hair(0.064)
fig.text(0.125, 0.050,
         "In (b), blue marks the frozen-embedding arms, whose only trained "
         "component is a",
         fontsize=9, color=INK2, va="top", ha="left")
fig.text(0.125, 0.028,
         "small head. Holm runs within each seed's own family, so a cell can "
         "move for either",
         fontsize=9, color=INK2, va="top", ha="left")
fig.text(0.125, 0.006,
         "reason: its own statistic moved, or its neighbours' did.",
         fontsize=9, color=INK2, va="top", ha="left")

finish(fig, "stability_seed_dispersion")
print("flips", int(m.sign_disagree.sum()), "of", len(m),
      "| wedge", int(wedge.sum()))
print(BLK.to_string())
print("genuine2026 n_sig_holm:", g.n_sig_holm.value_counts().to_dict())
