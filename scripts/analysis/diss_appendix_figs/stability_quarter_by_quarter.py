"""Appendix figure — the sixteen-quarter path of the text increment, cell by cell.

Every one of the 69 primary combination cells is drawn against every one of the
16 test quarters (2022Q1-2025Q4), under both weight schemes:

  * FIXED     -- combiner weights fit once on the 2020-21 validation block and
                 frozen across the whole test span (the study's primary recipe);
  * EXPANDING -- weights refit before each quarter on strictly earlier filings
                 only (the deployable pseudo-OOS path).

The point of drawing all 1,104 quarter-cells rather than a leaderboard is that
the *columns* are the finding: under the frozen recipe the cross-cell median is
flat and slightly positive in all sixteen quarters, whereas the deployable path
has two visible failure columns (2022Q1, 2025Q1) that no per-cell summary shows.

Source: results/tables/deployable_combiner_quarters.csv (per-quarter rel%)
        results/tables/deployable_combiner.csv         (cell membership, verdicts)
Out:    writing/dissertation/figures/stability_quarter_by_quarter.pdf
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, "scripts/analysis")
import supp_style
from supp_style import (
    BLUE,
    GREEN,
    GREEN_TXT,
    GREY,
    INK,
    INK2,
    LIGHT,
    PURPLE,
    REPO,
    RULE,
    TAB,
    VERM,
    annot,
    apply_style,
    finish,
    gate,
    note,
)

supp_style.OUTDIR = os.path.join(REPO, "writing", "dissertation", "figures")

KEY = ["disc", "model", "h"]
CLIP = 30.0          # colour saturates here; saturated cells carry a dot
LINTHRESH = 1.0      # symlog knee, in percentage points

SHORT = {
    "B1_bow_ridge": "BoW", "B2_tfidf_ridge": "TF-IDF", "B3_lm_linear": "LM-dict",
    "B4_lm_features": "LM-feat", "C1_bert_s1": "BERT", "C2_finbert_s1": "FinBERT-1",
    "C2_finbert_s2": "FinBERT-2", "C2_finbert_s3": "FinBERT-3",
    "C2_finbert_s4": "FinBERT-4", "C3_roberta_s1": "RoBERTa",
    "C4_longformer": "Longformer", "C6_llmtext": "LLM-text",
    "D1_concat_mlp": "Concat", "D2_gated_fusion": "Gated", "D4_llmfused": "LLM-fused",
}

# --------------------------------------------------------------------- data
cells = pd.read_csv(os.path.join(TAB, "deployable_combiner.csv"))
qtr = pd.read_csv(os.path.join(TAB, "deployable_combiner_quarters.csv"))

prim = cells[cells.in_primary_grid].copy()
qtr = qtr.merge(cells[KEY + ["in_primary_grid", "genuine_fixed", "genuine_exp"]], on=KEY)
qp = qtr[qtr.in_primary_grid].copy()

quarters = sorted(qp.quarter.unique())

gate({"cells": 69, "quarters": 16, "quarter_cells": 1104,
      "genuine_fixed": 36, "genuine_exp": 6},
     {"cells": int(prim.groupby(KEY).ngroups), "quarters": len(quarters),
      "quarter_cells": len(qp),
      "genuine_fixed": int(prim.genuine_fixed.sum()),
      "genuine_exp": int(prim.genuine_exp.sum())})

order = (prim.assign(_d=prim.disc.map({"event_driven": 0, "long_form": 1}))
             .sort_values(["_d", "model", "h"])
             .reset_index(drop=True))
rows = list(zip(order.disc, order.model, order.h, strict=False))
row_of = {r: i for i, r in enumerate(rows)}

M_fix = np.full((len(rows), len(quarters)), np.nan)
M_exp = np.full((len(rows), len(quarters)), np.nan)
for _, r in qp.iterrows():
    i = row_of[(r.disc, r.model, r.h)]
    j = quarters.index(r.quarter)
    M_fix[i, j] = r.fixed_rel
    M_exp[i, j] = r.exp_rel

med_fix = np.nanmedian(M_fix, axis=0)
med_exp = np.nanmedian(M_exp, axis=0)
q1_fix, q3_fix = np.nanpercentile(M_fix, [25, 75], axis=0)
q1_exp, q3_exp = np.nanpercentile(M_exp, [25, 75], axis=0)
pos_fix = (M_fix > 0).sum(axis=0)
pos_exp = (M_exp > 0).sum(axis=0)

n_clip_fix = int((np.abs(M_fix) > CLIP).sum())
n_clip_exp = int((np.abs(M_exp) > CLIP).sum())

# ------------------------------------------------------------------ canvas
apply_style(9)
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "vermblue", [VERM, "#E8B48A", "#F6F6F4", "#8FC0DE", BLUE])
NORM = mcolors.SymLogNorm(linthresh=LINTHRESH, vmin=-CLIP, vmax=CLIP, base=10)

# hspace 0.145 rather than 0.115, and 0.10 in more canvas to pay for it: the
# heat maps now carry their own year ticks, which need room under them.
fig = plt.figure(figsize=(6.10, 7.15))
gs = fig.add_gridspec(
    3, 3, width_ratios=[1.0, 1.0, 0.115], height_ratios=[1.0, 0.235, 0.048],
    left=0.160, right=0.980, top=0.900, bottom=0.132, wspace=0.075, hspace=0.145)

ax_f = fig.add_subplot(gs[0, 0])
ax_e = fig.add_subplot(gs[0, 1], sharey=ax_f)
ax_s = fig.add_subplot(gs[0, 2], sharey=ax_f)
ax_mf = fig.add_subplot(gs[1, 0], sharex=ax_f)
ax_me = fig.add_subplot(gs[1, 1], sharex=ax_e, sharey=ax_mf)
ax_cb = fig.add_subplot(gs[2, :2])

X = np.arange(len(quarters) + 1)
Y = np.arange(len(rows) + 1)

for ax, M, title in ((ax_f, M_fix, "(a) frozen validation weights"),
                     (ax_e, M_exp, "(b) expanding deployable weights")):
    ax.pcolormesh(X, Y, M, cmap=CMAP, norm=NORM, edgecolors="none")
    ii, jj = np.where(np.abs(M) > CLIP)
    if len(ii):
        ax.plot(jj + 0.5, ii + 0.5, ls="none", marker="o", ms=1.15,
                mfc="white", mec="white", mew=0.0)
    ax.set_ylim(len(rows), 0)
    ax.set_xlim(0, len(quarters))
    ax.set_title(title, fontsize=9, pad=3.5, color=GREY)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_edgecolor(GREY)
    ax.tick_params(length=0)
    ax.set_xticks([])

# horizon separators and the disclosure divide
n_ed = int((order.disc == "event_driven").sum())
for ax in (ax_f, ax_e, ax_s):
    for b in range(3, len(rows), 3):
        ax.axhline(b, color="white", lw=0.55)
    ax.axhline(n_ed, color=GREY, lw=1.0)

# model-group labels down the left edge
labels, centres = [], []
for start in range(0, len(rows), 3):
    labels.append(SHORT[rows[start][1]])
    centres.append(start + 1.5)
ax_f.set_yticks(centres)
ax_f.set_yticklabels(labels, fontsize=9)
ax_f.tick_params(axis="y", length=0, pad=2)
plt.setp(ax_e.get_yticklabels(), visible=False)

# Colour carries exactly two meanings in this figure and nothing else:
#   VERM..BLUE  the diverging heat scale -- blue = text helps (the colour bar
#               says so in words), so neither pole may be spent on anything
#               that is not the increment itself;
#   GREEN/PURPLE  the two weight schemes -- GREEN = frozen, PURPLE = expanding,
#               used for the strip's F/E columns and for the (c)/(d) medians,
#               and matching stability_freeze_point_deployability.
# These two row-group labels used to be drawn in VERM_TXT and BLUE, i.e. in the
# heat scale's own poles, which told the reader that 8-K rows were "text hurts"
# and long-form rows "text helps". They are structural labels, like axis
# labels, so they take primary ink and lean on the grey divide rule between
# them; recolouring changes no geometry.
bb = ax_f.get_position()
frac_ed = n_ed / len(rows)
fig.text(0.022, bb.y1 - bb.height * frac_ed / 2, "8-K  (event-driven)",
         rotation=90, ha="center", va="center", fontsize=9, color=INK)
fig.text(0.022, bb.y0 + bb.height * (1 - frac_ed) / 2, "10-K / 10-Q  (long-form)",
         rotation=90, ha="center", va="center", fontsize=9, color=INK)

# --------------------------------------------------- verdict strip, column (c)
ax_s.set_xlim(0, 2)
ax_s.set_xticks([])
for s in ax_s.spines.values():
    s.set_visible(False)
ax_s.tick_params(length=0)
plt.setp(ax_s.get_yticklabels(), visible=False)
for i, (_, r) in enumerate(order.iterrows()):
    ax_s.add_patch(plt.Rectangle((0.08, i + 0.12), 0.84, 0.76,
                                 fc=GREEN if r.genuine_fixed else LIGHT,
                                 ec="none"))
    ax_s.add_patch(plt.Rectangle((1.08, i + 0.12), 0.84, 0.76,
                                 fc=PURPLE if r.genuine_exp else LIGHT,
                                 ec="none"))
ax_s.set_ylim(len(rows), 0)
ax_s.text(0.25, 1.006, "F", transform=ax_s.transAxes, ha="center", va="bottom",
          fontsize=9, color=GREEN_TXT)
ax_s.text(0.75, 1.006, "E", transform=ax_s.transAxes, ha="center",
          va="bottom", fontsize=9, color=PURPLE)

# -------------------------------------------------------------- marginals
xc = np.arange(len(quarters)) + 0.5
years = sorted({q[:4] for q in quarters})
year_c = [np.mean([i + 0.5 for i, q in enumerate(quarters) if q[:4] == y])
          for y in years]

for ax, med, q1, q3, col in (
        (ax_mf, med_fix, q1_fix, q3_fix, GREEN),
        (ax_me, med_exp, q1_exp, q3_exp, PURPLE)):
    ax.axhline(0, color=GREY, lw=0.6)
    ax.fill_between(xc, q1, q3, color=col, alpha=0.16, lw=0)
    ax.plot(xc, med, color=col, lw=1.2, marker="o", ms=2.4, mfc=col, mec=col)
    ax.set_ylim(-18.0, 9.6)
    ax.set_xlim(0, len(quarters))
    ax.set_xticks(year_c)
    ax.set_xticklabels(years, fontsize=9)
    ax.tick_params(axis="x", length=2, pad=2)
    for b in range(4, len(quarters), 4):
        ax.axvline(b, color=LIGHT, lw=0.5, zorder=0)

ax_mf.set_ylabel("median across\ncells (pp)", fontsize=9, labelpad=1)
plt.setp(ax_me.get_yticklabels(), visible=False)
ax_mf.set_yticks([-15, -10, -5, 0, 5])

# The heat maps carry the year axis themselves.  Panels (c) and (d) share the
# x-axis with them, so the columns line up, but a reader cannot be asked to
# infer that: without these ticks neither "-14.5 in 2022Q1" nor any other
# per-quarter statement in the caption can be located in the maps.
for ax in (ax_f, ax_e):
    plt.setp(ax.get_xticklabels(), visible=True, fontsize=9)
    ax.tick_params(axis="x", length=2, width=0.5, color=GREY, pad=2)

lo = int(np.argmin(med_exp))
_lab = ax_me.annotate(f"{med_exp[lo]:+.1f}", xy=(lo + 0.5, med_exp[lo]),
                      xytext=(lo + 1.1, med_exp[lo] + 0.4), fontsize=9,
                      color=PURPLE, ha="left", va="center",
                      arrowprops=dict(arrowstyle="-", lw=0.5, color=PURPLE))
_lab.set_path_effects([mpe.withStroke(linewidth=2.0, foreground="white")])

# These four blocks sit inside the axes, and in panel (d) the upper edge of the
# expanding IQR band climbs through the title from 2024 onward -- measured on the
# rendered page, not guessed. annot() gives them a white outline so they survive
# being drawn over the band, which costs nothing in geometry; moving them out
# would widen the tight bounding box and shrink every glyph on the page.
annot(ax_mf, 0.985, 0.94, "(c) cross-cell median, IQR band",
      transform=ax_mf.transAxes, color=INK, va="top", ha="right")
annot(ax_mf, 0.985, 0.56,
      f"every quarter positive:\n{med_fix.min():+.1f} to {med_fix.max():+.1f} pp",
      transform=ax_mf.transAxes, color=GREEN_TXT, va="top", ha="right",
      linespacing=1.15)
annot(ax_me, 0.985, 0.94, "(d) same 69 cells, refit each quarter",
      transform=ax_me.transAxes, color=INK, va="top", ha="right")

# ------------------------------------------------------------- colour bar
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal",
                  ticks=[-30, -10, -3, -1, 0, 1, 3, 10, 30])
cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
cb.ax.set_xticklabels(["-30", "-10", "-3", "-1", "0", "+1", "+3", "+10", "+30"],
                      fontsize=9)
cb.set_label("QLIKE improvement from adding text, per cent "
             "(symmetric log; blue = text helps)", fontsize=9, labelpad=1.5)
cb.outline.set_linewidth(0.5)
cb.outline.set_edgecolor(GREY)
cb.ax.tick_params(length=2, width=0.5, pad=1)

# ---------------------------------------------------------------- headline
fig.text(0.160, 0.988,
         "Sixteen test quarters against 69 combination cells: the frozen",
         fontsize=9, color=GREY, va="top", ha="left")
fig.text(0.160, 0.966,
         "recipe is flat across time, the deployable one is not.",
         fontsize=9, color=GREY, va="top", ha="left")
# The first two lines above are the figure's claim and keep primary ink. This
# third line is the strip's key, not the argument, and it was printed in the
# same ink as the claim; INK2 tells the two apart without resizing anything.
# There is no room for a hairline between them: line two's ink ends at 0.9623
# and this line's begins at 0.9561, a 3.2 pt gap.
fig.text(0.160, 0.944,
         f"Strip: F = genuine, frozen ({int(prim.genuine_fixed.sum())}/69);  "
         f"E = genuine, expanding ({int(prim.genuine_exp.sum())}/69).",
         fontsize=9, color=INK2, va="top", ha="left")

# Apparatus block: what the dots are and how the rows are ordered. It read as
# more of the argument because it was set in the argument's ink and floated free
# under the colour bar with nothing dividing them. Recessive ink plus a hairline
# above, and the rule is placed by hand rather than by note(): note() draws at
# y + 0.012 = 0.0740, which lands on this block's own ascenders at 0.0742. The
# measured room is 0.0742 (text top) to 0.0916 (colour-bar label bottom), so the
# rule sits at 0.0825, inside content on all four sides -- under
# bbox_inches="tight" anything outside it would widen the page and shrink every
# glyph in the figure.
fig.lines.append(Line2D([0.160, 0.918], [0.0825, 0.0825],
                        transform=fig.transFigure, color=RULE,
                        linewidth=0.5, zorder=0.5))
note(fig, 0.160, 0.062,
     f"White dots: the {n_clip_exp} quarter-cells beyond +/-30 pp "
     f"({n_clip_fix} of them in panel a), drawn at the",
     rule=False)
note(fig, 0.160, 0.040,
     "colour limit. Each model band is h = 5, 10, 20 from top to bottom.",
     rule=False)

finish(fig, "stability_quarter_by_quarter")
print("per-quarter fixed median  :", np.round(med_fix, 2))
print("per-quarter expand median :", np.round(med_exp, 2))
print("cells positive, fixed     :", pos_fix)
print("cells positive, expanding :", pos_exp)
