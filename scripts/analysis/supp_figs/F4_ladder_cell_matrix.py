"""F4 -- The reference ladder, cell by cell: a 69 x 5 survivor matrix.

WHAT THIS DRAWS
    Every cell of the 69-cell combination grid, one row per (channel, model,
    horizon), against five rungs of the reference ladder, plus a right-hand
    margin bar carrying the primary-rung relative QLIKE improvement on a scale
    shared by both panels.

    The grid is drawn as two side-by-side panels, event-driven (24 rows) and
    long-form (45 rows), rather than as one 69-row column. A single column of
    69 rows on a 6.4 in canvas forces a ~7.7 pt row pitch, which pushes the
    per-row horizon digits below the house minimum of 9 pt that supp_style
    declares for the whole figure set. Splitting by channel puts the binding
    panel at 45 rows, which supports a 10.8 pt pitch and 9 pt type at 1:1
    inclusion scale. Nothing about the evidence changes: the same 345 marks
    are drawn, the bar scale is held common across the two panels, and the
    headline ladder counts are printed once for the whole grid.

SOURCE (single file, read at run time)
    results/tables/control_intersection_ensemble.csv
        columns: disc, model, h, n_seeds, vol_rel_impr_pct,
                 primary_genuine (38 True), maximal_holm (9), firm_holm (8),
                 AND_maximal_firm_holm (0), AND_full_holm (0)
    This file ONLY. results/tables/control_intersection.csv is the superseded
    seed-2026 single-seed basis (29 / 8 / 8) and would restate the headline
    ladder with the wrong rungs; it is reported separately in the supplement's
    basis section, never mixed into this matrix.

MAIN-TEXT SENTENCES SUBSTANTIATED (writing/paper/sections/06_results.tex)
    l.19 "against the upper-bound recalibrated HAR f_R the seed-ensemble
          primary leaves 38 of 69 cells surviving Holm and placebo (up to
          5.9%) ... No cell survives the maximal-pool and firm-identity
          controls under Holm."
    l.15 (caption of fig:cascade) "each rung a separate reference (final bar:
          their conjunction; survivor sets disjoint)."

CONVENTIONS
    Vol-unit QLIKE; day-clustered Diebold-Mariano; Holm within the 69-cell
    family; the primary column additionally requires the label-shuffle placebo
    |DM| < 2. Positive relative improvement = the text-augmented combiner
    lowers QLIKE against the recalibrated HAR.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from supp_style import (BLUE, GREEN, GREY, INK, INK2, LIGHT, PURPLE, RULE, SKY,
                        TAB, VERM, VERM_TXT, annot, apply_style, finish, gate,
                        note)
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

SRC = os.path.join(TAB, "control_intersection_ensemble.csv")
d = pd.read_csv(SRC)

COLS = ["primary_genuine", "maximal_holm", "firm_holm",
        "AND_maximal_firm_holm", "AND_full_holm"]
TAGS = ["(1)", "(2)", "(3)", "(4)", "(5)"]
KEY = ["primary rung (recalibrated HAR + placebo)",
       "maximal price pool (five price models)",
       "firm-identity reference",
       "(2) and (3)",
       "(1) and (2) and (3)"]
FILL = [BLUE, GREEN, PURPLE, GREY, GREY]

ED = d[d.disc == "event_driven"]
LF = d[d.disc == "long_form"]

# ---------------------------------------------------------------- evidence gate
# The paper's own counts. These are the ONLY hardcoded numbers in this script;
# if the evidence table drifts, the build aborts here instead of re-rendering.
overlap = int((d.maximal_holm & d.firm_holm).sum())
overlap_lf = int((LF.maximal_holm & LF.firm_holm).sum())
top = d.loc[d.vol_rel_impr_pct.idxmax()]
bot = d.loc[d.vol_rel_impr_pct.idxmin()]
gate(
    {"cells": 69, "event_driven": 24, "long_form": 45, "single_seed": 36,
     "single_seed_ed": 18, "single_seed_lf": 18,
     "primary_genuine": 38, "maximal_holm": 9, "firm_holm": 8,
     "and_maximal_firm_holm": 0, "and_full_holm": 0, "pool_firm_overlap": 0,
     "ed_counts": [8, 0, 6, 0, 0], "lf_counts": [30, 9, 2, 0, 0],
     "lf_pool_firm_overlap": 0,
     "max_cell": "long_form/B2_tfidf_ridge/20", "max_rel": 5.92,
     "min_cell": "long_form/C2_finbert_s3/20", "min_rel": -3.86},
    {"cells": len(d),
     "event_driven": len(ED), "long_form": len(LF),
     "single_seed": int((d.n_seeds == 1).sum()),
     "single_seed_ed": int((ED.n_seeds == 1).sum()),
     "single_seed_lf": int((LF.n_seeds == 1).sum()),
     "primary_genuine": int(d.primary_genuine.sum()),
     "maximal_holm": int(d.maximal_holm.sum()),
     "firm_holm": int(d.firm_holm.sum()),
     "and_maximal_firm_holm": int(d.AND_maximal_firm_holm.sum()),
     "and_full_holm": int(d.AND_full_holm.sum()),
     "pool_firm_overlap": overlap,
     "ed_counts": [int(ED[c].sum()) for c in COLS],
     "lf_counts": [int(LF[c].sum()) for c in COLS],
     "lf_pool_firm_overlap": overlap_lf,
     "max_cell": f"{top.disc}/{top.model}/{int(top.h)}",
     "max_rel": round(float(d.vol_rel_impr_pct.max()), 2),
     "min_cell": f"{bot.disc}/{bot.model}/{int(bot.h)}",
     "min_rel": round(float(d.vol_rel_impr_pct.min()), 2)},
)

COUNTS = [int(d[c].sum()) for c in COLS]
LO = float(d.vol_rel_impr_pct.min())
HI = float(d.vol_rel_impr_pct.max())
XLO, XHI = LO - 0.45, HI + 0.45          # bar scale, shared by both panels

# ----------------------------------------------------------------- page layout
# Every block is placed in inches from the top of the canvas so that no text
# can drift into another block, and the row pitch is set once for both panels.
apply_style(base_size=9)
W, H = 6.4, 9.0
PITCH = 0.148                             # inches per grid row = 10.7 pt
TOP = 1.18                                # inches from top to the first row
fig = plt.figure(figsize=(W, H))


def fy(inches_from_top):
    return 1.0 - inches_from_top / H


def fx(inches_from_left):
    return inches_from_left / W


# ------------------------------------------------------------------ hierarchy
# The figure is saved with bbox_inches="tight", so the PDF page IS the content's
# bounding box: enlarging any type widens the box, forces a harder down-scale in
# the document and makes every printed glyph SMALLER. Nothing below changes a
# size or moves a block outward. Emphasis comes from colour (INK for the
# argument, INK2 for the apparatus), from RULE hairlines that separate a heading
# from what it governs, and from `emph`.
def emph(t, color=INK, lw=0.30):
    """Give `t` weight without giving it width.

    Every fontweight in this document resolves to Helvetica.ttc's regular face,
    so `fontweight="bold"` renders as regular: the headings of this figure asked
    for weight and were drawn without any. A same-colour stroke supplies it. The
    advance width is unchanged (measured identical at 0.30), so no geometry moves
    and no printed point size falls.
    """
    t.set_path_effects([pe.withStroke(linewidth=lw, foreground=color)])
    return t


def hair(x0_in, x1_in, y_in, lw=0.5):
    """A RULE hairline, in inches of the layout grid. Interior white space only."""
    fig.add_artist(Line2D([fx(x0_in), fx(x1_in)], [fy(y_in), fy(y_in)],
                          transform=fig.transFigure, color=RULE,
                          linewidth=lw, zorder=0.5))


# columns: [label zone] [matrix] [bars] per channel panel
GEOM = {
    "event_driven": {"lab": 0.02, "mat": (1.26, 2.22), "bar": (2.28, 3.08)},
    "long_form": {"lab": 3.20, "mat": (4.44, 5.40), "bar": (5.46, 6.26)},
}
HEAD = {"event_driven": "event-driven, 24 cells",
        "long_form": "long-form, 45 cells"}


def draw_panel(disc):
    g = d[d.disc == disc].sort_values(["model", "h"]).reset_index(drop=True)
    n = len(g)
    geom = GEOM[disc]
    h_in = n * PITCH
    axm = fig.add_axes([fx(geom["mat"][0]), fy(TOP + h_in),
                        fx(geom["mat"][1] - geom["mat"][0]), h_in / H])
    axb = fig.add_axes([fx(geom["bar"][0]), fy(TOP + h_in),
                        fx(geom["bar"][1] - geom["bar"][0]), h_in / H])
    for ax in (axm, axb):
        ax.set_ylim(n - 0.5, -0.5)
    axm.set_xlim(-0.5, 4.5)
    for sp in axm.spines.values():
        sp.set_visible(False)
    axm.set_xticks([])
    axm.set_yticks([])

    # Emphasis 1: columns (4) and (5) are empty in all 69 rows. Shading the
    # band makes that read as a property of the panel, not as data that failed
    # to load.
    axm.add_patch(Rectangle((2.5, -0.5), 2.0, float(n),
                            facecolor=LIGHT, alpha=0.5, edgecolor="none",
                            zorder=0))

    for i, row in g.iterrows():
        for j, col in enumerate(COLS):
            if bool(row[col]):
                axm.plot([j], [i], marker="s", markersize=6.4,
                         markerfacecolor=FILL[j], markeredgecolor="none",
                         zorder=3)
            else:
                axm.plot([j], [i], marker="s", markersize=6.4,
                         markerfacecolor="white", markeredgecolor="#B8B8B8",
                         markeredgewidth=0.4, zorder=2)

    # ------------------------------------------------------ row/group labels
    lab_r = (geom["mat"][0] - 0.05 - geom["mat"][0]) / \
        (geom["mat"][1] - geom["mat"][0]) * 5.0 - 0.5      # x for the digits
    for i, row in g.iterrows():
        # the horizon digit is the row index inside a model group, so it recedes
        # one step behind the model name it sits under
        axm.text(lab_r, i, str(int(row.h)), ha="right", va="center",
                 fontsize=9, color=INK2, clip_on=False)
        # seed glyph: open marker = single seed, filled = 3-seed ensemble
        axm.plot([-0.62], [i], marker="o", markersize=3.0,
                 markerfacecolor=(GREY if row.n_seeds == 3 else "white"),
                 markeredgecolor=GREY, markeredgewidth=0.45, clip_on=False,
                 zorder=3)

    x_mod = (geom["lab"] - geom["mat"][0]) / \
        (geom["mat"][1] - geom["mat"][0]) * 5.0 - 0.5      # x for model names
    for model, grp in g.groupby("model", sort=False):
        i0, i1 = float(grp.index.min()), float(grp.index.max())
        axm.text(x_mod, (i0 + i1) / 2.0, model, ha="left", va="center",
                 fontsize=9, color=INK, clip_on=False)
        if i1 < n - 1:
            axm.plot([x_mod, 4.5], [i1 + 0.5, i1 + 0.5], color=RULE,
                     linewidth=0.4, clip_on=False, zorder=0)

    # ---------------------------------------------------------- panel header
    cnt = [int(g[c].sum()) for c in COLS]
    for j in range(5):
        col = VERM_TXT if cnt[j] == 0 else INK
        emph(fig.text(fx(geom["mat"][0]) + (j + 0.5) / 5.0
                      * fx(geom["mat"][1] - geom["mat"][0]), fy(0.83),
                      str(cnt[j]), ha="center", va="center", fontsize=10,
                      color=col), color=col, lw=0.35)
        fig.text(fx(geom["mat"][0]) + (j + 0.5) / 5.0
                 * fx(geom["mat"][1] - geom["mat"][0]), fy(1.03),
                 TAGS[j], ha="center", va="center", fontsize=9, color=INK)
    emph(fig.text(fx(geom["lab"]), fy(0.62), HEAD[disc], ha="left",
                  va="center", fontsize=9.5, color=INK))
    # "of n" is the denominator the counts above the columns are read against:
    # apparatus, not a count, so it recedes.
    fig.text(fx(geom["mat"][0] - 0.05), fy(0.83), f"of {n}", ha="right",
             va="center", fontsize=9, color=INK2)
    # the panel head is three stacked lines of apparatus (channel, count of
    # survivors, column tag); one hairline closes it and opens the data grid
    hair(geom["lab"], geom["mat"][1], 1.15)

    # ------------------------------------------------------------ margin bars
    for i, row in g.iterrows():
        v = float(row.vol_rel_impr_pct)
        axb.barh(i, v, height=0.66, color=(SKY if v >= 0 else VERM),
                 edgecolor="none", zorder=2)
    axb.axvline(0.0, color=GREY, linewidth=0.6, zorder=3)
    axb.set_xlim(XLO, XHI)
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    axb.spines["bottom"].set_position(("outward", 3))
    axb.tick_params(axis="x", labelsize=9, length=2.5, pad=2)
    axb.set_xticks([-3, 0, 3, 6])
    axb.set_xticklabels(["-3", "0", "+3", "+6"])

    fig.text(fx((geom["bar"][0] + geom["bar"][1]) / 2.0),
             fy(TOP + h_in + 0.32), "relative QLIKE", ha="center",
             va="center", fontsize=9, color=INK)
    fig.text(fx((geom["bar"][0] + geom["bar"][1]) / 2.0),
             fy(TOP + h_in + 0.49), "improvement (%)", ha="center",
             va="center", fontsize=9, color=INK)

    # the two extremes are labelled inside their own bar, which costs no width.
    # One decimal, not two: at 9 pt the five-character "-3.86" is wider than the
    # 3.86-unit bar that carries it, so its last digit crossed the zero rule
    # onto white page and the white stroke punched a gap in the rule. The exact
    # extremes (-3.86% to +5.92%) are carried by the caption. The positive label
    # is set in INK rather than white: white on the light SKY fill measures
    # 2.3:1, INK on the same fill 4.9:1; the negative label stays white, which
    # is the higher-contrast choice on the darker VERM fill (4.3:1).
    if disc == "long_form":
        imax = int(g.vol_rel_impr_pct.idxmax())
        axb.text(HI - 0.25, imax, f"+{HI:.1f}", ha="right", va="center",
                 fontsize=9, color=INK, zorder=4)
        imin = int(g.vol_rel_impr_pct.idxmin())
        axb.text(LO + 0.25, imin, f"{LO:.1f}", ha="left", va="center",
                 fontsize=9, color="white", zorder=4)
        # Emphasis 2: within this panel no row is filled in both (2) and (3).
        yb = n - 0.5 + 0.35
        axm.plot([1, 1, 2, 2], [yb, yb + 0.75, yb + 0.75, yb], color=VERM_TXT,
                 linewidth=0.8, clip_on=False)
        annot(axm, 1.5, yb + 2.3, "overlap 0", color=VERM_TXT, ha="center",
              va="center", clip_on=False)
    return axm


draw_panel("event_driven")
draw_panel("long_form")

# ------------------------------------------- whole-grid key and count strip
# Placed in the space the 24-row event-driven panel leaves free, so the
# headline ladder counts stay on the page once, for the whole 69-cell grid,
# and are never confused with a per-panel subtotal.
KY = 5.58
emph(fig.text(fx(0.06), fy(KY),
              "Whole grid, 69 cells: Holm survivors per rung",
              ha="left", va="center", fontsize=9.5, color=INK))
hair(0.06, 3.05, KY + 0.14)
for j in range(5):
    y = KY + 0.28 + 0.225 * j
    fig.text(fx(0.06), fy(y), TAGS[j], ha="left", va="center", fontsize=9,
             color=INK)
    # The key carries the rung's own mark, so the three channel hues in the
    # matrix and the always-open pair are read off the same line as the tag
    # that names them; without it the colours in the grid mean nothing.
    fig.add_artist(Line2D(
        [fx(0.2955)], [fy(y)], transform=fig.transFigure, marker="s",
        markersize=6.4, linestyle="none",
        markerfacecolor=(FILL[j] if j < 3 else "white"),
        markeredgecolor=("none" if j < 3 else "#B8B8B8"),
        markeredgewidth=0.4))
    fig.text(fx(0.38), fy(y), KEY[j], ha="left", va="center", fontsize=9,
             color=INK)
    ccol = VERM_TXT if COUNTS[j] == 0 else INK
    emph(fig.text(fx(3.05), fy(y), str(COUNTS[j]), ha="right", va="center",
                  fontsize=10, color=ccol), color=ccol, lw=0.35)

# ------------------------------------------------------------ apparatus notes
# Everything from here down is basis statement, not argument: the seed
# denominator, the bar convention, and the does-not-show clause on (2) and (3).
# It carries every word it carried before, in the recessive ink, under one
# hairline, so a reader can tell the claim from the ground it stands on.
note(fig, fx(0.06), fy(7.12),
     "Seeds: open circle = 1 seed (36 cells: 18 event-driven,\n"
     "18 long-form), filled circle = 3-seed ensemble (33 cells).",
     width=(3.05 - 0.06) / W)
note(fig, fx(0.06), fy(7.62),
     "Bars: primary rung only, one scale shared by both\n"
     "panels; the two extremes are labelled inside the bar.",
     rule=False)
note(fig, fx(0.06), fy(8.06),
     "No row of either panel is filled in both (2) and (3): the\n"
     "9 pool survivors and the 8 identity survivors are disjoint,\n"
     "not nested. Column (2) is empty in the event-driven panel,\n"
     "so disjointness there is forced; in the long-form panel 9\n"
     "pool and 2 identity survivors still do not intersect.",
     rule=False)

# -------------------------------------------------------------------- notes
# The basis statement and the mark legend. They have to stay at the head of the
# figure -- the foot of the page is already full to the bounding box, and moving
# them there would grow the box and shrink every glyph -- so they are held apart
# from the grid by ink and by a rule instead of by position: recessive INK2 above
# one full-width hairline, with the data beginning below it.
note(fig, fx(0.02), fy(0.17),
     "Seed-ensemble basis, 69-cell grid, vol-unit QLIKE. Filled square = "
     "cell survives that rung.",
     rule=False, va="center")
note(fig, fx(0.02), fy(0.36),
     "Open square = it does not. Columns (4) and (5) are conjunctions "
     "under Holm and are empty in all 69 rows.",
     rule=False, va="center")
hair(0.02, 6.26, 0.50)

finish(fig, "F4_ladder_cell_matrix")
