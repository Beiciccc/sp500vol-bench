"""AR2 — is the "maximal" price pool actually maximal?

Appendix figure (outside the 60-page cap).  The five-member price pool leaves two
computed price models out (HARQ and HAR-X with VIX).  This figure puts them back,
one at a time and together, and asks three things of the enlarged pools: are they
better forecasts, are their fitted weights stable, and what do they do to the
69-cell text cascade.

Sources
-------
results/tables/pool_frontier_audit.csv     4 pools x 6 panels: test QLIKE, fitted
                                           log weights, day-clustered DM vs pool5
results/tables/pool_frontier_cascade.csv   69 cells x 2 pools: text increment
results/tables/vix_control_harx.csv        HAR-X(VIX) against HAR-RV, 9 panels

Basis: the frontier recomputation exists only on the archived single-seed (2026)
basis, which the figure states on its face.
"""
import os
import sys

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import diss_style as ds  # noqa: E402
from supp_style import (BLUE, GREEN, GREY, INK, INK2, LIGHT,  # noqa: E402
                        PURPLE, RULE, TAB, VERM, VERM_TXT, YELLOW,
                        apply_style, gate)

POOLS = ["pool5 (paper)", "pool6 +HARQ", "pool6 +HARX", "pool7 (all)"]
SHORT = {"pool5 (paper)": "five members (the reported pool)",
         "pool6 +HARQ": "+ HARQ",
         "pool6 +HARX": "+ HAR-X (VIX)",
         "pool7 (all)": "+ both (seven members)"}
COL = {"pool5 (paper)": GREY, "pool6 +HARQ": YELLOW,
       "pool6 +HARX": GREEN, "pool7 (all)": PURPLE}
# Shape as well as hue: the four pools are otherwise separated by colour alone,
# which is the one channel a greyscale print does not carry.
MK = {"pool5 (paper)": "o", "pool6 +HARQ": "s",
      "pool6 +HARX": "^", "pool7 (all)": "D"}
MS = {"o": 26, "s": 24, "^": 30, "D": 22}
# Every pool used to land on the row's centre line, so where the pools agree --
# the ED horizons, which is the half where the change is small and therefore the
# half on which "the price side gets harder to beat" is actually tested -- the
# glyphs coincided: at ED h=20 the "+ HARQ" square kept about a dozen unoccluded
# pixels under the "+ both" diamond, and the row read as two pools rather than
# three.  The sub-lane offsets below are in the existing categorical y units, a
# fraction of the unit row pitch, so every marker stays inside its own row band
# and inside the axes; no axis limit, tick or label moves and the tight bounding
# box is unchanged.  HARQ and "+ both" take the outer lanes because they are the
# pair that coincides most often in x.
DODGE_A = {"pool6 +HARQ": -0.22, "pool6 +HARX": 0.0, "pool7 (all)": 0.22}
DODGE_B = {"pool5 (paper)": -0.24, "pool6 +HARQ": -0.08,
           "pool6 +HARX": 0.08, "pool7 (all)": 0.24}
# A white stroke laid under each marker's own outline, so any residual overlap
# reads as one glyph in front of another instead of as a single fused shape.
# Drawn behind the marker's fill and edge, and clipped with it, so it adds no
# extent to the axes.
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
PANELS = [("long_form", 5), ("long_form", 10), ("long_form", 20),
          ("event_driven", 5), ("event_driven", 10), ("event_driven", 20)]
DLAB = {"long_form": "LF", "event_driven": "ED", "combined": "both"}
PLAB = {("long_form", 5): "LF  h=5", ("long_form", 10): "LF  h=10",
        ("long_form", 20): "LF  h=20", ("event_driven", 5): "ED  h=5",
        ("event_driven", 10): "ED  h=10", ("event_driven", 20): "ED  h=20"}


def har_weight(s):
    return float(dict(kv.split("=") for kv in s.split(";"))["A2_har_rv"])


def main():
    aud = pd.read_csv(os.path.join(TAB, "pool_frontier_audit.csv"))
    cas = pd.read_csv(os.path.join(TAB, "pool_frontier_cascade.csv"))
    vix = pd.read_csv(os.path.join(TAB, "vix_control_harx.csv"))

    base = aud[aud.pool == POOLS[0]].set_index(["disc", "h"]).qlike_test
    aud["rel"] = [100 * (r.qlike_test / base[(r.disc, r.h)] - 1)
                  for r in aud.itertuples()]
    aud["w_har"] = aud.weights.map(har_weight)

    w = cas.pivot_table(index=["disc", "model", "h"], columns="pool",
                        values=["genuine_holm", "genuine_raw"], aggfunc="first")
    g5, g7 = w[("genuine_holm", POOLS[0])], w[("genuine_holm", "pool7 (all)")]

    got = {
        "panels": len(aud) // 4,
        "pool5_raw": int(cas[cas.pool == POOLS[0]].genuine_raw.sum()),
        "pool5_holm": int(g5.sum()),
        "pool7_raw": int(cas[cas.pool == "pool7 (all)"].genuine_raw.sum()),
        "pool7_holm": int(g7.sum()),
        "survive_both": int((g5 & g7).sum()),
        "lost": int((g5 & ~g7).sum()),
        "gained": int((~g5 & g7).sum()),
        "harq_better": int(aud[aud.pool == "pool6 +HARQ"].better_than_pool5.sum()),
        "harx_better": int(aud[aud.pool == "pool6 +HARX"].better_than_pool5.sum()),
        "pool7_better": int(aud[aud.pool == "pool7 (all)"].better_than_pool5.sum()),
        "max_abs_weight": round(float(aud.max_abs_weight.max()), 2),
        "vix_cells": len(vix),
        "vix_raw_sig": int((vix.p < 0.05).sum()),
        "vix_holm_sig": int((vix.holm < 0.05).sum()),
    }
    gate({"panels": 6, "pool5_raw": 23, "pool5_holm": 8, "pool7_raw": 17,
          "pool7_holm": 3, "survive_both": 2, "lost": 6, "gained": 1,
          "harq_better": 3, "harx_better": 4, "pool7_better": 3,
          "max_abs_weight": 1.82, "vix_cells": 9, "vix_raw_sig": 6,
          "vix_holm_sig": 0}, got)

    apply_style(9)
    H = 7.50
    fig = plt.figure(figsize=ds.canvas(H, max_h=7.7))

    def fy(t):
        return 1.0 - t / H

    def header(x, t_in, title, key):
        """A panel title at INK with its reading key beneath it at INK2.

        Each header used to be one fig.text block in which the panel title and
        the plotting convention that follows it drew at a single colour, so
        nothing told a reader where the claim stopped and the convention began.
        Colour is the only hierarchy channel available -- finish() writes with
        bbox_inches="tight", so a larger size would widen the page box and shrink
        every printed glyph in the figure -- and colour needs one text artist per
        ink.  Both artists therefore carry the block's FULL line count with the
        other's lines blanked, and share one va="top" anchor: line i lands in
        exactly the pixel row it occupied as one block, whatever line advance
        matplotlib resolves for the face.  Nothing reflows, no word moves, and the
        tight box is the union of two artists that span the same rows as the one
        they replace.
        """
        nt, nk = title.count("\n") + 1, key.count("\n") + 1
        fig.text(x, fy(t_in), title + "\n" * nk, fontsize=9, color=INK,
                 va="top", linespacing=1.3)
        fig.text(x, fy(t_in), "\n" * nt + key, fontsize=9, color=INK2,
                 va="top", linespacing=1.3)

    def rows_axes(rect):
        ax = fig.add_axes(rect)
        ax.set_ylim(5.6, -0.6)
        ax.set_yticks(range(6))
        ax.set_yticklabels([PLAB[p] for p in PANELS], fontsize=9)
        ax.tick_params(length=0, pad=2)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=LIGHT, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        return ax

    # ------------------------------------------- (a) is the pool a better forecast
    ax = rows_axes([0.135, fy(2.40), 0.335, 1.35 / H])
    ax.axvline(0, color=GREY, lw=0.9, zorder=2)
    for yi, key in enumerate(PANELS):
        for pool in POOLS[1:]:
            r = aud[(aud.disc == key[0]) & (aud.h == key[1])
                    & (aud.pool == pool)].iloc[0]
            sig = bool(r.better_than_pool5)
            # Open markers go last: their white face is the significance key,
            # and a later filled glyph used to punch straight through it.
            ax.scatter([r.rel], [yi + DODGE_A[pool]], s=MS[MK[pool]],
                       marker=MK[pool], zorder=3 if sig else 3.5,
                       color=COL[pool] if sig else "white",
                       edgecolor=COL[pool], linewidths=0.9,
                       path_effects=HALO)
    ax.set_xlim(-11.5, 2.3)
    ax.set_xticks([-10, -5, 0])
    ax.set_xlabel("test QLIKE vs the five-member pool (%)", fontsize=9,
                  labelpad=2)
    header(0.018, 0.62,
           "(a) enlarging the pool: change in test QLIKE",
           "     (left = better than the reported pool)")

    # ------------------------------------------------- (b) the fitted HAR weight
    bx = rows_axes([0.635, fy(2.40), 0.335, 1.35 / H])
    bx.axvline(0, color=GREY, lw=0.9, zorder=2)
    for yi, key in enumerate(PANELS):
        for pool in POOLS:
            r = aud[(aud.disc == key[0]) & (aud.h == key[1])
                    & (aud.pool == pool)].iloc[0]
            bx.scatter([r.w_har], [yi + DODGE_B[pool]], s=MS[MK[pool]],
                       marker=MK[pool], zorder=3, color=COL[pool],
                       edgecolor=COL[pool], linewidths=0.9,
                       path_effects=HALO)
    bx.set_xlim(-1.95, 0.75)
    bx.set_xticks([-1.5, -1.0, -0.5, 0, 0.5])
    bx.set_xlabel("fitted log-space weight on HAR-RV", fontsize=9, labelpad=2)
    bx.set_yticklabels([])
    header(0.545, 0.62,
           "(b) what the pool does with HAR-RV itself",
           "     (weight on the reference model)")

    # legend for (a) and (b)
    lg = fig.add_axes([0.018, fy(3.22), 0.964, 0.42 / H])
    lg.set_xlim(0, 100)
    lg.set_ylim(-1.1, 0.6)
    lg.axis("off")
    for x0, pool in [(1, POOLS[0]), (38, "pool6 +HARQ"), (52, "pool6 +HARX"),
                     (73, "pool7 (all)")]:
        lg.scatter([x0], [0.25], s=MS[MK[pool]], marker=MK[pool],
                   color=COL[pool], zorder=3)
        lg.text(x0 + 1.7, 0.25, SHORT[pool], ha="left", va="center",
                fontsize=9, color=GREY)
    lg.scatter([1], [-0.75], s=26, facecolor="white", edgecolor=GREY,
               linewidths=0.9, zorder=3)
    # The four pool names label data categories, so they stay at data ink; the
    # open-marker sentence is a plotting convention and recedes to INK2.
    # "(uncorrected)" is the fill rule's actual status: better_than_pool5 in
    # pool_frontier_audit.csv is raw day-clustered p < .05 over eighteen
    # pool-by-panel tests, so LF h=10 pool7 (p=0.010441) draws filled and would
    # not survive Holm over eighteen.  Without the word a reader takes every
    # filled marker for a Holm survivor, the one direction of error that makes
    # the evidence look friendlier.  The line ends at x=865 of an 1185-px
    # canvas and the widest artist reaches 1184, so the addition lands near
    # x=1010 and the tight box does not grow.
    lg.text(2.7, -0.75, "an open marker in (a) is a pool not significantly "
            "better than the reported five (uncorrected)", ha="left",
            va="center", fontsize=9, color=INK2)

    # ------------------------------------------------------ (c) the cascade
    cx = fig.add_axes([0.335, fy(6.05), 0.170, 1.80 / H])
    union = w[g5 | g7].index.tolist()
    cx.set_xlim(-0.6, 1.6)
    cx.set_ylim(len(union) - 0.4, -0.6)
    cx.axis("off")
    for yi, cell in enumerate(union):
        disc, model, h = cell
        for xi, (pool, flag) in enumerate([(POOLS[0], g5[cell]),
                                           ("pool7 (all)", g7[cell])]):
            cx.scatter([xi], [yi], marker="s", s=7.4 ** 2, zorder=3,
                       facecolor=BLUE if flag else "white", edgecolor=BLUE,
                       linewidths=0.7)
        # The stream prefix, which (a), (b) and (d) all carry and this panel
        # used to omit: all nine drawn cells are long-form, and no event-driven
        # cell survives Holm under either pool, so the 8-to-3 collapse the panel
        # shows is a long-form-only result while its header quotes a 69-cell
        # denominator (45 long-form + 24 event-driven) spanning both streams.
        # The labels are right-anchored and their left edge sits at x=139 of an
        # 1185-px canvas, so three characters stay well inside the box, whose
        # left edge is set by the header block at x=5.
        cx.text(-0.85, yi, f"{DLAB[disc]} {model}  h={h}", ha="right",
                va="center", fontsize=9, color=GREY)
    cx.text(0, -1.45, "five", ha="center", va="center", fontsize=9, color=GREY)
    cx.text(1, -1.45, "seven", ha="center", va="center", fontsize=9, color=GREY)
    cx.text(0, -0.95, f"{got['pool5_holm']}", ha="center", va="center",
            fontsize=9, color=BLUE)
    cx.text(1, -0.95, f"{got['pool7_holm']}", ha="center", va="center",
            fontsize=9, color=BLUE)
    header(0.018, 3.50,
           "(c) the 69-cell cascade under the five-member\n"
           "     pool and the completed seven-member one;",
           "     every cell genuine under either is drawn,\n"
           "     filled = adds at Holm, counts of 69.")

    # ----------------------------------------------------- (d) the VIX control
    dx = fig.add_axes([0.655, fy(6.05), 0.315, 1.80 / H])
    labs = []
    for i, r in enumerate(vix.itertuples()):
        labs.append(f"{DLAB[r.disc]}  h={r.h}")
        dx.barh([i], [r.rel_impr_pct], height=0.62, zorder=3,
                color=GREEN if r.p < 0.05 else "white", edgecolor=GREEN,
                linewidth=0.9)
        dx.text(r.rel_impr_pct + 0.35, i, f"{r.rel_impr_pct:.1f}", va="center",
                ha="left", fontsize=9, color=GREY)
    dx.set_ylim(len(labs) - 0.4, -0.6)
    dx.set_yticks(range(len(labs)))
    dx.set_yticklabels(labs, fontsize=9)
    dx.set_xlim(0, 12.5)
    dx.set_xticks([0, 5, 10])
    dx.tick_params(length=0, pad=2)
    dx.spines["left"].set_visible(False)
    dx.grid(axis="x", color=LIGHT, lw=0.5, zorder=0)
    dx.set_axisbelow(True)
    dx.set_xlabel("QLIKE improvement over HAR-RV (%)", fontsize=9, labelpad=2)
    header(0.545, 3.50,
           "(d) the VIX-augmented reference alone:",
           # The Holm column this panel is gated on is computed over the whole
           # nine-row frame (vix_control_harx.csv holm = p x 9: ED h=5 carries
           # p=0.008135 and holm=0.07322), not per stream.  The face used to
           # declare a three-horizon family, under which ED h=5 would clear
           # 0.05 and the "no panel survives" verdict would be false.  Naming
           # the family that was actually applied costs fewer characters on
           # both lines, so the tight box is neutral-to-narrower.
           "     HAR-X against HAR-RV; filled = raw\n"
           "     p < .05; no panel survives Holm in\n"
           "     the nine-panel family.")

    # --------------------------------------------------------------- the note
    note = (
        "Basis: this recomputation exists only on the archived single-seed "
        "(2026) basis, where the\n"
        "reported pool carries 8 of 69, not the primary's 9 of 69; what it "
        "licenses is the five-to-seven\n"
        "change, both halves computed alike. Completing the frontier makes the "
        "price side harder to\n"
        "beat: mean test QLIKE 0.0916 to 0.0887 once HAR-X joins, Holm "
        "survivors 8 to 3 (6 lost, 2 kept,\n"
        "1 gained). No weight explodes — largest absolute log weight 1.82 anywhere, "
        "every pool — so HARQ is of\n"
        "mixed value, not unstable. NOT shown: a seed-ensemble seven-member "
        "cascade, or effect sizes."
    )
    # A hairline across the existing gap between the lower panels and this block,
    # so a reader can see at a glance where the argument stops and the basis
    # statement starts.  It ends at 0.980, inside the widest artist already on
    # the canvas ((d)'s axis label reaches 0.988), so the tight box does not grow
    # and the inclusion scale is unchanged.  Every word of the note is kept; only
    # its ink drops to INK2, and its 1.42 linespacing is preserved rather than
    # handed to note(), which would force 1.32 and tighten the block.
    fig.lines.append(plt.Line2D([0.018, 0.980], [fy(6.375)] * 2,
                                transform=fig.transFigure, color=RULE,
                                linewidth=0.5, zorder=0.5))
    fig.text(0.018, fy(6.42), note, fontsize=9, color=INK2, va="top",
             linespacing=1.42)

    fig.text(0.018, fy(0.05),
             "Price-frontier completeness: what happens when the two archived "
             "price models are put back.",
             fontsize=9, color=GREY, va="top")
    # The banner used to claim "Holm within each family" for the whole figure,
    # but only (c) and (d) are Holm-gated: (a)'s open/filled split is raw
    # day-clustered p across eighteen pool-by-panel tests.  Naming the two
    # panels that carry a correction is three characters shorter, and the line
    # ends at x=1097 of an 1185-px canvas, so it never set the tight box.
    fig.text(0.018, fy(0.26),
             "Day-clustered DM throughout; Holm in (c) and (d); "
             "validation-fitted pool weights frozen on test.",
             fontsize=9, color=INK2, va="top")

    ds.finish(fig, "AR2_price_frontier_completeness", max_render_pt=595.0)


if __name__ == "__main__":
    main()
