"""F5 (dissertation variant) -- Is the maximal pool a shopped reference?

Why this file exists
--------------------
The supplement's generator at `scripts/analysis/supp_figs/F5_maximal_pool_audit.py`
still produces the supplement's PDF and is not edited.  This copy draws the same
figure for Appendix E and differs only in layout:

* Output goes to `writing/dissertation/figures/` through `diss_style.finish`.
* Panels (a) and (b) are 0.44 grid-inches shorter and every block below them
  moves up by the same amount.  At the supplement's geometry the emitted page
  was 625.7 pt tall against Appendix E's 595.1 pt height allowance, so the whole
  figure was scaled to 0.947 at inclusion and its 9 pt type printed at 8.53 pt,
  under the report's 9 pt floor.
* The two footer lines move up by 0.30 rather than 0.44, which opens the
  0.14 in the panel (d) x tick rows needed: the footer line "Both blocks:
  relative QLIKE improvement over the reference (%), one shared scale." was
  printed across the "-6" tick label of BOTH sub-blocks.
* The panel (d) legend anchor moves 0.12 in left so the emitted page is no
  wider than the A4 text block.

No datum changes and every `gate()` runs unchanged. The one re-wrap on this
canvas is the panel (a) caveat, under HIERARCHY below; it keeps every word.

Original docstring follows.
--------------------------------------------------------------------------
F5 -- Is the maximal pool a shopped reference?

WHAT THIS DRAWS
    (a) Holm survivors of 69 by reference specification, ordered by survivors
        left -- which is an absorption ordering, not a forecast-accuracy
        ordering; the panel title says so, because (b) shows the opposite
        ordering on test loss. Seed-ensemble bars with the seed-2026 counts
        behind as ghost bars and the single-recalibrated-HAR primary rung as a
        dashed rule.
    (b) One QLIKE axis per disclosure x horizon panel with four price
        references on it: the validation-selectable best member, the fitted
        five-model pool, the never-fitted equal-weight pool, and the test-best
        member, which is a hindsight selection and is labelled as such. The
        one panel in which the fitted pool is significantly WORSE than the
        validation-best member (long-form h=20) is named on the artefact.
    (c) The single-reference sweep: what the same 15 cells report against each
        of the five single price references and against the fitted pool.
    (d) The stronger-single-reference check: HAR against semivariance HAR on
        18 identical cells, drawn as two 9-row blocks on one shared scale.

SOURCES (all read at run time)
    results/tables/maximal_pool_robustness.csv        basis, ref, adds_holm
    results/tables/maximal_pool_robustness_panels.csv pool/valbest/eqw/oracle
                                                      test QLIKE, dm_*, p_*
    results/tables/maximal_reference_single_refs.csv  ref_price_model,
                                                      rel_impr_pct,
                                                      p_q_clustered
    results/tables/maximal_reference_ensemble.csv     rel_impr_pct_maximal_s26,
                                                      p_q_clustered_s26
    results/tables/stronger_baselines.csv (section = m1_incremental)
                                          A2_rel_impr_pct, A6_shar_rel_impr_pct
    results/tables/control_intersection_ensemble.csv  primary_genuine (38)
    results/tables/pool_frontier_cascade.csv          genuine_holm by pool
    results/tables/pool_frontier_audit.csv            qlike_test by pool
    results/tables/_rangebased_g1_pass.json           single_ref_a2_rank_orig

MAIN-TEXT SENTENCES SUBSTANTIATED
    06_results.tex l.19 "The pool absorbs through its own information rather
        than through weight fitting."
    05_protocol.tex l.26 text's credit is measured "against a single
        recalibrated HAR", the qualifier this figure prices.

BASIS NOTE (load-bearing)
    Panel (c) is drawn entirely on the SEED-2026 text basis: the single-
    reference sweep is written by fc.load's default single-seed path, so its
    pool column is taken from rel_impr_pct_maximal_s26, never from the
    ensemble column, which would put six of the fifteen rows on a different
    basis from their own row neighbours.

TYPE SIZE
    Every label on this canvas is >= 9 pt at 1:1 inclusion scale, the floor
    supp_style declares for the figure set. The canvas is held at 6.4 in wide
    so it is included without down-scaling; vertical space is allocated in
    inches from the top so no block can drift into another.

HIERARCHY (colour and rules only -- never size, never an outward move)
    `finish` writes with bbox_inches="tight", so the PDF page IS the content's
    bounding box and this figure is included height-bound at x1.0054, which is
    what makes its 9 pt type print at 9.05 pt. Enlarging any label, or moving
    the topmost block up or the bottom footer down, would widen the box, scale
    the graphic down and push every printed size under the 9 pt floor. So the
    reading order is carried by ink weight and hairlines, both of which are
    geometrically free:

    INK   panel titles and every data label / tick / axis label. supp_style
          defines INK == GREY, so the data labels below are left spelled GREY
          and are already at this level; the four panel titles say INK, and so
          does the (b) note, whose three sentences are all findings.
    INK2  apparatus: the ordering caveat under (a), the reading key under (b),
          the units-and-basis tail of the (c) title, the convention and
          shared-basis sentences of the (c) note, and the (d) footers.
          These were all set in the ink of the data before, so nothing told a
          reader which sentence was the argument and which the basis statement.
    RULE  three hairlines, each in whitespace the canvas already had, fencing
          the apparatus off from the data above it: over the (b) legend-plus-
          note block, over the (c) note, and over the (d) footers.

    Two blocks mix the roles inside one line, and both handovers are placed by
    measurement rather than by re-wrapping: `text_w` returns the head's advance
    width under the metrics the PDF writer itself uses, so the tail starts one
    space after the head and the line's width is unchanged. `baselines`
    reproduces `Text._get_layout`'s line arithmetic so a block drawn as several
    separately inked one-line texts keeps the baselines the single block had --
    verified against the committed PDF, where all nine affected baselines come
    back identical to 0.0000 pt.

    The panel (a) title needed the one wording change on this canvas: its three
    lines broke mid-clause ("... ordered by" / "survivors left --"), so the
    caveat is re-wrapped onto its own two lines and the comma after
    "specification" becomes the title/caveat boundary. Every word survives, in
    its original order; no note is shortened, dropped or reordered anywhere.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _inclusion_floor
import diss_style as ds
import matplotlib.pyplot as plt
from supp_style import (
    BLUE,
    GREEN,
    GREY,
    INK,
    INK2,
    PURPLE,
    RULE,
    TAB,
    VERM,
    VERM_TXT,
    YELLOW,
    apply_style,
    gate,
)

rob = pd.read_csv(os.path.join(TAB, "maximal_pool_robustness.csv"))
pan = pd.read_csv(os.path.join(TAB, "maximal_pool_robustness_panels.csv"))
sng = pd.read_csv(os.path.join(TAB, "maximal_reference_single_refs.csv"))
ens = pd.read_csv(os.path.join(TAB, "maximal_reference_ensemble.csv"))
sbl = pd.read_csv(os.path.join(TAB, "stronger_baselines.csv"))
cie = pd.read_csv(os.path.join(TAB, "control_intersection_ensemble.csv"))
fca = pd.read_csv(os.path.join(TAB, "pool_frontier_cascade.csv"))
fau = pd.read_csv(os.path.join(TAB, "pool_frontier_audit.csv"))
with open(os.path.join(TAB, "_rangebased_g1_pass.json")) as fh:
    g1 = json.load(fh)

surv = rob.groupby(["basis", "ref"]).adds_holm.sum()
stronger = sbl[sbl.section == "m1_incremental"].copy()
frontier_q = fau.groupby("pool").qlike_test.mean()
frontier_h = fca.groupby("pool").genuine_holm.sum()

# The one panel in which the fitted pool is beaten by the member a forecaster
# could actually have selected on validation. It is the strongest objection
# the panel raises against the reported rung, so it is gated and drawn.
rev = pan.loc[pan.dm_pool_vs_valbest.idxmax()]
REV_COLS = ["pool_test_qlike", "valbest_test_qlike", "eqw_test_qlike",
            "testbest_test_qlike_oracle"]

# ---------------------------------------------------------------- evidence gate
gate(
    {"valbest_ens": 34, "eqw_ens": 17, "fitted_ens": 9,
     "valbest_s26": 28, "eqw_s26": 19, "fitted_s26": 8,
     "primary_ens": 38, "panels": 6,
     "pool_beats_valbest": 5, "pool_loses_to_eqw": 6,
     "reversal_panel": "long_form/20", "reversal_dm": 8.49,
     "reversal_p_lt_1e15": True, "reversal_pool_is_worst": True,
     "single_ref_cells": 15, "stronger_cells": 18,
     "a2_rank1_panels": 0,
     "frontier_pool5_qlike": 0.0916, "frontier_harx_qlike": 0.0887,
     "frontier_harx_better": 4,
     "frontier_pool5_holm": 8, "frontier_pool7_holm": 3},
    {"valbest_ens": int(surv[("ens", "valbest_single")]),
     "eqw_ens": int(surv[("ens", "eqw_pool")]),
     "fitted_ens": int(surv[("ens", "fitted_pool")]),
     "valbest_s26": int(surv[("s26", "valbest_single")]),
     "eqw_s26": int(surv[("s26", "eqw_pool")]),
     "fitted_s26": int(surv[("s26", "fitted_pool")]),
     "primary_ens": int(cie.primary_genuine.sum()), "panels": len(pan),
     "pool_beats_valbest": int(((pan.dm_pool_vs_valbest < 0)
                                & (pan.p_pool_vs_valbest < .05)).sum()),
     "pool_loses_to_eqw": int(((pan.dm_fitted_vs_eqw > 0)
                               & (pan.p_fitted_vs_eqw < .05)).sum()),
     "reversal_panel": f"{rev.disc}/{int(rev.h)}",
     "reversal_dm": round(float(rev.dm_pool_vs_valbest), 2),
     "reversal_p_lt_1e15": bool(rev.p_pool_vs_valbest < 1e-15),
     "reversal_pool_is_worst": bool(float(rev.pool_test_qlike)
                                    == max(float(rev[c]) for c in REV_COLS)),
     "single_ref_cells": len(sng[["disc", "model", "h"]]
                                 .drop_duplicates()),
     "stronger_cells": len(stronger),
     "a2_rank1_panels": int(sum(r["a2_rank"] == 1
                                for r in g1["single_ref_a2_rank_orig"])),
     "frontier_pool5_qlike": round(float(frontier_q["pool5 (paper)"]), 4),
     "frontier_harx_qlike": round(float(frontier_q["pool6 +HARX"]), 4),
     "frontier_harx_better": int((fau[fau.pool == "pool6 +HARX"]
                                  .better_than_pool5 == True).sum()),
     "frontier_pool5_holm": int(frontier_h["pool5 (paper)"]),
     "frontier_pool7_holm": int(frontier_h["pool7 (all)"])},
)

PRIMARY = int(cie.primary_genuine.sum())
REV_TAG = ("LF" if rev.disc == "long_form" else "ED") + f" h={int(rev.h)}"

# ----------------------------------------------------------------- page layout
apply_style(base_size=9)
# W, H are the LAYOUT GRID -- the supplement's own -- and `fy()`/`axes_box()`
# read inches from the top of THAT grid.  HFIG is the height the figure is
# actually drawn at, so the layout compresses by HFIG/H while the type keeps its
# point size.
W, H = 6.4, 9.0
HFIG = 8.60
fig = plt.figure(figsize=(W, HFIG))


def fy(inches_from_top):
    return 1.0 - inches_from_top / H


def fx(inches_from_left):
    return inches_from_left / W


def axes_box(x0, x1, y_top, y_bot):
    return fig.add_axes([fx(x0), fy(y_bot), fx(x1 - x0), (y_bot - y_top) / H])


# ------------------------------------------------------------ hierarchy layer
# The note column: x0 is where every note block already starts and x1 is the
# right edge of the widest note line on the canvas (the (c) note's first line,
# 5.92 in of type from x0). The hairlines span exactly that column, so they add
# no ink outside the box the type already occupies -- the emitted page is the
# content bounding box, and this figure has only ~4.8 pt of width slack before
# width, not height, starts binding the inclusion scale.
NOTE_X0, NOTE_X1 = 0.10, 6.02


PDF_DPI = 72.0     # the dpi the PDF writer lays the page out at
PROP = FontProperties(size=9)      # every label on this canvas is 9 pt sans


def _renderer():
    if hasattr(fig.canvas, "get_renderer"):
        return fig.canvas.get_renderer()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    return FigureCanvasAgg(fig).get_renderer()


def _as_the_page_will(fn):
    """Run the measurement `fn` under the metrics the PDF writer will use.

    `backend_pdf.print_pdf` sets `figure.dpi = 72` and asks FreeType for
    unhinted glyphs; the Agg renderer that answers an interactive measurement
    runs at this canvas's 150 dpi and asks for the autohinter. Hinting is not
    scale-invariant, so those are three different answers to the same question:
    the block line drop below comes out 10.34 pt at 150 dpi autohinted,
    11.45 pt at 72 dpi autohinted, and 10.78 pt here -- against the 10.808 pt
    read back out of the emitted PDF. The first two put a split note's
    continuation lines half a point off their own leading; this one is within
    0.03 pt.
    """
    dpi, hint = fig.dpi, plt.rcParams["text.hinting"]
    fig.set_dpi(PDF_DPI)
    plt.rcParams["text.hinting"] = "none"
    try:
        return fn()
    finally:
        fig.set_dpi(dpi)
        plt.rcParams["text.hinting"] = hint


def text_w(s):
    """Advance width of `s`, in inches, under the metrics that reach the page.

    Used to hand a sentence-boundary split the exact x it must start at, so a
    line that changes ink half way along does not change width. Measured, not
    estimated: an estimate that ran long would widen the page and cost every
    label its printed point size.
    """
    return _as_the_page_will(
        lambda: _renderer().get_text_width_height_descent(s, PROP, False)[0]
        / fig.dpi)


SPACE_W = text_w("n n") - text_w("nn")


def baselines(y_top, lines, linespacing):
    """Figure-fraction baseline of each line of a block set with va="top".

    `Text._get_layout` lays a block out as: the first baseline one net height
    below the ink top, then each next baseline the previous line's descent plus
    max(one font line, this line's own net height) further down, with every
    metric floored at those of "lp". Reproducing that here is what lets a note
    whose sentences carry different roles be drawn as separately inked one-line
    texts that stand exactly where the single block stood.

    The alternative -- anchoring each continuation on leading blank lines in
    its own text -- places the type just as well but gives each text a bounding
    box that covers the lines above it, and `diss_style.collision_pairs` then
    reports overlaps where there is no ink: this figure went from 0 reported
    pairs to 7 that way, and that check earns its keep only while it is quiet.
    """
    def probe():
        r = _renderer()

        def metrics(s):
            return r.get_text_width_height_descent(s, PROP, False)

        _, lp_h, lp_d = metrics("lp")
        min_dy = (lp_h - lp_d) * linespacing
        hs, ds = [], []
        for s in lines:
            _, h, d = metrics(s)
            hs.append(max(h, lp_h))
            ds.append(max(d, lp_d))
        ys = [-(hs[0] - ds[0])]
        for i in range(1, len(lines)):
            ys.append(ys[-1] - ds[i - 1]
                      - max(min_dy, (hs[i] - ds[i]) * linespacing))
        return [y / fig.dpi for y in ys]

    return [y_top + dy / HFIG for dy in _as_the_page_will(probe)]


def rule(y_top):
    """Hairline across the note column, `y_top` grid-inches from the top."""
    fig.add_artist(plt.Line2D([fx(NOTE_X0), fx(NOTE_X1)], [fy(y_top)] * 2,
                              transform=fig.transFigure, color=RULE,
                              linewidth=0.5, zorder=0.5))


# 1.06 grid-inches tall, against the supplement's 1.50: the 0.44 recovered
# here is what brings the emitted page inside the height allowance.
ax_a = axes_box(0.55, 3.25, 0.82, 1.88)
ax_b = axes_box(3.60, 6.30, 0.82, 1.88)

# ------------------------------------------------------------------- panel (a)
specs = [("valbest_single", "validation-best\nsingle member"),
         ("eqw_pool", "equal-weight\n5-model pool"),
         ("fitted_pool", "fitted\n5-model pool")]
xs = np.arange(len(specs))
ens_v = [int(surv[("ens", k)]) for k, _ in specs]
s26_v = [int(surv[("s26", k)]) for k, _ in specs]

ax_a.bar(xs, s26_v, width=0.70, color="white", edgecolor=GREY, linewidth=0.6,
         hatch="////", zorder=1, label="seed-2026")
ax_a.bar(xs, ens_v, width=0.44, color=BLUE, edgecolor="none", zorder=2,
         label="seed-ensemble")
for x, ve, vs in zip(xs, ens_v, s26_v, strict=False):
    # Inside the bar, not above it.  On the shorter panel a label sitting 0.7
    # units above a 34-high bar has its cap height carried past the y=38
    # primary-rung rule, which then strikes the digits through; the ghost-bar
    # labels beside it stay outside because none of them reaches 38.
    ax_a.text(x, ve - 0.9, str(ve), ha="center", va="top", fontsize=9,
              color="white", fontweight="bold")
    ax_a.text(x + 0.30, vs + 0.7, str(vs), ha="center", va="bottom",
              fontsize=9, color=GREY)
ax_a.axhline(PRIMARY, color=VERM_TXT, linewidth=0.8, linestyle=(0, (4, 2.2)),
             zorder=3)
ax_a.text(-0.46, PRIMARY + 1.0, f"{PRIMARY} = primary rung\n(single "
          f"recalibrated HAR)", fontsize=9, color=VERM_TXT, va="bottom",
          ha="left", linespacing=1.3)
ax_a.set_xticks(xs)
ax_a.set_xticklabels([lab for _, lab in specs], fontsize=9, linespacing=1.3)
# 64 rather than the supplement's 52: on the shorter panel the upper-right
# legend reached down to the y=38 primary-rung rule, which then struck
# through the words "seed-ensemble".  Raising the ceiling lifts the legend
# clear of the rule; no bar and no tick value changes.
ax_a.set_ylim(0, 64)
ax_a.set_yticks([0, 20, 40, 60])
ax_a.set_ylabel("Holm survivors of 69", fontsize=9)
ax_a.tick_params(axis="y", labelsize=9)
ax_a.tick_params(axis="x", length=0, pad=3)
ax_a.legend(fontsize=9, loc="upper right", handlelength=1.2,
            borderpad=0.15, labelspacing=0.2, handletextpad=0.4)
# The title and its ordering caveat were one grey prose block whose line breaks
# fell mid-clause ("... ordered by" / "survivors left --"), so the panel had no
# title a reader could pick out and the caveat read as part of it. The title
# line is now a title in the primary ink; the caveat keeps every word, in
# order, and is set in the apparatus ink on the two lines below, re-wrapped so
# that break falls between clauses instead of inside one. Footprint is
# unchanged: three lines on the block's own baselines, and the longest of them
# is 0.03 in narrower than the line it replaces.
A_LINES = ["(a) survivors by reference specification",
           "ordered by survivors left -- an absorption ordering,",
           "not a test forecast-accuracy ordering; see (b)"]
A_YS = baselines(fy(0.14), A_LINES, 1.35)
fig.text(fx(0.10), fy(0.14), A_LINES[0],
         ha="left", va="top", fontsize=9, color=INK)
for y, line in zip(A_YS[1:], A_LINES[1:], strict=False):
    fig.text(fx(0.10), y, line,
             ha="left", va="baseline", fontsize=9, color=INK2)

# ------------------------------------------------------------------- panel (b)
pan = pan.sort_values(["disc", "h"]).reset_index(drop=True)
SER = [("valbest_test_qlike", "validation-best member", "o", YELLOW),
       ("pool_test_qlike", "fitted 5-model pool", "s", BLUE),
       ("eqw_test_qlike", "equal-weight pool", "^", GREEN),
       ("testbest_test_qlike_oracle", "test-best member (hindsight)", "X",
        VERM)]
X0, X1 = 0.10, 0.80
for i, row in pan.iterrows():
    vals = np.array([float(row[c]) for c, _, _, _ in SER])
    lo, hi = vals.min(), vals.max()
    ax_b.plot([X0, X1], [i, i], color="#C9C9C9", linewidth=0.7, zorder=1)
    for (c, lab, mk, col) in SER:
        xn = X0 + (float(row[c]) - lo) / (hi - lo) * (X1 - X0)
        ax_b.plot([xn], [i], marker=mk, markersize=5.2, color=col,
                  markeredgecolor="white", markeredgewidth=0.4, zorder=3,
                  label=lab if i == 0 else None)
    tag = ("LF" if row.disc == "long_form" else "ED") + f" h={int(row.h)}"
    is_rev = (row.disc == rev.disc) and (int(row.h) == int(rev.h))
    ax_b.text(-0.62, i, tag, ha="left", va="center", fontsize=9,
              color=(VERM_TXT if is_rev else GREY),
              fontweight=("bold" if is_rev else "normal"))
    ax_b.text(X0 - 0.035, i, f"{lo:.4f}", ha="right", va="center",
              fontsize=9, color=GREY)
    ax_b.text(X1 + 0.035, i, f"{hi:.4f}", ha="left", va="center", fontsize=9,
              color=GREY)
ax_b.set_xlim(-0.64, 1.24)
ax_b.set_ylim(len(pan) - 0.4, -0.6)
ax_b.set_xticks([])
ax_b.set_yticks([])
for sp in ax_b.spines.values():
    sp.set_visible(False)
# Same split, on the line break the text already had: the title in primary ink,
# the reading key -- which is apparatus, not a finding -- in the apparatus ink.
B_LINES = ["(b) test QLIKE, one axis per panel", "(left = lower QLIKE)"]
B_YS = baselines(fy(0.14), B_LINES, 1.35)
fig.text(fx(3.60), fy(0.14), B_LINES[0],
         ha="left", va="top", fontsize=9, color=INK)
fig.text(fx(3.60), B_YS[1], B_LINES[1],
         ha="left", va="baseline", fontsize=9, color=INK2)
fig.legend(*ax_b.get_legend_handles_labels(), fontsize=9, ncol=4,
           loc="upper left", bbox_to_anchor=(fx(0.10), fy(2.28)),
           handlelength=1.0, columnspacing=1.2, borderpad=0.1,
           handletextpad=0.35, frameon=False)
# Hairline between the (a)/(b) data and the apparatus under them, drawn on the
# shared legend's own anchor line: the legend is a key, so it belongs below the
# fence with the note prose, and that band is the one place here with room --
# measured on the emitted page it clears the panel (a) tick rows by 5.8 pt and
# the legend by 5.2 pt, where a rule below the legend had 1.1 pt of clearance.
# It costs no height either way; the gap was already on the canvas. Every one
# of the three sentences below is a finding, not a basis statement, so it keeps
# the primary ink -- the rule is what says it is caption prose rather
# than one more data label.
rule(2.28)
fig.text(fx(0.10), fy(2.54),
         "(b) The fitted pool beats the validation-best member in 5 of 6 "
         "panels (clustered DM -7.24 to -2.05). In the\n"
         "sixth, long-form h=20, it is significantly worse than that member "
         "(DM +8.49, p < 1e-15) and is the worst of\n"
         "the four references drawn. It loses to the never-fitted "
         "equal-weight pool in 6 of 6 panels, on test QLIKE.",
         fontsize=9, color=INK, va="top", ha="left", linespacing=1.4)

# ------------------------------------------------------------------- panel (c)
REFS = ["A2_har_rv", "A6_shar", "A3_garch", "A4_egarch", "A5_arima"]
cells = (sng[["disc", "model", "h"]].drop_duplicates()
         .sort_values(["disc", "model", "h"]).reset_index(drop=True))
piv = sng.pivot_table(index=["disc", "model", "h"], columns="ref_price_model",
                      values="rel_impr_pct")
pvl = sng.pivot_table(index=["disc", "model", "h"], columns="ref_price_model",
                      values="p_q_clustered")
pool = ens.set_index(["disc", "model", "h"])[["rel_impr_pct_maximal_s26",
                                              "p_q_clustered_s26"]]

M = np.zeros((len(cells), 6))
P = np.ones((len(cells), 6))
for i, r in cells.iterrows():
    key = (r.disc, r.model, r.h)
    for j, ref in enumerate(REFS):
        M[i, j] = piv.loc[key, ref]
        P[i, j] = pvl.loc[key, ref]
    M[i, 5] = pool.loc[key, "rel_impr_pct_maximal_s26"]
    P[i, 5] = pool.loc[key, "p_q_clustered_s26"]

C_TOP, C_PITCH = 3.34, 0.142
C_X0, C_X1 = 1.62, 5.52
ax_c = axes_box(C_X0, C_X1, C_TOP, C_TOP + len(cells) * C_PITCH)
cmap = LinearSegmentedColormap.from_list(
    "supp_div", [VERM, "#F2DCCB", "#FFFFFF", "#CFE2EF", BLUE])
lim = float(np.abs(M).max())
norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
ax_c.imshow(M, cmap=cmap, norm=norm, aspect="auto",
            extent=(-0.5, 5.5, len(cells) - 0.5, -0.5))
for i in range(len(cells)):
    for j in range(6):
        if P[i, j] < 0.05:
            ax_c.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor="none", edgecolor=GREY,
                                     linewidth=0.9, zorder=3))
        ax_c.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                  fontsize=9,
                  color=("white" if abs(M[i, j]) > 0.80 * lim else GREY),
                  zorder=4)
ax_c.set_xticks(range(6))
ax_c.set_xticklabels(REFS + ["fitted pool"], fontsize=9)
ax_c.set_yticks(range(len(cells)))
ax_c.set_yticklabels([f"h={int(r.h)}" for _, r in cells.iterrows()],
                     fontsize=9)
ax_c.tick_params(axis="both", length=0, pad=3)
# model block labels once per three rows, F14-style, so the row labels
# themselves stay short enough to be set at 9 pt
hi_row = int(cells.index[(cells.disc == "long_form")
                         & (cells.model == "B2_tfidf_ridge")
                         & (cells.h == 5)][0])
C_LAB_X = -0.5 - 0.46 * 6.0 / (C_X1 - C_X0)      # 0.46 in left of the axes
for (disc, model), grp in cells.groupby(["disc", "model"], sort=False):
    i0, i1 = float(grp.index.min()), float(grp.index.max())
    marked = (disc == "long_form") and (model == "B2_tfidf_ridge")
    ax_c.text(C_LAB_X, (i0 + i1) / 2.0,
              ("LF" if disc == "long_form" else "ED") + f" / {model}",
              ha="right", va="center", fontsize=9,
              color=(VERM_TXT if marked else GREY),
              fontweight=("bold" if marked else "normal"), clip_on=False)
# One line, two roles: the title, then what the numbers in the cells are and on
# which basis they were computed. The tail starts at the measured width of the
# head plus one space, so the line is set exactly where it was set before and
# only its ink changes at the colon.
C_TITLE = "(c) the same cell against each single price reference:"
fig.text(fx(0.10), fy(3.18), C_TITLE,
         ha="left", va="center", fontsize=9, color=INK)
fig.text(fx(0.10 + text_w(C_TITLE) + SPACE_W), fy(3.18),
         "relative QLIKE improvement (%), seed-2026 basis",
         ha="left", va="center", fontsize=9, color=INK2)

# The (c) note is the one block on this canvas that mixes the two roles: its
# first sentence is the panel's finding, its second and third are the outline
# convention and the shared-basis statement, and all three were set in the ink
# of the data. The finding keeps the primary ink; the convention and the basis
# move to the apparatus ink, and the mid-line handover is placed by measurement
# so line 2 keeps its width. Drawn as one-line texts on the baselines the
# three-line block itself would have used, so every line stands where it stood,
# the two halves of line 2 share one baseline exactly, and no text's box spans
# a line it does not ink.
rule(5.70)
C_L1 = ("Marked row (LF / B2_tfidf_ridge, h=5): the same cell reports +1.06 "
        "to +3.53 depending only on which single")
C_L2A = "price model is the reference."
C_L2B = ("Cell outline: clustered p < .05. "
         "Every column is computed on identical rows,")
C_L3 = ("and the pool column is the seed-2026 pool, so all six columns "
        "share one basis.")
C_LINES = [C_L1, C_L2A + " " + C_L2B, C_L3]
C_YS = baselines(fy(5.76), C_LINES, 1.4)
fig.text(fx(0.10), fy(5.76), C_LINES[0],
         fontsize=9, color=INK, ha="left", va="top")
fig.text(fx(0.10), C_YS[1], C_L2A,
         fontsize=9, color=INK, ha="left", va="baseline")
fig.text(fx(0.10 + text_w(C_L2A) + SPACE_W), C_YS[1], C_L2B,
         fontsize=9, color=INK2, ha="left", va="baseline")
fig.text(fx(0.10), C_YS[2], C_LINES[2],
         fontsize=9, color=INK2, ha="left", va="baseline")

# ------------------------------------------------------------------- panel (d)
# Two 9-row blocks rather than one 18-row column: at 18 rows the row labels
# would fall under 9 pt on this canvas. Both blocks carry the same x limits,
# so the two channels stay directly comparable.
stronger = stronger.sort_values(["disclosure", "text_model", "h"])
stronger = stronger.reset_index(drop=True)
D_TOP, D_PITCH = 6.80, 0.145
blocks = [("event_driven", 1.45, 3.05, "event-driven, 9 cells"),
          ("long_form", 4.50, 6.10, "long-form, 9 cells")]
dlo = float(min(stronger.A2_rel_impr_pct.min(),
                stronger.A6_shar_rel_impr_pct.min()))
dhi = float(max(stronger.A2_rel_impr_pct.max(),
                stronger.A6_shar_rel_impr_pct.max()))
pad = 0.09 * (dhi - dlo)
handles = None
for disc, x0, x1, head in blocks:
    sub = stronger[stronger.disclosure == disc].reset_index(drop=True)
    axd = axes_box(x0, x1, D_TOP, D_TOP + len(sub) * D_PITCH)
    axd.axvline(0.0, color=GREY, linewidth=0.6, zorder=1)
    for i, r in sub.iterrows():
        a2, sh = float(r.A2_rel_impr_pct), float(r.A6_shar_rel_impr_pct)
        axd.plot([a2, sh], [i, i], color=GREY, linewidth=1.0, zorder=2)
        # A2 is drawn as an open ring so that the A6 diamond, which sits on
        # top of it in every one of the 18 cells, cannot hide it.
        axd.plot([a2], [i], marker="o", markersize=6.0,
                 markerfacecolor="white", markeredgecolor=BLUE,
                 markeredgewidth=1.1, zorder=3,
                 label="recalibrated HAR (A2)" if i == 0 else None)
        axd.plot([sh], [i], marker="D", markersize=3.4, color=PURPLE,
                 markeredgecolor="none", zorder=4,
                 label="semivariance HAR (A6)" if i == 0 else None)
    axd.set_yticks(range(len(sub)))
    axd.set_yticklabels([f"h={int(r.h)}" for _, r in sub.iterrows()],
                        fontsize=9)
    axd.set_ylim(len(sub) - 0.5, -0.5)
    axd.set_xlim(dlo - pad, dhi + pad)
    axd.tick_params(axis="both", labelsize=9, length=2.5, pad=3)
    axd.set_xticks([-6, -3, 0, 3])
    span = (dhi + pad) - (dlo - pad)
    lab_x = (dlo - pad) - 0.46 * span / (x1 - x0)   # 0.46 in left of the axes
    for model, grp in sub.groupby("text_model", sort=False):
        i0, i1 = float(grp.index.min()), float(grp.index.max())
        axd.text(lab_x, (i0 + i1) / 2.0, model, ha="right", va="center",
                 fontsize=9, color=GREY, clip_on=False)
    fig.text(fx(x0), fy(D_TOP - 0.18), head, ha="left", va="center",
             fontsize=9, color=GREY)
    if disc == "long_form":
        handles = axd.get_legend_handles_labels()

fig.text(fx(0.10), fy(6.40),
         "(d) 18 identical cells, HAR against semivariance HAR",
         ha="left", va="center", fontsize=9, color=INK)
fig.legend(*handles, fontsize=9, ncol=2, loc="center left",
           bbox_to_anchor=(fx(3.10), fy(6.40)), handlelength=1.2,
           columnspacing=1.6, borderpad=0.1, handletextpad=0.4, frameon=False)
dshift = float((stronger.A6_shar_rel_impr_pct
                - stronger.A2_rel_impr_pct).abs().max())
# Both footers are pure apparatus -- what the axis measures, on which scale,
# and the denominator of the largest shift -- so they take the apparatus ink
# and sit under a hairline. The rule goes at 8.36, in the clearance between the
# panel (d) tick row and the first footer line: on the emitted page that leaves
# 3.2 pt above the rule and 3.7 pt below it. The footer lines themselves do not
# move, because the lower one sets the bottom edge of the emitted page.
rule(8.36)
fig.text(fx(0.10), fy(8.48),
         "Both blocks: relative QLIKE improvement over the reference (%), "
         "one shared scale.",
         fontsize=9, color=INK2, ha="left", va="center")
fig.text(fx(0.10), fy(8.64),
         f"Largest shift across all 18 cells: {dshift:.2f} percentage points.",
         fontsize=9, color=INK2, ha="left", va="center")

# diss_style's max_render_pt is its width-only overflow gate (the height a
# graphic WOULD occupy at the full \\textwidth); the gate that matters here is
# the printed type floor checked immediately below.
ds.finish(fig, "F5_maximal_pool_audit", max_render_pt=620.0,
          note="dissertation variant: panels (a)/(b) shortened, blocks below "
               "shifted up, panel (d) footer cleared of the tick row")
_inclusion_floor.check("F5_maximal_pool_audit", drawn_floor_pt=9.0)
