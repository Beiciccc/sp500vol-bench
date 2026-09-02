"""Appendix figure FP2 — the Yelp second-domain ladder, in full.

The same audit run on a domain where text is supposed to work.  Three things
the two Yelp tables hold that no single number in the report conveys:

  (a) the field-standard design's apparent gain, and how much of it a
      ZERO-TEXT business mean recovers by itself (139 % at one month, 148 %
      at three) — the cross-domain mirror of the SEC finding;
  (b) what survives the chronological protocol once the identity control is
      inside the reference: increments two orders of magnitude smaller than
      the field-design headline, each shown against its own
      minimum-detectable effect;
  (c) why the field's standard fix does not work on a prompted model: an
      entity-disjoint split kills the fitted arm and leaves the zero-content
      identity probe untouched.

Sources
-------
results/tables/yelp_cascade.csv          the five-row cascade at h = 1 m and 3 m
results/tables/yelp_cascade.md           MDEs, placebo means and the panel gates
results/tables/yelp_entity_disjoint.csv  the split-rule experiment
"""
import os
import re
import sys
import textwrap

import numpy as np
import pandas as pd

ANALYSIS = "scripts/analysis"
sys.path.insert(0, ANALYSIS)

from supp_style import (apply_style, gate, BLUE, SKY, VERM, VERM_TXT,  # noqa: E402
                        GREEN, YELLOW, PURPLE, GREY, LIGHT, TAB, REPO,
                        INK2, RULE)
import diss_style as ds  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- evidence
casc = pd.read_csv(os.path.join(TAB, "yelp_cascade.csv"))
disj = pd.read_csv(os.path.join(TAB, "yelp_entity_disjoint.csv"))

# the recovery ratios live in the cascade's row-1 note field
rec = {}
for _, r in casc[casc.row == 1].iterrows():
    rec[int(r.h)] = int(re.search(r"=\s*(\d+)%", str(r.note)).group(1))


def cell(row, h):
    r = casc[(casc.row == row) & (casc.h == h)].iloc[0]
    return float(r.delta_rel_pct)


# minimum detectable effects, from the cascade table's pre-registered note
MDE = {("ar", 1): 0.18, ("ar", 3): 0.40, ("entity", 1): 0.08,
       ("entity", 3): 0.08}

ARMS = ["TF-IDF (fitted)", "70B prompted (zero-shot)",
        "70B identity probe (zero-content)"]
SPLIT_A = "A: shared-entity (chronological)"
SPLIT_B = "B: entity-disjoint (standard fix)"


def dj(split, arm, h, col):
    r = disj[(disj.split == split) & (disj.arm == arm) & (disj.h == h)]
    return float(r.iloc[0][col])


# ------------------------------------------------------------------- gate
gate(
    {
        "apparent_gain": (28.43, 36.64),
        "identity_recovery_pct": (139, 148),
        "textalone_chrono": (-24.70, -35.29),
        "identity_control_cost": (-10.58, -9.48),
        "ar_stage_residual": (0.35, 0.87),
        "entity_stage_residual": (0.37, 0.61),
        "entity_coverage": (0.834, 0.0),
        "n_test_h1": (28134, 7183),
    },
    {
        "apparent_gain": (round(cell(1, 1), 2), round(cell(1, 3), 2)),
        "identity_recovery_pct": (rec[1], rec[3]),
        "textalone_chrono": (round(cell(2, 1), 2), round(cell(2, 3), 2)),
        "identity_control_cost": (round(cell(4, 1), 2), round(cell(4, 3), 2)),
        "ar_stage_residual": (round(cell(3, 1), 2), round(cell(3, 3), 2)),
        "entity_stage_residual": (round(cell(5, 1), 2), round(cell(5, 3), 2)),
        "entity_coverage": (
            round(dj(SPLIT_A, ARMS[0], 1, "entity_coverage_test"), 3),
            round(dj(SPLIT_B, ARMS[0], 1, "entity_coverage_test"), 3)),
        "n_test_h1": (int(dj(SPLIT_A, ARMS[0], 1, "n_test")),
                      int(dj(SPLIT_B, ARMS[0], 1, "n_test"))),
    },
)

# ------------------------------------------------------------------ canvas
apply_style(9)
fig = plt.figure(figsize=ds.canvas(7.55))
gs = fig.add_gridspec(3, 1, height_ratios=[2.60, 1.85, 2.25],
                      # bottom was 0.196, which left 6 pt between panel (c)'s
                      # axis label and the note block -- no room for the
                      # separating hairline to read as a separator.  Lifting the
                      # stack 1 % of the canvas moves type INWARD, into space the
                      # tight bounding box already encloses, so the page box and
                      # therefore every printed point size are unchanged.
                      left=0.375, right=0.972, top=0.930, bottom=0.206,
                      hspace=0.86)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
axC = fig.add_subplot(gs[2])

H1, H3 = BLUE, SKY
BARH = 0.30
# Panel (a) draws its bars thicker than panel (b) does, and that is forced.  Row
# 1's two value labels are printed INSIDE their bars, and at the figure's 8.9 pt
# the cap height of "+28.43 %" is 13.3 px against a 13.6 px bar whose top and
# bottom edges are already covered by the outline bar's stroke -- so the white
# glyph tops punched holes clean through the orange band above them and the "%"
# cleared the bar entirely and went white-on-white.  The type cannot shrink:
# the appendices's floor is 9 pt printed, which this figure meets at 8.9 pt drawn
# only because it is included at 1.04x.  So the bar grows instead.  0.38 y-units
# leaves about 1 px of dark fill above and below the cap height; the axes box,
# both limits and every label position are untouched, so the tight page box
# cannot move.  At panel (a)'s scale 0.38 units is 22.8 px, which also brings
# the printed bar thickness into line with panel (b)'s 22.5 px.
BARH_A = 0.38
# matplotlib's va="center" centres the text BOX, which carries the font's full
# descent even for a run of digits that has none, so a value label sits ~2 px
# above the bar it labels.  Subtracting it centres the INK on the bar, which is
# what the in-bar labels need to clear the outline stroke at both edges.
LAB_DY = 0.033


def val(v):
    """A signed percent label, with U+2212 rather than a hyphen for the minus.

    The x-axis ticks are drawn by matplotlib with a true minus, so a data label
    reading "-35.29 %" a few centimetres from a tick reading "-40" set two
    different glyphs for one operator.
    """
    return f"{v:+.2f} %".replace("-", "−")


# ============================ (a) the field design and the identity recovery
big = [
    ("1  pooled random split, text\n     vs. the pooled mean", 1),
    # One baseline, one name.  This row and panel (b)'s row 3 said "the
    # recalibrated AR" while row 4 below and panel (c)'s axis title said
    # "recal. AR", so the same reference was named three ways, twice within
    # two rows of itself.  The short form is the one that cannot widen the
    # label column: row 4's second line is already its widest line.
    ("2  chronological, text alone\n     vs. recal. AR", 2),
    ("4  chronological, AR + business\n     mean, zero text  vs. recal. AR", 4),
]
yb = np.arange(len(big))[::-1]
for y, (lab, row) in zip(yb, big):
    if row == 1:                                   # the zero-text recovery
        for k, h in enumerate((1, 3)):
            v = cell(row, h) * rec[h] / 100.0
            off = BARH_A / 1.9 if k == 0 else -BARH_A / 1.9
            axA.barh(y + off, v, height=BARH_A, color="none", edgecolor=VERM,
                     lw=1.1, zorder=4)
            # The ratio is NOT a position on this percent axis (the bar tip
            # is at 39.5 and 54.2), so it is printed in multiplication form:
            # "139 %" beside "+28.43 %" put two meanings of "%" in one row.
            # The multiplication form is also the narrower string, and this
            # label is the figure's right-most ink, so the tight box shrinks.
            axA.text(v + 1.2, y + off - LAB_DY, f"\u00d7{rec[h] / 100:.2f}",
                     va="center", ha="left",
                     fontsize=8.9, color=VERM_TXT, zorder=5)
    for k, (h, col) in enumerate(((1, H1), (3, H3))):
        v = cell(row, h)
        off = BARH_A / 1.9 if k == 0 else -BARH_A / 1.9
        axA.barh(y + off, v, height=BARH_A, color=col, edgecolor="none",
                 zorder=3)
        if row == 1:                               # inside, clear of the outline
            # White reads on the dark BLUE h = 1 bar (5.2:1) but not on the
            # light SKY h = 3 bar (2.3:1, and gone in greyscale), so the
            # h = 3 label takes the body-text GREY (4.9:1 on SKY).  Colour
            # only: same string, same coordinates, same alignment.
            axA.text(v - 1.2, y + off - LAB_DY, val(v), va="center",
                     ha="right", fontsize=8.9,
                     color="white" if k == 0 else GREY, zorder=5)
        else:
            axA.text(v - 1.2, y + off - LAB_DY, val(v), va="center",
                     ha="right", fontsize=8.9, color=GREY, zorder=5)

axA.set_yticks(yb)
axA.set_yticklabels([b[0] for b in big], fontsize=8.9)
# The zero rule spans the three bar rows only.  Drawn across the whole axes it
# ran through the legend entry "zero-text business mean, same split".
axA.plot([0, 0], [-0.50, len(big) - 0.45], color=GREY, lw=0.8, zorder=2)
# The left limit carries the "-35.29 %" value label as well as the bar: at -52
# the label started outside the axes and the spine struck out its minus sign.
axA.set_xlim(-58, 62)
axA.set_xticks([-40, -20, 0, 20, 40, 60])
axA.set_ylim(-1.95, len(big) - 0.45)
# delta_rel_pct is (reference - arm) / reference, so a POSITIVE value is a
# FALL in mean squared error.  "change in ..." left the sign to be guessed,
# and guessed the wrong way it inverts every reading in all three panels.
axA.set_xlabel("reduction in mean squared error vs. that row's reference, %")
axA.plot([], [], color=H1, lw=5, label="$h = 1$ month")
axA.plot([], [], color=H3, lw=5, label="$h = 3$ months")
axA.plot([], [], color="none", marker="s", ms=7, mfc="none", mec=VERM,
         mew=1.1, ls="none",
         label="zero-text business mean, same split")
axA.legend(loc="lower left", fontsize=8.9, handlelength=1.2,
           handletextpad=0.5, borderpad=0.25, labelspacing=0.25, ncol=2,
           columnspacing=1.2)

# ======================================= (b) what survives, against its MDE
small = [
    ("3  AR + text\n     vs. recal. AR", 3, "ar"),
    ("5  AR + business mean + text\n     vs. AR + business mean", 5, "entity"),
]
# The minimum-detectable-effect tick used to run BARH/1.7 either side of its
# bar's centre, i.e. past both bar edges.  Row 5's two effects are the same
# number (0.08 at both horizons), so the two ticks stood at one x, each
# overhung into the 1 px gap between the bars, and merged into a single
# unbroken 50 px rule across the pair -- one mark where the panel means two.
# The same overhang closed the gap under row 3's "+0.35 %" label, where the
# h = 3 tick's top stopped 1 px short of the glyphs and read as an underline.
# Confining the tick to the middle 72 % of its own bar breaks the row-5 pair
# with 7 px of white and opens 7 px under the row-3 label, and moves nothing:
# it is strictly less ink, inside bars that have not moved.
MDEH = BARH * 0.36
ys = np.arange(len(small))[::-1]
for y, (lab, row, stage) in zip(ys, small):
    for k, (h, col) in enumerate(((1, H1), (3, H3))):
        v = cell(row, h)
        off = BARH / 1.9 if k == 0 else -BARH / 1.9
        axB.barh(y + off, v, height=BARH, color=col, edgecolor="none",
                 zorder=3)
        m = MDE[(stage, h)]
        axB.plot([m, m], [y + off - MDEH, y + off + MDEH],
                 color=VERM, lw=1.4, zorder=6)
        axB.text(v + 0.012, y + off, val(v), va="center", ha="left",
                 fontsize=8.9, color=GREY, zorder=5)
axB.annotate("minimum detectable\neffect, 80 % power",
             # the leader now stops at the top edge of the bar the tick is
             # inside, directly above it, rather than at the old tick tip
             xy=(MDE[("ar", 1)], 1 + BARH / 1.9 + BARH / 2),
             xytext=(0.47, 1.66), fontsize=8.9, color=VERM_TXT,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=VERM, lw=0.8,
                             shrinkA=2, shrinkB=1))
axB.set_yticks(ys)
axB.set_yticklabels([s[0] for s in small], fontsize=8.9)
axB.axvline(0, color=GREY, lw=0.8)
axB.set_xlim(0, 1.30)
axB.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axB.set_ylim(-0.72, 2.05)
axB.set_xlabel("reduction in mean squared error vs. that row's reference, %")

# ================================== (c) the split rule, changed on its own
yc = np.array([6, 5, 4, 2, 1, 0])
labels = ARMS + ARMS

for y, (h, arm) in zip(yc, [(h, a) for h in (1, 3) for a in ARMS]):
    va = dj(SPLIT_A, arm, h, "rel_vs_recalAR")
    vb = dj(SPLIT_B, arm, h, "rel_vs_recalAR")
    pa = dj(SPLIT_A, arm, h, "p_month")
    pb = dj(SPLIT_B, arm, h, "p_month")
    axC.plot([va, vb], [y, y], color=GREY, lw=0.8, alpha=0.6, zorder=2)
    # At h = 1 the prompted pair sits 0.011 apart (3.6 pt) and the B square
    # was drawn over the A circle, leaving a crescent of rim -- and that
    # circle is the one HOLLOW A marker in the block (p_month = 0.0524), so
    # the fill-carries-significance key was unreadable exactly where it
    # works.  Fix is interior only: a white knock-out halo under each marker
    # and the circle above the square.  Both markers keep their data
    # coordinates and sit far inside the axes, so the box does not move.
    axC.plot([vb], [y], marker="s", ms=7.0, ls="none", zorder=3.5,
             mfc="white", mec="white", mew=1.0)
    axC.plot([vb], [y], marker="s", ms=5.6, zorder=4,
             mfc=VERM if pb < 0.05 else "white", mec=VERM, mew=1.1)
    axC.plot([va], [y], marker="o", ms=7.4, ls="none", zorder=4.5,
             mfc="white", mec="white", mew=1.0)
    axC.plot([va], [y], marker="o", ms=6.0, zorder=5,
             mfc=BLUE if pa < 0.05 else "white", mec=BLUE, mew=1.1)
    axC.axhline(y, color=LIGHT, lw=0.5, zorder=0)

axC.axhline(3.35, color=GREY, lw=0.6, ls=":")
axC.set_yticks(yc)
axC.set_yticklabels(labels, fontsize=8.9)
axC.set_xlim(0, 0.98)
axC.axvline(0, color=GREY, lw=0.8)
axC.set_xlabel("reduction in mean squared error vs. recal. AR, %")
axC.set_ylim(-0.80, 7.45)
axC.plot([], [], marker="o", ms=6.0, ls="none", mfc=BLUE, mec=BLUE,
         label="A: shared-entity split")
axC.plot([], [], marker="s", ms=5.6, ls="none", mfc=VERM, mec=VERM,
         label="B: entity-disjoint split")
# Marker FILL is a second channel -- it carries the month-clustered test -- and
# both key swatches were solid, so the key read as declaring fill for the A/B
# channel while three markers on the page were hollow.  The third entry is that
# channel's own key, and it is drawn as a BLUE open circle because that is a
# mark that actually appears in the panel (the h = 1 prompted arm, p = 0.0524).
axC.plot([], [], marker="o", ms=6.0, ls="none", mfc="white", mec=BLUE,
         mew=1.1, label="open: month-clustered p ≥ 0.05")
# The key was at "lower right", which put its two markers in the gaps between
# the h = 3 rows, on the same pale row rules as the data, close enough to the
# TF-IDF and identity-probe squares to be read as two more observations.  It
# moves to the h = 1 block's right half, which is empty from x = 0.35 out (the
# block's right-most marker is at 0.29), and takes a white patch so the row
# rules stop at it instead of running through it.  Inside the axes, no limit
# and no data coordinate changed, so the page box is untouched.
axC.legend(loc="upper right", fontsize=8.9, handletextpad=0.4,
           borderpad=0.3, labelspacing=0.25, borderaxespad=0.3,
           frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
axC.text(0.012, 6.95, "$h = 1$ month", fontsize=8.9, color=GREY,
         ha="left", va="center",
         bbox=dict(facecolor="white", edgecolor="none", pad=1.4))
axC.text(0.012, 2.72, "$h = 3$ months", fontsize=8.9, color=GREY,
         ha="left", va="center",
         bbox=dict(facecolor="white", edgecolor="none", pad=1.4))

# ---------------------------------------------------------- panel headings
fig.canvas.draw()


def heading(ax, text, dy=0.014):
    fig.text(0.004, ax.get_position().y1 + dy, text, fontsize=9.2, color=GREY,
             ha="left", va="bottom")


heading(axA, "(a)  The field-standard design, and what a zero-text business "
             "mean recovers from it")
# 114 was the span of the OLD xlim(-52, 62); axA now runs -58 to 62.
heading(axB, "(b)  What survives the chronological protocol — note the axis: "
             "0 to 1.3 %, against (a)'s 120 %")
# The panel shows both horizons; entity_coverage_test for split A is 0.8341
# at h = 1 and 0.8784 at h = 3, so one number mis-states the lower block.
heading(axC, "(c)  One thing changed — the split rule: 83–88 % of test "
             "entities seen, then 0.0 %")

NOTE = (
    "Yelp Open Dataset, 8,474 businesses and 407,385 review events; loss is "
    "squared error on stars; inference is month-clustered Diebold-Mariano "
    "(HAC lag h−1 months, Harvey-Leybourne-Newbold), with business × month "
    "two-way clustering as robustness. Combiner weights are fitted on "
    "validation and frozen on test; business means use train and validation "
    "observations only. In (a) the outline bars are the committed recovery "
    "ratios applied to the committed apparent gain, not separately reported "
    "losses. In (c) fill = the month-clustered test at 0.05; test rows "
    "28,134 (A) and 7,183 (B) at h = 1 m, 38,399 and 9,843 at h = 3 m."
)
# Three fixes in that last sentence, all inside the same seven wrapped lines.
# "filled = month-clustered p < .05" broke across the wrap as "p <" / ".05",
# left the significance level without its leading zero, and set the operator in
# "h-1" and the cross product in "business x month" with a hyphen and a letter
# where the figure's own ticks use U+2212.  Verified: textwrap.fill(NOTE, 92)
# still returns seven lines and no line now ends on an operator.
# The note is the figure's basis statement, not one of its readings.  It drew in
# the same ink as the value labels above it, so nothing on the page told the
# reader which text was the argument and which was the apparatus.  Recessive ink
# plus a hairline says it, and neither changes geometry: no fontsize changes, the
# note block itself does not move, and the rule is placed inside whitespace the
# tight bounding box already encloses (between panel (c)'s axis label and the
# note's own top), spanning no wider than the note block itself.
_note = fig.text(0.004, 0.006, textwrap.fill(NOTE, 92), fontsize=8.9,
                 color=INK2, ha="left", va="bottom", linespacing=1.34)
fig.canvas.draw()
_nb = _note.get_window_extent(
    renderer=fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
_xl = axC.xaxis.get_label().get_window_extent(
    renderer=fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
_ry = 0.5 * (_nb.y1 + _xl.y0)          # centred in the gap that already exists
fig.add_artist(plt.Line2D([0.004, min(0.972, _nb.x1)], [_ry, _ry],
                          transform=fig.transFigure, color=RULE,
                          linewidth=0.5, zorder=0.5))

ds.finish(fig, "FP2_yelp_ladder", max_render_pt=595.0,
          note="appendix figure: the Yelp cascade, its detectable minimum, and "
               "the entity-disjoint split experiment")
