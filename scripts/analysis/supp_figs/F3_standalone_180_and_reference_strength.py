"""F3 -- Standalone: 0 of 180 under three loss conventions, and the reference that
manufactures value.

Main-text sentences substantiated (06_results.tex, first paragraph, and
07_ablations.tex, penultimate paragraph):
  "0 of 180 standalone squared-error comparisons favour the challenger (raw
   p<.05 and Holm); 155 are significantly worse"
  "Variance-unit QLIKE leaves text and fusion 0 of 153 better."
  "models losing to HAR by DM +12.2--18.5 merely tie naive persistence: weak
   baselines manufacture apparent value."

Committed evidence read at run time (no number is hardcoded outside gate()):
  results/tables/variance_unit_standalone180.csv
      disclosure, horizon, challenger, dm_se_clu, dm_qlike_vol_clu,
      dm_qlike_var_clu, holm_se_clu, holm_qlike_vol_clu, holm_qlike_var_clu,
      better_*_holm, worse_*_holm. 180 rows = 20 challengers x 3 disclosures
      x 3 horizons; 3 of the challengers are price arms (27 rows) and 17 are
      text or fusion arms (153 rows).
  results/tables/dm_weak_vs_strong_baseline.md   (transcribed, not recomputed)
  results/tables/dm_stratified.md                (results/tables copy ONLY --
      the release/aggregate_results copy is a pre-retransformation vintage
      whose long-form D2 row flips two verdicts, so it is never read here)

Adversarial repair folded in
  * panel (c) is restricted to the models each channel actually carries --
    long-form has three, event-driven two (no C4_longformer) -- and the
    asymmetry is labelled on the artefact.
  * the per-row "significantly worse" counts (155 / 161 / 153) are printed on
    the three strip labels, so the volatility-unit convention is visibly the
    harshest on the challengers rather than only the variance-unit one being
    visibly the most permissive.
  * the Holm UNIT is now printed in panel (a)'s header. Reconstructing Holm
    within each (disclosure, horizon) panel of 20 reproduces holm_se_clu,
    holm_qlike_vol_clu and holm_qlike_var_clu to max abs error 1.2e-15, 6.0e-16
    and 7.0e-16; a 180-wide reconstruction does not (max abs error 0.755, 0.857,
    0.815) and would give 144 / 147 / 150 worse and 6 variance-unit wins instead
    of 155 / 161 / 153 and 7. The same within-20 adjustment carries the frozen
    main text's 155 via p_holm_clust in dm_pairwise_clustered.csv, so the paper's
    number is unaffected; holm_families.md declares family F1 as size 180, which
    is the source of the confusion and is itself inconsistent with the data.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from supp_style import (AGG, BLUE, GREEN, GREY, INK, INK2, LIGHT, PURPLE,  # noqa: E402,F401
                        REPO, RULE, SKY, TAB, VERM, VERM_TXT, YELLOW, annot,
                        apply_style, finish, gate)

# ------------------------------------------------------------------ evidence
sa = pd.read_csv(os.path.join(TAB, "variance_unit_standalone180.csv"))
PRICE = ["A3_garch", "A4_egarch", "A5_arima"]
sa["is_price"] = sa.challenger.isin(PRICE)

LOSSES = [("dm_se_clu", "holm_se_clu", "better_se_holm", "worse_se_holm",
           "squared error"),
          ("dm_qlike_vol_clu", "holm_qlike_vol_clu", "better_qlike_vol_holm",
           "worse_qlike_vol_holm", "volatility-unit QLIKE"),
          ("dm_qlike_var_clu", "holm_qlike_var_clu", "better_qlike_var_holm",
           "worse_qlike_var_holm", "variance-unit QLIKE")]

winners = sa[sa.better_qlike_var_holm]
win_counts = winners.challenger.value_counts().to_dict()

# --- weak vs strong reference (transcribed verbatim, never recomputed)
weak = []
for line in open(os.path.join(TAB, "dm_weak_vs_strong_baseline.md")):
    if not line.startswith("| C"):
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    model, ref = cells[0], cells[1]
    for h, cell in zip((5, 10, 20), cells[2:5]):
        m = re.match(r"([+-][\d.]+)(\*| ns)", cell)
        weak.append({"model": model, "ref": ref, "horizon": h,
                     "dm": float(m.group(1)), "sig": m.group(2) == "*"})
weak = pd.DataFrame(weak)
naive = weak[weak.ref.str.startswith("weak")]
strong = weak[weak.ref.str.startswith("strong")]

# --- stratified DM by test sub-period
strat, facet = [], None
for line in open(os.path.join(TAB, "dm_stratified.md")):
    if line.startswith("## "):
        facet = line[3:].strip()
    elif line.startswith("| C") or line.startswith("| D"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for h, cell in zip((5, 10, 20), cells[2:5]):
            m = re.match(r"([+-][\d.]+)(\*| ns)", cell)
            strat.append({"facet": facet, "model": cells[0], "period": cells[1],
                          "horizon": h, "dm": float(m.group(1)),
                          "sig": m.group(2) == "*"})
strat = pd.DataFrame(strat)

# --------------------------------------------------------------------- gates
gate(
    {"n_comparisons": 180,
     "n_price_rows": 27,
     "n_text_fusion_rows": 153,
     "better_holm_se_vol_var": (0, 0, 7),
     "worse_holm_se_vol_var": (155, 161, 153),
     "variance_unit_winners": {"A3_garch": 6, "A4_egarch": 1},
     "n_text_fusion_variance_unit_winners": 0,
     "naive_dm_range": (-1.44, 1.49),
     "har_dm_range": (12.22, 18.45),
     "n_weak_strong_cells": 6,
     "strat_cells_long_form": 27,
     "strat_cells_event_driven": 18,
     "strat_n_negative_cells": 0,
     "strat_models_long_form": ["C2_finbert_s1", "C4_longformer", "D2_gated_fusion"],
     "strat_models_event_driven": ["C2_finbert_s1", "D2_gated_fusion"]},
    {"n_comparisons": int(len(sa)),
     "n_price_rows": int(sa.is_price.sum()),
     "n_text_fusion_rows": int((~sa.is_price).sum()),
     "better_holm_se_vol_var": tuple(int(sa[c].sum()) for _, _, c, _, _ in LOSSES),
     "worse_holm_se_vol_var": tuple(int(sa[c].sum()) for _, _, _, c, _ in LOSSES),
     "variance_unit_winners": win_counts,
     "n_text_fusion_variance_unit_winners": int((~winners.is_price).sum()),
     "naive_dm_range": (float(naive.dm.min()), float(naive.dm.max())),
     "har_dm_range": (float(strong.dm.min()), float(strong.dm.max())),
     "n_weak_strong_cells": int(len(strong)),
     "strat_cells_long_form": int((strat.facet == "long_form").sum()),
     "strat_cells_event_driven": int((strat.facet == "event_driven").sum()),
     "strat_n_negative_cells": int((strat.dm <= 0).sum()),
     "strat_models_long_form": sorted(strat[strat.facet == "long_form"]
                                      .model.unique().tolist()),
     "strat_models_event_driven": sorted(strat[strat.facet == "event_driven"]
                                         .model.unique().tolist())},
)

# ------------------------------------------------------------------ plotting
#
# Layout note.  This figure is written UNTIGHT (finish(..., tight=False)), so the
# PDF page is the canvas rather than the content's bounding box: repositioning
# type cannot change the page box, the inclusion scale or the printed point size,
# but anything pushed past an edge is silently CLIPPED.  The committed version
# was losing the words "per row." off the right edge, because panel (a)'s title
# line measured 1.072 of the canvas width.  Every heading block below is
# therefore wrapped to reach at most 0.95 of the width, and the three panels give
# up dead white -- panel (a) and (b) the empty strip above their top row, panel
# (c) 0.85 pt of row pitch -- to pay for the third heading line that the rewrap
# needs.  Row pitch in (a) and (b) is unchanged; no type changes size.
apply_style()
fig = plt.figure(figsize=(6.4, 8.9))
ax_a = fig.add_axes([0.300, 0.8190, 0.685, 0.1245])
ax_b = fig.add_axes([0.300, 0.5060, 0.685, 0.1411])
ax_c = fig.add_axes([0.300, 0.0340, 0.520, 0.3430])
ax_cb = fig.add_axes([0.858, 0.0340, 0.022, 0.3430])

# ---------------------------------------------------------------- panel (a)
rng = np.random.default_rng(0)          # jitter only; carries no information
XLO_A, XHI_A = -6.2, 13.6
ax_a.axvspan(XLO_A, 0.0, color=LIGHT, alpha=0.45, lw=0, zorder=0)
ax_a.axvline(0.0, color=GREY, lw=1.0, zorder=2)

strips = []
for i, (dm_col, holm_col, better_col, worse_col, name) in enumerate(LOSSES):
    y0 = len(LOSSES) - 1 - i
    # 153 opaque text/fusion discs at one radius weld into a continuous orange
    # field from about +4 rightward, so the SHAPE of the mass is unreadable, and
    # the 27 price arms drawn underneath survive only as crescents -- which is
    # exactly the contrast the key sets up.  Both are fixed without moving a
    # single mark: the dense series takes alpha, so each disc gets a darker rim
    # where its same-colour edge stroke composites over its own face and
    # overlaps DARKEN instead of fusing, and the sparse price series keeps full
    # opacity and is lifted above it.  Same positions, same radius, same hues,
    # same canvas.  The draw order is deliberately unchanged so the shared
    # jitter generator is consumed in the same order and every point stays
    # exactly where it was.
    #
    # 0.80 is a floor, not a taste call.  This module's palette rule is 3:1 for
    # a mark on white -- it is why YELLOW was darkened from #E69F00 -- and VERM
    # at 4.33:1 buys only so much dilution: an ISOLATED disc measures 3.20:1 at
    # alpha 0.80 but 2.95:1 at 0.75 and 2.26:1 at 0.62, i.e. the softer values
    # that separate the blob slightly better would put every lone challenger in
    # the left of the variance-unit row under the floor.  Two coincident discs
    # already composite to 0.96 of full VERM, so the crowded right-hand mass --
    # the adverse result this panel exists to show -- loses no weight at all.
    # The legend keys stay opaque: a key carries hue identity, not density, and
    # a 3.6 pt dot at 0.80 alpha would be the weakest ink in the figure.
    for is_price, col in ((True, GREY), (False, VERM)):
        sub = sa[sa.is_price == is_price]
        jit = rng.uniform(-0.26, 0.26, len(sub))
        sig = sub[holm_col] < 0.05
        zo = 3.6 if is_price else 3.0
        al = None if is_price else 0.80
        ax_a.scatter(sub.loc[sig, dm_col], y0 + jit[sig.to_numpy()], s=9,
                     facecolors=col, edgecolors=col, linewidths=0.6,
                     alpha=al, zorder=zo)
        ax_a.scatter(sub.loc[~sig, dm_col], y0 + jit[(~sig).to_numpy()], s=9,
                     facecolors="white", edgecolors=col, linewidths=0.7,
                     zorder=zo)
    strips.append((y0, name, f"{int(sa[better_col].sum())} of 180 better, "
                             f"{int(sa[worse_col].sum())} worse"))

# The three strips are banded by a hairline so each label pairs with its row.
for edge in range(1, len(LOSSES)):
    ax_a.axhline(edge - 0.5, color=RULE, lw=0.5, zorder=1.5)

ax_a.set_yticks([])
ax_a.set_ylim(-0.45, 2.55)
ax_a.set_xlim(XLO_A, XHI_A)
ax_a.tick_params(axis="y", length=0)

# The row label carried two kinds of thing in one two-line tick label: the loss
# convention (identity) and its counts (the argument).  Same words, same size,
# same right edge -- but now two runs on two baselines, identity in bold.
lab_a = ax_a.get_yaxis_transform()
for y0, name, counts in strips:
    ax_a.text(-0.011, y0 + 0.05, name, transform=lab_a, fontsize=9, color=INK,
              family="Arial", fontweight="bold", ha="right", va="bottom")
    ax_a.text(-0.011, y0 - 0.05, counts, transform=lab_a, fontsize=9,
              color=INK, ha="right", va="top")

annot(ax_a, XLO_A + 0.25, 2.25, "challenger better", size=9, color=INK2,
      ha="left", va="center", style="italic")

leg_a = [plt.Line2D([], [], color=VERM, marker="o", ls="none", ms=3.6,
                    label="17 text / fusion arms (153)"),
         plt.Line2D([], [], color=GREY, marker="o", ls="none", ms=3.6,
                    label="3 price arms (27)"),
         plt.Line2D([], [], color=GREY, marker="o", ls="none", ms=3.6, mfc="white",
                    mew=0.7, label="open = not Holm-significant")]
lg_a = fig.legend(handles=leg_a, loc="upper center", bbox_to_anchor=(0.545, 0.797),
                  bbox_transform=fig.transFigure, ncol=3, fontsize=9,
                  handletextpad=0.35, columnspacing=1.4, borderpad=0.2)
for t in lg_a.get_texts():          # a key is apparatus, not the argument
    t.set_color(INK2)

# ---------------------------------------------------------------- panel (b)
cells = [(m, h) for m in ("C4_longformer", "C2_finbert_s1") for h in (5, 10, 20)]
# The connector was LIGHT (#D9D9D9) and the +-1.96 band is the SAME LIGHT at
# alpha 0.45, i.e. #EDEDED, so the stretch of connector that crosses the band --
# which is every row's first two data units, the part that carries the "merely
# ties the weak reference" reading -- had only 20 levels of separation against
# 38 in the white.  One shade darker restores it (47 levels inside the band)
# while staying subordinate to the blue dot and the vermillion diamond.  The
# band itself is left alone: its grey is shared with panel (a)'s shaded
# half-plane and repainting it would desynchronise the two.
CONN = "#BEBEBE"
for j, (m, h) in enumerate(cells):
    y = len(cells) - 1 - j
    a = float(naive[(naive.model == m) & (naive.horizon == h)].dm.iloc[0])
    b = float(strong[(strong.model == m) & (strong.horizon == h)].dm.iloc[0])
    ax_b.plot([a, b], [y, y], color=CONN, lw=1.6, zorder=2, solid_capstyle="round")
    ax_b.plot(a, y, marker="o", ms=5.0, color=SKY, mec="white", mew=0.6, zorder=3)
    ax_b.plot(b, y, marker="D", ms=5.0, color=VERM, mec="white", mew=0.6, zorder=3)

ax_b.axvspan(-1.96, 1.96, color=LIGHT, alpha=0.45, lw=0, zorder=0)
ax_b.axvline(0.0, color=GREY, lw=1.0, zorder=1)
ax_b.axhline(2.5, color=RULE, lw=0.5, zorder=1.5)   # the two models, banded
ax_b.set_yticks(range(len(cells) - 1, -1, -1))
ax_b.set_yticklabels([f"{m}   $h$ = {h}" for m, h in cells])
ax_b.set_ylim(-0.5, 5.85)
ax_b.set_xlim(-4.0, 21.0)
ax_b.tick_params(axis="y", length=0)
# The label names the shaded band but used to sit wholly OUTSIDE it, starting
# 0.19 data units past the band's right edge with no leader or tick, floating
# over live plotting space (+2.2 to +4.2) instead of over the region it names.
# It is now right-aligned to the band's inner right edge, which pulls ink
# leftward and moves no extent.
#
# The string had to lose its mathtext wrapper to make that fit, and the numbers
# are why: the band's right half is 1.96 data units wide, mathtext "$\pm$1.96"
# measures 2.033 units at 9 pt (mathtext pads the operator), so right-aligned at
# +1.90 its "+-" glyph would have straddled the zero rule.  The literal U+00B1
# measures 1.773, so the same anchor puts the string at +0.127 to +1.900: 4.5 px
# clear of the zero rule, 2 px inside the band's edge, and 4 px above the
# topmost connector.  Helvetica.ttc, which this style resolves to, carries
# U+00B1, and the glyph is outlined by ghostscript on write like every other.
# halo=False because the halo now hurts: annot()'s white stroke exists to keep a
# callout legible over DATA, and there is none here -- only the band's own fill,
# which the stroke punched a white box out of, and the zero rule, which it
# nibbled.  INK2 on the band measures 5.9:1 unaided.
annot(ax_b, 1.90, 5.50, "±1.96", size=9, color=INK2, ha="right",
      va="center", halo=False)

leg_b = [plt.Line2D([], [], color=SKY, marker="o", ls="none", ms=5.0, mec="white",
                    mew=0.6,
                    label=f"vs naive RV persistence ({naive.dm.min():+.2f} to "
                          f"{naive.dm.max():+.2f})"),
         plt.Line2D([], [], color=VERM, marker="D", ls="none", ms=5.0, mec="white",
                    mew=0.6,
                    label=f"vs HAR-RV ({strong.dm.min():+.2f} to "
                          f"{strong.dm.max():+.2f})")]
lg_b = fig.legend(handles=leg_b, loc="upper center", bbox_to_anchor=(0.60, 0.482),
                  bbox_transform=fig.transFigure, ncol=2, fontsize=9,
                  handletextpad=0.35, columnspacing=1.6, borderpad=0.2)
for t in lg_b.get_texts():
    t.set_color(INK2)

# ---------------------------------------------------------------- panel (c)
ORDER = [("long_form", "C2_finbert_s1"), ("long_form", "C4_longformer"),
         ("long_form", "D2_gated_fusion"), ("event_driven", "C2_finbert_s1"),
         ("event_driven", "D2_gated_fusion")]
PERIODS = ["2022", "2023", "24-25"]
rows = [(f, m, p) for f, m in ORDER for p in PERIODS]
grid = np.array([[float(strat[(strat.facet == f) & (strat.model == m) &
                              (strat.period == p) & (strat.horizon == h)].dm.iloc[0])
                  for h in (5, 10, 20)] for f, m, p in rows])
sigm = np.array([[bool(strat[(strat.facet == f) & (strat.model == m) &
                             (strat.period == p) & (strat.horizon == h)].sig.iloc[0])
                  for h in (5, 10, 20)] for f, m, p in rows])

cmap = LinearSegmentedColormap.from_list("verm", ["#FFFFFF", "#F6D6BF", VERM])
im = ax_c.imshow(grid, cmap=cmap, vmin=0.0, vmax=float(grid.max()),
                 aspect="auto", origin="upper")
for r in range(grid.shape[0]):
    for c in range(grid.shape[1]):
        v = grid[r, c]
        ax_c.text(c, r, f"{v:+.2f}" + ("*" if sigm[r, c] else " ns"), fontsize=9,
                  ha="center", va="center",
                  # White ink only on the three darkest fills.  Below about
                  # 0.88 of the maximum the fill is still a light-to-mid orange
                  # on which white measures under 3:1 while INK measures 4.2:1
                  # or better, so the 2022 cells the panel's claim rests on
                  # (+15.70, +16.98, +18.31, +19.39) take the dark ink their
                  # near-identical neighbours already use.  Colour only: same
                  # strings, same positions, canvas unchanged.
                  color="white" if v > 0.88 * grid.max() else INK)

ax_c.set_xticks(range(3))
ax_c.set_xticklabels(["$h$ = 5", "$h$ = 10", "$h$ = 20"])
ax_c.set_yticks(range(len(rows)))
ax_c.set_yticklabels([f"{m}  {p}" for _, m, p in rows])
ax_c.tick_params(axis="both", length=0)
n_lf = sum(1 for f, _, _ in rows if f == "long_form")
# Three levels of separator, so a reader can see the grid's nesting without
# reading the labels: a period row, a model's three periods, a disclosure
# channel.  Wider white for the model groups, the one grey rule for the channel.
for k in range(1, len(rows)):
    ax_c.axhline(k - 0.5, color="white",
                 lw=2.2 if k % len(PERIODS) == 0 and k != n_lf else 0.8)
for k in range(1, 3):
    ax_c.axvline(k - 0.5, color="white", lw=0.8)
ax_c.axhline(n_lf - 0.5, color=GREY, lw=1.2)
C_TOP, C_BOT = 0.377, 0.034
# These two run rotated alongside the rows they bracket, so their LENGTH has to
# fit the block's height.  "event-driven (2 models)" set 260 px of type against a
# 220 px block of six rows: correctly centred, but overhanging the heavy channel
# rule at the top by half a row and dropping below the bottom spine into the
# h-tick strip at the other end, so it read as if it also captioned the last
# long-form row.  The word "models" is what does not fit, and it is the one word
# here that is already carried elsewhere -- the rows themselves name three
# distinct models above the rule and two below it, and heading (c) states the
# asymmetry outright ("C4_longformer has no event-driven row").  Dropping it
# takes the longer string to about 181 px, inside its block at both ends.
# Shortening only: no size, colour or position change, and the labels stay
# centred on their own blocks.
fig.text(0.052, C_TOP - (n_lf / 2) / len(rows) * (C_TOP - C_BOT),
         "long-form (3)", fontsize=9, color=INK, rotation=90,
         ha="center", va="center")
fig.text(0.052, C_TOP - ((n_lf + len(rows)) / 2) / len(rows) * (C_TOP - C_BOT),
         "event-driven (2)", fontsize=9, color=INK, rotation=90,
         ha="center", va="center")

cb = fig.colorbar(im, cax=ax_cb)
cb.outline.set_linewidth(0.6)
cb.outline.set_edgecolor(GREY)
cb.set_label("DM vs A2_har_rv, squared error\n(positive = challenger worse)",
             fontsize=9)
cb.ax.tick_params(labelsize=9, width=0.6)

# ------------------------------------------------------------------- headings
#
# Every word of these blocks is the committed wording; what changes is that the
# marker, the claim and the basis statement are no longer one undifferentiated
# grey paragraph.  Marker in bold, claim in INK, basis in INK2, one role per
# line.  Nothing is resized: the three levels are colour and weight only.  In
# panel (c) the two apparatus sentences, which the original interleaved with the
# finding, are collected after it -- reordered, not rewritten.
LEAD, LINE = 0.012, 0.0185


def heading(y, letter, claim, basis):
    """A panel heading: bold marker and claim line share one baseline.

    `y` is the FIRST BASELINE, not the top of the block, and every run is set
    va="baseline" -- the marker is Arial Bold and the claim Helvetica, and the
    two faces do not share an ascent, so aligning their tops would leave the
    marker sitting half a point off the line it belongs to.
    """
    mark = fig.text(LEAD, y, f"({letter})", fontsize=9, color=INK,
                    family="Arial", fontweight="bold", ha="left", va="baseline")
    fig.canvas.draw()
    x1 = mark.get_window_extent(fig.canvas.get_renderer()).x1 / fig.bbox.width
    for i, line in enumerate(claim):
        fig.text(x1 + 0.011 if i == 0 else LEAD, y - i * LINE, line, fontsize=9,
                 color=INK, ha="left", va="baseline")
    for j, line in enumerate(basis):
        fig.text(LEAD, y - (len(claim) + j) * LINE, line, fontsize=9,
                 color=INK2, ha="left", va="baseline")


heading(0.9852, "a",
        ["all 180 standalone comparisons against HAR-RV as a standalone "
         "forecaster;",
         "three losses on one axis, counts per row."],
        ["x = day-clustered DM, negative = challenger better (shaded). Holm is "
         "applied within each 20-comparison panel."])

# The figure's own finding about panel (a): the argument, not the apparatus, so
# it keeps the vermillion of the marks it is about rather than being demoted to
# INK2, and a hairline fences the whole (a) block off from (b).
#
# The rule used to sit ABOVE this note, at 0.7675.  There is exactly one rule in
# the figure and it runs the full width, so at that height it read as "panel (a)
# ends here" and threw the note onto (b)'s side of the divider -- 7 px below the
# rule against 11 px above the "(b)" marker -- even though every clause of it is
# about (a) (the variance-unit row, the 0-of-153, the abstract's 0-of-180).
# Moving it below the note's last line puts the boundary where the subject
# actually changes.  Pure repositioning: same ink, same width, same span, and
# the figure is written untight, so the page box is the canvas either way.
# 0.7077 is the measured midpoint of the free band on the dissertation canvas:
# the note's rendered box bottoms at 0.7116 and the "(b)" marker tops at 0.7037,
# leaving 4.57 pt, so the rule clears type by about 2.3 pt on each side -- the
# same clearance it had at 0.7675, where the gaps were 6 px and 7 px.
fig.lines.append(plt.Line2D([LEAD, 0.945], [0.7077, 0.7077],
                            transform=fig.transFigure, color=RULE,
                            linewidth=0.5, zorder=0.5))
fig.text(LEAD, 0.763,
         "The only 7 Holm-significant wins in the figure are in the variance-unit "
         "row and every one is a GARCH-family\n"
         f"price arm (A3_garch {win_counts['A3_garch']}, A4_egarch "
         f"{win_counts['A4_egarch']}); the 17 text and fusion arms are 0 of 153 "
         "better under that\nconvention, the scope of the main text's 0-of-153. "
         "The abstract's 0 of 180 is the top two rows.",
         fontsize=9, color=VERM_TXT, ha="left", va="top", linespacing=1.4)

heading(0.6921, "b",
        ["a weak reference manufactures apparent value."],
        ["x = single-seed observation-level DM, positive = text worse than that "
         "reference; roughly three times",
         "the day-clustered primary of panel (a) and not comparable with it."])

heading(0.4372, "c",
        ["is the null a 2022 volatility-shock artefact? Every cell is positive;",
         "magnitudes fall from 2022 to 24-25 without crossing zero."],
        ["Test sub-period strata, single-seed observation-level DM. "
         "C4_longformer has no event-driven row."])

finish(fig, "F3_standalone_180_and_reference_strength", tight=False)
