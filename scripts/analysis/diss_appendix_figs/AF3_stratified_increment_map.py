"""AF3 -- the stratum map of the apparent increment over a recalibrated HAR.

Nine text and fusion arms x three horizons, each cell's frozen test residuals
partitioned along three axes (volatility tercile, filing period, disclosure
form) and rescored without refitting anything.  567 drawn cells: the question is
whether the apparent gain concentrates anywhere -- in the high-volatility
stratum, in the 2022 shock, in a document type -- or is broad and shallow.

Sources (every plotted number is read from this file in this run):
  results/tables/m1_stratified.csv
      model, disclosure, horizon, axis, stratum, n, rel_impr_pct, dm_q_stat,
      dm_q_p, ci_lo, ci_hi, placebo_rel_impr_pct

Dissertation sentences this must not contradict:
  chapters/04_results.tex:86 and chapters/05_validation.tex:49  the long-form
      TF-IDF ridge increments +3.33/+3.48/+5.92 per cent (this file's pooled
      cells reproduce them exactly: the B-block arms are deterministic, so
      their pooled values do not move between the single-seed and the
      seed-ensemble basis)
  appendices/C_full_results.tex  the full test supports 7,951/7,933/7,902
      long-form and 25,109/25,001/24,732 event-driven

Basis this figure must declare on its face: single seed 2026; observation-level
DM, not the report's day-clustered primary; no Holm family; the reference is the
recalibrated HAR alone -- the FIRST rung of the ladder.  The combined channel is
omitted (its rows are the union of the two channels drawn here).

CPU only; no model is refitted, and no combiner weight is re-estimated inside a
stratum.
"""
import os
import sys

import numpy as np
import pandas as pd

ANALYSIS = "scripts/analysis"
sys.path.insert(0, ANALYSIS)
from supp_style import (BLUE, GREY, INK2, LIGHT, RULE, TAB,  # noqa: E402
                        VERM, VERM_TXT, apply_style, gate)
import diss_style as ds  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --------------------------------------------------------------- evidence
D = pd.read_csv(os.path.join(TAB, "m1_stratified.csv"))
PART = D[D.axis != "all"]
POOL = D[D.axis == "all"]

MODELS = ["B1_bow_ridge", "B2_tfidf_ridge", "B3_lm_linear", "B4_lm_features",
          "C1_bert_s1", "C2_finbert_s1", "C4_longformer", "C5_qwen3",
          "D2_gated_fusion"]
MLAB = {"B1_bow_ridge": "B1 BoW", "B2_tfidf_ridge": "B2 TF-IDF",
        "B3_lm_linear": "B3 LM dict.", "B4_lm_features": "B4 LM feat.",
        "C1_bert_s1": "C1 BERT", "C2_finbert_s1": "C2 FinBERT",
        "C4_longformer": "C4 Longformer", "C5_qwen3": "C5 Qwen3-emb",
        "D2_gated_fusion": "D2 gated fus."}
HS = (5, 10, 20)

COLS = {
    "long_form": [("all", "ALL", "pooled"),
                  ("vol_regime", "low", "low"),
                  ("vol_regime", "mid", "mid"),
                  ("vol_regime", "high", "high"),
                  ("period", "<=2022", "$\\leq$2022"),
                  ("period", "2023", "2023"),
                  ("period", "2024-2025", "2024-25"),
                  ("form", "10-K", "10-K"),
                  ("form", "10-Q", "10-Q")],
    "event_driven": [("all", "ALL", "pooled"),
                     ("vol_regime", "low", "low"),
                     ("vol_regime", "mid", "mid"),
                     ("vol_regime", "high", "high"),
                     ("period", "<=2022", "$\\leq$2022"),
                     ("period", "2023", "2023"),
                     ("period", "2024-2025", "2024-25"),
                     ("form", "2.02,9.01", "2.02, 9.01"),
                     ("form", "5.02", "5.02"),
                     ("form", "7.01,9.01", "7.01, 9.01"),
                     ("form", "8.01,9.01", "8.01, 9.01"),
                     ("form", "Other", "other")],
}
GROUPS = [("volatility", 1, 3), ("period", 4, 3), ("form", 7, None)]

DRAWN = pd.concat([
    D[(D.disclosure == disc)
      & np.array([(a, s) in {(c[0], c[1]) for c in COLS[disc]}
                  for a, s in zip(D.axis, D.stratum)])]
    for disc in ("long_form", "event_driven")])
DRAWN_PART = DRAWN[DRAWN.axis != "all"]

CLIP = 5.0
EDGE = "#B0B0B0"            # cell lattice: visible over the near-white fills

# --------------------------------------------------------------- the gate
b2lf = POOL[(POOL.model == "B2_tfidf_ridge")
            & (POOL.disclosure == "long_form")].sort_values("horizon")
gate(
    {
        "n_rows": 756,
        "n_partition_cells": 675,
        "n_pooled_cells": 81,
        "n_drawn": 567,
        # Panel (b)'s subtitle reconciles itself against panel (a) with these
        # two: of the 675 partition cells, 513 are drawn on the map (567 less
        # the 54 pooled cells, which are not partition cells) and the other
        # 162 are the omitted combined channel.
        "n_drawn_partition": 513,
        "n_offmap_partition": 162,
        "n_models": 9,
        "lf_support": [7951, 7933, 7902],
        "ed_support": [25109, 25001, 24732],
        "b2_long_form_pooled": [3.33, 3.48, 5.92],
        "placebo_abs_max": 0.78,
        "worst_stratum_pct": -43.14,
        "best_stratum_pct": 7.17,
        "cells_beyond_clip": 29,
        "drawn_cells_beyond_clip": 28,
        # The carets are drawn over every cell of the matrix, the pooled column
        # included, so the count printed in the key must be over DRAWN and not
        # over DRAWN_PART: four pooled cells also exceed the clip.
        "drawn_cells_beyond_clip_incl_pooled": 32,
    },
    {
        "n_rows": len(D),
        "n_partition_cells": len(PART),
        "n_pooled_cells": len(POOL),
        "n_drawn": len(DRAWN),
        "n_drawn_partition": len(DRAWN_PART),
        "n_offmap_partition": len(PART) - len(DRAWN_PART),
        "n_models": int(D.model.nunique()),
        "lf_support": sorted(
            {int(v) for v in POOL[POOL.disclosure == "long_form"].n},
            reverse=True),
        "ed_support": sorted(
            {int(v) for v in POOL[POOL.disclosure == "event_driven"].n},
            reverse=True),
        "b2_long_form_pooled": [round(float(v), 2)
                                for v in b2lf.rel_impr_pct],
        "placebo_abs_max": round(float(D.placebo_rel_impr_pct.abs().max()), 2),
        "worst_stratum_pct": round(float(PART.rel_impr_pct.min()), 2),
        "best_stratum_pct": round(float(PART.rel_impr_pct.max()), 2),
        "cells_beyond_clip": int((PART.rel_impr_pct.abs() > CLIP).sum()),
        "drawn_cells_beyond_clip": int(
            (DRAWN_PART.rel_impr_pct.abs() > CLIP).sum()),
        "drawn_cells_beyond_clip_incl_pooled": int(
            (DRAWN.rel_impr_pct.abs() > CLIP).sum()),
    },
)

# --------------------------------------------------------------- geometry
apply_style(9)
plt.rcParams["hatch.linewidth"] = 0.45   # the sign hatch is a hint, not a rule
W, H = 6.10, 7.38
fig = plt.figure(figsize=(W, H))
LINE = 0.152

LABW = 1.26                 # row-label gutter
GAP = 0.24                  # gap between the two channel blocks
RIGHT = 0.12
NCOL = len(COLS["long_form"]) + len(COLS["event_driven"])
CW = (W - LABW - GAP - RIGHT) / NCOL
M_BOT, M_HGT = 3.34, 3.34
RP = M_HGT / 27.0           # row pitch

CMAP = LinearSegmentedColormap.from_list(
    "diss_div", [VERM, "#F2C6AE", "#FFFFFF", "#AFD2E8", BLUE])
NORM = Normalize(vmin=-CLIP, vmax=CLIP)


def rect(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


rows = [(m, h) for m in MODELS for h in HS]
x0 = {"long_form": LABW,
      "event_driven": LABW + len(COLS["long_form"]) * CW + GAP}

axM = {}
for disc in ("long_form", "event_driven"):
    n = len(COLS[disc])
    ax = fig.add_axes(rect(x0[disc], M_BOT, n * CW, M_HGT))
    axM[disc] = ax
    ax.set_xlim(0, n)
    ax.set_ylim(27, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    sub = D[D.disclosure == disc]
    for j, (axis, stratum, _lab) in enumerate(COLS[disc]):
        cell = sub[(sub.axis == axis) & (sub.stratum == stratum)]
        for i, (m, h) in enumerate(rows):
            r = cell[(cell.model == m) & (cell.horizon == h)]
            if r.empty:
                continue
            v = float(r.rel_impr_pct.iloc[0])
            p = float(r.dm_q_p.iloc[0])
            # Sign is not left to hue alone: cells where text HURTS are
            # hatched, so the diverging map survives a greyscale print, in
            # which #C85800 and #0072B2 sit at almost the same luminance.
            # The non-negative edge used to be white, which vanished over the
            # near-white fills that cover most of the map: whole 3x4 group
            # blocks had no lattice, so a significance dot could not be
            # assigned to a cell.  EDGE is light enough to stay subordinate to
            # the saturated fills and dark enough to read over the pale ones.
            ax.add_patch(plt.Rectangle(
                (j, i), 1, 1, facecolor=CMAP(NORM(np.clip(v, -CLIP, CLIP))),
                edgecolor=GREY if v < 0 else EDGE, lw=0.35, zorder=2,
                hatch="////" if v < 0 else None))
            if p >= 0.05:
                ax.plot([j + 0.5], [i + 0.5], marker="o", ms=1.9, mfc=GREY,
                        mec=GREY, zorder=4)
            if abs(v) > CLIP:
                ax.plot([j + 0.85], [i + 0.18], marker="v" if v < 0 else "^",
                        ms=2.4, mfc="white", mec=GREY, mew=0.5, zorder=4)
    for i in range(3, 27, 3):
        ax.axhline(i, color=GREY, lw=0.45, zorder=5)
    for _g, start, _n in GROUPS:
        ax.axvline(start, color=GREY, lw=0.45, zorder=5)
    ax.add_patch(plt.Rectangle((0, 0), n, 27, facecolor="none",
                               edgecolor=GREY, lw=0.6, zorder=6))
    # column labels, rotated, under the matrix
    for j, (_a, _s, lab) in enumerate(COLS[disc]):
        ax.text(j + 0.5, 27.35, lab, ha="right", va="top", rotation=60,
                rotation_mode="anchor", fontsize=8.6, color=GREY)

# row labels
for i, (m, h) in enumerate(rows):
    y = M_BOT + M_HGT - (i + 0.5) * RP
    fig.text((LABW - 0.06) / W, y / H, f"{MLAB[m]}  $h$={h}", ha="right",
             va="center", fontsize=8.6, color=GREY)

# block titles and axis-group headers
for disc, title in (("long_form", "long-form 10-K / 10-Q"),
                    ("event_driven", "event-driven 8-K")):
    n = len(COLS[disc])
    fig.text((x0[disc] + n * CW / 2) / W, (M_BOT + M_HGT + 0.50) / H, title,
             ha="center", va="center", fontsize=9, color=GREY, weight="bold")
    for gname, start, span in GROUPS:
        span = span if span is not None else n - start
        xc = x0[disc] + (start + span / 2) * CW
        fig.text(xc / W, (M_BOT + M_HGT + 0.31) / H, gname, ha="center",
                 va="center", fontsize=8.6, color=GREY, rotation=0)
        fig.lines.append(plt.Line2D(
            [(x0[disc] + start * CW + 0.015) / W,
             (x0[disc] + (start + span) * CW - 0.015) / W],
            [(M_BOT + M_HGT + 0.20) / H] * 2, color=GREY, lw=0.5,
            transform=fig.transFigure))

fig.text(0.010, (M_BOT + M_HGT + 0.70) / H,
         "(a)  567 stratum cells: the apparent increment over the recalibrated "
         "HAR, by stratum",
         ha="left", va="center", fontsize=9.5, color=GREY, weight="bold")

# ---------------------------------------------------- panel (b) the placebo
P_L, P_BOT, P_HGT = 0.62, 1.56, 0.78
P_W = 2.26
axP = fig.add_axes(rect(P_L, P_BOT, P_W, P_HGT))
axP.axhline(0, color=GREY, lw=0.6, zorder=3)
axP.axvline(0, color=GREY, lw=0.6, zorder=3)
axP.scatter(PART.rel_impr_pct, PART.placebo_rel_impr_pct, s=4.2,
            facecolor=BLUE, edgecolor="none", alpha=0.45, zorder=4)
axP.set_xscale("symlog", linthresh=2.0, linscale=0.9)
axP.set_xlim(-55, 12)
axP.set_ylim(-1.05, 1.05)
axP.set_xticks([-40, -10, -2, 0, 2, 10])
axP.set_xticklabels(["-40", "-10", "-2", "0", "+2", "+10"])
axP.set_yticks([-1, 0, 1])
axP.tick_params(axis="both", pad=1.5, labelsize=8.6)
# The x scale is symlog with a linear window of +/-2, so the near-zero crowd
# occupies half the axis width where a linear axis would give it eight per
# cent.  The axis says so on its face: unannotated, the panel exaggerates the
# spread of the real increments, and it does so in the flattering direction.
# The longer string is 2.05 in wide, still inside the 2.26 in axes, so the
# tight bounding box does not move.
axP.set_xlabel("real increment (%, symlog beyond $\\pm$2)", fontsize=8.6,
               labelpad=2)
axP.set_ylabel("placebo (%)", fontsize=8.6, labelpad=2)
axP.grid(True, color=LIGHT, lw=0.5, zorder=0)
axP.set_axisbelow(True)

fig.text(0.010, (P_BOT + P_HGT + 0.42) / H,
         "(b)  Shuffling the text kills every stratum",
         ha="left", va="center", fontsize=9.5, color=GREY, weight="bold")
# The denominator statement is basis, not argument: INK2 so a reader can tell
# at a glance which line is the panel's claim and which is its scope.
# It now reconciles the two counts on the figure's face instead of restating
# the two axis labels directly beneath it.  675 partition cells = the 513 that
# panel (a) draws (its 567 less the 54 pooled cells, which are not partition
# cells) + 162 from the combined channel, which panel (a) omits.  Both counts
# are asserted by the gate above.  The new string is 2.770 in against 2.771,
# and ends 0.5 in short of the right-hand key column, so nothing moves.
fig.text(0.010, (P_BOT + P_HGT + 0.21) / H,
         "{} partition cells: {} in (a), {} combined channel".format(
             len(PART), len(DRAWN_PART), len(PART) - len(DRAWN_PART)),
         ha="left", va="center", fontsize=8.6, color=INK2)

# ------------------------------------------------- colour key, at the right
KX = 3.34
cb_w, cb_h = 1.55, 0.115
axcb = fig.add_axes(rect(KX, P_BOT + 0.54, cb_w, cb_h))
grad = np.linspace(-CLIP, CLIP, 256)[None, :]
axcb.imshow(grad, aspect="auto", cmap=CMAP, norm=NORM,
            extent=[-CLIP, CLIP, 0, 1])
axcb.set_yticks([])
axcb.set_xticks([-5, -2.5, 0, 2.5, 5])
axcb.set_xticklabels(["-5", "", "0", "", "+5"])
axcb.tick_params(axis="x", pad=1.5, labelsize=8.6, length=2.5)
for sp in axcb.spines.values():
    sp.set_edgecolor(GREY)
    sp.set_linewidth(0.6)
# Key caption: apparatus (what the colour means, where it is clipped), so INK2.
# The two count lines below the legend are the panel's finding, not its basis,
# and stay at primary ink.
fig.text(KX / W, (P_BOT + P_HGT + 0.21) / H,
         "panel (a) colour: QLIKE improvement (%),",
         ha="left", va="center", fontsize=8.6, color=INK2)
fig.text(KX / W, (P_BOT + P_HGT + 0.055) / H,
         "clipped at $\\pm$5; carets mark the {} cells beyond it".format(
             int((DRAWN.rel_impr_pct.abs() > CLIP).sum())),
         ha="left", va="center", fontsize=8.6, color=INK2)

nonsig = int((DRAWN.dm_q_p >= 0.05).sum())
npos = int((DRAWN.rel_impr_pct > 0).sum())
fig.legend(handles=[
    Line2D([], [], ls="none", marker="o", ms=2.6, mfc=GREY, mec=GREY,
           label="grey dot: DM $p\\geq$.05"),
    # The hatch is a sign bit at one weight, so it falls on a -0.1 per cent
    # cell as heavily as on a clipped -5 one, and several hatched cells also
    # carry the not-significant dot.  The key says what the mark is, not what
    # it proves.  1.44 in against 0.98, ending 1.4 in short of the block that
    # sets this column's right edge.
    Patch(facecolor="white", edgecolor=GREY, lw=0.35, hatch="////",
          label="hatched: negative, any size")],
    ncol=1, loc="upper left",
    bbox_to_anchor=(KX / W, (P_BOT + 0.42) / H),
    handlelength=1.0, handletextpad=0.5, borderpad=0.0, labelspacing=0.35,
    fontsize=8.6)
fig.text(KX / W, (P_BOT - 0.07) / H,
         f"{nonsig} of {len(DRAWN)} drawn cells are not",
         ha="left", va="center", fontsize=8.6, color=GREY)
fig.text(KX / W, (P_BOT - 0.225) / H,
         f"significant at 5%; {npos} are positive.",
         ha="left", va="center", fontsize=8.6, color=GREY)

# --------------------------------------------------------------- footnote
foot = [
    "Basis. Single seed 2026; observation-level DM at HAC lag h-1 observations, "
    "not the report's day-clustered",
    # Matplotlib does not translate the TeX "---" convention, so it printed
    # three literal hyphens on the published page.  U+2014 direct; Helvetica
    # carries the glyph, and this is not the widest line of the block (5.56 in
    # against the 5.64 in of the supports line), so FOOT_R and the tight
    # bounding box are untouched.
    "primary, and no Holm family — an off-basis map. The reference is the "
    "recalibrated HAR alone, the FIRST",
    "rung of the ladder: a warm cell is an APPARENT increment of the kind the "
    "firm-identity and maximal-pool",
    "rungs go on to dissolve, and nothing here is attributable value. Weights "
    "are fitted once on validation",
    "and frozen; strata partition the test residuals and nothing is refitted "
    "inside a stratum. Supports",
    "7,951/7,933/7,902 long-form and 25,109/25,001/24,732 event-driven rows at "
    "h = 5/10/20. The event-driven",
    "form axis is the four most frequent literal item-code combinations plus a "
    "residual class.",
]
# The basis block used to start 2.5 pt under panel (b)'s x-axis label, in the
# same ink as the data labels, so nothing told the reader where the figure
# stopped and its basis statement began.  It drops by SEP and gains a hairline
# above it; the wording, the line pitch (LINE) and the 8.6 pt size are all
# untouched.  SEP is spent, not free -- it lengthens the tight bounding box --
# but width binds this figure's inclusion scale, so the height it costs does not
# shrink a single glyph.  FOOT_R covers the longest line, measured at 5.71 in.
SEP = 0.13
FOOT_TOP = 1.14 - SEP
FOOT_R = 0.940
fig.lines.append(plt.Line2D(
    [0.010, FOOT_R], [(FOOT_TOP + 0.12) / H] * 2, transform=fig.transFigure,
    color=RULE, lw=0.5, zorder=0.5))
for k, t in enumerate(foot):
    fig.text(0.010, (FOOT_TOP - LINE * k) / H, t, ha="left", va="center",
             fontsize=8.6, color=INK2)

ds.finish(fig, "AF3_stratified_increment_map",
          note=f"{len(DRAWN)} drawn cells, {nonsig} not significant")
