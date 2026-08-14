"""Appendix figure — what moves when the combiner's freeze point moves.

The study's primary recipe fits the log-space combination weights once, on the
2020-21 validation block, and freezes them for the whole 2022-25 test span.  A
deployable forecaster cannot do that: it must refit the weights before each
quarter on strictly earlier filings.  This figure puts the two side by side on
identical text forecasts -- only the weight-estimation scheme differs.

Panel (a) is the paired comparison, one point per cell.  Panel (b) is the flow
of verdicts across the 69 primary cells.  Panel (c) separates the two reasons
the headline count falls: widening the Holm family from 69 to 75 cells (38 ->
36, an accounting change) and refitting the weights (36 -> 6, a real one).

Source: results/tables/deployable_combiner.csv
Out:    writing/dissertation/figures/stability_freeze_point_deployability.pdf
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path

sys.path.insert(0, "scripts/analysis")
import supp_style
from supp_style import (
    GREEN,
    GREY,
    INK2,
    LIGHT,
    PURPLE,
    REPO,
    RULE,
    TAB,
    VERM,
    VERM_TXT,
    annot,
    apply_style,
    finish,
    gate,
)

supp_style.OUTDIR = os.path.join(REPO, "writing", "dissertation", "figures")

SHORT = {
    "B1_bow_ridge": "BoW", "B2_tfidf_ridge": "TF-IDF", "B3_lm_linear": "LM-dict",
    "B4_lm_features": "LM-feat", "C1_bert_s1": "BERT", "C2_finbert_s1": "FinBERT-1",
    "C2_finbert_s2": "FinBERT-2", "C2_finbert_s3": "FinBERT-3",
    "C2_finbert_s4": "FinBERT-4", "C3_roberta_s1": "RoBERTa",
    "C4_longformer": "Longformer", "C5_qwen3": "Qwen3-emb",
    "C6_llmtext": "LLM-text", "D1_concat_mlp": "Concat",
    "D2_gated_fusion": "Gated", "D4_llmfused": "LLM-fused",
}
DISC = {"long_form": "10-K/Q", "event_driven": "8-K"}

d = pd.read_csv(os.path.join(TAB, "deployable_combiner.csv"))
d["m1_genuine"] = d.m1_genuine.astype(str).eq("True")
p = d[d.in_primary_grid].copy()

n_surv = int((p.genuine_fixed & p.genuine_exp).sum())
n_lost = int((p.genuine_fixed & ~p.genuine_exp).sum())
n_gain = int((~p.genuine_fixed & p.genuine_exp).sum())
n_null = int((~p.genuine_fixed & ~p.genuine_exp).sum())

gate({"cells_all": 75, "cells_primary": 69, "m1_genuine": 38,
      "genuine_fixed": 36, "genuine_exp": 6,
      "survive": 5, "lost": 31, "gained": 1, "null": 32},
     {"cells_all": len(d), "cells_primary": len(p),
      "m1_genuine": int(p.m1_genuine.sum()),
      "genuine_fixed": int(p.genuine_fixed.sum()),
      "genuine_exp": int(p.genuine_exp.sum()),
      "survive": n_surv, "lost": n_lost, "gained": n_gain, "null": n_null})

# ------------------------------------------------------------------ canvas
apply_style(9)
# The lower row carries two-line bar notes set in absolute points, so it needs
# a taller share of the canvas than the 0.50 it had: at the old height the notes
# ran into the bar below them and the last one into the x-axis spine.
fig = plt.figure(figsize=(6.10, 7.10))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.92],
                      height_ratios=[1.0, 0.60],
                      left=0.105, right=0.985, top=0.878, bottom=0.062,
                      wspace=0.30, hspace=0.30)
ax = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])

# ----------------------------------------------------------- (a) scatter
STY = {
    "SURVIVES-DEPLOY": dict(c=GREEN, m="o", s=30, label="survives"),
    "GAINED-ON-DEPLOY": dict(c=PURPLE, m="^", s=32, label="gained on refit"),
    "LOST-ON-DEPLOY": dict(c=VERM, m="v", s=26, label="lost on refit"),
    "null-null": dict(c=GREY, m="o", s=13, label="null under both"),
}
ax.axhline(0, color=GREY, lw=0.6)
ax.axvline(0, color=GREY, lw=0.6)
lim = (-21.5, 10.5)
ax.plot([-4.6, 10], [-4.6, 10], color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
# The diagonal is apparatus, so its label recedes to INK2 rather than sharing the
# data labels' ink.
ax.text(7.35, 8.1, "no change", fontsize=9, color=INK2,
        ha="right", va="bottom")

for status, st in STY.items():
    for prim_flag, face, edge in ((True, st["c"], st["c"]), (False, "none", st["c"])):
        sub = d[(d.deploy_status == status) & (d.in_primary_grid == prim_flag)]
        if not len(sub):
            continue
        ax.scatter(sub.fixed_pooled_rel, sub.exp_pooled_rel, s=st["s"],
                   marker=st["m"], facecolors=face, edgecolors=edge,
                   linewidths=0.8, zorder=3 if prim_flag else 2)

ax.set_xlim(-7.5, 7.5)
ax.set_ylim(*lim)
ax.set_xlabel("pooled QLIKE gain from text, frozen validation weights (%)",
              fontsize=9)
ax.set_ylabel("... under expanding deployable weights (%)", fontsize=9)
ax.set_yticks([-20, -15, -10, -5, 0, 5, 10])
ax.set_title("(a) the same 75 cells under the two weight schemes",
             fontsize=9, color=GREY, pad=4, loc="left")

# Callout anchors.  The 45-degree "no change" rule runs through y = x, so a
# label whose baseline sits within ~1.2 y-units of its own x is struck through
# by it; the 10-K/Q h=5 callout is held 1.2 units below the rule for that reason.
LAB = {
    ("long_form", "C2_finbert_s1", 5): (-4.6, 8.6, "left"),
    ("long_form", "C6_llmtext", 5): (4.3, 3.1, "left"),
    ("long_form", "C6_llmtext", 10): (4.3, 1.1, "left"),
    ("long_form", "D1_concat_mlp", 5): (-6.9, 5.9, "left"),
    ("event_driven", "C6_llmtext", 5): (-3.5, -3.2, "right"),
    ("event_driven", "C6_llmtext", 10): (-3.5, -5.8, "right"),
    ("event_driven", "C5_qwen3", 20): (-6.9, 3.1, "left"),
}
for _, r in d[d.deploy_status.isin(["SURVIVES-DEPLOY", "GAINED-ON-DEPLOY"])].iterrows():
    tx, ty, ha = LAB[(r.disc, r.model, r.h)]
    col = GREEN if r.deploy_status == "SURVIVES-DEPLOY" else PURPLE
    tag = f"{SHORT[r.model]} {DISC[r.disc]} h={r.h}"
    if not r.in_primary_grid:
        tag += "*"
    ax.annotate(tag, xy=(r.fixed_pooled_rel, r.exp_pooled_rel),
                xytext=(tx, ty), fontsize=9, color=col, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=col,
                                shrinkA=1, shrinkB=3))

ax.annotate("long-form bag-of-words:\n+1.4 to +3.0 frozen,\n-13.5 to -17.4 refit",
            xy=(1.65, -17.44), xytext=(-2.4, -18.6), fontsize=9, color=VERM_TXT,
            ha="right", va="center", linespacing=1.15,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=VERM_TXT,
                            shrinkA=3, shrinkB=3))

handles = [Line2D([], [], ls="none", marker=st["m"], ms=np.sqrt(st["s"]),
                  mfc=st["c"], mec=st["c"], label=f'{st["label"]}')
           for st in STY.values()]
handles.append(Line2D([], [], ls="none", marker="o", ms=4.2, mfc="none",
                      mec=GREY, label="* outside primary grid"))
ax.legend(handles=handles, loc="lower right", fontsize=9, handletextpad=0.4,
          borderpad=0.2, labelspacing=0.28, borderaxespad=0.25)

# ----------------------------------------------------- (b) verdict flow
ax_b.set_xlim(0, 1)
ax_b.set_ylim(-9, 84)
ax_b.axis("off")
ax_b.set_title("(b) where the 69 primary cells go", fontsize=9, color=GREY,
               pad=4, loc="left")

BARW = 0.115
left_seg = [(0, n_lost + n_surv, GREEN), (n_lost + n_surv, 69, LIGHT)]
right_seg = [(0, n_surv + n_gain, PURPLE), (n_surv + n_gain, 69, LIGHT)]
for x0, segs in ((0.03, left_seg), (0.855, right_seg)):
    for a, b, c in segs:
        ax_b.add_patch(plt.Rectangle((x0, 69 - b), BARW, b - a, fc=c, ec="none"))

FLOW = [(0, n_surv, 0, n_surv, GREEN, f"{n_surv} survive"),
        (n_surv, n_surv + n_lost, n_surv + n_gain, 69, VERM, f"{n_lost} lost"),
        (n_surv + n_lost, 69, n_surv, n_surv + n_gain, PURPLE, f"{n_gain} gained"),
        (n_surv + n_lost, 69, n_surv + n_gain, 69, LIGHT, None)]
for a0, a1, b0, b1, col, lab in FLOW:
    if col is LIGHT:
        continue
    verts = [(0.03 + BARW, 69 - a0), (0.5, 69 - a0), (0.5, 69 - b0),
             (0.855, 69 - b0), (0.855, 69 - b1), (0.5, 69 - b1),
             (0.5, 69 - a1), (0.03 + BARW, 69 - a1), (0.03 + BARW, 69 - a0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax_b.add_patch(PathPatch(Path(verts, codes), fc=col, ec="none", alpha=0.42))

ax_b.text(0.03, 71.0, f"frozen\n{int(p.genuine_fixed.sum())} genuine", fontsize=9,
          color=GREEN, ha="left", va="bottom", linespacing=1.15)
ax_b.text(0.97, 71.0, f"expanding\n{int(p.genuine_exp.sum())} genuine", fontsize=9,
          color=PURPLE, ha="right", va="bottom", linespacing=1.15)
# Both of these sit on top of their own ribbon -- the "5" is taller than the
# 5-cell band it names and spills onto the band below it -- so they take the
# haloed in-axes callout instead of plain text.
annot(ax_b, 0.49, 69 - n_surv / 2, f"{n_surv}", color=GREEN,
      ha="center", va="center")
annot(ax_b, 0.49, 69 - (n_surv + n_lost / 2), f"{n_lost} lost",
      color=VERM_TXT, ha="center", va="center")
ax_b.text(0.49, -5.5, f"{n_gain} gained, {n_null} null throughout", fontsize=9,
          color=GREY, ha="center", va="center")

# ------------------------------------------------------- (c) count ladder
ax_c.set_title("(c) two different reasons the count falls", fontsize=9,
               color=GREY, pad=4, loc="left")
# One colour vocabulary across the three panels: GREEN is the frozen validation
# weights, PURPLE the deployable refit, VERM the loss, GREY the nulls.  These
# rungs were drawn in BLUE and SKY, two hues that appear nowhere else in the
# figure -- and the SKY one was the very same 36 that panel (b) draws in GREEN,
# so the same quantity carried two colours.  The frozen rungs now take GREEN and
# separate on fill, following panel (a)'s convention that solid is the primary
# 69-cell grid and hollow the wider 75-cell family; each bar also keeps its own
# printed value, so colour is never the only channel.
bars = [("38", 38, GREEN, True, "primary rung,\n69-cell Holm family"),
        ("36", 36, GREEN, False, "same evidence,\n75-cell Holm family"),
        ("6", 6, PURPLE, True, "weights refit\nquarter by quarter")]
ypos = [2, 1, 0]
for y, (txt, v, col, solid, blurb) in zip(ypos, bars, strict=False):
    ax_c.barh(y, v, height=0.36, color=col if solid else "none",
              edgecolor="none" if solid else col,
              linewidth=0.0 if solid else 0.9)
    ax_c.text(v + 1.2, y, txt, fontsize=9, color=col, va="center", ha="left")
    ax_c.text(0.6, y - 0.30, blurb, fontsize=9, color=GREY, va="top",
              ha="left", linespacing=1.15)
ax_c.set_xlim(0, 46)
# The bottom bar's two-line note hangs below y = 0; the axis floor has to clear
# it, or the x-spine is drawn straight through the second line.
ax_c.set_ylim(-1.05, 2.55)
ax_c.set_yticks([])
ax_c.set_xticks([0, 10, 20, 30, 40])
ax_c.set_xlabel("cells adding text (of 69)", fontsize=9)
for s in ("left", "right", "top"):
    ax_c.spines[s].set_visible(False)

# ------------------------------------------------------------------ notes
# This block is the figure's basis statement, not one of its findings, and it
# used to be set in GREY -- the same ink as the panel titles and the data labels
# -- with nothing between it and panel (a).  It now takes INK2 and a hairline,
# which is the only hierarchy available here: finish() writes with
# bbox_inches="tight", so any change of size or any outward move would widen the
# page and shrink every printed glyph.  The rule sits in whitespace the bounding
# box already contained, between the block and the (a) title, and spans exactly
# the width already inked by the axes below it -- so it costs no geometry.  The
# three lines keep their own 0.022 spacing rather than going through note(),
# whose 1.32 linespacing would reflow them.
fig.text(0.105, 0.988,
         "Genuine = day-clustered DM < 0, Holm p < .05 within the scheme's own "
         "family, and a",
         fontsize=9, color=INK2, va="top", ha="left")
fig.text(0.105, 0.966,
         "permuted-text placebo |DM| < 2. Text forecasts are identical "
         "throughout; only the",
         fontsize=9, color=INK2, va="top", ha="left")
fig.text(0.105, 0.944,
         "weight-estimation scheme changes.",
         fontsize=9, color=INK2, va="top", ha="left")
fig.lines.append(plt.Line2D([0.105, 0.985], [0.916, 0.916],
                            transform=fig.transFigure, color=RULE,
                            linewidth=0.5, zorder=0.5))

finish(fig, "stability_freeze_point_deployability")
print("survive/lost/gained/null (69 primary):", n_surv, n_lost, n_gain, n_null)
print("pooled rel ranges: fixed", round(d.fixed_pooled_rel.min(), 2),
      round(d.fixed_pooled_rel.max(), 2), " exp", round(d.exp_pooled_rel.min(), 2),
      round(d.exp_pooled_rel.max(), 2))
print("cells below the diagonal:", int((d.exp_pooled_rel < d.fixed_pooled_rel).sum()),
      "of", len(d))
