"""Appendix figure FP3 — the joint tests, panel by panel.

Pairwise Diebold-Mariano controls error only across the pairs actually tested.
Hansen's superior-predictive-ability test and the model confidence set answer
the aggregate data-snooping objection instead: is HAR-RV beaten by the BEST of
the whole leaderboard, and which models cannot be separated from the best.

  (a) 90 % model-confidence-set membership, every model against every one of
      the 18 loss x disclosure x horizon panels.  The whole text block is
      empty in all 18;
  (b) the SPA consistent p-value for each panel, against the full alternative
      set and against the text-and-fusion block alone.

Sources
-------
results/tables/row13_spa_mcs.csv          per-model, per-panel MCS membership
results/tables/row13_spa_mcs_panels.csv   per-panel SPA p-values and counts
"""
import os
import sys
import textwrap

import numpy as np
import pandas as pd

ANALYSIS = "scripts/analysis"
sys.path.insert(0, ANALYSIS)

import diss_style as ds
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from supp_style import (
    BLUE,
    GREY,
    LIGHT,
    PURPLE,
    TAB,
    VERM,
    VERM_TXT,
    apply_style,
    gate,
)

# ---------------------------------------------------------------- evidence
cells = pd.read_csv(os.path.join(TAB, "row13_spa_mcs.csv"))
pans = pd.read_csv(os.path.join(TAB, "row13_spa_mcs_panels.csv"))

DISC = ["long_form", "event_driven", "combined"]
DISC_LAB = {"long_form": "Long-form (10-K/Q)", "event_driven": "Event-driven (8-K)",
            "combined": "Combined"}
HS = [5, 10, 20]
LOSSES = [("qlike", "Q"), ("se", "SE")]

PANELS = [(d, h, ln) for d in DISC for h in HS for ln, _ in LOSSES]

ORDER = [
    ("price", "A1_hv", "A1  historical volatility"),
    ("price", "A2_har_rv", "A2  HAR-RV  (benchmark)"),
    ("price", "A3_garch", "A3  GARCH"),
    ("price", "A4_egarch", "A4  EGARCH"),
    ("price", "A5_arima", "A5  ARIMA"),
    ("price", "A6_harq", "A6  HARQ"),
    ("price", "A6_shar", "A6  semivariance HAR"),
    ("price", "A7_harx_vix", "A7  HAR-X with VIX"),
    ("text", "B1_bow_ridge", "B1  bag-of-words ridge"),
    ("text", "B2_tfidf_ridge", "B2  TF-IDF ridge"),
    ("text", "B3_lm_linear", "B3  Loughran-McDonald linear"),
    ("text", "B4_lm_features", "B4  Loughran-McDonald features"),
    ("text", "C1_bert_s1", "C1  BERT"),
    ("text", "C2_finbert_s1", "C2  FinBERT"),
    ("text", "C3_roberta_s1", "C3  RoBERTa"),
    ("text", "C4_longformer", "C4  Longformer"),
    ("text", "C5_e5mistral", "C5  E5-Mistral embeddings"),
    ("text", "C5_gteqwen2", "C5  gte-Qwen2 embeddings"),
    ("text", "C5_qwen3", "C5  Qwen3 embeddings"),
    ("text", "C6_llmtext", "C6  prompted Qwen3-32B"),
    ("text", "C6_llmtext_llama70", "C6  prompted Llama-3.1-70B"),
    ("fusion", "D1_concat_mlp", "D1  concat MLP"),
    ("fusion", "D2_gated_fusion", "D2  gated fusion"),
    ("fusion", "D3_e5mistral", "D3  price + E5-Mistral"),
    ("fusion", "D3_gteqwen2", "D3  price + gte-Qwen2"),
    ("fusion", "D3_qwen3", "D3  price + Qwen3"),
    ("fusion", "D4_llmfused", "D4  price + prompted LLM"),
]
BLOCK_COL = {"price": BLUE, "text": VERM, "fusion": PURPLE}

idx = cells.set_index(["disclosure", "horizon", "loss", "model"])


def member(model, panel):
    d, h, ln = panel
    try:
        return bool(idx.loc[(d, h, ln, model), "in_mcs90"])
    except KeyError:
        return None                      # the model was not run on that panel


counts = cells.groupby("block")["in_mcs90"].sum().to_dict()
n_text_cells = int((cells.block == "text").sum())
spa_reject = int((pans.spa_p_consistent < 0.05).sum())
tf_all_one = int((pans.spa_textfusion_p_consistent == 1.0).sum())
panels_with_fusion = int((pans.mcs90_fusion > 0).sum())

# ------------------------------------------------------------------- gate
gate(
    {"n_panels": 18, "n_text_cells": 222, "text_in_mcs": 0,
     "fusion_in_mcs": 24, "price_in_mcs": 73, "spa_rejects_full_set": 9,
     "spa_textfusion_p_is_one": 18, "panels_with_a_fusion_member": 12},
    {"n_panels": len(pans), "n_text_cells": n_text_cells,
     "text_in_mcs": int(counts.get("text", 0)),
     "fusion_in_mcs": int(counts.get("fusion", 0)),
     "price_in_mcs": int(counts.get("price", 0)),
     "spa_rejects_full_set": spa_reject,
     "spa_textfusion_p_is_one": tf_all_one,
     "panels_with_a_fusion_member": panels_with_fusion},
)

# ------------------------------------------------------------------ canvas
apply_style(9)
fig = plt.figure(figsize=ds.canvas(7.55))
gs = fig.add_gridspec(2, 1, height_ratios=[4.15, 1.00],
                      left=0.335, right=0.985, top=0.900, bottom=0.135,
                      hspace=0.46)
axM = fig.add_subplot(gs[0])
axS = fig.add_subplot(gs[1])

# column x positions: a gap between disclosure groups
xs, x = [], 0.0
for gi in range(3):
    for hi in range(3):
        for li in range(2):
            xs.append(x)
            x += 1.0
        x += 0.6
    x += 1.1
xs = np.array(xs)

ys = np.arange(len(ORDER))[::-1].astype(float)
# a gap between blocks
gap = {"price": 0.0, "text": -0.7, "fusion": -1.4}
ys = np.array([ys[i] + gap[ORDER[i][0]] for i in range(len(ORDER))])

for yi, (blk, mid, lab) in zip(ys, ORDER, strict=False):
    axM.axhline(yi, color=LIGHT, lw=0.45, zorder=0)
    for xi, panel in zip(xs, PANELS, strict=False):
        m = member(mid, panel)
        if m is None:
            axM.plot([xi], [yi], marker="x", ms=3.4, color=LIGHT, mew=1.0,
                     zorder=3)
        elif m:
            axM.add_patch(Rectangle((xi - 0.36, yi - 0.36), 0.72, 0.72,
                                    facecolor=BLOCK_COL[blk], edgecolor="none",
                                    zorder=4))
        else:
            axM.plot([xi], [yi], marker=".", ms=1.7, color="#B8B8B8",
                     zorder=3)

# block bands
for blk, col in (("text", VERM), ("price", BLUE), ("fusion", PURPLE)):
    sel = [i for i, o in enumerate(ORDER) if o[0] == blk]
    lo, hi = ys[sel].min(), ys[sel].max()
    axM.add_patch(Rectangle((xs[0] - 0.85, lo - 0.62),
                            xs[-1] - xs[0] + 1.7, hi - lo + 1.24,
                            facecolor="none", edgecolor=col, lw=0.7,
                            alpha=0.55, zorder=1))

axM.set_yticks(ys)
axM.set_yticklabels([o[2] for o in ORDER], fontsize=8.9)
for tick, o in zip(axM.get_yticklabels(), ORDER, strict=False):
    tick.set_color(GREY)
axM.set_xticks(xs)
axM.set_xticklabels([lab for _ in range(9) for _, lab in LOSSES], fontsize=8.9)
axM.set_xlim(xs[0] - 0.9, xs[-1] + 0.9)
axM.set_ylim(ys.min() - 0.95, ys.max() + 0.95)
axM.tick_params(axis="both", length=0)
for sp in ("left", "bottom"):
    axM.spines[sp].set_visible(False)

# horizon and disclosure headers
for gi, d in enumerate(DISC):
    grp = xs[gi * 6:(gi + 1) * 6]
    axM.text(grp.mean(), ys.max() + 2.05, DISC_LAB[d], ha="center",
             va="center", fontsize=8.9, color=GREY)
    for hi, h in enumerate(HS):
        pair = grp[hi * 2:hi * 2 + 2]
        axM.text(pair.mean(), ys.max() + 1.15, f"{h}", ha="center",
                 va="center", fontsize=8.9, color=GREY)
    axM.plot([grp[0] - 0.5, grp[-1] + 0.5],
             [ys.max() + 1.62, ys.max() + 1.62], color=GREY, lw=0.6,
             clip_on=False)

axM.text(xs[0] - 1.5, ys.max() + 1.15, "horizon, days:", ha="right",
         va="center", fontsize=8.9, color=GREY, clip_on=False)
# This count is set inside the plotting area, where it ran through two columns
# of the dot grid and over two row rules.  A white halo keeps both the label and
# the cells it covers readable; it stays vermillion because that is the text
# block's own colour, so the reader can see which block is being counted.  Pulled
# 0.35 units inside the right limit so the halo cannot widen the tight bbox.
ds.annot(axM, xs[-1] + 0.55,
         ys[[o[0] for o in ORDER].index("text")] - 6.0,
         f"0 of {n_text_cells}\ntext cells", size=8.9, color=VERM_TXT,
         ha="right", va="center")

# ------------------------------------------------------------ (b) SPA strip
p_full = []
p_tf = []
for d, h, ln in PANELS:
    r = pans[(pans.disclosure == d) & (pans.horizon == h) & (pans.loss == ln)]
    p_full.append(float(r.spa_p_consistent.iloc[0]))
    p_tf.append(float(r.spa_textfusion_p_consistent.iloc[0]))
p_full = np.array(p_full)
p_tf = np.array(p_tf)

axS.axhspan(0.0008, 0.05, color=LIGHT, alpha=0.55, lw=0, zorder=0)
axS.axhline(0.05, color=GREY, ls="--", lw=0.8, zorder=2)
for xi, pf in zip(xs, p_full, strict=False):
    axS.plot([xi, xi], [0.0009, pf], color=GREY, lw=0.7, zorder=2)
    axS.plot([xi], [pf], marker="o", ms=5.0, zorder=4,
             mfc=BLUE if pf < 0.05 else "white", mec=BLUE, mew=1.1)
axS.plot(xs, p_tf, marker="D", ms=4.4, ls="none", mfc="white", mec=VERM,
         mew=1.1, zorder=4)
axS.set_yscale("log")
axS.set_ylim(0.0009, 2.6)
axS.set_yticks([0.001, 0.01, 0.05, 1.0])
axS.set_yticklabels(["0.001", "0.01", "0.05", "1.0"], fontsize=9.0)
axS.set_xticks(xs)
axS.set_xticklabels([lab for _ in range(9) for _, lab in LOSSES], fontsize=8.9)
axS.set_xlim(xs[0] - 0.9, xs[-1] + 0.9)
axS.set_ylabel("SPA consistent $p$", fontsize=8.9)
axS.tick_params(axis="x", length=0)
axS.plot([], [], marker="o", ms=5.0, ls="none", mfc=BLUE, mec=BLUE,
         label="vs. the full alternative set")
axS.plot([], [], marker="D", ms=4.4, ls="none", mfc="white", mec=VERM,
         label="vs. the text and fusion block only")
axS.legend(loc="lower left", fontsize=8.9, handletextpad=0.35, borderpad=0.25,
           labelspacing=0.25, ncol=2, columnspacing=1.4,
           bbox_to_anchor=(0.0, 1.02))

# ---------------------------------------------------------- panel headings
fig.canvas.draw()


def heading(ax, text, dy):
    fig.text(0.004, ax.get_position().y1 + dy, text, fontsize=9.2, color=GREY,
             ha="left", va="bottom")


heading(axM, "(a)  90 % model-confidence-set membership, all 18 panels: "
             "filled = in the set", 0.070)
heading(axS, "(b)  Hansen's superior-predictive-ability test, panel by panel",
        0.030)

NOTE = (
    "Q = QLIKE in volatility units, SE = squared error. A filled square means "
    "the model is in that panel's 90 % set; a dot means it was run on the panel "
    "and excluded; a cross means it was not run on that disclosure. Membership "
    "means 'cannot be separated from the best model', not 'beats HAR-RV'."
)
# Basis statement, not argument: recessive ink and a hairline above it, so the
# definitions of Q/SE/square/dot/cross and the "membership is not victory"
# clause read as apparatus rather than as another data label.  Every word is
# kept; only the ink and the separator change, neither of which is geometry.
# 1.34 line spacing is preserved -- ds.note() would tighten it to 1.32.
ntxt = fig.text(0.004, 0.006, textwrap.fill(NOTE, 92), fontsize=8.9,
                color=ds.INK2, ha="left", va="bottom", linespacing=1.34)

# The rule is measured off the block's own rendered box and cut to its width, so
# it cannot push the tight bounding box outwards in either direction.
fig.canvas.draw()
_nb = ntxt.get_window_extent(fig.canvas.get_renderer()).transformed(
    fig.transFigure.inverted())
fig.lines.append(plt.Line2D([_nb.x0, _nb.x1], [_nb.y1 + 0.011] * 2,
                            transform=fig.transFigure, color=ds.RULE,
                            linewidth=0.5, zorder=0.5))

ds.finish(fig, "FP3_joint_spa_mcs", max_render_pt=595.0,
          note="appendix figure: MCS membership matrix and SPA p-values over "
               "the 18 loss x disclosure x horizon panels")
