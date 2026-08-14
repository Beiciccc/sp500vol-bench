"""F5 -- Is the maximal pool a shopped reference?

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
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from supp_style import (
    BLUE,
    GREEN,
    GREY,
    PURPLE,
    TAB,
    VERM,
    VERM_TXT,
    YELLOW,
    apply_style,
    finish,
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
W, H = 6.4, 9.0
fig = plt.figure(figsize=(W, H))


def fy(inches_from_top):
    return 1.0 - inches_from_top / H


def fx(inches_from_left):
    return inches_from_left / W


def axes_box(x0, x1, y_top, y_bot):
    return fig.add_axes([fx(x0), fy(y_bot), fx(x1 - x0), (y_bot - y_top) / H])


ax_a = axes_box(0.55, 3.25, 0.82, 2.32)
ax_b = axes_box(3.60, 6.30, 0.82, 2.32)

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
    ax_a.text(x, ve + 0.7, str(ve), ha="center", va="bottom", fontsize=9,
              color=BLUE, fontweight="bold")
    ax_a.text(x + 0.30, vs + 0.7, str(vs), ha="center", va="bottom",
              fontsize=9, color=GREY)
ax_a.axhline(PRIMARY, color=VERM_TXT, linewidth=0.8, linestyle=(0, (4, 2.2)),
             zorder=3)
ax_a.text(-0.46, PRIMARY + 1.0, f"{PRIMARY} = primary rung\n(single "
          f"recalibrated HAR)", fontsize=9, color=VERM_TXT, va="bottom",
          ha="left", linespacing=1.3)
ax_a.set_xticks(xs)
ax_a.set_xticklabels([lab for _, lab in specs], fontsize=9, linespacing=1.3)
ax_a.set_ylim(0, 52)
ax_a.set_ylabel("Holm survivors of 69", fontsize=9)
ax_a.tick_params(axis="y", labelsize=9)
ax_a.tick_params(axis="x", length=0, pad=3)
ax_a.legend(fontsize=9, loc="upper right", handlelength=1.2,
            borderpad=0.15, labelspacing=0.2, handletextpad=0.4)
fig.text(fx(0.10), fy(0.14),
         "(a) survivors by reference specification, ordered by\n"
         "survivors left -- an absorption ordering, not a test\n"
         "forecast-accuracy ordering; see (b)",
         ha="left", va="top", fontsize=9, color=GREY, linespacing=1.35)

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
fig.text(fx(3.60), fy(0.14),
         "(b) test QLIKE, one axis per panel\n(left = lower QLIKE)",
         ha="left", va="top", fontsize=9, color=GREY, linespacing=1.35)
fig.legend(*ax_b.get_legend_handles_labels(), fontsize=9, ncol=4,
           loc="upper left", bbox_to_anchor=(fx(0.10), fy(2.72)),
           handlelength=1.0, columnspacing=1.2, borderpad=0.1,
           handletextpad=0.35, frameon=False)
fig.text(fx(0.10), fy(2.98),
         "(b) The fitted pool beats the validation-best member in 5 of 6 "
         "panels (clustered DM -7.24 to -2.05). In the\n"
         "sixth, long-form h=20, it is significantly worse than that member "
         "(DM +8.49, p < 1e-15) and is the worst of\n"
         "the four references drawn. It loses to the never-fitted "
         "equal-weight pool in 6 of 6 panels, on test QLIKE.",
         fontsize=9, color=GREY, va="top", ha="left", linespacing=1.4)

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

C_TOP, C_PITCH = 3.78, 0.142
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
fig.text(fx(0.10), fy(3.62),
         "(c) the same cell against each single price reference: relative "
         "QLIKE improvement (%), seed-2026 basis",
         ha="left", va="center", fontsize=9, color=GREY)
fig.text(fx(0.10), fy(6.20),
         "Marked row (LF / B2_tfidf_ridge, h=5): the same cell reports +1.06 "
         "to +3.53 depending only on which single\n"
         "price model is the reference. Cell outline: clustered p < .05. "
         "Every column is computed on identical rows,\n"
         "and the pool column is the seed-2026 pool, so all six columns "
         "share one basis.",
         fontsize=9, color=GREY, ha="left", va="top", linespacing=1.4)

# ------------------------------------------------------------------- panel (d)
# Two 9-row blocks rather than one 18-row column: at 18 rows the row labels
# would fall under 9 pt on this canvas. Both blocks carry the same x limits,
# so the two channels stay directly comparable.
stronger = stronger.sort_values(["disclosure", "text_model", "h"])
stronger = stronger.reset_index(drop=True)
D_TOP, D_PITCH = 7.24, 0.145
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

fig.text(fx(0.10), fy(6.84),
         "(d) 18 identical cells, HAR against semivariance HAR",
         ha="left", va="center", fontsize=9, color=GREY)
fig.legend(*handles, fontsize=9, ncol=2, loc="center left",
           bbox_to_anchor=(fx(3.22), fy(6.84)), handlelength=1.2,
           columnspacing=1.6, borderpad=0.1, handletextpad=0.4, frameon=False)
dshift = float((stronger.A6_shar_rel_impr_pct
                - stronger.A2_rel_impr_pct).abs().max())
fig.text(fx(0.10), fy(8.78),
         "Both blocks: relative QLIKE improvement over the reference (%), "
         "one shared scale.",
         fontsize=9, color=GREY, ha="left", va="center")
fig.text(fx(0.10), fy(8.94),
         f"Largest shift across all 18 cells: {dshift:.2f} percentage points.",
         fontsize=9, color=GREY, ha="left", va="center")

finish(fig, "F5_maximal_pool_audit")
