"""F14 -- Does the residual pay? Portfolio Sharpe and value-at-risk.

Substantiates, and bounds the scope of, these frozen main-text sentences:
  06_results.tex, "What survives": "Economically, $\\Delta$Sharpe is significant
    in 1 of 18 grid cells and 0 of C6's 6 (median $+0.003$); VaR and utility
    tests agree."
  09_conclusion.tex: "The residual ... prices at $\\Delta$Sharpe 0 of C6's 6
    portfolio cells (grid-wide 1 of 18)".
  05_protocol.tex: "Surviving increments are also adjudicated economically, by
    inverse-variance portfolio $\\Delta$Sharpe, VaR backtests, a utility fee."

Evidence files read (every plotted number comes from one of these):
  results/tables/portfolio_econ.csv -- disc, model, h, n_periods, avg_names,
      sharpe_R, sharpe_U, sharpe_diff, sharpe_diff_lo, sharpe_diff_hi,
      sig_sharpe_diff
  results/tables/portfolio_econ.md  -- the committed verdict block (median, 1/18)
  results/tables/var_backtest.csv   -- disclosure, model, horizon, alpha,
      mu_mode, forecast, viol_rate, tick_loss, dm_tick_vs_fR_stat,
      dm_tick_vs_fR_p
  results/tables/var_backtest.md    -- the committed 72-cell tally block and the
      overlapping-window caveat on the Christoffersen statistics
  results/tables/utility_value.csv  -- fee_bps_ann, read only for the caption's
      cross-check that the utility table carries no prompted-arm (C6) row

Conventions on the artefact itself, because the two panels do NOT share one:
  panel (a): annualised Sharpe of a long-only inverse-variance book, difference
      (text-augmented minus recalibrated-HAR), non-overlapping h-day holding
      periods, day-block bootstrap over blocks of h periods;
  panel (b): realised left-tail violation rate of a Gaussian VaR, pooled-drift
      mode only, 72 (disclosure x model x horizon x alpha) cells.
  Neither panel is a QLIKE quantity; nothing here is comparable to the ladder's
  relative-QLIKE percentages.

Scope limits carried in the prose: the VaR and utility tables contain no C6
(prompted-arm) rows at all, so they corroborate the grid rather than adjudicate
the surviving 8-K residual; the Christoffersen conditional-coverage statistics
in the source are indicative only because the h-day return windows overlap.

Presentation (no number, no word of any note, changed by it)
-----------------------------------------------------------
The three prose blocks are apparatus, not argument, so they are set through
`note()` in the recessive ink INK2 and the panel markers they used to carry as a
"(a)  "/"(b)  " prefix are drawn instead by `panel()` (and, for panel (b), by the
one figure-coordinate call `panel()` cannot make) on one baseline with a short
title taken from the protocol's own nouns.  Each block is left-aligned with the
panel it describes -- the (a) blocks on panel (a)'s left spine, the (b) block on
panel (b)'s label column -- because this canvas has no vertical gap wide enough
to carry a RULE hairline with clearance: `regen.py --search` already bisected it
to the point where any further compression collides type.  Panel (b)'s legend is
its key alone, so it now shares a baseline with panel (b)'s own marker instead of
floating between two note blocks.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from supp_style import (
    BLUE,
    GREEN,
    GREY,
    INK,
    TAB,
    VERM,
    VERM_TXT,
    annot,
    apply_style,
    finish,
    gate,
    note,
    panel,
)

# --------------------------------------------------------------- load evidence
PE = pd.read_csv(os.path.join(TAB, "portfolio_econ.csv"))
VB = pd.read_csv(os.path.join(TAB, "var_backtest.csv"))
UV = pd.read_csv(os.path.join(TAB, "utility_value.csv"))

POOLED = VB[VB.mu_mode == "pooled"]
FU = POOLED[POOLED.forecast == "fU"]
FR = POOLED[POOLED.forecast == "fR"]
RAW = POOLED[POOLED.forecast == "rawHAR"]

# The reference-against-itself rows exist in the file (forecast == 'fR' carries
# dm_tick_vs_fR_stat == 0 with p == 1 by construction). They are excluded from
# every tally below and are never drawn as a tie; the violation-rate panel plots
# the reference's own violation rate, which is a level, not a self-comparison.
assert set(POOLED.forecast) == {"rawHAR", "fR", "fU"}
assert np.allclose(FR.dm_tick_vs_fR_stat.values, 0.0)
assert np.allclose(FR.dm_tick_vs_fR_p.values, 1.0)

KEY = ["disclosure", "model", "horizon", "alpha"]
MRG = FU.merge(FR, on=KEY, suffixes=("_U", "_R"))
CLOSER = int(((MRG.viol_rate_U - MRG.alpha).abs()
              < (MRG.viol_rate_R - MRG.alpha).abs()).sum())

N_LOWER = int((FU.dm_tick_vs_fR_stat < 0).sum())
N_LOWER_SIG = int(((FU.dm_tick_vs_fR_stat < 0) & (FU.dm_tick_vs_fR_p < .05)).sum())
N_WORSE_SIG = int(((FU.dm_tick_vs_fR_stat > 0) & (FU.dm_tick_vs_fR_p < .05)).sum())
N_RECAL = int((RAW.dm_tick_vs_fR_stat > 0).sum())
N_RECAL_SIG = int(((RAW.dm_tick_vs_fR_stat > 0) & (RAW.dm_tick_vs_fR_p < .05)).sum())

# The programme's draft caption asserted "at the 1% level every forecast
# over-violates". Adversarial repair: verified cell by cell instead of asserted,
# and stated as a count -- it is 107 of 108, not all of them. The single
# exception is long_form / C5_qwen3 / h=20 under fU.
A1 = POOLED[POOLED.alpha == 0.01]
N_OVER_1PC = int((A1.viol_rate > 0.01).sum())
N_CELLS_1PC = len(A1)
UNDER_1PC = A1[A1.viol_rate <= 0.01]
assert len(UNDER_1PC) == 1
EXC = UNDER_1PC.iloc[0]                      # the single under-violating cell

# Second adversarial repair: the draft prose called the recalibration move
# "visibly the smaller of the two", a verdict adjective the reader cannot in
# fact check -- the median text move is 0.05 percentage points on a facet
# spanning 2.6, i.e. under 1.5 pt, so the triangle usually covers the circle.
# Both moves are therefore counted here and the count is printed, not asserted.
PIV = POOLED.pivot_table(index=KEY, columns="forecast", values="viol_rate")
MOVE_RECAL = (PIV["rawHAR"] - PIV["fR"]).abs()
MOVE_TEXT = (PIV["fU"] - PIV["fR"]).abs()
N_RECAL_LARGER = int((MOVE_RECAL > MOVE_TEXT).sum())
MED_RECAL_PP = round(float(MOVE_RECAL.median()) * 100, 2)
MED_TEXT_PP = round(float(MOVE_TEXT.median()) * 100, 2)

MEDIAN_DSH = float(PE.sharpe_diff.median())
N_SIG = int(PE.sig_sharpe_diff.sum())
C6 = PE[PE.model == "C6_llmtext"]
NAN_CI = PE[PE.sharpe_diff_lo.isna()]

# Third adversarial repair: an undefined interval does not "cover zero", so the
# coverage count is taken over the cells that actually have an interval, and the
# prompted arm's count is taken the same way (one of its six has none).
HAS_CI = PE[PE.sharpe_diff_lo.notna()]
N_HAS_CI = len(HAS_CI)
N_COVER0 = int(((HAS_CI.sharpe_diff_lo <= 0) & (HAS_CI.sharpe_diff_hi >= 0)).sum())
C6_CI = C6[C6.sharpe_diff_lo.notna()]
N_C6_HAS_CI = len(C6_CI)
N_C6_COVER0 = int(((C6_CI.sharpe_diff_lo <= 0)
                   & (C6_CI.sharpe_diff_hi >= 0)).sum())

# ------------------------------------------------------------------------ gate
# The literal side of gate() is the only place a number is typed by hand; it
# exists to abort the build the moment the committed tables stop saying what the
# frozen main text says.
gate(
    {
        "n_portfolio_cells": 18, "n_sig_sharpe": 1, "n_c6_cells": 6,
        "n_c6_sig": 0, "median_dsharpe": 0.003,
        "sig_cell": ("event_driven", "B2_tfidf_ridge", 10),
        "sig_point_lo_hi": (0.0257, 0.0036, 0.0474),
        "n_undefined_ci": 3, "undefined_periods": 39, "undefined_sharpe_R": -0.104,
        "n_var_cells": 72, "tick_lower": 55, "tick_lower_sig": 29,
        "tick_worse_sig": 12, "viol_closer": 35,
        "recal_better": 60, "recal_better_sig": 54,
        "over_violate_1pc": 107, "cells_1pc": 108,
        "under_1pc_cell": ("long_form", "C5_qwen3", 20, "fU"),
        "under_1pc_rate": 0.0099,
        "n_with_ci": 15, "n_cover_zero": 14,
        "n_c6_with_ci": 5, "n_c6_cover_zero": 5,
        "recal_move_larger": 69, "n_move_pairs": 72,
        "median_move_recal_pp": 0.95, "median_move_text_pp": 0.05,
        "n_c6_rows_in_var": 0, "n_c6_rows_in_utility": 0,
    },
    {
        "n_portfolio_cells": len(PE), "n_sig_sharpe": N_SIG,
        "n_c6_cells": len(C6), "n_c6_sig": int(C6.sig_sharpe_diff.sum()),
        "median_dsharpe": round(MEDIAN_DSH, 3),
        "sig_cell": tuple(PE[PE.sig_sharpe_diff][["disc", "model", "h"]]
                          .itertuples(index=False, name=None))[0],
        "sig_point_lo_hi": tuple(
            round(float(v), 4) for v in
            PE[PE.sig_sharpe_diff][["sharpe_diff", "sharpe_diff_lo",
                                    "sharpe_diff_hi"]].iloc[0]),
        "n_undefined_ci": len(NAN_CI),
        "undefined_periods": int(NAN_CI.n_periods.unique()[0]),
        "undefined_sharpe_R": round(float(NAN_CI.sharpe_R.unique()[0]), 3),
        "n_var_cells": len(FU), "tick_lower": N_LOWER,
        "tick_lower_sig": N_LOWER_SIG, "tick_worse_sig": N_WORSE_SIG,
        "viol_closer": CLOSER,
        "recal_better": N_RECAL, "recal_better_sig": N_RECAL_SIG,
        "over_violate_1pc": N_OVER_1PC, "cells_1pc": N_CELLS_1PC,
        "under_1pc_cell": (EXC.disclosure, EXC.model, int(EXC.horizon),
                           EXC.forecast),
        "under_1pc_rate": round(float(EXC.viol_rate), 4),
        "n_with_ci": N_HAS_CI, "n_cover_zero": N_COVER0,
        "n_c6_with_ci": N_C6_HAS_CI, "n_c6_cover_zero": N_C6_COVER0,
        "recal_move_larger": N_RECAL_LARGER, "n_move_pairs": len(PIV),
        "median_move_recal_pp": MED_RECAL_PP,
        "median_move_text_pp": MED_TEXT_PP,
        "n_c6_rows_in_var": int((VB.model == "C6_llmtext").sum()),
        "n_c6_rows_in_utility": int((UV.model == "C6_llmtext").sum()),
    },
)

# ---------------------------------------------------------------- presentation
# Model codes are the paper's own; the key line under panel (b) expands them,
# which keeps 36 y labels inside a narrow gutter.
CODE = {"B2_tfidf_ridge": "B2", "C2_finbert_s1": "C2-1", "C2_finbert_s2": "C2-2",
        "C4_longformer": "C4", "C5_qwen3": "C5", "C6_llmtext": "C6",
        "D2_gated_fusion": "D2"}
LONG = {"B2_tfidf_ridge": "B2 TF-IDF ridge", "C2_finbert_s1": "C2 FinBERT s1",
        "C6_llmtext": "C6 prompted LLM"}
DISC = {"event_driven": "ED", "long_form": "LF"}
DISC_LONG = {"event_driven": "event-driven (8-K)",
             "long_form": "long-form (10-K / 10-Q)"}
MODEL_ORDER_A = ["B2_tfidf_ridge", "C2_finbert_s1", "C6_llmtext"]
MODEL_ORDER_B = ["B2_tfidf_ridge", "C2_finbert_s1", "C2_finbert_s2",
                 "C4_longformer", "C5_qwen3", "D2_gated_fusion"]

apply_style(base_size=9)
# Vertical space is allocated top-down in inches so that no text block can drift
# into another, and every free-text line is kept under ~92 characters so the
# finished canvas stays 6.4 in wide: a figure wider than the supplement's text
# block would be scaled down on inclusion, taking every label below 9 pt.
H = 8.55
fig = plt.figure(figsize=(6.4, H))


def frac(inches_from_top):
    return 1.0 - inches_from_top / H


gsa = fig.add_gridspec(1, 1, left=0.145, right=0.985,
                       top=frac(0.46), bottom=frac(2.86))
axa = fig.add_subplot(gsa[0, 0])

# ------------------------------------------- panel (a): portfolio Delta Sharpe
rows_a, y, labels_a = [], 0.0, []
for mi, mod in enumerate(MODEL_ORDER_A):
    if mi:
        y -= 0.85                              # blank line between model blocks
    for disc in ("event_driven", "long_form"):
        for h in (5, 10, 20):
            r = PE[(PE.disc == disc) & (PE.model == mod) & (PE.h == h)].iloc[0]
            rows_a.append((y, r))
            labels_a.append(f"{DISC[disc]}  $h{{=}}{h}$")
            y -= 1.0

axa.axvline(0.0, lw=0.9, color=GREY, zorder=1)
axa.axvline(MEDIAN_DSH, lw=1.1, color=GREEN, ls=(0, (4, 2)), zorder=1)
for yy, r in rows_a:
    sig = bool(r.sig_sharpe_diff)
    if not pd.isna(r.sharpe_diff_lo):
        axa.plot([float(r.sharpe_diff_lo), float(r.sharpe_diff_hi)], [yy, yy],
                 lw=1.2, color=(VERM if sig else GREY), zorder=2,
                 solid_capstyle="butt")
        for xx in (float(r.sharpe_diff_lo), float(r.sharpe_diff_hi)):
            axa.plot([xx, xx], [yy - 0.24, yy + 0.24], lw=1.0,
                     color=(VERM if sig else GREY), zorder=2)
    axa.plot([float(r.sharpe_diff)], [yy], marker="o", ms=5.0,
             mfc=(VERM if sig else "white"),
             mec=(VERM if sig else GREY), mew=1.0, zorder=3)

axa.set_yticks([yy for yy, _ in rows_a])
axa.set_yticklabels(labels_a, fontsize=9)
axa.set_ylim(rows_a[-1][0] - 0.70, rows_a[0][0] + 1.55)
axa.set_xlim(-0.058, 0.088)
axa.set_xticks([-0.05, -0.025, 0.0, 0.025, 0.05, 0.075])
axa.set_xlabel("change in annualised Sharpe, text-augmented minus "
               "recalibrated HAR", fontsize=9)
axa.tick_params(axis="y", length=0)

for mi, mod in enumerate(MODEL_ORDER_A):
    axa.text(0.0865, rows_a[mi * 6][0], LONG[mod], fontsize=9, color=INK,
             fontweight="bold", ha="right", va="center")
# The median callout is an in-axes callout, so it carries annot()'s white halo:
# the top row's whisker reaches within 0.1 in of it and a longer one would run
# under the words.
annot(axa, MEDIAN_DSH + 0.0020, rows_a[0][0] + 0.85,
      f"grid median $+${MEDIAN_DSH:.3f}", color=GREEN, ha="left", va="center")

# Marker and title on one baseline, inside the 1.55 blank rows the y limit leaves
# above the first cell: the only free space on this canvas is inside the axes,
# and putting the header there costs the tight bounding box nothing.  dy is set
# so the block clears the first row's whisker caps (y +/- 0.24 rows); dx insets it
# 0.05 in from the left spine, which the marker's opening bracket sat on top of
# when the header was placed at dx=0 (a marker above the axes, as elsewhere in
# this set, does not have that problem -- there is no spine beside it).
panel(axa, "a", "portfolio $\\Delta$Sharpe", dy=0.945, dx=0.010)

# Apparatus, not argument: INK2, and left-aligned on panel (a)'s own left spine
# so the reader can see at a glance which panel each basis statement supports.
note(fig, 0.145, frac(0.06),
     f"All {len(PE)} portfolio cells; whiskers are the day-block "
     f"bootstrap over non-overlapping\n"
     f"holding periods. Filled: interval excludes zero. All {len(C6)} "
     f"prompted-arm cells are hollow.",
     rule=False)

sig_r = PE[PE.sig_sharpe_diff].iloc[0]
# 3.30 rather than 3.28: at 3.28 this block's box overlapped the x label above it
# by 0.006 in and left 0.044 in below, so it read as a third line of the axis
# label. The gap below is now the larger of the two, which keeps the block with
# panel (a) rather than with panel (b)'s note under it.
note(fig, 0.145, frac(3.30),
     f"Filled: {DISC[sig_r.disc]} {LONG[sig_r.model]} "
     f"$h{{=}}{int(sig_r.h)}$, $+${float(sig_r.sharpe_diff):.4f} "
     f"[$+${float(sig_r.sharpe_diff_lo):.4f}, "
     f"$+${float(sig_r.sharpe_diff_hi):.4f}].  Bare dots: LF $h{{=}}20$,\n"
     f"interval undefined ({int(NAN_CI.n_periods.unique()[0])} holding "
     f"periods, {int(NAN_CI.h.unique()[0])}-period blocks).",
     rule=False)

# --------------------------------------- panel (b): realised violation rates
# Four facets in one row (disclosure x tail level) keep all 72 cells legible in
# a third of the height a single 36-row stack would need.
rows_b, labels_b = [], []
for mod in MODEL_ORDER_B:
    for h in (5, 10, 20):
        rows_b.append((-float(len(rows_b)), mod, h))
        labels_b.append(f"{CODE[mod]} $h{{=}}{h}$")

MK = [("rawHAR", "s", GREY, "raw HAR"),
      ("fR", "o", BLUE, "recalibrated HAR (the reference)"),
      ("fU", "^", VERM, "text-augmented")]
# The nominal-1% facets start at 0.0072 rather than 0.0084 so the bold rule at
# 0.01 stands clear of the left spine: a marker CAN be drawn to its left, and
# exactly one is. Widening cannot separate that marker from the rule -- the cell
# sits 0.014 of a percentage point below nominal -- so it is ringed instead, in
# the same convention this supplement uses elsewhere for a called-out cell.
PANELS = [("event_driven", 0.01, (0.0072, 0.0348), [0.01, 0.02, 0.03]),
          ("event_driven", 0.05, (0.0330, 0.0815), [0.04, 0.06, 0.08]),
          ("long_form", 0.01, (0.0072, 0.0348), [0.01, 0.02, 0.03]),
          ("long_form", 0.05, (0.0330, 0.0815), [0.04, 0.06, 0.08])]
# Figure-fraction left edges. The first group carried 0.110 and 0.292, which put
# its 36 row labels 0.047 in PAST the left edge of the canvas; the tight bounding
# box then grew leftwards, and a wider box is a harder down-scale on the page,
# i.e. smaller type everywhere. Shifted right by 0.05 in (0.0078 of the width) so
# the longest label starts inside the canvas. The middle gutter absorbs it: it
# keeps 0.30 in of clearance to the second group's labels.
LEFTS = [0.118, 0.300, 0.632, 0.814]
WIDTH = 0.168
axes_b = []
for k, (disc, alpha, xlim, xticks) in enumerate(PANELS):
    ax = fig.add_axes([LEFTS[k], frac(7.98), WIDTH,
                       (7.98 - 5.38) / H])
    axes_b.append(ax)
    ax.axvline(alpha, lw=1.7, color=GREY, zorder=1)
    for yy, mod, h in rows_b:
        for fc, mk, col, _ in MK:
            v = POOLED[(POOLED.disclosure == disc) & (POOLED.model == mod)
                       & (POOLED.horizon == h) & (POOLED.alpha == alpha)
                       & (POOLED.forecast == fc)].viol_rate
            ax.plot([float(v.iloc[0])], [yy], marker=mk, ms=4.4, mfc=col,
                    mec=col, mew=0.8, zorder=3, alpha=0.95)
            if (disc, mod, h, alpha, fc) == (EXC.disclosure, EXC.model,
                                             int(EXC.horizon), EXC.alpha,
                                             EXC.forecast):
                ax.plot([float(v.iloc[0])], [yy], marker="o", ms=10.5,
                        mfc="none", mec=VERM, mew=1.3, zorder=6)
    ax.set_ylim(rows_b[-1][0] - 0.7, rows_b[0][0] + 0.7)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{100 * t:g}%" for t in xticks], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.text(0.5, 1.018, f"nominal {100 * alpha:g}%", transform=ax.transAxes,
            fontsize=9, color=INK, ha="center", va="bottom")
    if k in (0, 2):
        ax.set_yticks([yy for yy, *_ in rows_b])
        tls = ax.set_yticklabels(labels_b, fontsize=9)
        # Colour is never the only channel: the ringed cell's row label is also
        # the only vermillion one, and the ring itself is the second channel.
        for tl, (_, mod, h) in zip(tls, rows_b, strict=False):
            if (disc, mod, h) == (EXC.disclosure, EXC.model, int(EXC.horizon)):
                tl.set_color(VERM_TXT)
        ax.text(0.0, 1.115, DISC_LONG[disc], transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=INK, ha="left",
                va="bottom")
    else:
        ax.set_yticks([])

# Centred on the four facets' data areas (0.550), not on the canvas (0.520): the
# axis title was sitting 0.16 in left of the axis it names.
fig.text(0.550, frac(8.42), "realised violation rate", fontsize=9, color=INK,
         ha="center", va="bottom")

# Panel (b)'s apparatus, flush with panel (b)'s own label column.
note(fig, 0.001, frac(3.66),
     f"All {len(FU)} cells ({len(rows_b) * 2} rows $\\times$ 2 tail "
     f"levels), pooled-drift mode. ED $=$ 8-K, LF $=$ 10-K/10-Q.\n"
     f"Markers are levels, not comparisons: reference-against-itself rows "
     f"are excluded, not ties.\n"
     f"Ringed: the {N_CELLS_1PC - N_OVER_1PC} cell of {N_CELLS_1PC} that "
     f"under-violates at the nominal 1% level "
     f"({DISC[EXC.disclosure]} {CODE[EXC.model]} "
     f"$h{{=}}{int(EXC.horizon)}$,\n"
     f"text-augmented, {100 * float(EXC.viol_rate):.2f}%); the other "
     f"{N_OVER_1PC} over-violate. Recalibration moves the rate further "
     f"than\n"
     f"text does in {N_RECAL_LARGER} of {len(PIV)} cells "
     f"(medians {MED_RECAL_PP:.2f} and {MED_TEXT_PP:.2f} percentage "
     f"points).\n"
     f"B2 TF-IDF ridge; C2-1, C2-2 FinBERT seeds 1, 2; C4 Longformer; "
     f"C5 Qwen3 embedding; D2 fusion.",
     rule=False)

# The three marker shapes are panel (b)'s key and nobody else's -- panel (a) uses
# neither the shapes nor the hues -- so the key now shares one baseline with panel
# (b)'s marker and title instead of floating between two note blocks as though it
# spoke for the whole figure. The legend slides right to make room for the title.
# panel() cannot place this pair: it works in axes coordinates, and its title
# offset (0.038 of the axes width) is calibrated for a wide panel, not for a
# 1.08 in facet, so the marker and title are set here as one string, which is one
# baseline by construction.
fig.text(0.001, frac(4.93), "(b)  VaR violation rates", fontsize=10,
         fontweight="semibold", color=INK, ha="left", va="bottom")

fig.legend(handles=[Line2D([], [], ls="none", marker=mk, ms=5, color=col,
                           label=lab) for _, mk, col, lab in MK],
           loc="upper center", bbox_to_anchor=(0.585, frac(4.66)), ncol=3,
           fontsize=9, handletextpad=0.3, columnspacing=1.4, frameon=False)

finish(fig, "F14_economic_adjudication")
