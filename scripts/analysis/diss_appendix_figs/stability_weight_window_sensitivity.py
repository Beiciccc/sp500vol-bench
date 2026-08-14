"""Appendix figure — how much of the increment depends on the window the
combination weights are fitted on.

Two separate experiments, both holding the trained text forecasts and the test
rows completely fixed and moving only the rows the weights are estimated on.

(a) FREEZE ORIGIN (results/tables/freeze_window_sensitivity.csv, 40 cells = the
    38 placebo-confirmed genuine cells plus the six prompted-LLM cells).  Three
    origins: the committed full validation block, the same block with the first
    COVID half removed, and the 2018-19 tail of the training era.  The last is a
    diagnostic, not a regime test: those rows are in-sample for every text model.

(b) VALIDATION HALF (results/tables/valwindow_sensitivity.csv, the full 69-cell
    grid).  The validation block splits into a COVID half (2020) and a calm half
    (2021).  Refitting on each answers the reviewer's question -- does fitting on
    a crisis window suppress the text coefficient? -- as a paired contrast.

(c) The scoreboard both experiments share: how often the sign of the verdict
    survives the move.

Out: writing/dissertation/figures/stability_weight_window_sensitivity.pdf
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, "scripts/analysis")
import supp_style
from supp_style import (
    BLUE,
    GREEN,
    GREY,
    LIGHT,
    PURPLE,
    REPO,
    SKY,
    TAB,
    VERM,
    VERM_TXT,
    apply_style,
    finish,
    gate,
)

supp_style.OUTDIR = os.path.join(REPO, "writing", "dissertation", "figures")
RNG = np.random.default_rng(20260802)

# ---------------------------------------------------------------- evidence
fz = pd.read_csv(os.path.join(TAB, "freeze_window_sensitivity.csv"))
vw = pd.read_csv(os.path.join(TAB, "valwindow_sensitivity.csv"))

pf = fz.pivot_table(index=["disc", "model", "h"], columns="fit_window",
                    values="rel_impr_pct")
sig = (fz.dm_clustered < 0) & (fz.p_clustered < 0.05)
fz = fz.assign(sig=sig)
sig_by_win = fz.dropna(subset=["rel_impr_pct"]).groupby("fit_window").sig.sum()
n_by_win = fz.dropna(subset=["rel_impr_pct"]).groupby("fit_window").size()

pv = vw.pivot_table(index=["disc", "model", "h"], columns="window",
                    values="rel_impr_pct")
base = pv["committed_val"]
dcc = (pv["calm_2021"] - pv["covid_2020"]).dropna()

gate({"freeze_cells": 40, "freeze_traintail_fitted": 32,
      "sig_val_full": 35, "sig_val_ex_h1": 27, "sig_train_tail": 3,
      "vw_cells": 69, "vw_alt_cells": 57, "calm_higher": 37},
     {"freeze_cells": len(pf),
      "freeze_traintail_fitted": int(pf["train_tail"].notna().sum()),
      "sig_val_full": int(sig_by_win["val_full"]),
      "sig_val_ex_h1": int(sig_by_win["val_ex_h1"]),
      "sig_train_tail": int(sig_by_win["train_tail"]),
      "vw_cells": int(base.notna().sum()),
      "vw_alt_cells": int(pv["alt_2018_19"].notna().sum()),
      "calm_higher": int((dcc > 0).sum())})

C6 = pf.index.get_level_values("model") == "C6_llmtext"

# ------------------------------------------------------------------ canvas
apply_style(9)
# 6.35 in of canvas rather than 6.10: the inclusion scale falls from 1.083 to
# 1.038 (9 pt still sets 9.3 pt on the page), which buys the vertical room panel
# (b) needs for its four row annotations without making the float any taller.
fig = plt.figure(figsize=(6.35, 7.62))
ax_a = fig.add_axes([0.1297, 0.6628, 0.8483, 0.2462])
ax_b = fig.add_axes([0.2594, 0.3585, 0.7186, 0.2047])
ax_c = fig.add_axes([0.2594, 0.1151, 0.6658, 0.1629])

# --------------------------------------------------- (a) freeze-origin slope
WINS = ["train_tail", "val_ex_h1", "val_full"]
WLAB = ["2018-19\ntraining tail", "validation minus\nits COVID half",
        "committed full\nvalidation block"]
xs = np.arange(3)

ax_a.axhline(0, color=GREY, lw=0.6)
for i, (idx, row) in enumerate(pf.iterrows()):
    y = [row[w] for w in WINS]
    ok = ~np.isnan(y)
    col = GREEN if C6[i] else GREY
    ax_a.plot(xs[ok], np.asarray(y)[ok], color=col, lw=0.75 if C6[i] else 0.5,
              alpha=0.95 if C6[i] else 0.42, zorder=3 if C6[i] else 2,
              marker="o", ms=2.2 if C6[i] else 1.6, mfc=col, mec=col)

med = [pf[w].median() for w in WINS]
ax_a.plot(xs, med, color=VERM, lw=2.0, marker="D", ms=4.6, mfc=VERM,
          mec="white", mew=0.7, zorder=5)

ax_a.set_yscale("symlog", linthresh=5, linscale=0.9)
# Headroom, not data: the column annotations sit in the band above +12, which
# is empty of cells (the largest is +11.42, on the training-tail column).  The
# left spine is bounded to the tick range so the empty band reads as margin.
ax_a.set_ylim(-160, 85)
ax_a.set_yticks([-100, -30, -10, 0, 5, 10])
ax_a.set_yticklabels(["-100", "-30", "-10", "0", "+5", "+10"])
ax_a.spines["left"].set_bounds(-160, 12)
ax_a.set_xlim(-0.42, 2.62)
ax_a.set_xticks(xs)
ax_a.set_xticklabels(WLAB, fontsize=9, linespacing=1.15)
ax_a.set_ylabel("QLIKE gain from text (%)", fontsize=9)
ax_a.set_title("(a) move the rows the weights are fitted on; text forecasts "
               "and test rows fixed (40 cells)",
               fontsize=9, color=GREY, pad=5, loc="left")

for i, w in enumerate(WINS):
    # y in axes fractions, x in data: the two lines are pinned to the top of the
    # panel so no per-cell line can run through them, whatever the y-scale does.
    ax_a.text(i, 0.995, f"median {pf[w].median():+.2f}", fontsize=9,
              color=VERM_TXT, ha="center", va="top",
              transform=ax_a.get_xaxis_transform())
    ax_a.text(i, 0.917, f"helps in {int(sig_by_win[w])}/{int(n_by_win[w])}",
              fontsize=9, color=GREY, ha="center", va="top",
              transform=ax_a.get_xaxis_transform())

ax_a.legend(handles=[
    Line2D([], [], color=GREY, lw=0.5, alpha=0.6, marker="o", ms=1.8,
           label="one cell"),
    Line2D([], [], color=GREEN, lw=0.9, marker="o", ms=2.4,
           label="prompted-LLM cells (6)"),
    Line2D([], [], color=VERM, lw=2.0, marker="D", ms=4.2, mec="white",
           label="median of the 40")],
    loc="lower right", fontsize=9, handletextpad=0.5, labelspacing=0.3,
    borderpad=0.25, borderaxespad=0.4)

# ------------------------------------------- (b) validation-half contrasts
ROWS = [
    (dcc.values, "calm  -  COVID", 3, BLUE,
     f"mean {dcc.mean():+.2f} pp; {int((dcc > 0).sum())}/69 favour calm; "
     f"t = +1.12, p = .268"),
    ((pv["calm_2021"] - base).dropna().values, "calm  -  committed", 2,
     SKY, f"mean {(pv['calm_2021'] - base).mean():+.2f} pp, 69 cells"),
    ((pv["covid_2020"] - base).dropna().values, "COVID  -  committed", 1,
     PURPLE, f"mean {(pv['covid_2020'] - base).mean():+.2f} pp, 69 cells"),
    ((pv["alt_2018_19"] - base).dropna().values, "2018-19  -  committed",
     0, VERM, f"mean {(pv['alt_2018_19'] - base).mean():+.1f} pp, 57 cells "
              f"(in-sample, not a regime test)"),
]
# The zero rule spans the four strips only: at full axes height it ran through
# the top row's annotation, between the "p" and the "= .268".
ax_b.plot([0, 0], [-0.45, 3.28], color=GREY, lw=0.6, zorder=1)
for vals, lab, y, col, note in ROWS:
    jit = RNG.uniform(-0.13, 0.13, size=len(vals))
    ax_b.plot(vals, y + jit, ls="none", marker="o", ms=2.6, mfc=col, mec=col,
              alpha=0.55)
    ax_b.plot([np.median(vals)], [y], ls="none", marker="|", ms=11,
              mec=GREY, mew=1.6)
    # Centred in the clear band between its own strip and the one above, so it
    # crosses neither the markers nor the median bar; the colour ties it to the
    # strip below it.
    ax_b.text(-172, y + 0.49, note, fontsize=9, color=col, ha="left",
              va="center")

ax_b.set_xscale("symlog", linthresh=1, linscale=0.85)
ax_b.set_xlim(-180, 26)
ax_b.set_xticks([-100, -10, -1, 0, 1, 10])
ax_b.set_xticklabels(["-100", "-10", "-1", "0", "+1", "+10"])
ax_b.set_yticks([r[2] for r in ROWS])
ax_b.set_yticklabels([r[1] for r in ROWS], fontsize=9)
ax_b.set_ylim(-0.45, 3.80)
ax_b.set_xlabel("difference in the text increment, percentage points "
                "(vertical bar = median)", fontsize=9)
ax_b.set_title("(b) which half of the validation block the weights see  "
               "(69-cell grid)", fontsize=9, color=GREY, pad=5, loc="left")
ax_b.tick_params(axis="y", length=0)

# ------------------------------------------------------- (c) sign scoreboard
SB = [("drop the COVID half", 39, 40, BLUE),
      ("COVID half only", 54, 69, BLUE),
      ("calm half only", 35, 69, BLUE),
      ("2018-19 tail (freeze)", 6, 32, VERM),
      ("2018-19 tail (val-window)", 22, 57, VERM)]
ypos = np.arange(len(SB))[::-1]
for y, (lab, keep, tot, col) in zip(ypos, SB, strict=False):
    ax_c.barh(y, 100 * keep / tot, height=0.56, color=col, edgecolor="none")
    ax_c.barh(y, 100 * (tot - keep) / tot, left=100 * keep / tot, height=0.56,
              color=LIGHT, edgecolor="none")
    ax_c.text(101.5, y, f"{keep}/{tot}", fontsize=9, color=col, va="center",
              ha="left")
ax_c.set_yticks(ypos)
ax_c.set_yticklabels([s[0] for s in SB], fontsize=9)
ax_c.set_xlim(0, 118)
ax_c.set_xticks([0, 25, 50, 75, 100])
ax_c.set_xticklabels(["0", "25", "50", "75", "100%"])
ax_c.set_xlabel("cells keeping the sign they have under the committed fit",
                fontsize=9)
ax_c.set_title("(c) sign survival (orange = the in-sample diagnostic arms)",
               fontsize=9, color=GREY,
               pad=5, loc="left")
ax_c.tick_params(axis="y", length=0)
for s in ("left", "right", "top"):
    ax_c.spines[s].set_visible(False)

fig.text(0.1297, 0.9924,
         "Denominators differ: (a) is the 40 cells that already show a "
         "confirmed increment,",
         fontsize=9, color=GREY, va="top", ha="left")
fig.text(0.1297, 0.9713,
         "(b) the whole 69-cell grid, most of which is null. The 2018-19 "
         "training tail is a",
         fontsize=9, color=GREY, va="top", ha="left")
fig.text(0.1297, 0.9501,
         "diagnostic, not a regime test: those rows are in-sample for every "
         "text model.",
         fontsize=9, color=GREY, va="top", ha="left")

finish(fig, "stability_weight_window_sensitivity")
print("freeze medians:", {w: round(pf[w].median(), 3) for w in WINS})
print("freeze sig:", dict(sig_by_win), "of", dict(n_by_win))
print("calm-covid: mean", round(dcc.mean(), 3), "median", round(dcc.median(), 3),
      "calm higher", int((dcc > 0).sum()), "/", len(dcc))
