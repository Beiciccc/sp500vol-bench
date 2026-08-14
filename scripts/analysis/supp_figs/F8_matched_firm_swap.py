"""F8 -- Swapping the document: what survives when the firm level is preserved
but the document-firm correspondence is destroyed.

Substantiates, and bounds the scope of, this frozen main-text sentence
(07_ablations.tex, Stress Tests):
  "Matched-firm swap: swapping documents between within-day firm pairs matched on
   validation RV (level preserved, correspondence destroyed) kills 84-93% of the
   residual in point estimate (content, not identity), yet retains a median 31% of
   the 38 genuine increments. Swap and anonymisation price different channels:
   masking strips identity strings (share 0.51); the swap preserves the level
   channel and retains a median 0.29, so the 0.71 it destroys is document-firm
   alignment, text calibrated to the firm's level without naming it."

Evidence files read (every plotted number comes from one of these):
  results/tables/matched_firm_swap.csv -- real_rel_pct, swap_rel_pct, genuine,
      swap_frac_test, retention_vs_real, wd_random_rel_pct, firmref_real_rel_pct,
      firmref_swap_rel_pct, firmref_swap_dm, firmref_swap_p, firmref_retention
  results/tables/matched_firm_swap.md  -- the committed headline medians
  results/tables/anon_arm.csv          -- the anonymisation arm's OWN swap-retention
      column, read only to prove on-figure that its 0.29 is a different median over
      a different denominator from this table's 0.31 (the two must never be merged)

  results/tables/firm_identity_ensemble.csv -- the firm-identity rung the frozen
      main text commits to (+0.52/+0.24/+0.21% on n = 23,855/22,785/22,318), read
      so panel (c) can print the row basis that separates it from the swap table's
      own row set (n = 25,109/25,001/24,732) instead of leaving the reader to
      collide the two triples
  results/tables/control_intersection_ensemble.csv -- primary_genuine, the
      seed-ensemble placebo-confirmed flag, read only to measure how far it
      diverges from the single-seed flag drawn here (both sum to 38; they agree on
      53 of 69 cells)

Loss convention: relative QLIKE improvement in per cent, volatility units, the
project's primary single-seed grid basis; `genuine` is that grid's placebo-confirmed
flag (38 of 69 cells).  05_protocol.tex fixes the counting convention -- untagged
counts are seed-ensemble -- so every drawn quantity that rests on the single-seed
flag is tagged as such on the artefact itself.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from supp_style import (
    BLUE,
    GREEN,
    GREY,
    INK2,
    PURPLE,
    TAB,
    VERM,
    VERM_TXT,
    apply_style,
    finish,
    gate,
)

# --------------------------------------------------------------- load evidence
D = pd.read_csv(os.path.join(TAB, "matched_firm_swap.csv"))
ARM = pd.read_csv(os.path.join(TAB, "anon_arm.csv"))
FID = pd.read_csv(os.path.join(TAB, "firm_identity_ensemble.csv"))
ENS = pd.read_csv(os.path.join(TAB, "control_intersection_ensemble.csv"))

GEN = D[D.genuine]
RET_ALL = float(GEN.retention_vs_real.median())
RET_LF = float(GEN[GEN.disc == "long_form"].retention_vs_real.median())
RET_ED = float(GEN[GEN.disc == "event_driven"].retention_vs_real.median())
COV_LO, COV_HI = float(D.swap_frac_test.min()), float(D.swap_frac_test.max())
WD_MED, WD_MEAN = float(D.wd_random_rel_pct.median()), float(D.wd_random_rel_pct.mean())
FR = D[D.firmref_real_rel_pct.notna()].sort_values("h").reset_index(drop=True)
ANON_RET = float(ARM.swap_retention.median())      # the OTHER median: 0.29

# The committed firm-identity residual and ITS row set.  Panel (c) draws the same
# three cells on the swap table's rows, which are more numerous, so both row counts
# are printed on the artefact and both triples are gated.
CFID = (FID[(FID.disc == "event_driven") & (FID.model == "C6_llmtext")]
        .sort_values("h").reset_index(drop=True))
COMMITTED = tuple(float(v) for v in CFID.rel_impr_pct_firm)
COMMITTED_N = tuple(int(v) for v in CFID.n_test)
PANEL_C_N = tuple(int(v) for v in FR.n_test)

# The seed-ensemble placebo-confirmed flag, for the basis divergence printed with
# the medians: both flags sum to 38, and they select the same cell in 53 of 69.
M = D.merge(ENS[["disc", "model", "h", "primary_genuine"]],
            on=["disc", "model", "h"], how="left", validate="1:1")
AGREE = int((M.genuine == M.primary_genuine).sum())
EGEN = M[M.primary_genuine]
ERET_ALL = float(EGEN.retention_vs_real.median())
ERET_LF = float(EGEN[EGEN.disc == "long_form"].retention_vs_real.median())
ERET_ED = float(EGEN[EGEN.disc == "event_driven"].retention_vs_real.median())
EN_LF = int((EGEN.disc == "long_form").sum())
EN_ED = int((EGEN.disc == "event_driven").sum())

# ------------------------------------------------------------------------ gate
# The literal side of gate() is the only place a number is typed by hand; it exists
# to abort the build the moment the committed table stops saying what the frozen
# main text says.
gate(
    {
        "n_cells": 69, "n_genuine": 38,
        "n_genuine_lf": 24, "n_genuine_ed": 14,
        "retention_median_all": 0.31, "retention_median_lf": 0.33,
        "retention_median_ed": 0.17,
        "coverage_lo": 0.940, "coverage_hi": 0.945,
        "firmref_real": (0.448, 0.253, 0.196),
        "firmref_swap": (0.051, 0.016, 0.031),
        "firmref_dm": (-2.26, -0.76, -1.56),
        "firmref_p": (0.024, 0.447, 0.119),
        "firmref_killed_pct": (89, 93, 84),
        "wd_random_median": -0.077, "wd_random_mean": -0.574,
        "anon_arm_swap_retention_median": 0.29,
        # panel (c)'s row basis, and the committed triple it must not be read as
        "panel_c_n_test": (25109, 25001, 24732),
        "committed_firm_residual": (0.52, 0.24, 0.21),
        "committed_firm_n_test": (23855, 22785, 22318),
        # basis divergence: the two placebo-confirmed flags both sum to 38 but
        # select the same cell in only 53 of 69
        "ens_genuine": 38, "flags_agree": 53,
        "ens_retention_median_all": 0.33,
        "ens_retention_median_lf": 0.33, "ens_retention_median_ed": 0.21,
        "ens_n_lf": 30, "ens_n_ed": 8,
    },
    {
        "n_cells": len(D), "n_genuine": int(D.genuine.sum()),
        "n_genuine_lf": int((GEN.disc == "long_form").sum()),
        "n_genuine_ed": int((GEN.disc == "event_driven").sum()),
        "retention_median_all": round(RET_ALL, 2),
        "retention_median_lf": round(RET_LF, 2),
        "retention_median_ed": round(RET_ED, 2),
        "coverage_lo": round(COV_LO, 3), "coverage_hi": round(COV_HI, 3),
        "firmref_real": tuple(round(float(v), 3) for v in FR.firmref_real_rel_pct),
        "firmref_swap": tuple(round(float(v), 3) for v in FR.firmref_swap_rel_pct),
        "firmref_dm": tuple(round(float(v), 2) for v in FR.firmref_swap_dm),
        "firmref_p": tuple(round(float(v), 3) for v in FR.firmref_swap_p),
        "firmref_killed_pct": tuple(int(round(100 * (1 - float(v))))
                                    for v in FR.firmref_retention),
        "wd_random_median": round(WD_MED, 3), "wd_random_mean": round(WD_MEAN, 3),
        "anon_arm_swap_retention_median": round(ANON_RET, 2),
        "panel_c_n_test": PANEL_C_N,
        "committed_firm_residual": tuple(round(v, 2) for v in COMMITTED),
        "committed_firm_n_test": COMMITTED_N,
        "ens_genuine": int(M.primary_genuine.sum()), "flags_agree": AGREE,
        "ens_retention_median_all": round(ERET_ALL, 2),
        "ens_retention_median_lf": round(ERET_LF, 2),
        "ens_retention_median_ed": round(ERET_ED, 2),
        "ens_n_lf": EN_LF, "ens_n_ed": EN_ED,
    },
)
# This table's 0.31 and the anonymisation arm's 0.29 are different medians over
# different denominators; assert they cannot be silently collapsed into one number.
assert round(RET_ALL, 2) != round(ANON_RET, 2)

# ------------------------------------------------------------------------ plot
apply_style(base_size=9)
STYLE = {"long_form": dict(marker="o", color=BLUE, label="long-form"),
         "event_driven": dict(marker="^", color=VERM, label="event-driven")}
ZX, ZY = (-0.62, 2.62), (-0.82, 1.62)          # zoom window drawn in panel (b)

fig = plt.figure(figsize=(6.4, 6.15))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.70],
              left=0.088, right=0.995, top=0.868, bottom=0.055,
              hspace=0.52, wspace=0.24)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[1, 0])
axd = fig.add_subplot(gs[1, 1])


def draw_cloud(ax, xlim, ylim, size=26):
    """Scatter the swap cells; fill marks the placebo-confirmed set."""
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], lw=0.9, color=GREY, zorder=1)
    ax.axhline(0.0, lw=0.9, color=GREY, ls=(0, (5, 2)), zorder=1)
    ax.plot([lo, hi], [RET_ALL * lo, RET_ALL * hi], lw=1.0, color=GREEN,
            ls=(0, (1, 1.6)), zorder=1)
    for disc, st in STYLE.items():
        for gen in (True, False):
            sub = D[(D.disc == disc) & (D.genuine == gen)]
            ax.scatter(sub.real_rel_pct, sub.swap_rel_pct, s=size,
                       marker=st["marker"],
                       facecolor=(st["color"] if gen else "none"),
                       edgecolor=st["color"], linewidths=0.9, zorder=3,
                       alpha=0.95)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


# ------------------------------------------------- panel (a): all 69, no cropping
draw_cloud(axa, (-10.4, 6.8), (-8.2, 3.2), size=22)
axa.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0],
                        fill=False, edgecolor=GREY, lw=0.8, ls=(0, (2, 2)),
                        zorder=4))
axa.text(ZX[1] + 0.30, ZY[0] - 0.15, "(b)", fontsize=9, color=GREY,
         ha="left", va="top")
axa.text(-10.1, 3.05, "$y=x$: the increment survives the swap\n"
                      "(firm level / identity shortcut)", fontsize=8.4,
         color=GREY, ha="left", va="top", linespacing=1.35)
axa.text(6.6, -2.00, "$y=0$: the increment dies\n(document content)",
         fontsize=8.4, color=GREY, ha="right", va="top", linespacing=1.35)
axa.text(-9.6, -3.05, f"locus of retention {RET_ALL:.2f}", fontsize=8.4,
         color=GREEN, ha="left", va="top")
axa.set_xlabel("real relative QLIKE improvement, %", fontsize=9)
axa.set_ylabel("after a within-day, validation-RV-matched\ndocument swap, %",
               fontsize=9, linespacing=1.35)
axa.text(0.0, 1.10, "(a)", transform=axa.transAxes, fontsize=10,
         fontweight="bold", color=GREY, ha="left", va="bottom")
axa.text(0.075, 1.10,
         f"All {len(D)} cells, nothing cropped; a matched partner exists\n"
         f"for {100 * COV_LO:.1f}-{100 * COV_HI:.1f}% of test rows",
         transform=axa.transAxes, fontsize=8.6, color=INK2, ha="left",
         va="bottom", linespacing=1.35)

handles = []
for disc, st in STYLE.items():
    n_gen = int(((D.disc == disc) & D.genuine).sum())
    n_oth = int(((D.disc == disc) & ~D.genuine).sum())
    handles.append(Line2D([], [], ls="none", marker=st["marker"], ms=5,
                          color=st["color"],
                          label=f"{st['label']}, confirmed ({n_gen})"))
    handles.append(Line2D([], [], ls="none", marker=st["marker"], ms=5,
                          mfc="none", mec=st["color"],
                          label=f"{st['label']}, not ({n_oth})"))
leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.082),
                 ncol=2, fontsize=8.4, handletextpad=0.3, columnspacing=1.6,
                 borderpad=0.2, frameon=False,
                 title="fill: placebo-confirmed on the single-seed combination "
                       f"grid ({int(D.genuine.sum())} of {len(D)} cells)")
leg.get_title().set_fontsize(8.4)
leg.get_title().set_color(GREY)


# ------------------------------------------ panel (b): the core, same encodings
draw_cloud(axb, ZX, ZY, size=30)
n_in = int((D.real_rel_pct.between(*ZX) & D.swap_rel_pct.between(*ZY)).sum())
axb.set_xlabel("real relative QLIKE improvement, %", fontsize=9)
axb.text(0.0, 1.10, "(b)", transform=axb.transAxes, fontsize=10,
         fontweight="bold", color=GREY, ha="left", va="bottom")
axb.text(0.075, 1.10,
         f"The crowded core of (a): {n_in} of {len(D)} cells;\n"
         "the other cells are visible in (a)",
         transform=axb.transAxes, fontsize=8.6, color=INK2, ha="left",
         va="bottom", linespacing=1.35)
axb.text(ZX[0] + 0.08, ZY[1] - 0.04,
         "median retention, placebo-confirmed cells,\n"
         "single-seed combination grid basis\n"
         f"all {int(D.genuine.sum())}: {RET_ALL:.2f}      "
         f"long-form {int((GEN.disc == 'long_form').sum())}: {RET_LF:.2f}\n"
         f"event-driven {int((GEN.disc == 'event_driven').sum())}: {RET_ED:.2f}",
         fontsize=8.2, color=GREY, ha="left", va="top", linespacing=1.45)
axb.text(ZX[0] + 0.08, ZY[1] - 0.60,
         f"dotted: locus of retention {RET_ALL:.2f}", fontsize=8.2, color=GREEN,
         ha="left", va="top")
axb.text(ZX[0] + 0.04, -0.66, "$y=x$", fontsize=8.4, color=GREY, ha="left",
         va="top")


# ------------- panel (c): the 8-K residual under the firm-identity reference ---
axc.axvline(0.0, lw=0.9, color=GREY, ls=(0, (5, 2)), zorder=1)
for i, r in FR.iterrows():
    y = -i
    filled = bool(r.genuine)
    axc.plot([float(r.firmref_swap_rel_pct), float(r.firmref_real_rel_pct)],
             [y, y], lw=1.3, color=VERM, zorder=2)
    axc.plot([float(r.firmref_real_rel_pct)], [y], marker="^", ms=6.5,
             color=(VERM if filled else "none"), mec=VERM, mew=1.0, zorder=3)
    axc.plot([float(r.firmref_swap_rel_pct)], [y], marker="^", ms=6.5,
             color="white", mec=VERM, mew=1.0, zorder=3)
    pstr = f"{float(r.firmref_swap_p):.3f}".lstrip("0")   # ".024", ".447", ".119"
    axc.text(0.60, y,
             f"$-${100 * (1 - float(r.firmref_retention)):.0f}%    "
             f"DM ${float(r.firmref_swap_dm):+.2f}$,  $p$ $=$ {pstr}",
             fontsize=8.4, color=GREY, ha="left", va="center")
axc.set_yticks([-i for i in range(len(FR))])
axc.set_yticklabels([f"$h={int(v)}$" for v in FR.h], fontsize=9)
axc.set_ylim(-len(FR) + 0.35, 0.95)
axc.set_xlim(-0.06, 1.62)
axc.set_xticks([0.0, 0.2, 0.4])
axc.set_xlabel("relative QLIKE improvement over the firm-identity reference, %",
               fontsize=8.6)
axc.text(float(FR.firmref_real_rel_pct.iloc[0]), 0.30, "real", fontsize=8.4,
         color=VERM_TXT, ha="center", va="bottom")
axc.text(float(FR.firmref_swap_rel_pct.iloc[0]), 0.30, "after swap", fontsize=8.4,
         color=VERM_TXT, ha="center", va="bottom")
axc.text(0.0, 1.16, "(c)", transform=axc.transAxes, fontsize=10,
         fontweight="bold", color=GREY, ha="left", va="bottom")
axc.text(0.085, 1.16,
         "The surviving prompted 8-K cells, measured against\n"
         "the firm-identity reference (not the (a)/(b) reference)",
         transform=axc.transAxes, fontsize=8.6, color=INK2, ha="left",
         va="bottom", linespacing=1.35)
axc.text(0.0, -0.40,
         f"$h{{=}}20$ is drawn hollow: it is not among the "
         f"{int(D.genuine.sum())}\nplacebo-confirmed cells of the primary "
         "vs-HAR screen.\n\n"
         # The swap table computes the firm-identity residual on more rows than the
         # firm-identity join does, so the two triples differ.  Both row sets are
         # printed here: a reader holding the artefact beside the paper must be able
         # to see that these are the same cells on a different basis, not a revision.
         "Row basis. Panel (c) is the swap table's own\n"
         f"row set, n $=$ {PANEL_C_N[0]:,} / {PANEL_C_N[1]:,} / "
         f"{PANEL_C_N[2]:,} at $h{{=}}5/10/20$.\n"
         "The committed firm-identity residual is\n"
         f"${COMMITTED[0]:+.2f}/{{+}}{COMMITTED[1]:.2f}/{{+}}{COMMITTED[2]:.2f}"
         "\\%$ on the firm-identity\n"
         f"join's rows ({COMMITTED_N[0]:,} / {COMMITTED_N[1]:,} / "
         f"{COMMITTED_N[2]:,}): the same\nthree cells on a different row set, "
         "not a\ndifferent result.",
         transform=axc.transAxes, fontsize=8.2, color=GREY, ha="left", va="top",
         linespacing=1.4)

# -------------------- panel (d): within-date random permutation, triangulation -
rng = np.random.default_rng(2026)
jit = rng.uniform(-0.32, 0.32, len(D))
axd.axvline(0.0, lw=0.9, color=GREY, zorder=1)
for disc, st in STYLE.items():
    m = (D.disc == disc).values
    axd.scatter(D.wd_random_rel_pct[m], jit[m], s=20, marker=st["marker"],
                facecolor="none", edgecolor=st["color"], linewidths=0.9, zorder=3)
axd.axvline(WD_MED, lw=1.2, color=GREEN, ls=(0, (4, 2)), zorder=2)
axd.axvline(WD_MEAN, lw=1.2, color=PURPLE, ls=(0, (1, 1.6)), zorder=2)
axd.set_ylim(-1.15, 1.30)
axd.set_yticks([])
axd.set_xlim(-8.4, 2.6)
axd.set_xlabel("relative QLIKE improvement after permutation, %", fontsize=8.6)
axd.text(WD_MED + 0.15, 0.62, f"median {WD_MED:.2f}", fontsize=8.2, color=GREEN,
         ha="left", va="bottom")
axd.text(WD_MEAN - 0.20, -0.62, f"mean {WD_MEAN:.2f}", fontsize=8.2, color=PURPLE,
         ha="right", va="top")
axd.spines["left"].set_visible(False)
axd.text(0.0, 1.16, "(d)", transform=axd.transAxes, fontsize=10,
         fontweight="bold", color=GREY, ha="left", va="bottom")
axd.text(0.085, 1.16,
         "Triangulation control: an unmatched\n"
         f"within-date random permutation, {len(D)} cells",
         transform=axd.transAxes, fontsize=8.6, color=INK2, ha="left",
         va="bottom", linespacing=1.35)
axd.text(0.0, -0.40,
         f"This table's median retention {RET_ALL:.2f} is over\n"
         f"its {int(D.genuine.sum())} placebo-confirmed cells; the "
         f"anonymisation\narm's {ANON_RET:.2f} is a different median over its "
         "6 cells.\n\n"
         # The flag drawn here is the single-seed grid's; the paper's untagged
         # counts are seed-ensemble.  Both sum to 38 but they are not the same 38,
         # so the ensemble medians are printed rather than left to be assumed.
         "Basis. The confirmed flag drawn here is the\n"
         "single-seed grid's. The seed-ensemble primary\n"
         f"flag also sums to {int(M.primary_genuine.sum())} of {len(D)}, but the "
         f"two agree on\n{AGREE} of {len(D)} cells; on that set the same table\n"
         f"gives median {ERET_ALL:.2f} over {EN_LF} long-form ({ERET_LF:.2f}) "
         f"and\n{EN_ED} event-driven ({ERET_ED:.2f}).",
         transform=axd.transAxes, fontsize=8.2, color=GREY, ha="left", va="top",
         linespacing=1.4)

finish(fig, "F8_matched_firm_swap")
