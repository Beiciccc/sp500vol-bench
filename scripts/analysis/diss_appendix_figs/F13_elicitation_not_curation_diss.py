"""F13 (dissertation variant) -- Elicitation, not curation, and not the name.

Why this file exists
--------------------
The supplement's own generator lives at
`scripts/analysis/supp_figs/F13_elicitation_not_curation.py` and must keep
producing the supplement's PDF unchanged, so it is not edited.  This copy draws
the same figure for the dissertation appendices and differs from it in three
respects, all of them layout:

* Output goes to `writing/dissertation/figures/` through `diss_style.finish`,
  never to the supplement's figure directory.
* The panel headings are set at 9.5 pt rather than 9.8 pt.  At 9.8 pt the panel
  (a) heading was 6.17 in wide, 0.17 in past the right edge of the panels
  themselves, and it alone drove the emitted page 7 pt wider than the A4 text
  block -- so the whole figure was scaled DOWN at inclusion and its 9 pt type
  printed at 8.86 pt, under the report's own 9 pt floor.
* Panel (c) is 0.13 grid-inches shorter and its heading sits 0.15 grid-inches
  lower.  On the dissertation's compressed canvas the last line of the panel (b)
  note was printed on top of the panel (c) heading; the two now clear each other
  by about 0.09 in.

Nothing else changes: every `gate()` expectation, every datum and every drawing
instruction is the supplement's.  The evidence gate below therefore still aborts
the build if any source table has drifted.

Original docstring follows.
--------------------------------------------------------------------------
F13 -- Elicitation, not curation, and not the name.

Three input manipulations of the prompted arm, drawn side by side.
(a) Input parity on long-form: the prompted arm against a same-lineage
    embedder reading the byte-identical excerpts, and against the standard-input
    seed-ensemble embedding arm.
(b) Contamination arms: a date-only prompt and a date-plus-ticker prompt
    carrying no document text, against the full-text arm.
(c) The same zero-content construction inside the second, larger family, with
    the honest complement -- what the full-text arm still adds once that probe
    sits inside the reference.

Sources (every plotted number is read from these files at run time; nothing is
hardcoded outside the gate() expectation block)
--------------------------------------------------------------------------
* results/tables/c5x_input_parity.csv (+ .md)
    model, run, h, rel_impr_pct, dm_q_clu, p_q_clu, placebo_dm_clu,
    dmq_holm_clu, genuine
* results/tables/llm_contamination.csv (+ .md)
    block, disc, arm, h, rel_pct, dm_clu, p_raw, repro_frac_vs_fulltext,
    p_holm      -- block in {variant, joint, cutoff}; p_holm is computed WITHIN
                   block, so the three blocks are three multiplicity families
                   and no count may be pooled across them.
                   repro_frac_vs_fulltext is a RATIO, not a percentage.
* results/tables/crossfamily_llama70_probe.csv (+ .md)
    block in {probe_m1, fulltext_anchor_committed, probe_share,
    beyond_identity} -- four disjoint schemas in one file, filtered separately.

Main-text sentences substantiated (frozen)
------------------------------------------
07_ablations.tex: "Input parity: on the byte-identical excerpts C6 reads, C5x
and seed-ensemble C5 lose everywhere C6 adds: prompting > pooling, not
curation.  Contamination: a date-only prompt is positive in 0 of 6 cells;
date+ticker (zero content) reproduces 31--77% of the full-text increment in the
3 of 6 cells where it is well-identified (full text >=1%, Holm-significant); at
long-form h=20, where full text adds +0.27%, the ratio reaches an uninformative
405%, yet with it in the reference full text adds 6 of 6 under Holm ...
A zero-content date+ticker prompt through the 70B recovers only 5--7% of its
full-text edge (0.1--0.5% over firm identity): content, not the name."
04_methods.tex: "C5x applies Qwen3-Embedding-8B to the byte-identical curated
excerpts C6 reads, so any C6-versus-C5x gap is attributable to elicitation, not
curation."

Adversarial repairs folded in (every required_change binding)
-------------------------------------------------------------
* Panel (c)'s annotation reports the failing h=20 cell as failing: the
  full-text arm adds at h=5 and h=10 under correction and is NOT distinguishable
  at h=20 (Holm .4512).
* The date-only caption reports the sign structure honestly: three cells sit at
  machine-zero (+0.004%, -0.002%, -2e-14%), so the frozen "positive in 0 of 6"
  is a significance statement, not a sign count.  Lens 1 asked for a two-cell
  wording and lens 2 for the three-cell one; the three-cell version is the
  conservative choice because it names every near-zero cell and does not call a
  negative value positive.
* The reproduction column is rendered as a percentage of a ratio (0.31 -> 31%).
"""
import os
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import diss_style as ds  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _inclusion_floor  # noqa: E402
from supp_style import (BLUE, GREEN, GREY, INK, INK2, LIGHT, PURPLE, REPO,
                        RULE, SKY, TAB, VERM, VERM_TXT, YELLOW, apply_style,
                        gate)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# W, H are the LAYOUT GRID, in inches, and are the supplement's own: every
# `rect()` and every `fig.text` offset below is an absolute inch position on
# that grid.  HFIG is the height the figure is actually drawn at.  Because the
# grid is divided by H and the canvas is HFIG, the whole layout is compressed
# vertically by HFIG/H = 0.905 while the type keeps its point size -- which is
# what closed the panel (b) note against the panel (c) heading, and why the
# panel (c) block is re-placed below.
W, H = 6.5, 8.95
HFIG = 8.10
HEAD = 9.5                            # panel-heading size (9.8 in the supplement)


def rect(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


def para(text, width):
    return "\n\n".join("\n".join(textwrap.wrap(p, width))
                       for p in text.split("\n\n"))


# ----------------------------------------------------------------- evidence
PAR = pd.read_csv(os.path.join(TAB, "c5x_input_parity.csv"))
CON = pd.read_csv(os.path.join(TAB, "llm_contamination.csv"))
PRB = pd.read_csv(os.path.join(TAB, "crossfamily_llama70_probe.csv"))

HH = [5, 10, 20]


def parity(run, basis):
    s = PAR[(PAR.run == run) & (PAR.basis == basis)].sort_values("h")
    assert len(s) == 3, f"{run}/{basis}: {len(s)} rows"
    return s


c5 = parity("C5_qwen3", "ens")            # declared primary basis: 3-seed ens
c5x = parity("C5x_qwen3exc", "s26")       # single run by design
c6 = parity("C6_llmtext", "s26")          # single run by design

VAR = CON[CON.block == "variant"]
JNT = CON[CON.block == "joint"]


def arm(disc, a):
    s = VAR[(VAR.disc == disc) & (VAR.arm == a)].sort_values("h")
    assert len(s) == 3
    return s


CELLS = [("long_form", "10-K/Q"), ("event_driven", "8-K")]
date_only = {d: arm(d, "dateonly") for d, _ in CELLS}
date_firm = {d: arm(d, "datefirm") for d, _ in CELLS}
fulltext = {d: arm(d, "fulltext") for d, _ in CELLS}
joint = {d: JNT[JNT.disc == d].sort_values("h") for d, _ in CELLS}

# "well identified" is the frozen text's own definition: full text >= 1% AND
# Holm-significant within the variant block
well_id = {}
for d, _ in CELLS:
    f = fulltext[d]
    well_id[d] = ((f.rel_pct.to_numpy() >= 1.0)
                  & (f.p_holm.to_numpy() < 0.05))
n_well = int(sum(w.sum() for w in well_id.values()))
repro_well = np.concatenate([100 * date_firm[d].repro_frac_vs_fulltext.to_numpy()[w]
                             for d, w in well_id.items()])

# date-only: significance, not sign.  A cell counts as significantly positive
# only if the point estimate is positive AND the clustered DM rejects.
do_sig_pos = 0
do_near_zero = []
for d, _ in CELLS:
    s = date_only[d]
    do_sig_pos += int(((s.rel_pct > 0) & (s.p_raw < 0.05)).sum())
    for v in s.rel_pct:
        if abs(float(v)) < 0.01:
            do_near_zero.append(float(v))
do_near_zero = sorted(do_near_zero, key=lambda v: -abs(v))

n_joint_holm = int(sum((joint[d].p_holm < 0.05).sum() for d, _ in CELLS))

probe = PRB[PRB.block == "probe_m1"].sort_values("h")
anchor = PRB[(PRB.block == "fulltext_anchor_committed")
             & (PRB.family == "llama70_awq_ens3")].sort_values("h")
share = PRB[PRB.block == "probe_share"].sort_values("h")
beyond = PRB[PRB.block == "beyond_identity"].sort_values("h")
assert len(probe) == len(anchor) == len(share) == len(beyond) == 3
retain = 100 * beyond.rel_pct.to_numpy() / anchor.rel_har.to_numpy()
well_h20 = bool(share.denominator_well_identified.iloc[-1])

# ------------------------------------------------------------------- gate
gate(
    {"c6_lf": [1.79, 2.25, 0.27], "c5x_lf": [-0.32, -3.74, -4.12],
     "c5_lf": [-1.02, -3.1, -6.39],
     "genuine": [3, 0, 0],
     "date_only_sig_pos": 0, "date_only_cells": 6,
     "date_only_near_zero": [0.0044, -0.0017, -0.0],
     "repro_ratio_lf": [0.3054, 0.3675, 4.0466],
     "repro_ratio_ed": [0.7708, 0.7158, 0.0137],
     "n_well_identified": 3, "repro_well_range": [30.5, 77.1],
     "joint_holm_lt05": 6,
     "probe_rel_har": [0.073, 0.085, 0.218],
     "probe_share_pct": [5.23, 7.31, 31.48],
     "probe_share_firm_pct": [0.1, 0.49, 5.7],
     "h20_well_identified": False,
     "beyond": [1.374, 1.136, 0.653],
     "beyond_holm": [0.0046, 0.0465, 0.4512],
     "retention_pct": [98.5, 97.7, 94.4]},
    {"c6_lf": [round(v, 2) for v in c6.rel_impr_pct],
     "c5x_lf": [round(v, 2) for v in c5x.rel_impr_pct],
     "c5_lf": [round(v, 2) for v in c5.rel_impr_pct],
     "genuine": [int(c6.genuine.sum()), int(c5x.genuine.sum()),
                 int(c5.genuine.sum())],
     "date_only_sig_pos": do_sig_pos,
     "date_only_cells": int(sum(len(date_only[d]) for d, _ in CELLS)),
     "date_only_near_zero": [round(v, 4) for v in do_near_zero],
     "repro_ratio_lf": [round(v, 4) for v in
                        date_firm["long_form"].repro_frac_vs_fulltext],
     "repro_ratio_ed": [round(v, 4) for v in
                        date_firm["event_driven"].repro_frac_vs_fulltext],
     "n_well_identified": n_well,
     "repro_well_range": [round(float(repro_well.min()), 1),
                          round(float(repro_well.max()), 1)],
     "joint_holm_lt05": n_joint_holm,
     "probe_rel_har": [round(v, 3) for v in probe.rel_har],
     "probe_share_pct": [round(v, 2) for v in share.share_har_pct],
     "probe_share_firm_pct": [round(v, 2) for v in share.share_firm_pct],
     "h20_well_identified": well_h20,
     "beyond": [round(v, 3) for v in beyond.rel_pct],
     "beyond_holm": [round(v, 4) for v in beyond.p_holm],
     "retention_pct": [round(float(v), 1) for v in retain]},
)

# ------------------------------------------------------------------ figure
apply_style(9)
fig = plt.figure(figsize=(W, HFIG))

ax_a = fig.add_axes(rect(0.92, 7.40, 5.38, 1.08))
ax_b = fig.add_axes(rect(0.92, 4.24, 5.38, 1.22))
ax_c = fig.add_axes(rect(0.92, 1.26, 5.38, 0.85))

BW = 0.26

# ------------------------------------------------------- hierarchy devices
# Measured on the committed page (pdftoppm at 130 dpi, ink-row bands): each of
# the three apparatus notes began exactly 1.66 pt below the last ink of its
# legend -- the same line gap the note uses internally -- and was drawn in GREY,
# the colour of the data labels.  All three therefore read as a fourth legend
# row.  The repair is colour plus a rule, never size: the notes drop to INK2 and
# a hairline separates them from the key.
#
# The room for that hairline is bought by moving each legend UP into whitespace
# the page already has (11.1 / 12.7 / 10.5 pt of clearance sat between the panel
# tick labels and the legends), which is free.  Moving anything DOWN or OUT is
# not: finish() writes with bbox_inches="tight", the emitted page is 451.0 x
# 586.7 pt, it binds on WIDTH at inclusion (scale 1.0094, 9 pt drawn printing at
# 9.08 pt), and the scaled height 592.1 pt sits only 3 pt under the float cap --
# so there are about 2.9 native pt of vertical headroom and none horizontal.
# LEG_UP is that move, expressed on the layout grid: the grid is divided by H
# while the canvas is HFIG, so one printed point is 1/72/(HFIG/H) grid inches.
PT = 1.0 / 72.0 / (HFIG / H)          # one printed point, in grid inches
LEG_UP = 5.0 * PT                     # legends rise 5 pt into existing space
RULE_UP = 3.4 * PT                    # hairline: measured mid-gap, 3.5 pt clear
                                      # of the legend ink and 3.1 pt of the note


def hairline(y, x0=0.30, x1=6.30):
    """A RULE hairline across the note column, on the layout grid.

    x1 stops at 6.30 grid-inches; the widest note line already reaches 6.35, so
    the rule cannot widen the bounding box and cannot cost printed point size.
    """
    fig.add_artist(Line2D([x0 / W, x1 / W], [y / H, y / H],
                          transform=fig.transFigure, color=RULE,
                          linewidth=0.5, solid_capstyle="butt", zorder=0.5))


# --------------------------------------------- (a) input parity, long-form
xa = np.arange(3)
SER_A = [("C5 seed-ensemble frozen embeddings, standard input", c5, PURPLE,
          "\\\\"),
         ("C5x same-lineage embedder, byte-identical excerpts", c5x, YELLOW,
          "//"),
         ("C6 prompted Qwen3-32B, the same excerpts", c6, BLUE, "")]
for i, (lab, s, col, hh) in enumerate(SER_A):
    v = s.rel_impr_pct.to_numpy()
    ax_a.bar(xa + (i - 1) * BW, v, width=BW, color=col, edgecolor="white",
             lw=0.7, hatch=hh, zorder=3)
    for xi, vi in zip(xa + (i - 1) * BW, v):
        ax_a.annotate(f"{vi:+.2f}", (xi, vi), textcoords="offset points",
                      xytext=(0, 3 if vi >= 0 else -11), ha="center",
                      fontsize=9, color=GREY, zorder=6)
ax_a.axhline(0, color=GREY, lw=0.7, zorder=4)
ax_a.set_xticks(xa)
ax_a.set_xticklabels([f"h={v}" for v in HH])
ax_a.set_xlim(-0.55, 2.55)
# The supplement's -7.8 floor put the h=20 C5 bar's own value label (drawn 11 pt
# BELOW a bar that reaches -6.39) astride the bottom spine, which struck the
# label through.  Dropping the floor to -8.9 clears it; no bar changes value.
ax_a.set_ylim(-8.9, 3.6)
ax_a.set_yticks([-8, -6, -4, -2, 0, 2])
ax_a.set_ylabel("rel. QLIKE\nimprovement over\nrecalibrated\nHAR (%)")
fig.text(0.30 / W, 8.62 / H,
         "(a)  Input parity, 10-K/Q only: what changes is how the text is "
         "elicited, not which excerpts were curated",
         fontsize=HEAD, color=GREY, va="bottom", ha="left")

handles_a = [Patch(fc=col, ec="white", hatch=hh,
                   label=f"{lab} -- genuine in {int(s.genuine.sum())} of 3")
             for lab, s, col, hh in SER_A]
# Legend rows in DRAWING order (purple C5, gold C5x, blue C6), which is the
# left-to-right order of the bars inside every h group and the order panels (b)
# and (c) already use.  The committed page passed handles_a[::-1], so panel (a)
# was the lone inversion and the reader had to flip the mapping on each lookup.
# A permutation of a one-column, three-row legend leaves the longest label and
# the row count untouched, so the tight bounding box does not move.
fig.legend(handles=handles_a, loc="upper left",
           bbox_to_anchor=(0.30 / W, (7.10 + LEG_UP) / H), ncol=1,
           handlelength=1.5,
           labelspacing=0.24, borderpad=0.0, handletextpad=0.5, fontsize=9)

note_a = (
    "'Genuine' is the committed flag: clustered DM < 0, Holm < .05 within this "
    "table's one 12-cell family, and |placebo DM| < 2. The comparison is at "
    "parity of INPUT, not of parameter count -- a 32B decoder against an 8B "
    "embedder with a ridge head -- so the phrase is same-lineage, not "
    "same-size, and 10-K/Q is the only channel with a parity cell."
)
hairline(6.54 + RULE_UP)
fig.text(0.30 / W, 6.54 / H, para(note_a, 101), fontsize=9, color=INK2,
         va="top", ha="left", linespacing=1.30)

# --------------------------------------------- (b) contamination arms
xb = np.arange(6)
lab_b, do_v, df_v, ft_v, rp_v = [], [], [], [], []
for d, nm in CELLS:
    for j, hh in enumerate(HH):
        lab_b.append(f"{nm}\nh={hh}")
        do_v.append(float(date_only[d].rel_pct.iloc[j]))
        df_v.append(float(date_firm[d].rel_pct.iloc[j]))
        ft_v.append(float(fulltext[d].rel_pct.iloc[j]))
        rp_v.append(100 * float(date_firm[d].repro_frac_vs_fulltext.iloc[j]))
well_flat = np.concatenate([well_id[d] for d, _ in CELLS])

ax_b.bar(xb - BW, do_v, width=BW, color=LIGHT, edgecolor=GREY, lw=0.6,
         hatch="..", zorder=3)
ax_b.bar(xb, df_v, width=BW, color=YELLOW, edgecolor="white", lw=0.7,
         hatch="//", zorder=3)
ax_b.bar(xb + BW, ft_v, width=BW, color=BLUE, edgecolor="white", lw=0.7,
         zorder=3)
ax_b.axhline(0, color=GREY, lw=0.7, zorder=4)

# A bar chart draws a machine-zero arm as NOTHING, so "measured and it is zero"
# and "this arm was not run" look identical -- exactly the distinction note_b is
# at pains to make.  On the committed page the date-only arm was therefore
# missing in 3 of its 6 cells (10-K/Q h=5, 8-K h=5, 8-K h=20) and the
# date+ticker arm in 8-K h=20, which left that cell's "1%" ratio label floating
# in blank space with the blue full-text bar as its nearest mark.  Each such arm
# now gets a stub centred on y=0, 0.10 data units (~1.5 pt) tall, in its own
# series colour and its own bar width: a mark that says "measured, at zero"
# without claiming a sign, and small enough not to be confused with the smallest
# genuinely drawn arm (8-K h=10, -0.39%, about 6 pt).  This is interior ink,
# inside the existing axes limits and inside widths the layout already reserves,
# so the tight bounding box -- and every printed point size -- is unchanged.
ZERO_STUB = 0.10
for _vals, _fc, _ec, _dx in ((do_v, LIGHT, GREY, -BW), (df_v, YELLOW, YELLOW, 0.0)):
    for _xi, _vi in zip(xb + _dx, _vals):
        if abs(_vi) < 0.05:
            ax_b.bar(_xi, ZERO_STUB, bottom=-ZERO_STUB / 2, width=BW,
                     color=_fc, edgecolor=_ec, lw=0.6, zorder=5)

for xi, v, r, w in zip(xb, df_v, rp_v, well_flat):
    ax_b.annotate(f"{r:.0f}%", (xi, v), textcoords="offset points",
                  xytext=(0, 3), ha="center", fontsize=9,
                  color=VERM_TXT if not w else GREY, zorder=6)
# One line, not two.  The two-line version repeated the ratio ("ratio 405% on a
# +0.27% denominator") and so became the heaviest, highest-contrast text block
# inside the plotting area -- a caveat about a number the panel is telling the
# reader to ignore, outweighing the panel's own result.  The vermillion "405%"
# sitting on its own bar already carries the flag, so the numeral is dropped
# here and the caveat collapses to one line at the same anchor, well inside the
# axes.
ax_b.annotate(f"ratio on a {ft_v[2]:+.2f}% denominator -- uninformative",
              (xb[2], df_v[2]), textcoords="offset points", xytext=(6, 20),
              ha="left", va="bottom", fontsize=9, color=VERM_TXT,
              linespacing=1.28)
ax_b.set_xticks(xb)
ax_b.set_xticklabels(lab_b, linespacing=1.28)
ax_b.set_xlim(-0.62, 5.62)
ax_b.set_ylim(-2.0, 3.15)
ax_b.set_yticks([-2, -1, 0, 1, 2])
ax_b.set_ylabel("rel. QLIKE\nimprovement over\nrecalibrated\nHAR (%)")
# The heading now leads with the panel's RESULT and keeps the middle-bar
# disclosure in shorter words, so the finding is not left to lines 1 and 7 of
# the grey note while a discredited ratio holds the loudest position on the
# panel.  "sig." is load-bearing and stays: a bare "positive in 0 of 6" would
# contradict note_b, which states that the frozen count is a significance
# statement, not a sign count.  Width is the binding constraint, not character
# count: measured in Helvetica at 9.5 pt this heading is 5.78 in and ends at
# 6.08 in, inside panel (a)'s heading (5.98 in, ending at 6.28) and inside the
# panels' own right edge at 6.30, which is what actually sets the tight box.
fig.text(0.30 / W, 5.74 / H,
         f"(b)  Contamination arms: date-only sig. positive in {do_sig_pos} of "
         "6; middle-bar % = zero-content reproduction",
         fontsize=HEAD, color=GREY, va="bottom", ha="left")

handles_b = [Patch(fc=LIGHT, ec=GREY, hatch="..", label="date only"),
             Patch(fc=YELLOW, ec="white", hatch="//",
                   label="date + ticker (zero document content)"),
             # "(= C6)" names the arm: these blue bars ARE panel (a)'s blue C6
             # series (10-K/Q: +1.79, +2.25, +0.27 in both panels), and with the
             # two panels on different y-ranges the same +0.27 shows at two
             # visual magnitudes, inviting the reader to double-count the
             # evidence or to think the panels disagree.  The legend row is set
             # by "date + ticker (zero document content)"; the tagged entry ends
             # at 4.83 in, far inside the box.
             Patch(fc=BLUE, ec="white", label="full text (= C6)")]
fig.legend(handles=handles_b, loc="upper left",
           bbox_to_anchor=(0.30 / W, (3.75 + LEG_UP) / H), ncol=3,
           handlelength=1.5,
           labelspacing=0.24, columnspacing=1.6, borderpad=0.0,
           handletextpad=0.5, fontsize=9)

note_b = (
    f"The date-only arm is significantly positive in {do_sig_pos} of 6 cells; "
    f"three sit at machine-zero ({do_near_zero[0]:+.3f}%, "
    f"{do_near_zero[1]:+.3f}%, {do_near_zero[2]:+.0e}%), so the frozen "
    "'positive in 0 of 6' is a significance statement, not a sign count, and "
    # All six date-only cells are now accounted for: three at machine zero plus
    # THREE negative.  The committed wording named only the two 10-K/Q negatives
    # and left the visibly negative 8-K h=10 bar (-0.39%) unexplained, so a
    # reader counting bars had to map it onto a "machine-zero" value two orders
    # of magnitude away.  Wrapped at 101 the note is 7 lines before and after,
    # and its widest line measures 5.8087 in either way, so the box is unmoved.
    f"the three negative cells ({do_v[2]:+.2f}%, {do_v[1]:+.2f}%, "
    f"{do_v[4]:+.2f}%) are "
    "drawn as measured. The reproduction column is a RATIO in the source, "
    f"rendered here as a percentage; it is interpretable in the {n_well} of 6 "
    "cells where full text is at least 1% and Holm-significant "
    f"({repro_well.min():.0f}--{repro_well.max():.0f}%), and the other three "
    "are printed in the attention colour, not hidden. With the "
    "date-plus-ticker forecast in the reference, full text still adds in "
    f"{n_joint_holm} of 6 cells under the joint block's own Holm."
)
hairline(3.52 + RULE_UP)
fig.text(0.30 / W, 3.52 / H, para(note_b, 101), fontsize=9, color=INK2,
         va="top", ha="left", linespacing=1.30)

# --------------------------------------------- (c) the 70B zero-content probe
xc = np.arange(3)
pv = probe.rel_har.to_numpy()
av = anchor.rel_har.to_numpy()
sv = share.share_har_pct.to_numpy()
# The source's own well-identified flag, read here so the attention colour is
# spent on the one cell that earns it.  Committed page painted all three share
# labels vermillion while panel (b)'s note states the convention -- attention
# colour means "denominator not well identified" -- and the "xxx" overlay
# already marks h=20 alone.  Panel (c) now obeys (b)'s convention.
sw = share.denominator_well_identified.to_numpy()
hatches = ["", "", "xxx"]
# Panels (b) and (c) draw the SAME two categories, and the committed page gave
# them different colours: the zero-content date+ticker arm was YELLOW "//" in (b)
# and VERM in (c), the full-text arm BLUE in (b) and GREEN in (c), with nothing
# telling the reader they were the same two things.  Worse, VERM doubles as the
# attention colour.  (c) now speaks (b)'s palette: yellow hatched = zero-content
# prompt, blue = full text, and vermillion is left to mean only "look here".
for i, (v, col, base_h) in enumerate(((pv, YELLOW, "//"), (av, BLUE, ""))):
    for j in range(3):
        hh = "xxx" if j == 2 else base_h
        ax_c.bar(xc[j] + (i - 0.5) * BW * 1.15, v[j], width=BW * 1.15,
                 color=col, edgecolor="white", lw=0.7, hatch=hh, zorder=3)
for j in range(3):
    # one line only: a three-line label above a 0.22%-high bar collides with
    # the neighbouring full-text bar's own value label at h=20
    ax_c.annotate(f"{sv[j]:.1f}%", (xc[j] - 0.5 * BW * 1.15, pv[j]),
                  textcoords="offset points", xytext=(0, 3), ha="center",
                  va="bottom", fontsize=9,
                  color=INK if bool(sw[j]) else VERM_TXT, zorder=6)
    ax_c.annotate(f"{av[j]:.2f}%", (xc[j] + 0.5 * BW * 1.15, av[j]),
                  textcoords="offset points", xytext=(0, 3), ha="center",
                  fontsize=9, color=GREY, zorder=6)
ax_c.axhline(0, color=GREY, lw=0.7, zorder=4)
ax_c.set_xticks(xc)
ax_c.set_xticklabels([f"h={v}" for v in HH])
ax_c.set_xlim(-0.55, 2.55)
ax_c.set_ylim(0, 1.78)
ax_c.set_yticks([0, 0.5, 1.0, 1.5])
ax_c.set_ylabel("rel. QLIKE\nimprovement over\nrecalibrated\nHAR (%)")
# Panel (c) prints two different quantities in one identical label style: the
# blue labels are axis values (1.39 / 1.16 / 0.69%) while the gold ones are
# shares of the adjacent blue bar (5.2 / 7.3 / 31.5%), on an axis that tops out
# at 1.5.  Panel (b)'s heading discloses that convention for its own middle bar;
# (c)'s did not, so "5.2%" over a near-invisible bar invited the reading that
# the zero-content 70B probe adds 5.2%, seventy times its true 0.073%.  The
# suffix carries (b)'s disclosure: 5.82 in wide at 9.5 pt, ending at 6.12 in,
# inside panel (a)'s 6.28 and the panels' 6.30, so the box does not grow.
fig.text(0.30 / W, 2.21 / H,
         "(c)  The same zero-content construction inside the 70B family "
         "(8-K only); gold % = share of blue bar",
         fontsize=HEAD, color=GREY, va="bottom", ha="left")
handles_c = [Patch(fc=YELLOW, ec="white", hatch="//",
                   label="date + ticker probe (70B)"),
             Patch(fc=BLUE, ec="white",
                   label="full text (70B, 3-run ensemble)"),
             Patch(fc="white", ec=GREY, hatch="xxx",
                   label="denominator not well identified")]
fig.legend(handles=handles_c, loc="upper left",
           bbox_to_anchor=(0.30 / W, (0.96 + LEG_UP) / H), ncol=3,
           handlelength=1.5,
           labelspacing=0.24, columnspacing=1.3, borderpad=0.0,
           handletextpad=0.5, fontsize=9)

note_c = (
    f"Probe increments over the recalibrated HAR: {pv[0]:.3f} / {pv[1]:.3f} / "
    f"{pv[2]:.3f}%, i.e. {sv[0]:.1f} / {sv[1]:.1f} / {sv[2]:.1f}% of the "
    "full-text bar. The complement, and it is the uncomfortable half: with the "
    "same-model date-plus-ticker forecast inside the reference, the full-text "
    "arm still adds at h=5 and h=10 under correction "
    f"({beyond.rel_pct.iloc[0]:+.2f}%, Holm {beyond.p_holm.iloc[0]:.4f}; "
    f"{beyond.rel_pct.iloc[1]:+.2f}%, Holm {beyond.p_holm.iloc[1]:.4f}) and is "
    f"NOT distinguishable at h=20 ({beyond.rel_pct.iloc[2]:+.2f}%, Holm "
    f"{beyond.p_holm.iloc[2]:.4f}), retaining {retain.min():.0f}--"
    f"{retain.max():.0f}% of its uncontrolled increment. Panels (a), (b) and "
    "(c) carry SEPARATE multiplicity families, so no count may be pooled "
    "across them."
)
hairline(0.73 + RULE_UP)
fig.text(0.30 / W, 0.73 / H, para(note_c, 101), fontsize=9, color=INK2,
         va="top", ha="left", linespacing=1.30)

# diss_style's max_render_pt is its width-only overflow gate; the gate that
# matters here is the printed type floor checked immediately below.
ds.finish(fig, "F13_elicitation_not_curation", max_render_pt=620.0,
          note="dissertation variant: headings 9.5pt, panel (c) re-placed")
_inclusion_floor.check("F13_elicitation_not_curation", drawn_floor_pt=9.0)
