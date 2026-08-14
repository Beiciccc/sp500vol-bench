"""F9 -- What the null is powered to see.

Three panels: (a) injected-signal recovery per reference-ladder rung,
(b) each cell's real (no-injection) effect against its own minimum
detectable effect, (c) the close-to-close label's own estimator-noise
budget and the R-squared ceiling it implies.

Sources (every plotted number is read from these files at run time; nothing
is hardcoded outside the gate() expectation block)
------------------------------------------------------------------------
* results/tables/signal_injection_power.csv
    disc, model, h, target_pct, delta_negative, converged, mde_rel_pct,
    har_detect, firm_detect, pool_detect, all3_detect,
    har0_rel, har0_detect, firm0_detect, pool0_detect
* results/tables/label_noise_budget.csv
    h, n_windows, var_signal, var_noise, noise_share, r2_ceiling,
    overnight_share_of_label

Main-text sentences substantiated
---------------------------------
06_results.tex: "The conjunction is a strict selection device, not evidence
of absence: injected firm-orthogonal signals clear it in 2 of 69 cells at
0.3%, 13 of 69 at 1.0% (median 80%-power MDE 0.84--1.27% long-form, 0.44%
event-driven h=5)."
05_protocol.tex: "the protocol ships its own power calibration: an oracle
firm-orthogonal signal injected at known effect sizes measures each
control's recovery rate, and a per-cell minimum detectable effect
accompanies every null count."

Scope carried on the artefact: the injection is an oracle construction built
from test labels (the one declared exception to the no-look-ahead rule); it
measures sensitivity and is never citable as forecasting performance.
"""
import os
import re
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supp_style import (AGG, BLUE, GREEN, GREY, INK, LIGHT, PURPLE, REPO, SKY,
                        TAB, VERM, VERM_TXT, YELLOW, annot, apply_style, finish,
                        gate, note)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W, H = 6.5, 8.60                      # canvas inches (portrait supplement page)


def rect(x, y, w, h):
    """Axes rectangle from inches (origin bottom-left) to figure fractions."""
    return [x / W, y / H, w / W, h / H]


# Emphasis inside a wrapped note block.  A word is marked with em() while the
# note is written as plain prose; the mark is invisible to the wrap measure, and
# only after wrapping does ital() turn it into mathtext italics.  So emphasis
# can neither move a line break nor be split across two lines.
EM = "\x00"


def em(word):
    """Mark one word of a note for italic emphasis."""
    return EM + word + EM


def para(text, width):
    """Wrap a multi-paragraph string, keeping the blank line between paras."""
    out = []
    for p in text.split("\n\n"):
        lines = textwrap.wrap(p.replace(EM, ""), width)
        words = p.split()
        if EM in p:
            # re-attach the marks to the words the prose wrap has just laid
            # out; the wrap never splits a word, so the two sequences agree
            if len(words) != sum(len(ln.split()) for ln in lines):
                sys.exit("F9: emphasis marks lost the wrap alignment")
            it = iter(words)
            lines = [" ".join(next(it) for _ in ln.split()) for ln in lines]
        out.append("\n".join(lines))
    return "\n\n".join(out)


def ital(text):
    """Turn the emphasis marks of an already-wrapped note into italics."""
    return re.sub(EM + r"(\S+?)" + EM, lambda m: r"$\it{%s}$" % m.group(1),
                  text)


# ----------------------------------------------------------------- evidence
inj = pd.read_csv(os.path.join(TAB, "signal_injection_power.csv"))
lab = pd.read_csv(os.path.join(TAB, "label_noise_budget.csv")).sort_values("h")

LEVELS = [0.3, 0.5, 1.0]
# the delta=0 (real-data) columns are repeated on every level's rows, so any
# single level's slice carries exactly one row per cell
base = inj[inj.target_pct == LEVELS[0]].copy()
n_cells = len(base)

real = {"har": int(base.har0_detect.sum()),
        "firm": int(base.firm0_detect.sum()),
        "pool": int(base.pool0_detect.sum())}
real["all3"] = int((base.har0_detect & base.firm0_detect
                    & base.pool0_detect).sum())

rec = {k: [real[k]] for k in ("har", "firm", "pool", "all3")}
subtracted = []
for lv in LEVELS:
    s = inj[inj.target_pct == lv]
    rec["har"].append(int(s.har_detect.sum()))
    rec["firm"].append(int(s.firm_detect.sum()))
    rec["pool"].append(int(s.pool_detect.sum()))
    rec["all3"].append(int(s.all3_detect.sum()))
    subtracted.append(int(s.delta_negative.sum()))
n_conv = int(inj.converged.sum())

mde = base.mde_rel_pct.to_numpy()
eff = base.har0_rel.to_numpy()
det = base.har0_detect.to_numpy(dtype=bool)
is_lf = (base.disc == "long_form").to_numpy()
above = int((eff >= mde).sum())
below = int((eff < mde).sum())
det_below = int((eff[det] < mde[det]).sum())

h_lab = lab.h.to_numpy()
v_sig = lab.var_signal.to_numpy()
v_noi = lab.var_noise.to_numpy()
share = lab.noise_share.to_numpy()
ceil = lab.r2_ceiling.to_numpy()
wins = lab.n_windows.to_numpy()
overnight = lab.overnight_share_of_label.to_numpy()

# ------------------------------------------------------------------- gate
# The counts the frozen main text and the committed source tables state.
# Any drift aborts the build instead of silently re-rendering.
gate(
    {"n_cells": 69, "converged_pairs": 207,
     "rec_har": [38, 12, 20, 41], "rec_firm": [8, 7, 11, 20],
     "rec_pool": [9, 6, 12, 19], "rec_all3": [0, 2, 6, 13],
     "subtracted": [50, 47, 42],
     "mde_median": 0.823, "mde_q1": 0.372, "mde_q3": 1.268,
     "mde_min": 0.012, "mde_max": 3.647,
     "eff_min": -3.859, "eff_max": 5.920,
     "above_mde": 40, "below_mde": 29, "detected": 38, "detected_below": 4,
     "noise_share": [0.548, 0.438, 0.349],
     "r2_ceiling": [0.452, 0.562, 0.651],
     "windows": [100485, 49748, 24617]},
    {"n_cells": n_cells, "converged_pairs": n_conv,
     "rec_har": rec["har"], "rec_firm": rec["firm"],
     "rec_pool": rec["pool"], "rec_all3": rec["all3"],
     "subtracted": subtracted,
     "mde_median": round(float(np.median(mde)), 3),
     "mde_q1": round(float(np.quantile(mde, 0.25)), 3),
     "mde_q3": round(float(np.quantile(mde, 0.75)), 3),
     "mde_min": round(float(mde.min()), 3),
     "mde_max": round(float(mde.max()), 3),
     "eff_min": round(float(eff.min()), 3),
     "eff_max": round(float(eff.max()), 3),
     "above_mde": above, "below_mde": below,
     "detected": int(det.sum()), "detected_below": det_below,
     "noise_share": [round(float(x), 3) for x in share],
     "r2_ceiling": [round(float(x), 3) for x in ceil],
     "windows": [int(x) for x in wins]},
)

# ------------------------------------------------------------------ figure
apply_style(9)
fig = plt.figure(figsize=(W, H))

ax_a = fig.add_axes(rect(0.66, 6.95, 2.86, 1.45))
ax_b = fig.add_axes(rect(0.80, 3.71, 5.52, 1.89))
ax_c = fig.add_axes(rect(0.66, 1.10, 2.44, 1.42))

RCOL = 3.88          # inches: left edge of the right-hand note column
RWID = 43            # wrap width, characters, for that column

# Widths, in figure fractions, of the hairlines note() draws above each
# apparatus block.  They are chosen, not defaulted, because finish() writes with
# bbox_inches="tight": a rule that runs past the widest piece of type widens the
# page box, which forces a harder down-scale at inclusion and shrinks every
# glyph in the figure.  Each value below stops just inside the type it caps.
#
#   RRULE  the right-hand note column (RCOL -> 6.415 in), so note (a) and note
#          (c) are capped by one aligned pair of rules and read as one register
#   FRULE  the scope footnote, stopped left of the right-hand column so it
#          passes under nothing.
#
# Panel (b)'s note gets no rule at all, and that is a measurement rather than an
# oversight: its first line clears panel (b)'s x-axis label by 0.016 in, while
# note() sets its hairline 0.10 in above the text -- i.e. across that label.
# Rendered short of the label instead, the rule read as a stray dash prefixed to
# the axis label, which is worse than no rule; and the block cannot move down,
# because its last line already meets panel (c)'s title.  Ink alone carries it.
RRULE = 0.390
FRULE = 0.815

# ---------------------------------------------------- (a) recovery per rung
x = np.arange(4)
# The four rungs run within one count of each other in the 6-20 band (7 vs 6
# at 0.3%, 11 vs 12 at 0.5%, 20 vs 19 at 1.0%), which is about two points on
# this panel. No in-plot value label can be bound to its own series by
# position at that separation, so the sixteen counts are printed once, as a
# numeric strip carried by the key below the panel, and the panel itself
# carries only the four step curves.
series = [("Recalibrated-HAR rung", rec["har"], BLUE, "o"),
          ("Firm-identity rung", rec["firm"], GREEN, "s"),
          ("Maximal-pool rung", rec["pool"], PURPLE, "^"),
          ("Conjunction (all three)", rec["all3"], VERM, "D")]
for lbl, ys, col, mk in series:
    ax_a.step(x, ys, where="mid", color=col, lw=1.2, zorder=3)
    ax_a.plot(x, ys, ls="none", marker=mk, ms=4.8, color=col, mfc=col,
              mec="white", mew=0.6, zorder=4,
              label="%s   %s" % (lbl, " / ".join(str(v) for v in ys)))

ax_a.axvline(0.5, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
ax_a.set_xticks(x)
ax_a.set_xticklabels(["real\n(none)", "0.3", "0.5", "1.0"])
ax_a.set_xlim(-0.55, 3.55)
ax_a.set_ylim(-2.0, 45)
ax_a.set_yticks([0, 10, 20, 30, 40])
ax_a.set_ylabel("cells detected, of 69")
ax_a.set_xlabel("injected firm-orthogonal signal (realised rel-QLIKE, %)")
# below the axes: the key doubles as the value table, one row per rung, the
# four numbers in the column order of the panel above
leg = ax_a.legend(loc="upper left", bbox_to_anchor=(-0.115, -0.52), ncol=2,
                  handlelength=1.3, columnspacing=1.1, labelspacing=0.30,
                  borderpad=0.0, handletextpad=0.5, fontsize=9)
for txt, (_, _, col, _) in zip(leg.get_texts(), series):
    txt.set_color(col)
# The strip's header is apparatus, not data: it names the denominator and the
# column order of the sixteen counts printed in the key.  It drops to INK2 with
# the rest of the apparatus, and takes no rule -- panel (a)'s x-axis label sits
# 0.10 in above it, exactly where note() would put one.
note(fig, 0.33 / W, 6.20 / H,
     "cells detected, of 69, in the order real / 0.3 / 0.5 / 1.0:",
     rule=False, va="bottom")
# Panel marker and title are one string on one baseline, so panel() would only
# re-do what this call already does correctly -- and it sets 10 pt, which on a
# tight bounding box would enlarge the page and shrink every printed glyph.  The
# marker/title pair therefore keeps its own size and takes panel()'s ink: the
# titles are the only INK type outside the data, now that the notes are INK2.
fig.text(0.66 / W, 8.35 / H,
         "(a)  Recovery of a known injected signal, per rung",
         fontsize=9.6, color=INK, va="bottom", ha="left")

note_a = (
    f"The injection equalises the {em('realised')} effect at exactly the "
    f"target level in every cell, so signal is {em('subtracted')} "
    "(delta < 0) in "
    f"{subtracted[0]} / {subtracted[1]} / {subtracted[2]} of the "
    f"{n_cells} cells at 0.3 / 0.5 / 1.0%. Each curve is therefore a "
    "recovery rate at one fixed effect size, not a comparison of effect "
    f"sizes, and it does {em('not')} imply that the real effects exceed "
    "1%.\n\n"
    f"That is why the HAR rung falls from {rec['har'][0]} to "
    f"{rec['har'][1]} at 0.3%: in every cell whose real effect already "
    "exceeded 0.3%, signal was removed down to it."
)
# Apparatus block: INK2 under a hairline, so a reader can tell the basis
# statement from the argument without either one changing size.  The line
# spacing is restored to the 1.30 this block was laid out at -- note()'s 1.32
# would push the wrapped block a couple of points further down the page, and on
# a tight bounding box that is a real cost in printed point size.
t_a = note(fig, RCOL / W, 8.29 / H, ital(para(note_a, RWID)), width=RRULE)
t_a.set_linespacing(1.30)

# ---------------------------------- (b) real effect against per-cell MDE
lim_hi = 3.88
ax_b.plot([0, lim_hi], [0, lim_hi], color=GREY, lw=0.8, ls=(0, (4, 3)),
          zorder=2)
ax_b.axhline(0, color=GREY, lw=0.6, zorder=1)
for mask, mk in ((is_lf, "o"), (~is_lf, "^")):
    idx = np.where(mask)[0]
    for keep, fc, ec, mew in ((det[mask], BLUE, BLUE, 0.7),
                              (~det[mask], "none", SKY, 0.9)):
        sel = idx[keep]
        ax_b.plot(mde[sel], eff[sel], ls="none", marker=mk, ms=4.6,
                  mfc=fc, mec=ec, mew=mew, zorder=3)
low = np.where(det & (eff < mde))[0]          # detected yet under-powered
ax_b.plot(mde[low], eff[low], ls="none", marker="o", ms=10.0, mfc="none",
          mec=VERM, mew=1.0, zorder=5)

ax_b.set_xlim(-0.12, lim_hi)
ax_b.set_ylim(-4.7, 6.7)
ax_b.set_xlabel("per-cell minimum detectable effect at 80% power, 5% "
                "two-sided size (rel-QLIKE, %)")
ax_b.set_ylabel("real (no-injection) relative\nQLIKE improvement over "
                "$f_R$ (%)")
# An in-axes callout, so it gets a halo: it sits in the upper right of the cloud
# where the 45-degree reference line and the sparse high-MDE cells pass through
# it, and a halo keeps it readable there without moving it out of the panel and
# reflowing the canvas.
annot(ax_b, 2.98, 1.62, "45\u00b0 line:\neffect $=$ MDE",
      ha="left", va="top", linespacing=1.30)
shape_key = [Line2D([], [], ls="none", marker="o", ms=4.6, mfc=BLUE,
                    mec=BLUE, label="long-form 10-K/Q"),
             Line2D([], [], ls="none", marker="^", ms=4.6, mfc=BLUE,
                    mec=BLUE, label="event-driven 8-K")]
ax_b.legend(handles=shape_key, loc="upper left", handlelength=0.9,
            labelspacing=0.26, borderpad=0.0, handletextpad=0.5, fontsize=9)
fig.text(0.80 / W, 5.66 / H,
         "(b)  Every cell's observed effect against its own detectability "
         "threshold", fontsize=9.6, color=INK, va="bottom", ha="left")

note_b = (
    "Filled = Holm-detected at the recalibrated-HAR rung; hollow = not "
    "detected; ringed = detected yet below its own MDE. "
    f"{above} of {n_cells} cells sit at or above their own MDE on the "
    f"signed comparison, {below} below it; of the {int(det.sum())} "
    f"detected, {det_below} lie below their prospective MDE, which is an "
    "80%-power planning quantity from a HAC daily standard error, not a "
    "test. Negative effects are plotted, not clipped."
)
t_b = note(fig, 0.80 / W, 3.30 / H, para(note_b, 96), rule=False)
t_b.set_linespacing(1.30)

# --------------------------------------------- (c) label-noise budget
xc = np.arange(len(h_lab))
ax_c.bar(xc, v_sig, width=0.58, color=SKY, edgecolor="white", lw=0.6,
         label="signal", zorder=3)
ax_c.bar(xc, v_noi, width=0.58, bottom=v_sig, color=VERM, edgecolor="white",
         lw=0.6, hatch="///", label="estimator noise", zorder=3)
ax_c.set_xticks(xc)
# the noise share sits under its own bar rather than inside it: the h=20
# noise block is too thin to hold 9pt text without overprinting the hatch
ax_c.set_xticklabels([f"h={int(v)}\nnoise {100 * s:.1f}%"
                      for v, s in zip(h_lab, share)])
ax_c.set_xlim(-0.62, 2.55)
ax_c.set_ylim(0, 1.70)
ax_c.set_yticks([0.0, 0.4, 0.8, 1.2])
ax_c.set_ylabel("Var(log close-to-close RV)")
ax_c.legend(loc="upper left", handlelength=1.1, labelspacing=0.22,
            borderpad=0.0, handletextpad=0.5, fontsize=9)
fig.text(0.66 / W, 2.58 / H,
         "(c)  The label's own noise, and the R$^2$ it caps",
         fontsize=9.6, color=INK, va="bottom", ha="left")

ax_c2 = ax_c.twinx()
ax_c2.spines["top"].set_visible(False)
ax_c2.spines["right"].set_visible(True)
ax_c2.spines["right"].set_color(GREY)
ax_c2.plot(xc, ceil, color=GREY, lw=1.2, marker="D", ms=4.4, mfc="white",
           mec=GREY, mew=1.0, zorder=6)
for xi, c_ in zip(xc, ceil):
    ax_c2.annotate(f"{c_:.3f}", (xi, c_), textcoords="offset points",
                   xytext=(-8, 0), ha="right", va="center", fontsize=9,
                   color=GREY)
ax_c2.set_ylim(0, 0.75)
ax_c2.set_yticks([0.0, 0.2, 0.4, 0.6])
ax_c2.set_ylabel("implied R$^2$ ceiling", color=GREY)
ax_c2.tick_params(axis="y", colors=GREY)

note_c = (
    "Signal = the part of Var(log close-to-close RV) shared with a "
    "Garman\u2013Klass estimator of the same window; noise = the remainder. "
    f"Non-overlapping test-era windows: {wins[0]:,} (h=5), {wins[1]:,} "
    f"(h=10), {wins[2]:,} (h=20).\n\n"
    f"The identity prices the {em('intraday')} component only: about "
    f"{100 * overnight.mean():.0f}% of the label's variance is the "
    "overnight return, which Garman\u2013Klass cannot see and which is not "
    f"identified here, so this is a bounded {em('partial')} accounting."
)
# 2.44 -> 2.523: the block moves *up* 0.08 in inside the white space it already
# had (the right-hand column is empty from note (a)'s last line down to here),
# which is free on a tight bounding box, and it is what lets the scope footnote
# below carry a rule across the full width of its own type instead of stopping
# under this column's last line.
t_c = note(fig, RCOL / W, 2.523 / H, ital(para(note_c, RWID)), width=RRULE)
t_c.set_linespacing(1.30)

foot = (
    "Oracle injection: the synthetic signal is the within-firm-demeaned "
    "test log-residual of the recalibrated HAR reference \u2013 the one declared "
    "exception to the no-look-ahead rule. It calibrates sensitivity and may "
    f"never be read as forecasting performance. {n_conv} of {n_conv} "
    "(cell, level) pairs converged to within 0.02pp of their target."
)
# The scope statement is apparatus too, so it takes the same hairline as the two
# right-column notes -- but it keeps its vermillion rather than dropping to INK2.
# Vermillion is this figure's one warning hue, and what this block warns about
# (an oracle built from test labels, never citable as forecasting performance) is
# the single thing a reader must not mis-take from the panels above it.
t_foot = note(fig, 0.30 / W, 0.57 / H, para(foot, 99), width=FRULE)
t_foot.set_linespacing(1.30)
t_foot.set_color(VERM_TXT)

finish(fig, "F9_power_mde_label_noise")
