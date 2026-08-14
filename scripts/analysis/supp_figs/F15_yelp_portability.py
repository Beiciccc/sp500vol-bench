"""F15 -- Portability I: the Yelp panel, where the identity control is itself costly.

Substantiates, and bounds the scope of, these frozen main-text sentences:
  00_abstract.tex: "The audit travels, and reads *oppositely* across panels ...
    while on Yelp the same control is itself costly and a content residual
    survives."
  08_discussion.tex: "The identity control reads *oppositely* across domains: it
    helps the SEC reference but costs 9.5--10.6% on Yelp, whose baseline already
    encodes the entity ... Two fitted arms agree on Yelp's surviving content
    residual (+0.37% TF-IDF; +0.38% frozen-embedding; h=1 month, placebo-clean),
    far above the 0.18% MDE and orders below the published +28--37%; its
    prompted Llama-70B arm is fully recovered by its own zero-content probe."
  01_intro.tex: "on Yelp the same control is itself *costly* (9.5--10.6%) while
    a content residual survives (+0.37/+0.38%); the shortcut's size is a
    property of the panel, not a constant".

Evidence files read (every plotted number comes from one of these):
  results/tables/yelp_cascade.csv  -- row, arm, reference, h, mse,
      delta_rel_pct, dm_p (the five ladder rungs at both horizons)
  results/tables/yelp_cascade.md   -- the minimum-detectable-effect line (AR and
      entity stage, per horizon), the G1 panel-shape gate (entities, events),
      the G4 label-shuffle gate, and the G4b within-month text-swap flag that
      makes the three-month residual borderline
  results/tables/yelp_multiarm.csv -- arm, h, text_alone, combiner, residual,
      res_dm, res_p2way, placebo_dm, placebo_p (four arms x two horizons)
  results/tables/yelp_multiarm.md  -- the panel description (entities,
      business-months, period, chronological split)

Loss convention: squared error on stars (MSE), month-clustered Diebold-Mariano
with a business x month two-way robustness variant; every percentage is a
relative change in MSE against the reference named on the artefact itself,
because the reference CHANGES from rung to rung. Nothing here is a QLIKE
quantity and nothing here shares a denominator with the SEC panel.

What the figure does not show: the machinery-validation fixture on an injected
data-generating process is a separate object and is not a Yelp result; the
h=3-month content residual is drawn hatched because its within-month text-swap
diagnostic is borderline, and it is therefore not claimed.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supp_style import (apply_style, finish, gate, annot, note, BLUE, SKY,  # noqa: E402
                        VERM, VERM_TXT, GREEN, YELLOW, PURPLE, GREY, LIGHT,
                        INK, INK2, RULE, TAB, AGG, REPO)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --------------------------------------------------------------- load evidence
CAS = pd.read_csv(os.path.join(TAB, "yelp_cascade.csv"))
ARM = pd.read_csv(os.path.join(TAB, "yelp_multiarm.csv"))
CAS_MD = open(os.path.join(TAB, "yelp_cascade.md")).read()
ARM_MD = open(os.path.join(TAB, "yelp_multiarm.md")).read()

# minimum detectable effects, per horizon and per ladder stage
m = re.search(r"h=1m: AR stage ([\d.]+)%, entity stage ([\d.]+)%; "
              r"h=3m: AR stage ([\d.]+)%, entity stage ([\d.]+)%", CAS_MD)
MDE_AR = {1: float(m.group(1)), 3: float(m.group(3))}
MDE_ENT = {1: float(m.group(2)), 3: float(m.group(4))}

m = re.search(r"([\d,]+) entities, ([\d,]+) events", CAS_MD)
N_ENT, N_OBS = (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")))
m = re.search(r"(\d{4})-(\d{4}); train<=(\d{4})/val (\d{4})/test (\d{4})-(\d{2})",
              ARM_MD)
YR0, YR1, TRAIN_TO, VAL_YR, TEST0, TEST1 = (int(g) for g in m.groups())
m = re.search(r"max \|mean DM\| = ([\d.]+) \(threshold ([\d.]+)\)", CAS_MD)
PLACEBO_MAX, PLACEBO_THR = float(m.group(1)), float(m.group(2))
m = re.search(r"h=3 row5_swap: mean DM ([+\-][\d.]+), mean p ([\d.]+) "
              r"\(BORDERLINE\)", CAS_MD)
SWAP_DM3, SWAP_P3 = float(m.group(1)), float(m.group(2))

H1 = CAS[CAS.h == 1].sort_values("row").reset_index(drop=True)
H3 = CAS[CAS.h == 3].sort_values("row").reset_index(drop=True)

ARMS = ["TF-IDF ridge", "Qwen3-Emb-8B", "Llama-70B prompt", "70B probe (0-text)"]
PROMPT_ROWS = ARM[ARM.arm == "Llama-70B prompt"].set_index("h")
PROBE_ROWS = ARM[ARM.arm == "70B probe (0-text)"].set_index("h")
# probe combination increment as a share of the prompted arm's own increment
REPRO = {h: 100.0 * float(PROBE_ROWS.loc[h, "combiner"])
         / float(PROMPT_ROWS.loc[h, "combiner"]) for h in (1, 3)}

# ------------------------------------------------------------------------ gate
# The literal side of gate() is the only place a number is typed by hand; it
# exists to abort the build the moment the committed tables stop saying what the
# frozen main text says.
gate(
    {
        "n_rungs": 5, "n_arm_rows": 8,
        "ladder_h1": (28.431, -24.702, 0.346, -10.581, 0.365),
        "ladder_h3": (36.638, -35.294, 0.871, -9.484, 0.612),
        "mde_ar": (0.18, 0.40), "mde_entity": (0.08, 0.08),
        "residual_h1": (0.365, 0.376, 0.025, -0.056),
        "residual_h3": (0.612, 0.786, 0.176, 0.057),
        "probe_h1_dm_p": (5.65, 0.00074),
        "reproduction_pct": (149, 103),
        "entities": 8474, "observations": 407385,
        "period": (2005, 2022), "split": (2016, 2017, 2018, 21),
        "placebo_max_abs_dm": 1.31, "placebo_threshold": 2.0,
        "swap_h3_row5_p": 0.047,
    },
    {
        "n_rungs": int(H1.row.nunique()), "n_arm_rows": int(len(ARM)),
        "ladder_h1": tuple(round(float(v), 3) for v in H1.delta_rel_pct),
        "ladder_h3": tuple(round(float(v), 3) for v in H3.delta_rel_pct),
        "mde_ar": (MDE_AR[1], MDE_AR[3]), "mde_entity": (MDE_ENT[1], MDE_ENT[3]),
        "residual_h1": tuple(round(float(ARM[(ARM.arm == a) & (ARM.h == 1)]
                                          .residual.iloc[0]), 3) for a in ARMS),
        "residual_h3": tuple(round(float(ARM[(ARM.arm == a) & (ARM.h == 3)]
                                          .residual.iloc[0]), 3) for a in ARMS),
        "probe_h1_dm_p": (round(float(PROBE_ROWS.loc[1, "res_dm"]), 2),
                          round(float(PROBE_ROWS.loc[1, "res_p2way"]), 5)),
        "reproduction_pct": (int(round(REPRO[1])), int(round(REPRO[3]))),
        "entities": N_ENT, "observations": N_OBS,
        "period": (YR0, YR1), "split": (TRAIN_TO, VAL_YR, TEST0, TEST1),
        "placebo_max_abs_dm": PLACEBO_MAX, "placebo_threshold": PLACEBO_THR,
        "swap_h3_row5_p": SWAP_P3,
    },
)

# ---------------------------------------------------------------- presentation
# Rung labels carry their OWN reference, because the reference changes from rung
# to rung and the five numbers are therefore not on one scale.
#
# Hue is semantic and is shared with panel (b), which is the whole reason the
# ladder had to stop using one hue per rung: rung 5 at h=1 (+0.365%) IS panel
# (b)'s TF-IDF point at h=1, and the two panels used to draw that single number
# in two different colours. Three hues now, each meaning one thing:
#   BLUE  the arm gains on its own reference   (panel b: DM favours text)
#   VERM  the arm loses to its own reference   (panel b: DM points the wrong way)
#   GREY  the field design, exhibited not claimed
# Sign is therefore carried twice, by position about zero and by hue, and colour
# is never the only channel: every bar is also labelled with its own value.
RUNG = [("1 naive split", "(field design)", "vs pooled mean", GREY),
        ("2 text alone", "chronological", "vs recal. AR", VERM),
        ("3 AR + text", "combiner", "vs recal. AR", BLUE),
        ("4 AR + entity", "mean, 0 text", "vs recal. AR", VERM),
        ("5 AR + entity", "+ text", "vs AR + entity", BLUE)]


def fmt(v):
    """Two decimals down to 1%, three below it: the residual rungs are small."""
    return f"{v:+.2f}" if abs(v) >= 1 else f"{v:+.3f}"


def marked(fig, x, y, letter, body, **kw):
    """A panel marker in primary ink, then its apparatus block in recessive ink.

    The two used to be one string, so the marker that identifies the panel and
    the basis statement that supports it carried exactly the same weight and
    colour. The body is placed where the old "(a)  " prefix put it -- the offset
    is measured from the very glyphs it replaces rather than guessed -- so the
    tight bounding box, and with it every printed point size in the figure, is
    unchanged by the split.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    def measure(s):
        probe = fig.text(0.0, -1.0, s, fontsize=9)
        w = probe.get_window_extent(renderer=r).width
        probe.remove()
        return w

    first = body.split("\n")[0]
    dx = ((measure(f"({letter})  {first}") - measure(first))
          / (fig.dpi * fig.get_figwidth()))
    fig.text(x, y, f"({letter})", fontsize=9, fontweight="semibold", color=INK,
             ha="left", va="top")
    return note(fig, x + dx, y, body, rule=False, **kw)


def hairline(fig, x, y, w):
    """The RULE hairline that closes the figure and opens its basis block.

    `note()` sets its own rule 0.012 of the figure height above the text, which
    is more air than this canvas has between panel (b)'s axis label and the
    closing block; the offset is therefore given here instead. The rule itself is
    the same 0.5 pt RULE hairline, and it moves nothing: it is drawn inside white
    space the layout already reserved.
    """
    fig.lines.append(plt.Line2D([x, x + w], [y, y], transform=fig.transFigure,
                                color=RULE, linewidth=0.5, zorder=0.5))


apply_style(base_size=9)
# Vertical space is allocated top-down in inches so that no text block can drift
# into another, and every free-text line is kept under ~92 characters so the
# finished canvas stays 6.4 in wide.
H = 7.62
fig = plt.figure(figsize=(6.4, H))


def frac(inches_from_top):
    return 1.0 - inches_from_top / H


gs1 = fig.add_gridspec(1, 1, left=0.115, right=0.985,
                       top=frac(0.52), bottom=frac(1.92))
gs2 = fig.add_gridspec(1, 1, left=0.115, right=0.985,
                       top=frac(2.10), bottom=frac(3.50))
ax1 = fig.add_subplot(gs1[0, 0])
ax2 = fig.add_subplot(gs2[0, 0])

# --------------------------------------------- panel (a): the five-rung ladder
# The five values span two orders of magnitude, so the y axis is symmetric-log
# with a linear zone below 1%; the zone is drawn and labelled on the artefact so
# no reader can mistake the small rungs for large ones.
for ax, dat, h in ((ax1, H1, 1), (ax2, H3, 3)):
    ax.axhspan(-1, 1, color=LIGHT, alpha=0.55, zorder=0, lw=0)
    ax.axhline(0.0, lw=0.9, color=GREY, zorder=2)
    for i, r in dat.iterrows():
        val = float(r.delta_rel_pct)
        hatch = (h == 3 and int(r.row) == 5)
        ax.bar(i, val, width=0.56, color=RUNG[i][3], zorder=3,
               edgecolor=("white" if hatch else RUNG[i][3]),
               hatch=("////" if hatch else None), linewidth=0.0)
        ax.annotate(fmt(val), xy=(i, val), xycoords="data",
                    xytext=(0, 3 if val > 0 else -3), textcoords="offset points",
                    fontsize=9, color=RUNG[i][3], ha="center",
                    va=("bottom" if val > 0 else "top"))
    ax.axhline(MDE_AR[h], lw=1.0, color=VERM_TXT, ls=(0, (4, 2)), zorder=4)
    # The dashed rule used to run straight through the parentheses of "(AR
    # stage)". A white halo opens the rule around the glyphs instead of hiding
    # the line, and the anchor moves 0.06 data units INWARD to make room for the
    # halo, so the panel's bounding box is not touched.
    annot(ax, 5.80, MDE_AR[h], f"MDE {MDE_AR[h]:.2f}%\n(AR stage)",
          color=VERM_TXT, ha="right", va="bottom", linespacing=1.35)
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.85)
    ax.set_yticks([-30, -10, -1, 0, 1, 10, 30])
    ax.set_yticklabels(["-30", "-10", "-1", "0", "1", "10", "30"], fontsize=9)
    ax.set_ylim(-72, 72)
    ax.set_xlim(-0.62, 5.90)
    ax.set_ylabel("relative change\nin MSE, %", fontsize=9, linespacing=1.35)
    ax.text(0.995, 0.96, f"$h = {h}$ month" + ("s" if h == 3 else ""),
            transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK,
            ha="right", va="top")

# The key for the ladder's hues, set in white space the h=1 panel already has:
# without it the reader met a grey, a vermillion and a blue rung with nothing
# saying they are different KINDS of thing rather than four series. Two rows and
# not three, because a third row reaches down into the "+0.346" data label -- the
# grey rung is the one the x-axis label already names as the field design.
key = ax1.legend(handles=[
    Patch(facecolor=BLUE, edgecolor="none",
          label="gains on its own reference"),
    Patch(facecolor=VERM, edgecolor="none",
          label="loses to its own reference")],
    loc="upper left", bbox_to_anchor=(0.185, 0.995), frameon=False, fontsize=9,
    handlelength=1.1, handleheight=0.85, handletextpad=0.55, labelspacing=0.32,
    borderpad=0.0, borderaxespad=0.0, labelcolor=INK2)
key.set_zorder(5)

ax1.set_xticks([])
ax2.set_xticks(range(5))
ax2.set_xticklabels([f"{a}\n{b}\n{c}" for a, b, c, _ in RUNG], fontsize=9,
                    linespacing=1.35)
# This caveat used to be set at y=30, where its first line crossed the panel's
# own top spine and belonged to neither panel; it now sits wholly inside the
# h=3 axes, in recessive ink and with a halo, so it reads as a note about the
# hatch rather than as plotted content.
annot(ax2, 2.55, 30.0, f"rung 5 hatched: within-month text-swap borderline\n"
                       f"(mean $p$ = {SWAP_P3:.3f}), so this residual is not "
                       f"claimed",
      color=INK2, ha="center", va="center", linespacing=1.35)

marked(fig, 0.022, frac(0.06), "a",
       "Every bar names its own reference: the five numbers are "
       "increments over four different\n"
       "baselines, not one series. The y axis is linear inside the shaded "
       "band and logarithmic outside it.")

# ------------------------------------- panel (b): four arms, residual increment
gs3 = fig.add_gridspec(1, 1, left=0.255, right=0.985,
                       top=frac(5.00), bottom=frac(6.40))
ax3 = fig.add_subplot(gs3[0, 0])

rows, labels = [], []
for a in ARMS:
    for h in (1, 3):
        rows.append((-float(len(rows)), a, h))
        labels.append(f"{a}   $h={h}$m")

ax3.axvline(0.0, lw=0.9, color=GREY, zorder=2)
ax3.axvline(MDE_ENT[1], lw=1.0, color=VERM_TXT, ls=(0, (4, 2)), zorder=2)
for yy, a, h in rows:
    r = ARM[(ARM.arm == a) & (ARM.h == h)].iloc[0]
    wrong_way = float(r.res_dm) > 0            # DM > 0 = text hurts here
    sig = float(r.res_p2way) < .05
    col = VERM if wrong_way else BLUE
    ax3.plot([float(r.residual)], [yy], marker=("X" if wrong_way else "o"),
             ms=(7.0 if wrong_way else 6.0),
             mfc=(col if sig else "white"), mec=col, mew=1.1, zorder=4)
    ax3.text(0.92, yy, f"DM {float(r.res_dm):+.2f},  "
                       f"2-way $p$ = {float(r.res_p2way):.3g}",
             fontsize=9, color=INK, ha="left", va="center")

ax3.set_yticks([yy for yy, *_ in rows])
ax3.set_yticklabels(labels, fontsize=9)
ax3.set_ylim(rows[-1][0] - 0.60, rows[0][0] + 0.60)
ax3.set_xlim(-0.16, 1.74)
ax3.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax3.set_xlabel("identity-controlled residual, % of the AR + entity-mean "
               "reference MSE", fontsize=9, labelpad=1.5)
ax3.tick_params(axis="y", length=0)
# The axis label belongs to its axis, so it is pulled up towards it; the white
# space that buys is what the closing block's hairline is set in. Nothing moves
# outward and the figure's bounding box is unchanged.
ax3.tick_params(axis="x", pad=2.5)
fig.legend(handles=[
    Line2D([], [], ls="none", marker="o", ms=6, color=BLUE,
           label="DM favours text, two-way $p<.05$"),
    Line2D([], [], ls="none", marker="o", ms=6, mfc="white", mec=BLUE,
           label="not significant"),
    Line2D([], [], ls="none", marker="X", ms=7, color=VERM,
           label="DM points the wrong way")],
    loc="upper left", bbox_to_anchor=(0.022, frac(4.68)), ncol=3, fontsize=9,
    handletextpad=0.3, columnspacing=1.3, frameon=False, labelcolor=INK2)

marked(fig, 0.022, frac(4.10), "b",
       f"Four arms, both horizons, one shared legend. The zero-content "
       f"probe is a diagnostic, not a\n"
       f"challenger: at $h$ = 1 month it is significantly harmful once the "
       f"entity mean is in the reference.\n"
       f"The dashed rule is the entity-stage minimum detectable effect, "
       f"{MDE_ENT[1]:.2f}% at both horizons.")

# The closing block is the figure's basis statement -- the reproduction share,
# the denominator, the split and the placebo -- and it is the one block with no
# panel marker of its own, so the hairline is what tells a reader where the
# argument stops and the basis begins.
hairline(fig, 0.022, frac(6.836), 0.85)
note(fig, 0.022, frac(6.90),
     f"The probe reproduces {REPRO[1]:.0f}% ($h$ = 1m) and "
     f"{REPRO[3]:.0f}% ($h$ = 3m) of the prompted arm's own combination\n"
     f"increment. Panel: {N_ENT:,} businesses, {N_OBS:,} business-months, "
     f"{YR0}-{YR1}; chronological split,\n"
     f"train to {TRAIN_TO}, validation {VAL_YR}, test {TEST0}-{TEST1}. A "
     f"20-seed label-shuffle placebo is clean at every rung\n"
     f"(max |mean DM| = {PLACEBO_MAX:.2f} against a threshold of "
     f"{PLACEBO_THR:.1f}).",
     rule=False)

finish(fig, "F15_yelp_portability")
