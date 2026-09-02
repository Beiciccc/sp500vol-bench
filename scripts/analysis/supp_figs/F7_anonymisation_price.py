"""F7 -- What anonymisation prices, and the arms where it goes the other way.

Substantiates, and bounds the scope of, these frozen main-text sentences:
  * 00_abstract.tex : "An anonymisation arm prices the identity share at 0.51."
  * 01_intro.tex    : "An anonymisation arm turns the bound into an estimate of
                       the identity share (pooled median 0.51, CI [0.44,0.56];
                       prompted-arm median 0.56)."
  * 06_results.tex  : "...per-horizon identity shares 0.51/0.56/0.71 (day-block-
                       bootstrap 95% CIs [0.40,0.62], [0.43,0.70], [0.57,0.84])...
                       The same arm prices FinBERT's increment at share 0.02
                       ([-0.14,0.19])..."
  * 08_discussion.tex: "a pooled identity share of 0.51 ([0.44,0.56]; prompted arm
                       0.56), between the matched-firm swap's 0.29 and the per-cell
                       interval bound (0.72)."
  * 10_limitations.tex: "...the anonymisation arm supplies the lower bound, pricing
                       the identity share at 0.51."

Evidence files read (every plotted number comes from one of these):
  results/tables/anon_share_ci.csv          -- the 9 share readouts, day-block
                                               bootstrap CIs, undefined-draw fractions
  results/tables/anon_arm.csv               -- masked/unmasked increment pairs, DM,
                                               interval bound, swap retention, arm status
  results/tables/anon_arm.md                -- the event-driven G2 masking gate line
  results/tables/anon_arm_lf.csv            -- long-form arm statuses (zero executed)
  results/tables/anon_arm_lf.md             -- the long-form G2 masking gate line
  results/tables/anon_annex_samelineage.csv -- the exploratory same-lineage annex
  results/tables/matched_firm_swap.csv      -- cross-check of the 0.29 swap marker

Never rendered: the `prereg` and `g1_deviation` columns of anon_arm.csv, and the
`box_ctrl_dir` / `masked_dir` columns of anon_annex_samelineage.csv (those hold
local filesystem paths and are dropped at load time, before any use).
"""
import os
import re
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supp_style import (apply_style, finish, gate, note, panel,  # noqa: E402
                        BLUE, SKY, VERM, VERM_TXT, GREEN, YELLOW, PURPLE,
                        GREY, INK, INK2, LIGHT, RULE, TAB, AGG, REPO)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402

# --------------------------------------------------------------- load evidence
CI = pd.read_csv(os.path.join(TAB, "anon_share_ci.csv"))

# anon_arm.csv: drop the two narrative columns the spec forbids rendering.
ARM = pd.read_csv(os.path.join(TAB, "anon_arm.csv")).drop(
    columns=["prereg", "g1_deviation"])
ARM_LF = pd.read_csv(os.path.join(TAB, "anon_arm_lf.csv")).drop(
    columns=["prereg", "g1_deviation"])

# anon_annex_samelineage.csv: box_ctrl_dir / masked_dir hold absolute local paths
# that would de-anonymise the submission.  Dropped here, before any use.
ANNEX = pd.read_csv(os.path.join(TAB, "anon_annex_samelineage.csv")).drop(
    columns=["box_ctrl_dir", "masked_dir"])

SWAP = pd.read_csv(os.path.join(TAB, "matched_firm_swap.csv"))

G2_RE = (r"G2 masking: (\d+) docs, ([\d.]+)% with >=1 mask, mean masked-char "
         r"fraction ([\d.]+)%; leak rates: own-ticker ([\d.]+)%, "
         r"own-name-token ([\d.]+)%")


def read_g2(md_name):
    with open(os.path.join(TAB, md_name), encoding="utf-8") as fh:
        m = re.search(G2_RE, fh.read())
    if m is None:
        sys.exit(f"GATE FAIL - no G2 masking line in {md_name}")
    return dict(docs=int(m.group(1)), any_mask_pct=float(m.group(2)),
                masked_char_pct=float(m.group(3)),
                own_ticker_pct=float(m.group(4)), own_name_pct=float(m.group(5)))


G2_ED = read_g2("anon_arm.md")
G2_LF = read_g2("anon_arm_lf.md")


def ci_row(readout, arm, h):
    r = CI[(CI.readout == readout) & (CI.arm == arm) & (CI.h.astype(str) == str(h))]
    assert len(r) == 1, (readout, arm, h, len(r))
    return r.iloc[0]


def arm_row(arm, h):
    r = ARM[(ARM.arm == arm) & (ARM.h == h)]
    assert len(r) == 1, (arm, h, len(r))
    return r.iloc[0]


POOLED = ci_row("pooled_median", "c6+c2", "5/10/20")
C2H20 = ci_row("cell", "c2", "20")
# The six (arm, horizon) cells panel (b) draws, in the order they are drawn.
CELLS = [("c6", 5), ("c6", 10), ("c6", 20), ("c2", 5), ("c2", 10), ("c2", 20)]
SWAP_MED = float(ARM.swap_retention.median())          # 0.29 triangulation marker
BOUND_MED = float(ARM.refinterval_bound.median())      # 0.72 triangulation marker
ANNEX_LF = ANNEX[ANNEX.channel == "lf"].sort_values("h").reset_index(drop=True)

# ------------------------------------------------------------------------ gate
# The literal side of gate() is the ONLY place a number is written by hand: its
# whole purpose is to abort the build if the committed evidence has drifted from
# what the frozen main text states.  Every value the artefact draws is gated, not
# only the ones the main text quotes: the nine (point, lo, hi) triples of panel
# (a), and the twelve increment pairs and ten defined share labels of panel (b).
gate(
    {
        "n_share_readouts": 9,
        "ci_triples": ((0.51, 0.40, 0.62), (0.56, 0.43, 0.69),
                       (0.71, 0.57, 0.84), (0.02, -0.14, 0.19),
                       (0.51, 0.45, 0.57), (-0.39, -4.94, 0.13),
                       (0.56, 0.47, 0.69), (0.02, -0.13, 0.26),
                       (0.51, 0.44, 0.56)),
        "har_pairs": ((1.21, 0.60), (1.00, 0.44), (0.66, 0.19),
                      (2.14, 2.08), (2.10, 1.03), (0.92, 1.28)),
        "firm_pairs": ((0.45, 0.33), (0.25, 0.22), (0.20, 0.08),
                       (-0.31, 0.13), (-1.22, -1.97), (0.66, -0.35)),
        "har_shares": (0.51, 0.56, 0.71, 0.02, 0.51, -0.39),
        "firm_shares_defined": (0.25, 0.12, 0.59, 1.53),
        "firm_shares_undefined": 2,
        "pooled_share": 0.51, "pooled_ci": (0.44, 0.56),
        "c6_cell_shares": (0.51, 0.56, 0.71),
        "c6_arm_median": 0.56,
        "c2_h5_share": 0.02, "c2_h5_ci": (-0.14, 0.19),
        "c2_h20_undefined_frac": 0.028,
        "bootstrap_B": 2000,
        "swap_retention_median": 0.29,
        "refinterval_bound_median": 0.72,
        "n_ed_arms_registered": 3, "n_ed_executed": 2, "n_ed_exited": 1,
        "n_lf_executed": 0,
        "ed_gate_docs": 112528, "ed_masked_char_pct": 9.48,
        "ed_own_ticker_pct": 0.00, "ed_own_name_pct": 0.17,
        "lf_gate_docs": 31601, "lf_masked_char_pct": 5.24,
        "lf_any_mask_pct": 100.0, "lf_own_ticker_pct": 0.00,
        "annex_lf_shares": (-0.879, -1.228, -0.717),
        "c2_h20_har_pair": (0.92, 1.28), "c2_h20_har_share": -0.39,
        "c2_h20_firm_pair": (0.66, -0.35), "c2_h20_firm_share": 1.53,
        "c2_h20_firm_dm": 3.07,
    },
    {
        "n_share_readouts": int(len(CI)),
        "ci_triples": tuple((round(float(r.share_point), 2), round(float(r.ci_lo), 2),
                             round(float(r.ci_hi), 2)) for _, r in CI.iterrows()),
        "har_pairs": tuple((round(float(arm_row(a, h).rel_har_unmasked), 2),
                            round(float(arm_row(a, h).rel_har_masked), 2))
                           for a, h in CELLS),
        "firm_pairs": tuple((round(float(arm_row(a, h).rel_firm_unmasked), 2),
                             round(float(arm_row(a, h).rel_firm_masked), 2))
                            for a, h in CELLS),
        "har_shares": tuple(round(float(arm_row(a, h).share_anon_har), 2)
                            for a, h in CELLS),
        "firm_shares_defined": tuple(
            round(float(arm_row(a, h).share_anon_firm), 2) for a, h in CELLS
            if not pd.isna(arm_row(a, h).share_anon_firm)),
        "firm_shares_undefined": int(sum(pd.isna(arm_row(a, h).share_anon_firm)
                                         for a, h in CELLS)),
        "pooled_share": round(float(POOLED.share_point), 2),
        "pooled_ci": (round(float(POOLED.ci_lo), 2), round(float(POOLED.ci_hi), 2)),
        "c6_cell_shares": tuple(round(float(ci_row("cell", "c6", h).share_point), 2)
                                for h in (5, 10, 20)),
        "c6_arm_median": round(float(ci_row("arm_median", "c6", "5/10/20")
                                     .share_point), 2),
        "c2_h5_share": round(float(ci_row("cell", "c2", 5).share_point), 2),
        "c2_h5_ci": (round(float(ci_row("cell", "c2", 5).ci_lo), 2),
                     round(float(ci_row("cell", "c2", 5).ci_hi), 2)),
        "c2_h20_undefined_frac": round(float(C2H20.undefined_frac), 3),
        "bootstrap_B": int(POOLED.B),
        "swap_retention_median": round(SWAP_MED, 2),
        "refinterval_bound_median": round(BOUND_MED, 2),
        "n_ed_arms_registered": int(ARM.arm.nunique()),
        "n_ed_executed": int(ARM[ARM.status == "executed"].arm.nunique()),
        "n_ed_exited": int(ARM[ARM.status == "g1-fail"].arm.nunique()),
        "n_lf_executed": int((ARM_LF.status == "executed").sum()),
        "ed_gate_docs": G2_ED["docs"], "ed_masked_char_pct": G2_ED["masked_char_pct"],
        "ed_own_ticker_pct": G2_ED["own_ticker_pct"],
        "ed_own_name_pct": G2_ED["own_name_pct"],
        "lf_gate_docs": G2_LF["docs"], "lf_masked_char_pct": G2_LF["masked_char_pct"],
        "lf_any_mask_pct": G2_LF["any_mask_pct"],
        "lf_own_ticker_pct": G2_LF["own_ticker_pct"],
        "annex_lf_shares": tuple(round(float(v), 3) for v in ANNEX_LF.share_har),
        "c2_h20_har_pair": (round(float(arm_row("c2", 20).rel_har_unmasked), 2),
                            round(float(arm_row("c2", 20).rel_har_masked), 2)),
        "c2_h20_har_share": round(float(arm_row("c2", 20).share_anon_har), 2),
        "c2_h20_firm_pair": (round(float(arm_row("c2", 20).rel_firm_unmasked), 2),
                             round(float(arm_row("c2", 20).rel_firm_masked), 2)),
        "c2_h20_firm_share": round(float(arm_row("c2", 20).share_anon_firm), 2),
        "c2_h20_firm_dm": round(float(arm_row("c2", 20).dm_firm_masked), 2),
    },
)
# Cross-check: the 0.29 marker also appears as matched_firm_swap's own median over
# its 38 genuine cells (0.31, a DIFFERENT denominator) -- assert they are distinct
# so the two medians can never be silently conflated on the artefact.
assert round(SWAP_MED, 2) != round(
    float(SWAP[SWAP.genuine].retention_vs_real.median()), 2)

# ------------------------------------------------------------------------ plot
apply_style(base_size=9)
H_COL = {5: BLUE, 10: YELLOW, 20: PURPLE}
WHITE_BOX = dict(facecolor="white", edgecolor="none", pad=1.2)


def minus(s):
    """ASCII hyphen -> U+2212 in a hand-formatted number.

    matplotlib's own tick formatter sets the typographic minus (axes.unicode_
    minus), so every f-string number written by hand here has to be converted or
    the same axis carries two minus signs of different length -- "-4.94" beside
    "-0.5" on panel (a), "-39%" beside the "-1"/"-2" y ticks on panel (b).  Only
    ever applied to numeric strings, never to prose, which needs its real hyphens
    ("day-block", "fine-tuned").  Costs at most 2pt on strings that are nowhere
    near the outermost extent, so the tight bounding box is unaffected.
    """
    return s.replace("-", "−")


fig = plt.figure(figsize=(6.4, 8.05))
gs = GridSpec(3, 1, figure=fig, height_ratios=[3.00, 2.72, 2.20],
              left=0.152, right=0.995, top=0.955, bottom=0.015, hspace=0.56)

# =========================================================== panel (a) forest =
gsa = gs[0].subgridspec(1, 3, width_ratios=[0.50, 3.05, 1.66], wspace=0.115)
axL = fig.add_subplot(gsa[0])      # broken-axis tail: the c2 h=20 lower whisker
axR = fig.add_subplot(gsa[1], sharey=axL)
axN = fig.add_subplot(gsa[2], sharey=axL)

# One colour system for the whole artefact, which is what this panel used to
# break: hue is the HORIZON (H_COL, the same three hues panels (b) and (c) use
# and panel (b)'s legend names), a dashed interval is the fine-tuned arm and a
# solid one the prompted arm (again panel (b)'s legend), ink marks the two arm
# medians and the pooled median because they are aggregates rather than cells,
# green is reserved for the two triangulating instruments and vermillion for the
# adverse readings alone.  Before this, panel (a) drew the fine-tuned arm in the
# vermillion that means "adverse" three inches away on the same panel, and drew
# horizons in an arm hue that means "h = 5" on the panel below.  Every row is
# also labelled with its own horizon, so no series rests on colour.
SLOTS = [
    ("head", "Prompted arm  C6 (LLM-written text), 8-K channel", None, None, None),
    ("cell", "h = 5", "c6", 5, H_COL[5]),
    ("cell", "h = 10", "c6", 10, H_COL[10]),
    ("cell", "h = 20", "c6", 20, H_COL[20]),
    ("armmed", "arm median", "c6", "5/10/20", INK),
    ("gap", None, None, None, None),
    ("head", "Fine-tuned arm  C2 (FinBERT), 8-K channel", None, None, None),
    ("cell", "h = 5", "c2", 5, H_COL[5]),
    ("cell", "h = 10", "c2", 10, H_COL[10]),
    ("cell", "h = 20", "c2", 20, H_COL[20]),
    ("gap", None, None, None, None),
    ("armmed", "arm median", "c2", "5/10/20", INK),
    ("gap", None, None, None, None),
    ("head", "Registered, exited at its reproduction gate", None, None, None),
    ("dead", "B2 (TF-IDF ridge)", None, None, None),
    ("gap", None, None, None, None),
    ("pooled", "pooled median", "c6+c2", "5/10/20", INK),
]
ypos = {i: -i for i in range(len(SLOTS))}
XLO, XHI = -0.80, 1.32          # main segment
BLO, BHI = -5.12, -4.62         # broken tail segment (compressed, see break marks)
YBOT = min(ypos.values()) - 0.75

yticks, ylabels = [], []
for i, (kind, lab, arm, h, col) in enumerate(SLOTS):
    y = ypos[i]
    if kind == "gap":
        continue
    if kind == "head":
        # Structure, not apparatus: weight rather than italics, so the three row
        # groups read as the panel's spine at a glance.  Weight costs no width
        # the tight bounding box can charge for.
        axR.text(XLO + 0.03, y, lab, ha="left", va="center", fontsize=9,
                 color=INK, fontweight="semibold", bbox=WHITE_BOX, zorder=5)
        continue
    if kind == "dead":
        yticks.append(y)
        ylabels.append(lab)
        axR.plot([XLO + 0.06, XHI - 0.06], [y, y], ls=(0, (1, 2.6)), lw=0.9,
                 color=LIGHT, zorder=1)
        # Recessive ink because there is no statistic to read, not because the
        # exit is a detail: the empty row is the point.
        axN.text(0.015, y, "exited at G1: no statistic", ha="left",
                 va="center", fontsize=9, color=INK2)
        continue

    readout = {"cell": "cell", "armmed": "arm_median", "pooled": "pooled_median"}[kind]
    r = ci_row(readout, arm, h)
    pt, lo, hi = float(r.share_point), float(r.ci_lo), float(r.ci_hi)
    mk = {"cell": "o", "armmed": "D", "pooled": "s"}[kind]
    ms = {"cell": 4.6, "armmed": 5.2, "pooled": 5.6}[kind]
    yticks.append(y)
    ylabels.append(lab)

    # Dashed for the fine-tuned arm, solid for the prompted one: the same two
    # line styles panel (b)'s legend already names, so hue is free to mean the
    # horizon on both panels.  The end-caps stay solid, so a short interval is
    # still read from its caps rather than from the dash phase.
    ils = (0, (4.5, 2.2)) if arm == "c2" else "-"
    for ax, (a, b) in ((axR, (XLO, XHI)), (axL, (BLO, BHI))):
        if hi < a or lo > b:
            continue
        ax.plot([max(lo, a), min(hi, b)], [y, y], lw=1.6, color=col, ls=ils,
                solid_capstyle="butt", dash_capstyle="butt", zorder=3)
        if a <= pt <= b:
            ax.plot([pt], [y], marker=mk, ms=ms, color=col, mec="white",
                    mew=0.7, zorder=4)
    for v in (lo, hi):                       # interval end-caps
        for ax, (a, b) in ((axR, (XLO, XHI)), (axL, (BLO, BHI))):
            if a <= v <= b:
                ax.plot([v, v], [y - 0.21, y + 0.21], lw=1.2, color=col, zorder=3)

    axN.text(0.015, y, minus(f"{pt:+.2f}  [{lo:+.2f}, {hi:+.2f}]"), ha="left",
             va="center", fontsize=9, color=GREY)
    if float(r.undefined_frac) > 0:
        # A full row pitch below the interval it qualifies, i.e. centred in the
        # blank slot that already follows this row, rather than 0.80 of one: at
        # 0.80 the caveat's ascenders ran into the brackets of the interval above
        # (shared pixels on "[", "]" and the comma) while 0.34 of a row of white
        # sat idle beneath it.  Purely a move into existing space -- the block is
        # still bounded by the h=20 row above and the arm-median row below, so
        # the readout column's extent, and the figure's box, do not change.
        axN.text(0.015, y - 1.00,
                 f"{100 * float(r.undefined_frac):.1f}% of draws undefined",
                 ha="left", va="center", fontsize=9,
                 color=VERM_TXT)

for ax in (axL, axR):
    ax.axvline(0.0, color=GREY, lw=0.7, zorder=2)
    ax.axvline(1.0, color=GREY, lw=0.7, zorder=2)
    ax.axvline(SWAP_MED, color=GREEN, lw=1.1, ls=(0, (4, 2)), zorder=2)
    ax.axvline(BOUND_MED, color=GREEN, lw=1.1, ls=(0, (1, 1.7)), zorder=2)

axR.set_xlim(XLO, XHI)
axL.set_xlim(BLO, BHI)
axL.set_ylim(YBOT, 0.95)
axR.set_xticks([-0.5, 0.0, 0.5, 1.0])
axL.set_xticks([float(C2H20.ci_lo)])
axL.set_xticklabels([minus(f"{float(C2H20.ci_lo):.2f}")], fontsize=9)
axR.tick_params(labelleft=False)
axN.set_xlim(0, 1)
axN.axis("off")
for ax in (axL, axR):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
axL.set_yticks(yticks)
axL.set_yticklabels(ylabels, fontsize=9)

# axis-break marks on the facing edges of the two segments, top and bottom, so
# the break is unmistakable wherever the reader enters the panel
for ax, xf in ((axL, 1.0), (axR, 0.0)):
    for base in (0.0, 1.0):
        for off in (-0.030, 0.006):
            ax.plot([xf - 0.030, xf + 0.030],
                    [base + off, base + off + 0.036],
                    transform=ax.transAxes, clip_on=False, color=GREY, lw=0.9,
                    zorder=6)

# Two lines, so the label still fits inside the artefact's own width once every
# on-artefact string is set at the house minimum of 9pt.
axR.set_xlabel("identity share $=1-$ (masked increment / unmasked increment)\n"
               "0 $=$ no identity,   1 $=$ all identity",
               fontsize=9, labelpad=3, linespacing=1.35)
# Panel (a)'s header used to be a marker on one baseline, a two-line basis
# statement on another, and an "axis break" label stranded between them, all in
# the same ink as the interval readouts on the right.  It is now two tiers: marker
# and title on one baseline, then a hairline, then one apparatus baseline in
# recessive ink carrying both the basis statement and the axis-break label.  Not
# one word is dropped; the sentence is split at its own semicolon.
#
# The marker column axL is a tenth of the panel wide, so panel()'s own title
# offset (a fraction of the axes it is called on) would set the title on top of
# the marker.  The title therefore goes on axR at the marker's own baseline:
# axL and axR are one gridspec row sharing a y axis, so a transAxes y is the
# same figure y on both.
#
# The whole header is re-tiered INSIDE the box it already occupied.  The two-line
# statement became a one-line title plus a one-line basis statement, so the title
# is set on the first line's own baseline and the artefact's top edge does not
# move by a point: a taller box would mean a harder down-scale on the page and
# therefore smaller printed type everywhere on this figure.
_posR = axR.get_position()      # gridspec-relative, so identical on both canvases
LINE_A = (9 * 1.35 / 72.0) / (_posR.height * fig.get_figheight())
Y_TITLE = 1.155 + 0.86 * LINE_A  # the old first line's baseline, less the leading
                                 # a single line no longer needs above it: this is
                                 # the number that keeps the top edge where it was
Y_APPAR = 1.194                 # apparatus block, top-aligned, under the hairline
panel(axL, "a", dy=Y_TITLE)
axR.text(0.0, Y_TITLE, "Nine registered readouts", transform=axR.transAxes,
         fontsize=9, fontweight="semibold", color=INK, ha="left", va="bottom")
_y_appar = _posR.y0 + Y_APPAR * _posR.height
note(fig, _posR.x0, _y_appar,
     f"day-block bootstrap 95% CI, B $=$ {int(POOLED.B):,}, resampling unit "
     f"the effective trading day",
     rule=False)
# note()'s own hairline sits 0.012 of the figure above its text, which is more
# room than this header band has between the title's descenders and the green
# triangulation labels below; the artefact may not grow to make room, so the
# hairline is drawn here at the same weight and colour, half that offset up, and
# spanning the panel it heads -- marker column to readout column -- rather than
# stopping short of the basis statement it separates from the data.
fig.lines.append(plt.Line2D([axL.get_position().x0, axN.get_position().x1],
                            [_y_appar + 0.005] * 2,
                            transform=fig.transFigure, color=RULE,
                            linewidth=0.5, zorder=0.5))
# Parenthesised: the label shares this baseline with the basis statement to its
# right, and unbracketed the two ran together as one phrase ("axis break
# day-block bootstrap ...").  Two glyphs on a centred string a full column clear
# of the artefact's left edge, so the box does not move.
axL.text(0.5, Y_APPAR, "(axis break)", transform=axL.transAxes, fontsize=9,
         color=INK2, ha="center", va="top")
# The triangulation labels keep their own baseline: it is the only one clear of
# both the axis-break marks above the frame and the apparatus tier above those.
blend = mtransforms.blended_transform_factory(axR.transData, axR.transAxes)
axR.text(SWAP_MED - 0.015, 1.055,
         f"matched-firm swap retention {SWAP_MED:.2f}", transform=blend,
         fontsize=9, color=GREEN, ha="right", va="bottom", clip_on=False)
axR.text(BOUND_MED + 0.015, 1.055,
         f"per-cell interval bound {BOUND_MED:.2f}", transform=blend,
         fontsize=9, color=GREEN, ha="left", va="bottom", clip_on=False)


# ================================================== panel (b) paired slope, ED =
gsb = gs[1].subgridspec(2, 1, height_ratios=[3.9, 1.0], hspace=0.10)
axb = fig.add_subplot(gsb[0])
axbn = fig.add_subplot(gsb[1])
axbn.axis("off")

BLK = {"har": (0.0, 1.0), "firm": (2.55, 3.55)}
lines = []
for arm, ls in (("c6", "-"), ("c2", (0, (4.5, 2.2)))):
    for h in (5, 10, 20):
        r = arm_row(arm, h)
        for ref, (x0, x1) in BLK.items():
            u = float(r[f"rel_{ref}_unmasked"])
            m = float(r[f"rel_{ref}_masked"])
            s = r[f"share_anon_{ref}"]
            axb.plot([x0, x1], [u, m], ls=ls, lw=1.5, color=H_COL[h],
                     marker="o", ms=3.4, mec="white", mew=0.6, zorder=3)
            lines.append(dict(arm=arm, h=h, ref=ref, x=x1, y=m, col=H_COL[h],
                              # The callout is an identity SHARE, so it is
                              # printed the way panel (a)'s readout column and
                              # this panel's own note print it -- +0.51, -0.39 --
                              # not as a percentage.  Set as "51%" it collided
                              # with the y axis, which is itself in QLIKE rel. %,
                              # and with the note four lines below reading
                              # "share -0.39" for the same number.
                              lab=("n/a" if pd.isna(s)
                                   else minus(f"{float(s):+.2f}")),
                              adverse=(arm == "c2" and h == 20)))

# The zero rule spans the two reference blocks and stops just short of each
# block's callout column, rather than running the full width of the axes.  Drawn
# edge to edge it passed behind the labels, whose own opaque background knocked
# two gaps in it, and then carried on as an unattached line across the empty
# right-hand third of the panel.  Ink only: the axes box is unchanged, so the
# artefact's bounding box is too.
for _z0, _z1 in ((-0.32, 1.28), (2.18, 3.83)):
    axb.plot([_z0, _z1], [0.0, 0.0], color=GREY, lw=0.7, zorder=2,
             solid_capstyle="butt")


def nudge(vals, gap, rounds=60):
    """Separate label positions by `gap`, keeping order and staying centred.

    Alternating up/down relaxation, so labels are displaced symmetrically about
    their true values instead of all being pushed upward off the lowest one.
    """
    order = np.argsort(vals)
    srt = np.array(vals, dtype=float)[order]
    lo, hi = srt.min(), srt.max()
    span = max(hi - lo, gap * (len(srt) - 1))
    lo -= (span - (hi - lo)) / 2.0
    hi = lo + span
    for _ in range(rounds):
        for i in range(1, len(srt)):
            if srt[i] - srt[i - 1] < gap:
                srt[i] = srt[i - 1] + gap
        srt[-1] = min(srt[-1], hi)
        for i in range(len(srt) - 2, -1, -1):
            if srt[i + 1] - srt[i] < gap:
                srt[i] = srt[i + 1] - gap
        srt[0] = max(srt[0], lo)
    out = np.empty_like(srt)
    out[order] = srt
    return out


# Zero is the load-bearing boundary of this panel -- the note under it turns on
# one line falling past the rule -- so a label may be displaced off its datum to
# clear its neighbours, but never across the rule.  Positive-datum labels are
# relaxed as one stack starting at +LAB_KEEP, negative-datum ones as a second
# stack ending at -LAB_KEEP; the two never mix, so a cell whose masked increment
# is positive can no longer be printed below zero (before this, four of the six
# firm-side labels sat under the rule where only two of the six endpoints do).
# LAB_TOP is the highest a 9pt label can be centred and still sit inside the
# frame; when the positive stack will not fit under it the pitch tightens (0.51
# rather than 0.55 data units, still ~2.8pt of white between rows of digits)
# instead of labels spilling over the frame or the type shrinking.
LAB_KEEP, LAB_TOP = 0.30, 2.86


def stack_signed(vals, gap):
    """Relax label positions by `gap` without carrying any label across zero."""
    vals = np.asarray(vals, dtype=float)
    out = np.empty_like(vals)
    for sel, up in ((vals >= 0, True), (vals < 0, False)):
        if not sel.any():
            continue
        sub = vals[sel]
        rel = nudge(sub, gap) if sub.size > 1 else sub.astype(float).copy()
        if up:
            rel = rel + max(0.0, LAB_KEEP - rel.min())
            if rel.max() > LAB_TOP:
                rank = np.argsort(np.argsort(sub))
                rel = np.linspace(LAB_KEEP, LAB_TOP, sub.size)[rank]
        else:
            rel = rel - max(0.0, rel.max() + LAB_KEEP)
        out[sel] = rel
    return out


for ref, (x0, x1) in BLK.items():
    grp = [d for d in lines if d["ref"] == ref]
    # 0.55 data units is the pitch six 9pt labels need in this panel: at 0.40 the
    # rows of digits stood 0.7pt apart and the middle two overprinted.  The room
    # comes out of the label column's own spacing, never out of the type size.
    # stack_signed then keeps each label on its datum's own side of the zero rule,
    # which is what lifts the HAR-side column off the legend and holds the
    # firm-side column clear of the bottom spine as well.
    ys = stack_signed([d["y"] for d in grp], 0.55)
    # The label is set on the panel's own background, so nothing behind it ever
    # strikes through the digits.  zorder sits above the zero rule (2) and below
    # the spines (2.5), so the label knocks out what it covers without ever
    # nicking the axis frame; no plotted line reaches this gutter.
    lab_bg = dict(facecolor=("white" if ref == "har" else "#F3F3F3"),
                  edgecolor="none", pad=1.0)
    for d, yy in zip(grp, ys):
        # Leader in the line's own colour, so a label can always be traced back
        # to its horizon even where two labels read the same ("51%") -- but as a
        # dotted hairline that starts a clear gap after the endpoint marker, not
        # as a solid stroke flush with it.  Drawn the old way, in series colour
        # and near series weight, the elbows read as the lines themselves
        # continuing past `masked` and plunging through zero: three apparent sign
        # crossings on a panel whose whole claim is about one.
        axb.plot([x1 + 0.07, x1 + 0.13, x1 + 0.22, x1 + 0.28],
                 [d["y"], d["y"], yy, yy], lw=0.55, color=d["col"], zorder=2,
                 ls=(0, (1.6, 2.0)), dash_capstyle="butt")
        axb.text(x1 + 0.32, yy, d["lab"], fontsize=9, va="center", ha="left",
                 color=(VERM_TXT if d["adverse"] else GREY), bbox=lab_bg,
                 zorder=2.2)

axb.set_xlim(-0.32, 4.62)
axb.set_ylim(-2.45, 3.05)
axb.set_xticks([0.0, 1.0, 2.55, 3.55])
axb.set_xticklabels(["unmasked", "masked", "unmasked", "masked"], fontsize=9)
axb.set_yticks([-2, -1, 0, 1, 2])
axb.set_ylabel("M1 increment,\nQLIKE rel. %", fontsize=9)
axb.axvspan(2.18, 4.62, color="#F3F3F3", zorder=0)
# The two reference blocks are structure, like panel (a)'s row-group heads, and
# are set in the same weight so a reader sorts the panel before reading a number.
axb.text(0.5, 2.92, "vs recalibrated HAR", ha="center", va="top", fontsize=9,
         color=INK, fontweight="semibold")
axb.text(3.05, 2.92, "vs HAR $+$ firm identity", ha="center", va="top",
         fontsize=9, color=INK, fontweight="semibold")
# The title wraps to two lines and the marker was pinned to the same baseline as
# the block, i.e. beside the SECOND line.  One text line of the title, measured
# in this panel's own axes fraction, is what lifts the marker onto the first.
LINE_B = (9 * 1.35 / 72.0) / (axb.get_position().height * fig.get_figheight())
panel(axb, "b", dy=1.055 + LINE_B)
# Size is untouchable here (a wider box means a harder down-scale means smaller
# printed type everywhere), so the title gains weight, never points.
axb.text(0.048, 1.055,
         "Event-driven arms: where the increment goes when every firm name,\n"
         "ticker, executive, product and CIK is masked",
         transform=axb.transAxes, fontsize=9, color=INK, ha="left",
         va="bottom", linespacing=1.35, fontweight="semibold")

handles = [Line2D([], [], color=H_COL[h], lw=1.5, marker="o", ms=3.4,
                  label=f"h = {h}") for h in (5, 10, 20)]
handles += [Line2D([], [], color=GREY, lw=1.5, ls="-", label="prompted C6"),
            Line2D([], [], color=GREY, lw=1.5, ls=(0, (4.5, 2.2)),
                   label="fine-tuned C2")]
axb.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.002, 0.005),
           ncol=2, fontsize=9, handlelength=1.9, columnspacing=1.0,
           handletextpad=0.5, borderpad=0.2, labelspacing=0.35)

# Adversarial repair (both lenses raised the same direction error and agree on the
# fix; the wording below is the union of the two, i.e. the more explicit and
# therefore more conservative version): the fine-tuned h=20 line FALLS against the
# firm-identity reference and RISES against HAR.  Both are drawn.
c2_20 = arm_row("c2", 20)
note_b = (
    "The fine-tuned $h{=}20$ line rises against the HAR reference "
    f"(${float(c2_20.rel_har_unmasked):+.2f}\\%$ to "
    f"${float(c2_20.rel_har_masked):+.2f}\\%$,\n"
    f"share ${float(c2_20.share_anon_har):+.2f}$), and falls past zero against "
    "the firm-identity reference "
    f"(${float(c2_20.rel_firm_unmasked):+.2f}\\%$ to\n"
    f"${float(c2_20.rel_firm_masked):+.2f}\\%$, share "
    f"${float(c2_20.share_anon_firm):+.2f}$, DM "
    f"${float(c2_20.dm_firm_masked):+.2f}$): masking there removes more than\n"
    "the whole increment. Both lines are drawn.")
# Word for word the block that used to sit under the panel as prose in the data's
# own ink, where nothing told a reader it was the basis rather than the finding.
# note() sets it in recessive ink under a hairline that runs the width of the
# panel, so everything below the rule is visibly the basis and everything above it
# the data.  The block drops from 0.35 to 0.20 of the spacer axes: the rule needs
# the room the note used to take, and 0.20 is what leaves the hairline clear of
# the x tick labels instead of underlining them.  Downward is free -- the
# artefact's bottom edge is set by panel (c), two rows below.
_posbn = axbn.get_position()
note(fig, _posbn.x0, _posbn.y0 + 0.20 * _posbn.height, note_b,
     width=_posbn.width)

# ============================== panel (c) long-form annex + figure note (row 3) =
gsc = gs[2].subgridspec(1, 2, width_ratios=[1.0, 1.72], wspace=0.12)
axc = fig.add_subplot(gsc[0])
axnote = fig.add_subplot(gsc[1])
axnote.axis("off")

axc.add_patch(Rectangle((0, 0), 1, 1, transform=axc.transAxes, facecolor="#EFEEEB",
                        edgecolor="none", zorder=0))
ends = []
for _, r in ANNEX_LF.iterrows():
    h = int(r.h)
    axc.plot([0, 1], [float(r.rel_har_ctrl), float(r.rel_har_masked)],
             ls=(0, (1.7, 1.7)), lw=1.7, color=H_COL[h], marker="o", ms=3.4,
             mec="white", mew=0.6, zorder=3)
    ends.append((h, float(r.rel_har_masked), float(r.share_har)))
# 0.71 rather than 0.62 data units: the pitch is specified in data units and the
# panel now carries 8.35 of them instead of 7.35, so this is the number that
# keeps the same ~2pt of white between the printed rows as before.
lab_y = nudge([v for _, v, _ in ends], 0.71)
for (h, yv, sh), yy in zip(ends, lab_y):
    # Leader in its own line's colour, as in panel (b): the grey hairline it used
    # to be belonged to no series in particular.  Nothing here can be mistaken
    # for a line continuing -- the stub is a few points long, it already starts
    # clear of the endpoint marker, and this panel carries no zero rule -- so it
    # keeps its solid stroke rather than panel (b)'s dotted one.
    axc.plot([1.03, 1.10], [yv, yy], lw=0.7, color=H_COL[h], zorder=2)
    # Printed as a share, as on panels (a) and (b), and one space tighter: the
    # percentage form overran the panel's own shaded box ("-123%" straddled the
    # grey/white edge).  Both changes only remove width.
    axc.text(1.13, yy, f"$h{{=}}{h}$  " + minus(f"{sh:+.2f}"), fontsize=9,
             va="center", ha="left", color=VERM_TXT)
axc.set_xlim(-0.20, 2.15)
# Head-room above the highest drawn point, so the in-panel note clears every line
# and label now that it is set at the house minimum of 9pt rather than 8.6.  The
# floor is zero, not the 1.0 it used to be: this panel carries the same axis
# label as panel (b) directly above it, and panel (b) is anchored on a zero rule,
# so a truncated floor here drew these slopes as full-height climbs and invited
# the reader to compare them with (b)'s.  Same axes box, same bounding box; the
# rise is still the panel's point and is still labelled on the panel.
axc.set_ylim(0.0, 8.35)
axc.set_xticks([0, 1])
axc.set_xticklabels(["control", "masked"], fontsize=9)
axc.set_ylabel("M1 increment,\nQLIKE rel. %", fontsize=9)
axc.set_yticks([0, 2, 4, 6])
# "a registered arm", not just "registered": the note beside this panel counts
# executed ARMS (zero of them), and naming the unit is what stops a reader taking
# the three series drawn here for those arms.
axc.text(-0.16, 8.15, "not a registered arm;\nmasking raises the increment",
         fontsize=9, color=GREY, ha="left", va="top", linespacing=1.35)
panel(axc, "c", dy=1.045)
axc.text(0.115, 1.045, "Long-form same-lineage annex (exploratory)",
         transform=axc.transAxes, fontsize=9, color=INK, ha="left",
         va="bottom", fontweight="semibold")

# The note now carries ONLY the long-form masking gate, i.e. the audit of the
# treatment panel (c) is drawn from and the one thing on this artefact that no
# other line of it and no caption states.  Everything else the longer note used
# to carry was already said twice: the registered/exited arm count is in the
# caption and drawn as panel (a)'s own empty row, the event-driven gate is quoted
# in the section text that introduces the figure, and panel (c)'s "descriptive
# only" caveat now sits in the caption, where a reader of the page still meets it.
# Panel (c)'s own "not registered; masking raises the increment" label is
# untouched: the inconvenient direction stays on the panel.
note_c = (
    f"Long-form masking gate. Panel (c)'s channel closed with "
    f"{int((ARM_LF.status == 'executed').sum())} executed arms although its "
    f"gate ran: {G2_LF['any_mask_pct']:.0f}% of {G2_LF['docs']:,} documents "
    f"carry at least one mask, {G2_LF['masked_char_pct']:.2f}% of characters "
    f"are masked, own-ticker leakage {G2_LF['own_ticker_pct']:.2f}%, "
    f"own-name tokens {G2_LF['own_name_pct']:.2f}%.")

# Same words, same wrap, same column; recessive ink and a hairline of its own, so
# the gate audit reads as the basis for panel (c) rather than as more of panel
# (c).  Dropped a hair below the panel title's baseline so the rule clears it.
_posN = axnote.get_position()
note(fig, _posN.x0, _posN.y0 + 0.965 * _posN.height,
     "\n".join(textwrap.wrap(note_c, 52)), width=_posN.width)

# --------------------------------------------------------------- type-size gate
# supp_style declares the system as ">= 9pt text at 1:1 inclusion scale".  The
# artefact ships 6.46in wide, i.e. about 1:1 in the supplement's text block, so
# every non-empty string on it must be at least 9pt.  Checked, not asserted in a
# comment: the build aborts if any string is set smaller.
small = sorted({round(t.get_fontsize(), 2) for t in fig.findobj(plt.Text)
                if t.get_text().strip() and t.get_fontsize() < 9})
if small:
    sys.exit(f"TYPE GATE FAIL - on-artefact strings below 9pt: {small}")

finish(fig, "F7_anonymisation_price")
