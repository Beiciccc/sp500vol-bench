"""Appendix figure FP1 — the prompted arms, family by family.

Four things the cross-family tables answer only cell by cell:

  (a) which prompted runs are healthy enough to be evidence at all
      (the registered screen: variance-unit QLIKE < 4 AND modal share of the
      two-decimal forecast < 60 % at every horizon);
  (b) what each family's increment is over the recalibrated HAR reference and
      over the firm-identity-augmented reference, on the one channel every
      family ran (8-K);
  (c) how much of the replication family's own edge a ZERO-CONTENT date+ticker
      prompt reproduces;
  (d) what the full text still adds once that same-model probe is inside the
      reference.

Panels (c) and (d) are the 70B's own identity probe — committed evidence
(`crossfamily_llama70_probe.csv`) that the report has nowhere reported.

Sources
-------
results/tables/crossfamily_llama70.csv        health + increments: qwen, yi, phi4, llama70
results/tables/crossfamily_llama70_ens.csv    llama70 3-seed ensemble rows
results/tables/crossfamily_gemma27.csv        health + increments + Holm: gemma27, mistral24
results/tables/crossfamily_mistral24.csv      cross-check of the mistral ensemble rows
results/tables/crossfamily_llama70_probe.csv  the zero-content probe and the
                                              text-beyond-identity readout
"""
import os
import sys

import numpy as np
import pandas as pd

ANALYSIS = "scripts/analysis"
sys.path.insert(0, ANALYSIS)

import textwrap

import diss_style as ds
import matplotlib.pyplot as plt
from supp_style import (
    BLUE,
    GREEN,
    GREY,
    INK,
    INK2,
    LIGHT,
    RULE,
    TAB,
    VERM,
    VERM_TXT,
    apply_style,
    gate,
)

# ---------------------------------------------------------------- evidence
d70 = pd.read_csv(os.path.join(TAB, "crossfamily_llama70.csv"))
ens = pd.read_csv(os.path.join(TAB, "crossfamily_llama70_ens.csv"))
g27 = pd.read_csv(os.path.join(TAB, "crossfamily_gemma27.csv"))
m24 = pd.read_csv(os.path.join(TAB, "crossfamily_mistral24.csv"))
prb = pd.read_csv(os.path.join(TAB, "crossfamily_llama70_probe.csv"))

HEALTH_COLS = ["disc", "family", "h", "qlike_var", "mode_share_pct"]
health = pd.concat([
    d70[HEALTH_COLS],
    g27[g27.family.isin(["gemma27_bf16", "mistral24_bf16"])][HEALTH_COLS],
], ignore_index=True)

QLIKE_CEIL, MODAL_CEIL = 4.0, 60.0


def run_health(fam, disc):
    r = health[(health.family == fam) & (health.disc == disc)]
    return (r.qlike_var.min(), r.qlike_var.max(),
            r.mode_share_pct.min(), r.mode_share_pct.max())


def passes(fam, disc):
    qlo, qhi, mlo, mhi = run_health(fam, disc)
    return (qhi < QLIKE_CEIL) and (mhi < MODAL_CEIL)


RUNS = [                       # in the order Appendix C prints them
    ("Qwen3-32B", "qwen3_32b", "event_driven", "8-K"),
    ("Qwen3-32B", "qwen3_32b", "long_form", "10-K/Q"),
    ("Llama-3.1-70B", "llama70_awq", "event_driven", "8-K"),
    ("Yi-1.5-34B", "yi_34b", "event_driven", "8-K"),
    ("Yi-1.5-34B", "yi_34b", "long_form", "10-K/Q"),
    ("Phi-4-14B", "phi4_14b", "event_driven", "8-K"),
    ("Mistral-24B", "mistral24_bf16", "event_driven", "8-K"),
    ("Gemma-3-27B", "gemma27_bf16", "event_driven", "8-K"),
]

PROBE_FAMILIES = ["llama70_awq", "yi_34b", "phi4_14b",
                  "mistral24_bf16", "gemma27_bf16"]
n_probe_fail = sum(
    0 if any(passes(f, d) for d in health[health.family == f].disc.unique())
    else 1 for f in PROBE_FAMILIES)


# ------------------------------------------------------- 8-K increment rows
def rows_8k(df, fam):
    return df[(df.disc == "event_driven") & (df.family == fam)].sort_values("h")


INC = []
_q = rows_8k(g27, "qwen3_32b")
INC.append(("Qwen3-32B  (anchor)", _q.rel_har.values, _q.p_har.values,
            _q.p_har_holm.values, _q.rel_firm.values, _q.p_firm.values,
            _q.p_firm_holm.values, True))
_l = rows_8k(ens, "llama70_awq_ens3")
INC.append(("Llama-3.1-70B (ens. 3)", _l.rel_har.values, _l.p_har.values,
            _l.p_har_holm.values, _l.rel_firm.values, _l.p_firm.values,
            _l.p_firm_holm.values, True))
_g = rows_8k(g27, "gemma27_ens3")
INC.append(("Gemma-3-27B (ens. 3)", _g.rel_har.values, _g.p_har.values,
            _g.p_har_holm.values, _g.rel_firm.values, _g.p_firm.values,
            _g.p_firm_holm.values, False))
_m = rows_8k(m24, "mistral24_ens3")
INC.append(("Mistral-24B (ens. 3)", _m.rel_har.values, _m.p_har.values,
            _m.p_har_holm.values, _m.rel_firm.values, _m.p_firm.values,
            _m.p_firm_holm.values, False))
_y = rows_8k(d70, "yi_34b")
INC.append(("Yi-1.5-34B", _y.rel_har.values, _y.p_har.values, None,
            _y.rel_firm.values, _y.p_firm.values, None, False))
_p = rows_8k(d70, "phi4_14b")
INC.append(("Phi-4-14B", _p.rel_har.values, _p.p_har.values, None,
            _p.rel_firm.values, _p.p_firm.values, None, False))

# --------------------------------------------------------------- the probe
pm = prb[prb.block == "probe_m1"].sort_values("h")
fa = prb[(prb.block == "fulltext_anchor_committed")
         & (prb.family == "llama70_awq_ens3")].sort_values("h")
sh = prb[prb.block == "probe_share"].sort_values("h")
bi = prb[prb.block == "beyond_identity"].sort_values("h")

# ------------------------------------------------------------------- gate
gate(
    {
        "n_probe_families": 5,
        "n_probe_health_fail": 4,
        "anchor_firm_triple": (0.45, 0.25, 0.20),
        "llama70_ens_firm_triple": (0.84, 0.64, 0.38),
        "probe_share_har_pct_wellid": (5, 7),
        "beyond_identity_retained_pct": (98, 98, 94),
        "probe_n_test": (25109, 25001, 24732),
    },
    {
        "n_probe_families": len(PROBE_FAMILIES),
        "n_probe_health_fail": n_probe_fail,
        "anchor_firm_triple": tuple(np.round(_q.rel_firm.values, 2)),
        "llama70_ens_firm_triple": tuple(np.round(_l.rel_firm.values, 2)),
        "probe_share_har_pct_wellid": tuple(
            int(round(v)) for v in
            sh[sh.denominator_well_identified == True].share_har_pct.values),
        "beyond_identity_retained_pct": tuple(
            int(round(100 * r / f)) for r, f in
            zip(bi.rel_pct.values, fa.rel_har.values, strict=False)),
        "probe_n_test": tuple(int(v) for v in pm.n_test.values),
    },
)

# ------------------------------------------------------------------ canvas
apply_style(9)
fig = plt.figure(figsize=ds.canvas(7.58))
gs = fig.add_gridspec(
    3, 2, height_ratios=[2.00, 2.45, 1.45], width_ratios=[1.0, 1.0],
    left=0.300, right=0.975, top=0.935, bottom=0.222,
    hspace=0.80, wspace=0.10)

axQ = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[0, 1], sharey=axQ)
axH = fig.add_subplot(gs[1, 0])
axF = fig.add_subplot(gs[1, 1], sharey=axH)
axP = fig.add_subplot(gs[2, 0])
axB = fig.add_subplot(gs[2, 1])

HFMT = {5: ("o", 5.2), 10: ("s", 4.6), 20: ("^", 5.4)}

# =========================================================== (a) the screen
ypos = np.arange(len(RUNS))[::-1]
for y, (lab, fam, disc, chan) in zip(ypos, RUNS, strict=False):
    qlo, qhi, mlo, mhi = run_health(fam, disc)
    ok = (qhi < QLIKE_CEIL) and (mhi < MODAL_CEIL)
    col = BLUE if ok else VERM
    # Pattern as well as hue: a run that fails the screen is drawn broken, so
    # pass and fail stay apart in greyscale.  A run can fail on the other
    # sub-panel's term, so the verdict is not recoverable from position alone.
    for ax, (lo, hi) in ((axQ, (qlo, qhi)), (axM, (mlo, mhi))):
        ax.plot([lo, hi], [y, y], color=col, lw=3.0, solid_capstyle="butt",
                ls="-" if ok else (0, (2.1, 1.5)), zorder=3)
        ax.plot([lo, hi], [y, y], marker="|", ms=6, color=col, lw=0, zorder=4)

axQ.plot([3.66], [ypos[-1] - 1], marker="D", ms=4.2, color=GREY, zorder=4)
axM.plot([45.2], [ypos[-1] - 1], marker="D", ms=4.2, color=GREY, zorder=4)

axQ.set_yticks(list(ypos) + [ypos[-1] - 1])
axQ.set_yticklabels([f"{lab}  ·  {chan}" for lab, _, _, chan in RUNS]
                    + ["Gemma-3-27B  ·  pilot"])
axQ.set_ylim(ypos[-1] - 1.7, ypos[0] + 0.8)

axQ.set_xscale("log")
axQ.set_xlim(0.3, 12)
axQ.set_xticks([0.5, 1, 2, 4, 8])
axQ.set_xticklabels(["0.5", "1", "2", "4", "8"])
# The shaded band is the REJECTED side of the ceiling, not the accepted one.
# It used to shade 0.3 -> 4 here and 18 -> 60 on the sub-panel beside it, i.e.
# the pass region, while panel (b) shades a row to mean the run FAILED this
# screen and panel (d) fills a bar with the same tint to mean the Holm test did
# not reject.  One tint carried "counts as evidence" in (a) and "does not count"
# in (b) and (d), two panels apart, with nothing saying they differed.  Shading
# the fail side instead makes LIGHT mean exactly one thing across all four
# panels; the ceiling itself is still carried by the dashed rule and its label,
# so nothing is lost, and a span inside an axes changes no geometry.
axQ.axvspan(QLIKE_CEIL, 12, color=LIGHT, alpha=0.45, zorder=0, lw=0)
axQ.axvline(QLIKE_CEIL, color=GREY, ls="--", lw=0.8)
axQ.set_xlabel("QLIKE, variance units")
# Both ceiling labels now hug their rule from the same side, the accepted side,
# which is also the unshaded one: this one used to sit to the right of its rule
# and its twin to the left, so the reader compared two sub-panels whose ceiling
# annotation was mirrored, and after the span was inverted this label would have
# been the only one set over grey.  It moves within the axes, into the empty
# strip above the top row, so no geometry moves.
axQ.text(QLIKE_CEIL * 0.93, ypos[0] + 0.42, "ceiling 4", fontsize=8.9,
         color=INK2, ha="right", va="center")

axM.set_xlim(18, 95)
axM.set_xticks([20, 40, 60, 80])
axM.axvspan(MODAL_CEIL, 95, color=LIGHT, alpha=0.45, zorder=0, lw=0)
axM.axvline(MODAL_CEIL, color=GREY, ls="--", lw=0.8)
axM.set_xlabel("modal share of the 2-dp forecast, %")
axM.text(MODAL_CEIL - 2.0, ypos[0] + 0.42, "ceiling 60 %", fontsize=8.9,
         color=INK2, ha="right", va="center")
plt.setp(axM.get_yticklabels(), visible=False)
axM.tick_params(axis="y", length=0)

for ax in (axQ, axM):
    for y in ypos:
        ax.axhline(y, color=LIGHT, lw=0.5, zorder=0)

# ================================================= (b) increments, 8-K only
n_inc = len(INC)
rows = np.arange(n_inc)[::-1]
OFF = {5: +0.25, 10: 0.0, 20: -0.25}

for ax, which in ((axH, "har"), (axF, "firm")):
    for y, rec in zip(rows, INC, strict=False):
        lab, rh, ph, hh, rf, pf, hf, healthy = rec
        rel = rh if which == "har" else rf
        praw = ph if which == "har" else pf
        pholm = hh if which == "har" else hf
        if not healthy:
            ax.axhspan(y - 0.45, y + 0.45, color=LIGHT, alpha=0.5, zorder=0,
                       lw=0)
        ax.axhline(y, color=LIGHT, lw=0.5, zorder=1)
        col = BLUE if healthy else VERM
        for k, h in enumerate((5, 10, 20)):
            mk, msz = HFMT[h]
            yy = y + OFF[h]
            ax.plot([0, rel[k]], [yy, yy], color=col, lw=0.8, alpha=0.55,
                    zorder=2)
            ax.plot([rel[k]], [yy], marker=mk, ms=msz, zorder=4,
                    mfc=col if praw[k] < 0.05 else "white", mec=col, mew=1.0)
            if pholm is not None and not np.isnan(pholm[k]) and pholm[k] < 0.05:
                ax.plot([rel[k]], [yy], marker="o", ms=msz + 4.4, zorder=3,
                        mfc="none", mec=GREY, mew=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_ylim(-0.72, n_inc - 0.28)

axH.set_yticks(rows)
axH.set_yticklabels([r[0] for r in INC])
axH.set_xlim(-0.95, 3.35)
axH.set_xticks([0, 1, 2, 3])
axH.set_xlabel("over the recalibrated HAR, %")
axF.set_xlim(-0.95, 1.85)
axF.set_xticks([0, 1])
axF.set_xlabel("over the firm-identity reference, %")
plt.setp(axF.get_yticklabels(), visible=False)
axF.tick_params(axis="y", length=0)

for h in (5, 10, 20):
    mk, msz = HFMT[h]
    axF.plot([], [], marker=mk, ms=msz, ls="none", color=GREY,
             label=f"$h={h}$")
# Lower right, not upper left: at the upper left the three entries sat on the
# zero rule, on the Qwen3-32B h=10 ring and on the Llama-3.1-70B h=20 connector.
# The lower right of this sub-panel carries no marker (every cell in the bottom
# three rows lies within |0.4| of zero).
axF.legend(loc="lower right", fontsize=8.9, handletextpad=0.3, borderpad=0.2,
           labelspacing=0.22, borderaxespad=0.5)

# ============================== (c) the replication family's own zero probe
hs = [5, 10, 20]
yb = np.arange(3)[::-1]
full_har = fa.rel_har.values
probe_har = pm.rel_har.values
shares = sh.share_har_pct.values
wellid = sh.denominator_well_identified.values

axP.barh(yb, full_har, height=0.50, color="white", edgecolor=BLUE, lw=1.2,
         zorder=3)
axP.barh(yb, probe_har, height=0.50, color=VERM, edgecolor=VERM, lw=0,
         zorder=4)
for y, f, s, ok in zip(yb, full_har, shares, wellid, strict=False):
    axP.text(f + 0.06, y, (f"{s:.0f} %" if ok else f"{s:.0f} %*"),
             va="center", ha="left", fontsize=8.9,
             color=VERM_TXT if ok else GREY)
axP.set_yticks(yb)
axP.set_yticklabels([f"$h={h}$" for h in hs])
axP.set_xlim(0, 2.30)
axP.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
axP.set_xlabel("over the recalibrated HAR, %")

axB_full = bi.rel_pct.values
bp = bi.p_holm.values
retained = 100 * axB_full / full_har
axB.barh(yb, axB_full, height=0.50, edgecolor=GREY, lw=0.6, zorder=3,
         color=[GREEN if q < 0.05 else LIGHT for q in bp])
for y, b, q, r in zip(yb, axB_full, bp, retained, strict=False):
    axB.text(b + 0.05, y, ("Holm .005" if q < 0.005 else
                           (f"Holm .{int(round(q*1000)):03d}" if q < 0.05
                            else f"Holm .{int(round(q*100)):02d} ns"))
             + f"  ·  {r:.0f} %",
             va="center", ha="left", fontsize=8.9, color=GREY)
axB.set_yticks(yb)
axB.set_yticklabels([f"$h={h}$" for h in hs])
axB.set_xlim(0, 2.30)
axB.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
axB.set_xlabel("over HAR + the zero-content forecast, %")

# ---------------------------------------------------------- panel headings
fig.canvas.draw()


def heading(ax, text, dy=0.016, x=0.004):
    """Marker and title, one call, one baseline.

    Rows (a) and (b) are a single heading over a pair of sub-panels that share a
    y axis, and their titles are as wide as the figure, so they sit flush with
    the figure's left edge.  Panels (c) and (d) are independent, and they used to
    share ONE padded string: "(d)  What the full text adds beyond it" began at
    0.397 of the figure width, which is inside panel (c)'s span (0.300-0.621), so
    the marker for the fourth panel printed above the third panel's bars.  Each
    now starts at its own panel's left spine.

    The titles wrap rather than run: a panel is 141.2 pt wide, "(c) ..." sets at
    156.1 pt and "(d) ..." at 147.5 pt on one line, and finish() writes with
    bbox_inches="tight", so a title pushed past its panel would widen the page
    and shrink every glyph in the figure.  Moving a block inward into whitespace
    the box already contains is free; moving one outward is not.  Both wrap to
    two lines and `va="bottom"` keeps their last lines on one baseline.
    """
    fig.text(x, ax.get_position().y1 + dy, text, fontsize=9.2,
             color=INK, ha="left", va="bottom")


heading(axQ, "(a)  The registered forecaster-health screen: a run passes only "
             "if it clears both ceilings")
heading(axH, "(b)  Increment on the 8-K channel — the one channel every "
             "family ran")
heading(axP, "(c)  The 70B's own\nzero-content probe",
        x=axP.get_position().x0)
heading(axB, "(d)  What the full text\nadds beyond it",
        x=axB.get_position().x0)

# ------------------------------------------------------------- foot matter
NOTE_SRC = (
    "(a) bars span the run's three horizons; blue = the run passes the screen, "
    "orange = it fails; the diamond is the registered 2,000-filing Gemma pilot, "
    "which cleared both terms.  (b) filled = raw p < .05; grey ring = also below "
    ".05 under that family's own committed Holm family, and those families "
    "differ and are never pooled; a shaded row is a run that fails (a).  "
    "(c) outline = the committed 3-seed full-text arm, solid = the identical "
    "prompt with the filing text deleted; * marks a cell whose denominator falls "
    "below the 1 % stable-denominator rule - shown, not quotable."
)
# Recessive ink and a hairline, so a reader can tell at a glance which text is
# the figure's argument and which is its basis statement.  Every word is kept:
# this block carries the pass/fail encoding, the pilot's denominator, the Holm
# families that are never pooled, and the "shown, not quotable" clause.
# Deliberately NOT ds.note(): that helper hardcodes linespacing=1.32 and this
# block is set at 1.34, so calling it would tighten the leading.  Its two
# devices -- INK2 and a RULE hairline -- are taken by hand instead.
_note = fig.text(0.004, 0.006, textwrap.fill(NOTE_SRC, 92), fontsize=8.9,
                 color=INK2, ha="left", va="bottom", linespacing=1.34)

# The rule is measured from the block's own rendered box and laid in the gap the
# canvas already has between the panel (c)/(d) axis labels and the note, so it
# reaches past nothing and moves nothing: under bbox_inches="tight" the page is
# the content's bounding box, and this adds no content outside it.
fig.canvas.draw()
_nb = _note.get_window_extent(renderer=fig.canvas.get_renderer())
_nb = _nb.transformed(fig.transFigure.inverted())
_ry = _nb.y1 + 0.020
fig.lines.append(plt.Line2D([_nb.x0, _nb.x1], [_ry, _ry],
                            transform=fig.transFigure, color=RULE,
                            linewidth=0.5, zorder=0.5))

ds.finish(fig, "FP1_prompted_family_panel", max_render_pt=595.0,
          note="appendix figure: prompted-arm health screen, cross-family "
               "increments on 8-K, and the 70B zero-content probe")
