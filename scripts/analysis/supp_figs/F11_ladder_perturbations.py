"""F11 -- What moves the reference ladder: unit convention, label proxy, panel
composition and weight scheme.

Four perturbations applied to the SAME frozen forecasts. Every rung of the
reference ladder is redrawn under each perturbation, favourable or not.

Evidence files read (all repo-relative, all committed):
  results/tables/control_intersection_ensemble.csv  -- volatility-unit ladder on
        the declared primary basis (46 raw / 38 Holm / 38 genuine; pool 9;
        firm 8; full conjunction 0)
  results/tables/variance_unit_cascade.csv          -- the same frozen forecasts
        rescored under Patton-proxy-robust variance-unit QLIKE
  results/tables/firm_identity_ensemble.csv         -- volatility-unit Holm p for
        the event-driven C6 residual against the firm-identity reference
  results/tables/rangebased_cascade.csv             -- Parkinson / Garman-Klass
        relabelling of the whole ladder (old_*, pk_*, gk_* columns)
  results/tables/public_variant_cascade.csv         -- panels A / B / C
  results/tables/public_variant_cascade.md          -- the coverage strata, which
        live in the prose of that file and nowhere else
  results/tables/deployable_combiner.csv            -- fixed validation-frozen vs
        expanding deployable combiner weights over 75 cells
  results/tables/valwindow_sensitivity.csv          -- combiner refit on the calm
        (2021) vs COVID (2020) half of validation
  results/tables/maximal_reference_ensemble.csv     -- used only to establish that
        the weight-window arm's reference is the maximal five-price pool on the
        seed-2026 text basis (gated below)

Main-text sentences this figure substantiates:
  06_results.tex: "0 of 69 survive in both unit conventions, with disjoint
     survivor sets; under range-based label proxies and a licence-free
     survivorship subsample the composition moves but not the substance: none of
     the per-perturbation survivors clears all label proxies".
  06_results.tex: "Deployable expanding weights erase most of the grid (6 of 69
     genuine, vs. 36 of 69 for frozen weights under Holm in the same deployable
     family) yet keep event-driven C6 h=5/10 (+1.19/+0.98%, Holm .000/.033,
     placebo clean; h=20 null)."
  06_results.tex: "Variance units keep raw 3 of 3 but Holm 1 of 3."
  10_limitations.tex: "38 of 69 genuine in volatility units, 20 in variance, 19
     under a Parkinson relabeling with an empty cross-proxy conjunction".
  07_ablations.tex: "one dictionary cell newly clears the conjunction, and a
     covered-row CRSP panel isolates that flip to survivorship composition".
  05_protocol.tex: "the 38->20 variance-unit tightening".

Two adversarial repairs are folded in, and where they pull in different
directions the more conservative reading is taken:
  * the primary rung is drawn on the PLACEBO-GATED GENUINE convention in every
    block (38 / 19 / 21 for the label proxies, matching 10_limitations.tex, and
    38 / 32 / 31 for the panels), with the Holm-only count carried as a separate
    tick.  The programme text quotes the Holm-only numbers for the panel block
    (38 / 35 / 36); both are shown rather than choosing one, so no block asserts
    a primary count on a convention different from its neighbours.
  * the TEXT arms are frozen everywhere. The price side is not: checked
    against the sources rather than assumed, rangebased_cascade.md L3 refits
    "A2 + A6_shar ... on range-based features+labels via the committed
    fitting code", and public_variant_cascade.md L3 refits the same two on
    public features+labels in panel C (panel B only recalibrates them).
    Recalibration, the firm mean and the combiner weights are refit wherever
    the labels, the rows or the weight scheme change. Block (a) is the only
    block in which nothing at all is re-estimated: the loss alone changes.
    The closing note on the artefact states this; an earlier draft carried
    the false blanket claim that nothing is refitted anywhere.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.patheffects as _pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from supp_style import (
    BLUE,
    GREEN,
    GREY,
    INK,
    INK2,
    LIGHT,
    PURPLE,
    RULE,
    SKY,
    TAB,
    VERM,
    VERM_TXT,
    YELLOW,
    apply_style,
    finish,
    gate,
)

# --------------------------------------------------------------- evidence
CIE = pd.read_csv(os.path.join(TAB, "control_intersection_ensemble.csv"))
VU = pd.read_csv(os.path.join(TAB, "variance_unit_cascade.csv"))
FIE = pd.read_csv(os.path.join(TAB, "firm_identity_ensemble.csv"))
RB = pd.read_csv(os.path.join(TAB, "rangebased_cascade.csv"))
PV = pd.read_csv(os.path.join(TAB, "public_variant_cascade.csv"))
DC = pd.read_csv(os.path.join(TAB, "deployable_combiner.csv"))
VW = pd.read_csv(os.path.join(TAB, "valwindow_sensitivity.csv"))
MRE = pd.read_csv(os.path.join(TAB, "maximal_reference_ensemble.csv"))
with open(os.path.join(TAB, "public_variant_cascade.md"), encoding="utf-8") as fh:
    PV_MD = fh.read()

# --- (a) unit convention -------------------------------------------------
a_gen = [int(CIE.primary_genuine.sum()), int(VU.primary_var_genuine.sum())]
a_holm = [int(CIE.primary_holm.sum()), int(VU.primary_var_holm.sum())]
a_firm = [int(CIE.firm_holm.sum()), int(VU.firm_var_adds_holm.sum())]
a_pool = [int(CIE.maximal_holm.sum()), int(VU.max_var_adds_holm.sum())]
a_conj = [int(CIE.AND_full_holm.sum()), int(VU.AND_full_var_holm.sum())]

res_vol = FIE[(FIE.disc == "event_driven") & (FIE.model == "C6_llmtext")].sort_values("h")
res_var = VU[(VU.disc == "event_driven") & (VU.model == "C6_llmtext")].sort_values("h")
res_holm_vol = res_vol.holm_p.to_numpy()
res_holm_var = res_var.firm_var_holm.to_numpy()

# --- (b) label proxy -----------------------------------------------------
b_gen = [int(RB.old_genuine.sum()), int(RB.pk_genuine.sum()), int(RB.gk_genuine.sum())]
b_holm = [int(RB.old_primary_detect.sum()), int(RB.pk_primary_detect.sum()),
          int(RB.gk_primary_detect.sum())]
b_firm = [int(RB.old_firm_detect.sum()), int(RB.pk_firm_detect.sum()),
          int(RB.gk_firm_detect.sum())]
b_pool = [int(RB.old_pool_detect.sum()), int(RB.pk_pool_detect.sum()),
          int(RB.gk_pool_detect.sum())]
b_conj = [int(RB.old_conj.sum()), int(RB.pk_conj.sum()), int(RB.gk_conj.sum())]
b_mde = [float(RB.old_mde.median()), float(RB.pk_mde.median()), float(RB.gk_mde.median())]
b_shrink = [100.0 * (m - b_mde[0]) / b_mde[0] for m in b_mde]


def _cells(frame, flag):
    """Compact 'channel model h' labels for the cells carrying a boolean flag."""
    short = {"event_driven": "8-K", "long_form": "10-K/Q"}
    return ["%s %s h%d" % (short[r.disc], r.model, int(r.h))
            for r in frame[frame[flag]].itertuples()]


def _compact(cells):
    """Fold cells that share a channel and model into one 'h5 and h10' phrase."""
    heads = {c.rsplit(" h", 1)[0] for c in cells}
    if len(heads) == 1 and len(cells) > 1:
        hs = [c.rsplit(" h", 1)[1] for c in cells]
        return "%s h%s" % (cells[0].rsplit(" h", 1)[0], " and h".join(hs))
    return "; ".join(cells)


pk_conj_cells = _cells(RB, "pk_conj")
gk_conj_cells = _cells(RB, "gk_conj")
# The committed survivor set is empty, so the three-way intersection is empty
# too; the two non-empty sets are also disjoint from each other.
xproxy = set(_cells(RB, "old_conj")) & set(pk_conj_cells) & set(gk_conj_cells)

# --- (c) panel composition ----------------------------------------------
c_gen = [int(PV.a_genuine.sum()), int(PV.b_genuine.sum()), int(PV.c_genuine.sum())]
c_holm = [int(PV.a_primary_detect.sum()), int(PV.b_primary_detect.sum()),
          int(PV.c_primary_detect.sum())]
c_firm = [int(PV.a_firm_detect.sum()), int(PV.b_firm_detect.sum()),
          int(PV.c_firm_detect.sum())]
c_pool = [int(PV.a_pool_detect.sum()), int(PV.b_pool_detect.sum()),
          int(PV.c_pool_detect.sum())]
c_conj = [int(PV.a_conj.sum()), int(PV.b_conj.sum()), int(PV.c_conj.sum())]
c_conj_cells = _cells(PV, "b_conj")

_cov = re.search(
    r"train ([\d.]+)% / val ([\d.]+)% / test ([\d.]+)% of modelled rows; "
    r"benchmark-row clean coverage ([\d.]+)%; exit-firm rows ([\d.]+)% vs active ([\d.]+)%",
    PV_MD)
cov_train, cov_val, cov_test, cov_bench, cov_exit, cov_active = (
    float(g) for g in _cov.groups())

# --- (d) weight scheme ---------------------------------------------------
d_fixed75, d_exp75 = int(DC.genuine_fixed.sum()), int(DC.genuine_exp.sum())
_p = DC[DC.in_primary_grid]
d_fixed69, d_exp69 = int(_p.genuine_fixed.sum()), int(_p.genuine_exp.sum())
c6 = DC[(DC.disc == "event_driven") & (DC.model == "C6_llmtext")].sort_values("h")
# row basis of the middle panel, distinct from the firm-reference residual in
# block (a) (23,855 / 22,785 / 22,318 merged-grid rows)
c6_n = [int(v) for v in c6.n_test]
res_n = [int(v) for v in res_vol.n_test]

VWp = VW.pivot_table(index=["disc", "model", "h"], columns="window",
                     values="rel_impr_pct")
vw_diff = (VWp["calm_2021"] - VWp["covid_2020"]).to_numpy()
vw_t, vw_p = stats.ttest_rel(VWp["calm_2021"], VWp["covid_2020"])
# The DIFFERENCE is positive on average; the increment itself is negative
# under either half, which the difference alone would let a reader deny.
vw_calm_mean = float(VWp["calm_2021"].mean())
vw_covid_mean = float(VWp["covid_2020"].mean())
VW_CALM_TXT = "%+.3fpp" % vw_calm_mean
VW_COVID_TXT = "%+.3fpp" % vw_covid_mean
# The alt_2018_19 arm is in-sample for every text model and is deliberately not
# drawn (valwindow_sensitivity.md: "must not be read as a regime test").

# The weight-window arm's reference is the maximal five-price pool on the
# seed-2026 text basis: its committed_val rows reproduce that column exactly.
_join = pd.concat([VWp["committed_val"],
                   MRE.set_index(["disc", "model", "h"])["rel_impr_pct_maximal_s26"]],
                  axis=1, join="inner")
vw_ref_maxdiff = float((_join["committed_val"]
                        - _join["rel_impr_pct_maximal_s26"]).abs().max())

# ------------------------------------------------------------------- gate
gate(
    {"vol_primary_genuine": 38, "var_primary_genuine": 20,
     "vol_primary_holm": 38, "var_primary_holm": 21,
     "vol_firm_holm": 8, "var_firm_holm": 3,
     "vol_pool_holm": 9, "var_pool_holm": 2,
     "vol_conj_holm": 0, "var_conj_holm": 0,
     "residual_holm_vol": 3, "residual_holm_var": 1,
     "cc_gen": 38, "pk_gen": 19, "gk_gen": 21,
     "cc_holm": 38, "pk_holm": 21, "gk_holm": 23,
     "cc_firm": 8, "pk_firm": 7, "gk_firm": 8,
     "cc_pool": 9, "pk_pool": 15, "gk_pool": 15,
     "cc_conj": 0, "pk_conj": 1, "gk_conj": 2,
     "mde_cc": 0.823, "mde_pk": 0.913, "mde_gk": 1.018,
     "cross_proxy_conjunction": 0,
     "A_gen": 38, "B_gen": 32, "C_gen": 31,
     "A_holm": 38, "B_holm": 35, "C_holm": 36,
     "A_firm": 8, "B_firm": 9, "C_firm": 9,
     "A_pool": 9, "B_pool": 11, "C_pool": 13,
     "A_conj": 0, "B_conj": 1, "C_conj": 1,
     "cov_bench": 79.83, "cov_exit": 31.5, "cov_active": 97.3,
     "deploy_fixed_75": 36, "deploy_exp_75": 7,
     "deploy_fixed_69": 36, "deploy_exp_69": 6,
     "vw_cells": 69, "vw_calm_higher": 37,
     "vw_mean_pp": 0.433, "vw_median_pp": 0.064,
     "vw_t": 1.12, "vw_p": 0.268,
     "vw_calm_mean_pp": -0.087, "vw_covid_mean_pp": -0.519,
     "vw_reference_is_maximal_pool_s26": True,
     "c6_deploy_rows": [25109, 25001, 24732],
     "residual_rows": [23855, 22785, 22318]},
    {"vol_primary_genuine": a_gen[0], "var_primary_genuine": a_gen[1],
     "vol_primary_holm": a_holm[0], "var_primary_holm": a_holm[1],
     "vol_firm_holm": a_firm[0], "var_firm_holm": a_firm[1],
     "vol_pool_holm": a_pool[0], "var_pool_holm": a_pool[1],
     "vol_conj_holm": a_conj[0], "var_conj_holm": a_conj[1],
     "residual_holm_vol": int((res_holm_vol < 0.05).sum()),
     "residual_holm_var": int((res_holm_var < 0.05).sum()),
     "cc_gen": b_gen[0], "pk_gen": b_gen[1], "gk_gen": b_gen[2],
     "cc_holm": b_holm[0], "pk_holm": b_holm[1], "gk_holm": b_holm[2],
     "cc_firm": b_firm[0], "pk_firm": b_firm[1], "gk_firm": b_firm[2],
     "cc_pool": b_pool[0], "pk_pool": b_pool[1], "gk_pool": b_pool[2],
     "cc_conj": b_conj[0], "pk_conj": b_conj[1], "gk_conj": b_conj[2],
     "mde_cc": round(b_mde[0], 3), "mde_pk": round(b_mde[1], 3),
     "mde_gk": round(b_mde[2], 3),
     "cross_proxy_conjunction": len(xproxy),
     "A_gen": c_gen[0], "B_gen": c_gen[1], "C_gen": c_gen[2],
     "A_holm": c_holm[0], "B_holm": c_holm[1], "C_holm": c_holm[2],
     "A_firm": c_firm[0], "B_firm": c_firm[1], "C_firm": c_firm[2],
     "A_pool": c_pool[0], "B_pool": c_pool[1], "C_pool": c_pool[2],
     "A_conj": c_conj[0], "B_conj": c_conj[1], "C_conj": c_conj[2],
     "cov_bench": cov_bench, "cov_exit": cov_exit, "cov_active": cov_active,
     "deploy_fixed_75": d_fixed75, "deploy_exp_75": d_exp75,
     "deploy_fixed_69": d_fixed69, "deploy_exp_69": d_exp69,
     "vw_cells": int(vw_diff.size), "vw_calm_higher": int((vw_diff > 0).sum()),
     "vw_mean_pp": round(float(vw_diff.mean()), 3),
     "vw_median_pp": round(float(np.median(vw_diff)), 3),
     "vw_t": round(float(vw_t), 2), "vw_p": round(float(vw_p), 3),
     "vw_calm_mean_pp": round(vw_calm_mean, 3),
     "vw_covid_mean_pp": round(vw_covid_mean, 3),
     "vw_reference_is_maximal_pool_s26": bool(vw_ref_maxdiff < 1e-9),
     "c6_deploy_rows": c6_n, "residual_rows": res_n})

# ------------------------------------------------------------------ layout
apply_style(9)
FW, FH = 6.4, 8.61
fig = plt.figure(figsize=(FW, FH))

RUNGS = ["primary", "firm\nidentity", "maximal\npool", "all\nthree"]
YMAX = 78.0
COLX, AXW, AXH = 0.95, 2.85, 0.94   # count block: left edge, width, height
CMPX, CMPW = 4.95, 1.15             # companion block: left edge, width


def axes_in(x, y, w, h):
    """Add axes from inch coordinates measured down from the top-left corner."""
    return fig.add_axes([x / FW, 1.0 - (y + h) / FH, w / FW, h / FH])


def fig_text(x, y, text, **kw):
    kw.setdefault("fontsize", 9)
    kw.setdefault("color", GREY)
    return fig.text(x / FW, 1.0 - y / FH, text, ha="left", va="top", **kw)


WRAP_IN = 6.40        # inches of text measure for the wrapped note blocks


def fig_lines(x, y, text, step=0.15, **kw):
    """Wrap `text` against the renderer and draw it as consecutive lines.

    Hand-counted line breaks drift the moment a formatted number changes
    width, and a line that overruns the canvas silently widens the tight
    bounding box on save, so the wrap is measured rather than guessed.
    Returns the y just past the last line.
    """
    r = fig.canvas.get_renderer()
    probe = fig.text(0.0, 0.0, "", fontsize=kw.get("fontsize", 9))
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word) if cur else word
        probe.set_text(trial)
        too_wide = probe.get_window_extent(renderer=r).width / fig.dpi > WRAP_IN
        if cur and too_wide:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    probe.remove()
    for i, ln in enumerate(lines):
        fig_text(x, y + i * step, ln, **kw)
    return y + len(lines) * step


RULE_IN = 0.42 * FW   # hairline length, the `note()` default rule width


def fig_rule(y, x=0.62, w=RULE_IN):
    """A RULE hairline divider, drawn in the same inch-from-the-top frame.

    The four blocks stack with their apparatus notes between them, and nothing
    used to mark where one block ended and the next began. The divider goes in
    the gap that already exists between a note's last line and the next block
    title, so it adds no geometry: `finish()` saves on a tight bounding box, and
    that box is already set by the note lines, which run wider and lower than
    this. It cannot go directly above a note, where `note()` would put it --
    there the two-line x tick labels leave 0.003 in of clearance.
    """
    yy = 1.0 - y / FH
    fig.add_artist(Line2D([x / FW, (x + w) / FW], [yy, yy],
                          transform=fig.transFigure, color=RULE,
                          linewidth=0.5, zorder=0.5))


def fig_runs(x, y, runs, **kw):
    """Draw one line as consecutive coloured runs on a single baseline.

    Two of the note blocks broke colour mid-sentence: a basis statement and the
    finding that follows it were split at a hand-set line break, so the finding
    started in one ink and ended in another. The line breaks cannot move -- the
    float is within 6 pt of its height cap, so re-wrapping a block into one more
    line would fail the gate -- and every word has to stay. Measuring the runs
    against the renderer therefore lets the colour boundary fall at the sentence
    boundary while the line keeps the width and the wording it already had.
    """
    r = fig.canvas.get_renderer()
    probe = fig.text(0.0, 0.0, "", fontsize=kw.get("fontsize", 9))

    def measure(s):
        probe.set_text(s)
        return probe.get_window_extent(renderer=r).width / fig.dpi

    space = measure("x x") - measure("xx")
    texts, seen = [], []
    for text, colour in runs:
        dx = (measure(" ".join(seen)) + space) if seen else 0.0
        texts.append(fig_text(x + dx, y, text, color=colour, **kw))
        seen.append(text)
    probe.remove()
    return texts


def _strip(ax):
    ax.tick_params(length=2.5)
    ax.spines["left"].set_visible(False)


def rung_block(ax, names, values, colours, hatches, holm_primary, groups=RUNGS):
    """Vertical grouped bars: groups are ladder rungs, bars are the arms.

    Vertical rather than horizontal bars, because the horizontal variant cannot
    give all twelve bars a legible 9pt count label inside a portrait page.
    """
    n = len(names)
    base = np.arange(len(groups), dtype=float)
    bw = 0.78 / n
    for i, name in enumerate(names):
        x = base - 0.39 + bw / 2.0 + i * bw
        v = np.array([values[k][i] for k in range(len(groups))], dtype=float)
        ax.bar(x, v, width=bw * 0.86, color=colours[i], hatch=hatches[i],
               edgecolor="white", linewidth=0.5, label=name, zorder=3)
        hv = holm_primary[i]
        for k, (xx, vv) in enumerate(zip(x, v, strict=False)):
            # on the primary rung the count sits clear of the Holm tick
            top = max(vv, hv) if k == 0 else vv
            ax.text(xx, top + 2.6, "%d" % vv, va="bottom", ha="center",
                    fontsize=9, color=GREY, zorder=4)
        ax.plot([x[0] - bw * 0.46, x[0] + bw * 0.46], [hv, hv], color=GREY,
                lw=1.3, zorder=5, solid_capstyle="butt")
    ax.axhline(69, color=LIGHT, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.set_xticks(base)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, YMAX)
    ax.set_xlim(-0.62, len(groups) - 0.38)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("cells, of 69", labelpad=2)
    ax.tick_params(length=2.5)
    ax.tick_params(axis="x", length=0)


def series_key(ax, x_in, y_in, ncol):
    """Series key placed on the block-title line, to the right of the title."""
    ax.legend(loc="upper left", bbox_to_anchor=(x_in / FW, 1.0 - y_in / FH),
              bbox_transform=fig.transFigure, ncol=ncol, handlelength=1.0,
              handletextpad=0.35, columnspacing=1.1, borderpad=0.0,
              borderaxespad=0.0, fontsize=9)


def companion_title(ax, text):
    ax.set_title(text, loc="left", fontsize=9, color=GREY, pad=3.5)


# ============================================================== block (a)
fig_text(0.62, 0.14, "(a)  Unit convention", fontweight="bold")
axA = axes_in(COLX, 0.38, AXW, AXH)
rung_block(axA, ["volatility unit  q(y, f)", "variance unit  q(y^2, f^2)"],
           [[a_gen[0], a_gen[1]], a_firm, a_pool, a_conj],
           [BLUE, SKY], [None, "///"], a_holm)
series_key(axA, 2.20, 0.13, 2)

axA2 = axes_in(CMPX, 0.38, CMPW, AXH)
companion_title(axA2, "8-K residual: Holm p")
yy = np.arange(3)[::-1].astype(float)
axA2.set_xscale("log")
axA2.set_xlim(8e-6, 4.0)
axA2.set_ylim(-1.75, 2.55)
# Survival is NOT left to which side of the rule a marker lands on. On this
# log axis the variance-unit h=5 point (Holm p .032) sits 0.19 decades from
# .05, about 2.8pt, while its glyph is 4.6pt: at 150 dpi the square straddles
# the rule. Fill therefore encodes Holm < .05 on both series, the surviving
# region is shaded, and the one variance-unit survivor carries its value.
axA2.axvspan(8e-6, 0.05, ymin=0.36, ymax=1.0, color=LIGHT, alpha=0.55,
             lw=0, zorder=0)
for a, b, y0 in zip(res_holm_vol, res_holm_var, yy, strict=False):
    axA2.plot([a, b], [y0, y0], color=GREY, lw=0.7, zorder=2)
for p_, y0 in zip(res_holm_vol, yy, strict=False):
    axA2.plot([p_], [y0], "o", ms=4.6, color=BLUE, mec=BLUE,
              mfc=BLUE if p_ < 0.05 else "white", mew=1.0, zorder=3)
for p_, y0 in zip(res_holm_var, yy, strict=False):
    axA2.plot([p_], [y0], "s", ms=4.6, color=SKY, mec=GREY,
              mfc=SKY if p_ < 0.05 else "white", mew=0.8, zorder=3)
axA2.plot([0.05, 0.05], [-0.2, 2.55], color=GREY, lw=0.8, ls=(0, (2, 2)),
          zorder=1)   # stops above the two annotation lines
axA2.annotate("%.3f" % res_holm_var[0], (res_holm_var[0], yy[0]),
              textcoords="offset points", xytext=(-6, 0), ha="right",
              va="center", fontsize=9, color=GREY, zorder=6,
              bbox=dict(fc="white", ec="none", pad=0.5))
axA2.set_xticks([1e-4, 1e-2, 1.0])
axA2.set_xticklabels(["1e-4", "0.01", "1"])
axA2.set_yticks(yy)
axA2.set_yticklabels(["h=5", "h=10", "h=20"])
axA2.text(0.075, 2.20, "0.05", fontsize=9, color=GREY, ha="left", va="center")
axA2.text(8e-6, -0.62, "filled = Holm < .05", fontsize=9, color=INK2,
          ha="left", va="center")   # the key is apparatus, not a data label
axA2.text(8e-6, -1.35, "survivors: %d of 3 -> %d of 3"
          % (int((res_holm_vol < 0.05).sum()), int((res_holm_var < 0.05).sum())),
          fontsize=9, color=VERM_TXT, ha="left", va="center")
axA2.tick_params(axis="y", length=0)
_strip(axA2)

A_END = fig_lines(0.62, 1.66,
                  "Holm-only primary counts (grey ticks): %d / %d; the bars "
                  "draw genuine %d / %d, the frozen table's variance-unit row."
                  % (a_holm[0], a_holm[1], a_gen[0], a_gen[1]), color=INK2)

# ============================================================== block (b)
fig_rule(1.845)
fig_text(0.62, 1.90, "(b)  Label proxy", fontweight="bold")
axB = axes_in(COLX, 2.14, AXW, AXH)
rung_block(axB, ["close-to-close", "Parkinson", "Garman-Klass"],
           [[b_gen[0], b_gen[1], b_gen[2]], b_firm, b_pool, b_conj],
           [BLUE, YELLOW, PURPLE], [None, "\\\\\\", "xxx"], b_holm)
series_key(axB, 1.92, 1.89, 3)

axB2 = axes_in(CMPX, 2.14, CMPW, AXH)
companion_title(axB2, "median MDE, rel %")
axB2.barh([2, 1, 0], b_mde, height=0.55, color=[BLUE, YELLOW, PURPLE],
          hatch=[None, "\\\\\\", "xxx"], edgecolor="white", linewidth=0.5, zorder=3)
for y0, v in zip([2, 1, 0], b_mde, strict=False):
    axB2.text(v + 0.05, y0, "%.3f" % v, va="center", ha="left", fontsize=9, color=GREY)
axB2.set_yticks([2, 1, 0])
axB2.set_yticklabels(["close-close", "Parkinson", "Garman-K."])
axB2.set_xlim(0, 1.62)
axB2.set_xticks([0, 0.5, 1.0])
axB2.set_ylim(-0.62, 2.62)
axB2.tick_params(axis="y", length=0)
_strip(axB2)

fig_text(0.62, 3.42, "Holm-only primary counts (grey ticks): %d / %d / %d."
         % (b_holm[0], b_holm[1], b_holm[2]), color=INK2)
fig_text(0.62, 3.57, "Cells clearing all three rungs: Parkinson %s; Garman-Klass %s;"
         % (_compact(pk_conj_cells), _compact(gk_conj_cells)), color=VERM_TXT)
# the survivor list ends mid-line, so the ink changes back there and not at the
# line break: "close-to-close none" closes the finding, the disjointness clause
# after it is the basis statement
fig_runs(0.62, 3.72,
         [("close-to-close none.", VERM_TXT),
          ("The three survivor sets are pairwise disjoint, so the "
           "cross-proxy conjunction is %d of 69." % len(xproxy), INK2)])

# ============================================================== block (c)
fig_rule(3.900)
fig_text(0.62, 3.96, "(c)  Panel composition", fontweight="bold")
axC = axes_in(COLX, 4.20, AXW, AXH)
rung_block(axC, ["A full panel", "B covered, CRSP", "C covered, public"],
           [[c_gen[0], c_gen[1], c_gen[2]], c_firm, c_pool, c_conj],
           [BLUE, GREEN, VERM], [None, "///", "..."], c_holm)
series_key(axC, 2.34, 3.95, 3)

axC2 = axes_in(CMPX, 4.20, CMPW, AXH)
companion_title(axC2, "coverage, % of rows")
strata = ["train", "validation", "test", "benchmark clean", "active firms",
          "exiting firms"]
covals = [cov_train, cov_val, cov_test, cov_bench, cov_active, cov_exit]
# No arm hue in a companion panel: green and vermillion are panels B and C in
# the key three inches to the left, and a green "active firms" bar invited the
# reader to take 97.3% for panel B's coverage. Grey carries the three audit
# strata, light grey the split that only gives them context.
cols = [LIGHT, LIGHT, LIGHT, GREY, GREY, GREY]
ypos = np.arange(len(strata))[::-1].astype(float)
axC2.barh(ypos, covals, height=0.62, color=cols, edgecolor="white", linewidth=0.5,
          zorder=3)
for y0, v in zip(ypos, covals, strict=False):
    # keep the value clear of the benchmark-clean reference line
    xt = v + 2.5 if abs(v - cov_bench) > 9.0 else cov_bench + 3.0
    axC2.text(xt, y0, "%.4g" % v, va="center", ha="left", fontsize=9, color=GREY)
_ref = axC2.axvline(cov_bench, color=GREY, lw=0.8, ls=(0, (2, 2)), zorder=4)
# The bars either side of this rule are now grey, so a grey dash on a grey fill
# disappeared where it crosses "active firms". A white casing restores it
# without spending one of the arm hues on a reference line.
_ref.set_path_effects([_pe.withStroke(linewidth=2.2, foreground="white")])
axC2.set_yticks(ypos)
axC2.set_yticklabels(strata)
axC2.set_xlim(0, 138)
axC2.set_xticks([0, 50, 100])
axC2.set_ylim(-0.62, len(strata) - 0.38)
axC2.tick_params(axis="y", length=0)
_strip(axC2)

fig_runs(0.62, 5.48,
         [("Holm-only primary counts (grey ticks): %d / %d / %d."
           % (c_holm[0], c_holm[1], c_holm[2]), INK2),
          ("Panels B and C both clear all three rungs at", VERM_TXT)])
fig_text(0.62, 5.63, "%s, already present in B: row composition, not label source."
         % c_conj_cells[0], color=VERM_TXT)

# ============================================================== block (d)
fig_rule(5.810)
fig_text(0.62, 5.87, "(d)  Weight scheme", fontweight="bold")
axD = axes_in(COLX, 6.11, 1.30, AXH)
dnames = ["fixed weights", "expanding weights"]
dvals = [[d_fixed75, d_exp75], [d_fixed69, d_exp69]]
dbase = np.arange(2, dtype=float)
for i, name in enumerate(dnames):
    x = dbase - 0.34 + 0.17 + i * 0.34
    v = np.array([dvals[0][i], dvals[1][i]], dtype=float)
    axD.bar(x, v, width=0.29, color=[BLUE, VERM][i], hatch=[None, "///"][i],
            edgecolor="white", linewidth=0.5, label=name, zorder=3)
    for xx, vv in zip(x, v, strict=False):
        axD.text(xx, vv + 2.4, "%d" % vv, va="bottom", ha="center", fontsize=9,
                 color=GREY)
axD.axhline(69, color=LIGHT, lw=0.8, ls=(0, (2, 2)), zorder=1)
axD.set_xticks(dbase)
axD.set_xticklabels(["of 75", "of 69"])
axD.set_ylim(0, YMAX)
axD.set_xlim(-0.62, 1.62)
axD.set_yticks([0, 20, 40, 60])
axD.set_ylabel("genuine cells", labelpad=2)
axD.tick_params(length=2.5)
axD.tick_params(axis="x", length=0)
# one column, two rows: a two-column key would run under the middle panel's
# two-line title
series_key(axD, 2.06, 5.86, 1)

axD2 = axes_in(3.60, 6.11, 0.95, AXH)
# the reference is spelled out on every block-(d) panel: the middle panel is
# the C6 cell against the recalibrated HAR on the deployable path, NOT the
# 8-K residual against the firm-identity reference drawn in block (a)
companion_title(axD2, "8-K C6, pooled rel %\nvs recalibrated HAR")
yy = np.arange(3)[::-1].astype(float)
for y0, row in zip(yy, c6.itertuples(), strict=False):
    axD2.plot([row.fixed_pooled_rel, row.exp_pooled_rel], [y0 + 0.17, y0 - 0.17],
              color=GREY, lw=0.7, zorder=2)
    axD2.plot(row.fixed_pooled_rel, y0 + 0.17, "o", ms=4.6, color=BLUE,
              mfc=BLUE if row.genuine_fixed else "white", mew=1.0, zorder=3)
    axD2.plot(row.exp_pooled_rel, y0 - 0.17, "D", ms=4.2, color=VERM,
              mfc=VERM if row.genuine_exp else "white", mew=1.0, zorder=3)
axD2.axvline(0, color=LIGHT, lw=0.8, zorder=1)   # passive reference, like the
#                                                  dotted 69-cell guide above
axD2.set_yticks(yy)
axD2.set_yticklabels(["h=5", "h=10", "h=20"])
axD2.set_ylim(-0.62, 2.62)
axD2.set_xlim(-0.12, 1.9)
axD2.set_xticks([0, 1])
axD2.tick_params(axis="y", length=0)
_strip(axD2)

axD3 = axes_in(5.15, 6.11, 0.95, AXH)
companion_title(axD3, "calm - COVID, pp\nvs maximal price pool")
rng = np.random.default_rng(0)
axD3.plot(vw_diff, rng.uniform(-0.62, 0.62, size=vw_diff.size), "o", ms=2.8,
          color=BLUE, alpha=0.75, mew=0, zorder=3)
# Vermillion is the expanding arm two panels to the left, and these two lines
# are not an arm: they are the median and the mean of the cloud. Primary ink at
# double the reference weight says "computed locus"; the zero line drops to the
# passive-reference grey so the median, which lands on it, still reads.
axD3.axvline(0, color=LIGHT, lw=0.8, zorder=1)
axD3.axvline(float(np.median(vw_diff)), color=INK, lw=1.3, zorder=4)
axD3.axvline(float(vw_diff.mean()), color=INK, lw=1.3, ls=(0, (3, 2)), zorder=4)
axD3.set_yticks([])
axD3.set_ylim(-1.0, 1.0)
axD3.set_xlim(-6.8, 15.6)
axD3.set_xticks([-5, 0, 5, 10])
axD3.tick_params(axis="y", length=0)
_strip(axD3)

D_END = fig_lines(
    0.62, 7.27,
    "Genuine = pooled day-clustered DM < 0, Holm < .05, abs(placebo DM) < 2. "
    "Middle: filled = genuine, circle fixed, diamond expanding. Right: calm "
    "minus COVID differences over %d cells, against the MAXIMAL FIVE-PRICE "
    "POOL on the seed-2026 text basis, NOT the recalibrated HAR: mean "
    "+%.3fpp (dashed), median +%.3fpp (solid), calm higher in %d of %d, "
    "t = +%.2f, p = %.3f; negative under BOTH halves (calm %s, COVID %s)."
    % (vw_diff.size, vw_diff.mean(), np.median(vw_diff),
       int((vw_diff > 0).sum()), vw_diff.size, vw_t, vw_p,
       VW_CALM_TXT, VW_COVID_TXT), color=INK2)

# ------------------------------------------------------------ closing note
# The refit statement is the one the sources support: the TEXT arms are frozen
# everywhere, the price references are not (rangebased_cascade.md L3,
# public_variant_cascade.md L3).
fig_rule(7.905)
C_END = fig_lines(
    0.62, 7.96,
    "Primary-rung bars are placebo-gated genuine counts; the grey tick is "
    "the Holm-only count. Control rungs are Holm counts; dotted guide = %d "
    "cells. Text arms are frozen throughout and block (a) changes only the "
    "loss, but blocks (b) and (c) panel C refit the price references A2 "
    "(HAR) and A6 (SHAR) on the new labels or features, and recalibration, "
    "the firm mean and the combiner weights are refit wherever labels, rows "
    "or weights change." % vw_diff.size, color=INK2)
print("block ends: a %.2f  d %.2f  closing %.2f  (FH %.2f)"
      % (A_END, D_END, C_END, FH))

finish(fig, "F11_ladder_perturbations")
