"""F10 -- Detectable, not attributable.

Two panels: (a) the fitted text coefficient g in all 48 forecast-encompassing
regressions, grouped by disclosure and sorted within group, coloured by the
HAC t statistic; (b) the pooled omnibus power curve, with the real-data
pooled statistics beside it.

Sources (parsed at run time; nothing is hardcoded outside gate())
-----------------------------------------------------------------
* results/tables/encompassing_regression.md -- 48 rows, no CSV companion
  exists, so the markdown tables are parsed here. Columns taken: model, h,
  g (text coef), HAC t, p. The source's per-row `verdict` column is
  deliberately NOT read or reproduced (it editorialises every row).
  Transcription risk is handled two ways: the parser is an exact match on
  the pipe tables, and gate() pins the row count, the extreme g values and
  the extreme t values, so a silent edit to the source aborts the build.
* results/tables/omnibus_m1.csv -- section in {omnibus, power, mde}:
  subfamily, n_cells, n_days, approx_rel_pct, t, p, level_pct,
  n_kappa_negative, reject_rate, n_reps, mde_80_empirical_pct,
  mde_80_analytic_pct

Main-text sentences substantiated
---------------------------------
08_discussion.tex: "Conditional on HAR, text is detectable, which is not the
same as 'text is noise': encompassing regressions return g>0 in 48 of 48
tests. But detection is not attribution."
06_results.tex: "while a pooled test over all 69 does detect the systematic
micro-increment (t=+8.29, MDE 0.375%): detectable, not attributable."

Scope carried on the artefact (binding disclosure): a, b and g are fitted on
the TEST sample itself, so panel (a) is an in-sample conditional-correlation
statistic, not a validation-frozen out-of-sample loss reduction. It is the
only such quantity in the paper.

Functional form, checked against the generator rather than assumed: the
source builds y = label_realised_vol and X = [1, f_HAR, f_text] with no log
transform anywhere (scripts/analysis/encompassing.py), and its own header
reads "RV = a + b*f_HAR + g*f_text + e (test, HAC lag h-1)". The regression
is therefore in LEVELS. It is not the log-space nesting of Eq. (1) in
05_protocol.tex, which is the M1 forecast-combination object; conflating the
two is the error this artefact previously carried on its axis label.
"""
import os
import re
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from supp_style import BLUE, GREY, SKY, TAB, VERM, VERM_TXT, apply_style, finish, gate

W, H = 6.5, 9.0


def rect(x, y, w, h):
    """Axes rectangle from inches (origin bottom-left) to figure fractions."""
    return [x / W, y / H, w / W, h / H]


def para(text, width):
    """Wrap a multi-paragraph string, keeping the blank line between paras."""
    return "\n\n".join("\n".join(textwrap.wrap(p, width))
                       for p in text.split("\n\n"))


# ------------------------------------------------- parse the markdown table
ENC = os.path.join(TAB, "encompassing_regression.md")
ROW = re.compile(r"^\|\s*([A-Za-z0-9_]+)\s*\|\s*(\d+)\s*\|\s*([+-][\d.]+)\s*\|"
                 r"\s*([+-][\d.]+)\s*\|\s*([\d.]+)\s*\|")
raw = open(ENC, encoding="utf-8").read()
rows, disc = [], None
for line in raw.splitlines():
    if line.startswith("## "):
        disc = line[3:].strip()
        continue
    m = ROW.match(line.strip())
    if m and disc:
        rows.append({"disc": disc, "model": m.group(1), "h": int(m.group(2)),
                     "g": float(m.group(3)), "t": float(m.group(4)),
                     "p": float(m.group(5))})
enc = pd.DataFrame(rows)
# the source's own summary counts, parsed rather than trusted from memory
summ = re.search(r"of (\d+) encompassing tests, (\d+) show a significantly "
                 r"positive text coefficient", raw)
n_claimed, n_pos_claimed = int(summ.group(1)), int(summ.group(2))
n_pos = int(((enc.g > 0) & (enc.p < 0.05)).sum())

# ------------------------------------------------------- omnibus and power
omn = pd.read_csv(os.path.join(TAB, "omnibus_m1.csv"))
om = omn[omn.section == "omnibus"].set_index("subfamily")
pw = omn[omn.section == "power"].sort_values("level_pct")
md = omn[omn.section == "mde"].iloc[0]

lev = pw.level_pct.to_numpy()
rate = pw.reject_rate.to_numpy()
kneg = pw.n_kappa_negative.to_numpy().astype(int)
n_reps = int(pw.n_reps.iloc[0])
mde_emp = float(md.mde_80_empirical_pct)
mde_ana = float(md.mde_80_analytic_pct)

# ------------------------------------------------------------------- gate
gate(
    {"n_tests": 48, "n_claimed": 48, "n_pos": 48, "n_pos_claimed": 48,
     "n_long_form": 33, "n_event_driven": 15,
     "g_min": 0.052, "g_max": 2.882, "t_min": 2.28, "t_max": 21.12,
     "p_max": 0.0225,
     "t_all69": 8.286, "t_lf": 6.899, "t_ed": 4.565,
     "cells": [69, 45, 24], "days": [996, 809, 996],
     "rel_all69": 0.843,
     "levels": [0.1, 0.2, 0.3, 0.5, 1.0],
     "rates": [0.10, 0.46, 0.68, 1.00, 1.00],
     "kappa_negative": [51, 45, 40, 37, 32], "n_reps": 100,
     "mde_empirical": 0.375, "mde_analytic": 0.302},
    {"n_tests": len(enc), "n_claimed": n_claimed, "n_pos": n_pos,
     "n_pos_claimed": n_pos_claimed,
     "n_long_form": int((enc.disc == "long_form").sum()),
     "n_event_driven": int((enc.disc == "event_driven").sum()),
     "g_min": round(float(enc.g.min()), 3),
     "g_max": round(float(enc.g.max()), 3),
     "t_min": round(float(enc.t.min()), 2),
     "t_max": round(float(enc.t.max()), 2),
     "p_max": round(float(enc.p.max()), 4),
     "t_all69": round(float(om.loc["all_69", "t"]), 3),
     "t_lf": round(float(om.loc["long_form", "t"]), 3),
     "t_ed": round(float(om.loc["event_driven", "t"]), 3),
     "cells": [int(om.loc[k, "n_cells"])
               for k in ("all_69", "long_form", "event_driven")],
     "days": [int(om.loc[k, "n_days"])
              for k in ("all_69", "long_form", "event_driven")],
     "rel_all69": round(float(om.loc["all_69", "approx_rel_pct"]), 3),
     "levels": [round(float(v), 2) for v in lev],
     "rates": [round(float(v), 2) for v in rate],
     "kappa_negative": list(kneg), "n_reps": n_reps,
     "mde_empirical": round(mde_emp, 3), "mde_analytic": round(mde_ana, 3)},
)

# ------------------------------------------------------------------ figure
apply_style(9)
fig = plt.figure(figsize=(W, H))

PITCH = 0.150                      # inches per bar row, both sub-panels
BW = 0.90                          # inches, bar-axes width, both sub-panels
TOP = 8.50                         # inches, shared top edge of both columns
COL0, COL1 = 1.19, 4.46            # inches, left edge of each bar axes
lf = enc[enc.disc == "long_form"].sort_values("g").reset_index(drop=True)
ed = enc[enc.disc == "event_driven"].sort_values("g").reset_index(drop=True)

ax_lf = fig.add_axes(rect(COL0, TOP - PITCH * len(lf), BW, PITCH * len(lf)))
ax_ed = fig.add_axes(rect(COL1, TOP - PITCH * len(ed), BW, PITCH * len(ed)))

cmap = LinearSegmentedColormap.from_list(
    "hac_t", ["#D3E5F1", "#8FC0DC", SKY, BLUE, "#00456E"])
norm = Normalize(0.0, 22.0)
GX, TX = 1 + 0.50 / BW, 1 + 1.00 / BW     # axes-fraction x of the two number
                                          # columns printed beside the bars

for ax, dat, ttl in ((ax_lf, lf, f"long-form 10-K/Q, {len(lf)} tests"),
                     (ax_ed, ed, f"event-driven 8-K, {len(ed)} tests")):
    y = np.arange(len(dat))
    ax.barh(y, dat.g, height=0.66, color=cmap(norm(dat.t.to_numpy())),
            edgecolor=GREY, lw=0.25, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} h={h}" for m, h in zip(dat.model, dat.h, strict=False)],
                       fontsize=9)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.66, len(dat) - 0.34)
    ax.set_xlim(0, 3.05)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("text coefficient g (levels, not logs)")
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=GREY, lw=0.6, zorder=4)
    # g and the HAC t printed for every row, so nothing is read off colour
    tr = ax.get_yaxis_transform()
    for yi, gv, tv in zip(y, dat.g, dat.t, strict=False):
        ax.text(GX, yi, f"{gv:+.3f}", transform=tr, ha="right", va="center",
                fontsize=9, color=GREY, clip_on=False)
        ax.text(TX, yi, f"{tv:+.2f}", transform=tr, ha="right", va="center",
                fontsize=9, color=GREY, clip_on=False)

for x0, ttl in ((COL0, f"long-form 10-K/Q, {len(lf)} tests"),
                (COL1, f"event-driven 8-K, {len(ed)} tests")):
    fig.text((x0 - 1.17) / W, (TOP + 0.07) / H, ttl, fontsize=9, color=GREY,
             va="bottom", ha="left")
    fig.text((x0 + BW * GX) / W, (TOP + 0.07) / H, "g", fontsize=9,
             color=GREY, va="bottom", ha="right")
    fig.text((x0 + BW * TX) / W, (TOP + 0.07) / H, "HAC t", fontsize=9,
             color=GREY, va="bottom", ha="right")

fig.text(0.10 / W, 8.82 / H,
         "(a)  The text coefficient g conditional on the HAR forecast, all "
         f"{len(enc)} encompassing tests",
         fontsize=9.6, color=GREY, va="bottom", ha="left")

# colour key, with the 1.96 threshold marked on it
cax = fig.add_axes(rect(COL1, 5.48, 1.10, 0.10))
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  orientation="horizontal")
cb.outline.set_linewidth(0.5)
cb.outline.set_edgecolor(GREY)
cb.set_ticks([1.96, 10, 20])
cb.ax.set_xticklabels(["1.96", "10", "20"], fontsize=9)
cb.ax.tick_params(length=2.2, width=0.6, color=GREY, pad=2)
cb.ax.plot([1.96, 1.96], [0, 1], color=VERM, lw=1.4, zorder=5)
fig.text(3.29 / W, 5.64 / H,
         "bar colour: HAC t; 5% two-sided threshold 1.96 marked",
         fontsize=9, color=GREY, va="bottom", ha="left")

disc_txt = (
    "RV, f_HAR and f_text all enter in LEVELS, not logs (source header: "
    "RV = a + b*f_HAR + g*f_text, HAC lag h-1); this is NOT the log-space "
    "nesting of Eq. (1), which is the forecast-combination object. a, b and "
    "g are fitted on the TEST sample itself, so panel (a) is an in-sample "
    "conditional-correlation statistic, not a validation-frozen out-of-sample "
    "loss reduction, and it is the only such quantity in the paper. g is a "
    "level-space regression loading, not a QLIKE improvement, and entitles "
    "no cell to a survivor count."
)
fig.text(3.29 / W, 5.30 / H, para(disc_txt, 54), fontsize=9, color=VERM_TXT,
         va="top", ha="left", linespacing=1.30)

range_txt = (
    f"All {len(enc)} coefficients are positive at p < .05 (largest p "
    f"{enc.p.max():.4f}). g runs from {enc.g.min():+.3f} "
    f"({lf.model.iloc[0]} h={lf.h.iloc[0]}, long-form) to "
    f"{enc.g.max():+.3f} ({ed.model.iloc[-1]} h={ed.h.iloc[-1]}, "
    f"event-driven); the HAC t from {enc.t.min():+.2f} to "
    f"{enc.t.max():+.2f}."
)
fig.text(3.29 / W, 3.60 / H, para(range_txt, 54), fontsize=9, color=GREY,
         va="top", ha="left", linespacing=1.30)

# --------------------------------------------------- (b) pooled power curve
ax_b = fig.add_axes(rect(0.85, 1.20, 2.95, 1.50))
ax_b.axhline(0.80, color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=2)
ax_b.plot([mde_emp, mde_emp], [0, 0.80], color=VERM, lw=1.2,
          ls=(0, (1, 1.6)), zorder=4)
ax_b.plot(lev, rate, color=BLUE, lw=1.3, marker="o", ms=4.6, mfc=BLUE,
          mec="white", mew=0.6, zorder=5)
ax_b.plot([mde_emp], [0.80], marker="D", ms=5.2, mfc=VERM, mec="white",
          mew=0.7, ls="none", zorder=6)
for xi, yi in zip(lev, rate, strict=False):
    ax_b.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                  xytext=(0, 7), ha="center", fontsize=9, color=BLUE,
                  bbox=dict(fc="white", ec="none", pad=0.6), zorder=7)
ax_b.annotate("80% power", (1.045, 0.80), textcoords="offset points",
              xytext=(0, 4), ha="right", va="bottom", fontsize=9, color=GREY)
ax_b.annotate(f"empirical MDE {mde_emp:.3f}%\n(analytic {mde_ana:.3f}%)",
              (mde_emp, 0.03), textcoords="offset points", xytext=(7, 0),
              ha="left", va="bottom", fontsize=9, color=VERM_TXT,
              linespacing=1.30)
ax_b.set_xlim(0, 1.06)
ax_b.set_ylim(0, 1.14)
ax_b.set_xticks(list(lev))
ax_b.set_xticklabels([f"{v:g}" for v in lev])
ax_b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_b.set_xlabel("injected level (realised rel-QLIKE, %)")
ax_b.set_ylabel(f"rejection rate\nover {n_reps} day-block\n"
                "bootstrap replications")
fig.text(0.10 / W, 2.86 / H,
         "(b)  What the pooled omnibus is powered to detect",
         fontsize=9.6, color=GREY, va="bottom", ha="left")

box = (
    "REAL DATA, no injection -- pooled omnibus on the day-clustered daily "
    "differential:\n\n"
    f"all {int(om.loc['all_69', 'n_cells'])} cells:  t = "
    f"{om.loc['all_69', 't']:+.2f},  p = {om.loc['all_69', 'p']:.1e},  "
    f"{int(om.loc['all_69', 'n_days'])} days, mean daily improvement "
    f"{om.loc['all_69', 'approx_rel_pct']:+.2f}% (descriptive scale)\n\n"
    f"long-form:  t = {om.loc['long_form', 't']:+.2f}, "
    f"{int(om.loc['long_form', 'n_cells'])} cells, "
    f"{int(om.loc['long_form', 'n_days'])} days\n\n"
    f"event-driven:  t = {om.loc['event_driven', 't']:+.2f}, "
    f"{int(om.loc['event_driven', 'n_cells'])} cells, "
    f"{int(om.loc['event_driven', 'n_days'])} days"
)
fig.text(4.05 / W, 2.70 / H, para(box, 40), fontsize=9, color=GREY,
         va="top", ha="left", linespacing=1.30)

foot = (
    "At the five injected levels signal was SUBTRACTED in "
    + " / ".join(str(k) for k in kneg) +
    f" of the {int(om.loc['all_69', 'n_cells'])} cells, so panel (b) is a "
    "recovery rate at a fixed effect. It is also a within-basis curve: its "
    "rejection rates say nothing about which cells the effect lives in, and "
    "the injection is the same oracle construction from test labels. "
    "Detection at the pooled level and attribution at the cell level are "
    "different claims."
)
fig.text(0.30 / W, 0.72 / H, para(foot, 105), fontsize=9, color=GREY,
         va="top", ha="left", linespacing=1.30)

finish(fig, "F10_encompassing_pooled_power")
