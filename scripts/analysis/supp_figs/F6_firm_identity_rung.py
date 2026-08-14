"""F6 -- the firm-identity rung, and what a zero-text firm mean achieves alone.

Sources (every plotted number is read from one of these in this run; nothing is
hardcoded outside gate() expectations, whose job is to abort on evidence drift):
  results/tables/firm_identity_ensemble.csv
      disc, model, h, n_test, rel_impr_pct_firm, verdict, adds_holm, hurts_holm,
      firm_val_coverage, firm_val_coverage_test_obs,
      rel_impr_firmMeanOnly_vs_fR, dm_firmMeanOnly_vs_fR,
      p_firmMeanOnly_vs_fR, firm_beats_fR
  results/tables/firm_identity_control_zerotext.csv
      disc, h, n_test, rel_pct, dm_clustered, p_clustered   (full A2 panel)

Main-text sentences substantiated (frozen, must not be contradicted):
  00_abstract.tex  "a zero-text firm-identity reference reproduces the apparent
                    gain in 4 of the 6 channel x horizon comparisons"
  01_intro.tex     the same 4-of-6 claim
  06_results.tex   "the firm's mean validation-period RV, a zero-cost and
                    zero-text term, flips 38 of the 45 negative"
  06_results.tex   event-driven C6 "+0.52/+0.24/+0.21%" against firm identity
  08_discussion.tex table row "8/69 Holm; firm mean alone beats f_R in 4 of 6"

Adversarial repairs folded in (all three lenses; every required_change binding):
  * Panel (b) is drawn from firm_identity_ensemble.csv's firmMeanOnly columns
    (the estimand the frozen 4-of-6 refers to), NOT from the full-A2-panel file.
  * The full-A2 file is still shown, in its own labelled panel (c), with its own
    count (3 of 6) printed, so the basis-dependence is visible rather than
    cropped.  Two repairs conflict here -- one lens says "put it in a separate
    labelled panel", another says "never publish DM<0 and p<.05 = 4 of 6" (a
    statement true only of the full-A2 file).  The conservative resolution is
    taken: BOTH panels print all six rows with rel%, DM and p, and each prints
    its own count beside its own row basis, so neither count can be read off
    the wrong file.
  * Six rows, not 69: the firmMeanOnly columns take one distinct value per
    (channel, horizon), so the denominator drawn is 6 comparisons.  The 53-cell
    expansion of the same flag is never placed on an axis.
  * Channel means are the seed-ensemble values (-2.25% long-form, -0.27%
    event-driven); the seed-2026 event-driven mean (-0.10%) is not used here.
  * Axis extremes are the ensemble minimum (-11.38%, long_form/C4_longformer/
    h=10) and the second-most-negative cell (-8.09%, long_form/B2_tfidf_ridge/
    h=20).  The seed-2026 figure -10.42% is not used.
  * The third colour is labelled "no verdict recorded", not "null": the file
    leaves the verdict column empty for those 26 rows.
  * The "percentage sign and verdict can disagree" rationale is dropped for
    panel (a); the script instead gates on the fact that no cell in this file
    has a verdict opposing the sign of its percentage.  The observation-weighted
    vs day-weighted divergence is real in panel (b) and is annotated there, at
    long-form h=20.

CPU only, single-threaded, no model is re-fitted.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from supp_style import BLUE, GREY, LIGHT, PURPLE, TAB, VERM, VERM_TXT, apply_style, finish, gate

# --------------------------------------------------------------- evidence
ENS = pd.read_csv(os.path.join(TAB, "firm_identity_ensemble.csv"))
ZT = pd.read_csv(os.path.join(TAB, "firm_identity_control_zerotext.csv"))

ENS["verdict"] = ENS["verdict"].fillna("")
lf = ENS[ENS.disc == "long_form"]
ed = ENS[ENS.disc == "event_driven"]

# the six channel x horizon comparisons; the firmMeanOnly columns take exactly
# one distinct value per (channel, horizon), so 6 is the honest denominator.
SIX = (ENS.drop_duplicates(["disc", "h"])
          .sort_values(["disc", "h"], ascending=[False, True])
          [["disc", "h", "n_test", "rel_impr_firmMeanOnly_vs_fR",
            "dm_firmMeanOnly_vs_fR", "p_firmMeanOnly_vs_fR", "firm_beats_fR"]]
          .reset_index(drop=True))
assert len(SIX) == 6
beats_ens = int(((SIX.dm_firmMeanOnly_vs_fR < 0)
                 & (SIX.p_firmMeanOnly_vs_fR < 0.05)).sum())

ZT = ZT.sort_values(["disc", "h"], ascending=[False, True]).reset_index(drop=True)
beats_zt = int(((ZT.dm_clustered < 0) & (ZT.p_clustered < 0.05)).sum())

srt = ENS.sort_values("rel_impr_pct_firm", kind="mergesort").reset_index(drop=True)
c6ed = ENS[(ENS.disc == "event_driven") & (ENS.model == "C6_llmtext")] \
    .sort_values("h")
lf20 = SIX[(SIX.disc == "long_form") & (SIX.h == 20)].iloc[0]
ed10z = ZT[(ZT.disc == "event_driven") & (ZT.h == 10)].iloc[0]
# The comparison that actually changes status between panels (b) and (c) is
# long-form h=5 (p .024 -> .231); event-driven h=10 is hollow on both bases.
lf5z = ZT[(ZT.disc == "long_form") & (ZT.h == 5)].iloc[0]

# --------------------------------------------------------------- the gate
gate(
    {
        "n_cells": 69,
        "adds_holm": 8,
        "hurts_holm": 35,
        "no_verdict": 26,
        "lf_cells": 45,
        "lf_negative": 38,
        "lf_mean_pct": -2.25,
        "ed_mean_pct": -0.27,
        "worst_pct": -11.38,
        "worst_cell": "long_form/C4_longformer/h10",
        "second_worst_pct": -8.09,
        "second_worst_cell": "long_form/B2_tfidf_ridge/h20",
        "cov_firm_lf": 0.629,
        "cov_firm_ed": 0.630,
        "cov_obs_lf": 0.915,
        "cov_obs_ed": 0.919,
        "c6_event_driven": [0.52, 0.24, 0.21],
        # the frozen abstract/intro count, on the grid-row basis
        "firm_beats_fR_flag": 4,
        "firm_beats_fR_recomputed": 4,
        # the same test on the wider full-A2 row basis returns a different count
        "zerotext_full_panel_beats": 3,
        # the colouring channel: the verdict never opposes the sign here
        "adds_with_negative_pct": 0,
        "hurts_with_positive_pct": 0,
    },
    {
        "n_cells": len(ENS),
        "adds_holm": int(ENS.adds_holm.sum()),
        "hurts_holm": int(ENS.hurts_holm.sum()),
        "no_verdict": int((ENS.verdict == "").sum()),
        "lf_cells": len(lf),
        "lf_negative": int((lf.rel_impr_pct_firm < 0).sum()),
        "lf_mean_pct": round(float(lf.rel_impr_pct_firm.mean()), 2),
        "ed_mean_pct": round(float(ed.rel_impr_pct_firm.mean()), 2),
        "worst_pct": round(float(srt.rel_impr_pct_firm.iloc[0]), 2),
        "worst_cell": "{}/{}/h{}".format(*srt.loc[0, ["disc", "model", "h"]]),
        "second_worst_pct": round(float(srt.rel_impr_pct_firm.iloc[1]), 2),
        "second_worst_cell": "{}/{}/h{}".format(*srt.loc[1, ["disc", "model", "h"]]),
        "cov_firm_lf": round(float(lf.firm_val_coverage.iloc[0]), 3),
        "cov_firm_ed": round(float(ed.firm_val_coverage.iloc[0]), 3),
        "cov_obs_lf": round(float(lf.firm_val_coverage_test_obs.iloc[0]), 3),
        "cov_obs_ed": round(float(ed.firm_val_coverage_test_obs.iloc[0]), 3),
        "c6_event_driven": [round(float(v), 2) for v in c6ed.rel_impr_pct_firm],
        "firm_beats_fR_flag": int(SIX.firm_beats_fR.sum()),
        "firm_beats_fR_recomputed": beats_ens,
        "zerotext_full_panel_beats": beats_zt,
        "adds_with_negative_pct": int(((ENS.verdict == "text adds")
                                       & (ENS.rel_impr_pct_firm < 0)).sum()),
        "hurts_with_positive_pct": int(((ENS.verdict == "text HURTS")
                                        & (ENS.rel_impr_pct_firm > 0)).sum()),
    },
)

# --------------------------------------------------------------- geometry
apply_style()
W, H = 6.4, 8.45
LABW, BARW = 1.66, 1.46          # inches: y-label gutter, bar plotting width
A_BOT, A_HGT = 3.12, 4.80        # panel (a) axes box
B_BOT, B_HGT = 0.92, 0.95        # panels (b)/(c) axes box
BLABW, BBARW = 0.52, 1.45        # (b)/(c) label gutter, DM plotting width
LINE = 0.155                     # inter-line pitch for 9pt figure text

fig = plt.figure(figsize=(W, H))


def rect(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


axL = fig.add_axes(rect(LABW, A_BOT, BARW, A_HGT))
axR = fig.add_axes(rect(W / 2 + LABW - 0.06, A_BOT, BARW, A_HGT))
axB = fig.add_axes(rect(BLABW, B_BOT, BBARW, B_HGT))
axC = fig.add_axes(rect(W / 2 + BLABW, B_BOT, BBARW, B_HGT))

SHORT = {"long_form": "LF", "event_driven": "ED"}
COL = {"text adds": BLUE, "text HURTS": VERM, "": LIGHT}
CODE = {"text adds": "A", "text HURTS": "H", "": " "}

XMIN, XMAX = -13.5, 2.6
NROW = 35                        # rows per sub-column (35 + 34 = 69)

# --------------------------------------------------------- panel (a) bars
for ax, block, rank0 in ((axL, srt.iloc[:NROW], 1), (axR, srt.iloc[NROW:], NROW + 1)):
    ypos = np.arange(len(block))
    vals = block.rel_impr_pct_firm.to_numpy()
    cols = [COL[v] for v in block.verdict]
    edges = [GREY if v == "text adds" else "none" for v in block.verdict]
    lws = [0.8 if v == "text adds" else 0.0 for v in block.verdict]
    ax.axvspan(-1, 1, color=LIGHT, alpha=0.16, lw=0, zorder=0)
    for b in (-1, 1):
        ax.axvline(b, color=GREY, lw=0.5, ls=(0, (1, 2)), zorder=1)
    ax.barh(ypos, vals, height=0.68, color=cols, edgecolor=edges,
            linewidth=lws, zorder=3)
    ax.axvline(0, color=GREY, lw=0.7, zorder=4)
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.15)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(NROW - 0.5, -0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [f"{SHORT[r.disc]} {r.model} h={r.h}  {CODE[r.verdict]}"
         for r in block.itertuples()])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks([-10, -3, -1, 0, 1])
    ax.set_xticklabels(["-10", "-3", "-1", "0", "+1"])
    ax.xaxis.grid(True, color=LIGHT, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.text(0.0, 1.006, f"cells {rank0}-{rank0 + len(block) - 1} of 69",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            color=GREY)

# value labels: the two extremes and the three event-driven C6 cells.  The two
# extremes sit inside their own bar (there is no room outside), the small
# positive cells sit just past the bar tip.
NAMED = {(srt.disc[0], srt.model[0], srt.h[0]),
         (srt.disc[1], srt.model[1], srt.h[1])} | \
        {(r.disc, r.model, r.h) for r in c6ed.itertuples()}
for ax, block in ((axL, srt.iloc[:NROW]), (axR, srt.iloc[NROW:])):
    for i, r in enumerate(block.itertuples()):
        if (r.disc, r.model, r.h) not in NAMED:
            continue
        v = r.rel_impr_pct_firm
        inside = v < -3.0
        ax.annotate(f"{v:+.2f}", (v, i),
                    xytext=(4 if (inside or v > 0) else -3, 0),
                    textcoords="offset points",
                    ha="right" if (v < 0 and not inside) else "left",
                    va="center", fontsize=9,
                    color="white" if inside else (VERM_TXT if v < 0 else BLUE),
                    zorder=6)

fig.text(0.5, (A_BOT - 0.33) / H,
         "QLIKE improvement of text over the firm-identity-augmented reference "
         "(%); linear on [-1, +1], logarithmic outside",
         ha="center", va="center", fontsize=9, color=GREY)

handles = [Patch(facecolor=BLUE, edgecolor=GREY, lw=0.8,
                 label="A = text adds, Holm (8)"),
           Patch(facecolor=VERM, label="H = text hurts, Holm (35)"),
           Patch(facecolor=LIGHT, label="no code = no verdict recorded (26)")]
fig.legend(handles=handles, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, (A_BOT + A_HGT + 0.14) / H),
           handlelength=1.1, handletextpad=0.5, columnspacing=1.5,
           borderpad=0.0, fontsize=9)

fig.text(0.5, (A_BOT + A_HGT + 0.38) / H,
         "(a)  All 69 grid cells against the firm-identity reference, sorted; "
         "seed-ensemble basis",
         ha="center", va="center", fontsize=9.5, color=GREY, weight="bold")

note = [
    f"Channel means {lf.rel_impr_pct_firm.mean():+.2f}% long-form, {ed.rel_impr_pct_firm.mean():+.2f}% event-driven. The firm "
    f"mean turns {int((lf.rel_impr_pct_firm < 0).sum())} of {len(lf)} long-form cells negative.",
    f"Firm coverage {lf.firm_val_coverage.iloc[0]:.3f} / {ed.firm_val_coverage.iloc[0]:.3f} of firms and {lf.firm_val_coverage_test_obs.iloc[0]:.3f} / "
    f"{ed.firm_val_coverage_test_obs.iloc[0]:.3f} of test rows (long-form / event-driven).",
]
for k, t in enumerate(note):
    fig.text(0.5, (A_BOT - 0.57 - LINE * k) / H, t, ha="center", va="center",
             fontsize=9, color=GREY)


# --------------------------------------------------- panels (b) and (c)
def lollipop(ax, rows, rel_key, dm_key, p_key):
    lab, dm, rel, pv = [], [], [], []
    for r in rows.itertuples():
        lab.append(f"{SHORT[r.disc]} h={r.h}")
        dm.append(float(getattr(r, dm_key)))
        rel.append(float(getattr(r, rel_key)))
        pv.append(float(getattr(r, p_key)))
    y = np.arange(len(lab))
    ax.axvline(0, color=GREY, lw=0.7, zorder=4)
    for i, (d, p) in enumerate(zip(dm, pv, strict=False)):
        sig = p < 0.05
        ax.plot([0, d], [i, i], color=PURPLE if sig else GREY,
                lw=1.6 if sig else 0.9, zorder=3, solid_capstyle="butt")
        ax.plot([d], [i], marker="o", ms=5.2, zorder=5,
                mfc=PURPLE if sig else "white",
                mec=PURPLE if sig else GREY, mew=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(lab)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(len(lab) - 0.5, -0.5)
    ax.set_xlim(-5.2, 0.9)
    ax.set_xticks([-4, -2, 0])
    ax.xaxis.grid(True, color=LIGHT, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    for i, (d, r, p) in enumerate(zip(dm, rel, pv, strict=False)):
        ptxt = "p<.001" if p < 0.0005 else f"p={p:.3f}"
        ax.text(1.05, i, f"{r:+.2f}%  {ptxt}",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=9, color=GREY, clip_on=False, zorder=6)


lollipop(axB, SIX, "rel_impr_firmMeanOnly_vs_fR", "dm_firmMeanOnly_vs_fR",
         "p_firmMeanOnly_vs_fR")
lollipop(axC, ZT, "rel_pct", "dm_clustered", "p_clustered")

for ax in (axB, axC):
    ax.set_xlabel("day-clustered DM", fontsize=9, labelpad=2)

n_lf = SIX[SIX.disc == "long_form"].n_test.tolist()
n_ed = SIX[SIX.disc == "event_driven"].n_test.tolist()
z_lf = ZT[ZT.disc == "long_form"].n_test.tolist()
z_ed = ZT[ZT.disc == "event_driven"].n_test.tolist()
ROWS = "LF n {:,}-{:,}, ED n {:,}-{:,} by horizon".format

blocks = [
    (0.0, "(b)  Firm mean alone vs " + r"$f_R$" + ", grid rows: "
     f"{beats_ens} of 6",
     ROWS(max(n_lf), min(n_lf), max(n_ed), min(n_ed))),
    (W / 2, f"(c)  The same test, full A2 panel: {beats_zt} of 6",
     ROWS(max(z_lf), min(z_lf), max(z_ed), min(z_ed))),
]
for x0, t1, t2 in blocks:
    for k, (t, sz, wt) in enumerate(((t1, 9.5, "bold"), (t2, 9, "normal"))):
        fig.text(x0 / W, (B_BOT + B_HGT + 0.32 - LINE * k) / H, t, ha="left",
                 va="center", fontsize=sz, color=GREY, weight=wt)

foot = [
    "Lollipop: day-clustered DM, filled where p < .05. Both counts use DM < 0 "
    "and p < .05 (the zero-text",
    "reference beats the recalibrated HAR). Panel (b) is the reported four of "
    "six: long-form h=5, 10, 20 and",
    "event-driven h=5; panel (c) runs the same test on every A2 row and returns "
    "three of six (long-form h=5",
    f"falls from p=0.024 to p={float(lf5z.p_clustered):.3f}), so the count moves with the row basis. "
    f"Long-form h=20 in (b): {float(lf20.rel_impr_firmMeanOnly_vs_fR):+.2f}% against DM {float(lf20.dm_firmMeanOnly_vs_fR):+.2f}.",
]
for k, t in enumerate(foot):
    fig.text(0.5, (0.50 - LINE * k) / H, t, ha="center", va="center",
             fontsize=9, color=GREY)

# guard: the artefact must ship at ~1:1, otherwise every 9pt label shrinks when
# the supplement includes it at \linewidth (6.5in).
fig.canvas.draw()
bb = fig.get_tightbbox(fig.canvas.get_renderer())
if bb.width > 6.62 or bb.height > 8.70:
    sys.exit(f"LAYOUT GATE FAIL - tight bbox {bb.width:.2f} x {bb.height:.2f} in")

finish(fig, "F6_firm_identity_rung")
