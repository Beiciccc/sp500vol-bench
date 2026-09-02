"""F13 -- Elicitation, not curation, and not the name.

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
from supp_style import (BLUE, GREEN, GREY, LIGHT, PURPLE, REPO, SKY, TAB,
                        VERM, VERM_TXT, YELLOW, apply_style, finish, gate)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

W, H = 6.5, 8.95                      # canvas inches (portrait supplement page)


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
fig = plt.figure(figsize=(W, H))

ax_a = fig.add_axes(rect(0.92, 7.40, 5.38, 1.08))
ax_b = fig.add_axes(rect(0.92, 4.24, 5.38, 1.22))
ax_c = fig.add_axes(rect(0.92, 1.26, 5.38, 0.98))

BW = 0.26

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
ax_a.set_ylim(-7.8, 3.6)
ax_a.set_yticks([-6, -4, -2, 0, 2])
ax_a.set_ylabel("rel. QLIKE\nimprovement over\nrecalibrated\nHAR (%)")
fig.text(0.30 / W, 8.62 / H,
         "(a)  Input parity, 10-K/Q only: what changes is how the text is "
         "elicited, not which excerpts were curated",
         fontsize=9.8, color=GREY, va="bottom", ha="left")

handles_a = [Patch(fc=col, ec="white", hatch=hh,
                   label=f"{lab} -- genuine in {int(s.genuine.sum())} of 3")
             for lab, s, col, hh in SER_A]
fig.legend(handles=handles_a[::-1], loc="upper left",
           bbox_to_anchor=(0.30 / W, 7.10 / H), ncol=1, handlelength=1.5,
           labelspacing=0.24, borderpad=0.0, handletextpad=0.5, fontsize=9)

note_a = (
    "'Genuine' is the committed flag: clustered DM < 0, Holm < .05 within this "
    "table's one 12-cell family, and |placebo DM| < 2. The comparison is at "
    "parity of INPUT, not of parameter count -- a 32B decoder against an 8B "
    "embedder with a ridge head -- so the phrase is same-lineage, not "
    "same-size, and 10-K/Q is the only channel with a parity cell."
)
fig.text(0.30 / W, 6.54 / H, para(note_a, 101), fontsize=9, color=GREY,
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
for xi, v, r, w in zip(xb, df_v, rp_v, well_flat):
    ax_b.annotate(f"{r:.0f}%", (xi, v), textcoords="offset points",
                  xytext=(0, 3), ha="center", fontsize=9,
                  color=VERM_TXT if not w else GREY, zorder=6)
ax_b.annotate(f"ratio {rp_v[2]:.0f}% on a {ft_v[2]:+.2f}%\n"
              "denominator -- uninformative",
              (xb[2], df_v[2]), textcoords="offset points", xytext=(6, 20),
              ha="left", va="bottom", fontsize=9, color=VERM_TXT,
              linespacing=1.28)
ax_b.set_xticks(xb)
ax_b.set_xticklabels(lab_b, linespacing=1.28)
ax_b.set_xlim(-0.62, 5.62)
ax_b.set_ylim(-2.0, 3.15)
ax_b.set_yticks([-2, -1, 0, 1, 2])
ax_b.set_ylabel("rel. QLIKE\nimprovement over\nrecalibrated\nHAR (%)")
fig.text(0.30 / W, 5.74 / H,
         "(b)  Contamination arms: percentages on the middle bar are what a "
         "zero-content prompt reproduces",
         fontsize=9.8, color=GREY, va="bottom", ha="left")

handles_b = [Patch(fc=LIGHT, ec=GREY, hatch="..", label="date only"),
             Patch(fc=YELLOW, ec="white", hatch="//",
                   label="date + ticker (zero document content)"),
             Patch(fc=BLUE, ec="white", label="full text")]
fig.legend(handles=handles_b, loc="upper left",
           bbox_to_anchor=(0.30 / W, 3.75 / H), ncol=3, handlelength=1.5,
           labelspacing=0.24, columnspacing=1.6, borderpad=0.0,
           handletextpad=0.5, fontsize=9)

note_b = (
    f"The date-only arm is significantly positive in {do_sig_pos} of 6 cells; "
    f"three sit at machine-zero ({do_near_zero[0]:+.3f}%, "
    f"{do_near_zero[1]:+.3f}%, {do_near_zero[2]:+.0e}%), so the frozen "
    "'positive in 0 of 6' is a significance statement, not a sign count, and "
    f"the two negative 10-K/Q cells ({do_v[1]:+.2f}%, {do_v[2]:+.2f}%) are "
    "drawn as measured. The reproduction column is a RATIO in the source, "
    f"rendered here as a percentage; it is interpretable in the {n_well} of 6 "
    "cells where full text is at least 1% and Holm-significant "
    f"({repro_well.min():.0f}--{repro_well.max():.0f}%), and the other three "
    "are printed in the attention colour, not hidden. With the "
    "date-plus-ticker forecast in the reference, full text still adds in "
    f"{n_joint_holm} of 6 cells under the joint block's own Holm."
)
fig.text(0.30 / W, 3.52 / H, para(note_b, 101), fontsize=9, color=GREY,
         va="top", ha="left", linespacing=1.30)

# --------------------------------------------- (c) the 70B zero-content probe
xc = np.arange(3)
pv = probe.rel_har.to_numpy()
av = anchor.rel_har.to_numpy()
sv = share.share_har_pct.to_numpy()
hatches = ["", "", "xxx"]
for i, (v, col, base_h) in enumerate(((pv, VERM, "//"), (av, GREEN, ""))):
    for j in range(3):
        hh = "xxx" if j == 2 else base_h
        ax_c.bar(xc[j] + (i - 0.5) * BW * 1.15, v[j], width=BW * 1.15,
                 color=col, edgecolor="white", lw=0.7, hatch=hh, zorder=3)
for j in range(3):
    # one line only: a three-line label above a 0.22%-high bar collides with
    # the neighbouring full-text bar's own value label at h=20
    ax_c.annotate(f"{sv[j]:.1f}%", (xc[j] - 0.5 * BW * 1.15, pv[j]),
                  textcoords="offset points", xytext=(0, 3), ha="center",
                  va="bottom", fontsize=9, color=VERM_TXT, zorder=6)
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
fig.text(0.30 / W, 2.33 / H,
         "(c)  The same zero-content construction inside the 70B family "
         "(8-K only)", fontsize=9.8, color=GREY, va="bottom", ha="left")
handles_c = [Patch(fc=VERM, ec="white", hatch="//",
                   label="date + ticker probe (70B)"),
             Patch(fc=GREEN, ec="white",
                   label="full text (70B, 3-run ensemble)"),
             Patch(fc="white", ec=GREY, hatch="xxx",
                   label="denominator not well identified")]
fig.legend(handles=handles_c, loc="upper left",
           bbox_to_anchor=(0.30 / W, 0.96 / H), ncol=3, handlelength=1.5,
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
fig.text(0.30 / W, 0.73 / H, para(note_c, 101), fontsize=9, color=VERM_TXT,
         va="top", ha="left", linespacing=1.30)

finish(fig, "F13_elicitation_not_curation")
