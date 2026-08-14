"""THE paper figure — reference-ladder survivor waterfall + power calibration.

One glance = survivors collapse as references strengthen, and even known
injected signals rarely clear the last rung (the conjunction is a strict
selection device with low power, not evidence of absence).

Sources (committed tables; the script ABORTS if any count drifts):
  results/tables/control_intersection_ensemble.csv
      -> Holm survivors on the ensemble basis: primary 38/69, firm 8/69,
         maximal pool 9/69, full conjunction 0/69.
  results/tables/vix_control.csv
      -> 19 of the 38 originally-genuine cells survive the VIX-augmented
         reference (denominator 38, not 69 — disclosed on the tick label).
  results/tables/signal_injection_power.csv
      -> full-conjunction recovery of a firm-orthogonal signal injected at
         0.3/0.5/1.0% rel-QLIKE: 2/6/13 of 69 (column all3_detect).
  results/tables/firm_identity_ensemble.csv
      -> zero-text firm-mean reference vs plain f_R. firm_beats_fR is a
         clustered-DM test on a quantity with ONE distinct value per
         (channel, horizon), so the honest unit is the comparison: 4 of 6,
         not 53 of 69 cells. Deliberately NOT drawn on the survivor axis --
         it is not a Holm-survivor count and a rule at 53 would invite a
         false reading against 38/19/8/9/0.

Writes the manuscript figure at native size. Typographic compliance:
3.15 x 1.75 in (included at 0.95\\columnwidth ~ 1:1, so native pt sizes are
the effective ones), ALL text >= 9pt, and every glyph is converted to
outlines via ghostscript (-dNoOutputFonts) so the final PDF embeds NO fonts.
Okabe-Ito colour-blind-safe palette; distinctions never colour-only.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TAB = os.path.join(REPO, "results", "tables")
OUT = os.path.join(REPO, "writing", "paper", "figures", "cascade_power.pdf")

# ---------------------------------------------------------------- load counts
ci = pd.read_csv(os.path.join(TAB, "control_intersection_ensemble.csv"))
n_cells = len(ci)
n_primary = int(ci.primary_holm.sum())      # 38
n_firm = int(ci.firm_holm.sum())            # 8
n_pool = int(ci.maximal_holm.sum())         # 9
n_conj = int(ci.AND_full_holm.sum())        # 0

vx = pd.read_csv(os.path.join(TAB, "vix_control.csv"))
n_vix_base = int(vx.orig_genuine.sum())                          # 38
n_vix = int((vx.orig_genuine & vx.survives_vix).sum())           # 19

sp = pd.read_csv(os.path.join(TAB, "signal_injection_power.csv"))
rec = sp.groupby("target_pct").all3_detect.sum()                 # 2 / 6 / 13
recovery = {lvl: int(rec[lvl]) for lvl in (0.3, 0.5, 1.0)}

# Zero-text firm-mean control. firm_beats_fR is (clustered DM < 0 and p < .05)
# on a quantity that takes ONE distinct value per (channel, horizon) -- every
# model in a group shares the same firm-mean-vs-f_R comparison. So the honest
# unit is the comparison, not the cell: 53 cells are 4 comparisons, and the 15
# from long-form h=20 come from a group whose point estimate is negative
# (-1.39) against DM -3.13. This is NOT a Holm-survivor count and must not be
# drawn on the survivor axis.
fi = pd.read_csv(os.path.join(TAB, "firm_identity_ensemble.csv"))
_g = fi.groupby(["disc", "h"]).firm_beats_fR.sum()
ZEROTEXT_GROUPS = int((_g > 0).sum())      # 4
ZEROTEXT_TOTAL_GROUPS = len(_g)       # 6
ZEROTEXT_CELLS = int(fi.firm_beats_fR.sum())  # 53, kept for the gate only

# ------------------------------------------------------- gates (paper-bound)
expect = dict(n_cells=69, n_primary=38, n_firm=8, n_pool=9, n_conj=0,
              n_vix_base=38, n_vix=19)
got = dict(n_cells=n_cells, n_primary=n_primary, n_firm=n_firm, n_pool=n_pool,
           n_conj=n_conj, n_vix_base=n_vix_base, n_vix=n_vix)
if got != expect or recovery != {0.3: 2, 0.5: 6, 1.0: 13}:
    sys.exit(f"GATE FAIL — table counts drifted from paper: {got}, {recovery}")

# ------------------------------------------------------------------- palette
BLUE = "#0072B2"      # Okabe-Ito blue: survivor bars
SKY = "#3B8FC4"       # darkened Okabe-Ito sky blue (3.55:1 on white, clears the 3:1 graphics floor): VIX rung (denominator 38, not 69)
VERM = "#C85800"      # deepened Okabe-Ito vermillion, hue kept: injected-signal
                      # diamonds (graphic floor 3:1 — 4.33:1 white, 3.87:1 on the
                      # #F2F2F2 band; deepened from #D55E00 for headroom)
VERM_TXT = "#A34700"  # darkened vermillion for TEXT (6.07:1, clears 4.5:1)
GREY = "#3A3A3A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,               # floor: nothing below 9pt at 1:1 scale
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,           # irrelevant post-outlining, kept for the raw
    "ps.fonttype": 42,
})

FS = 10  # single knob — every text element uses >= this

fig, ax = plt.subplots(figsize=(3.15, 1.75))
fig.subplots_adjust(left=0.18, right=0.99, top=0.98, bottom=0.205)

xs = [0, 1, 2, 3, 4]
vals = [n_primary, n_vix, n_firm, n_pool, n_conj]
# Exemplar-polish redesign (74-chart census of accepted papers): single-hue
# dark->light ramp over the rungs — the cascade "fades out to nothing", which
# IS the story; every shade >= 3:1 on white (kit graphics floor). The VIX bar
# keeps its dashed outline as the redundant denominator-switch cue.
RAMP = ["#0072B2", "#3B8FC4", "#4191C5", "#4796C8", BLUE]
bars = ax.bar(xs, vals, width=0.62, color=RAMP, zorder=3)
bars[1].set_edgecolor("#005A8C"); bars[1].set_linewidth(0.8)
bars[1].set_linestyle((0, (2.5, 1.5)))

# neutral "conclusion zone" band behind the conjunction rung + power diamonds
# (census: tinted verdict band; gray, never a warm hue — the verdict is a null)
ax.axvspan(3.58, 4.92, color="#F2F2F2", zorder=0)

# count labels: 10pt bold near-black; VIX two-layer (bold 19 + small /38);
# the money number "0" is the largest glyph in the chart — typographic
# emphasis, no color (warm hues stay reserved for the injected-signal system)
INK = "#1A1A1A"
for x, v in zip(xs[:4], vals[:4], strict=False):
    if x == 1:
        ax.text(x - 0.06, v + 1.5, str(v), ha="right", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(x - 0.02, v + 1.7, "/38", ha="left", va="bottom",
                fontsize=FS, color="#595959")
    else:
        ax.text(x, v + 1.5, str(v), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
ax.text(4, 1.6, "0", ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=INK)

# ---- (a) zero-text firm-mean marker line -----------------------------------
# Reference line labeled in place, same gray family (census: color-matched
# labeled baseline, no legend lookup); one line, ylim tightened 74 -> 64.
# No rule: this count lives on a different scale from the survivor axis, so a
# horizontal line would invite a false comparison with 38/19/8/9/0. Stated as
# a note in its own units instead, anchored above the firm-identity rung.
ax.annotate(f"zero-text firm mean alone beats $f_R$\n"
            f"in {ZEROTEXT_GROUPS} of {ZEROTEXT_TOTAL_GROUPS} "
            f"channel\u00d7horizon comparisons",
            xy=(2, n_firm + 5.0), xytext=(-0.46, 47),
            fontsize=FS, color="#595959", ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#A6A6A6",
                            shrinkA=2, shrinkB=3))

# ---- (b) injected-signal recovery at the conjunction rung ------------------
dx = 4.3  # diamond column, just right of the (empty) conjunction bar
ys = [recovery[0.3], recovery[0.5], recovery[1.0]]
ax.plot([dx, dx], [min(ys), max(ys)], lw=0.7, color=VERM, zorder=4)
ax.scatter([dx] * 3, ys, marker="D", s=12, facecolor="white",
           edgecolor=VERM, linewidth=1.0, zorder=5)
# label y-positions nudged apart (2/6 are only ~5pt apart at this scale —
# 9pt digits would collide); each stays adjacent to its own diamond
label_y = {recovery[0.3]: 2.0, recovery[0.5]: 7.6, recovery[1.0]: 13.9}
for y in ys:
    ax.text(dx + 0.14, label_y[y], str(y), ha="left", va="center",
            fontsize=FS, color=VERM_TXT)
ax.text(4.92, 18.5, "recovery of signal\ninjected at 0.3/0.5/1.0%",
        ha="right", va="bottom", fontsize=FS, color=VERM_TXT, linespacing=1.15)

# ------------------------------------------------------------------ cosmetics
# Minimal chrome (census norm): top/right spines off, thin left spine, a
# heavier true-zero floor the "0" visibly sits on, sparse ticks.
ax.set_xlim(-0.6, 4.95)
ax.set_ylim(0, 64)
ax.set_yticks([0, 20, 40])
ax.set_ylabel("Holm survivors\n(of 69 cells)", fontsize=FS, color="#333333",
              linespacing=1.05)
ax.set_xticks(xs)
ax.set_xticklabels(["recalib.\nHAR", "+VIX\n(of 38)", "firm\nidentity",
                    "maximal\npool", "full\nconjunction"],
                   fontsize=FS, linespacing=1.05)
ax.tick_params(axis="x", length=0, pad=2)
# "conjunction" (widest tick line) would sit flush against "pool" at 9pt:
# shift the last tick label ~5pt right into the free space under the diamonds
_off = mtransforms.ScaledTranslation(5 / 72, 0, fig.dpi_scale_trans)
_last = ax.get_xticklabels()[4]
_last.set_transform(_last.get_transform() + _off)
ax.tick_params(axis="y", labelsize=FS, pad=1.5, colors="#333333")
ax.grid(axis="y", lw=0.4, color="#CCCCCC", alpha=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#595959"); ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_color("#333333"); ax.spines["bottom"].set_linewidth(1.0)

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ------------------------------------------- save + outline (zero-font PDF)
# Submission checkers reject the CID TrueType/Identity-H fonts matplotlib embeds,
# so the committed figure carries NO fonts at all: ghostscript pdfwrite with
# -dNoOutputFonts converts every glyph to vector outlines.
GS = next((g for g in (shutil.which("gs"), "/usr/local/bin/gs",
                       "/opt/homebrew/bin/gs")
           if g and os.path.exists(g)), None)
if GS is None:
    sys.exit("GATE FAIL — ghostscript not found; cannot outline fonts "
             "(the committed figure must embed zero fonts)")

with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    raw = tmp.name
try:
    fig.savefig(raw)
    subprocess.run([GS, "-o", OUT, "-sDEVICE=pdfwrite", "-dNoOutputFonts",
                    "-dQUIET", "-dBATCH", "-dNOPAUSE", raw], check=True)
finally:
    os.unlink(raw)

fig.savefig(os.path.splitext(OUT)[0] + "_preview.png", dpi=300)
print(f"wrote {OUT} (fonts outlined via {GS})")
