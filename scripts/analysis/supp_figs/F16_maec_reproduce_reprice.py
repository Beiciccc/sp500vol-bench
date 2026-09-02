"""F16 -- Portability II: MAEC, reproduce then reprice.

Substantiates, and bounds the scope of, these frozen main-text sentences:
  00_abstract.tex: "on MAEC it reprices a published gain to nothing detectable
    (0 of 24)".
  08_discussion.tex: "On MAEC (3,443 calls) we supply the forecast-level
    counterpart to Yu et al.: the published gain reproduces, then reprices to
    nothing detectable under recalibration plus a same-ticker mean (0 of 24
    cells; MDE 3.0--12.8%), and a zero-content probe matches or exceeds the
    full-transcript arm at three of four horizons. Each null sits beside its
    MDE; the audit is powered against the published gains, effects below them
    not ruled out."
  02_related.tex: "on which MAEC's published gain reprices to nothing
    detectable."

Evidence files read (every plotted number comes from one of these):
  results/tables/maec_audit.md  -- section 1, the published-convention context
      block (raw pooled MSE, text pooled MSE, per-cent of raw, and the G1 counts
      12/12 and 4/12); section 5, the minimum-detectable-effect adjudication and
      the proxy disclosure; section 7, the alignment and gate disclosures
  results/tables/maec_audit.csv -- arm, alignment, horizon, ref, stage, rel_pct,
      p_holm, mde_ent_pct, boot_ci, gate, identity_share

Provenance note: results/second_domain/maec/published_readings.json is the
upstream provenance of the section-1 block. It sits outside the declared
evidence directories and its preds_dir field holds an author-side absolute path,
so it is neither read nor quoted here; the bars come from the audit table.

Conventions on the artefact itself, because the two panels do NOT share one:
  panel (a): pooled mean squared error in v-squared units, transported from the
      published three annual panels (2015, 2016, 2017-18) merged row-equal-
      weight. The published panels differ from this audit's test window
      (2017-05 to 2018-06) in period and in level, so the transport is an
      approximation the source itself discloses; the bars are descriptive and
      enter no correction.
  panel (b): relative MSE improvement over a reference that already contains a
      recalibrated price forecast AND a same-ticker expanding mean, per cent of
      that reference's own MSE -- the same denominator and unit as the minimum
      detectable effect drawn beside it. Call-date-clustered Diebold-Mariano,
      Holm over 8 cells per arm, date-block bootstrap interval.

What the figure does not show: nothing here is a QLIKE quantity and nothing
shares a denominator with the SEC panel; the identity share tabulated in the
source is the entity term's share of what text adds over an ALREADY-recalibrated
reference, not a share of the published-convention gain.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supp_style import (apply_style, finish, gate, note, BLUE, SKY, VERM,  # noqa: E402
                        VERM_TXT, GREEN, YELLOW, PURPLE, GREY, LIGHT, INK,
                        INK2, RULE, TAB, AGG, REPO)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# --------------------------------------------------------------- load evidence
MD = open(os.path.join(TAB, "maec_audit.md")).read()
CSV = pd.read_csv(os.path.join(TAB, "maec_audit.csv"))

# section 1: the published-convention block (raw vs text pooled MSE, % of raw)
PUB = pd.DataFrame(
    [(a, int(h), float(raw), float(txt), float(dpub), float(pct))
     for a, h, raw, txt, dpub, pct in re.findall(
         r"\| (tfidf|prompted_qwen) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| "
         r"([+\-][\d.]+) \| ([+\-][\d.]+)% \|", MD)],
    columns=["arm", "h", "raw_mse", "text_mse", "d_pub", "pct_of_raw"])
m = re.search(r"tfidf (\d+)/(\d+) cells", MD)
G1_TFIDF = (int(m.group(1)), int(m.group(2)))
m = re.search(r"prompted_qwen (\d+)/(\d+) \(pooled", MD)
G1_PROMPT = (int(m.group(1)), int(m.group(2)))
TEX = open(os.path.join(REPO, "writing", "paper", "sections",
                        "08_discussion.tex")).read()
m = re.search(r"\(([\d{},]+) calls\)", TEX)
N_CALLS = int(re.sub(r"[{},]", "", m.group(1)))
m = re.search(r"n_test=(\d+), (\d+) call-date clusters, (\d+) entities, "
              r"n_val=(\d+)", MD)
N_TEST, N_CLUST, N_ENT, N_VAL = (int(g) for g in m.groups())

# section 2/3: the 24 primary identity-controlled residual cells
R5 = CSV[(CSV.alignment == "primary") & (CSV.stage == "row5")
         & (CSV.arm != "identity_probe")].copy()
ORDER = {"tfidf": 0, "qwen_emb": 1, "prompted_qwen": 2}
R5["ao"] = R5.arm.map(ORDER)
R5["ro"] = R5.ref.map({"r_ar": 0, "r_har": 1})
R5 = R5.sort_values(["ao", "horizon", "ro"]).reset_index(drop=True)
CI = R5.boot_ci.str.strip("[]").str.split(",", expand=True).astype(float)
R5["ci_lo"], R5["ci_hi"] = CI[0], CI[1]
R5["dirty"] = R5.gate.str.contains("G4b:dirty")
R5["inside_mde"] = R5.rel_pct.abs() <= R5.mde_ent_pct
R5["ci_excludes_zero"] = (R5.ci_lo > 0) | (R5.ci_hi < 0)

# section 7: the two shifted-alignment prompted cells that are significantly
# harmful, kept in frame rather than dropped
SH = CSV[(CSV.alignment == "shifted") & (CSV.stage == "row5")
         & (CSV.arm == "prompted_qwen") & (CSV.gate.str.startswith("neg_sig"))]

# ------------------------------------------------------------------------ gate
# The literal side of gate() is the only place a number is typed by hand; it
# exists to abort the build the moment the committed tables stop saying what the
# frozen main text says.
gate(
    {
        "n_published_cells": 8,
        "tfidf_pct_of_raw": (53.1, 38.0, 32.8, 13.6),
        "prompt_pct_of_raw": (27.2, -28.6, -50.0, -72.4),
        "g1_tfidf": (12, 12), "g1_prompt": (4, 12),
        "n_primary_cells": 24, "n_clear_holm": 0, "n_inside_mde": 24,
        "mde_lo_hi": (2.97, 12.80), "n_dirty_gate": 4,
        "n_ci_excludes_zero": 2,
        "n_shifted_neg_sig": 2, "shifted_neg_sig_h": (15, 15),
        "n_calls": 3443, "panel_shape": (672, 143, 461, 333),
    },
    {
        "n_published_cells": int(len(PUB)),
        "tfidf_pct_of_raw": tuple(
            round(float(v), 1) for v in
            PUB[PUB.arm == "tfidf"].sort_values("h", key=lambda s:
                                                s.map({3: 0, 7: 1, 15: 2, 30: 3})
                                                ).pct_of_raw),
        "prompt_pct_of_raw": tuple(
            round(float(v), 1) for v in
            PUB[PUB.arm == "prompted_qwen"].sort_values(
                "h", key=lambda s: s.map({3: 0, 7: 1, 15: 2, 30: 3})).pct_of_raw),
        "g1_tfidf": G1_TFIDF, "g1_prompt": G1_PROMPT,
        "n_primary_cells": int(len(R5)),
        "n_clear_holm": int((R5.p_holm < .05).sum()),
        "n_inside_mde": int(R5.inside_mde.sum()),
        "mde_lo_hi": (round(float(R5.mde_ent_pct.min()), 2),
                      round(float(R5.mde_ent_pct.max()), 2)),
        "n_dirty_gate": int(R5.dirty.sum()),
        "n_ci_excludes_zero": int(R5.ci_excludes_zero.sum()),
        "n_shifted_neg_sig": int(len(SH)),
        "shifted_neg_sig_h": tuple(int(v) for v in SH.horizon),
        "n_calls": N_CALLS,
        "panel_shape": (N_TEST, N_CLUST, N_ENT, N_VAL),
    },
)

# ---------------------------------------------------------------- presentation
ARM_LABEL = {"tfidf": "TF-IDF ridge", "qwen_emb": "frozen Qwen3 embedding",
             "prompted_qwen": "prompted Qwen3"}
ARM_COL = {"tfidf": BLUE, "qwen_emb": SKY, "prompted_qwen": PURPLE}
REF_LABEL = {"r_ar": "AR", "r_har": "HAR"}
HORIZONS = [3, 7, 15, 30]

apply_style(base_size=9)
# Vertical space is allocated top-down in inches so that no text block can drift
# into another, and every free-text line is kept under ~92 characters so the
# finished canvas stays 6.4 in wide.
H = 8.55
fig = plt.figure(figsize=(6.4, H))


def frac(inches_from_top):
    return 1.0 - inches_from_top / H


# ------------------------------------------------------------- apparatus band
# The three long blocks at the foot of each panel are basis statements, not the
# figure's argument, and they used to be set in the same ink as the data labels.
# They are now recessive (INK2) with a hairline closing the panel above them, and
# the panel marker is lifted out of the prose into a hanging position in primary
# ink so a reader can find "(a)" without reading the sentence. Every word of
# every block is unchanged.
#
# Two constraints shaped how this is done, and both are geometric. This figure is
# included height-bound, so the page scales it by the ratio of the float cap to
# the content box: any block that grows downward or upward lowers every printed
# point size in the figure. Hence (i) the marker stays at 9 pt -- panel() sets
# 10 pt, which would buy a findable marker by shrinking all the other type -- and
# (ii) the marker hangs into the gutter left of the text rather than occupying a
# line of its own, which would have cost a whole line of height.
X_MARK = 0.0145          # hanging panel marker, flush with panel (a)'s y-label
X_BODY = 0.0535          # apparatus prose, indented clear of the marker


def hairline(inches_from_top, x0=0.0140, x1=0.9845):
    """note()'s own hairline, placed at a measured height instead of a fixed one.

    note() offsets its rule a fixed distance above the text anchor; for the foot
    block that offset lands on panel (b)'s x-label, so the two rules are drawn
    here in note()'s colour and weight at heights measured from the rendered
    figure. Both ends sit inside content that is already present -- panel (a)'s
    y-label on the left, panel (b)'s frame on the right -- so the tight bounding
    box, and with it every printed point size, is unchanged.
    """
    fig.lines.append(Line2D([x0, x1], [frac(inches_from_top)] * 2,
                            transform=fig.transFigure, color=RULE,
                            linewidth=0.5, zorder=0.5))


def marker(letter, inches_from_top):
    """The panel letter, in primary ink, on the first baseline of its note."""
    return fig.text(X_MARK, frac(inches_from_top), f"({letter})", fontsize=9,
                    fontweight="bold", color=INK, ha="left", va="top")


# ------------------------- panel (a): the published-convention reproduction ---
axa = [fig.add_axes([0.090, frac(2.74), 0.375, 1.60 / H]),
       fig.add_axes([0.580, frac(2.74), 0.375, 1.60 / H])]
for k, arm in enumerate(("tfidf", "prompted_qwen")):
    ax = axa[k]
    sub = PUB[PUB.arm == arm].set_index("h")
    for i, h in enumerate(HORIZONS):
        ax.bar(i - 0.19, float(sub.loc[h, "raw_mse"]), width=0.34, color=GREY,
               zorder=3)
        ax.bar(i + 0.19, float(sub.loc[h, "text_mse"]), width=0.34,
               color=ARM_COL[arm], zorder=3)
        top = max(float(sub.loc[h, "raw_mse"]), float(sub.loc[h, "text_mse"]))
        ax.annotate(f"{float(sub.loc[h, 'pct_of_raw']):+.1f}%", xy=(i, top),
                    xytext=(0, 3), textcoords="offset points", fontsize=9,
                    color=(GREEN if float(sub.loc[h, "pct_of_raw"]) > 0
                           else VERM_TXT),
                    ha="center", va="bottom")
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"{h}" for h in HORIZONS], fontsize=9)
    ax.set_xlabel("horizon, trading days", fontsize=9)
    ax.set_ylim(0, 1.72)
    ax.set_xlim(-0.62, 3.62)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
if True:
    axa[0].set_ylabel("pooled MSE, $v^2$ units", fontsize=9)
    axa[1].set_yticklabels([])
# Each sub-panel's headline is set in its own arm's hue, the way panel (b)
# already labels its three arm blocks. That is the only thing binding a headline
# to the bars it counts, and it costs no geometry: colour, not size.
axa[0].text(0.0, 1.045,
            f"TF-IDF: beats raw in {G1_TFIDF[0]} of {G1_TFIDF[1]} cells",
            transform=axa[0].transAxes, fontsize=9, fontweight="bold",
            color=ARM_COL["tfidf"], ha="left", va="bottom")
axa[1].text(0.0, 1.045,
            f"prompted: beats raw in {G1_PROMPT[0]} of {G1_PROMPT[1]} cells",
            transform=axa[1].transAxes, fontsize=9, fontweight="bold",
            color=ARM_COL["prompted_qwen"], ha="left", va="bottom")

fig.legend(handles=[
    Line2D([], [], ls="none", marker="s", ms=7, color=GREY,
           label="raw past-volatility baseline"),
    Line2D([], [], ls="none", marker="s", ms=7, color=BLUE, label="TF-IDF text"),
    Line2D([], [], ls="none", marker="s", ms=7, color=PURPLE,
           label="prompted text")],
    loc="upper left", bbox_to_anchor=(0.022, frac(0.46)), ncol=3, fontsize=9,
    handletextpad=0.3, columnspacing=1.4, frameon=False)

marker("a", 0.06)
note(fig, X_BODY, frac(0.06),
     f"The published convention reproduced. Counts are over the "
     f"{G1_TFIDF[1]} panel $\\times$ horizon cells;\n"
     f"the bars merge the three annual panels row-equal-weight, a transport "
     f"the source calls approximate.",
     rule=False)

# --------------------- panel (b): the identity-controlled residual, 24 cells ---
axb = fig.add_axes([0.200, frac(7.52), 0.785, (7.52 - 3.96) / H])
rows, labels, y = [], [], 0.0
blocks = []
for arm in ("tfidf", "qwen_emb", "prompted_qwen"):
    if rows:
        y -= 0.9
    blocks.append((arm, y))
    for _, r in R5[R5.arm == arm].iterrows():
        rows.append((y, r))
        labels.append(f"$h$={int(r.horizon)}  {REF_LABEL[r.ref]}"
                      + ("$^{\\dagger}$" if bool(r.dirty) else ""))
        y -= 1.0

axb.axvline(0.0, lw=0.9, color=GREY, zorder=3)
for yy, r in rows:
    mde = float(r.mde_ent_pct)
    col = ARM_COL[r.arm]
    axb.add_patch(Rectangle((-mde, yy - 0.34), 2 * mde, 0.68, facecolor=LIGHT,
                            edgecolor="none", zorder=1))
    # MDE values sit in a fixed right-hand column, never beside the band, so
    # they cannot collide with a bootstrap whisker that runs past the band.
    # The MDE column is the instrument, not the estimate: recessive ink, the
    # same register as the grey band it quantifies, so the coloured point and
    # its whisker read as the finding and the band as what could be resolved.
    axb.text(13.4, yy, f"$\\pm${mde:.2f}", fontsize=9, color=INK2,
             ha="left", va="center", zorder=4)
    axb.plot([float(r.ci_lo), float(r.ci_hi)], [yy, yy], lw=1.2, color=col,
             zorder=4, solid_capstyle="butt")
    for xx in (float(r.ci_lo), float(r.ci_hi)):
        axb.plot([xx, xx], [yy - 0.22, yy + 0.22], lw=1.0, color=col, zorder=4)
    axb.plot([float(r.rel_pct)], [yy], marker="o", ms=5.0, mfc=col, mec=col,
             zorder=5)
    if bool(r.ci_excludes_zero):
        axb.plot([float(r.rel_pct)], [yy], marker="o", ms=10.0, mfc="none",
                 mec=VERM, mew=1.2, zorder=6)

axb.set_yticks([yy for yy, _ in rows])
axb.set_yticklabels(labels, fontsize=9)
axb.set_ylim(rows[-1][0] - 0.75, rows[0][0] + 1.25)
axb.set_xlim(-14.6, 17.2)
axb.set_xticks([-10, -5, 0, 5, 10])
axb.set_xlabel("relative MSE improvement over the recalibrated reference plus "
               "the same-ticker mean, %", fontsize=9)
axb.tick_params(axis="y", length=0)
for arm, yy in blocks:
    axb.text(-14.3, yy + 0.80, ARM_LABEL[arm], fontsize=9, fontweight="bold",
             color=ARM_COL[arm], ha="left", va="center")
axb.text(13.4, rows[0][0] + 0.80, "MDE", fontsize=9, color=INK2, ha="left",
         va="center")

# The hairline closes panel (a); the block below it belongs to panel (b), which
# is drawn under it. The block is raised 0.06 in from where it used to sit: its
# last line was touching the "TF-IDF ridge" label, and the gap above it was
# empty. Moving a block into white space already inside the bounding box costs
# nothing, so the crowding is repaired without a single glyph changing size.
hairline(3.186)
marker("b", 3.24)
note(fig, X_BODY, frac(3.24),
     f"All {len(R5)} primary-alignment cells (3 arms $\\times$ 4 "
     f"horizons $\\times$ 2 references). Grey band: that cell's own\n"
     f"minimum detectable effect, $\\pm${float(R5.mde_ent_pct.min()):.2f}% "
     f"to $\\pm${float(R5.mde_ent_pct.max()):.2f}%. Whiskers: date-block "
     f"bootstrap. Every residual falls\n"
     f"inside its own band and {int((R5.p_holm < .05).sum())} of "
     f"{len(R5)} clear the Holm correction, so this bounds the residual at "
     f"the size of the\n"
     f"published gain and does not demonstrate a zero. "
     f"$\\dagger$ = within-date-swap gate recorded unclean "
     f"({int(R5.dirty.sum())} cells).",
     rule=False)

ring = R5[R5.ci_excludes_zero]
# Adversarial repair: "while the clustered test does not" was true only after
# Holm. One of the two ringed cells rejects at the raw 5% level (p = .027) and
# is removed only by the 8-cell correction, so the qualifier and both raw p are
# printed here rather than left to the prose.

def p_(v, nd=3):
    """Format a p-value the way the supplement's tables do (no leading zero)."""
    return f"{float(v):.{nd}f}".lstrip("0")


# The second hairline closes panel (b). The block below it carries no panel
# letter because it is figure-level: the empty marker gutter is what says so.
hairline(7.944)
note(fig, X_BODY, frac(7.98),
     f"Ringed: {len(ring)} cells whose bootstrap interval excludes zero "
     f"while the Holm-corrected clustered test does\n"
     f"not (raw $p$ "
     + ", ".join(p_(v) for v in ring.p_raw)
     + "; Holm "
     + ", ".join(p_(v) for v in ring.p_holm)
     + f"). The prompted arm is negative at three of four horizons;\n"
       f"under the shifted alignment its two $h$="
     + "/".join(str(int(v)) for v in sorted(set(SH.horizon)))
     + f" cells are significantly harmful (Holm "
     + ", ".join(p_(v, 4) for v in SH.p_holm) + ").",
     rule=False)

finish(fig, "F16_maec_reproduce_reprice")
