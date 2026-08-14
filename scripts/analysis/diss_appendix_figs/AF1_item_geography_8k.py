"""AF1 -- the item geography of the surviving 8-K residual.

Where inside the event-driven channel does the prompted arm's increment over the
firm-identity reference actually live?  The pooled residual is decomposed by 8-K
item code into six disjoint groups, for two prompted families, at three horizons.

Sources (every plotted number is read from this file in this run; nothing is
hardcoded outside the gate() expectations, whose job is to abort on drift):
  results/tables/row11_item_stratified.csv
      family, item_group, kind, horizon, is_partition, n_test, n_days,
      share_of_filings_pct, rel_firm_pct, dm_firm, p_firm, p_firm_holm,
      share_of_pooled_residual_pct, abs_reduction_firm, rel_har_pct

Dissertation sentences this must not contradict:
  chapters/04_results.tex:172  "Those figures sit on the full event-driven panel
      (25,109/25,001/24,732 rows), where the plain firm-identity residual reads
      +0.45/+0.25/+0.20 per cent"
  chapters/04_results.tex:178  the Llama-3.1-70B replication, +0.84/+0.64/+0.38
      on the three-seed ensemble basis (this table is the SINGLE-PASS basis,
      +0.83/+0.64/+0.39; the difference is declared on the figure's face)
  appendices/C_full_results.tex  the full event-driven supports and day counts

Basis trap this figure is built to declare on its face: this table sits on the
FULL event-driven panel (25,109/25,001/24,732), while the reference-ladder rungs
are counted on merged-grid rows (23,855/22,785/22,318).  Percentages are not
interchangeable across those two supports.

CPU only; no model is refitted.
"""
import os
import sys

import pandas as pd

ANALYSIS = "scripts/analysis"
sys.path.insert(0, ANALYSIS)
import diss_style as ds
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from supp_style import (
    BLUE,
    GREY,
    INK,
    INK2,
    LIGHT,
    RULE,
    TAB,
    VERM,
    apply_style,
    gate,
)

# --------------------------------------------------------------- evidence
D = pd.read_csv(os.path.join(TAB, "row11_item_stratified.csv"))
PART = D[D.is_partition].copy()
QW, LL = "qwen3_32b", "llama70_awq"
HS = (5, 10, 20)


def pooled(fam, group, col):
    s = D[(D.family == fam) & (D.item_group == group)].sort_values("horizon")
    return s[col].to_numpy()


def holm_survivors(fam, only_non_earnings=False):
    p = PART[PART.family == fam]
    if only_non_earnings:
        p = p[p.item_group != "2.02_earnings"]
    return int(((p.dm_firm < 0) & (p.p_firm_holm < 0.05)).sum())


def summed_share(fam):
    """Share of the horizon-summed absolute QLIKE reduction carried by 2.02."""
    tot = float(D[(D.family == fam) & (D.item_group == "ALL")]
                .abs_reduction_firm.sum())
    earn = float(PART[(PART.family == fam)
                      & (PART.item_group == "2.02_earnings")]
                 .abs_reduction_firm.sum())
    return 100.0 * earn / tot


def narrative_sig(fam):
    s = D[(D.family == fam) & (D.item_group == "narrative_ALL")]
    return int(((s.dm_firm < 0) & (s.p_firm < 0.05)).sum())


# --------------------------------------------------------------- the gate
gate(
    {
        "n_rows": 48,
        "n_partition_cells": 36,
        "n_test": [25109, 25001, 24732],
        "n_days": [996, 991, 981],
        "qwen_pooled_rel_firm": [0.45, 0.25, 0.20],
        "llama_pooled_rel_firm": [0.83, 0.64, 0.39],
        "qwen_holm": 4,
        "qwen_holm_non_earnings": 2,
        "llama_holm": 1,
        "llama_holm_non_earnings": 0,
        "qwen_narrative_all_sig": 3,
        "llama_narrative_all_sig": 1,
        "qwen_earnings_share": 53.6,
        "llama_earnings_share": 69.3,
        "earnings_filing_share": 33.4,
        "n_item_groups": 6,
    },
    {
        "n_rows": len(D),
        "n_partition_cells": len(PART),
        "n_test": [int(v) for v in pooled(QW, "ALL", "n_test")],
        "n_days": [int(v) for v in pooled(QW, "ALL", "n_days")],
        "qwen_pooled_rel_firm": [round(float(v), 2)
                                 for v in pooled(QW, "ALL", "rel_firm_pct")],
        "llama_pooled_rel_firm": [round(float(v), 2)
                                  for v in pooled(LL, "ALL", "rel_firm_pct")],
        "qwen_holm": holm_survivors(QW),
        "qwen_holm_non_earnings": holm_survivors(QW, True),
        "llama_holm": holm_survivors(LL),
        "llama_holm_non_earnings": holm_survivors(LL, True),
        "qwen_narrative_all_sig": narrative_sig(QW),
        "llama_narrative_all_sig": narrative_sig(LL),
        "qwen_earnings_share": round(summed_share(QW), 1),
        "llama_earnings_share": round(summed_share(LL), 1),
        "earnings_filing_share": round(float(
            PART[PART.item_group == "2.02_earnings"]
            .share_of_filings_pct.mean()), 1),
        "n_item_groups": int(PART.item_group.nunique()),
    },
)

# --------------------------------------------------------------- geometry
apply_style(9)
W, H = 6.10, 7.60
fig = plt.figure(figsize=(W, H))
LINE = 0.152                      # inter-line pitch for 9pt figure text


def rect(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


def yfig(ax, ydata):
    """Figure-fraction y of a data coordinate; call only after set_ylim."""
    y0, y1 = ax.get_ylim()
    p = ax.get_position()
    return p.y0 + (ydata - y0) / (y1 - y0) * p.height


A_LAB, A_GAP = 1.34, 0.075
A_W = (W - A_LAB - 0.10 - 2 * A_GAP) / 3.0
A_BOT, A_HGT = 4.62, 2.42

B_LAB = 1.34
B_BOT, B_HGT = 3.30, 0.70
B_W = W - B_LAB - 0.34

C_LAB = 1.34
C_BOT, C_HGT = 1.56, 0.80
C_W = 1.95

# ---------------------------------------------------------- panel (a) rows
ORDER = ["2.02_earnings", "8.01_other_events", "5.02_leadership",
         "7.01_regFD", "other_narrative", "5.07_shareholder_vote"]
ROWLAB = {
    "2.02_earnings": "2.02 results release",
    "8.01_other_events": "8.01 other events",
    "5.02_leadership": "5.02 leadership",
    "7.01_regFD": "7.01 Regulation FD",
    "other_narrative": "other narrative",
    "5.07_shareholder_vote": "5.07 shareholder vote",
}
#  Row labels are all one ink.  They used to carry a `kind` hue -- earnings in
#  VERM_TXT, narrative and procedural both plain GREY -- so the only coloured
#  words in the label column were vermillion, sitting in the same panel row as
#  the vermillion Llama series and its long vermillion bar.  Vermillion already
#  means Llama-3.1-70B here; it cannot also mean "this item group is earnings".
#  The 2.02 row keeps its prominence from its own bar length and from panel (b),
#  which is entirely about it.
YROWS = ["ALL"] + ORDER
ypos = {g: i for i, g in enumerate(YROWS)}
XMIN, XMAX = -1.15, 3.35

axA = []
for k, h in enumerate(HS):
    ax = fig.add_axes(rect(A_LAB + k * (A_W + A_GAP), A_BOT, A_W, A_HGT))
    axA.append(ax)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(len(YROWS) - 0.55, -0.62)
    ax.axvspan(XMIN, 0, color=LIGHT, alpha=0.30, lw=0, zorder=0)
    ax.axvline(0, color=GREY, lw=0.7, zorder=4)
    for g in YROWS:
        ax.axhline(ypos[g], color=LIGHT, lw=0.5, zorder=0)
    #  The title promises six disjoint groups, but seven identically-ruled rows
    #  were drawn: the pooled total sat in the same visual series as the parts.
    #  A hairline divider says which row is the total and which six sum to it,
    #  and it costs no geometry (it is drawn inside an existing axes).
    ax.axhline(ypos["ALL"] + 0.5, color=RULE, lw=0.9, zorder=1)
    for fam, dy, mk, col in ((QW, -0.19, "o", BLUE), (LL, +0.19, "s", VERM)):
        for r in D[(D.family == fam) & (D.horizon == h)].itertuples():
            if r.item_group not in ypos:
                continue
            y = ypos[r.item_group] + dy
            v = float(r.rel_firm_pct)
            keep = (r.item_group != "ALL" and float(r.dm_firm) < 0
                    and float(r.p_firm_holm) < 0.05)
            ax.plot([0, v], [y, y], color=col, lw=1.5 if keep else 0.8,
                    alpha=1.0 if keep else 0.75, zorder=3,
                    solid_capstyle="butt")
            ax.plot([v], [y], marker=mk, ms=4.4, zorder=5,
                    mfc=col if keep else "white", mec=col,
                    mew=1.4 if keep else 0.9)
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_xticklabels(["-1", "0", "+1", "+2", "+3"])
    ax.tick_params(axis="x", pad=1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(0.5, 1.012, f"$h$ = {h}", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=9, color=GREY)

share = PART.groupby("item_group").share_of_filings_pct.mean().to_dict()
for g in YROWS:
    yf = yfig(axA[0], ypos[g])
    lab = "all 8-K filings" if g == "ALL" else ROWLAB[g]
    sub = "100% of rows" if g == "ALL" else f"{share[g]:.1f}% of rows"
    #  Name in primary ink, denominator recessed: the share of rows is a basis
    #  statement, not a second half of the row's name, and at one ink the two
    #  lines read as a single two-line label of equal standing.
    fig.text((A_LAB - 0.07) / W, yf + 0.0060, lab, ha="right", va="center",
             fontsize=9, color=INK)
    fig.text((A_LAB - 0.07) / W, yf - 0.0074, sub, ha="right", va="center",
             fontsize=8.6, color=INK2)

fig.text((A_LAB + (W - A_LAB - 0.10) / 2) / W, (A_BOT - 0.315) / H,
         "QLIKE improvement over the firm-identity reference (%)",
         ha="center", va="center", fontsize=9, color=GREY)

fig.legend(handles=[
    Line2D([], [], color=BLUE, marker="o", ms=4.4, lw=1.5, mfc=BLUE,
           label="Qwen3-32B"),
    Line2D([], [], color=VERM, marker="s", ms=4.4, lw=1.5, mfc=VERM,
           label="Llama-3.1-70B"),
    # 这个键此前是 GREY:页面上的开口标记是 BLUE 圆与 VERM 方,灰色开口标记
    # 一个都没有,所以键的色相在图里不存在。用 BLUE 让它描述真实的标记形态;
    # "open:" 说的是填充约定,两族通用。
    Line2D([], [], color=BLUE, marker="o", ms=4.4, lw=0.8, mfc="white",
           label="open: Holm $p\\geq$.05")],
    ncol=3, loc="lower center",
    bbox_to_anchor=(0.53, (A_BOT + A_HGT + 0.185) / H),
    handlelength=1.5, handletextpad=0.45, columnspacing=1.6,
    borderpad=0.0, fontsize=9)

fig.text(0.012, (A_BOT + A_HGT + 0.44) / H,
         "(a)  The pooled 8-K residual, split into six disjoint item groups",
         ha="left", va="center", fontsize=9.5, color=GREY, weight="bold")

# ------------------------------------------------- panel (b) composition
axB = fig.add_axes(rect(B_LAB, B_BOT, B_W, B_HGT))
#  The two family bars carry the SAME hues as the two families in panels (a) and
#  (c).  They were SKY and YELLOW, i.e. a second blue and a gold, for the very
#  series that panels (a) and (c) draw in BLUE and VERM -- so a reader met
#  Qwen3-32B in two different blues and Llama-70B in two unrelated hues inside
#  one figure, with nothing saying they were the same two models.
comp = [("share of test rows",
         float(PART[PART.item_group == "2.02_earnings"]
               .share_of_filings_pct.mean()), LIGHT),
        ("Qwen3-32B residual", summed_share(QW), BLUE),
        ("Llama-70B residual", summed_share(LL), VERM)]
axB.set_xlim(0, 103)
axB.set_ylim(-0.62, len(comp) - 0.38)
for i, (lab, val, col) in enumerate(comp):
    y = len(comp) - 1 - i
    axB.barh(y, val, height=0.50, color=col, edgecolor=GREY, lw=0.6, zorder=3)
    axB.barh(y, 100 - val, left=val, height=0.50, color="white",
             edgecolor=GREY, lw=0.6, hatch="////", zorder=3)
    axB.text(val - 1.8, y, f"{val:.1f}%", ha="right", va="center", fontsize=9,
             color=GREY if i == 0 else "white", zorder=6)
axB.set_xticks([0, 25, 50, 75, 100])
axB.set_xticklabels(["0", "25", "50", "75", "100%"])
axB.tick_params(axis="x", pad=1.5)
axB.set_yticks([])
axB.spines["left"].set_visible(False)
for i, (lab, _v, _c) in enumerate(comp):
    fig.text((B_LAB - 0.07) / W, yfig(axB, len(comp) - 1 - i), lab,
             ha="right", va="center", fontsize=9, color=GREY)

fig.text(0.012, (B_BOT + B_HGT + 0.19) / H,
         "(b)  Item 2.02 (solid) is a third of the rows and over half of the "
         "residual",
         ha="left", va="center", fontsize=9.5, color=GREY, weight="bold")

qs = PART[(PART.family == QW) & (PART.item_group == "2.02_earnings")] \
    .sort_values("horizon").share_of_pooled_residual_pct.to_numpy()
ls = PART[(PART.family == LL) & (PART.item_group == "2.02_earnings")] \
    .sort_values("horizon").share_of_pooled_residual_pct.to_numpy()
#  This two-line block is apparatus for panel (b): it says what the bars pool and
#  gives the per-horizon shares the pooling hides.  It was set in the same ink as
#  every data label, and it sat 0.09 in below the "(c)" title against 0.18 in
#  below panel (b)'s tick labels -- twice as close to the panel it does NOT
#  describe.  Recessed to INK2 and lifted 0.10 in, which reverses the proximity
#  and is free: it moves into whitespace the bounding box already contains.
B_NOTE = B_BOT - 0.30
fig.text(0.012, B_NOTE / H,
         "Bars pool the three horizons. Per horizon (h = 5/10/20) Item 2.02's "
         "share of the residual",
         ha="left", va="center", fontsize=8.6, color=INK2)
fig.text(0.012, (B_NOTE - LINE) / H,
         "is " + " / ".join(f"{v:+.1f}" for v in qs) + " % (Qwen3-32B) and "
         + " / ".join(f"{v:+.1f}" for v in ls) + " % (Llama-70B).",
         ha="left", va="center", fontsize=8.6, color=INK2)

# ------------------------------------- panel (c) the earnings-free residual
axC = fig.add_axes(rect(C_LAB, C_BOT, C_W, C_HGT))
rows = [(fam, h) + tuple(
    D[(D.family == fam) & (D.item_group == "narrative_ALL")
      & (D.horizon == h)][["rel_firm_pct", "dm_firm", "p_firm",
                           "n_test"]].iloc[0])
        for fam in (QW, LL) for h in HS]
axC.set_xlim(-5.1, 0.8)
axC.set_ylim(len(rows) - 0.45, -0.55)
#  Non-significant rows keep their family hue and are marked by the open face and
#  the thinner stem -- exactly panel (a)'s convention.  They used to be recoloured
#  GREY, which meant three hues shared this one panel with nothing to say that
#  grey was a state rather than a third model, and it made grey mean "fails DM"
#  here while meaning ink everywhere else.  The p-value is printed on every row,
#  so nothing rests on the recolour.
for i, (fam, h, rel, dm, p, n) in enumerate(rows):
    sig = (dm < 0) and (p < 0.05)
    col = BLUE if fam == QW else VERM
    axC.plot([0, dm], [i, i], color=col, lw=1.6 if sig else 0.9,
             alpha=1.0 if sig else 0.75, zorder=3, solid_capstyle="butt")
    axC.plot([dm], [i], marker="o" if fam == QW else "s", ms=4.4, zorder=5,
             mfc=col if sig else "white", mec=col,
             mew=1.4 if sig else 0.9)
    ptxt = "p<.001" if p < 0.0005 else f"p={p:.3f}"
    axC.text(1.05, i, f"{rel:+.2f}%   {ptxt}   n = {int(n):,}",
             transform=axC.get_yaxis_transform(), ha="left", va="center",
             fontsize=9, color=GREY, clip_on=False, zorder=6)
axC.axvline(0, color=GREY, lw=0.7, zorder=4)
axC.set_xticks([-4, -2, 0])
axC.tick_params(axis="x", pad=1.5)
axC.set_yticks([])
axC.spines["left"].set_visible(False)
axC.xaxis.grid(True, color=LIGHT, lw=0.5, zorder=0)
axC.set_axisbelow(True)
axC.set_xlabel("day-clustered DM", fontsize=9, labelpad=2)
for i, (fam, h, *_rest) in enumerate(rows):
    fig.text((C_LAB - 0.07) / W, yfig(axC, i),
             ("Qwen3-32B" if fam == QW else "Llama-70B") + f"   $h$ = {h}",
             ha="right", va="center", fontsize=9, color=GREY)

fig.text(0.012, (C_BOT + C_HGT + 0.20) / H,
         "(c)  The earnings-free residual: the five non-2.02 groups pooled",
         ha="left", va="center", fontsize=9.5, color=GREY, weight="bold")

# --------------------------------------------------------------- footnote
foot = [
    "Basis. Event-driven panel only, on the FULL 25,109 / 25,001 / 24,732 test "
    "rows over 996 /",
    "991 / 981 trading days --- not the 23,855 / 22,785 / 22,318 merged-grid "
    "rows on which the",
    "reference ladder is counted. Reference: firm-identity-augmented "
    "recalibrated HAR, fitted on",
    "validation, frozen on test; residuals partitioned by item code, nothing "
    "refitted per stratum.",
    "Volatility-unit QLIKE; day-clustered DM; Holm within family over the 18 "
    "group x horizon cells.",
]
#  The whole-figure basis block.  Five lines of prose in the data ink, with no
#  boundary of any kind above them, so the support counts and the "not the
#  merged-grid rows" warning arrived looking like more figure content.  A hairline
#  above plus INK2 says: below this line is the basis, not the argument.  Both are
#  free -- the ink is not geometry, and the rule and the lift are drawn into
#  whitespace already inside the tight bounding box.  Every word is kept, at its
#  own size, on its own line, at the LINE pitch the block was set with (note() is
#  deliberately not used: it would impose linespacing=1.32 on lines that are
#  positioned individually).
FOOT_TOP = 0.95 + LINE * 0.90
fig.add_artist(Line2D([0.012, 0.905], [FOOT_TOP / H, FOOT_TOP / H],
                      transform=fig.transFigure, color=RULE, lw=0.5,
                      zorder=0.5))
for k, t in enumerate(foot):
    fig.text(0.012, (0.95 - LINE * k) / H, t, ha="left", va="center",
             fontsize=8.6, color=INK2)

ds.finish(fig, "AF1_item_geography_8k",
          note=(f"qwen narrative_ALL sig {narrative_sig(QW)}/3; llama {narrative_sig(LL)}/3; Holm survivors "
                f"{holm_survivors(QW)} and {holm_survivors(LL)} of 18"))
