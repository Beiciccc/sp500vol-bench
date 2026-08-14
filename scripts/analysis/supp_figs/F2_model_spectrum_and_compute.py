"""F2 -- The model spectrum: standalone skill across representations, and what the
fine-tuning budget bought.

Panel (a) substantiates the frozen main text (06_results.tex, first paragraph):
    "text-alone test $R^2$ is negative on long-form at every horizon"
and the protocol's statement (05_protocol.tex) that the primary basis is a
seed-ensemble rather than any single trained run.

Panel (b) prices the same spectrum against recorded compute.

Committed evidence read at run time (no number is hardcoded outside gate()):
  results/tables/seed_aggregate.csv
      disclosure=long_form; columns model, horizon, n_seeds, r2_mean, r2_std.
      Supplies the 15 long-form text-only challengers and the four price arms
      A1_hv, A3_garch, A4_egarch, A5_arima (all n_seeds = 1).
  results/tables/compare_full_long_form_test.md
      Supplies the A2_har_rv reference row (R-squared 0.0890/0.1210/0.2080) and
      that table's A2 QLIKE row. A2 is absent from seed_aggregate.csv; HAR-RV is
      a deterministic fit.
  results/tables/cost_accuracy.csv
      model, block, gpu_hours_total, best_qlike, on_pareto_frontier.
  results/tables/cost_accuracy.md
      Block C / Block D / total GPU-hour line and the CPU wall-clock seconds.

Adversarial repairs folded in
  * "the 8 seed-invariant A/B arms" -> only the 4 B arms are single-seed inside
    this forest; the price arms are drawn in a separate reference block.
  * "the five 0-GPU-hour CPU arms" -> cost_accuracy.csv holds NINE rows at
    gpu_hours_total = 0 (five price, four bag-of-words).
  * "the price baseline stays positive" -> all five price arms that have a
    committed long-form R^2 row are drawn. HAR-RV is the only one positive at
    every horizon. These five are NOT the maximal pool's five members: the pool
    of 05_protocol.tex is HAR, SHAR, GARCH, EGARCH, ARIMA, whereas A6_shar has
    no row in seed_aggregate.csv or compare_full_long_form_test.md and A1_hv is
    not a pool member. The motivating fact is taken verbatim in scope from
    06_results.tex ("HAR-RV is not QLIKE-strongest: SHAR is clustered-
    significantly better ... which motivates the maximal pool"), not from the
    R^2/QLIKE ranking disagreement, which is a separate observation.
  * panel (b)'s zero-GPU band merged nine arms into four blobs and the Pareto
    ring then sat level with the wrong name. x carries no information inside
    that band, so the nine rows are dodged into three columns by rank; the tick
    label reads "0 GPU-h" and the footnote says the spread is not a cost.
  * CONFLICT (adversary 1 vs adversary 3) on the unit of cost_accuracy's QLIKE:
    adversary 3 notes it matches m1_variance_unit's variance-unit figure;
    adversary 1 notes cost_accuracy.md never declares a unit. The MORE
    CONSERVATIVE reading is taken: the axis is labelled as recorded in
    cost_accuracy.csv with the unit explicitly not declared in the source.
  * CONFLICT (adversary 1: "either drop A2 from panel (b) or annotate the
    disagreement"). The MORE CONSERVATIVE choice is to keep A2 and annotate --
    dropping a row would crop committed evidence -- so both A2 vintages are
    printed on the artefact and no cross-panel A2 comparison is drawn.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
from supp_style import (  # noqa: F401
    AGG,
    BLUE,
    GREEN,
    GREY,
    LIGHT,
    PURPLE,
    REPO,
    SKY,
    TAB,
    VERM,
    VERM_TXT,
    YELLOW,
    apply_style,
    finish,
    gate,
)

# ------------------------------------------------------------------ evidence
seed_agg = pd.read_csv(os.path.join(TAB, "seed_aggregate.csv"))
lf = seed_agg[seed_agg.disclosure == "long_form"].copy()

TEXT_ARMS = ["B1_bow_ridge", "B2_tfidf_ridge", "B3_lm_linear", "B4_lm_features",
             "C1_bert_s1", "C1_bert_s2", "C2_finbert_s1", "C2_finbert_s2",
             "C2_finbert_s3", "C2_finbert_s4", "C3_roberta_s1", "C4_longformer",
             "C5_e5mistral", "C5_gteqwen2", "C5_qwen3"]
PRICE_FROM_AGG = ["A1_hv", "A3_garch", "A4_egarch", "A5_arima"]
HORIZONS = [5, 10, 20]

text = lf[lf.model.isin(TEXT_ARMS)].copy()
price = lf[lf.model.isin(PRICE_FROM_AGG)].copy()

# A2_har_rv is not in seed_aggregate.csv; read it out of the compare table.
cmp_path = os.path.join(TAB, "compare_full_long_form_test.md")
a2_fields = None
for line in open(cmp_path):
    if line.startswith("A2_har_rv"):
        a2_fields = line.split()
        break
if a2_fields is None or len(a2_fields) != 13:
    sys.exit("could not parse the A2_har_rv row of compare_full_long_form_test.md")
# column order in that table: mae 5/10/20, qlike 5/10/20, r2 5/10/20, rmse 5/10/20
A2_R2 = [float(a2_fields[7]), float(a2_fields[8]), float(a2_fields[9])]
A2_QLIKE_CMP = [float(a2_fields[4]), float(a2_fields[5]), float(a2_fields[6])]

cost = pd.read_csv(os.path.join(TAB, "cost_accuracy.csv"))
cost_md = open(os.path.join(TAB, "cost_accuracy.md")).read()
tot = re.search(r"all C runs = (\d+\.\d+); all D runs = (\d+\.\d+); C\+D = (\d+\.\d+)",
                cost_md)
if tot is None:
    sys.exit("could not parse the GPU-hour totals line of cost_accuracy.md")
GPU_C, GPU_D, GPU_TOT = (float(tot.group(1)), float(tot.group(2)), float(tot.group(3)))

cpu_sec = {}
in_cpu = False
for line in cost_md.splitlines():
    if line.startswith("### CPU baselines"):
        in_cpu = True
        continue
    if in_cpu and line.startswith("|"):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 3 and re.match(r"^[AB]\d", cells[0]):
            cpu_sec[cells[0]] = float(cells[2])

pareto = cost[cost.on_pareto_frontier == "yes"]
c4 = cost[cost.model == "C4_longformer"].iloc[0]
a2_cost = cost[cost.model == "A2_har_rv"].iloc[0]

# --------------------------------------------------------------------- gates
gate(
    {"n_text_arms": 15,
     "n_text_cells": 45,
     "n_text_cells_negative_r2": 45,
     "text_best_r2_mean": -0.0149,
     "n_single_seed_text_arms": 4,
     "A2_r2": (0.0890, 0.1210, 0.2080),
     "n_cost_rows": 25,
     "n_zero_gpu_rows": 9,
     "n_pareto": 1,
     "pareto_model": "A3_garch",
     "pareto_gpu_h": 0.0,
     "pareto_qlike": 0.2686,
     "c4_gpu_h": 254.715,
     "c4_qlike": 0.6051,
     "gpu_C_D_total": (566.7, 23.6, 590.4),
     "cpu_seconds_A2_A3_A5_B2": (3.9, 212.1, 2862.6, 4474.1)},
    {"n_text_arms": int(text.model.nunique()),
     "n_text_cells": len(text),
     "n_text_cells_negative_r2": int((text.r2_mean < 0).sum()),
     "text_best_r2_mean": round(float(text.r2_mean.max()), 4),
     "n_single_seed_text_arms": int((text.groupby("model").n_seeds.max() == 1).sum()),
     "A2_r2": tuple(round(v, 4) for v in A2_R2),
     "n_cost_rows": len(cost),
     "n_zero_gpu_rows": int((cost.gpu_hours_total == 0).sum()),
     "n_pareto": len(pareto),
     "pareto_model": str(pareto.iloc[0].model),
     "pareto_gpu_h": float(pareto.iloc[0].gpu_hours_total),
     "pareto_qlike": float(pareto.iloc[0].best_qlike),
     "c4_gpu_h": float(c4.gpu_hours_total),
     "c4_qlike": float(c4.best_qlike),
     "gpu_C_D_total": (GPU_C, GPU_D, GPU_TOT),
     "cpu_seconds_A2_A3_A5_B2": (cpu_sec["A2_har_rv"], cpu_sec["A3_garch"],
                                 cpu_sec["A5_arima"], cpu_sec["B2_tfidf_ridge"])},
)


# --------------------------------------------------------------- label helper
def spread_labels(ax, pts, side="r", pad_px=8.0, bias_px=0.0, fontsize=9,
                  colour=GREY):
    """Place arm names beside their markers, de-collided in display space.

    pts: iterable of (x_data, y_data, text). Labels keep their vertical order,
    are pushed apart to at least one line-height, re-centred on the group and
    then shifted by bias_px; a hairline leader joins each label to its marker.
    """
    fig = ax.figure
    fig.canvas.draw()
    tr = ax.transData
    inv = tr.inverted()
    disp = sorted(((tr.transform((x, y)), t) for x, y, t in pts),
                  key=lambda d: d[0][1])
    gap = fontsize * 1.45 * fig.dpi / 72.0
    ys, cur = [], -1e18
    for (px, py), _ in disp:
        cur = max(py, cur + gap)
        ys.append(cur)
    shift = (sum(d[0][1] for d in disp) - sum(ys)) / len(ys) + bias_px
    ys = [y + shift for y in ys]
    for ((px, py), t), ly in zip(disp, ys, strict=False):
        lx = px + pad_px if side == "r" else px - pad_px
        ax.annotate(t, xy=inv.transform((px, py)), xytext=inv.transform((lx, ly)),
                    fontsize=fontsize, color=colour, va="center",
                    ha="left" if side == "r" else "right",
                    arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.6,
                                    shrinkA=1.5, shrinkB=3.0))


def spread_labels_gutter(ax, pts, x_anchor, fontsize=9, colour=GREY):
    """Name every point in a y-ordered column at x_anchor, with hairline leaders.

    Used where the markers cluster too tightly for beside-the-point labels; the
    column preserves the vertical order of the markers, so a leader identifies
    its own point unambiguously.
    """
    fig = ax.figure
    fig.canvas.draw()
    tr = ax.transData
    inv = tr.inverted()
    disp = sorted(((tr.transform((x, y)), t) for x, y, t in pts),
                  key=lambda d: d[0][1])
    gap = fontsize * 1.42 * fig.dpi / 72.0
    ys, cur = [], -1e18
    for (px, py), _ in disp:
        cur = max(py, cur + gap)
        ys.append(cur)
    shift = (sum(d[0][1] for d in disp) - sum(ys)) / len(ys)
    ys = [y + shift for y in ys]
    lo_lim, hi_lim = ax.transData.transform((x_anchor, ax.get_ylim()[0]))[1], \
        ax.transData.transform((x_anchor, ax.get_ylim()[1]))[1]
    over = max(0.0, max(ys) - (hi_lim - gap * 0.4))
    ys = [y - over for y in ys]
    under = max(0.0, (lo_lim + gap * 0.4) - min(ys))
    ys = [y + under for y in ys]
    for ((px, py), t), ly in zip(disp, ys, strict=False):
        ax.annotate(t, xy=inv.transform((px, py)),
                    xytext=(x_anchor, inv.transform((px, ly))[1]),
                    fontsize=fontsize, color=colour, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.6,
                                    shrinkA=1.5, shrinkB=3.0))


# ------------------------------------------------------------------ plotting
apply_style()
fig = plt.figure(figsize=(6.4, 9.0))
gs_a = fig.add_gridspec(1, 1, left=0.205, right=0.988, top=0.905, bottom=0.590)
gs_b = fig.add_gridspec(1, 2, left=0.205, right=0.988, top=0.437, bottom=0.186,
                        width_ratios=[1.45, 3.28], wspace=0.04)
ax_a = fig.add_subplot(gs_a[0])

H_STYLE = {5: (BLUE, "o", "$h$ = 5"),
           10: (YELLOW, "^", "$h$ = 10"),
           20: (PURPLE, "s", "$h$ = 20")}
OFFSET = {5: -0.22, 10: 0.0, 20: 0.22}

order = (text.groupby("model").r2_mean.mean().sort_values(ascending=False)
         .index.tolist())
ypos = {m: i for i, m in enumerate(order)}
XLO, XHI = -0.53, 0.27


def draw_rows(models, ypos_map):
    for m in models:
        sub = text if m in TEXT_ARMS else price
        rows = sub[sub.model == m]
        for h in HORIZONS:
            r = rows[rows.horizon == h]
            if r.empty:
                continue
            r = r.iloc[0]
            col, mk, _ = H_STYLE[h]
            y = ypos_map[m] + OFFSET[h]
            if int(r.n_seeds) > 1:
                ax_a.errorbar(r.r2_mean, y, xerr=r.r2_std, fmt=mk, ms=3.9,
                              color=col, ecolor=col, elinewidth=0.8, capsize=1.8,
                              mec=col, mew=0.8, zorder=3)
            elif r.r2_mean < XLO:   # off-scale: exact value printed alongside
                ax_a.plot(XLO + 0.007, y, marker="<", ms=4.6, color=col,
                          mfc="white", mew=1.0, zorder=3)
            else:
                ax_a.plot(r.r2_mean, y, marker=mk, ms=4.1, color=col, mfc="white",
                          mew=1.0, zorder=3)


draw_rows(order, ypos)

# --- reference block: all five single price references, not A2 alone
ref_order = ["A2_har_rv", "A5_arima", "A3_garch", "A4_egarch", "A1_hv"]
GAP = 1.5
ref_y = {m: len(order) - 1 + GAP + i for i, m in enumerate(ref_order)}
for h, r2 in zip(HORIZONS, A2_R2, strict=False):
    col, mk, _ = H_STYLE[h]
    ax_a.plot(r2, ref_y["A2_har_rv"] + OFFSET[h], marker=mk, ms=4.1, color=col,
              mfc="white", mew=1.0, zorder=3)
draw_rows(["A5_arima", "A3_garch", "A4_egarch", "A1_hv"], ref_y)

ax_a.axvline(0.0, color=GREY, lw=1.0, zorder=2)
ax_a.axhline(len(order) - 1 + GAP / 2, color=LIGHT, lw=0.8, zorder=1)

ax_a.set_yticks([ypos[m] for m in order] + [ref_y[m] for m in ref_order])
ax_a.set_yticklabels(order + ref_order)
ax_a.set_ylim(len(order) - 1 + GAP + len(ref_order) + 0.35, -0.65)
ax_a.set_xlim(XLO, XHI)
ax_a.set_xlabel("long-form test $R^2$   (0 = predicting the sample mean)")
ax_a.tick_params(axis="y", length=0)

a1 = price[price.model == "A1_hv"].set_index("horizon").r2_mean
ax_a.text(XLO + 0.020, ref_y["A1_hv"] + 0.85,
          f"A1_hv $h$=5 ({a1[5]:.3f}), $h$=10 ({a1[10]:.3f}): left of the axis",
          fontsize=9, color=VERM_TXT, va="center", ha="left")
ax_a.text(XHI - 0.005, ref_y["A2_har_rv"] - 0.95, "five single price arms (A1-A5)",
          fontsize=9, color=GREY, ha="right", va="center", style="italic")

leg_h = [plt.Line2D([], [], color=H_STYLE[h][0], marker=H_STYLE[h][1], ls="none",
                    ms=4.1, label=H_STYLE[h][2]) for h in HORIZONS]
leg_h += [plt.Line2D([], [], color=GREY, marker="o", ls="none", ms=4.1,
                     label="3 seeds: mean $\\pm$ 1 s.d."),
          plt.Line2D([], [], color=GREY, marker="o", ls="none", ms=4.1, mfc="white",
                     mew=1.0, label="single deterministic fit")]
ax_a.legend(handles=leg_h, loc="upper center", bbox_to_anchor=(0.44, -0.115),
            ncol=5, fontsize=9, handletextpad=0.35, columnspacing=1.1,
            borderpad=0.2)

fig.text(0.012, 0.992,
         "(a)  15 long-form text-only challengers: all 45 (arm $\\times$ horizon) "
         "seed-mean cells are negative. Open markers\n"
         "carry no seed dispersion (the 4 single-seed B arms; the five price "
         "arms). HAR-RV is the only one of the five\n"
         "positive at every horizon, and it is not QLIKE-strongest (SHAR is "
         "clustered-significantly better), which is\n"
         "what motivates the maximal pool.",
         fontsize=9, color=GREY, ha="left", va="top", linespacing=1.4)

fig.text(0.012, 0.528,
         "Below the rule: the five single price arms with a committed long-form "
         "$R^2$ row (A1-A5) -- not the maximal\n"
         "pool's five: SHAR has no row in seed_aggregate.csv or "
         "compare_full_long_form_test.md, and A1_hv is not one.",
         fontsize=9, color=GREY, ha="left", va="top", linespacing=1.4)

# ------------------------------------------------------------------- panel b
ax_b0 = fig.add_subplot(gs_b[0])
ax_b1 = fig.add_subplot(gs_b[1], sharey=ax_b0)

BLOCK = {"A": (GREY, "o"), "B": (YELLOW, "s"), "C": (VERM, "^"), "D": (BLUE, "D")}
zero = (cost[cost.gpu_hours_total == 0].sort_values("best_qlike")
        .reset_index(drop=True))
nz = cost[cost.gpu_hours_total > 0].sort_values("gpu_hours_total")

# Inside the zero-GPU band the horizontal coordinate carries no information --
# every one of these nine arms is at 0 GPU-hours -- and seven of them lie within
# one marker width (4.8 pt) of a neighbour on the log QLIKE axis, so drawing
# them on one vertical
# merges nine arms into four blobs and leaves the Pareto ring ambiguous. The
# rows are therefore dodged into three columns by QLIKE rank, which guarantees
# that any two vertically adjacent arms sit in different columns. The band's
# tick label reads "0 GPU-h" and the footnote states that the spread is a
# de-collision device rather than a cost.
DODGE = (0.0, 0.30, 0.60)
zero["x_dodge"] = [DODGE[i % len(DODGE)] for i in range(len(zero))]

for _, r in zero.iterrows():
    col, mk = BLOCK[r.block]
    ax_b0.plot(r.x_dodge, r.best_qlike, marker=mk, ms=4.8, color=col, mfc=col,
               mec="white", mew=0.6, zorder=3)
for _, r in nz.iterrows():
    col, mk = BLOCK[r.block]
    ax_b1.plot(r.gpu_hours_total, r.best_qlike, marker=mk, ms=4.8, color=col,
               mfc=col, mec="white", mew=0.6, zorder=3)

ax_b1.set_xscale("log")
ax_b1.set_xlim(0.11, 26000)
ax_b1.set_xticks([0.2, 1, 10, 100])
ax_b1.set_xticklabels(["0.2", "1", "10", "100"])
GUT = 420.0                     # arm names live in a gutter right of the data
ax_b1.axvline(GUT, color=LIGHT, lw=0.7, ls=(0, (2, 2)), zorder=1)
ax_b0.set_xlim(-0.35, 2.55)
ax_b0.set_xticks([DODGE[1]])
ax_b0.set_xticklabels(["0 GPU-h"])
ax_b0.set_yscale("log")
ax_b0.set_ylim(0.222, 1.48)
ax_b0.set_yticks([0.25, 0.35, 0.5, 0.7, 1.0])
ax_b0.set_yticklabels(["0.25", "0.35", "0.50", "0.70", "1.00"])
ax_b0.minorticks_off()
ax_b0.set_ylabel("best long-form test QLIKE, log scale\n"
                 "(as recorded in cost_accuracy.csv;\nunit not declared in the "
                 "source)\nlower is better", fontsize=9)
ax_b1.tick_params(axis="y", length=0, labelleft=False)
ax_b1.spines["left"].set_visible(False)
for ax in (ax_b0, ax_b1):
    ax.grid(axis="y", color=LIGHT, lw=0.5, zorder=0)

spread_labels_gutter(ax_b0, [(r.x_dodge, r.best_qlike, r.model)
                             for _, r in zero.iterrows()], x_anchor=0.88)
# the eight arms above QLIKE 0.5 crowd together, so they are named in the gutter,
# y-ordered, and the eight below are named beside their markers in three groups
spread_labels_gutter(ax_b1, [(r.gpu_hours_total, r.best_qlike, r.model)
                             for _, r in nz[nz.best_qlike >= 0.5].iterrows()],
                     x_anchor=GUT * 1.12)
for grp, bias in ((nz[nz.gpu_hours_total < 1], 0.0),
                  (nz[(nz.gpu_hours_total >= 4) & (nz.gpu_hours_total < 7)], -21.0),
                  (nz[(nz.gpu_hours_total >= 7) & (nz.best_qlike < 0.5)], 13.0)):
    spread_labels(ax_b1, [(r.gpu_hours_total, r.best_qlike, r.model)
                          for _, r in grp.iterrows()], side="r", pad_px=7,
                  bias_px=bias)

p = pareto.iloc[0]
p_x = float(zero.loc[zero.model == p.model, "x_dodge"].iloc[0])
ax_b0.plot(p_x, p.best_qlike, marker="o", ms=10.5, mfc="none", mec=GREEN, mew=1.2,
           zorder=5)
ax_b1.plot(c4.gpu_hours_total, c4.best_qlike, marker="o", ms=10.5, mfc="none",
           mec=VERM_TXT, mew=1.2, zorder=4)
ax_b1.text(0.128, 1.45,
           "ringed green: sole Pareto point\n"
           f"{p.model}, {p.gpu_hours_total:.0f} GPU-h, {p.best_qlike:.4f}",
           fontsize=9, color=GREEN, ha="left", va="top", linespacing=1.35)
ax_b1.text(0.128, 1.10,
           "ringed orange: largest single-arm budget\n"
           f"{c4.model}, {c4.gpu_hours_total:.1f} GPU-h, {c4.best_qlike:.4f}",
           fontsize=9, color=VERM_TXT, ha="left", va="top", linespacing=1.35)

leg_b = [plt.Line2D([], [], color=BLOCK[b][0], marker=BLOCK[b][1], ls="none", ms=4.8,
                    mec="white", mew=0.6, label=f"block {b}") for b in "ABCD"]
fig.legend(handles=leg_b, loc="lower center", ncol=4, fontsize=9,
           columnspacing=1.6, handletextpad=0.4, bbox_to_anchor=(0.60, 0.113))

fig.text(0.012, 0.484,
         "(b)  cost against accuracy: the accuracy frontier is held at zero "
         "GPU-hours.\n"
         f"Block C {GPU_C:.1f} GPU-h $+$ Block D {GPU_D:.1f} $=$ {GPU_TOT:.1f} in "
         "total; prompted-LLM inference is not in this table.",
         fontsize=9, color=GREY, ha="left", va="top", linespacing=1.4)

fig.text(0.205, 0.157,
         "total GPU-hours summed over that arm's runs (log scale right of the break)",
         fontsize=9, color=GREY, ha="left", va="top")
fig.text(0.012, 0.103,
         "0 GPU-h band: nine arms -- five price (A1-A5) and four bag-of-words "
         "(B1-B4); they are spread over three\n"
         "columns only to stop the markers merging, and all nine are at 0 "
         "GPU-hours. CPU wall-clock seconds\n"
         f"over 3 disclosures: A2 {cpu_sec['A2_har_rv']:.1f}, "
         f"A3 {cpu_sec['A3_garch']:.1f}, A5 {cpu_sec['A5_arima']:,.1f}, "
         f"B2 {cpu_sec['B2_tfidf_ridge']:,.1f}. Panels (a) and (b) carry different\n"
         "quantities from different committed tables and share no axis: "
         "cost_accuracy.csv records A2_har_rv\n"
         f"QLIKE {a2_cost.qlike_long_form_h5:.4f}/"
         f"{a2_cost.qlike_long_form_h10:.4f}/{a2_cost.qlike_long_form_h20:.4f} "
         "while compare_full_long_form_test.md records "
         f"{A2_QLIKE_CMP[0]:.4f}/{A2_QLIKE_CMP[1]:.4f}/{A2_QLIKE_CMP[2]:.4f}\n"
         "for the same arm, so no cross-panel A2 comparison is made.",
         fontsize=9, color=GREY, ha="left", va="top", linespacing=1.4)

finish(fig, "F2_model_spectrum_and_compute", tight=False)
