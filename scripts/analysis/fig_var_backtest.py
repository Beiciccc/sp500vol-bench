"""Publication figure: VaR backtest of the text increment.

Grouped bars of empirical violation rate per forecast (rawHAR / recal fR / +text fU)
at alpha=5% and 1%, faceted by horizon (long_form, two representative finance models),
dashed nominal-alpha line, Kupiec significance annotated. Companion panel: tick-loss
DM (fU vs fR) per cell.

Source: results/tables/var_backtest.csv
Out:    results/figures/fig_var_backtest.{png,pdf}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = "."
SRC = os.path.join(ROOT, "results/tables/var_backtest.csv")
OUTP = os.path.join(ROOT, "results/figures/fig_var_backtest")

# ---- config -------------------------------------------------------------
DISC = "long_form"
MU = "pooled"                       # pooled-mean drift convention
MODELS = ["B2_tfidf_ridge", "C2_finbert_s1"]  # two best finance-domain reps
MODEL_LABEL = {"B2_tfidf_ridge": "B2 TF-IDF+Ridge", "C2_finbert_s1": "C2 FinBERT"}
HORIZONS = [5, 10, 20]
ALPHAS = [0.05, 0.01]
FORECASTS = ["rawHAR", "fR", "fU"]
FC_LABEL = {"rawHAR": "raw HAR", "fR": "recal $f_R$", "fU": "+text $f_U$"}
# colorblind-safe (Okabe-Ito)
FC_COLOR = {"rawHAR": "#E69F00", "fR": "#0072B2", "fU": "#009E73"}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.titlesize": 14,
})

df = pd.read_csv(SRC)
df = df[(df.disclosure == DISC) & (df.mu_mode == MU)].copy()
assert not df.empty, "no rows after filter"


def cell(model, h, alpha, fc):
    r = df[(df.model == model) & (df.horizon == h) & (df.alpha == alpha) & (df.forecast == fc)]
    if r.empty:
        return None
    return r.iloc[0]


# figure: rows = alpha (5%, 1%), cols = model  -> violation-rate bars;
# plus a bottom-spanning companion panel for tick-loss DM.
fig = plt.figure(figsize=(11.5, 9.0))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85], hspace=0.42, wspace=0.22)

bar_axes = {}
for ai, alpha in enumerate(ALPHAS):
    for mi, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[ai, mi])
        bar_axes[(alpha, model)] = ax
        x = np.arange(len(HORIZONS))
        w = 0.26
        for fi, fc in enumerate(FORECASTS):
            vals, sig = [], []
            for h in HORIZONS:
                c = cell(model, h, alpha, fc)
                vals.append(np.nan if c is None else c.viol_rate)
                sig.append(False if c is None else (c.kupiec_p < 0.05))
            xpos = x + (fi - 1) * w
            bars = ax.bar(xpos, vals, w, color=FC_COLOR[fc], label=FC_LABEL[fc],
                          edgecolor="black", linewidth=0.5, zorder=3)
            # Kupiec significance star above each bar (rejects nominal coverage)
            for xp, v, s in zip(xpos, vals, sig):
                if s and not np.isnan(v):
                    ax.text(xp, v + 0.0015, "*", ha="center", va="bottom",
                            fontsize=12, color="black", zorder=4)
        # nominal alpha line
        ax.axhline(alpha, ls="--", color="#555555", lw=1.4, zorder=2)
        ax.text(len(HORIZONS) - 0.55, alpha, f"  nominal {alpha:.0%}",
                va="bottom", ha="right", fontsize=9, color="#555555")
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in HORIZONS])
        ax.set_ylabel("empirical violation rate")
        ax.set_title(f"{MODEL_LABEL[model]}  —  VaR at {alpha:.0%}")
        ax.set_ylim(0, max(0.09, alpha * 1.9))
        ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)

# shared legend (bar panels)
handles = [Patch(facecolor=FC_COLOR[fc], edgecolor="black", label=FC_LABEL[fc]) for fc in FORECASTS]
handles.append(plt.Line2D([0], [0], ls="--", color="#555555", label="nominal $\\alpha$"))
handles.append(plt.Line2D([0], [0], marker="*", ls="none", color="black",
                           markersize=10, label="Kupiec rejects (p<.05)"))
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 0.975))

# ---- companion panel: tick-loss DM (fU vs fR) ---------------------------
axd = fig.add_subplot(gs[2, :])
# build cells: model x alpha x horizon; negative stat = text lowers tick loss (better)
labels, stats, colors = [], [], []
for model in MODELS:
    for alpha in ALPHAS:
        for h in HORIZONS:
            c = cell(model, h, alpha, "fU")
            if c is None:
                continue
            st = c.dm_tick_vs_fR_stat
            p = c.dm_tick_vs_fR_p
            labels.append(f"{MODEL_LABEL[model].split()[0]}\n{alpha:.0%} h{h}")
            stats.append(st)
            if st < 0 and p < 0.05:
                colors.append("#009E73")   # text sig better
            elif st > 0 and p < 0.05:
                colors.append("#D55E00")   # text sig worse
            else:
                colors.append("#BBBBBB")   # ns
xd = np.arange(len(labels))
axd.bar(xd, stats, color=colors, edgecolor="black", linewidth=0.4, zorder=3)
axd.axhline(0, color="black", lw=0.9, zorder=2)
axd.set_xticks(xd)
axd.set_xticklabels(labels, fontsize=8)
axd.set_ylabel("DM stat: $f_U$ vs $f_R$\n(<0 = text lowers tick loss)")
axd.set_title("Tick-loss Diebold-Mariano: does +text beat recalibrated HAR? (long_form, pooled $\\mu$)")
axd.grid(axis="y", ls=":", alpha=0.5, zorder=0)
dh = [Patch(facecolor="#009E73", edgecolor="black", label="text better (p<.05)"),
      Patch(facecolor="#D55E00", edgecolor="black", label="text worse (p<.05)"),
      Patch(facecolor="#BBBBBB", edgecolor="black", label="n.s.")]
axd.legend(handles=dh, loc="upper right", frameon=True, framealpha=0.9, ncol=3)

fig.suptitle("VaR backtest of the text increment: raw HAR over-forecasts violations, "
             "recalibration fixes most, +text trims tick loss\n"
             "(long_form, pooled-$\\mu$ drift; lower bars = fewer VaR breaches, closer to nominal is better)",
             y=1.02)

fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUTP + ".png", dpi=150, bbox_inches="tight")
fig.savefig(OUTP + ".pdf", bbox_inches="tight")
plt.close(fig)

# verify non-empty
for ext in ("png", "pdf"):
    p = OUTP + "." + ext
    sz = os.path.getsize(p)
    print(f"{p}  {sz} bytes")
    assert sz > 1000, f"empty file {p}"

# print a few numbers for the record
print("\n-- key numbers --")
for model in MODELS:
    for alpha in ALPHAS:
        row = []
        for fc in FORECASTS:
            c5 = cell(model, 5, alpha, fc)
            row.append(f"{fc}={c5.viol_rate:.4f}(kp={c5.kupiec_p:.1e})")
        print(model, f"a={alpha} h5:", " ".join(row))
