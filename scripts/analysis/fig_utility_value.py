#!/usr/bin/env python
"""Publication figure: annualized performance fee (bps) for the text increment (f_U vs f_R).

Bars = mean annualized performance fee (bps) an investor pays for switching from the
recalibrated price forecast f_R to the text-augmented forecast f_U, per model x horizon,
grouped by risk aversion gamma in {2,10}, for long_form and event_driven disclosures.
Positive fee = text is worth paying for. Fee averaged over mu_annual in {0.04,0.06,0.08}.

Source: results/tables/utility_value.csv
Out:    results/figures/fig_utility_value.{png,pdf}
"""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(".")
SRC = ROOT / "results/tables/utility_value.csv"
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC)

# Average fee over mu_annual (fee scales ~linearly with mu); one value per cell.
cell = (df.groupby(["disclosure", "model", "horizon", "gamma"], as_index=False)
          .agg(fee_bps_ann=("fee_bps_ann", "mean")))

MODELS = ["B2_tfidf_ridge", "C2_finbert_s1", "C2_finbert_s2",
          "C4_longformer", "C5_qwen3", "D2_gated_fusion"]
MLABEL = {"B2_tfidf_ridge": "B2 TF-IDF", "C2_finbert_s1": "C2 FinBERT-s1",
          "C2_finbert_s2": "C2 FinBERT-s2", "C4_longformer": "C4 Longformer",
          "C5_qwen3": "C5 Qwen3", "D2_gated_fusion": "D2 Gated"}
HORIZONS = [5, 10, 20]
GAMMAS = [2, 10]
DISCS = ["long_form", "event_driven"]
DLABEL = {"long_form": "Long-form (10-K/10-Q)", "event_driven": "Event-driven (8-K)"}

# Colorblind-safe (Okabe-Ito): blue for gamma=2, vermillion for gamma=10.
GCOLOR = {2: "#0072B2", 10: "#D55E00"}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 10})

n_models = len(MODELS)
n_h = len(HORIZONS)
# x positions: groups of horizons per model
group_w = 1.0
inner = np.linspace(-0.28, 0.28, n_h)  # horizon offset within a model group
bar_w = 0.22

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

hatch = {5: "", 10: "//", 20: ".."}

for ax, disc in zip(axes, DISCS, strict=False):
    d = cell[cell.disclosure == disc]
    xticks = []
    for mi, m in enumerate(MODELS):
        base = mi * 1.4
        xticks.append(base)
        for hi, h in enumerate(HORIZONS):
            for gi, g in enumerate(GAMMAS):
                row = d[(d.model == m) & (d.horizon == h) & (d.gamma == g)]
                if row.empty:
                    continue
                val = row.fee_bps_ann.values[0]
                # gamma sub-position within horizon slot
                xoff = inner[hi] + (gi - 0.5) * bar_w
                ax.bar(base + xoff, val, width=bar_w, color=GCOLOR[g],
                       hatch=hatch[h], edgecolor="white", linewidth=0.4,
                       alpha=0.95, zorder=3)
    ax.axhline(0, color="0.25", linewidth=1.0, zorder=4)
    ax.set_xticks([mi * 1.4 for mi in range(n_models)])
    ax.set_xticklabels([MLABEL[m] for m in MODELS], rotation=30, ha="right")
    ax.set_title(DLABEL[disc])
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Annualized performance fee (bps)")

# Legend: gamma via color, horizon via hatch
from matplotlib.patches import Patch

gamma_handles = [Patch(facecolor=GCOLOR[g], label=f"$\\gamma={g}$") for g in GAMMAS]
horizon_handles = [Patch(facecolor="0.75", hatch=hatch[h], edgecolor="white",
                         label=f"h={h}d") for h in HORIZONS]
leg1 = axes[1].legend(handles=gamma_handles, title="Risk aversion", loc="upper right",
                      framealpha=0.9)
axes[1].add_artist(leg1)
axes[0].legend(handles=horizon_handles, title="Horizon", loc="upper left",
               framealpha=0.9)

fig.suptitle("Economic value of the text increment: performance fee for f$_U$ (with text) over f$_R$ (recalibrated price)\n"
             "Positive = investor would pay for text; magnitudes are small (a few bps), consistent with the ~0.1-4.6% QLIKE gain",
             fontsize=12.5, y=1.02)
fig.tight_layout()

png = OUT / "fig_utility_value.png"
pdf = OUT / "fig_utility_value.pdf"
fig.savefig(png, dpi=150, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

# summary numbers
print("PNG bytes:", png.stat().st_size)
print("PDF bytes:", pdf.stat().st_size)
print("fee_bps_ann (mu-avg) describe:\n", cell.fee_bps_ann.describe())
print("median |fee| bps:", cell.fee_bps_ann.abs().median())
print("frac cells fee>0:", (cell.fee_bps_ann > 0).mean())
