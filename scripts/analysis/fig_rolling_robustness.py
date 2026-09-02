"""Publication figure: rolling-window robustness of the M1 text QLIKE increment.

Per-quarter relative QLIKE improvement % (text fU vs recalibrated fR) across the
16 test quarters 2022Q1..2025Q4, for a few representative (disc, model, h) cells.
EXPANDING scheme as the primary line (moving-block CI band, markers filled when
DM-significant); FIXED scheme overlaid as a dashed reference line.

Source: results/tables/rolling_robustness.csv
Out:    results/figures/fig_rolling_robustness.{png,pdf}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(".")
SRC = ROOT / "results/tables/rolling_robustness.csv"
OUT = ROOT / "results/figures/fig_rolling_robustness"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# Colorblind-safe (Okabe-Ito)
C_EXP = "#0072B2"   # blue  -> expanding (primary)
C_FIX = "#E69F00"   # orange-> fixed (reference)
C_BAND = "#56B4E9"  # light blue band

df = pd.read_csv(SRC)

# Representative cells: (disc, model, h, human title)
CELLS = [
    ("long_form",    "C2_finbert_s1",   10, "FinBERT-s1 · long-form · h=10"),
    ("event_driven", "C2_finbert_s1",   10, "FinBERT-s1 · event-driven · h=10"),
    ("long_form",    "B2_tfidf_ridge",  20, "TF-IDF+Ridge · long-form · h=20"),
    ("long_form",    "C4_longformer",   10, "Longformer · long-form · h=10"),
]

SIG = 0.05
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.ravel()

# consistent quarter axis
quarters = (df[df["scheme"] == "expanding"]
            .sort_values("quarter_idx")["quarter"].drop_duplicates().tolist())
x = np.arange(len(quarters))

for ax, (disc, model, h, title) in zip(axes, CELLS):
    exp = (df[(df.disc == disc) & (df.model == model) & (df.h == h)
              & (df.scheme == "expanding")].sort_values("quarter_idx"))
    fix = (df[(df.disc == disc) & (df.model == model) & (df.h == h)
              & (df.scheme == "fixed")].sort_values("quarter_idx"))

    # ci_lo/ci_hi are in raw-loss units; convert the MB-CI band to the same
    # relative % scale as rel_impr_pct by rescaling the half-widths to the point.
    ye = exp["rel_impr_pct"].to_numpy()
    lo_raw = exp["ci_lo"].to_numpy(); hi_raw = exp["ci_hi"].to_numpy()
    # half-width fraction of the raw band mapped onto the % point estimate
    mid = 0.5 * (lo_raw + hi_raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(mid != 0, ye / mid, np.nan)
    band_lo = np.where(np.isfinite(scale), lo_raw * scale, ye)
    band_hi = np.where(np.isfinite(scale), hi_raw * scale, ye)
    b_lo = np.minimum(band_lo, band_hi)
    b_hi = np.maximum(band_lo, band_hi)

    ax.axhline(0, color="0.4", lw=1.0, ls="-", zorder=1)
    ax.fill_between(x, b_lo, b_hi, color=C_BAND, alpha=0.25, lw=0,
                    zorder=2, label="Expanding 95% MB-CI")

    # fixed reference (dashed)
    ax.plot(x, fix["rel_impr_pct"].to_numpy(), color=C_FIX, lw=1.6, ls="--",
            marker=None, zorder=3, label="Fixed-origin (M1)")

    # expanding primary line
    ax.plot(x, ye, color=C_EXP, lw=1.8, zorder=4, label="Expanding")
    sig = exp["dm_p"].to_numpy() < SIG
    # filled markers where DM-significant, open otherwise
    ax.scatter(x[sig], ye[sig], s=42, facecolors=C_EXP, edgecolors=C_EXP,
               zorder=5)
    ax.scatter(x[~sig], ye[~sig], s=42, facecolors="white", edgecolors=C_EXP,
               linewidths=1.4, zorder=5)

    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=90)
    ax.set_xlim(-0.5, len(x) - 0.5)

for ax in (axes[0], axes[2]):
    ax.set_ylabel("Rel. QLIKE improvement (%)\n(text vs recalibrated price)")

# shared legend
handles = [
    Line2D([0], [0], color=C_EXP, lw=1.8, marker="o", markerfacecolor=C_EXP,
           markeredgecolor=C_EXP, label="Expanding (filled = DM sig. p<.05)"),
    Line2D([0], [0], color=C_EXP, lw=0, marker="o", markerfacecolor="white",
           markeredgecolor=C_EXP, label="Expanding (not sig.)"),
    Line2D([0], [0], color=C_FIX, lw=1.6, ls="--", label="Fixed-origin (M1)"),
    plt.Rectangle((0, 0), 1, 1, fc=C_BAND, alpha=0.25, label="Expanding 95% MB-CI"),
    Line2D([0], [0], color="0.4", lw=1.0, label="Zero (no text gain)"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Rolling-window robustness of the text QLIKE increment over 2022Q1–2025Q4\n"
    "positive = text lowers QLIKE vs a recalibrated price forecast (expanding refit vs frozen M1 combiner)",
    fontsize=13, y=1.0)

fig.tight_layout(rect=(0, 0.04, 1, 0.97))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
print("saved", OUT)

# quick numeric summary for the return payload
for disc, model, h, title in CELLS:
    exp = df[(df.disc == disc) & (df.model == model) & (df.h == h) & (df.scheme == "expanding")]
    fix = df[(df.disc == disc) & (df.model == model) & (df.h == h) & (df.scheme == "fixed")]
    print(f"{title}: exp mean={exp.rel_impr_pct.mean():.2f}% sig={(exp.dm_p<SIG).sum()}/16 "
          f"pos={(exp.rel_impr_pct>0).sum()}/16 | fix mean={fix.rel_impr_pct.mean():.2f}% "
          f"sig={(fix.dm_p<SIG).sum()}/16")
