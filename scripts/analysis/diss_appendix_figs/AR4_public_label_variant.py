"""AR4 — could a reader without the licensed price data reproduce the verdicts?

Appendix figure (outside the 60-page cap).  The labels, the recalibrated reference
and every price baseline are built from licensed daily returns.  This figure asks
what a licence-free rebuild would and would not recover: it re-derives the labels
from a public adjusted-close source on the benchmark's own label windows and
re-runs the verdict objects on three panels.

Panels of the underlying study
    A  full test panel, licensed labels          (the reported basis)
    B  public-coverage intersection, licensed labels  (isolates survivorship)
    C  the same intersection, public labels           (adds label measurement error)

Sources
-------
results/tables/label_parity.csv             coverage, label agreement, verdicts
results/tables/public_variant_leaderboard.csv  180 standalone comparisons x 2 panels
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import diss_style as ds  # noqa: E402
from supp_style import (BLUE, GREEN, GREY, LIGHT, PURPLE, TAB, VERM,  # noqa: E402
                        VERM_TXT, YELLOW, apply_style, gate)


def main():
    d = pd.read_csv(os.path.join(TAB, "label_parity.csv"), low_memory=False)
    lb = pd.read_csv(os.path.join(TAB, "public_variant_leaderboard.csv"))

    cov = d[d.section == "coverage"].iloc[0]
    ex = d[d.section == "coverage_by_exit_year"].copy()
    tick = d[d.section == "ticker_status"].iloc[0]
    pc = d[d.section == "parity_corr"]
    split_h = pc[pc.scope == "by_split_h"]
    overall = pc[pc.scope == "modelled_all_splits"].iloc[0]
    raw_all = pc[pc.scope == "modelled_all_splits_RAW_incl_mismatch"].iloc[0]
    ag = d[d.section == "verdict_agreement"]
    combo = d[d.section == "verdict_combo"]

    active = ex[ex.exit_year == "active"].iloc[0]
    exited = ex[ex.exit_year != "active"]

    got = {
        "rows": int(cov.n_rows),
        "coverage_clean_pct": round(100 * cov.coverage_clean, 2),
        "active_cov_pct": round(100 * active.coverage_clean, 1),
        "exit_cov_pct": round(100 * exited.n_rows.mul(exited.coverage_clean).sum()
                              / exited.n_rows.sum(), 1),
        "tickers": int(tick.n_tickers),
        "no_public": int(tick.n_no_data),
        "mismatch": int(tick.n_mismatch),
        "pearson_clean": round(float(overall.pearson_logRV), 4),
        "pearson_raw": round(float(raw_all.pearson_logRV), 4),
        "stand_full_B": int(ag[(ag.family == "F-STAND")
                               & (ag.panel_vs_A == "B")].full_agree.iloc[0]),
        "stand_full_C": int(ag[(ag.family == "F-STAND")
                               & (ag.panel_vs_A == "C")].full_agree.iloc[0]),
        "combo_sign_C": int(ag[(ag.family == "F-COMBO")
                               & (ag.panel_vs_A == "C")].sign_agree.iloc[0]),
        "combo_holm_C": int(ag[(ag.family == "F-COMBO")
                               & (ag.panel_vs_A == "C")].holm_sig_agree.iloc[0]),
        "rank_C": int(ag[(ag.family == "RANKING")
                         & (ag.panel_vs_A == "C")].full_agree.iloc[0]),
        "genuine_A": int(combo[combo.panel == "A"].genuine.sum()),
        "genuine_C": int(combo[combo.panel == "C"].genuine.sum()),
        "leaderboard_flips": int(lb.verdict_flip_vs_A.sum()),
        "leaderboard_cells": int(len(lb[lb.panel == "B"])),
    }
    gate({"rows": 431245, "coverage_clean_pct": 80.19, "active_cov_pct": 97.7,
          "exit_cov_pct": 31.9, "tickers": 848, "no_public": 200,
          "mismatch": 18, "pearson_clean": 0.9981, "pearson_raw": 0.9714,
          "stand_full_B": 18, "stand_full_C": 18, "combo_sign_C": 11,
          "combo_holm_C": 12, "rank_C": 5, "genuine_A": 8, "genuine_C": 7,
          "leaderboard_flips": 0, "leaderboard_cells": 180}, got)

    apply_style(9)
    H = 7.55
    fig = plt.figure(figsize=ds.canvas(H, max_h=7.7))

    def fy(t):
        return 1.0 - t / H

    fig.text(0.018, fy(0.05),
             "Rebuilding the price layer from a public source: what a "
             "licence-free variant of the benchmark",
             fontsize=9, color=GREY, va="top")
    fig.text(0.018, fy(0.26),
             "would recover. Labels recomputed from public adjusted closes on "
             "the benchmark's own windows.",
             fontsize=9, color=GREY, va="top")

    # ------------------------------------------------- (a) coverage by cohort
    ax = fig.add_axes([0.075, fy(2.62), 0.905, 1.70 / H])
    labs = [str(int(y)) for y in exited.exit_year] + ["still\nlisted"]
    vals = list(100 * exited.coverage_clean) + [100 * active.coverage_clean]
    rows = list(exited.n_rows) + [active.n_rows]
    xs = np.arange(len(vals))
    colours = [VERM] * len(exited) + [GREEN]
    ax.bar(xs, vals, color=colours, width=0.72, zorder=3)
    ax.axhline(got["coverage_clean_pct"], color=GREY, lw=0.8, ls=(0, (4, 2)),
               zorder=4)
    ax.text(0.0, got["coverage_clean_pct"] + 2.5,
            f"all benchmark rows: {got['coverage_clean_pct']:.1f}%", ha="left",
            va="bottom", fontsize=9, color=GREY)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.6, f"{v:.0f}", ha="center", va="bottom", fontsize=9,
                color=GREY)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a}\n{int(n) / 1000:.0f}k" for a, n in zip(labs, rows)],
                       fontsize=9, linespacing=1.25)
    ax.set_xlim(-0.7, len(vals) - 0.3)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("rows with a clean\npublic label (%)", fontsize=9,
                  linespacing=1.2)
    ax.tick_params(length=2.5, pad=2)
    ax.spines["bottom"].set_visible(True)
    fig.text(0.018, fy(0.62),
             "(a) coverage of the public source by the year a firm left the "
             "index (rows in thousands beneath)",
             fontsize=9, color=GREY, va="top")

    # ---------------------------------------------------- (b) where rows go
    bx = fig.add_axes([0.075, fy(3.72), 0.905, 0.32 / H])
    parts = [("covered", cov.rows_covered, GREEN),
             ("no public data", cov.rows_no_public_data, VERM),
             ("window incomplete", cov.rows_window_incomplete, YELLOW),
             ("symbol reused", cov.rows_symbol_mismatch, PURPLE)]
    left = 0.0
    for lab, n, colour in parts:
        share = 100 * n / cov.n_rows
        bx.barh([0], [share], left=[left], color=colour, height=0.62, zorder=3)
        if share > 10:
            bx.text(left + share / 2, 0, f"{share:.1f}%", ha="center",
                    va="center", fontsize=9, color="white", zorder=4)
        left += share
    bx.set_xlim(0, 100)
    bx.set_ylim(-0.55, 0.55)
    bx.axis("off")
    key = fig.add_axes([0.075, fy(4.06), 0.905, 0.20 / H])
    key.set_xlim(0, 100)
    key.set_ylim(-0.5, 0.5)
    key.axis("off")
    # Measured layout: the four entries need 78.5 of the 100 axes units, which
    # leaves 7 units of gap between them.  At the old fixed stops the third
    # entry ran into the fourth swatch.  A share large enough to carry its own
    # in-bar label does not repeat it here.
    for x0, (lab, n, colour) in zip([0.8, 18.2, 41.7, 76.5], parts):
        share = 100 * n / cov.n_rows
        key.scatter([x0], [0], marker="s", s=42, color=colour)
        key.text(x0 + 1.6, 0, lab if share > 10 else f"{lab}  {share:.1f}%",
                 ha="left", va="center", fontsize=9, color=GREY)

    fig.text(0.018, fy(3.12),
             "(b) the 431,245 benchmark rows by what the public source could "
             "supply. The reused-symbol screen",
             fontsize=9, color=GREY, va="top")
    fig.text(0.018, fy(3.33),
             "     itself needs the licensed data to run: 18 of 848 tickers, "
             "and 200 more have no public history.",
             fontsize=9, color=GREY, va="top")

    # ------------------------------------------------- (c) label agreement
    cx = fig.add_axes([0.135, fy(5.92), 0.320, 1.35 / H])
    order = [("train", 5), ("train", 10), ("train", 20), ("val", 5),
             ("val", 10), ("val", 20), ("test", 5), ("test", 10), ("test", 20)]
    ys = np.arange(len(order))
    vals = [float(split_h[(split_h.split == s) & (split_h.h == h)]
                  .pearson_logRV.iloc[0]) for s, h in order]
    cx.scatter(vals, ys, s=24, color=BLUE, zorder=3)
    cx.axvline(got["pearson_clean"], color=GREY, lw=0.8, ls=(0, (4, 2)),
               zorder=2)
    cx.set_yticks(ys)
    cx.set_yticklabels([f"{s}  h={h}" for s, h in order], fontsize=9)
    cx.set_ylim(len(order) - 0.4, -1.25)
    cx.set_xlim(0.9955, 1.0004)
    cx.set_xticks([0.996, 0.998, 1.000])
    cx.set_xticklabels(["0.996", "0.998", "1.000"], fontsize=9)
    cx.grid(axis="x", color=LIGHT, lw=0.5, zorder=0)
    cx.set_axisbelow(True)
    cx.tick_params(length=0, pad=2)
    cx.spines["left"].set_visible(False)
    cx.set_xlabel("correlation of public and licensed log RV", fontsize=9,
                  labelpad=2)
    cx.text(got["pearson_clean"] + 0.0002, -0.85,
            f"all rows {got['pearson_clean']:.4f}", ha="left", va="center",
            fontsize=9, color=GREY)
    fig.text(0.018, fy(4.16),
             "(c) the labels themselves, on rows the public\n"
             "     source covers",
             fontsize=9, color=GREY, va="top", linespacing=1.3)

    # --------------------------------------------- (d) verdict preservation
    dx = fig.add_axes([0.585, fy(5.92), 0.395, 1.35 / H])
    dx.set_xlim(0, 10)
    dx.set_ylim(5.6, -1.5)
    dx.axis("off")
    rows_d = [
        ("standalone DM sign", "18/18", "18/18", True),
        ("standalone Holm verdict", "18/18", "18/18", True),
        ("leaderboard verdicts", "180/180", "180/180", True),
        ("QLIKE ranking per panel", "5/6", "5/6", False),
        ("increment sign", "11/12", "11/12", False),
        ("increment Holm verdict", "12/12", "12/12", True),
    ]
    dx.text(7.4, -1.15, "panel B", ha="center", va="center", fontsize=9,
            color=GREY)
    dx.text(9.4, -1.15, "panel C", ha="center", va="center", fontsize=9,
            color=GREY)
    for i, (lab, b, c, full) in enumerate(rows_d):
        dx.text(0, i, lab, ha="left", va="center", fontsize=9, color=GREY)
        col = GREEN if full else VERM_TXT
        dx.text(7.4, i, b, ha="center", va="center", fontsize=9, color=col)
        dx.text(9.4, i, c, ha="center", va="center", fontsize=9, color=col)
    fig.text(0.560, fy(4.16),
             "(d) verdict objects reproduced, against panel A\n"
             "     (green = every case agrees)",
             fontsize=9, color=GREY, va="top", linespacing=1.3)

    note = (
        "The failure mode is coverage, not measurement. On rows the public "
        "source covers, the labels are\n"
        "near-duplicates — correlation 0.9981 overall, 1.0000 on the test "
        "years — and every verdict object\n"
        "survives: no standalone verdict of 180 flips, and genuine "
        "combination increments move only from 8\n"
        "of 12 to 7 of 12, through a placebo term rather than a DM statistic. "
        "What breaks is the sample. A fifth\n"
        "of the rows have no clean public label, and the loss concentrates in "
        "firms that left the index, whose\n"
        "coverage is 31.9 per cent against 97.7 per cent for firms still "
        "listed, so a licence-free variant is a\n"
        "survivorship-tilted subsample and must be labelled one. NOT shown: "
        "any rerun of the reference\n"
        "ladder on public labels, and any redistribution route — this "
        "snapshot permits research use only."
    )
    fig.text(0.018, fy(6.35), note, fontsize=9, color=GREY, va="top",
             linespacing=1.42)

    ds.finish(fig, "AR4_public_label_variant", max_render_pt=595.0)


if __name__ == "__main__":
    main()
