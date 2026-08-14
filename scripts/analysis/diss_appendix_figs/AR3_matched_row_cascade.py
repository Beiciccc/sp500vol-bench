"""AR3 — is the collapse a control effect or a sample effect?

Appendix figure (outside the 60-page cap).  The primary rung is scored on the
HAR-plus-text join; the firm-identity rung is scored on the smaller five-price-model
intersection, so the reported 38-to-8 fall confounds a change of reference with a
change of sample.  This figure re-scores both rungs on both row sets and separates
the two.

Sources
-------
results/tables/matched_row_cascade.csv       69 cells x 2 row sets, both rungs
results/tables/firm_identity_ensemble.csv    the committed firm-identity rung,
                                             used only to verify that the matched
                                             arm reproduces it cell for cell
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import diss_style as ds  # noqa: E402
from supp_style import (BLUE, GREEN, GREY, LIGHT, PURPLE, TAB, VERM,  # noqa: E402
                        VERM_TXT, apply_style, gate)

KEY = ["disc", "model", "h"]
PANELS = [("long_form", 5), ("long_form", 10), ("long_form", 20),
          ("event_driven", 5), ("event_driven", 10), ("event_driven", 20)]
PLAB = {("long_form", 5): "LF  h=5", ("long_form", 10): "LF  h=10",
        ("long_form", 20): "LF  h=20", ("event_driven", 5): "ED  h=5",
        ("event_driven", 10): "ED  h=10", ("event_driven", 20): "ED  h=20"}


def main():
    m = pd.read_csv(os.path.join(TAB, "matched_row_cascade.csv"))
    firm = pd.read_csv(os.path.join(TAB, "firm_identity_ensemble.csv"))

    piv = m.pivot_table(index=KEY, columns="arm",
                        values=["adds_primary", "adds_firm", "rel_primary",
                                "rel_firm"], aggfunc="first")
    cp = piv[("adds_primary", "committed_rows")]
    mp = piv[("adds_primary", "matched")]
    cf = piv[("adds_firm", "committed_rows")]
    mf = piv[("adds_firm", "matched")]

    committed = m[m.arm == "committed_rows"].groupby(["disc", "h"]).first()
    matched = m[m.arm == "matched"].groupby(["disc", "h"]).first()

    got = {
        "cells": len(piv),
        "committed_primary": int(cp.sum()),
        "matched_primary": int(mp.sum()),
        "committed_firm": int(cf.sum()),
        "matched_firm": int(mf.sum()),
        "rows_lost_primary": int((cp & ~mp).sum()),
        "rows_gained_primary": int((~cp & mp).sum()),
        "control_lost": int((mp & ~mf).sum()),
        "control_gained": int((~mp & mf).sum()),
        "matched_firm_matches_committed_table": (
            set(map(tuple, m[(m.arm == "matched") & m.adds_firm][KEY].values))
            == set(map(tuple, firm[firm.adds_holm][KEY].values))),
        "row_retention_pct": round(100 * matched.n_test.sum()
                                   / committed.n_test.sum(), 1),
        "placebo_drops_matched": int(m[m.arm == "matched"].adds2_primary.sum()
                                     - m[m.arm == "matched"].adds_primary.sum()),
    }
    gate({"cells": 69, "committed_primary": 38, "matched_primary": 30,
          "committed_firm": 11, "matched_firm": 8, "rows_lost_primary": 8,
          "rows_gained_primary": 0, "control_lost": 26, "control_gained": 4,
          "matched_firm_matches_committed_table": True,
          "row_retention_pct": 92.0, "placebo_drops_matched": 3}, got)

    apply_style(9)
    H = 7.55
    fig = plt.figure(figsize=ds.canvas(H, max_h=7.7))

    def fy(t):
        return 1.0 - t / H

    fig.text(0.018, fy(0.05),
             "Separating the reference from the sample. Holm survivors of 69 "
             "combination cells, seed-ensemble",
             fontsize=9, color=GREY, va="top")
    fig.text(0.018, fy(0.26),
             "basis, day-clustered DM, validation-fitted weights frozen on "
             "test; only the labelled thing changes.",
             fontsize=9, color=GREY, va="top")

    # ------------------------------------------------- (a) the 2 x 2 diagram
    ax = fig.add_axes([0.018, fy(2.75), 0.964, 1.95 / H])
    ax.set_xlim(0, 10)
    ax.set_ylim(0.55, 6.15)
    ax.axis("off")

    cols = {"committed": 3.05, "matched": 7.35}
    rows_y = {"har": 4.15, "firm": 1.35}
    boxes = {("har", "committed"): got["committed_primary"],
             ("har", "matched"): got["matched_primary"],
             ("firm", "committed"): got["committed_firm"],
             ("firm", "matched"): got["matched_firm"]}
    bw, bh = 1.45, 0.92
    for (r, c), n in boxes.items():
        x, y = cols[c], rows_y[r]
        final = (r, c) == ("firm", "matched")
        ax.add_patch(Rectangle((x - bw / 2, y - bh / 2), bw, bh,
                               facecolor="white", zorder=3,
                               edgecolor=BLUE if final else GREY,
                               linewidth=1.4 if final else 0.8))
        ax.text(x, y + 0.13, str(n), ha="center", va="center", fontsize=12,
                color=BLUE if final else GREY, zorder=4)
        ax.text(x, y - 0.26, "of 69", ha="center", va="center", fontsize=9,
                color=GREY, zorder=4)

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                     color=GREY, lw=0.9, shrinkA=2, shrinkB=2,
                                     zorder=2))

    arrow((cols["committed"] + bw / 2, rows_y["har"]),
          (cols["matched"] - bw / 2, rows_y["har"]))
    arrow((cols["committed"] + bw / 2, rows_y["firm"]),
          (cols["matched"] - bw / 2, rows_y["firm"]))
    arrow((cols["committed"], rows_y["har"] - bh / 2),
          (cols["committed"], rows_y["firm"] + bh / 2))
    arrow((cols["matched"], rows_y["har"] - bh / 2),
          (cols["matched"], rows_y["firm"] + bh / 2))

    ax.text(5.2, rows_y["har"] + 0.22, "same reference,", ha="center",
            va="bottom", fontsize=9, color=GREY)
    ax.text(5.2, rows_y["har"] - 0.44, "smaller row set:  −8", ha="center",
            va="bottom", fontsize=9, color=GREY)
    ax.text(5.2, rows_y["firm"] + 0.22, "same reference,", ha="center",
            va="bottom", fontsize=9, color=GREY)
    ax.text(5.2, rows_y["firm"] - 0.44, "smaller row set:  −3", ha="center",
            va="bottom", fontsize=9, color=GREY)
    ax.text(cols["committed"] - 0.15, 2.75, "add firm\nidentity:  −27",
            ha="right", va="center", fontsize=9, color=GREY, linespacing=1.25)
    ax.text(cols["matched"] + 0.15, 2.75,
            "add firm identity:\n−22 net (26 lost, 4 gained)", ha="left",
            va="center", fontsize=9, color=GREY, linespacing=1.25)

    ax.text(cols["committed"], 5.75, "committed row set", ha="center",
            va="center", fontsize=9, color=GREY)
    ax.text(cols["committed"], 5.35, "(HAR-plus-text join)", ha="center",
            va="center", fontsize=9, color=GREY)
    ax.text(cols["matched"], 5.75, "matched row set", ha="center",
            va="center", fontsize=9, color=GREY)
    ax.text(cols["matched"], 5.35, "(five-price-model intersection)",
            ha="center", va="center", fontsize=9, color=GREY)
    ax.text(0.6, rows_y["har"], "reference:\nrecalibrated\nHAR-RV",
            ha="center", va="center", fontsize=9, color=GREY, linespacing=1.25)
    ax.text(0.6, rows_y["firm"], "reference:\nHAR-RV plus\nfirm identity",
            ha="center", va="center", fontsize=9, color=GREY, linespacing=1.25)

    fig.text(0.018, fy(0.62),
             "(a) the reported fall from 38 to 8 decomposed. Both paths reach "
             "the same 8 cells.",
             fontsize=9, color=GREY, va="top")

    # ------------------------------------------------ (b) cell by cell
    bx = fig.add_axes([0.115, fy(5.85), 0.365, 1.90 / H])
    xp = piv[("rel_primary", "matched")]
    yp = piv[("rel_firm", "matched")]
    kill = (mp & ~mf)
    gain = (~mp & mf)
    keep = (mp & mf)
    rest = ~(kill | gain | keep)
    bx.axhline(0, color=GREY, lw=0.7, zorder=2)
    ylo, yhi, xlo, xhi = -6.6, 2.0, -4.1, 6.6
    # The zero rule stops above the clipped-cell note; drawn full height it ran
    # through the words "5 cells below the axis".
    bx.plot([0, 0], [ylo + 0.78, yhi], color=GREY, lw=0.7, zorder=2)
    bx.plot([xlo, yhi], [xlo, yhi], color=LIGHT, lw=0.9, zorder=1)
    off = yp < ylo
    yc = yp.clip(lower=ylo + 0.12)
    # Shape, not only hue: "adds only before" is a square and "adds under
    # both" a disc, so the two verdicts survive a greyscale print.  Clipped
    # cells keep the caret that marks them clipped and carry the verdict in
    # their fill (filled = adds before the firm term enters).
    for mask, kw in [(rest & ~off, dict(s=17, facecolor="white",
                                        edgecolor=GREY, linewidths=0.7)),
                     (kill & ~off, dict(s=18, marker="s", color=VERM)),
                     (keep & ~off, dict(s=19, color=BLUE)),
                     (gain & ~off, dict(s=25, marker="^", color=GREEN))]:
        bx.scatter(xp[mask], yc[mask], zorder=4, **kw)
    for i in np.where(off)[0]:
        adds_before = bool(kill.iloc[i] or keep.iloc[i])
        c = VERM if kill.iloc[i] else (BLUE if keep.iloc[i] else GREY)
        bx.scatter([xp.iloc[i]], [ylo + 0.20], marker="v", s=22,
                   facecolor=c if adds_before else "white", edgecolor=c,
                   linewidths=0.7, zorder=5)
    bx.text(xhi - 0.15, ylo + 0.30,
            f"{int(off.sum())} cells below the axis, to "
            f"{yp.min():.1f}".replace("-", "\u2212"),
            ha="right", va="bottom", fontsize=9, color=GREY)
    bx.set_xlim(xlo, xhi)
    bx.set_ylim(ylo, yhi)
    bx.set_xlabel("increment over recalibrated HAR-RV (%)", fontsize=9,
                  labelpad=2)
    bx.set_ylabel("over HAR-RV plus firm identity (%)", fontsize=9, labelpad=2)
    bx.tick_params(length=2.5, pad=2)
    fig.text(0.018, fy(3.05),
             "(b) the same 69 cells on the matched row set, before and\n"
             "     after the firm-identity term enters the reference",
             fontsize=9, color=GREY, va="top", linespacing=1.3)

    lg = fig.add_axes([0.115, fy(3.85), 0.480, 0.34 / H])
    lg.set_xlim(0, 100)
    lg.set_ylim(-1.2, 0.6)
    lg.axis("off")
    lg.scatter([1], [0.3], s=19, color=BLUE)
    lg.text(3.5, 0.3, "adds under both (4)", fontsize=9, color=GREY,
            va="center")
    lg.scatter([52], [0.3], s=18, marker="s", color=VERM)
    lg.text(54.5, 0.3, "adds only before (26)", fontsize=9, color=GREY,
            va="center")
    lg.scatter([1], [-0.8], s=25, marker="^", color=GREEN)
    lg.text(3.5, -0.8, "adds only after (4)", fontsize=9, color=GREY,
            va="center")
    lg.scatter([52], [-0.8], s=17, facecolor="white", edgecolor=GREY,
               linewidths=0.7)
    lg.text(54.5, -0.8, "adds under neither (35)", fontsize=9, color=GREY,
            va="center")

    # ----------------------------------------------------- (c) the support
    cx = fig.add_axes([0.635, fy(5.85), 0.335, 1.90 / H])
    y = np.arange(6)
    rows_pct = [100 * matched.loc[p].n_test / committed.loc[p].n_test
                for p in PANELS]
    days_pct = [100 * matched.loc[p].n_days / committed.loc[p].n_days
                for p in PANELS]
    cx.barh(y - 0.19, rows_pct, height=0.34, color=PURPLE, zorder=3,
            label="test rows")
    cx.barh(y + 0.19, days_pct, height=0.34, color=LIGHT, edgecolor=GREY,
            linewidth=0.5, zorder=3, label="trading days")
    for yi, (a, b) in enumerate(zip(rows_pct, days_pct)):
        cx.text(a - 1.2, yi - 0.19, f"{a:.0f}", ha="right", va="center",
                fontsize=9, color="white", zorder=4)
        cx.text(b - 1.2, yi + 0.19, f"{b:.0f}", ha="right", va="center",
                fontsize=9, color=GREY, zorder=4)
    cx.set_ylim(5.6, -0.6)
    cx.set_yticks(y)
    cx.set_yticklabels([PLAB[p] for p in PANELS], fontsize=9)
    # Zero baseline: at an 80 per cent origin a 90 per cent bar and a 95 per
    # cent bar differed by about 2:1 in drawn length, which overstates a cost
    # this panel exists to call small.
    cx.set_xlim(0, 101)
    cx.set_xticks([0, 25, 50, 75, 100])
    cx.tick_params(length=0, pad=2)
    cx.spines["left"].set_visible(False)
    cx.grid(axis="x", color=LIGHT, lw=0.5, zorder=0)
    cx.set_axisbelow(True)
    cx.set_xlabel("kept by the matched row set (%)", fontsize=9, labelpad=2)
    cx.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), fontsize=9,
              ncol=2, handlelength=1.1, handletextpad=0.4, columnspacing=1.2,
              borderpad=0.0)
    fig.text(0.545, fy(3.05),
             # 92.0 is the pooled retention, not a drawn bar: the six cells
             # keep 95.0/90.3/89.8/95.0/91.1/90.2 per cent of their test rows.
             # src: results/tables/matched_row_cascade.csv (n_test, matched
             # against committed_rows, by disc and h)
             "(c) what the matched row set costs: 90 to 95 per\n"
             "     cent of test rows (92.0 pooled), and 95.4 to 100\n"
             "     per cent of trading days",
             fontsize=9, color=GREY, va="top", linespacing=1.3)

    note = (
        "Read the diagram either way round: the control, not the sample, does "
        "most of the work. On\n"
        "one row set the fall is 30 to 8; on the committed rows the reference "
        "alone takes 38 to 11.\n"
        "Two checks make the arms comparable — the same code path on the "
        "committed rows returns\n"
        "the committed 38 of 69, and the matched arm's firm rung reproduces "
        "the committed firm-\n"
        "identity rung cell for cell, the same 8 cells. The second step is not "
        "monotone: 26 fall and 4\n"
        "rise, so 22 is a net. Survivorship here includes the placebo gate, "
        "which binds only on the\n"
        "matched rows, removing 3 of 33. NOT shown: the maximal-pool rung; the "
        "firm-mean term is itself\n"
        "fitted on validation and covers 91.5 to 91.9 per cent of test "
        "observations."
    )
    fig.text(0.018, fy(6.25), note, fontsize=9, color=GREY, va="top",
             linespacing=1.42)

    ds.finish(fig, "AR3_matched_row_cascade", max_render_pt=595.0)


if __name__ == "__main__":
    main()
