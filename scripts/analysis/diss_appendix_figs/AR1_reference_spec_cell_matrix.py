"""AR1 — does the verdict depend on which price reference is chosen?

Appendix figure (outside the 60-page cap).  Draws all 69 combination cells
against four price references ordered by how much they absorb, so a reader can
see *which* cells die as the reference is strengthened rather than only how many.

Sources
-------
results/tables/m1_ensemble_primary.csv       the A2-only (primary) rung, per cell
results/tables/maximal_pool_robustness.csv   three pool specifications x two seed
                                             bases, per cell (414 = 69 x 3 x 2)

Every count the figure prints is gated against the values the dissertation
states (38 / 34 / 17 / 9 of 69 on the seed-ensemble basis).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import diss_style as ds  # noqa: E402
from supp_style import (BLUE, GREY, INK, INK2, LIGHT, RULE,  # noqa: E402
                        TAB, VERM, apply_style, gate)

KEY = ["disc", "model", "h"]
HS = [5, 10, 20]
MID = "#4E9BCB"
PALE = "#9DC6E0"


# ------------------------------------------------------------------ evidence
def load():
    prim = pd.read_csv(os.path.join(TAB, "m1_ensemble_primary.csv"))
    rob = pd.read_csv(os.path.join(TAB, "maximal_pool_robustness.csv"))

    p = prim[KEY].copy()
    p["dm"], p["p"], p["holm"] = (prim.vol_dm_q_clu, prim.vol_p_q_clu,
                                  prim.vol_dmq_holm_clu)
    p["ref"], p["basis"] = "primary", "ens"

    q = prim[KEY].copy()
    q["dm"], q["p"], q["holm"] = (prim.s26_dm_q_clu, prim.s26_p_q_clu,
                                  prim.s26_dmq_holm_clu)
    q["ref"], q["basis"] = "primary", "s26"

    r = rob[KEY + ["basis", "ref"]].copy()
    r["dm"], r["p"], r["holm"] = (rob.dm_q_clustered, rob.p_q_clustered,
                                  rob.p_holm)
    return pd.concat([p, q, r], ignore_index=True)


def state(row):
    """Five-state verdict for one cell against one reference."""
    if row.dm < 0 and row.holm < 0.05:
        return "ADD"
    if row.dm < 0 and row.p < 0.05:
        return "add_raw"
    if row.dm > 0 and row.holm < 0.05:
        return "HURT"
    if row.dm > 0 and row.p < 0.05:
        return "hurt_raw"
    return "null"


def counts(d, basis, ref, what="ADD"):
    g = d[(d.basis == basis) & (d.ref == ref)]
    return int((g.verdict == what).sum())


# --------------------------------------------------------------------- draw
def marker(ax, x, y, verdict, side=7.4):
    """One verdict glyph, drawn at a size fixed in points (never stretched)."""
    face = {"ADD": BLUE, "HURT": VERM}.get(verdict, "white")
    edge = {"ADD": BLUE, "add_raw": BLUE,
            "HURT": "white", "hurt_raw": VERM}.get(verdict, GREY)
    # Hue is not the only channel carrying the verdict: both "hurts" states are
    # hatched as well, so an adds cell and a hurts cell stay distinguishable in
    # greyscale and under photocopying.
    hatch = "///" if verdict in ("HURT", "hurt_raw") else None
    ax.scatter([x], [y], marker="s", s=side ** 2, facecolor=face,
               edgecolor=edge, linewidths=0.7, zorder=3, clip_on=False,
               hatch=hatch)
    if verdict in ("add_raw", "hurt_raw"):
        ax.scatter([x], [y], marker="o", s=3.2, color=edge, zorder=4,
                   clip_on=False)


def main():
    d = load()
    d["verdict"] = d.apply(state, axis=1)

    wide = (d[d.basis == "ens"]
            .pivot_table(index=KEY, columns="ref", values="verdict",
                         aggfunc="first"))
    adds = wide.isin(["ADD"])
    n_spec = adds[["valbest_single", "eqw_pool", "fitted_pool"]].sum(axis=1)

    got = {
        "cells": len(wide),
        "primary_holm": counts(d, "ens", "primary"),
        "primary_raw_or_holm": int(((d.basis == "ens") & (d.ref == "primary")
                                    & d.verdict.isin(["ADD", "add_raw"])).sum()),
        "vbs_holm": counts(d, "ens", "valbest_single"),
        "eqw_holm": counts(d, "ens", "eqw_pool"),
        "fitted_holm": counts(d, "ens", "fitted_pool"),
        # Both directions, not only the favourable one: under the equal-weight
        # pool the significant hurts (11) are within six of the significant adds
        # (17), and a header that prints only the adds hides that.
        "primary_hurt": counts(d, "ens", "primary", "HURT"),
        "vbs_hurt": counts(d, "ens", "valbest_single", "HURT"),
        "eqw_hurt": counts(d, "ens", "eqw_pool", "HURT"),
        "fitted_hurt": counts(d, "ens", "fitted_pool", "HURT"),
        "s26_primary": counts(d, "s26", "primary"),
        "s26_vbs": counts(d, "s26", "valbest_single"),
        "s26_eqw": counts(d, "s26", "eqw_pool"),
        "s26_fitted": counts(d, "s26", "fitted_pool"),
        "all_three": int((n_spec == 3).sum()),
        "union_any_pool": int((n_spec > 0).sum()),
        "rescued_by_stronger": int((n_spec.gt(0) & ~adds["primary"]).sum()),
    }
    gate({
        "cells": 69,
        "primary_holm": 38, "primary_raw_or_holm": 46,
        "vbs_holm": 34, "eqw_holm": 17, "fitted_holm": 9,
        "primary_hurt": 8, "vbs_hurt": 10, "eqw_hurt": 11, "fitted_hurt": 5,
        "s26_primary": 29, "s26_vbs": 28, "s26_eqw": 19, "s26_fitted": 8,
        "all_three": 6, "union_any_pool": 35, "rescued_by_stronger": 0,
    }, got)

    apply_style(9)
    H = 7.60
    fig = plt.figure(figsize=ds.canvas(H, max_h=7.7))

    def fy(inches_from_top):
        return 1.0 - inches_from_top / H

    ed_models = list(d[d.disc == "event_driven"].model.unique())
    lf_models = list(d[d.disc == "long_form"].model.unique())
    gap = 1.15                                  # blank slot between the panels
    rows = ([(("event_driven", m), i) for i, m in enumerate(ed_models)]
            + [(("long_form", m), len(ed_models) + gap + i)
               for i, m in enumerate(lf_models)])
    y_max = len(ed_models) + gap + len(lf_models) - 1

    refs = [("primary", "recalibrated\nHAR-RV alone", got["primary_holm"]),
            ("valbest_single", "validation-best\nsingle member", got["vbs_holm"]),
            ("eqw_pool", "equal-weight\n5-model pool", got["eqw_holm"]),
            ("fitted_pool", "fitted\n5-model pool", got["fitted_holm"])]

    ax = fig.add_axes([0.265, fy(5.20), 0.715, (5.20 - 1.25) / H])
    xs, x = {}, 0.0
    for ref, _, _ in refs:
        for hi, h in enumerate(HS):
            xs[(ref, h)] = x + hi
        x += 3 + 0.85
    ax.set_xlim(-0.55, max(xs.values()) + 0.55)
    ax.set_ylim(y_max + 0.6, -0.6)
    ax.axis("off")

    look = {(r.disc, r.model, r.h, r.ref): r.verdict
            for r in d[d.basis == "ens"].itertuples()}
    for (disc, model), yi in rows:
        for ref, _, _ in refs:
            for h in HS:
                marker(ax, xs[(ref, h)], yi, look[(disc, model, h, ref)])
        ax.text(-0.85, yi, model, ha="right", va="center", fontsize=9,
                color=GREY)

    # block headers, drawn in figure space above the matrix.  The count line
    # carries both verdict directions and names them: "38 of 69" was a bare
    # number with no noun (adds? resolved cells?) and reported only the
    # favourable direction, so a reader of the headers alone never learned that
    # the equal-weight pool's 17 adds sit against 11 significant hurts.  The
    # denominator drops out of the header because it is already stated twice
    # above it, in the panel title ("all 69 combination cells") and in the basis
    # line ("Holm within each 69-cell family"); the widest of the four strings
    # measures 63.2 pt against an 82.5 pt block pitch, so no header touches its
    # neighbour and the tight bounding box (set on the right by the legend's
    # "hurts (Holm)") is untouched.  One colour for all four: the count is the
    # same quantity in every block, and BLUE here is the "text adds" verdict
    # colour, which made three of the four counts look like a class of their own.
    hurt = {"primary": got["primary_hurt"], "valbest_single": got["vbs_hurt"],
            "eqw_pool": got["eqw_hurt"], "fitted_pool": got["fitted_hurt"]}
    for ref, lab, n in refs:
        x0, x1 = xs[(ref, 5)], xs[(ref, 20)]
        xm = (x0 + x1) / 2
        ax.text(xm, -3.35, lab, ha="center", va="center", fontsize=9,
                color=GREY, linespacing=1.15, clip_on=False)
        ax.text(xm, -2.05, f"{n} add, {hurt[ref]} hurt", ha="center",
                va="center", fontsize=9, color=GREY, clip_on=False)
        for h in HS:
            ax.text(xs[(ref, h)], -1.2, str(h), ha="center", va="center",
                    fontsize=9, color=GREY, clip_on=False)
        ax.plot([x0 - 0.45, x1 + 0.45], [-1.72, -1.72], color=LIGHT, lw=0.8,
                clip_on=False)
    ax.text(-0.85, -1.2, "h =", ha="right", va="center", fontsize=9,
            color=GREY, clip_on=False)

    # Panel separators.  These two are the block's structural labels, not data,
    # so they take INK2 and hang left of the model-name column: right-aligned at
    # -0.85 like the model names, "event-driven panel" ran into "h =" directly
    # above it -- the ascender of its final "l" crossed the lower bar of the
    # "=" (verified at 400 dpi).  -1.75 moves them into the blank strip that
    # already exists left of the labels, so the bounding box does not grow.
    ax.text(-1.75, -0.05 + len(ed_models) + gap - 0.72, "long-form panel",
            ha="right", va="center", fontsize=9, color=INK2, style="italic",
            clip_on=False)
    ax.text(-1.75, -0.72, "event-driven panel", ha="right", va="center",
            fontsize=9, color=INK2, style="italic", clip_on=False)
    ax.plot([-0.5, max(xs.values()) + 0.5],
            [len(ed_models) + gap / 2 - 0.5] * 2, color=LIGHT, lw=0.8,
            clip_on=False)

    # The panel title and the basis statement were one run of prose broken
    # mid-sentence across two lines in one colour, so nothing told a reader where
    # the title stopped and the apparatus began.  Same words, same two lines,
    # same size: the break now falls at the end of the title sentence and the
    # basis statement recedes to INK2.
    fig.text(0.018, fy(0.05),
             "(a) all 69 combination cells against four price references, "
             "ordered left to right by how much they absorb.",
             fontsize=9, color=INK, va="top")
    fig.text(0.018, fy(0.23),
             "     Seed-ensemble basis, day-clustered DM, Holm within "
             "each 69-cell family.",
             fontsize=9, color=INK2, va="top")

    # legend
    lg = fig.add_axes([0.018, fy(0.60), 0.964, 0.20 / H])
    lg.set_xlim(0, 100)
    lg.set_ylim(-0.5, 0.5)
    lg.axis("off")
    for x0, v, lab in [(1, "ADD", "text adds (Holm)"),
                       (27, "add_raw", "adds, raw only"),
                       (50, "null", "no effect"),
                       (67, "hurt_raw", "hurts, raw only"),
                       (89, "HURT", "hurts (Holm)")]:
        marker(lg, x0, 0, v, side=7.0)
        lg.text(x0 + 1.8, 0, lab, ha="left", va="center", fontsize=9,
                color=GREY)

    # ------------------------------------------------- (b) consensus counts
    # (b) and (c) sit 0.08 in higher than they did.  That is paid for out of the
    # blank strip already inside the bounding box between the matrix's last row
    # and this band, not by growing the canvas: the hairline that now separates
    # the apparatus note from the data needs clear air above it, and buying that
    # air below instead would push the note past the page box, widen nothing but
    # lengthen the float, and lower every printed glyph.
    BAND = 6.32 - 0.08
    bx = fig.add_axes([0.135, fy(BAND), 0.275, 0.68 / H])
    cons = [int((n_spec == k).sum()) for k in range(4)]
    y = np.arange(4)
    bx.barh(y, cons, color=[LIGHT, PALE, MID, BLUE], edgecolor=GREY,
            linewidth=0.5, height=0.74)
    for yi, c in zip(y, cons):
        bx.text(c + 0.9, yi, str(c), va="center", ha="left", fontsize=9,
                color=GREY)
    bx.set_yticks(y)
    bx.set_yticklabels(["none", "one", "two", "all three"], fontsize=9)
    bx.set_xlim(0, 41)
    bx.set_xticks([])
    bx.invert_yaxis()
    for s in ("bottom", "left"):
        bx.spines[s].set_visible(False)
    bx.tick_params(length=0)
    # "all three pools" had no referent on the page: panel (a) draws four price
    # references and only two of them are pools, the third set being counted
    # here is the validation-best *single* member.  "stronger references" is the
    # figure's own vocabulary (the note opens "No cell is rescued by a stronger
    # reference") and picks out exactly the three blocks to the right of the
    # HAR-only baseline.  The string is 27 pt narrower than the one it replaces,
    # so the gap to panel (c)'s title widens rather than closes.
    fig.text(0.018, fy(5.42 - 0.08),
             "(b) cells adding under 0-3 stronger references",
             fontsize=9, color=INK, va="top")

    # ------------------------------------------------------ (c) basis strip
    cx = fig.add_axes([0.665, fy(BAND), 0.300, 0.68 / H])
    ens = [got["primary_holm"], got["vbs_holm"], got["eqw_holm"],
           got["fitted_holm"]]
    s26 = [got["s26_primary"], got["s26_vbs"], got["s26_eqw"],
           got["s26_fitted"]]
    y = np.arange(4)
    for yi, (a, b) in enumerate(zip(ens, s26)):
        cx.plot([min(a, b), max(a, b)], [yi, yi], color=LIGHT, lw=2.0,
                zorder=1, solid_capstyle="round")
        cx.text(50, yi, f"{a} / {b}", ha="right", va="center", fontsize=9,
                color=GREY)
    cx.scatter(ens, y, s=22, color=BLUE, zorder=3)
    cx.scatter(s26, y, s=22, facecolor="white", edgecolor=GREY, linewidth=0.8,
               zorder=3)
    cx.set_yticks(y)
    cx.set_yticklabels(["HAR alone", "val-best single", "equal-weight pool",
                        "fitted pool"], fontsize=9)
    cx.set_xlim(3, 51)
    cx.set_xticks([])
    cx.invert_yaxis()
    for s in ("bottom", "left"):
        cx.spines[s].set_visible(False)
    cx.tick_params(length=0)
    # "2026" alone reads as a year on a 2010-2025 sample, which left the open
    # markers meaningless; it is the training seed of the archived single-seed
    # basis.  "seed " is paid for by dropping "same ", which is the wider of the
    # two strings, so the title ends 2.5 pt further left than it did -- required,
    # not cosmetic: this title's right end sits close behind the legend entry
    # that sets the content edge, and adding five characters unpaid would push
    # past it and force the whole figure to be scaled down at inclusion.
    fig.text(0.575, fy(5.42 - 0.08),
             "(c) counts: ensemble (filled), seed 2026 (open)",
             fontsize=9, color=INK, va="top")

    # Two wording repairs inside the existing six lines, both of which shorten
    # the block: "pool specification" -> "stronger reference" (only two of the
    # four references drawn are pools, and the set being counted includes the
    # validation-best single member), and "prompted arm" -> "C6_llmtext", which
    # is the row label panel (a) actually prints, so the six named cells can now
    # be found in the matrix.  Re-wrapped by measurement, not by eye: the widest
    # line falls from 361.3 pt to 358.9 pt and the block stays at six lines.
    note = (
        "No cell is rescued by a stronger reference: the 35 cells adding under at "
        "least one\n"
        "stronger reference are a subset of the 38 adding against recalibrated "
        "HAR-RV alone. Six\n"
        "add under all three — TF-IDF ridge h=5/10/20, Longformer h=5, C6_llmtext "
        "h=5/10 — and\n"
        "all six are long-form. NOT shown: the firm-identity and conjunction "
        "rungs (Chapter\n"
        "4), the placebo term (it removes no cell here; the primary rung is 38 with "
        "and without\n"
        "it), and effect size — a filled square is a verdict, not a magnitude."
    )
    # This block is apparatus: it carries the subset claim, the six named cells,
    # and the four things the figure cannot show.  Set in the data ink and with
    # no rule, it read as one more unframed paragraph of the same standing as the
    # panel titles.  A hairline above it and INK2 inside it separate argument from
    # basis without touching a word, a size or the page box.  supp_style.note()
    # is not used here only because it hardcodes linespacing=1.32 and this block
    # was set at 1.42; the ink and the rule are its.
    fig.lines.append(plt.Line2D([0.018, 0.858], [fy(6.38)] * 2,
                                transform=fig.transFigure, color=RULE,
                                linewidth=0.5, zorder=0.5))
    fig.text(0.018, fy(6.47), note, fontsize=9, color=INK2, va="top",
             linespacing=1.42)

    ds.finish(fig, "AR1_reference_spec_cell_matrix", max_render_pt=595.0)


if __name__ == "__main__":
    main()
