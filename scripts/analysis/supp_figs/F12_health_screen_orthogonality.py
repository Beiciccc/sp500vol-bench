"""F12 -- The forecaster-health screen reads the forecast distribution, not the label.

Panel (a) puts the screen's two axes in one plane: the share of forecasts taking
the modal 2-decimal value (gate < 60%) against the standalone variance-unit
QLIKE of the same forecasts (gate < 4).  Panel (b) puts the increments the
screen decides on beneath the same x-axis, so a reader can check that the screen
is orthogonal to increment sign.  A separate lower band carries the
4K-TRUNCATED Yi long-form rows on their own scale.

Sources (every plotted number is read from these files at run time; nothing is
hardcoded outside the gate() expectation block)
--------------------------------------------------------------------------
* results/tables/crossfamily_llama70.csv
    disc, family, h, rel_har, qlike_var, mode_share_pct, parse_fail_rate,
    clipped_rate, flag        -- Qwen3-32B (both channels), Yi-1.5-34B (both
                                 channels), Phi-4-14B, Llama-3.1-70B-AWQ
* results/tables/crossfamily_gemma27.csv
    same columns              -- Mistral-Small-24B and Gemma-3-27B, bf16 and
                                 3-run-ensemble rows, plus the 70B ens3 rows
                                 whose health columns are NaN
* results/tables/crossfamily_mistral24.csv
    read only to assert that the Mistral and 70B rows agree row-for-row with
    crossfamily_gemma27.csv (the union the spec asks for)
* results/tables/crossfamily_standalone.csv
    context_flag              -- 4K-TRUNCATED (Yi long-form), PARTIAL(4K)
                                 (Yi combined)
* results/tables/crossfamily_gemma27_pilot.json
    healthy, max_qlike_var, max_mode_share_pct, n_docs, split
    NOTE the file also holds model_path_resolved, an absolute filesystem path.
    It is never read into any plotted or printed string.

Main-text sentences substantiated
---------------------------------
07_ablations.tex: "of five probes, four fail a forecaster-health screen and are
instrument-dead rather than disconfirming.  The screen reads the forecast
distribution, never the label, and is orthogonal to increment-sign by
construction, so it discards the table's largest positive (Gemma +2.97%), which
an agreement-selecting filter never would; the healthy instruments stay below
the 60% modal-share ceiling on the 8-K channel (Qwen 49--50%, Llama-70B
40--51%) while dead-probe signs are mixed."
04_methods.tex: "Five cross-family replications test whether any increment
generalises beyond Qwen3, each screened for forecaster health before its
reading counts."

Scope carried on the artefact: the gate axis is variance-unit QLIKE (a
standalone diagnostic) while the increment axis is volatility-unit relative
QLIKE against the recalibrated HAR; the two conventions are labelled separately
and are never mixed on one axis.

Presentation
------------
A presentation-only pass gave this figure two clear ranks of type without touching
a number, a word or a font size.  Panel titles and data labels hold primary ink;
the three apparatus blocks, and the two legend rows that state a convention rather
than name a run, drop to INK2, so a reader can tell the argument from the basis
statement at a glance.  The five in-axes callouts go through annot(), which keeps
them legible where they lie over data, and the registered-pilot block moved down
into the empty floor of the healthy quadrant, out from under the marker that was
eating its last glyph.  Weight is deliberately unused: see the note at panel (a)'s
title for the measurement that shows this environment cannot render it.

Every one of those changes is weight, colour, or an INWARD move.  finish() writes
with bbox_inches="tight", so the PDF page IS the content's bounding box and the
dissertation includes it at width=\\textwidth: one point added to any label widens
that box and scales every glyph in the figure DOWN on the page.  The page came out
of this pass at 463.9 x 595.6 pt against 463.9 x 596.1 before it: identical width,
identical 0.9813x inclusion scale, identical 8.83 pt of printed type, and half a
point shorter, so the float it makes is 584.5 pt instead of 585.0 pt.
"""
import json
import os
import re
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supp_style import (BLUE, GREEN, GREY, INK, INK2, LIGHT, PURPLE, REPO, SKY,
                        TAB, VERM, VERM_TXT, YELLOW, annot, apply_style, finish,
                        gate, note)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import offset_copy

W, H = 6.5, 8.80                      # canvas inches (portrait supplement page)


def rect(x, y, w, h):
    """Axes rectangle from inches (origin bottom-left) to figure fractions."""
    return [x / W, y / H, w / W, h / H]


def para(text, width):
    """Wrap a multi-paragraph string, keeping the blank line between paras."""
    return "\n\n".join("\n".join(textwrap.wrap(p, width))
                       for p in text.split("\n\n"))


# ----------------------------------------------------------------- evidence
l70 = pd.read_csv(os.path.join(TAB, "crossfamily_llama70.csv"))
gem = pd.read_csv(os.path.join(TAB, "crossfamily_gemma27.csv"))
mis = pd.read_csv(os.path.join(TAB, "crossfamily_mistral24.csv"))
std = pd.read_csv(os.path.join(TAB, "crossfamily_standalone.csv"))
with open(os.path.join(TAB, "crossfamily_gemma27_pilot.json")) as fh:
    pilot = json.load(fh)

# the spec asks for the union of the three cross-family tables; the Mistral and
# 70B rows are duplicated across gemma27.csv and mistral24.csv, so assert they
# agree rather than choosing arbitrarily between them
_key = ["disc", "family", "h"]
_shared = sorted(set(gem.family) & set(mis.family))
_a = gem[gem.family.isin(_shared)].sort_values(_key).reset_index(drop=True)
_b = mis[mis.family.isin(_shared)].sort_values(_key).reset_index(drop=True)
for _c in ("rel_har", "qlike_var", "mode_share_pct"):
    assert np.allclose(_a[_c].to_numpy(dtype=float),
                       _b[_c].to_numpy(dtype=float),
                       rtol=1e-12, equal_nan=True), f"union disagrees on {_c}"


def rows(df, family, disc):
    s = df[(df.family == family) & (df.disc == disc)].sort_values("h")
    assert len(s) == 3, f"{family}/{disc}: expected 3 horizons, got {len(s)}"
    return s


# panel (a): single-pass runs only -- these are the rows that carry the full
# health block (variance-unit QLIKE, modal share, parse-fail and clipped rate).
# The registered pilot was also a single pass, so the arrow below compares like
# with like.
HEALTH = [
    # key, label, colour, marker, frame (df), family tag, channel, hollow?
    ("qwen_ed", "Qwen3-32B", BLUE, "*", l70, "qwen3_32b", "event_driven", False),
    ("qwen_lf", "Qwen3-32B", BLUE, "*", l70, "qwen3_32b", "long_form", True),
    ("l70_ed", "Llama-3.1-70B", GREEN, "o", l70, "llama70_awq", "event_driven", False),
    ("yi_ed", "Yi-1.5-34B", VERM, "s", l70, "yi_34b", "event_driven", False),
    ("yi_lf", "Yi-1.5-34B", VERM, "s", l70, "yi_34b", "long_form", True),
    ("phi_ed", "Phi-4-14B", YELLOW, "^", l70, "phi4_14b", "event_driven", False),
    ("mis_ed", "Mistral-Small-24B", PURPLE, "v", gem, "mistral24_bf16",
     "event_driven", False),
    ("gem_ed", "Gemma-3-27B", SKY, "D", gem, "gemma27_bf16", "event_driven", False),
]
health = {}
for key, lab, col, mk, df, fam, disc, hollow in HEALTH:
    s = rows(df, fam, disc)
    health[key] = dict(label=lab, colour=col, marker=mk, hollow=hollow,
                       x=s.mode_share_pct.to_numpy(dtype=float),
                       y=s.qlike_var.to_numpy(dtype=float),
                       h=s.h.to_numpy(dtype=int))

# The two thresholds are PARSED out of the registered formula the pilot JSON
# carries verbatim, so they have a source outside this script and a silent
# edit of either number here cannot pass the gate below.
_gate_str = str(pilot["gate"])
_m_q = re.search(r"QLIKE\s*<\s*([0-9.]+)", _gate_str)
_m_m = re.search(r"modal share[^<]*<\s*([0-9.]+)\s*%", _gate_str)
assert _m_q and _m_m, "pilot JSON no longer states the registered gate formula"
GATE_QLIKE = float(_m_q.group(1))
GATE_MODE = float(_m_m.group(1))
for k, d in health.items():
    d["pass"] = bool(d["x"].max() < GATE_MODE and d["y"].max() < GATE_QLIKE)

# the 70B's 3-run ensemble rows carry no health columns at all, so they cannot
# be placed on the gate plane (adversarial repair, lens 1)
ens70 = gem[gem.family == "llama70_awq_ens3"]
ens70_health_missing = bool(ens70[["qlike_var", "mode_share_pct",
                                   "parse_fail_rate", "clipped_rate"]]
                            .isna().all().all())

# how far the 3-run ensembles of Mistral and Gemma sit from their single-pass
# rows, so the single-pass choice in panel (a) can be priced on the artefact
ens_drift_mode, ens_drift_qlike = 0.0, 0.0
for single, ens in (("mistral24_bf16", "mistral24_ens3"),
                    ("gemma27_bf16", "gemma27_ens3")):
    a, b = rows(gem, single, "event_driven"), rows(gem, ens, "event_driven")
    ens_drift_mode = max(ens_drift_mode,
                         float(np.abs(a.mode_share_pct.to_numpy()
                                      - b.mode_share_pct.to_numpy()).max()))
    ens_drift_qlike = max(ens_drift_qlike,
                          float(np.abs(a.qlike_var.to_numpy()
                                       - b.qlike_var.to_numpy()).max()))

# panel (b): increments on exactly the basis the frozen main table quotes --
# Gemma on its 3-run ensemble, Mistral on bf16, the 70B single-seed, Yi and
# Phi-4 as committed.  Yi is quoted on the context-clean 8-K channel; its
# long-form rows go to the truncated band below (adversarial repair, lens 3).
INCR = [
    ("qwen_ed", "Qwen3-32B", BLUE, "*", l70, "qwen3_32b", "event_driven", True),
    ("qwen_lf", "Qwen3-32B", BLUE, "*", l70, "qwen3_32b", "long_form", False),
    ("l70_ed", "Llama-3.1-70B", GREEN, "o", l70, "llama70_awq", "event_driven", True),
    ("yi_ed", "Yi-1.5-34B", VERM, "s", l70, "yi_34b", "event_driven", False),
    ("phi_ed", "Phi-4-14B", YELLOW, "^", l70, "phi4_14b", "event_driven", False),
    ("mis_ed", "Mistral-Small-24B", PURPLE, "v", gem, "mistral24_bf16",
     "event_driven", False),
    ("gem_ed", "Gemma-3-27B", SKY, "D", gem, "gemma27_ens3", "event_driven", False),
]
incr = {}
for key, lab, col, mk, df, fam, disc, healthy in INCR:
    s = rows(df, fam, disc)
    incr[key] = dict(label=lab, colour=col, marker=mk, healthy=healthy,
                     long_form=(disc == "long_form"),
                     x=s.mode_share_pct.to_numpy(dtype=float),
                     y=s.rel_har.to_numpy(dtype=float),
                     h=s.h.to_numpy(dtype=int))
# the Gemma ensemble rows carry modal share but the screen verdict is the run's,
# so re-derive it from the same rows rather than inheriting the bf16 verdict
for key in incr:
    if key in health:
        incr[key]["healthy"] = health[key]["pass"]
incr["gem_ed"]["healthy"] = bool(
    rows(gem, "gemma27_ens3", "event_driven").mode_share_pct.max() < GATE_MODE
    and rows(gem, "gemma27_ens3", "event_driven").qlike_var.max() < GATE_QLIKE)

yi_lf = rows(l70, "yi_34b", "long_form")
band_x = yi_lf.mode_share_pct.to_numpy(dtype=float)
band_y = yi_lf.rel_har.to_numpy(dtype=float)

# context flags, read rather than typed
flag_yi_lf = std[(std.family == "yi_34b")
                 & (std.disc == "long_form")].context_flag.unique()
flag_yi_cb = std[(std.family == "yi_34b")
                 & (std.disc == "combined")].context_flag.unique()
flag_l70 = l70[l70.family == "llama70_awq"].flag.unique()
assert len(flag_yi_lf) == len(flag_yi_cb) == len(flag_l70) == 1

# the two QLIKE-ranking statistics the adversaries disagreed about.  Lens 1 asks
# for the lowest MAX-over-horizons value (Phi-4); lens 2 asks for the lowest
# single-cell value (Mistral).  They are different statistics, not a
# disagreement about the data, so the conservative resolution is to name both
# explicitly rather than pick one and leave the reader to guess.
probe_maxq = {"Llama-3.1-70B": health["l70_ed"]["y"].max(),
              "Yi-1.5-34B": max(health["yi_ed"]["y"].max(),
                                health["yi_lf"]["y"].max()),
              "Phi-4-14B": health["phi_ed"]["y"].max(),
              "Mistral-Small-24B": float(gem[gem.family.str.startswith("mistral24")]
                                         .qlike_var.max()),
              "Gemma-3-27B": float(gem[gem.family.str.startswith("gemma27")]
                                   .qlike_var.max())}
lowest_maxq = min(probe_maxq, key=probe_maxq.get)

# The lowest SINGLE CELL is an argmin over every committed probe row, never a
# named family: Gemma-3-27B holds it (0.377, 3-run ensemble h=20), below
# Mistral-Small-24B's 0.421, under both the all-rows and the plotted single-
# pass reading.  Naming a family here would contradict Table T11.
PROBE_LABEL = {"llama70_awq": "Llama-3.1-70B", "yi_34b": "Yi-1.5-34B",
               "phi4_14b": "Phi-4-14B",
               "mistral24_bf16": "Mistral-Small-24B",
               "mistral24_ens3": "Mistral-Small-24B",
               "gemma27_bf16": "Gemma-3-27B", "gemma27_ens3": "Gemma-3-27B"}
all_probe_q = pd.concat([
    l70[l70.family.isin(["yi_34b", "phi4_14b", "llama70_awq"])],
    gem[gem.family.str.startswith(("mistral24", "gemma27"))]],
    ignore_index=True)
all_probe_q = all_probe_q[all_probe_q.qlike_var.notna()]
_lo = all_probe_q.loc[all_probe_q.qlike_var.astype(float).idxmin()]
lowest_cell_fam = PROBE_LABEL[str(_lo.family)]
lowest_cell_q = float(_lo.qlike_var)
# the same argmin restricted to the single-pass rows panel (a) actually plots
_plotted = all_probe_q[~all_probe_q.family.astype(str).str.endswith("_ens3")]
_lo_p = _plotted.loc[_plotted.qlike_var.astype(float).idxmin()]
lowest_cell_plotted_q = float(_lo_p.qlike_var)
assert PROBE_LABEL[str(_lo_p.family)] == lowest_cell_fam, \
    "plotted and all-rows single-cell minima name different families"

pilot_x = float(pilot["max_mode_share_pct"])
pilot_y = float(pilot["max_qlike_var"])
gem_bf16 = rows(gem, "gemma27_bf16", "event_driven")
arrow_x = float(gem_bf16[gem_bf16.h == 5].mode_share_pct.iloc[0])
arrow_y = float(gem_bf16[gem_bf16.h == 5].qlike_var.iloc[0])

# The probe count is per FAMILY, not per run: the screen's verdict attaches to
# a (family, channel) run, and Yi-1.5-34B contributes two of them, so summing
# over runs returns 5 where the paper says 4.  Both readings are derived from
# the plotted health columns and asserted equal, so a drifting column moves the
# printed count instead of being papered over by a literal.
PROBE_ED = {"yi": "yi_ed", "phi": "phi_ed", "mis": "mis_ed",
            "gem": "gem_ed", "l70": "l70_ed"}
PROBE_ALL = {"yi": ("yi_ed", "yi_lf"), "phi": ("phi_ed",), "mis": ("mis_ed",),
             "gem": ("gem_ed",), "l70": ("l70_ed",)}
n_probes = len(PROBE_ED)
n_probe_fail = sum(1 for k in PROBE_ED.values() if not health[k]["pass"])
assert n_probe_fail == sum(1 for ks in PROBE_ALL.values()
                           if not all(health[k]["pass"] for k in ks)), \
    "8-K-channel and all-runs probe counts disagree"

# ------------------------------------------------------------------- gate
# The counts the frozen main text and the committed source tables state.
# Any drift aborts the build instead of silently re-rendering.
gate(
    {"n_probes": 5, "n_probes_failing": 4, "n_probes_healthy": 1,
     "gate_mode_pct": 60.0, "gate_qlike_var": 4.0,
     "pilot": [True, 3.66, 45.22, 2000, "val (no test rows read)"],
     "gemma_full_h5": [71.54, 0.9005],
     "qwen_ed_mode": [49.2, 50.3], "qwen_lf_mode": [45.5, 76.8],
     "l70_ed_mode": [40.4, 51.2],
     "yi_ed_rel": [0.37, 0.07, -0.62], "yi_lf_rel": [-0.64, -2.71, -9.86],
     "phi_ed_rel": [0.38, 0.18, -0.12],
     "mis_ed_rel": [0.27, 0.14, 0.14],
     "gem_ens3_rel": [1.84, 2.59, 2.97],
     "qwen_ed_rel": [1.21, 1.0, 0.66], "l70_ed_rel": [1.39, 1.17, 0.7],
     "lowest_max_qlike": ["Phi-4-14B", 0.677],
     "lowest_cell_qlike": ["Gemma-3-27B", 0.377, 0.378],
     "ens70_health_missing": True,
     "flags": ["4K-TRUNCATED", "PARTIAL(4K)", "AWQ-INT4"]},
    {"n_probes": n_probes, "n_probes_failing": n_probe_fail,
     "n_probes_healthy": n_probes - n_probe_fail,
     "gate_mode_pct": GATE_MODE, "gate_qlike_var": GATE_QLIKE,
     "pilot": [bool(pilot["healthy"]), round(pilot_y, 2), round(pilot_x, 2),
               int(pilot["n_docs"]), str(pilot["split"])],
     "gemma_full_h5": [round(arrow_x, 2), round(arrow_y, 4)],
     "qwen_ed_mode": [round(health["qwen_ed"]["x"].min(), 1),
                      round(health["qwen_ed"]["x"].max(), 1)],
     "qwen_lf_mode": [round(health["qwen_lf"]["x"].min(), 1),
                      round(health["qwen_lf"]["x"].max(), 1)],
     "l70_ed_mode": [round(health["l70_ed"]["x"].min(), 1),
                     round(health["l70_ed"]["x"].max(), 1)],
     "yi_ed_rel": [round(v, 2) for v in incr["yi_ed"]["y"]],
     "yi_lf_rel": [round(v, 2) for v in band_y],
     "phi_ed_rel": [round(v, 2) for v in incr["phi_ed"]["y"]],
     "mis_ed_rel": [round(v, 2) for v in incr["mis_ed"]["y"]],
     "gem_ens3_rel": [round(v, 2) for v in incr["gem_ed"]["y"]],
     "qwen_ed_rel": [round(v, 2) for v in incr["qwen_ed"]["y"]],
     "l70_ed_rel": [round(v, 2) for v in incr["l70_ed"]["y"]],
     "lowest_max_qlike": [lowest_maxq, round(probe_maxq[lowest_maxq], 3)],
     "lowest_cell_qlike": [lowest_cell_fam, round(lowest_cell_q, 3),
                           round(lowest_cell_plotted_q, 3)],
     "ens70_health_missing": ens70_health_missing,
     "flags": [str(flag_yi_lf[0]), str(flag_yi_cb[0]), str(flag_l70[0])]},
)

# ------------------------------------------------------------------ figure
# Per-run verdicts are DERIVED from the plotted columns, never typed, so the
# legend cannot drift away from the markers it labels.
def verdict(d):
    bad = []
    if d["y"].max() >= GATE_QLIKE:
        bad.append("Q")
    if d["x"].max() >= GATE_MODE:
        bad.append("M")
    return "passes" if not bad else "fails " + "+".join(bad)


LAB = {
    "qwen": ("Qwen3-32B -- anchor: 8-K %s, 10-K/Q %s"
             % (verdict(health["qwen_ed"]), verdict(health["qwen_lf"]))),
    "l70": ("Llama-3.1-70B [%s] -- 8-K only: %s"
            % (flag_l70[0], verdict(health["l70_ed"]))),
    "yi": ("Yi-1.5-34B -- 8-K %s, 10-K/Q %s"
           % (verdict(health["yi_ed"]), verdict(health["yi_lf"]))),
    "phi": "Phi-4-14B -- 8-K %s" % verdict(health["phi_ed"]),
    "mis": "Mistral-Small-24B -- 8-K %s" % verdict(health["mis_ed"]),
    "gem": "Gemma-3-27B -- 8-K %s" % verdict(health["gem_ed"]),
}

apply_style(9)
fig = plt.figure(figsize=(W, H))

ax_a = fig.add_axes(rect(0.98, 6.40, 5.32, 1.96))
ax_b = fig.add_axes(rect(0.98, 2.58, 5.32, 1.20))
ax_c = fig.add_axes(rect(0.98, 2.02, 5.32, 0.46))

XLO, XHI = 20.0, 95.0

# ------------------------------------------------- (a) the gate plane
ax_a.add_patch(plt.Rectangle((XLO, 0.30), GATE_MODE - XLO, GATE_QLIKE - 0.30,
                             color=GREEN, alpha=0.09, lw=0, zorder=0))
ax_a.axvline(GATE_MODE, color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=2)
ax_a.axhline(GATE_QLIKE, color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=2)

for key, d in health.items():
    fc = "none" if d["hollow"] else d["colour"]
    ms = 9.0 if d["marker"] == "*" else 5.2
    ax_a.plot(d["x"], d["y"], ls="none", marker=d["marker"], ms=ms,
              mfc=fc, mec=d["colour"], mew=1.0, zorder=5)

# the registered pilot point and the arrow to the full-panel h=5 row.  The
# endpoint is pinned to ONE row (gemma27_bf16, h=5) rather than mixing a modal
# share from one horizon with a QLIKE from another; the pilot was a single
# inference pass, so the single-seed full-panel row is the like-for-like
# comparator.  The pilot carries a glyph of its own -- a filled plus, which is
# no family's marker and is not a hollow copy of one -- because fill on this
# plane encodes channel: a hollow SKY diamond here would read as a Gemma
# long-form run when it is a 2,000-filing 8-K validation pass.
ax_a.plot([pilot_x], [pilot_y], ls="none", marker="P", ms=8.0, mfc=SKY,
          mec=SKY, mew=1.0, zorder=6)
ax_a.add_patch(FancyArrowPatch((pilot_x, pilot_y), (arrow_x, arrow_y),
                               connectionstyle="arc3,rad=-0.30",
                               arrowstyle="-|>", mutation_scale=9,
                               color=SKY, lw=1.0, zorder=4))
# The pilot block used to sit at (21.0, 2.75), where the tail of its last line
# ran underneath the Llama-3.1-70B h=5 marker at (40.41, 1.403) and the marker
# (zorder 5) ate the final glyph of "PASSES".  The bottom-left of the healthy
# quadrant is the only empty rectangle in this panel wide enough for three
# lines, and it is also where a PASSES verdict belongs, so the block moves down
# into it: an INWARD move, which costs the tight bounding box nothing.  annot()
# carries a white halo so a later data shift degrades legibility instead of
# destroying it.
annot(ax_a, 21.0, 0.72,
      "Gemma-3-27B registered pilot\n"
      f"{pilot['n_docs']:,} validation filings\n"
      f"{pilot_x:.1f}%, {pilot_y:.2f}  ->  PASSES",
      size=9, color=SKY, va="top", ha="left", linespacing=1.28, zorder=6)
# Same 4pt/13pt offset from the arrow head as the ax.annotate call this replaces;
# offset_copy reproduces it exactly while letting annot() add the halo, which is
# what keeps the text legible where the arrow's tail passes under it.
annot(ax_a, arrow_x, arrow_y,
      "full 8-K test panel, h=5\n"
      f"{arrow_x:.2f}%, {arrow_y:.3f}  ->  FAILS",
      size=9, color=SKY, ha="left", va="bottom", linespacing=1.28, zorder=6,
      transform=offset_copy(ax_a.transData, fig=fig, x=4 / 72.0, y=13 / 72.0,
                            units="inches"))

ax_a.set_yscale("log")
ax_a.set_xlim(XLO, XHI)
ax_a.set_ylim(0.30, 9.6)
ax_a.set_yticks([0.4, 0.6, 1.0, 2.0, 4.0, 8.0])
ax_a.set_yticklabels(["0.4", "0.6", "1.0", "2.0", "4.0", "8.0"])
ax_a.set_xticks([20, 30, 40, 50, 60, 70, 80, 90])
ax_a.set_xlabel("modal share of the 2-decimal forecast value (%);  "
                "gate M: below 60%")
ax_a.set_ylabel("standalone variance-unit QLIKE\n(log scale);  gate Q: below 4")
ax_a.text(94.2, 4.22, "gate Q = 4", fontsize=9, color=GREY, va="bottom",
          ha="right")
ax_a.text(60.9, 9.3, "gate M = 60%", fontsize=9, color=GREY, va="top",
          ha="left")
ax_a.text(59.0, 0.335, "healthy quadrant", fontsize=9, color=GREY,
          va="bottom", ha="right")

# Marker and title are one string on one baseline already, which is what panel()
# exists to guarantee, so this keeps its own call and takes panel()'s INK.  Not its
# size: panel() sets 10pt, and finish() writes with bbox_inches="tight", so every
# point added to a title widens the page box and scales every glyph in the figure
# DOWN at inclusion.  And NOT its weight, because weight does not exist in this
# environment: with font.sans-serif = Helvetica, findfont returns the same
# Helvetica.ttc for normal, medium, semibold, bold and heavy (matplotlib addresses
# face 0 of a .ttc), and the rendered width is identical at every one of them --
# measured, not assumed.  A fontweight= here would read as hierarchy in the source
# and print none, so rank is carried entirely by ink: titles at INK, every
# apparatus block at INK2, and nothing in between competing with either.
fig.text(0.30 / W, 8.44 / H,
         "(a)  The screen's own plane: both gate terms are properties of the "
         "forecast distribution", fontsize=9.8, color=INK, va="bottom",
         ha="left")

keys = [Line2D([], [], ls="none", marker="*", ms=9.0, mfc=BLUE, mec=BLUE,
               label=LAB["qwen"]),
        Line2D([], [], ls="none", marker="o", ms=5.2, mfc=GREEN, mec=GREEN,
               label=LAB["l70"]),
        Line2D([], [], ls="none", marker="s", ms=5.2, mfc=VERM, mec=VERM,
               label=LAB["yi"]),
        Line2D([], [], ls="none", marker="^", ms=5.2, mfc=YELLOW, mec=YELLOW,
               label=LAB["phi"]),
        Line2D([], [], ls="none", marker="v", ms=5.2, mfc=PURPLE, mec=PURPLE,
               label=LAB["mis"]),
        Line2D([], [], ls="none", marker="D", ms=5.2, mfc=SKY, mec=SKY,
               label=LAB["gem"]),
        Line2D([], [], ls="none", marker="P", ms=8.0, mfc=SKY, mec=SKY,
               label="Gemma-3-27B registered pilot (8-K validation)"),
        Line2D([], [], ls="none", marker="o", ms=5.2, mfc=GREY, mec=GREY,
               label="filled = 8-K (event-driven)"),
        Line2D([], [], ls="none", marker="o", ms=5.2, mfc="none", mec=GREY,
               label="hollow = 10-K/Q (long-form)")]
leg = fig.legend(handles=keys, loc="upper left",
                 bbox_to_anchor=(0.30 / W, 5.98 / H), ncol=2, handlelength=1.0,
                 labelspacing=0.22, columnspacing=1.4, borderpad=0.0,
                 handletextpad=0.5, fontsize=9)
# The last two rows are not series: they state what FILL encodes.  Recessing them
# to apparatus ink is what tells a reader that seven rows name runs and two name a
# convention -- the colour-semantics problem this legend had, fixed without
# moving a glyph.
for _t in leg.get_texts()[-2:]:
    _t.set_color(INK2)

note_a = (
    "Five cross-family probes with the primary Qwen3-32B family as anchor "
    "(own glyph, not a probe); "
    f"{n_probe_fail} of {n_probes} fail, per family: Yi-1.5-34B runs two "
    "channels and fails on both. Neither gate term reads the label or the "
    "increment, and the screen takes the MAXIMUM over a run's three "
    "horizons, so its verdict is per (family, channel) run. It "
    "is per channel too: Qwen3-32B clears it on 8-Ks "
    f"({health['qwen_ed']['x'].min():.0f}--{health['qwen_ed']['x'].max():.0f}%) "
    "and breaches it on long-form "
    f"({health['qwen_lf']['x'][0]:.1f}/{health['qwen_lf']['x'][1]:.1f}%), which "
    "is why the residual is scoped to 8-Ks. The arrow spans two "
    "splits and two document counts, so it compares gate "
    f"values, not effect sizes: QLIKE falls {pilot_y:.2f} -> {arrow_y:.3f} "
    "while modal share crosses the gate, so that failure is on modal "
    "share alone."
)
# Apparatus ink, not data ink: this block is the basis statement for panel (a) and
# used to be set in the same GREY as the axis labels and the data callouts.
#
# rule=False for all three note blocks, deliberately.  The dissertation canvas is
# the collision limit found by regen.py --search: this block's first line sits 1px
# under the legend's last row and the two lower blocks sit 1px under ax_c's axis
# label and 2px above each other, so note()'s hairline (12pt above the anchor)
# would strike through the type above rather than separate anything.  A rule in
# one of the three positions and not the others would assert a boundary the
# layout does not have, so the register change carries the whole distinction.
_t = note(fig, 0.30 / W, 5.18 / H, para(note_a, 101), rule=False, size=9)
# note() sets 1.32.  1.28 is this figure's own leading -- every callout and key in
# it is set at 1.28 -- and taking it here does three things at once: it unifies the
# figure's leading, it lifts each block's last line a fraction so the two blocks at
# the foot of the page stop touching, and because the bottom block's descender is
# what the tight bounding box rests on, it makes the page 0.8pt SHORTER rather than
# taller.  Anything above 1.30 would have grown the page and scaled the type down.
_t.set_linespacing(1.28)

# ------------------------------------- (b) the increments the screen decides on
ax_b.axvspan(XLO, GATE_MODE, color=GREEN, alpha=0.09, lw=0, zorder=0)
ax_b.axvline(GATE_MODE, color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=2)
ax_b.axhline(0.0, color=GREY, lw=0.7, zorder=2)

# Fill means the SAME thing here as in (a) -- the channel -- because the two
# panels share an x-axis and a reader carries the (a) legend down it.  The
# health verdict rides entirely on the 'x' overlay, keyed inside the axes.
for key, d in incr.items():
    ms = 9.0 if d["marker"] == "*" else 5.2
    fc = "none" if d["long_form"] else d["colour"]
    ax_b.plot(d["x"], d["y"], ls="none", marker=d["marker"], ms=ms,
              mfc=fc, mec=d["colour"], mew=1.0, zorder=5)
    if not d["healthy"]:
        ax_b.plot(d["x"], d["y"], ls="none", marker="x", ms=ms + 2.4,
                  mec=GREY, mew=0.9, zorder=7)

# Both callouts keep their exact anchors and offsets and gain annot()'s halo.  The
# Gemma one reaches back across the gate-M rule, which used to cut through the
# word "block"; a halo breaks the dashed rule around the glyphs instead of hiding
# it behind an opaque box, so the rule still reads as continuous.
gmax_i = int(np.argmax(incr["gem_ed"]["y"]))
annot(ax_b, incr["gem_ed"]["x"][gmax_i], incr["gem_ed"]["y"][gmax_i],
      f"Gemma-3-27B h={incr['gem_ed']['h'][gmax_i]}: "
      f"{incr['gem_ed']['y'][gmax_i]:+.2f}% -- the largest\n"
      "positive in the block, discarded",
      size=9, color=VERM_TXT, ha="right", va="center", linespacing=1.28,
      zorder=6, transform=offset_copy(ax_b.transData, fig=fig, x=-9 / 72.0,
                                      y=0.0, units="inches"))
annot(ax_b, incr["qwen_lf"]["x"][1], incr["qwen_lf"]["y"][1],
      "anchor on 10-K/Q:\nbreaches gate M",
      size=9, color=BLUE, ha="left", va="center", linespacing=1.28, zorder=6,
      transform=offset_copy(ax_b.transData, fig=fig, x=8 / 72.0, y=0.0,
                            units="inches"))

ax_b.set_xlim(XLO, XHI)
ax_b.set_ylim(-1.15, 3.75)
ax_b.set_yticks([-1, 0, 1, 2, 3])
ax_b.set_xticks([20, 30, 40, 50, 60, 70, 80, 90])
ax_b.set_xticklabels([])
ax_b.set_ylabel("rel. QLIKE\nimprovement over\nrecalibrated HAR (%)")
# A key, not a reading: apparatus ink, like the two encoding rows of (a)'s legend.
ax_b.text(20.9, -1.03,
          "fill = channel, as in (a)\nx = fails the health screen",
          fontsize=9, color=INK2, va="bottom", ha="left", linespacing=1.28)

fig.text(0.30 / W, 3.84 / H,
         "(b)  Same x-axis, against what the screen decides on",
         fontsize=9.8, color=INK, va="bottom", ha="left")

# ------------------------------------- (b, lower band) truncated-context rows
ax_c.axvspan(XLO, GATE_MODE, color=GREEN, alpha=0.09, lw=0, zorder=0)
ax_c.axvline(GATE_MODE, color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=2)
ax_c.axhline(0.0, color=GREY, lw=0.7, zorder=2)
ax_c.plot(band_x, band_y, ls="none", marker="s", ms=5.2, mfc="none", mec=VERM,
          mew=1.0, zorder=5)
ax_c.plot(band_x, band_y, ls="none", marker="x", ms=8.2, mec=GREY, mew=1.0,
          zorder=7)
# Same anchor and the same 9pt/1pt offset, routed through annot() like the other
# four callouts -- but this is the one that KEEPS its opaque plate, halo=False.
# It is the only callout whose crossing of the gate-M rule lands inside a printed
# number: with a halo, a dash of the (4,3) pattern sits in the gap between the 0
# and the .64 and prints as punctuation inside the value ("-0|.64", checked at
# 400 dpi).  A rule interrupted by a label is a cosmetic cost; a digit string that
# can be misread is not, so the plate stays and the rule gives way.
annot(ax_c, band_x[2], band_y[2],
      f"Yi-1.5-34B, 10-K/Q, {flag_yi_lf[0]}: "
      f"{band_y[0]:+.2f} / {band_y[1]:+.2f} / {band_y[2]:+.2f}%"
      "  -- not citable for the family claim",
      size=9, color=VERM_TXT, ha="left", va="center", zorder=8, halo=False,
      bbox=dict(fc="white", ec="none", pad=0.8),
      transform=offset_copy(ax_c.transData, fig=fig, x=9 / 72.0, y=1 / 72.0,
                            units="inches"))
ax_c.set_xlim(XLO, XHI)
ax_c.set_ylim(-12.5, 2.5)
ax_c.set_yticks([-10, -5, 0])
ax_c.set_xticks([20, 30, 40, 50, 60, 70, 80, 90])
ax_c.set_xlabel("modal share of the 2-decimal forecast value (%);  "
                "gate M: below 60%")
ax_c.set_ylabel("own scale")

note_b = (
    "Increments are volatility-unit relative QLIKE against the recalibrated "
    "HAR, the main table's convention; the gate axis in (a) is variance-unit "
    "and the two are not interchangeable. The screen discards the largest "
    "positive in the block and retains a family whose increments run "
    f"{incr['l70_ed']['y'].min():+.2f} to {incr['l70_ed']['y'].max():+.2f}%; an "
    "agreement-selecting filter would do the opposite, and the signs among the "
    "screened-out families are mixed. Ranking by QLIKE alone would invert the "
    "verdict: the lowest maximum-over-horizons variance-unit QLIKE among the "
    f"five probes is {lowest_maxq}'s ({probe_maxq[lowest_maxq]:.3f}) and the "
    f"lowest single cell over every committed probe row is {lowest_cell_fam}'s "
    f"({lowest_cell_q:.3f}; {lowest_cell_plotted_q:.3f} among the single-pass "
    "rows plotted above), both screened out on modal share alone."
)
_t = note(fig, 0.30 / W, 1.60 / H, para(note_b, 101), rule=False, size=9)
_t.set_linespacing(1.28)

foot = (
    "Basis. (a) plots single-pass rows: the Mistral-Small-24B and "
    f"Gemma-3-27B 3-run ensembles sit within {ens_drift_mode:.2f}pp of modal "
    f"share and {ens_drift_qlike:.4f} of variance-unit QLIKE of them; the 70B's "
    "ensemble rows record no health columns at all. (b) plots the basis the "
    "main table quotes: Gemma-3-27B on its 3-run ensemble, Mistral-Small-24B on "
    "bf16, the 70B single-seed, Yi-1.5-34B on its three context-clean 8-K rows "
    f"(its combined rows carry {flag_yi_cb[0]})."
)
# The basis statement was the most saturated block on the page: VERM_TXT, which in
# this palette means "attention: failures, the residual".  It is the most recessive
# thing the figure says, and it now reads that way.  Vermillion is left carrying
# exactly one meaning here -- a reading the screen discards (the Gemma callout and
# the truncated-context band).
#
# It is indented 0.14 in from the flush-left edge every other block shares.  With
# both blocks in the same register there is no colour break between them any more,
# and the leading between this block's first line and the note above it is nearly
# all the canvas has left to give, so the indent is what tells the reader a new,
# subordinate block starts here.  Indenting moves the block INWARD, so the tight
# bounding box does not notice.
_t = note(fig, 0.44 / W, 0.40 / H, para(foot, 101), rule=False, size=9)
_t.set_linespacing(1.28)

finish(fig, "F12_health_screen_orthogonality")
