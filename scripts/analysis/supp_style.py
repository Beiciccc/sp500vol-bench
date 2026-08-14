"""Shared house style for Technical Supplement figures.

Every supplement figure imports this so the whole set reads as one system and
matches the main paper's two figures (Okabe-Ito colour-blind-safe palette,
Helvetica-family sans, >= 9pt text at 1:1 inclusion scale, hairline axes).

Usage
-----
    from supp_style import (BLUE, VERM, GREY, SKY, GREEN, YELLOW, PURPLE,
                            apply_style, finish, gate)

    gate({"n_primary": 38}, {"n_primary": int(df.primary_holm.sum())})
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ...
    finish(fig, "fig_name")     # writes the outlined PDF into the supplement

Design rules this module enforces
---------------------------------
* Colour is never the only channel: callers must also vary hatch, marker, or
  label. The palette is Okabe-Ito-derived and is CHECKED, not asserted: run
      node <dataviz skill>/scripts/validate_palette.js "$(python3 -c 'import supp_style as s;
      print(",".join([s.BLUE,s.SKY,s.VERM,s.GREEN,s.YELLOW,s.PURPLE]))')" --mode light
  All six checks pass. SKY carries a contrast WARN at 2.25:1, which this figure
  set discharges the way the validator requires: every series is also carried by
  marker shape or a direct label, never by colour alone.
* Fonts are outlined by ghostscript (-dNoOutputFonts) on write, so the shipped
  PDFs embed no fonts at all -- the supplement build's font gate then passes
  and no glyph can be substituted on a reviewer's machine.
* `gate()` aborts the build if a source table's counts have drifted from the
  values the paper states. A figure that silently re-renders against changed
  evidence is worse than no figure.
"""
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TAB = os.path.join(REPO, "results", "tables")
AGG = os.path.join(REPO, "release", "aggregate_results")
OUTDIR = os.path.join(REPO, "writing", "paper", "supplementary", "figures")

# ------------------------------------------------------------------ palette
BLUE = "#0072B2"     # primary series / survivors
SKY = "#56B4E9"      # secondary series, different denominator
#   Was #3B8FC4 until the palette was actually validated rather than assumed:
#   against BLUE it scored deltaE 9.3 for NORMAL vision, below the 15 floor, i.e.
#   hard to tell apart even with full colour vision. Okabe-Ito's own sky blue
#   scores 20.4. SKY and BLUE co-occur in 32 of the 33 generators, so the old
#   value was a document-wide defect.
VERM = "#C85800"     # attention: injected signal, failures, the residual
VERM_TXT = "#A34700"  # vermillion for text (6.07:1 on white)
GREEN = "#009E73"    # "survives" / healthy instrument (marks only: 3.42:1)
GREEN_TXT = "#00734F"  # green for TEXT (5.89:1 on white)
#   GREEN is a mark colour and fails as body type: 3.42:1, under the 4.5:1 a
#   9 pt line needs. The module already had VERM_TXT for exactly this split;
#   green lacked its counterpart, and a redraw moved a panel statement from
#   BLUE (5.19:1) to GREEN and lost the contrast without anything catching it.
YELLOW = "#B87E00"   # third series
#   Was #E69F00, which sits at 2.19:1 on white -- below the 3:1 this module's
#   own docstring claimed for every entry. Darkened until it clears.
PURPLE = "#8055A6"   # fourth series (darkened Okabe-Ito reddish purple)
GREY = "#3A3A3A"     # rules, annotations, dead instruments

# ------------------------------------------------- what a hue does and does not mean
# THE PALETTE IS PANEL-LOCAL, NOT DOCUMENT-GLOBAL, and that is a design decision
# rather than an oversight. Two of the seven slots say so outright: YELLOW is
# "third series" and PURPLE is "fourth series" -- positional, with no semantics
# attached. A generator needing a third category takes YELLOW, so YELLOW means
# h=10 in one panel and block B in another.
#
# This gets raised as a defect ("colour carries no shared semantics") and it was
# audited in full: every hue's legend labels were extracted from every generator
# by AST -- only calls carrying BOTH a colour kwarg and a literal label= -- and
# every apparent within-figure collision turned out to be a SECOND CHANNEL with
# its own declared key:
#     F15   BLUE filled = significant, BLUE hollow = not; fill carries it
#     F12   GREY filled = 8-K, GREY hollow = 10-K/Q; fill carries it
#     AF3   GREY dot vs GREY hatched; texture carries it
#     F9    BLUE for both channels; marker shape carries it
#     F2    GREY with and without error bars; the bars carry it
#     F13   the hatched bar's label is a reproduction share, and note_c prints
#           the bar heights and the shares side by side, naming the second
# So the rule that actually governs is the one at the top of this file: colour is
# never the only channel. Uniformity of hue across figures is NOT the rule, and
# imposing it would be worse -- F9 needs VERM for estimator noise and F12 needs
# its hues for models, so repainting either to match F8's channel colours would
# create a real same-figure collision to remove an imagined cross-figure one.
#
# What IS a defect, and the one the audit found: a legend key whose hue appears
# nowhere on the page. AF1's third key was a GREY open circle while the page's
# open markers are BLUE circles and VERM squares. Check keys against marks, not
# hues against other figures.
LIGHT = "#D9D9D9"    # inactive background bars


def apply_style(base_size=9):
    """Set the rcParams every supplement figure shares."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": base_size,
        "axes.linewidth": 0.6,
        "axes.edgecolor": GREY,
        "axes.labelcolor": GREY,
        "text.color": GREY,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "legend.fontsize": base_size,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
    })


def gate(expected, got):
    """Abort unless the source tables still hold the values the paper states.

    expected/got are dicts of the same keys; any mismatch exits non-zero with
    the drift printed, so a stale evidence table can never be plotted silently.
    """
    if expected != got:
        diff = {k: (expected[k], got.get(k)) for k in expected
                if expected[k] != got.get(k)}
        sys.exit(f"GATE FAIL — evidence drifted from the paper: {diff}")



# ---------------------------------------------------------------- hierarchy
# These figures were legible but undesigned: every element drew at one 9pt size,
# so panel titles, data labels and the long apparatus notes carried equal weight
# and the set read as analysis output rather than as figures.
#
# WEIGHT IS NOT AVAILABLE ON THIS BUILD, and that is measured, not assumed:
# font.sans-serif puts Helvetica first, matplotlib resolves every weight to the
# same Helvetica.ttc face, and "(a) Panel title" renders BYTE-IDENTICAL at
# normal, semibold and bold -- 461 ink pixels in all three. So every
# fontweight="bold" already in these generators has been doing nothing, and INK
# is deliberately == GREY, which makes a GREY -> INK edit a no-op too.
# Arial DOES carry a real bold (238 -> 336 ink pixels), so weight can be bought
# per-call with family="Arial", fontweight="bold" -- but bold Arial is WIDER than
# regular Helvetica, and under bbox_inches="tight" that widens the page and
# scales the whole figure down. It cost F3 a collapsed label gap and a 10.8 pt
# overflow before it was reverted. Buy weight only where the figure has measured
# horizontal room.
#
# SO HIERARCHY HERE IS COLOUR AND RULES, AND NEVER SIZE, and
# that is a hard constraint rather than a preference. finish() writes with
# bbox_inches="tight", so the PDF page is the CONTENT's bounding box; enlarging
# any text widens that box, which forces a harder down-scale on the page, which
# LOWERS the printed point size and can push a figure past its float cap. Raising
# a title from 9pt to 11pt can therefore make every glyph in the figure smaller.
# Weight and colour change no geometry at all.
INK = GREY            # primary: titles, data labels, axis labels
INK2 = "#5A5A5A"      # secondary: apparatus notes, basis statements (7.0:1 on white)
RULE = "#C8C8C8"      # hairline separating apparatus from data


def panel(ax, letter, title=None, dy=1.10, dx=0.0):
    """Draw a panel marker, and its title on the same baseline.

    One call replaces the two-or-three-call idiom the generators grew
    independently, where the marker and a wrapped title were placed separately
    and drifted out of alignment.
    """
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=10,
            fontweight="semibold", color=INK, ha="left", va="bottom")
    if title:
        ax.text(dx + 0.038, dy, title, transform=ax.transAxes, fontsize=10,
                fontweight="semibold", color=INK, ha="left", va="bottom")


def note(fig, x, y, text, width=None, rule=True, ha="left", va="top", size=9):
    """An apparatus note: recessive grey, optional hairline above it.

    The rule and the lighter ink are what let a reader tell at a glance which
    text is the figure's argument and which is its basis statement, without
    either one shrinking. Every word of these notes is load-bearing -- they carry
    the row support, the denominator and the does-not-show clause -- so nothing
    here abbreviates them.
    """
    if rule:
        w = width if width is not None else 0.42
        fig.lines.append(plt.Line2D([x, x + w], [y + 0.012, y + 0.012],
                                    transform=fig.transFigure, color=RULE,
                                    linewidth=0.5, zorder=0.5))
    return fig.text(x, y, text, fontsize=size, color=INK2, ha=ha, va=va,
                    linespacing=1.32)


def annot(ax, x, y, text, size=9, color=None, halo=True, **kw):
    """An in-axes callout that survives being drawn over data.

    Several panels put multi-line explanation inside the plot area where it
    collided with marks. A white halo is cheaper than moving the block out and
    reflowing the canvas, and unlike a filled bbox it does not hide the data
    underneath.
    """
    t = ax.text(x, y, text, fontsize=size, color=color or INK, **kw)
    if halo:
        import matplotlib.patheffects as pe
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    return t

def finish(fig, name, tight=True):
    """Write `name`.pdf into the supplement figure dir with fonts outlined."""
    os.makedirs(OUTDIR, exist_ok=True)
    raw = os.path.join(OUTDIR, f"_{name}_raw.pdf")
    out = os.path.join(OUTDIR, f"{name}.pdf")
    if tight:
        fig.savefig(raw, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(raw)
    plt.close(fig)

    gs = shutil_which_gs()
    subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
                    "-dNoOutputFonts", "-o", out, raw], check=True)
    os.remove(raw)

    fonts = subprocess.run(["pdffonts", out], check=False, capture_output=True, text=True)
    n = len([ln for ln in fonts.stdout.splitlines()[2:] if ln.strip()])
    if n:
        sys.exit(f"FONT GATE FAIL — {out} still embeds {n} font(s)")
    print(f"wrote {os.path.relpath(out, REPO)} (0 fonts embedded)")


def shutil_which_gs():
    import shutil
    for cand in ("gs", "gswin64c"):
        p = shutil.which(cand)
        if p:
            return p
    sys.exit("ghostscript not found — required to outline figure fonts")
