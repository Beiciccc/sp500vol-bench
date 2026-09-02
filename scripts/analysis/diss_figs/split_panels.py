"""Split a composite dissertation figure into its panels, without redrawing it.

Why cropping and not re-plotting
--------------------------------
The eleven main-text figures are the Technical Supplement's own gated
generators; `regen.py` deliberately does not fork them, so that every
`gate(...)` evidence check runs exactly as the supplement runs it.  Splitting a
figure by restructuring its GridSpec would fork them.

Cropping does not.  A crop changes only the page box: the drawing is untouched,
so a panel carries the identical vector content it carries inside the composite.
And because a crop keeps the *width*, the panel is included at the same
`width=\\textwidth` scale as the composite was -- 5.92in / 6.4in = 0.925 -- so
every printed point size is bit-identical to what the composite printed.  That
is the property that makes this safe: splitting cannot shrink type.

What it costs: a text block drawn by the generator cannot be divided.  A note
that spans the cut belongs wholly to one side, and the caption on the other side
has to carry the fact instead.  `--bands` reports where the cuts can go; the
caller decides which band is a seam and which is a gap inside a panel.

The other cost is size.  A crop narrows the page box but leaves the content
stream whole, so each of a figure's two parts weighs what the composite weighed.
Re-emitting the parts through Ghostscript drops what the box excludes and halves
them, and it was tried: the page boxes come back the right size, but the
rendering does not survive the round trip.  Downsampled 8x, where antialiasing
should have averaged away, the re-emitted parts still differ from the originals
by up to 23 grey levels over 0.06% of the image -- something in the paths or the
stroke widths is being re-rounded.  These figures are gate-checked evidence, so
the drawing is kept bit-exact and the file size is paid.

Usage
-----
    python3 split_panels.py --bands F16_maec_reproduce_reprice
    python3 split_panels.py --cut F16_maec_reproduce_reprice 330.5
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pypdf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIGDIR = os.path.join(ROOT, "writing", "dissertation", "figures")

# The composites are included at width=\textwidth.  Every geometry number this
# script prints in "printed points" is a canvas measure times this factor.
TEXTWIDTH_PT = 455.24411


# Adjudicated seams, canvas points from the top of each figure.  One cut each:
# every figure divides in two.  F4_ladder_cell_matrix is absent on purpose --
# it is a two-column figure whose right column runs the full height, so it has
# no horizontal seam, and its two columns share one scale, which is the
# comparison the figure exists to make.
SPLITS = {
    "F1_membership_panel": 326,
    "F3_standalone_180_and_reference_strength": 316,
    "F16_maec_reproduce_reprice": 224,
    # 228.5 is the paragraph break inside the note block under panel (a):
    # the 3.1pt gap between (a)'s own "Filled: ED B2 ..." note and the six-line
    # block that explains (b).  Cutting at the 299pt seam below that block left
    # (b)'s row-code key and ring definition stranded in part a.
    "F14_economic_adjudication": 228.5,
    # F12 is deliberately absent.  Its panel-(a) note block abuts the "(b) Same
    # x-axis" heading with no whitespace between them at any canvas height the
    # generator will draw, so every candidate seam either strands that heading at
    # the foot of part a or divides the note itself.  Its two panels are coupled
    # anyway -- (b) is annotated "fill = channel, as in (a)" and shares (a)'s
    # marker key -- so it stays whole, regenerated at 8.40in to clear the
    # overprint and included at 0.90\textwidth to keep its former footprint.
    # 228.4, not the wider 299.5 seam: 299.5 falls below the "(b) Four arms ..."
    # heading, which would strand (b)'s own caption line at the foot of part a.
    "F15_yelp_portability": 260.9,
    "F7_anonymisation_price": 422,
    "F8_matched_firm_swap": 273,
    "F9_power_mde_label_noise": 187,
    # F11 adjudicated by hand: the only legal seam that separates whole blocks.
    # Its note strips are full-width light grey, so an all-white seam does not
    # exist; the cut sits in the white gap above the rule that opens block (c).
    "F11_ladder_perturbations": 234.4,
}


def _raster(path, dpi):
    """Render to a greyscale array, rows top-down, plus points-per-pixel."""
    out = subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-gray", "-singlefile", "-png", path,
         "/tmp/_splitpanels"], capture_output=True)
    if out.returncode != 0:
        sys.exit(f"pdftoppm failed on {path}: {out.stderr.decode()[:200]}")
    from PIL import Image
    a = np.asarray(Image.open("/tmp/_splitpanels.png").convert("L"))
    return a, 72.0 / dpi


def bands(path, dpi=200, min_gap_pt=3.0):
    """Horizontal all-white runs, as (top_pt, bottom_pt, height_pt) from the top.

    A seam candidate is a run of rows that carry no ink at all.  Runs shorter
    than `min_gap_pt` are inter-line leading, not panel seams, and are dropped.
    """
    a, ppp = _raster(path, dpi)
    ink = (a < 250).any(axis=1)          # a row carries ink anywhere
    out, run = [], None
    for i, has in enumerate(ink):
        if not has and run is None:
            run = i
        elif has and run is not None:
            out.append((run, i))
            run = None
    if run is not None:
        out.append((run, len(ink)))
    res = []
    for lo, hi in out:
        h = (hi - lo) * ppp
        if h >= min_gap_pt:
            res.append((lo * ppp, hi * ppp, h))
    return res, a.shape[0] * ppp


def cut(name, ys, outnames=None, dry=False):
    """Write one PDF per band between the cut heights `ys` (pt from the top).

    The page box is narrowed in y only.  Width, and therefore the include scale
    and every printed point size, is untouched.
    """
    src = os.path.join(FIGDIR, f"{name}.pdf")
    r = pypdf.PdfReader(src)
    if len(r.pages) != 1:
        sys.exit(f"{name}: expected 1 page, found {len(r.pages)}")
    box = r.pages[0].mediabox
    x0, y0 = float(box.left), float(box.bottom)
    W, H = float(box.width), float(box.height)
    edges = [0.0] + list(ys) + [H]
    made = []
    for i in range(len(edges) - 1):
        top, bot = edges[i], edges[i + 1]          # from the top, downwards
        nm = (outnames[i] if outnames else f"{name}__p{i + 1}")
        # PDF y grows upward: a band [top, bot] from the top is
        # [H - bot, H - top] from the bottom.
        lo, hi = y0 + (H - bot), y0 + (H - top)
        made.append((nm, bot - top, (bot - top) * TEXTWIDTH_PT / W))
        if dry:
            continue
        w = pypdf.PdfWriter()
        p = pypdf.PdfReader(src).pages[0]
        for b in (p.mediabox, p.cropbox):
            b.lower_left, b.upper_right = (x0, lo), (x0 + W, hi)
        w.add_page(p)
        with open(os.path.join(FIGDIR, f"{nm}.pdf"), "wb") as fh:
            w.write(fh)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--all", action="store_true",
                    help="cut every figure in SPLITS into its two parts")
    ap.add_argument("--bands", action="store_true")
    ap.add_argument("--cut", nargs="+", type=float, default=None,
                    help="cut heights in canvas points from the top")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    if a.all:
        for n, y in sorted(SPLITS.items()):
            for nm, hpt, pr in cut(n, [y], [f"{n}__a", f"{n}__b"], dry=a.dry):
                print(f"  {nm:<46}{pr:6.1f}pt printed "
                      f"({pr / 717.0 * 100:3.0f}% of block)")
        return
    if not a.name:
        ap.error("give a figure name, or --all")
    src = os.path.join(FIGDIR, f"{a.name}.pdf")

    if a.bands:
        bs, H = bands(src)
        r = pypdf.PdfReader(src)
        W = float(r.pages[0].mediabox.width)
        k = TEXTWIDTH_PT / W
        print(f"{a.name}: canvas {W:.1f} x {H:.1f}pt, include scale {k:.4f}")
        print(f"  whole figure prints {H * k:.1f}pt "
              f"({H * k / 717.0 * 100:.0f}% of the 717pt text block)")
        print(f"  {'band top':>9} {'bottom':>8} {'gap':>6}   cut here -> "
              f"upper piece prints")
        for lo, hi, h in bs:
            mid = (lo + hi) / 2
            print(f"  {lo:9.1f} {hi:8.1f} {h:6.1f}   {mid:7.1f}pt "
                  f"-> {mid * k:6.1f}pt ({mid * k / 717.0 * 100:3.0f}%)")
        return

    if a.cut:
        made = cut(a.name, a.cut, a.names, dry=a.dry)
        for nm, hpt, pr in made:
            print(f"  {nm:<44} {hpt:7.1f}pt canvas -> {pr:6.1f}pt printed "
                  f"({pr / 717.0 * 100:3.0f}% of block)")
        return
    ap.error("give --bands or --cut")


if __name__ == "__main__":
    main()
