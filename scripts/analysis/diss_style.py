"""Dissertation house style — a thin wrapper over `supp_style`.

The Technical Supplement's sixteen figures were drawn for a single-column
supplement page and are included there at their natural size.  The dissertation
is an A4 book at 11pt with 25 mm side margins and 1.5 line spacing, so its text
block is 160 mm x 252 mm (455.24 pt x 717.01 pt).  A supplement figure dropped
into that block at `width=\\textwidth` renders 596-645 pt tall; add the
dissertation's 6-9 line caption and every one of them overflows the page, which
is what the eight "Float too large for page" warnings were reporting.

This module changes two things and nothing else:

* `OUTDIR` points at `writing/dissertation/figures/` instead of the supplement's
  figure directory, so regenerating a dissertation figure can never overwrite the
  supplement's copy;
* `finish()` measures the written PDF, converts its page box into the height the
  float will actually occupy at `width=\\textwidth`, and aborts if that exceeds
  the dissertation-safe cap.  A figure that silently reintroduces a page overflow
  is exactly the failure this module exists to prevent.

Palette, rcParams and the evidence `gate()` are re-exported unchanged from
`supp_style`: the two documents must remain visually one system, and no figure
may be redrawn against drifted evidence in either.

Usage
-----
    import diss_style as ds

    ds.gate({"n_primary": 38}, {"n_primary": int(df.primary_holm.sum())})
    ds.apply_style(9)
    fig = plt.figure(figsize=ds.canvas(6.55))
    ...
    ds.finish(fig, "F4_ladder_cell_matrix")

Geometry contract
-----------------
    text width           455.24411 pt  (160 mm)
    text height          717.00946 pt  (252 mm)
    canvas width         6.10 in       (= 439.2 pt, renders at 1.036x)
    canvas height cap    7.70 in       (~195 mm, 0.80 of the text height)
    rendered height cap  573.61 pt     (0.80 of the text height)

`MAX_RENDER_PT` is the number that matters.  Figures that cannot reach it without
crushing their annotation blocks pass an explicit, larger `max_render_pt`; the
value used is recorded per figure in the regeneration manifest.

Two caveats the numbers above hide.  The canvas *width* is a default, not a law:
several of these generators wrap their note blocks against a character count or
an absolute measure tuned to the generator's own width, and forcing 6.10 in
pushes those blocks past the canvas edge, whereupon the tight bounding box grows
back to the original width and nothing has been gained.  `regen.py` therefore
tries both widths per figure and keeps whichever makes the shorter float.  And
the dissertation includes every figure as

    \\includegraphics[width=\\textwidth,height=0.83\\textheight,keepaspectratio]

so the page, not the generator, has the last word on height: 0.83 x 717.01 =
595.1 pt of graphic plus the longest caption in the report still clears the text
block.  A figure that comes out taller than that is scaled down at inclusion
rather than allowed to overflow.
"""
import json
import os
import subprocess
import sys

import supp_style as _supp
from supp_style import (  # noqa: F401  (re-exported: the two documents share one palette)
    AGG,
    BLUE,
    GREEN,
    GREEN_TXT,
    GREY,
    INK,
    INK2,
    LIGHT,
    PURPLE,
    REPO,
    RULE,
    SKY,
    TAB,
    VERM,
    VERM_TXT,
    YELLOW,
    annot,
    note,
    panel,
    apply_style,
    gate,
    shutil_which_gs,
)

# ------------------------------------------------------------------ geometry
PT_PER_IN = 72.0
TEXTWIDTH_PT = 455.24411          # \the\textwidth  (a4paper, left=right=25mm)
TEXTHEIGHT_PT = 717.00946         # \the\textheight (top=25mm, bottom=20mm)

DISS_W = 6.10                     # canvas width, inches
MAX_H = 7.70                      # canvas height cap, inches (~195 mm)
MAX_RENDER_PT = 0.80 * TEXTHEIGHT_PT     # 573.61 pt of page, graphic only

OUTDIR = os.path.join(REPO, "writing", "dissertation", "figures")
MANIFEST = os.path.join(OUTDIR, "_geometry_manifest.json")

# Per-figure escape hatches, filled in by the regeneration driver.  A figure only
# appears in CAP_OVERRIDE when its in-figure annotation blocks (which are set in
# absolute points and therefore do not shrink with the canvas) stop the canvas
# from reaching MAX_RENDER_PT without text collisions.
CAP_OVERRIDE = {}
NOTE_OVERRIDE = {}


# Filled by `finish()`: figure name -> sorted list of overlapping text-pair keys.
# The regeneration driver compares a compressed canvas against the generator's
# own uncompressed rendering and refuses any canvas that introduces a new pair.
# Shrinking a canvas moves axes but not the type set on them, so a collision
# check is the only honest way to know how far a figure can be compressed.
COLLISIONS = {}


def _text_boxes(fig):
    """Every visible piece of type in the figure, with its rendered box."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []

    def add(artist, owner):
        if artist is None or not artist.get_visible():
            return
        s = artist.get_text()
        if not s.strip():
            return
        try:
            bb = artist.get_window_extent(renderer=r)
        except (RuntimeError, ValueError, AttributeError):
            return
        if bb.width <= 0 or bb.height <= 0:
            return
        items.append((owner, " ".join(s.split())[:48], bb))

    for t in fig.texts:
        add(t, "fig")
    for i, ax in enumerate(fig.axes):
        owner = f"ax{i}"
        for t in ax.texts:
            add(t, owner)
        add(ax.title, owner)
        add(ax.xaxis.label, owner)
        add(ax.yaxis.label, owner)
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            add(t, owner)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                add(t, owner)
    for lg in fig.legends:
        for t in lg.get_texts():
            add(t, "fig")
    return items


def collision_pairs(fig, min_overlap=0.06):
    """Keys of every pair of text boxes that overlap by more than a sliver.

    A panel's own data area is added as a pseudo-box so that type driven into a
    *neighbouring* panel by a shorter canvas is caught as well; type over its own
    panel is how these figures annotate, so same-owner pairs are skipped there.
    """
    items = _text_boxes(fig)
    for i, ax in enumerate(fig.axes):
        items.append((f"ax{i}", f"<panel {i} plotting area>", ax.bbox))
    pairs = set()
    for i in range(len(items)):
        oi, si, bi = items[i]
        for j in range(i + 1, len(items)):
            oj, sj, bj = items[j]
            panel = si.startswith("<panel") or sj.startswith("<panel")
            if panel and (oi == oj or (si.startswith("<panel")
                                       and sj.startswith("<panel"))):
                continue
            w = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            h = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if w <= 0 or h <= 0:
                continue
            area = w * h
            smallest = min(bi.width * bi.height, bj.width * bj.height)
            if smallest > 0 and area / smallest >= min_overlap:
                a, b = sorted([f"{oi}|{si}", f"{oj}|{sj}"])
                pairs.add(f"{a}  <>  {b}")
    return sorted(pairs)


def canvas(height_in, width_in=DISS_W, max_h=MAX_H):
    """Return a `figsize` clipped to the dissertation-safe canvas.

    Width is fixed rather than negotiated: every dissertation figure is included
    at `width=\\textwidth`, so a common canvas width is what makes the rendered
    text size common across figures too.
    """
    if height_in > max_h:
        height_in = max_h
    return (width_in, height_in)


def page_size_pt(path):
    """(width, height) of a PDF's page box, in points."""
    out = subprocess.run(["pdfinfo", path], capture_output=True, text=True,
                         check=True).stdout
    for line in out.splitlines():
        if line.startswith("Page size:"):
            body = line.split(":", 1)[1].strip()
            w, _, h = body.split()[:3]
            return float(w), float(h)
    sys.exit(f"could not read a page size out of {path}")


def rendered_height_pt(w_pt, h_pt):
    """Height the graphic occupies at `width=\\textwidth`, in points."""
    return TEXTWIDTH_PT * h_pt / w_pt


def inclusion_scale(w_pt):
    """Factor applied to every glyph when the PDF is set to `\\textwidth`."""
    return TEXTWIDTH_PT / w_pt


def finish(fig, name, tight=True, max_render_pt=None, note=""):
    """Write `name`.pdf into the dissertation figure dir with fonts outlined.

    Mirrors `supp_style.finish` (tight box, ghostscript `-dNoOutputFonts`,
    zero-embedded-font gate) and then adds the dissertation's own gate: the
    height the float will occupy on an A4 page must not exceed `max_render_pt`.
    """
    if max_render_pt is None:
        # Resolved at call time, not at definition time, so the regeneration
        # driver can raise the cap for a named figure without editing its
        # generator's `finish(...)` call.
        max_render_pt = CAP_OVERRIDE.get(name, MAX_RENDER_PT)
    note = note or NOTE_OVERRIDE.get(name, "")
    COLLISIONS[name] = collision_pairs(fig)
    os.makedirs(OUTDIR, exist_ok=True)
    raw = os.path.join(OUTDIR, f"_{name}_raw.pdf")
    out = os.path.join(OUTDIR, f"{name}.pdf")
    if tight:
        fig.savefig(raw, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(raw)
    import matplotlib.pyplot as plt
    plt.close(fig)

    gs = shutil_which_gs()
    subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
                    "-dNoOutputFonts", "-o", out, raw], check=True)
    os.remove(raw)

    fonts = subprocess.run(["pdffonts", out], capture_output=True, text=True)
    n = len([ln for ln in fonts.stdout.splitlines()[2:] if ln.strip()])
    if n:
        sys.exit(f"FONT GATE FAIL — {out} still embeds {n} font(s)")

    w_pt, h_pt = page_size_pt(out)
    render = rendered_height_pt(w_pt, h_pt)
    scale = inclusion_scale(w_pt)
    record = {"figure": name, "page_w_pt": round(w_pt, 2),
              "page_h_pt": round(h_pt, 2),
              "rendered_h_pt": round(render, 2),
              "rendered_h_frac_textheight": round(render / TEXTHEIGHT_PT, 4),
              "inclusion_scale": round(scale, 4),
              "cap_pt": round(max_render_pt, 2),
              "fonts_embedded": 0, "note": note}
    _write_manifest(record)

    print(f"wrote {os.path.relpath(out, REPO)} "
          f"(0 fonts; page {w_pt:.1f}x{h_pt:.1f} pt; "
          f"renders {render:.1f} pt = {render / TEXTHEIGHT_PT:.3f} textheight; "
          f"glyph scale {scale:.3f}x)")

    if render > max_render_pt:
        sys.exit(f"HEIGHT GATE FAIL — {name} renders {render:.1f} pt at "
                 f"\\textwidth, over the {max_render_pt:.1f} pt cap "
                 f"({render - max_render_pt:.1f} pt too tall)")


def _write_manifest(record):
    data = {}
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST) as fh:
                data = json.load(fh)
        except (ValueError, OSError):
            data = {}
    data[record["figure"]] = record
    with open(MANIFEST, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


# `supp_style` is imported for its side-effect-free constants only; the name is
# kept bound so `diss_style` can be substituted for it in `sys.modules` by the
# regeneration driver without losing access to the original module.
_ = _supp
