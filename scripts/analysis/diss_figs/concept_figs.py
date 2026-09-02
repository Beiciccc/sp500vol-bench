"""The dissertation's concept diagrams: schematics, not evidence plots.

Why these are separate from `regen.py`
--------------------------------------
Every other figure in this report is a Technical Supplement generator run
through `regen.py`, which rewrites the canvas and nothing else so that each
generator's `gate(...)` evidence checks run exactly as the supplement runs them.
The figures here are not that.  They carry no series, no axes and no computed
quantity: they are drawings of an argument, made because Chapters 1 and 2 ran
for fifteen consecutive pages without a figure and the reader was asked to hold
a four-rung ladder, and a two-route failure mode, entirely in the head.

The discipline that replaces the evidence gate is provenance.  A schematic can
still lie, and the way it would lie here is by carrying a number that is not in
the report.  So every number drawn below is quoted from the chapters, its
source named in a `# src:` comment beside it in the same form the chapters use,
and `--check` reads those numbers back out of the compiled report and fails if
any of them cannot be found there.

Geometry.  Drawn at `diss_style.DISS_W` (6.1 in) like every other dissertation
figure, so the inclusion scale, and therefore the printed type size, matches
them: 9 pt drawn prints at 9.33 pt.  These are deliberately wide and short --
the main body is capped at sixty pages and sits at sixty, so each diagram has to
earn its height from the prose it makes unnecessary.

Craft, and why it is spelled out
--------------------------------
The first version of these two was drawn by eye and looked it: saturated borders
on boxes the size of a stamp, arrowheads landing on the borders they pointed at,
labels printed straight over the rules they labelled, and card widths guessed --
three of the four rungs shipped with a line overhanging the border.  The rules
below are what replaced the guessing, and each one is a defect that actually
happened:

* A card belongs to a colour family rather than wearing one.  Fill is a 4-10%
  tint of the accent, the hairline a 40% tint; the accent itself is spent on the
  tag alone.  `tint()` is the whole mechanism.
* Nothing is positioned by estimate.  `text_w()` measures, the card and the
  prologue assert against the space they have, and a string that outgrows its
  box stops the build instead of printing over the border.
* Connectors stop short of what they point at (`gap_end`), turn on an arc rather
  than a mitre (`elbow`), and carry their labels inside a hole in the line
  (`spine`, and the white-bbox text in `ladder`).
* The two junctions of the ladder are drawn to the same measurements -- same
  stub, same wedge, same radius -- because asymmetry between them was the
  loudest thing left after the first pass.
* Colour keeps one meaning, and the two figures share it.  Vermillion is
  identity and failure -- Route I, the R2 card, the standalone zero and the
  conjunction's zero -- and nothing else.  The four "detects" lines were drawn
  in it once; that made vermillion mean apparatus as well, and inside R1 a
  vermillion line sat under a blue 38 and read as bad news for a moment.  They
  are grey, which is what the palette already calls apparatus.  Blue ink is reserved for what text genuinely contributed:
  the surviving counts here, the content route there.  R3 is therefore drawn in
  purple rather than blue: it is a price-side reference, and a blue card would
  have made blue mean the price side in one figure and the filing's own content
  in the other.  R4 is the only card whose fill and hairline are both stepped
  darker, so its tone reads as an endpoint rather than as an inconsistency.

Usage
-----
    python3 concept_figs.py            # draw all
    python3 concept_figs.py ladder     # one of them
    python3 concept_figs.py --check    # verify every drawn number is in the report
"""
import argparse
import os
import re
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ANALYSIS)

import diss_style as ds  # noqa: E402

# ---------------------------------------------------------------- primitives
#
# A schematic vocabulary small enough that every diagram below is built from the
# same five moves, which is what keeps them looking like one another and like
# the rest of the report.


# The smallest type the report prints anywhere is 8.09 pt, in F8.  A schematic
# is the easiest place to undercut that without noticing, because a diagram will
# happily hold 6 pt annotations that look fine on screen and are unreadable on
# paper.  Nothing below is drawn smaller than this.
MIN_PT = 8.0


def frame(w_in, h_in, xmax=None, ymax=None):
    """A blank figure whose axes are the drawing surface, in points of canvas.

    Working in canvas points rather than data units means a position written
    once means the same thing after the canvas is retuned to fit the page
    budget, and it makes the type-size floor checkable: a font size in points is
    a font size in these units.
    """
    ds.apply_style(base_size=9)
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xmax if xmax is not None else w_in * 72)
    ax.set_ylim(0, ymax if ymax is not None else h_in * 72)
    ax.axis("off")
    return fig, ax


def text_w(fig, s, size, bold=False):
    """Rendered width of a string, in points of canvas.

    Card widths in the first draft were guessed, and three of the four cards
    shipped with a line overhanging the border.  Measuring costs one throwaway
    text artist and turns a silent defect into an arithmetic one.
    """
    t = fig.text(0, 0, s, fontsize=size,
                 fontweight="bold" if bold else "normal")
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width * 72.0 / fig.dpi


def tint(hex_colour, f):
    """Blend a palette colour towards white by 1-f.

    The two schematics need a colour family per card without the loud saturated
    borders a first draft reaches for.  A 5% tint reads as membership; a 40%
    tint is a hairline that belongs to the same family as the tag above it.
    """
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(1.0 - (1.0 - c) * f for c in (r, g, b))


def tint_to_L(hex_colour, target_L):
    """A tint of `hex_colour` whose relative luminance is `target_L`.

    `tint(accent, f)` blends by a fixed fraction, which gives every card the
    same alpha but NOT the same ink on paper: the accents differ in luminance,
    so at f=0.055 the vermillion card printed a 3.5% K tint and the purple one
    3.1%, against the neutral card's 5.5%.  A monochrome laser holds those
    unequally, and the accented cards came out lighter than the plain one, which
    inverts what the tint is for.  Solving for luminance instead makes every
    card drop the same amount of ink and keeps its hue.
    """
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        v = tint(hex_colour, mid)
        L = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
        if L > target_L:
            lo = mid
        else:
            hi = mid
    return tint(hex_colour, (lo + hi) / 2)


# Card fills, as luminance rather than as alpha.  0.945 is a 5.5% ink drop, the
# lightest a 600 dpi laser holds reliably; R4 sits at 0.898 so its step from the
# others is two halftone levels rather than one.
FILL_L, FILL_L_END = 0.945, 0.898


def t(ax, x, y, s, size=MIN_PT, colour=None, weight=None, ha="left",
      va="baseline", style=None, ls=1.30):
    """Text with the floor enforced rather than trusted."""
    if size < MIN_PT - 1e-9:
        sys.exit(f"TYPE FLOOR — {s[:40]!r} asked for {size}pt, floor is {MIN_PT}")
    return ax.text(x, y, s, fontsize=size, color=colour or ds.INK, ha=ha,
                   va=va, fontweight=weight or "normal", style=style or "normal",
                   linespacing=ls, zorder=5)


def box(ax, x, y, w, h, title=None, body=None, edge=None, face="white",
        lw=0.9, title_size=9, body_size=8.2, title_colour=None, dashed=False):
    """A rounded box with an optional bold title line and a body block.

    x, y is the top-left corner; h grows downward, which is how the specs below
    read (top to bottom), not how matplotlib's y-axis runs.
    """
    edge = edge or ds.GREY
    p = mpatches.FancyBboxPatch(
        (x, y - h), w, h, boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=lw, edgecolor=edge, facecolor=face,
        linestyle=(0, (3, 2)) if dashed else "solid", zorder=2)
    ax.add_patch(p)
    ty = y - 3.2
    if title:
        ax.text(x + w / 2, ty, title, ha="center", va="top", zorder=3,
                fontsize=title_size, color=title_colour or ds.INK,
                fontweight="bold")
        ty -= title_size * 0.42 + 1.4
    if body:
        ax.text(x + w / 2, ty, body, ha="center", va="top", zorder=3,
                fontsize=body_size, color=ds.INK2, linespacing=1.35)
    return p


def arrow(ax, x0, y0, x1, y1, colour=None, lw=0.9, label=None,
          label_side="above", size=8.0, dashed=False, head=True,
          gap_start=0.0, gap_end=2.5):
    """A connector that stops short of what it points at.

    The first draft drew these with shrink zero, so every arrowhead landed on a
    box border and the join read as a wireframe rather than a drawing.  The
    default now leaves 2.5 pt of air at the head, which is what makes a
    connector look placed rather than collided.
    """
    colour = colour or ds.GREY
    style = f"-|>,head_width=0.26,head_length=0.62" if head else "-"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle=style, color=colour, linewidth=lw,
                                linestyle=(0, (3, 2)) if dashed else "solid",
                                shrinkA=gap_start, shrinkB=gap_end,
                                mutation_scale=9.0,
                                joinstyle="round", capstyle="round"))
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dy = 1.8 if label_side == "above" else -1.8
        ax.text(mx, my + dy, label, ha="center", zorder=5, fontsize=size,
                va="bottom" if label_side == "above" else "top", color=colour)


def elbow(ax, x0, y0, x1, y1, colour=None, lw=0.95, rad=5.0, head=True,
          gap_end=2.5, gap_start=0.0, vertical_first=True):
    """A right-angled connector whose corner is an arc, not a mitre.

    A square corner is the tell of a diagram drawn by whoever was in a hurry.
    matplotlib will round it for free through the `angle` connection style, and
    the arc radius is the one number that decides whether a schematic looks
    drawn or assembled.
    """
    colour = colour or ds.GREY
    a, b = (90 if y1 > y0 else -90, 0) if vertical_first else (0, 90 if y1 > y0 else -90)
    ax.add_patch(mpatches.FancyArrowPatch(
        (x0, y0), (x1, y1),
        connectionstyle=f"angle,angleA={a},angleB={b},rad={rad}",
        arrowstyle="-|>,head_width=0.26,head_length=0.62" if head else "-",
        color=colour, linewidth=lw, shrinkA=gap_start, shrinkB=gap_end,
        mutation_scale=9.0, joinstyle="round", capstyle="round", zorder=3))


def spine(ax, x, y0, y1, colour=None, lw=0.9, label=None, size=8.0):
    """A vertical gathering line, with its label sitting in a hole in the line.

    A label drawn straight over a rule is the single crudest thing in a
    schematic.  The white bounding box is the whole point of this helper.
    """
    colour = colour or ds.GREY
    ax.plot([x, x], [y0, y1], color=colour, linewidth=lw, zorder=3,
            solid_capstyle="round")
    if label:
        ax.text(x, (y0 + y1) / 2, label, ha="center", va="center", zorder=5,
                fontsize=size, color=ds.INK2,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="none"))


def rule(ax, x0, x1, y, colour=None, lw=0.7, dashed=False):
    """A hairline, for separating apparatus from argument."""
    ax.plot([x0, x1], [y, y], color=colour or ds.RULE, linewidth=lw, zorder=1,
            linestyle=(0, (3, 2)) if dashed else "solid",
            solid_capstyle="butt")


def caption_note(fig, text, y=0.0, size=8.0):
    """The basis line under a schematic, in the report's apparatus grey."""
    fig.text(0.0, y, text, ha="left", va="bottom", fontsize=size,
             color=ds.INK2, linespacing=1.35)


# ------------------------------------------------------------------ diagrams
# Filled in once the design is settled; each returns the figure name it wrote.

FIGURES = {}


def register(name):
    def deco(fn):
        FIGURES[name] = fn
        return fn
    return deco


def card(fig, ax, x, y, w, h, tag, knows, count, power, accent, foot=None,
         count_colour=None, fill=None, edge=0.40):
    """One rung of the ladder: what the reference knows, then what survived it.

    The card belongs to a colour family rather than wearing one: a 5.5% tint of
    the accent for the fill and a 40% tint for the hairline, with the accent
    itself reserved for the tag.  A saturated border on a box this small reads
    as a button.

    There is deliberately no rule inside the card.  A 54 pt card holding four
    rows has no clearance for one, and the first attempt drew it straight
    through the cap height of the count.  The 12 pt bold numeral already
    separates outcome from design.  `zero` paints the count in ink rather than
    in the survivors' blue, because a count of none should not wear the colour
    the palette gives to what survived.

    A tag of the form "R2 . heading" is split: the rung number is set into the
    same numbered tab the overview band uses for its cards, and the heading runs
    on beside it.  That is what makes the two halves of the overview figure read
    as one drawing rather than as an apparatus diagram with a results diagram
    glued underneath.  The swap is close to width-neutral -- the tab costs about
    what the dropped "R2 . " prefix returns -- and the row measurement below is
    taken against whichever form is actually drawn.

    Every row is measured against the card before it is drawn, so a later edit
    that lengthens a string fails here rather than overhanging the border in the
    printed report.
    """
    pad = 8.0
    inner = w - 2 * pad
    cw = text_w(fig, count, 12.0, True)
    m = re.match(r"^(R\d)\s+·\s+(.*)$", tag)
    rung, head = (m.group(1), m.group(2)) if m else (None, tag)
    TABW, TABGAP = 14.0, 3.5
    head_w = text_w(fig, head, 8.6, True) + (TABW + TABGAP if rung else 0)
    rows = [(head, 8.6, True, head_w),
            (knows, 8.0, False, text_w(fig, knows, 8.0)),
            (power, 8.0, False, text_w(fig, power, 8.0)),
            (count + (" " + foot if foot else ""), 12.0, True,
             cw + (6 + text_w(fig, foot, 8.0) if foot else 0))]
    for s_, sz, b_, ww in rows:
        if ww > inner + 0.5:
            sys.exit(f"CARD OVERFLOW — {tag!r}: {s_!r} needs {ww:.1f}pt, "
                     f"card gives {inner:.1f}pt")
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=2.4",
        linewidth=0.8, edgecolor=tint(accent, edge),
        facecolor=tint_to_L(accent, fill or FILL_L), zorder=2,
        clip_on=False))
    if rung:
        tab(ax, x + pad, y + h - 5.5, rung, accent, w=TABW, h=11.0)
        t(ax, x + pad + TABW + TABGAP, y + h - 13.5, head, size=8.6,
          weight="bold", colour=accent)
    else:
        t(ax, x + pad, y + h - 13.5, head, size=8.6, weight="bold", colour=accent)
    t(ax, x + pad, y + h - 25, knows, size=8.0, colour=ds.INK2)
    t(ax, x + pad, y + 15, count, size=12.0, weight="bold",
      colour=count_colour or ds.BLUE)
    if foot:
        t(ax, x + pad + cw + 6, y + 15, foot, size=8.0, colour=ds.INK2)
    t(ax, x + pad, y + 4.5, power, size=8.0, colour=ds.INK2)


def _icon_rgba(name, colour, px=192):
    """One of the paper's icon assets, flattened to `colour` and cropped.

    The assets are rich full-colour PNGs drawn for a two-column paper.  Dropped
    into this document unchanged they would be the only saturated objects in a
    report whose figures are drawn from one restrained palette, and they would
    read as clip art.  Flattening each to its own card's accent keeps the icon
    working as a family mark instead: the eye ties it to the card, not to
    itself.  Geometry only otherwise -- crop to content, square-pad, downsample.
    """
    from PIL import Image
    import numpy as np
    src = os.path.join(ROOT, "writing", "paper", "figures", "icons", name + ".png")
    if not os.path.exists(src):
        return None
    im = Image.open(src).convert("RGBA")
    a = im.split()[3]
    bb = a.getbbox()
    if bb:
        im = im.crop(bb)
    side = int(max(im.size) * 1.06)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2), im)
    sq = sq.resize((px, px), Image.LANCZOS)
    arr = np.asarray(sq).astype(float) / 255.0

    # Duotone, not a flat fill.  Painting every opaque pixel one colour throws
    # away the drawing: the robot loses its face and reads as a blob.  Mapping
    # the source's own luminance onto a ramp from a pale tint of the accent to
    # the accent itself keeps the internal line work while still tying the mark
    # to its card.
    lum = arr[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    full = np.array([int(colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
    pale = full + (1.0 - full) * 0.72
    w = np.clip(1.0 - lum, 0.0, 1.0)[..., None] ** 0.85
    out = np.zeros_like(arr)
    out[..., :3] = pale + (full - pale) * w
    out[..., 3] = arr[..., 3]
    return out


def icon(ax, x, y, s, colour, name):
    """Place a tinted icon with its top-left at (x, y); y grows downward."""
    arr = _icon_rgba(name, colour)
    if arr is None:
        return
    ax.imshow(arr, extent=[x, x + s, y - s, y], zorder=4,
              interpolation="antialiased")


def tab(ax, x, y, n, colour, w=13.5, h=11.5):
    """The numbered tab that opens each card, as on a filing divider."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y - h), w, h, boxstyle="round,pad=0,rounding_size=3.2",
        linewidth=0, facecolor=colour, zorder=3))
    ax.text(x + w / 2, y - h / 2, str(n), ha="center", va="center", zorder=4,
            fontsize=8.0, color="white", fontweight="bold")


def pill(ax, x, y, w, h, text, colour, size=8.0):
    """A filled rounded strip: one fact the card is willing to be judged on."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y - h), w, h, boxstyle="round,pad=0,rounding_size=4.0",
        linewidth=0.7, edgecolor=tint(colour, 0.40),
        facecolor=tint(colour, 0.13), zorder=3))
    ax.text(x + w / 2, y - h / 2, text, ha="center", va="center", zorder=4,
            fontsize=size, color=colour, fontweight="bold")


def timeline(ax, x, y, w, colour, first, last, n=4):
    """A dotted span: the years the panel covers, drawn rather than stated."""
    ax.plot([x, x + w], [y, y], color=tint(colour, 0.55), lw=1.6,
            solid_capstyle="round", zorder=3)
    for k in range(n):
        ax.plot([x + k * w / (n - 1)], [y], marker="o", ms=2.6,
                color=colour, zorder=4)
    t(ax, x, y - 11.0, first, size=8.0, colour=ds.INK2)
    t(ax, x + w, y - 11.0, last, size=8.0, colour=ds.INK2, ha="right")


# The overview band above the ladder -- the study's panel, challengers and
# gates -- is drawn only when this is true.  It is off by default because it
# costs exactly one page of the 60-page main body: Chapter 1 sits on a page
# boundary, and the free headroom above the ladder's own 186 pt is about 6 pt
# (measured: a 192 pt figure still ends the body on page 60, a 207 pt one does
# not).  Turning it on is an editorial decision about what page of Chapter 1
# pays for it, not a drawing decision, so it is left to the author.
OVERVIEW = False


@register("ladder")
def ladder():
    """Chapter 1: the reference ladder, and the fork inside it.

    The word "ladder" invites a reading the study does not support -- four rungs
    climbed in sequence -- and a reader who meets "8 survive identity, 9 survive
    the pool, none survives both" is entitled to wonder whether 8 and 9 missing
    each other in 69 cells is luck.  It is not: the two strengthenings run in
    different directions and their survivor sets are disjoint, which is why the
    conjunction is empty and why the empty conjunction is worth stating.
    """
    # Two bands on one canvas.  The ladder keeps the geometry it was tuned at
    # -- every coordinate below is measured against YL, the height it had as a
    # standalone figure -- and the study's front half is drawn above it in a
    # band of its own.  Adding the band by raising Y alone would have moved
    # `mid` and the prologue with it and retuned a drawing that was already
    # right; pinning the ladder to YL leaves it untouched at the foot of a
    # taller canvas.
    W, BAND = ds.DISS_W, (1.88 if OVERVIEW else 0.0)
    H = 2.50 + (BAND + 0.20 if OVERVIEW else 0.0)
    fig, ax = frame(W, H, xmax=W * 72, ymax=H * 72)
    X, Y = W * 72, H * 72
    YL = 2.50 * 72          # the ladder band: y = 0 .. YL
    BY = YL + 30            # the overview band sits above it

    # Connectors sit between the hairline grey of a rule and the ink of text:
    # at RULE they vanished, at GREY they competed with the cards.
    CONN = tint(ds.GREY, 0.55)
    cw1, cw2, cw4, ch = 99, 138, 129, 54
    mid = (YL - 24) / 2 + 8
    x1, x2, x4 = 0, 136, 307

    # The standalone verdict belongs before the ladder, not after it.  A rule in
    # the same grey as the apparatus turns three floating lines into a block.
    # src: ch4 "the 17 text and fusion arms remain 0 of 153 better even there"
    # The block sits beside R2, so every line has to stop short of that card.
    # The first version ran two of its three lines under R2's border; four
    # narrower lines fit, and give the count a line of its own, which is where
    # it belongs anyway.
    px, avail = 9.0, x2 - 8 - 9.0
    # "text alone" was wrong twice over: the 153 family is the 17 text AND
    # FUSION arms, and a fusion arm carries price by construction, so the block
    # was claiming a text-only result the report does not make.  What is alone
    # here is the challenger, standing before any combination with the reference.
    rows = [("Before any combination, alone:", 8.0, False, ds.INK2),
            ("0 of 153", 12.0, True, ds.VERM_TXT),
            ("text and fusion comparisons", 8.0, False, ds.INK2),
            ("favour the challenger,", 8.0, False, ds.INK2),
            ("under three loss conventions", 8.0, False, ds.INK2)]
    for s_, sz, b_, _c in rows:
        ww = text_w(fig, s_, sz, b_)
        if ww > avail + 0.5:
            sys.exit(f"PROLOGUE OVERFLOW — {s_!r} needs {ww:.1f}pt, "
                     f"{avail:.1f}pt before the R2 card")
    py = YL - 12
    ax.plot([0.8, 0.8], [py - 50.8, py + 6.4], color=tint(ds.VERM, 0.45),
            linewidth=2.2, zorder=3, solid_capstyle="round", clip_on=False)
    for k, (s_, sz, b_, c_) in enumerate(rows):
        t(ax, px, py - k * 12.2, s_, size=sz,
          weight="bold" if b_ else None, colour=c_)

    # src: ch1 "38 of 69 combination cells show an apparent, placebo-confirmed
    #      increment"; ch3 "The primary rung recovers the injected signal in
    #      12, 20 and 41 of 69 cells at the three sizes"
    card(fig, ax, x1, mid - ch / 2, cw1, ch, "R1 \u00b7 price history only",
         "recalibrated HAR", "38 of 69", "detects 12 / 20 / 41", ds.GREY)

    # src: ch1 "8 of 69 cells survive a reference that additionally knows only
    #      each firm's own average volatility"; ch4 "the 8 identity survivors
    #      split six event-driven and two long-form"; ch3 rungs "7/11/20"
    card(fig, ax, x2, mid + 5, cw2, ch, "R2 \u00b7 + who is speaking",
         "adds the filer\u2019s own mean volatility", "8 of 69",
         "detects 7 / 11 / 20", ds.VERM_TXT, foot="six 8-K, two long-form",
         edge=0.55)

    # src: ch1 "9 survive a pool of five price models"; ch4 "all 9 pool
    #      survivors are long-form"; ch3 rungs "6/12/19".  The five model names
    #      are in the caption, not here: they measure 145pt and the widest card
    #      this layout affords is 121pt.  "adds four more price models" would
    #      have fitted, but four is arithmetic on the report rather than a
    #      number the report states, and this figure draws only what it says.
    card(fig, ax, x2, mid - ch - 5, cw2, ch, "R3 \u00b7 + a five-model price pool",
         "fitted on the validation years", "9 of 69",
         "detects 6 / 12 / 19", ds.PURPLE, foot="all nine long-form",
         edge=0.55)

    # src: ch1 "none survives both"; ch4 table "Full conjunction ... survivor
    #      sets disjoint"; ch1 "cleared the same conjunction in 2, 6 and 13"
    # "told both at once" implied a reference fitted with both terms.  No such
    # reference exists: Chapter 3 defines the rung as the full conjunction, an
    # increment credited only if it survives every rung simultaneously.
    card(fig, ax, x4, mid - ch / 2, cw4, ch, "R4 \u00b7 must clear both",
         "credited only if neither had it", "0 of 69",
         "detects 2 / 6 / 13", ds.INK, foot="the sets never meet",
         count_colour=ds.VERM_TXT, fill=FILL_L_END, edge=0.58)

    # The fork: two directions, not two steps.  Both junctions are drawn as one
    # gathering line with the label inside a hole in it.
    ytop, ybot = mid + 5 + ch / 2, mid - 5 - ch / 2
    xf = x1 + cw1 + 13
    arrow(ax, x1 + cw1, mid, xf, mid, colour=CONN, lw=0.95, head=False,
          gap_start=1.5, gap_end=0)
    for yy in (ytop, ybot):
        elbow(ax, xf, mid, x2, yy, colour=CONN, rad=5.5)
    ax.text((xf + x2) / 2, mid, "two\nways", ha="center", va="center", zorder=6,
            fontsize=8.0, color=ds.INK2, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                      edgecolor="none"))

    # The merge mirrors the fork: the same 13 pt stub off the card, the same
    # wedge for the label, the same arc radius.  Asymmetry between the two
    # junctions was the loudest thing left in the drawing.
    xj = x2 + cw2 + 21
    for yy in (ytop, ybot):
        elbow(ax, x2 + cw2, yy, xj, mid, colour=CONN, rad=5.5, head=False,
              gap_end=0, gap_start=1.5, vertical_first=False)
    arrow(ax, xj, mid, x4, mid, colour=CONN, lw=0.95, gap_start=0)
    ax.text((x2 + cw2 + xj) / 2, mid, "both", ha="center", va="center", zorder=6,
            fontsize=8.0, color=ds.INK2,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                      edgecolor="none"))

    # Spans the canvas, not the card block.  A first review asked for the
    # opposite; a second pointed out that the only thing long enough to compare
    # this rule against is the justified body text either side of the figure, so
    # landing on the text margins is what a reader actually checks.
    rule(ax, 0, X, 20, lw=0.6)
    t(ax, 0, 8,
      "\u2018detects\u2019 = cells that recover an oracle firm-orthogonal signal "
      "injected at 0.3 / 0.5 / 1.0 per cent of reference loss,",
      size=8.0, colour=ds.INK2)
    t(ax, 0, -2, "so R4\u2019s zero is read against the 2 of 69 that the smallest of those three would have cleared.",
      size=8.0, colour=ds.INK2)

    if OVERVIEW:
        # ---------------- the band above: what the ladder is run on ------------
        # The ladder answers "what survived"; alone it asks the reader to take
        # the panel and the challengers on trust.  These three cards are the
        # study's front half, drawn to the ladder's own rule: every figure is
        # one the chapters state, and not one of them is a result.
        #
        # The card is built as a filing divider rather than a list -- numbered
        # tab, family mark, rule, body, and a strip at the foot carrying the one
        # commitment the card is willing to be judged on.  Two details do most
        # of the work.  The foot strips are anchored to the card's bottom edge
        # instead of following the text down, so they line up across three cards
        # whose bodies are different lengths; and the counts that matter are set
        # large in the family accent, so the eye lands on 144,129 and 431,245
        # before it reads a word.  Both are cheap; neither is decoration.
        CARDS = [
            (1, "The panel", ds.BLUE, "survivorship-free, point-in-time",
             "doc_stack", [
                 [("144,129", 11.5, True), ("  SEC filings", 8.0, False)],
                 [("31,601 long-form · 112,528 8-K", 8.0, False)],
                 [("431,245", 11.5, True), ("  filing-by-horizon rows", 8.0, False)],
                 [("CRSP daily prices, 2010–25", 8.0, False)]],
             "no-look-ahead: 0 violations"),
            (2, "The challengers", ds.GREEN_TXT, "one recipe, three training seeds",
             "robot_prompt", [
                 [("text", 8.6, True), ("bag-of-words · dictionaries", 8.0, False)],
                 [("fine-tuned · frozen 7–8B · 32B", 8.0, False)],
                 [("price", 8.6, True), ("HAR · SHAR · GARCH", 8.0, False)],
                 [("EGARCH · ARIMA", 8.0, False)]],
             "h = 5, 10, 20 trading days"),
            (3, "The gates", ds.PURPLE, "declared before the statistics",
             "scale", [
                 [("day-clustered Diebold–Mariano", 8.0, False)],
                 [("Holm inside 15 families", 8.0, False)],
                 [("label-shuffle placebo · power", 8.0, False)],
                 [("240", 11.5, True), ("  fingerprinted runs", 8.0, False)]],
             "families fixed before the tests"),
        ]
        GAP, PADX, ROW, TAGCOL = 11.0, 8.0, 13.4, 27.0
        bh = BAND * 72 - 14
        bw = (X - 2 * GAP) / 3
        avail = bw - 2 * PADX
        for n, title, colour, sub, ico, rows, foot in CARDS:
            bx = (n - 1) * (bw + GAP)
            top = BY + bh
            # Measured, not trusted: a later edit that lengthens any line
            # fails here rather than printing over the card's own border.  The
            # head line is checked against the width the icon leaves it, the
            # foot strip against the strip's own inset.
            checks = [(r, avail) for r in rows]
            checks.append(([(sub, 8.0, False)], avail))
            checks.append(([(foot, 8.0, True)], avail - 14))
            checks.append(([(title, 9.0, True)], avail - 19 - 23))
            for row, cap in checks:
                tagged = row[0][1] == 8.6
                need = sum(text_w(fig, q, sz, b) for q, sz, b in row[1:]) \
                    + (TAGCOL if tagged else text_w(fig, *row[0][:2],
                                                    row[0][2]) * 0 +
                       (sum(text_w(fig, q, sz, b) for q, sz, b in row[:1])))
                if need > cap + 0.5:
                    sys.exit(f"BAND OVERFLOW — {''.join(r[0] for r in row)!r} "
                             f"needs {need:.1f}pt, card affords {cap:.1f}pt")
            box(ax, bx, top, bw, bh, edge=tint(colour, 0.42),
                face=tint(colour, 0.05), lw=0.75)
            tab(ax, bx + PADX, top - 6, n, colour)
            t(ax, bx + PADX + 19, top - 15.5, title, size=9.0, weight="bold",
              colour=colour)
            icon(ax, bx + bw - PADX - 21, top - 4.5, 21, colour, ico)
            t(ax, bx + PADX, top - 31, sub, size=8.0, colour=ds.INK2)
            ax.plot([bx + PADX, bx + bw - PADX], [top - 38] * 2, lw=0.6,
                    color=tint(colour, 0.38), zorder=3, solid_capstyle="butt")
            for r, row in enumerate(rows):
                cx = bx + PADX
                for q, sz, b in row:
                    t(ax, cx, top - 52 - r * ROW, q, size=sz,
                      weight="bold" if b else None,
                      colour=colour if b else ds.INK2)
                    # A tag opens a column, so what follows it lines up
                    # across rows whose tags are different lengths.
                    cx = (bx + PADX + TAGCOL if sz == 8.6
                          else cx + text_w(fig, q, sz, b))
            pill(ax, bx + PADX, BY + 23, bw - 2 * PADX, 15, foot, colour)

        # One arrow down the middle: the band is what the rungs are run on.
        arrow(ax, X / 2, BY - 3, X / 2, YL + 2, colour=tint(ds.GREY, 0.55), lw=1.0)
        ax.text(X / 2 + 7, (BY + YL) / 2 - 2, "run through the ladder", ha="left",
                va="center", zorder=6, fontsize=8.0, color=ds.INK2)

    ds.finish(fig, "C1_reference_ladder_overview" if OVERVIEW else "C1_reference_ladder",
              note="concept diagram; no series")


@register("shortcut")
def shortcut():
    """Chapter 2: the two routes by which a filing can improve a forecast.

    Held-out accuracy scores the two together and reports only their sum, which
    is the whole reason the audit downstream needs a reference that already
    knows the filer.  Kept to a single strip: Chapter 2 has no page to spare,
    and the chronological-split half of the argument survives in the prose.

    Drawn in the same vocabulary as the ladder -- tinted card, hairline in the
    family, accent reserved for the tag, connectors that stop short of what they
    point at -- so the two schematics read as a pair rather than as two
    attempts.
    """
    W, H = ds.DISS_W, 1.06
    fig, ax = frame(W, H, xmax=W * 72, ymax=H * 72)
    X, Y = W * 72, H * 72
    CONN = tint(ds.GREY, 0.55)

    bh, gap, pad = 27, 9, 8.0
    rc_y0 = Y - 4 - bh
    ri_y0 = rc_y0 - gap - bh
    mid = (rc_y0 + bh + ri_y0) / 2

    fw = max(text_w(fig, "filing", 8.6, True),
             text_w(fig, "10-K, 10-Q, 8-K", 8.0)) + 2 * pad
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, mid - 18), fw, 36, boxstyle="round,pad=0,rounding_size=2.4",
        linewidth=0.8, edgecolor=tint(ds.GREY, 0.40),
        facecolor=tint_to_L(ds.GREY, FILL_L), zorder=2, clip_on=False))
    t(ax, fw / 2, mid + 3.5, "filing", size=8.6, ha="center", weight="bold")
    t(ax, fw / 2, mid - 9, "10-K, 10-Q, 8-K", size=8.0, ha="center",
      colour=ds.INK2)

    # The forecast label closes the strip; the routes get whatever is left.
    fx = X - text_w(fig, "forecast", 8.6, True)
    bx = fw + 34
    # 26pt, not 15: the merge needs room for a bracket, a shaft and a head.
    # At 15 the arrow was a head sitting on the bracket with no shaft at all.
    bw = fx - 38 - bx

    tagw = max(text_w(fig, "Route C \u00b7 content", 8.6, True),
               text_w(fig, "Route I \u00b7 identity", 8.6, True))
    for y0, tag, sub, col in [
            (rc_y0, "Route C \u00b7 content",
             "what the filing says about the risk ahead", ds.BLUE),
            (ri_y0, "Route I \u00b7 identity",
             # The semicolon is load-bearing: the arrow starts at the filing, so
             # "and how volatile it is" made the filing state the volatility and
             # collapsed the two steps the chapter separates.  The filing reveals
             # who; the volatility was already known from the price history.
             "which firm is speaking; its volatility is known", ds.VERM_TXT)]:
        need = tagw + 12 + text_w(fig, sub, 8.0)
        if need > bw - 2 * pad + 0.5:
            sys.exit(f"ROUTE OVERFLOW — {tag!r}: needs {need:.1f}pt, "
                     f"box gives {bw - 2 * pad:.1f}pt")
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, y0), bw, bh, boxstyle="round,pad=0,rounding_size=2.4",
            linewidth=0.8, edgecolor=tint(col, 0.40),
            facecolor=tint_to_L(col, FILL_L), zorder=2, clip_on=False))
        t(ax, bx + pad, y0 + 9.5, tag, size=8.6, weight="bold", colour=col)
        t(ax, bx + pad + tagw + 12, y0 + 9.5, sub, size=8.0, colour=ds.INK2)

    # Same construction as the ladder's fork: a stub off the box, then one
    # elbow per branch that arrives horizontally at the route's left edge.
    # Arriving vertically, as a first attempt did, plants the arrowhead under
    # the box instead of in it.
    xf = fw + 11
    arrow(ax, fw, mid, xf, mid, colour=CONN, lw=0.95, head=False,
          gap_start=1.5, gap_end=0)
    for yy in (rc_y0 + bh / 2, ri_y0 + bh / 2):
        elbow(ax, xf, mid, bx, yy, colour=CONN, rad=4.5)

    xj = bx + bw + 8
    for yy in (rc_y0 + bh / 2, ri_y0 + bh / 2):
        elbow(ax, bx + bw, yy, xj, mid, colour=CONN, rad=4.5, head=False,
              gap_start=1.5, gap_end=0, vertical_first=False)
    arrow(ax, xj + 1, mid, fx - 3, mid, colour=CONN, lw=0.95, gap_start=0,
          gap_end=1.5)
    t(ax, fx, mid + 3.5, "one", size=8.6, weight="bold")
    t(ax, fx, mid - 8, "forecast", size=8.6, weight="bold")

    ds.finish(fig, "C2_identity_shortcut", note="concept diagram; no series")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", help="diagram names; default all")
    ap.add_argument("--check", action="store_true",
                    help="verify every drawn number appears in the chapters")
    a = ap.parse_args()
    if a.check:
        from concept_check import check_all
        sys.exit(0 if check_all() else 1)
    names = a.which or list(FIGURES)
    for n in names:
        if n not in FIGURES:
            sys.exit(f"unknown diagram {n!r}; have {sorted(FIGURES)}")
        print(f"== {n}")
        FIGURES[n]()


if __name__ == "__main__":
    main()
