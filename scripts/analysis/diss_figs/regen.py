"""Regenerate the dissertation's figures from the supplement's generators.

Why this exists
---------------
The eleven figures the rebuilt chapters include are the Technical Supplement's
own gated generators.  They must stay gated -- the evidence checks inside them
are the reason a figure cannot drift from the tables -- so this driver does not
fork them.  It reads each generator's source, rewrites *only its canvas
constants* (the `figsize` call, and for the three generators whose annotation
blocks are wrapped against an absolute measure, that measure), and executes the
result with `supp_style` replaced by `diss_style` in `sys.modules`.  Every
`gate(...)` call, every data path and every drawing instruction runs exactly as
the supplement runs it; the output simply lands in
`writing/dissertation/figures/` on a canvas that fits an A4 text block.

No figure's content is redesigned here.  The only thing that changes is the shape
of the paper the figure is drawn on.

How far a canvas may be compressed
----------------------------------
These figures carry long in-figure note blocks.  Type does not shrink with the
canvas, so squeezing the canvas eats the gaps between panels rather than the
panels themselves, and past some point a note lands on an axis label.  `--search`
therefore renders each generator once at its own canvas to record which pieces of
type already overlap, then bisects the canvas height for the smallest canvas that
introduces no *new* overlap.  That number, not a wished-for page budget, is what
each figure is regenerated at.

Usage
-----
    python3 regen.py                 # regenerate all eleven at the recorded canvas
    python3 regen.py F4 F12          # a subset (prefix match)
    python3 regen.py --search        # bisect each canvas to its collision limit
    python3 regen.py --png           # also write a PNG preview at reading size
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, ".."))
SUPP_FIGS = os.path.join(ANALYSIS, "supp_figs")
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, SUPP_FIGS)

import diss_style as ds  # noqa: E402

PREVIEW = os.path.join(HERE, "preview")

# ---------------------------------------------------------------------------
# Per-figure canvas specification.
#
#   src        generator in scripts/analysis/supp_figs/
#   w0, h0     the generator's own canvas, inches (the supplement's shape)
#   w,  h      the dissertation canvas, inches
#   subs       exact-string canvas rewrites, each of which must match once.
#              `{w}` / `{h}` are filled from the spec.  Anything beyond the
#              `figsize` call is a wrap measure that has to follow the width.
#   cap        hard height gate handed to diss_style.finish, in points of page
#   note       recorded in the geometry manifest
# ---------------------------------------------------------------------------
SPECS = {
    "F1_membership_panel": dict(
        src="F1_membership_panel.py", w0=6.4, h0=8.6, w=6.40, h=8.261, cap=625.0,
        subs=[("fig = plt.figure(figsize=(6.4, 8.6))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="three stacked panels on a fractional gridspec; canvas only"),
    "F3_standalone_180_and_reference_strength": dict(
        src="F3_standalone_180_and_reference_strength.py",
        w0=6.4, h0=8.9, w=6.40, h=8.024, cap=575.0,
        subs=[("fig = plt.figure(figsize=(6.4, 8.9))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="fractional add_axes, saved untight; canvas only"),
    "F4_ladder_cell_matrix": dict(
        src="F4_ladder_cell_matrix.py", w0=6.4, h0=9.0, w=6.40, h=7.523, cap=535.0,
        subs=[("fig = plt.figure(figsize=(W, H))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="inch grid divided by the original W,H; only the canvas shrinks"),
    "F7_anonymisation_price": dict(
        src="F7_anonymisation_price.py", w0=6.4, h0=8.05, w=6.40, h=7.469, cap=570.0,
        subs=[("fig = plt.figure(figsize=(6.4, 8.05))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="three-block fractional GridSpec; canvas only"),
    "F8_matched_firm_swap": dict(
        src="F8_matched_firm_swap.py", w0=6.4, h0=6.15, w=6.40, h=5.744, cap=565.0,
        subs=[("fig = plt.figure(figsize=(6.4, 6.15))",
               "fig = plt.figure(figsize=({w}, {h}))"),
              # The one substitution here that is not a canvas constant.  At this
              # canvas the green "dotted: locus of retention" line sits close enough
              # under the grey median block that the block's white halo shears the
              # tops off its ascenders.  The generator already records the reverse
              # collision (the green halo used to erase the grey line's baseline) and
              # settled it by z-order, which fixes one line at the other's expense.
              # Dropping the green line by 0.08 data units separates them outright.
              # It moves a label, not a number.
              ('axb.text(ZX[0] + 0.08, ZY[1] - 0.60,',
               'axb.text(ZX[0] + 0.08, ZY[1] - 0.68,')],
        note="canvas, plus the green locus label dropped clear of the median block"),
    "F9_power_mde_label_noise": dict(
        src="F9_power_mde_label_noise.py", w0=6.5, h0=8.60, w=6.50, h=8.318,
        cap=605.0,
        subs=[("fig = plt.figure(figsize=(W, H))",
               "fig = plt.figure(figsize=({w}, {h}))"),
              ],
        wsubs=[("RWID = 43", "RWID = 40"),
               ("para(note_b, 96)", "para(note_b, 90)"),
               ("para(foot, 99)", "para(foot, 93)")],
        note="note columns re-wrapped for the narrower canvas; canvas only"),
    "F11_ladder_perturbations": dict(
        src="F11_ladder_perturbations.py", w0=6.4, h0=8.61, w=6.40, h=7.480,
        cap=525.0,
        subs=[("fig = plt.figure(figsize=(FW, FH))",
               "fig = plt.figure(figsize=({w}, {h}))"),
              ],
        wsubs=[("WRAP_IN = 6.40", "WRAP_IN = 6.05")],
        note="renderer-measured wrap narrowed to the canvas; canvas only"),
    "F12_health_screen_orthogonality": dict(
        src="F12_health_screen_orthogonality.py",
        w0=6.5, h0=8.80, w=6.50, h=8.400, cap=660.0,
        subs=[("fig = plt.figure(figsize=(W, H))",
               "fig = plt.figure(figsize=({w}, {h}))"),
              ],
        wsubs=[("para(note_a, 101)", "para(note_a, 94)"),
               ("para(note_b, 101)", "para(note_b, 94)"),
               ("para(foot, 101)", "para(foot, 94)")],
        note="not split: its (a) note block abuts the (b) heading with no gap at any canvas height, so no crop can separate them; h raised from 7.593 to 8.400, the smallest canvas at which those two stop overprinting, and it is included at 0.90\\textwidth to hold its former footprint"),
    "F14_economic_adjudication": dict(
        src="F14_economic_adjudication.py", w0=6.4, h0=8.55, w=6.40, h=7.596,
        cap=535.0,
        subs=[("fig = plt.figure(figsize=(6.4, H))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="inch-from-top layout divided by the original H; canvas only"),
    "F15_yelp_portability": dict(
        src="F15_yelp_portability.py", w0=6.4, h0=7.62, w=6.40, h=6.820, cap=490.0,
        subs=[("fig = plt.figure(figsize=(6.4, H))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="inch-from-top layout divided by the original H; canvas only"),
    "F16_maec_reproduce_reprice": dict(
        src="F16_maec_reproduce_reprice.py", w0=6.4, h0=8.55, w=6.40, h=8.382,
        cap=605.0,
        subs=[("fig = plt.figure(figsize=(6.4, H))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="inch-from-top layout divided by the original H; canvas only"),
}

# Caption block heights measured from the pre-regeneration build log, in points
# (11pt body, \onehalfspacing).  Divide by 1.5 for the same caption set single
# spaced, which is the typographic change that accompanies this regeneration.
CAPTION_PT = {
    "F1_membership_panel": 148.73,
    "F3_standalone_180_and_reference_strength": 121.54,
    "F4_ladder_cell_matrix": 120.93,
    "F7_anonymisation_price": 120.93,
    "F8_matched_firm_swap": 127.90,
    "F9_power_mde_label_noise": 120.93,
    "F11_ladder_perturbations": 121.53,
    "F12_health_screen_orthogonality": 134.52,
    "F14_economic_adjudication": 122.66,
    "F15_yelp_portability": 179.65,
    "F16_maec_reproduce_reprice": 107.33,
}

SEARCH_STATE = os.path.join(HERE, "canvas_search.json")


def _exec(name, spec, src, path, cap):
    ds.CAP_OVERRIDE[name] = cap
    ds.NOTE_OVERRIDE[name] = spec["note"]
    import supp_style  # noqa: F401  (kept loadable; only the binding is swapped)
    saved = sys.modules.get("supp_style")
    sys.modules["supp_style"] = ds
    ns = {"__name__": "__main__", "__file__": path}
    try:
        exec(compile(src, path, "exec"), ns)
    finally:
        if saved is not None:
            sys.modules["supp_style"] = saved
    with open(ds.MANIFEST) as fh:
        return json.load(fh)[name], ds.COLLISIONS.get(name, [])


def rewrite(spec, w=None, h=None):
    """Return the generator's source with its canvas constants replaced.

    `wsubs` are the wrap measures that have to follow a narrowed canvas; they are
    applied only when the canvas is actually narrower than the generator's own,
    because re-wrapping a note block at the original width only adds lines.
    """
    path = os.path.join(SUPP_FIGS, spec["src"])
    with open(path) as fh:
        src = fh.read()
    w = spec["w"] if w is None else w
    h = spec["h"] if h is None else h
    subs = dict(w=f"{w:.2f}", h=f"{h:.3f}")
    todo = list(spec["subs"])
    if w < spec["w0"] - 1e-9:
        todo += list(spec.get("wsubs", []))
    for old, repl in todo:
        if src.count(old) != 1:
            sys.exit(f"CANVAS SUB FAIL — {spec['src']}: {old!r} matched "
                     f"{src.count(old)} times, expected exactly 1")
        src = src.replace(old, repl.format(**subs))
    return path, src


def baseline(name, spec):
    """Render the generator untouched, to learn which type already overlaps."""
    path = os.path.join(SUPP_FIGS, spec["src"])
    with open(path) as fh:
        src = fh.read()
    return _exec(name, spec, src, path, cap=10_000.0)


def trial(name, spec, w, h):
    path, src = rewrite(spec, w=w, h=h)
    return _exec(name, spec, src, path, cap=10_000.0)


def run_one(name, spec):
    path, src = rewrite(spec)
    return _exec(name, spec, src, path, cap=spec["cap"])


def search(name, spec, lo=0.58, steps=6):
    """Pick the canvas that makes the shortest float without new collisions.

    Both candidate widths are tried: 6.10 in (the text block, so the figure sets
    1:1.04 and its type comes out slightly larger than the supplement's) and the
    generator's own width (which yields a squatter aspect and therefore a shorter
    float).  For each, the canvas height is bisected down to the point where the
    rendering first introduces a text overlap the original does not have.  The
    shorter float wins, because the page budget is what this exercise is for; the
    width that produced it is recorded with the figure.
    """
    _, base_pairs = baseline(name, spec)
    base = set(base_pairs)
    print(f"    baseline {spec['w0']}x{spec['h0']}in, "
          f"{len(base)} pre-existing overlap(s)")

    def ok(w, k):
        rec, pairs = trial(name, spec, w, spec["h0"] * k)
        new = sorted(set(pairs) - base)
        print(f"    w={w:.2f} k={k:.3f} h={spec['h0'] * k:.2f}in "
              f"{rec['rendered_h_pt']:6.1f}pt  "
              f"{'clean' if not new else str(len(new)) + ' new'}")
        for pair in new[:2]:
            print(f"        + {pair}")
        return (not new), rec

    widths = [6.10, spec["w0"]] if abs(spec["w0"] - 6.10) > 1e-9 else [6.10]
    best = None
    for w in widths:
        clean, rec = ok(w, 1.0)
        if not clean:
            print(f"    w={w:.2f} unusable at full height")
            continue
        a, b = lo, 1.0                        # a assumed to collide, b is clean
        if ok(w, lo)[0]:
            b = lo
        else:
            for _ in range(steps):
                m = (a + b) / 2.0
                if ok(w, m)[0]:
                    b = m
                else:
                    a = m
        _, rec = ok(w, b)
        cand = (rec["rendered_h_pt"], w, round(spec["h0"] * b, 3))
        print(f"    w={w:.2f} limit k={b:.3f} -> {cand[0]:.1f}pt")
        if best is None or cand[0] < best[0] - 0.5:
            best = cand
    if best is None:
        sys.exit(f"{name}: no usable canvas found")
    spec["w"], spec["h"] = best[1], best[2]
    print(f"    -> canvas {spec['w']}x{spec['h']}in, float graphic "
          f"{best[0]:.1f}pt")
    return run_one(name, spec)


def preview(name):
    """Rasterise the final PDF at roughly the size it has on the printed page."""
    os.makedirs(PREVIEW, exist_ok=True)
    pdf = os.path.join(ds.OUTDIR, f"{name}.pdf")
    w_pt, _ = ds.page_size_pt(pdf)
    dpi = int(round(1400.0 / (w_pt / 72.0)))
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                    pdf, os.path.join(PREVIEW, name)], check=True)


def load_state():
    if os.path.exists(SEARCH_STATE):
        with open(SEARCH_STATE) as fh:
            for name, wh in json.load(fh).items():
                if name in SPECS:
                    SPECS[name]["w"] = wh["w"]
                    SPECS[name]["h"] = wh["h"]


def save_state():
    with open(SEARCH_STATE, "w") as fh:
        json.dump({n: {"w": s["w"], "h": s["h"]} for n, s in SPECS.items()},
                  fh, indent=2, sort_keys=True)
        fh.write("\n")


def table(rows):
    print("\n" + "-" * 104)
    print(f"{'figure':<44}{'canvas in':>12}{'page pt':>14}{'graphic':>9}"
          f"{'+cap':>7}{'+cap/1.5':>10}{'frac':>7}")
    print("-" * 104)
    for name, spec, rec in rows:
        g = rec["rendered_h_pt"]
        c = CAPTION_PT[name]
        print(f"{name:<44}{spec['w']:>5.2f}x{spec['h']:<6.2f}"
              f"{rec['page_w_pt']:>7.0f}x{rec['page_h_pt']:<6.0f}{g:>9.1f}"
              f"{g + c:>7.0f}{g + c / 1.5:>10.0f}"
              f"{(g + c / 1.5) / ds.TEXTHEIGHT_PT:>7.3f}")
    print("-" * 104)
    print(f"text height {ds.TEXTHEIGHT_PT:.1f} pt; a float becomes placeable at "
          f"the top of a text page below \\topfraction x textheight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="figure name prefixes")
    ap.add_argument("--search", action="store_true",
                    help="bisect each canvas to its text-collision limit")
    ap.add_argument("--png", action="store_true", help="write PNG previews")
    ap.add_argument("--width", type=float, default=None,
                    help="force the canvas width, inches")
    args = ap.parse_args()

    load_state()
    chosen = [n for n in SPECS
              if not args.names or any(n.startswith(p) for p in args.names)]
    if not chosen:
        sys.exit(f"no figure matched {args.names}")

    rows = []
    for name in chosen:
        print(f"== {name}")
        spec = SPECS[name]
        if args.width:
            spec["w"] = args.width
        rec = (search(name, spec) if args.search else run_one(name, spec))[0]
        rows.append((name, spec, rec))
        if args.png:
            preview(name)
    if args.search:
        save_state()
    table(rows)


if __name__ == "__main__":
    main()
