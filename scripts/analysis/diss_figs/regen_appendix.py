"""Regenerate the five supplement figures Appendix E adopts, at A4 geometry.

Companion to `regen.py`, which did the same job for the eleven figures the
rebuilt chapters include.  The five here -- F2, F5, F6, F10 and F13 -- were
never used in the main text and are drawn for the supplement's page, so at
`width=\\textwidth` in the dissertation they render 596-640 pt tall and would be
scaled down at inclusion (shrinking their 9 pt type below the floor) or overflow
the page once a caption is added.

The machinery is `regen`'s: only each generator's `figsize` call is rewritten,
every `gate()` and every drawing instruction runs unchanged, and the output
lands in `writing/dissertation/figures/`.  The width is held at each
generator's own width so that no note block has to be re-wrapped -- the four
generators that place their annotation blocks at absolute inch offsets divided
by the module-level `W`/`H` keep those originals, so the whole layout scales
proportionally rather than drifting.

Usage
-----
    python3 regen_appendix.py            # regenerate all five at the recorded canvas
    python3 regen_appendix.py --search   # bisect each canvas to its collision limit
    python3 regen_appendix.py F5 F6      # a subset (prefix match)


STATUS: THIS DRIVER IS NOW DEAD, and deliberately so. Its SPECS name exactly
five figures -- F2, F5, F6, F10, F13 -- and all five are in NOT_OURS below,
so every name it can accept it now refuses. All twenty Appendix E figures are
owned by diss_appendix_figs/<name>.py and are run directly. The file is kept
rather than deleted because its refusal message is the thing that stops the
next person repeating the mistake it used to cause.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import diss_style as ds
import regen

# Same shape as regen.SPECS.  `w` is pinned to the generator's own width in
# every case: these five wrap their note blocks against a character count tuned
# to that width, and narrowing the canvas only pushes the blocks past its edge,
# whereupon the tight bounding box grows back and nothing is gained.
SPECS = {
    "F2_model_spectrum_and_compute": dict(
        src="F2_model_spectrum_and_compute.py",
        w0=6.4, h0=9.0, w=6.40, h=7.594, cap=626.0,
        subs=[("fig = plt.figure(figsize=(6.4, 9.0))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="two stacked panels on a fractional gridspec; canvas only"),
    "F5_maximal_pool_audit": dict(
        src="F5_maximal_pool_audit.py",
        w0=6.4, h0=9.0, w=6.40, h=7.594, cap=622.0,
        subs=[("fig = plt.figure(figsize=(W, H))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="inch grid divided by the original W,H; only the canvas shrinks"),
    "F6_firm_identity_rung": dict(
        src="F6_firm_identity_rung.py",
        w0=6.4, h0=8.45, w=6.40, h=7.129, cap=565.0,
        subs=[("fig = plt.figure(figsize=(W, H))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="inch grid divided by the original W,H; only the canvas shrinks"),
    "F10_encompassing_pooled_power": dict(
        src="F10_encompassing_pooled_power.py",
        w0=6.5, h0=9.0, w=6.50, h=7.594, cap=570.0,
        subs=[("fig = plt.figure(figsize=(W, H))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="inch-from-top note blocks divided by the original H; canvas only"),
    "F13_elicitation_not_curation": dict(
        src="F13_elicitation_not_curation.py",
        w0=6.5, h0=8.95, w=6.50, h=7.552, cap=580.0,
        subs=[("fig = plt.figure(figsize=(W, H))",
               "fig = plt.figure(figsize=({w}, {h}))")],
        note="inch-from-top note blocks divided by the original H; canvas only"),
}

# ---------------------------------------------------------------- ownership
# These five figures are NOT ours. Their authoritative source is
# diss_appendix_figs/<name>_diss.py, which imports diss_style directly, carries
# its own figsize, and calls ds.finish() with the SAME output name we would use.
# Two code paths writing one file is how five Appendix E figures silently fell
# from ~9.05 pt printed type to 8.47-8.86 pt: regenerating them HERE rewrites a
# canvas that was never theirs. Verified by running the _diss.py path, which
# reproduces the committed geometry to the hundredth of a point (F2 452.62x592.11,
# floor 9.05, identical to HEAD).
NOT_OURS = {
    "F2_model_spectrum_and_compute": "F2_model_spectrum_and_compute_diss.py",
    "F5_maximal_pool_audit": "F5_maximal_pool_audit_diss.py",
    "F6_firm_identity_rung": "F6_firm_identity_rung_diss.py",
    "F10_encompassing_pooled_power": "F10_encompassing_pooled_power_diss.py",
    "F13_elicitation_not_curation": "F13_elicitation_not_curation_diss.py",
}

STATE = os.path.join(HERE, "canvas_search_appendix.json")


def search(name, spec, lo=0.62, steps=6):
    """Bisect the canvas height for the shortest float with no new type overlap."""
    _, base_pairs = regen.baseline(name, spec)
    base = set(base_pairs)
    print(f"    baseline {spec['w0']}x{spec['h0']}in, "
          f"{len(base)} pre-existing overlap(s)")

    def ok(k):
        rec, pairs = regen.trial(name, spec, spec["w0"], spec["h0"] * k)
        new = sorted(set(pairs) - base)
        print(f"    k={k:.3f} h={spec['h0'] * k:.3f}in "
              f"{rec['rendered_h_pt']:6.1f}pt  "
              f"{'clean' if not new else str(len(new)) + ' new'}")
        for pair in new[:2]:
            print(f"        + {pair}")
        return (not new), rec

    clean, _ = ok(1.0)
    if not clean:
        sys.exit(f"{name}: the generator's own canvas already fails its own baseline")
    a, b = lo, 1.0
    if ok(lo)[0]:
        b = lo
    else:
        for _ in range(steps):
            m = (a + b) / 2.0
            if ok(m)[0]:
                b = m
            else:
                a = m
    spec["h"] = round(spec["h0"] * b, 3)
    _, rec = ok(b)
    print(f"    -> canvas {spec['w']}x{spec['h']}in, float graphic "
          f"{rec['rendered_h_pt']:.1f}pt")
    return regen.run_one(name, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--use-search-state", action="store_true",
                    help="apply canvas_search_appendix.json over the tuned specs")
    args = ap.parse_args()

    # canvas_search_appendix.json is the OUTPUT of --search, not an input to a
    # normal run. It used to be applied unconditionally, and it had gone stale:
    # it recorded h=8.786in for F2 where the tuned spec says 7.594in. A taller
    # native canvas is scaled down harder on the page, so a plain regeneration
    # silently pushed five Appendix E figures from ~9.05 pt printed type to
    # 8.47-8.86 pt, under the 9 pt floor, with nothing in the output saying so.
    # The specs are now authoritative unless the state file is asked for.
    if args.use_search_state and os.path.exists(STATE):
        with open(STATE) as fh:
            for name, wh in json.load(fh).items():
                if name in SPECS:
                    SPECS[name]["w"], SPECS[name]["h"] = wh["w"], wh["h"]
        print("NOTE: canvas overridden from canvas_search_appendix.json; "
              "re-run audit_inclusion_geometry.py and check the 9 pt floor.")

    chosen = [n for n in SPECS
              if not args.names or any(n.startswith(p) for p in args.names)]
    stolen = [n for n in chosen if n in NOT_OURS]
    if stolen:
        lines = [f"    python3 scripts/analysis/diss_appendix_figs/{NOT_OURS[n]}"
                 for n in stolen]
        sys.exit("REFUSING: these figures are owned by diss_appendix_figs, not by "
                 "this driver.\nRegenerating them here rewrites a canvas that is "
                 "not theirs and drops their printed\ntype below the 9 pt floor. "
                 "Run instead:\n" + "\n".join(lines))
    if not chosen:
        sys.exit(f"no figure matched {args.names}")

    rows = []
    for name in chosen:
        print(f"== {name}")
        spec = SPECS[name]
        rec = (search(name, spec) if args.search else regen.run_one(name, spec))[0]
        rows.append((name, spec, rec))

    if args.search:
        with open(STATE, "w") as fh:
            json.dump({n: {"w": s["w"], "h": s["h"]} for n, s in SPECS.items()},
                      fh, indent=2, sort_keys=True)
            fh.write("\n")

    print("\n" + "-" * 92)
    print(f"{'figure':<44}{'canvas in':>12}{'page pt':>14}{'graphic':>9}{'frac':>9}")
    print("-" * 92)
    for name, spec, rec in rows:
        g = rec["rendered_h_pt"]
        print(f"{name:<44}{spec['w']:>5.2f}x{spec['h']:<6.2f}"
              f"{rec['page_w_pt']:>7.0f}x{rec['page_h_pt']:<6.0f}{g:>9.1f}"
              f"{g / ds.TEXTHEIGHT_PT:>9.3f}")
    print("-" * 92)
    print(f"text height {ds.TEXTHEIGHT_PT:.1f} pt; inclusion caps the graphic at "
          f"0.83 x textheight = {0.83 * ds.TEXTHEIGHT_PT:.1f} pt")


if __name__ == "__main__":
    main()
