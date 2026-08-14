"""Check that a dissertation figure's type survives inclusion at >= 9 pt.

Appendix E includes every figure as

    \\includegraphics[width=\\textwidth,height=0.83\\textheight,keepaspectratio]

so `keepaspectratio` applies whichever of the two limits binds harder:

    scale = min(TEXTWIDTH / page_w,  0.83 x TEXTHEIGHT / page_h)

`diss_style.finish` records only the width ratio, which overstates the scale of
any figure the height clamps, and nothing in the manifest then reveals that a
generator's 9 pt type is printing at 8.4 pt.  This helper recomputes the honest
scale from the emitted PDF and aborts if the smallest type the generator draws
prints below the floor.

Geometry is the dissertation body's, `writing/dissertation/main.log` lines
760-761.  (The 714.164 pt \\textheight also in that log is the title page's
`\\newgeometry`, restored at `config.tex:65`; it is not the figures' geometry.)
"""
import os
import subprocess
import sys

TEXTWIDTH_PT = 455.24411
TEXTHEIGHT_PT = 717.00946
CAP_PT = 0.83 * TEXTHEIGHT_PT          # 595.118 pt of graphic
FLOOR_PT = 9.0


def page_size_pt(path):
    out = subprocess.run(["pdfinfo", path], capture_output=True, text=True,
                         check=True).stdout
    for line in out.splitlines():
        if line.startswith("Page size:"):
            w, _, h = line.split(":", 1)[1].split()[:3]
            return float(w), float(h)
    sys.exit(f"could not read a page size out of {path}")


def check(name, drawn_floor_pt=9.0, outdir=None):
    """Abort unless `name`.pdf prints its smallest type at >= 9 pt."""
    if outdir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.abspath(os.path.join(
            here, "..", "..", "..", "writing", "dissertation", "figures"))
    pdf = os.path.join(outdir, f"{name}.pdf")
    w, h = page_size_pt(pdf)
    sw, sh = TEXTWIDTH_PT / w, CAP_PT / h
    scale = min(sw, sh)
    printed = drawn_floor_pt * scale
    print(f"  inclusion: page {w:.1f}x{h:.1f} pt; width x{sw:.4f}, "
          f"height x{sh:.4f}; binding {'width' if sw < sh else 'height'} "
          f"x{scale:.4f}; {drawn_floor_pt:g} pt type prints {printed:.2f} pt")
    if printed < FLOOR_PT - 0.005:
        sys.exit(f"TYPE FLOOR FAIL — {name} prints {printed:.2f} pt, "
                 f"under the {FLOOR_PT:g} pt floor "
                 f"(page {w:.1f}x{h:.1f} pt; needs w <= {TEXTWIDTH_PT:.1f} "
                 f"and h <= {CAP_PT:.1f})")
    return scale
