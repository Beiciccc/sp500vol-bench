"""Measure what every dissertation figure's type actually prints at.

Why this exists
---------------
`diss_style.finish` records an `inclusion_scale` in
`writing/dissertation/figures/_geometry_manifest.json`, but it computes that
scale from the page *width* alone:

    inclusion_scale = TEXTWIDTH_PT / page_w_pt

Every figure in the report is included as

    \\includegraphics[width=\\textwidth,height=0.83\\textheight,keepaspectratio]

so `keepaspectratio` applies whichever of the two constraints binds harder, and
for a tall figure that is the height.  The recorded number therefore overstates
the scale of every height-clamped figure -- F2 is recorded at 0.9879 when the
graphic is actually set at 0.9408 -- and a figure whose type is drawn at 9 pt
prints below 9 pt without anything in the manifest saying so.

This script recomputes the geometry honestly from the emitted PDFs and writes
the corrected fields back into the manifest.  It reads; it never draws, so it
cannot disturb a figure another regeneration is mid-way through.

Geometry (dissertation body, `main.log` lines 760-761; the 714.164 pt
\\textheight also in that log belongs to the title page's `\\newgeometry` and is
restored at `config.tex:65`, so it is not the figures' geometry):

    \\textwidth   455.24411 pt
    \\textheight  717.00946 pt
    height cap   0.83 x \\textheight = 595.11785 pt

Fields added per figure:

    page_w_pt / page_h_pt      re-read from the PDF now on disk
    inclusion_scale_true       min(TW / w, CAP / h)
    binding_constraint         "width" or "height"
    printed_graphic_h_pt       height the graphic occupies on the page
    drawn_type_floor_pt        smallest type the generator sets (see FLOOR)
    printed_type_floor_pt      drawn floor x inclusion_scale_true

Usage
-----
    python3 audit_inclusion_geometry.py            # report only
    python3 audit_inclusion_geometry.py --write    # also repair the manifest
"""
import argparse
import json
import os
import subprocess
import sys

REPO = "."
FIGDIR = os.path.join(REPO, "writing", "dissertation", "figures")
MANIFEST = os.path.join(FIGDIR, "_geometry_manifest.json")

TEXTWIDTH_PT = 455.24411
TEXTHEIGHT_PT = 717.00946
CAP_PT = 0.83 * TEXTHEIGHT_PT

# Smallest type each generator sets, in points on its own canvas.  The five
# reused supplement generators run `supp_style.apply_style(9)` and set no
# explicit `fontsize=` below 9; the fifteen new ones pass their own base size to
# the same call.  Read from the generator, not guessed.
FLOOR = {}

# Appendix E, in the order the chapter includes them.
APPENDIX_E = [
    "F2_model_spectrum_and_compute",
    "FP4_compute_accuracy",
    "F5_maximal_pool_audit",
    "AR1_reference_spec_cell_matrix",
    "AR2_price_frontier_completeness",
    "F6_firm_identity_rung",
    "AR3_matched_row_cascade",
    "AF1_item_geography_8k",
    "AF2_longform_section_geography",
    "AF3_stratified_increment_map",
    "stability_quarter_by_quarter",
    "stability_freeze_point_deployability",
    "stability_weight_window_sensitivity",
    "stability_seed_dispersion",
    "F13_elicitation_not_curation",
    "FP1_prompted_family_panel",
    "F10_encompassing_pooled_power",
    "FP3_joint_spa_mcs",
    "AR4_public_label_variant",
    "FP2_yelp_ladder",
]
APPENDIX_C = ["AP1_label_proxy_cascade"]
MAIN_TEXT = [
    "F1_membership_panel",
    "F3_standalone_180_and_reference_strength",
    "F4_ladder_cell_matrix",
    "F7_anonymisation_price",
    "F8_matched_firm_swap",
    "F14_economic_adjudication",
    "F9_power_mde_label_noise",
    "F11_ladder_perturbations",
    "F12_health_screen_orthogonality",
    "F15_yelp_portability",
    "F16_maec_reproduce_reprice",
]

SUPP_SRC = os.path.join(REPO, "scripts", "analysis", "supp_figs")
NEW_SRC = os.path.dirname(os.path.abspath(__file__))


def drawn_floor(name):
    """Smallest point size the generator sets, read out of its source."""
    import re
    # 查找顺序要紧,而且这里曾经是错的。F2/F5/F6/F10/F13 的真实生成器是
    # <name>_diss.py;只找 <name>.py 会命中 supp_figs/ 里论文附录的同名源文件,
    # 于是这道闸给恰好是全部麻烦来源的那五张图读的是错的文件。_diss 优先。
    for d, stem in ((NEW_SRC, f"{name}_diss.py"), (NEW_SRC, f"{name}.py"),
                    (SUPP_SRC, f"{name}.py")):
        p = os.path.join(d, stem)
        if os.path.exists(p):
            with open(p) as fh:
                src = fh.read()
            # size= 也要认:supp_style.annot()/note() 用 size= 传字号,只认
            # fontsize=/labelsize= 会让 annot(..., size=8.6) 对这道闸完全隐形。
            sizes = [float(m) for m in
                     re.findall(r"\b(?:fontsize|labelsize|size)="
                                r"([0-9]+(?:\.[0-9]+)?)", src)]
            m = re.search(r"apply_style\(\s*(?:base_size\s*=\s*)?([0-9.]+)?\s*\)",
                          src)
            if m:
                sizes.append(float(m.group(1)) if m.group(1) else 9.0)
            return min(sizes) if sizes else None
    return None


def page_size(name):
    path = os.path.join(FIGDIR, f"{name}.pdf")
    if not os.path.exists(path):
        return None
    out = subprocess.run(["pdfinfo", path], capture_output=True, text=True,
                         check=True).stdout
    for line in out.splitlines():
        if line.startswith("Page size:"):
            body = line.split(":", 1)[1].split()
            return float(body[0]), float(body[2])
    return None


def measure(name):
    wh = page_size(name)
    if wh is None:
        return None
    w, h = wh
    sw, sh = TEXTWIDTH_PT / w, CAP_PT / h
    scale = min(sw, sh)
    floor = drawn_floor(name)
    return {
        "figure": name,
        "page_w_pt": round(w, 2),
        "page_h_pt": round(h, 2),
        "inclusion_scale_true": round(scale, 4),
        "binding_constraint": "width" if sw <= sh else "height",
        "printed_graphic_h_pt": round(h * scale, 2),
        "drawn_type_floor_pt": floor,
        "printed_type_floor_pt": round(floor * scale, 2) if floor else None,
    }


def report(title, names, rows):
    print(f"\n{title}")
    print("-" * 108)
    print(f"{'figure':<42}{'native pt':>16}{'binds':>8}{'scale':>9}"
          f"{'drawn':>8}{'final':>8}{'graphic pt':>12}")
    print("-" * 108)
    for n in names:
        r = rows.get(n)
        if r is None:
            print(f"{n:<42}{'MISSING':>16}")
            continue
        flag = "  <-- under 9 pt" if (r["printed_type_floor_pt"] or 9) < 9 else ""
        print(f"{n:<42}{r['page_w_pt']:>7.1f}x{r['page_h_pt']:<8.1f}"
              f"{r['binding_constraint']:>8}{r['inclusion_scale_true']:>9.4f}"
              f"{r['drawn_type_floor_pt']:>8.1f}{r['printed_type_floor_pt']:>8.2f}"
              f"{r['printed_graphic_h_pt']:>12.1f}{flag}")
    print("-" * 108)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the corrected geometry back into the manifest")
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero if any Appendix E float prints under 9 pt. "
                         "Run this after ANY figure regeneration: the failure mode "
                         "it catches is silent, because a figure regenerated by the "
                         "wrong driver still renders and still builds.")
    args = ap.parse_args()

    allnames = APPENDIX_E + APPENDIX_C + MAIN_TEXT
    rows = {}
    for n in allnames:
        m = measure(n)
        if m:
            rows[n] = m

    report("Appendix E (20 floats)", APPENDIX_E, rows)
    report("Appendix C", APPENDIX_C, rows)
    report("Chapters 3-5 (frozen: the 60-page cap forbids re-geometry)",
           MAIN_TEXT, rows)

    e_floor = min(rows[n]["printed_type_floor_pt"] for n in APPENDIX_E
                  if n in rows)
    all_floor = min(rows[n]["printed_type_floor_pt"] for n in rows)
    print(f"\nsmallest type printed in Appendix E : {e_floor:.2f} pt")
    print(f"smallest type printed in the report  : {all_floor:.2f} pt")
    under_e = [n for n in APPENDIX_E
               if n in rows and rows[n]["printed_type_floor_pt"] < 9]
    print(f"Appendix E floats under 9 pt         : {len(under_e)} "
          f"({', '.join(under_e) if under_e else 'none'})")

    if args.gate:
        if under_e:
            sys.exit("GATE FAILED: Appendix E floats printing under 9 pt: "
                     + ", ".join(under_e)
                     + "\nThe usual cause is a figure regenerated by the wrong "
                       "driver: five of these are owned by diss_appendix_figs/"
                       "<name>_diss.py, not by diss_figs/regen_appendix.py. "
                       "See NOT_OURS in that driver.")
        print("GATE PASSED: no Appendix E float prints under 9 pt.")

    if args.write:
        data = {}
        if os.path.exists(MANIFEST):
            with open(MANIFEST) as fh:
                data = json.load(fh)
        for n, r in rows.items():
            rec = data.get(n, {"figure": n})
            rec.update(r)
            rec["inclusion_scale_note"] = (
                "inclusion_scale is width-only and is what diss_style.finish "
                "records; inclusion_scale_true also applies the "
                "height=0.83\\textheight clamp, which is what keepaspectratio "
                "actually uses. Re-run "
                "scripts/analysis/diss_appendix_figs/audit_inclusion_geometry.py "
                "--write after any regeneration.")
            data[n] = rec
        with open(MANIFEST, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote corrected geometry for {len(rows)} figures into "
              f"{os.path.relpath(MANIFEST, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
