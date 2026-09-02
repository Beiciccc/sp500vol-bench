#!/usr/bin/env python3
"""Build main.pdf (and the Reproducibility Checklist) behind hard gates.

Written after an adversarial audit found three literal ``\\n`` strings that a
Python replacement had injected into 10_limitations.tex. They raised three
"Undefined control sequence" errors, LaTeX carried on, and a broken paragraph
shipped -- because the build was a bare ``latexmk`` run whose log nobody read.

Every check below is one I had been doing by hand and would eventually skip.
Hard gates exit non-zero:

  log        no TeX errors, no undefined control sequences (this is what a
             stray Python escape becomes), no undefined references or
             citations, no missing characters, no overfull boxes
  rendering  no ``??`` in the extracted text, no Type 3 fonts, all fonts embedded
  budget     at most 7 pages of content, references may run to page 9

Underfull boxes are reported but do not fail: they are looseness, not error.

Usage:
    python3 scripts/release/build_paper.py            # build + gate
    python3 scripts/release/build_paper.py --check    # gate the existing build
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = ROOT / "writing/paper"
MAIN = "main"
CHECKLIST = "ReproducibilityChecklist"
CONTENT_PAGE_MAX = 7          # template rule: content may not pass page 7
TOTAL_PAGE_MAX = 9            # references may run to page 9

def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def scrub_figures() -> None:
    """Included figure PDFs travel into main.pdf as PTEX.InfoDict objects, so a
    figure's own CreationDate/Title (drawio's 'export3.html', matplotlib's
    Creator, all with the +01'00' stamp) ends up embedded in the paper. Strip
    each figure's Info before latexmk includes it; idempotent, content-only
    rewrite."""
    from pypdf import PdfReader, PdfWriter
    for fig in sorted((PAPER / "figures").glob("*.pdf")):
        reader = PdfReader(str(fig))
        md = reader.metadata or {}
        if not any(k in md for k in ("/CreationDate", "/ModDate", "/Title", "/Creator")):
            continue
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        tmp = fig.with_suffix(".scrub")
        with open(tmp, "wb") as fh:
            writer.write(fh)
        tmp.replace(fig)
        print(f"figure metadata stripped: {fig.name}")


def build(stem: str) -> None:
    scrub_figures()
    r = run(["latexmk", "-pdf", "-interaction=nonstopmode", f"{stem}.tex"], PAPER)
    if not (PAPER / f"{stem}.pdf").exists():
        print(r.stdout[-3000:])
        sys.exit(f"BUILD FAILED: {stem}.pdf was not produced")
    scrub_metadata(stem)


def scrub_metadata(stem: str) -> None:
    """Drop CreationDate/ModDate (they carried a +01'00' BST stamp -- a weak
    geographic signal under double blind), the PTEX.Fullbanner TeX Live banner,
    and Creator/Producer. The kit-required /TemplateVersion is re-added; pypdf
    copies page content verbatim so fonts and layout are untouched, and the
    hard gates below run on the scrubbed file."""
    from pypdf import PdfReader, PdfWriter
    pdf = PAPER / f"{stem}.pdf"
    reader = PdfReader(str(pdf))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/TemplateVersion": "2027.1"})
    tmp = PAPER / f"_{stem}_scrub.pdf"
    with open(tmp, "wb") as fh:
        writer.write(fh)
    tmp.replace(pdf)


def gate_log(stem: str) -> tuple[list[str], int]:
    log = (PAPER / f"{stem}.log").read_text(errors="ignore")
    lines = log.splitlines()
    fails = []
    checks = [
        ("TeX error", r"^! "),
        ("undefined control sequence", r"Undefined control sequence"),
        ("undefined reference", r"Reference `[^']*' on page \d+ undefined"),
        ("undefined citation", r"Citation `[^']*' on page \d+ undefined"),
        ("missing character", r"Missing character:"),
        ("overfull box", r"Overfull \\[hv]box"),
    ]
    for name, pat in checks:
        hits = [l for l in lines if re.search(pat, l)]
        if hits:
            fails.append(f"{stem}.log: {len(hits)} {name}(s); first: {hits[0][:110]}")
    underfull = sum(1 for l in lines if re.search(r"Underfull \\[hv]box", l))
    return fails, underfull


def gate_pdf(stem: str) -> list[str]:
    pdf = PAPER / f"{stem}.pdf"
    fails = []
    txt = run(["pdftotext", str(pdf), "-"], PAPER).stdout
    if "??" in txt:
        fails.append(f"{stem}.pdf: unresolved cross-reference rendered as '??'")
    fonts = run(["pdffonts", str(pdf)], PAPER).stdout.splitlines()
    if any("Type 3" in l for l in fonts):
        fails.append(f"{stem}.pdf: Type 3 font present (template requires Type 1/TrueType)")
    for l in fonts[2:]:
        if l.strip() and l.split()[-4:-3] == ["no"]:
            fails.append(f"{stem}.pdf: non-embedded font {l.split()[0]}")
    return fails


def gate_budget() -> list[str]:
    pdf = PAPER / f"{MAIN}.pdf"
    info = run(["pdfinfo", str(pdf)], PAPER).stdout
    pages = int(re.search(r"^Pages:\s*(\d+)", info, re.M).group(1))
    fails = []
    if pages > TOTAL_PAGE_MAX:
        fails.append(f"main.pdf: {pages} pages, the template allows {TOTAL_PAGE_MAX}")
    # Content ends where the References heading is. If the heading is NOT the
    # first thing on its page, that page still carries content, so the content
    # end is that page -- not the page before it. An earlier version of this
    # gate only asked which page the heading was on, and passed a build with
    # four lines of Conclusion sitting above it on page 8.
    refs_page, refs_opens_page = None, False
    for p in range(1, pages + 1):
        txt = run(["pdftotext", "-f", str(p), "-l", str(p), str(pdf), "-"], PAPER).stdout
        lines = [l for l in txt.splitlines() if l.strip()]
        if any(l.strip() == "References" for l in lines):
            refs_page = p
            refs_opens_page = lines[0].strip() == "References"
            break
    if refs_page is None:
        fails.append("main.pdf: the References heading was not found on any page")
    else:
        content_end = refs_page - 1 if refs_opens_page else refs_page
        if content_end > CONTENT_PAGE_MAX:
            where = ("References open page {}".format(refs_page) if refs_opens_page
                     else "content runs above the References heading on page {}".format(refs_page))
            fails.append(f"main.pdf: content ends on page {content_end} ({where}); "
                         f"the template allows {CONTENT_PAGE_MAX}")
    return fails, pages, refs_page


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="gate the existing build without recompiling")
    args = ap.parse_args()

    if not args.check:
        for stem in (MAIN, CHECKLIST):
            build(stem)

    fails: list[str] = []
    underfull = {}
    for stem in (MAIN, CHECKLIST):
        f, u = gate_log(stem)
        fails += f
        underfull[stem] = u
        fails += gate_pdf(stem)
    budget_fails, pages, refs_page = gate_budget()
    fails += budget_fails

    print(f"main.pdf: {pages} pages; References heading on page {refs_page}")
    print(f"underfull boxes (looseness, not gated): "
          + ", ".join(f"{k} {v}" for k, v in underfull.items()))

    if fails:
        print()
        for f in fails:
            print(f"  FAIL  {f}")
        return 1
    print("all gates green: no TeX errors, no undefined sequences/refs/citations, "
          "no missing characters, no overfull boxes, no '??', no Type 3, "
          "all fonts embedded, page budget met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
