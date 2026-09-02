#!/usr/bin/env python3
"""Extract the 60-page main body of the dissertation as a standalone PDF.

The examiner-facing copy of the main text lives outside the repository, and it
has to be regenerated whenever the body changes.  Doing that by hand means
retyping a page range, and the range is not stable: it is an offset into
main.pdf that moves whenever the front matter does.  Two numbers therefore come
from the build's own artefacts rather than from memory:

  offset  the printed-to-PDF page offset, recovered by finding the PDF page
          whose running header reads "CHAPTER 1" and reading the printed page
          number off that same header.  main.pdf's front matter is numbered in
          roman, so printed page 1 sits at PDF page 1 + offset.
  last    the last printed page of the body, taken as the References entry in
          main.toc minus one -- the same rule the build script gates on.

Both are asserted against the 60-page cap before anything is written, so a
front-matter change that silently shifts the window fails here instead of
producing a plausible-looking PDF that starts mid-sentence.

Usage:  python3 scripts/release/export_main_text.py [--out PATH]
Exit code non-zero == the window could not be established, or is not 60 pages.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISS = os.path.join(ROOT, "writing", "dissertation")
SRC = os.path.join(DISS, "main.pdf")
DEFAULT_OUT = "dist/SP500VolBench_dissertation_main_text.pdf"
CAP = 60


def page_text(pdf, page):
    return subprocess.run(["pdftotext", "-layout", "-f", str(page), "-l", str(page), pdf, "-"],
                          capture_output=True, text=True).stdout


def n_pages(pdf):
    out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    return int(out.split("Pages:")[1].split()[0])


def printed_offset(pdf):
    """PDF page of printed page 1, from the first CHAPTER running header."""
    for p in range(1, min(n_pages(pdf), 40) + 1):
        head = page_text(pdf, p).split("\n")[0]
        if "CHAPTER" not in head:
            continue
        nums = re.findall(r"\b(\d{1,3})\b", head)
        if not nums:
            continue
        return p - int(nums[-1])
    raise SystemExit("FAIL: no CHAPTER running header found; cannot fix the page offset")


def body_last_page():
    toc = open(os.path.join(DISS, "main.toc")).read()
    m = re.search(r"contentsline \{chapter\}\{References\}\{(\d+)\}", toc)
    if not m:
        raise SystemExit("FAIL: no References entry in main.toc; cannot fix the body's last page")
    return int(m.group(1)) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not os.path.isfile(SRC):
        raise SystemExit(f"FAIL: {SRC} not found -- build the dissertation first")

    off = printed_offset(SRC)
    last = body_last_page()
    first_pdf, last_pdf = 1 + off, last + off
    span = last_pdf - first_pdf + 1
    print(f"  printed 1..{last} = PDF {first_pdf}..{last_pdf}  (offset +{off}, {span} pages)")
    if last > CAP:
        raise SystemExit(f"FAIL: body is {last} printed pages, over the {CAP}-page cap")
    if span != CAP:
        raise SystemExit(f"FAIL: window is {span} pages, expected {CAP}")

    # The window must open on the chapter it claims to, or the offset is wrong
    # in a way the arithmetic above cannot see.
    head = page_text(SRC, first_pdf)
    if "Chapter 1" not in head and "Introduction" not in head:
        raise SystemExit(f"FAIL: PDF page {first_pdf} does not open Chapter 1")

    from pypdf import PdfReader, PdfWriter
    reader, writer = PdfReader(SRC), PdfWriter()
    for p in range(first_pdf - 1, last_pdf):
        writer.add_page(reader.pages[p])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        writer.write(fh)
    print(f"  -> {args.out}  ({n_pages(args.out)} pages, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
