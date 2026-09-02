"""Find floats whose caption prints a number with no source comment anywhere in it.

Every printed number in this report is meant to carry a same-line `% src:`
comment naming where it was read from, and `src_number_audit.py` checks those
against their sources. Neither notices the case where a caption states a
measured quantity and the float carries no source comment at all: there is
nothing to check, so nothing complains.

That is the gap this closes. It cannot compare a caption against its drawing --
the figure PDFs carry no text layer, every glyph having been converted to a
vector path -- so it asks the one question that can be answered from the sources:
does this number have a provenance trail at all?
"""
import glob
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DISS = os.path.join(ROOT, "writing", "dissertation")

SRC = re.compile(r"""(?<![\\"])%\s*(?:caption\s+)?(?:numbers\s+)?src:""")
NUM = re.compile(r"[-+\u2212]?\d{1,3}(?:\{,\}\d{3})+(?:\.\d+)?|[-+\u2212]?\d+(?:\.\d+)?")


def floats(src):
    """(kind, label, caption text, whole float body) for every float."""
    out = []
    for m in re.finditer(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", src, flags=re.S):
        body = m.group(0)
        c = re.search(r"\\caption(?:\[[^\]]*\])?\{", body)
        if not c:
            continue
        i, depth = c.end(), 1
        while i < len(body) and depth:
            if body[i] == "{" and body[i - 1] != "\\":
                depth += 1
            elif body[i] == "}" and body[i - 1] != "\\":
                depth -= 1
            i += 1
        lab = re.search(r"\\label\{([^}]+)\}", body)
        out.append((m.group(1), lab.group(1) if lab else "?", body[c.end():i - 1], body))
    return out


def clean(cap):
    cap = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in cap.split("\n"))
    cap = re.sub(r"\\(?:ref|label|cite\w*|eqref)\{[^}]*\}", " ", cap)
    cap = re.sub(r"\\[^a-zA-Z\s]", " ", cap)
    cap = re.sub(r"\\[a-zA-Z]+", " ", cap)
    return re.sub(r"[{}$]", " ", cap)


def measured(cap):
    """The numbers in a caption that look like measurements, not labels."""
    out = []
    for m in NUM.finditer(clean(cap)):
        s = m.group(0).replace("{,}", "").replace("\u2212", "-").lstrip("+")
        try:
            v = float(s)
        except ValueError:
            continue
        before = clean(cap)[max(0, m.start() - 1):m.start()]
        if before and (before.isalpha() or before in "=-"):
            continue                          # an arm code, h=5, a hyphenated name
        if 1900 <= v <= 2100 and "." not in s:
            continue                          # a year
        if abs(v) < 3 and "." not in s:
            continue                          # a panel index, a small count of panels
        out.append(s)
    return list(dict.fromkeys(out))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None,
                    help="exit non-zero above this many; the baseline is not zero "
                         "because a caption may legitimately restate a denominator "
                         "or a design constant declared in Chapter 3")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    bad, n_floats, n_nums = [], 0, 0
    for p in (sorted(glob.glob(os.path.join(DISS, "chapters", "*.tex"))) +
              sorted(glob.glob(os.path.join(DISS, "appendices", "*.tex")))):
        for kind, lab, cap, body in floats(open(p).read()):
            n_floats += 1
            nums = measured(cap)
            n_nums += len(nums)
            if nums and not SRC.search(body):
                bad.append((os.path.relpath(p, ROOT), kind, lab, nums[:8]))
    print(f"  {n_floats} floats, {n_nums} measured numbers in captions")
    print(f"  {len(bad)} float(s) print a number with no source comment anywhere inside\n")
    if not a.quiet:
        for f, kind, lab, nums in bad:
            print(f"  FAIL {f}  {kind} {lab}")
            print(f"       quotes {', '.join(nums)} and carries no % src:")
    if a.max is not None and len(bad) > a.max:
        print(f"\n  GATE FAIL: {len(bad)} floats print a number with no provenance, "
              f"above the baseline of {a.max}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
