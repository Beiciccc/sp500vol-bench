"""Check that every printed number can be found in the file its source comment names.

The source comments make the report auditable in principle; nothing so far has
audited them.  This does, mechanically: for each line carrying a `% src:` naming
a file under results/ or release/, it takes the numbers in the prose and asks
whether the named file contains a number that ROUNDS to each one.  Rounding is
the whole difficulty --- the report prints 0.51 where the table holds 0.5612,
and +1.21 per cent where the table holds 1.2134 --- so a match is any file value
whose rounding, at the printed number's own precision, equals the printed one.

Percentages are tried both ways (1.21 against 1.21 and against 0.0121), and a
printed integer is additionally allowed to match a count of rows rather than a
cell value, because many printed integers are denominators.

Output is a ranked report with a baseline, not a pass/fail gate at zero: a number
that fails to match is a lead to check by hand, not proof of an error.  Plenty of
legitimate numbers are derived (a sum, a median, a difference) and will not appear
literally anywhere.

A baseline with no record of what it covers is a hiding place, so the thirteen
that stand at the time of writing are enumerated here, each checked by hand:

  ch3:25, :26, :39   45, 42, 16, 41 -- percentages and per-year churn ranges
                     derived from crsp_restricted/membership_intervals.csv.  Recomputed
                     from that file: 411 of 914 intervals close inside the sample
                     (45.0%), entries run 18-42 and exits 16-41 per year over
                     2011-2025, month-end active membership 497-505.
  ch4:50             1.96 -- the standard normal critical value, not a table
                     value.
  ch4:230            21,200 -- its source is a parquet shard, which has no text
                     to scan.
  D:201              29,000 -- printed with a tilde as an approximation of the
                     29,278-token long-form median its source records.
  E:1661             92.6 -- its source is predictions.parquet, again binary.
  D:349              1.076655 -- a ratio between two prediction files, so it is
                     in neither.  Checked by hand: dividing the committed
                     prediction_realised_vol by the superseded one at h=20 gives
                     a value between 1.0766552112791599 and 1.07665521128133 on
                     every one of the 94,237 rows, which rounds to the printed
                     figure.  The two companion constants sit on the preceding
                     source line and are derived the same way.

If the count rises, the new entry is the one to look at.
"""
import argparse
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DISS = os.path.join(ROOT, "writing", "dissertation")

SRC = re.compile(r"""(?<![\\"])%\s*(?:caption\s+)?(?:numbers\s+)?src:\s*(.+)$""")
PATH = re.compile(r"(?:results|release|configs|scripts|src|data)/[\w./*+-]+"
                  r"|\b[\w.+-]+\.(?:csv|md|txt|json|tex|yaml|yml|py)\b")
# a printed number: optional sign, digits, optional {,} thousands, optional decimals
NUM = re.compile(r"[-+\u2212]?\d{1,3}(?:\{,\}\d{3})+(?:\.\d+)?"
                 r"|[-+\u2212]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
                 r"|[-+\u2212]?\d+(?:\.\d+)?")
# things that are not measurements
SKIP_CTX = re.compile(r"\\(?:ref|label|cite|eqref|autoref)\{[^}]*$")


def clean_prose(line):
    """The printable part of a source line, with LaTeX plumbing removed."""
    prose = SRC.split(line)[0]
    # a LaTeX en-dash range reads as a minus sign to a naive number scanner:
    # "1.02--1.17" would come out as 1.02 and -1.17
    prose = prose.replace("--", " ")
    prose = re.sub(r"\\(?:ref|label|cite\w*|eqref|autoref)\{[^}]*\}", " ", prose)
    prose = re.sub(r"\\[a-zA-Z]+", " ", prose)
    return prose


def numbers_in(text):
    """The measurements in a line, which is not the same as its digits.

    Arm codes (A3, C6, D4, S5), horizons written h=5, hyphenated names like
    Nasdaq-100 and citation years all put digits in a sentence without putting a
    measurement in it.  A digit glued to a letter or to a hyphen inside a word is
    a label, not a number this audit can check.
    """
    out = []
    for m in NUM.finditer(text):
        before = text[max(0, m.start() - 1):m.start()]
        if before and (before.isalpha() or before in "=-"):
            continue
        s = m.group(0).replace("{,}", "").replace(",", "").replace("\u2212", "-")
        try:
            v = float(s)
        except ValueError:
            continue
        if 1900 <= v <= 2100 and "." not in s:
            continue  # a year, not a measurement
        dec = len(s.split(".")[1]) if "." in s else 0
        out.append((m.group(0), v, dec))
    return out


def resolve(name, index={}):
    """Where a source comment's file name actually lives.

    Comments name a file three ways: by repository path, by a path with a glob
    left in, and -- often enough to matter -- by bare file name.  A marker would
    resolve the bare one by looking; so does this.
    """
    if os.path.isfile(os.path.join(ROOT, name)):
        return name
    if not index:
        import glob as _g
        roots = ["results", "release", "configs",
                 os.path.join("writing", "dissertation", "paper_full_sections"),
                 os.path.join("writing", "paper")]
        cands = []
        for r in roots:
            cands += _g.glob(os.path.join(ROOT, r, "**", "*"), recursive=True)
        cands += _g.glob(os.path.join(ROOT, "*.md"))
        for p in cands:
            if os.path.isfile(p):
                index.setdefault(os.path.basename(p), os.path.relpath(p, ROOT))
    return index.get(os.path.basename(name))


def file_numbers(path, cache={}):
    if path in cache:
        return cache[path]
    path = resolve(path) or path
    full = os.path.join(ROOT, path)
    vals = set()
    if os.path.isfile(full):
        try:
            txt = open(full, errors="replace").read()
        except OSError:
            txt = ""
        # comma-grouped integers: the markdown tables write 19,668 where the
        # report writes 19{,}668, and a plain scan reads that as 19 and 668
        # LaTeX sources -- including the frozen drafting sources several comments
        # name -- group thousands as 215{,}785, which a plain scan reads as 215
        for m in re.finditer(r"\d{1,3}(?:\{,\}\d{3})+", txt):
            try:
                vals.add(float(m.group(0).replace("{,}", "")))
            except ValueError:
                pass
        for m in re.finditer(r"\d{1,3}(?:,\d{3})+", txt):
            try:
                vals.add(float(m.group(0).replace(",", "")))
            except ValueError:
                pass
        for m in re.finditer(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", txt):
            try:
                vals.add(float(m.group(0)))
            except ValueError:
                continue
            # the report often prints the mantissa of a value the table stores in
            # scientific notation ("1.062e-05" printed as $1.062\times10^{-5}$)
            if "e" in m.group(0).lower():
                try:
                    vals.add(float(m.group(0).lower().split("e")[0]))
                except ValueError:
                    pass
        # A printed integer is very often not a value in the file but a count OF
        # the file: how many rows it has, or how many distinct firms, panels or
        # cells appear in one of its columns.  Without these the audit drowns in
        # false positives on exactly the counts the report leads with.
        vals.add(float(txt.count("\n")))
        vals.add(float(max(txt.count("\n") - 1, 0)))   # minus the header
        if path.endswith(".csv"):
            try:
                import csv as _csv
                import io as _io
                rows = list(_csv.reader(_io.StringIO(txt)))
                if rows:
                    hdr, body = rows[0], rows[1:]
                    vals.add(float(len(body)))
                    for j in range(len(hdr)):
                        col = {r[j] for r in body if len(r) > j}
                        vals.add(float(len(col)))
                        # counts of rows taking each value: "411 of the 914 close inside"
                        seen = {}
                        for r in body:
                            if len(r) > j:
                                seen[r[j]] = seen.get(r[j], 0) + 1
                        for c in set(seen.values()):
                            vals.add(float(c))
            except Exception:
                pass
    cache[path] = vals
    return vals


def matches(v, dec, vals):
    """Does any file value round to this printed one, at its own precision?"""
    for cand in (v, v / 100.0, v * 100.0):
        for f in vals:
            if round(f, dec) == round(cand, dec):
                return True
            # the printed number may itself be the rounding of a percentage
            if dec == 0 and abs(f - cand) < 0.5:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dec", type=int, default=0,
                    help="only audit numbers with at least this many decimals")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--max-miss", type=int, default=None,
                    help="exit non-zero if more numbers than this fail to match; "
                         "the baseline is not zero because a legitimately derived "
                         "number (a median, a sum, a percentage) is not in any file "
                         "literally, and two sources are binary parquet")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    import glob
    paths = a.files or (sorted(glob.glob(os.path.join(DISS, "chapters", "*.tex"))) +
                        sorted(glob.glob(os.path.join(DISS, "appendices", "*.tex"))) +
                        [os.path.join(DISS, "summary.tex")])
    miss, checked, nosrc = [], 0, 0
    for p in paths:
        if not os.path.isfile(p):
            continue
        lines = open(p).readlines()
        for i, line in enumerate(lines, 1):
            m = SRC.search(line)
            if not m:
                continue
            # a long source comment is continued on following lines that start
            # with "%" and carry no "src:" of their own; those name files too
            tail = m.group(1)
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("%") \
                    and not SRC.search(lines[j]):
                tail += " " + lines[j]
                j += 1
            # LaTeX escapes are still LaTeX inside a comment: TECHNICAL\_SUPPLEMENT.md
            # names a real file, and a scanner that does not undo the backslash
            # looks for _SUPPLEMENT.md instead
            tail = tail.replace("\\_", "_").replace("\\%", "%").replace("\\&", "&")
            names = PATH.findall(tail)
            if not names:
                nosrc += 1
                continue
            vals = set()
            for n in names:
                vals |= file_numbers(n)
            if not vals:
                continue
            for raw, v, dec in numbers_in(clean_prose(line)):
                if dec < a.min_dec:
                    continue
                checked += 1
                if not matches(v, dec, vals):
                    miss.append((os.path.basename(p), i, raw, names[0],
                                 " ".join(clean_prose(line).split())[:110]))

    print(f"  checked {checked} printed numbers against their named sources")
    print(f"  {len(miss)} not found in the file the comment names")
    print(f"  ({nosrc} source comments name no results/ or release/ path)\n")
    if not a.quiet:
        for f, i, raw, src, ctx in miss:
            print(f"  {f}:{i}  {raw:<12} not in {src}")
            print(f"      {ctx}")
    if a.max_miss is not None and len(miss) > a.max_miss:
        print(f"\n  GATE FAIL: {len(miss)} unmatched numbers exceeds the baseline of "
              f"{a.max_miss}; a printed number no longer traces to the file its "
              f"source comment names")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
