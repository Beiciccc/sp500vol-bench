"""Mechanical reconciliation of every appendix table's PRINTED values against the
fields of the evidence file its own "% src:" comment names.

Two independent nets, because they catch different failures:

  NET 1 -- FIELD INFERENCE.  For each printed numeric column, find which field of
  the cited evidence file best covers its values (order-independent: what fraction
  of the column's printed values appear somewhere in that field, at the precision
  the table prints).  Then compare the winning field's NAME against the column
  HEADER.  This is the net that catches the Tables C.31/C.32 defect, where the
  column headed "Holm" was populated from fixed_p_clu: the value is in the file,
  so an existence check passes, but the winning field is named p_clu, not holm.

  NET 2 -- EXISTENCE.  Any printed value that appears in NO field of any cited
  file, at its own printed precision.  Catches transcription errors and drift.

Neither net is a proof: a low-coverage column may simply be derived rather than
transcribed, and a header/field name clash may be a synonym.  The output is a
ranked worklist for adjudication, not a verdict.

Usage:  python3 reconcile.py            # full run, writes a report
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

ROOT = "."
DISS = os.path.join(ROOT, "writing/dissertation")
SEARCH = ["", "results/tables/", "results/", "release/", "writing/paper/supplementary/"]

# Header token -> field-name tokens that would be a semantically CONSISTENT source.
# Used only to decide whether a winning field's name contradicts the header.
CONSISTENT = {
    "holm": ["holm"],
    "p": ["_p", "p_", "pval", "p_clu", "pv"],
    "dm": ["dm"],
    "rel": ["rel", "impr"],
    "n": ["n_", "count", "size"],
    "qlike": ["qlike"],
    "share": ["share", "frac", "pct"],
    "plac": ["placebo", "plac"],
}


def load_fields(path):
    """Return {field_name: [floats]} for a .csv or a markdown pipe table."""
    out = defaultdict(list)
    if path.endswith(".csv"):
        try:
            rows = list(csv.DictReader(open(path)))
        except Exception:
            return {}
        for r in rows:
            for k, v in r.items():
                if k is None or v is None:
                    continue
                try:
                    out[k].append(float(v))
                except (TypeError, ValueError):
                    pass
    else:  # markdown: every pipe table in the file
        header = None
        for line in open(path, errors="replace"):
            if line.count("|") < 2:
                header = None
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if header is None:
                header = [c.strip("* `") for c in cells]
                continue
            for k, v in zip(header, cells):
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v.replace(",", ""))
                if m:
                    try:
                        out[k].append(float(m.group()))
                    except ValueError:
                        pass
    return {k: v for k, v in out.items() if v}


def resolve(tok):
    tok = tok.strip().strip("`,;").replace("\\_", "_").replace("\\allowbreak", "")
    for pre in SEARCH:
        p = os.path.join(ROOT, pre, tok)
        if os.path.isfile(p):
            return p
        p2 = os.path.join(ROOT, pre, os.path.basename(tok))
        if os.path.isfile(p2):
            return p2
    return None


def strip_tex(c):
    c = re.sub(r"\\(?:textbf|emph|texttt|mathbf|text)\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\[a-zA-Z]+", " ", c)
    c = c.replace("$", "").replace("{", "").replace("}", "").replace("~", " ")
    return c.strip()


# Leading-dot decimals (".0357") are the house style for p-values here. A pattern
# requiring a leading digit silently reads ".0357" as 357, which inflated the
# orphan count badly on first run.
NUMRE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


SCI = re.compile(r"([-+]?\d*\.?\d+)\s*\\times\s*10\^\{?\s*(-?\d+)\s*\}?")
LT = re.compile(r"\$?<\$?\s*\.?(\d+)")


DATEISH = re.compile(r"\d{4}-\d{2}(-\d{2})?")
MODELISH = re.compile(r"[A-Za-z]-?\d+(\.\d+)?-\d+[BbMm]?")


def cell_numbers(c):
    """Numbers printed in a cell, with the decimal precision each was printed at.

    Handles three LaTeX forms a naive scanner gets wrong, and getting them wrong
    made this reconciler report correct tables as unmatched:
      a\\times10^{-b}  -> a*10**-b, significant to a's decimals + |b|
      $<$.0001         -> the threshold, matched as "file value below it"
    """
    out = []
    raw = DATEISH.sub(" ", MODELISH.sub(" ", c))
    for m in SCI.finditer(raw):
        mant, ex = m.group(1), int(m.group(2))
        dec = (len(mant.split(".")[1]) if "." in mant else 0) + abs(ex)
        try:
            out.append((float(mant) * (10.0 ** ex), dec))
        except (ValueError, OverflowError):
            pass
    if out:
        return out
    m = LT.search(raw)
    if m and "<" in raw:
        thr = float("0." + m.group(1)) if not m.group(1).startswith("0") or True else 0.0
        return [(-abs(thr), -1)]  # sentinel: "strictly less than thr"
    for m in NUMRE.finditer(strip_tex(raw)):
        t = m.group().replace(",", "")
        if t in ("-", "+", ""):
            continue
        try:
            v = float(t)
        except ValueError:
            continue
        dec = len(t.split(".")[1]) if "." in t else 0
        out.append((v, dec))
    return out


def matches(v, dec, vals):
    """Does some file value round to the printed value at its printed precision?"""
    if dec == -1:  # printed as "$<$ thr": any file value strictly below thr matches
        thr = abs(v)
        return any(0 <= f < thr for f in vals)
    for f in vals:
        try:
            if round(f, dec) == round(v, dec):
                return True
            if dec and abs(f - v) <= 0.5 * 10 ** (-dec):
                return True
        except (ValueError, OverflowError):
            pass
    return False


def parse_tables(path):
    s = open(path).read()
    out = []
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", s, re.S):
        blk = m.group(1)
        lab = re.search(r"\\label\{([^}]*)\}", blk)
        srcs = []
        for sm in re.findall(r"%\s*(?:caption numbers\s*)?src:\s*([^\n]*)", blk):
            srcs += re.findall(r"[\w./\\-]+\.(?:csv|md|json)", sm)
        tm = re.search(r"\\begin\{tabular\}(?:\[[^\]]*\])?\{[^}]*\}(.*?)\\end\{tabular\}", blk, re.S)
        if not tm:
            continue
        body = tm.group(1)
        lines = [l for l in body.split("\\\\") if l.strip()]
        hdr, data = None, []
        for l in lines:
            clean = re.sub(r"\\(?:toprule|midrule|bottomrule|addlinespace|cmidrule\(?[^)]*\)?\{[^}]*\})", " ", l)
            if "&" not in clean:
                continue
            cells = [c.strip() for c in clean.split("&")]
            joined = strip_tex(clean)
            if hdr is None and not NUMRE.search(joined):
                hdr = [strip_tex(c) for c in cells]
                continue
            if hdr is not None:
                data.append(cells)
        out.append(dict(file=os.path.basename(path), label=lab.group(1) if lab else "?",
                        srcs=srcs, header=hdr or [], rows=data))
    return out


def header_consistent(header, field):
    h, f = header.lower(), field.lower()
    for key, toks in CONSISTENT.items():
        if key in h:
            # header names a known quantity: the field should look like it
            if any(t in f for t in toks):
                return True
            # a p-column sourced from a holm field, or vice versa, is the red flag
            other = [k for k in CONSISTENT if k != key and any(t in f for t in CONSISTENT[k])]
            if other:
                return False
            return None  # unknown, do not judge
    return None


def main():
    tabs = []
    for fn in ("A_external_materials.tex", "C_full_results.tex", "D_hyperparams.tex"):
        tabs += parse_tables(os.path.join(DISS, "appendices", fn))

    cache = {}
    flags, orphans, summary = [], [], []
    for t in tabs:
        fields = {}
        resolved, unresolved = [], []
        for s in t["srcs"]:
            p = resolve(s)
            if not p:
                unresolved.append(s)
                continue
            resolved.append(os.path.relpath(p, ROOT))
            if p not in cache:
                cache[p] = load_fields(p)
            for k, v in cache[p].items():
                fields.setdefault(f"{os.path.basename(p)}::{k}", v)
        allvals = [x for v in fields.values() for x in v]
        ncols = max((len(r) for r in t["rows"]), default=0)
        col_report, n_orphan, n_num = [], 0, 0
        for j in range(ncols):
            printed = []
            for r in t["rows"]:
                if j < len(r):
                    printed += cell_numbers(r[j])
            if len(printed) < 3:
                continue
            n_num += len(printed)
            best, bestcov = None, 0.0
            for k, vals in fields.items():
                cov = sum(1 for v, d in printed if matches(v, d, vals)) / len(printed)
                if cov > bestcov:
                    best, bestcov = k, cov
            miss = [v for v, d in printed if not matches(v, d, allvals)]
            n_orphan += len(miss)
            head = t["header"][j] if j < len(t["header"]) else f"col{j}"
            if best and bestcov >= 0.6:
                verdict = header_consistent(head, best.split("::")[-1])
                if verdict is False:
                    flags.append(dict(table=t["label"], file=t["file"], col=j, header=head,
                                      winning_field=best, coverage=round(bestcov, 3),
                                      n=len(printed)))
            col_report.append((head, best, round(bestcov, 2), len(printed)))
        if n_orphan:
            orphans.append(dict(table=t["label"], file=t["file"], n_orphan=n_orphan,
                                n_numbers=n_num, srcs=resolved, unresolved=unresolved))
        summary.append(dict(table=t["label"], srcs=resolved, unresolved=unresolved,
                            nrows=len(t["rows"]), ncols=ncols, cols=col_report))

    out = dict(n_tables=len(tabs), header_field_flags=flags, orphan_tables=orphans,
               summary=summary)
    with open("/tmp/reconcile_report.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"appendix tables parsed: {len(tabs)}")
    print(f"\n=== NET 1: columns whose winning field CONTRADICTS the header ({len(flags)}) ===")
    for fl in flags:
        print(f"  {fl['table']:28s} col{fl['col']:<2} header={fl['header']!r:16s} "
              f"-> {fl['winning_field']}  cov={fl['coverage']} n={fl['n']}")
    print(f"\n=== NET 2: tables with printed numbers absent from every cited source ===")
    orphans.sort(key=lambda d: -d["n_orphan"])
    for o in orphans[:20]:
        print(f"  {o['table']:28s} {o['n_orphan']:4d}/{o['n_numbers']:4d} orphan  "
              f"srcs={o['srcs'] or o['unresolved']}")
    print(f"\n  ({len(orphans)} tables have at least one orphan; full detail in "
          f"/tmp/reconcile_report.json)")


if __name__ == "__main__":
    main()
