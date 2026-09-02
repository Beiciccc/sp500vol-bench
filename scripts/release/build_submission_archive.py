#!/usr/bin/env python3
"""Stage and gate the code-and-data archive lodged with the dissertation.

Appendix A tells the marker that the archive is the authoritative copy and that
nothing in Sections A.1-A.6 depends on a resource they cannot open.  Two rules
follow, and neither survives being remembered rather than enforced:

  1. Working notes written for the companion manuscript must not travel.  Those
     files are not scrubbed -- rewriting a strategy document to hide what it was
     for is falsification -- they are simply excluded, and excluded by name here
     so the exclusion is auditable.
  2. Nothing that does travel may name a publication venue or a review process.

The scan is the point.  It runs over the staged tree, not over an author's
intentions, and a single hit aborts the build.  Run it before lodging anything:

    python3 scripts/release/build_submission_archive.py            # stage + gate
    python3 scripts/release/build_submission_archive.py --zip OUT  # and pack

Restricted data (CRSP/WRDS-derived rows) is a separate matter: crsp_restricted/
is excluded here and is handed to the supervisor and assessors inside the
subscribing institution, as Appendix B records.
"""
import argparse
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directories that never enter the archive.  Each carries a reason, because a
# bare list rots: the next person needs to know whether an entry is a licence
# problem, a venue problem or simply bulk.
EXCLUDE_DIRS = {
    "writing/paper": "companion manuscript sources; sibling filenames name a venue",
    "tmp": "scratch, including a downloaded author kit",
    "reference_papers": "third-party PDFs, redistribution not ours to grant",
    "design": "early planning notes written against the manuscript cycle",
    "crsp_restricted": "CRSP-derived rows; licence bars redistribution",
    ".git": "history carries commit messages written for the manuscript cycle",
    "slide": "a talk deck with Chinese speaker notes; not a deliverable",
}

# Individual files that never enter the archive.  All four are strategy
# documents about the companion manuscript's reception.  None is cited by any
# "% src:" comment in the dissertation, so excluding them breaks no promise --
# verified by scripts/analysis/src_comment_census.py.
EXCLUDE_FILES = {
    "results/ACCEPTANCE_ROI_PANEL.md",
    "results/PAPER_WRITING_PLAYBOOK.md",
    "results/DATA_COMPLETENESS_FINAL.md",
    "results/SECOND_DOMAIN_PLAN.md",
    "results/COLD_READ_GAPS.md",
}

# Whole families excluded by pattern: accept-probability estimates and the
# archived review-panel simulations.
EXCLUDE_PATTERNS = [
    re.compile(r"^results/AAAI27_"),
    re.compile(r"^results/REVIEW"),
    # The two CRSP-derived crosswalks exist twice in this checkout: as CSV under
    # crsp_restricted/ (excluded above by directory) and as parquet under
    # data/universe/. Same rows, same licence bar. Excluding one form and not
    # the other would ship the data anyway.
    re.compile(r"^data/universe/.*\.parquet$"),
]

# The venue names are the obvious half.  The other half is process vocabulary:
# a file that says "upon acceptance" or "anonymised supplementary material"
# announces a manuscript under review without naming where, and the first pass
# of this gate missed exactly that in a supporting file it had just started
# shipping.
VENUE = re.compile(
    r"\bAAAI\b|\bNeurIPS\b|\bICML\b|\bICLR\b|\bACL\b|\bEMNLP\b|\bKDD\b"
    r"|double-?blind|\u53cc\u76f2|camera-?ready|area chair|\u6295\u7a3f|desk-?reject"
    r"|upon acceptance|under review|during review|anonymi[sz]ed supplement",
    re.IGNORECASE,
)

TEXT_EXT = {".md", ".txt", ".csv", ".py", ".yaml", ".yml", ".json", ".toml",
            ".tex", ".cfg", ".sh", ".bib", ".lock", ".sty", ".bst", ""}

# Real disclosure text legitimately contains "double-blind" (clinical trials)
# and "rebuttal" (regulatory testimony).  Excluding the corpus from the scan
# would be the wrong fix -- it has to ship -- so the corpus is scanned for
# venue names only, which it has never contained.
CORPUS_PREFIXES = ("results/second_domain/",)

# A bibliography names venues by construction -- that is what a citation is --
# and so does any line citing someone else's paper.  Flagging those would train
# the reader to ignore the gate, which is worse than not having one.
# Hits the author has seen and decided to ship.  An exception is recorded here,
# with its reason, rather than by loosening the pattern: a gate that is quietly
# weakened stops meaning anything, and a gate that always fails gets ignored.
#
# results/HPO_ARM_SPEC.md is a member of results/prereg/hpo_prereg_v1.1.tar,
# whose SHA-256 is printed in Appendix A and whose OpenTimestamps proof commits
# to that digest.  Editing the file would void both.  Editing only the working
# copy would be worse than either option: a marker who extracted the tar and
# diffed it against the checkout would see precisely which lines had been
# removed.  The exposure is bounded -- the public mirror carries neither this
# file nor results/prereg/, so the two lines reach only a marker who unpacks the
# lodged archive.  Author's decision, 2026-08-28: ship as is.
#
# writing/dissertation/acknowledge.tex names the venues of the author's first
# two papers and the venue this project was submitted to.  The no-venue rule
# existed to keep the dissertation from linking itself to a manuscript under
# blind review; the author weighed that against thanking the supervisor who
# made both possible and chose the thanks.  Author's decision, 2026-08-29.
ACCEPTED = {
    ("writing/dissertation/acknowledge.tex", 4),
}

SCANNERS = {
    "scripts/release/build_submission_archive.py",
    "scripts/release/sync_public_mirror.py",
    "scripts/release/blind_scan.py",
}

BIB_LINE = re.compile(
    r"booktitle|journal\s*=|Proceedings of|Findings of|arXiv:|@(?:in)?proceedings"
    r"|Anthology|bib key|\\cite", re.IGNORECASE)


def tracked():
    out = subprocess.run(["git", "-C", ROOT, "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    return out.split("\n")[:-1] if out.endswith("\n") else out.split("\n")


def excluded(path):
    for d, why in EXCLUDE_DIRS.items():
        if path == d or path.startswith(d + "/"):
            return why
    if path in EXCLUDE_FILES:
        return "strategy document about the companion manuscript's reception"
    for pat in EXCLUDE_PATTERNS:
        if pat.match(path):
            if path.startswith("data/universe/"):
                return "CRSP-derived crosswalk in parquet form; licence bars redistribution"
            return "accept-probability estimate or archived review simulation"
    return None


def scan(paths):
    """Return [(path, lineno, line)] for every venue mention in the staged set."""
    hits, accepted = [], []
    for p in paths:
        if os.path.splitext(p)[1].lower() not in TEXT_EXT:
            continue
        if p.endswith(".bib"):
            continue
        # A leak scanner necessarily contains the strings it hunts for.
        if p in SCANNERS:
            continue
        full = os.path.join(ROOT, p)
        if not os.path.isfile(full):
            continue
        corpus = p.startswith(CORPUS_PREFIXES)
        try:
            with open(full, errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    m = VENUE.search(line)
                    if not m:
                        continue
                    # inside the disclosure corpus, only a named venue counts
                    if corpus and not re.search(
                            r"\bAAAI\b|\bNeurIPS\b|\bICML\b|\bICLR\b", line, re.I):
                        continue
                    if BIB_LINE.search(line):
                        continue
                    if (p, i) in ACCEPTED:
                        accepted.append((p, i))
                        continue
                    hits.append((p, i, line.strip()[:160]))
        except OSError:
            continue
    return hits, accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", metavar="OUT", help="also write a ZIP of the staged set")
    ap.add_argument("--root", help="gate a different checkout (e.g. the public mirror)")
    args = ap.parse_args()
    if args.root:
        global ROOT
        ROOT = os.path.abspath(args.root)

    all_paths = tracked()
    staged, dropped = [], {}
    for p in all_paths:
        why = excluded(p)
        if why:
            dropped.setdefault(why, []).append(p)
        else:
            staged.append(p)

    print(f"tracked {len(all_paths)}  staged {len(staged)}  excluded {len(all_paths) - len(staged)}")
    for why, ps in sorted(dropped.items()):
        print(f"  - {len(ps):>4} {why}")

    hits, accepted = scan(staged)
    if accepted:
        print(f"\nnote {len(accepted)} recorded exception(s), shipped by decision:")
        for p, i in accepted:
            print(f"  {p}:{i}")
    if hits:
        print(f"\nFAIL: {len(hits)} venue or review mention(s) inside the staged set")
        for p, i, line in hits[:40]:
            print(f"  {p}:{i}: {line}")
        if len(hits) > 40:
            print(f"  ... and {len(hits) - 40} more")
        return 1
    print("\nok   no venue or review mention in any staged text file")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for p in staged:
                full = os.path.join(ROOT, p)
                if os.path.isfile(full):
                    z.write(full, p)
        mb = os.path.getsize(args.zip) / 1e6
        print(f"ok   wrote {args.zip} ({mb:.1f} MB, {len(staged)} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
