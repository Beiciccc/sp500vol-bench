#!/usr/bin/env python3
"""Refresh the public mirror from this checkout, sanitised, and gate the result.

The mirror and the lodged archive are not the same object and must not be
sanitised the same way.  The archive goes to a supervisor and assessors inside
the institution, where the author is named on the title page and a path like
/Users/<name> discloses nothing they do not already know.  The mirror is world
readable, so the same string is a small permanent disclosure of the author's
machine, storage layout, rented GPU host and personal address.

So the substitutions below apply to the mirror only.  Running them over this
checkout instead would break the author's own scripts for no benefit, and
rewriting a committed evidence table to hide a directory name is not something
that should happen quietly in the working tree.

The mirror keeps its own curation: this script refreshes the files the mirror
already tracks and never introduces new ones, apart from the explicit ADDITIONS
below.  Anything the mirror should stop carrying has to be removed there.

    python3 scripts/release/sync_public_mirror.py PATH/TO/MIRROR          # dry run
    python3 scripts/release/sync_public_mirror.py PATH/TO/MIRROR --write  # apply
"""
import argparse
import filecmp
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The strings to scrub are the author's home directory, external volume, rented
# GPU hosts and personal address. Hard-coding them here published every one of
# them the moment this file itself started travelling to the mirror -- the
# sanitiser leaking exactly what it sanitises. They now live in a local file
# that git ignores, so this script can be read without disclosing them.
#
# Format: one "literal<TAB>replacement" pair per line. Order matters: put the
# longer literal first, or a hostname that is also the prefix of a longer
# directory name rewrites the directory into a half-substituted mess.
SCRUB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_scrub.tsv")


def load_substitutions():
    if not os.path.isfile(SCRUB_FILE):
        sys.exit(f"missing {SCRUB_FILE}: the scrub list is deliberately not in git.\n"
                 "Recreate it as literal<TAB>replacement lines before syncing.")
    out = []
    for line in open(SCRUB_FILE, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        lit, _, rep = line.partition("\t")
        out.append((lit, rep))
    return out


SUBSTITUTIONS = None   # filled by main() from SCRUB_FILE

# Files present in this checkout that the mirror should carry even if it does
# not have them yet.
# Files and directories present in this checkout that the mirror should carry
# even if it does not have them yet. Directories are expanded from git ls-files.
# results/prereg/ is the externally timestamped pre-registration bundle, and
# writing/dissertation/supporting/ holds the six sources that 105 "% src:"
# comments point at: Appendix A promises a marker can open both, so once the
# repository became the delivery route rather than a convenience copy, both had
# to travel. Neither contains licensed data.
ADDITIONS = [
    "release/tag_manifest.txt",
    "results/prereg/",
    "results/HPO_ARM_SPEC.md",
    "writing/dissertation/supporting/",
    # Appendix A.5 tells a marker to open the LaTeX sources and follow a
    # "% src:" comment from a printed number to the artefact behind it. That is
    # the report's central traceability claim and it needs the sources to be
    # here. Listed by directory and by file so that writing/dissertation/figures/
    # -- which holds a scanned signature -- is never swept in.
    "writing/dissertation/chapters/",
    "writing/dissertation/appendices/",
    "writing/dissertation/paper_full_sections/",
    "writing/dissertation/main.tex",
    "writing/dissertation/prelude.tex",
    "writing/dissertation/acknowledge.tex",
    "writing/dissertation/summary.tex",
    "writing/dissertation/nomenclature.tex",
    "writing/dissertation/config.tex",
    "writing/dissertation/figuresetup.tex",
    "writing/dissertation/refs.bib",
    # The census the same section quotes, and the gate scripts the appendix
    # names by filename.
    "results/tables/src_comment_census.txt",
    "results/tables/recal_slope_logspace.md",
    "scripts/analysis/",
    "scripts/release/",
]

# Never mirrored, whatever the mirror happens to track.  The mirror carries no
# writing/ directory today, so this costs nothing; it exists because a scanned
# signature is reusable by anyone who downloads it, and "the mirror does not
# track that directory" is a fact that could change without anyone noticing.
NEVER = {
    "writing/dissertation/figures/signature.png",
    # CRSP-derived crosswalks in parquet form; the licence bars redistribution.
    "data/universe/sp500_membership.parquet",
    "data/universe/crsp_cik_links.parquet",
    # The manuscript packaging pipeline. It has no bearing on how the
    # dissertation's materials are delivered, and each of these hard-codes the
    # author's home directory, GPU hosts and personal address as literals to
    # search for -- publishing them would republish exactly what the scrub list
    # removes.
    "scripts/release/blind_scan.py",
    "scripts/release/build_code_data_zip.py",
    "scripts/release/build_supplement_pdf.py",
    "scripts/release/stage_release_payload.py",
}

# A leak scanner necessarily contains the strings it hunts for, so it must be
# exempt twice over: substituting into its own patterns would publish a scanner
# that can no longer detect what it was written to catch, and scanning it would
# flag its own source. Both were observed before this set existed.
SCANNERS = {
    "scripts/release/sync_public_mirror.py",
    "scripts/release/build_submission_archive.py",
}

# Lines the author has seen and chosen to publish. Recorded here with the
# reason rather than by loosening the pattern.
#
# results/HPO_ARM_SPEC.md is a member of the externally timestamped bundle
# results/prereg/hpo_prereg_v1.1.tar, whose SHA-256 is printed in Appendix A.
# Editing it would void that digest and the OpenTimestamps proof, so the two
# lines travel or the evidence does not. The earlier decision to ship them was
# taken when the exposure was a marker unpacking a lodged archive; once the
# repository became the delivery route the exposure became world-readable.
# What they say -- that the project was submitted to a conference -- the
# Acknowledgements now say in English on page iv of the dissertation itself,
# so they disclose nothing the author has not already published under his own
# name. Author's decision, 2026-08-29.
ACCEPTED = {
    # The Acknowledgements name the venues of the author's first two papers and
    # the venue this project was submitted to. That is the author's explicit
    # decision and it is printed on page iv of the dissertation itself, so the
    # source carrying it adds nothing.
    ("writing/dissertation/acknowledge.tex", 4),
}

TEXT_EXT = {".md", ".txt", ".csv", ".py", ".yaml", ".yml", ".json", ".toml",
            ".tex", ".cfg", ".sh", ".bib", ".lock", ".sty", ".bst", ".example", ""}

# What must not survive into a world-readable copy.  The venue patterns are the
# archive gate's; these add the identity and machine strings that only matter
# once the repository is public.
# Venue and process vocabulary is generic and safe to publish. The identity
# half is appended at run time from SCRUB_FILE, for the same reason.
VENUE_PAT = (r"\bAAAI\b|\bNeurIPS\b|\bICML\b|\bICLR\b|double-?blind|\u53cc\u76f2|camera-?ready"
             r"|area chair|\u6295\u7a3f|upon acceptance|under review|during review"
             r"|anonymi[sz]ed supplement")
LEAK = None   # filled by main() once the scrub list is known
BIB_LINE = re.compile(
    r"booktitle|journal\s*=|Proceedings of|Findings of|arXiv:|Anthology|bib key",
    re.IGNORECASE,
)


def is_text(path):
    return os.path.splitext(path)[1].lower() in TEXT_EXT


def sanitise(text):
    for a, b in SUBSTITUTIONS:
        text = text.replace(a, b)
    return text


def mirror_files(mirror):
    out = subprocess.run(["git", "-C", mirror, "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    return [p for p in out.split("\n") if p]


def scan(mirror, paths):
    hits, accepted = [], []
    for p in paths:
        if not is_text(p) or p.endswith(".bib") or p in SCANNERS:
            continue
        full = os.path.join(mirror, p)
        if not os.path.isfile(full):
            continue
        try:
            with open(full, errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if LEAK.search(line) and not BIB_LINE.search(line):
                        if (p, i) in ACCEPTED:
                            accepted.append((p, i))
                            continue
                        hits.append((p, i, line.strip()[:150]))
        except OSError:
            continue
    return hits, accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mirror", help="path to a checkout of the public mirror")
    ap.add_argument("--write", action="store_true", help="apply; otherwise dry run")
    args = ap.parse_args()
    mirror = os.path.abspath(args.mirror)

    global SUBSTITUTIONS, LEAK
    SUBSTITUTIONS = load_substitutions()
    ident = "|".join(re.escape(lit) for lit, _ in SUBSTITUTIONS)
    LEAK = re.compile(VENUE_PAT + "|" + ident, re.IGNORECASE)

    paths = mirror_files(mirror)
    tracked_here = subprocess.run(["git", "-C", ROOT, "ls-files"],
                                  capture_output=True, text=True, check=True).stdout.split("\n")
    for extra in ADDITIONS:
        if extra.endswith("/"):
            for p in tracked_here:
                if p.startswith(extra) and p not in paths:
                    paths.append(p)
        elif extra not in paths and os.path.isfile(os.path.join(ROOT, extra)):
            paths.append(extra)

    copied = changed = 0
    for p in paths:
        # macOS sidecars are junk to every reader and reached the mirror once
        # already through a bulk sync.
        if os.path.basename(p).startswith("._") or os.path.basename(p) == ".DS_Store":
            continue
        if p in NEVER:
            print(f"  withheld: {p}")
            continue
        src, dst = os.path.join(ROOT, p), os.path.join(mirror, p)
        if not os.path.isfile(src):
            continue
        if is_text(p):
            try:
                raw = open(src, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            new = raw if p in SCANNERS else sanitise(raw)
            if new != raw:
                changed += 1
            old = open(dst, encoding="utf-8", errors="ignore").read() if os.path.isfile(dst) else None
            if old != new:
                copied += 1
                if args.write:
                    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                    open(dst, "w", encoding="utf-8").write(new)
        else:
            if not os.path.isfile(dst) or not filecmp.cmp(src, dst, shallow=False):
                copied += 1
                if args.write:
                    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                    shutil.copy2(src, dst)

    verb = "updated" if args.write else "would update"
    print(f"{verb} {copied} of {len(paths)} mirrored paths; {changed} needed sanitising")

    if not args.write:
        print("dry run: nothing written (pass --write to apply)")
        return 0

    hits, accepted = scan(mirror, paths)
    if accepted:
        print(f"note {len(accepted)} recorded exception(s), published by decision:")
        for p, i in accepted:
            print(f"  {p}:{i}")
    if hits:
        print(f"\nFAIL: {len(hits)} leak(s) survive in the mirror")
        for p, i, line in hits[:40]:
            print(f"  {p}:{i}: {line}")
        if len(hits) > 40:
            print(f"  ... and {len(hits) - 40} more")
        return 1
    print("ok   no venue, machine-path or personal-address string in the mirror")
    return 0


if __name__ == "__main__":
    sys.exit(main())
