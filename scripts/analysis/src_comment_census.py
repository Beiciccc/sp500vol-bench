"""Census of the source comments that make this report's numbers checkable.

Appendix A tells a marker that every printed number carries a same-line
`% src:` comment, and then quotes how many of those comments point at a file
they can actually open.  That second number is the one claim in the section a
marker can check in a single command, so it must not be typed by hand: it moves
every time a sentence is edited.  This script is that command.

The split is "does the named path resolve to a file the archive ships".  The
archive is the whole submitted code-and-data tree, not just `release/` -- the
appendix calls `release/` its licence-safe core, and `results/tables/` is
tracked and travels with it -- so the test is git-tracked existence, resolved
from the repository root.  A comment naming several sources is classified on
the first one, which is the one a marker would open first.

`--check` gates the four printed figures against the recount; `--sync` writes
the recount back into Appendix A first.  The two are meant to be used together:
adding one `% src:` comment anywhere moves two of the four numbers, and typing
them by hand went wrong six times in one afternoon.  `--sync` never invents a
number -- it substitutes the recount into the two sentences that already print
one, and leaves the gate to say whether the file now agrees.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DISS = os.path.join(ROOT, "writing", "dissertation")
OUT = os.path.join(ROOT, "results", "tables", "src_comment_census.txt")

# any of the three markers used in the sources; `numbers` and `caption` are the
# captions' variants and are source comments in exactly the same sense.  A
# quoted "% src:" is prose about the convention, not an instance of it.
MARK = re.compile(r"""(?<![\\"])%\s*(?:caption\s+)?(?:numbers\s+)?src:\s*(.+)$""")
PATHY = re.compile(r"[\w./+-]*/[\w./+-]+|\b[\w.+-]+\.(?:csv|md|json|txt|tex|py|sh|yml|yaml|parquet|toml|lock)\b")


def compiled_files():
    """The .tex files main.tex and prelude.tex actually pull in."""
    inc = []
    for f in ("main.tex", "prelude.tex"):
        p = os.path.join(DISS, f)
        if not os.path.exists(p):
            continue
        for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", open(p).read()):
            n = m.group(1)
            n = n if n.endswith(".tex") else n + ".tex"
            if os.path.exists(os.path.join(DISS, n)) and n not in inc:
                inc.append(n)
    return inc


def _unescape(payload):
    """LaTeX escapes inside a comment are still LaTeX: TECHNICAL\\_SUPPLEMENT.md
    names a real file, but a path scanner that does not undo the backslash reads
    it as _SUPPLEMENT.md and reports a file the archive does not ship."""
    return payload.replace("\\_", "_").replace("\\%", "%").replace("\\&", "&")


def first_path(payload):
    """The first path-looking token in a source comment, punctuation stripped.

    A bare file name counts: many comments name the frozen drafting sources or
    the technical dossier by name alone, and those are paths a marker resolves
    just as readily as a slashed one.
    """
    m = PATHY.search(_unescape(payload))
    return m.group(0).rstrip(".,;:)") if m else None


# One artefact ships under a different name than the comments call it: each run's
# config.json travels as release/run_configs/<run_id>.json, byte-identical. A
# bare "config.json" therefore names a file the repository does carry, and
# resolving it by basename alone finds it in the authoring checkout (where
# results/runs/ exists) but not in the published mirror -- which is how the
# printed count and the delivered count came to differ by one.
ALIASES = {"config.json": "release/run_configs/"}


def resolves(tok, tracked, tracked_list):
    alias = ALIASES.get(os.path.basename(tok))
    if alias and any(t.startswith(alias) for t in tracked_list):
        return True
    """Whether a named token points at a file the archive actually ships.

    Three ways it can: the token is a tracked path; it is tracked relative to
    the dissertation directory, which is how the .tex sources name their
    siblings; or it is the literal prefix of a tracked path, which is what a
    comment naming a glob (`results/tables/yelp_*.csv`) leaves behind once the
    wildcard is stripped.
    """
    if tok in tracked:
        return True
    rel = os.path.relpath(os.path.join(DISS, tok), ROOT)
    if rel in tracked:
        return True
    if "/" not in tok:
        # a bare file name: the frozen drafting sources are named this way and
        # ship under writing/dissertation/paper_full_sections/
        if any(os.path.basename(t) == tok for t in tracked_list):
            return True
    return len(tok) >= 8 and any(t.startswith(tok) for t in tracked_list)


# Paths a "% src:" comment can legitimately name that the repository cannot
# carry. Counting them as reachable was the flaw this list closes: the census
# used to resolve against the author's checkout while Appendix A's sentence
# claimed the repository, and the two diverged by 55 the moment the repository
# became the delivery route instead of a convenience copy.
WITHHELD = (
    "crsp_restricted/",   # CRSP-derived crosswalks; the licence bars redistribution
    "results/runs/",      # per-run predictions, row-level and therefore licensed
    "results/anon/",      # anonymisation-arm predictions, same basis
    "results/hpo/",       # tuned-arm predictions, same basis
)


CITEKEY = re.compile(r"^[a-z]+\d{4}[a-z]+\b")


def kind_of(payload):
    """What a source comment names when it does not name a file.

    Three things legitimately are not files: a cited paper, the repository's own
    git history, and this report's own chapters.  The breakdown Appendix A prints
    has to come from here rather than from a hand count, for the same reason the
    totals do.
    """
    t = payload.strip()
    if t.startswith("git ") or re.match(r"^git\b", t):
        return "git-history"
    if CITEKEY.match(t) or "literature" in t.lower():
        return "citation"
    if re.match(r"^Chapters?\b|^Sections?\b", t):
        return "own-chapters"
    return "other"


def check(total, in_archive, in_release, kinds=None, withheld=None):
    """Fail if Appendix A's printed census no longer matches the sources.

    The numbers are self-referential -- the sentence that quotes them carries a
    source comment of its own, so it counts itself -- which is exactly why they
    need a gate rather than a careful author.
    """
    a = open(os.path.join(DISS, "appendices", "A_external_materials.tex")).read()
    # the sentence is line-wrapped in the source, so compare on collapsed spaces
    a = re.sub(r"\s+", " ", a)
    want = [(f"Of the {total} such comments, {in_archive} name a file", "total and archive count"),
            (_word(in_release) + " of those lead with a file in", "release/ subset")]
    # The sentence also prints a breakdown of the unresolved comments. Gating the
    # totals but not the breakdown leaves four printed numbers ungated, which is
    # exactly where a hand-typed figure would drift.
    if withheld is not None:
        want.append((f"A further {withheld} name an artefact the licence withholds",
                     "licence-withheld count"))
    if kinds:
        n = {"citation": kinds.get("citation", 0), "git": kinds.get("git-history", 0),
             "own": kinds.get("own-chapters", 0)}
        want.append((f"{_num(n['citation'])} name a cited paper", "citation count"))
        want.append((f"{_num(n['git'])} the repository's own commit history", "git-history count"))
    # Chapter 4's own pair, printed in the same appendix section. It drifted
    # unnoticed once (152/115 against a true 150/113) precisely because it was
    # the one census claim outside this list.
    ch4 = os.path.join(DISS, "chapters", "04_results.tex")
    hits = [m.group(1) for line in open(ch4) for m in [MARK.search(line)] if m]
    n_ch4 = len(hits)
    n_tab = sum(1 for h in hits if (first_path(h) or "").startswith("results/tables/"))
    want.append((f"where {n_tab} of Chapter~\\ref{{ch:results}}'s {n_ch4} source comments lead",
                 "chapter 4 tables pair"))

    bad = [w for pat, w in want if pat not in a]
    for pat, w in want:
        print(f"  {'ok  ' if pat in a else 'FAIL'} {w:<24} {pat[:52]}")
    return not bad


def _num(n):
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
             12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen"}
    return words.get(n, str(n))


WORDS = {
        40: "Forty", 41: "Forty-one", 42: "Forty-two", 43: "Forty-three", 44: "Forty-four",
        45: "Forty-five", 46: "Forty-six", 47: "Forty-seven", 48: "Forty-eight",
        49: "Forty-nine", 50: "Fifty", 51: "Fifty-one", 52: "Fifty-two", 53: "Fifty-three",
        54: "Fifty-four", 55: "Fifty-five", 56: "Fifty-six", 57: "Fifty-seven",
        58: "Fifty-eight", 59: "Fifty-nine", 60: "Sixty", 61: "Sixty-one", 62: "Sixty-two",
        63: "Sixty-three", 64: "Sixty-four", 65: "Sixty-five", 66: "Sixty-six",
        67: "Sixty-seven", 68: "Sixty-eight", 69: "Sixty-nine", 70: "Seventy",
        71: "Seventy-one", 72: "Seventy-two", 73: "Seventy-three", 74: "Seventy-four",
        75: "Seventy-five", 76: "Seventy-six", 77: "Seventy-seven", 78: "Seventy-eight",
        79: "Seventy-nine", 80: "Eighty", 81: "Eighty-one", 82: "Eighty-two",
        83: "Eighty-three", 84: "Eighty-four", 85: "Eighty-five", 86: "Eighty-six",
        87: "Eighty-seven", 88: "Eighty-eight", 89: "Eighty-nine"}
# The range matters: the release/ subset count has already travelled from the
# fifties into the sixties and back to the forties as files moved, and a table
# that stops short leaves --sync writing a numeral where the prose spells a word.


def _word(n):
    """Spell a count the way the prose does.

    This used to carry its own table, which stopped at sixty-three while the one
    `sync` writes from reached seventy.  The two then disagreed in exactly the
    band the count was drifting through: --sync wrote "Sixty-four" and the gate
    went looking for "64" and failed on a sentence it had itself just corrected.
    One table, read by both.
    """
    return WORDS.get(n, str(n))


def sync(total, in_archive, in_release):
    """Write the recount into the two sentences of Appendix A that print it."""
    p = os.path.join(DISS, "appendices", "A_external_materials.tex")
    s = open(p).read()
    before = s
    s = re.sub(r"Of the \d+ such comments, \d+ name a file",
               f"Of the {total} such comments, {in_archive} name a file", s)
    if in_release in WORDS:
        s = re.sub(r"\b(?:Forty|Fifty|Sixty|Seventy|Eighty)(?:-\w+)? of those lead with a file in",
                   f"{WORDS[in_release]} of those lead with a file in", s)
    else:
        print(f"  --sync: {in_release} has no spelled form in WORDS; left alone")
    if s != before:
        open(p, "w").write(s)
        print(f"  --sync: Appendix A updated to {total}/{in_archive}, release {in_release}")
    else:
        print("  --sync: Appendix A already agrees")


def main():
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.split("\n"))
    tracked_list = sorted(t for t in tracked if t)
    files = compiled_files()
    rows, in_archive, elsewhere, nopath, in_release = [], 0, 0, 0, 0
    withheld = 0
    per = {}
    for f in files:
        n = 0
        for ln in open(os.path.join(DISS, f)):
            m = MARK.search(ln)
            if not m:
                continue
            n += 1
            p = first_path(m.group(1))
            # a token with no letter in it is a number the regex mistook for a
            # path -- "4{,}096/512" reads as one
            if p is not None and not re.search(r"[A-Za-z]", p):
                p = None
            if p is None:
                nopath += 1
                rows.append((f, kind_of(m.group(1)), "no-path"))
            elif p.startswith(WITHHELD):
                withheld += 1
                rows.append((f, p, "withheld"))
            elif resolves(p, tracked, tracked_list):
                in_archive += 1
                # the licence-safe subset: what could be published as-is
                if p.startswith("release/"):
                    in_release += 1
                rows.append((f, p, "archive"))
            else:
                elsewhere += 1
                rows.append((f, p, "elsewhere"))
        if n:
            per[f] = n
    total = in_archive + withheld + elsewhere + nopath
    with open(OUT, "w") as fh:
        fh.write("# Census of `% src:` source comments in the compiled dissertation\n")
        fh.write(f"total_source_comments= {total}\n")
        fh.write(f"files_carrying_them= {len(per)}\n")
        fh.write(f"first_source_names_a_file_the_repository_carries= {in_archive}\n")
        fh.write(f"first_source_is_withheld_under_licence= {withheld}\n")
        fh.write(f"  of_those_under_release_the_licence_safe_core= {in_release}\n")
        fh.write(f"first_source_is_not_in_the_archive= {elsewhere}\n")
        fh.write(f"no_path_named= {nopath}\n")
        kinds = {}
        for _, k, kind in rows:
            if kind == "no-path":
                kinds[k] = kinds.get(k, 0) + 1
        for k in sorted(kinds):
            fh.write(f"  no_path_{k.replace('-', '_')}= {kinds[k]}\n")
        fh.write("\n")
        for f in files:
            if f in per:
                fh.write(f"{f}= {per[f]}\n")
        fh.write("\n# not in the archive, by named path\n")
        seen = {}
        for _, p, k in rows:
            if k == "elsewhere":
                seen[p] = seen.get(p, 0) + 1
        for p, c in sorted(seen.items(), key=lambda x: -x[1]):
            fh.write(f"  {c:>4}  {p}\n")
    print(f"  {total} source comments across {len(per)} files")
    print(f"    {in_archive} name a tracked file in the archive ({in_release} under release/)")
    print(f"    {elsewhere} name something outside it")
    print(f"    {nopath} name no path at all")
    print(f"  -> {os.path.relpath(OUT, ROOT)}")
    if "--sync" in sys.argv:
        sync(total, in_archive, in_release)
    if "--check" in sys.argv:
        kinds = {}
        for _, k, kind in rows:
            if kind == "no-path":
                kinds[k] = kinds.get(k, 0) + 1
        return 0 if check(total, in_archive, in_release, kinds, withheld) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
