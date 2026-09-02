"""What a prose pass is allowed to change, and what it is not.

A rewrite for readability should move words and leave evidence alone.  The
failure mode is quiet: a sentence gets tightened, and a number, a citation or a
concession goes with it, and nothing in the build complains because the document
still compiles and still has sixty pages.

This compares a chapter against its committed self and reports every number,
citation key, cross-reference and provenance comment that the working copy no
longer contains.  It says nothing about words: those are meant to change.

    python3 prose_diff_audit.py                      # all chapters vs HEAD
    python3 prose_diff_audit.py --ref HEAD~1         # against an older commit
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CH = "writing/dissertation/chapters"
# The appendices carry more than half the report's numbers and all of its result
# tables, and an edit there can lose a clause exactly as easily; a line-by-line
# rewrite in Appendix D dropped half a sentence and its source comment before
# this was widened.
APX = "writing/dissertation/appendices"
EXTRA = ["writing/dissertation/summary.tex",
         "writing/dissertation/nomenclature.tex"]


def _strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.split("\n"))


def facts(src):
    """The things a prose edit must not lose, keyed by kind."""
    prose = _strip_comments(src)
    # Numbers as the reader meets them: 112,528 and 0.51 and 8-K all count, and
    # the thousands brace is normalised so {,} and , are the same number.
    # A grouped number is ONE number. The brace form is normalised to a comma
    # first, so the comma alternative has to come before the bare one, or
    # "112{,}528" is recorded as "112" and "528" and its loss goes unseen.
    nums = re.findall(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])"
                      r"|(?<![\w.])\d[\d.]*(?![\w])",
                      prose.replace("{,}", ","))
    return {
        "number": sorted(set(n.rstrip(".") for n in nums)),
        "cite": sorted(set(k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", prose)
                           for k in m.split(","))),
        "ref": sorted(set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]*)\}", prose))),
        # provenance comments are the one thing read OUT of the comments
        # A quoted or escaped "% src:" is prose ABOUT the convention -- Appendix A
        # describes it in a sentence -- not an instance of it.
        "src": sorted(set(m.strip() for m in
                          re.findall(r"(?<![\\\"])%\s*(?:caption\s+)?(?:numbers\s+)?src:\s*([^\n]*)",
                                     src))),
    }


def sentences(src):
    """The prose split into sentences, comments and floats removed."""
    t = _strip_comments(src)
    # Captions live inside float environments and carry a large share of the
    # report's claims -- and, being edited line by line, a large share of its
    # truncations. Strip the body of a float (the tabular, the graphics) but
    # keep its caption text in the sentence stream.
    t = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", t, flags=re.S)
    t = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\(?:begin|end)\{(?:figure|table)\*?\}(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\caption\[[^\]]*\]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", t) if len(x.split()) >= 8]


def truncated(before, after):
    """Sentences whose head survived an edit but whose body was cut away.

    This is what a line-by-line rewrite does when it drops a line: the sentence
    still starts where it did and then stops mid-clause, and nothing else
    notices -- no number is lost, no citation, no source comment, and LaTeX
    compiles a headless sentence without a murmur. It has happened twice here.

    A deliberate rewrite also keeps the head sometimes, so the test is not
    "the tail is gone" but "the sentence that still starts this way is now much
    shorter" -- truncation removes words, rephrasing trades them.
    """
    idx = {}
    for s in after:
        idx.setdefault(" ".join(s.split()[:6]), []).append(len(s.split()))
    joined = " ".join(after)
    heads_after = {" ".join(s.split()[:6]) for s in after}
    out = []
    for n, s in enumerate(before):
        w = s.split()
        head, tail = " ".join(w[:6]), " ".join(w[-5:])
        if head not in idx:
            continue
        now = max(idx[head])
        # the sentence that FOLLOWED this one: if the full stop was lost, that
        # one is no longer a sentence of its own, it is inside this one
        nxt = before[n + 1] if n + 1 < len(before) else ""
        swallowed = bool(nxt) and " ".join(nxt.split()[:6]) not in heads_after
        # cut short: the sentence still starts here and stops early
        if now < 0.6 * len(w) and len(w) - now >= 6:
            out.append(f"{head} ... ({len(w)} words -> {now})")
        # cut at the full stop: losing the last clause takes the full stop with
        # it, so the sentence runs on and SWALLOWS the next one instead
        elif now > 1.4 * len(w) and tail not in joined and swallowed:
            out.append(f"{head} ... (runs on: {len(w)} words -> {now}, tail gone)")
    # A third shape: the deleted span sits ACROSS a full stop, so the sentence
    # before it loses its ending and the sentence after it loses its opening,
    # and the weld comes out about the same length as either. Neither of the
    # branches above fires; what gives it away is a sentence whose head is gone
    # while its tail survives, welded onto its predecessor.
    heads = {" ".join(s.split()[:6]) for s in after}
    for n, s in enumerate(before):
        w = s.split()
        if len(w) < 7:
            continue
        head, tail = " ".join(w[:6]), " ".join(w[-6:])
        if head in heads or tail not in joined:
            continue
        # Rewriting a sentence's opening while keeping its ending looks the same
        # from here, and is legitimate. What separates the two is the sentence
        # BEFORE: a deletion that crosses a full stop takes that one's ending
        # with it, a rewrite leaves it alone.
        prev = before[n - 1] if n else ""
        pw = prev.split()
        if len(pw) >= 7 and " ".join(pw[-6:]) not in joined:
            out.append(f"HEAD EATEN: ...{tail}  (opened '{head}'; "
                       f"the sentence before it also lost its ending)")
    return out


def unbalanced(src):
    """Sentences that open a bracket and never close it."""
    out = []
    for s in sentences(src):
        plain = re.sub(r"\\[a-zA-Z]+", " ", s)
        if plain.count("(") != plain.count(")"):
            out.append(" ".join(s.split()[:12]))
    return out


def committed(path, ref):
    out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"cannot read {path} at {ref}: {out.stderr.strip()[:160]}")
    return out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--files", nargs="*", default=None)
    a = ap.parse_args()

    if a.files:
        rels = [f"{CH}/{f}" if "/" not in f else f for f in a.files]
    else:
        rels = ([f"{CH}/{f}" for f in sorted(os.listdir(os.path.join(ROOT, CH)))
                 if f.endswith(".tex")] +
                [f"{APX}/{f}" for f in sorted(os.listdir(os.path.join(ROOT, APX)))
                 if f.endswith(".tex")] +
                [f for f in EXTRA if os.path.exists(os.path.join(ROOT, f))])
    bad = 0
    for rel in rels:
        n = os.path.basename(rel)
        before = facts(committed(rel, a.ref))
        after = facts(open(os.path.join(ROOT, rel)).read())
        lost = {k: [x for x in before[k] if x not in after[k]] for k in before}
        src_before = committed(rel, a.ref)
        src_after = open(os.path.join(ROOT, rel)).read()
        cut = truncated(sentences(src_before), sentences(src_after))
        unb = [x for x in unbalanced(src_after) if x not in unbalanced(src_before)]
        gained_num = [x for x in after["number"] if x not in before["number"]]
        n_lost = sum(len(v) for v in lost.values())
        if not n_lost and not gained_num and not cut and not unb:
            print(f"  ok   {n:<44} nothing lost")
            continue
        bad += 1
        print(f"  FAIL {n}")
        for k, v in lost.items():
            if v:
                print(f"         lost {k}: {', '.join(v[:14])}"
                      + (" ..." if len(v) > 14 else ""))
        for t in cut:
            print(f"         TRUNCATED: {t}")
        for u in unb:
            print(f"         UNCLOSED BRACKET: {u}")
        if gained_num:
            print(f"         NEW numbers (must be quoted from elsewhere in the "
                  f"report, never invented): {', '.join(gained_num[:14])}")
    print("\n  " + ("evidence intact across the prose pass" if not bad
                    else f"{bad} file(s) lost evidence — see above"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
