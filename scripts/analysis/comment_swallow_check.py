"""Catch prose and cross-references that a source comment has silently eaten.

LaTeX comments run to end of line, so anything after an unescaped `%` vanishes
from the page without any error.  This project writes a `% src:` comment on the
same line as the sentence it documents, which puts a `%` in the middle of almost
every line that carries a number -- and three times now a `\\ref`, a closing
brace or a whole clause has ended up on the wrong side of it.  The failure is
invisible in the log and in the page count; it shows up only as a sentence that
starts mid-thought.

The test: after the first unescaped `%` on a line, is there anything that could
only be typeset material?  A `\\ref`/`\\cite`/`\\label` command, a `\\emph`, or a
run of words ending in a full stop followed by a capitalised word.  A source
comment naming files and numbers trips none of those.
"""
import glob
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DISS = os.path.join(ROOT, "writing", "dissertation")

# the first % that is neither \% nor inside \texttt{\%  ...}
PCT = re.compile(r'(?<![\\"])%')
TYPESET = re.compile(r"\\(?:ref|autoref|eqref|cite\w*|label|emph|textbf|texttt|footnote)\{")
# "...text. Word" -- a sentence boundary inside a comment is prose, not provenance
PROSE = re.compile(r"[a-z]{3,}\.\s+[A-Z][a-z]{2,}")


SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])")

# Fourth trigger.  The three tests above share two blind spots, and the same
# defect walked through both: a comment whose swallowed clause ended in a
# function word, where the next line opened with a DIGIT ("1.60 per cent")
# rather than a lower-case letter, and a comment-continuation line beginning
# with "%", where there is no text before the % whose sentence could be tested.
# What both leave is unmistakable once looked for: a source comment that ends
# mid-clause.  Provenance names files, rows and numbers; it never ends on a
# preposition, article or conjunction, so a tail ending in one is prose that
# fell behind the %.
DANGLE = re.compile(r"\b(?:a|an|the|to|of|in|on|at|by|for|from|with|into|over|under|"
                    r"between|and|or|but|that|which|is|are|was|were|be|been|as|than|"
                    r"its|their)\s*$", re.I)   # not "it"/"this"/"those": those
                                                #   end a clause legitimately


def unbalanced_sentences(path):
    """Sentences that open a round bracket and never close it.

    A dropped line usually takes a closing bracket with it, which is the one
    part of a truncation that can be seen without a baseline to compare against.
    """
    src = open(path).read()
    src = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in src.split("\n"))
    src = re.sub(r"\\begin\{(figure|table|tabular)\}.*?\\end\{\1\}", " ", src, flags=re.S)
    # spacing commands such as \! carry a character the sentence splitter reads
    # as a full stop, which would cut a sentence open inside its own maths
    src = re.sub(r"\\[^a-zA-Z\s]", " ", src)
    src = re.sub(r"\\[a-zA-Z]+", " ", re.sub(r"\s+", " ", src))
    out = []
    for s in SENT.split(src):
        if len(s.split()) >= 8 and s.count("(") != s.count(")"):
            out.append(" ".join(s.split()[:12]))
    return out


def main():
    files = (sorted(glob.glob(os.path.join(DISS, "chapters", "*.tex"))) +
             sorted(glob.glob(os.path.join(DISS, "appendices", "*.tex"))) +
             [os.path.join(DISS, f) for f in ("summary.tex", "nomenclature.tex",
                                              "main.tex", "prelude.tex")])
    bad = []
    for p in files:
        if not os.path.isfile(p):
            continue
        lines = open(p).readlines()
        for i, line in enumerate(lines, 1):
            # A continuation line of a multi-line source comment starts with "%"
            # and is exactly where the last swallow happened, so it is not skipped.
            m = PCT.search(line)
            if not m:
                continue
            tail = line[m.end():]
            hit = TYPESET.search(tail) or PROSE.search(tail)
            # A source comment may legitimately name a chapter with \ref.  What
            # marks the real failure is the sentence continuing on the next line:
            # the swallowed words were the head of a clause, so what follows
            # begins in lower case, mid-thought.
            nxt = lines[i].lstrip() if i < len(lines) else ""
            # A swallowed clause is not always followed by a lower-case word:
            # it is followed by whatever the sentence said next, which may be a
            # figure ("1.60 per cent") or a formula.  Both continue mid-thought.
            continues = bool(re.match(r"[a-z0-9$]", nxt)) and not nxt.startswith("%")
            # A swallow at the end of a sentence or a paragraph leaves nothing
            # continuing in lower case, so the test above cannot see it. What it
            # does leave is a cross-reference that now appears ONLY inside a
            # comment: the label is still defined, but no reader is ever sent to
            # it. That is the same defect seen from the other side.
            lost_ref = False
            if hit and not continues:
                for m2 in re.finditer(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", tail):
                    lab = m2.group(1)
                    live = sum(1 for ln2 in lines
                               if re.search(r"\\(?:ref|autoref|eqref)\{" + re.escape(lab) + r"\}",
                                            PCT.split(ln2)[0]))
                    if live == 0:
                        lost_ref = True
            # Third trigger, and the one that caught eleven at once.  The two
            # tests above need the swallowed words to look like prose ON THEIR
            # OWN (a "word. Word" boundary, or a lost \\ref).  They cannot see the
            # commonest way this defect is made: appending a "% src:" gloss to a
            # line whose sentence had not finished, which pushes the rest of that
            # line behind the %.  What that always leaves is a signature the
            # regexes above miss --- the text BEFORE the % ends a sentence, yet
            # the NEXT line opens in lower case, mid-thought.  Nothing bridges
            # them on the page, because what bridged them is inside the comment.
            # Legitimate multi-line provenance is excluded by requiring the
            # comment to END in prose: its last tokens must be plain words, not
            # a path, filename, identifier or bare figure.
            bridged = False
            if not hit and continues and re.search(r"[.:;!?]$", line[:m.start()].rstrip()):
                # Walk back from the end collecting the TRAILING run of plain
                # words, stopping at the first token that is a path, filename,
                # identifier or figure.  Scanning a fixed window instead fails:
                # the provenance's own "C2_finbert_s1)" sits inside the last five
                # tokens and vetoes a run that is plainly prose.
                run = []
                for t in reversed(tail.split()):
                    if re.fullmatch(r"[A-Za-z][A-Za-z'-]*[.,;:]?", t):
                        run.append(t)
                    else:
                        break
                run.reverse()
                # Three plain words are prose on their own; one or two are prose
                # when the run opens a clause with a capital ("The relabelling").
                bridged = len(run) >= 3 or (len(run) >= 1 and run[0][:1].isupper())
            # Fourth trigger: the comment itself ends mid-clause.  This one
            # needs neither a sentence boundary before the % nor a lower-case
            # word after it, which is exactly why it sees what the others miss.
            # A multi-line EXPLANATORY comment ends its lines mid-sentence by
            # design, and the next line carries on inside the comment.  What
            # separates it from a swallow is what follows: prose that will be
            # typeset, not another "%".  Without this the trigger fires on every
            # decision note in the preamble -- 44 of them at the time of writing.
            dangling = (bool(DANGLE.search(tail.rstrip()))
                        and bool(re.search(r"[A-Za-z]{3}", tail))
                        and nxt != "" and not nxt.startswith("%"))
            if hit and (continues or lost_ref) or bridged or dangling:
                    bad.append((os.path.relpath(p, ROOT), i,
                            hit.group(0)[:24] if hit else
                            ("dangling-tail" if dangling else "bridged-sentence"),
                            tail.strip()[-100:]))
    unb = []
    for p in files:
        if os.path.isfile(p):
            for u in unbalanced_sentences(p):
                unb.append((os.path.relpath(p, ROOT), u))
    for f, u in unb:
        print(f"  FAIL {f}  unclosed bracket: {u}")
    for f, i, what, tail in bad:
        print(f"  FAIL {f}:{i}  swallowed {what!r}")
        print(f"       after %: {tail}")
    print(f"\n  {len(bad)} line(s) with typeset material after a comment mark"
          if bad else "\n  no comment swallows prose or a cross-reference")
    if unb:
        print(f"  {len(unb)} sentence(s) with an unclosed bracket")
    return 1 if (bad or unb) else 0


if __name__ == "__main__":
    sys.exit(main())
