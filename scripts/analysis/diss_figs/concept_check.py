"""Provenance gate for the concept diagrams.

The evidence figures cannot drift from the tables because their generators read
the tables and assert on them.  A schematic reads nothing, so it has no such
gate; the way a schematic would go wrong is by carrying a number the report does
not actually contain, or one the report contains attached to something else.

`NUMBERS` below is the register: for every quantity drawn on a concept diagram,
the string as drawn, the chapter file it was taken from, and enough of the
surrounding sentence to pin it to its own claim rather than to a coincidence of
digits elsewhere.  `check_all()` re-reads the chapters and fails if any register
entry can no longer be found, which is what makes the diagrams survive an edit
to the prose they were drawn from.
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DISS = os.path.join(ROOT, "writing", "dissertation")
CH = os.path.join(DISS, "chapters")
APX = os.path.join(DISS, "appendices")

# figure -> list of (chapter token, anchor phrase that must appear somewhere in
# the chapters [, the literal as it is drawn]).  Deliberately not pinned to one chapter: a number legitimately moves
# between chapters as the prose is edited -- indeed several of these moved out of
# Chapter 1 and into the diagram itself -- and what must stay true is that the
# report still says it, attached to this claim.
#
# The optional third element closes the other half of the loop.  The two-element
# form asserts only that the report says the thing; it cannot tell that a figure
# stopped drawing it, and that is not hypothetical -- editing a card's subtitle
# from "three training seeds" to "three seeds" left this register describing a
# string no longer on the page, and every check still passed.  So each entry is
# also matched against the generator source, whitespace-insensitively: the third
# element supplies the literal when the drawn form differs from the chapters'
# wording, and an explicit None records that the caption carries it instead.
NUMBERS = {
    "C1_reference_ladder": [
        ("0 of 153", "17 text and fusion arms remain 0 of 153 better"),
        # The figure says the standalone zero holds under three loss conventions,
        # not just the one Chapter 4 names; the scope is Appendix C's.
        ("180", "under three loss conventions and three error structures", None),
        ("153", "0 of 153 for text and fusion, and zero per-comparison verdict flips"),
        ("38", "38 of 69 combination cells show an apparent, placebo-confirmed increment"),
        # The chapters count these out in words; the rungs print them as a
        # slash triple under the cell count, so the drawn literal differs.
        ("12, 20 and 41", "recovers the injected signal in 12, 20 and 41 of 69 cells",
         "detects 12 / 20 / 41"),
        ("8", "the 8 identity survivors split six event-driven and two long-form"),
        ("7/11/20", "the identity and pool rungs in 7/11/20 and 6/12/19"),
        ("9", "all 9 pool survivors are long-form"),
        ("6/12/19", "the identity and pool rungs in 7/11/20 and 6/12/19"),
        ("2, 6 and 13", "the full conjunction in 2, 6 and 13", "detects 2 / 6 / 13"),
        ("69", "of 69 combination cells"),
        ("0.3", "injected into the text forecast at calibrated effect sizes of 0.3"),
        # Back on the figure, but split: the five names set as one line measure
        # 145pt and the widest card the layout affords is 121pt, so card 2 runs
        # them over two rows.  The ladder-only variant still has no room and
        # leaves them to the caption, which is why the literal checked here is
        # the first row rather than the whole list.
        ("HAR, SHAR, GARCH, EGARCH, ARIMA",
         "pool of five price models (HAR, SHAR, GARCH, EGARCH, ARIMA)",
         "HAR · SHAR · GARCH"),
        # "Disjoint" is the chapters' word; the rung says it in plain English.
        ("disjoint", "survivor sets disjoint", "the sets never meet"),
        # The band added above the ladder: the study's front half.  These are
        # design facts, not results, which is why they can sit in Chapter 1
        # without pre-empting Chapter 4.
        ("144,129", "144,129 of them align to the benchmark"),
        ("31,601", "31,601 long-form"),
        ("112,528", "112,528 event-driven"),
        ("431,245", "431,245 aligned filing-by-horizon rows"),
        ("2010", "2010"),
        ("no-look-ahead", "no-look-ahead audit found zero violations"),
        ("three training seeds", "three seeds"),
        ("5, 10, 20", "horizon h 5, 10, 20 trading days"),
        ("15", "15 pre-declared families"),
        ("240", "240 production runs"),
    ],
    # C2_identity_shortcut carries no quantity at all, which is correct for a
    # background chapter: it must not pre-empt a result.
    "C2_identity_shortcut": [],
}


def _drawn_literals(path):
    """Every string the generator can actually put on a page, run together.

    Matching against the raw source would be worthless: a comment explaining
    that "the eye lands on 144,129 and 431,245" satisfies a search for either
    number, so a figure could stop drawing a quantity while the prose about the
    figure kept the check green.  Verified by injection -- editing the drawn
    431,245 fails here only once comments and docstrings are excluded.  String
    literals are kept, because they are what gets drawn; docstrings are dropped
    by position rather than by content.
    """
    tree = ast.parse(open(path).read())
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node, clean=False):
            first = node.body[0].value
            doc.add((first.lineno, first.col_offset))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and (node.lineno, node.col_offset) not in doc):
            out.append(node.value)
    return re.sub(r"\s+", "", "\x00".join(out))


def _clean(path):
    """A chapter's prose with comments and maths spacing removed."""
    s = open(path).read()
    s = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.split("\n"))
    s = s.replace("{,}", ",").replace("~", " ").replace("\\,", "")
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = re.sub(r"[{}$\\]", "", s)
    return re.sub(r"\s+", " ", s)


def check_all(verbose=True):
    import glob
    ok = True
    # The appendices are part of the report.  A first version searched the
    # chapters only, which let a scope qualifier drawn on Figure 1.1 -- that the
    # standalone zero holds under all three loss conventions -- go unchecked,
    # because the sentence carrying it lives in Appendix C.
    bodies = {os.path.basename(p)[:2]: _clean(p)
              for p in sorted(glob.glob(os.path.join(CH, "0*.tex")))}
    bodies.update({"app-" + os.path.basename(p)[0]: _clean(p)
                   for p in sorted(glob.glob(os.path.join(APX, "*.tex")))})
    gen = _drawn_literals(os.path.join(HERE, "concept_figs.py"))
    for fig, entries in sorted(NUMBERS.items()):
        for entry in entries:
            drawn, anchor = entry[0], entry[1]
            lit = entry[2] if len(entry) > 2 else drawn
            if lit is not None and re.sub(r"\s+", "", lit) not in gen:
                ok = False
                print(f"  FAIL {fig:<22} {drawn:<32} registered as drawn, but "
                      f"{lit!r} is not in the generator")
            where = [k for k, b in bodies.items()
                     if drawn in b and anchor.lower() in b.lower()]
            if where:
                if verbose:
                    print(f"  ok   {fig:<22} {drawn:<32} ch{','.join(where)}")
            else:
                ok = False
                loose = [k for k, b in bodies.items() if anchor.lower() in b.lower()]
                why = ("anchor found in ch" + ",".join(loose) +
                       " but the number is not beside it" if loose
                       else "anchor sentence is in no chapter")
                print(f"  FAIL {fig:<22} {drawn:<32} {why}")
                print(f"       anchor: {anchor[:96]}")
    if verbose:
        print(f"\n  {'all concept-diagram numbers trace to the report' if ok else 'PROVENANCE GATE FAILED'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check_all() else 1)
