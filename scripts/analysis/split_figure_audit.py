#!/usr/bin/env python3
"""Split-figure gate: a multi-panel figure cut into -b/-c/-d fails silently in three ways.

Why this needs a gate of its own. Cutting a composite figure into several is done
line by line, and inside the caption the `% src:` comments are interleaved with the
prose, so it cannot be reflowed; the splitter can therefore only cut on line
boundaries. All three kinds of accident have happened, and all of them compile fine,
render fine, and say not one word in the log:

  1) Fragments. The shared sentence explaining the baseline names both (a) and (b),
     so whichever part it lands in reads wrong: nothing follows "…Bases differ:",
     or a caption is left with just the six characters "panel~(b)".
     Test: the caption ends on an unclosed colon/semicolon/comma, or starts with
     "panel~(x)" while the preceding sentence has already ended in a full stop,
     or the whole thing is under 10 words.
  2) Panel ownership. The prose says "Panel~(c) of Figure~\\ref{X}", while after the
     split (c) is no longer in X. Test: compute the set of panel letters each figure's
     own caption declares, then check whether the figure named in an ownership
     reference really holds that letter. Only "a figure listed that contains that
     panel not at all" is gated.
  3) Subject-verb. When `Figure~\\ref{X}` was bulk-expanded into
     `Figures~\\ref{X} and~\\ref{X-b}`, the singular verb after it stayed put --
     "Figures A and B draws the frontier audit".
     Test: a multi-reference `Figures~` group at the start of a sentence, immediately
     followed by a singular verb. A group preceded by "of", "charge" and the like
     (the real subject is further back, as in "The stratum map of Figures A
     and B is…") does not count; those sentences are correct.

  4) Mute captions. One of the split captions is left with nothing but the shared
     opening and never says which panel it draws -- two of the three elicitation ones
     are like this, and so is the second of the quarterly ones. This kind is the
     hardest to catch by eye: the caption reads fluently, it just describes the figure
     next door. Test: within a family, as soon as one split figure names a panel
     letter, every one of them must name at least one **in a sentence that carries no
     sibling figure's \\ref**. A family that uses no panel letters at all (there are a
     few in the main text) is skipped automatically.

All baselines are 0. Any count that is not 0 is damage left by the split, not a
tolerable normal state.
Usage:  python3 scripts/analysis/split_figure_audit.py [--quiet]
An exit code that is not 0 == damage.
"""
import re, sys, glob, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISS = os.path.join(ROOT, "writing", "dissertation")

FLOAT = re.compile(r"\\begin\{(figure)\*?\}(.*?)\\end\{\1\*?\}", re.S)
# Declaration forms for panel letters: the prose form "Panel~(a)"/"Panels (a) and (b)",
# and the bare form "(c) redraws..." / "(b): each cell...". The latter counts only at the
# start of a clause or immediately before a colon; otherwise cross-references such as
# "as in panel (b), (c) is not corrected" would be counted in as well.
DECL = re.compile(r"[Pp]anels?[~ ]?\(([a-e])\)(?:\s*and[~ ]?\(([a-e])\))?")
BARE = re.compile(r"^\s*\(([a-e])\)|\(([a-e])\)\s*:")
SING_VERB = (r"(is|was|has|does|draws|takes|reads|carries|shows|gives|plots|counts|labels|makes|"
             r"answers|separates|reports|sits|holds|adds|leaves|puts|turns|moves|prints|traces|"
             r"stratifies|says|needs|uses|earns|marks|lists|names|tells|records)")
AGREE = re.compile(r"(?:(?<=\.\s)|(?<=\.\n)|(?<=\n\n))\s*Figures~(?:\\ref\{[^}]+\}[,\s]*(?:and~?)?)+\s*"
                   + SING_VERB + r"\b")
OWN = re.compile(r"[Pp]anels?[~ ]?\(([a-e])\)((?:\s*and[~ ]?\([a-e]\))?)\s+of\s+"
                 r"Figures?~((?:\\ref\{fig:[^}]+\}[,\s]*(?:and~?)?)+)")


def strip_comments(s):
    """Drop % comments, but keep \\%."""
    return re.sub(r"(?<!\\)%[^\n]*", "", s)


def caption_text(body):
    """Text inside \\caption's braces, minus the optional short title and src comments."""
    i = body.find("\\caption")
    if i < 0:
        return ""
    j = i + len("\\caption")
    if j < len(body) and body[j] == "[":          # skip the short title
        d = 0
        while j < len(body):
            if body[j] == "[":
                d += 1
            elif body[j] == "]":
                d -= 1
                if d == 0:
                    j += 1
                    break
            j += 1
    while j < len(body) and body[j] != "{":
        j += 1
    d, k = 0, j
    while k < len(body):
        if body[k] == "{" and body[k - 1] != "\\":
            d += 1
        elif body[k] == "}" and body[k - 1] != "\\":
            d -= 1
            if d == 0:
                break
        k += 1
    return strip_comments(body[j + 1:k])


def own_letters(cap):
    """Which panels this caption claims to draw.

    Decided sentence by sentence: a sentence carrying \\ref{fig:...} is pointing at a
    sibling figure, so the panel letters in it do not count as this figure's. The
    distinction is required -- after a split, a caption legitimately uses
    "panel~(a), Figure~\\ref{...}" to point at another figure, and that is not a claim
    to have drawn (a) itself.
    """
    txt = " ".join(cap.split())
    here = set()
    for sent in re.split(r"(?<=[.;])\s+", txt):
        if re.search(r"\\ref\{fig:", sent):
            continue
        for mm in DECL.finditer(sent):
            for g in mm.groups():
                if g:
                    here.add(g)
        for mm in BARE.finditer(sent):
            for g in mm.groups():
                if g:
                    here.add(g)
    return here


def collect(files):
    """label -> (file, line number, caption text, set of panel letters it declares itself)"""
    out = {}
    for f in files:
        s = open(f).read()
        for m in FLOAT.finditer(s):
            body = m.group(2)
            lab = re.search(r"\\label\{([^}]+)\}", body)
            if not lab:
                continue
            cap = caption_text(body)
            lets = own_letters(cap)
            ln = s[:m.start()].count("\n") + 1
            out[lab.group(1)] = (f, ln, cap, lets)
    return out


def check_fragments(floats):
    """Fragments. Only members of a split family (same name with -b/-c/-d siblings) are checked."""
    bad = []
    fams = {}
    for lab in floats:
        fams.setdefault(re.sub(r"-[bcd]$", "", lab), []).append(lab)
    for base, members in fams.items():
        if len(members) < 2:
            continue
        for lab in members:
            f, ln, cap, _ = floats[lab]
            t = " ".join(cap.split())
            if not t:
                continue
            why = None
            if re.search(r"[:;,]\s*$", t):
                why = "ends on a colon/semicolon/comma"
            elif len(t.split()) < 10:
                why = f"only {len(t.split())} words"
            else:
                # a fragment starting with "panel~(x)": the previous sentence already ended in "."
                for mm in re.finditer(r"(?<=\.)\s+panel[~ ]?\([a-e]\)", t):
                    why = "starts a new sentence with lower-case panel~(x)"
                    break
            if why:
                bad.append((f, ln, lab, why, t[-70:]))
    return bad


def check_mute(floats):
    """Mute caption: one member of a split family never says which panel it draws."""
    fams = {}
    for lab in floats:
        fams.setdefault(re.sub(r"-[bcd]$", "", lab), []).append(lab)
    bad = []
    for base, mem in sorted(fams.items()):
        if len(mem) < 2:
            continue
        if not any(floats[l][3] for l in mem):      # whole family uses no panel letters
            continue
        for lab in sorted(mem):
            if not floats[lab][3]:
                f, ln = floats[lab][0], floats[lab][1]
                bad.append((f, ln, lab))
    return bad


def check_orphan_continuation(floats):
    """A continuation caption does not name which figure it continues.

    A float numbered -b whose panel letters start at (c), landing alone on a page, leaves
    the reader no way of knowing where (a) and (b) are. The convention itself is sound --
    one conceptual figure cut into two numbered floats with continuous lettering shows
    they are a single whole better than restarting each at (a) -- but it only holds if the
    continuation announces itself. Before this gate was added, only one of the nine
    continuation figures managed both "says continued" and "names the parent"; four opened
    their caption straight at (c). This is what a convention kept by memory looks like.
    """
    bad = []
    for lab in sorted(floats):
        base = re.sub(r"-[bcd]$", "", lab)
        if base == lab or base not in floats:
            continue
        f, ln, cap, _ = floats[lab]
        names_parent = re.search(r"\\ref\{" + re.escape(base) + r"\}", cap)
        says_continued = re.search(r"\bcontinu", cap, re.I)
        if not (names_parent and says_continued):
            why = []
            if not says_continued:
                why.append("does not say it is a continuation")
            if not names_parent:
                why.append("does not name the parent figure")
            bad.append((f, ln, lab, ", ".join(why)))
    return bad


def check_ownership(files, floats):
    bad = []
    for f in files:
        t = open(f).read()
        for m in OWN.finditer(t):
            if t.rfind("\n%", 0, m.start()) > t.rfind("\n", 0, m.start()) - 2:
                continue
            lets = {m.group(1)} | set(re.findall(r"\(([a-e])\)", m.group(2)))
            refs = re.findall(r"\\ref\{(fig:[^}]+)\}", m.group(3))
            extra = [r for r in refs if r in floats and not (lets & floats[r][3])]
            if extra:
                ln = t[:m.start()].count("\n") + 1
                bad.append((f, ln, sorted(lets), extra))
    return bad


def check_agreement(files):
    bad = []
    for f in files:
        t = open(f).read()
        for m in AGREE.finditer(t):
            ln = t[:m.start()].count("\n") + 1
            bad.append((f, ln, m.group(1), " ".join(m.group(0).split())[:100]))
    return bad


def main():
    quiet = "--quiet" in sys.argv
    files = sorted(glob.glob(os.path.join(DISS, "chapters", "*.tex"))
                   + glob.glob(os.path.join(DISS, "appendices", "*.tex")))
    floats = collect(files)
    frag = check_fragments(floats)
    own = check_ownership(files, floats)
    agree = check_agreement(files)
    mute = check_mute(floats)
    orph = check_orphan_continuation(floats)

    def rel(p):
        return os.path.relpath(p, ROOT)

    print(f"split figures {sum(1 for l in floats if re.search(r'-[bcd]$', l))}, "
          f"floats in total {len(floats)}")
    print(f"\n[1] Caption fragments: {len(frag)}")
    for f, ln, lab, why, tail in frag:
        print(f"  {rel(f)}:{ln}  {lab}  {why}\n      …{tail}")
    print(f"[2] Panel ownership misplaced: {len(own)}")
    for f, ln, lets, extra in own:
        print(f"  {rel(f)}:{ln}  panel{lets} lists figures not containing it {extra}")
    print(f"[3] Multi-reference Figures~ at sentence start, then a singular verb: {len(agree)}")
    for f, ln, v, txt in agree:
        print(f"  {rel(f)}:{ln}  '{v}'  {txt}")

    print(f"[4] Split caption never says which panel it draws: {len(mute)}")
    for f, ln, lab in mute:
        print(f"  {rel(f)}:{ln}  {lab}")

    print(f"[5] Continuation caption does not name which figure it continues: {len(orph)}")
    for f, ln, lab, why in orph:
        print(f"  {rel(f)}:{ln}  {lab}  {why}")

    n = len(frag) + len(own) + len(agree) + len(mute) + len(orph)
    if n:
        print(f"\nGATE FAIL: the split left {n} points of damage (baseline 0)")
        return 1
    if not quiet:
        print("\nGATE PASS: all five classes are 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
