"""Orphans and unattributed claims: the checks a marker runs by hand.

Four questions, each of which a finished report should be able to answer without
the author present:

  ORPHAN FLOAT      is every figure and table referred to from somewhere?  An
                    appendix figure nobody points at is a figure the reader never
                    reaches, whatever the appendix preamble promises.
  UNCITED METHOD    does every named estimator, test and procedure carry a
                    citation the first time the report uses its name?  Later
                    mentions do not need one; the first does.
  UNREACHED SECTION is every appendix section pointed to from the main text?
                    The sixty-page cap is the reason the appendices exist, so a
                    section the chapters never send anyone to is dead weight.
  DEAD LABEL        is every \\label used by at least one \\ref?

The point is not to reach zero on all four.  A cross-reference the argument does
not need is worse than a missing one.  The point is that each survivor is a
decision on the record rather than an oversight.  The survivors, and why:

  26 appendix floats reachable only from inside the appendices.  Almost all are
  Appendix C tables, and Section app:res-basis states the convention they follow:
  the chapters compress to headline counts and name the table carrying each one,
  while the rest are the full cell-level backing a reader consults after arriving
  at the appendix.  Naming all 26 in the chapters would cost about eighty words
  of a sixty-page main body to tell a reader something the appendix already says.

  14 appendix sections the chapters do not name.  These are navigational or
  administrative -- repository layout, run commands, the appendix's own "how to
  read these tables" -- reached by opening the appendix rather than by being sent
  there mid-argument.

  eq:m1, the recalibrated combiner, is labelled and never referenced.  The prose
  that uses it stands immediately beneath it and says "the same equation without
  the text term"; a number there would point at what the reader is already
  looking at.  The label is kept so a later edit can reference it from a distance.

    python3 completeness_audit.py            # report
    python3 completeness_audit.py --gate     # exit non-zero on orphan floats
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "writing", "dissertation")

# Named methods the report uses by name.  The list is explicit rather than
# discovered, because "named method" is a judgement: Holm and Diebold--Mariano
# are methods, "the ladder" is this report's own construction and needs no
# citation.
NAMED = [
    "Parkinson", "Garman", "Giacomini", "Duan", "ASHA", "Sobol",
    "Diebold", "Holm", "Newey", "Harvey", "Clark and West", "Hansen",
    "Corsi", "Bollerslev", "Engle", "Nelson", "Patton", "Andersen",
    "Ljung", "Bonferroni", "Benjamini", "Hyperband", "Optuna",
]


def strip(s):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.split("\n"))


def load():
    ch = {os.path.basename(p): strip(open(p).read())
          for p in sorted(glob.glob(os.path.join(D, "chapters", "0*.tex")))}
    ap = {os.path.basename(p): strip(open(p).read())
          for p in sorted(glob.glob(os.path.join(D, "appendices", "*.tex")))}
    return ch, ap


def floats_in(src):
    """(label, kind, short caption) for every float that carries a label."""
    out = []
    for m in re.finditer(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}",
                         src, re.S):
        lab = re.search(r"\\label\{([^}]*)\}", m.group(2))
        cap = re.search(r"\\caption\[([^\]]*)\]", m.group(2))
        if lab:
            out.append((lab.group(1), m.group(1),
                        cap.group(1) if cap else ""))
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--gate", action="store_true")
    a = ap_.parse_args()
    ch, apx = load()
    main_text = "\n".join(ch.values())
    everything = main_text + "\n" + "\n".join(apx.values())
    bad = 0

    print("=== orphan floats: labelled but never \\ref'd ===")
    orphans = []
    for where, files in (("main", ch), ("appendix", apx)):
        for fn, src in files.items():
            for lab, kind, cap in floats_in(src):
                if not re.search(r"\\ref\{" + re.escape(lab) + r"\}", everything):
                    orphans.append((where, fn, lab, kind, cap))
    for w, fn, lab, kind, cap in orphans:
        print(f"  {w:<9}{fn[:2]}  {kind:<7}{lab:<34}{cap[:44]}")
    print(f"  {len(orphans)} orphan(s)")
    bad += len(orphans)

    print("\n=== appendix floats never reached FROM the main text ===")
    far = []
    for fn, src in apx.items():
        for lab, kind, cap in floats_in(src):
            if not re.search(r"\\ref\{" + re.escape(lab) + r"\}", main_text):
                far.append((fn, lab, kind, cap))
    for fn, lab, kind, cap in far:
        print(f"  {fn[:2]}  {kind:<7}{lab:<34}{cap[:52]}")
    print(f"  {len(far)} of {sum(len(floats_in(s)) for s in apx.values())} "
          f"appendix floats are reachable only from inside the appendices")

    print("\n=== named methods without a citation at first use ===")
    miss = []
    for name in NAMED:
        m = re.search(re.escape(name), main_text)
        if not m:
            continue
        window = main_text[max(0, m.start() - 120):m.start() + 260]
        if not re.search(r"\\cite[tp]?\{", window):
            miss.append((name, re.sub(r"\s+", " ",
                                      main_text[max(0, m.start() - 70):
                                                m.start() + 90])))
    for n, ctx in miss:
        print(f"  {n:<16}{ctx[:96]}")
    print(f"  {len(miss)} uncited at first use")
    bad += len(miss)

    print("\n=== appendix sections the main text cannot reach at all ===")
    # A section is reached if the chapters name it, OR name any float inside it:
    # a reader who follows "Figure E.7" lands in the section that holds it, and
    # asking for a second, section-level cross-reference beside every figure
    # reference would be cross-referencing for the audit rather than the reader.
    unreached = []
    for fn, src in apx.items():
        secs = [(m.start(), m.group(2), m.group(1)) for m in
                re.finditer(r"\\section\{([^}]*)\}\s*\\label\{([^}]*)\}", src)]
        for k, (pos, lab, title) in enumerate(secs):
            end = secs[k + 1][0] if k + 1 < len(secs) else len(src)
            inside = [l for l, _, _ in floats_in(src[pos:end])]
            reachable = re.search(r"\\ref\{" + re.escape(lab) + r"\}", main_text) \
                or any(re.search(r"\\ref\{" + re.escape(l) + r"\}", main_text)
                       for l in inside)
            if not reachable:
                unreached.append((fn, lab, title))
    for fn, lab, title in unreached:
        print(f"  {fn[:2]}  {lab:<36}{title[:48]}")
    print(f"  {len(unreached)} unreached")

    print("\n=== dead labels: declared, never referenced ===")
    dead = []
    for fn, src in list(ch.items()) + list(apx.items()):
        for lab in re.findall(r"\\label\{([^}]*)\}", src):
            if lab.startswith(("sec:", "app:", "ch:")):
                continue          # structural anchors are allowed to be unused
            if not re.search(r"\\ref\{" + re.escape(lab) + r"\}", everything):
                dead.append((fn, lab))
    for fn, lab in dead:
        print(f"  {fn[:2]}  {lab}")
    print(f"  {len(dead)} dead")

    if a.gate and bad:
        print(f"\nGATE FAIL: {bad} orphan float(s) or uncited method(s)")
        return 1
    print("\naudit complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
