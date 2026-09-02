"""Measure the cadence of a section, so a prose pass can be checked rather than felt.

Three numbers, chosen because they are what actually went wrong in this draft
rather than because they are the usual suspects:

  article_open   share of sentences beginning "The" or "A".  At 38% the prose
                 reads as a list of definitions, because an abstract noun is
                 doing the work in nearly every subject position.
  colon_rate     mid-sentence colon appositions per hundred words.  One is a
                 definition; six in a subsection is a rhythm the author fell into.
  len_sd         standard deviation of sentence length in words.  A low value is
                 the giveaway: every sentence the same size means nothing lands.

Run before and after; the point is the direction of travel, not a threshold.

    python3 prose_cadence.py                      # the nine edited sections
    python3 prose_cadence.py --ref HEAD           # compare against a commit
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CH = "writing/dissertation/chapters"

SECTIONS = [
    ("01_introduction.tex", "section", "Project Aim"),
    ("01_introduction.tex", "section", "Structure of the Report"),
    ("02_background.tex", "subsection", "Price-Based Volatility Forecasting"),
    ("02_background.tex", "subsection", "Shortcut Learning and Evaluation Hygiene"),
    ("02_background.tex", "subsection", "Encompassing, Combination, Recalibration"),
    ("02_background.tex", "subsection", "Fine-Tuning, Frozen Embeddings and Prompting"),
    ("03_datasets_and_experimental_design.tex", "section", "Datasets and Data Sources"),
    ("03_datasets_and_experimental_design.tex", "subsection", "The Point-in-Time Firm Universe"),
    ("05_validation.tex", "subsection", "Within-Date Permutation"),
]


def slice_section(src, lvl, name):
    i = src.find("\\" + lvl + "{" + name)
    if i < 0:
        i = src.find(name)
        if i < 0:
            return ""
        i = src.rfind("\\", 0, i)
    nxt = [src.find(t, i + 10)
           for t in ("\\section{", "\\subsection{", "\\subsubsection{")]
    j = min([x for x in nxt if x > 0], default=len(src))
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in src[i:j].split("\n"))
    body = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", " ", body, flags=re.S)
    return re.sub(r"\s+", " ", body)


def cadence(body):
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", body)
             if len(x.split()) > 3]
    if not sents:
        return None
    lens = [len(x.split()) for x in sents]
    art = sum(1 for x in sents if re.match(r"^(The|A|An)\b", x))
    # a colon between two lower-case words is apposition, not a list header
    colon = len(re.findall(r"[a-z]\w*:\s+[a-z]", body))
    return dict(n=len(sents), words=sum(lens),
                article_open=100.0 * art / len(sents),
                colon_rate=100.0 * colon / max(sum(lens), 1),
                len_sd=statistics.pstdev(lens) if len(lens) > 1 else 0.0,
                len_mean=statistics.mean(lens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="git ref to compare against")
    a = ap.parse_args()

    def read(fn, ref):
        if ref is None:
            return open(os.path.join(ROOT, CH, fn)).read()
        out = subprocess.run(["git", "show", f"{ref}:{CH}/{fn}"], cwd=ROOT,
                             capture_output=True, text=True)
        return out.stdout

    hdr = f"{'section':<44}{'sent':>5}{'words':>6}{'The/A %':>9}{'colon':>7}{'len sd':>8}"
    print(hdr if a.ref is None else hdr + "     (before -> after)")
    agg = []
    for fn, lvl, name in SECTIONS:
        now = cadence(slice_section(read(fn, None), lvl, name))
        if now is None:
            print(f"  {name[:40]:<42} (not found)")
            continue
        agg.append(now)
        if a.ref is None:
            print(f"  {name[:40]:<42}{now['n']:>5}{now['words']:>6}"
                  f"{now['article_open']:>9.0f}{now['colon_rate']:>7.2f}"
                  f"{now['len_sd']:>8.1f}")
        else:
            was = cadence(slice_section(read(fn, a.ref), lvl, name)) or now
            print(f"  {name[:40]:<42}{now['n']:>5}{now['words']:>6}"
                  f"{was['article_open']:>5.0f}->{now['article_open']:<3.0f}"
                  f"{was['colon_rate']:>4.1f}->{now['colon_rate']:<3.1f}"
                  f"{was['len_sd']:>5.1f}->{now['len_sd']:<4.1f}")
    if agg:
        tot_w = sum(x["words"] for x in agg)
        tot_s = sum(x["n"] for x in agg)
        art = sum(x["article_open"] * x["n"] for x in agg) / tot_s
        col = sum(x["colon_rate"] * x["words"] for x in agg) / tot_w
        print(f"\n  overall: {tot_s} sentences, {tot_w} words, "
              f"{art:.0f}% open with an article, {col:.2f} colons per 100 words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
