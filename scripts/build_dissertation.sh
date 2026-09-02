#!/bin/zsh
# Build the dissertation and print every invariant that must not be broken.
#
# Why this script lives in the repo and not in /tmp: its predecessor sat in a session temp
# directory and was cleaned out once; more to the point, its predecessor's latexmk call once
# failed silently with "command not found", and the script cheerfully went on to report a
# main.pdf that was five hours old -- I took that report for a passing build. The giveaway
# was that the per-chapter margins were byte-for-byte unchanged after edits that plainly move
# the layout. So PATH is set here by hand, and the presence of latexmk is a hard gate.
#
# Usage:  scripts/build_dissertation.sh [--clean] [--no-figures]
# An exit code other than 0 == an invariant was broken. Can be wired straight into CI or a Makefile.
set -u
export PATH="/Library/TeX/texbin:$PATH"
command -v latexmk >/dev/null || { echo "FAIL: latexmk not on PATH"; exit 1; }

ROOT=${0:A:h:h}
D="$ROOT/writing/dissertation"
FAIL=0
note() { printf "  %-34s %s\n" "$1" "$2" }
bad()  { printf "  %-34s %s   <-- FAIL\n" "$1" "$2"; FAIL=1 }

# ---- Figure geometry gate. It runs before the build because the failure it catches is
#      silent: a figure regenerated with the wrong driver still renders, still builds, and
#      LaTeX issues not one warning.
if [[ "${*}" != *--no-figures* ]]; then
  echo "=== Figures: minimum font-size gate ==="
  if python3 "$ROOT/scripts/analysis/diss_appendix_figs/audit_inclusion_geometry.py" --gate \
       > /tmp/diss_figgate.out 2>&1; then
    note "appendix figure print size >= 9pt" "PASS"
  else
    bad "appendix figure print size >= 9pt" "$(grep -m1 GATE /tmp/diss_figgate.out)"
  fi
fi

# ---- Provenance gate for the concept figures. Those two figures are not generator-drawn
#      evidence figures, so there is no gate(...) to run; the way they go wrong is by carrying
#      a number that does not exist anywhere in the report. This check reads every number on
#      the figures back into the chapters and checks it.
# ---- Completeness: orphan floats, named methods uncited at first appearance. The criteria
#      and the reasons for the retained items are written up in the script's docstring; only
#      these two are gated here, the rest is read as a report.
echo "=== Completeness: orphans and uncited ==="
if python3 "$ROOT/scripts/analysis/completeness_audit.py" --gate \
     > /tmp/diss_complete.out 2>&1; then
  note "no orphan floats/uncited first use" "PASS"
else
  bad "no orphan floats/uncited first use" "$(grep -m1 'GATE FAIL' /tmp/diss_complete.out)"
fi

echo "=== Concept figures: provenance gate ==="
if python3 "$ROOT/scripts/analysis/diss_figs/concept_check.py" > /tmp/diss_concept.out 2>&1; then
  note "figure numbers traceable to text" "PASS"
else
  bad "figure numbers traceable to text" "$(grep -m1 FAIL /tmp/diss_concept.out)"
fi

# ---- src comment census: the two numbers printed in appendix A are self-referential (the
# sentence citing them carries a src of its own), so they must be recomputed and checked by
# script rather than relying on the author remembering to update them.
echo "=== src comments: census gate ==="
if python3 "$ROOT/scripts/analysis/src_comment_census.py" --check > /tmp/diss_census.out 2>&1; then
  note "appendix A src census matches src" "PASS"
else
  bad "appendix A src census matches src" "$(grep -m1 FAIL /tmp/diss_census.out)"
fi

# ---- % eats the line: if anything that still needs typesetting is left after a comment
# (\ref, \emph, half a sentence), LaTeX swallows it silently, neither the log nor the page
# count shows it, and it surfaces only as a sentence in the body that starts halfway through.
# Fallen into this hole three times now.
echo "=== Comments: have they swallowed body text or cross-references? ==="
if python3 "$ROOT/scripts/analysis/comment_swallow_check.py" > /tmp/diss_swallow.out 2>&1; then
  note "comments swallow no body text" "PASS"
else
  bad "comments swallow no body text" "$(grep -m1 FAIL /tmp/diss_swallow.out)"
fi

# ---- Caption provenance: a caption prints a measured value while the whole float carries not
# one src. The baseline is not 0 -- captions sometimes restate a denominator or a design
# constant already defined in chapter 3 (69, 180, 0.05, 80% power).
echo "=== Captions: numbers printed with no provenance ==="
if python3 "$ROOT/scripts/analysis/caption_figure_audit.py" --max 7 --quiet \
     > /tmp/diss_capaudit.out 2>&1; then
  note "caption numbers have a source" "$(grep -m1 'no source comment' /tmp/diss_capaudit.out | sed 's/^ *//')"
else
  bad "caption numbers have a source" "$(grep -m1 'GATE FAIL' /tmp/diss_capaudit.out)"
fi

# ---- Split figures: the silent damage left after a composite figure is split into -b/-c/-d
# (fragmented captions, misplaced panel attribution, a singular verb left behind after
# references were expanded in bulk). Baseline 0 for all of them, details in the script's docstring.
echo "=== Split figures: fragments/attribution/agreement/dumb captions/orphan continuations ==="
if python3 "$ROOT/scripts/analysis/split_figure_audit.py" --quiet \
     > /tmp/diss_splitfig.out 2>&1; then
  note "all five split-figure faults 0" "PASS"
else
  bad "all five split-figure faults 0" "$(grep -m1 'GATE FAIL' /tmp/diss_splitfig.out)"
fi

# ---- Printed-number reconciliation: for every number printed, can a value equal to it after
# rounding be found in the file its src names. The baseline is not 0 -- derived quantities
# (medians, sums, percentages) do not appear literally in the first place, and two more sources
# are binary parquet. It is a problem only when the baseline goes up: that means a newly
# written number did not land in the source it claims.
echo "=== Printed numbers: reconciled against the file named by src ==="
if python3 "$ROOT/scripts/analysis/src_number_audit.py" --min-dec 0 --max-miss 13 --quiet \
     > /tmp/diss_numaudit.out 2>&1; then
  note "printed numbers traceable to src" "$(grep -m1 'not found' /tmp/diss_numaudit.out | sed 's/^ *//')"
else
  bad "printed numbers traceable to src" "$(grep -m1 'GATE FAIL' /tmp/diss_numaudit.out)"
fi

cd "$D" || exit 1
[[ "${*}" == *--clean* ]] && latexmk -C >/dev/null 2>&1

echo "=== Build ==="
if latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > /tmp/diss_build.out 2>&1; then
  note "latexmk" "rc=0"
else
  bad "latexmk" "rc!=0"; grep -n "^!" /tmp/diss_build.out | head -10
fi

echo "=== Pages and caps ==="
note "total pages" "$(pdfinfo main.pdf | awk '/^Pages/{print $2}')"
# Printed page number != PDF page index: the front matter uses roman numerals, so body p1 falls
# on PDF page 1+offset. main.toc records printed page numbers, so the chapter-start check below
# is right; but a manual pdftotext -f/-l uses the PDF page index. I once measured the per-chapter
# margins by PDF page index and the whole set of numbers was worthless. So the offset is printed.
python3 - <<'PY'
import re, subprocess
n = int(subprocess.run(["pdfinfo", "main.pdf"], capture_output=True, text=True)
        .stdout.split("Pages:")[1].split()[0])
for p in range(1, n + 1):
    h = subprocess.run(["pdftotext", "-layout", "-f", str(p), "-l", str(p), "main.pdf", "-"],
                       capture_output=True, text=True).stdout.split("\n")[0]
    if "CHAPTER" in h or "APPENDIX" in h:
        nums = re.findall(r"\b(\d{1,3})\b", h)
        if nums:
            off = p - int(nums[-1])
            print(f"  {'printed p1 = PDF page':<34}{1 + off}  (offset +{off}; "
                  f"body 1..60 = PDF {1 + off}..{60 + off})")
            break
PY
python3 - "$D" <<'PY'
import re, sys
toc = open(sys.argv[1] + "/main.toc").read()
st = {m.group(1): int(m.group(2)) for m in
      re.finditer(r'contentsline \{chapter\}\{\\numberline \{([0-9])\}[^}]*\}\{(\d+)\}', toc)}
ref = re.search(r'contentsline \{chapter\}\{References\}\{(\d+)\}', toc)
base = {'1': 1, '2': 6, '3': 16, '4': 28, '5': 44, '6': 58}
ok = st == base
print(f"  {'chapter starts 1/6/16/28/44/58':<34}{'match' if ok else 'changed ' + str(st)}"
      + ("" if ok else "   <-- FAIL"))
if ref:
    body = int(ref.group(1)) - 1
    print(f"  {'body pages':<34}{body}/60"
          + ("" if body <= 60 else "   <-- FAIL over hard cap"))
sys.exit(0 if ok and ref and int(ref.group(1)) - 1 <= 60 else 3)
PY
[[ $? -ne 0 ]] && FAIL=1

echo "=== References and floats ==="
for spec in "undefined reference:Reference \`" "unresolved citation:Citation \`" \
            "duplicate label:multiply defined" "duplicate destination:destination with the same" \
            "overfull hbox:^Overfull \\\\hbox" "overfull vbox:^Overfull \\\\vbox" \
            "float too large:Float too large" "geometry warning:Package geometry Warning"; do
  lbl=${spec%%:*}; pat=${spec#*:}
  n=$(grep -c "$pat" main.log)
  [[ "$n" -eq 0 ]] && note "$lbl" "0" || bad "$lbl" "$n"
done
note "bibliography entries" "$(grep -c '^\\bibitem' main.bbl)"

echo
[[ $FAIL -eq 0 ]] && echo "All invariants pass." || echo "An invariant was broken, see the FAIL above."
exit $FAIL
