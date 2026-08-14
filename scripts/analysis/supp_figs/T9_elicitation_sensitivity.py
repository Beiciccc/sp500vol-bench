"""T9 -- elicitation and decoding sensitivity.

Emits the markdown of Technical Supplement Table T9 and gates every count
against the values the paper states.  Nothing is re-fitted; CPU only, under a
second.

Sources
  results/tables/elicitation_sensitivity.csv
      section in {repeat, agreement, m1}; arm, metric, exact, spearman,
      reldiff, rel_pct, dm, p, n_days
  results/tables/elicitation_sensitivity.md
      the parse-success column of the agreement block, which the CSV does not
      carry; parsed here rather than typed

Main-text sentences substantiated (frozen)
  07_ablations.tex "on a 4,000-filing subsample the event-driven h=5 increment
    is positive in all five elicitation arms (+0.63--1.13%) while the long-form
    one flips under paraphrase."
  04_methods.tex "Repeat decodes are 94--97% exact-equal (Stress Tests)."

Unit trap this script exists to prevent: the source stores `reldiff` as a
FRACTION while `rel_pct` is already a percentage.  The two must never share a
formatter, so block (a)'s mean relative difference is multiplied by 100 here
and block (c)'s increments are not.

Scope carried on the artefact: this family carries RAW p only, with no
multiplicity correction, and runs on a 4,000-filing subsample (286-290
long-form and 787-800 event-driven trading days), so nothing in it may be
upgraded to a significance statement or compared against the full-panel
increments elsewhere in the supplement.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp_style import TAB, gate

E = pd.read_csv(os.path.join(TAB, "elicitation_sensitivity.csv"))
MD = open(os.path.join(TAB, "elicitation_sensitivity.md")).read()

HH = [5, 10, 20]
REP = E[E.section == "repeat"].copy()
REP["h"] = REP.metric.str.extract(r"vol_(\d+)d").astype(int)
REP = REP.sort_values("h")

AGR = E[E.section == "agreement"].copy()
ARMS = [("base_rep1", "base prompt (decode 1)"),
        ("rep2", "repeat decode (identical config)"),
        ("para1", "paraphrase 1"),
        ("para2", "paraphrase 2"),
        ("think", "thinking mode")]
AGR_NAME = {"para1": "paraphrase 1", "para2": "paraphrase 2",
            "think": "thinking mode"}

M1 = E[E.section == "m1"].copy()
M1["disc"] = M1.metric.str.replace(r"_h\d+$", "", regex=True)
M1["h"] = M1.metric.str.extract(r"_h(\d+)$").astype(int)

# parse success lives only in the committed markdown; read it, do not type it
parse_ok = {}
for m in re.finditer(r"^\|\s*(para1|para2|think)\s*\|\s*(\d+)\s*\|\s*"
                     r"([\d.]+)\s*\|\s*([\d.]+)\s*\|", MD, re.M):
    parse_ok[m.group(1)] = (int(m.group(2)), float(m.group(4)))
assert set(parse_ok) == {"para1", "para2", "think"}, parse_ok


def cell(disc, arm, h):
    s = M1[(M1.disc == disc) & (M1.arm == arm) & (M1.h == h)]
    assert len(s) == 1, f"{disc}/{arm}/h={h}: {len(s)} rows"
    return s.iloc[0]


ed_h5 = np.array([float(cell("event_driven", a, 5).rel_pct) for a, _ in ARMS])
lf_h20 = np.array([float(cell("long_form", a, 20).rel_pct) for a, _ in ARMS])
nd = {d: sorted({int(v) for v in M1[M1.disc == d].n_days})
      for d in ("long_form", "event_driven")}

# Adversarial repair: the draft prose called the subsample "roughly a third of
# the event-driven trading days and a third of the long-form ones". Only the
# long-form figure is a third. Both coverages are derived here from the
# committed full-panel file and gated, so no fraction is typed into the prose.
FULL = pd.read_csv(os.path.join(TAB, "m1_ensemble_primary.csv"))
FULL = FULL[FULL.model == "C6_llmtext"]
nd_full = {d: {int(r.h): int(r.n_days)
               for r in FULL[FULL.disc == d].itertuples()}
           for d in ("long_form", "event_driven")}
nd_sub = {d: {int(r.h): int(r.n_days) for r in M1[M1.disc == d].itertuples()}
          for d in ("long_form", "event_driven")}
cover = {d: [round(100 * nd_sub[d][h] / nd_full[d][h], 1) for h in HH]
         for d in ("long_form", "event_driven")}

# ------------------------------------------------------------------- gate
gate(
    {"n_subsample": 4000, "n_arms": 5, "n_horizons": 3, "n_channels": 2,
     "exact": [0.97, 0.945, 0.936],
     "spearman_repeat": [0.9639, 0.9538, 0.9244],
     "reldiff_pct": [0.59, 0.68, 1.08],
     "agreement": {"para1": 0.578, "para2": 0.413, "think": 0.391},
     "parse_ok": {"para1": 1.0, "para2": 1.0, "think": 1.0},
     "agreement_n": {"para1": 4000, "para2": 4000, "think": 4000},
     "ed_h5_all_positive": True, "ed_h5_range": [0.63, 1.13],
     "lf_h20_max_arm": "para1", "lf_h20_max": 3.14,
     "lf_h20_min_arm": "para2", "lf_h20_min": -3.77,
     "n_days_lf": [286, 287, 290], "n_days_ed": [787, 798, 800],
     "n_days_full_lf": [809, 803, 794], "n_days_full_ed": [996, 991, 981],
     "cover_lf_pct": [35.8, 35.7, 36.0], "cover_ed_pct": [80.3, 80.5, 80.2],
     "holm_columns_present": False},
    {"n_subsample": int(parse_ok["para1"][0]), "n_arms": int(M1.arm.nunique()),
     "n_horizons": int(M1.h.nunique()), "n_channels": int(M1.disc.nunique()),
     "exact": [round(float(v), 3) for v in REP.exact],
     "spearman_repeat": [round(float(v), 4) for v in REP.spearman],
     "reldiff_pct": [round(100 * float(v), 2) for v in REP.reldiff],
     "agreement": {r.arm: round(float(r.spearman), 3)
                   for r in AGR.itertuples()},
     "parse_ok": {k: v[1] for k, v in parse_ok.items()},
     "agreement_n": {k: v[0] for k, v in parse_ok.items()},
     "ed_h5_all_positive": bool((ed_h5 > 0).all()),
     "ed_h5_range": [round(float(ed_h5.min()), 2), round(float(ed_h5.max()), 2)],
     "lf_h20_max_arm": ARMS[int(np.argmax(lf_h20))][0],
     "lf_h20_max": round(float(lf_h20.max()), 2),
     "lf_h20_min_arm": ARMS[int(np.argmin(lf_h20))][0],
     "lf_h20_min": round(float(lf_h20.min()), 2),
     "n_days_lf": nd["long_form"], "n_days_ed": nd["event_driven"],
     "n_days_full_lf": [nd_full["long_form"][h] for h in HH],
     "n_days_full_ed": [nd_full["event_driven"][h] for h in HH],
     "cover_lf_pct": cover["long_form"], "cover_ed_pct": cover["event_driven"],
     "holm_columns_present": bool(
         any(c.lower().startswith("p_") and "holm" in c.lower()
             for c in E.columns))},
)

# ------------------------------------------------------------------ markdown
print("**Basis for the whole table.** All three blocks run on the same "
      f"{parse_ok['para1'][0]:,}-filing subsample. This family carries RAW p "
      "only: no multiplicity correction is applied anywhere in it. The "
      "subsample gives "
      f"{nd['long_form'][0]}--{nd['long_form'][-1]} long-form and "
      f"{nd['event_driven'][0]}--{nd['event_driven'][-1]} event-driven trading "
      f"days, against {nd_full['long_form'][20]}--{nd_full['long_form'][5]} and "
      f"{nd_full['event_driven'][20]}--{nd_full['event_driven'][5]} in the full "
      f"panel ({min(cover['long_form']):.1f}--{max(cover['long_form']):.1f}% "
      f"and {min(cover['event_driven']):.1f}--"
      f"{max(cover['event_driven']):.1f}% of the days), so these percentages "
      "are not comparable with the full-panel cells elsewhere in the "
      "supplement, and nothing here may be upgraded to a significance "
      "statement.\n")

print("### (a) Repeat-decode stability at temperature 0\n")
print("| horizon | exact-equal share | Spearman rank correlation | mean "
      "relative difference |")
print("|---|--:|--:|--:|")
for r in REP.itertuples():
    print(f"| h={r.h} | {r.exact:.3f} | {r.spearman:.4f} | "
          f"{100 * r.reldiff:.2f}% |")
print("\nThe source stores the last column as a FRACTION while the increment "
      "columns below are already percentages; the two do not share a "
      "formatter.\n")

print("### (b) Cross-template rank agreement against the base arm, at the "
      "individual filing level\n")
print("| arm | n | Spearman vs base (h=10) | parse success |")
print("|---|--:|--:|--:|")
for r in AGR.itertuples():
    n, ok = parse_ok[r.arm]
    print(f"| {AGR_NAME[r.arm]} | {n:,} | {r.spearman:.3f} | {ok:.3f} |")
print()

print("### (c) Increment over the recalibrated HAR, by elicitation arm "
      "(rel%, with day-clustered DM and raw p)\n")
print("| arm | channel | h=5 | h=10 | h=20 |")
print("|---|---|--:|--:|--:|")
for disc, nm in (("event_driven", "8-K"), ("long_form", "10-K/Q")):
    for a, lab in ARMS:
        cells = []
        for h in HH:
            r = cell(disc, a, h)
            cells.append(f"{r.rel_pct:+.2f} ({r.dm:+.2f}; {r.p:.4f})")
        print(f"| {lab} | {nm} | " + " | ".join(cells) + " |")
print()

print("### Verification lines (not part of the table)\n")
print("8-K h=5 across the five arms: "
      + " / ".join(f"{v:+.2f}" for v in ed_h5)
      + f"  -- positive in {int((ed_h5 > 0).sum())} of 5, range "
        f"{ed_h5.min():+.2f} to {ed_h5.max():+.2f}%")
print("10-K/Q h=20 across the five arms: "
      + " / ".join(f"{v:+.2f}" for v in lf_h20)
      + f"  -- swings from {lf_h20.max():+.2f}% ({ARMS[int(np.argmax(lf_h20))][1]}) "
        f"to {lf_h20.min():+.2f}% ({ARMS[int(np.argmin(lf_h20))][1]})")
print("trading days: long-form " + "/".join(map(str, nd["long_form"]))
      + ", event-driven " + "/".join(map(str, nd["event_driven"])))
print("subsample coverage of the full-panel C6 days, h=5/10/20: long-form "
      + "/".join(f"{nd_sub['long_form'][h]}of{nd_full['long_form'][h]}"
                 for h in HH)
      + " = " + "/".join(f"{v:.1f}%" for v in cover["long_form"])
      + "; event-driven "
      + "/".join(f"{nd_sub['event_driven'][h]}of{nd_full['event_driven'][h]}"
                 for h in HH)
      + " = " + "/".join(f"{v:.1f}%" for v in cover["event_driven"]))
print("committed columns: " + ", ".join(E.columns)
      + "  -- no Holm column exists in this family")
