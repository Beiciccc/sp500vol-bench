"""T6 -- four ways to write down firm identity: the specification battery.

Emits the markdown of Technical Supplement Table T6 and gates every count
against the values the paper states, so a stale evidence table can never be
tabulated silently.  Nothing is re-fitted; CPU only, under a second.

Sources
  results/tables/firm_fe_battery.csv
      variant, disc, model, h, n_test, n_days, cov_firm, cov_obs,
      rel_firm_pct, rel_zerotext_pct, survives_raw, survives_holm,
      lf_flip_neg, zerotext_beats
  results/tables/firm_identity_ensemble.csv
      rel_impr_pct_firm  (used only for the basis line: the paper's primary
      rung is the seed-ensemble validation-mean specification, 38 of 45
      long-form cells negative, against 40 of 45 on this single-seed battery)

Main-text sentence substantiated (frozen):
  06_results.tex  "One cell survives all four identity specifications
                   (validation-mean, train-mean, expanding point-in-time,
                   shrunk)."

Adversarial repairs folded in (both lenses; every required_change binding):
  * The point-in-time specification is NOT described as "the most defensible".
    The source file's own measured diagnostic is used instead: it is the most
    permissive (14 survivors) and its zero-text term improves the plain
    reference in only 23 of 69 cells against 53-61 for the three fixed-window
    means, because it largely duplicates HAR's own rv_22d input.
  * A basis line is printed: the battery is the single-seed 2026 basis, so its
    validation-mean row reads 40 of 45 long-form cells negative where the
    frozen main text (seed-ensemble) says 38 of 45.
  * Firm coverage for the point-in-time specification is printed per channel
    (0.987 long-form / 0.995 event-driven) and pooled (0.990); the single
    number 0.987 is never presented as the pooled figure.
  * The zero-text column is reported as comparisons of 6 first, with the cell
    expansion in brackets: the flag takes one distinct value per (channel,
    horizon), so 53 is not a count of independent survivors.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp_style import TAB, gate

B = pd.read_csv(os.path.join(TAB, "firm_fe_battery.csv"))
E = pd.read_csv(os.path.join(TAB, "firm_identity_ensemble.csv"))
B["cell"] = B.disc + " / " + B.model + " / h=" + B.h.astype(str)

ORDER = ["valmean", "trainmean", "pit252", "ebshrink"]
NAME = {
    "valmean": "(i) validation-window firm mean [primary rung]",
    "trainmean": "(ii) train-period (2010-19) firm mean",
    "pit252": "(iii) expanding point-in-time pre-filing mean",
    "ebshrink": "(iv) empirical-Bayes shrunk validation mean (k=10)",
}


def per(v, disc, col):
    return float(B[(B.variant == v) & (B.disc == disc)][col].iloc[0])


rows = []
for v in ORDER:
    d = B[B.variant == v]
    dlf, ded = d[d.disc == "long_form"], d[d.disc == "event_driven"]
    six = d.drop_duplicates(["disc", "h"])
    rows.append({
        "spec": NAME[v],
        "cov_firm": f"{per(v, 'long_form', 'cov_firm'):.3f} / "
                    f"{per(v, 'event_driven', 'cov_firm'):.3f}",
        "cov_firm_pool": f"{d.cov_firm.mean():.3f}",
        "cov_obs": f"{per(v, 'long_form', 'cov_obs'):.3f} / "
                   f"{per(v, 'event_driven', 'cov_obs'):.3f}",
        "holm": int(d.survives_holm.sum()),
        "raw": int(d.survives_raw.sum()),
        "lfneg": int((dlf.rel_firm_pct < 0).sum()),
        "zt": f"{int(six.zerotext_beats.sum())} of 6 ({int(d.zerotext_beats.sum())})",
        "rel": f"{dlf.rel_firm_pct.mean():+.2f} / {ded.rel_firm_pct.mean():+.2f}",
        "ztrel": f"{dlf.rel_zerotext_pct.mean():+.2f} / "
                 f"{ded.rel_zerotext_pct.mean():+.2f}",
    })

piv = (B.pivot_table(index="cell", columns="variant", values="survives_holm",
                     aggfunc="first")[ORDER])
ncleared = piv.sum(axis=1)
tiers = {k: sorted(ncleared[ncleared == k].index.tolist()) for k in (4, 3, 2, 1)}

# The three fixed-window specifications (valmean, trainmean, ebshrink) intersect
# in MORE cells than all four do: the single cell the frozen main text reports is
# an all-four count, and only the expanding point-in-time column removes the
# second cell.  Derived here so the distinction can never be hand-typed.
FIXED = ["valmean", "trainmean", "ebshrink"]
three_fixed = sorted(piv.index[piv[FIXED].all(axis=1)].tolist())

elf = E[E.disc == "long_form"]

gate(
    {
        "n_variants": 4,
        "cells_per_variant": 69,
        "holm": [8, 10, 14, 8],
        "raw": [14, 17, 27, 14],
        "lf_negative": [40, 24, 14, 39],
        "lf_mean_rel": [-2.25, -0.98, 0.39, -2.18],
        "zerotext_cells": [53, 61, 23, 61],
        "zerotext_comparisons": [4, 5, 2, 5],
        "cov_firm_lf": [0.629, 0.856, 0.987, 0.629],
        "cov_firm_pooled": [0.629, 0.857, 0.990, 0.629],
        "all_four": ["event_driven / C6_llmtext / h=5"],
        "three_fixed_window": ["event_driven / C6_llmtext / h=5",
                               "long_form / B3_lm_linear / h=10"],
        "n_any": 22,
        "survivor_slots": 40,
        "tier_sizes": [1, 2, 11, 8],
        # basis: the battery is single-seed 2026, the paper's rung is ensemble
        "ensemble_lf_negative": 38,
        "battery_valmean_lf_negative": 40,
    },
    {
        "n_variants": B.variant.nunique(),
        "cells_per_variant": int(B.groupby("variant").size().unique()[0]),
        "holm": [int(B[B.variant == v].survives_holm.sum()) for v in ORDER],
        "raw": [int(B[B.variant == v].survives_raw.sum()) for v in ORDER],
        "lf_negative": [int((B[(B.variant == v) & (B.disc == "long_form")]
                             .rel_firm_pct < 0).sum()) for v in ORDER],
        "lf_mean_rel": [round(float(B[(B.variant == v) & (B.disc == "long_form")]
                                    .rel_firm_pct.mean()), 2) for v in ORDER],
        "zerotext_cells": [int(B[B.variant == v].zerotext_beats.sum())
                           for v in ORDER],
        "zerotext_comparisons": [
            int(B[B.variant == v].drop_duplicates(["disc", "h"])
                .zerotext_beats.sum()) for v in ORDER],
        "cov_firm_lf": [round(per(v, "long_form", "cov_firm"), 3) for v in ORDER],
        "cov_firm_pooled": [round(float(B[B.variant == v].cov_firm.mean()), 3)
                            for v in ORDER],
        "all_four": tiers[4],
        "three_fixed_window": three_fixed,
        "n_any": int((ncleared > 0).sum()),
        "survivor_slots": int(piv.values.sum()),
        "tier_sizes": [len(tiers[k]) for k in (4, 3, 2, 1)],
        "ensemble_lf_negative": int((elf.rel_impr_pct_firm < 0).sum()),
        "battery_valmean_lf_negative": int(
            (B[(B.variant == "valmean") & (B.disc == "long_form")]
             .rel_firm_pct < 0).sum()),
    },
)

HDR = ("| specification | firm cov. LF / ED | pooled | test-row cov. LF / ED | "
       "Holm /69 | raw /69 | LF negative /45 | zero-text beats f_R | "
       "mean rel% LF / ED |")
print(HDR)
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| {spec} | {cov_firm} | {cov_firm_pool} | {cov_obs} | {holm} | "
          "{raw} | {lfneg} | {zt} | {rel} |".format(**r))
print()
print("| zero-text term's own mean rel% vs plain f_R (LF / ED) |")
for r, v in zip(rows, ORDER):
    print("| {}: {} |".format(NAME[v], r["ztrel"]))
print()
print("| specifications cleared | cells | which |")
print("|---|---|---|")
for k in (4, 3, 2, 1):
    which = "; ".join(tiers[k]) if k >= 3 else "-"
    print(f"| {k} of 4 | {len(tiers[k])} | {which} |")
print(f"| at least one | {int((ncleared > 0).sum())} | - |")
print(f"| survivor-slots summed over the four columns | "
      f"{int(piv.values.sum())} | - |")
print()
print("two-of-four cells: " + "; ".join(tiers[2]))
print("one-of-four cells: " + "; ".join(tiers[1]))
print()
print(f"intersection of the three fixed-window specifications "
      f"({', '.join(FIXED)}): {len(three_fixed)} cells -- "
      + "; ".join(three_fixed)
      + f"; adding the expanding point-in-time column leaves {len(tiers[4])} "
        "(the all-four count the main text reports).")
print()
print("BASIS: battery is the single-seed 2026 basis; the ensemble primary rung "
      f"has {int((elf.rel_impr_pct_firm < 0).sum())} of {len(elf)} long-form "
      f"cells negative against "
      f"{int((B[(B.variant == 'valmean') & (B.disc == 'long_form')].rel_firm_pct < 0).sum())}"
      f" of 45 here.")
