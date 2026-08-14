"""T10 -- where the 8-K residual lives, by SEC item code.

Emits the markdown of Technical Supplement Table T10 and gates every count
against the values the paper states.  Nothing is re-fitted: the committed test
residual is PARTITIONED by item code, and no model is estimated per stratum.
CPU only, under a second.

Sources
  results/tables/row11_item_stratified.csv (+ .md)
      family, item_group, kind, horizon, is_partition, n_test, n_days,
      share_of_filings_pct, rel_firm_pct, dm_firm, p_firm, p_firm_holm,
      share_of_pooled_residual_pct, abs_reduction_firm, rel_har_pct
  results/tables/crossfamily_llm.csv, results/tables/crossfamily_llama70.csv
      the pooled ALL cells are asserted to reproduce these committed anchors to
      a relative tolerance of 1e-12
  results/tables/itemcode_control.csv
      rel_202, p_202_holm -- the separate control that puts an Item-2.02
      indicator in the reference alongside the firm term (reported in S4, not
      derived here)

Main-text sentence substantiated (frozen)
  06_results.tex "With the firm-identity term and an Item-2.02 earnings-window
  indicator both in the reference, C6 retains +0.21/+0.18/+0.20% at h=5/10/20
  (Holm <= 2.7e-4), so this is not an earnings-announcement artefact."

Scope carried on the artefact: the two pooled rows (ALL, narrative-ALL) are
derived, are NOT part of the partition, and are excluded from the Holm family
of 18 disjoint cells per model family; signed shares outside [0, 100] are
printed as measured and never normalised.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp_style import TAB, gate

D = pd.read_csv(os.path.join(TAB, "row11_item_stratified.csv"))
QW = pd.read_csv(os.path.join(TAB, "crossfamily_llm.csv"))
L70 = pd.read_csv(os.path.join(TAB, "crossfamily_llama70.csv"))
IC = pd.read_csv(os.path.join(TAB, "itemcode_control.csv")).sort_values("h")

# Adversarial repair: this table's ALL rows sit on the FULL event-driven panel
# (+0.45/+0.25/+0.20), whereas the reference ladder's headline +0.52/+0.24/+0.21
# and the maximal-pool cross-reference sit on the firm-coverage-restricted merged
# grid rows. The two bases are read here so the basis line under the table is a
# derived statement rather than a typed one.
FI = pd.read_csv(os.path.join(TAB, "firm_identity_ensemble.csv"))
FI = FI[(FI.disc == "event_driven")
        & (FI.model == "C6_llmtext")].sort_values("h")
CI = pd.read_csv(os.path.join(TAB, "control_intersection_ensemble.csv"))
CI = CI[(CI.disc == "event_driven")
        & (CI.model == "C6_llmtext")].sort_values("h")

FAM = [("qwen3_32b", "Qwen3-32B"), ("llama70_awq", "Llama-3.1-70B-AWQ")]
GROUPS = [("2.02_earnings", "Item 2.02 earnings"),
          ("5.02_leadership", "Item 5.02 leadership"),
          ("7.01_regFD", "Item 7.01 Reg FD"),
          ("8.01_other_events", "Item 8.01 other events"),
          ("5.07_shareholder_vote", "Item 5.07 shareholder vote"),
          ("other_narrative", "other narrative")]
POOLED = [("ALL", "ALL (pooled, non-partition)"),
          ("narrative_ALL", "narrative-ALL (pooled, non-partition)")]
HH = [5, 10, 20]

# ------------------------------------------------- pooled anchors reproduce
for tag, src, sel in (("qwen3_32b", QW, dict(family="qwen3_32b")),
                      ("llama70_awq", L70, dict(family="llama70_awq"))):
    a = D[(D.family == tag) & (D.item_group == "ALL")].sort_values("horizon")
    b = src[(src.family == sel["family"])
            & (src.disc == "event_driven")].sort_values("h")
    for col_a, col_b in (("rel_firm_pct", "rel_firm"), ("dm_firm", "dm_firm"),
                         ("p_firm", "p_firm"), ("rel_har_pct", "rel_har"),
                         ("n_test", "n_test")):
        assert np.allclose(a[col_a].astype(float), b[col_b].astype(float),
                           rtol=1e-12), f"{tag}: pooled anchor drifted on {col_a}"

# --------------------------------------------------------------- summaries
summary = {}
for tag, name in FAM:
    f = D[D.family == tag]
    part = f[f.is_partition]
    allrow = f[f.item_group == "ALL"]
    narr = f[f.item_group == "narrative_ALL"]
    e = part[part.item_group == "2.02_earnings"]
    # the horizon-pooled share is a share of the SUMMED absolute QLIKE
    # reduction, not a mean of the per-horizon shares
    share_202 = 100 * e.abs_reduction_firm.sum() / allrow.abs_reduction_firm.sum()
    nar_part = part[part.item_group != "2.02_earnings"]
    summary[tag] = dict(
        name=name,
        filings=(e.share_of_filings_pct.min(), e.share_of_filings_pct.max()),
        share_202=share_202,
        share_narr=100 - share_202,
        per_h_202=e.sort_values("horizon").share_of_pooled_residual_pct.to_numpy(),
        narr_raw=int(((narr.rel_firm_pct > 0) & (narr.dm_firm < 0)
                      & (narr.p_firm < 0.05)).sum()),
        narr_holm=int((nar_part.p_firm_holm < 0.05).sum()),
        n_narr_cells=len(nar_part),
        holm_all=int((part.p_firm_holm < 0.05).sum()),
        n_part_cells=len(part))

# ------------------------------------------------------------------- gate
gate(
    {"n_partition_cells_per_family": 18, "n_families": 2,
     "n_item_groups": 6, "n_pooled_rows_per_family": 2,
     "share_202_qwen": 54, "share_202_llama": 69,
     "filings_202": [33.2, 33.6],
     "narr_raw_qwen": 3, "narr_raw_llama": 1,
     "narr_holm_qwen": 2, "narr_holm_llama": 0, "n_narr_cells": 15,
     "llama_h20_202_share": -25.6, "llama_h20_narr_share": 125.6,
     "itemcode_rel": [0.21, 0.18, 0.2],
     "itemcode_holm_max": 0.00027,
     "panel_n_test": [25109, 25001, 24732],
     "panel_rel_firm": [0.45, 0.25, 0.2],
     "grid_n_test": [23855, 22785, 22318],
     "grid_rel_firm": [0.52, 0.24, 0.21],
     "grid_rel_maximal": [0.4, 0.12, 0.09],
     "maximal_absorbs_raw": False, "maximal_absorbs_holm": False,
     "itemcode_n_test": [25109, 25001, 24732]},
    {"n_partition_cells_per_family": summary["qwen3_32b"]["n_part_cells"],
     "n_families": int(D.family.nunique()),
     "n_item_groups": int(D[D.is_partition].item_group.nunique()),
     "n_pooled_rows_per_family":
         int(D[(D.family == "qwen3_32b") & ~D.is_partition].item_group.nunique()),
     "share_202_qwen": int(round(summary["qwen3_32b"]["share_202"])),
     "share_202_llama": int(round(summary["llama70_awq"]["share_202"])),
     "filings_202": [round(float(summary["qwen3_32b"]["filings"][0]), 1),
                     round(float(summary["qwen3_32b"]["filings"][1]), 1)],
     "narr_raw_qwen": summary["qwen3_32b"]["narr_raw"],
     "narr_raw_llama": summary["llama70_awq"]["narr_raw"],
     "narr_holm_qwen": summary["qwen3_32b"]["narr_holm"],
     "narr_holm_llama": summary["llama70_awq"]["narr_holm"],
     "n_narr_cells": summary["qwen3_32b"]["n_narr_cells"],
     "llama_h20_202_share": round(float(
         D[(D.family == "llama70_awq") & (D.item_group == "2.02_earnings")
           & (D.horizon == 20)].share_of_pooled_residual_pct.iloc[0]), 1),
     "llama_h20_narr_share": round(float(
         D[(D.family == "llama70_awq") & (D.item_group == "narrative_ALL")
           & (D.horizon == 20)].share_of_pooled_residual_pct.iloc[0]), 1),
     "itemcode_rel": [round(v, 2) for v in IC.rel_202],
     "itemcode_holm_max": round(float(IC.p_202_holm.max()), 5),
     "panel_n_test": [int(v) for v in D[(D.family == "qwen3_32b")
                                        & (D.item_group == "ALL")]
                      .sort_values("horizon").n_test],
     "panel_rel_firm": [round(float(v), 2) for v in
                        D[(D.family == "qwen3_32b") & (D.item_group == "ALL")]
                        .sort_values("horizon").rel_firm_pct],
     "grid_n_test": [int(v) for v in FI.n_test],
     "grid_rel_firm": [round(float(v), 2) for v in FI.rel_impr_pct_firm],
     "grid_rel_maximal": [round(float(v), 2) for v in CI.rel_impr_pct_maximal],
     "maximal_absorbs_raw": bool(CI.maximal_raw.any()),
     "maximal_absorbs_holm": bool(CI.maximal_holm.any()),
     "itemcode_n_test": [int(v) for v in IC.n_test]},
)


# ------------------------------------------------------------------ markdown
def pfmt(p):
    if pd.isna(p):
        return "--"
    p = float(p)
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def row(r, label):
    holm = r.p_firm_holm
    surv = (not pd.isna(holm)) and float(holm) < 0.05 and float(r.dm_firm) < 0
    b = "**" if surv else ""
    return (f"| {b}{label}{b} | {r.kind} | {int(r.horizon)} | {int(r.n_test)} | "
            f"{r.share_of_filings_pct:.1f} | {b}{r.rel_firm_pct:+.2f}{b} | "
            f"{r.dm_firm:+.2f} | {pfmt(r.p_firm)} | {pfmt(holm)} | "
            f"{r.share_of_pooled_residual_pct:+.1f} |")


for tag, name in FAM:
    f = D[D.family == tag]
    print(f"### {name} -- 8-K test residual partitioned by item code "
          f"(firm-identity-augmented reference)\n")
    print("| item group | kind | h | n_test | % of filings | rel% vs firm-ID "
          "reference | day-clustered DM | raw p | Holm p (18 cells) | share of "
          "pooled residual % |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for h in HH:
        for key, lab in [POOLED[0]] + GROUPS + [POOLED[1]]:
            r = f[(f.item_group == key) & (f.horizon == h)].iloc[0]
            print(row(r, lab))
    print()

print("### Summary (all three horizons pooled; shares of the SUMMED absolute "
      "QLIKE reduction)\n")
print("| family | Item 2.02 share of filings | Item 2.02 share of pooled "
      "residual | narrative share | narrative-ALL positive with DM<0 and raw "
      "p<.05 | narrative partition cells clearing Holm |")
print("|---|--:|--:|--:|--:|--:|")
for tag, name in FAM:
    s = summary[tag]
    print(f"| {name} | {s['filings'][0]:.1f}--{s['filings'][1]:.1f}% | "
          f"{s['share_202']:+.0f}% | {s['share_narr']:+.0f}% | "
          f"{s['narr_raw']} of 3 horizons | {s['narr_holm']} of "
          f"{s['n_narr_cells']} |")

print("\n### Verification lines (not part of the table)\n")
for tag, name in FAM:
    s = summary[tag]
    print(f"{name}: per-horizon Item 2.02 share of pooled residual "
          f"{' / '.join(f'{v:+.1f}%' for v in s['per_h_202'])}; "
          f"Holm survivors among the {s['n_part_cells']} partition cells "
          f"{s['holm_all']}")
print("pooled ALL cells reproduce the committed crossfamily anchors to "
      "rtol 1e-12 (asserted in-script)")
print("row basis of this table (full event-driven panel): n_test "
      + " / ".join(str(int(v)) for v in D[(D.family == "qwen3_32b")
                                          & (D.item_group == "ALL")]
                   .sort_values("horizon").n_test)
      + ", rel% vs firm identity "
      + " / ".join(f"{v:+.2f}" for v in D[(D.family == "qwen3_32b")
                                          & (D.item_group == "ALL")]
                   .sort_values("horizon").rel_firm_pct))
print("row basis of the reference-ladder headline and of the maximal-pool "
      "cross-reference (merged grid rows): n_test "
      + " / ".join(str(int(v)) for v in FI.n_test)
      + ", rel% vs firm identity "
      + " / ".join(f"{v:+.2f}" for v in FI.rel_impr_pct_firm)
      + ", rel% vs maximal pool "
      + " / ".join(f"{v:+.2f}" for v in CI.rel_impr_pct_maximal)
      + f" (maximal_raw any {bool(CI.maximal_raw.any())}, maximal_holm any "
        f"{bool(CI.maximal_holm.any())})")
print("itemcode_control.csv is on this table's basis: n_test "
      + " / ".join(str(int(v)) for v in IC.n_test))
print("separate Item-2.02-in-reference control (itemcode_control.csv, reported "
      "in S4): rel% "
      + " / ".join(f"{v:+.2f}" for v in IC.rel_202)
      + "; Holm p " + " / ".join(f"{v:.2e}" for v in IC.p_202_holm))
