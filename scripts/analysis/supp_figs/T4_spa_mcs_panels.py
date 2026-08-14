"""T4 -- Joint inference: SPA and the model confidence set over 18 loss-panels.

Emits supplement Table T4 (Technical Supplement, S11) and gates every headline
count against the committed evidence before printing a single cell.

Sources read (all committed)
----------------------------
results/tables/row13_spa_mcs_panels.csv
    disclosure, horizon, loss, n_obs, n_days, n_models,
    spa_p_consistent, spa_best_challenger, spa_best_tstat,
    spa_textfusion_p_consistent, spa_textfusion_best,
    spa_textfusion_best_tstat, mcs90_price, mcs90_text, mcs90_fusion,
    n_price, n_text, n_fusion, arch_note
results/tables/row13_spa_mcs.csv
    disclosure, horizon, loss, model, block, mcs_p, in_mcs90 -- the per-model
    membership lists that the panel counts aggregate
results/tables/variance_unit_standalone180.csv
    better_qlike_var_holm -- used only for the scope-labelled cross-check that
    the challengers which do beat HAR under the variance-unit convention are
    price models, never text or fusion

Main-text sentence substantiated
--------------------------------
06_results.tex: "Across 18 loss-panels no pure-text model enters the 90%
    model-confidence set, the SPA p for 'no text beats HAR' is 1.000, and all
    9 full-set SPA rejections favour price models."

Specification notes
-------------------
* Every SPA column is pinned to spa_p_consistent. spa_p_lower and
  spa_p_upper exist in the source and are deliberately never printed.
* MCS membership means "cannot be statistically separated from the best
  model", not "beats HAR"; the fusion admissions are printed in full rather
  than summarised.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp_style import TAB, gate

p = pd.read_csv(os.path.join(TAB, "row13_spa_mcs_panels.csv"))
m = pd.read_csv(os.path.join(TAB, "row13_spa_mcs.csv"))
vu = pd.read_csv(os.path.join(TAB, "variance_unit_standalone180.csv"))

# the panel counts must be exactly the per-model membership flags aggregated
agg = (m[m.in_mcs90].groupby(["disclosure", "horizon", "loss", "block"])
       .size().unstack(fill_value=0).reset_index())
for blk in ("price", "text", "fusion"):
    if blk not in agg:
        agg[blk] = 0
chk = p.merge(agg, on=["disclosure", "horizon", "loss"], how="left").fillna(0)
if not ((chk.mcs90_price == chk.price).all()
        and (chk.mcs90_text == chk.text).all()
        and (chk.mcs90_fusion == chk.fusion).all()):
    sys.exit("ALIGN FAIL - panel MCS counts differ from the per-model flags")

vu_better = vu.loc[vu.better_qlike_var_holm, "challenger"].unique().tolist()

# n_models counts the HAR-RV benchmark; the SPA alternative set is n_models-1
# (row13_spa_mcs.py: n_models = len(present), alt_all excludes bench_idx).
# Check the arithmetic that licenses printing the alternative-set size: one
# benchmark row per panel, the benchmark is A2_har_rv, it is a price arm, and
# the per-panel model row count equals n_models.
bench_per_panel = (m[m.is_benchmark].groupby(["disclosure", "horizon", "loss"])
                   .size())
rows_per_panel = m.groupby(["disclosure", "horizon", "loss"]).size()
n_models_by_panel = p.set_index(["disclosure", "horizon", "loss"]).n_models
bench_names = sorted(m.loc[m.is_benchmark, "model"].unique().tolist())
bench_blocks = sorted(m.loc[m.is_benchmark, "block"].unique().tolist())
bench_in_mcs = m[(m.model == "A2_har_rv") & m.in_mcs90]

# the combined channel is the union of the two source channels, so the 18
# panels are not 18 independent samples; check it rather than assert it
wide = p.pivot_table(index=["horizon", "loss"], columns="disclosure",
                     values="n_obs")
combined_is_union = bool(
    (wide["combined"] == wide["long_form"] + wide["event_driven"]).all())
extra_ed = sorted(set(m[m.disclosure == "event_driven"].model)
                  - set(m[m.disclosure == "long_form"].model))

gate(
    {"n_panels": 18, "tf_p_all_one": True, "text_in_mcs90_total": 0,
     "full_set_rejections": 9, "rejection_winners_all_price": True,
     "fusion_panels": 12, "fusion_slots": 24, "price_slots": 73,
     "arch_note_clean": True, "tf_best_t_all_negative": True,
     "pool_sizes": {(8, 12, 6), (8, 13, 6)},
     "vu_better_are_price_only": True,
     "combined_is_union": True, "extra_event_driven_model":
         ["C6_llmtext_llama70"],
     "one_benchmark_per_panel": True, "benchmark_names": ["A2_har_rv"],
     "benchmark_blocks": ["price"], "model_rows_equal_n_models": True,
     "alt_set_sizes": {25, 26}, "benchmark_outside_mcs90_panels": 8},
    {"n_panels": len(p),
     "tf_p_all_one": bool((p.spa_textfusion_p_consistent == 1.0).all()),
     "text_in_mcs90_total": int(p.mcs90_text.sum()),
     "full_set_rejections": int((p.spa_p_consistent < 0.05).sum()),
     "rejection_winners_all_price": bool(
         p.loc[p.spa_p_consistent < 0.05, "spa_best_challenger"]
         .str.startswith("A").all()),
     "fusion_panels": int((p.mcs90_fusion > 0).sum()),
     "fusion_slots": int(p.mcs90_fusion.sum()),
     "price_slots": int(p.mcs90_price.sum()),
     "arch_note_clean": bool((p.arch_note == "ok").all()),
     "tf_best_t_all_negative": bool((p.spa_textfusion_best_tstat < 0).all()),
     "pool_sizes": set(map(tuple, p[["n_price", "n_text", "n_fusion"]]
                           .drop_duplicates().values.tolist())),
     "vu_better_are_price_only": all(c.startswith("A") for c in vu_better),
     "combined_is_union": combined_is_union,
     "extra_event_driven_model": extra_ed,
     "one_benchmark_per_panel": bool((bench_per_panel == 1).all()
                                     and len(bench_per_panel) == len(p)),
     "benchmark_names": bench_names, "benchmark_blocks": bench_blocks,
     "model_rows_equal_n_models": bool(
         (rows_per_panel == n_models_by_panel.reindex(rows_per_panel.index))
         .all()),
     "alt_set_sizes": set((p.n_models - 1).tolist()),
     "benchmark_outside_mcs90_panels": len(p) - len(bench_in_mcs)},
)

LOSS = {"qlike": "QLIKE (vol)", "se": "SE (vol^2)"}
DISC = {"long_form": "Long-form", "event_driven": "Event-driven",
        "combined": "Combined"}
order = {"long_form": 0, "event_driven": 1, "combined": 2}
p = p.sort_values(by=["disclosure", "horizon", "loss"],
                  key=lambda s: s.map(order) if s.name == "disclosure" else s)

out = []
out.append("| Disclosure | h | Loss | n_obs | n_days | K models "
           "(alternatives) | SPA p, all "
           "alternatives | Best challenger (t) | SPA p, text/fusion only | "
           "MCS90 price / text / fusion | Pool price / text / fusion |")
out.append("|---|---:|---|---:|---:|---:|---:|---|---:|---|---|")
for _, r in p.iterrows():
    dag = " †" if r.spa_p_consistent < 0.05 else ""
    out.append(
        f"| {DISC[r.disclosure]} | {int(r.horizon)} | {LOSS[r.loss]} | "
        f"{int(r.n_obs):,} | {int(r.n_days)} | "
        f"{int(r.n_models)} ({int(r.n_models) - 1}) | "
        f"{r.spa_p_consistent:.4f}{dag} | {r.spa_best_challenger} "
        f"({r.spa_best_tstat:+.2f}) | "
        f"{r.spa_textfusion_p_consistent:.3f} | "
        f"{int(r.mcs90_price)} / {int(r.mcs90_text)} / "
        f"{int(r.mcs90_fusion)} | {int(r.n_price)} / {int(r.n_text)} / "
        f"{int(r.n_fusion)} |")
out.append("")
out.append(f"† full-set SPA p < .05: {int((p.spa_p_consistent < 0.05).sum())} "
           f"of {len(p)} panels. Column totals across the 18 panels: MCS90 "
           f"price {int(p.mcs90_price.sum())} slots, text "
           f"{int(p.mcs90_text.sum())} slots, fusion "
           f"{int(p.mcs90_fusion.sum())} slots in "
           f"{int((p.mcs90_fusion > 0).sum())} panels.")

# the fusion admissions, printed rather than summarised
fus = m[(m.in_mcs90) & (m.block == "fusion")]
names = fus.groupby("model").size().sort_values(ascending=False)
out.append("")
out.append("Fusion arms admitted to the 90% model confidence set, by model "
           "and number of panels: " + "; ".join(
               f"{k} {v}" for k, v in names.items()) + ".")
pri = (m[(m.in_mcs90) & (m.block == "price")].groupby("model").size()
       .sort_values(ascending=False))
out.append("Price arms admitted, by model and number of panels: " +
           "; ".join(f"{k} {v}" for k, v in pri.items()) + ".")
out.append(f"Text arms admitted, any block-B or block-C model, any panel: "
           f"{int(m[(m.in_mcs90) & (m.block == 'text')].shape[0])}.")
a2 = m[m.model == "A2_har_rv"]
a2_out = a2[~a2.in_mcs90][["disclosure", "horizon", "loss"]]
out.append(f"The A2 HAR-RV benchmark is itself outside the 90% set in "
           f"{len(a2_out)} of {len(a2)} panels: " + "; ".join(
               f"{r.disclosure} h{int(r.horizon)} {r.loss}"
               for _, r in a2_out.iterrows()) + ".")
out.append("Best text/fusion alternative in every panel, with its t: " +
           "; ".join(sorted({f"{a} ({b:+.2f} to {c:+.2f})" for a, b, c in
                             [(g, gg.spa_textfusion_best_tstat.min(),
                               gg.spa_textfusion_best_tstat.max())
                              for g, gg in
                              p.groupby("spa_textfusion_best")]})) + ".")
print("\n".join(out))

# ------------------------------------------------- numbers quoted in prose
win = (p.loc[p.spa_p_consistent < 0.05, "spa_best_challenger"]
       .value_counts())
print("\n[footnote facts]")
print("  winners in the rejecting panels: " +
      "; ".join(f"{k} {v}" for k, v in win.items()))
print(f"  text/fusion best t range: "
      f"{p.spa_textfusion_best_tstat.min():+.2f} to "
      f"{p.spa_textfusion_best_tstat.max():+.2f}")
print(f"  combined n_obs equals long_form + event_driven in all "
      f"{len(wide)} horizon x loss pairs: {combined_is_union}")
print(f"  the one model event-driven carries and long-form does not: "
      f"{extra_ed}")
vc = vu.loc[vu.better_qlike_var_holm, "challenger"].value_counts()
print("  variance-unit QLIKE, challengers beating A2 under Holm (scope: "
      "variance unit, all 180 standalone comparisons): " +
      "; ".join(f"{k} {v}" for k, v in vc.items()) +
      f" -- {int(vu.better_qlike_var_holm.sum())} of {len(vu)} in total")
print(f"  arch cross-check note distinct values: "
      f"{sorted(p.arch_note.unique())}")
print("\n[gate passed] T4 markdown emitted from committed evidence only")
