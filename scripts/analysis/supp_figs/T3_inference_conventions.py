"""T3 -- One forecast set, three inference conventions (Supplement, S11).

Emits the two markdown blocks of supplement Table T3 and gates every headline
count against the committed evidence, so a drifted table aborts the build
instead of being typeset silently.

Sources read (all committed)
----------------------------
results/tables/dm_pairwise_clustered.csv
    disclosure, horizon, challenger, n_obs, n_days,
    dm_obs, p_obs, p_holm_obs, dm_clust, p_clust, p_holm_clust,
    better_obs, better_clust, better_clust_raw, sig_worse_obs, sig_worse_clust
results/tables/dm_pairwise_clustered.md
    "median |DM| shrink factor ... 0.31" (day-clustered over observation-level)
results/tables/twoway_cluster.csv
    panel (a_m1_grid / b_firm_ref / c_pairwise_vsA2), disc, model, h,
    n_obs, n_firms, n_days, dm_day, verdict_day, dm_2way, verdict_2way, flip,
    se_infl_2way_vs_day, share_Vfirm, share_Vday, share_Vcell, guard_hit
results/tables/m1_clustered.md
    grid-level median |DM| shrink factor 0.49 (quoted in a footnote so the
    standalone set's 0.31 is not read as contradicting the protocol's
    "clustering roughly halves |DM|")

Main-text sentences substantiated
---------------------------------
06_results.tex: "0 of 180 standalone squared-error comparisons favour the
    challenger (raw p<.05 and Holm); 155 are significantly worse (152 under
    two-way clustering)".
06_results.tex Table 1 (reference ladder), row "two-way clustering ... 27,
    24 on seed 2026".
05_protocol.tex: day-clustered DM is the primary; "794--996 days per panel";
    "recompute under two-way firm x day clustering".

Adversarial repairs folded in
-----------------------------
R1 (binding): the committed quantity is median |DM_day-clustered| /
    |DM_obs-level| = 0.31, i.e. day-clustered OVER observation-level. The
    direction is stated that way here and nowhere inverted.
Conservative choice where the specification and the evidence disagree on a
    label rather than a number: the specification asked for every block-(b)
    count to carry a "seed 2026" tag. Panels (a) and (b) are indeed the
    seed-2026 basis (twoway_cluster.md headers; firm_identity_control.csv is
    the seed2026 firm-identity file, 14 raw / 8 Holm, against the ensemble
    file's 15 raw / 8 Holm), but panel (c) is explicitly the SEED-ENSEMBLE
    basis of dm_pairwise_clustered. Tagging it "seed 2026" would contradict
    both the source header and the frozen main text, whose 152 carries no
    seed note. Each row therefore carries its own true basis tag.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp_style import TAB, gate

dm = pd.read_csv(os.path.join(TAB, "dm_pairwise_clustered.csv"))
tw = pd.read_csv(os.path.join(TAB, "twoway_cluster.csv"))

a = tw[tw.panel == "a_m1_grid"]
b = tw[tw.panel == "b_firm_ref"]
c = tw[tw.panel == "c_pairwise_vsA2"]

# ---------------------------------------------------------------- alignment
# The two files must describe the SAME 180 forecasts; if they ever drift
# apart the table would silently mix two forecast sets.
mg = dm.merge(c, left_on=["disclosure", "horizon", "challenger"],
              right_on=["disc", "h", "model"], how="inner")
if len(mg) != 180 or float((mg.dm_clust - mg.dm_day).abs().max()) != 0.0:
    sys.exit("ALIGN FAIL - dm_pairwise_clustered and twoway_cluster disagree")

# "better" is a committed flag for the Holm columns; the raw-p counterpart on
# the observation-level side is derived here from sign(dm_obs) and p_obs.
better_obs_raw = int(((dm.dm_obs < 0) & (dm.p_obs < 0.05)).sum())
better_2way_raw = int(((c.dm_2way < 0) & (c.p_2way < 0.05)).sum())
better_2way_holm = int(((c.dm_2way < 0) & (c.holm_2way < 0.05)).sum())

gate(
    # values the frozen main text and the source .md files state
    {"n_comparisons": 180, "better_obs_raw": 0, "better_obs_holm": 0,
     "better_clust_raw": 0, "better_clust_holm": 0,
     "better_2way_raw": 0, "better_2way_holm": 0,
     "worse_obs": 174, "worse_clust": 155, "worse_2way": 152,
     "n_days_lo": 794, "n_days_hi": 996,
     "n_obs_lo": 7902, "n_obs_hi": 33060,
     "n_firms_lo": 567, "n_firms_hi": 569,
     "shrink_day_over_obs": 0.31, "se_infl_standalone": 1.181,
     "grid_genuine_day": 29, "grid_genuine_2way": 24, "grid_flips": 5,
     "firm_adds_day": 8, "firm_adds_2way": 5,
     "firm_hurts_day": 29, "firm_hurts_2way": 19, "firm_flips": 13,
     "standalone_flips": 3, "guard_hits": 0, "n_rows_total": 318},
    {"n_comparisons": len(dm), "better_obs_raw": better_obs_raw,
     "better_obs_holm": int(dm.better_obs.sum()),
     "better_clust_raw": int(dm.better_clust_raw.sum()),
     "better_clust_holm": int(dm.better_clust.sum()),
     "better_2way_raw": better_2way_raw, "better_2way_holm": better_2way_holm,
     "worse_obs": int(dm.sig_worse_obs.sum()),
     "worse_clust": int(dm.sig_worse_clust.sum()),
     "worse_2way": int((c.verdict_2way == "sig worse").sum()),
     "n_days_lo": int(dm.n_days.min()), "n_days_hi": int(dm.n_days.max()),
     "n_obs_lo": int(dm.n_obs.min()), "n_obs_hi": int(dm.n_obs.max()),
     "n_firms_lo": int(c.n_firms.min()), "n_firms_hi": int(c.n_firms.max()),
     "shrink_day_over_obs": round(
         float((dm.dm_clust.abs() / dm.dm_obs.abs()).median()), 2),
     "se_infl_standalone": round(float(c.se_infl_2way_vs_day.median()), 3),
     "grid_genuine_day": int((a.verdict_day == "genuine").sum()),
     "grid_genuine_2way": int((a.verdict_2way == "genuine").sum()),
     "grid_flips": int(a.flip.sum()),
     "firm_adds_day": int((b.verdict_day == "text adds").sum()),
     "firm_adds_2way": int((b.verdict_2way == "text adds").sum()),
     "firm_hurts_day": int((b.verdict_day == "text HURTS").sum()),
     "firm_hurts_2way": int((b.verdict_2way == "text HURTS").sum()),
     "firm_flips": int(b.flip.sum()),
     "standalone_flips": int(c.flip.sum()),
     "guard_hits": int(tw.guard_hit.sum()), "n_rows_total": len(tw)},
)

# every row satisfies share_firm + share_day - share_intersection = 1 exactly;
# the column-wise medians reported below therefore do NOT satisfy it.
ident = (tw.share_Vfirm + tw.share_Vday - tw.share_Vcell)
if not (abs(ident - 1.0) < 1e-9).all():
    sys.exit("SHARE FAIL - two-way variance shares do not decompose to 1")

med_abs_obs = float(dm.dm_obs.abs().median())
med_abs_day = float(dm.dm_clust.abs().median())
med_abs_2w = float(c.dm_2way.abs().median())
r_day_obs = float((dm.dm_clust.abs() / dm.dm_obs.abs()).median())
r_2w_obs = float((mg.dm_2way.abs() / mg.dm_obs.abs()).median())
# The ratio column holds medians of the 180 per-comparison ratios. The naive
# ratio of the two column medians is a DIFFERENT number, and for the two-way
# row it happens to equal the value printed one row above (0.31), so both are
# computed here and the collision is stated in the footnote rather than left
# for a reader to trip over.
q_day_obs = med_abs_day / med_abs_obs
q_2w_obs = med_abs_2w / med_abs_obs

out = []
out.append("**(a) The 180 standalone squared-error comparisons, "
           "re-adjudicated under three error structures.**\n")
out.append("| Error structure for the loss differential | Challenger better,"
           " raw p<.05 | Challenger better, Holm p<.05 | Challenger worse,"
           " Holm p<.05 | Median \\|DM\\| | Median per-comparison \\|DM\\|"
           " ratio to observation-level | Median SE relative to day-clustered"
           " |")
out.append("|---|---:|---:|---:|---:|---:|---:|")
out.append(f"| Observation-level HAC, lag h-1 observations | "
           f"{better_obs_raw} of 180 | {int(dm.better_obs.sum())} of 180 | "
           f"{int(dm.sig_worse_obs.sum())} of 180 | {med_abs_obs:.2f} | "
           f"1.00 | -- |")
out.append(f"| Day-clustered, lag h-1 days (primary) | "
           f"{int(dm.better_clust_raw.sum())} of 180 | "
           f"{int(dm.better_clust.sum())} of 180 | "
           f"{int(dm.sig_worse_clust.sum())} of 180 | {med_abs_day:.2f} | "
           f"{r_day_obs:.2f} | 1.000 |")
out.append(f"| Two-way firm x day (Cameron-Gelbach-Miller) | "
           f"{better_2way_raw} of 180 | {better_2way_holm} of 180 | "
           f"{int((c.verdict_2way == 'sig worse').sum())} of 180 | "
           f"{med_abs_2w:.2f} | {r_2w_obs:.2f} | "
           f"{float(c.se_infl_2way_vs_day.median()):.3f} |")
out.append("")
out.append(f"Panel geometry, identical under all three rows: 9 disclosure x "
           f"horizon panels x 20 challengers; {int(dm.n_days.min())}-"
           f"{int(dm.n_days.max())} trading days, {int(dm.n_obs.min()):,}-"
           f"{int(dm.n_obs.max()):,} filing-horizon observations, "
           f"{int(c.n_firms.min())}-{int(c.n_firms.max())} firms per panel.\n")

out.append("**(b) The same estimator applied to the verdict grids, on both "
           "forecast bases.**\n")
out.append("| Grid (forecast basis) | Verdict counted | Day-clustered | "
           "Two-way firm x day | Cells lost | Median SE inflation | "
           "Median variance share firm / day / intersection | "
           "Non-PSD guard hits |")
out.append("|---|---|---:|---:|---:|---:|---|---:|")


def shares(g):
    return (f"{g.share_Vfirm.median():.2f} / {g.share_Vday.median():.2f} / "
            f"{g.share_Vcell.median():.2f}")


def row(label, verdict, n_day, n_2way, denom, se, sh, guard):
    out.append(f"| {label} | {verdict} | {n_day} of {denom} | "
               f"{n_2way} of {denom} | {n_day - n_2way} | {se} | {sh} | "
               f"{guard} |")


# seed-ensemble basis: residual_symmetric_holm carries both panels; this is
# the basis the frozen reference-ladder table reports as 38 -> 27.
rs = pd.read_csv(os.path.join(TAB, "residual_symmetric_holm.csv"))
ens_a = rs[(rs.basis == "ens") & (rs.panel == "A_singleHAR")]
ens_b = rs[(rs.basis == "ens") & (rs.panel == "B_firmref")]
s26_a = rs[(rs.basis == "s26") & (rs.panel == "A_singleHAR")]
s26_b = rs[(rs.basis == "s26") & (rs.panel == "B_firmref")]

# cross-file gate: the seed-2026 rows of residual_symmetric_holm must be the
# same numbers twoway_cluster commits, or the two files have diverged.
for panel_tw, panel_rs, key in ((a, s26_a, "a"), (b, s26_b, "b")):
    j = panel_tw.merge(panel_rs, on=["disc", "model", "h"],
                       suffixes=("_tw", "_rs"))
    if len(j) != 69 or float((j.dm_2way_tw - j.dm_2way_rs).abs().max()) != 0.0:
        sys.exit(f"ALIGN FAIL - twoway_cluster and residual_symmetric_holm "
                 f"disagree on panel {key}")

gate({"ens_grid_genuine_day": 38, "ens_grid_genuine_2way": 27,
      "ens_firm_adds_day": 8, "ens_firm_adds_2way": 5,
      "ens_firm_hurts_day": 35, "ens_firm_hurts_2way": 20,
      "rs_guard_hits": 0, "rs_rows": 276},
     {"ens_grid_genuine_day": int(ens_a.genuine_day.sum()),
      "ens_grid_genuine_2way": int(ens_a["genuine_2way"].sum()),
      "ens_firm_adds_day": int(ens_b.genuine_day.sum()),
      "ens_firm_adds_2way": int(ens_b["genuine_2way"].sum()),
      "ens_firm_hurts_day": int(ens_b.hurts_day.sum()),
      "ens_firm_hurts_2way": int(ens_b["hurts_2way"].sum()),
      "rs_guard_hits": int(rs.guard_hit.sum()), "rs_rows": len(rs)})

NA = "not carried on this basis"
row("69-cell combination grid vs recalibrated HAR (seed ensemble)",
    "genuine", int(ens_a.genuine_day.sum()), int(ens_a["genuine_2way"].sum()),
    len(ens_a), f"{float(ens_a.se_infl_2way_vs_day.median()):.3f}", NA,
    int(ens_a.guard_hit.sum()))
row("69-cell combination grid vs recalibrated HAR (seed ensemble)",
    "text hurts", int(ens_a.hurts_day.sum()), int(ens_a["hurts_2way"].sum()),
    len(ens_a), f"{float(ens_a.se_infl_2way_vs_day.median()):.3f}", NA,
    int(ens_a.guard_hit.sum()))
row("69-cell combination grid vs recalibrated HAR (seed 2026)",
    "genuine", int((a.verdict_day == "genuine").sum()),
    int((a.verdict_2way == "genuine").sum()), len(a),
    f"{float(a.se_infl_2way_vs_day.median()):.3f}", shares(a),
    int(a.guard_hit.sum()))
row("69-cell combination grid vs recalibrated HAR (seed 2026)",
    "text hurts", int(s26_a.hurts_day.sum()), int(s26_a["hurts_2way"].sum()),
    len(s26_a), f"{float(a.se_infl_2way_vs_day.median()):.3f}", shares(a),
    int(a.guard_hit.sum()))
row("Firm-identity reference (seed ensemble)", "text adds",
    int(ens_b.genuine_day.sum()), int(ens_b["genuine_2way"].sum()),
    len(ens_b), f"{float(ens_b.se_infl_2way_vs_day.median()):.3f}", NA,
    int(ens_b.guard_hit.sum()))
row("Firm-identity reference (seed ensemble)", "text hurts",
    int(ens_b.hurts_day.sum()), int(ens_b["hurts_2way"].sum()), len(ens_b),
    f"{float(ens_b.se_infl_2way_vs_day.median()):.3f}", NA,
    int(ens_b.guard_hit.sum()))
row("Firm-identity reference (seed 2026)", "text adds",
    int((b.verdict_day == "text adds").sum()),
    int((b.verdict_2way == "text adds").sum()), len(b),
    f"{float(b.se_infl_2way_vs_day.median()):.3f}", shares(b),
    int(b.guard_hit.sum()))
row("Firm-identity reference (seed 2026)", "text hurts",
    int((b.verdict_day == "text HURTS").sum()),
    int((b.verdict_2way == "text HURTS").sum()), len(b),
    f"{float(b.se_infl_2way_vs_day.median()):.3f}", shares(b),
    int(b.guard_hit.sum()))
row("180 standalone comparisons vs A2-HAR (seed ensemble)",
    "challenger better", int(dm.better_clust.sum()), better_2way_holm,
    len(c), f"{float(c.se_infl_2way_vs_day.median()):.3f}", shares(c),
    int(c.guard_hit.sum()))
row("180 standalone comparisons vs A2-HAR (seed ensemble)",
    "significantly worse", int((c.verdict_day == "sig worse").sum()),
    int((c.verdict_2way == "sig worse").sum()), len(c),
    f"{float(c.se_infl_2way_vs_day.median()):.3f}", shares(c),
    int(c.guard_hit.sum()))
out.append("")
flip_dirs = tw[tw.flip].groupby(["verdict_day", "verdict_2way"]).size()
# the "none reverses a sign" claim is checked on BOTH files, not asserted:
# in twoway_cluster every flip must land on "ns", and in
# residual_symmetric_holm no cell may gain a significant verdict.
if not (tw[tw.flip].verdict_2way == "ns").all():
    sys.exit("CLAIM FAIL - a twoway_cluster flip does not land on ns")
gained = (int((rs["genuine_2way"] & ~rs.genuine_day).sum())
          + int((rs["hurts_2way"] & ~rs.hurts_day).sum()))
if gained:
    sys.exit(f"CLAIM FAIL - {gained} residual_symmetric_holm cells gain "
             f"significance under two-way clustering")
out.append(f"Every verdict change in the table moves a cell out of "
           f"significance; none reverses a sign. In twoway_cluster.csv all "
           f"{int(tw.flip.sum())} changes across its {len(tw)} rows are: " +
           "; ".join(f"{k[0]} -> not significant, {v}"
                     for k, v in flip_dirs.items()) +
           f". The non-PSD guard, which floors the two-way variance when the "
           f"Cameron-Gelbach-Miller combination turns negative, fires in "
           f"{int(tw.guard_hit.sum())} of {len(tw)} rows there and "
           f"{int(rs.guard_hit.sum())} of {len(rs)} rows in "
           f"residual_symmetric_holm.csv.")
print("\n".join(out))

# ------------------------------------------------- numbers quoted in prose
# The placebo gate is part of the "genuine" definition; check whether it does
# any work on either side of the grid restatement, so the prose can say so.
cand_2w = a[(a.dm_2way < 0) & (a.holm_2way < 0.05)]
m1 = pd.read_csv(os.path.join(TAB, "m1_clustered.csv"))
grid_shrink = float((m1.dm_q_clust.abs() / m1.dm_q.abs()).median())
# the day-clustered placebo column lives in m1_clustered, not twoway_cluster
cand_day = m1[(m1.dm_q_clust < 0) & (m1.dmq_holm_clust < 0.05)]
if int(m1.genuine_clust.sum()) != len(cand_day):
    sys.exit("GATE FAIL - m1_clustered genuine_clust no longer equals the "
             "day-clustered DM<0 and Holm<.05 candidate set")
print("\n[footnote facts]")
print(f"  grid median |DM_day|/|DM_obs| on the 69 cells: {grid_shrink:.2f} "
      f"(standalone set: {r_day_obs:.2f})")
print(f"  ratio-column footnote: medians of per-comparison ratios are "
      f"{r_day_obs:.2f} and {r_2w_obs:.2f}; the ratios of the two column "
      f"medians would be {q_day_obs:.2f} and {q_2w_obs:.2f}")
print(f"  placebo |DM|<2 among day-clustered candidates: "
      f"{int((cand_day.placebo_dm_clust.abs() < 2).sum())} of {len(cand_day)}")
print(f"  placebo |DM|<2 among two-way candidates: "
      f"{int((cand_2w.placebo_dm_2way.abs() < 2).sum())} of {len(cand_2w)}")
print(f"  median |DM| by row: obs {med_abs_obs:.2f}, day {med_abs_day:.2f}, "
      f"two-way {med_abs_2w:.2f}")
print(f"  challengers with a favourable point estimate (DM<0), any "
      f"estimator: obs {int((dm.dm_obs < 0).sum())}, day "
      f"{int((dm.dm_clust < 0).sum())}, two-way {int((c.dm_2way < 0).sum())}")
print("\n[gate passed] T3 markdown emitted from committed evidence only")
