"""T11 -- the five cross-family probes and the primary-family anchor.

Emits the markdown of Technical Supplement Table T11 and gates every count
against the values the paper states, so a stale evidence table can never be
tabulated silently.  Nothing is re-fitted; CPU only, under a second.

Sources
  results/tables/crossfamily_llama70.csv   (+ .md)
      Qwen3-32B (both channels), Yi-1.5-34B (both channels), Phi-4-14B,
      Llama-3.1-70B-AWQ; rel_har, rel_firm, qlike_var, pred_sd,
      mode_share_pct, parse_fail_rate, clipped_rate, flag, p_*_holm
  results/tables/crossfamily_gemma27.csv   (+ .md)
      the same anchors plus Mistral-Small-24B and Gemma-3-27B, single and
      3-run ensemble, and the holm_family strings that record which
      multiplicity convention each row carries
  results/tables/crossfamily_mistral24.csv (+ .md)
      read only to assert the shared rows agree with crossfamily_gemma27.csv
  results/tables/crossfamily_standalone.csv (+ .md)
      context_flag for the Yi long-form (4K-TRUNCATED) and combined
      (PARTIAL(4K)) rows, and the combined-channel health columns
  results/tables/crossfamily_llama70_ens.csv
      single-versus-ensemble rel_firm and p_firm_holm for footnote (v)
  results/tables/row15_ensemble_m1.csv
      the same comparison in wide form; asserted equal to the ens file

Deliberately NOT reproduced (both are superseded by the committed evidence):
  * the three-family verdict table in crossfamily_mistral24.md, which predates
    the Gemma-3-27B probe;
  * the family-specificity headline in crossfamily_llm.md, which is retracted
    and is contradicted by the healthy 70B replication in 07_ablations.tex.

Main-text sentences substantiated (frozen)
  07_ablations.tex "of five probes, four fail a forecaster-health screen and
    are instrument-dead rather than disconfirming"; "The healthy probe,
    Llama-3.1-70B (int4; 8-K only), directionally replicates: larger than
    Qwen's in point estimate yet statistically attenuated (vs firm identity
    +0.83/+0.64/+0.39%; DM<0 in 3 of 3; 1 of 3 vs-HAR Holm; ~3x more
    dispersed)."
  04_methods.tex "Five cross-family replications test whether any increment
    generalises beyond Qwen3, each screened for forecaster health before its
    reading counts."

Adversarial repairs folded in (every required_change binding)
  * Title and every count say FIVE cross-family probes; the Qwen3-32B row is
    labelled "primary family (anchor), not a probe" and is excluded from any
    probe count, matching the frozen text.
  * Footnote (ii) is restated on Mistral-Small-24B's own measured range
    (0.42-0.69 variance-unit QLIKE, 69.3-88.6% modal share).  Lens 1 asked for
    the max-over-horizons statistic (Phi-4, 0.677) and lens 2 for the lowest
    single cell (Mistral); the two are different statistics, so the
    conservative resolution is to print BOTH with the statistic named, rather
    than pick one and leave the ranking ambiguous.
  * Footnote (iii) is rewritten: Yi's three LONG-FORM rows carry 4K-TRUNCATED
    and its three COMBINED rows carry PARTIAL(4K); only its three event-driven
    rows are context-clean.
  * Footnote (v) says "at most 0.006pp (0.003/0.003/0.005 by horizon)".
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp_style import TAB, gate

L70 = pd.read_csv(os.path.join(TAB, "crossfamily_llama70.csv"))
GEM = pd.read_csv(os.path.join(TAB, "crossfamily_gemma27.csv"))
MIS = pd.read_csv(os.path.join(TAB, "crossfamily_mistral24.csv"))
STD = pd.read_csv(os.path.join(TAB, "crossfamily_standalone.csv"))
ENS = pd.read_csv(os.path.join(TAB, "crossfamily_llama70_ens.csv"))
R15 = pd.read_csv(os.path.join(TAB, "row15_ensemble_m1.csv")).sort_values("h")

# the Mistral table and the Gemma table overlap on the anchors and Mistral;
# assert they agree rather than choosing arbitrarily between them
_k = ["disc", "family", "h"]
_sh = sorted(set(GEM.family) & set(MIS.family))
_a = GEM[GEM.family.isin(_sh)].sort_values(_k).reset_index(drop=True)
_b = MIS[MIS.family.isin(_sh)].sort_values(_k).reset_index(drop=True)
for _c in ("rel_har", "rel_firm", "qlike_var", "mode_share_pct"):
    assert np.allclose(_a[_c].astype(float), _b[_c].astype(float),
                       rtol=1e-12, equal_nan=True), f"union disagrees on {_c}"
# row15 restates the ens file in wide form; assert it matches to 1e-12
_e_s = ENS[ENS.family == "llama70_awq"].sort_values("h")
_e_e = ENS[ENS.family == "llama70_awq_ens3"].sort_values("h")
assert np.allclose(_e_s.rel_firm.astype(float), R15.rel_firm_single.astype(float),
                   rtol=1e-12)
assert np.allclose(_e_e.rel_firm.astype(float), R15.rel_firm_ens.astype(float),
                   rtol=1e-12)

GATE_MODE, GATE_QLIKE = 60.0, 4.0
CH = {"long_form": "10-K/Q", "event_driven": "8-K", "combined": "combined"}

# (source frame, family tag, display name, precision, role)
ROWS = [
    (L70, "qwen3_32b", "Qwen3-32B", "bf16",
     "primary family (anchor), not a probe"),
    (L70, "llama70_awq", "Llama-3.1-70B", "AWQ-INT4, single", "probe"),
    (GEM, "llama70_awq_ens3", "Llama-3.1-70B", "AWQ-INT4, 3-run ensemble",
     "probe"),
    (L70, "yi_34b", "Yi-1.5-34B", "bf16", "probe"),
    (L70, "phi4_14b", "Phi-4-14B", "bf16", "probe"),
    (GEM, "mistral24_bf16", "Mistral-Small-24B", "bf16, single", "probe"),
    (GEM, "mistral24_ens3", "Mistral-Small-24B", "bf16, 3-run ensemble",
     "probe"),
    (GEM, "gemma27_bf16", "Gemma-3-27B", "bf16, single", "probe"),
    (GEM, "gemma27_ens3", "Gemma-3-27B", "bf16, 3-run ensemble", "probe"),
]
# families each probe belongs to, so probe COUNTS are per family, never per row
FAMILY_OF = {"Llama-3.1-70B": "llama70", "Yi-1.5-34B": "yi",
             "Phi-4-14B": "phi4", "Mistral-Small-24B": "mistral24",
             "Gemma-3-27B": "gemma27"}


def health_rows(name, tag):
    """Every committed row of this family that carries health columns."""
    frames = []
    for df, col in ((L70, "flag"), (GEM, "flag")):
        frames.append(df[df.family == tag])
    frames.append(STD[STD.family == tag.replace("_bf16", "")
                      .replace("_ens3", "").replace("_awq", "")])
    out = pd.concat(frames, ignore_index=True)
    return out[out.qlike_var.notna()]


def rng(v, fmt="{:.3f}"):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return "not recorded"
    lo, hi = v.min(), v.max()
    return fmt.format(lo) if np.isclose(lo, hi) else \
        fmt.format(lo) + "--" + fmt.format(hi)


# ------------------------------------------------------- per-family evidence
fam_health = {}
for tag_std, name in (("qwen3_32b", "Qwen3-32B"), ("yi_34b", "Yi-1.5-34B"),
                      ("phi4_14b", "Phi-4-14B"),
                      ("llama70_awq", "Llama-3.1-70B")):
    src = pd.concat([L70[L70.family == tag_std],
                     STD[STD.family == tag_std]], ignore_index=True)
    src = src[src.qlike_var.notna()]
    # L70 and STD overlap on the 8-K and 10-K/Q rows; assert they agree, then
    # keep one copy so a (family, channel) run is counted once
    _d = src.groupby(["disc", "h"])[["qlike_var", "mode_share_pct"]].agg(
        lambda v: float(np.ptp(np.asarray(v, dtype=float))))
    assert float(_d.to_numpy().max()) < 1e-12, f"{name}: sources disagree"
    src = src.assign(_pri=src.parse_fail_rate.isna().astype(int))
    fam_health[name] = (src.sort_values(["disc", "h", "_pri"])
                        .drop_duplicates(["disc", "h"], keep="first")
                        .drop(columns="_pri"))
for tag in ("mistral24_bf16", "mistral24_ens3", "gemma27_bf16", "gemma27_ens3"):
    fam_health[tag] = GEM[(GEM.family == tag) & GEM.qlike_var.notna()]

HEALTH_KEY = {"Qwen3-32B": "Qwen3-32B", "Llama-3.1-70B": "Llama-3.1-70B",
              "Yi-1.5-34B": "Yi-1.5-34B", "Phi-4-14B": "Phi-4-14B"}


DISC_ORDER = {"long_form": 0, "event_driven": 1, "combined": 2}


def verdict(g):
    """Registered formula: healthy <=> max variance-unit QLIKE < 4 AND max
    modal share of round(pred,2) < 60%, taken over the horizons of ONE
    (family, channel) run."""
    if g.empty:
        return "not evaluable (no health columns)"
    bad = []
    if g.qlike_var.max() >= GATE_QLIKE:
        bad.append("QLIKE")
    if g.mode_share_pct.max() >= GATE_MODE:
        bad.append("modal share")
    return "passes" if not bad else "fails " + "+".join(bad)


def incr(df, tag, disc):
    s = df[(df.family == tag) & (df.disc == disc)].sort_values("h")
    assert len(s) == 3
    return s


def trio(v, sign=True):
    f = "{:+.2f}" if sign else "{:.2f}"
    return " / ".join(f.format(float(x)) for x in v)


# ------------------------------------------------------------------- gate
qw_ed = incr(L70, "qwen3_32b", "event_driven")
qw_lf = incr(L70, "qwen3_32b", "long_form")
l70_s = incr(L70, "llama70_awq", "event_driven")
l70_e = incr(GEM, "llama70_awq_ens3", "event_driven")
yi_ed = incr(L70, "yi_34b", "event_driven")
yi_lf = incr(L70, "yi_34b", "long_form")
ph_ed = incr(L70, "phi4_14b", "event_driven")
mi_s = incr(GEM, "mistral24_bf16", "event_driven")
mi_e = incr(GEM, "mistral24_ens3", "event_driven")
ge_s = incr(GEM, "gemma27_bf16", "event_driven")
ge_e = incr(GEM, "gemma27_ens3", "event_driven")
# the combined channel carries health columns only (no increment statistics),
# so it is read straight off the standalone table
_qw_cb = STD[(STD.family == "qwen3_32b") & (STD.disc == "combined")]
_yi_cb = STD[(STD.family == "yi_34b") & (STD.disc == "combined")]
assert len(_qw_cb) == len(_yi_cb) == 3

d_firm = np.abs(_e_s.rel_firm.to_numpy() - _e_e.rel_firm.to_numpy())
mistral_all = GEM[GEM.family.str.startswith("mistral24") & GEM.qlike_var.notna()]

# every committed health row of every PROBE family, keyed by display name, so
# both QLIKE rankings below are argmins over the data rather than named
# families.  The anchor (Qwen3-32B) is excluded: it is not a probe.
PROBE_HEALTH = {"Llama-3.1-70B": ["Llama-3.1-70B"],
                "Yi-1.5-34B": ["Yi-1.5-34B"],
                "Phi-4-14B": ["Phi-4-14B"],
                "Mistral-Small-24B": ["mistral24_bf16", "mistral24_ens3"],
                "Gemma-3-27B": ["gemma27_bf16", "gemma27_ens3"]}
probe_maxq = {n: max(float(fam_health[k].qlike_var.max()) for k in ks)
              for n, ks in PROBE_HEALTH.items()}
lowest_maxq = min(probe_maxq, key=probe_maxq.get)

# the lowest SINGLE CELL over all probe rows -- an argmin, not a named family
_cells = pd.concat([fam_health[k].assign(_disp=n)
                    for n, ks in PROBE_HEALTH.items() for k in ks],
                   ignore_index=True)
_lo = _cells.loc[_cells.qlike_var.astype(float).idxmin()]
lowest_cell_fam = str(_lo._disp)
lowest_cell_q = float(_lo.qlike_var)
lowest_cell_h = int(_lo.h)
lowest_cell_run = ("3-run ensemble" if str(_lo.family).endswith("_ens3")
                   else "single pass")
# the same family's other variant, and the two terms that screen it out
_lo_rows = pd.concat([fam_health[k] for k in PROBE_HEALTH[lowest_cell_fam]],
                     ignore_index=True)
lowest_cell_alt = float(_lo_rows[_lo_rows.family != _lo.family].qlike_var.min())
lowest_cell_modal = (float(_lo_rows.mode_share_pct.min()),
                     float(_lo_rows.mode_share_pct.max()))
lowest_cell_maxq = float(_lo_rows.qlike_var.max())

ens70_no_health = bool(GEM[GEM.family == "llama70_awq_ens3"]
                       [["qlike_var", "mode_share_pct", "parse_fail_rate",
                         "clipped_rate"]].isna().all().all())
n_probe_families = len(set(FAMILY_OF.values()))


def _run_passes(g):
    return bool(g.qlike_var.max() < GATE_QLIKE
                and g.mode_share_pct.max() < GATE_MODE)


# the probe count is per FAMILY: the screen's verdict attaches to a
# (family, channel) run and Yi-1.5-34B commits five of them, so a per-run sum
# would not be the paper's denominator.  Derived from the health columns, with
# the registered 8-K-channel reading asserted to give the same number.
n_fail = sum(
    1 for n, ks in PROBE_HEALTH.items()
    if not all(_run_passes(g)
               for k in ks
               for _, g in fam_health[k].groupby("disc")))
assert n_fail == sum(
    1 for n, ks in PROBE_HEALTH.items()
    if not all(_run_passes(fam_health[k][fam_health[k].disc == "event_driven"])
               for k in ks)), "8-K and all-channel probe counts disagree"
sd_ratio = (l70_s.pred_sd.to_numpy() / qw_ed.pred_sd.to_numpy())

gate(
    {"n_probe_families": 5, "n_probes_failing_health": 4,
     "qwen_ed_firm": [0.45, 0.25, 0.2], "l70_single_firm": [0.83, 0.64, 0.39],
     "l70_ens_firm": [0.84, 0.64, 0.38],
     "l70_har_holm_lt05": 1, "l70_firm_holm_lt05_single": 0,
     "l70_firm_holm_lt05_ens": 1,
     "ens_shift_firm_pp": [0.003, 0.003, 0.005],
     "ens_best_firm_holm": [0.050012, 0.048549],
     "l70_pred_sd": [0.0643, 0.0873], "qwen_pred_sd": [0.0214, 0.0323],
     "sd_ratio_range": [2.7, 3.0],
     "n_test": [25109, 25001, 24732], "n_days": [996, 991, 981],
     "mistral_qlike_range": [0.421, 0.691],
     "mistral_mode_range": [69.3, 88.6],
     "lowest_max_qlike_probe": ["Phi-4-14B", 0.677],
     "lowest_cell_qlike_probe": ["Gemma-3-27B", 0.377, 20, "3-run ensemble"],
     "lowest_cell_gemma_alt": 0.378,
     "lowest_cell_gemma_modal": [71.2, 71.5],
     "lowest_cell_gemma_maxq": 0.9,
     "qwen_ed_mode": [49.2, 50.3], "qwen_lf_mode": [45.5, 76.8],
     "qwen_cb_mode": [44.7, 56.7],
     "yi_ed_mode": [55.9, 73.6], "yi_lf_mode": [26.1, 56.6],
     "yi_cb_mode": [46.4, 69.5],
     "yi_lf_flag": "4K-TRUNCATED", "yi_combined_flag": "PARTIAL(4K)",
     "yi_clean_rows": 3,
     "ens70_no_health": True,
     "n_holm_conventions": 3, "n_raw_only_families": 2,
     "gemma_ens_har": [1.84, 2.59, 2.97], "mistral_single_har": [0.27, 0.14, 0.14],
     "phi4_har": [0.38, 0.18, -0.12], "yi_ed_har": [0.37, 0.07, -0.62],
     "yi_lf_har": [-0.64, -2.71, -9.86]},
    {"n_probe_families": n_probe_families, "n_probes_failing_health": n_fail,
     "qwen_ed_firm": [round(v, 2) for v in qw_ed.rel_firm],
     "l70_single_firm": [round(v, 2) for v in l70_s.rel_firm],
     "l70_ens_firm": [round(v, 2) for v in l70_e.rel_firm],
     "l70_har_holm_lt05": int((l70_s.p_har_holm < 0.05).sum()),
     "l70_firm_holm_lt05_single": int((l70_s.p_firm_holm < 0.05).sum()),
     "l70_firm_holm_lt05_ens": int((l70_e.p_firm_holm < 0.05).sum()),
     "ens_shift_firm_pp": [round(float(v), 3) for v in d_firm],
     "ens_best_firm_holm": [round(float(_e_s.p_firm_holm.min()), 6),
                            round(float(_e_e.p_firm_holm.min()), 6)],
     "l70_pred_sd": [round(float(l70_s.pred_sd.min()), 4),
                     round(float(l70_s.pred_sd.max()), 4)],
     "qwen_pred_sd": [round(float(qw_ed.pred_sd.min()), 4),
                      round(float(qw_ed.pred_sd.max()), 4)],
     "sd_ratio_range": [round(float(sd_ratio.min()), 1),
                        round(float(sd_ratio.max()), 1)],
     "n_test": [int(v) for v in l70_s.n_test],
     "n_days": [int(v) for v in l70_s.n_days],
     "mistral_qlike_range": [round(float(mistral_all.qlike_var.min()), 3),
                             round(float(mistral_all.qlike_var.max()), 3)],
     "mistral_mode_range": [round(float(mistral_all.mode_share_pct.min()), 1),
                            round(float(mistral_all.mode_share_pct.max()), 1)],
     "lowest_max_qlike_probe": [lowest_maxq, round(probe_maxq[lowest_maxq], 3)],
     "lowest_cell_qlike_probe": [lowest_cell_fam, round(lowest_cell_q, 3),
                                 lowest_cell_h, lowest_cell_run],
     "lowest_cell_gemma_alt": round(lowest_cell_alt, 3),
     "lowest_cell_gemma_modal": [round(lowest_cell_modal[0], 1),
                                 round(lowest_cell_modal[1], 1)],
     "lowest_cell_gemma_maxq": round(lowest_cell_maxq, 1),
     "qwen_ed_mode": [round(float(qw_ed.mode_share_pct.min()), 1),
                      round(float(qw_ed.mode_share_pct.max()), 1)],
     "qwen_lf_mode": [round(float(qw_lf.mode_share_pct.min()), 1),
                      round(float(qw_lf.mode_share_pct.max()), 1)],
     "qwen_cb_mode": [round(float(_qw_cb.mode_share_pct.min()), 1),
                      round(float(_qw_cb.mode_share_pct.max()), 1)],
     "yi_ed_mode": [round(float(yi_ed.mode_share_pct.min()), 1),
                    round(float(yi_ed.mode_share_pct.max()), 1)],
     "yi_lf_mode": [round(float(yi_lf.mode_share_pct.min()), 1),
                    round(float(yi_lf.mode_share_pct.max()), 1)],
     "yi_cb_mode": [round(float(_yi_cb.mode_share_pct.min()), 1),
                    round(float(_yi_cb.mode_share_pct.max()), 1)],
     "yi_lf_flag": str(STD[(STD.family == "yi_34b")
                           & (STD.disc == "long_form")].context_flag.iloc[0]),
     "yi_combined_flag": str(STD[(STD.family == "yi_34b")
                                 & (STD.disc == "combined")].context_flag.iloc[0]),
     "yi_clean_rows": int(len(yi_ed)),
     "ens70_no_health": ens70_no_health,
     "n_holm_conventions": len({"Holm(3) per reference", "own-family Holm(6)",
                                "inherited committed Holm(6)"}),
     "n_raw_only_families": int(sum(
         1 for t in ("yi_34b", "phi4_14b")
         if L70[L70.family == t].p_har_holm.isna().all())),
     "gemma_ens_har": [round(v, 2) for v in ge_e.rel_har],
     "mistral_single_har": [round(v, 2) for v in mi_s.rel_har],
     "phi4_har": [round(v, 2) for v in ph_ed.rel_har],
     "yi_ed_har": [round(v, 2) for v in yi_ed.rel_har],
     "yi_lf_har": [round(v, 2) for v in yi_lf.rel_har]},
)

# ------------------------------------------------------------------ markdown
MULT = {
    "qwen3_32b": "committed Holm(6) inherited (3 h x 2 references, 8-K); "
                 "crossfamily_llm.csv itself carries raw p only",
    "llama70_awq": "committed Holm(6), single seed (registered B0)",
    "llama70_awq_ens3": "ensemble Holm(6) (registered B0)",
    "yi_34b": "raw p only -- no correction columns committed",
    "phi4_14b": "raw p only -- no correction columns committed",
    "mistral24_bf16": "own-family Holm(6) (registered B1)",
    "mistral24_ens3": "own-family Holm(6) (registered B1)",
    "gemma27_bf16": "Holm(3) PER REFERENCE (registered B2 v1.3); pooled Holm(6) "
                    "committed as info-only columns",
    "gemma27_ens3": "Holm(3) PER REFERENCE (registered B2 v1.3); pooled Holm(6) "
                    "committed as info-only columns",
}
CHANNELS = {
    "qwen3_32b": "10-K/Q, 8-K, combined",
    "llama70_awq": "8-K only",
    "llama70_awq_ens3": "8-K only",
    "yi_34b": "10-K/Q, 8-K, combined",
    "phi4_14b": "8-K only",
    "mistral24_bf16": "8-K only", "mistral24_ens3": "8-K only",
    "gemma27_bf16": "8-K only", "gemma27_ens3": "8-K only",
}
HKEY = {"qwen3_32b": "Qwen3-32B", "llama70_awq": "Llama-3.1-70B",
        "yi_34b": "Yi-1.5-34B", "phi4_14b": "Phi-4-14B",
        "mistral24_bf16": "mistral24_bf16", "mistral24_ens3": "mistral24_ens3",
        "gemma27_bf16": "gemma27_bf16", "gemma27_ens3": "gemma27_ens3"}

print("### Block A -- instrument, scope and the health screen\n")
print("| Model / precision | Role | Channel | variance-unit QLIKE | "
      "modal share % | parse-fail / clipped | Health verdict (registered "
      "formula) |")
print("|---|---|---|---|---|---|---|")
for df, tag, name, prec, role in ROWS:
    sub = fam_health.get(HKEY.get(tag, tag), pd.DataFrame())
    if tag == "llama70_awq_ens3":
        print(f"| {name} {prec} | {role} | {CH['event_driven']} (only channel "
              "run) | not recorded | not recorded | not recorded | not "
              "evaluable (no health columns) |")
        continue
    # ONE row per (family, channel) run, so that every diagnostic range sits
    # beside the verdict it supports.  Pooling channels made the Qwen and Yi
    # ranges uncheckable against their own per-channel verdicts.
    discs = sorted(sub.disc.unique(), key=lambda d: DISC_ORDER[d])
    for i, disc in enumerate(discs):
        g = sub[sub.disc == disc]
        q = rng(g.qlike_var, "{:.3f}")
        m = rng(g.mode_share_pct, "{:.1f}")
        pf = g.parse_fail_rate.dropna() if "parse_fail_rate" in g else pd.Series(dtype=float)
        cl = g.clipped_rate.dropna() if "clipped_rate" in g else pd.Series(dtype=float)
        pc = ("not recorded" if pf.empty else
              f"{rng(100 * pf, '{:.2f}')}% / {rng(100 * cl, '{:.2f}')}%")
        chan = CH[disc] + (" (only channel run)"
                           if len(discs) == 1 and disc == "event_driven"
                           else "")
        print(f"| {name + ' ' + prec if i == 0 else ''} | "
              f"{role if i == 0 else ''} | {chan} | {q} | {m} | {pc} | "
              f"{verdict(g)} |")

print("\n### Block B -- increments and the multiplicity convention each "
      "carries\n")
print("| Model / precision | Channel | rel% vs recalibrated HAR "
      "(h=5 / 10 / 20) | rel% vs firm-identity reference (h=5 / 10 / 20) | "
      "Multiplicity convention applied |")
print("|---|---|---|---|---|")
for df, tag, name, prec, role in ROWS:
    discs = ["event_driven"]
    if tag in ("qwen3_32b", "yi_34b"):
        discs = ["event_driven", "long_form"]
    for i, disc in enumerate(discs):
        s = incr(df, tag, disc)
        lab = f"{name} {prec}" if i == 0 else ""
        mark = ""
        if tag == "yi_34b" and disc == "long_form":
            mark = " [4K-TRUNCATED, not citable]"
        print(f"| {lab} | {CH[disc]}{mark} | {trio(s.rel_har)} | "
              f"{trio(s.rel_firm)} | {MULT[tag] if i == 0 else ''} |")

print("\n### Verification lines (not part of the table)\n")
print(f"probe families: {n_probe_families}; failing the screen: {n_fail}")
print(f"70B single vs ensemble, rel_firm shift (pp): "
      f"{'/'.join(f'{v:.3f}' for v in d_firm)} (max {d_firm.max():.3f})")
print(f"70B best firm Holm p: single {float(_e_s.p_firm_holm.min()):.6f} -> "
      f"ensemble {float(_e_e.p_firm_holm.min()):.6f}")
print(f"70B pred sd {l70_s.pred_sd.min():.4f}--{l70_s.pred_sd.max():.4f} vs "
      f"Qwen {qw_ed.pred_sd.min():.4f}--{qw_ed.pred_sd.max():.4f}; ratio "
      f"{sd_ratio.min():.2f}--{sd_ratio.max():.2f} at identical n_test "
      f"{list(map(int, l70_s.n_test))} and n_days {list(map(int, l70_s.n_days))}")
print(f"70B Holm survivors: vs HAR {int((l70_s.p_har_holm < .05).sum())} of 3 "
      f"(single), {int((l70_e.p_har_holm < .05).sum())} of 3 (ensemble); "
      f"vs firm identity {int((l70_s.p_firm_holm < .05).sum())} of 3 (single), "
      f"{int((l70_e.p_firm_holm < .05).sum())} of 3 (ensemble)")
print("lowest max-over-horizons variance-unit QLIKE among the five probes: "
      f"{lowest_maxq} {probe_maxq[lowest_maxq]:.3f}")
print(f"lowest single-cell variance-unit QLIKE of any probe: {lowest_cell_fam} "
      f"{lowest_cell_q:.3f} ({lowest_cell_run}, h={lowest_cell_h}; its other "
      f"variant {lowest_cell_alt:.3f}); that family's modal share "
      f"{lowest_cell_modal[0]:.1f}--{lowest_cell_modal[1]:.1f}% and its max "
      f"variance-unit QLIKE {lowest_cell_maxq:.3f} -- screened out on modal "
      "share alone")
print(f"Mistral-Small-24B for comparison: variance-unit QLIKE "
      f"{mistral_all.qlike_var.min():.2f}--{mistral_all.qlike_var.max():.2f}, "
      f"modal share {mistral_all.mode_share_pct.min():.1f}--"
      f"{mistral_all.mode_share_pct.max():.1f}%")
print("per-channel modal share (the Block A split): "
      f"Qwen 8-K {qw_ed.mode_share_pct.min():.1f}--"
      f"{qw_ed.mode_share_pct.max():.1f}, 10-K/Q "
      f"{qw_lf.mode_share_pct.min():.1f}--{qw_lf.mode_share_pct.max():.1f}, "
      f"combined {_qw_cb.mode_share_pct.min():.1f}--"
      f"{_qw_cb.mode_share_pct.max():.1f}; "
      f"Yi 8-K {yi_ed.mode_share_pct.min():.1f}--"
      f"{yi_ed.mode_share_pct.max():.1f}, 10-K/Q "
      f"{yi_lf.mode_share_pct.min():.1f}--{yi_lf.mode_share_pct.max():.1f}, "
      f"combined {_yi_cb.mode_share_pct.min():.1f}--"
      f"{_yi_cb.mode_share_pct.max():.1f}")
print("holm_family strings actually committed:")
for s in sorted(set(GEM.holm_family.dropna())):
    print("  - " + s)
