"""C-anon step 3 (prereg-ea-v1.0, configs/prereg_swap_lf_and_anon.md §C-anon) —
SINGLE-SHOT scoring of the entity-anonymisation arms. Run LOCALLY after
pulling the box run dirs; refuses to run twice.

CHANNELS (--channel, each with its OWN single-shot output file — the ED table
is scored when the ED arms land and must stay write-once, while the long-form
stretch lands later; a shared file would force either delaying the registered
primary readout or rewriting an already-fired single shot):
  ed (default)  the registered primary on event_driven: C6/C2 executed;
                B2 EXITED at G1 (2026-07-16 ruling — see the exit-ruling
                block in the source constants; one status=g1-fail row with
                the official box G1 numbers enters the table, no share, no
                tests; Holm/medians run on c2+c6 only)
                -> results/tables/anon_arm.{csv,md}
  lf            the registered stretch ("long-form as stretch, only when ED completes"),
                trimmed to B2-ONLY pre-statistic and then CLOSED with zero
                executed arms: B2-lf exited at the same G1 (box lineage,
                g1_control_b2_lf_boxvenv.json) and C5/C2/C6-lf are
                not-executed -> results/tables/anon_arm_lf.{csv,md} = one
                g1-fail row + not-executed rows + the registered
                disclosures; branch adjudication: undefined (no defined
                share cell).
                Exited/not-executed arms enter the tables AS ROWS (status
                column, numeric cells n/a, reason recorded verbatim) rather
                than being silently absent.

LF-STRETCH OPERATIONALISATION, B2-only — REGISTERED as prereg-ea v1.4
(configs/prereg_swap_lf_and_anon.md §v1.4 is the binding text; summary):
  the ONLY executed arm = B2-lf: TF-IDF fixed recipe (configs/models/B2_tfidf_ridge.yaml),
  masked retrain, CPU; control = an unmasked reproduction-run comparison vs committed
  B2_tfidf_ridge_full_long_form_seed2026 (committed model.pkl missing, so the control IS
  the reproduction). **G1 (CPU arm, no deviation escape hatch) = 1e-8 full-panel reproduction gate**: the runner writes as usual
  the exact-match rate + max|diff|, this script judges by max|diff| <= 1e-8; exceeding it is a pipeline-invariance
  failure, the arm exits truthfully (G1-fail, no share estimates); --record-g1-deviation is limited to ED's
  GPU arms (c6/c2), explicitly rejected for the LF channel.
  readout = B2-lf share^anon: Holm(6) = 3 horizons x 2 refs (executed
  firmID cells with share=n/a stay in the family carrying day-clustered DM); share median = the median over
  defined cells (the 3 HAR cells). **share estimation surface shrinks (v1.4, constructive)**: B2-lf's
  unmasked increment vs the firm-identity reference = −0.615/−3.892/−8.089% (h=5/10/20,
  committed firm_identity_control.csv) — all non-positive, so the 3 firmID-side share cells are, prior to
  execution, n/a; and branch (b)'s "absorption" clause is expected-true for LF and weakly discriminating (LF degeneracy,
  paper wording softened accordingly).
  Branch quantification (v1.4, shared by both channels, "Holm-significantly positive" = masked increment rel%>0 AND
  Holm<.05): (a) ⇔ defined-cell share median >=0.75 AND the masked arm vs the HAR reference has
  0/3 cells Holm-significantly positive; (b) ⇔ median <=0.50 AND the masked arm vs the firmID reference has
  <=1/3 cells Holm-significantly positive; (c) ⇔ otherwise (incl. median falling in (0.50,0.75), or firmID
  Holm-significant >=2/3 = absorption broken), mixed cell by cell.
  Not-executed arms (all decided before any LF statistic; C5-lf enters the table as not-executed rows):
  (i)  C2-lf — two fixed-recipe FinBERT trainings ~20-30 GPU-h, small marginal defensive value;
       the ED C2 arm already covers the fine-tuned lineage, the E-lf C2 arm is registered artefact-lost;
  (ii) C5-lf (HAR side) — constructively n/a: primary citation committed m1_multiseed.csv long_form
       C5_qwen3 seed-2026 rel_impr_pct = −1.0347/−3.1346/−6.6467 (same single-seed
       basis as at scoring time); deployable_combiner FIXED mean rel% =
       −0.85/−2.48/−5.97 corroborates (3-seed basis). All negative → the HAR-side share cells are, prior to execution,
       already empty. C5-lf (firmID side) — no committed table carries that increment → **does NOT invoke** constructive
       n/a, excluded on GPU budget/scope as C6-lf; and whether it executes cannot change the branch adjudication
       (the median takes HAR cells only);
  (iii) C6-lf — the committed unmasked C6 long-form run exists and is genuine (11,907/11,907 filings
       genuinely generated, not rv22 padding), the masked arm is well-defined; excluded on GPU budget/scope
       (needs another Qwen3-32B bf16 TP=2 block), not impossible;
  (iv) therefore this channel trains and runs NO neural text model (B2 is a CPU classical model; the deep
       fine-tune/frozen-embedding/prompted-LLM lineages are covered by the ED channel only).
  Gate mapping: S-A = only B2's three cells vs committed forecast_combination_grid.csv (machine precision
  rtol 1e-12); S-B n/a (the crossfamily anchor is C6/ED-exclusive); G1 = the 1e-8 CPU gate above
  (g1_control_b2_lf.json); G2 = mask_stats_lf.json; G3 = not-executed (this channel
  has no GPU arm; if a GPU arm is reopened later, reuse the frozen-artefact hash-invariance file's slot).
  Triangulation column (operationalised) = the E-lf B2 arm's same-horizon document-level retention (committed
  swap_longform.csv); if that cell is absent **or any of E-lf G1–G3 not passed** (read
  the gate status from swap_longform_meta.json, a missing status counts as not passed) record n/a with the reason noted.
  Hard preconditions: both swap_longform.csv and anon_arm.csv (the ED table fires first) already exist.

Registered readouts (all committed before any statistic):
  * per arm (C6 prompted / C2 FinBERT-S1 / B2 TF-IDF, event_driven) x horizon
    (5/10/20): M1 log-space increment of the MASKED arm vs BOTH references —
    (a) single recalibrated HAR, (b) firm-identity-augmented HAR (val-window
    firm-mean spec) — with day-clustered DM; the M1 block is copied VERBATIM
    from scripts/analysis/crossfamily_llama70.py.
  * Holm within each arm's pre-declared family of 6 tests (3 horizons x 2 refs).
  * Pre-declared point estimate:
        identity_share_anon = 1 - masked increment / unmasked increment
    per cell (arm x h x ref) + aggregate (median), unmasked increments
    recomputed in-script from the committed runs and anchored to the
    committed tables at machine precision (sanity gates below).
  * Triangulation vs (i) the reference-interval bound
    1 - rel_firm_unmasked / rel_har_unmasked (committed firm-identity
    reference logic) and (ii) the matched-swap retention
    (results/tables/matched_firm_swap.csv).

Pre-registered branches (adjudicated per arm; the numeric thresholds below
are a POST-REGISTRATION OPERATIONALISATION — the prereg text gives the three
branches without cutoffs — and are disclosed in the output):
  (a) masked increment ~ 0 (share -> 1): identity-dominated; the title's
      identity narrative gains a point estimate.        [median share_HAR >=
      0.75 AND no masked HAR cell Holm-significant]
  (b) masked increment largely retained AND still absorbed by the firm-ID
      reference: genuinely firm-stable content; title wording SOFTENS
      (fix-that-reduces, committed).                    [median share_HAR <=
      0.50 AND <=1/3 masked firm-ref cells Holm-significant]
  (c) masked increment retained AND NO LONGER absorbed by firm-ID: masking
      broke the firm-stable channel alignment — reported as a methodological
      finding.                                          [median share_HAR <=
      0.50 AND >=2/3 masked firm-ref cells Holm-significant]
  otherwise: mixed — reported cell by cell. ALL branches go in the paper.

HARD GATES (abort before writing anything):
  S-A  unmasked rel% vs single HAR reproduces the committed
       forecast_combination_grid.csv rel_impr_pct at machine precision
       (9 cells);
  S-B  unmasked C6 cells reproduce crossfamily_llm.csv (qwen3_32b,
       event_driven) on all M1 columns at machine precision;
  G1   results/anon/g1_control_<arm>.json exists with pass=true for every
       arm (pipeline invariance; the registered bit-identity control) —
       a documented deviation can be recorded with --record-g1-deviation
       "reason", which is stamped into the outputs;
  G2   results/anon/mask_stats.json exists (non-smoke) — leak rates surfaced;
  G3   results/anon/g3_truncation_stats.json exists — truncation surfaced.

Run from repo root (ONCE):  .venv/bin/python scripts/analysis/anon_score.py
Pre-flight without scoring: --check-inputs
Outputs (NEW files): results/tables/anon_arm.{csv,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # noqa: E402
import clustered_dm as cdm  # noqa: E402

KEY = ["ticker", "accession", "horizon_days"]
EPS = 1e-8
HORIZONS = (5, 10, 20)
RTOL = 1e-12
ATOL_XPATH = 1e-8  # cross-code-path anchor (E-lf/matched_firm_swap convention)
ANON = Path("results/anon")
M1_COLS = ["rel_har", "dm_har", "p_har", "rel_firm", "dm_firm", "p_firm", "g_text"]
BRANCH_HI, BRANCH_LO = 0.75, 0.50  # v1.4-registered branch cutoffs (both channels)
G1_LF_TOL = 1e-8       # v1.4: LF CPU-arm G1 = 1e-8 full-panel reproduction gate
ED_GPU_ARMS = {"c6", "c2"}  # v1.1 deviation recording is GPU-arm-only (ED)

# ---- B2 exit ruling (2026-07-16, decided BEFORE either table fired) --------
# The registered B2 arm executed on the BOX, so the box control G1 is the
# arm's OFFICIAL G1. It failed far beyond any numeric-noise reading, and the
# diagnosis (all facts reproducible from the files cited below) closed as:
#   * box control (sklearn 1.9.0/py3.12, box text cache): exact-match 0,
#     max|diff| 1.926e-01                       -> g1_control_b2_boxvenv.json
#   * local reproduction (sklearn 1.8.0/py3.11.15, numpy 2.4.6, /Volumes/Z
#     cache): max|diff| 1.303e-03      -> g1_control_b2_local_numpy246.json
#   * trial venv (numpy downgraded to 2.3.5): deviations BIT-IDENTICAL to
#     numpy 2.4.6 -> numpy ruled out   -> g1_control_b2_local_numpy235.json
#   * both control model.pkl CV alphas identical (1.0 at h=5/10/20)
#     -> the alpha hypothesis is dead;
#   * BUT the fitted TF-IDF vocabularies are DIFFERENT SETS and idf
#     max|diff| = 7.5 -> text-store LINEAGE DRIFT confirmed;
#   * the committed June run's (env x cache) pair is UNRECONSTRUCTIBLE:
#     env.json records no package versions (pip_freeze_hash=null), and
#     scipy 1.18 does not exist on py3.11, so the June box scipy <=1.17.1.
# Ruling: B2 exits BOTH channels under the v1.4-registered CPU rule (no
# deviation path). The box same-lineage ctrl/masked pair goes to the
# NON-REGISTERED annex (anon_annex_samelineage.py), never into these tables.
B2_EXIT_REASON_ED = (
    "arm exited at G1 (v1.4 CPU rule, no deviation path). Official G1 = the "
    "box control (g1_control_b2_boxvenv.json; the registered arm executed on "
    "the box): exact-match 0, max|diff| 1.926e-01. Diagnosis: the committed "
    "June (env x cache) pair is unreconstructible (env.json has no package "
    "versions, pip_freeze_hash=null; scipy 1.18 does not exist on py3.11 so "
    "the June box scipy was <=1.17.1) and the text-store lineage has drifted "
    "— the box-control and local-control TF-IDF fits agree on CV alphas "
    "(1.0 at h=5/10/20) yet carry DIFFERENT vocabulary SETS with idf "
    "max|diff| 7.5, so the two text caches are different lineages and the "
    "committed fit's lineage is a third, unreconstructible state. Local "
    "reproduction attempts bottom out at max|diff| 1.303e-03 (sklearn "
    "1.8.0/py3.11.15; numpy 2.4.6 and 2.3.5 deviations bit-identical -> "
    "numpy ruled out). No share, no tests; see Disclosures.")
B2_EXIT_REASON_LF = (
    "arm exited at G1 (v1.4 CPU rule: 1e-8 full-panel reproduction, no "
    "deviation path). Official G1 = the box control "
    "(g1_control_b2_lf_boxvenv.json; same box venv/cache lineage as the ED "
    "control): exact-match 0, max|diff| far above 1e-8 (row g1 columns). "
    "Same root cause as the ED exit — committed June (env x cache) pair "
    "unreconstructible + text-store lineage drift (vocabulary-set/idf "
    "evidence in the ED table's b2 diagnostics). No share, no tests.")

CHANNELS = {
    "ed": dict(
        disc="event_driven",
        arms=[("c6", "C6_llmtext", "C6_llmtext_anonmask"),
              ("c2", "C2_finbert_s1", "C2_finbert_s1_anonmask")],
        csv=Path("results/tables/anon_arm.csv"),
        md=Path("results/tables/anon_arm.md"),
        g1_suffix="",
        mask_stats="mask_stats.json",
        prereg="prereg-ea-v1.0 §C-anon",
        # arm(s) exited at G1 (see the B2 exit ruling block above): one row
        # per arm enters the table (status=g1-fail, exact rate + max|diff|
        # from the OFFICIAL box json, no share, no tests); Holm/medians run
        # on the remaining executed arms only.
        g1_failed={
            "b2": dict(model="B2_tfidf_ridge",
                       g1_json="g1_control_b2_boxvenv.json",
                       diagnostics={
                           "box sklearn1.9.0/py3.12 box-cache (OFFICIAL)":
                               "g1_control_b2_boxvenv.json",
                           "local sklearn1.8.0/py3.11.15 numpy2.4.6":
                               "g1_control_b2_local_numpy246.json",
                           "local trial venv numpy2.3.5":
                               "g1_control_b2_local_numpy235.json",
                       },
                       reason=B2_EXIT_REASON_ED),
        },
    ),
    "lf": dict(
        disc="long_form",
        arms=[],  # B2-lf exited at G1 -> the channel has NO executed arm
        csv=Path("results/tables/anon_arm_lf.csv"),
        md=Path("results/tables/anon_arm_lf.md"),
        g1_suffix="_lf",
        mask_stats="mask_stats_lf.json",
        prereg="prereg-ea v1.4 §C-anon (long_form stretch, B2-only)",
        g1_failed={
            "b2": dict(model="B2_tfidf_ridge",
                       g1_json="g1_control_b2_lf_boxvenv.json",
                       diagnostics={},  # full chain lives in the ED table
                       reason=B2_EXIT_REASON_LF),
        },
        # arm decided-not-executed BEFORE any LF statistic: it enters the
        # output table as rows (status=not-executed, numeric n/a) with the
        # reason recorded verbatim — never silently absent. C2-lf / C6-lf
        # were never candidate LF arms of this scorer and stay as Disclosure
        # lines (the runner docstring carries all four exclusion reasons).
        not_executed={
            "c5": dict(model="C5_qwen3", reason=(
                "excluded pre-statistic (prereg-ea v1.4): HAR side "
                "constructively n/a — committed m1_multiseed.csv long_form "
                "C5_qwen3 seed-2026 rel_impr_pct = -1.0347/-3.1346/-6.6467 "
                "(primary, same single-seed basis as this scorer; "
                "deployable_combiner FIXED mean rel% -0.85/-2.48/-5.97 "
                "corroborates, 3-seed basis), all negative, so every HAR "
                "share cell is empty before execution; firmID side — no "
                "committed table carries that increment, so constructive n/a "
                "is NOT invoked: excluded on GPU budget/scope (as C6-lf), "
                "and its execution could not change the branch adjudication "
                "(the share median takes HAR cells only)")),
        },
    ),
}
# channel-scoped globals, set by _set_channel() before any scoring
CHANNEL = "ed"
DISC = CHANNELS["ed"]["disc"]
ARMS = CHANNELS["ed"]["arms"]
FINAL_CSV = CHANNELS["ed"]["csv"]
FINAL_MD = CHANNELS["ed"]["md"]
G1_SUFFIX = CHANNELS["ed"]["g1_suffix"]
MASK_STATS = CHANNELS["ed"]["mask_stats"]
PREREG = CHANNELS["ed"]["prereg"]
NOT_EXECUTED = {}
G1_FAILED = {}


def _set_channel(ch: str) -> None:
    global CHANNEL, DISC, ARMS, FINAL_CSV, FINAL_MD, G1_SUFFIX, MASK_STATS, \
        PREREG, NOT_EXECUTED, G1_FAILED
    c = CHANNELS[ch]
    CHANNEL = ch
    DISC, ARMS = c["disc"], c["arms"]
    FINAL_CSV, FINAL_MD = c["csv"], c["md"]
    G1_SUFFIX, MASK_STATS, PREREG = c["g1_suffix"], c["mask_stats"], c["prereg"]
    NOT_EXECUTED = c.get("not_executed", {})
    G1_FAILED = c.get("g1_failed", {})


def ols(y, X):  # verbatim crossfamily_llama70.py
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def holm(ps):  # verbatim crossfamily_llama70.py
    ps = np.asarray(ps, float)
    n = len(ps)
    order = np.argsort(ps)
    out = np.empty(n)
    for rank, idx in enumerate(order):
        out[idx] = ps[idx] * (n - rank)
    run = 0.0
    for idx in order:
        run = max(run, out[idx])
        out[idx] = min(run, 1.0)
    return out


def close(a, b):
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= RTOL * max(abs(a), abs(b), 1.0)


def m1_cells(run: str) -> dict[int, dict]:
    """Per-horizon M1 vs both references — the crossfamily_llama70.py block
    verbatim (same merge, same dropna, same firm-mean spec, same clustering)."""
    a2 = fc.load("A2_har_rv", DISC)[KEY + ["split", "label_realised_vol",
                                           "prediction_realised_vol",
                                           "effective_trading_day"]] \
        .rename(columns={"prediction_realised_vol": "fh"})
    p = fc.load(run, DISC)
    t = p[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "ft"})
    out = {}
    for h in HORIZONS:
        m = a2[a2.horizon_days == h].merge(t[t.horizon_days == h], on=KEY).dropna()
        v, te = m[m.split == "val"], m[m.split == "test"]
        y = te.label_realised_vol.values
        fR, fU, g = fc.log_combo(v.label_realised_vol.values, v.fh.values,
                                 v.ft.values, te.fh.values, te.ft.values)
        qR, qU = fc.qlike(y, fR), fc.qlike(y, fU)
        rel = 100 * np.mean(qR - qU) / np.mean(qR)
        dm, pv, nd = cdm.dm_test_clustered(qU, qR, te.effective_trading_day.values, h)
        fm = v.groupby("ticker").label_realised_vol.mean()
        gmean = v.label_realised_vol.mean()
        fid_v = v.ticker.map(fm).fillna(gmean).values
        fid_t = te.ticker.map(fm).fillna(gmean).values
        L = lambda x: np.log(np.clip(x, EPS, None))  # noqa: E731
        ly = L(v.label_realised_vol.values)
        bR = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v)]))
        bU = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v),
                                      L(v.ft.values)]))
        fRf = np.exp(bR[0] + bR[1] * L(te.fh.values) + bR[2] * L(fid_t))
        fUf = np.exp(bU[0] + bU[1] * L(te.fh.values) + bU[2] * L(fid_t)
                     + bU[3] * L(te.ft.values))
        qRf, qUf = fc.qlike(y, fRf), fc.qlike(y, fUf)
        relf = 100 * np.mean(qRf - qUf) / np.mean(qRf)
        dmf, pf, _ = cdm.dm_test_clustered(qUf, qRf, te.effective_trading_day.values, h)
        out[h] = dict(n_test=len(te), n_days=nd, rel_har=rel, dm_har=dm, p_har=pv,
                      rel_firm=relf, dm_firm=dmf, p_firm=pf, g_text=float(g))
    return out


def nmed(vals) -> float:
    """Median over finite values; NaN (silently) when none — the all-n/a case
    (e.g. an arm whose unmasked increments are all non-positive) is expected
    and disclosed, not a warning."""
    x = np.asarray(list(vals), float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def share(masked_rel: float, unmasked_rel: float) -> float:
    """identity_share_anon = 1 - masked/unmasked; NaN when the unmasked cell
    had no positive increment to decompose (disclosed, not clipped)."""
    if not np.isfinite(unmasked_rel) or unmasked_rel <= 0:
        return float("nan")
    return 1.0 - masked_rel / unmasked_rel


def check_inputs(record_g1_dev: str | None) -> tuple[list[str], list[str]]:
    problems, notes = [], []
    runner = "anon_run_arms.py" if CHANNEL == "ed" else "anon_run_arms_lf.py"
    for _arm, base, masked in ARMS:
        for run in (base, masked):
            p = Path(f"results/runs/{run}_full_{DISC}_seed2026/predictions.parquet")
            if not p.exists():
                problems.append(f"missing run: {p}")
    tables = (("results/tables/forecast_combination_grid.csv",
               "results/tables/crossfamily_llm.csv",
               "results/tables/matched_firm_swap.csv") if CHANNEL == "ed" else
              ("results/tables/forecast_combination_grid.csv",
               "results/tables/swap_longform.csv",
               "results/tables/anon_arm.csv"))
    for t in tables:
        if not Path(t).exists():
            hint = ""
            if t.endswith("swap_longform.csv"):
                if Path("results/tables/swap_longform_retirement.md").exists():
                    # prereg-ea v1.5: E-lf retired at its own gates; the
                    # retirement record is the registered alternative
                    # precondition (the triangulation column reports n/a).
                    notes.append("swap_longform.csv absent; retirement record "
                                 "accepted per v1.5 — triangulation n/a "
                                 "(E-lf retired at its own gates)")
                    continue
                hint = " (score E-lf first — the triangulation column)"
            elif CHANNEL == "lf" and t.endswith("anon_arm.csv"):
                hint = (" (v1.4 timing gate: the ED anon table must fire "
                        "BEFORE any LF statistic)")
            problems.append(f"missing committed table: {t}{hint}")
    for arm, _b, _m in ARMS:
        g1 = ANON / f"g1_control_{arm}{G1_SUFFIX}.json"
        if not g1.exists():
            problems.append(f"G1 missing: {g1} (run {runner} --arm {arm} "
                            "--control on the box)")
            continue
        v = json.loads(g1.read_text())
        if CHANNEL == "lf":
            # v1.4: CPU-arm gate = 1e-8 full-panel reproduction, NO deviation
            # path — above tolerance the arm exits (G1-fail, no share
            # estimates), which for the B2-only channel blocks scoring.
            mad = float(v.get("max_abs_diff", float("inf")))
            if not (v.get("rows_align") and mad <= G1_LF_TOL):
                problems.append(
                    f"G1 FAIL (1e-8 CPU reproduction gate) for {arm}: "
                    f"rows_align={v.get('rows_align')}, max|diff| {mad:.3e} > "
                    f"{G1_LF_TOL:g} — the arm exits (G1-fail, no share "
                    "estimates); the deviation path is ED/GPU-arm-only (v1.4)")
        elif not v.get("pass"):
            msg = (f"G1 FAIL for {arm}: exact {v.get('n_exact')}/"
                   f"{v.get('n_joined')}, max|diff| {v.get('max_abs_diff')}")
            if record_g1_dev and arm in ED_GPU_ARMS:
                notes.append(msg + f" — DEVIATION RECORDED: {record_g1_dev}")
            elif arm in ED_GPU_ARMS:
                problems.append(msg + " — registered gate is bit-identity; "
                                "pass --record-g1-deviation 'reason' to "
                                "proceed with a documented deviation "
                                "(GPU arms only, v1.1)")
            else:
                problems.append(msg + " — CPU arm: bit-identity is the "
                                "registered gate and the v1.1 deviation "
                                "path is GPU-arm-only; a CPU mismatch is a "
                                "real pipeline-invariance failure")
    # ---- g1-failed arms: the exit ruling needs its evidence on disk ----
    for arm, info in G1_FAILED.items():
        p = ANON / info["g1_json"]
        if not p.exists():
            problems.append(f"g1-fail evidence missing for {arm}: {p} (the "
                            "official box G1 backing the exit ruling)")
            continue
        v = json.loads(p.read_text())
        if CHANNEL == "ed":
            gate_ok = bool(v.get("pass"))
        else:
            gate_ok = (bool(v.get("rows_align")) and
                       float(v.get("max_abs_diff", float("inf"))) <= G1_LF_TOL)
        if gate_ok:
            problems.append(f"registry marks {arm} g1-failed but the official "
                            f"G1 ({p.name}) PASSES its gate — the exit ruling "
                            "must be revisited before scoring")
        for label, fn in info.get("diagnostics", {}).items():
            if not (ANON / fn).exists():
                problems.append(f"g1-fail diagnostic missing for {arm}: "
                                f"results/anon/{fn} ({label})")
    if ARMS and not (ANON / MASK_STATS).exists():
        # G2 gates EXECUTED statistics; a channel with no executed arm has
        # none to gate (the g1-fail/not-executed rows carry no statistic).
        problems.append(f"G2 missing: results/anon/{MASK_STATS} (non-smoke "
                        f"mask build{' --panel lf' if CHANNEL == 'lf' else ''})")
    if CHANNEL == "ed":
        if not (ANON / "g3_truncation_stats.json").exists():
            problems.append("G3 missing: results/anon/g3_truncation_stats.json")
    else:
        # B2-only channel: no GPU arm -> no frozen-artefact hash-invariance
        # file is EXPECTED; its absence is the not-executed state, not a gap.
        # If the file exists (a GPU arm was run despite the exclusion), a
        # recorded FAIL still blocks.
        g3p = ANON / "g3_c5_lf_hash_invariance.json"
        if not g3p.exists():
            notes.append("G3: not-executed — no GPU arm in this channel "
                         "(B2-only; C5-lf excluded pre-statistic)")
        elif not json.loads(g3p.read_text()).get("pass"):
            problems.append(f"G3 FAIL recorded in {g3p}")
    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=sorted(CHANNELS), default="ed",
                    help="ed = registered primary (event_driven C6/C2/B2, "
                         "default); lf = registered stretch (long_form "
                         "B2-lf/C5-lf) -> anon_arm_lf.{csv,md}")
    ap.add_argument("--check-inputs", action="store_true",
                    help="pre-flight: verify every input exists; compute nothing")
    ap.add_argument("--record-g1-deviation", default=None, metavar="REASON",
                    help="ED GPU arms (c6/c2) ONLY: proceed despite a G1 "
                         "bit-identity failure; REASON is stamped into the "
                         "output table (documented deviation). REJECTED for "
                         "--channel lf (v1.4: the CPU-arm gate is a 1e-8 "
                         "reproduction gate with no deviation path).")
    args = ap.parse_args()
    if args.channel == "lf" and args.record_g1_deviation:
        ap.error("--record-g1-deviation is ED/GPU-arm-only (prereg-ea v1.4): "
                 "the LF CPU-arm G1 is a 1e-8 full-panel reproduction gate "
                 "with NO deviation path — above tolerance the arm exits.")
    _set_channel(args.channel)

    # ---------------- single-shot + prerequisite gates ----------------
    if FINAL_CSV.exists() or FINAL_MD.exists():
        print(f"REFUSED: {FINAL_CSV} / {FINAL_MD} already exists — single-shot "
              "discipline (family convention). Any re-run requires a recorded "
              "prereg revision first.")
        sys.exit(2)
    problems, notes = check_inputs(args.record_g1_deviation)
    if args.check_inputs:
        for p in problems:
            print("MISSING/FAIL:", p)
        for n in notes:
            print("NOTE:", n)
        print("check-inputs:", "NOT READY" if problems else "READY")
        sys.exit(3 if problems else 0)
    if problems:
        for p in problems:
            print("ABORT:", p)
        sys.exit(1)

    grid = pd.read_csv("results/tables/forecast_combination_grid.csv") \
        .set_index(["disc", "model", "h"])
    if CHANNEL == "ed":
        cfl = pd.read_csv("results/tables/crossfamily_llm.csv")
        swap = pd.read_csv("results/tables/matched_firm_swap.csv") \
            .set_index(["disc", "model", "h"])
        g3 = json.loads((ANON / "g3_truncation_stats.json").read_text())
    else:
        cfl = None
        swap_p = Path("results/tables/swap_longform.csv")
        if swap_p.exists():
            swap = pd.read_csv(swap_p).set_index(["disc", "model", "h"])
        else:
            # prereg-ea v1.5: E-lf retired at its own gates; the retirement
            # record substitutes and the triangulation column is n/a.
            swap = pd.DataFrame(
                columns=["disc", "model", "h"]).set_index(
                ["disc", "model", "h"])
        g3p = ANON / "g3_c5_lf_hash_invariance.json"
        g3 = json.loads(g3p.read_text()) if g3p.exists() else None
        # v1.4 triangulation conditionality: retention is reported ONLY if
        # the E-lf gates G1-G3 are verified PASS in swap_longform_meta.json;
        # a missing/incomplete status counts as NOT passed -> n/a + reason.
        meta_p = Path("results/tables/swap_longform_meta.json")
        if meta_p.exists():
            gm = json.loads(meta_p.read_text()).get("gates", {})
            _g1, _g2, _g3 = gm.get("g1"), gm.get("g2"), gm.get("g3")
            g1ok = isinstance(_g1, dict) and bool(_g1.get("pass"))
            g2ok = isinstance(_g2, dict) and bool(_g2.get("pass"))
            # E-lf G3 is fatal-on-fail: a recorded dict means it ran through
            g3ok = isinstance(_g3, dict) and bool(_g3) and not _g3.get("skipped")
            elf_gates_ok = g1ok and g2ok and g3ok
            if not elf_gates_ok:
                elf_gate_note = ("E-lf gates unverified in swap_longform_meta"
                                 f".json (g1={g1ok}, g2={g2ok}, g3={g3ok})")
        else:
            elf_gates_ok = False
            elf_gate_note = ("swap_longform_meta.json absent — E-lf gate "
                             "status unknown (v1.4: missing status counts as "
                             "not passed)")
    g2p = ANON / MASK_STATS
    g2 = json.loads(g2p.read_text()) if g2p.exists() else None
    if ARMS and g2 is None:
        print(f"ABORT: G2 missing at scoring time: {g2p}")
        sys.exit(1)
    g1v = {arm: json.loads((ANON / f"g1_control_{arm}{G1_SUFFIX}.json").read_text())
           for arm, _b, _m in ARMS}
    g1f = {arm: json.loads((ANON / info["g1_json"]).read_text())
           for arm, info in G1_FAILED.items()}

    rows = []
    branch_by_arm = {}
    for arm, base, masked in ARMS:
        un = m1_cells(base)
        ma = m1_cells(masked)

        # ---- S-A: unmasked vs-HAR rel reproduces the committed anchor ----
        # In-grid cells: machine precision vs forecast_combination_grid.csv.
        # LF C5 has NO committed grid cell — its anchor is the committed E-lf
        # real_rel_pct (swap_longform.csv) at atol 1e-8, the same cross-code-
        # path tolerance matched_firm_swap.py / swap_longform_score.py use.
        for h in HORIZONS:
            if (DISC, base, h) in grid.index:
                gref = float(grid.loc[(DISC, base, h), "rel_impr_pct"])
                if not close(un[h]["rel_har"], gref):
                    print(f"SANITY S-A FAIL {base} h{h}: recomputed "
                          f"{un[h]['rel_har']!r} vs grid {gref!r}")
                    sys.exit(1)
            elif CHANNEL == "lf" and (DISC, base, h) in swap.index:
                sref = float(swap.loc[(DISC, base, h), "real_rel_pct"])
                if not np.isclose(un[h]["rel_har"], sref, atol=ATOL_XPATH,
                                  rtol=0.0):
                    print(f"SANITY S-A FAIL {base} h{h}: recomputed "
                          f"{un[h]['rel_har']!r} vs swap_longform real_rel_pct "
                          f"{sref!r} (atol {ATOL_XPATH:g})")
                    sys.exit(1)
            else:
                print(f"SANITY S-A FAIL {base} h{h}: no committed anchor "
                      "(neither grid nor swap_longform carries the cell)")
                sys.exit(1)
        # ---- S-B: unmasked C6 cells reproduce crossfamily_llm.csv (ED only) ----
        if arm == "c6":
            ref = cfl[(cfl.disc == DISC) & (cfl.family == "qwen3_32b")]
            for _, r in ref.iterrows():
                for c in M1_COLS:
                    if not close(un[int(r.h)][c], r[c]):
                        print(f"SANITY S-B FAIL C6 h{int(r.h)} {c}: "
                              f"{un[int(r.h)][c]!r} vs {r[c]!r}")
                        sys.exit(1)

        # ---- pre-declared Holm(6) within this arm ----
        ph = holm([ma[h]["p_har"] for h in HORIZONS]
                  + [ma[h]["p_firm"] for h in HORIZONS])
        for i, h in enumerate(HORIZONS):
            ma[h]["p_har_holm"] = float(ph[i])
            ma[h]["p_firm_holm"] = float(ph[3 + i])

        for h in HORIZONS:
            # triangulation: ED = committed forecast-level matched swap;
            # LF = E-lf DOCUMENT-level retention (the LF-relevant readout)
            if (DISC, base, h) in swap.index:
                srow = swap.loc[(DISC, base, h)]
                if CHANNEL == "ed":
                    sret = float(srow["retention_vs_real"])
                    sretf = float(srow["firmref_retention"]) \
                        if "firmref_retention" in srow and pd.notna(
                            srow.get("firmref_retention")) else float("nan")
                else:
                    # v1.4: n/a unless the E-lf gates are verified PASS
                    sret = (float(srow["retention_doc"]) if elf_gates_ok
                            else float("nan"))
                    sretf = float("nan")  # no firm-ref swap variant for LF
            else:
                sret, sretf = float("nan"), float("nan")
            rows.append({
                "arm": arm, "model": base, "h": h, "channel": CHANNEL,
                "n_test_masked": ma[h]["n_test"], "n_days": ma[h]["n_days"],
                "rel_har_unmasked": un[h]["rel_har"],
                "rel_har_masked": ma[h]["rel_har"],
                "dm_har_masked": ma[h]["dm_har"], "p_har_masked": ma[h]["p_har"],
                "p_har_holm": ma[h]["p_har_holm"],
                "share_anon_har": share(ma[h]["rel_har"], un[h]["rel_har"]),
                "rel_firm_unmasked": un[h]["rel_firm"],
                "rel_firm_masked": ma[h]["rel_firm"],
                "dm_firm_masked": ma[h]["dm_firm"],
                "p_firm_masked": ma[h]["p_firm"],
                "p_firm_holm": ma[h]["p_firm_holm"],
                "share_anon_firm": share(ma[h]["rel_firm"], un[h]["rel_firm"]),
                "refinterval_bound": share(un[h]["rel_firm"], un[h]["rel_har"]),
                "swap_retention": sret,
                "swap_retention_firmref": sretf,
                "g_text_masked": ma[h]["g_text"],
            })

        # ---- branch adjudication (prereg-ea v1.4 quantified conditions,
        #      BOTH channels; registered pre-fire for both tables).
        #      "Holm-significantly positive" = the masked increment is POSITIVE (rel% > 0)
        #      AND Holm p < .05 — the increment's own sign carries the
        #      direction, NOT a bare two-sided significance reading. ----
        arm_rows = [r for r in rows if r["arm"] == arm]
        med_share = nmed(r["share_anon_har"] for r in arm_rows)  # defined (HAR) cells
        n_sig_har = sum(1 for r in arm_rows
                        if r["rel_har_masked"] > 0 and r["p_har_holm"] < .05)
        n_sig_firm = sum(1 for r in arm_rows
                         if r["rel_firm_masked"] > 0 and r["p_firm_holm"] < .05)
        if np.isnan(med_share):
            b = ("undefined — no defined share cell (no positive unmasked "
                 "increment to decompose); cells reported as-is")
        elif med_share >= BRANCH_HI and n_sig_har == 0:
            b = ("(a) identity-dominated: defined-cell share median "
                 f"{med_share:.2f} >= {BRANCH_HI:g} AND 0/3 masked HAR cells "
                 "Holm-significantly positive; the identity narrative gains "
                 "a point estimate")
        elif med_share <= BRANCH_LO and n_sig_firm <= 1:
            b = ("(b) firm-stable content: defined-cell share median "
                 f"{med_share:.2f} <= {BRANCH_LO:g} AND the masked increment "
                 "is still absorbed by the firm-ID reference (<=1/3 firmID "
                 "cells Holm-significantly positive) — title wording softens "
                 "(committed)")
        else:
            b = ("(c) otherwise (incl. median in (0.50,0.75) or >=2/3 firmID "
                 "cells Holm-significantly positive = absorption broken): "
                 f"median share {med_share:.2f}, Holm-significantly-positive "
                 f"masked cells {n_sig_har}/3 vs HAR, {n_sig_firm}/3 vs "
                 "firm-ID — mixed, reported cell by cell")
        branch_by_arm[arm] = dict(median_share_har=med_share,
                                  n_sig_har=n_sig_har, n_sig_firm=n_sig_firm,
                                  branch=b)

    df = pd.DataFrame(rows)
    if NOT_EXECUTED or G1_FAILED:
        # non-executed/exited arms enter the table AS ROWS, never silently
        if len(df):
            df["status"] = "executed"
            df["reason"] = ""
        extra = [{"arm": arm, "model": info["model"], "h": h,
                  "channel": CHANNEL, "status": "not-executed",
                  "reason": info["reason"]}
                 for arm, info in NOT_EXECUTED.items() for h in HORIZONS]
        # g1-fail: ONE row per arm (the G1 verdict is arm-level, not
        # per-horizon); carries the OFFICIAL box G1 numbers, no share, no test
        extra += [{"arm": arm, "model": info["model"],
                   "channel": CHANNEL, "status": "g1-fail",
                   "reason": info["reason"],
                   "g1_json": info["g1_json"],
                   "g1_exact_match_rate": g1f[arm].get("exact_match_rate"),
                   "g1_max_abs_diff": g1f[arm].get("max_abs_diff"),
                   "g1_n_joined": g1f[arm].get("n_joined")}
                  for arm, info in G1_FAILED.items()]
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True) \
            if len(df) else pd.DataFrame(extra)

    def dcol(name):  # column accessor tolerant of all-exited channels
        return df[name] if name in df.columns else pd.Series(dtype=float)

    agg = {
        "median_share_har_all": nmed(dcol("share_anon_har")),
        "median_share_firm_all": nmed(dcol("share_anon_firm")),
        "median_share_har_by_arm": {a: v["median_share_har"]
                                    for a, v in branch_by_arm.items()},
        "median_refinterval_bound": nmed(dcol("refinterval_bound")),
        "median_swap_retention": nmed(dcol("swap_retention")),
    }

    # ---------------- write (single-shot re-check just before) ----------------
    if FINAL_CSV.exists():
        print(f"REFUSED at write time: {FINAL_CSV} appeared concurrently")
        sys.exit(2)
    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = {
        "prereg": PREREG,
        "g1_deviation": args.record_g1_deviation or "",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = df.copy()
    for k, v in meta_cols.items():
        out[k] = v
    out.to_csv(FINAL_CSV, index=False)

    def cell(r, ref):
        rel_u, rel_m = r[f"rel_{ref}_unmasked"], r[f"rel_{ref}_masked"]
        dm, hp = r[f"dm_{ref}_masked"], r[f"p_{ref}_holm"]
        s = r[f"share_anon_{ref}"]
        # v1.4 direction convention: Holm-significantly POSITIVE increment
        star = "**" if (rel_m > 0 and hp < .05) else ""
        return (f"{rel_u:+.3f} | {rel_m:+.3f}{star} | {dm:+.2f} | {hp:.4g} | "
                f"{s:+.2f}" if np.isfinite(s) else
                f"{rel_u:+.3f} | {rel_m:+.3f}{star} | {dm:+.2f} | {hp:.4g} | n/a")

    n_cells = len(ARMS) * len(HORIZONS)
    if CHANNEL == "ed":
        title = ("# C-anon — entity-anonymisation arms (prereg-ea-v1.0): "
                 "identity share, bound -> point estimate")
        sa_line = (f"- S-A PASS: all {n_cells} unmasked vs-HAR cells reproduce "
                   f"forecast_combination_grid.csv at machine precision "
                   f"(rtol {RTOL:g}).")
        sb_line = ("- S-B PASS: unmasked C6 cells reproduce crossfamily_llm.csv "
                   "(qwen3_32b, event_driven) on all M1 columns at machine "
                   "precision.")
    else:
        title = ("# C-anon-lf — entity-anonymisation, LONG-FORM stretch, "
                 "B2-only (prereg-ea v1.4 §C-anon: long-form stretch): identity "
                 "share, bound -> point estimate")
        if ARMS:
            sa_line = (f"- S-A PASS: all {n_cells} unmasked vs-HAR cells "
                       f"(B2-only channel) reproduce the committed "
                       f"forecast_combination_grid.csv at machine precision "
                       f"(rtol {RTOL:g}).")
        else:
            sa_line = ("- S-A n/a: no executed arm in this channel (B2-lf "
                       "exited at G1; C5/C2/C6-lf not-executed) — no unmasked "
                       "cell was recomputed.")
        sb_line = ("- S-B n/a: the crossfamily anchor is C6/event_driven-only "
                   "and C6-lf is not an arm in this channel (see Disclosures).")
    md = [
        title,
        "",
        f"Committed before any statistic; scored once on "
        f"{meta_cols['generated_utc']}. identity_share_anon = 1 - masked/"
        "unmasked M1 increment (log-space combiner, val-fit test-frozen, "
        "day-clustered DM — the crossfamily_llama70.py block verbatim). "
        "Holm(6) per arm (3 horizons x 2 references), the pre-declared family; "
        "share=n/a cells stay in the family carrying their DM test. "
        "`**` = masked increment Holm-significantly POSITIVE (rel% > 0 and "
        "Holm p<.05 — the v1.4 direction convention).",
        "",
        "## Gates",
        "",
        sa_line,
        sb_line,
    ]
    for arm, v in g1v.items():
        if CHANNEL == "lf":
            # check_inputs already enforced the 1e-8 gate; label it honestly
            md.append(f"- G1 {arm} (v1.4 CPU gate: 1e-8 full-panel "
                      f"reproduction, NO deviation path): PASS — max|diff| "
                      f"{v['max_abs_diff']:.3e} <= {G1_LF_TOL:g}; exact "
                      f"{v['n_exact']}/{v['n_joined']} predictions "
                      f"(bit-identity {'also holds' if v['pass'] else 'not required'}).")
        else:
            md.append(f"- G1 {arm}: {'PASS' if v['pass'] else 'FAIL'} — exact "
                      f"{v['n_exact']}/{v['n_joined']} predictions, max|diff| "
                      f"{v['max_abs_diff']:.3e}."
                      + (f" DEVIATION RECORDED: {args.record_g1_deviation}"
                         if (not v['pass'] and args.record_g1_deviation) else ""))
    for arm, info in G1_FAILED.items():
        v = g1f[arm]
        md.append(f"- G1 {arm}: FAIL — **ARM EXITED** (v1.4 CPU rule, no "
                  f"deviation path). Official box G1 ({info['g1_json']}): "
                  f"exact {v['n_exact']}/{v['n_joined']}, max|diff| "
                  f"{v['max_abs_diff']:.3e}. No share, no tests; full "
                  "diagnostic chain in Disclosures.")
    for arm in NOT_EXECUTED:
        md.append(f"- G1 {arm}: not-executed — arm excluded pre-statistic "
                  "(see Not-executed arms).")
    if CHANNEL == "lf":
        md.append("- Triangulation source (E-lf document-swap): "
                  + ("gates G1-G3 verified PASS via swap_longform_meta.json"
                     if elf_gates_ok else
                     f"retention column n/a — {elf_gate_note}")
                  + "; cells absent from swap_longform.csv also render n/a.")
    audit_name = ("mask_audit_sample.md" if CHANNEL == "ed"
                  else "mask_audit_sample_lf.md")
    if g2 is not None:
        md += [
            f"- G2 masking: {g2['n_docs']} docs, "
            f"{g2['docs_with_any_mask_pct']:.1f}% "
            f"with >=1 mask, mean masked-char fraction "
            f"{g2['mean_mask_char_frac']*100:.2f}%; leak rates: own-ticker "
            f"{g2['leak_own_ticker_pct']:.2f}%, own-name-token "
            f"{g2['leak_own_name_token_pct']:.2f}% (audit sample: "
            f"results/anon/{audit_name}).",
        ]
    else:
        md += [
            "- G2 masking: n/a — no executed arm in this channel, so no "
            "masked statistic to gate (the mask build evidence, when it "
            "exists, is an engineering artefact only here).",
        ]
    if CHANNEL == "ed":
        md.append(
            f"- G3 truncation: masked prompt_chars median "
            f"{g3['masked']['prompt_chars_median']:.0f} vs committed "
            f"{g3['committed']['prompt_chars_median']:.0f}; parse_ok "
            f"{g3['masked']['parse_ok_pct']:.1f}% vs "
            f"{g3['committed']['parse_ok_pct']:.1f}%.")
    elif g3 is None:
        md.append(
            "- G3 (LF form): not-executed — no GPU arm in this channel "
            "(B2-only; C5-lf excluded pre-statistic), so there is no frozen "
            "artefact whose hash invariance could be gated; no C6 arm -> no "
            "truncation stats either.")
    else:
        md.append(
            f"- G3 (LF form): frozen-artefact hash invariance through the "
            f"masked C5 pass — {'PASS' if g3.get('pass') else 'FAIL'} "
            f"({len(g3.get('post', {}))} artefacts incl. the original "
            f"embedding cache re-hashed unchanged; no C6 arm -> no truncation "
            f"stats in this channel).")
    swap_col = "swap retention" if CHANNEL == "ed" else "E-lf doc-swap retention"
    md += [
        "",
        "## Table — masked vs unmasked M1 increment and identity share",
        "",
        "| arm | h | vs HAR: unmask rel% | masked rel% | DM | Holm p | share | "
        "vs HAR+firmID: unmask rel% | masked rel% | DM | Holm p | share | "
        f"interval bound | {swap_col} |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, r in df.iterrows():
        if "status" in df.columns and r["status"] == "g1-fail":
            lead = (f"g1-fail: exact-match {r.g1_exact_match_rate:.4f}, "
                    f"max|diff| {r.g1_max_abs_diff:.3e} ({r.g1_json})")
            md.append(f"| {r.arm} | — | {lead} | "
                      + " | ".join(["—"] * 11) + " |")
            continue
        if "status" in df.columns and r["status"] == "not-executed":
            md.append(f"| {r.arm} | {int(r.h)} | "
                      + " | ".join(["n/a (not-executed)"] + ["—"] * 11) + " |")
            continue
        bound = (f"{r.refinterval_bound:+.2f}"
                 if np.isfinite(r.refinterval_bound) else "n/a")
        sret = (f"{r.swap_retention:+.2f}"
                if np.isfinite(r.swap_retention) else "n/a")
        md.append(f"| {r.arm} | {int(r.h)} | {cell(r, 'har')} | "
                  f"{cell(r, 'firm')} | {bound} | {sret} |")
    if NOT_EXECUTED:
        md += ["", "## Not-executed arms", ""]
        for arm, info in NOT_EXECUTED.items():
            md.append(f"- **{arm}** ({info['model']}): {info['reason']}")
    branch_hdr = ("## Pre-registered branches (per arm)" if CHANNEL == "ed" else
                  "## Pre-registered branches (per arm; stretch-channel readout "
                  "— the ED table carries the binding adjudication)")
    md += ["", branch_hdr, ""]
    for arm, v in branch_by_arm.items():
        md.append(f"- **{arm}**: {v['branch']} "
                  f"(Holm-significantly-positive masked cells: "
                  f"{v['n_sig_har']}/3 vs HAR, "
                  f"{v['n_sig_firm']}/3 vs firm-ID).")
    for arm in G1_FAILED:
        md.append(f"- **{arm}**: g1-fail — arm exited at G1; contributes no "
                  "defined share cell and no test (see Gates/Disclosures).")
    for arm in NOT_EXECUTED:
        md.append(f"- **{arm}**: not-executed (see Not-executed arms).")
    if not ARMS:
        md.append("- **channel adjudication: undefined** — no executed arm "
                  "and hence no defined share cell in this channel (all arms "
                  "g1-failed or not-executed); the registered branch "
                  "conditions have no domain to act on.")
    swap_label = ("matched-swap retention" if CHANNEL == "ed"
                  else "E-lf document-level swap retention")
    if ARMS:
        agg_lines = [
            f"- identity share^anon, median across the {n_cells} HAR cells: "
            f"**{agg['median_share_har_all']:+.2f}** "
            f"(per arm: {agg['median_share_har_by_arm']}).",
            f"- median across the {n_cells} firm-ID cells: "
            f"{agg['median_share_firm_all']:+.2f}.",
            f"- triangulation: reference-interval bound median "
            f"{agg['median_refinterval_bound']:+.2f}; {swap_label} "
            f"median {agg['median_swap_retention']:+.2f} (share and "
            "1-retention should bracket/agree if the interval logic is "
            "sound).",
        ]
    else:
        agg_lines = [
            "- none: no executed arm in this channel — no share cell, no "
            "aggregate point estimate (see Gates and Not-executed arms).",
        ]
    md += [
        "",
        "## Aggregate point estimates",
        "",
        *agg_lines,
        "",
        "## Disclosures",
        "",
        "- Branch adjudication follows the prereg-ea v1.4 quantified "
        "conditions (registered BEFORE either channel's table fired; shared "
        "by both channels): (a) <=> defined-cell share median >= 0.75 AND "
        "0/3 masked HAR cells Holm-significantly positive; (b) <=> median "
        "<= 0.50 AND masked increment still absorbed by firmID (<=1/3 firmID "
        "cells Holm-significantly positive); (c) <=> otherwise (incl. median "
        "in (0.50,0.75) or >=2/3 firmID cells Holm-significantly positive = "
        "absorption broken). 'Holm-significantly positive' = masked rel% > 0 "
        "AND Holm p < .05. The registered quantities themselves (per-cell "
        "shares, Holm p) are reported unconditionally above.",
        "- share is undefined (n/a) where the unmasked increment is <= 0 — "
        "no clipping, no exclusion from the table.",
    ]
    for arm, info in G1_FAILED.items():
        md.append(f"- **{arm} ({info['model']}) EXITED at G1**: "
                  f"{info['reason']}")
        if info.get("diagnostics"):
            parts = []
            for label, fn in info["diagnostics"].items():
                d = json.loads((ANON / fn).read_text())
                parts.append(f"{label}: exact {d['n_exact']}/{d['n_joined']}, "
                             f"max|diff| {d['max_abs_diff']:.3e} [{fn}]")
            md.append(f"- {arm} G1 reproduction attempts, side by side — "
                      + "; ".join(parts)
                      + ". Vocabulary drift in one sentence: the box-control "
                        "and local-control model.pkl fits share identical CV "
                        "alphas yet hold DIFFERENT vocabulary sets with idf "
                        "max|diff| 7.5 — text-store lineage drift, not a "
                        "numeric-library effect. The same-lineage box "
                        "ctrl/masked pair lives in the NON-REGISTERED annex "
                        "(anon_annex_samelineage.{csv,md}), outside this "
                        "table and outside branch adjudication.")
    if CHANNEL == "ed":
        md += [
            "- Scope (registered): event_driven only; long_form is a stretch "
            "goal, scored separately (anon_arm_lf.{csv,md}, --channel lf); "
            "C5x not run (GPU budget), pre-registered as not-done.",
        ]
    else:
        md += [
            "- Channel: the registered STRETCH (\"long-form as stretch, only "
            "when ED completes\"), trimmed to B2-ONLY before any LF statistic; the "
            "event_driven table (anon_arm.{csv,md}) is the registered "
            "primary and carries the branch adjudication that binds the "
            "paper's wording — this table triangulates it on the long-form "
            "panel.",
            "- Registered executed arm: B2-lf ONLY (masked RETRAIN under the "
            "fixed committed recipe; CPU classical model — no neural text "
            "model in this channel). It EXITED at G1 under the v1.4 CPU "
            "rule (see Gates + the b2 disclosure above), so the channel "
            "closes with ZERO executed arms: this table documents the exit "
            "and the pre-statistic exclusions; it reports no masked "
            "statistic.",
            "- LF degeneracy (v1.4, registered): B2-lf's UNMASKED increment "
            "is already fully absorbed by the firm-identity reference — "
            "committed firm_identity_control.csv long_form B2 "
            "rel_impr_pct_firm = -0.615/-3.892/-8.089% (h=5/10/20), all "
            "non-positive — so the firmID-side shares are constructively "
            "n/a before execution, the share median draws on the 3 HAR "
            "cells only, and branch (b)'s absorption clause is expected-true "
            "and weakly discriminating in this channel; the paper wording "
            "is softened accordingly.",
            "- C5-lf NOT run (prereg-ea v1.4, decided before any LF "
            "statistic; rows above marked not-executed). HAR side: "
            "constructively n/a — primary citation committed m1_multiseed.csv "
            "long_form C5_qwen3 seed-2026 rel_impr_pct = "
            "-1.0347/-3.1346/-6.6467 (the same single-seed basis this scorer "
            "uses); deployable_combiner FIXED mean rel% -0.85/-2.48/-5.97 "
            "corroborates (3-seed basis); all negative, so every HAR share "
            "cell is empty before execution. firmID side: no committed table "
            "carries that increment, so constructive n/a is NOT invoked — "
            "excluded on GPU budget/scope (as C6-lf), and its execution "
            "could not change the branch adjudication (the share median "
            "takes HAR cells only).",
            "- C2-lf NOT run (decided before any LF statistic): two "
            "fixed-recipe FinBERT trainings (~20-30 GPU-h) for marginal "
            "defensive value; the ED C2 arm already covers the fine-tuned "
            "lineage, and the E-lf C2 arm is registered artefact-lost.",
            "- C6-lf NOT run: the committed UNMASKED C6 long-form run exists "
            "(genuine LLM output over all 11,907 long-form val+test filings), "
            "so a masked C6-lf arm is well-defined; it was excluded on GPU "
            "budget/scope (another Qwen3-32B bf16 TP=2 block), NOT because "
            "the baseline is missing. The LF channel therefore lacks the "
            "prompted-LLM lineage.",
            "- Triangulation column (v1.4 operationalisation): the E-lf B2 "
            "arm's same-horizon DOCUMENT-swap retention from committed "
            "swap_longform.csv, reported ONLY where the cell exists AND the "
            "E-lf gates G1-G3 are verified PASS in swap_longform_meta.json "
            "(missing status counts as not passed -> n/a, reason in the "
            "Gates section). The firm-ref swap variant exists only for the "
            "ED forecast-level swap, hence n/a here.",
        ]
    md += [
        "- Single-shot: this file is written once; re-running the script "
        "refuses while it exists.",
    ]
    FINAL_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {FINAL_CSV} ({len(df)} rows) and {FINAL_MD}")
    print(json.dumps(agg, indent=2))
    for arm, v in branch_by_arm.items():
        print(f"BRANCH {arm}: {v['branch']}")


if __name__ == "__main__":
    main()
