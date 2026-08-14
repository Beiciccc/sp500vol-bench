"""hpo_test_eval.py — the ONE-SHOT pre-registered test evaluation of the HPO
tuned arm (protocol: configs/pretest_evaluation_protocol.md, tag hpo-pretest-v1.0).

Implements §3 of the protocol EXACTLY, for the two in-scope tasks
    T1a (FinBERT-S1, long_form)   and   T1c (FinBERT-S1, event_driven):

  (a) tuned_standalone : vol-unit QLIKE of the tuned 3-seed ensemble vs the
      val-recalibrated A2 HAR (log_recal val-fit, test-frozen — the fc code
      path), per horizon, day-clustered DM (HAC lag h-1, HLN).
  (b) tuned_cascade(M1): log_combo (val-fit, test-frozen) increment of the
      tuned text arm over the single recalibrated HAR — the verbatim M1 block
      of scripts/analysis/crossfamily_llama70.py / m1_clustered.py.
  Holm(6) WITHIN each pre-declared family (2 tasks x 3 horizons).
  Placebo gate (configs/hpo_arm.yaml placebo_gate rule): any cell entering a
  "win" statement (DM<0 & Holm<.05 in either family) must pass BOTH the
  label-shuffle placebo and the within-date placebo; failure -> artefact.

SINGLE-SHOT DISCIPLINE (§5): this script refuses to run a second time — if
results/tables/hpo_test_eval.csv (or .md) already exists it aborts. The
override flag --i-know-this-violates-prereg forces a rerun but appends a
RERUN DISCLOSURE block to the md (§5 requires the bug-fix diff + reason to be
recorded there).

Code provenance (stated per pre-implementation review):
  * standalone recalibration : mirrors log_recal/apply_recal of
    scripts/experiments/second_domain/yelp_entity_disjoint.py (WITHOUT its
    domain clip) — this is byte-for-byte the fR branch of fc.log_combo, and
    every cell hard-asserts equality against fc.log_combo's fR.
  * M1 block                 : verbatim pattern of
    scripts/analysis/crossfamily_llama70.py (fc.log_combo + fc.qlike +
    clustered_dm.dm_test_clustered on effective_trading_day).
  * label-shuffle placebo    : the committed gate mechanism of
    scripts/analysis/row3_tuned_m1.py / m1_ensemble_primary.py — whole-sample
    rng.permutation of the text arm on val AND test, seeds fc.PLACEBO_SEEDS
    (1000-1004), day-clustered DM; committed pass threshold |mean DM| < 2.
  * within-date placebo      : scripts/analysis/withindate_placebo.py
    permute_within_day (imported, not copied), same seeds, clustered DM.
  * standalone placebo NOTE  : the committed |DM|<2 two-sided rule is defined
    for INCREMENT-form comparisons, where destroying the text information
    drives DM -> 0. In the standalone form, destroying the tuned forecast
    drives DM -> +large BY CONSTRUCTION (a shuffled forecast is garbage), so
    the two-sided rule would mechanically fail every standalone win. The gate
    is therefore applied one-sided for the standalone family: PASS unless the
    placebo REPRODUCES the win (mean placebo DM < -2). Disclosed in the md.

Validation modes (neither touches results/tables/hpo_test_eval.*):
  --selftest     : fabricates a tiny synthetic panel IN MEMORY (3 seeds,
                   2 horizons, ~40 rows per split, planted increments), runs
                   the full pipeline end-to-end into a temp dir, asserts the
                   planted signs are recovered and the Holm/placebo paths
                   execute. Reads/writes NOTHING under results/.
  --check-inputs : existence check of the 6 frozen input files (+ winners +
                   single-shot guard). Loads no test row into any statistic.

Run (the one shot, from repo root):
    .venv/bin/python scripts/analysis/hpo_test_eval.py
"""
from __future__ import annotations

# --- thread caps BEFORE numpy/pandas are imported (<=4 cores discipline) ----
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)  # fc.load & friends use repo-root-relative paths
sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import forecast_combination as fc  # noqa: E402
from clustered_dm import dm_test_clustered  # noqa: E402
from withindate_placebo import day_key, permute_within_day  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants (protocol §1-§3)
# ---------------------------------------------------------------------------
TASKS = {"T1a": "long_form", "T1c": "event_driven"}
SEEDS = (2026, 2027, 2028)          # final_seeds, hpo_arm.yaml
RUNG = 2                            # winner checkpoints = rung 2
HORIZONS = fc.HORIZONS              # (5, 10, 20)
EPS = fc.EPS                        # 1e-8
SORT = fc.SORT                      # deterministic obs order for permutations
MERGE_KEY = ["accession", "horizon_days"]   # predictions_fulltest has NO ticker
PLACEBO_SEEDS = fc.PLACEBO_SEEDS    # (1000..1004) — committed seed stream
PLACEBO_T = 2.0                     # committed threshold (fc genuine flag / row3)
HPO_ROOT = Path("results/hpo")
OUT_CSV = Path("results/tables/hpo_test_eval.csv")
OUT_MD = Path("results/tables/hpo_test_eval.md")

FAM_STANDALONE = "tuned_standalone"
FAM_CASCADE = "tuned_cascade"


def _fatal(msg: str) -> None:
    raise SystemExit(f"[hpo_test_eval] FATAL: {msg}")


# ---------------------------------------------------------------------------
# Recalibration — mirrors scripts/experiments/second_domain/yelp_entity_disjoint.py
# log_recal/apply_recal (WITHOUT its domain clip: exp() already guarantees
# positivity and fc.log_combo's fR applies no clip). Identical by construction
# to the fR branch of fc.log_combo; asserted per cell in _eval_cell.
# ---------------------------------------------------------------------------
def log_recal(yv, fv):
    ly = np.log(np.clip(yv, EPS, None))
    lf = np.log(np.clip(fv, EPS, None))
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(lf)), lf]), ly, rcond=None)
    return float(beta[0]), float(beta[1])


def apply_recal(a, b, f):
    return np.exp(a + b * np.log(np.clip(np.asarray(f, float), EPS, None)))


def qlike_var(y, f):
    """QLIKE in VARIANCE units, q(y^2, f^2) — report-only column (protocol §3)."""
    return fc.qlike(np.asarray(y, float) ** 2, np.asarray(f, float) ** 2)


# ---------------------------------------------------------------------------
# Seed ensemble — protocol §1: per-observation LOG-space mean,
#     exp(mean_seeds(log(clip(pred, 1e-8)))).
# This mirrors the HPO SELECTION rule (results/hpo/T1a/seed_validation.json
# "rule": lower 3-seed-ensemble val_select QLIKE) so the tested object is the
# selected object. NOTE — DELIBERATE DIFFERENCE, disclosed per protocol §1:
# the paper's C-model primary tables use the ARITHMETIC per-observation mean
# (m1_ensemble_primary.ensemble_text). The two conventions are each internally
# consistent and are never mixed; the md discloses both side by side.
# ---------------------------------------------------------------------------
def log_ensemble(frame: pd.DataFrame) -> np.ndarray:
    cols = [f"f{s}" for s in SEEDS]
    logs = np.log(np.clip(frame[cols].to_numpy(dtype=float), EPS, None))
    return np.exp(logs.mean(axis=1))


# ---------------------------------------------------------------------------
# Input resolution & hard asserts (protocol §2)
# ---------------------------------------------------------------------------
def rid_of(task: str, trial: int, seed: int) -> str:
    """Mirror of predict_winner_test._rid."""
    rid = f"{task}_trial{trial:03d}_rung{RUNG}"
    if seed != 2026:
        rid += f"_s{seed}"
    return rid


def winner_trial(task: str):
    p = HPO_ROOT / task / "seed_validation.json"
    if not p.exists():
        return None, p
    return int(json.loads(p.read_text())["winner_trial"]), p


def input_paths():
    """Resolve the 6 frozen input files. Returns (paths, missing_seedval)."""
    paths, missing_sv = {}, []
    for task in TASKS:
        w, sv = winner_trial(task)
        if w is None:
            missing_sv.append(str(sv))
            continue
        for seed in SEEDS:
            rd = HPO_ROOT / task / rid_of(task, w, seed)
            paths[(task, seed)] = (w, rd / "predictions_fulltest.parquet", rd / "summary.json")
    return paths, missing_sv


def assemble_task(task, disc, seed_frames, a2, summaries):
    """All §2 hard asserts + the A2 merge for one task.

    seed_frames: {seed: DataFrame[accession, horizon_days, split,
                  label_realised_vol, prediction_realised_vol]}
    a2:          DataFrame[split, ticker, accession, horizon_days, fhar,
                  label_realised_vol, filing_time_utc, effective_trading_day]
    summaries:   {seed: n_test_dropped or None}  (summary.json cross-check)

    Returns (merged panel, sanity dict).
    """
    keysets = {}
    for seed, df in seed_frames.items():
        splits = set(df["split"].unique())
        if splits != {"train", "val", "test"}:
            _fatal(f"{task} seed {seed}: splits {sorted(splits)} != train/val/test")
        keys = pd.MultiIndex.from_frame(df[MERGE_KEY])
        if keys.has_duplicates:
            _fatal(f"{task} seed {seed}: duplicate (accession, horizon_days) keys")
        keysets[seed] = frozenset(keys)
    base = keysets[SEEDS[0]]
    for seed in SEEDS[1:]:
        if keysets[seed] != base:
            d1 = len(base - keysets[seed]); d2 = len(keysets[seed] - base)
            _fatal(f"{task}: key set of seed {seed} differs from seed {SEEDS[0]} "
                   f"({d1} missing / {d2} extra)")

    # summary.json n_test_dropped cross-check (protocol §4-iv sanity)
    n_test_parquet = int((seed_frames[SEEDS[0]]["split"] == "test").sum())
    checked = {}
    for seed in SEEDS:
        n_seed = int((seed_frames[seed]["split"] == "test").sum())
        if n_seed != n_test_parquet:
            _fatal(f"{task} seed {seed}: n_test {n_seed} != seed {SEEDS[0]} {n_test_parquet}")
        exp = summaries.get(seed)
        if exp is not None:
            checked[seed] = int(exp)
            if int(exp) != n_test_parquet:
                _fatal(f"{task} seed {seed}: parquet n_test {n_test_parquet} != "
                       f"summary.json n_test_dropped {exp}")
    if not checked:
        _fatal(f"{task}: no summary.json n_test_dropped available for cross-check")

    # wide frame: one column per seed, 1:1 across seeds by construction
    wide = None
    for seed in SEEDS:
        f = seed_frames[seed].rename(columns={"prediction_realised_vol": f"f{seed}"})
        cols = MERGE_KEY + ([f"f{seed}", "split", "label_realised_vol"]
                            if wide is None else [f"f{seed}", "label_realised_vol"])
        f = f[cols].rename(columns={"label_realised_vol": f"y{seed}"})
        wide = f if wide is None else wide.merge(f, on=MERGE_KEY, validate="one_to_one")
    for seed in SEEDS[1:]:
        if not np.allclose(wide[f"y{SEEDS[0]}"], wide[f"y{seed}"], rtol=1e-9, atol=0):
            _fatal(f"{task}: labels differ between seed {SEEDS[0]} and seed {seed}")
    wide = wide.rename(columns={f"y{SEEDS[0]}": "label_hpo"}).drop(
        columns=[f"y{s}" for s in SEEDS[1:]])

    # A2 merge on [accession, horizon_days] (fulltest parquet has NO ticker):
    # ticker / effective_trading_day / filing_time_utc come from the A2 side.
    if pd.MultiIndex.from_frame(a2[MERGE_KEY]).has_duplicates:
        _fatal(f"{task}: A2 panel has duplicate (accession, horizon_days) keys")
    n_val_hpo = int((wide["split"] == "val").sum())
    m = wide.merge(a2, on=MERGE_KEY, how="inner", validate="one_to_one",
                   suffixes=("", "_a2"))
    n_test_m = int((m["split"] == "test").sum())
    n_val_m = int((m["split"] == "val").sum())
    if n_test_m != n_test_parquet:
        _fatal(f"{task}: A2 merge lost test rows ({n_test_parquet} -> {n_test_m})")
    if n_val_m != n_val_hpo:  # val rows fit every frozen weight — loss would corrupt them
        _fatal(f"{task}: A2 merge lost val rows ({n_val_hpo} -> {n_val_m})")
    if (m["split"] != m["split_a2"]).any():
        _fatal(f"{task}: split labels disagree between fulltest parquet and A2")
    if not np.allclose(m["label_hpo"], m["label_realised_vol"], rtol=1e-6):
        _fatal(f"{task}: labels disagree between fulltest parquet and A2")
    n_train_lost = int((wide["split"] == "train").sum()) - int((m["split"] == "train").sum())

    sanity = {"task": task, "disc": disc, "n_test": n_test_parquet,
              "n_val": n_val_hpo, "keyset_equal": True, "merge_one_to_one": True,
              "test_rows_lost": 0, "val_rows_lost": 0,
              "train_rows_lost_info": n_train_lost,
              "summary_n_test_dropped": checked}
    return m.drop(columns=["split_a2", "label_hpo"]), sanity


def load_real_inputs():
    """Disk loader for the real run. HARD ASSERTS (§2) before anything else:
    all 6 files exist -> then load -> per-task asserts in assemble_task."""
    paths, missing_sv = input_paths()
    if missing_sv:
        _fatal("winner not resolvable — missing: " + ", ".join(missing_sv))
    missing = [str(p) for (_t, _s), (_w, p, _sm) in sorted(paths.items()) if not p.exists()]
    if missing:
        _fatal("frozen inputs incomplete (protocol §2 requires all 6):\n  "
               + "\n  ".join(missing))

    panels, sanities, winners = {}, [], {}
    for task, disc in TASKS.items():
        seed_frames, summaries = {}, {}
        for seed in SEEDS:
            w, pq_path, sm_path = paths[(task, seed)]
            winners[task] = w
            seed_frames[seed] = pd.read_parquet(pq_path)
            summaries[seed] = (json.loads(sm_path.read_text()).get("n_test_dropped")
                               if sm_path.exists() else None)
        a2 = fc.load("A2_har_rv", disc)[
            ["split", "ticker", "accession", "horizon_days", "prediction_realised_vol",
             "label_realised_vol", "filing_time_utc", "effective_trading_day"]
        ].rename(columns={"prediction_realised_vol": "fhar"})
        panel, sanity = assemble_task(task, disc, seed_frames, a2, summaries)
        hs = sorted(panel["horizon_days"].astype(int).unique().tolist())
        if hs != sorted(HORIZONS):
            _fatal(f"{task}: horizons {hs} != {sorted(HORIZONS)}")
        panels[task] = {"disc": disc, "df": panel}
        sanities.append(sanity)
    return panels, sanities, winners


# ---------------------------------------------------------------------------
# §3 statistics
# ---------------------------------------------------------------------------
def _eval_cell(task, disc, v, te, h):
    """One (task, horizon): standalone row + cascade row + placebo context."""
    yv, fhv, ftv = v.label_realised_vol.to_numpy(), v.fhar.to_numpy(), v.fens.to_numpy()
    yt, fhr, ftt = te.label_realised_vol.to_numpy(), te.fhar.to_numpy(), te.fens.to_numpy()
    days_t = te.effective_trading_day.values      # verbatim crossfamily_llama70 M1 block
    days_v_key, days_t_key = day_key(v), day_key(te)

    # (a) standalone: tuned ensemble vs val-recalibrated A2 (log_recal, test-frozen)
    a, b = log_recal(yv, fhv)
    fA2r = apply_recal(a, b, fhr)
    # ---- verbatim M1 block (crossfamily_llama70.py) -------------------------
    fR, fU, g = fc.log_combo(yv, fhv, ftv, fhr, ftt)
    # consistency: our recal IS fc.log_combo's fR code path (protocol §3a "the current fc function")
    if not np.allclose(fA2r, fR, rtol=1e-10, atol=0):
        _fatal(f"{task} h={h}: log_recal drifted from fc.log_combo fR")
    lS, lA2 = fc.qlike(yt, ftt), fc.qlike(yt, fA2r)
    dm_s, p_s, n_days = dm_test_clustered(lS, lA2, days_t, h)
    rel_s = 100.0 * float(lA2.mean() - lS.mean()) / float(lA2.mean())
    vS, vA2 = qlike_var(yt, ftt), qlike_var(yt, fA2r)          # report-only
    # (b) M1 cascade increment vs the single recalibrated HAR
    lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
    rel_c = 100.0 * float(lR.mean() - lU.mean()) / float(lR.mean())
    dm_c, p_c, _ = dm_test_clustered(lU, lR, days_t, h)
    vR, vU = qlike_var(yt, fR), qlike_var(yt, fU)              # report-only

    common = dict(task=task, disc=disc, h=int(h), n_test=len(te), n_days=int(n_days))
    row_s = dict(common, family=FAM_STANDALONE,
                 qlike_ref_vol=float(lA2.mean()), qlike_arm_vol=float(lS.mean()),
                 rel_pct_vol=rel_s, dm_clu=float(dm_s), p_clu=float(p_s),
                 qlike_ref_var=float(vA2.mean()), qlike_arm_var=float(vS.mean()),
                 rel_pct_var=100.0 * float(vA2.mean() - vS.mean()) / float(vA2.mean()),
                 g_log=float("nan"), recal_a=a, recal_b=b)
    row_c = dict(common, family=FAM_CASCADE,
                 qlike_ref_vol=float(lR.mean()), qlike_arm_vol=float(lU.mean()),
                 rel_pct_vol=rel_c, dm_clu=float(dm_c), p_clu=float(p_c),
                 qlike_ref_var=float(vR.mean()), qlike_arm_var=float(vU.mean()),
                 rel_pct_var=100.0 * float(vR.mean() - vU.mean()) / float(vR.mean()),
                 g_log=float(g), recal_a=a, recal_b=b)
    ctx = dict(yv=yv, fhv=fhv, ftv=ftv, yt=yt, fhr=fhr, ftt=ftt, fA2r=fA2r,
               days_t=days_t, days_v_key=days_v_key, days_t_key=days_t_key, h=h)
    return row_s, row_c, ctx


def placebo_cascade(ctx):
    """Committed gate mechanisms for the INCREMENT form.
    label-shuffle: row3_tuned_m1.py block (whole-sample permutation, val+test);
    within-date : withindate_placebo.py block (permute_within_day, val+test)."""
    ls, wd = [], []
    for s in PLACEBO_SEEDS:
        rng = np.random.default_rng(s)
        pv, pt = rng.permutation(ctx["ftv"]), rng.permutation(ctx["ftt"])
        pR, pU, _ = fc.log_combo(ctx["yv"], ctx["fhv"], pv, ctx["fhr"], pt)
        st, _, _ = dm_test_clustered(fc.qlike(ctx["yt"], pU), fc.qlike(ctx["yt"], pR),
                                     ctx["days_t"], ctx["h"])
        ls.append(st)
        rng = np.random.default_rng(s)  # same per-seed stream as withindate_placebo.main
        pv = permute_within_day(ctx["ftv"], ctx["days_v_key"], rng)
        pt = permute_within_day(ctx["ftt"], ctx["days_t_key"], rng)
        pR, pU, _ = fc.log_combo(ctx["yv"], ctx["fhv"], pv, ctx["fhr"], pt)
        st, _, _ = dm_test_clustered(fc.qlike(ctx["yt"], pU), fc.qlike(ctx["yt"], pR),
                                     ctx["days_t"], ctx["h"])
        wd.append(st)
    ls_m, wd_m = float(np.mean(ls)), float(np.mean(wd))
    return ls_m, wd_m, bool(abs(ls_m) < PLACEBO_T), bool(abs(wd_m) < PLACEBO_T)


def placebo_standalone(ctx):
    """Same two shuffle mechanisms applied to the STANDALONE comparison
    (shuffled tuned forecast vs the frozen recalibrated A2). One-sided pass
    rule (see module docstring): FAIL only if the placebo reproduces the win."""
    ls, wd = [], []
    for s in PLACEBO_SEEDS:
        rng = np.random.default_rng(s)
        pt = rng.permutation(ctx["ftt"])
        st, _, _ = dm_test_clustered(fc.qlike(ctx["yt"], pt),
                                     fc.qlike(ctx["yt"], ctx["fA2r"]),
                                     ctx["days_t"], ctx["h"])
        ls.append(st)
        rng = np.random.default_rng(s)
        pt = permute_within_day(ctx["ftt"], ctx["days_t_key"], rng)
        st, _, _ = dm_test_clustered(fc.qlike(ctx["yt"], pt),
                                     fc.qlike(ctx["yt"], ctx["fA2r"]),
                                     ctx["days_t"], ctx["h"])
        wd.append(st)
    ls_m, wd_m = float(np.mean(ls)), float(np.mean(wd))
    return ls_m, wd_m, bool(ls_m > -PLACEBO_T), bool(wd_m > -PLACEBO_T)


def evaluate(panels, horizons):
    """Full §3 pass. Returns the long-format cell table."""
    rows, ctxs = [], {}
    for task, P in panels.items():
        d = P["df"].copy()
        d["fens"] = log_ensemble(d)  # protocol §1 selection-rule ensemble (see helper)
        for h in horizons:
            v = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
            te = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
            if len(v) < 2 or len(te) < 2:
                _fatal(f"{task} h={h}: empty val/test slice")
            row_s, row_c, ctx = _eval_cell(task, P["disc"], v, te, h)
            rows += [row_s, row_c]
            ctxs[(task, int(h))] = ctx
    df = pd.DataFrame(rows)

    # Holm WITHIN each pre-declared family (protocol §3: two Holm(6) families)
    df["p_holm"] = np.nan
    for fam in (FAM_STANDALONE, FAM_CASCADE):
        m = df.family == fam
        df.loc[m, "p_holm"] = fc.holm(df.loc[m, "p_clu"].fillna(1.0).values)

    # placebo gate: every cell with DM<0 & Holm<.05 in EITHER family
    for c in ("placebo_ls_dm", "placebo_wd_dm"):
        df[c] = np.nan
    df["placebo_ls_pass"] = pd.array([None] * len(df), dtype="boolean")
    df["placebo_wd_pass"] = pd.array([None] * len(df), dtype="boolean")
    df["flagged"] = (df.dm_clu < 0) & (df.p_holm < 0.05)
    for i, r in df[df.flagged].iterrows():
        ctx = ctxs[(r.task, int(r.h))]
        fn = placebo_cascade if r.family == FAM_CASCADE else placebo_standalone
        ls_m, wd_m, ls_ok, wd_ok = fn(ctx)
        df.loc[i, ["placebo_ls_dm", "placebo_wd_dm"]] = [ls_m, wd_m]
        df.loc[i, "placebo_ls_pass"] = ls_ok
        df.loc[i, "placebo_wd_pass"] = wd_ok
    df["gate_pass"] = df.flagged & (df.placebo_ls_pass == True) & (df.placebo_wd_pass == True)  # noqa: E712
    return df


def branch_of(df):
    """§5 rewrite branch. B1: standalone still loses AND no newly genuine M1
    cell. B2: >=1 newly genuine (Holm+both placebos) cascade cell. B3: mixed."""
    std_win = df[(df.family == FAM_STANDALONE) & df.gate_pass]
    cas_gen = df[(df.family == FAM_CASCADE) & df.gate_pass]
    if len(cas_gen):
        return "B2", std_win, cas_gen
    if not len(std_win):
        return "B1", std_win, cas_gen
    return "B3", std_win, cas_gen


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
DISCLOSURES = [
    ("(i) Scope trimming", "amendment_1's T6d3/T6c5, Track-B, T4/T3/T2/T1d and the ~51-cell Holm family were not executed"
     " (post-R8 ROI ruling). Consequence: the paper's rewrite-sentence scope narrows to \"an ASHA-tuned FinBERT (both channels)\","
     " and the full phrase \"validation-tuned challengers\" must not be written. This table's Holm family is accordingly 2 tasks × 3 horizons = 6."),
    ("(ii) s_strategy_recheck not executed", "S1 was frozen into retraining without a val recheck of S2/S3/S4; if some S* beats S1 on val,"
     " the tuned arm may be underestimated — the direction is conservative for the null (the tuned arm could only be stronger)."),
    ("(iii) Selection convention", "track_a is the pooled-across-horizons val_select argmin (not per-horizon selection);"
     " if the main text mentions it, it must be written as pooled."),
    ("(iv) Code wording", "asha_hpo.py's vol_unit_qlike(pred, \"test\") reads as test; physically it is val-select "
     "borrowing an empty slot (test rows are physically deleted in prepare_data, manifest sha256 on record); predict_winner_test.py "
     "has re-verified that n_test matches summary.json (re-checked in this table's SANITY)."),
    ("(v) pilot touched test", "the yaml pilot_disclosure is transcribed verbatim into the paper appendix (a single-seed lr+early-stopping pilot "
     "touched test before pre-registration, with a null result; the bias-direction argument is in the yaml)."),
]


def _fmt(x, spec="+.3f"):
    return "-" if (x is None or (isinstance(x, float) and np.isnan(x))) else format(x, spec)


def write_outputs(df, branch, std_win, cas_gen, sanities, winners, out_csv, out_md,
                  rerun_note=None):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = [
        "# HPO tuned-arm — the ONE-SHOT pre-registered test evaluation "
        "(hpo-pretest-v1.0)",
        "",
        f"Generated {now} by scripts/analysis/hpo_test_eval.py. "
        "Protocol: configs/pretest_evaluation_protocol.md (§3 statistics, §4 "
        "disclosures, §5 single-shot discipline). This file is written ONCE; "
        "the script refuses a second run.",
        "",
        f"Winners: " + ", ".join(f"{t} = trial {w}" for t, w in sorted(winners.items()))
        + f"; seeds {SEEDS}, rung {RUNG} checkpoints.",
        "",
        "## Ensemble-rule disclosure (protocol §1)",
        "",
        "- Tuned arm here = per-observation **log-space** mean of the 3 seed "
        "forecasts, `exp(mean(log(clip(pred, 1e-8))))` — identical to the HPO "
        "selection rule (seed_validation.json `rule`), so the object tested is "
        "the object selected.",
        "- The paper's C-model primary convention is the **arithmetic** "
        "per-observation mean (m1_ensemble_primary.py). The two conventions are "
        "each internally consistent and are NOT mixed anywhere; do not compare "
        "rows across the two conventions.",
        "",
        "## (a) tuned_standalone — tuned ensemble vs val-recalibrated A2 "
        "(vol-unit decides; Holm(6) within family)",
        "",
        "| task | disc | h | n_test | n_days | QLIKE recal-A2 | QLIKE tuned | rel% "
        "| DM(clu) | p | Holm | var-unit rel% | placebo ls DM | placebo wd DM | gate |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for _, r in df[df.family == FAM_STANDALONE].sort_values(["task", "h"]).iterrows():
        gate = ("PASS" if r.gate_pass else ("ARTEFACT" if r.flagged else "-"))
        md.append(f"| {r.task} | {r.disc} | {int(r.h)} | {int(r.n_test)} | {int(r.n_days)} "
                  f"| {r.qlike_ref_vol:.4f} | {r.qlike_arm_vol:.4f} | {r.rel_pct_vol:+.2f} "
                  f"| {r.dm_clu:+.2f} | {r.p_clu:.4f} | {r.p_holm:.4f} "
                  f"| {r.rel_pct_var:+.2f} | {_fmt(r.placebo_ls_dm, '+.2f')} "
                  f"| {_fmt(r.placebo_wd_dm, '+.2f')} | {gate} |")
    md += [
        "",
        "## (b) tuned_cascade (M1) — log_combo increment over the single "
        "recalibrated HAR (vol-unit decides; Holm(6) within family)",
        "",
        "| task | disc | h | n_test | n_days | QLIKE fR | QLIKE fU | rel% | g_log "
        "| DM(clu) | p | Holm | var-unit rel% | placebo ls DM | placebo wd DM | gate |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for _, r in df[df.family == FAM_CASCADE].sort_values(["task", "h"]).iterrows():
        gate = ("GENUINE" if r.gate_pass else ("ARTEFACT" if r.flagged else "-"))
        md.append(f"| {r.task} | {r.disc} | {int(r.h)} | {int(r.n_test)} | {int(r.n_days)} "
                  f"| {r.qlike_ref_vol:.4f} | {r.qlike_arm_vol:.4f} | {r.rel_pct_vol:+.2f} "
                  f"| {r.g_log:+.3f} | {r.dm_clu:+.2f} | {r.p_clu:.4f} | {r.p_holm:.4f} "
                  f"| {r.rel_pct_var:+.2f} | {_fmt(r.placebo_ls_dm, '+.2f')} "
                  f"| {_fmt(r.placebo_wd_dm, '+.2f')} | {gate} |")
    md += [
        "",
        "Reading guide: DM(clu) < 0 = tuned arm/text-augmented combiner better; "
        "day-clustered DM (daily-mean loss differentials, HAC lag h-1, HLN, over "
        "effective_trading_day). rel% > 0 = tuned arm lowers QLIKE vs the "
        "reference. vol-unit is the DECISION unit; var-unit columns are "
        "report-only (protocol §3). Placebo gate (hpo_arm.yaml `placebo_gate`) "
        "runs on every cell with DM<0 & Holm<.05: label-shuffle (whole-sample "
        "permutation, the committed row3_tuned_m1/m1_ensemble_primary mechanism, "
        "seeds 1000-1004) AND within-date (withindate_placebo.permute_within_day). "
        f"Committed pass threshold |mean clustered placebo DM| < {PLACEBO_T:.0f} for the "
        "increment form; for the standalone form the rule is applied ONE-SIDED "
        f"(fail only if placebo DM < -{PLACEBO_T:.0f}) because destroying a standalone "
        "forecast drives DM to +large by construction — a two-sided rule would "
        "mechanically fail every standalone win.",
        "",
        f"## Branch verdict (§5): **{branch}**",
        "",
    ]
    if branch == "B1":
        md.append("B1 (expected): standalone still loses to the recalibrated HAR and "
                  "the tuned arm creates NO newly genuine M1 cell (Holm + both "
                  "placebos). Rewrite: tuning closes ~74% of the standalone gap on "
                  "val-select, still loses out of sample, and creates no new genuine "
                  "cell — the null is tuning-robust (scope: an ASHA-tuned FinBERT, "
                  "both channels).")
    elif branch == "B2":
        md.append(f"B2: {len(cas_gen)} newly genuine tuned-cascade cell(s) "
                  f"({', '.join(f'{r.task}/h{int(r.h)}' for _, r in cas_gen.iterrows())}). "
                  "Per §5 these enter the main text alongside the committed 69-cell "
                  "table; headline counts and abstract must be re-estimated — no "
                  "hiding, no downplaying.")
    else:
        md.append(f"B3 (mixed): standalone win(s) "
                  f"({', '.join(f'{r.task}/h{int(r.h)}' for _, r in std_win.iterrows())}) "
                  "without a newly genuine cascade cell. Report cell-by-cell; rewrite "
                  "the sentence in its weakest defensible form (§5).")
    md += [
        "",
        "## Disclosures (protocol §4, recorded before evaluation)",
        "",
    ]
    for title, body in DISCLOSURES:
        md.append(f"- **{title}** — {body}")
    md += [
        "",
        "## SANITY",
        "",
    ]
    for s in sanities:
        md.append(f"- **{s['task']}** ({s['disc']}): 3-seed (accession, horizon_days) "
                  f"key sets identical = {s['keyset_equal']}; A2 merge 1:1 = "
                  f"{s['merge_one_to_one']}, test rows lost = {s['test_rows_lost']}, "
                  f"val rows lost = {s['val_rows_lost']} (train rows outside A2 panel: "
                  f"{s['train_rows_lost_info']}, informational); n_test = {s['n_test']} "
                  f"vs summary.json n_test_dropped = "
                  + ", ".join(f"seed{k}: {v}" for k, v in sorted(s["summary_n_test_dropped"].items()))
                  + " — MATCH.")
    md.append("- Per-cell hard assert: the standalone recalibration (log_recal "
              "val-fit, test-frozen; pattern of scripts/experiments/second_domain/"
              "yelp_entity_disjoint.py) reproduces fc.log_combo's fR to rtol 1e-10 "
              "in every cell.")
    if rerun_note:
        md += ["", "## RERUN DISCLOSURE (protocol §5 violation — operator-forced)", "",
               rerun_note,
               "",
               "§5 permits a rerun only to fix a bug in this script; the operator "
               "must record the code diff and the reason here."]
    out_md.write_text("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# --selftest : synthetic end-to-end run. Touches NOTHING under results/.
# ---------------------------------------------------------------------------
def synth_inputs():
    """3 seeds, 2 horizons, ~40 rows per (split, horizon), planted signs:
      T1a: text arm scale-biased  -> standalone LOSES (DM>0); combiner absorbs
           the bias -> cascade increment GENUINE (DM<0, placebo-null).
      T1c: text arm unbiased      -> standalone WINS and cascade GENUINE.
    Cross-sectional signal c is per-observation (dies under within-date
    shuffle); HAR carries the day signal fully -> both placebos ~0."""
    rng = np.random.default_rng(20260715)
    horizons = (5, 10)
    plan = {"T1a": ("long_form", 0.7), "T1c": ("event_driven", 0.0)}  # log-bias
    splits = {"train": ("2015-01-05", 6), "val": ("2020-01-06", 20), "test": ("2023-01-02", 20)}
    obs_per_day = 2
    panels = {}
    for task, (disc, bias) in plan.items():
        seed_frames = {s: [] for s in SEEDS}
        a2_rows = []
        for split, (start, n_days) in splits.items():
            days = pd.bdate_range(start, periods=n_days)
            for h in horizons:
                for di, day in enumerate(days):
                    for oi in range(obs_per_day):
                        acc = f"{task}-{split}-{di:03d}-{oi}"
                        s_d = 0.5 * np.sin(0.7 * di + 0.2 * h)      # day signal
                        c = rng.normal(0.0, 0.3)                    # cross-sectional
                        ly = -3.0 + s_d + c
                        y = float(np.exp(ly))
                        fhar = float(np.exp(-0.5 + 0.6 * (ly - 0.7 * c)
                                            + rng.normal(0.0, 0.05)))
                        lt = ly + rng.normal(0.0, 0.05) + bias      # informative text
                        for s in SEEDS:
                            f_s = float(np.exp(lt + rng.normal(0.0, 0.02)))
                            seed_frames[s].append([acc, h, split, y, f_s])
                        a2_rows.append([split, f"TK{(di * obs_per_day + oi) % 7}", acc, h,
                                        fhar, y, day + pd.Timedelta(hours=15), day])
        seed_dfs = {s: pd.DataFrame(r, columns=["accession", "horizon_days", "split",
                                                "label_realised_vol", "prediction_realised_vol"])
                    for s, r in seed_frames.items()}
        a2 = pd.DataFrame(a2_rows, columns=["split", "ticker", "accession", "horizon_days",
                                            "fhar", "label_realised_vol",
                                            "filing_time_utc", "effective_trading_day"])
        # exercise the identical hard-assert/merge path as the real run
        panel, sanity = assemble_task(task, disc, seed_dfs, a2,
                                      {s: int((seed_dfs[s]["split"] == "test").sum())
                                       for s in SEEDS})
        panels[task] = {"disc": disc, "df": panel, "sanity": sanity}
    return panels, horizons


def run_selftest():
    print("[selftest] fabricating synthetic panel in memory "
          "(3 seeds, 2 horizons, 40 rows per (split, horizon))")
    panels, horizons = synth_inputs()
    sanities = [P.pop("sanity") for P in panels.values()]
    df = evaluate(panels, horizons)
    branch, std_win, cas_gen = branch_of(df)

    tmp = Path(tempfile.mkdtemp(prefix="hpo_test_eval_selftest_"))
    assert "results" not in tmp.parts, "selftest output dir must not be under results/"
    write_outputs(df, branch, std_win, cas_gen, sanities,
                  winners={t: -1 for t in panels}, out_csv=tmp / "hpo_test_eval.csv",
                  out_md=tmp / "hpo_test_eval.md")

    show = df[["task", "family", "h", "n_test", "n_days", "rel_pct_vol", "dm_clu",
               "p_clu", "p_holm", "placebo_ls_dm", "placebo_wd_dm", "gate_pass"]]
    print(show.to_string(index=False,
                         float_format=lambda x: f"{x:+.3f}" if np.isfinite(x) else "nan"))

    # ---- planted-sign assertions -------------------------------------------
    cas = df[df.family == FAM_CASCADE]
    std = df[df.family == FAM_STANDALONE]
    assert (cas.dm_clu < 0).all() and (cas.rel_pct_vol > 0).all(), \
        "planted cascade increment sign not recovered"
    assert (std[std.task == "T1a"].dm_clu > 0).all(), \
        "planted T1a standalone LOSS not recovered (biased arm should lose)"
    t1c_std = std[std.task == "T1c"]
    assert (t1c_std.dm_clu < 0).all() and (t1c_std.p_holm < 0.05).all(), \
        "planted T1c standalone WIN not recovered"
    # ---- Holm / placebo code-path assertions -------------------------------
    assert df.p_holm.notna().all(), "Holm did not run on every cell"
    assert (cas.p_holm < 0.05).any(), "no cascade cell Holm-significant — placebo path untested"
    flagged = df[df.flagged]
    assert len(flagged) >= 3, f"expected >=3 flagged cells, got {len(flagged)}"
    assert flagged.placebo_ls_dm.notna().all() and flagged.placebo_wd_dm.notna().all(), \
        "placebo did not execute on a flagged cell"
    assert (flagged.family == FAM_STANDALONE).any() and (flagged.family == FAM_CASCADE).any(), \
        "both placebo mechanisms (standalone + cascade form) must execute"
    assert bool(df[df.family == FAM_CASCADE].gate_pass.any()), \
        "planted genuine cascade cell failed its placebo gate"
    assert branch == "B2", f"planted branch B2 not recovered (got {branch})"
    assert (tmp / "hpo_test_eval.csv").exists() and (tmp / "hpo_test_eval.md").exists()

    print(f"[selftest] branch fired: {branch} "
          f"({len(cas_gen)} genuine cascade, {len(std_win)} standalone win)")
    print(f"[selftest] outputs written to {tmp} (NOT under results/)")
    print("[selftest] PASS — planted signs recovered; Holm + both placebo "
          "mechanisms + gate + branch + writer all executed")
    return 0


# ---------------------------------------------------------------------------
# --check-inputs : existence only; loads no test row into any statistic
# ---------------------------------------------------------------------------
def run_check_inputs():
    paths, missing_sv = input_paths()
    n_present = 0
    for task, disc in TASKS.items():
        sv = HPO_ROOT / task / "seed_validation.json"
        if str(sv) in missing_sv:
            print(f"[check-inputs] {task} ({disc}): PENDING — {sv} not found "
                  "(winner not yet selected)")
            continue
        w = paths[(task, SEEDS[0])][0]
        print(f"[check-inputs] {task} ({disc}): winner trial {w} ({sv})")
        for seed in SEEDS:
            _, pq_path, sm_path = paths[(task, seed)]
            ok = pq_path.exists()
            n_present += int(ok)
            extra = "" if sm_path.exists() else "  [summary.json MISSING]"
            print(f"  seed {seed}: {pq_path}  {'EXISTS' if ok else 'MISSING'}{extra}")
    guard = OUT_CSV.exists() or OUT_MD.exists()
    print(f"[check-inputs] single-shot guard: results/tables/hpo_test_eval.csv/.md "
          f"{'ALREADY EXIST — the one shot has been fired' if guard else 'absent — evaluation not yet run'}")
    ready = (not missing_sv) and n_present == len(TASKS) * len(SEEDS)
    print(f"STATUS: {'READY' if ready else 'NOT READY'} "
          f"({n_present}/{len(TASKS) * len(SEEDS)} prediction files present"
          f"{'' if not missing_sv else '; winner pending: ' + ', '.join(missing_sv)})")
    return 0 if ready else 2


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="ONE-SHOT pre-registered test evaluation of the HPO tuned arm "
                    "(configs/pretest_evaluation_protocol.md, hpo-pretest-v1.0). "
                    "Runs ONCE; refuses if results/tables/hpo_test_eval.csv exists.")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic in-memory end-to-end validation; reads/writes "
                         "nothing under results/")
    ap.add_argument("--check-inputs", action="store_true",
                    help="existence check of the 6 frozen inputs only; loads no "
                         "test row into any statistic (exit 0 ready / 2 pending)")
    ap.add_argument("--i-know-this-violates-prereg", action="store_true",
                    help="override the single-run refusal (§5). The md gets a "
                         "RERUN DISCLOSURE block; §5 requires the bug-fix diff "
                         "and reason to be recorded there")
    args = ap.parse_args()

    if args.selftest and args.check_inputs:
        _fatal("--selftest and --check-inputs are mutually exclusive")
    if args.selftest:
        return run_selftest()
    if args.check_inputs:
        return run_check_inputs()

    # ---- single-shot refusal (§5) -------------------------------------------
    rerun_note = None
    if OUT_CSV.exists() or OUT_MD.exists():
        if not args.i_know_this_violates_prereg:
            _fatal(f"{OUT_CSV} (and/or .md) already exists. The pre-registered "
                   "protocol (configs/pretest_evaluation_protocol.md §5) forbids a "
                   "second run of this evaluation. If — and only if — you are "
                   "rerunning to fix a bug in THIS script, pass "
                   "--i-know-this-violates-prereg; the rerun will be disclosed in "
                   "the md and §5 requires you to record the diff and reason there.")
        h_csv = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest() if OUT_CSV.exists() else "absent"
        h_md = hashlib.sha256(OUT_MD.read_bytes()).hexdigest() if OUT_MD.exists() else "absent"
        rerun_note = (f"- --i-know-this-violates-prereg supplied at "
                      f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}.\n"
                      f"- Overwritten artifacts: csv sha256={h_csv}; md sha256={h_md}.")

    panels, sanities, winners = load_real_inputs()
    df = evaluate(panels, HORIZONS)
    branch, std_win, cas_gen = branch_of(df)
    write_outputs(df, branch, std_win, cas_gen, sanities, winners, OUT_CSV, OUT_MD,
                  rerun_note=rerun_note)
    print(f"[hpo_test_eval] branch fired: {branch} "
          f"({len(cas_gen)} genuine cascade cell(s), {len(std_win)} standalone win(s))")
    print(f"[hpo_test_eval] wrote {OUT_CSV} and {OUT_MD}")
    print("[hpo_test_eval] §5: this was the one shot. Do not run again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
