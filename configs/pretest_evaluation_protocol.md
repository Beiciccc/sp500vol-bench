# Pre-registration: one-shot test evaluation protocol for the HPO tuned arm (hpo-pretest-v1.0)

Date: 2026-07-15. This file was committed and tagged **before** any test metric was computed. Upstream pre-registration:
`configs/hpo_arm.yaml` (tag hpo-prereg-v1.0 + amendment_1). Motivation: the paper's current sentence
"Nor is the null a tuning artefact: validation-tuned challengers create no newly genuine cell"
rests on a **pre-registration pilot** (single seed, lr + early stopping; tuning made it worse); the real HPO search shows that tuning
closes ~74% of the FinBERT-vs-HAR standalone gap on val-select (0.4402 -> ensemble 0.3423 vs HAR
0.3075). As written the sentence **does not hold**; it must be rewritten after adjudication by the one-shot test evaluation specified here.

## 1. Evaluation subjects (scope-trimming disclosure -> §4-i)

- **T1a** (FinBERT-S1, long_form): winner = trial 21 (seed_validation.json, the pre-registered
  3-seed ensemble rule), seeds 2026/2027/2028, rung 2 checkpoints.
- **T1c** (FinBERT-S1, event_driven): winner = pending the output of `results/hpo/T1c/seed_validation.json`
  (retraining in flight; same rule as T1a, ahead of any test metric).
- Ensemble rule: **identical to the one used at selection time** — per-observation log-space mean (the "rule" field of
  seed_validation.json; note this differs from the arithmetic-mean convention of the paper's C-model primary; the two are each
  internally consistent, are disclosed side by side in the md output, and must never be mixed).

## 2. Frozen inputs

- `results/hpo/<task>/<rid>/predictions_fulltest.parquet` x 3 seeds x 2 tasks
  (produced by `predict_winner_test.py`: strict fingerprint validation, manifest sha256 asserted consistent with summary.json,
  structurally metric-free). Before running, the evaluation script asserts that all 6 files exist and each contains all three splits.
- Reference: `fc.load("A2_har_rv", <disc>)` (the same code path as the committed M1 tables).

## 3. Pre-declared statistics (one execution; everything enters the table)

For each task (disc = long_form / event_driven), on the **real test split**:
- **(a) standalone**: vol-unit QLIKE of the tuned 3-seed ensemble vs **the val-recalibrated A2**
  (log_recal fitted on val, frozen on test, current fc function), per horizon, day-clustered DM (HAC lag
  h-1, HLN).
- **(b) M1 combination increment**: under log_combo (fitted on val, frozen on test), the tuned-ensemble text arm's increment over the single
  recalibrated HAR, per horizon, day-clustered DM — mechanically identical to the committed 69-cell grid.
- **Holm families (pre-declared)**: `tuned_standalone` = 2 tasks x 3 horizons = 6 cells, Holm(6);
  `tuned_cascade(M1)` = an isomorphic 6 cells, Holm(6). Same names and meanings as the yaml's holm_families, with the size
  trimmed per §4-i.
- **placebo gate** (yaml rule verbatim): any cell entering a "win" statement must pass both the label-shuffle and
  within-date placebos; failure is recorded as an artefact. With no win, placebos run only on M1-significant cells.
- Units: vol-unit is the adjudication unit (consistent with the selection metric); variance-unit is tabled alongside (report-only).

## 4. Deviation disclosures (relative to hpo_arm.yaml; all recorded before evaluation)

- **(i) Scope trimming**: amendment_1's T6d3/T6c5, Track-B, T4/T3/T2/T1d and the ~51-cell Holm family
  were not executed (ROI adjudication after an internal adversarial dry-run: the ceiling is set by project scope, so the HPO queue retains only the
  correctness obligations; degradation_ladder extended in the spirit of wall-clock). **Consequence**: the paper's rewritten sentence narrows in scope to
  "an ASHA-tuned FinBERT (both channels)", and must not use the general phrase "validation-tuned challengers".
- **(ii) s_strategy_recheck not executed**: S1 was frozen into retraining; no val recheck of S2/S3/S4. Disclosure:
  if some S* beats S1 on val, the tuned arm may be understated — a direction that is conservative for the null (the tuned arm could only get stronger); written truthfully.
- **(iii) Selection convention**: track_a follows the yaml as a pooled-across-horizons val_select argmin
  (not a per-horizon selection); if the main text mentions it, it must say pooled.
- **(iv) Code wording**: `vol_unit_qlike(pred, "test")` in asha_hpo.py reads as test but is physically
  a val-select borrowing an empty slot (firewall: test rows are physically deleted in prepare_data, manifest sha256 on file);
  predict_winner_test.py has re-verified n_test=23786 (T1a) consistent with summary.json.
- **(v) The pilot touched test**: the yaml's pilot_disclosure is reproduced verbatim in the paper's appendix.

## 5. Single-shot discipline and rewrite branches (committed ahead of results)

The evaluation script `scripts/analysis/hpo_test_eval.py` **runs exactly once**; all numbers, in whichever direction, enter
`results/tables/hpo_test_eval.{csv,md}` + FACTS.md + the paper. Branches:
- **B1 (expected)**: standalone still loses + M1 yields no newly genuine cell (Holm+placebo) -> the sentence is rewritten as
  "tuning closes ~74% of the standalone gap on val, still loses out of sample to the recalibrated HAR, and produces no new
  genuine cell — the null is tuning-robust" (stronger than the original sentence, and true).
- **B2**: a newly genuine cell appears -> reported truthfully in the main text, with the headline counts and the abstract re-assessed (nothing hidden, nothing softened);
  that cell enters a tuned-family table placed alongside the committed 69-cell grid.
- **B3 (mixed)**: reported cell by cell, with the sentence rewritten in the weakest defensible form.
Forbidden: adjusting the convention, units, or family structure conditional on results; forbidden: a second run (unless the script itself has a bug, whose
fix must be recorded in the md with the diff and the reason).
