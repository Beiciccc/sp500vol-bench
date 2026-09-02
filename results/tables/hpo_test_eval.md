# HPO tuned-arm — the ONE-SHOT pre-registered test evaluation (hpo-pretest-v1.0)

Generated 2026-07-15T04:26:24+00:00 by scripts/analysis/hpo_test_eval.py. Protocol: configs/pretest_evaluation_protocol.md (§3 statistics, §4 disclosures, §5 single-shot discipline). This file is written ONCE; the script refuses a second run.

Winners: T1a = trial 21, T1c = trial 11; seeds (2026, 2027, 2028), rung 2 checkpoints.

## Ensemble-rule disclosure (protocol §1)

- Tuned arm here = per-observation **log-space** mean of the 3 seed forecasts, `exp(mean(log(clip(pred, 1e-8))))` — identical to the HPO selection rule (seed_validation.json `rule`), so the object tested is the object selected.
- The paper's C-model primary convention is the **arithmetic** per-observation mean (m1_ensemble_primary.py). The two conventions are each internally consistent and are NOT mixed anywhere; do not compare rows across the two conventions.

## (a) tuned_standalone — tuned ensemble vs val-recalibrated A2 (vol-unit decides; Holm(6) within family)

| task | disc | h | n_test | n_days | QLIKE recal-A2 | QLIKE tuned | rel% | DM(clu) | p | Holm | var-unit rel% | placebo ls DM | placebo wd DM | gate |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| T1a | long_form | 5 | 7951 | 809 | 0.1209 | 0.1444 | -19.46 | +5.36 | 0.0000 | 0.0000 | -19.09 | - | - | - |
| T1a | long_form | 10 | 7933 | 803 | 0.0873 | 0.1204 | -37.93 | +3.20 | 0.0014 | 0.0028 | -64.41 | - | - | - |
| T1a | long_form | 20 | 7902 | 794 | 0.0701 | 0.0875 | -24.79 | +1.01 | 0.3115 | 0.3115 | -47.79 | - | - | - |
| T1c | event_driven | 5 | 25109 | 996 | 0.1265 | 0.1451 | -14.72 | +8.27 | 0.0000 | 0.0000 | -6.85 | - | - | - |
| T1c | event_driven | 10 | 25001 | 991 | 0.0883 | 0.1106 | -25.33 | +5.89 | 0.0000 | 0.0000 | -31.46 | - | - | - |
| T1c | event_driven | 20 | 24732 | 981 | 0.0645 | 0.0875 | -35.58 | +5.09 | 0.0000 | 0.0000 | -48.15 | - | - | - |

## (b) tuned_cascade (M1) — log_combo increment over the single recalibrated HAR (vol-unit decides; Holm(6) within family)

| task | disc | h | n_test | n_days | QLIKE fR | QLIKE fU | rel% | g_log | DM(clu) | p | Holm | var-unit rel% | placebo ls DM | placebo wd DM | gate |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| T1a | long_form | 5 | 7951 | 809 | 0.1209 | 0.1190 | +1.57 | +0.429 | -0.88 | 0.3812 | 1.0000 | +4.17 | - | - | - |
| T1a | long_form | 10 | 7933 | 803 | 0.0873 | 0.0872 | +0.10 | +0.074 | +0.53 | 0.5961 | 1.0000 | +0.70 | - | - | - |
| T1a | long_form | 20 | 7902 | 794 | 0.0701 | 0.0726 | -3.54 | +0.324 | +3.14 | 0.0018 | 0.0088 | -1.16 | - | - | - |
| T1c | event_driven | 5 | 25109 | 996 | 0.1265 | 0.1229 | +2.88 | +0.322 | -4.89 | 0.0000 | 0.0000 | +5.87 | -2.40 | -2.46 | ARTEFACT |
| T1c | event_driven | 10 | 25001 | 991 | 0.0883 | 0.0864 | +2.13 | +0.241 | -2.54 | 0.0111 | 0.0446 | +5.93 | +0.40 | +1.38 | GENUINE |
| T1c | event_driven | 20 | 24732 | 981 | 0.0645 | 0.0648 | -0.33 | +0.255 | +0.03 | 0.9730 | 1.0000 | +4.18 | - | - | - |

Reading guide: DM(clu) < 0 = tuned arm/text-augmented combiner better; day-clustered DM (daily-mean loss differentials, HAC lag h-1, HLN, over effective_trading_day). rel% > 0 = tuned arm lowers QLIKE vs the reference. vol-unit is the DECISION unit; var-unit columns are report-only (protocol §3). Placebo gate (hpo_arm.yaml `placebo_gate`) runs on every cell with DM<0 & Holm<.05: label-shuffle (whole-sample permutation, the committed row3_tuned_m1/m1_ensemble_primary mechanism, seeds 1000-1004) AND within-date (withindate_placebo.permute_within_day). Committed pass threshold |mean clustered placebo DM| < 2 for the increment form; for the standalone form the rule is applied ONE-SIDED (fail only if placebo DM < -2) because destroying a standalone forecast drives DM to +large by construction — a two-sided rule would mechanically fail every standalone win.

## Branch verdict (§5): **B2**

B2: 1 newly genuine tuned-cascade cell(s) (T1c/h10). Per §5 these enter the main text alongside the committed 69-cell table; headline counts and abstract must be re-estimated — no hiding, no downplaying.

## Disclosures (protocol §4, recorded before evaluation)

- **(i) Scope trimming** — amendment_1's T6d3/T6c5, Track-B, T4/T3/T2/T1d and the ~51-cell Holm family were not executed (ROI ruling made after the internal adversarial dry-run). Consequence: the scope of the paper's rewritten sentence narrows to "an ASHA-tuned FinBERT (both channels)"; the full phrase "validation-tuned challengers" must not be written. The Holm family for this table is correspondingly 2 tasks × 3 horizons = 6.
- **(ii) s_strategy_recheck not executed** — S1 was frozen and taken into retraining; no val recheck of S2/S3/S4 was performed; if S* beats S1 on val, the tuned arm may be underestimated — the direction is conservative with respect to the null (the tuned arm would only become stronger).
- **(iii) Selection convention** — track_a is the pooled-across-horizons val_select argmin (not a per-horizon selection); if the main text mentions it, it must say pooled.
- **(iv) Code wording** — the vol_unit_qlike(pred, "test") in asha_hpo.py reads as test, but physically it is val-select borrowing an empty slot (the test rows are physically deleted in prepare_data; the manifest sha256 is on record); predict_winner_test.py has already re-checked that n_test is consistent with summary.json (re-verified again in the SANITY section of this table).
- **(v) The pilot touched test** — the yaml pilot_disclosure is transcribed verbatim into the paper's appendix (a single-seed lr+early-stopping pilot touched test before pre-registration, and the result was null; the argument on the direction of the bias is given in the yaml).

## SANITY

- **T1a** (long_form): 3-seed (accession, horizon_days) key sets identical = True; A2 merge 1:1 = True, test rows lost = 0, val rows lost = 0 (train rows outside A2 panel: 0, informational); n_test = 23786 vs summary.json n_test_dropped = seed2026: 23786, seed2027: 23786, seed2028: 23786 — MATCH.
- **T1c** (event_driven): 3-seed (accession, horizon_days) key sets identical = True; A2 merge 1:1 = True, test rows lost = 0, val rows lost = 0 (train rows outside A2 panel: 0, informational); n_test = 74842 vs summary.json n_test_dropped = seed2026: 74842, seed2027: 74842, seed2028: 74842 — MATCH.
- Per-cell hard assert: the standalone recalibration (log_recal val-fit, test-frozen; pattern of scripts/experiments/second_domain/yelp_entity_disjoint.py) reproduces fc.log_combo's fR to rtol 1e-10 in every cell.
