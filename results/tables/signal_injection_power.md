# ROW 1 — Signal-injection power calibration + per-cell MDE for the 0/69 cascade headline

> **ORACLE INJECTION — POWER CALIBRATION, NOT A FORECAST.** The synthetic text forecast `f_synth = exp(log f_text + delta*s)` uses **test labels by design** (s is the within-firm-demeaned test log-residual of the recalibrated HAR reference). This is the ONE declared exception to the no-look-ahead rule in the round-3 remediation plan; it calibrates the cascade's detection power and may never be cited as forecasting performance. All combiner/reference weights remain genuine validation-only fits on REAL text (the oracle content enters only through the test-side text array).

## RESTATED vs BEFORE

| | BEFORE (committed) | RESTATED (this table) |
|---|---|---|
| power of the cascade | never calibrated — panel CRITICAL: "0/69 is not interpretable as evidence of absence" | recovery rates measured at known injected firm-orthogonal signal of 0.3/0.5/1.0% rel-QLIKE + per-cell MDE |
| delta=0 baseline (real data) | HAR 38/69, firm 8/69, pool 9/69, full AND 0/69 (control_intersection_ensemble) | reproduced exactly: 38/8/9, see SANITY |

## Design

Grid and text object = the declared primary (`m1_ensemble_primary.py`: 69 cells, per-observation 3-seed-ensemble text; A/B, C6, D4 single-run). Injected signal `s` = test-split log-residual of the single recalibrated-HAR reference, demeaned WITHIN FIRM on the test split — so a firm-level identity regressor cannot mechanically absorb it (verified: max within-firm |mean s| < 1e-12). Per cell, `delta` is bisected (on kappa = g1*delta, tolerance 0.02pp) so the realised test rel-QLIKE improvement of the stage-HAR combined forecast hits the target exactly; cells whose REAL improvement already exceeds the target receive delta<0 (signal REMOVED down to the target) — the design equalises the realised effect at exactly X% in every cell, making each recovery rate a power estimate at that effect size. The same `f_synth` then runs through the firm-identity reference (val-window firm-mean spec) and the maximal 5-price pool with their own REAL validation-fit text loadings (`kappa_stage = g_stage*delta` reported per cell): the cascade is stress-tested exactly as deployed. Stage FIRM/POOL run on the 5-price inner-join panel (test rows a subset of the HAR-stage panel; n_test2 reported per cell). No subsampling anywhere. **Unit convention:** every QLIKE in this table is in VOLATILITY units, q(y, f) on realised vol — the same convention as the committed cascade tables it gates against (`vol_qlike_R` / `qlike_Rfirm` / `qlike_Rstar`); the variance-unit convention q(y^2, f^2) is treated separately in the variance-unit remediation line.

## PRE-DECLARED Holm families

Nine families, declared before any result was inspected: for each injection target level in {0.3%, 0.5%, 1.0%} and each cascade stage in {single recalibrated HAR, firm-identity-augmented reference, maximal 5-price pool}, one family = the 69 day-clustered DM p-values of that (stage, level) grid. Detection = clustered DM < 0 AND Holm-adjusted p < .05 within the family (same 'detected' criterion as the committed cascade tables; the placebo gate is not part of the detection criterion here, matching the row-1 brief).

## SANITY

All gates enforced in-script; the run aborts before writing any table if one fails. Status: **PASS**.

| gate | committed table reproduced at delta=0 | max abs diff | verdict |
|---|---|---|---|
| A | m1_ensemble_primary.csv `vol_qlike_R` | 8.33e-17 | PASS |
| A | m1_ensemble_primary.csv `vol_rel_impr_pct` | 2.22e-16 | PASS |
| B | firm_identity_ensemble.csv `qlike_Rfirm` / `rel_impr_pct_firm` / `dm` | 4.44e-16 | PASS |
| C | maximal_reference_ensemble.csv `qlike_Rstar` / `rel_impr_pct_maximal` / `dm` | 1.78e-15 | PASS |
| D | control_intersection_ensemble Holm counts at delta=0 (HAR/firm/pool) | [38, 8, 9] == [38, 8, 9] | PASS |
| D | injected s within-firm mean-zero | 1.48e-16 | PASS |

Calibration: 207/207 (cell, level) pairs converged within 0.02pp (max |achieved - target| = 0.0200pp).

## HEADLINE — cascade recovery of a known firm-orthogonal signal

| injected level (realised rel-QLIKE of f_U over f_R) | HAR stage (DM<0 & Holm<.05) | firm-identity stage | maximal-pool stage | full conjunction (all 3) | cells with delta<0 (signal removed to target) |
|---|---|---|---|---|---|
| 0 (real data) | 38/69 | 8/69 | 9/69 | 0/69 | — |
| 0.3% | **12/69** | **7/69** | **6/69** | **2/69** | 50 |
| 0.5% | **20/69** | **11/69** | **12/69** | **6/69** | 47 |
| 1.0% | **41/69** | **20/69** | **19/69** | **13/69** | 42 |

**Headline:** at 0.3% injected signal the cascade recovers 12/69 (HAR) / 7/69 (firm) / 6/69 (pool), full conjunction 2/69; at 0.5% injected signal the cascade recovers 20/69 (HAR) / 11/69 (firm) / 12/69 (pool), full conjunction 6/69; at 1.0% injected signal the cascade recovers 41/69 (HAR) / 20/69 (firm) / 19/69 (pool), full conjunction 13/69.

## Per-cell MDE (80% power, 5% two-sided size)

`MDE_rel% = (1.96+0.84) * SE_daily / mean(QLIKE_R) * 100`, with SE_daily the day-clustered (HAC lag = h-1 days) standard error of the mean daily loss differential of the REAL (delta=0) stage-HAR comparison; denominator = per-observation mean QLIKE of f_R (= `vol_qlike_R`).

| disclosure | h | n cells | median MDE_rel% | IQR | min | max |
|---|---|---|---|---|---|---|
| long_form | 5 | 15 | 1.07 | [0.46, 1.28] | 0.14 | 2.99 |
| long_form | 10 | 15 | 0.84 | [0.59, 1.18] | 0.03 | 2.36 |
| long_form | 20 | 15 | 1.27 | [0.66, 1.86] | 0.02 | 3.65 |
| event_driven | 5 | 8 | 0.44 | [0.20, 0.77] | 0.01 | 0.86 |
| event_driven | 10 | 8 | 0.46 | [0.27, 0.83] | 0.04 | 0.93 |
| event_driven | 20 | 8 | 0.96 | [0.50, 1.59] | 0.20 | 2.63 |
| **all** | — | 69 | **0.82** | [0.37, 1.27] | 0.01 | 3.65 |

**Observed effects vs detectability.** The real (delta=0) stage-HAR effects span -3.86% to +5.92% (the 38 Holm-detected cells span +0.02% to +5.92%). **40/69** cells have an observed effect at or above their own MDE (above detectability); **29/69** sit below it. Of the 38 detected cells, 4 lie below their prospective MDE (detected despite <80% ex-ante power — expected, since MDE is an 80%-power threshold, not a significance bound). Cells powered (MDE <= target) for each injected level: 0.3% -> 14/69, 0.5% -> 21/69, 1.0% -> 43/69 — compare these analytic counts with the empirical HAR-stage recovery rates above.

## Per-cell detail — real data (delta=0): observed effect vs MDE

| disc | model | h | n_days | rel%(real) | MDE_rel% | above MDE? | Holm-detected (HAR/firm/pool) |
|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | 996 | +1.33 | 0.80 | YES | Y/Y/. |
| event_driven | B1_bow_ridge | 10 | 991 | +1.23 | 0.82 | YES | Y/Y/. |
| event_driven | B1_bow_ridge | 20 | 981 | +1.53 | 1.22 | YES | Y/./. |
| event_driven | B2_tfidf_ridge | 5 | 996 | +1.21 | 0.76 | YES | ././. |
| event_driven | B2_tfidf_ridge | 10 | 991 | +1.35 | 0.87 | YES | ./Y/. |
| event_driven | B2_tfidf_ridge | 20 | 981 | +1.84 | 1.36 | YES | Y/./. |
| event_driven | B3_lm_linear | 5 | 996 | +0.25 | 0.20 | YES | ././. |
| event_driven | B3_lm_linear | 10 | 991 | +0.20 | 0.36 | no | ././. |
| event_driven | B3_lm_linear | 20 | 981 | +0.26 | 0.55 | no | ././. |
| event_driven | B4_lm_features | 5 | 996 | +0.18 | 0.18 | YES | ././. |
| event_driven | B4_lm_features | 10 | 991 | +0.08 | 0.08 | YES | ././. |
| event_driven | B4_lm_features | 20 | 981 | +0.25 | 0.37 | no | ././. |
| event_driven | C2_finbert_s1 | 5 | 996 | +2.57 | 0.86 | YES | Y/./. |
| event_driven | C2_finbert_s1 | 10 | 991 | +2.42 | 0.93 | YES | Y/./. |
| event_driven | C2_finbert_s1 | 20 | 981 | +1.58 | 2.63 | no | ././. |
| event_driven | C6_llmtext | 5 | 996 | +1.21 | 0.50 | YES | Y/Y/. |
| event_driven | C6_llmtext | 10 | 991 | +1.00 | 0.56 | YES | Y/Y/. |
| event_driven | C6_llmtext | 20 | 981 | +0.66 | 0.70 | no | ./Y/. |
| event_driven | D2_gated_fusion | 5 | 996 | +0.56 | 0.37 | YES | ././. |
| event_driven | D2_gated_fusion | 10 | 991 | +0.18 | 0.33 | no | ././. |
| event_driven | D2_gated_fusion | 20 | 981 | -2.76 | 2.25 | no | ././. |
| event_driven | D4_llmfused | 5 | 996 | -0.01 | 0.01 | no | ././. |
| event_driven | D4_llmfused | 10 | 991 | -0.01 | 0.04 | no | ././. |
| event_driven | D4_llmfused | 20 | 981 | -0.35 | 0.20 | no | ././. |
| long_form | B1_bow_ridge | 5 | 809 | +1.65 | 1.11 | YES | Y/./. |
| long_form | B1_bow_ridge | 10 | 803 | +1.44 | 0.81 | YES | Y/./. |
| long_form | B1_bow_ridge | 20 | 794 | +2.99 | 1.38 | YES | Y/./. |
| long_form | B2_tfidf_ridge | 5 | 809 | +3.33 | 1.67 | YES | Y/./Y |
| long_form | B2_tfidf_ridge | 10 | 803 | +3.48 | 1.25 | YES | Y/./Y |
| long_form | B2_tfidf_ridge | 20 | 794 | +5.92 | 1.86 | YES | Y/./Y |
| long_form | B3_lm_linear | 5 | 809 | +0.49 | 1.70 | no | ././. |
| long_form | B3_lm_linear | 10 | 803 | +1.79 | 1.70 | YES | Y/Y/. |
| long_form | B3_lm_linear | 20 | 794 | +3.48 | 1.87 | YES | Y/./. |
| long_form | B4_lm_features | 5 | 809 | +0.11 | 0.21 | no | ././. |
| long_form | B4_lm_features | 10 | 803 | -0.92 | 0.57 | no | ././. |
| long_form | B4_lm_features | 20 | 794 | -1.92 | 0.70 | no | ././. |
| long_form | C1_bert_s1 | 5 | 809 | +3.42 | 2.99 | YES | Y/./. |
| long_form | C1_bert_s1 | 10 | 803 | -0.85 | 0.84 | no | ././. |
| long_form | C1_bert_s1 | 20 | 794 | +2.70 | 1.27 | YES | Y/./. |
| long_form | C2_finbert_s1 | 5 | 809 | +1.90 | 0.95 | YES | Y/./. |
| long_form | C2_finbert_s1 | 10 | 803 | +2.62 | 1.27 | YES | Y/./. |
| long_form | C2_finbert_s1 | 20 | 794 | -0.59 | 2.82 | no | ././. |
| long_form | C2_finbert_s2 | 5 | 809 | +1.21 | 0.59 | YES | Y/./Y |
| long_form | C2_finbert_s2 | 10 | 803 | +0.48 | 0.26 | YES | Y/./. |
| long_form | C2_finbert_s2 | 20 | 794 | +1.68 | 0.93 | YES | Y/./. |
| long_form | C2_finbert_s3 | 5 | 809 | +2.90 | 1.35 | YES | Y/./. |
| long_form | C2_finbert_s3 | 10 | 803 | +2.31 | 2.36 | no | Y/./. |
| long_form | C2_finbert_s3 | 20 | 794 | -3.86 | 3.65 | no | ././. |
| long_form | C2_finbert_s4 | 5 | 809 | +1.46 | 1.07 | YES | Y/./. |
| long_form | C2_finbert_s4 | 10 | 803 | +0.36 | 0.22 | YES | Y/./. |
| long_form | C2_finbert_s4 | 20 | 794 | +3.08 | 1.31 | YES | Y/./Y |
| long_form | C3_roberta_s1 | 5 | 809 | +0.30 | 0.14 | YES | Y/./. |
| long_form | C3_roberta_s1 | 10 | 803 | +1.84 | 1.11 | YES | Y/./. |
| long_form | C3_roberta_s1 | 20 | 794 | +0.02 | 0.02 | no | Y/./. |
| long_form | C4_longformer | 5 | 809 | +1.47 | 0.69 | YES | Y/./Y |
| long_form | C4_longformer | 10 | 803 | -2.95 | 0.96 | no | ././. |
| long_form | C4_longformer | 20 | 794 | +0.91 | 0.63 | YES | Y/./. |
| long_form | C6_llmtext | 5 | 809 | +1.79 | 1.11 | YES | Y/./Y |
| long_form | C6_llmtext | 10 | 803 | +2.25 | 1.07 | YES | Y/./Y |
| long_form | C6_llmtext | 20 | 794 | +0.27 | 0.37 | no | Y/Y/. |
| long_form | D1_concat_mlp | 5 | 809 | -1.04 | 1.21 | no | ././. |
| long_form | D1_concat_mlp | 10 | 803 | -0.46 | 0.62 | no | ././. |
| long_form | D1_concat_mlp | 20 | 794 | +0.12 | 0.09 | YES | Y/./Y |
| long_form | D2_gated_fusion | 5 | 809 | +0.19 | 0.29 | no | ././. |
| long_form | D2_gated_fusion | 10 | 803 | -0.02 | 0.03 | no | ././. |
| long_form | D2_gated_fusion | 20 | 794 | -2.27 | 3.04 | no | ././. |
| long_form | D4_llmfused | 5 | 809 | +0.15 | 0.32 | no | ././. |
| long_form | D4_llmfused | 10 | 803 | -0.15 | 0.84 | no | ././. |
| long_form | D4_llmfused | 20 | 794 | +0.60 | 0.73 | no | Y/./. |

## Per-cell detail — injected level 0.3%

| disc | model | h | delta | kappa(HAR) | kappa(firm) | kappa(pool) | rel%(HAR) | DM(HAR) | Holm | rel%(firm) | DM(firm) | Holm | rel%(pool) | DM(pool) | Holm | detect H/F/P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | -0.022 | -0.0051 | -0.0018 | -0.0038 | +0.31 | +0.14 | 1.000 | +0.11 | -0.06 | 1.000 | -0.46 | +2.74 | 0.273 | ././. |
| event_driven | B1_bow_ridge | 10 | -0.020 | -0.0047 | -0.0007 | -0.0035 | +0.32 | -0.16 | 1.000 | +0.11 | -1.68 | 1.000 | -0.84 | +3.76 | 0.010 | ././. |
| event_driven | B1_bow_ridge | 20 | -0.029 | -0.0070 | +0.0006 | -0.0051 | +0.31 | -0.23 | 1.000 | -0.10 | +2.56 | 0.566 | -1.17 | +4.18 | 0.002 | ././. |
| event_driven | B2_tfidf_ridge | 5 | -0.018 | -0.0045 | -0.0016 | -0.0034 | +0.31 | +0.44 | 1.000 | +0.13 | -0.02 | 1.000 | -0.55 | +3.79 | 0.008 | ././. |
| event_driven | B2_tfidf_ridge | 10 | -0.022 | -0.0055 | -0.0009 | -0.0041 | +0.29 | +0.42 | 1.000 | +0.11 | -0.98 | 1.000 | -0.86 | +4.52 | 0.000 | ././. |
| event_driven | B2_tfidf_ridge | 20 | -0.033 | -0.0088 | +0.0001 | -0.0067 | +0.31 | +0.11 | 1.000 | -0.02 | +1.95 | 1.000 | -1.30 | +4.56 | 0.000 | ././. |
| event_driven | B3_lm_linear | 5 | +0.000 | +0.0002 | +0.0001 | +0.0002 | +0.29 | -2.93 | 0.190 | +0.13 | -3.48 | 0.031 | +0.19 | -2.27 | 0.901 | ./Y/. |
| event_driven | B3_lm_linear | 10 | +0.001 | +0.0006 | +0.0002 | +0.0005 | +0.31 | -2.05 | 1.000 | +0.15 | -2.67 | 0.417 | +0.18 | -1.33 | 1.000 | ././. |
| event_driven | B3_lm_linear | 20 | +0.000 | +0.0002 | +0.0001 | +0.0002 | +0.29 | -1.29 | 1.000 | +0.18 | -2.27 | 1.000 | +0.12 | -0.52 | 1.000 | ././. |
| event_driven | B4_lm_features | 5 | +0.001 | +0.0006 | +0.0007 | +0.0006 | +0.30 | -2.84 | 0.233 | +0.35 | -3.41 | 0.039 | +0.32 | -3.14 | 0.086 | ./Y/. |
| event_driven | B4_lm_features | 10 | +0.003 | +0.0012 | +0.0019 | +0.0012 | +0.31 | -9.19 | 0.000 | +0.50 | -9.12 | 0.000 | +0.33 | -9.07 | 0.000 | Y/Y/Y |
| event_driven | B4_lm_features | 20 | +0.001 | +0.0004 | +0.0002 | +0.0004 | +0.32 | -2.48 | 0.630 | +0.14 | -2.80 | 0.286 | +0.15 | -1.63 | 1.000 | ././. |
| event_driven | C2_finbert_s1 | 5 | -0.049 | -0.0115 | +0.0003 | -0.0071 | +0.29 | +1.43 | 1.000 | -0.00 | -1.85 | 1.000 | -0.67 | +6.15 | 0.000 | ././. |
| event_driven | C2_finbert_s1 | 10 | -0.049 | -0.0109 | +0.0028 | -0.0073 | +0.29 | +0.62 | 1.000 | -0.13 | +0.13 | 1.000 | -1.36 | +6.17 | 0.000 | ././. |
| event_driven | C2_finbert_s1 | 20 | -0.027 | -0.0076 | +0.0020 | -0.0061 | +0.29 | -0.56 | 1.000 | -0.28 | +1.28 | 1.000 | -2.65 | +3.80 | 0.008 | ././. |
| event_driven | C6_llmtext | 5 | -0.017 | -0.0045 | -0.0019 | -0.0032 | +0.31 | -0.11 | 1.000 | +0.13 | -0.27 | 1.000 | -0.25 | +2.90 | 0.170 | ././. |
| event_driven | C6_llmtext | 10 | -0.013 | -0.0035 | -0.0008 | -0.0024 | +0.31 | -0.32 | 1.000 | +0.10 | -1.09 | 1.000 | -0.35 | +3.03 | 0.122 | ././. |
| event_driven | C6_llmtext | 20 | -0.008 | -0.0020 | -0.0004 | -0.0016 | +0.32 | -0.56 | 1.000 | +0.14 | -2.12 | 1.000 | -0.20 | +1.30 | 1.000 | ././. |
| event_driven | D2_gated_fusion | 5 | -0.004 | -0.0014 | -0.0005 | -0.0011 | +0.28 | +0.31 | 1.000 | +0.12 | -0.22 | 1.000 | +0.07 | +1.10 | 1.000 | ././. |
| event_driven | D2_gated_fusion | 10 | +0.001 | +0.0006 | +0.0001 | -0.0001 | +0.29 | -1.43 | 1.000 | +0.08 | -1.94 | 1.000 | -0.03 | +2.05 | 1.000 | ././. |
| event_driven | D2_gated_fusion | 20 | -0.025 | +0.0176 | +0.0419 | +0.0287 | +0.30 | -0.35 | 1.000 | -1.22 | +0.44 | 1.000 | +2.42 | -1.64 | 1.000 | ././. |
| event_driven | D4_llmfused | 5 | +0.705 | +0.0016 | +0.0097 | -0.0421 | +0.30 | -19.04 | 0.000 | +1.87 | -19.27 | 0.000 | -8.66 | +18.63 | 0.000 | Y/Y/. |
| event_driven | D4_llmfused | 10 | +0.109 | +0.0016 | +0.0028 | -0.0019 | +0.30 | -12.10 | 0.000 | +0.52 | -11.73 | 0.000 | -0.40 | +11.70 | 0.000 | Y/Y/. |
| event_driven | D4_llmfused | 20 | -0.056 | +0.0037 | +0.0035 | +0.0026 | +0.30 | -4.76 | 0.000 | +0.33 | -5.21 | 0.000 | +0.21 | -4.48 | 0.000 | Y/Y/Y |
| long_form | B1_bow_ridge | 5 | -0.073 | -0.0078 | +0.0006 | -0.0066 | +0.30 | -0.73 | 1.000 | -0.05 | -0.49 | 1.000 | -0.27 | +0.70 | 1.000 | ././. |
| long_form | B1_bow_ridge | 10 | -0.120 | -0.0070 | +0.0127 | -0.0044 | +0.32 | -0.63 | 1.000 | -0.81 | -0.92 | 1.000 | -0.19 | +1.30 | 1.000 | ././. |
| long_form | B1_bow_ridge | 20 | -0.214 | -0.0211 | +0.0310 | -0.0223 | +0.31 | -0.44 | 1.000 | -1.57 | -0.65 | 1.000 | -1.09 | +2.15 | 1.000 | ././. |
| long_form | B2_tfidf_ridge | 5 | -0.074 | -0.0176 | +0.0029 | -0.0156 | +0.30 | -0.69 | 1.000 | -0.13 | -0.77 | 1.000 | -1.07 | +1.33 | 1.000 | ././. |
| long_form | B2_tfidf_ridge | 10 | -0.116 | -0.0197 | +0.0201 | -0.0199 | +0.31 | -2.19 | 1.000 | -0.87 | +0.18 | 1.000 | -1.24 | +0.49 | 1.000 | ././. |
| long_form | B2_tfidf_ridge | 20 | -0.214 | -0.0438 | +0.0555 | -0.0647 | +0.30 | -1.13 | 1.000 | -2.07 | +0.15 | 1.000 | -4.08 | +2.24 | 0.946 | ././. |
| long_form | B3_lm_linear | 5 | -0.002 | -0.0012 | -0.0003 | -0.0011 | +0.29 | -2.28 | 1.000 | +0.16 | -1.53 | 1.000 | -0.41 | -1.20 | 1.000 | ././. |
| long_form | B3_lm_linear | 10 | -0.016 | -0.0094 | -0.0012 | -0.0086 | +0.29 | -2.38 | 0.832 | +0.08 | -1.36 | 1.000 | -0.72 | -0.87 | 1.000 | ././. |
| long_form | B3_lm_linear | 20 | -0.043 | -0.0250 | +0.0001 | -0.0262 | +0.30 | -0.36 | 1.000 | -0.01 | -0.45 | 1.000 | -1.26 | +1.66 | 1.000 | ././. |
| long_form | B4_lm_features | 5 | +0.008 | +0.0012 | -0.0017 | +0.0006 | +0.31 | -1.52 | 1.000 | -0.42 | +1.09 | 1.000 | +0.14 | -0.90 | 1.000 | ././. |
| long_form | B4_lm_features | 10 | -0.038 | +0.0078 | +0.0212 | +0.0100 | +0.32 | -2.04 | 1.000 | +0.39 | -1.17 | 1.000 | +0.83 | -2.74 | 0.273 | ././. |
| long_form | B4_lm_features | 20 | -0.095 | +0.0178 | +0.0614 | +0.0213 | +0.30 | -4.35 | 0.001 | -0.58 | -3.00 | 0.158 | +0.94 | -5.82 | 0.000 | Y/./Y |
| long_form | C1_bert_s1 | 5 | -0.033 | -0.0180 | -0.0081 | -0.0184 | +0.31 | -0.56 | 1.000 | +0.03 | +1.51 | 1.000 | -2.19 | +1.04 | 1.000 | ././. |
| long_form | C1_bert_s1 | 10 | -0.056 | +0.0072 | +0.0319 | +0.0141 | +0.31 | -0.34 | 1.000 | +0.11 | -0.24 | 1.000 | +2.10 | -2.17 | 1.000 | ././. |
| long_form | C1_bert_s1 | 20 | -0.122 | -0.0189 | +0.0469 | -0.0293 | +0.30 | -1.91 | 1.000 | -1.37 | +0.50 | 1.000 | -2.69 | +2.10 | 1.000 | ././. |
| long_form | C2_finbert_s1 | 5 | -0.028 | -0.0094 | +0.0013 | -0.0071 | +0.31 | -0.29 | 1.000 | -0.05 | -1.06 | 1.000 | -0.61 | +2.16 | 1.000 | ././. |
| long_form | C2_finbert_s1 | 10 | -0.096 | -0.0145 | +0.0185 | -0.0129 | +0.30 | -1.92 | 1.000 | -0.40 | -0.17 | 1.000 | -1.12 | +0.94 | 1.000 | ././. |
| long_form | C2_finbert_s1 | 20 | +0.031 | +0.0072 | -0.0074 | +0.0109 | +0.30 | -1.23 | 1.000 | -0.86 | +0.89 | 1.000 | -4.23 | +1.74 | 1.000 | ././. |
| long_form | C2_finbert_s2 | 5 | -0.030 | -0.0053 | +0.0092 | -0.0019 | +0.30 | -1.57 | 1.000 | -0.69 | -0.41 | 1.000 | -0.10 | +0.45 | 1.000 | ././. |
| long_form | C2_finbert_s2 | 10 | -0.022 | -0.0012 | +0.0086 | +0.0008 | +0.30 | -5.63 | 0.000 | -2.44 | +3.14 | 0.099 | -0.02 | +2.97 | 0.142 | Y/./. |
| long_form | C2_finbert_s2 | 20 | -0.051 | -0.0109 | +0.0212 | -0.0162 | +0.30 | -1.06 | 1.000 | -1.44 | -0.77 | 1.000 | -2.53 | +4.58 | 0.000 | ././. |
| long_form | C2_finbert_s3 | 5 | -0.054 | -0.0152 | +0.0040 | -0.0132 | +0.32 | -0.39 | 1.000 | -0.15 | -1.83 | 1.000 | -1.03 | +2.32 | 0.814 | ././. |
| long_form | C2_finbert_s3 | 10 | -0.048 | -0.0125 | +0.0053 | -0.0134 | +0.31 | -2.92 | 0.191 | -0.41 | +1.36 | 1.000 | -2.26 | -0.38 | 1.000 | ././. |
| long_form | C2_finbert_s3 | 20 | +0.116 | +0.0350 | -0.0268 | +0.0533 | +0.30 | -1.23 | 1.000 | -2.33 | +1.65 | 1.000 | -6.74 | +2.09 | 1.000 | ././. |
| long_form | C2_finbert_s4 | 5 | -0.022 | -0.0066 | +0.0005 | -0.0045 | +0.31 | -1.97 | 1.000 | -0.04 | -0.05 | 1.000 | -0.29 | -0.43 | 1.000 | ././. |
| long_form | C2_finbert_s4 | 10 | -0.007 | -0.0004 | +0.0034 | +0.0002 | +0.30 | -5.98 | 0.000 | -3.55 | +3.78 | 0.010 | -0.03 | +2.55 | 0.451 | Y/./. |
| long_form | C2_finbert_s4 | 20 | -0.179 | -0.0219 | +0.0494 | -0.0341 | +0.29 | -2.54 | 0.560 | -1.50 | +0.42 | 1.000 | -2.51 | +1.07 | 1.000 | ././. |
| long_form | C3_roberta_s1 | 5 | +0.000 | +0.0000 | -0.0000 | -0.0000 | +0.30 | -5.13 | 0.000 | -2.78 | +3.71 | 0.013 | -0.35 | +3.13 | 0.089 | Y/./. |
| long_form | C3_roberta_s1 | 10 | -0.038 | -0.0098 | +0.0047 | -0.0083 | +0.28 | -1.51 | 1.000 | -0.07 | -0.68 | 1.000 | -0.89 | +1.17 | 1.000 | ././. |
| long_form | C3_roberta_s1 | 20 | +0.617 | +0.0023 | -0.4362 | +0.0082 | +0.32 | -16.07 | 0.000 | -66.10 | +13.23 | 0.000 | +0.95 | -14.33 | 0.000 | Y/./Y |
| long_form | C4_longformer | 5 | -0.043 | -0.0068 | +0.0106 | -0.0043 | +0.31 | -1.90 | 1.000 | -0.46 | +0.74 | 1.000 | -0.27 | +0.57 | 1.000 | ././. |
| long_form | C4_longformer | 10 | -0.166 | +0.0205 | +0.0752 | +0.0384 | +0.30 | +1.78 | 1.000 | -0.40 | -0.07 | 1.000 | +2.35 | -0.47 | 1.000 | ././. |
| long_form | C4_longformer | 20 | -0.043 | -0.0049 | +0.0266 | -0.0069 | +0.30 | -1.92 | 1.000 | -2.02 | +0.23 | 1.000 | -1.12 | +3.21 | 0.070 | ././. |
| long_form | C6_llmtext | 5 | -0.035 | -0.0088 | +0.0008 | -0.0081 | +0.29 | -2.89 | 0.206 | -0.05 | +2.08 | 1.000 | -0.37 | -1.37 | 1.000 | ././. |
| long_form | C6_llmtext | 10 | -0.036 | -0.0121 | +0.0023 | -0.0105 | +0.31 | -3.12 | 0.106 | -0.08 | +1.74 | 1.000 | -0.60 | -0.71 | 1.000 | ././. |
| long_form | C6_llmtext | 20 | +0.003 | +0.0002 | +0.0000 | +0.0002 | +0.30 | -3.40 | 0.042 | +0.08 | -4.33 | 0.001 | +0.26 | -2.95 | 0.152 | Y/Y/. |
| long_form | D1_concat_mlp | 5 | -0.011 | +0.0078 | +0.0100 | +0.0092 | +0.28 | -0.14 | 1.000 | +0.25 | -0.65 | 1.000 | +0.69 | -0.48 | 1.000 | ././. |
| long_form | D1_concat_mlp | 10 | -0.013 | +0.0047 | +0.0134 | +0.0072 | +0.29 | +0.51 | 1.000 | +0.33 | +0.57 | 1.000 | +0.74 | +0.28 | 1.000 | ././. |
| long_form | D1_concat_mlp | 20 | +0.031 | +0.0014 | -0.0288 | +0.0000 | +0.30 | -10.63 | 0.000 | -6.02 | +9.49 | 0.000 | +0.00 | -10.21 | 0.000 | Y/./Y |
| long_form | D2_gated_fusion | 5 | +0.004 | +0.0006 | -0.0005 | +0.0002 | +0.29 | -3.00 | 0.152 | -0.24 | +1.65 | 1.000 | +0.06 | -2.57 | 0.437 | ././. |
| long_form | D2_gated_fusion | 10 | +0.109 | +0.0020 | -0.0316 | -0.0166 | +0.29 | -11.20 | 0.000 | -4.71 | +10.22 | 0.000 | -2.50 | +9.85 | 0.000 | Y/./. |
| long_form | D2_gated_fusion | 20 | -0.032 | +0.0207 | +0.0467 | +0.0245 | +0.31 | +3.31 | 0.055 | +1.53 | +1.45 | 1.000 | +3.61 | +0.74 | 1.000 | ././. |
| long_form | D4_llmfused | 5 | -0.030 | +0.0008 | -0.0000 | +0.0021 | +0.28 | -1.79 | 1.000 | -0.00 | +1.90 | 1.000 | +0.25 | +0.35 | 1.000 | ././. |
| long_form | D4_llmfused | 10 | +0.029 | +0.0027 | +0.0025 | +0.0016 | +0.29 | -1.76 | 1.000 | +0.23 | -1.90 | 1.000 | +0.54 | -4.52 | 0.000 | ././Y |
| long_form | D4_llmfused | 20 | -0.028 | -0.0023 | -0.0004 | -0.0014 | +0.31 | -2.54 | 0.560 | +0.04 | -1.79 | 1.000 | +0.20 | -2.31 | 0.815 | ././. |

## Per-cell detail — injected level 0.5%

| disc | model | h | delta | kappa(HAR) | kappa(firm) | kappa(pool) | rel%(HAR) | DM(HAR) | Holm | rel%(firm) | DM(firm) | Holm | rel%(pool) | DM(pool) | Holm | detect H/F/P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | -0.018 | -0.0041 | -0.0014 | -0.0031 | +0.51 | -0.52 | 1.000 | +0.17 | -0.69 | 1.000 | -0.31 | +2.09 | 1.000 | ././. |
| event_driven | B1_bow_ridge | 10 | -0.016 | -0.0037 | -0.0006 | -0.0028 | +0.51 | -0.79 | 1.000 | +0.14 | -2.28 | 0.947 | -0.70 | +3.18 | 0.065 | ././. |
| event_driven | B1_bow_ridge | 20 | -0.024 | -0.0059 | +0.0005 | -0.0043 | +0.51 | -0.71 | 1.000 | -0.11 | +3.05 | 0.113 | -1.02 | +3.66 | 0.012 | ././. |
| event_driven | B2_tfidf_ridge | 5 | -0.014 | -0.0035 | -0.0012 | -0.0027 | +0.50 | -0.23 | 1.000 | +0.20 | -0.65 | 1.000 | -0.40 | +3.12 | 0.079 | ././. |
| event_driven | B2_tfidf_ridge | 10 | -0.018 | -0.0045 | -0.0007 | -0.0034 | +0.48 | -0.16 | 1.000 | +0.14 | -1.55 | 1.000 | -0.72 | +3.98 | 0.004 | ././. |
| event_driven | B2_tfidf_ridge | 20 | -0.029 | -0.0078 | +0.0001 | -0.0060 | +0.48 | -0.26 | 1.000 | -0.02 | +2.30 | 0.930 | -1.17 | +4.18 | 0.002 | ././. |
| event_driven | B3_lm_linear | 5 | +0.002 | +0.0012 | +0.0005 | +0.0010 | +0.48 | -5.20 | 0.000 | +0.21 | -5.69 | 0.000 | +0.36 | -4.55 | 0.000 | Y/Y/Y |
| event_driven | B3_lm_linear | 10 | +0.002 | +0.0016 | +0.0006 | +0.0013 | +0.50 | -3.29 | 0.050 | +0.23 | -3.82 | 0.008 | +0.34 | -2.57 | 0.387 | ./Y/. |
| event_driven | B3_lm_linear | 20 | +0.002 | +0.0014 | +0.0005 | +0.0011 | +0.50 | -2.24 | 0.802 | +0.26 | -3.10 | 0.097 | +0.29 | -1.51 | 1.000 | ././. |
| event_driven | B4_lm_features | 5 | +0.001 | +0.0016 | +0.0018 | +0.0016 | +0.50 | -5.83 | 0.000 | +0.58 | -6.38 | 0.000 | +0.52 | -6.11 | 0.000 | Y/Y/Y |
| event_driven | B4_lm_features | 10 | +0.005 | +0.0021 | +0.0036 | +0.0023 | +0.50 | -12.38 | 0.000 | +0.81 | -12.05 | 0.000 | +0.53 | -11.99 | 0.000 | Y/Y/Y |
| event_driven | B4_lm_features | 20 | +0.003 | +0.0016 | +0.0007 | +0.0015 | +0.52 | -3.78 | 0.009 | +0.22 | -3.95 | 0.005 | +0.34 | -2.99 | 0.119 | Y/Y/. |
| event_driven | C2_finbert_s1 | 5 | -0.045 | -0.0105 | +0.0003 | -0.0065 | +0.48 | +0.77 | 1.000 | -0.01 | -1.22 | 1.000 | -0.55 | +5.43 | 0.000 | ././. |
| event_driven | C2_finbert_s1 | 10 | -0.043 | -0.0098 | +0.0025 | -0.0065 | +0.52 | +0.03 | 1.000 | -0.19 | +0.69 | 1.000 | -1.20 | +5.71 | 0.000 | ././. |
| event_driven | C2_finbert_s1 | 20 | -0.022 | -0.0063 | +0.0016 | -0.0050 | +0.52 | -0.81 | 1.000 | -0.34 | +1.55 | 1.000 | -2.46 | +3.46 | 0.025 | ././. |
| event_driven | C6_llmtext | 5 | -0.013 | -0.0035 | -0.0015 | -0.0025 | +0.51 | -1.13 | 1.000 | +0.22 | -1.22 | 1.000 | -0.11 | +1.94 | 1.000 | ././. |
| event_driven | C6_llmtext | 10 | -0.009 | -0.0025 | -0.0005 | -0.0017 | +0.50 | -1.21 | 1.000 | +0.14 | -1.95 | 1.000 | -0.22 | +2.28 | 0.824 | ././. |
| event_driven | C6_llmtext | 20 | -0.004 | -0.0010 | -0.0002 | -0.0008 | +0.49 | -1.25 | 1.000 | +0.18 | -2.83 | 0.223 | -0.06 | +0.66 | 1.000 | ././. |
| event_driven | D2_gated_fusion | 5 | -0.001 | -0.0002 | -0.0001 | -0.0002 | +0.52 | -1.40 | 1.000 | +0.21 | -1.86 | 1.000 | +0.27 | -0.62 | 1.000 | ././. |
| event_driven | D2_gated_fusion | 10 | +0.002 | +0.0016 | +0.0004 | -0.0002 | +0.48 | -3.26 | 0.056 | +0.12 | -3.73 | 0.011 | -0.06 | +4.44 | 0.001 | ./Y/. |
| event_driven | D2_gated_fusion | 20 | -0.027 | +0.0188 | +0.0447 | +0.0306 | +0.50 | -0.59 | 1.000 | -0.78 | +0.21 | 1.000 | +2.75 | -1.88 | 1.000 | ././. |
| event_driven | D4_llmfused | 5 | +1.146 | +0.0025 | +0.0158 | -0.0683 | +0.50 | -19.48 | 0.000 | +3.07 | -19.87 | 0.000 | -14.38 | +18.59 | 0.000 | Y/Y/. |
| event_driven | D4_llmfused | 10 | +0.177 | +0.0025 | +0.0046 | -0.0031 | +0.49 | -13.73 | 0.000 | +0.85 | -13.53 | 0.000 | -0.63 | +12.93 | 0.000 | Y/Y/. |
| event_driven | D4_llmfused | 20 | -0.074 | +0.0049 | +0.0046 | +0.0034 | +0.50 | -7.19 | 0.000 | +0.51 | -7.29 | 0.000 | +0.36 | -6.75 | 0.000 | Y/Y/Y |
| long_form | B1_bow_ridge | 5 | -0.062 | -0.0066 | +0.0005 | -0.0056 | +0.51 | -1.19 | 1.000 | -0.06 | -0.05 | 1.000 | -0.10 | +0.24 | 1.000 | ././. |
| long_form | B1_bow_ridge | 10 | -0.100 | -0.0059 | +0.0106 | -0.0037 | +0.51 | -1.17 | 1.000 | -1.12 | -0.55 | 1.000 | -0.07 | +0.93 | 1.000 | ././. |
| long_form | B1_bow_ridge | 20 | -0.198 | -0.0195 | +0.0287 | -0.0207 | +0.51 | -0.76 | 1.000 | -1.81 | -0.34 | 1.000 | -0.89 | +1.87 | 1.000 | ././. |
| long_form | B2_tfidf_ridge | 5 | -0.069 | -0.0164 | +0.0028 | -0.0145 | +0.50 | -0.99 | 1.000 | -0.16 | -0.46 | 1.000 | -0.89 | +1.03 | 1.000 | ././. |
| long_form | B2_tfidf_ridge | 10 | -0.109 | -0.0186 | +0.0189 | -0.0187 | +0.50 | -2.54 | 0.425 | -1.05 | +0.50 | 1.000 | -1.04 | +0.20 | 1.000 | ././. |
| long_form | B2_tfidf_ridge | 20 | -0.206 | -0.0422 | +0.0535 | -0.0624 | +0.51 | -1.37 | 1.000 | -2.28 | +0.38 | 1.000 | -3.78 | +2.03 | 1.000 | ././. |
| long_form | B3_lm_linear | 5 | +0.000 | +0.0000 | +0.0000 | +0.0000 | +0.49 | -2.58 | 0.389 | +0.21 | -1.81 | 1.000 | -0.22 | -1.49 | 1.000 | ././. |
| long_form | B3_lm_linear | 10 | -0.014 | -0.0080 | -0.0010 | -0.0073 | +0.51 | -2.70 | 0.290 | +0.10 | -1.68 | 1.000 | -0.52 | -1.18 | 1.000 | ././. |
| long_form | B3_lm_linear | 20 | -0.040 | -0.0234 | +0.0001 | -0.0246 | +0.50 | -0.62 | 1.000 | -0.01 | -0.22 | 1.000 | -1.05 | +1.41 | 1.000 | ././. |
| long_form | B4_lm_features | 5 | +0.017 | +0.0023 | -0.0034 | +0.0013 | +0.51 | -4.17 | 0.002 | -0.69 | +3.79 | 0.009 | +0.25 | -3.60 | 0.015 | Y/./Y |
| long_form | B4_lm_features | 10 | -0.044 | +0.0090 | +0.0244 | +0.0115 | +0.50 | -2.79 | 0.234 | +0.86 | -1.86 | 1.000 | +1.06 | -3.40 | 0.032 | ././Y |
| long_form | B4_lm_features | 20 | -0.104 | +0.0195 | +0.0674 | +0.0234 | +0.52 | -4.99 | 0.000 | +0.04 | -3.62 | 0.016 | +1.18 | -6.42 | 0.000 | Y/Y/Y |
| long_form | C1_bert_s1 | 5 | -0.031 | -0.0168 | -0.0076 | -0.0172 | +0.52 | -0.73 | 1.000 | +0.12 | +1.33 | 1.000 | -1.97 | +0.87 | 1.000 | ././. |
| long_form | C1_bert_s1 | 10 | -0.065 | +0.0084 | +0.0370 | +0.0164 | +0.49 | -0.87 | 1.000 | +0.88 | -0.79 | 1.000 | +2.46 | -2.66 | 0.311 | ././. |
| long_form | C1_bert_s1 | 20 | -0.112 | -0.0174 | +0.0430 | -0.0269 | +0.50 | -2.29 | 0.751 | -1.78 | +0.90 | 1.000 | -2.39 | +1.77 | 1.000 | ././. |
| long_form | C2_finbert_s1 | 5 | -0.024 | -0.0082 | +0.0012 | -0.0062 | +0.51 | -0.83 | 1.000 | -0.08 | -0.50 | 1.000 | -0.46 | +1.60 | 1.000 | ././. |
| long_form | C2_finbert_s1 | 10 | -0.088 | -0.0133 | +0.0170 | -0.0118 | +0.49 | -2.27 | 0.766 | -0.62 | +0.18 | 1.000 | -0.96 | +0.65 | 1.000 | ././. |
| long_form | C2_finbert_s1 | 20 | +0.038 | +0.0090 | -0.0092 | +0.0136 | +0.51 | -1.43 | 1.000 | -1.07 | +1.09 | 1.000 | -3.93 | +1.55 | 1.000 | ././. |
| long_form | C2_finbert_s2 | 5 | -0.023 | -0.0041 | +0.0072 | -0.0015 | +0.50 | -2.42 | 0.544 | -1.03 | +0.44 | 1.000 | -0.03 | -0.35 | 1.000 | ././. |
| long_form | C2_finbert_s2 | 10 | +0.000 | +0.0000 | -0.0000 | -0.0000 | +0.48 | -7.59 | 0.000 | -3.73 | +5.25 | 0.000 | -0.14 | +4.97 | 0.000 | Y/./. |
| long_form | C2_finbert_s2 | 20 | -0.044 | -0.0094 | +0.0182 | -0.0139 | +0.50 | -1.58 | 1.000 | -1.77 | -0.21 | 1.000 | -2.25 | +4.10 | 0.002 | ././. |
| long_form | C2_finbert_s3 | 5 | -0.049 | -0.0141 | +0.0037 | -0.0122 | +0.52 | -0.77 | 1.000 | -0.21 | -1.42 | 1.000 | -0.85 | +1.95 | 1.000 | ././. |
| long_form | C2_finbert_s3 | 10 | -0.043 | -0.0113 | +0.0048 | -0.0121 | +0.50 | -3.13 | 0.082 | -0.48 | +1.57 | 1.000 | -2.06 | -0.56 | 1.000 | ././. |
| long_form | C2_finbert_s3 | 20 | +0.121 | +0.0367 | -0.0281 | +0.0560 | +0.50 | -1.37 | 1.000 | -2.49 | +1.79 | 1.000 | -6.46 | +1.94 | 1.000 | ././. |
| long_form | C2_finbert_s4 | 5 | -0.018 | -0.0055 | +0.0004 | -0.0037 | +0.51 | -2.46 | 0.524 | -0.05 | +0.41 | 1.000 | -0.15 | -0.90 | 1.000 | ././. |
| long_form | C2_finbert_s4 | 10 | +0.015 | +0.0008 | -0.0069 | -0.0005 | +0.49 | -8.54 | 0.000 | -5.12 | +6.44 | 0.000 | -0.14 | +4.93 | 0.000 | Y/./. |
| long_form | C2_finbert_s4 | 20 | -0.166 | -0.0203 | +0.0459 | -0.0316 | +0.49 | -2.91 | 0.161 | -1.87 | +0.78 | 1.000 | -2.21 | +0.74 | 1.000 | ././. |
| long_form | C3_roberta_s1 | 5 | +0.039 | +0.0012 | -0.0110 | -0.0026 | +0.50 | -8.55 | 0.000 | -4.61 | +7.24 | 0.000 | -0.80 | +6.80 | 0.000 | Y/./. |
| long_form | C3_roberta_s1 | 10 | -0.033 | -0.0084 | +0.0040 | -0.0071 | +0.50 | -1.99 | 1.000 | -0.17 | -0.16 | 1.000 | -0.71 | +0.72 | 1.000 | ././. |
| long_form | C3_roberta_s1 | 20 | +1.029 | +0.0039 | -0.7269 | +0.0137 | +0.51 | -15.88 | 0.000 | -125.32 | +11.93 | 0.000 | +1.59 | -14.52 | 0.000 | Y/./Y |
| long_form | C4_longformer | 5 | -0.035 | -0.0057 | +0.0088 | -0.0035 | +0.51 | -2.65 | 0.325 | -0.75 | +1.53 | 1.000 | -0.15 | -0.20 | 1.000 | ././. |
| long_form | C4_longformer | 10 | -0.178 | +0.0219 | +0.0802 | +0.0410 | +0.51 | +1.26 | 1.000 | +0.29 | -0.53 | 1.000 | +2.74 | -0.90 | 1.000 | ././. |
| long_form | C4_longformer | 20 | -0.029 | -0.0033 | +0.0181 | -0.0047 | +0.50 | -2.72 | 0.281 | -2.97 | +1.11 | 1.000 | -0.85 | +2.32 | 0.764 | ././. |
| long_form | C6_llmtext | 5 | -0.030 | -0.0076 | +0.0007 | -0.0070 | +0.49 | -3.36 | 0.042 | -0.07 | +2.47 | 0.596 | -0.18 | -1.84 | 1.000 | Y/./. |
| long_form | C6_llmtext | 10 | -0.033 | -0.0109 | +0.0021 | -0.0095 | +0.50 | -3.56 | 0.021 | -0.12 | +2.10 | 1.000 | -0.44 | -1.05 | 1.000 | Y/./. |
| long_form | C6_llmtext | 20 | +0.023 | +0.0018 | +0.0004 | +0.0018 | +0.50 | -4.68 | 0.000 | +0.12 | -5.75 | 0.000 | +0.45 | -4.22 | 0.001 | Y/Y/Y |
| long_form | D1_concat_mlp | 5 | -0.013 | +0.0090 | +0.0115 | +0.0106 | +0.48 | -0.55 | 1.000 | +0.49 | -1.06 | 1.000 | +0.93 | -0.88 | 1.000 | ././. |
| long_form | D1_concat_mlp | 10 | -0.016 | +0.0061 | +0.0173 | +0.0092 | +0.51 | -0.33 | 1.000 | +0.91 | -0.22 | 1.000 | +1.07 | -0.61 | 1.000 | ././. |
| long_form | D1_concat_mlp | 20 | +0.070 | +0.0031 | -0.0657 | +0.0000 | +0.52 | -14.78 | 0.000 | -10.33 | +14.34 | 0.000 | +0.00 | -14.15 | 0.000 | Y/./Y |
| long_form | D2_gated_fusion | 5 | +0.012 | +0.0018 | -0.0016 | +0.0005 | +0.49 | -4.81 | 0.000 | -0.41 | +3.46 | 0.028 | +0.12 | -4.38 | 0.001 | Y/./Y |
| long_form | D2_gated_fusion | 10 | +0.186 | +0.0033 | -0.0537 | -0.0282 | +0.51 | -12.92 | 0.000 | -8.26 | +11.31 | 0.000 | -4.38 | +11.12 | 0.000 | Y/./. |
| long_form | D2_gated_fusion | 20 | -0.035 | +0.0223 | +0.0502 | +0.0263 | +0.51 | +3.15 | 0.079 | +1.90 | +1.29 | 1.000 | +3.82 | +0.59 | 1.000 | ././. |
| long_form | D4_llmfused | 5 | -0.076 | +0.0020 | -0.0000 | +0.0051 | +0.48 | -3.30 | 0.050 | -0.00 | +3.64 | 0.015 | +0.78 | -1.33 | 1.000 | Y/./. |
| long_form | D4_llmfused | 10 | +0.043 | +0.0041 | +0.0038 | +0.0024 | +0.51 | -2.45 | 0.524 | +0.43 | -2.67 | 0.357 | +0.67 | -5.19 | 0.000 | ././Y |
| long_form | D4_llmfused | 20 | -0.009 | -0.0008 | -0.0001 | -0.0005 | +0.51 | -3.24 | 0.059 | +0.07 | -2.51 | 0.553 | +0.31 | -2.88 | 0.162 | ././. |

## Per-cell detail — injected level 1.0%

| disc | model | h | delta | kappa(HAR) | kappa(firm) | kappa(pool) | rel%(HAR) | DM(HAR) | Holm | rel%(firm) | DM(firm) | Holm | rel%(pool) | DM(pool) | Holm | detect H/F/P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | -0.007 | -0.0016 | -0.0005 | -0.0012 | +1.02 | -2.26 | 0.530 | +0.35 | -2.40 | 0.511 | +0.07 | +0.32 | 1.000 | ././. |
| event_driven | B1_bow_ridge | 10 | -0.005 | -0.0012 | -0.0002 | -0.0009 | +1.00 | -2.46 | 0.337 | +0.21 | -3.93 | 0.004 | -0.33 | +1.59 | 1.000 | ./Y/. |
| event_driven | B1_bow_ridge | 20 | -0.013 | -0.0031 | +0.0003 | -0.0023 | +0.99 | -1.84 | 0.968 | -0.15 | +4.20 | 0.001 | -0.67 | +2.41 | 0.552 | ././. |
| event_driven | B2_tfidf_ridge | 5 | -0.004 | -0.0010 | -0.0003 | -0.0007 | +1.01 | -2.04 | 0.743 | +0.38 | -2.37 | 0.520 | -0.00 | +1.25 | 1.000 | ././. |
| event_driven | B2_tfidf_ridge | 10 | -0.007 | -0.0018 | -0.0003 | -0.0013 | +1.01 | -1.85 | 0.968 | +0.23 | -3.19 | 0.052 | -0.32 | +2.35 | 0.631 | ././. |
| event_driven | B2_tfidf_ridge | 20 | -0.018 | -0.0049 | +0.0001 | -0.0037 | +0.99 | -1.35 | 1.000 | -0.03 | +3.36 | 0.030 | -0.78 | +2.98 | 0.117 | ././. |
| event_driven | B3_lm_linear | 5 | +0.005 | +0.0037 | +0.0015 | +0.0031 | +0.99 | -9.67 | 0.000 | +0.41 | -10.13 | 0.000 | +0.78 | -9.02 | 0.000 | Y/Y/Y |
| event_driven | B3_lm_linear | 10 | +0.006 | +0.0041 | +0.0016 | +0.0034 | +1.00 | -5.85 | 0.000 | +0.42 | -6.23 | 0.000 | +0.75 | -5.15 | 0.000 | Y/Y/Y |
| event_driven | B3_lm_linear | 20 | +0.006 | +0.0043 | +0.0017 | +0.0036 | +1.01 | -4.22 | 0.001 | +0.44 | -4.85 | 0.000 | +0.72 | -3.57 | 0.016 | Y/Y/Y |
| event_driven | B4_lm_features | 5 | +0.004 | +0.0041 | +0.0048 | +0.0042 | +1.01 | -12.04 | 0.000 | +1.17 | -12.51 | 0.000 | +1.05 | -12.16 | 0.000 | Y/Y/Y |
| event_driven | B4_lm_features | 10 | +0.011 | +0.0047 | +0.0078 | +0.0049 | +1.00 | -14.77 | 0.000 | +1.61 | -14.54 | 0.000 | +1.05 | -14.32 | 0.000 | Y/Y/Y |
| event_driven | B4_lm_features | 20 | +0.008 | +0.0043 | +0.0018 | +0.0041 | +1.00 | -6.12 | 0.000 | +0.41 | -6.07 | 0.000 | +0.80 | -5.45 | 0.000 | Y/Y/Y |
| event_driven | C2_finbert_s1 | 5 | -0.034 | -0.0080 | +0.0002 | -0.0050 | +0.99 | -0.93 | 1.000 | -0.02 | +0.44 | 1.000 | -0.23 | +3.50 | 0.020 | ././. |
| event_driven | C2_finbert_s1 | 10 | -0.032 | -0.0072 | +0.0018 | -0.0048 | +1.01 | -1.33 | 1.000 | -0.31 | +2.00 | 1.000 | -0.87 | +4.61 | 0.000 | ././. |
| event_driven | C2_finbert_s1 | 20 | -0.012 | -0.0035 | +0.0009 | -0.0028 | +0.98 | -1.32 | 1.000 | -0.46 | +2.07 | 0.922 | -2.08 | +2.80 | 0.198 | ././. |
| event_driven | C6_llmtext | 5 | -0.004 | -0.0010 | -0.0004 | -0.0007 | +1.02 | -3.93 | 0.004 | +0.43 | -3.89 | 0.004 | +0.26 | -0.81 | 1.000 | Y/Y/. |
| event_driven | C6_llmtext | 10 | +0.000 | +0.0000 | +0.0000 | +0.0000 | +1.00 | -3.76 | 0.007 | +0.24 | -4.43 | 0.000 | +0.12 | +0.05 | 1.000 | Y/Y/. |
| event_driven | C6_llmtext | 20 | +0.008 | +0.0020 | +0.0004 | +0.0016 | +1.00 | -3.53 | 0.015 | +0.29 | -5.18 | 0.000 | +0.37 | -1.50 | 1.000 | Y/Y/. |
| event_driven | D2_gated_fusion | 5 | +0.006 | +0.0021 | +0.0008 | +0.0017 | +0.99 | -5.17 | 0.000 | +0.39 | -5.47 | 0.000 | +0.65 | -4.51 | 0.000 | Y/Y/Y |
| event_driven | D2_gated_fusion | 10 | +0.007 | +0.0043 | +0.0010 | -0.0005 | +1.02 | -8.96 | 0.000 | +0.24 | -9.17 | 0.000 | -0.12 | +11.01 | 0.000 | Y/Y/. |
| event_driven | D2_gated_fusion | 20 | -0.031 | +0.0217 | +0.0517 | +0.0354 | +1.00 | -1.17 | 1.000 | +0.30 | -0.34 | 1.000 | +3.54 | -2.44 | 0.539 | ././. |
| event_driven | D4_llmfused | 5 | +2.291 | +0.0051 | +0.0317 | -0.1367 | +1.01 | -19.69 | 0.000 | +6.15 | -20.28 | 0.000 | -30.18 | +18.26 | 0.000 | Y/Y/. |
| event_driven | D4_llmfused | 10 | +0.354 | +0.0051 | +0.0091 | -0.0061 | +0.98 | -14.66 | 0.000 | +1.72 | -14.68 | 0.000 | -1.23 | +13.91 | 0.000 | Y/Y/. |
| event_driven | D4_llmfused | 20 | -0.118 | +0.0078 | +0.0074 | +0.0055 | +1.01 | -10.78 | 0.000 | +0.97 | -10.48 | 0.000 | +0.71 | -10.20 | 0.000 | Y/Y/Y |
| long_form | B1_bow_ridge | 5 | -0.035 | -0.0037 | +0.0003 | -0.0031 | +1.01 | -2.35 | 0.438 | -0.09 | +1.09 | 1.000 | +0.33 | -0.95 | 1.000 | ././. |
| long_form | B1_bow_ridge | 10 | -0.047 | -0.0027 | +0.0049 | -0.0017 | +1.01 | -2.70 | 0.178 | -1.97 | +0.51 | 1.000 | +0.24 | -0.13 | 1.000 | ././. |
| long_form | B1_bow_ridge | 20 | -0.159 | -0.0156 | +0.0229 | -0.0165 | +1.01 | -1.60 | 1.000 | -2.44 | +0.46 | 1.000 | -0.39 | +1.10 | 1.000 | ././. |
| long_form | B2_tfidf_ridge | 5 | -0.057 | -0.0135 | +0.0023 | -0.0119 | +1.01 | -1.75 | 1.000 | -0.24 | +0.33 | 1.000 | -0.43 | +0.26 | 1.000 | ././. |
| long_form | B2_tfidf_ridge | 10 | -0.091 | -0.0154 | +0.0157 | -0.0156 | +1.01 | -3.51 | 0.015 | -1.52 | +1.41 | 1.000 | -0.54 | -0.63 | 1.000 | Y/./. |
| long_form | B2_tfidf_ridge | 20 | -0.188 | -0.0385 | +0.0488 | -0.0569 | +1.00 | -1.96 | 0.854 | -2.77 | +0.94 | 1.000 | -3.08 | +1.51 | 1.000 | ././. |
| long_form | B3_lm_linear | 5 | +0.005 | +0.0029 | +0.0007 | +0.0026 | +1.00 | -3.35 | 0.026 | +0.32 | -2.51 | 0.397 | +0.24 | -2.23 | 0.842 | Y/./. |
| long_form | B3_lm_linear | 10 | -0.009 | -0.0051 | -0.0007 | -0.0046 | +0.98 | -3.40 | 0.023 | +0.16 | -2.39 | 0.514 | -0.10 | -1.85 | 1.000 | Y/./. |
| long_form | B3_lm_linear | 20 | -0.034 | -0.0195 | +0.0001 | -0.0205 | +1.01 | -1.29 | 1.000 | -0.01 | +0.37 | 1.000 | -0.55 | +0.76 | 1.000 | ././. |
| long_form | B4_lm_features | 5 | +0.037 | +0.0051 | -0.0073 | +0.0028 | +0.98 | -10.12 | 0.000 | -1.34 | +10.14 | 0.000 | +0.51 | -9.76 | 0.000 | Y/./Y |
| long_form | B4_lm_features | 10 | -0.059 | +0.0121 | +0.0329 | +0.0155 | +0.99 | -4.62 | 0.000 | +2.09 | -3.52 | 0.019 | +1.67 | -4.94 | 0.000 | Y/Y/Y |
| long_form | B4_lm_features | 20 | -0.125 | +0.0234 | +0.0809 | +0.0281 | +1.00 | -6.27 | 0.000 | +1.40 | -4.86 | 0.000 | +1.72 | -7.61 | 0.000 | Y/Y/Y |
| long_form | C1_bert_s1 | 5 | -0.026 | -0.0141 | -0.0063 | -0.0144 | +0.99 | -1.13 | 1.000 | +0.32 | +0.93 | 1.000 | -1.47 | +0.46 | 1.000 | ././. |
| long_form | C1_bert_s1 | 10 | -0.089 | +0.0115 | +0.0508 | +0.0225 | +0.99 | -2.24 | 0.539 | +2.91 | -2.17 | 0.852 | +3.40 | -3.88 | 0.005 | ././Y |
| long_form | C1_bert_s1 | 20 | -0.087 | -0.0135 | +0.0333 | -0.0208 | +1.00 | -3.28 | 0.033 | -2.82 | +1.97 | 1.000 | -1.64 | +0.90 | 1.000 | Y/./. |
| long_form | C2_finbert_s1 | 5 | -0.016 | -0.0053 | +0.0007 | -0.0040 | +1.00 | -2.19 | 0.583 | -0.14 | +0.89 | 1.000 | -0.07 | +0.20 | 1.000 | ././. |
| long_form | C2_finbert_s1 | 10 | -0.067 | -0.0102 | +0.0130 | -0.0091 | +0.99 | -3.26 | 0.034 | -1.22 | +1.16 | 1.000 | -0.51 | -0.18 | 1.000 | Y/./. |
| long_form | C2_finbert_s1 | 20 | +0.055 | +0.0129 | -0.0131 | +0.0195 | +0.99 | -1.88 | 0.965 | -1.53 | +1.55 | 1.000 | -3.24 | +1.12 | 1.000 | ././. |
| long_form | C2_finbert_s2 | 5 | -0.007 | -0.0012 | +0.0021 | -0.0004 | +1.01 | -4.65 | 0.000 | -1.86 | +2.69 | 0.239 | +0.15 | -2.49 | 0.482 | Y/./. |
| long_form | C2_finbert_s2 | 10 | +0.059 | +0.0031 | -0.0229 | -0.0021 | +0.98 | -12.62 | 0.000 | -7.26 | +10.84 | 0.000 | -0.47 | +10.58 | 0.000 | Y/./. |
| long_form | C2_finbert_s2 | 20 | -0.026 | -0.0055 | +0.0106 | -0.0081 | +0.99 | -2.94 | 0.091 | -2.62 | +1.25 | 1.000 | -1.55 | +2.85 | 0.177 | ././. |
| long_form | C2_finbert_s3 | 5 | -0.040 | -0.0113 | +0.0030 | -0.0098 | +0.98 | -1.65 | 1.000 | -0.33 | -0.44 | 1.000 | -0.44 | +1.05 | 1.000 | ././. |
| long_form | C2_finbert_s3 | 10 | -0.031 | -0.0082 | +0.0035 | -0.0088 | +1.00 | -3.70 | 0.008 | -0.68 | +2.16 | 0.852 | -1.52 | -1.06 | 1.000 | Y/./. |
| long_form | C2_finbert_s3 | 20 | +0.136 | +0.0410 | -0.0314 | +0.0625 | +1.00 | -1.71 | 1.000 | -2.88 | +2.12 | 0.893 | -5.75 | +1.59 | 1.000 | ././. |
| long_form | C2_finbert_s4 | 5 | -0.009 | -0.0027 | +0.0002 | -0.0019 | +0.98 | -3.61 | 0.011 | -0.09 | +1.51 | 1.000 | +0.18 | -2.02 | 1.000 | Y/./. |
| long_form | C2_finbert_s4 | 10 | +0.074 | +0.0039 | -0.0345 | -0.0024 | +0.99 | -15.12 | 0.000 | -9.41 | +13.75 | 0.000 | -0.44 | +12.24 | 0.000 | Y/./. |
| long_form | C2_finbert_s4 | 20 | -0.134 | -0.0164 | +0.0370 | -0.0255 | +0.99 | -3.89 | 0.004 | -2.81 | +1.74 | 1.000 | -1.45 | -0.15 | 1.000 | Y/./. |
| long_form | C3_roberta_s1 | 5 | +0.137 | +0.0041 | -0.0384 | -0.0091 | +1.00 | -15.00 | 0.000 | -9.30 | +13.90 | 0.000 | -1.94 | +14.04 | 0.000 | Y/./. |
| long_form | C3_roberta_s1 | 10 | -0.021 | -0.0053 | +0.0025 | -0.0045 | +1.00 | -3.13 | 0.051 | -0.39 | +1.05 | 1.000 | -0.28 | -0.36 | 1.000 | ././. |
| long_form | C3_roberta_s1 | 20 | +2.057 | +0.0078 | -1.4539 | +0.0274 | +1.00 | -15.47 | 0.000 | -354.29 | +9.50 | 0.000 | +3.19 | -14.47 | 0.000 | Y/./Y |
| long_form | C4_longformer | 5 | -0.017 | -0.0027 | +0.0042 | -0.0017 | +1.01 | -4.52 | 0.000 | -1.50 | +3.47 | 0.021 | +0.17 | -2.14 | 1.000 | Y/./. |
| long_form | C4_longformer | 10 | -0.203 | +0.0250 | +0.0916 | +0.0469 | +1.00 | +0.14 | 1.000 | +1.86 | -1.51 | 1.000 | +3.61 | -1.82 | 1.000 | ././. |
| long_form | C4_longformer | 20 | +0.007 | +0.0008 | -0.0043 | +0.0011 | +1.01 | -4.76 | 0.000 | -5.52 | +3.30 | 0.037 | -0.16 | -0.04 | 1.000 | Y/./. |
| long_form | C6_llmtext | 5 | -0.018 | -0.0047 | +0.0004 | -0.0043 | +0.99 | -4.51 | 0.000 | -0.11 | +3.44 | 0.023 | +0.29 | -3.02 | 0.108 | Y/./. |
| long_form | C6_llmtext | 10 | -0.023 | -0.0078 | +0.0015 | -0.0068 | +1.00 | -4.76 | 0.000 | -0.21 | +3.10 | 0.068 | -0.01 | -2.01 | 1.000 | Y/./. |
| long_form | C6_llmtext | 20 | +0.075 | +0.0059 | +0.0012 | +0.0060 | +1.01 | -7.64 | 0.000 | +0.21 | -9.09 | 0.000 | +0.94 | -7.25 | 0.000 | Y/Y/Y |
| long_form | D1_concat_mlp | 5 | -0.017 | +0.0121 | +0.0154 | +0.0142 | +1.01 | -1.61 | 1.000 | +1.14 | -2.10 | 0.893 | +1.55 | -1.94 | 1.000 | ././. |
| long_form | D1_concat_mlp | 10 | -0.025 | +0.0092 | +0.0262 | +0.0140 | +1.01 | -2.12 | 0.654 | +2.24 | -1.88 | 1.000 | +1.81 | -2.43 | 0.539 | ././. |
| long_form | D1_concat_mlp | 20 | +0.158 | +0.0070 | -0.1479 | +0.0000 | +1.01 | -16.91 | 0.000 | -20.59 | +16.02 | 0.000 | +0.01 | -15.97 | 0.000 | Y/./Y |
| long_form | D2_gated_fusion | 5 | +0.031 | +0.0047 | -0.0041 | +0.0013 | +0.99 | -9.19 | 0.000 | -0.84 | +7.94 | 0.000 | +0.26 | -8.83 | 0.000 | Y/./Y |
| long_form | D2_gated_fusion | 10 | +0.360 | +0.0064 | -0.1043 | -0.0547 | +1.00 | -14.02 | 0.000 | -16.74 | +11.87 | 0.000 | -8.78 | +11.97 | 0.000 | Y/./. |
| long_form | D2_gated_fusion | 20 | -0.041 | +0.0262 | +0.0590 | +0.0309 | +0.99 | +2.74 | 0.162 | +2.81 | +0.90 | 1.000 | +4.34 | +0.23 | 1.000 | ././. |
| long_form | D4_llmfused | 5 | -0.197 | +0.0051 | -0.0000 | +0.0133 | +1.01 | -6.94 | 0.000 | -0.01 | +7.66 | 0.000 | +2.19 | -5.45 | 0.000 | Y/./Y |
| long_form | D4_llmfused | 10 | +0.075 | +0.0072 | +0.0067 | +0.0043 | +1.00 | -4.07 | 0.002 | +0.86 | -4.46 | 0.000 | +0.96 | -6.66 | 0.000 | Y/Y/Y |
| long_form | D4_llmfused | 20 | +0.038 | +0.0031 | +0.0005 | +0.0019 | +1.00 | -4.94 | 0.000 | +0.14 | -4.29 | 0.001 | +0.59 | -4.28 | 0.001 | Y/Y/Y |

## Caveats (read before citing)

1. **Oracle**: s is built from test labels; the calibrated deltas quantify detection power only, never achievable forecast gains.
2. **Transmission**: one f_synth per (cell, level), calibrated on the HAR stage; the firm/pool stages receive it through their own REAL validation-fit text loadings (kappa_stage = g_stage*delta, tabulated). A stage can therefore miss an injected signal either through statistical noise or through a small/opposite-signed deployed loading — both are properties of the cascade as actually run, which is what this calibration measures.
3. **delta<0 cells**: where the real effect already exceeds the target, signal is removed down to the target so every cell realises exactly X%; the recovery rate is then a clean power estimate at that effect size.
4. Stage FIRM/POOL evaluate on the 5-price inner-join panel (n_test2 <= n_test); the within-firm demeaning is exact on the HAR-stage test panel and carries over to the subset up to dropped rows (the firm reference is a single global loading on a firm-level regressor, so exact per-firm zero mean is not required for non-absorption).
5. MDE uses the normal-approximation (1.96+0.84) factor on the HAC daily SE; it is an 80%-power planning quantity, not a test.

## Bottom line

- at 0.3% injected signal the cascade recovers 12/69 (HAR) / 7/69 (firm) / 6/69 (pool), full conjunction 2/69; at 0.5% injected signal the cascade recovers 20/69 (HAR) / 11/69 (firm) / 12/69 (pool), full conjunction 6/69; at 1.0% injected signal the cascade recovers 41/69 (HAR) / 20/69 (firm) / 19/69 (pool), full conjunction 13/69.
- Median per-cell MDE is 0.82% rel-QLIKE (IQR [0.37, 1.27]); 14/69 cells are 80%-powered at 0.3%, 21/69 at 0.5%, 43/69 at 1.0%.
- Interpretation for the 0/69 headline: a genuinely firm-orthogonal signal of 1.0% would have been flagged by the HAR stage in 41/69 cells and survived the full conjunction in 13/69; at 0.3% the corresponding counts are 12/69 and 2/69. Observed real effects (-3.86% to +5.92%) must be read against the per-cell MDE table above: effects below their cell's MDE were never detectable at 80% power, and the conjunction's specificity (placebo-validated) is now complemented by measured sensitivity.
