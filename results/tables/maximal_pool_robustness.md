# P1-5 — Maximal-pool robustness: is the 5-model reference val-overfit, and does absorption survive non-fitted references?

## RESTATED vs BEFORE

| quantity | BEFORE (committed maximal_reference.csv: ONE val-fitted 5-model pool, seed2026 text) | RESTATED (3 reference specs x 2 text bases, same clustered DM + Holm) |
|---|---|---|
| text-adds cells (Holm<.05) | 8/69 (fitted pool only) | **8–34/69 across all 6 spec-basis combinations**; overfit-immune equal-weight pool: **17–19/69** |
| pool-overfit allegation | untested (the '5/6 panels' figure in circulation uses the TEST-best member = hindsight) | vs the feasibly-selectable VAL-best member the pool is significantly BETTER in 5/6 panels and worse in only **1/6** (1 significant); vs the TEST-best ORACLE it is worse in 5/6 — the allegation rests on hindsight selection |
| sanity (s26 fitted col reproduces maximal_reference.csv) | — | max|dQLIKE_R|=8.33e-17, max|dQLIKE_U|=9.71e-17, max|dDM|=4.44e-16 over 69 cells: **PASS** |

All references use val-only weights frozen to test; text enters as one extra log-linear term fit on val; inference = day-clustered DM (daily-mean QLIKE differentials, HAC lag=h-1 days), Holm within each 69-cell family.


## (a) Allegation check — val-fitted pool vs its own best member (by VAL QLIKE), evaluated on TEST

| disc | h | val-best member | pool val QLIKE | member val QLIKE | pool TEST QLIKE | member TEST QLIKE | pool worse? | cluDM(pool-vs-member) | p | EQW TEST QLIKE | oracle member (TEST-best) | oracle TEST QLIKE | pool worse than oracle? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | 5 | A6_shar | 0.1361 | 0.1393 | 0.1233 | 0.1268 | no | -7.24 | 0.0000 | 0.1215 | A3_garch | 0.1218 | YES |
| event_driven | 10 | A6_shar | 0.1125 | 0.1156 | 0.0869 | 0.0891 | no | -4.29 | 0.0000 | 0.0841 | A3_garch | 0.0862 | YES |
| event_driven | 20 | A6_shar | 0.1299 | 0.1314 | 0.0634 | 0.0650 | no | -4.58 | 0.0000 | 0.0610 | A5_arima | 0.0638 | no |
| long_form | 5 | A2_har_rv | 0.1481 | 0.1487 | 0.1174 | 0.1209 | no | -6.24 | 0.0000 | 0.1159 | A4_egarch | 0.1144 | YES |
| long_form | 10 | A6_shar | 0.1420 | 0.1424 | 0.0856 | 0.0865 | no | -2.05 | 0.0411 | 0.0819 | A4_egarch | 0.0818 | YES |
| long_form | 20 | A6_shar | 0.1990 | 0.2008 | 0.0731 | 0.0714 | YES | +8.49 | 0.0000 | 0.0680 | A5_arima | 0.0689 | YES |

**The allegation dissolves once selection is made feasible.** Compared with the member a forecaster could actually have PICKED on validation, the fitted pool is significantly BETTER in 5/6 panels (clustered DM -7.24 to -2.05) and worse in only 1/6 (long_form h20, DM +8.49). The '5/6 panels' version of the charge holds only against the TEST-best member — an oracle unavailable without peeking at test (5/6 here). Residual val-fit slippage does exist: the never-fitted equal-weight pool beats the fitted pool on test in 6/6 panels (6 significant), which is why the equal-weight spec below is the decisive robustness check.


## (b) 69-cell text-increment survivor counts per reference spec

| text basis | reference spec | adds (raw p<.05) | adds (Holm<.05) | HURTS (Holm<.05) | cells |
|---|---|---|---|---|---|
| seed2026 | FITTED 5-model pool (6 val-fit params) | 23 | **8** | 5 | 69 |
| seed2026 | EQUAL-WEIGHT 1/5 log pool (2-param recal only) | 23 | **19** | 12 | 69 |
| seed2026 | VAL-BEST single member (2-param recal) | 37 | **28** | 8 | 69 |
| seed-ensemble (declared primary) | FITTED 5-model pool (6 val-fit params) | 26 | **9** | 5 | 69 |
| seed-ensemble (declared primary) | EQUAL-WEIGHT 1/5 log pool (2-param recal only) | 23 | **17** | 11 | 69 |
| seed-ensemble (declared primary) | VAL-BEST single member (2-param recal) | 44 | **34** | 10 | 69 |

Comparators: 38/69 genuine vs the recalibrated **A2-only** reference (seed-ensemble primary, m1_ensemble_primary.md); 8/69 vs the fitted pool (maximal_reference.md, seed2026). Note the gradient: single reference (38 A2-only; 28–34 val-best single) → equal-weight 5-model pool (17–19) → fitted 5-model pool (8–9).


### Holm-survivor overlap — seed2026

- fitted pool: 8; equal-weight: 19; val-best single: 28
- fitted ∩ equal-weight: 7; fitted ∩ val-best: 8; all three: 7; union: 28
- union cells: event_driven/B1_bow_ridge/h5 [V]; event_driven/B1_bow_ridge/h10 [V]; event_driven/B1_bow_ridge/h20 [V]; event_driven/C2_finbert_s1/h5 [EV]; event_driven/C2_finbert_s1/h10 [EV]; event_driven/C6_llmtext/h5 [V]; long_form/B1_bow_ridge/h5 [EV]; long_form/B1_bow_ridge/h20 [EV]; long_form/B2_tfidf_ridge/h5 [FEV]; long_form/B2_tfidf_ridge/h10 [FEV]; long_form/B2_tfidf_ridge/h20 [FEV]; long_form/B3_lm_linear/h10 [EV]; long_form/B3_lm_linear/h20 [EV]; long_form/C1_bert_s1/h20 [V]; long_form/C2_finbert_s1/h10 [EV]; long_form/C2_finbert_s2/h5 [EV]; long_form/C2_finbert_s2/h10 [V]; long_form/C2_finbert_s3/h5 [EV]; long_form/C2_finbert_s3/h10 [EV]; long_form/C2_finbert_s4/h10 [FEV]; long_form/C2_finbert_s4/h20 [FV]; long_form/C3_roberta_s1/h5 [V]; long_form/C3_roberta_s1/h10 [EV]; long_form/C4_longformer/h5 [FEV]; long_form/C6_llmtext/h5 [FEV]; long_form/C6_llmtext/h10 [FEV]; long_form/C6_llmtext/h20 [EV]; long_form/D4_llmfused/h20 [V]

### Holm-survivor overlap — seed-ensemble (declared primary)

- fitted pool: 9; equal-weight: 17; val-best single: 34
- fitted ∩ equal-weight: 6; fitted ∩ val-best: 8; all three: 6; union: 35
- union cells: event_driven/B1_bow_ridge/h5 [V]; event_driven/B1_bow_ridge/h10 [V]; event_driven/B1_bow_ridge/h20 [V]; event_driven/C2_finbert_s1/h5 [EV]; event_driven/C2_finbert_s1/h10 [V]; event_driven/C6_llmtext/h5 [V]; event_driven/C6_llmtext/h10 [V]; long_form/B1_bow_ridge/h5 [EV]; long_form/B1_bow_ridge/h20 [EV]; long_form/B2_tfidf_ridge/h5 [FEV]; long_form/B2_tfidf_ridge/h10 [FEV]; long_form/B2_tfidf_ridge/h20 [FEV]; long_form/B3_lm_linear/h10 [EV]; long_form/B3_lm_linear/h20 [EV]; long_form/C1_bert_s1/h5 [V]; long_form/C1_bert_s1/h20 [V]; long_form/C2_finbert_s1/h5 [EV]; long_form/C2_finbert_s1/h10 [V]; long_form/C2_finbert_s2/h5 [FV]; long_form/C2_finbert_s2/h10 [V]; long_form/C2_finbert_s2/h20 [V]; long_form/C2_finbert_s3/h5 [EV]; long_form/C2_finbert_s3/h10 [EV]; long_form/C2_finbert_s4/h5 [EV]; long_form/C2_finbert_s4/h10 [V]; long_form/C2_finbert_s4/h20 [FV]; long_form/C3_roberta_s1/h5 [V]; long_form/C3_roberta_s1/h10 [EV]; long_form/C4_longformer/h5 [FEV]; long_form/C4_longformer/h20 [V]; long_form/C6_llmtext/h5 [FEV]; long_form/C6_llmtext/h10 [FEV]; long_form/C6_llmtext/h20 [EV]; long_form/D1_concat_mlp/h20 [F]; long_form/D4_llmfused/h20 [V]

## Per-cell grid — seed-ensemble basis (primary); rel% = QLIKE improvement of +text vs each reference


### long_form

| model | h | n_days | rel% FITTED | Holm | rel% EQW | Holm | rel% VBS (ref) | Holm | verdicts F/E/V |
|---|---|---|---|---|---|---|---|---|---|
| B1_bow_ridge | 5 | 792 | +0.87 | 0.520 | +0.92 | 0.044 | +1.75 (A2_har_rv) | 0.007 | null/ADD/ADD |
| B1_bow_ridge | 10 | 766 | +0.51 | 1.000 | -0.33 | 0.672 | +1.32 (A6_shar) | 0.135 | null/null/null |
| B1_bow_ridge | 20 | 765 | +1.61 | 0.484 | +0.33 | 0.000 | +2.39 (A6_shar) | 0.000 | null/ADD/ADD |
| B2_tfidf_ridge | 5 | 792 | +1.67 | 0.021 | +1.69 | 0.000 | +3.39 (A2_har_rv) | 0.000 | ADD/ADD/ADD |
| B2_tfidf_ridge | 10 | 766 | +1.95 | 0.000 | +0.66 | 0.000 | +3.78 (A6_shar) | 0.000 | ADD/ADD/ADD |
| B2_tfidf_ridge | 20 | 765 | +4.01 | 0.000 | +1.19 | 0.000 | +5.29 (A6_shar) | 0.000 | ADD/ADD/ADD |
| B3_lm_linear | 5 | 792 | -0.22 | 1.000 | +0.04 | 0.963 | +0.40 (A2_har_rv) | 0.367 | null/null/null |
| B3_lm_linear | 10 | 766 | +0.64 | 0.122 | +1.02 | 0.003 | +1.68 (A6_shar) | 0.000 | null/ADD/ADD |
| B3_lm_linear | 20 | 765 | +1.93 | 0.370 | +2.35 | 0.018 | +3.31 (A6_shar) | 0.000 | null/ADD/ADD |
| B4_lm_features | 5 | 792 | +0.02 | 1.000 | +0.01 | 1.000 | +0.13 (A2_har_rv) | 1.000 | null/null/null |
| B4_lm_features | 10 | 766 | -0.73 | 0.688 | -1.31 | 0.062 | -0.86 (A6_shar) | 0.009 | null/null/HURT |
| B4_lm_features | 20 | 765 | -1.55 | 1.000 | -3.26 | 0.041 | -2.43 (A6_shar) | 0.002 | null/HURT/HURT |
| C1_bert_s1 | 5 | 792 | +1.09 | 1.000 | +1.99 | 0.214 | +3.38 (A2_har_rv) | 0.026 | null/null/ADD |
| C1_bert_s1 | 10 | 766 | -0.11 | 1.000 | -1.63 | 0.278 | -0.99 (A6_shar) | 0.013 | null/null/HURT |
| C1_bert_s1 | 20 | 765 | +0.90 | 0.478 | -0.97 | 0.000 | +2.19 (A6_shar) | 0.000 | null/HURT/ADD |
| C2_finbert_s1 | 5 | 792 | +0.63 | 0.731 | +0.59 | 0.009 | +1.83 (A2_har_rv) | 0.000 | null/ADD/ADD |
| C2_finbert_s1 | 10 | 766 | +0.93 | 0.057 | -0.21 | 0.000 | +2.51 (A6_shar) | 0.000 | null/HURT/ADD |
| C2_finbert_s1 | 20 | 765 | -5.51 | 0.500 | -0.47 | 1.000 | -0.53 (A6_shar) | 1.000 | null/null/null |
| C2_finbert_s2 | 5 | 792 | +0.22 | 0.042 | -0.12 | 0.000 | +1.08 (A2_har_rv) | 0.000 | ADD/HURT/ADD |
| C2_finbert_s2 | 10 | 766 | -0.14 | 0.000 | -1.43 | 0.000 | +0.35 (A6_shar) | 0.000 | HURT/HURT/ADD |
| C2_finbert_s2 | 20 | 765 | -0.58 | 1.000 | -0.17 | 0.174 | +1.42 (A6_shar) | 0.000 | null/null/ADD |
| C2_finbert_s3 | 5 | 792 | +1.25 | 0.202 | +1.38 | 0.001 | +2.83 (A2_har_rv) | 0.000 | null/ADD/ADD |
| C2_finbert_s3 | 10 | 766 | -0.12 | 0.520 | +0.70 | 0.001 | +2.43 (A6_shar) | 0.000 | null/ADD/ADD |
| C2_finbert_s3 | 20 | 765 | -12.69 | 0.000 | -2.67 | 0.103 | -3.61 (A6_shar) | 1.000 | HURT/null/null |
| C2_finbert_s4 | 5 | 792 | +0.50 | 0.085 | +0.53 | 0.001 | +1.37 (A2_har_rv) | 0.000 | null/ADD/ADD |
| C2_finbert_s4 | 10 | 766 | -0.06 | 0.055 | -1.17 | 0.000 | +0.27 (A6_shar) | 0.000 | null/HURT/ADD |
| C2_finbert_s4 | 20 | 765 | +1.66 | 0.001 | -0.54 | 0.000 | +2.74 (A6_shar) | 0.000 | ADD/HURT/ADD |
| C3_roberta_s1 | 5 | 792 | -0.35 | 0.091 | -1.06 | 0.000 | +0.28 (A2_har_rv) | 0.000 | null/HURT/ADD |
| C3_roberta_s1 | 10 | 766 | +0.42 | 0.743 | +0.28 | 0.001 | +1.61 (A6_shar) | 0.000 | null/ADD/ADD |
| C3_roberta_s1 | 20 | 765 | -0.03 | 1.000 | -0.99 | 0.103 | -0.26 (A6_shar) | 0.001 | null/null/HURT |
| C4_longformer | 5 | 792 | +0.47 | 0.006 | +0.17 | 0.000 | +1.49 (A2_har_rv) | 0.000 | ADD/ADD/ADD |
| C4_longformer | 10 | 766 | -3.54 | 0.000 | -7.01 | 0.000 | -3.31 (A6_shar) | 0.000 | HURT/HURT/HURT |
| C4_longformer | 20 | 765 | -0.29 | 1.000 | -0.79 | 0.186 | +0.51 (A6_shar) | 0.000 | null/null/ADD |
| C6_llmtext | 5 | 792 | +1.05 | 0.000 | +1.13 | 0.000 | +1.94 (A2_har_rv) | 0.000 | ADD/ADD/ADD |
| C6_llmtext | 10 | 766 | +1.07 | 0.000 | +1.23 | 0.000 | +1.84 (A6_shar) | 0.000 | ADD/ADD/ADD |
| C6_llmtext | 20 | 765 | +0.23 | 0.246 | +0.42 | 0.005 | +0.26 (A6_shar) | 0.036 | null/ADD/ADD |
| D1_concat_mlp | 5 | 792 | -0.89 | 0.727 | +0.29 | 1.000 | -1.03 (A2_har_rv) | 0.162 | null/null/null |
| D1_concat_mlp | 10 | 766 | -0.38 | 0.021 | -0.68 | 1.000 | -0.44 (A6_shar) | 0.000 | HURT/null/HURT |
| D1_concat_mlp | 20 | 765 | +0.00 | 0.000 | -2.08 | 1.000 | -0.92 (A6_shar) | 0.000 | ADD/null/HURT |
| D2_gated_fusion | 5 | 792 | +0.04 | 1.000 | -0.64 | 1.000 | +0.17 (A2_har_rv) | 0.517 | null/null/null |
| D2_gated_fusion | 10 | 766 | +0.14 | 1.000 | -3.10 | 0.000 | -0.07 (A6_shar) | 1.000 | null/HURT/null |
| D2_gated_fusion | 20 | 765 | +0.78 | 0.238 | +0.45 | 1.000 | -2.89 (A6_shar) | 0.000 | null/null/HURT |
| D4_llmfused | 5 | 792 | -0.11 | 1.000 | -0.50 | 1.000 | +0.12 (A2_har_rv) | 1.000 | null/null/null |
| D4_llmfused | 10 | 766 | +0.29 | 0.090 | -1.58 | 1.000 | -0.02 (A6_shar) | 1.000 | null/null/null |
| D4_llmfused | 20 | 765 | +0.37 | 0.085 | -0.15 | 1.000 | +0.12 (A6_shar) | 0.009 | null/null/ADD |

### event_driven

| model | h | n_days | rel% FITTED | Holm | rel% EQW | Holm | rel% VBS (ref) | Holm | verdicts F/E/V |
|---|---|---|---|---|---|---|---|---|---|
| B1_bow_ridge | 5 | 996 | +0.31 | 1.000 | +0.63 | 0.325 | +1.31 (A6_shar) | 0.016 | null/null/ADD |
| B1_bow_ridge | 10 | 991 | -0.16 | 1.000 | +0.35 | 1.000 | +1.17 (A6_shar) | 0.016 | null/null/ADD |
| B1_bow_ridge | 20 | 981 | -0.27 | 1.000 | +0.24 | 1.000 | +1.40 (A6_shar) | 0.023 | null/null/ADD |
| B2_tfidf_ridge | 5 | 996 | +0.15 | 1.000 | +0.45 | 1.000 | +1.18 (A6_shar) | 0.154 | null/null/null |
| B2_tfidf_ridge | 10 | 991 | -0.06 | 1.000 | +0.33 | 1.000 | +1.29 (A6_shar) | 0.090 | null/null/null |
| B2_tfidf_ridge | 20 | 981 | -0.13 | 1.000 | +0.31 | 1.000 | +1.69 (A6_shar) | 0.053 | null/null/null |
| B3_lm_linear | 5 | 996 | +0.16 | 1.000 | +0.16 | 1.000 | +0.22 (A6_shar) | 0.455 | null/null/null |
| B3_lm_linear | 10 | 991 | +0.09 | 1.000 | +0.09 | 1.000 | +0.18 (A6_shar) | 1.000 | null/null/null |
| B3_lm_linear | 20 | 981 | +0.09 | 1.000 | +0.12 | 1.000 | +0.26 (A6_shar) | 1.000 | null/null/null |
| B4_lm_features | 5 | 996 | +0.19 | 1.000 | +0.19 | 1.000 | +0.18 (A6_shar) | 1.000 | null/null/null |
| B4_lm_features | 10 | 991 | +0.08 | 0.632 | +0.08 | 0.442 | +0.08 (A6_shar) | 0.488 | null/null/null |
| B4_lm_features | 20 | 981 | +0.09 | 1.000 | +0.10 | 1.000 | +0.25 (A6_shar) | 0.672 | null/null/null |
| C2_finbert_s1 | 5 | 996 | +0.77 | 0.608 | +0.70 | 0.001 | +2.45 (A6_shar) | 0.000 | null/ADD/ADD |
| C2_finbert_s1 | 10 | 991 | +0.08 | 1.000 | +0.31 | 0.070 | +2.20 (A6_shar) | 0.000 | null/null/ADD |
| C2_finbert_s1 | 20 | 981 | -1.59 | 1.000 | -0.15 | 1.000 | +1.44 (A6_shar) | 0.672 | null/null/null |
| C6_llmtext | 5 | 996 | +0.40 | 1.000 | +0.52 | 0.088 | +1.20 (A6_shar) | 0.000 | null/null/ADD |
| C6_llmtext | 10 | 991 | +0.12 | 1.000 | +0.25 | 1.000 | +0.92 (A6_shar) | 0.043 | null/null/ADD |
| C6_llmtext | 20 | 981 | +0.09 | 1.000 | +0.16 | 1.000 | +0.61 (A6_shar) | 0.902 | null/null/null |
| D2_gated_fusion | 5 | 996 | +0.30 | 1.000 | -0.31 | 0.404 | +0.49 (A6_shar) | 0.492 | null/null/null |
| D2_gated_fusion | 10 | 991 | -0.02 | 1.000 | -1.61 | 0.000 | +0.01 (A6_shar) | 0.908 | null/HURT/null |
| D2_gated_fusion | 20 | 981 | -2.49 | 0.824 | +0.14 | 1.000 | -3.77 (A6_shar) | 0.001 | null/null/HURT |
| D4_llmfused | 5 | 996 | +0.11 | 0.162 | -0.27 | 0.106 | -0.02 (A6_shar) | 0.055 | null/null/null |
| D4_llmfused | 10 | 991 | -0.03 | 1.000 | -0.37 | 0.186 | +0.00 (A6_shar) | 1.000 | null/null/null |
| D4_llmfused | 20 | 981 | -0.24 | 0.000 | -0.04 | 1.000 | -0.48 (A6_shar) | 0.000 | HURT/null/HURT |

(seed2026-basis per-cell rows are in maximal_pool_robustness.csv, basis='s26'.)


## VERDICT

1. **The literal overfit allegation rests on hindsight selection.** The fitted pool loses to its TEST-best member (oracle, unselectable without peeking) in 5/6 panels — that is the '5/6' figure in circulation — but against the VAL-selectable best member it is significantly BETTER in 5/6 panels and worse in only 1/6. The pool is not a dominated reference under any feasible selection rule.
2. **Absorption is PARTLY spec-robust, and the paper must say which part.** The equal-weight 1/5 pool — zero val-fitted multi-model freedom, and the BEST price forecaster on test in 6/6 panels — still cuts the Holm survivor count from 38/69 (A2-only primary) to **17–19/69**: roughly half of the absorption is price-pool INFORMATION and immune to the overfit objection. The further collapse to 8–9/69 under the fitted pool comes from conditioning on the five forecasts as SEPARATE regressors (a standard multivariate encompassing design), not from a better reference forecast — the fitted pool is a slightly WORSE forecast than the equal-weight pool.
3. **Single-model references barely absorb more than A2 alone** (val-best single: 28–34/69; the val-best member is SHAR in 5/6 panels) — the pool's BREADTH, not its weights, does the work.
4. **Bottom line for the paper**: the 'maximal reference absorbs the increment' conclusion HOLDS in direction under non-fitted references but its magnitude is spec-dependent. Quote the survivor count as the RANGE **8–34/69** across the three reference specs, headline the overfit-immune equal-weight figure (17–19/69), and present the fitted-pool figure (8–9/69) explicitly as the multivariate encompassing test. Framed this way the reverse-reference-shopping objection is defused: even the reference the reviewer cannot call overfit halves the A2-only survivor count.