# P1-b — VIX control on the M1 incremental-text finding

## RESTATED vs ORIGINAL

| | ORIGINAL (obs-level DM, no VIX) | RESTATED (day-clustered DM) | RESTATED (day-clustered DM + VIX-augmented reference) |
|---|---|---|---|
| Significant text increment (Holm<.05, placebo-clean) | 38/69 cells (of which 38 re-tested here) | 29/38 of the original genuine cells survive | 19/38 of the original genuine cells survive |
| All 40 re-tested cells (38 genuine + C6 h20 x2) | — | 30/40 | 19/40 (both filters: 17) |

**Reading:** ORIGINAL inference ran HAC(lag=h-1) over observation order — with ~10-25 same-day filings sharing market shocks, those t-stats are inflated. RESTATED collapses losses to daily means (equal weight per day), runs DM with HAC lag = h-1 DAYS, and additionally puts log VIX (point-in-time: last close STRICTLY BEFORE the label-window start, effective_trading_day) into the recalibrated reference, so the text forecast must add information BEYOND market-level implied volatility. The within-date placebo logic predicted the increment survives a pure 'when' control; the table below shows cell-by-cell whether it does.

**VIX data:** FRED VIXCLS, 4073 rows 2009-12-01..2025-12-31 (results/tables/_vix_daily.csv). Validation: 2020-03 peak=82.69 (>80 OK), 2017 low=9.14 (~9 OK).

## (a) Text increment under the VIX-augmented recalibrated reference (log space, val-fit)

DM sign: negative = text-augmented f_U_vix BETTER than reference f_R_vix. n_days = clustered sample size. `c_vix_ref` = log-VIX loading in the reference; `g_text_vix` = text elasticity given HAR+VIX. Holm within this table. `survives_vix` requires DM<0, Holm<.05 and |placebo|<2.

| disc | model | h | n_test | n_days | orig DM | orig Holm | clu DM | clu Holm | VIX DM | VIX p | VIX Holm | rel%(vix) | g_text | c_vix | MBB 95% CI (daily dQLIKE) | placebo DM | orig genuine | surv. clu | surv. VIX |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | 25109 | 996 | -7.43 | 4.89e-12 | -3.35 | 0.0134 | -1.38 | 0.1682 | 1 | +1.14 | +0.313 | +0.286 | [-0.00199, +0.00035] | +0.23 | Y | Y | no |
| event_driven | B1_bow_ridge | 10 | 25001 | 991 | -6.01 | 5.7e-08 | -3.25 | 0.0178 | -1.77 | 0.0775 | 0.698 | +1.38 | +0.313 | +0.224 | [-0.00172, +0.00010] | -0.07 | Y | Y | no |
| event_driven | B1_bow_ridge | 20 | 24732 | 981 | -6.50 | 2.71e-09 | -3.10 | 0.0254 | -5.33 | 0.0000 | 4e-06 | +2.94 | +0.258 | +0.010 | [-0.00229, -0.00107] | +0.55 | Y | Y | **Y** |
| event_driven | B2_tfidf_ridge | 5 | 25109 | 996 | -7.05 | 7.04e-11 | -2.76 | 0.0589 | -0.81 | 0.4188 | 1 | +0.95 | +0.354 | +0.286 | [-0.00157, +0.00063] | +0.32 | Y | no | no |
| event_driven | B2_tfidf_ridge | 10 | 25001 | 991 | -7.21 | 2.26e-11 | -2.97 | 0.034 | -1.90 | 0.0584 | 0.642 | +1.62 | +0.328 | +0.224 | [-0.00167, +0.00009] | +0.34 | Y | Y | no |
| event_driven | B2_tfidf_ridge | 20 | 24732 | 981 | -8.22 | 9.65e-15 | -3.11 | 0.0254 | -5.72 | 0.0000 | 4.82e-07 | +3.36 | +0.285 | +0.010 | [-0.00246, -0.00123] | +0.54 | Y | Y | **Y** |
| event_driven | B3_lm_linear | 5 | 25109 | 996 | -4.82 | 3.27e-05 | -2.43 | 0.137 | -2.23 | 0.0262 | 0.367 | +0.30 | +0.892 | +0.286 | [-0.00054, -0.00003] | -0.26 | Y | no | no |
| event_driven | B3_lm_linear | 10 | 25001 | 991 | -3.03 | 0.0361 | -1.23 | 1 | -1.58 | 0.1145 | 0.916 | +0.29 | +0.895 | +0.224 | [-0.00052, +0.00004] | -0.54 | Y | no | no |
| event_driven | B4_lm_features | 5 | 25109 | 996 | -3.95 | 0.00152 | -0.99 | 1 | -0.60 | 0.5500 | 1 | +0.18 | +1.043 | +0.286 | [-0.00018, +0.00010] | +0.52 | Y | no | no |
| event_driven | B4_lm_features | 10 | 25001 | 991 | -3.77 | 0.00298 | -2.10 | 0.285 | -1.88 | 0.0606 | 0.642 | +0.07 | +0.354 | +0.224 | [-0.00008, +0.00000] | -0.47 | Y | no | no |
| event_driven | B4_lm_features | 20 | 24732 | 981 | -3.11 | 0.03 | -2.01 | 0.313 | -2.72 | 0.0066 | 0.132 | +0.35 | +0.577 | +0.010 | [-0.00043, -0.00009] | +0.39 | Y | no | no |
| event_driven | C2_finbert_s1 | 10 | 25001 | 991 | -14.19 | 9.91e-44 | -5.52 | 1.36e-06 | -3.26 | 0.0012 | 0.0255 | +3.08 | +0.286 | +0.224 | [-0.00285, -0.00061] | -0.29 | Y | Y | **Y** |
| event_driven | C6_llmtext | 5 | 25109 | 996 | -9.86 | 3.81e-21 | -5.04 | 1.3e-05 | -5.38 | 0.0000 | 3.1e-06 | +1.44 | +0.298 | +0.286 | [-0.00186, -0.00087] | +0.49 | Y | Y | **Y** |
| event_driven | C6_llmtext | 10 | 25001 | 991 | -7.98 | 6.68e-14 | -3.76 | 0.00327 | -3.48 | 0.0005 | 0.013 | +1.12 | +0.311 | +0.224 | [-0.00110, -0.00031] | +0.35 | Y | Y | **Y** |
| event_driven | C6_llmtext | 20 | 24732 | 981 | -5.39 | 1.71e-06 | -1.98 | 0.313 | -2.28 | 0.0229 | 0.344 | +0.79 | +0.246 | +0.010 | [-0.00071, -0.00006] | +1.09 | - | no | no |
| long_form | B1_bow_ridge | 5 | 7951 | 809 | -6.75 | 5.47e-10 | -3.83 | 0.00258 | -1.36 | 0.1733 | 1 | +1.67 | +0.153 | +0.431 | [-0.00247, +0.00048] | +0.14 | Y | Y | no |
| long_form | B1_bow_ridge | 10 | 7933 | 803 | -9.18 | 2.77e-18 | -4.15 | 0.000744 | -2.48 | 0.0134 | 0.228 | +2.91 | +0.116 | +0.388 | [-0.00270, -0.00027] | +0.47 | Y | Y | no |
| long_form | B1_bow_ridge | 20 | 7902 | 794 | -13.08 | 7e-37 | -5.45 | 2.06e-06 | -6.67 | 0.0000 | 1.74e-09 | +4.45 | +0.108 | +0.036 | [-0.00383, -0.00205] | -0.28 | Y | Y | **Y** |
| long_form | B2_tfidf_ridge | 5 | 7951 | 809 | -11.34 | 8.45e-28 | -5.39 | 2.58e-06 | -2.57 | 0.0104 | 0.187 | +3.59 | +0.319 | +0.431 | [-0.00479, -0.00061] | +0.13 | Y | Y | no |
| long_form | B2_tfidf_ridge | 10 | 7933 | 803 | -15.30 | 2.82e-50 | -8.89 | 1.56e-16 | -6.65 | 0.0000 | 2e-09 | +5.57 | +0.266 | +0.388 | [-0.00581, -0.00330] | +0.33 | Y | Y | **Y** |
| long_form | B3_lm_linear | 10 | 7933 | 803 | -5.45 | 1.33e-06 | -4.62 | 0.0001 | -3.50 | 0.0005 | 0.013 | +1.30 | +0.597 | +0.388 | [-0.00295, -0.00090] | +0.20 | Y | Y | **Y** |
| long_form | B3_lm_linear | 20 | 7902 | 794 | -9.63 | 4.09e-20 | -4.69 | 7.3e-05 | -4.81 | 0.0000 | 5.7e-05 | +3.57 | +0.582 | +0.036 | [-0.00320, -0.00139] | -1.19 | Y | Y | **Y** |
| long_form | C1_bert_s1 | 20 | 7902 | 794 | -15.11 | 4.4e-49 | -7.42 | 1.09e-11 | -8.68 | 0.0000 | 8.84e-16 | +4.56 | +0.140 | +0.036 | [-0.00409, -0.00260] | +0.13 | Y | Y | **Y** |
| long_form | C2_finbert_s1 | 5 | 7951 | 809 | -5.57 | 6.7e-07 | -0.84 | 1 | -3.48 | 0.0005 | 0.013 | +1.99 | +0.226 | +0.431 | [-0.00231, -0.00067] | +0.94 | Y | no | **Y** |
| long_form | C2_finbert_s1 | 10 | 7933 | 803 | -12.77 | 3.36e-35 | -6.46 | 6.31e-09 | -3.87 | 0.0001 | 0.00314 | +5.12 | +0.315 | +0.388 | [-0.00593, -0.00196] | +0.60 | Y | Y | **Y** |
| long_form | C2_finbert_s2 | 5 | 7951 | 809 | -7.25 | 1.8e-11 | -4.30 | 0.000409 | +0.08 | 0.9391 | 1 | +1.38 | +0.466 | +0.431 | [-0.00179, +0.00192] | -0.24 | Y | Y | no |
| long_form | C2_finbert_s2 | 10 | 7933 | 803 | -7.92 | 1.14e-13 | -6.58 | 3.07e-09 | -6.93 | 0.0000 | 3.28e-10 | +1.60 | +0.122 | +0.388 | [-0.00215, -0.00124] | +0.45 | Y | Y | **Y** |
| long_form | C2_finbert_s3 | 5 | 7951 | 809 | -9.85 | 5.17e-21 | -5.43 | 2.16e-06 | -2.23 | 0.0262 | 0.367 | +3.25 | +0.286 | +0.431 | [-0.00393, -0.00028] | +0.19 | Y | Y | no |
| long_form | C2_finbert_s3 | 10 | 7933 | 803 | -9.38 | 4.57e-19 | -5.13 | 9.15e-06 | -1.27 | 0.2055 | 1 | +3.25 | +0.481 | +0.388 | [-0.00385, +0.00103] | +0.76 | Y | Y | no |
| long_form | C2_finbert_s4 | 10 | 7933 | 803 | -10.36 | 3e-23 | -6.02 | 8.57e-08 | -2.40 | 0.0168 | 0.269 | +2.56 | +0.191 | +0.388 | [-0.00493, -0.00049] | -0.25 | Y | Y | no |
| long_form | C2_finbert_s4 | 20 | 7902 | 794 | -16.89 | 4.94e-61 | -8.82 | 2.76e-16 | -9.28 | 0.0000 | 6.3e-18 | +4.96 | +0.155 | +0.036 | [-0.00514, -0.00335] | +0.57 | Y | Y | **Y** |
| long_form | C3_roberta_s1 | 5 | 7951 | 809 | -10.47 | 9.94e-24 | -5.37 | 2.83e-06 | -2.16 | 0.0313 | 0.375 | +1.76 | +0.158 | +0.431 | [-0.00276, -0.00014] | -0.28 | Y | Y | no |
| long_form | C3_roberta_s1 | 10 | 7933 | 803 | -6.37 | 6.8e-09 | -5.18 | 7.15e-06 | -3.30 | 0.0010 | 0.0235 | +3.17 | +0.480 | +0.388 | [-0.00475, -0.00121] | +0.59 | Y | Y | **Y** |
| long_form | C3_roberta_s1 | 20 | 7902 | 794 | -6.89 | 2.19e-10 | -1.07 | 1 | -4.61 | 0.0000 | 0.00014 | +0.45 | +0.159 | +0.036 | [-0.00025, -0.00010] | +0.02 | Y | no | **Y** |
| long_form | C4_longformer | 5 | 7951 | 809 | -8.73 | 1.54e-16 | -5.99 | 1.02e-07 | -4.13 | 0.0000 | 0.00114 | +2.47 | +0.347 | +0.431 | [-0.00457, -0.00168] | +0.14 | Y | Y | **Y** |
| long_form | C6_llmtext | 5 | 7951 | 809 | -10.27 | 7.39e-23 | -6.31 | 1.58e-08 | -3.94 | 0.0001 | 0.00248 | +1.67 | +0.303 | +0.431 | [-0.00401, -0.00144] | -0.32 | Y | Y | **Y** |
| long_form | C6_llmtext | 10 | 7933 | 803 | -7.06 | 7.04e-11 | -7.92 | 3.11e-13 | -4.82 | 0.0000 | 5.48e-05 | +1.88 | +0.372 | +0.388 | [-0.00290, -0.00124] | -0.13 | Y | Y | **Y** |
| long_form | C6_llmtext | 20 | 7902 | 794 | -4.15 | 0.000678 | -3.23 | 0.0179 | -2.72 | 0.0066 | 0.132 | +0.20 | +0.076 | +0.036 | [-0.00041, -0.00007] | -1.47 | - | Y | no |
| long_form | D4_llmfused | 5 | 7951 | 809 | -3.29 | 0.0169 | -0.75 | 1 | -1.53 | 0.1273 | 0.916 | +0.09 | -0.015 | +0.431 | [-0.00024, +0.00004] | +0.93 | Y | no | no |
| long_form | D4_llmfused | 20 | 7902 | 794 | -6.07 | 4.2e-08 | -3.58 | 0.00617 | -2.97 | 0.0030 | 0.064 | +0.46 | +0.080 | +0.036 | [-0.00093, -0.00022] | -0.49 | Y | Y | no |

## (b) HAR-X(VIX) standalone baseline (A7) vs A2 HAR-RV — test QLIKE (vol-unit), clustered DM

Pooled per-horizon log-OLS [1, log rv1, log rv5, log rv22, log VIX_pit] + Duan smearing, fit on train, har_rv.py conventions. Run dirs: results/runs/A7_harx_vix_full_<disc>_seed2026 (metrics.json uses the pipeline's variance-unit QLIKE; this table uses M1's vol-unit QLIKE). DM sign: negative = A7 better than A2.

| disc | h | n_test | n_days | QLIKE A2 | QLIKE A7 | rel% | DM(A7 vs A2) | p | Holm | log-VIX coef | smear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| long_form | 5 | 7951 | 809 | 0.1226 | 0.1149 | +6.31 | -1.03 | 0.3015 | 0.582 | +0.460 | 1.136 |
| long_form | 10 | 7933 | 803 | 0.0893 | 0.0807 | +9.63 | -1.06 | 0.2910 | 0.582 | +0.390 | 1.094 |
| long_form | 20 | 7902 | 794 | 0.0646 | 0.0587 | +9.08 | -1.34 | 0.1808 | 0.542 | +0.331 | 1.072 |
| event_driven | 5 | 25109 | 996 | 0.1235 | 0.1197 | +3.09 | -2.65 | 0.0081 | 0.0732 | +0.272 | 1.139 |
| event_driven | 10 | 25001 | 991 | 0.0890 | 0.0859 | +3.50 | -2.08 | 0.0379 | 0.265 | +0.235 | 1.098 |
| event_driven | 20 | 24732 | 981 | 0.0660 | 0.0636 | +3.60 | -2.04 | 0.0416 | 0.265 | +0.173 | 1.074 |
| combined | 5 | 33060 | 996 | 0.1234 | 0.1188 | +3.72 | -2.37 | 0.0180 | 0.144 | +0.315 | 1.139 |
| combined | 10 | 32934 | 991 | 0.0891 | 0.0850 | +4.59 | -1.96 | 0.0498 | 0.265 | +0.272 | 1.097 |
| combined | 20 | 32634 | 981 | 0.0660 | 0.0630 | +4.59 | -2.00 | 0.0457 | 0.265 | +0.210 | 1.074 |

## PIT sanity spot-check (3 test filings, long_form h=5)

Discipline: VIX value used is the last close with date STRICTLY BEFORE effective_trading_day (= label-window start); no same-day information.

| ticker | accession | filing_time_utc | effective_trading_day | VIX date used | VIX close | PIT ok |
|---|---|---|---|---|---|---|
| DRI | 0000940944-22-000007 | 2022-01-05 20:48:28+00:00 | 2022-01-05 | 2022-01-04 | 16.91 | True |
| URI | 0001067701-24-000007 | 2024-01-24 21:27:28+00:00 | 2024-01-25 | 2024-01-24 | 13.14 | True |
| PAYX | 0001193125-25-328698 | 2025-12-22 21:03:09+00:00 | 2025-12-23 | 2025-12-22 | 14.08 | True |

## Bottom line

- Day-clustered DM alone: 29/38 originally-genuine cells stay significant (Holm<.05 within this table).
- Adding the VIX control on top: 19/38 originally-genuine cells survive a market-level implied-volatility control (placebo-clean).
- HAR-X(VIX) shows how much of the field a pure market-vol regressor claims on its own (table b); the text increment in (a) is measured NET of that signal.
- Weights fit on validation only, applied frozen to test; VIX strictly point-in-time.