# P1-c — Freeze-window sensitivity of the M1 log-space text increment

## RESTATED vs ORIGINAL

- **ORIGINAL (grid, forecast_combination_grid.csv):** all combiner weights frozen on the full COVID validation window (2020-2021); significance from observation-order DM (HAC lag over filings, reviewer-verified as ~2x inflated).
- **RESTATED (this table):** same val-fit/test-apply protocol, but the freeze window is varied — (i) `train_tail` 2018-2019 pre-COVID, (ii) `val_ex_h1` val minus Jan-Jun 2020, (iii) `val_full` original — and ALL inference is day-clustered DM (daily-mean differentials on effective_trading_day, HAC lag=h-1 days, n=days). Variant (iii) reproduces the grid rel% exactly (max |dev| = 2.22e-16; sanity PASS), so any difference in significance vs the grid is the clustering fix, and any difference across rows (i)/(ii) is purely the freeze origin.

Cells: 40 (all 38 placebo-confirmed genuine cells + all 6 C6_llmtext cells, deduplicated). C6_llmtext/D4_llmfused runs contain no train rows, so `train_tail` is structurally NA for them.

## Verdict by fit window (cells with a computable fit)

| fit_window | median rel% | median clustered DM | sig (DM<0, p<.05) |
|---|---|---|---|
| train_tail | -15.96 | +4.89 | 3/32 |
| val_ex_h1 | +1.48 | -2.54 | 27/40 |
| val_full | +1.48 | -4.22 | 35/40 |

- Sign agreement of rel% with `val_full`: train_tail 6/32 pairable cells; val_ex_h1 39/40 cells.
- Median rel% shift vs val_full: train_tail -18.12pp, val_ex_h1 +0.10pp (negative = the COVID window INFLATES the measured increment; positive = it deflates it).

## Interpretation (read before citing the train_tail row)

- **`val_ex_h1` is the clean freeze-origin test** and the increment survives it: rel% is essentially unchanged (median shift +0.10pp, 39/40 sign agreement) and 27/40 cells stay significant under day-clustered DM (vs 35/40 on the full val window). The acute COVID H1 window mildly inflates significance (clustered DM median -4.22 -> -2.54) but NOT the size of the increment.
- **`train_tail` is structurally confounded, not a clean refit:** the text models were TRAINED on the train split, so their train-split predictions are IN-SAMPLE fitted values. The combiner therefore over-weights text (median g_log +0.78 on train_tail vs +0.24 on val) and that inflated weight fails out of sample (3/32 sig, median rel% -15.96). This says combiner weights must be estimated on data where the text forecasts are themselves out-of-sample (which the val window provides); it is NOT evidence that the increment depends on COVID. The pre-registered protocol (val-fit) is the correct one, and (ii) shows its conclusion does not hinge on the COVID crash months.
- C6_llmtext caveat: at h=20 (a non-genuine cell included only for completeness) C6 loses significance without 2020H1; the four genuine C6 cells (h=5/10) all remain negative, three of four significant.

## Per-cell grid (rel% = QLIKE improvement of U over R; DM day-clustered, negative = text helps; n_days = test days)

| disc | model | h | fit_window | n_fit | rel% | g_log | DM(clust) | p | n_days | note |
|---|---|---|---|---|---|---|---|---|---|---|
| long_form | B1_bow_ridge | 5 | train_tail | 3935 | -51.95 | +0.811 | +7.75 | 0.0000 | 809 |  |
| long_form | B1_bow_ridge | 5 | val_ex_h1 | 2971 | +1.07 | +0.185 | -1.34 | 0.1803 | 809 |  |
| long_form | B1_bow_ridge | 5 | val_full | 3956 | +1.65 | +0.107 | -3.83 | 0.0001 | 809 |  |
| long_form | B1_bow_ridge | 10 | train_tail | 3930 | -50.81 | +0.811 | +5.42 | 0.0000 | 803 |  |
| long_form | B1_bow_ridge | 10 | val_ex_h1 | 2967 | +1.94 | +0.189 | -1.38 | 0.1669 | 803 |  |
| long_form | B1_bow_ridge | 10 | val_full | 3950 | +1.44 | +0.059 | -4.15 | 0.0000 | 803 |  |
| long_form | B1_bow_ridge | 20 | train_tail | 3919 | -47.99 | +0.776 | +4.40 | 0.0000 | 794 |  |
| long_form | B1_bow_ridge | 20 | val_ex_h1 | 2961 | +3.47 | +0.237 | -1.50 | 0.1350 | 794 |  |
| long_form | B1_bow_ridge | 20 | val_full | 3943 | +2.99 | +0.098 | -5.45 | 0.0000 | 794 |  |
| long_form | B2_tfidf_ridge | 5 | train_tail | 3935 | -11.17 | +0.860 | +5.27 | 0.0000 | 809 |  |
| long_form | B2_tfidf_ridge | 5 | val_ex_h1 | 2971 | +2.18 | +0.422 | -1.20 | 0.2318 | 809 |  |
| long_form | B2_tfidf_ridge | 5 | val_full | 3956 | +3.33 | +0.237 | -5.39 | 0.0000 | 809 |  |
| long_form | B2_tfidf_ridge | 10 | train_tail | 3930 | -13.67 | +0.880 | +4.24 | 0.0000 | 803 |  |
| long_form | B2_tfidf_ridge | 10 | val_ex_h1 | 2967 | +3.57 | +0.385 | -3.57 | 0.0004 | 803 |  |
| long_form | B2_tfidf_ridge | 10 | val_full | 3950 | +3.48 | +0.170 | -8.89 | 0.0000 | 803 |  |
| long_form | B3_lm_linear | 10 | train_tail | 3930 | +1.93 | +0.455 | -5.30 | 0.0000 | 803 |  |
| long_form | B3_lm_linear | 10 | val_ex_h1 | 2967 | +0.13 | +0.718 | -2.58 | 0.0099 | 803 |  |
| long_form | B3_lm_linear | 10 | val_full | 3950 | +1.79 | +0.591 | -4.62 | 0.0000 | 803 |  |
| long_form | B3_lm_linear | 20 | train_tail | 3919 | +3.14 | +0.553 | -4.61 | 0.0000 | 794 |  |
| long_form | B3_lm_linear | 20 | val_ex_h1 | 2961 | +1.52 | +0.664 | -2.32 | 0.0208 | 794 |  |
| long_form | B3_lm_linear | 20 | val_full | 3943 | +3.48 | +0.582 | -4.69 | 0.0000 | 794 |  |
| long_form | C1_bert_s1 | 20 | train_tail | 3919 | -38.00 | +0.724 | +4.52 | 0.0000 | 794 |  |
| long_form | C1_bert_s1 | 20 | val_ex_h1 | 2961 | +4.84 | +0.251 | -3.91 | 0.0001 | 794 |  |
| long_form | C1_bert_s1 | 20 | val_full | 3943 | +2.95 | +0.124 | -7.42 | 0.0000 | 794 |  |
| long_form | C2_finbert_s1 | 5 | train_tail | 3935 | +11.42 | +0.456 | -7.94 | 0.0000 | 809 |  |
| long_form | C2_finbert_s1 | 5 | val_ex_h1 | 2971 | +1.87 | +0.234 | -3.33 | 0.0009 | 809 |  |
| long_form | C2_finbert_s1 | 5 | val_full | 3956 | +0.56 | +0.110 | -0.84 | 0.4014 | 809 |  |
| long_form | C2_finbert_s1 | 10 | train_tail | 3930 | -25.38 | +0.714 | +5.86 | 0.0000 | 803 |  |
| long_form | C2_finbert_s1 | 10 | val_ex_h1 | 2967 | +2.08 | +0.397 | -1.90 | 0.0583 | 803 |  |
| long_form | C2_finbert_s1 | 10 | val_full | 3950 | +4.56 | +0.242 | -6.46 | 0.0000 | 803 |  |
| long_form | C2_finbert_s2 | 5 | train_tail | 3935 | -4.58 | +0.721 | +5.43 | 0.0000 | 809 |  |
| long_form | C2_finbert_s2 | 5 | val_ex_h1 | 2971 | +1.26 | +0.369 | -2.09 | 0.0371 | 809 |  |
| long_form | C2_finbert_s2 | 5 | val_full | 3956 | +1.73 | +0.309 | -4.30 | 0.0000 | 809 |  |
| long_form | C2_finbert_s2 | 10 | train_tail | 3930 | +2.44 | +0.633 | -1.90 | 0.0579 | 803 |  |
| long_form | C2_finbert_s2 | 10 | val_ex_h1 | 2967 | +1.80 | +0.247 | -4.95 | 0.0000 | 803 |  |
| long_form | C2_finbert_s2 | 10 | val_full | 3950 | +0.24 | +0.023 | -6.58 | 0.0000 | 803 |  |
| long_form | C2_finbert_s3 | 5 | train_tail | 3935 | -14.78 | +0.504 | +8.13 | 0.0000 | 809 |  |
| long_form | C2_finbert_s3 | 5 | val_ex_h1 | 2971 | +2.98 | +0.279 | -3.31 | 0.0010 | 809 |  |
| long_form | C2_finbert_s3 | 5 | val_full | 3956 | +2.39 | +0.176 | -5.43 | 0.0000 | 809 |  |
| long_form | C2_finbert_s3 | 10 | train_tail | 3930 | -17.14 | +0.774 | +6.12 | 0.0000 | 803 |  |
| long_form | C2_finbert_s3 | 10 | val_ex_h1 | 2967 | +1.44 | +0.432 | -1.53 | 0.1273 | 803 |  |
| long_form | C2_finbert_s3 | 10 | val_full | 3950 | +3.03 | +0.308 | -5.13 | 0.0000 | 803 |  |
| long_form | C2_finbert_s4 | 10 | train_tail | 3930 | -110.62 | +0.942 | +8.93 | 0.0000 | 803 |  |
| long_form | C2_finbert_s4 | 10 | val_ex_h1 | 2967 | +0.88 | +0.087 | -2.86 | 0.0043 | 803 |  |
| long_form | C2_finbert_s4 | 10 | val_full | 3950 | +2.28 | +0.105 | -6.02 | 0.0000 | 803 |  |
| long_form | C2_finbert_s4 | 20 | train_tail | 3919 | -28.86 | +0.769 | +4.28 | 0.0000 | 794 |  |
| long_form | C2_finbert_s4 | 20 | val_ex_h1 | 2961 | +3.48 | +0.253 | -3.60 | 0.0003 | 794 |  |
| long_form | C2_finbert_s4 | 20 | val_full | 3943 | +3.45 | +0.139 | -8.82 | 0.0000 | 794 |  |
| long_form | C3_roberta_s1 | 5 | train_tail | 3935 | -20.84 | +0.720 | +7.54 | 0.0000 | 809 |  |
| long_form | C3_roberta_s1 | 5 | val_ex_h1 | 2971 | +1.44 | +0.177 | -2.73 | 0.0064 | 809 |  |
| long_form | C3_roberta_s1 | 5 | val_full | 3956 | +0.67 | +0.050 | -5.37 | 0.0000 | 809 |  |
| long_form | C3_roberta_s1 | 10 | train_tail | 3930 | -1.28 | +0.777 | +1.95 | 0.0520 | 803 |  |
| long_form | C3_roberta_s1 | 10 | val_ex_h1 | 2967 | +2.24 | +0.367 | -4.22 | 0.0000 | 803 |  |
| long_form | C3_roberta_s1 | 10 | val_full | 3950 | +1.98 | +0.312 | -5.18 | 0.0000 | 803 |  |
| long_form | C3_roberta_s1 | 20 | train_tail | 3919 | -1.39 | +0.950 | +0.95 | 0.3406 | 794 |  |
| long_form | C3_roberta_s1 | 20 | val_ex_h1 | 2961 | +0.53 | +0.261 | -4.24 | 0.0000 | 794 |  |
| long_form | C3_roberta_s1 | 20 | val_full | 3943 | +0.24 | +0.132 | -1.07 | 0.2857 | 794 |  |
| long_form | C4_longformer | 5 | train_tail | 3935 | -4.26 | +0.755 | +2.15 | 0.0317 | 809 |  |
| long_form | C4_longformer | 5 | val_ex_h1 | 2971 | +2.13 | +0.312 | -4.92 | 0.0000 | 809 |  |
| long_form | C4_longformer | 5 | val_full | 3956 | +1.53 | +0.186 | -5.99 | 0.0000 | 809 |  |
| long_form | C6_llmtext | 5 | train_tail | 0 | — | — | — | — | — | NA: fit window has <100 rows (C6/D4 runs contain no train split) |
| long_form | C6_llmtext | 5 | val_ex_h1 | 2971 | +1.68 | +0.361 | -4.53 | 0.0000 | 809 |  |
| long_form | C6_llmtext | 5 | val_full | 3956 | +1.79 | +0.254 | -6.31 | 0.0000 | 809 |  |
| long_form | C6_llmtext | 10 | train_tail | 0 | — | — | — | — | — | NA: fit window has <100 rows (C6/D4 runs contain no train split) |
| long_form | C6_llmtext | 10 | val_ex_h1 | 2967 | +1.71 | +0.527 | -4.37 | 0.0000 | 803 |  |
| long_form | C6_llmtext | 10 | val_full | 3950 | +2.25 | +0.333 | -7.92 | 0.0000 | 803 |  |
| long_form | D4_llmfused | 5 | train_tail | 0 | — | — | — | — | — | NA: fit window has <100 rows (C6/D4 runs contain no train split) |
| long_form | D4_llmfused | 5 | val_ex_h1 | 2971 | +0.24 | -0.101 | +0.08 | 0.9352 | 809 |  |
| long_form | D4_llmfused | 5 | val_full | 3956 | +0.15 | -0.026 | -0.75 | 0.4512 | 809 |  |
| long_form | D4_llmfused | 20 | train_tail | 0 | — | — | — | — | — | NA: fit window has <100 rows (C6/D4 runs contain no train split) |
| long_form | D4_llmfused | 20 | val_ex_h1 | 2961 | +0.72 | +0.185 | -2.72 | 0.0066 | 794 |  |
| long_form | D4_llmfused | 20 | val_full | 3943 | +0.60 | +0.083 | -3.58 | 0.0004 | 794 |  |
| event_driven | B1_bow_ridge | 5 | train_tail | 13770 | -20.59 | +0.849 | +9.88 | 0.0000 | 996 |  |
| event_driven | B1_bow_ridge | 5 | val_ex_h1 | 9915 | +1.57 | +0.311 | -2.50 | 0.0126 | 996 |  |
| event_driven | B1_bow_ridge | 5 | val_full | 14213 | +1.33 | +0.230 | -3.35 | 0.0008 | 996 |  |
| event_driven | B1_bow_ridge | 10 | train_tail | 13740 | -27.04 | +0.821 | +8.90 | 0.0000 | 991 |  |
| event_driven | B1_bow_ridge | 10 | val_ex_h1 | 9902 | +1.35 | +0.334 | -2.07 | 0.0392 | 991 |  |
| event_driven | B1_bow_ridge | 10 | val_full | 14196 | +1.23 | +0.236 | -3.25 | 0.0012 | 991 |  |
| event_driven | B1_bow_ridge | 20 | train_tail | 13695 | -37.24 | +0.816 | +7.44 | 0.0000 | 981 |  |
| event_driven | B1_bow_ridge | 20 | val_ex_h1 | 9874 | +1.63 | +0.330 | -2.13 | 0.0337 | 981 |  |
| event_driven | B1_bow_ridge | 20 | val_full | 14156 | +1.53 | +0.241 | -3.10 | 0.0020 | 981 |  |
| event_driven | B2_tfidf_ridge | 5 | train_tail | 13770 | -17.87 | +0.798 | +9.26 | 0.0000 | 996 |  |
| event_driven | B2_tfidf_ridge | 5 | val_ex_h1 | 9915 | +1.41 | +0.368 | -1.66 | 0.0974 | 996 |  |
| event_driven | B2_tfidf_ridge | 5 | val_full | 14213 | +1.21 | +0.251 | -2.76 | 0.0059 | 996 |  |
| event_driven | B2_tfidf_ridge | 10 | train_tail | 13740 | -25.41 | +0.775 | +7.91 | 0.0000 | 991 |  |
| event_driven | B2_tfidf_ridge | 10 | val_ex_h1 | 9902 | +1.62 | +0.353 | -2.11 | 0.0347 | 991 |  |
| event_driven | B2_tfidf_ridge | 10 | val_full | 14196 | +1.35 | +0.243 | -2.97 | 0.0031 | 991 |  |
| event_driven | B2_tfidf_ridge | 20 | train_tail | 13695 | -34.65 | +0.766 | +6.51 | 0.0000 | 981 |  |
| event_driven | B2_tfidf_ridge | 20 | val_ex_h1 | 9874 | +2.18 | +0.361 | -2.81 | 0.0050 | 981 |  |
| event_driven | B2_tfidf_ridge | 20 | val_full | 14156 | +1.84 | +0.265 | -3.11 | 0.0020 | 981 |  |
| event_driven | B3_lm_linear | 5 | train_tail | 13770 | -0.52 | +1.114 | +3.46 | 0.0006 | 996 |  |
| event_driven | B3_lm_linear | 5 | val_ex_h1 | 9915 | +0.35 | +0.945 | -2.83 | 0.0047 | 996 |  |
| event_driven | B3_lm_linear | 5 | val_full | 14213 | +0.25 | +0.680 | -2.43 | 0.0152 | 996 |  |
| event_driven | B3_lm_linear | 10 | train_tail | 13740 | -0.51 | +0.849 | +2.16 | 0.0310 | 991 |  |
| event_driven | B3_lm_linear | 10 | val_ex_h1 | 9902 | +0.34 | +0.891 | -2.34 | 0.0195 | 991 |  |
| event_driven | B3_lm_linear | 10 | val_full | 14196 | +0.20 | +0.710 | -1.23 | 0.2193 | 991 |  |
| event_driven | B4_lm_features | 5 | train_tail | 13770 | -0.18 | +1.762 | +3.99 | 0.0001 | 996 |  |
| event_driven | B4_lm_features | 5 | val_ex_h1 | 9915 | +0.27 | +1.531 | -0.46 | 0.6431 | 996 |  |
| event_driven | B4_lm_features | 5 | val_full | 14213 | +0.18 | +1.140 | -0.99 | 0.3205 | 996 |  |
| event_driven | B4_lm_features | 10 | train_tail | 13740 | +0.04 | +1.373 | +1.83 | 0.0674 | 991 |  |
| event_driven | B4_lm_features | 10 | val_ex_h1 | 9902 | +0.14 | +0.656 | -1.35 | 0.1786 | 991 |  |
| event_driven | B4_lm_features | 10 | val_full | 14196 | +0.08 | +0.428 | -2.10 | 0.0356 | 991 |  |
| event_driven | B4_lm_features | 20 | train_tail | 13695 | +0.29 | +0.757 | -1.80 | 0.0720 | 981 |  |
| event_driven | B4_lm_features | 20 | val_ex_h1 | 9874 | +0.49 | +0.772 | -2.81 | 0.0050 | 981 |  |
| event_driven | B4_lm_features | 20 | val_full | 14156 | +0.25 | +0.569 | -2.01 | 0.0447 | 981 |  |
| event_driven | C2_finbert_s1 | 10 | train_tail | 13740 | -45.59 | +0.912 | +7.73 | 0.0000 | 991 |  |
| event_driven | C2_finbert_s1 | 10 | val_ex_h1 | 9902 | +3.48 | +0.283 | -4.86 | 0.0000 | 991 |  |
| event_driven | C2_finbert_s1 | 10 | val_full | 14196 | +2.10 | +0.177 | -5.52 | 0.0000 | 991 |  |
| event_driven | C6_llmtext | 5 | train_tail | 0 | — | — | — | — | — | NA: fit window has <100 rows (C6/D4 runs contain no train split) |
| event_driven | C6_llmtext | 5 | val_ex_h1 | 9915 | +1.16 | +0.347 | -2.86 | 0.0044 | 996 |  |
| event_driven | C6_llmtext | 5 | val_full | 14213 | +1.21 | +0.264 | -5.04 | 0.0000 | 996 |  |
| event_driven | C6_llmtext | 10 | train_tail | 0 | — | — | — | — | — | NA: fit window has <100 rows (C6/D4 runs contain no train split) |
| event_driven | C6_llmtext | 10 | val_ex_h1 | 9902 | +0.83 | +0.330 | -1.92 | 0.0554 | 991 |  |
| event_driven | C6_llmtext | 10 | val_full | 14196 | +1.00 | +0.281 | -3.76 | 0.0002 | 991 |  |
| long_form | C6_llmtext | 20 | train_tail | 0 | — | — | — | — | — | NA: fit window has <100 rows (C6/D4 runs contain no train split) |
| long_form | C6_llmtext | 20 | val_ex_h1 | 2961 | -0.16 | +0.091 | -0.10 | 0.9193 | 794 |  |
| long_form | C6_llmtext | 20 | val_full | 3943 | +0.27 | +0.078 | -3.23 | 0.0013 | 794 |  |
| event_driven | C6_llmtext | 20 | train_tail | 0 | — | — | — | — | — | NA: fit window has <100 rows (C6/D4 runs contain no train split) |
| event_driven | C6_llmtext | 20 | val_ex_h1 | 9874 | +0.28 | +0.161 | -0.97 | 0.3317 | 981 |  |
| event_driven | C6_llmtext | 20 | val_full | 14156 | +0.66 | +0.245 | -1.98 | 0.0477 | 981 |  |
