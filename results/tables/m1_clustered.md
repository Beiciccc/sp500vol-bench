# M1 Family-1 — CLUSTERED-DM restatement (day-clustered inference)

## RESTATED vs ORIGINAL

| quantity | ORIGINAL (obs-level HAC) | RESTATED (day-clustered) |
|---|---|---|
| genuine cells (DM-QLIKE<0, Holm<.05, placebo null) | **38/69** | **29/69** |
| DM-QLIKE helps (Holm<.05) | 43 | 29 |
| DM-QLIKE worse (Holm<.05) | 12 | 8 |
| Clark-West adds (Holm<.05) | 50 | 44 |
| placebo-gate failures (\|DM\|>=2) | 7 | 2 |
| genuine effect-size range (rel QLIKE %) | 0.08-4.56 | 0.24-5.92 |
| median \|DM\| shrink factor (clust/obs) | 1.00 | 0.49 |

The forecasts are IDENTICAL (fc.log_combo, weights fit on validation, frozen on test); only the inference changes. Obs-level columns of this rerun reproduce `forecast_combination_grid.csv` exactly (hard assertion, max column diff 1.8e-15). Clustering: per-obs loss differentials averaged within calendar day of `effective_trading_day`; DM run on the daily series with HAC lag=h-1 in DAYS; day-block (h-day) moving bootstrap CI; placebo permutations identical seeds (1000-1004).

**n_obs vs n_days (test):**
| disclosure | h | n_obs | n_days |
|---|---|---|---|
| event_driven | 5 | 25109 | 996 |
| event_driven | 10 | 25001 | 991 |
| event_driven | 20 | 24732 | 981 |
| long_form | 5 | 7951 | 809 |
| long_form | 10 | 7933 | 803 |
| long_form | 20 | 7902 | 794 |

## Reviewer-flagged cells

| cell | dm_q obs | p obs | Holm obs | dm_q clust | p clust | Holm clust | genuine obs->clust |
|---|---|---|---|---|---|---|---|
| long_form C2_finbert_s1 h5 | -5.57 | 2.58e-08 | 0.000 | -0.84 | 4.01e-01 | 1.000 | YES -> no |
| long_form B2_tfidf_ridge h5 | -11.34 | 1.43e-29 | 0.000 | -5.39 | 9.21e-08 | 0.000 | YES -> YES |
| long_form B2_tfidf_ridge h10 | -15.30 | 4.27e-52 | 0.000 | -8.89 | 3.90e-18 | 0.000 | YES -> YES |
| long_form B2_tfidf_ridge h20 | -19.62 | 1.07e-83 | 0.000 | -9.04 | 1.17e-18 | 0.000 | no -> YES |
| long_form C6_llmtext h5 | -10.27 | 1.32e-24 | 0.000 | -6.31 | 4.66e-10 | 0.000 | YES -> YES |
| long_form C6_llmtext h10 | -7.06 | 1.85e-12 | 0.000 | -7.92 | 8.18e-15 | 0.000 | YES -> YES |
| long_form C6_llmtext h20 | -4.15 | 3.39e-05 | 0.001 | -3.23 | 1.28e-03 | 0.042 | no -> YES |
| event_driven B2_tfidf_ridge h5 | -7.05 | 1.88e-12 | 0.000 | -2.76 | 5.89e-03 | 0.165 | YES -> no |
| event_driven B2_tfidf_ridge h10 | -7.21 | 5.80e-13 | 0.000 | -2.97 | 3.10e-03 | 0.090 | YES -> no |
| event_driven B2_tfidf_ridge h20 | -8.22 | 2.10e-16 | 0.000 | -3.11 | 1.96e-03 | 0.063 | YES -> no |
| event_driven C6_llmtext h5 | -9.86 | 6.93e-23 | 0.000 | -5.04 | 5.42e-07 | 0.000 | YES -> YES |
| event_driven C6_llmtext h10 | -7.98 | 1.48e-15 | 0.000 | -3.76 | 1.81e-04 | 0.008 | YES -> YES |
| event_driven C6_llmtext h20 | -5.39 | 7.14e-08 | 0.000 | -1.98 | 4.77e-02 | 1.000 | no -> no |

## Cells that FLIP (genuine -> not genuine under clustering): 12

| disc | model | h | rel% | dm_q obs | Holm obs | dm_q clust | p clust | Holm clust |
|---|---|---|---|---|---|---|---|---|
| long_form | C2_finbert_s1 | 5 | +0.56 | -5.57 | 0.0000 | -0.84 | 0.4014 | 1.0000 |
| long_form | C3_roberta_s1 | 20 | +0.24 | -6.89 | 0.0000 | -1.07 | 0.2857 | 1.0000 |
| long_form | D4_llmfused | 5 | +0.15 | -3.29 | 0.0169 | -0.75 | 0.4512 | 1.0000 |
| event_driven | B1_bow_ridge | 20 | +1.53 | -6.50 | 0.0000 | -3.10 | 0.0020 | 0.0626 |
| event_driven | B2_tfidf_ridge | 5 | +1.21 | -7.05 | 0.0000 | -2.76 | 0.0059 | 0.1650 |
| event_driven | B2_tfidf_ridge | 10 | +1.35 | -7.21 | 0.0000 | -2.97 | 0.0031 | 0.0898 |
| event_driven | B2_tfidf_ridge | 20 | +1.84 | -8.22 | 0.0000 | -3.11 | 0.0020 | 0.0626 |
| event_driven | B3_lm_linear | 5 | +0.25 | -4.82 | 0.0000 | -2.43 | 0.0152 | 0.3944 |
| event_driven | B3_lm_linear | 10 | +0.20 | -3.03 | 0.0361 | -1.23 | 0.2193 | 1.0000 |
| event_driven | B4_lm_features | 5 | +0.18 | -3.95 | 0.0015 | -0.99 | 0.3205 | 1.0000 |
| event_driven | B4_lm_features | 10 | +0.08 | -3.77 | 0.0030 | -2.10 | 0.0356 | 0.8555 |
| event_driven | B4_lm_features | 20 | +0.25 | -3.11 | 0.0300 | -2.01 | 0.0447 | 0.9845 |

Cells newly genuine under clustering: 3 (long_form B2_tfidf_ridge h20, long_form C6_llmtext h20, event_driven C2_finbert_s1 h5)

## Surviving genuine cells (clustered)

| disc | model | h | rel% | dm_q clust | Holm clust | daily-boot 95% CI (QLIKE diff) | placebo DM clust |
|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | +1.33 | -3.35 | 0.0304 | [-0.00193, -0.00053] | +0.59 |
| event_driven | B1_bow_ridge | 10 | +1.23 | -3.25 | 0.0403 | [-0.00136, -0.00034] | +0.05 |
| event_driven | C2_finbert_s1 | 5 | +2.14 | -4.95 | 0.0000 | [-0.00254, -0.00111] | -0.94 |
| event_driven | C2_finbert_s1 | 10 | +2.10 | -5.52 | 0.0000 | [-0.00189, -0.00091] | -0.16 |
| event_driven | C6_llmtext | 5 | +1.21 | -5.04 | 0.0000 | [-0.00157, -0.00069] | +0.10 |
| event_driven | C6_llmtext | 10 | +1.00 | -3.76 | 0.0076 | [-0.00101, -0.00031] | +0.95 |
| long_form | B1_bow_ridge | 5 | +1.65 | -3.83 | 0.0058 | [-0.00276, -0.00089] | +0.43 |
| long_form | B1_bow_ridge | 10 | +1.44 | -4.15 | 0.0016 | [-0.00153, -0.00056] | +0.39 |
| long_form | B1_bow_ridge | 20 | +2.99 | -5.45 | 0.0000 | [-0.00255, -0.00122] | -0.25 |
| long_form | B2_tfidf_ridge | 5 | +3.33 | -5.39 | 0.0000 | [-0.00529, -0.00248] | +0.51 |
| long_form | B2_tfidf_ridge | 10 | +3.48 | -8.89 | 0.0000 | [-0.00426, -0.00281] | +0.33 |
| long_form | B2_tfidf_ridge | 20 | +5.92 | -9.04 | 0.0000 | [-0.00522, -0.00343] | +0.30 |
| long_form | B3_lm_linear | 10 | +1.79 | -4.62 | 0.0002 | [-0.00351, -0.00147] | +0.27 |
| long_form | B3_lm_linear | 20 | +3.48 | -4.69 | 0.0002 | [-0.00318, -0.00135] | -1.26 |
| long_form | C1_bert_s1 | 20 | +2.95 | -7.42 | 0.0000 | [-0.00266, -0.00158] | +0.24 |
| long_form | C2_finbert_s1 | 10 | +4.56 | -6.46 | 0.0000 | [-0.00595, -0.00327] | +1.01 |
| long_form | C2_finbert_s2 | 5 | +1.73 | -4.30 | 0.0009 | [-0.00299, -0.00114] | +0.10 |
| long_form | C2_finbert_s2 | 10 | +0.24 | -6.58 | 0.0000 | [-0.00040, -0.00022] | +0.21 |
| long_form | C2_finbert_s3 | 5 | +2.39 | -5.43 | 0.0000 | [-0.00366, -0.00174] | +0.30 |
| long_form | C2_finbert_s3 | 10 | +3.03 | -5.13 | 0.0000 | [-0.00412, -0.00189] | +1.00 |
| long_form | C2_finbert_s4 | 10 | +2.28 | -6.02 | 0.0000 | [-0.00425, -0.00216] | -0.66 |
| long_form | C2_finbert_s4 | 20 | +3.45 | -8.82 | 0.0000 | [-0.00374, -0.00241] | +0.65 |
| long_form | C3_roberta_s1 | 5 | +0.67 | -5.37 | 0.0000 | [-0.00119, -0.00057] | +0.37 |
| long_form | C3_roberta_s1 | 10 | +1.98 | -5.18 | 0.0000 | [-0.00374, -0.00174] | +0.53 |
| long_form | C4_longformer | 5 | +1.53 | -5.99 | 0.0000 | [-0.00285, -0.00147] | -0.21 |
| long_form | C6_llmtext | 5 | +1.79 | -6.31 | 0.0000 | [-0.00406, -0.00218] | +0.83 |
| long_form | C6_llmtext | 10 | +2.25 | -7.92 | 0.0000 | [-0.00333, -0.00202] | -1.45 |
| long_form | C6_llmtext | 20 | +0.27 | -3.23 | 0.0422 | [-0.00049, -0.00013] | -1.44 |
| long_form | D4_llmfused | 20 | +0.60 | -3.58 | 0.0149 | [-0.00108, -0.00035] | -0.49 |

## Full 69-cell grid (original vs clustered)

| disc | model | h | n_obs | n_days | rel% | dm_q | p_q | Holm | dm_q_cl | p_cl | Holm_cl | dm_se | dm_se_cl | genuine | genuine_cl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long_form | B1_bow_ridge | 5 | 7951 | 809 | +1.65 | -6.75 | 1.6e-11 | 0.000 | -3.83 | 1.4e-04 | 0.006 | -8.57 | -4.43 | Y | Y |
| long_form | B1_bow_ridge | 10 | 7933 | 803 | +1.44 | -9.18 | 5.4e-20 | 0.000 | -4.15 | 3.7e-05 | 0.002 | -9.98 | -3.81 | Y | Y |
| long_form | B1_bow_ridge | 20 | 7902 | 794 | +2.99 | -13.08 | 1.1e-38 | 0.000 | -5.45 | 6.9e-08 | 0.000 | -12.66 | -4.64 | Y | Y |
| long_form | B2_tfidf_ridge | 5 | 7951 | 809 | +3.33 | -11.34 | 1.4e-29 | 0.000 | -5.39 | 9.2e-08 | 0.000 | -11.11 | -6.63 | Y | Y |
| long_form | B2_tfidf_ridge | 10 | 7933 | 803 | +3.48 | -15.30 | 4.3e-52 | 0.000 | -8.89 | 3.9e-18 | 0.000 | -12.83 | -7.97 | Y | Y |
| long_form | B2_tfidf_ridge | 20 | 7902 | 794 | +5.92 | -19.62 | 1.1e-83 | 0.000 | -9.04 | 1.2e-18 | 0.000 | -15.85 | -8.04 | n | Y |
| long_form | B3_lm_linear | 5 | 7951 | 809 | +0.49 | -1.70 | 8.9e-02 | 0.802 | -2.58 | 1.0e-02 | 0.269 | -4.14 | -3.45 | n | n |
| long_form | B3_lm_linear | 10 | 7933 | 803 | +1.79 | -5.45 | 5.3e-08 | 0.000 | -4.62 | 4.6e-06 | 0.000 | -6.29 | -4.28 | Y | Y |
| long_form | B3_lm_linear | 20 | 7902 | 794 | +3.48 | -9.63 | 7.7e-22 | 0.000 | -4.69 | 3.2e-06 | 0.000 | -8.60 | -4.31 | Y | Y |
| long_form | B4_lm_features | 5 | 7951 | 809 | +0.11 | -2.80 | 5.2e-03 | 0.072 | +1.02 | 3.1e-01 | 1.000 | -6.14 | -1.19 | n | n |
| long_form | B4_lm_features | 10 | 7933 | 803 | -0.92 | +8.66 | 5.6e-18 | 0.000 | +3.32 | 9.5e-04 | 0.033 | +9.66 | +4.05 | n | n |
| long_form | B4_lm_features | 20 | 7902 | 794 | -1.92 | +14.66 | 5.2e-48 | 0.000 | +3.38 | 7.5e-04 | 0.029 | +13.59 | +4.49 | n | n |
| long_form | C1_bert_s1 | 5 | 7951 | 809 | +0.12 | -0.96 | 3.4e-01 | 1.000 | -2.03 | 4.3e-02 | 0.984 | -2.97 | -3.00 | n | n |
| long_form | C1_bert_s1 | 10 | 7933 | 803 | -1.28 | +4.37 | 1.3e-05 | 0.000 | +2.35 | 1.9e-02 | 0.476 | +2.60 | +1.92 | n | n |
| long_form | C1_bert_s1 | 20 | 7902 | 794 | +2.95 | -15.11 | 6.8e-51 | 0.000 | -7.42 | 3.0e-13 | 0.000 | -12.87 | -5.28 | Y | Y |
| long_form | C2_finbert_s1 | 5 | 7951 | 809 | +0.56 | -5.57 | 2.6e-08 | 0.000 | -0.84 | 4.0e-01 | 1.000 | -2.68 | +0.25 | Y | n |
| long_form | C2_finbert_s1 | 10 | 7933 | 803 | +4.56 | -12.77 | 5.6e-37 | 0.000 | -6.46 | 1.8e-10 | 0.000 | -11.65 | -6.14 | Y | Y |
| long_form | C2_finbert_s1 | 20 | 7902 | 794 | -0.08 | +5.98 | 2.3e-09 | 0.000 | +0.88 | 3.8e-01 | 1.000 | +4.90 | +1.48 | n | n |
| long_form | C2_finbert_s2 | 5 | 7951 | 809 | +1.73 | -7.25 | 4.5e-13 | 0.000 | -4.30 | 1.9e-05 | 0.001 | -9.35 | -4.56 | Y | Y |
| long_form | C2_finbert_s2 | 10 | 7933 | 803 | +0.24 | -7.92 | 2.6e-15 | 0.000 | -6.58 | 8.5e-11 | 0.000 | -7.93 | -6.18 | Y | Y |
| long_form | C2_finbert_s2 | 20 | 7902 | 794 | -5.19 | +7.93 | 2.5e-15 | 0.000 | -0.06 | 9.5e-01 | 1.000 | +7.07 | +0.60 | n | n |
| long_form | C2_finbert_s3 | 5 | 7951 | 809 | +2.39 | -9.85 | 9.6e-23 | 0.000 | -5.43 | 7.4e-08 | 0.000 | -9.16 | -4.74 | Y | Y |
| long_form | C2_finbert_s3 | 10 | 7933 | 803 | +3.03 | -9.38 | 8.8e-21 | 0.000 | -5.13 | 3.7e-07 | 0.000 | -9.44 | -5.88 | Y | Y |
| long_form | C2_finbert_s3 | 20 | 7902 | 794 | -0.76 | +2.07 | 3.8e-02 | 0.419 | -1.77 | 7.8e-02 | 1.000 | +0.98 | -1.86 | n | n |
| long_form | C2_finbert_s4 | 5 | 7951 | 809 | -1.09 | +5.94 | 2.9e-09 | 0.000 | +3.36 | 8.2e-04 | 0.030 | +3.77 | +1.78 | n | n |
| long_form | C2_finbert_s4 | 10 | 7933 | 803 | +2.28 | -10.36 | 5.3e-25 | 0.000 | -6.02 | 2.6e-09 | 0.000 | -10.93 | -7.07 | Y | Y |
| long_form | C2_finbert_s4 | 20 | 7902 | 794 | +3.45 | -16.89 | 7.3e-63 | 0.000 | -8.82 | 7.1e-18 | 0.000 | -14.23 | -8.33 | Y | Y |
| long_form | C3_roberta_s1 | 5 | 7951 | 809 | +0.67 | -10.47 | 1.7e-25 | 0.000 | -5.37 | 1.0e-07 | 0.000 | -8.90 | -5.60 | Y | Y |
| long_form | C3_roberta_s1 | 10 | 7933 | 803 | +1.98 | -6.37 | 2.1e-10 | 0.000 | -5.18 | 2.7e-07 | 0.000 | -5.48 | -4.16 | Y | Y |
| long_form | C3_roberta_s1 | 20 | 7902 | 794 | +0.24 | -6.89 | 6.1e-12 | 0.000 | -1.07 | 2.9e-01 | 1.000 | -5.46 | -1.44 | Y | n |
| long_form | C4_longformer | 5 | 7951 | 809 | +1.53 | -8.73 | 3.1e-18 | 0.000 | -5.99 | 3.2e-09 | 0.000 | -7.44 | -5.26 | Y | Y |
| long_form | C4_longformer | 10 | 7933 | 803 | -2.61 | +16.84 | 1.6e-62 | 0.000 | +10.55 | 1.8e-24 | 0.000 | +14.01 | +8.86 | n | n |
| long_form | C4_longformer | 20 | 7902 | 794 | -9.34 | +8.43 | 4.1e-17 | 0.000 | +3.46 | 5.7e-04 | 0.023 | +8.22 | +4.90 | n | n |
| long_form | C6_llmtext | 5 | 7951 | 809 | +1.79 | -10.27 | 1.3e-24 | 0.000 | -6.31 | 4.7e-10 | 0.000 | -10.21 | -6.97 | Y | Y |
| long_form | C6_llmtext | 10 | 7933 | 803 | +2.25 | -7.06 | 1.9e-12 | 0.000 | -7.92 | 8.2e-15 | 0.000 | -7.92 | -6.82 | Y | Y |
| long_form | C6_llmtext | 20 | 7902 | 794 | +0.27 | -4.15 | 3.4e-05 | 0.001 | -3.23 | 1.3e-03 | 0.042 | -5.82 | -3.90 | n | Y |
| long_form | D1_concat_mlp | 5 | 7951 | 809 | -0.01 | +0.63 | 5.3e-01 | 1.000 | +0.10 | 9.2e-01 | 1.000 | +0.17 | +0.15 | n | n |
| long_form | D1_concat_mlp | 10 | 7933 | 803 | +0.18 | -1.68 | 9.4e-02 | 0.802 | -0.67 | 5.1e-01 | 1.000 | +2.04 | +1.22 | n | n |
| long_form | D1_concat_mlp | 20 | 7902 | 794 | -0.08 | +0.33 | 7.4e-01 | 1.000 | -0.63 | 5.3e-01 | 1.000 | -0.72 | -0.63 | n | n |
| long_form | D2_gated_fusion | 5 | 7951 | 809 | -0.12 | +1.12 | 2.6e-01 | 1.000 | -0.29 | 7.7e-01 | 1.000 | -2.93 | -1.61 | n | n |
| long_form | D2_gated_fusion | 10 | 7933 | 803 | +0.13 | -0.56 | 5.7e-01 | 1.000 | +0.96 | 3.4e-01 | 1.000 | +1.93 | +1.46 | n | n |
| long_form | D2_gated_fusion | 20 | 7902 | 794 | -6.00 | +4.28 | 1.9e-05 | 0.000 | +4.27 | 2.2e-05 | 0.001 | +5.54 | +3.62 | n | n |
| long_form | D4_llmfused | 5 | 7951 | 809 | +0.15 | -3.29 | 1.0e-03 | 0.017 | -0.75 | 4.5e-01 | 1.000 | +1.17 | +0.92 | Y | n |
| long_form | D4_llmfused | 10 | 7933 | 803 | -0.15 | +1.10 | 2.7e-01 | 1.000 | -0.43 | 6.7e-01 | 1.000 | -0.75 | -0.12 | n | n |
| long_form | D4_llmfused | 20 | 7902 | 794 | +0.60 | -6.07 | 1.3e-09 | 0.000 | -3.58 | 3.6e-04 | 0.015 | -3.86 | -3.22 | Y | Y |
| event_driven | B1_bow_ridge | 5 | 25109 | 996 | +1.33 | -7.43 | 1.2e-13 | 0.000 | -3.35 | 8.4e-04 | 0.030 | -4.07 | -2.17 | Y | Y |
| event_driven | B1_bow_ridge | 10 | 25001 | 991 | +1.23 | -6.01 | 1.8e-09 | 0.000 | -3.25 | 1.2e-03 | 0.040 | -2.76 | -1.38 | Y | Y |
| event_driven | B1_bow_ridge | 20 | 24732 | 981 | +1.53 | -6.50 | 8.0e-11 | 0.000 | -3.10 | 2.0e-03 | 0.063 | -2.83 | -1.51 | Y | n |
| event_driven | B2_tfidf_ridge | 5 | 25109 | 996 | +1.21 | -7.05 | 1.9e-12 | 0.000 | -2.76 | 5.9e-03 | 0.165 | -6.71 | -3.29 | Y | n |
| event_driven | B2_tfidf_ridge | 10 | 25001 | 991 | +1.35 | -7.21 | 5.8e-13 | 0.000 | -2.97 | 3.1e-03 | 0.090 | -5.88 | -2.88 | Y | n |
| event_driven | B2_tfidf_ridge | 20 | 24732 | 981 | +1.84 | -8.22 | 2.1e-16 | 0.000 | -3.11 | 2.0e-03 | 0.063 | -5.25 | -2.09 | Y | n |
| event_driven | B3_lm_linear | 5 | 25109 | 996 | +0.25 | -4.82 | 1.4e-06 | 0.000 | -2.43 | 1.5e-02 | 0.394 | -2.41 | -1.28 | Y | n |
| event_driven | B3_lm_linear | 10 | 25001 | 991 | +0.20 | -3.03 | 2.4e-03 | 0.036 | -1.23 | 2.2e-01 | 1.000 | -1.44 | -0.62 | Y | n |
| event_driven | B3_lm_linear | 20 | 24732 | 981 | +0.26 | -2.64 | 8.2e-03 | 0.106 | -1.12 | 2.6e-01 | 1.000 | -0.89 | -0.42 | n | n |
| event_driven | B4_lm_features | 5 | 25109 | 996 | +0.18 | -3.95 | 8.0e-05 | 0.002 | -0.99 | 3.2e-01 | 1.000 | -4.51 | -2.92 | Y | n |
| event_driven | B4_lm_features | 10 | 25001 | 991 | +0.08 | -3.77 | 1.7e-04 | 0.003 | -2.10 | 3.6e-02 | 0.856 | -4.53 | -3.55 | Y | n |
| event_driven | B4_lm_features | 20 | 24732 | 981 | +0.25 | -3.11 | 1.9e-03 | 0.030 | -2.01 | 4.5e-02 | 0.984 | -2.86 | -1.88 | Y | n |
| event_driven | C2_finbert_s1 | 5 | 25109 | 996 | +2.14 | -14.36 | 1.4e-46 | 0.000 | -4.95 | 8.8e-07 | 0.000 | -9.25 | -3.18 | n | Y |
| event_driven | C2_finbert_s1 | 10 | 25001 | 991 | +2.10 | -14.19 | 1.6e-45 | 0.000 | -5.52 | 4.4e-08 | 0.000 | -10.99 | -4.80 | Y | Y |
| event_driven | C2_finbert_s1 | 20 | 24732 | 981 | +0.92 | -2.03 | 4.2e-02 | 0.422 | -0.50 | 6.2e-01 | 1.000 | +1.32 | +0.72 | n | n |
| event_driven | C6_llmtext | 5 | 25109 | 996 | +1.21 | -9.86 | 6.9e-23 | 0.000 | -5.04 | 5.4e-07 | 0.000 | -5.84 | -2.82 | Y | Y |
| event_driven | C6_llmtext | 10 | 25001 | 991 | +1.00 | -7.98 | 1.5e-15 | 0.000 | -3.76 | 1.8e-04 | 0.008 | -5.58 | -2.72 | Y | Y |
| event_driven | C6_llmtext | 20 | 24732 | 981 | +0.66 | -5.39 | 7.1e-08 | 0.000 | -1.98 | 4.8e-02 | 1.000 | -5.33 | -2.22 | n | n |
| event_driven | D2_gated_fusion | 5 | 25109 | 996 | +0.51 | -5.58 | 2.5e-08 | 0.000 | -0.69 | 4.9e-01 | 1.000 | -8.77 | -4.17 | n | n |
| event_driven | D2_gated_fusion | 10 | 25001 | 991 | +0.23 | -2.22 | 2.7e-02 | 0.321 | -0.56 | 5.8e-01 | 1.000 | -4.06 | -2.28 | n | n |
| event_driven | D2_gated_fusion | 20 | 24732 | 981 | -2.87 | +8.58 | 9.9e-18 | 0.000 | +3.00 | 2.8e-03 | 0.083 | +6.77 | +3.07 | n | n |
| event_driven | D4_llmfused | 5 | 25109 | 996 | -0.01 | +5.66 | 1.5e-08 | 0.000 | +3.37 | 7.7e-04 | 0.029 | +0.58 | +1.15 | n | n |
| event_driven | D4_llmfused | 10 | 25001 | 991 | -0.01 | +0.74 | 4.6e-01 | 1.000 | +0.66 | 5.1e-01 | 1.000 | -1.25 | -0.44 | n | n |
| event_driven | D4_llmfused | 20 | 24732 | 981 | -0.35 | +7.28 | 3.5e-13 | 0.000 | +4.69 | 3.1e-06 | 0.000 | +5.31 | +4.24 | n | n |