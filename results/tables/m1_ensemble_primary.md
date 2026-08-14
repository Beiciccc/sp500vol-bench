# M1 PRIMARY (restated) — seed-ensemble predictions + day-clustered DM

## RESTATED vs ORIGINAL

| | ORIGINAL (seed2026 only, obs-order HAC DM) | RESTATED (3-seed ensemble, day-clustered DM) |
|---|---|---|
| genuine text-increment cells | 38/69 | **38/69** |
| inference unit | n_obs (~10-25 same-day filings treated independent) | n_days (median 809 days), HAC lag=h-1 in DAYS |
| seed handling | hardcoded seed2026 | per-observation mean across seeds (2026, 2027, 2028) for 3-seed C/D models |

Attribution: seed2026-only + clustered DM (clustering alone) gives 29/69 genuine cells; the ensemble step then moves this to 38/69. Of the 38 originally-genuine cells: **30 SURVIVE**, **8 are LOST**, and **8 cells are newly GAINED**.

Genuine = clustered DM-QLIKE < 0, Holm(clustered p) < .05 across the 69-cell grid, |clustered placebo DM| < 2. Combiner weights fit on validation only, frozen on test (unchanged). This table replaces the hardcoded-seed2026 primary and neutralizes the seed2027-flip objection: the primary object is now the seed-averaged forecast.

**Sanity (a): single-seed rows (A/B-anchored text models, C6, D4) reproduce the original grid QLIKE columns exactly — max|dQLIKE_R|=8.33e-17, max|dQLIKE_U|=8.33e-17, max|dg_log|=8.33e-17 over 36 cells: PASS.**


## long_form — ensemble primary (vol-unit QLIKE, day-clustered)
| model | h | seeds | n_test | n_days | QLIKE(R) | QLIKE(U) | rel% | g_log | DM-Q(clu) | p(clu) | Holm | placebo DM(clu) | CW t(clu) | orig DM-Q | orig Holm | orig genuine | NEW genuine | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_bow_ridge | 5 | 2026 | 7951 | 809 | 0.1209 | 0.1189 | +1.65 | +0.107 | -3.83 | 0.0001 | 0.005 | +0.43 | +5.73 | -6.75 | 0.000 | YES | YES | SURVIVES |
| B1_bow_ridge | 10 | 2026 | 7933 | 803 | 0.0873 | 0.0860 | +1.44 | +0.059 | -4.15 | 0.0000 | 0.001 | +0.39 | +4.40 | -9.18 | 0.000 | YES | YES | SURVIVES |
| B1_bow_ridge | 20 | 2026 | 7902 | 794 | 0.0701 | 0.0680 | +2.99 | +0.098 | -5.45 | 0.0000 | 0.000 | -0.25 | +5.79 | -13.08 | 0.000 | YES | YES | SURVIVES |
| B2_tfidf_ridge | 5 | 2026 | 7951 | 809 | 0.1209 | 0.1168 | +3.33 | +0.237 | -5.39 | 0.0000 | 0.000 | +0.51 | +8.28 | -11.34 | 0.000 | YES | YES | SURVIVES |
| B2_tfidf_ridge | 10 | 2026 | 7933 | 803 | 0.0873 | 0.0843 | +3.48 | +0.170 | -8.89 | 0.0000 | 0.000 | +0.33 | +8.86 | -15.30 | 0.000 | YES | YES | SURVIVES |
| B2_tfidf_ridge | 20 | 2026 | 7902 | 794 | 0.0701 | 0.0659 | +5.92 | +0.204 | -9.04 | 0.0000 | 0.000 | +0.30 | +9.21 | -19.62 | 0.000 | no | YES | GAINED |
| B3_lm_linear | 5 | 2026 | 7951 | 809 | 0.1209 | 0.1203 | +0.49 | +0.641 | -2.58 | 0.0100 | 0.189 | +0.59 | +5.09 | -1.70 | 0.802 | no | no | null-null |
| B3_lm_linear | 10 | 2026 | 7933 | 803 | 0.0873 | 0.0857 | +1.79 | +0.591 | -4.62 | 0.0000 | 0.000 | +0.27 | +5.72 | -5.45 | 0.000 | YES | YES | SURVIVES |
| B3_lm_linear | 20 | 2026 | 7902 | 794 | 0.0701 | 0.0676 | +3.48 | +0.582 | -4.69 | 0.0000 | 0.000 | -1.26 | +6.09 | -9.63 | 0.000 | YES | YES | SURVIVES |
| B4_lm_features | 5 | 2026 | 7951 | 809 | 0.1209 | 0.1207 | +0.11 | +0.139 | +1.02 | 0.3080 | 1.000 | +0.43 | +1.41 | -2.80 | 0.072 | no | no | null-null |
| B4_lm_features | 10 | 2026 | 7933 | 803 | 0.0873 | 0.0881 | -0.92 | -0.204 | +3.32 | 0.0010 | 0.028 | -1.69 | -3.66 | +8.66 | 0.000 | no | no | null-null |
| B4_lm_features | 20 | 2026 | 7902 | 794 | 0.0701 | 0.0714 | -1.92 | -0.188 | +3.38 | 0.0008 | 0.024 | -3.17 | -4.08 | +14.66 | 0.000 | no | no | null-null |
| C1_bert_s1 | 5 | 2026+2027+2028 | 7951 | 809 | 0.1209 | 0.1167 | +3.42 | +0.546 | -3.25 | 0.0012 | 0.033 | -0.03 | +6.39 | -0.96 | 1.000 | no | YES | GAINED |
| C1_bert_s1 | 10 | 2026+2027+2028 | 7933 | 803 | 0.0873 | 0.0880 | -0.85 | -0.129 | +3.06 | 0.0023 | 0.053 | -1.02 | -2.07 | +4.37 | 0.000 | no | no | null-null |
| C1_bert_s1 | 20 | 2026+2027+2028 | 7902 | 794 | 0.0701 | 0.0682 | +2.70 | +0.156 | -6.96 | 0.0000 | 0.000 | +0.39 | +6.91 | -15.11 | 0.000 | YES | YES | SURVIVES |
| C2_finbert_s1 | 5 | 2026+2027+2028 | 7951 | 809 | 0.1209 | 0.1186 | +1.90 | +0.339 | -4.53 | 0.0000 | 0.000 | -0.72 | +4.55 | -5.57 | 0.000 | YES | YES | SURVIVES |
| C2_finbert_s1 | 10 | 2026+2027+2028 | 7933 | 803 | 0.0873 | 0.0850 | +2.62 | +0.151 | -6.67 | 0.0000 | 0.000 | +0.80 | +6.72 | -12.77 | 0.000 | YES | YES | SURVIVES |
| C2_finbert_s1 | 20 | 2026+2027+2028 | 7902 | 794 | 0.0701 | 0.0705 | -0.59 | +0.234 | -0.40 | 0.6894 | 1.000 | +0.78 | +1.41 | +5.98 | 0.000 | no | no | null-null |
| C2_finbert_s2 | 5 | 2026+2027+2028 | 7951 | 809 | 0.1209 | 0.1194 | +1.21 | +0.175 | -5.57 | 0.0000 | 0.000 | +0.21 | +6.08 | -7.25 | 0.000 | YES | YES | SURVIVES |
| C2_finbert_s2 | 10 | 2026+2027+2028 | 7933 | 803 | 0.0873 | 0.0869 | +0.48 | +0.053 | -7.59 | 0.0000 | 0.000 | +0.52 | +6.90 | -7.92 | 0.000 | YES | YES | SURVIVES |
| C2_finbert_s2 | 20 | 2026+2027+2028 | 7902 | 794 | 0.0701 | 0.0689 | +1.68 | +0.213 | -4.89 | 0.0000 | 0.000 | -0.39 | +5.71 | +7.93 | 0.000 | no | YES | GAINED |
| C2_finbert_s3 | 5 | 2026+2027+2028 | 7951 | 809 | 0.1209 | 0.1174 | +2.90 | +0.284 | -5.36 | 0.0000 | 0.000 | +0.38 | +6.31 | -9.85 | 0.000 | YES | YES | SURVIVES |
| C2_finbert_s3 | 10 | 2026+2027+2028 | 7933 | 803 | 0.0873 | 0.0853 | +2.31 | +0.262 | -5.26 | 0.0000 | 0.000 | +0.55 | +7.10 | -9.38 | 0.000 | YES | YES | SURVIVES |
| C2_finbert_s3 | 20 | 2026+2027+2028 | 7902 | 794 | 0.0701 | 0.0728 | -3.86 | +0.302 | +1.68 | 0.0941 | 0.988 | +1.06 | +0.19 | +2.07 | 0.419 | no | no | null-null |
| C2_finbert_s4 | 5 | 2026+2027+2028 | 7951 | 809 | 0.1209 | 0.1191 | +1.46 | +0.300 | -4.78 | 0.0000 | 0.000 | +0.37 | +6.62 | +5.94 | 0.000 | no | YES | GAINED |
| C2_finbert_s4 | 10 | 2026+2027+2028 | 7933 | 803 | 0.0873 | 0.0870 | +0.36 | +0.053 | -6.82 | 0.0000 | 0.000 | -0.24 | +7.12 | -10.36 | 0.000 | YES | YES | SURVIVES |
| C2_finbert_s4 | 20 | 2026+2027+2028 | 7902 | 794 | 0.0701 | 0.0679 | +3.08 | +0.122 | -8.34 | 0.0000 | 0.000 | +0.24 | +8.66 | -16.89 | 0.000 | YES | YES | SURVIVES |
| C3_roberta_s1 | 5 | 2026+2027+2028 | 7951 | 809 | 0.1209 | 0.1205 | +0.30 | +0.030 | -5.13 | 0.0000 | 0.000 | +0.06 | +5.64 | -10.47 | 0.000 | YES | YES | SURVIVES |
| C3_roberta_s1 | 10 | 2026+2027+2028 | 7933 | 803 | 0.0873 | 0.0857 | +1.84 | +0.254 | -5.06 | 0.0000 | 0.000 | -0.73 | +5.26 | -6.37 | 0.000 | YES | YES | SURVIVES |
| C3_roberta_s1 | 20 | 2026+2027+2028 | 7902 | 794 | 0.0701 | 0.0701 | +0.02 | +0.004 | -3.90 | 0.0001 | 0.004 | -0.44 | +4.14 | -6.89 | 0.000 | YES | YES | SURVIVES |
| C4_longformer | 5 | 2026+2027+2028 | 7951 | 809 | 0.1209 | 0.1191 | +1.47 | +0.161 | -6.18 | 0.0000 | 0.000 | +0.09 | +6.45 | -8.73 | 0.000 | YES | YES | SURVIVES |
| C4_longformer | 10 | 2026+2027+2028 | 7933 | 803 | 0.0873 | 0.0899 | -2.95 | -0.123 | +10.32 | 0.0000 | 0.000 | +0.23 | -7.62 | +16.84 | 0.000 | no | no | null-null |
| C4_longformer | 20 | 2026+2027+2028 | 7902 | 794 | 0.0701 | 0.0694 | +0.91 | +0.114 | -4.38 | 0.0000 | 0.001 | +1.20 | +5.42 | +8.43 | 0.000 | no | YES | GAINED |
| C6_llmtext | 5 | 2026 | 7951 | 809 | 0.1209 | 0.1187 | +1.79 | +0.254 | -6.31 | 0.0000 | 0.000 | +0.83 | +7.66 | -10.27 | 0.000 | YES | YES | SURVIVES |
| C6_llmtext | 10 | 2026 | 7933 | 803 | 0.0873 | 0.0853 | +2.25 | +0.333 | -7.92 | 0.0000 | 0.000 | -1.45 | +7.38 | -7.06 | 0.000 | YES | YES | SURVIVES |
| C6_llmtext | 20 | 2026 | 7902 | 794 | 0.0701 | 0.0699 | +0.27 | +0.078 | -3.23 | 0.0013 | 0.033 | -1.44 | +4.30 | -4.15 | 0.001 | no | YES | GAINED |
| D1_concat_mlp | 5 | 2026+2027+2028 | 7951 | 809 | 0.1209 | 0.1221 | -1.04 | -0.719 | +2.66 | 0.0081 | 0.161 | +0.79 | -1.43 | +0.63 | 1.000 | no | no | null-null |
| D1_concat_mlp | 10 | 2026+2027+2028 | 7933 | 803 | 0.0873 | 0.0877 | -0.46 | -0.371 | +3.54 | 0.0004 | 0.014 | -0.76 | -2.31 | -1.68 | 0.802 | no | no | null-null |
| D1_concat_mlp | 20 | 2026+2027+2028 | 7902 | 794 | 0.0701 | 0.0700 | +0.12 | +0.045 | -5.57 | 0.0000 | 0.000 | -0.61 | +4.06 | +0.33 | 1.000 | no | YES | GAINED |
| D2_gated_fusion | 5 | 2026+2027+2028 | 7951 | 809 | 0.1209 | 0.1206 | +0.19 | +0.150 | -2.11 | 0.0355 | 0.604 | +0.76 | +2.55 | +1.12 | 1.000 | no | no | null-null |
| D2_gated_fusion | 10 | 2026+2027+2028 | 7933 | 803 | 0.0873 | 0.0873 | -0.02 | +0.018 | +2.02 | 0.0438 | 0.657 | -0.41 | +0.50 | -0.56 | 1.000 | no | no | null-null |
| D2_gated_fusion | 20 | 2026+2027+2028 | 7902 | 794 | 0.0701 | 0.0717 | -2.27 | -0.638 | +5.56 | 0.0000 | 0.000 | -0.18 | -3.92 | +4.28 | 0.000 | no | no | null-null |
| D4_llmfused | 5 | 2026 | 7951 | 809 | 0.1209 | 0.1207 | +0.15 | -0.026 | -0.75 | 0.4512 | 1.000 | +1.00 | -0.84 | -3.29 | 0.017 | YES | no | LOST |
| D4_llmfused | 10 | 2026 | 7933 | 803 | 0.0873 | 0.0874 | -0.15 | +0.096 | -0.43 | 0.6687 | 1.000 | +0.87 | +0.71 | +1.10 | 1.000 | no | no | null-null |
| D4_llmfused | 20 | 2026 | 7902 | 794 | 0.0701 | 0.0697 | +0.60 | +0.083 | -3.58 | 0.0004 | 0.013 | -0.49 | +3.28 | -6.07 | 0.000 | YES | YES | SURVIVES |

## event_driven — ensemble primary (vol-unit QLIKE, day-clustered)
| model | h | seeds | n_test | n_days | QLIKE(R) | QLIKE(U) | rel% | g_log | DM-Q(clu) | p(clu) | Holm | placebo DM(clu) | CW t(clu) | orig DM-Q | orig Holm | orig genuine | NEW genuine | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_bow_ridge | 5 | 2026 | 25109 | 996 | 0.1265 | 0.1248 | +1.33 | +0.230 | -3.35 | 0.0008 | 0.025 | +0.59 | +7.09 | -7.43 | 0.000 | YES | YES | SURVIVES |
| B1_bow_ridge | 10 | 2026 | 25001 | 991 | 0.0883 | 0.0872 | +1.23 | +0.236 | -3.25 | 0.0012 | 0.033 | +0.05 | +6.70 | -6.01 | 0.000 | YES | YES | SURVIVES |
| B1_bow_ridge | 20 | 2026 | 24732 | 981 | 0.0645 | 0.0636 | +1.53 | +0.241 | -3.10 | 0.0020 | 0.049 | +0.45 | +6.43 | -6.50 | 0.000 | YES | YES | SURVIVES |
| B2_tfidf_ridge | 5 | 2026 | 25109 | 996 | 0.1265 | 0.1250 | +1.21 | +0.251 | -2.76 | 0.0059 | 0.124 | +0.39 | +7.42 | -7.05 | 0.000 | YES | no | LOST |
| B2_tfidf_ridge | 10 | 2026 | 25001 | 991 | 0.0883 | 0.0871 | +1.35 | +0.243 | -2.97 | 0.0031 | 0.068 | +0.38 | +7.10 | -7.21 | 0.000 | YES | no | LOST |
| B2_tfidf_ridge | 20 | 2026 | 24732 | 981 | 0.0645 | 0.0634 | +1.84 | +0.265 | -3.11 | 0.0020 | 0.049 | +0.42 | +5.98 | -8.22 | 0.000 | YES | YES | SURVIVES |
| B3_lm_linear | 5 | 2026 | 25109 | 996 | 0.1265 | 0.1262 | +0.25 | +0.680 | -2.43 | 0.0152 | 0.273 | +0.01 | +2.73 | -4.82 | 0.000 | YES | no | LOST |
| B3_lm_linear | 10 | 2026 | 25001 | 991 | 0.0883 | 0.0881 | +0.20 | +0.710 | -1.23 | 0.2193 | 1.000 | +0.16 | +1.95 | -3.03 | 0.036 | YES | no | LOST |
| B3_lm_linear | 20 | 2026 | 24732 | 981 | 0.0645 | 0.0644 | +0.26 | +0.726 | -1.12 | 0.2611 | 1.000 | +0.84 | +2.10 | -2.64 | 0.106 | no | no | null-null |
| B4_lm_features | 5 | 2026 | 25109 | 996 | 0.1265 | 0.1263 | +0.18 | +1.140 | -0.99 | 0.3205 | 1.000 | +0.46 | +4.16 | -3.95 | 0.002 | YES | no | LOST |
| B4_lm_features | 10 | 2026 | 25001 | 991 | 0.0883 | 0.0882 | +0.08 | +0.428 | -2.10 | 0.0356 | 0.604 | -0.05 | +4.19 | -3.77 | 0.003 | YES | no | LOST |
| B4_lm_features | 20 | 2026 | 24732 | 981 | 0.0645 | 0.0644 | +0.25 | +0.569 | -2.01 | 0.0447 | 0.657 | +0.47 | +3.19 | -3.11 | 0.030 | YES | no | LOST |
| C2_finbert_s1 | 5 | 2026+2027+2028 | 25109 | 996 | 0.1265 | 0.1233 | +2.57 | +0.233 | -5.92 | 0.0000 | 0.000 | -1.72 | +6.06 | -14.36 | 0.000 | no | YES | GAINED |
| C2_finbert_s1 | 10 | 2026+2027+2028 | 25001 | 991 | 0.0883 | 0.0861 | +2.42 | +0.225 | -5.68 | 0.0000 | 0.000 | +0.96 | +8.56 | -14.19 | 0.000 | YES | YES | SURVIVES |
| C2_finbert_s1 | 20 | 2026+2027+2028 | 24732 | 981 | 0.0645 | 0.0635 | +1.58 | +0.284 | -1.94 | 0.0521 | 0.657 | +0.33 | +3.81 | -2.03 | 0.422 | no | no | null-null |
| C6_llmtext | 5 | 2026 | 25109 | 996 | 0.1265 | 0.1250 | +1.21 | +0.264 | -5.04 | 0.0000 | 0.000 | +0.10 | +5.49 | -9.86 | 0.000 | YES | YES | SURVIVES |
| C6_llmtext | 10 | 2026 | 25001 | 991 | 0.0883 | 0.0874 | +1.00 | +0.281 | -3.76 | 0.0002 | 0.007 | +0.95 | +5.21 | -7.98 | 0.000 | YES | YES | SURVIVES |
| C6_llmtext | 20 | 2026 | 24732 | 981 | 0.0645 | 0.0641 | +0.66 | +0.245 | -1.98 | 0.0477 | 0.657 | +1.11 | +4.39 | -5.39 | 0.000 | no | no | null-null |
| D2_gated_fusion | 5 | 2026+2027+2028 | 25109 | 996 | 0.1265 | 0.1258 | +0.56 | +0.362 | -1.70 | 0.0898 | 0.988 | +3.00 | +5.67 | -5.58 | 0.000 | no | no | null-null |
| D2_gated_fusion | 10 | 2026+2027+2028 | 25001 | 991 | 0.0883 | 0.0881 | +0.18 | +0.629 | -0.42 | 0.6769 | 1.000 | -0.20 | +3.95 | -2.22 | 0.321 | no | no | null-null |
| D2_gated_fusion | 20 | 2026+2027+2028 | 24732 | 981 | 0.0645 | 0.0663 | -2.76 | -0.694 | +3.55 | 0.0004 | 0.014 | -0.44 | -1.85 | +8.58 | 0.000 | no | no | null-null |
| D4_llmfused | 5 | 2026 | 25109 | 996 | 0.1265 | 0.1265 | -0.01 | +0.002 | +3.37 | 0.0008 | 0.024 | +1.58 | -1.12 | +5.66 | 0.000 | no | no | null-null |
| D4_llmfused | 10 | 2026 | 25001 | 991 | 0.0883 | 0.0883 | -0.01 | +0.014 | +0.66 | 0.5109 | 1.000 | -0.08 | +0.70 | +0.74 | 1.000 | no | no | null-null |
| D4_llmfused | 20 | 2026 | 24732 | 981 | 0.0645 | 0.0648 | -0.35 | -0.066 | +4.69 | 0.0000 | 0.000 | -0.19 | -4.00 | +7.28 | 0.000 | no | no | null-null |

## Flips

**LOST** (genuine under seed2026+obs-order DM, NOT genuine under ensemble+clustered): long_form/D4_llmfused/h5 (orig Holm=0.017 -> clu Holm=1.000); event_driven/B2_tfidf_ridge/h5 (orig Holm=0.000 -> clu Holm=0.124); event_driven/B2_tfidf_ridge/h10 (orig Holm=0.000 -> clu Holm=0.068); event_driven/B3_lm_linear/h5 (orig Holm=0.000 -> clu Holm=0.273); event_driven/B3_lm_linear/h10 (orig Holm=0.036 -> clu Holm=1.000); event_driven/B4_lm_features/h5 (orig Holm=0.002 -> clu Holm=1.000); event_driven/B4_lm_features/h10 (orig Holm=0.003 -> clu Holm=0.604); event_driven/B4_lm_features/h20 (orig Holm=0.030 -> clu Holm=0.657)

**GAINED**: long_form/B2_tfidf_ridge/h20; long_form/C1_bert_s1/h5; long_form/C2_finbert_s2/h20; long_form/C2_finbert_s4/h5; long_form/C4_longformer/h20; long_form/C6_llmtext/h20; long_form/D1_concat_mlp/h20; event_driven/C2_finbert_s1/h5


## Bottom line
- **38/69** cells keep a genuine, placebo-confirmed text increment under the honest primary (seed-ensemble + day-clustered DM), vs 38/69 originally; 8 lost, 8 gained, 30 survive.
- Clustering alone (seed2026): 29/69 — the drop from 38 is the t-inflation the reviewer flagged; the ensemble step recovers/moves cells on top of that.
