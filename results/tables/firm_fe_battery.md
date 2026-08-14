# P1-a — Firm-identity SPECIFICATION BATTERY (4 identity specs x 69-cell M1 grid)

## RESTATED vs BEFORE

| quantity | BEFORE (single spec: val-window firm mean; committed firm_identity_control.{csv,md}) | RESTATED (battery of 4 identity specs, byte-identical pipeline) |
|---|---|---|
| Holm<.05 survivors | 8/69 | **[8, 14]/69 across specs** |
| raw p<.05 survivors | 14/69 | [14, 27]/69 |
| long_form cells with NEGATIVE rel% (text hurts the firm-augmented ref) | 40/45 | 14-40/45 |
| zero-text firm-only ref beats plain f_R (raw p<.05) | 53/69 | 23-61/69 |
| cells surviving ALL 4 specs | — | **1**: event_driven/C6_llmtext/h5 |

Basis: seed2026 (identical to the committed control, so the identity-spec axis is the ONLY thing varied; seed-ensemble basis alignment is P0-2's deliverable). Rows: the 5-price-model intersection panel merged with each text model (same n_test per cell as the committed table). Reference f_R_firm = exp(a + b*log fA2 + c*log FIRM_STAT), val-fit/test-frozen; f_U_firm adds g*log f_text; day-clustered DM; Holm within each variant's 69-cell family.

## The four identity specifications

| variant | FIRM_STAT | imputation | information window | kills which objection |
|---|---|---|---|---|
| (i) valmean | firm mean label RV over its own val rows (pooled across horizons) | global val mean | 2020-21 (COVID) | — (baseline) |
| (ii) trainmean | firm mean label RV over its own train rows | global train mean | 2010-19, disjoint from val AND test | 'fitted on the same violent COVID val window' |
| (iii) pit252 | mean feature_rv_22d over the firm's strictly-prior filings within 365 calendar days (~252 trading days); else expanding mean of all prior filings; first-ever filing: own feature_rv_22d. Computed per filing on the FULL A2 panel; no label used; every input predates the filing timestamp | none needed (defined for 100% of rows) | fully point-in-time, per filing | 'coverage + global-mean imputation' AND 'val-window' simultaneously |
| (iv) ebshrink | w*firm_val_mean + (1-w)*global_val_mean, w = n/(n+10), n = firm's distinct val filings | global val mean (w=0) | 2020-21 | 'noisy per-firm val means with thin coverage' |

## Per-variant survivor summary (69 cells each)

| variant | raw p<.05 | Holm<.05 | long_form rel%<0 | zero-text beats f_R | firm cov (lf/ed) | test-obs cov (lf/ed) | mean rel% lf | mean rel% ed |
|---|---|---|---|---|---|---|---|---|
| (i) val-window firm mean [baseline] | 14/69 | **8/69** | 40/45 | 53/69 | 0.629/0.630 | 0.915/0.919 | -2.25% | -0.10% |
| (ii) train-period (2010-19) firm mean | 17/69 | **10/69** | 24/45 | 61/69 | 0.856/0.858 | 0.835/0.858 | -0.98% | -0.11% |
| (iii) PIT expanding pre-filing ~252td realized mean | 27/69 | **14/69** | 14/45 | 23/69 | 0.987/0.995 | 0.991/0.997 | +0.39% | +0.10% |
| (iv) EB-shrunk val mean (k=10) | 14/69 | **8/69** | 39/45 | 61/69 | 0.629/0.630 | 0.915/0.919 | -2.18% | -0.10% |

## Survivor identity across specs (Holm<.05)

- **(i) val-window firm mean [baseline]** (8): event_driven/B1_bow_ridge/h5; event_driven/B1_bow_ridge/h10; event_driven/B2_tfidf_ridge/h10; event_driven/C6_llmtext/h5; event_driven/C6_llmtext/h10; event_driven/C6_llmtext/h20; long_form/B3_lm_linear/h10; long_form/C6_llmtext/h20
- **(ii) train-period (2010-19) firm mean** (10): event_driven/B1_bow_ridge/h20; event_driven/C6_llmtext/h5; long_form/B2_tfidf_ridge/h5; long_form/B2_tfidf_ridge/h10; long_form/B3_lm_linear/h10; long_form/B3_lm_linear/h20; long_form/C6_llmtext/h5; long_form/C6_llmtext/h10; long_form/C6_llmtext/h20; long_form/D4_llmfused/h20
- **(iii) PIT expanding pre-filing ~252td realized mean** (14): event_driven/C2_finbert_s1/h5; event_driven/C6_llmtext/h5; long_form/B2_tfidf_ridge/h5; long_form/B2_tfidf_ridge/h10; long_form/B2_tfidf_ridge/h20; long_form/C2_finbert_s2/h10; long_form/C2_finbert_s4/h10; long_form/C2_finbert_s4/h20; long_form/C3_roberta_s1/h5; long_form/C4_longformer/h5; long_form/C6_llmtext/h5; long_form/C6_llmtext/h10; long_form/C6_llmtext/h20; long_form/D4_llmfused/h20
- **(iv) EB-shrunk val mean (k=10)** (8): event_driven/B1_bow_ridge/h5; event_driven/B1_bow_ridge/h10; event_driven/B2_tfidf_ridge/h10; event_driven/C6_llmtext/h5; event_driven/C6_llmtext/h10; event_driven/C6_llmtext/h20; long_form/B3_lm_linear/h10; long_form/B3_lm_linear/h20
- **ALL 4 specs** (1): event_driven/C6_llmtext/h5
- **ANY spec** (22): event_driven/B1_bow_ridge/h5; event_driven/B1_bow_ridge/h10; event_driven/B1_bow_ridge/h20; event_driven/B2_tfidf_ridge/h10; event_driven/C2_finbert_s1/h5; event_driven/C6_llmtext/h5; event_driven/C6_llmtext/h10; event_driven/C6_llmtext/h20; long_form/B2_tfidf_ridge/h5; long_form/B2_tfidf_ridge/h10; long_form/B2_tfidf_ridge/h20; long_form/B3_lm_linear/h10; long_form/B3_lm_linear/h20; long_form/C2_finbert_s2/h10; long_form/C2_finbert_s4/h10; long_form/C2_finbert_s4/h20; long_form/C3_roberta_s1/h5; long_form/C4_longformer/h5; long_form/C6_llmtext/h5; long_form/C6_llmtext/h10; long_form/C6_llmtext/h20; long_form/D4_llmfused/h20

## Reproduction gate — variant (i) vs committed firm_identity_control.csv

| metric | battery (i) | committed | match |
|---|---|---|---|
| raw count | 14 | 14 | PASS |
| holm count | 8 | 8 | PASS |
| lf_neg count | 40 | 40 | PASS |
| zerotext count | 53 | 53 | PASS |
| firm coverage (long_form) | 0.628707 | 0.628707 | PASS |
| max cell-wise |d(clustered DM)| over 69 cells | 4.44e-16 | 0 | PASS |
| max cell-wise |d(rel%)| / |d(Holm p)| / |d(zero-text rel%)| | 8.33e-17 / 9.78e-17 / 2.22e-16 | 0 | PASS |

## long_form — per-cell grid: rel% / clustered DM / Holm p per spec (bold = survives Holm<.05)
| model | h | valmean | trainmean | pit252 | ebshrink |
|---|---|---|---|---|---|
| B1_bow_ridge | 5 | -0.14 / +2.58 / 0.264 | +0.52 / -2.18 / 1.000 | +1.28 / -2.45 / 0.596 | -0.07 / +2.64 / 0.227 |
| B1_bow_ridge | 10 | -2.72 / +1.54 / 1.000 | -0.82 / +1.46 / 1.000 | +0.98 / -0.85 / 1.000 | -2.70 / +1.64 / 1.000 |
| B1_bow_ridge | 20 | -4.98 / +4.28 / 0.001 | -1.23 / +4.76 / 0.000 | +1.21 / -1.26 / 1.000 | -4.92 / +4.49 / 0.000 |
| B2_tfidf_ridge | 5 | -0.61 / +4.19 / 0.002 | **+1.21 / -3.51 / 0.024** | **+2.67 / -3.57 / 0.021** | -0.55 / +4.30 / 0.001 |
| B2_tfidf_ridge | 10 | -3.89 / +7.02 / 0.000 | **+0.18 / -7.07 / 0.000** | **+3.10 / -4.90 / 0.000** | -3.92 / +7.33 / 0.000 |
| B2_tfidf_ridge | 20 | -8.09 / +8.15 / 0.000 | -1.09 / +8.81 / 0.000 | **+3.68 / -3.46 / 0.030** | -8.25 / +8.39 / 0.000 |
| B3_lm_linear | 5 | +0.21 / -1.81 / 1.000 | +0.03 / -1.52 / 1.000 | -0.07 / -1.37 / 1.000 | +0.22 / -1.75 / 1.000 |
| B3_lm_linear | 10 | **+0.26 / -3.63 / 0.012** | **+1.07 / -3.56 / 0.021** | +0.74 / -2.67 / 0.329 | **+0.30 / -3.62 / 0.012** |
| B3_lm_linear | 20 | -0.02 / +3.47 / 0.020 | **+2.36 / -3.51 / 0.024** | +1.49 / -1.58 / 1.000 | **+0.08 / -3.48 / 0.019** |
| B4_lm_features | 5 | -0.14 / -1.41 / 1.000 | +0.01 / +1.25 / 1.000 | +0.05 / +1.84 / 1.000 | -0.20 / -1.34 / 1.000 |
| B4_lm_features | 10 | -2.77 / +4.10 / 0.002 | -1.15 / +3.13 / 0.080 | -0.48 / +1.88 / 1.000 | -2.84 / +4.00 / 0.003 |
| B4_lm_features | 20 | -7.13 / +4.62 / 0.000 | -3.32 / +3.19 / 0.067 | -0.32 / +0.75 / 1.000 | -7.44 / +4.54 / 0.000 |
| C1_bert_s1 | 5 | -0.35 / +1.07 / 1.000 | -0.00 / -1.21 / 1.000 | +0.04 / -1.77 / 1.000 | -0.33 / +0.73 / 1.000 |
| C1_bert_s1 | 10 | -6.06 / +3.26 / 0.039 | -2.59 / +1.24 / 1.000 | -0.33 / +0.93 / 1.000 | -5.71 / +3.14 / 0.058 |
| C1_bert_s1 | 20 | -5.80 / +5.92 / 0.000 | -6.88 / +5.90 / 0.000 | +1.99 / -3.11 / 0.100 | -5.37 / +6.22 / 0.000 |
| C2_finbert_s1 | 5 | -0.18 / +1.89 / 1.000 | -0.27 / -0.09 / 1.000 | +0.33 / -0.31 / 1.000 | -0.09 / +2.08 / 0.869 |
| C2_finbert_s1 | 10 | -0.60 / +4.09 / 0.002 | +1.22 / -3.10 / 0.085 | +3.00 / -2.75 / 0.268 | -0.33 / +4.24 / 0.001 |
| C2_finbert_s1 | 20 | -0.92 / +0.17 / 1.000 | -0.51 / -0.47 / 1.000 | +0.00 / +1.49 / 1.000 | -0.84 / +0.72 / 1.000 |
| C2_finbert_s2 | 5 | -0.31 / +2.44 / 0.373 | +0.28 / -1.13 / 1.000 | +1.15 / -2.73 / 0.277 | -0.25 / +2.64 / 0.227 |
| C2_finbert_s2 | 10 | -2.80 / +4.23 / 0.001 | -1.44 / +3.62 / 0.017 | **+0.13 / -3.95 / 0.005** | -2.74 / +4.69 / 0.000 |
| C2_finbert_s2 | 20 | -4.99 / +0.80 / 1.000 | -4.75 / +0.45 / 1.000 | -1.20 / -2.31 / 0.847 | -4.72 / +0.81 / 1.000 |
| C2_finbert_s3 | 5 | -1.16 / +4.03 / 0.003 | +0.19 / -1.91 / 1.000 | +1.75 / -3.32 / 0.050 | -0.99 / +3.96 / 0.003 |
| C2_finbert_s3 | 10 | -1.56 / +2.66 / 0.223 | +0.68 / -2.69 / 0.296 | +2.29 / -2.98 / 0.147 | -1.54 / +2.94 / 0.105 |
| C2_finbert_s3 | 20 | -0.74 / +1.23 / 1.000 | +2.14 / -0.51 / 1.000 | -6.28 / +0.93 / 1.000 | -0.43 / +1.11 / 1.000 |
| C2_finbert_s4 | 5 | -1.32 / +3.97 / 0.003 | -1.36 / +3.51 / 0.024 | -1.16 / +2.89 / 0.192 | -1.27 / +4.00 / 0.003 |
| C2_finbert_s4 | 10 | -3.24 / +4.32 / 0.001 | -0.96 / +4.30 / 0.001 | **+1.84 / -4.40 / 0.001** | -2.80 / +4.30 / 0.001 |
| C2_finbert_s4 | 20 | -5.18 / +6.51 / 0.000 | -3.89 / +8.32 / 0.000 | **+3.25 / -5.73 / 0.000** | -5.05 / +6.84 / 0.000 |
| C3_roberta_s1 | 5 | -2.56 / +3.84 / 0.005 | -1.34 / +3.09 / 0.086 | **+0.40 / -4.08 / 0.003** | -2.43 / +3.96 / 0.003 |
| C3_roberta_s1 | 10 | -0.45 / +4.14 / 0.002 | +0.08 / -1.64 / 1.000 | +0.65 / -2.81 / 0.232 | -0.35 / +4.46 / 0.000 |
| C3_roberta_s1 | 20 | -0.57 / +0.44 / 1.000 | -0.45 / -0.60 / 1.000 | +0.03 / +1.56 / 1.000 | -0.46 / +0.82 / 1.000 |
| C4_longformer | 5 | -0.97 / +4.59 / 0.000 | -0.12 / +3.25 / 0.057 | **+1.08 / -4.22 / 0.002** | -0.85 / +4.81 / 0.000 |
| C4_longformer | 10 | -10.84 / +9.10 / 0.000 | -10.60 / +7.76 / 0.000 | -2.31 / +8.50 / 0.000 | -10.38 / +9.42 / 0.000 |
| C4_longformer | 20 | -10.42 / +4.15 / 0.002 | -9.45 / +4.16 / 0.002 | -4.73 / +2.27 / 0.923 | -10.53 / +4.13 / 0.002 |
| C6_llmtext | 5 | -0.18 / +4.94 / 0.000 | **+1.23 / -5.24 / 0.000** | **+1.51 / -5.17 / 0.000** | -0.30 / +4.84 / 0.000 |
| C6_llmtext | 10 | -0.43 / +5.79 / 0.000 | **+1.60 / -6.34 / 0.000** | **+1.54 / -5.54 / 0.000** | -0.68 / +5.86 / 0.000 |
| C6_llmtext | 20 | **+0.08 / -4.15 / 0.002** | **+0.68 / -4.13 / 0.002** | **+0.19 / -3.63 / 0.017** | -0.03 / +4.29 / 0.001 |
| D1_concat_mlp | 5 | -0.06 / +0.55 / 1.000 | +0.01 / -0.18 / 1.000 | -0.01 / +0.35 / 1.000 | -0.03 / +0.31 / 1.000 |
| D1_concat_mlp | 10 | +0.01 / +0.60 / 1.000 | +0.30 / -0.35 / 1.000 | +0.29 / -0.63 / 1.000 | +0.06 / +0.53 / 1.000 |
| D1_concat_mlp | 20 | -0.04 / -0.59 / 1.000 | -0.14 / +1.08 / 1.000 | -0.91 / +1.64 / 1.000 | -0.04 / -0.29 / 1.000 |
| D2_gated_fusion | 5 | -0.39 / +0.82 / 1.000 | -0.17 / -0.02 / 1.000 | -0.12 / -0.47 / 1.000 | -0.43 / +0.87 / 1.000 |
| D2_gated_fusion | 10 | -0.58 / +1.19 / 1.000 | +0.19 / +0.72 / 1.000 | +0.42 / -0.01 / 1.000 | -0.50 / +1.16 / 1.000 |
| D2_gated_fusion | 20 | -8.47 / +3.43 / 0.023 | -5.94 / +3.75 / 0.011 | -1.76 / +3.20 / 0.075 | -8.21 / +3.38 / 0.028 |
| D4_llmfused | 5 | -0.00 / +0.69 / 1.000 | +0.08 / -0.63 / 1.000 | +0.12 / -0.75 / 1.000 | +0.01 / -0.67 / 1.000 |
| D4_llmfused | 10 | -0.15 / -0.45 / 1.000 | -0.13 / -0.63 / 1.000 | -0.11 / -0.75 / 1.000 | -0.13 / -0.47 / 1.000 |
| D4_llmfused | 20 | +0.08 / -2.87 / 0.122 | **+0.36 / -3.53 / 0.023** | **+0.25 / -3.77 / 0.010** | +0.10 / -2.89 / 0.117 |

## event_driven — per-cell grid: rel% / clustered DM / Holm p per spec (bold = survives Holm<.05)
| model | h | valmean | trainmean | pit252 | ebshrink |
|---|---|---|---|---|---|
| B1_bow_ridge | 5 | **+0.46 / -3.50 / 0.018** | +0.30 / -1.65 / 1.000 | +1.07 / -2.18 / 1.000 | **+0.45 / -3.34 / 0.030** |
| B1_bow_ridge | 10 | **+0.25 / -4.71 / 0.000** | +0.21 / -2.24 / 0.881 | +0.32 / -0.32 / 1.000 | **+0.25 / -4.48 / 0.000** |
| B1_bow_ridge | 20 | -0.19 / +5.50 / 0.000 | **+0.15 / -3.38 / 0.037** | -0.67 / +1.22 / 1.000 | -0.19 / +5.48 / 0.000 |
| B2_tfidf_ridge | 5 | +0.44 / -3.06 / 0.072 | +0.21 / -0.60 / 1.000 | +0.94 / -1.45 / 1.000 | +0.44 / -2.87 / 0.121 |
| B2_tfidf_ridge | 10 | **+0.28 / -4.28 / 0.001** | +0.23 / -1.40 / 1.000 | +0.53 / -0.25 / 1.000 | **+0.29 / -4.07 / 0.002** |
| B2_tfidf_ridge | 20 | -0.05 / +5.08 / 0.000 | +0.25 / -2.26 / 0.855 | -0.27 / +0.83 / 1.000 | -0.03 / +4.99 / 0.000 |
| B3_lm_linear | 5 | +0.11 / -3.00 / 0.086 | +0.11 / -1.92 / 1.000 | +0.21 / -2.18 / 1.000 | +0.11 / -2.85 / 0.127 |
| B3_lm_linear | 10 | +0.11 / -1.91 / 1.000 | +0.08 / -0.90 / 1.000 | +0.16 / -0.89 / 1.000 | +0.10 / -1.81 / 1.000 |
| B3_lm_linear | 20 | +0.17 / -2.12 / 0.793 | +0.15 / -1.03 / 1.000 | +0.17 / -0.57 / 1.000 | +0.16 / -2.03 / 0.942 |
| B4_lm_features | 5 | +0.22 / -1.56 / 1.000 | +0.17 / -1.28 / 1.000 | +0.19 / -1.21 / 1.000 | +0.21 / -1.45 / 1.000 |
| B4_lm_features | 10 | +0.13 / -2.60 / 0.253 | +0.07 / -2.55 / 0.431 | +0.08 / -2.80 / 0.237 | +0.13 / -2.52 / 0.298 |
| B4_lm_features | 20 | +0.11 / -2.38 / 0.418 | +0.16 / -1.81 / 1.000 | +0.20 / -1.61 / 1.000 | +0.10 / -2.27 / 0.568 |
| C2_finbert_s1 | 5 | -0.28 / +4.88 / 0.000 | -0.42 / +3.58 / 0.020 | **+1.86 / -3.47 / 0.030** | -0.16 / +4.84 / 0.000 |
| C2_finbert_s1 | 10 | -1.22 / +6.33 / 0.000 | -0.89 / +3.23 / 0.059 | +1.24 / -2.02 / 1.000 | -1.02 / +6.18 / 0.000 |
| C2_finbert_s1 | 20 | +0.58 / -0.81 / 1.000 | -1.38 / +1.35 / 1.000 | -4.88 / +3.04 / 0.123 | +0.57 / -0.83 / 1.000 |
| C6_llmtext | 5 | **+0.52 / -4.98 / 0.000** | **+0.67 / -3.64 / 0.016** | **+1.08 / -3.83 / 0.008** | **+0.50 / -5.18 / 0.000** |
| C6_llmtext | 10 | **+0.24 / -4.43 / 0.001** | +0.50 / -2.41 / 0.606 | +0.59 / -1.58 / 1.000 | **+0.19 / -4.79 / 0.000** |
| C6_llmtext | 20 | **+0.21 / -3.58 / 0.014** | +0.44 / -1.64 / 1.000 | +0.21 / -0.37 / 1.000 | **+0.14 / -4.00 / 0.003** |
| D2_gated_fusion | 5 | +0.38 / -1.34 / 1.000 | +0.25 / -0.54 / 1.000 | +0.43 / -0.46 / 1.000 | +0.35 / -1.16 / 1.000 |
| D2_gated_fusion | 10 | +0.14 / -0.63 / 1.000 | +0.04 / -0.26 / 1.000 | +0.07 / +0.01 / 1.000 | +0.12 / -0.54 / 1.000 |
| D2_gated_fusion | 20 | -4.60 / +2.93 / 0.105 | -3.48 / +2.35 / 0.701 | -0.81 / +0.96 / 1.000 | -4.65 / +3.01 / 0.085 |
| D4_llmfused | 5 | -0.07 / +3.19 / 0.048 | -0.01 / +3.11 / 0.082 | -0.00 / +2.84 / 0.216 | -0.08 / +3.21 / 0.047 |
| D4_llmfused | 10 | -0.02 / +0.91 / 1.000 | -0.01 / +0.40 / 1.000 | +0.00 / -0.17 / 1.000 | -0.02 / +0.90 / 1.000 |
| D4_llmfused | 20 | -0.25 / +3.48 / 0.019 | -0.39 / +4.50 / 0.000 | -0.33 / +4.11 / 0.003 | -0.24 / +3.50 / 0.019 |

## VERDICT

**ROBUST, with one sharpening. Under EVERY one of the four identity specifications at least 55/69 cells (80%) lose their Holm-significant text increment — the collapse from 29/69 (m1_clustered, same seed2026 basis, no identity control) to a single-digit-to-low-teens survivor set is NOT an artifact of the val-window/coverage/imputation choices of the committed control. The sharpening: WHICH cells survive is spec-dependent — only 1 cell(s) survive all four specs (event_driven/C6_llmtext/h5), so the only spec-robust residual is thinner than the committed 8/69.**

- Holm survivor interval across the 4 identity specs: **[8, 14]/69** (committed single-spec figure: 8/69; no-identity-control clustered baseline on the same seed2026 basis: 29/69 per m1_clustered — that grid uses the plain A2-merge panel, this battery the 5-price-model intersection panel of the committed control). Raw-p interval: [14, 27]/69 (was 14/69).
- Spec-robust survivors (ALL 4 specs, 1 cell(s)): event_driven/C6_llmtext/h5 — only these are citable as a residual increment that no identity specification absorbs. Union across specs: 22/69.
- Interpretive caveat that favours the absorption claim: the two specs with more survivors are the WEAKER identity instruments, not evidence of revived text signal. The zero-text check shows the PIT rolling stat improves the plain reference in only 23/69 cells (mean rel% -3.97 long_form / -1.27 event_driven — it mostly DUPLICATES HAR's own rv_22d input and adds noise), whereas the val/train/EB firm means improve it in 53-61/69 cells. What text proxies is the PERSISTENT firm level, which fixed-window firm means capture and a 1-year trailing RV mean does not.
- The train-period spec (ii) draws firm identity from 2010-19, disjoint from both val and test — its survivor count (10/69) shows the absorption is NOT a 2020-21 COVID-window artifact.
- The fully point-in-time spec (iii) is defined for 100% of test rows from strictly pre-filing information; even this deliberately conservative, HAR-overlapping proxy still kills 55/69 cells.
- Note for the residual-claim wording: the event_driven C6 cells survive the two val-based specs at all horizons, but h=10/h=20 fail trainmean and pit252 — the only horizon-robust C6 residual is h=5.