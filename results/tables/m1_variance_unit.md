# M1 — VARIANCE-UNIT primary (same seed-ensemble forecasts, QLIKE on variances)

## RESTATED vs ORIGINAL

The original grid evaluated QLIKE in VOLATILITY units, q(y, f) — a non-standard convention (the QLIKE literature works on variances). Here the SAME seed-ensemble forecasts and the SAME log-space val-fit combiner (unit-free) are evaluated with q(y^2, f^2), day-clustered DM. Only the evaluation loss changes.

| | vol-unit (restated primary) | variance-unit (this table) |
|---|---|---|
| genuine cells | 38/69 | **20/69** |
| convention-dependent cells (verdict differs) | — | **20** |


## long_form — variance-unit evaluation (ensemble, day-clustered)
| model | h | QLIKEv(R) | QLIKEv(U) | rel% | DM-Qv(clu) | p | Holm | placebo DM | genuine(var) | genuine(vol) | convention-dep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_bow_ridge | 5 | 0.5109 | 0.5049 | +1.18 | -1.42 | 0.1569 | 1.000 | +0.11 | no | YES | FLIP |
| B1_bow_ridge | 10 | 0.3863 | 0.3795 | +1.76 | -1.90 | 0.0581 | 1.000 | +0.71 | no | YES | FLIP |
| B1_bow_ridge | 20 | 0.2647 | 0.2570 | +2.90 | -3.15 | 0.0017 | 0.067 | -0.07 | no | YES | FLIP |
| B2_tfidf_ridge | 5 | 0.5109 | 0.4976 | +2.60 | -2.22 | 0.0270 | 0.674 | +0.10 | no | YES | FLIP |
| B2_tfidf_ridge | 10 | 0.3863 | 0.3762 | +2.60 | -6.32 | 0.0000 | 0.000 | -0.02 | YES | YES | - |
| B2_tfidf_ridge | 20 | 0.2647 | 0.2503 | +5.43 | -7.95 | 0.0000 | 0.000 | +0.05 | YES | YES | - |
| B3_lm_linear | 5 | 0.5109 | 0.5154 | -0.88 | -0.28 | 0.7776 | 1.000 | -0.81 | no | no | - |
| B3_lm_linear | 10 | 0.3863 | 0.3811 | +1.35 | -2.31 | 0.0212 | 0.552 | -0.16 | no | YES | FLIP |
| B3_lm_linear | 20 | 0.2647 | 0.2572 | +2.80 | -3.35 | 0.0008 | 0.038 | -1.06 | YES | YES | - |
| B4_lm_features | 5 | 0.5109 | 0.5115 | -0.11 | +1.56 | 0.1190 | 1.000 | -1.60 | no | no | - |
| B4_lm_features | 10 | 0.3863 | 0.3883 | -0.53 | +2.18 | 0.0295 | 0.709 | -0.23 | no | no | - |
| B4_lm_features | 20 | 0.2647 | 0.2687 | -1.53 | +2.81 | 0.0051 | 0.163 | -2.21 | no | no | - |
| C1_bert_s1 | 5 | 0.5109 | 0.5024 | +1.67 | -0.34 | 0.7339 | 1.000 | -1.51 | no | YES | FLIP |
| C1_bert_s1 | 10 | 0.3863 | 0.3921 | -1.52 | +3.90 | 0.0001 | 0.005 | +0.27 | no | no | - |
| C1_bert_s1 | 20 | 0.2647 | 0.2575 | +2.71 | -5.82 | 0.0000 | 0.000 | +0.23 | YES | YES | - |
| C2_finbert_s1 | 5 | 0.5109 | 0.4966 | +2.79 | -4.11 | 0.0000 | 0.002 | -0.14 | YES | YES | - |
| C2_finbert_s1 | 10 | 0.3863 | 0.3768 | +2.45 | -4.09 | 0.0000 | 0.003 | +0.37 | YES | YES | - |
| C2_finbert_s1 | 20 | 0.2647 | 0.2620 | +1.00 | -0.82 | 0.4101 | 1.000 | +0.58 | no | no | - |
| C2_finbert_s2 | 5 | 0.5109 | 0.5073 | +0.71 | -1.28 | 0.1995 | 1.000 | -0.35 | no | YES | FLIP |
| C2_finbert_s2 | 10 | 0.3863 | 0.3847 | +0.41 | -5.23 | 0.0000 | 0.000 | +0.18 | YES | YES | - |
| C2_finbert_s2 | 20 | 0.2647 | 0.2590 | +2.15 | -5.51 | 0.0000 | 0.000 | -0.47 | YES | YES | - |
| C2_finbert_s3 | 5 | 0.5109 | 0.4918 | +3.73 | -3.23 | 0.0013 | 0.056 | +0.34 | no | YES | FLIP |
| C2_finbert_s3 | 10 | 0.3863 | 0.3781 | +2.12 | -2.94 | 0.0034 | 0.118 | +0.38 | no | YES | FLIP |
| C2_finbert_s3 | 20 | 0.2647 | 0.2665 | -0.68 | +0.26 | 0.7920 | 1.000 | +0.71 | no | no | - |
| C2_finbert_s4 | 5 | 0.5109 | 0.5107 | +0.04 | -0.16 | 0.8699 | 1.000 | -0.65 | no | YES | FLIP |
| C2_finbert_s4 | 10 | 0.3863 | 0.3854 | +0.23 | -3.86 | 0.0001 | 0.006 | -0.05 | YES | YES | - |
| C2_finbert_s4 | 20 | 0.2647 | 0.2567 | +3.02 | -7.37 | 0.0000 | 0.000 | +0.09 | YES | YES | - |
| C3_roberta_s1 | 5 | 0.5109 | 0.5090 | +0.36 | -3.03 | 0.0025 | 0.092 | +0.34 | no | YES | FLIP |
| C3_roberta_s1 | 10 | 0.3863 | 0.3794 | +1.79 | -3.49 | 0.0005 | 0.024 | -0.08 | YES | YES | - |
| C3_roberta_s1 | 20 | 0.2647 | 0.2646 | +0.02 | -3.60 | 0.0003 | 0.016 | -0.29 | YES | YES | - |
| C4_longformer | 5 | 0.5109 | 0.5012 | +1.90 | -5.39 | 0.0000 | 0.000 | +0.69 | YES | YES | - |
| C4_longformer | 10 | 0.3863 | 0.3959 | -2.49 | +9.09 | 0.0000 | 0.000 | -0.07 | no | no | - |
| C4_longformer | 20 | 0.2647 | 0.2613 | +1.28 | -4.62 | 0.0000 | 0.000 | +0.98 | YES | YES | - |
| C6_llmtext | 5 | 0.5109 | 0.5044 | +1.28 | -3.16 | 0.0016 | 0.066 | -0.71 | no | YES | FLIP |
| C6_llmtext | 10 | 0.3863 | 0.3737 | +3.26 | -3.94 | 0.0001 | 0.005 | +0.03 | YES | YES | - |
| C6_llmtext | 20 | 0.2647 | 0.2642 | +0.17 | -2.45 | 0.0145 | 0.421 | -0.78 | no | YES | FLIP |
| D1_concat_mlp | 5 | 0.5109 | 0.5147 | -0.75 | +0.67 | 0.5059 | 1.000 | -0.12 | no | no | - |
| D1_concat_mlp | 10 | 0.3863 | 0.3886 | -0.60 | +2.87 | 0.0042 | 0.143 | +0.51 | no | no | - |
| D1_concat_mlp | 20 | 0.2647 | 0.2645 | +0.06 | -3.36 | 0.0008 | 0.037 | -0.25 | YES | YES | - |
| D2_gated_fusion | 5 | 0.5109 | 0.5100 | +0.18 | +0.23 | 0.8146 | 1.000 | -0.13 | no | no | - |
| D2_gated_fusion | 10 | 0.3863 | 0.3863 | -0.01 | +0.64 | 0.5247 | 1.000 | +0.65 | no | no | - |
| D2_gated_fusion | 20 | 0.2647 | 0.2713 | -2.52 | +4.99 | 0.0000 | 0.000 | -0.01 | no | no | - |
| D4_llmfused | 5 | 0.5109 | 0.5097 | +0.23 | -1.47 | 0.1421 | 1.000 | +0.66 | no | no | - |
| D4_llmfused | 10 | 0.3863 | 0.3866 | -0.08 | +0.35 | 0.7299 | 1.000 | +0.93 | no | no | - |
| D4_llmfused | 20 | 0.2647 | 0.2633 | +0.51 | -3.56 | 0.0004 | 0.019 | -0.17 | YES | YES | - |

## event_driven — variance-unit evaluation (ensemble, day-clustered)
| model | h | QLIKEv(R) | QLIKEv(U) | rel% | DM-Qv(clu) | p | Holm | placebo DM | genuine(var) | genuine(vol) | convention-dep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_bow_ridge | 5 | 0.6126 | 0.6014 | +1.84 | -2.80 | 0.0053 | 0.164 | +0.19 | no | YES | FLIP |
| B1_bow_ridge | 10 | 0.4163 | 0.4102 | +1.47 | -2.38 | 0.0174 | 0.471 | -0.11 | no | YES | FLIP |
| B1_bow_ridge | 20 | 0.2793 | 0.2744 | +1.77 | -2.65 | 0.0083 | 0.248 | +0.34 | no | YES | FLIP |
| B2_tfidf_ridge | 5 | 0.6126 | 0.6014 | +1.83 | -2.82 | 0.0049 | 0.161 | -0.08 | no | no | - |
| B2_tfidf_ridge | 10 | 0.4163 | 0.4084 | +1.90 | -3.18 | 0.0015 | 0.063 | +0.54 | no | no | - |
| B2_tfidf_ridge | 20 | 0.2793 | 0.2730 | +2.27 | -3.22 | 0.0013 | 0.056 | +0.46 | no | YES | FLIP |
| B3_lm_linear | 5 | 0.6126 | 0.6101 | +0.41 | -3.13 | 0.0018 | 0.069 | -0.31 | no | no | - |
| B3_lm_linear | 10 | 0.4163 | 0.4145 | +0.43 | -1.79 | 0.0736 | 1.000 | -0.81 | no | no | - |
| B3_lm_linear | 20 | 0.2793 | 0.2772 | +0.77 | -2.02 | 0.0434 | 0.999 | +0.41 | no | no | - |
| B4_lm_features | 5 | 0.6126 | 0.6113 | +0.22 | +0.42 | 0.6781 | 1.000 | +0.52 | no | no | - |
| B4_lm_features | 10 | 0.4163 | 0.4158 | +0.12 | -1.19 | 0.2345 | 1.000 | -0.34 | no | no | - |
| B4_lm_features | 20 | 0.2793 | 0.2773 | +0.73 | -1.81 | 0.0702 | 1.000 | +0.18 | no | no | - |
| C2_finbert_s1 | 5 | 0.6126 | 0.5840 | +4.68 | -7.83 | 0.0000 | 0.000 | -4.69 | no | YES | FLIP |
| C2_finbert_s1 | 10 | 0.4163 | 0.4045 | +2.83 | -4.29 | 0.0000 | 0.001 | +0.57 | YES | YES | - |
| C2_finbert_s1 | 20 | 0.2793 | 0.2666 | +4.55 | -4.38 | 0.0000 | 0.001 | +0.02 | YES | no | FLIP |
| C6_llmtext | 5 | 0.6126 | 0.6029 | +1.58 | -3.95 | 0.0001 | 0.004 | +0.47 | YES | YES | - |
| C6_llmtext | 10 | 0.4163 | 0.4121 | +1.01 | -2.41 | 0.0159 | 0.446 | +0.01 | no | YES | FLIP |
| C6_llmtext | 20 | 0.2793 | 0.2778 | +0.54 | -1.11 | 0.2672 | 1.000 | -0.40 | no | no | - |
| D2_gated_fusion | 5 | 0.6126 | 0.6109 | +0.29 | +1.24 | 0.2166 | 1.000 | +6.45 | no | no | - |
| D2_gated_fusion | 10 | 0.4163 | 0.4168 | -0.13 | +1.95 | 0.0516 | 1.000 | -1.03 | no | no | - |
| D2_gated_fusion | 20 | 0.2793 | 0.2889 | -3.43 | +3.64 | 0.0003 | 0.014 | +0.99 | no | no | - |
| D4_llmfused | 5 | 0.6126 | 0.6127 | -0.02 | +3.12 | 0.0019 | 0.071 | +3.60 | no | no | - |
| D4_llmfused | 10 | 0.4163 | 0.4165 | -0.04 | +1.58 | 0.1141 | 1.000 | +0.87 | no | no | - |
| D4_llmfused | 20 | 0.2793 | 0.2801 | -0.27 | +3.08 | 0.0022 | 0.080 | +0.29 | no | no | - |

## Convention-dependent cells

| disc | model | h | vol verdict | var verdict | vol Holm | var Holm |
|---|---|---|---|---|---|---|
| long_form | B1_bow_ridge | 5 | genuine | no | 0.005 | 1.000 |
| long_form | B1_bow_ridge | 10 | genuine | no | 0.001 | 1.000 |
| long_form | B1_bow_ridge | 20 | genuine | no | 0.000 | 0.067 |
| long_form | B2_tfidf_ridge | 5 | genuine | no | 0.000 | 0.674 |
| long_form | B3_lm_linear | 10 | genuine | no | 0.000 | 0.552 |
| long_form | C1_bert_s1 | 5 | genuine | no | 0.033 | 1.000 |
| long_form | C2_finbert_s2 | 5 | genuine | no | 0.000 | 1.000 |
| long_form | C2_finbert_s3 | 5 | genuine | no | 0.000 | 0.056 |
| long_form | C2_finbert_s3 | 10 | genuine | no | 0.000 | 0.118 |
| long_form | C2_finbert_s4 | 5 | genuine | no | 0.000 | 1.000 |
| long_form | C3_roberta_s1 | 5 | genuine | no | 0.000 | 0.092 |
| long_form | C6_llmtext | 5 | genuine | no | 0.000 | 0.066 |
| long_form | C6_llmtext | 20 | genuine | no | 0.033 | 0.421 |
| event_driven | B1_bow_ridge | 5 | genuine | no | 0.025 | 0.164 |
| event_driven | B1_bow_ridge | 10 | genuine | no | 0.033 | 0.471 |
| event_driven | B1_bow_ridge | 20 | genuine | no | 0.049 | 0.248 |
| event_driven | B2_tfidf_ridge | 20 | genuine | no | 0.049 | 0.056 |
| event_driven | C2_finbert_s1 | 5 | genuine | no | 0.000 | 0.000 |
| event_driven | C2_finbert_s1 | 20 | no | genuine | 0.657 | 0.001 |
| event_driven | C6_llmtext | 10 | genuine | no | 0.007 | 0.446 |

## Bottom line
- Variance-unit evaluation gives **20/69** genuine cells vs 38/69 in vol units; **20** cells are convention-dependent (listed above); conclusions for those cells must not be cited as convention-robust.
