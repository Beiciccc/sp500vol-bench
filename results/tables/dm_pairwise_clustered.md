# Pairwise DM vs A2_har_rv on squared error — CLUSTERED restatement

## RESTATED vs ORIGINAL

| quantity | ORIGINAL (obs-level HAC) | RESTATED (day-clustered) |
|---|---|---|
| challengers significantly BETTER than A2 (Holm<.05) | 0/180 | **0/180** |
| ... even at RAW p<.05 (no Holm) | - | 0/180 |
| challengers significantly WORSE than A2 (Holm<.05) | 174/180 | 155/180 |
| median \|DM\| shrink factor | 1.00 | 0.31 |

Same seed-ensembled test-split forecasts and inner-joined sample as `dm_pairwise.csv`; only inference changes (daily-mean SE differential, HAC lag=h-1 in days). Holm here is applied WITHIN each (disclosure,horizon) group over the vs-A2 challenger set only — a WEAKER correction than the original 420-pair family, i.e. conservative for the '0 challengers beat A2' headline. Original obs-level columns are copied from dm_pairwise.csv for the before/after.

| disclosure | h | challenger | n_obs | n_days | dm obs | p_holm obs | dm clust | p clust | p_holm clust | verdict clust |
|---|---|---|---|---|---|---|---|---|---|---|
| combined | 5 | D3_gteqwen2 | 33060 | 996 | +9.50 | 0.0000 | +2.69 | 0.0072 | 0.0072 | sig worse |
| combined | 5 | D3_qwen3 | 33060 | 996 | +15.43 | 0.0000 | +3.80 | 0.0002 | 0.0003 | sig worse |
| combined | 5 | D3_e5mistral | 33060 | 996 | +18.30 | 0.0000 | +4.07 | 0.0001 | 0.0002 | sig worse |
| combined | 5 | A3_garch | 33060 | 996 | +8.22 | 0.0000 | +6.08 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | A5_arima | 33060 | 996 | +9.84 | 0.0000 | +6.17 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | A4_egarch | 33060 | 996 | +5.58 | 0.0000 | +6.26 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | D1_concat_mlp | 33060 | 996 | +26.85 | 0.0000 | +6.85 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | D2_gated_fusion | 33060 | 996 | +27.82 | 0.0000 | +7.22 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C2_finbert_s1 | 33060 | 996 | +27.62 | 0.0000 | +8.19 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C3_roberta_s1 | 33060 | 996 | +29.67 | 0.0000 | +8.66 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C2_finbert_s2 | 33060 | 996 | +29.06 | 0.0000 | +8.73 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C5_qwen3 | 33060 | 996 | +27.98 | 0.0000 | +8.78 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C2_finbert_s3 | 33060 | 996 | +30.00 | 0.0000 | +8.89 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C5_gteqwen2 | 33060 | 996 | +28.39 | 0.0000 | +8.92 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C4_longformer | 33060 | 996 | +31.09 | 0.0000 | +9.01 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C5_e5mistral | 33060 | 996 | +30.40 | 0.0000 | +9.07 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C1_bert_s1 | 33060 | 996 | +31.60 | 0.0000 | +9.19 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | B2_tfidf_ridge | 33060 | 996 | +33.24 | 0.0000 | +9.47 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C2_finbert_s4 | 33060 | 996 | +33.84 | 0.0000 | +9.49 | 0.0000 | 0.0000 | sig worse |
| combined | 5 | C1_bert_s2 | 33060 | 996 | +33.16 | 0.0000 | +9.56 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | D3_gteqwen2 | 32934 | 991 | +3.83 | 0.0083 | +0.99 | 0.3232 | 0.3232 | worse(ns) |
| combined | 10 | D3_qwen3 | 32934 | 991 | +7.83 | 0.0000 | +1.90 | 0.0573 | 0.1146 | worse(ns) |
| combined | 10 | D3_e5mistral | 32934 | 991 | +9.50 | 0.0000 | +2.14 | 0.0324 | 0.0972 | worse(ns) |
| combined | 10 | A5_arima | 32934 | 991 | +7.70 | 0.0000 | +4.12 | 0.0000 | 0.0002 | sig worse |
| combined | 10 | D1_concat_mlp | 32934 | 991 | +20.48 | 0.0000 | +4.74 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | A3_garch | 32934 | 991 | +7.76 | 0.0000 | +4.78 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | D2_gated_fusion | 32934 | 991 | +22.69 | 0.0000 | +5.34 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | A4_egarch | 32934 | 991 | +4.80 | 0.0001 | +5.46 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C2_finbert_s1 | 32934 | 991 | +21.68 | 0.0000 | +5.97 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C1_bert_s1 | 32934 | 991 | +20.75 | 0.0000 | +6.05 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C2_finbert_s3 | 32934 | 991 | +26.09 | 0.0000 | +6.50 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C2_finbert_s4 | 32934 | 991 | +25.61 | 0.0000 | +6.65 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C4_longformer | 32934 | 991 | +25.78 | 0.0000 | +6.80 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C1_bert_s2 | 32934 | 991 | +27.51 | 0.0000 | +6.85 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C2_finbert_s2 | 32934 | 991 | +27.51 | 0.0000 | +6.89 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C3_roberta_s1 | 32934 | 991 | +29.97 | 0.0000 | +7.54 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C5_qwen3 | 32934 | 991 | +26.84 | 0.0000 | +7.63 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C5_e5mistral | 32934 | 991 | +28.56 | 0.0000 | +7.70 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | C5_gteqwen2 | 32934 | 991 | +27.72 | 0.0000 | +7.79 | 0.0000 | 0.0000 | sig worse |
| combined | 10 | B2_tfidf_ridge | 32934 | 991 | +31.51 | 0.0000 | +8.06 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | D3_gteqwen2 | 32634 | 981 | +0.52 | 1.0000 | +0.24 | 0.8131 | 0.8131 | worse(ns) |
| combined | 20 | D3_qwen3 | 32634 | 981 | +7.60 | 0.0000 | +1.93 | 0.0539 | 0.1078 | worse(ns) |
| combined | 20 | A5_arima | 32634 | 981 | +5.98 | 0.0000 | +2.27 | 0.0235 | 0.0705 | worse(ns) |
| combined | 20 | D3_e5mistral | 32634 | 981 | +8.92 | 0.0000 | +2.42 | 0.0155 | 0.0620 | worse(ns) |
| combined | 20 | A3_garch | 32634 | 981 | +8.30 | 0.0000 | +3.49 | 0.0005 | 0.0025 | sig worse |
| combined | 20 | D1_concat_mlp | 32634 | 981 | +17.93 | 0.0000 | +3.92 | 0.0001 | 0.0006 | sig worse |
| combined | 20 | A4_egarch | 32634 | 981 | +4.82 | 0.0001 | +4.18 | 0.0000 | 0.0002 | sig worse |
| combined | 20 | C2_finbert_s4 | 32634 | 981 | +22.19 | 0.0000 | +4.98 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | C2_finbert_s1 | 32634 | 981 | +22.96 | 0.0000 | +5.25 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | C2_finbert_s3 | 32634 | 981 | +23.46 | 0.0000 | +5.35 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | C4_longformer | 32634 | 981 | +22.85 | 0.0000 | +5.36 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | C2_finbert_s2 | 32634 | 981 | +24.82 | 0.0000 | +5.50 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | D2_gated_fusion | 32634 | 981 | +24.57 | 0.0000 | +5.65 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | C1_bert_s2 | 32634 | 981 | +24.82 | 0.0000 | +5.67 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | C3_roberta_s1 | 32634 | 981 | +25.29 | 0.0000 | +5.88 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | C1_bert_s1 | 32634 | 981 | +26.32 | 0.0000 | +6.12 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | C5_qwen3 | 32634 | 981 | +25.97 | 0.0000 | +6.38 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | C5_e5mistral | 32634 | 981 | +27.08 | 0.0000 | +6.52 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | C5_gteqwen2 | 32634 | 981 | +26.37 | 0.0000 | +6.61 | 0.0000 | 0.0000 | sig worse |
| combined | 20 | B2_tfidf_ridge | 32634 | 981 | +29.19 | 0.0000 | +6.64 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | D3_gteqwen2 | 25109 | 996 | +5.81 | 0.0000 | +2.24 | 0.0252 | 0.0252 | sig worse |
| event_driven | 5 | D3_qwen3 | 25109 | 996 | +13.55 | 0.0000 | +3.73 | 0.0002 | 0.0004 | sig worse |
| event_driven | 5 | D3_e5mistral | 25109 | 996 | +15.11 | 0.0000 | +3.92 | 0.0001 | 0.0003 | sig worse |
| event_driven | 5 | A3_garch | 25109 | 996 | +4.84 | 0.0001 | +4.33 | 0.0000 | 0.0001 | sig worse |
| event_driven | 5 | A5_arima | 25109 | 996 | +8.02 | 0.0000 | +5.48 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | A4_egarch | 25109 | 996 | +5.38 | 0.0000 | +5.75 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | D2_gated_fusion | 25109 | 996 | +21.38 | 0.0000 | +6.47 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | D1_concat_mlp | 25109 | 996 | +24.18 | 0.0000 | +6.98 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C2_finbert_s1 | 25109 | 996 | +18.80 | 0.0000 | +7.18 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C2_finbert_s3 | 25109 | 996 | +20.32 | 0.0000 | +7.43 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C3_roberta_s1 | 25109 | 996 | +20.75 | 0.0000 | +7.47 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C4_longformer | 25109 | 996 | +22.67 | 0.0000 | +7.87 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C2_finbert_s4 | 25109 | 996 | +23.71 | 0.0000 | +8.08 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C1_bert_s1 | 25109 | 996 | +24.68 | 0.0000 | +8.13 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C5_e5mistral | 25109 | 996 | +22.92 | 0.0000 | +8.38 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C2_finbert_s2 | 25109 | 996 | +27.20 | 0.0000 | +8.54 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C1_bert_s2 | 25109 | 996 | +25.98 | 0.0000 | +8.71 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C5_qwen3 | 25109 | 996 | +21.16 | 0.0000 | +9.20 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | B2_tfidf_ridge | 25109 | 996 | +29.55 | 0.0000 | +9.54 | 0.0000 | 0.0000 | sig worse |
| event_driven | 5 | C5_gteqwen2 | 25109 | 996 | +21.78 | 0.0000 | +9.57 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | D3_gteqwen2 | 25001 | 991 | +7.06 | 0.0000 | +2.11 | 0.0348 | 0.0360 | sig worse |
| event_driven | 10 | D3_e5mistral | 25001 | 991 | +9.43 | 0.0000 | +2.51 | 0.0123 | 0.0360 | sig worse |
| event_driven | 10 | D3_qwen3 | 25001 | 991 | +8.85 | 0.0000 | +2.52 | 0.0120 | 0.0360 | sig worse |
| event_driven | 10 | A3_garch | 25001 | 991 | +4.69 | 0.0002 | +3.37 | 0.0008 | 0.0031 | sig worse |
| event_driven | 10 | A5_arima | 25001 | 991 | +6.79 | 0.0000 | +3.75 | 0.0002 | 0.0009 | sig worse |
| event_driven | 10 | A4_egarch | 25001 | 991 | +4.89 | 0.0001 | +5.08 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | D2_gated_fusion | 25001 | 991 | +21.27 | 0.0000 | +5.73 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C2_finbert_s3 | 25001 | 991 | +18.20 | 0.0000 | +6.00 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C2_finbert_s4 | 25001 | 991 | +20.37 | 0.0000 | +6.29 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | D1_concat_mlp | 25001 | 991 | +23.65 | 0.0000 | +6.43 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C4_longformer | 25001 | 991 | +21.34 | 0.0000 | +6.51 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C1_bert_s1 | 25001 | 991 | +22.41 | 0.0000 | +6.83 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C2_finbert_s1 | 25001 | 991 | +24.14 | 0.0000 | +7.07 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C3_roberta_s1 | 25001 | 991 | +23.98 | 0.0000 | +7.11 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C2_finbert_s2 | 25001 | 991 | +24.36 | 0.0000 | +7.29 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C1_bert_s2 | 25001 | 991 | +24.71 | 0.0000 | +7.31 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C5_e5mistral | 25001 | 991 | +22.25 | 0.0000 | +7.44 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | B2_tfidf_ridge | 25001 | 991 | +28.04 | 0.0000 | +8.18 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C5_qwen3 | 25001 | 991 | +21.77 | 0.0000 | +8.66 | 0.0000 | 0.0000 | sig worse |
| event_driven | 10 | C5_gteqwen2 | 25001 | 991 | +22.50 | 0.0000 | +9.02 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | D3_gteqwen2 | 24732 | 981 | +2.65 | 0.3518 | +1.16 | 0.2447 | 0.2447 | worse(ns) |
| event_driven | 20 | D3_qwen3 | 24732 | 981 | +6.11 | 0.0000 | +1.80 | 0.0719 | 0.2035 | worse(ns) |
| event_driven | 20 | D3_e5mistral | 24732 | 981 | +5.93 | 0.0000 | +1.83 | 0.0678 | 0.2035 | worse(ns) |
| event_driven | 20 | A5_arima | 24732 | 981 | +5.35 | 0.0000 | +2.20 | 0.0280 | 0.1119 | worse(ns) |
| event_driven | 20 | A3_garch | 24732 | 981 | +5.60 | 0.0000 | +2.58 | 0.0100 | 0.0502 | worse(ns) |
| event_driven | 20 | A4_egarch | 24732 | 981 | +4.88 | 0.0001 | +3.96 | 0.0001 | 0.0005 | sig worse |
| event_driven | 20 | D2_gated_fusion | 24732 | 981 | +16.10 | 0.0000 | +4.30 | 0.0000 | 0.0001 | sig worse |
| event_driven | 20 | C2_finbert_s3 | 24732 | 981 | +19.97 | 0.0000 | +5.04 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | D1_concat_mlp | 24732 | 981 | +18.66 | 0.0000 | +5.07 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | C2_finbert_s2 | 24732 | 981 | +19.33 | 0.0000 | +5.14 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | C1_bert_s1 | 24732 | 981 | +19.61 | 0.0000 | +5.28 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | C2_finbert_s1 | 24732 | 981 | +19.13 | 0.0000 | +5.34 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | C4_longformer | 24732 | 981 | +21.37 | 0.0000 | +5.73 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | C1_bert_s2 | 24732 | 981 | +22.08 | 0.0000 | +5.82 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | C3_roberta_s1 | 24732 | 981 | +22.19 | 0.0000 | +5.84 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | C2_finbert_s4 | 24732 | 981 | +22.42 | 0.0000 | +6.06 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | C5_e5mistral | 24732 | 981 | +22.31 | 0.0000 | +6.62 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | B2_tfidf_ridge | 24732 | 981 | +26.87 | 0.0000 | +7.01 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | C5_qwen3 | 24732 | 981 | +22.63 | 0.0000 | +7.70 | 0.0000 | 0.0000 | sig worse |
| event_driven | 20 | C5_gteqwen2 | 24732 | 981 | +23.89 | 0.0000 | +8.77 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | D1_concat_mlp | 7951 | 809 | +5.94 | 0.0000 | +1.61 | 0.1077 | 0.1141 | worse(ns) |
| long_form | 5 | D3_gteqwen2 | 7951 | 809 | +6.12 | 0.0000 | +1.91 | 0.0571 | 0.1141 | worse(ns) |
| long_form | 5 | A4_egarch | 7951 | 809 | +3.46 | 0.0597 | +2.74 | 0.0063 | 0.0188 | sig worse |
| long_form | 5 | D3_e5mistral | 7951 | 809 | +9.94 | 0.0000 | +3.19 | 0.0015 | 0.0060 | sig worse |
| long_form | 5 | D2_gated_fusion | 7951 | 809 | +9.80 | 0.0000 | +3.53 | 0.0004 | 0.0022 | sig worse |
| long_form | 5 | D3_qwen3 | 7951 | 809 | +10.52 | 0.0000 | +3.81 | 0.0002 | 0.0009 | sig worse |
| long_form | 5 | A3_garch | 7951 | 809 | +7.10 | 0.0000 | +5.22 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C5_e5mistral | 7951 | 809 | +11.04 | 0.0000 | +5.24 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | A5_arima | 7951 | 809 | +6.35 | 0.0000 | +5.84 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C5_qwen3 | 7951 | 809 | +11.05 | 0.0000 | +5.92 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C4_longformer | 7951 | 809 | +12.57 | 0.0000 | +6.06 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | B2_tfidf_ridge | 7951 | 809 | +13.64 | 0.0000 | +6.35 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C2_finbert_s3 | 7951 | 809 | +13.25 | 0.0000 | +6.68 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C2_finbert_s1 | 7951 | 809 | +14.45 | 0.0000 | +6.68 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C5_gteqwen2 | 7951 | 809 | +11.71 | 0.0000 | +6.71 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C2_finbert_s4 | 7951 | 809 | +14.81 | 0.0000 | +7.07 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C1_bert_s2 | 7951 | 809 | +15.44 | 0.0000 | +7.21 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C2_finbert_s2 | 7951 | 809 | +16.13 | 0.0000 | +7.63 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C1_bert_s1 | 7951 | 809 | +16.55 | 0.0000 | +7.64 | 0.0000 | 0.0000 | sig worse |
| long_form | 5 | C3_roberta_s1 | 7951 | 809 | +16.26 | 0.0000 | +7.68 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | D3_gteqwen2 | 7933 | 803 | +5.53 | 0.0000 | +0.77 | 0.4388 | 0.4388 | worse(ns) |
| long_form | 10 | D2_gated_fusion | 7933 | 803 | +8.80 | 0.0000 | +1.86 | 0.0626 | 0.1521 | worse(ns) |
| long_form | 10 | D3_e5mistral | 7933 | 803 | +8.18 | 0.0000 | +1.96 | 0.0507 | 0.1521 | worse(ns) |
| long_form | 10 | A4_egarch | 7933 | 803 | +2.76 | 0.6220 | +2.39 | 0.0173 | 0.0692 | worse(ns) |
| long_form | 10 | D3_qwen3 | 7933 | 803 | +8.81 | 0.0000 | +2.62 | 0.0091 | 0.0535 | worse(ns) |
| long_form | 10 | D1_concat_mlp | 7933 | 803 | +10.31 | 0.0000 | +2.62 | 0.0089 | 0.0535 | worse(ns) |
| long_form | 10 | A5_arima | 7933 | 803 | +4.64 | 0.0005 | +4.21 | 0.0000 | 0.0002 | sig worse |
| long_form | 10 | C1_bert_s2 | 7933 | 803 | +13.09 | 0.0000 | +4.42 | 0.0000 | 0.0001 | sig worse |
| long_form | 10 | C4_longformer | 7933 | 803 | +13.14 | 0.0000 | +4.56 | 0.0000 | 0.0001 | sig worse |
| long_form | 10 | C1_bert_s1 | 7933 | 803 | +13.54 | 0.0000 | +4.66 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | A3_garch | 7933 | 803 | +6.47 | 0.0000 | +4.73 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | C2_finbert_s4 | 7933 | 803 | +13.66 | 0.0000 | +4.89 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | B2_tfidf_ridge | 7933 | 803 | +13.45 | 0.0000 | +4.92 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | C2_finbert_s2 | 7933 | 803 | +14.47 | 0.0000 | +5.11 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | C2_finbert_s3 | 7933 | 803 | +14.24 | 0.0000 | +5.50 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | C2_finbert_s1 | 7933 | 803 | +15.46 | 0.0000 | +5.53 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | C3_roberta_s1 | 7933 | 803 | +15.09 | 0.0000 | +5.61 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | C5_e5mistral | 7933 | 803 | +13.46 | 0.0000 | +6.96 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | C5_qwen3 | 7933 | 803 | +13.42 | 0.0000 | +7.08 | 0.0000 | 0.0000 | sig worse |
| long_form | 10 | C5_gteqwen2 | 7933 | 803 | +13.73 | 0.0000 | +7.31 | 0.0000 | 0.0000 | sig worse |
| long_form | 20 | D3_gteqwen2 | 7902 | 794 | +1.80 | 1.0000 | +0.24 | 0.8116 | 1.0000 | worse(ns) |
| long_form | 20 | D3_e5mistral | 7902 | 794 | +4.27 | 0.0034 | +0.63 | 0.5295 | 1.0000 | worse(ns) |
| long_form | 20 | D3_qwen3 | 7902 | 794 | +5.63 | 0.0000 | +1.26 | 0.2067 | 0.6201 | worse(ns) |
| long_form | 20 | D2_gated_fusion | 7902 | 794 | +7.76 | 0.0000 | +2.03 | 0.0431 | 0.1725 | worse(ns) |
| long_form | 20 | D1_concat_mlp | 7902 | 794 | +9.61 | 0.0000 | +2.44 | 0.0151 | 0.0755 | worse(ns) |
| long_form | 20 | C2_finbert_s4 | 7902 | 794 | +9.33 | 0.0000 | +2.86 | 0.0044 | 0.0261 | sig worse |
| long_form | 20 | C4_longformer | 7902 | 794 | +10.43 | 0.0000 | +2.92 | 0.0036 | 0.0250 | sig worse |
| long_form | 20 | A5_arima | 7902 | 794 | +4.10 | 0.0071 | +3.10 | 0.0020 | 0.0171 | sig worse |
| long_form | 20 | C2_finbert_s2 | 7902 | 794 | +10.07 | 0.0000 | +3.12 | 0.0019 | 0.0171 | sig worse |
| long_form | 20 | C2_finbert_s1 | 7902 | 794 | +10.38 | 0.0000 | +3.32 | 0.0009 | 0.0095 | sig worse |
| long_form | 20 | C2_finbert_s3 | 7902 | 794 | +10.73 | 0.0000 | +3.41 | 0.0007 | 0.0076 | sig worse |
| long_form | 20 | C1_bert_s2 | 7902 | 794 | +11.59 | 0.0000 | +3.46 | 0.0006 | 0.0068 | sig worse |
| long_form | 20 | A4_egarch | 7902 | 794 | +3.01 | 0.3619 | +3.61 | 0.0003 | 0.0044 | sig worse |
| long_form | 20 | B2_tfidf_ridge | 7902 | 794 | +11.86 | 0.0000 | +3.62 | 0.0003 | 0.0044 | sig worse |
| long_form | 20 | C1_bert_s1 | 7902 | 794 | +11.79 | 0.0000 | +3.66 | 0.0003 | 0.0041 | sig worse |
| long_form | 20 | C3_roberta_s1 | 7902 | 794 | +12.57 | 0.0000 | +3.87 | 0.0001 | 0.0019 | sig worse |
| long_form | 20 | A3_garch | 7902 | 794 | +6.95 | 0.0000 | +4.27 | 0.0000 | 0.0004 | sig worse |
| long_form | 20 | C5_gteqwen2 | 7902 | 794 | +13.50 | 0.0000 | +6.67 | 0.0000 | 0.0000 | sig worse |
| long_form | 20 | C5_e5mistral | 7902 | 794 | +13.85 | 0.0000 | +6.76 | 0.0000 | 0.0000 | sig worse |
| long_form | 20 | C5_qwen3 | 7902 | 794 | +13.20 | 0.0000 | +6.79 | 0.0000 | 0.0000 | sig worse |