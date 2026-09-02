# Pairwise Diebold-Mariano significance matrix (per-obs squared error)

DM on per-observation squared error `fc.se` on split==test, joined on KEY=[ticker,accession,horizon_days] (inner join across the full model set within each disclosure, so all pairs share one sample). 3-seed models (C*/D*) are **seed-ensembled**: per-observation prediction averaged across seeds {2026,2027,2028} BEFORE squared error. A/B are seed-invariant (seed2026 only).

**Sign convention:** dm_stat>0 => challenger has HIGHER loss (WORSE) than baseline; dm_stat<0 => challenger BETTER. Holm applied WITHIN each (disclosure,horizon) group over all ordered pairs. `sig better` = dm_stat<0 AND p_holm<0.05.


## long_form


### horizon = 5 days (n=7951, best-QLIKE model = **A2_har_rv**)

**Challenger vs A2_har_rv** (does the challenger beat the HAR price baseline?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A4_egarch | +3.465 | 0.0005 | 0.0597 | worse |
| D1_concat_mlp | +5.939 | 0.0000 | 0.0000 | worse |
| D3_gteqwen2 | +6.115 | 0.0000 | 0.0000 | worse |
| A5_arima | +6.355 | 0.0000 | 0.0000 | worse |
| A3_garch | +7.098 | 0.0000 | 0.0000 | worse |
| D2_gated_fusion | +9.804 | 0.0000 | 0.0000 | worse |
| D3_e5mistral | +9.937 | 0.0000 | 0.0000 | worse |
| D3_qwen3 | +10.524 | 0.0000 | 0.0000 | worse |
| C5_e5mistral | +11.040 | 0.0000 | 0.0000 | worse |
| C5_qwen3 | +11.048 | 0.0000 | 0.0000 | worse |
| C5_gteqwen2 | +11.710 | 0.0000 | 0.0000 | worse |
| C4_longformer | +12.570 | 0.0000 | 0.0000 | worse |
| C2_finbert_s3 | +13.252 | 0.0000 | 0.0000 | worse |
| B2_tfidf_ridge | +13.644 | 0.0000 | 0.0000 | worse |
| C2_finbert_s1 | +14.452 | 0.0000 | 0.0000 | worse |
| C2_finbert_s4 | +14.811 | 0.0000 | 0.0000 | worse |
| C1_bert_s2 | +15.443 | 0.0000 | 0.0000 | worse |
| C2_finbert_s2 | +16.132 | 0.0000 | 0.0000 | worse |
| C3_roberta_s1 | +16.265 | 0.0000 | 0.0000 | worse |
| C1_bert_s1 | +16.545 | 0.0000 | 0.0000 | worse |

**Challenger vs best model (A2_har_rv)** (who is statistically indistinct from / beats the best?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A4_egarch | +3.465 | 0.0005 | 0.0597 | indistinct from best |
| D1_concat_mlp | +5.939 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_gteqwen2 | +6.115 | 0.0000 | 0.0000 | sig WORSE than best |
| A5_arima | +6.355 | 0.0000 | 0.0000 | sig WORSE than best |
| A3_garch | +7.098 | 0.0000 | 0.0000 | sig WORSE than best |
| D2_gated_fusion | +9.804 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_e5mistral | +9.937 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_qwen3 | +10.524 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_e5mistral | +11.040 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_qwen3 | +11.048 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_gteqwen2 | +11.710 | 0.0000 | 0.0000 | sig WORSE than best |
| C4_longformer | +12.570 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s3 | +13.252 | 0.0000 | 0.0000 | sig WORSE than best |
| B2_tfidf_ridge | +13.644 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s1 | +14.452 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s4 | +14.811 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s2 | +15.443 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s2 | +16.132 | 0.0000 | 0.0000 | sig WORSE than best |
| C3_roberta_s1 | +16.265 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s1 | +16.545 | 0.0000 | 0.0000 | sig WORSE than best |

### horizon = 10 days (n=7933, best-QLIKE model = **A2_har_rv**)

**Challenger vs A2_har_rv** (does the challenger beat the HAR price baseline?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A4_egarch | +2.762 | 0.0058 | 0.6220 | worse |
| A5_arima | +4.644 | 0.0000 | 0.0005 | worse |
| D3_gteqwen2 | +5.531 | 0.0000 | 0.0000 | worse |
| A3_garch | +6.470 | 0.0000 | 0.0000 | worse |
| D3_e5mistral | +8.180 | 0.0000 | 0.0000 | worse |
| D2_gated_fusion | +8.798 | 0.0000 | 0.0000 | worse |
| D3_qwen3 | +8.806 | 0.0000 | 0.0000 | worse |
| D1_concat_mlp | +10.306 | 0.0000 | 0.0000 | worse |
| C1_bert_s2 | +13.091 | 0.0000 | 0.0000 | worse |
| C4_longformer | +13.138 | 0.0000 | 0.0000 | worse |
| C5_qwen3 | +13.416 | 0.0000 | 0.0000 | worse |
| B2_tfidf_ridge | +13.452 | 0.0000 | 0.0000 | worse |
| C5_e5mistral | +13.462 | 0.0000 | 0.0000 | worse |
| C1_bert_s1 | +13.538 | 0.0000 | 0.0000 | worse |
| C2_finbert_s4 | +13.655 | 0.0000 | 0.0000 | worse |
| C5_gteqwen2 | +13.734 | 0.0000 | 0.0000 | worse |
| C2_finbert_s3 | +14.238 | 0.0000 | 0.0000 | worse |
| C2_finbert_s2 | +14.466 | 0.0000 | 0.0000 | worse |
| C3_roberta_s1 | +15.086 | 0.0000 | 0.0000 | worse |
| C2_finbert_s1 | +15.457 | 0.0000 | 0.0000 | worse |

**Challenger vs best model (A2_har_rv)** (who is statistically indistinct from / beats the best?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A4_egarch | +2.762 | 0.0058 | 0.6220 | indistinct from best |
| A5_arima | +4.644 | 0.0000 | 0.0005 | sig WORSE than best |
| D3_gteqwen2 | +5.531 | 0.0000 | 0.0000 | sig WORSE than best |
| A3_garch | +6.470 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_e5mistral | +8.180 | 0.0000 | 0.0000 | sig WORSE than best |
| D2_gated_fusion | +8.798 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_qwen3 | +8.806 | 0.0000 | 0.0000 | sig WORSE than best |
| D1_concat_mlp | +10.306 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s2 | +13.091 | 0.0000 | 0.0000 | sig WORSE than best |
| C4_longformer | +13.138 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_qwen3 | +13.416 | 0.0000 | 0.0000 | sig WORSE than best |
| B2_tfidf_ridge | +13.452 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_e5mistral | +13.462 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s1 | +13.538 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s4 | +13.655 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_gteqwen2 | +13.734 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s3 | +14.238 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s2 | +14.466 | 0.0000 | 0.0000 | sig WORSE than best |
| C3_roberta_s1 | +15.086 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s1 | +15.457 | 0.0000 | 0.0000 | sig WORSE than best |

### horizon = 20 days (n=7902, best-QLIKE model = **A2_har_rv**)

**Challenger vs A2_har_rv** (does the challenger beat the HAR price baseline?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| D3_gteqwen2 | +1.802 | 0.0715 | 1.0000 | worse |
| A4_egarch | +3.014 | 0.0026 | 0.3619 | worse |
| A5_arima | +4.102 | 0.0000 | 0.0071 | worse |
| D3_e5mistral | +4.274 | 0.0000 | 0.0034 | worse |
| D3_qwen3 | +5.632 | 0.0000 | 0.0000 | worse |
| A3_garch | +6.946 | 0.0000 | 0.0000 | worse |
| D2_gated_fusion | +7.759 | 0.0000 | 0.0000 | worse |
| C2_finbert_s4 | +9.332 | 0.0000 | 0.0000 | worse |
| D1_concat_mlp | +9.606 | 0.0000 | 0.0000 | worse |
| C2_finbert_s2 | +10.069 | 0.0000 | 0.0000 | worse |
| C2_finbert_s1 | +10.381 | 0.0000 | 0.0000 | worse |
| C4_longformer | +10.433 | 0.0000 | 0.0000 | worse |
| C2_finbert_s3 | +10.732 | 0.0000 | 0.0000 | worse |
| C1_bert_s2 | +11.588 | 0.0000 | 0.0000 | worse |
| C1_bert_s1 | +11.787 | 0.0000 | 0.0000 | worse |
| B2_tfidf_ridge | +11.860 | 0.0000 | 0.0000 | worse |
| C3_roberta_s1 | +12.573 | 0.0000 | 0.0000 | worse |
| C5_qwen3 | +13.195 | 0.0000 | 0.0000 | worse |
| C5_gteqwen2 | +13.496 | 0.0000 | 0.0000 | worse |
| C5_e5mistral | +13.850 | 0.0000 | 0.0000 | worse |

**Challenger vs best model (A2_har_rv)** (who is statistically indistinct from / beats the best?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| D3_gteqwen2 | +1.802 | 0.0715 | 1.0000 | indistinct from best |
| A4_egarch | +3.014 | 0.0026 | 0.3619 | indistinct from best |
| A5_arima | +4.102 | 0.0000 | 0.0071 | sig WORSE than best |
| D3_e5mistral | +4.274 | 0.0000 | 0.0034 | sig WORSE than best |
| D3_qwen3 | +5.632 | 0.0000 | 0.0000 | sig WORSE than best |
| A3_garch | +6.946 | 0.0000 | 0.0000 | sig WORSE than best |
| D2_gated_fusion | +7.759 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s4 | +9.332 | 0.0000 | 0.0000 | sig WORSE than best |
| D1_concat_mlp | +9.606 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s2 | +10.069 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s1 | +10.381 | 0.0000 | 0.0000 | sig WORSE than best |
| C4_longformer | +10.433 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s3 | +10.732 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s2 | +11.588 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s1 | +11.787 | 0.0000 | 0.0000 | sig WORSE than best |
| B2_tfidf_ridge | +11.860 | 0.0000 | 0.0000 | sig WORSE than best |
| C3_roberta_s1 | +12.573 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_qwen3 | +13.195 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_gteqwen2 | +13.496 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_e5mistral | +13.850 | 0.0000 | 0.0000 | sig WORSE than best |

## event_driven


### horizon = 5 days (n=25109, best-QLIKE model = **A2_har_rv**)

**Challenger vs A2_har_rv** (does the challenger beat the HAR price baseline?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A3_garch | +4.838 | 0.0000 | 0.0001 | worse |
| A4_egarch | +5.385 | 0.0000 | 0.0000 | worse |
| D3_gteqwen2 | +5.810 | 0.0000 | 0.0000 | worse |
| A5_arima | +8.016 | 0.0000 | 0.0000 | worse |
| D3_qwen3 | +13.551 | 0.0000 | 0.0000 | worse |
| D3_e5mistral | +15.112 | 0.0000 | 0.0000 | worse |
| C2_finbert_s1 | +18.803 | 0.0000 | 0.0000 | worse |
| C2_finbert_s3 | +20.322 | 0.0000 | 0.0000 | worse |
| C3_roberta_s1 | +20.752 | 0.0000 | 0.0000 | worse |
| C5_qwen3 | +21.156 | 0.0000 | 0.0000 | worse |
| D2_gated_fusion | +21.382 | 0.0000 | 0.0000 | worse |
| C5_gteqwen2 | +21.779 | 0.0000 | 0.0000 | worse |
| C4_longformer | +22.673 | 0.0000 | 0.0000 | worse |
| C5_e5mistral | +22.924 | 0.0000 | 0.0000 | worse |
| C2_finbert_s4 | +23.714 | 0.0000 | 0.0000 | worse |
| D1_concat_mlp | +24.184 | 0.0000 | 0.0000 | worse |
| C1_bert_s1 | +24.676 | 0.0000 | 0.0000 | worse |
| C1_bert_s2 | +25.982 | 0.0000 | 0.0000 | worse |
| C2_finbert_s2 | +27.199 | 0.0000 | 0.0000 | worse |
| B2_tfidf_ridge | +29.546 | 0.0000 | 0.0000 | worse |

**Challenger vs best model (A2_har_rv)** (who is statistically indistinct from / beats the best?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A3_garch | +4.838 | 0.0000 | 0.0001 | sig WORSE than best |
| A4_egarch | +5.385 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_gteqwen2 | +5.810 | 0.0000 | 0.0000 | sig WORSE than best |
| A5_arima | +8.016 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_qwen3 | +13.551 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_e5mistral | +15.112 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s1 | +18.803 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s3 | +20.322 | 0.0000 | 0.0000 | sig WORSE than best |
| C3_roberta_s1 | +20.752 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_qwen3 | +21.156 | 0.0000 | 0.0000 | sig WORSE than best |
| D2_gated_fusion | +21.382 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_gteqwen2 | +21.779 | 0.0000 | 0.0000 | sig WORSE than best |
| C4_longformer | +22.673 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_e5mistral | +22.924 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s4 | +23.714 | 0.0000 | 0.0000 | sig WORSE than best |
| D1_concat_mlp | +24.184 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s1 | +24.676 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s2 | +25.982 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s2 | +27.199 | 0.0000 | 0.0000 | sig WORSE than best |
| B2_tfidf_ridge | +29.546 | 0.0000 | 0.0000 | sig WORSE than best |

### horizon = 10 days (n=25001, best-QLIKE model = **A3_garch**)

**Challenger vs A2_har_rv** (does the challenger beat the HAR price baseline?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A3_garch | +4.695 | 0.0000 | 0.0002 | worse |
| A4_egarch | +4.888 | 0.0000 | 0.0001 | worse |
| A5_arima | +6.789 | 0.0000 | 0.0000 | worse |
| D3_gteqwen2 | +7.061 | 0.0000 | 0.0000 | worse |
| D3_qwen3 | +8.849 | 0.0000 | 0.0000 | worse |
| D3_e5mistral | +9.433 | 0.0000 | 0.0000 | worse |
| C2_finbert_s3 | +18.199 | 0.0000 | 0.0000 | worse |
| C2_finbert_s4 | +20.367 | 0.0000 | 0.0000 | worse |
| D2_gated_fusion | +21.273 | 0.0000 | 0.0000 | worse |
| C4_longformer | +21.337 | 0.0000 | 0.0000 | worse |
| C5_qwen3 | +21.766 | 0.0000 | 0.0000 | worse |
| C5_e5mistral | +22.247 | 0.0000 | 0.0000 | worse |
| C1_bert_s1 | +22.413 | 0.0000 | 0.0000 | worse |
| C5_gteqwen2 | +22.497 | 0.0000 | 0.0000 | worse |
| D1_concat_mlp | +23.646 | 0.0000 | 0.0000 | worse |
| C3_roberta_s1 | +23.975 | 0.0000 | 0.0000 | worse |
| C2_finbert_s1 | +24.139 | 0.0000 | 0.0000 | worse |
| C2_finbert_s2 | +24.363 | 0.0000 | 0.0000 | worse |
| C1_bert_s2 | +24.711 | 0.0000 | 0.0000 | worse |
| B2_tfidf_ridge | +28.041 | 0.0000 | 0.0000 | worse |

**Challenger vs best model (A3_garch)** (who is statistically indistinct from / beats the best?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A2_har_rv | -4.695 | 0.0000 | 0.0002 | sig BETTER than best* |
| D3_gteqwen2 | -3.071 | 0.0021 | 0.1067 | indistinct from best |
| D3_qwen3 | -2.667 | 0.0077 | 0.3368 | indistinct from best |
| D3_e5mistral | -2.659 | 0.0078 | 0.3368 | indistinct from best |
| D2_gated_fusion | +2.724 | 0.0064 | 0.2965 | indistinct from best |
| A5_arima | +3.425 | 0.0006 | 0.0358 | sig WORSE than best |
| A4_egarch | +3.837 | 0.0001 | 0.0077 | sig WORSE than best |
| D1_concat_mlp | +5.621 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s3 | +7.017 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_qwen3 | +8.023 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_gteqwen2 | +8.363 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s4 | +9.050 | 0.0000 | 0.0000 | sig WORSE than best |
| C4_longformer | +9.551 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_e5mistral | +9.612 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s1 | +11.029 | 0.0000 | 0.0000 | sig WORSE than best |
| C3_roberta_s1 | +11.578 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s1 | +12.088 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s2 | +12.196 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s2 | +12.633 | 0.0000 | 0.0000 | sig WORSE than best |
| B2_tfidf_ridge | +15.490 | 0.0000 | 0.0000 | sig WORSE than best |

### horizon = 20 days (n=24732, best-QLIKE model = **A3_garch**)

**Challenger vs A2_har_rv** (does the challenger beat the HAR price baseline?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| D3_gteqwen2 | +2.652 | 0.0080 | 0.3518 | worse |
| A4_egarch | +4.877 | 0.0000 | 0.0001 | worse |
| A5_arima | +5.354 | 0.0000 | 0.0000 | worse |
| A3_garch | +5.602 | 0.0000 | 0.0000 | worse |
| D3_e5mistral | +5.931 | 0.0000 | 0.0000 | worse |
| D3_qwen3 | +6.108 | 0.0000 | 0.0000 | worse |
| D2_gated_fusion | +16.097 | 0.0000 | 0.0000 | worse |
| D1_concat_mlp | +18.657 | 0.0000 | 0.0000 | worse |
| C2_finbert_s1 | +19.127 | 0.0000 | 0.0000 | worse |
| C2_finbert_s2 | +19.331 | 0.0000 | 0.0000 | worse |
| C1_bert_s1 | +19.605 | 0.0000 | 0.0000 | worse |
| C2_finbert_s3 | +19.971 | 0.0000 | 0.0000 | worse |
| C4_longformer | +21.366 | 0.0000 | 0.0000 | worse |
| C1_bert_s2 | +22.080 | 0.0000 | 0.0000 | worse |
| C3_roberta_s1 | +22.191 | 0.0000 | 0.0000 | worse |
| C5_e5mistral | +22.308 | 0.0000 | 0.0000 | worse |
| C2_finbert_s4 | +22.422 | 0.0000 | 0.0000 | worse |
| C5_qwen3 | +22.632 | 0.0000 | 0.0000 | worse |
| C5_gteqwen2 | +23.892 | 0.0000 | 0.0000 | worse |
| B2_tfidf_ridge | +26.868 | 0.0000 | 0.0000 | worse |

**Challenger vs best model (A3_garch)** (who is statistically indistinct from / beats the best?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A2_har_rv | -5.602 | 0.0000 | 0.0000 | sig BETTER than best* |
| D3_gteqwen2 | -4.461 | 0.0000 | 0.0006 | sig BETTER than best* |
| D3_e5mistral | -3.958 | 0.0001 | 0.0051 | sig BETTER than best* |
| D3_qwen3 | -3.861 | 0.0001 | 0.0075 | sig BETTER than best* |
| A5_arima | +0.174 | 0.8615 | 1.0000 | indistinct from best |
| D2_gated_fusion | +2.621 | 0.0088 | 0.3680 | indistinct from best |
| A4_egarch | +3.503 | 0.0005 | 0.0249 | sig WORSE than best |
| D1_concat_mlp | +4.189 | 0.0000 | 0.0021 | sig WORSE than best |
| C2_finbert_s1 | +8.741 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_qwen3 | +8.784 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s2 | +8.870 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_gteqwen2 | +9.098 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s1 | +9.659 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_e5mistral | +10.215 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s3 | +10.361 | 0.0000 | 0.0000 | sig WORSE than best |
| C4_longformer | +11.120 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s4 | +11.614 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s2 | +11.785 | 0.0000 | 0.0000 | sig WORSE than best |
| C3_roberta_s1 | +12.170 | 0.0000 | 0.0000 | sig WORSE than best |
| B2_tfidf_ridge | +16.072 | 0.0000 | 0.0000 | sig WORSE than best |

## combined


### horizon = 5 days (n=33060, best-QLIKE model = **A2_har_rv**)

**Challenger vs A2_har_rv** (does the challenger beat the HAR price baseline?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A4_egarch | +5.582 | 0.0000 | 0.0000 | worse |
| A3_garch | +8.218 | 0.0000 | 0.0000 | worse |
| D3_gteqwen2 | +9.498 | 0.0000 | 0.0000 | worse |
| A5_arima | +9.845 | 0.0000 | 0.0000 | worse |
| D3_qwen3 | +15.432 | 0.0000 | 0.0000 | worse |
| D3_e5mistral | +18.301 | 0.0000 | 0.0000 | worse |
| D1_concat_mlp | +26.852 | 0.0000 | 0.0000 | worse |
| C2_finbert_s1 | +27.615 | 0.0000 | 0.0000 | worse |
| D2_gated_fusion | +27.822 | 0.0000 | 0.0000 | worse |
| C5_qwen3 | +27.980 | 0.0000 | 0.0000 | worse |
| C5_gteqwen2 | +28.394 | 0.0000 | 0.0000 | worse |
| C2_finbert_s2 | +29.058 | 0.0000 | 0.0000 | worse |
| C3_roberta_s1 | +29.668 | 0.0000 | 0.0000 | worse |
| C2_finbert_s3 | +30.003 | 0.0000 | 0.0000 | worse |
| C5_e5mistral | +30.400 | 0.0000 | 0.0000 | worse |
| C4_longformer | +31.089 | 0.0000 | 0.0000 | worse |
| C1_bert_s1 | +31.596 | 0.0000 | 0.0000 | worse |
| C1_bert_s2 | +33.162 | 0.0000 | 0.0000 | worse |
| B2_tfidf_ridge | +33.238 | 0.0000 | 0.0000 | worse |
| C2_finbert_s4 | +33.843 | 0.0000 | 0.0000 | worse |

**Challenger vs best model (A2_har_rv)** (who is statistically indistinct from / beats the best?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| A4_egarch | +5.582 | 0.0000 | 0.0000 | sig WORSE than best |
| A3_garch | +8.218 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_gteqwen2 | +9.498 | 0.0000 | 0.0000 | sig WORSE than best |
| A5_arima | +9.845 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_qwen3 | +15.432 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_e5mistral | +18.301 | 0.0000 | 0.0000 | sig WORSE than best |
| D1_concat_mlp | +26.852 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s1 | +27.615 | 0.0000 | 0.0000 | sig WORSE than best |
| D2_gated_fusion | +27.822 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_qwen3 | +27.980 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_gteqwen2 | +28.394 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s2 | +29.058 | 0.0000 | 0.0000 | sig WORSE than best |
| C3_roberta_s1 | +29.668 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s3 | +30.003 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_e5mistral | +30.400 | 0.0000 | 0.0000 | sig WORSE than best |
| C4_longformer | +31.089 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s1 | +31.596 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s2 | +33.162 | 0.0000 | 0.0000 | sig WORSE than best |
| B2_tfidf_ridge | +33.238 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s4 | +33.843 | 0.0000 | 0.0000 | sig WORSE than best |

### horizon = 10 days (n=32934, best-QLIKE model = **A2_har_rv**)

**Challenger vs A2_har_rv** (does the challenger beat the HAR price baseline?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| D3_gteqwen2 | +3.826 | 0.0001 | 0.0083 | worse |
| A4_egarch | +4.803 | 0.0000 | 0.0001 | worse |
| A5_arima | +7.703 | 0.0000 | 0.0000 | worse |
| A3_garch | +7.759 | 0.0000 | 0.0000 | worse |
| D3_qwen3 | +7.835 | 0.0000 | 0.0000 | worse |
| D3_e5mistral | +9.497 | 0.0000 | 0.0000 | worse |
| D1_concat_mlp | +20.477 | 0.0000 | 0.0000 | worse |
| C1_bert_s1 | +20.752 | 0.0000 | 0.0000 | worse |
| C2_finbert_s1 | +21.684 | 0.0000 | 0.0000 | worse |
| D2_gated_fusion | +22.686 | 0.0000 | 0.0000 | worse |
| C2_finbert_s4 | +25.611 | 0.0000 | 0.0000 | worse |
| C4_longformer | +25.783 | 0.0000 | 0.0000 | worse |
| C2_finbert_s3 | +26.090 | 0.0000 | 0.0000 | worse |
| C5_qwen3 | +26.838 | 0.0000 | 0.0000 | worse |
| C1_bert_s2 | +27.510 | 0.0000 | 0.0000 | worse |
| C2_finbert_s2 | +27.512 | 0.0000 | 0.0000 | worse |
| C5_gteqwen2 | +27.715 | 0.0000 | 0.0000 | worse |
| C5_e5mistral | +28.559 | 0.0000 | 0.0000 | worse |
| C3_roberta_s1 | +29.973 | 0.0000 | 0.0000 | worse |
| B2_tfidf_ridge | +31.509 | 0.0000 | 0.0000 | worse |

**Challenger vs best model (A2_har_rv)** (who is statistically indistinct from / beats the best?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| D3_gteqwen2 | +3.826 | 0.0001 | 0.0083 | sig WORSE than best |
| A4_egarch | +4.803 | 0.0000 | 0.0001 | sig WORSE than best |
| A5_arima | +7.703 | 0.0000 | 0.0000 | sig WORSE than best |
| A3_garch | +7.759 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_qwen3 | +7.835 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_e5mistral | +9.497 | 0.0000 | 0.0000 | sig WORSE than best |
| D1_concat_mlp | +20.477 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s1 | +20.752 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s1 | +21.684 | 0.0000 | 0.0000 | sig WORSE than best |
| D2_gated_fusion | +22.686 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s4 | +25.611 | 0.0000 | 0.0000 | sig WORSE than best |
| C4_longformer | +25.783 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s3 | +26.090 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_qwen3 | +26.838 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s2 | +27.510 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s2 | +27.512 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_gteqwen2 | +27.715 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_e5mistral | +28.559 | 0.0000 | 0.0000 | sig WORSE than best |
| C3_roberta_s1 | +29.973 | 0.0000 | 0.0000 | sig WORSE than best |
| B2_tfidf_ridge | +31.509 | 0.0000 | 0.0000 | sig WORSE than best |

### horizon = 20 days (n=32634, best-QLIKE model = **A2_har_rv**)

**Challenger vs A2_har_rv** (does the challenger beat the HAR price baseline?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| D3_gteqwen2 | +0.516 | 0.6060 | 1.0000 | worse |
| A4_egarch | +4.819 | 0.0000 | 0.0001 | worse |
| A5_arima | +5.979 | 0.0000 | 0.0000 | worse |
| D3_qwen3 | +7.605 | 0.0000 | 0.0000 | worse |
| A3_garch | +8.300 | 0.0000 | 0.0000 | worse |
| D3_e5mistral | +8.916 | 0.0000 | 0.0000 | worse |
| D1_concat_mlp | +17.933 | 0.0000 | 0.0000 | worse |
| C2_finbert_s4 | +22.192 | 0.0000 | 0.0000 | worse |
| C4_longformer | +22.850 | 0.0000 | 0.0000 | worse |
| C2_finbert_s1 | +22.956 | 0.0000 | 0.0000 | worse |
| C2_finbert_s3 | +23.464 | 0.0000 | 0.0000 | worse |
| D2_gated_fusion | +24.570 | 0.0000 | 0.0000 | worse |
| C2_finbert_s2 | +24.816 | 0.0000 | 0.0000 | worse |
| C1_bert_s2 | +24.824 | 0.0000 | 0.0000 | worse |
| C3_roberta_s1 | +25.294 | 0.0000 | 0.0000 | worse |
| C5_qwen3 | +25.968 | 0.0000 | 0.0000 | worse |
| C1_bert_s1 | +26.319 | 0.0000 | 0.0000 | worse |
| C5_gteqwen2 | +26.373 | 0.0000 | 0.0000 | worse |
| C5_e5mistral | +27.077 | 0.0000 | 0.0000 | worse |
| B2_tfidf_ridge | +29.189 | 0.0000 | 0.0000 | worse |

**Challenger vs best model (A2_har_rv)** (who is statistically indistinct from / beats the best?)

| challenger | dm_stat | p_raw | p_holm | verdict |
|---|---|---|---|---|
| D3_gteqwen2 | +0.516 | 0.6060 | 1.0000 | indistinct from best |
| A4_egarch | +4.819 | 0.0000 | 0.0001 | sig WORSE than best |
| A5_arima | +5.979 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_qwen3 | +7.605 | 0.0000 | 0.0000 | sig WORSE than best |
| A3_garch | +8.300 | 0.0000 | 0.0000 | sig WORSE than best |
| D3_e5mistral | +8.916 | 0.0000 | 0.0000 | sig WORSE than best |
| D1_concat_mlp | +17.933 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s4 | +22.192 | 0.0000 | 0.0000 | sig WORSE than best |
| C4_longformer | +22.850 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s1 | +22.956 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s3 | +23.464 | 0.0000 | 0.0000 | sig WORSE than best |
| D2_gated_fusion | +24.570 | 0.0000 | 0.0000 | sig WORSE than best |
| C2_finbert_s2 | +24.816 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s2 | +24.824 | 0.0000 | 0.0000 | sig WORSE than best |
| C3_roberta_s1 | +25.294 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_qwen3 | +25.968 | 0.0000 | 0.0000 | sig WORSE than best |
| C1_bert_s1 | +26.319 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_gteqwen2 | +26.373 | 0.0000 | 0.0000 | sig WORSE than best |
| C5_e5mistral | +27.077 | 0.0000 | 0.0000 | sig WORSE than best |
| B2_tfidf_ridge | +29.189 | 0.0000 | 0.0000 | sig WORSE than best |

## Sanity — text/neural models vs A2_har_rv on SE

- 144/144 (challenger, A2) cells have dm_stat>0 (challenger WORSE than HAR on SE), matching seed_aggregate.md (all text models WORSE than HAR on SE).
