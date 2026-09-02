# DM robustness: QLIKE-loss vs A2_har_rv (test)

Positive DM = worse than HAR-RV. n.s. = p>=0.05.


| disclosure | model | h | DM | p | Holm | BH |
|---|---|---|---|---|---|---|
| long_form | B1_bow_ridge | 5 | +20.17 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B1_bow_ridge | 10 | +18.30 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B1_bow_ridge | 20 | +16.39 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B2_tfidf_ridge | 5 | +19.87 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B2_tfidf_ridge | 10 | +17.28 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B2_tfidf_ridge | 20 | +15.10 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B3_lm_linear | 5 | +20.96 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B3_lm_linear | 10 | +19.17 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B3_lm_linear | 20 | +16.56 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B4_lm_features | 5 | +22.32 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B4_lm_features | 10 | +19.47 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B4_lm_features | 20 | +17.34 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C1_bert_s1 | 5 | +25.93 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C1_bert_s1 | 10 | +16.90 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C1_bert_s1 | 20 | +17.53 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s1 | 5 | +14.93 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s1 | 10 | +19.31 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s1 | 20 | +19.19 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s2 | 5 | +22.47 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s2 | 10 | +16.13 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s2 | 20 | +17.03 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s3 | 5 | +22.06 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s3 | 10 | +18.50 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s3 | 20 | +13.13 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s4 | 5 | +21.98 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s4 | 10 | +21.99 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s4 | 20 | +13.65 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C3_roberta_s1 | 5 | +24.15 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C3_roberta_s1 | 10 | +21.84 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C3_roberta_s1 | 20 | +15.88 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C4_longformer | 5 | +18.57 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C4_longformer | 10 | +20.01 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C4_longformer | 20 | +16.43 | 0.0000 | 0.0000 | 0.0000 |
| long_form | D1_concat_mlp | 5 | +9.42 | 0.0000 | 0.0000 | 0.0000 |
| long_form | D1_concat_mlp | 10 | +17.24 | 0.0000 | 0.0000 | 0.0000 |
| long_form | D1_concat_mlp | 20 | +13.48 | 0.0000 | 0.0000 | 0.0000 |
| long_form | D2_gated_fusion | 5 | +18.00 | 0.0000 | 0.0000 | 0.0000 |
| long_form | D2_gated_fusion | 10 | +15.47 | 0.0000 | 0.0000 | 0.0000 |
| long_form | D2_gated_fusion | 20 | +11.30 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B1_bow_ridge | 5 | +34.92 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B1_bow_ridge | 10 | +32.68 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B1_bow_ridge | 20 | +29.31 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B2_tfidf_ridge | 5 | +39.15 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B2_tfidf_ridge | 10 | +35.48 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B2_tfidf_ridge | 20 | +32.01 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B3_lm_linear | 5 | +37.61 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B3_lm_linear | 10 | +33.51 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B3_lm_linear | 20 | +30.07 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B4_lm_features | 5 | +37.16 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B4_lm_features | 10 | +32.89 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B4_lm_features | 20 | +29.54 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | C2_finbert_s1 | 5 | +26.39 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | C2_finbert_s1 | 10 | +33.34 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | C2_finbert_s1 | 20 | +30.14 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | D2_gated_fusion | 5 | +26.00 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | D2_gated_fusion | 10 | +31.20 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | D2_gated_fusion | 20 | +17.32 | 0.0000 | 0.0000 | 0.0000 |