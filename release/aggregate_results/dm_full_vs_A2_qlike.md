# DM robustness: QLIKE-loss vs A2_har_rv (test)

Positive DM = worse than HAR-RV. n.s. = p>=0.05.


| disclosure | model | h | DM | p | Holm | BH |
|---|---|---|---|---|---|---|
| long_form | B1_bow_ridge | 5 | +19.12 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B1_bow_ridge | 10 | +17.39 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B1_bow_ridge | 20 | +16.15 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B2_tfidf_ridge | 5 | +17.43 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B2_tfidf_ridge | 10 | +16.36 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B2_tfidf_ridge | 20 | +14.82 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B3_lm_linear | 5 | +18.85 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B3_lm_linear | 10 | +18.41 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B3_lm_linear | 20 | +16.11 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B4_lm_features | 5 | +20.94 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B4_lm_features | 10 | +19.03 | 0.0000 | 0.0000 | 0.0000 |
| long_form | B4_lm_features | 20 | +17.20 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C1_bert_s1 | 5 | +31.54 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C1_bert_s1 | 10 | +28.88 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C1_bert_s1 | 20 | +24.06 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s1 | 5 | +22.72 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s1 | 10 | +16.89 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s1 | 20 | +23.66 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s2 | 5 | +15.39 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s2 | 10 | +23.58 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s2 | 20 | +17.26 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s3 | 5 | +24.86 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s3 | 10 | +15.99 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s3 | 20 | +20.09 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s4 | 5 | +26.50 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s4 | 10 | +20.81 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C2_finbert_s4 | 20 | +17.46 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C3_roberta_s1 | 5 | +20.83 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C3_roberta_s1 | 10 | +21.73 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C3_roberta_s1 | 20 | +16.86 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C4_longformer | 5 | +15.30 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C4_longformer | 10 | +15.87 | 0.0000 | 0.0000 | 0.0000 |
| long_form | C4_longformer | 20 | +17.46 | 0.0000 | 0.0000 | 0.0000 |
| long_form | D1_concat_mlp | 5 | +1.44 | 0.1491 | 0.2981 | 0.1517 |
| long_form | D1_concat_mlp | 10 | +6.98 | 0.0000 | 0.0000 | 0.0000 |
| long_form | D1_concat_mlp | 20 | +18.20 | 0.0000 | 0.0000 | 0.0000 |
| long_form | D2_gated_fusion | 5 | +0.07 | 0.9457 | 0.9457 | 0.9457 |
| long_form | D2_gated_fusion | 10 | +20.25 | 0.0000 | 0.0000 | 0.0000 |
| long_form | D2_gated_fusion | 20 | +18.92 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B1_bow_ridge | 5 | +32.84 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B1_bow_ridge | 10 | +32.21 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B1_bow_ridge | 20 | +29.21 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B2_tfidf_ridge | 5 | +38.02 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B2_tfidf_ridge | 10 | +35.70 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B2_tfidf_ridge | 20 | +32.51 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B3_lm_linear | 5 | +35.78 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B3_lm_linear | 10 | +33.17 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B3_lm_linear | 20 | +30.26 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B4_lm_features | 5 | +35.06 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B4_lm_features | 10 | +32.32 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | B4_lm_features | 20 | +29.59 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | C2_finbert_s1 | 5 | +23.89 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | C2_finbert_s1 | 10 | +28.51 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | C2_finbert_s1 | 20 | +24.70 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | D2_gated_fusion | 5 | +21.30 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | D2_gated_fusion | 10 | +18.81 | 0.0000 | 0.0000 | 0.0000 |
| event_driven | D2_gated_fusion | 20 | +34.03 | 0.0000 | 0.0000 | 0.0000 |