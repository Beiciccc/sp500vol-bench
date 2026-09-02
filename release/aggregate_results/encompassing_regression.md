# E1 Forecast-encompassing regression: RV = a + b*f_HAR + g*f_text + e (test, HAC lag h-1)
g = coefficient on the text-only forecast conditional on the HAR-RV forecast. g not significant (p>=0.05) => text carries NO incremental information beyond HAR-RV.


## long_form
| model | h | g (text coef) | HAC t | p | verdict |
|---|---|---|---|---|---|
| B1_bow_ridge | 5 | +0.134 | +4.76 | 0.0000 | text adds (g>0,sig) |
| B1_bow_ridge | 10 | +0.179 | +4.18 | 0.0000 | text adds (g>0,sig) |
| B1_bow_ridge | 20 | +0.199 | +4.25 | 0.0000 | text adds (g>0,sig) |
| B2_tfidf_ridge | 5 | +0.698 | +16.23 | 0.0000 | text adds (g>0,sig) |
| B2_tfidf_ridge | 10 | +0.668 | +17.26 | 0.0000 | text adds (g>0,sig) |
| B2_tfidf_ridge | 20 | +0.690 | +21.12 | 0.0000 | text adds (g>0,sig) |
| B3_lm_linear | 5 | +0.652 | +6.82 | 0.0000 | text adds (g>0,sig) |
| B3_lm_linear | 10 | +0.722 | +7.31 | 0.0000 | text adds (g>0,sig) |
| B3_lm_linear | 20 | +0.707 | +9.99 | 0.0000 | text adds (g>0,sig) |
| B4_lm_features | 5 | +1.123 | +5.93 | 0.0000 | text adds (g>0,sig) |
| B4_lm_features | 10 | +0.573 | +5.99 | 0.0000 | text adds (g>0,sig) |
| B4_lm_features | 20 | +0.566 | +8.60 | 0.0000 | text adds (g>0,sig) |
| C1_bert_s1 | 5 | +0.626 | +12.96 | 0.0000 | text adds (g>0,sig) |
| C1_bert_s1 | 10 | +0.706 | +13.21 | 0.0000 | text adds (g>0,sig) |
| C1_bert_s1 | 20 | +0.611 | +16.01 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s1 | 5 | +0.601 | +12.69 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s1 | 10 | +0.444 | +12.80 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s1 | 20 | +0.554 | +16.14 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s2 | 5 | +0.052 | +2.28 | 0.0225 | text adds (g>0,sig) |
| C2_finbert_s2 | 10 | +0.560 | +14.11 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s2 | 20 | +0.350 | +11.23 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s3 | 5 | +0.181 | +6.77 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s3 | 10 | +0.336 | +12.79 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s3 | 20 | +0.364 | +13.82 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s4 | 5 | +0.294 | +7.00 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s4 | 10 | +0.112 | +4.52 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s4 | 20 | +0.397 | +13.10 | 0.0000 | text adds (g>0,sig) |
| C3_roberta_s1 | 5 | +0.569 | +11.09 | 0.0000 | text adds (g>0,sig) |
| C3_roberta_s1 | 10 | +0.417 | +11.04 | 0.0000 | text adds (g>0,sig) |
| C3_roberta_s1 | 20 | +0.334 | +10.49 | 0.0000 | text adds (g>0,sig) |
| C4_longformer | 5 | +0.484 | +12.82 | 0.0000 | text adds (g>0,sig) |
| C4_longformer | 10 | +0.476 | +12.59 | 0.0000 | text adds (g>0,sig) |
| C4_longformer | 20 | +0.602 | +17.37 | 0.0000 | text adds (g>0,sig) |

## event_driven
| model | h | g (text coef) | HAC t | p | verdict |
|---|---|---|---|---|---|
| B1_bow_ridge | 5 | +0.054 | +3.85 | 0.0001 | text adds (g>0,sig) |
| B1_bow_ridge | 10 | +0.053 | +4.20 | 0.0000 | text adds (g>0,sig) |
| B1_bow_ridge | 20 | +0.089 | +3.59 | 0.0003 | text adds (g>0,sig) |
| B2_tfidf_ridge | 5 | +0.372 | +13.95 | 0.0000 | text adds (g>0,sig) |
| B2_tfidf_ridge | 10 | +0.366 | +14.77 | 0.0000 | text adds (g>0,sig) |
| B2_tfidf_ridge | 20 | +0.376 | +17.22 | 0.0000 | text adds (g>0,sig) |
| B3_lm_linear | 5 | +1.184 | +5.71 | 0.0000 | text adds (g>0,sig) |
| B3_lm_linear | 10 | +1.024 | +5.73 | 0.0000 | text adds (g>0,sig) |
| B3_lm_linear | 20 | +1.023 | +6.73 | 0.0000 | text adds (g>0,sig) |
| B4_lm_features | 5 | +2.882 | +6.81 | 0.0000 | text adds (g>0,sig) |
| B4_lm_features | 10 | +1.953 | +6.10 | 0.0000 | text adds (g>0,sig) |
| B4_lm_features | 20 | +1.236 | +6.63 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s1 | 5 | +0.194 | +12.74 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s1 | 10 | +0.365 | +19.66 | 0.0000 | text adds (g>0,sig) |
| C2_finbert_s1 | 20 | +0.303 | +20.37 | 0.0000 | text adds (g>0,sig) |

**Summary:** of 48 encompassing tests, 48 show a significantly positive text coefficient (incremental value), 0 significantly negative, 0 no significant incremental information beyond HAR-RV.