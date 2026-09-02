# Matched-row cascade: is the collapse a control effect or a sample effect?

The committed primary rung is scored on the A2+text join; the firm-identity
rung is scored on the five-price-model intersection, which is ~9.7% smaller
in rows and ~2.1% smaller in trading days. That makes the headline collapse
confounded with a sample change. Here BOTH rungs are re-scored on ONE row
set --- the identity rung's own support --- so only the reference differs.
Predictions are frozen; weights are val-fitted and frozen to test; inference
is day-clustered DM with Holm within the 69-cell family.

**Validation arm.** The identical code path run on the committed row set
(A2+text join, no five-model intersection) returns 38/69, matching the
committed table's 38/69. Independently, the matched arm's FIRM rung
runs on the committed firm table's own support and reproduces it cell for cell
(8 cells, set-identical to its `adds_holm`). The difference below is therefore the
row set, not a reimplementation difference.

| rung | Holm survivors of 69 |
|---|---|
| recalibrated HAR, committed (unmatched rows) | 38 |
| recalibrated HAR, **matched rows** | **30** |
| $+$ firm identity, matched rows | **8** |

So on a single row set the collapse is **30 to 8**. The sample
change accounts for 8 of the committed
38 to 8 drop; the control accounts for the rest.

The second step is **not monotone**: 26 cells fall but 4 rise,
which is why the 22 is a net. Only 3 of the 4
are cells the identity term sharpens -- their *raw* clustered-DM p falls by an
order of magnitude once the identity term is in the reference, and the firm
family's median raw p is the larger of the two, so Holm is if anything stricter
there. The remaining riser(s) leave the primary rung on the placebo term alone:
   - long_form/B3_lm_linear/h=10: p_primary 6.94e-06 -> p_firm 3.02e-04 (weaker), excluded from the primary rung by placebo -2.29, not by DM or Holm.

| disc | model | h | primary rel% | firm rel% | primary adds | firm adds |
|---|---|---|---|---|---|---|
| long_form | B1_bow_ridge | 5 | +1.65 | +0.02 | yes | no |
| long_form | B1_bow_ridge | 10 | +1.44 | -2.09 | yes | no |
| long_form | B1_bow_ridge | 20 | +2.99 | -3.18 | yes | no |
| long_form | B2_tfidf_ridge | 5 | +3.33 | -0.34 | yes | no |
| long_form | B2_tfidf_ridge | 10 | +3.48 | -3.04 | yes | no |
| long_form | B2_tfidf_ridge | 20 | +5.92 | -5.56 | yes | no |
| long_form | B3_lm_linear | 5 | +0.49 | +0.24 | no | no |
| long_form | B3_lm_linear | 10 | +1.79 | +0.43 | yes | yes |
| long_form | B3_lm_linear | 20 | +3.48 | +0.57 | yes | no |
| long_form | B4_lm_features | 5 | +0.11 | -0.19 | no | no |
| long_form | B4_lm_features | 10 | -0.92 | -2.83 | no | no |
| long_form | B4_lm_features | 20 | -1.92 | -5.92 | no | no |
| long_form | C1_bert_s1 | 5 | +3.42 | +1.38 | yes | no |
| long_form | C1_bert_s1 | 10 | -0.85 | -3.85 | no | no |
| long_form | C1_bert_s1 | 20 | +2.70 | -4.75 | yes | no |
| long_form | C2_finbert_s1 | 5 | +1.90 | -0.05 | yes | no |
| long_form | C2_finbert_s1 | 10 | +2.62 | -2.36 | yes | no |
| long_form | C2_finbert_s1 | 20 | -0.59 | +0.10 | no | no |
| long_form | C2_finbert_s2 | 5 | +1.21 | -1.81 | yes | no |
| long_form | C2_finbert_s2 | 10 | +0.48 | -2.84 | yes | no |
| long_form | C2_finbert_s2 | 20 | +1.68 | -2.75 | yes | no |
| long_form | C2_finbert_s3 | 5 | +2.90 | -0.53 | yes | no |
| long_form | C2_finbert_s3 | 10 | +2.31 | -0.58 | yes | no |
| long_form | C2_finbert_s3 | 20 | -3.86 | +0.75 | no | no |
| long_form | C2_finbert_s4 | 5 | +1.46 | +0.11 | yes | no |
| long_form | C2_finbert_s4 | 10 | +0.36 | -3.34 | yes | no |
| long_form | C2_finbert_s4 | 20 | +3.08 | -5.48 | yes | no |
| long_form | C3_roberta_s1 | 5 | +0.30 | -2.51 | yes | no |
| long_form | C3_roberta_s1 | 10 | +1.84 | -0.41 | yes | no |
| long_form | C3_roberta_s1 | 20 | +0.02 | -2.71 | yes | no |
| long_form | C4_longformer | 5 | +1.47 | -2.07 | yes | no |
| long_form | C4_longformer | 10 | -2.95 | -10.04 | no | no |
| long_form | C4_longformer | 20 | +0.91 | -3.78 | yes | no |
| long_form | C6_llmtext | 5 | +1.79 | -0.16 | yes | no |
| long_form | C6_llmtext | 10 | +2.25 | -0.06 | yes | no |
| long_form | C6_llmtext | 20 | +0.27 | +0.15 | yes | yes |
| long_form | D1_concat_mlp | 5 | -1.04 | -1.41 | no | no |
| long_form | D1_concat_mlp | 10 | -0.46 | -1.48 | no | no |
| long_form | D1_concat_mlp | 20 | +0.12 | -2.54 | yes | no |
| long_form | D2_gated_fusion | 5 | +0.19 | -0.14 | no | no |
| long_form | D2_gated_fusion | 10 | -0.02 | +0.23 | no | no |
| long_form | D2_gated_fusion | 20 | -2.27 | -3.43 | no | no |
| long_form | D4_llmfused | 5 | +0.15 | +0.02 | no | no |
| long_form | D4_llmfused | 10 | -0.15 | -0.16 | no | no |
| long_form | D4_llmfused | 20 | +0.60 | +0.31 | yes | no |
| event_driven | B1_bow_ridge | 5 | +1.33 | +0.51 | yes | yes |
| event_driven | B1_bow_ridge | 10 | +1.23 | +0.37 | yes | yes |
| event_driven | B1_bow_ridge | 20 | +1.53 | +0.11 | yes | yes |
| event_driven | B2_tfidf_ridge | 5 | +1.21 | +0.46 | no | no |
| event_driven | B2_tfidf_ridge | 10 | +1.35 | +0.37 | no | yes |
| event_driven | B2_tfidf_ridge | 20 | +1.84 | +0.27 | yes | yes |
| event_driven | B3_lm_linear | 5 | +0.25 | +0.14 | no | no |
| event_driven | B3_lm_linear | 10 | +0.20 | +0.12 | no | no |
| event_driven | B3_lm_linear | 20 | +0.26 | +0.17 | no | no |
| event_driven | B4_lm_features | 5 | +0.18 | +0.23 | no | no |
| event_driven | B4_lm_features | 10 | +0.08 | +0.14 | no | no |
| event_driven | B4_lm_features | 20 | +0.25 | +0.09 | no | no |
| event_driven | C2_finbert_s1 | 5 | +2.57 | +0.10 | yes | yes |
| event_driven | C2_finbert_s1 | 10 | +2.42 | -0.49 | yes | no |
| event_driven | C2_finbert_s1 | 20 | +1.58 | -0.28 | no | no |
| event_driven | C6_llmtext | 5 | +1.21 | +0.53 | yes | yes |
| event_driven | C6_llmtext | 10 | +1.00 | +0.30 | yes | yes |
| event_driven | C6_llmtext | 20 | +0.66 | +0.24 | no | yes |
| event_driven | D2_gated_fusion | 5 | +0.56 | +0.27 | no | no |
| event_driven | D2_gated_fusion | 10 | +0.18 | +0.09 | no | no |
| event_driven | D2_gated_fusion | 20 | -2.76 | -7.27 | no | no |
| event_driven | D4_llmfused | 5 | -0.01 | -0.08 | no | no |
| event_driven | D4_llmfused | 10 | -0.01 | -0.03 | no | no |
| event_driven | D4_llmfused | 20 | -0.35 | -0.22 | no | no |
| long_form | B1_bow_ridge | 5 | +1.75 | -0.14 | yes | no |
| long_form | B1_bow_ridge | 10 | +1.23 | -2.72 | no | no |
| long_form | B1_bow_ridge | 20 | +2.39 | -4.98 | yes | no |
| long_form | B2_tfidf_ridge | 5 | +3.39 | -0.61 | yes | no |
| long_form | B2_tfidf_ridge | 10 | +3.63 | -3.89 | yes | no |
| long_form | B2_tfidf_ridge | 20 | +5.27 | -8.09 | no | no |
| long_form | B3_lm_linear | 5 | +0.40 | +0.21 | no | no |
| long_form | B3_lm_linear | 10 | +1.62 | +0.26 | no | yes |
| long_form | B3_lm_linear | 20 | +3.26 | -0.02 | yes | no |
| long_form | B4_lm_features | 5 | +0.13 | -0.14 | no | no |
| long_form | B4_lm_features | 10 | -0.89 | -2.77 | no | no |
| long_form | B4_lm_features | 20 | -2.39 | -7.13 | no | no |
| long_form | C1_bert_s1 | 5 | +3.38 | +1.38 | yes | no |
| long_form | C1_bert_s1 | 10 | -1.05 | -4.75 | no | no |
| long_form | C1_bert_s1 | 20 | +2.14 | -6.49 | yes | no |
| long_form | C2_finbert_s1 | 5 | +1.83 | -0.27 | yes | no |
| long_form | C2_finbert_s1 | 10 | +2.41 | -3.18 | yes | no |
| long_form | C2_finbert_s1 | 20 | -0.56 | -0.01 | no | no |
| long_form | C2_finbert_s2 | 5 | +1.08 | -2.20 | yes | no |
| long_form | C2_finbert_s2 | 10 | +0.22 | -3.73 | yes | no |
| long_form | C2_finbert_s2 | 20 | +1.38 | -3.82 | yes | no |
| long_form | C2_finbert_s3 | 5 | +2.83 | -0.82 | yes | no |
| long_form | C2_finbert_s3 | 10 | +2.35 | -1.21 | yes | no |
| long_form | C2_finbert_s3 | 20 | -3.63 | +0.80 | no | no |
| long_form | C2_finbert_s4 | 5 | +1.37 | -0.12 | yes | no |
| long_form | C2_finbert_s4 | 10 | +0.13 | -4.07 | yes | no |
| long_form | C2_finbert_s4 | 20 | +2.66 | -6.87 | yes | no |
| long_form | C3_roberta_s1 | 5 | +0.28 | -2.78 | yes | no |
| long_form | C3_roberta_s1 | 10 | +1.54 | -0.78 | yes | no |
| long_form | C3_roberta_s1 | 20 | -0.30 | -3.56 | no | no |
| long_form | C4_longformer | 5 | +1.49 | -2.21 | yes | no |
| long_form | C4_longformer | 10 | -3.67 | -11.38 | no | no |
| long_form | C4_longformer | 20 | +0.46 | -5.03 | yes | no |
| long_form | C6_llmtext | 5 | +1.94 | -0.18 | yes | no |
| long_form | C6_llmtext | 10 | +1.84 | -0.43 | yes | no |
| long_form | C6_llmtext | 20 | +0.26 | +0.08 | yes | yes |
| long_form | D1_concat_mlp | 5 | -1.03 | -1.38 | no | no |
| long_form | D1_concat_mlp | 10 | -0.51 | -1.70 | no | no |
| long_form | D1_concat_mlp | 20 | -0.05 | -2.80 | no | no |
| long_form | D2_gated_fusion | 5 | +0.17 | -0.16 | no | no |
| long_form | D2_gated_fusion | 10 | -0.09 | +0.20 | no | no |
| long_form | D2_gated_fusion | 20 | -2.45 | -3.49 | no | no |
| long_form | D4_llmfused | 5 | +0.12 | -0.00 | no | no |
| long_form | D4_llmfused | 10 | -0.14 | -0.15 | no | no |
| long_form | D4_llmfused | 20 | +0.35 | +0.08 | no | no |
| event_driven | B1_bow_ridge | 5 | +1.31 | +0.46 | yes | yes |
| event_driven | B1_bow_ridge | 10 | +1.16 | +0.25 | yes | yes |
| event_driven | B1_bow_ridge | 20 | +1.39 | -0.19 | yes | no |
| event_driven | B2_tfidf_ridge | 5 | +1.17 | +0.44 | no | no |
| event_driven | B2_tfidf_ridge | 10 | +1.27 | +0.28 | no | yes |
| event_driven | B2_tfidf_ridge | 20 | +1.68 | -0.05 | no | no |
| event_driven | B3_lm_linear | 5 | +0.22 | +0.11 | no | no |
| event_driven | B3_lm_linear | 10 | +0.19 | +0.11 | no | no |
| event_driven | B3_lm_linear | 20 | +0.27 | +0.17 | no | no |
| event_driven | B4_lm_features | 5 | +0.17 | +0.22 | no | no |
| event_driven | B4_lm_features | 10 | +0.07 | +0.13 | no | no |
| event_driven | B4_lm_features | 20 | +0.26 | +0.11 | no | no |
| event_driven | C2_finbert_s1 | 5 | +2.44 | -0.06 | yes | no |
| event_driven | C2_finbert_s1 | 10 | +2.19 | -0.67 | yes | no |
| event_driven | C2_finbert_s1 | 20 | +1.43 | -0.61 | no | no |
| event_driven | C6_llmtext | 5 | +1.20 | +0.52 | yes | yes |
| event_driven | C6_llmtext | 10 | +0.92 | +0.24 | no | yes |
| event_driven | C6_llmtext | 20 | +0.60 | +0.21 | no | yes |
| event_driven | D2_gated_fusion | 5 | +0.51 | +0.23 | no | no |
| event_driven | D2_gated_fusion | 10 | +0.13 | +0.05 | no | no |
| event_driven | D2_gated_fusion | 20 | -3.48 | -7.99 | no | no |
| event_driven | D4_llmfused | 5 | -0.00 | -0.07 | no | no |
| event_driven | D4_llmfused | 10 | -0.00 | -0.02 | no | no |
| event_driven | D4_llmfused | 20 | -0.38 | -0.25 | no | no |
