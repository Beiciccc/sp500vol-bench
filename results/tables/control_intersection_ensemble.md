# P0-2 — Control intersection on the SEED-ENSEMBLE basis (controls now kill the DECLARED primary)

## RESTATED vs BEFORE

BEFORE = committed control_intersection.{csv,md}: primary marginal from m1_clustered (seed2026, 29/69) and controls on seed2026 text — an internal basis mismatch with the declared primary (m1_ensemble_primary, 38/69). RESTATED: every ingredient of the AND is now the SAME seed-ensemble forecast object: primary marginal = m1_ensemble_primary flags; maximal + firm controls = {maximal_reference,firm_identity}_ensemble. Raw p<.05 and Holm<.05 both reported; the within-date column is retained for information ONLY (seed2026 basis, excluded from every AND).

| quantity | BEFORE (seed2026 basis) | RESTATED (seed-ensemble basis) |
|---|---|---|
| primary marginal, raw / Holm | 39 / 29 (m1_clustered) | **46 / 38** (declared primary; placebo-gated genuine = 38) |
| maximal-pool survivors, raw / Holm | 23 / 8 | **26 / 9** |
| firm-identity survivors, raw / Holm | 14 / 8 | **15 / 8** |
| AND maximal & firm, raw / Holm | 4 / 0 | **4 / 0** |
| FULL AND (primary & maximal & firm), raw / Holm | 4 / 0 | **4 / 0** |
| strictest: placebo-gated genuine & both Holm controls | — | **0** |
| maximal vs firm Holm-survivor overlap | 0 (disjoint) | **0 (disjoint)** |

## Does the headline hold on the declared-primary basis?

1. **Holm AND = 0**: YES — holds (full AND = 0/69 under Holm; maximal&firm AND = 0/69; strictest placebo-gated version = 0/69).
2. **Survivor sets disjoint**: YES — holds (overlap = empty).
3. Raw-p full AND = 4/69: long_form/B3_lm_linear/h10; long_form/C6_llmtext/h20; long_form/D4_llmfused/h20; event_driven/B4_lm_features/h10.


## Per-cell flags (ensemble basis)
| disc | model | h | seeds | primary raw/Holm/genuine | maximal raw/Holm | firm raw/Holm | AND mf Holm | AND full Holm | within-date (s26, info) |
|---|---|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | 1 | Y/Y/Y | ./. | Y/Y | . | . | cross-sectional |
| event_driven | B1_bow_ridge | 10 | 1 | Y/Y/Y | ./. | Y/Y | . | . | cross-sectional |
| event_driven | B1_bow_ridge | 20 | 1 | Y/Y/Y | ./. | ./. | . | . | cross-sectional |
| event_driven | B2_tfidf_ridge | 5 | 1 | Y/./. | ./. | Y/. | . | . | cross-sectional |
| event_driven | B2_tfidf_ridge | 10 | 1 | Y/./. | ./. | Y/Y | . | . | cross-sectional |
| event_driven | B2_tfidf_ridge | 20 | 1 | Y/Y/Y | ./. | ./. | . | . | cross-sectional |
| event_driven | B3_lm_linear | 5 | 1 | Y/./. | ./. | Y/. | . | . | regime-timing |
| event_driven | B3_lm_linear | 10 | 1 | ././. | ./. | ./. | . | . | cross-sectional |
| event_driven | B3_lm_linear | 20 | 1 | ././. | ./. | Y/. | . | . | n/a (not genuine) |
| event_driven | B4_lm_features | 5 | 1 | ././. | ./. | ./. | . | . | regime-timing |
| event_driven | B4_lm_features | 10 | 1 | Y/./. | Y/. | Y/. | . | . | cross-sectional |
| event_driven | B4_lm_features | 20 | 1 | Y/./. | ./. | Y/. | . | . | cross-sectional |
| event_driven | C2_finbert_s1 | 5 | 3 | Y/Y/Y | Y/. | ./. | . | . | n/a (not genuine) |
| event_driven | C2_finbert_s1 | 10 | 3 | Y/Y/Y | ./. | ./. | . | . | cross-sectional |
| event_driven | C2_finbert_s1 | 20 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| event_driven | C6_llmtext | 5 | 1 | Y/Y/Y | ./. | Y/Y | . | . | mixed |
| event_driven | C6_llmtext | 10 | 1 | Y/Y/Y | ./. | Y/Y | . | . | cross-sectional |
| event_driven | C6_llmtext | 20 | 1 | Y/./. | ./. | Y/Y | . | . | n/a (not genuine) |
| event_driven | D2_gated_fusion | 5 | 3 | ././. | ./. | Y/. | . | . | n/a (not genuine) |
| event_driven | D2_gated_fusion | 10 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| event_driven | D2_gated_fusion | 20 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| event_driven | D4_llmfused | 5 | 1 | ././. | Y/. | ./. | . | . | n/a (not genuine) |
| event_driven | D4_llmfused | 10 | 1 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| event_driven | D4_llmfused | 20 | 1 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | B1_bow_ridge | 5 | 1 | Y/Y/Y | Y/. | ./. | . | . | cross-sectional |
| long_form | B1_bow_ridge | 10 | 1 | Y/Y/Y | ./. | ./. | . | . | cross-sectional |
| long_form | B1_bow_ridge | 20 | 1 | Y/Y/Y | Y/. | ./. | . | . | cross-sectional |
| long_form | B2_tfidf_ridge | 5 | 1 | Y/Y/Y | Y/Y | ./. | . | . | cross-sectional |
| long_form | B2_tfidf_ridge | 10 | 1 | Y/Y/Y | Y/Y | ./. | . | . | cross-sectional |
| long_form | B2_tfidf_ridge | 20 | 1 | Y/Y/Y | Y/Y | ./. | . | . | n/a (not genuine) |
| long_form | B3_lm_linear | 5 | 1 | Y/./. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | B3_lm_linear | 10 | 1 | Y/Y/Y | Y/. | Y/Y | . | . | regime-timing |
| long_form | B3_lm_linear | 20 | 1 | Y/Y/Y | Y/. | ./. | . | . | cross-sectional |
| long_form | B4_lm_features | 5 | 1 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | B4_lm_features | 10 | 1 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | B4_lm_features | 20 | 1 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | C1_bert_s1 | 5 | 3 | Y/Y/Y | ./. | ./. | . | . | n/a (not genuine) |
| long_form | C1_bert_s1 | 10 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | C1_bert_s1 | 20 | 3 | Y/Y/Y | Y/. | ./. | . | . | cross-sectional |
| long_form | C2_finbert_s1 | 5 | 3 | Y/Y/Y | Y/. | ./. | . | . | cross-sectional |
| long_form | C2_finbert_s1 | 10 | 3 | Y/Y/Y | Y/. | ./. | . | . | cross-sectional |
| long_form | C2_finbert_s1 | 20 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | C2_finbert_s2 | 5 | 3 | Y/Y/Y | Y/Y | ./. | . | . | cross-sectional |
| long_form | C2_finbert_s2 | 10 | 3 | Y/Y/Y | ./. | ./. | . | . | cross-sectional |
| long_form | C2_finbert_s2 | 20 | 3 | Y/Y/Y | ./. | ./. | . | . | n/a (not genuine) |
| long_form | C2_finbert_s3 | 5 | 3 | Y/Y/Y | Y/. | ./. | . | . | cross-sectional |
| long_form | C2_finbert_s3 | 10 | 3 | Y/Y/Y | Y/. | ./. | . | . | cross-sectional |
| long_form | C2_finbert_s3 | 20 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | C2_finbert_s4 | 5 | 3 | Y/Y/Y | Y/. | ./. | . | . | n/a (not genuine) |
| long_form | C2_finbert_s4 | 10 | 3 | Y/Y/Y | ./. | ./. | . | . | cross-sectional |
| long_form | C2_finbert_s4 | 20 | 3 | Y/Y/Y | Y/Y | ./. | . | . | cross-sectional |
| long_form | C3_roberta_s1 | 5 | 3 | Y/Y/Y | ./. | ./. | . | . | cross-sectional |
| long_form | C3_roberta_s1 | 10 | 3 | Y/Y/Y | Y/. | ./. | . | . | cross-sectional |
| long_form | C3_roberta_s1 | 20 | 3 | Y/Y/Y | ./. | ./. | . | . | cross-sectional |
| long_form | C4_longformer | 5 | 3 | Y/Y/Y | Y/Y | ./. | . | . | cross-sectional |
| long_form | C4_longformer | 10 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | C4_longformer | 20 | 3 | Y/Y/Y | ./. | ./. | . | . | n/a (not genuine) |
| long_form | C6_llmtext | 5 | 1 | Y/Y/Y | Y/Y | ./. | . | . | cross-sectional |
| long_form | C6_llmtext | 10 | 1 | Y/Y/Y | Y/Y | ./. | . | . | cross-sectional |
| long_form | C6_llmtext | 20 | 1 | Y/Y/Y | Y/. | Y/Y | . | . | n/a (not genuine) |
| long_form | D1_concat_mlp | 5 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | D1_concat_mlp | 10 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | D1_concat_mlp | 20 | 3 | Y/Y/Y | Y/Y | ./. | . | . | n/a (not genuine) |
| long_form | D2_gated_fusion | 5 | 3 | Y/./. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | D2_gated_fusion | 10 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | D2_gated_fusion | 20 | 3 | ././. | ./. | ./. | . | . | n/a (not genuine) |
| long_form | D4_llmfused | 5 | 1 | ././. | ./. | ./. | . | . | regime-timing |
| long_form | D4_llmfused | 10 | 1 | ././. | Y/. | ./. | . | . | n/a (not genuine) |
| long_form | D4_llmfused | 20 | 1 | Y/Y/Y | Y/. | Y/. | . | . | cross-sectional |

## Survivor sets (Holm, ensemble basis)

**Maximal (n=9):** long_form/B2_tfidf_ridge/h10, long_form/B2_tfidf_ridge/h20, long_form/B2_tfidf_ridge/h5, long_form/C2_finbert_s2/h5, long_form/C2_finbert_s4/h20, long_form/C4_longformer/h5, long_form/C6_llmtext/h10, long_form/C6_llmtext/h5, long_form/D1_concat_mlp/h20

**Firm-identity (n=8):** event_driven/B1_bow_ridge/h10, event_driven/B1_bow_ridge/h5, event_driven/B2_tfidf_ridge/h10, event_driven/C6_llmtext/h10, event_driven/C6_llmtext/h20, event_driven/C6_llmtext/h5, long_form/B3_lm_linear/h10, long_form/C6_llmtext/h20

**Overlap:** EMPTY — the two survivor sets are disjoint


## Sanity
- Single-seed cells (36/69) bit-reproduce the committed seed2026 control tables (gate enforced in scripts/analysis/basis_alignment_ensemble.py; run aborts on drift).
- 0/69 cells lose observations to the 3-seed inner join (max 0 obs).
- Primary marginals reconcile with m1_ensemble_primary.md by construction (same CSV, same flags).