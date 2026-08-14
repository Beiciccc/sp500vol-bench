# P1-b — TWO-WAY (FIRM x DAY) CLUSTER ROBUSTNESS of the DM inference

## RESTATED vs BEFORE

| panel (identical forecasts; only the variance estimator changes) | BEFORE (day-clustered, committed) | RESTATED (two-way firm x day, CGM) | flips |
|---|---|---|---|
| (a) M1 69-cell grid — genuine text-increment cells (DM<0, Holm<.05, placebo gate) | **29/69** (m1_clustered.csv) | **24/69** | 5 |
| (b) firm-identity-reference grid — text survives (Holm<.05) | **8/69** (firm_identity_control.csv) | **5/69** | 13 |
| (b) — text HURTS (Holm<.05) | 29/69 | 19/69 | — |
| (c) pairwise vs A2 (SE) — challengers significantly BETTER (Holm<.05) | **0/180** (dm_pairwise_clustered.csv) | **0/180** (raw p<.05: 0/180) | 3 |
| (c) — challengers significantly WORSE (Holm<.05) | 155/180 | 152/180 | — |

**Method.** Cameron-Gelbach-Miller two-way clustered variance on the mean loss differential (equal weight per day, matching the day-clustered primary): `V_2way = V_firm + V_day - V_firm∩day`, where `V_C = Σ_c (Σ_{i∈c} w_i(d_i - d̄))²` with `w_i = 1/(T·n_day(i))`, and the day component is the Newey-West HAC (lag = h-1 trading days) long-run variance of the daily-mean differential series divided by T — at lag 0 this equals the CGM day component exactly, so the serial-correlation treatment is identical to the committed day-clustered DM. Non-PSD guard: `V_2way <- max(V_2way, 1e-30)` (flagged); reference distribution t(min(#firms, #days) - 1). The firm∩day intersection is subtracted at lag 0 only (Thompson's lagged own-firm overlap terms omitted), which can only WIDEN the SEs — conservative for every significance claim. No HLN correction on the two-way stat (immaterial at n_days≈800; the day column keeps it, as committed). Full details: `scripts/analysis/twoway_dm.py`.

**SANITY (hard assertions, all PASS):** the recomputed day-clustered columns reproduce the committed tables exactly — (a) `m1_clustered.csv` (dm_q_clust, p_q_clust, dmq_holm_clust), (b) `firm_identity_control.csv` (dm_q_clustered, p_q_clustered, holm_p, survivor set), (c) `dm_pairwise_clustered.csv` (dm_clust, p_clust, p_holm_clust).

**Variance anatomy (medians):** SE inflation two-way vs day-only sqrt(V_2way/V_day): (a) 1.191, (b) 1.219, (c) 1.181. Median variance shares (a): firm 0.75, day 0.71, intersection (subtracted) 0.40; (c): firm 0.44, day 0.72, intersection 0.08. Non-PSD guard hits: 0/318 cells.

## Verdict flips (day-clustered -> two-way)

| panel | disc | model | h | dm_day | Holm(day) | dm_2way | p_2way | Holm(2way) | verdict day -> 2way |
|---|---|---|---|---|---|---|---|---|---|
| a_m1_grid | long_form | B3_lm_linear | 20 | -4.69 | 0.0002 | -2.87 | 0.0043 | 0.1592 | genuine -> ns |
| a_m1_grid | long_form | C2_finbert_s2 | 5 | -4.30 | 0.0009 | -3.20 | 0.0014 | 0.0579 | genuine -> ns |
| a_m1_grid | long_form | C6_llmtext | 20 | -3.23 | 0.0422 | -2.76 | 0.0060 | 0.2047 | genuine -> ns |
| a_m1_grid | event_driven | B1_bow_ridge | 5 | -3.35 | 0.0304 | -3.07 | 0.0023 | 0.0865 | genuine -> ns |
| a_m1_grid | event_driven | B1_bow_ridge | 10 | -3.25 | 0.0403 | -2.81 | 0.0052 | 0.1812 | genuine -> ns |
| b_firm_ref | long_form | B3_lm_linear | 10 | -3.63 | 0.0121 | -2.56 | 0.0108 | 0.3452 | text adds -> ns |
| b_firm_ref | long_form | B3_lm_linear | 20 | +3.47 | 0.0201 | +2.39 | 0.0172 | 0.5163 | text HURTS -> ns |
| b_firm_ref | long_form | B4_lm_features | 10 | +4.10 | 0.0021 | +2.99 | 0.0029 | 0.1121 | text HURTS -> ns |
| b_firm_ref | long_form | B4_lm_features | 20 | +4.62 | 0.0003 | +3.15 | 0.0017 | 0.0740 | text HURTS -> ns |
| b_firm_ref | long_form | C1_bert_s1 | 10 | +3.26 | 0.0393 | +2.25 | 0.0249 | 0.7221 | text HURTS -> ns |
| b_firm_ref | long_form | C2_finbert_s2 | 10 | +4.23 | 0.0013 | +3.05 | 0.0024 | 0.0955 | text HURTS -> ns |
| b_firm_ref | long_form | C2_finbert_s3 | 5 | +4.03 | 0.0027 | +2.98 | 0.0030 | 0.1142 | text HURTS -> ns |
| b_firm_ref | long_form | C3_roberta_s1 | 5 | +3.84 | 0.0054 | +2.42 | 0.0159 | 0.4924 | text HURTS -> ns |
| b_firm_ref | long_form | C3_roberta_s1 | 10 | +4.14 | 0.0018 | +2.62 | 0.0091 | 0.3102 | text HURTS -> ns |
| b_firm_ref | event_driven | B1_bow_ridge | 5 | -3.50 | 0.0184 | -3.15 | 0.0017 | 0.0740 | text adds -> ns |
| b_firm_ref | event_driven | C6_llmtext | 20 | -3.58 | 0.0141 | -3.23 | 0.0013 | 0.0598 | text adds -> ns |
| b_firm_ref | event_driven | D4_llmfused | 5 | +3.19 | 0.0477 | +3.13 | 0.0019 | 0.0763 | text HURTS -> ns |
| b_firm_ref | event_driven | D4_llmfused | 20 | +3.48 | 0.0194 | +3.21 | 0.0014 | 0.0626 | text HURTS -> ns |
| c_pairwise_vsA2 | event_driven | D3_qwen3 | 10 | +2.52 | 0.0360 | +2.38 | 0.0177 | 0.0531 | sig worse -> ns |
| c_pairwise_vsA2 | event_driven | D3_gteqwen2 | 10 | +2.11 | 0.0360 | +2.01 | 0.0447 | 0.0531 | sig worse -> ns |
| c_pairwise_vsA2 | event_driven | D3_e5mistral | 10 | +2.51 | 0.0360 | +2.37 | 0.0179 | 0.0531 | sig worse -> ns |


## (a) M1 69-cell grid — day-clustered vs two-way (seed2026 basis of m1_clustered)
| disc | model | h | n_firms | n_days | dm_day | Holm(day) | dm_2way | p_2way | Holm(2way) | placebo 2way | SEx | genuine day->2way |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | 569 | 996 | -3.35 | 0.030 | -3.07 | 0.0023 | 0.086 | +0.56 | 1.10 | Y->n |
| event_driven | B1_bow_ridge | 10 | 569 | 991 | -3.25 | 0.040 | -2.81 | 0.0052 | 0.181 | +0.02 | 1.17 | Y->n |
| event_driven | B1_bow_ridge | 20 | 568 | 981 | -3.10 | 0.063 | -2.53 | 0.0116 | 0.373 | +0.52 | 1.25 | n->n |
| event_driven | B2_tfidf_ridge | 5 | 569 | 996 | -2.76 | 0.165 | -2.31 | 0.0212 | 0.642 | +0.38 | 1.20 | n->n |
| event_driven | B2_tfidf_ridge | 10 | 569 | 991 | -2.97 | 0.090 | -2.25 | 0.0247 | 0.717 | +0.33 | 1.33 | n->n |
| event_driven | B2_tfidf_ridge | 20 | 568 | 981 | -3.11 | 0.063 | -2.06 | 0.0398 | 1.000 | +0.49 | 1.54 | n->n |
| event_driven | B3_lm_linear | 5 | 569 | 996 | -2.43 | 0.394 | -2.20 | 0.0279 | 0.783 | +0.01 | 1.11 | n->n |
| event_driven | B3_lm_linear | 10 | 569 | 991 | -1.23 | 1.000 | -1.16 | 0.2466 | 1.000 | +0.16 | 1.07 | n->n |
| event_driven | B3_lm_linear | 20 | 568 | 981 | -1.12 | 1.000 | -1.03 | 0.3026 | 1.000 | +0.91 | 1.11 | n->n |
| event_driven | B4_lm_features | 5 | 569 | 996 | -0.99 | 1.000 | -0.92 | 0.3566 | 1.000 | +0.45 | 1.08 | n->n |
| event_driven | B4_lm_features | 10 | 569 | 991 | -2.10 | 0.856 | -1.93 | 0.0547 | 1.000 | -0.03 | 1.10 | n->n |
| event_driven | B4_lm_features | 20 | 568 | 981 | -2.01 | 0.984 | -1.67 | 0.0956 | 1.000 | +0.53 | 1.23 | n->n |
| event_driven | C2_finbert_s1 | 5 | 569 | 996 | -4.95 | 0.000 | -4.39 | 0.0000 | 0.001 | -0.93 | 1.13 | Y->Y |
| event_driven | C2_finbert_s1 | 10 | 569 | 991 | -5.52 | 0.000 | -4.51 | 0.0000 | 0.000 | -0.17 | 1.24 | Y->Y |
| event_driven | C2_finbert_s1 | 20 | 568 | 981 | -0.50 | 1.000 | -0.32 | 0.7490 | 1.000 | -0.45 | 1.58 | n->n |
| event_driven | C6_llmtext | 5 | 569 | 996 | -5.04 | 0.000 | -4.56 | 0.0000 | 0.000 | +0.05 | 1.11 | Y->Y |
| event_driven | C6_llmtext | 10 | 569 | 991 | -3.76 | 0.008 | -3.49 | 0.0005 | 0.024 | +0.96 | 1.09 | Y->Y |
| event_driven | C6_llmtext | 20 | 568 | 981 | -1.98 | 1.000 | -1.82 | 0.0692 | 1.000 | +1.12 | 1.11 | n->n |
| event_driven | D2_gated_fusion | 5 | 569 | 996 | -0.69 | 1.000 | -0.65 | 0.5173 | 1.000 | +2.96 | 1.08 | n->n |
| event_driven | D2_gated_fusion | 10 | 569 | 991 | -0.56 | 1.000 | -0.54 | 0.5863 | 1.000 | -0.18 | 1.03 | n->n |
| event_driven | D2_gated_fusion | 20 | 568 | 981 | +3.00 | 0.083 | +2.83 | 0.0048 | 0.173 | -0.35 | 1.08 | n->n |
| event_driven | D4_llmfused | 5 | 569 | 996 | +3.37 | 0.029 | +3.32 | 0.0010 | 0.040 | +1.52 | 1.02 | n->n |
| event_driven | D4_llmfused | 10 | 569 | 991 | +0.66 | 1.000 | +0.64 | 0.5221 | 1.000 | -0.08 | 1.04 | n->n |
| event_driven | D4_llmfused | 20 | 568 | 981 | +4.69 | 0.000 | +4.36 | 0.0000 | 0.001 | -0.18 | 1.10 | n->n |
| long_form | B1_bow_ridge | 5 | 568 | 809 | -3.83 | 0.006 | -3.35 | 0.0009 | 0.039 | +0.43 | 1.15 | Y->Y |
| long_form | B1_bow_ridge | 10 | 568 | 803 | -4.15 | 0.002 | -3.74 | 0.0002 | 0.010 | +0.36 | 1.12 | Y->Y |
| long_form | B1_bow_ridge | 20 | 567 | 794 | -5.45 | 0.000 | -4.34 | 0.0000 | 0.001 | -0.16 | 1.29 | Y->Y |
| long_form | B2_tfidf_ridge | 5 | 568 | 809 | -5.39 | 0.000 | -4.83 | 0.0000 | 0.000 | +0.49 | 1.12 | Y->Y |
| long_form | B2_tfidf_ridge | 10 | 568 | 803 | -8.89 | 0.000 | -7.06 | 0.0000 | 0.000 | +0.39 | 1.28 | Y->Y |
| long_form | B2_tfidf_ridge | 20 | 567 | 794 | -9.04 | 0.000 | -6.82 | 0.0000 | 0.000 | +0.38 | 1.36 | Y->Y |
| long_form | B3_lm_linear | 5 | 568 | 809 | -2.58 | 0.269 | -1.95 | 0.0515 | 1.000 | +0.59 | 1.33 | n->n |
| long_form | B3_lm_linear | 10 | 568 | 803 | -4.62 | 0.000 | -3.35 | 0.0009 | 0.039 | +0.19 | 1.40 | Y->Y |
| long_form | B3_lm_linear | 20 | 567 | 794 | -4.69 | 0.000 | -2.87 | 0.0043 | 0.159 | -1.24 | 1.68 | Y->n |
| long_form | B4_lm_features | 5 | 568 | 809 | +1.02 | 1.000 | +0.80 | 0.4245 | 1.000 | +0.47 | 1.28 | n->n |
| long_form | B4_lm_features | 10 | 568 | 803 | +3.32 | 0.033 | +2.57 | 0.0105 | 0.346 | -1.67 | 1.31 | n->n |
| long_form | B4_lm_features | 20 | 567 | 794 | +3.38 | 0.029 | +2.32 | 0.0207 | 0.642 | -3.31 | 1.49 | n->n |
| long_form | C1_bert_s1 | 5 | 568 | 809 | -2.03 | 0.984 | -1.63 | 0.1045 | 1.000 | +0.57 | 1.25 | n->n |
| long_form | C1_bert_s1 | 10 | 568 | 803 | +2.35 | 0.476 | +1.94 | 0.0525 | 1.000 | -1.71 | 1.22 | n->n |
| long_form | C1_bert_s1 | 20 | 567 | 794 | -7.42 | 0.000 | -4.77 | 0.0000 | 0.000 | +0.38 | 1.60 | Y->Y |
| long_form | C2_finbert_s1 | 5 | 568 | 809 | -0.84 | 1.000 | -0.72 | 0.4735 | 1.000 | +0.18 | 1.18 | n->n |
| long_form | C2_finbert_s1 | 10 | 568 | 803 | -6.46 | 0.000 | -5.14 | 0.0000 | 0.000 | +1.04 | 1.27 | Y->Y |
| long_form | C2_finbert_s1 | 20 | 567 | 794 | +0.88 | 1.000 | +0.41 | 0.6791 | 1.000 | -0.54 | 2.19 | n->n |
| long_form | C2_finbert_s2 | 5 | 568 | 809 | -4.30 | 0.001 | -3.20 | 0.0014 | 0.058 | +0.15 | 1.35 | Y->n |
| long_form | C2_finbert_s2 | 10 | 568 | 803 | -6.58 | 0.000 | -4.88 | 0.0000 | 0.000 | +0.22 | 1.36 | Y->Y |
| long_form | C2_finbert_s2 | 20 | 567 | 794 | -0.06 | 1.000 | -0.05 | 0.9629 | 1.000 | +0.11 | 1.28 | n->n |
| long_form | C2_finbert_s3 | 5 | 568 | 809 | -5.43 | 0.000 | -4.24 | 0.0000 | 0.001 | +0.28 | 1.29 | Y->Y |
| long_form | C2_finbert_s3 | 10 | 568 | 803 | -5.13 | 0.000 | -4.01 | 0.0001 | 0.004 | +1.14 | 1.30 | Y->Y |
| long_form | C2_finbert_s3 | 20 | 567 | 794 | -1.77 | 1.000 | -1.48 | 0.1382 | 1.000 | +0.69 | 1.22 | n->n |
| long_form | C2_finbert_s4 | 5 | 568 | 809 | +3.36 | 0.030 | +3.33 | 0.0009 | 0.039 | +0.01 | 1.01 | n->n |
| long_form | C2_finbert_s4 | 10 | 568 | 803 | -6.02 | 0.000 | -5.34 | 0.0000 | 0.000 | -0.56 | 1.14 | Y->Y |
| long_form | C2_finbert_s4 | 20 | 567 | 794 | -8.82 | 0.000 | -6.53 | 0.0000 | 0.000 | +0.64 | 1.38 | Y->Y |
| long_form | C3_roberta_s1 | 5 | 568 | 809 | -5.37 | 0.000 | -3.79 | 0.0002 | 0.008 | +0.38 | 1.42 | Y->Y |
| long_form | C3_roberta_s1 | 10 | 568 | 803 | -5.18 | 0.000 | -3.56 | 0.0004 | 0.019 | +0.52 | 1.48 | Y->Y |
| long_form | C3_roberta_s1 | 20 | 567 | 794 | -1.07 | 1.000 | -0.52 | 0.6012 | 1.000 | -0.05 | 2.09 | n->n |
| long_form | C4_longformer | 5 | 568 | 809 | -5.99 | 0.000 | -5.06 | 0.0000 | 0.000 | -0.23 | 1.19 | Y->Y |
| long_form | C4_longformer | 10 | 568 | 803 | +10.55 | 0.000 | +7.63 | 0.0000 | 0.000 | +0.21 | 1.40 | n->n |
| long_form | C4_longformer | 20 | 567 | 794 | +3.46 | 0.023 | +3.15 | 0.0017 | 0.067 | -0.08 | 1.13 | n->n |
| long_form | C6_llmtext | 5 | 568 | 809 | -6.31 | 0.000 | -5.92 | 0.0000 | 0.000 | +0.78 | 1.07 | Y->Y |
| long_form | C6_llmtext | 10 | 568 | 803 | -7.92 | 0.000 | -7.59 | 0.0000 | 0.000 | -1.42 | 1.06 | Y->Y |
| long_form | C6_llmtext | 20 | 567 | 794 | -3.23 | 0.042 | -2.76 | 0.0060 | 0.205 | -1.60 | 1.20 | Y->n |
| long_form | D1_concat_mlp | 5 | 568 | 809 | +0.10 | 1.000 | +0.09 | 0.9276 | 1.000 | +0.80 | 1.05 | n->n |
| long_form | D1_concat_mlp | 10 | 568 | 803 | -0.67 | 1.000 | -0.57 | 0.5697 | 1.000 | -1.00 | 1.18 | n->n |
| long_form | D1_concat_mlp | 20 | 567 | 794 | -0.63 | 1.000 | -0.41 | 0.6843 | 1.000 | -0.28 | 1.59 | n->n |
| long_form | D2_gated_fusion | 5 | 568 | 809 | -0.29 | 1.000 | -0.28 | 0.7770 | 1.000 | +0.90 | 1.04 | n->n |
| long_form | D2_gated_fusion | 10 | 568 | 803 | +0.96 | 1.000 | +0.90 | 0.3692 | 1.000 | -1.07 | 1.08 | n->n |
| long_form | D2_gated_fusion | 20 | 567 | 794 | +4.27 | 0.001 | +4.13 | 0.0000 | 0.002 | -0.04 | 1.06 | n->n |
| long_form | D4_llmfused | 5 | 568 | 809 | -0.75 | 1.000 | -0.73 | 0.4660 | 1.000 | +1.06 | 1.04 | n->n |
| long_form | D4_llmfused | 10 | 568 | 803 | -0.43 | 1.000 | -0.39 | 0.6941 | 1.000 | +0.82 | 1.10 | n->n |
| long_form | D4_llmfused | 20 | 567 | 794 | -3.58 | 0.015 | -3.26 | 0.0012 | 0.049 | -0.49 | 1.13 | Y->Y |

## (b) Firm-identity-reference grid — day-clustered vs two-way
| disc | model | h | n_firms | n_days | dm_day | Holm(day) | dm_2way | p_2way | Holm(2way) | SEx | verdict day -> 2way |
|---|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | 569 | 996 | -3.50 | 0.018 | -3.15 | 0.0017 | 0.074 | 1.12 | text adds -> ns |
| event_driven | B1_bow_ridge | 10 | 569 | 991 | -4.71 | 0.000 | -3.81 | 0.0002 | 0.009 | 1.25 | text adds -> text adds |
| event_driven | B1_bow_ridge | 20 | 568 | 981 | +5.50 | 0.000 | +4.12 | 0.0000 | 0.003 | 1.36 | text HURTS -> text HURTS |
| event_driven | B2_tfidf_ridge | 5 | 569 | 996 | -3.06 | 0.072 | -2.62 | 0.0091 | 0.310 | 1.18 | ns -> ns |
| event_driven | B2_tfidf_ridge | 10 | 569 | 991 | -4.28 | 0.001 | -3.36 | 0.0008 | 0.039 | 1.29 | text adds -> text adds |
| event_driven | B2_tfidf_ridge | 20 | 568 | 981 | +5.08 | 0.000 | +3.64 | 0.0003 | 0.016 | 1.42 | text HURTS -> text HURTS |
| event_driven | B3_lm_linear | 5 | 569 | 996 | -3.00 | 0.086 | -2.72 | 0.0068 | 0.238 | 1.11 | ns -> ns |
| event_driven | B3_lm_linear | 10 | 569 | 991 | -1.91 | 1.000 | -1.76 | 0.0788 | 1.000 | 1.10 | ns -> ns |
| event_driven | B3_lm_linear | 20 | 568 | 981 | -2.12 | 0.793 | -1.91 | 0.0569 | 1.000 | 1.13 | ns -> ns |
| event_driven | B4_lm_features | 5 | 569 | 996 | -1.56 | 1.000 | -1.40 | 0.1609 | 1.000 | 1.11 | ns -> ns |
| event_driven | B4_lm_features | 10 | 569 | 991 | -2.60 | 0.253 | -2.20 | 0.0280 | 0.784 | 1.19 | ns -> ns |
| event_driven | B4_lm_features | 20 | 568 | 981 | -2.38 | 0.418 | -2.01 | 0.0447 | 1.000 | 1.21 | ns -> ns |
| event_driven | C2_finbert_s1 | 5 | 569 | 996 | +4.88 | 0.000 | +4.27 | 0.0000 | 0.001 | 1.15 | text HURTS -> text HURTS |
| event_driven | C2_finbert_s1 | 10 | 569 | 991 | +6.33 | 0.000 | +5.18 | 0.0000 | 0.000 | 1.23 | text HURTS -> text HURTS |
| event_driven | C2_finbert_s1 | 20 | 568 | 981 | -0.81 | 1.000 | -0.58 | 0.5595 | 1.000 | 1.42 | ns -> ns |
| event_driven | C6_llmtext | 5 | 569 | 996 | -4.98 | 0.000 | -4.52 | 0.0000 | 0.000 | 1.11 | text adds -> text adds |
| event_driven | C6_llmtext | 10 | 569 | 991 | -4.43 | 0.001 | -4.03 | 0.0001 | 0.004 | 1.11 | text adds -> text adds |
| event_driven | C6_llmtext | 20 | 568 | 981 | -3.58 | 0.014 | -3.23 | 0.0013 | 0.060 | 1.13 | text adds -> ns |
| event_driven | D2_gated_fusion | 5 | 569 | 996 | -1.34 | 1.000 | -1.22 | 0.2218 | 1.000 | 1.10 | ns -> ns |
| event_driven | D2_gated_fusion | 10 | 569 | 991 | -0.63 | 1.000 | -0.61 | 0.5426 | 1.000 | 1.05 | ns -> ns |
| event_driven | D2_gated_fusion | 20 | 568 | 981 | +2.93 | 0.105 | +2.80 | 0.0053 | 0.195 | 1.07 | ns -> ns |
| event_driven | D4_llmfused | 5 | 569 | 996 | +3.19 | 0.048 | +3.13 | 0.0019 | 0.076 | 1.03 | text HURTS -> ns |
| event_driven | D4_llmfused | 10 | 569 | 991 | +0.91 | 1.000 | +0.89 | 0.3755 | 1.000 | 1.04 | ns -> ns |
| event_driven | D4_llmfused | 20 | 568 | 981 | +3.48 | 0.019 | +3.21 | 0.0014 | 0.063 | 1.11 | text HURTS -> ns |
| long_form | B1_bow_ridge | 5 | 568 | 792 | +2.58 | 0.264 | +2.12 | 0.0342 | 0.890 | 1.22 | ns -> ns |
| long_form | B1_bow_ridge | 10 | 568 | 766 | +1.54 | 1.000 | +1.47 | 0.1421 | 1.000 | 1.06 | ns -> ns |
| long_form | B1_bow_ridge | 20 | 567 | 765 | +4.28 | 0.001 | +3.56 | 0.0004 | 0.021 | 1.23 | text HURTS -> text HURTS |
| long_form | B2_tfidf_ridge | 5 | 568 | 792 | +4.19 | 0.002 | +3.46 | 0.0006 | 0.029 | 1.22 | text HURTS -> text HURTS |
| long_form | B2_tfidf_ridge | 10 | 568 | 766 | +7.02 | 0.000 | +6.02 | 0.0000 | 0.000 | 1.18 | text HURTS -> text HURTS |
| long_form | B2_tfidf_ridge | 20 | 567 | 765 | +8.15 | 0.000 | +6.19 | 0.0000 | 0.000 | 1.35 | text HURTS -> text HURTS |
| long_form | B3_lm_linear | 5 | 568 | 792 | -1.81 | 1.000 | -1.18 | 0.2386 | 1.000 | 1.54 | ns -> ns |
| long_form | B3_lm_linear | 10 | 568 | 766 | -3.63 | 0.012 | -2.56 | 0.0108 | 0.345 | 1.44 | text adds -> ns |
| long_form | B3_lm_linear | 20 | 567 | 765 | +3.47 | 0.020 | +2.39 | 0.0172 | 0.516 | 1.49 | text HURTS -> ns |
| long_form | B4_lm_features | 5 | 568 | 792 | -1.41 | 1.000 | -1.08 | 0.2813 | 1.000 | 1.32 | ns -> ns |
| long_form | B4_lm_features | 10 | 568 | 766 | +4.10 | 0.002 | +2.99 | 0.0029 | 0.112 | 1.39 | text HURTS -> ns |
| long_form | B4_lm_features | 20 | 567 | 765 | +4.62 | 0.000 | +3.15 | 0.0017 | 0.074 | 1.51 | text HURTS -> ns |
| long_form | C1_bert_s1 | 5 | 568 | 792 | +1.07 | 1.000 | +0.83 | 0.4063 | 1.000 | 1.29 | ns -> ns |
| long_form | C1_bert_s1 | 10 | 568 | 766 | +3.26 | 0.039 | +2.25 | 0.0249 | 0.722 | 1.47 | text HURTS -> ns |
| long_form | C1_bert_s1 | 20 | 567 | 765 | +5.92 | 0.000 | +3.74 | 0.0002 | 0.011 | 1.63 | text HURTS -> text HURTS |
| long_form | C2_finbert_s1 | 5 | 568 | 792 | +1.89 | 1.000 | +1.42 | 0.1547 | 1.000 | 1.34 | ns -> ns |
| long_form | C2_finbert_s1 | 10 | 568 | 766 | +4.09 | 0.002 | +3.43 | 0.0007 | 0.032 | 1.21 | text HURTS -> text HURTS |
| long_form | C2_finbert_s1 | 20 | 567 | 765 | +0.17 | 1.000 | +0.10 | 0.9176 | 1.000 | 1.69 | ns -> ns |
| long_form | C2_finbert_s2 | 5 | 568 | 792 | +2.44 | 0.373 | +1.75 | 0.0805 | 1.000 | 1.40 | ns -> ns |
| long_form | C2_finbert_s2 | 10 | 568 | 766 | +4.23 | 0.001 | +3.05 | 0.0024 | 0.095 | 1.41 | text HURTS -> ns |
| long_form | C2_finbert_s2 | 20 | 567 | 765 | +0.80 | 1.000 | +0.65 | 0.5151 | 1.000 | 1.26 | ns -> ns |
| long_form | C2_finbert_s3 | 5 | 568 | 792 | +4.03 | 0.003 | +2.98 | 0.0030 | 0.114 | 1.36 | text HURTS -> ns |
| long_form | C2_finbert_s3 | 10 | 568 | 766 | +2.66 | 0.223 | +2.14 | 0.0324 | 0.876 | 1.26 | ns -> ns |
| long_form | C2_finbert_s3 | 20 | 567 | 765 | +1.23 | 1.000 | +1.01 | 0.3134 | 1.000 | 1.25 | ns -> ns |
| long_form | C2_finbert_s4 | 5 | 568 | 792 | +3.97 | 0.003 | +3.57 | 0.0004 | 0.020 | 1.12 | text HURTS -> text HURTS |
| long_form | C2_finbert_s4 | 10 | 568 | 766 | +4.32 | 0.001 | +4.08 | 0.0001 | 0.003 | 1.07 | text HURTS -> text HURTS |
| long_form | C2_finbert_s4 | 20 | 567 | 765 | +6.51 | 0.000 | +5.29 | 0.0000 | 0.000 | 1.26 | text HURTS -> text HURTS |
| long_form | C3_roberta_s1 | 5 | 568 | 792 | +3.84 | 0.005 | +2.42 | 0.0159 | 0.492 | 1.60 | text HURTS -> ns |
| long_form | C3_roberta_s1 | 10 | 568 | 766 | +4.14 | 0.002 | +2.62 | 0.0091 | 0.310 | 1.60 | text HURTS -> ns |
| long_form | C3_roberta_s1 | 20 | 567 | 765 | +0.44 | 1.000 | +0.28 | 0.7777 | 1.000 | 1.60 | ns -> ns |
| long_form | C4_longformer | 5 | 568 | 792 | +4.59 | 0.000 | +3.61 | 0.0003 | 0.018 | 1.28 | text HURTS -> text HURTS |
| long_form | C4_longformer | 10 | 568 | 766 | +9.10 | 0.000 | +6.47 | 0.0000 | 0.000 | 1.43 | text HURTS -> text HURTS |
| long_form | C4_longformer | 20 | 567 | 765 | +4.15 | 0.002 | +3.96 | 0.0001 | 0.005 | 1.08 | text HURTS -> text HURTS |
| long_form | C6_llmtext | 5 | 568 | 792 | +4.94 | 0.000 | +4.37 | 0.0000 | 0.001 | 1.14 | text HURTS -> text HURTS |
| long_form | C6_llmtext | 10 | 568 | 766 | +5.79 | 0.000 | +5.64 | 0.0000 | 0.000 | 1.04 | text HURTS -> text HURTS |
| long_form | C6_llmtext | 20 | 567 | 765 | -4.15 | 0.002 | -3.37 | 0.0008 | 0.038 | 1.26 | text adds -> text adds |
| long_form | D1_concat_mlp | 5 | 568 | 792 | +0.55 | 1.000 | +0.55 | 0.5810 | 1.000 | 1.00 | ns -> ns |
| long_form | D1_concat_mlp | 10 | 568 | 766 | +0.60 | 1.000 | +0.58 | 0.5648 | 1.000 | 1.06 | ns -> ns |
| long_form | D1_concat_mlp | 20 | 567 | 765 | -0.59 | 1.000 | -0.43 | 0.6683 | 1.000 | 1.40 | ns -> ns |
| long_form | D2_gated_fusion | 5 | 568 | 792 | +0.82 | 1.000 | +0.81 | 0.4198 | 1.000 | 1.02 | ns -> ns |
| long_form | D2_gated_fusion | 10 | 568 | 766 | +1.19 | 1.000 | +1.12 | 0.2644 | 1.000 | 1.07 | ns -> ns |
| long_form | D2_gated_fusion | 20 | 567 | 765 | +3.43 | 0.023 | +3.42 | 0.0007 | 0.032 | 1.03 | text HURTS -> text HURTS |
| long_form | D4_llmfused | 5 | 568 | 792 | +0.69 | 1.000 | +0.64 | 0.5196 | 1.000 | 1.08 | ns -> ns |
| long_form | D4_llmfused | 10 | 568 | 766 | -0.45 | 1.000 | -0.40 | 0.6918 | 1.000 | 1.14 | ns -> ns |
| long_form | D4_llmfused | 20 | 567 | 765 | -2.87 | 0.122 | -2.79 | 0.0054 | 0.196 | 1.06 | ns -> ns |

## (c) Pairwise vs A2 on squared error — day-clustered vs two-way (seed-ensemble basis of dm_pairwise_clustered)
| disc | h | challenger | n_firms | n_days | dm_day | Holm(day) | dm_2way | p_2way | Holm(2way) | SEx | verdict day -> 2way |
|---|---|---|---|---|---|---|---|---|---|---|---|
| combined | 5 | D3_gteqwen2 | 569 | 996 | +2.69 | 0.0072 | +2.48 | 0.0135 | 0.0135 | 1.09 | sig worse -> sig worse |
| combined | 5 | D3_qwen3 | 569 | 996 | +3.80 | 0.0003 | +3.44 | 0.0006 | 0.0013 | 1.11 | sig worse -> sig worse |
| combined | 5 | D3_e5mistral | 569 | 996 | +4.07 | 0.0002 | +3.74 | 0.0002 | 0.0006 | 1.09 | sig worse -> sig worse |
| combined | 5 | A4_egarch | 569 | 996 | +6.26 | 0.0000 | +5.11 | 0.0000 | 0.0000 | 1.23 | sig worse -> sig worse |
| combined | 5 | A3_garch | 569 | 996 | +6.08 | 0.0000 | +5.15 | 0.0000 | 0.0000 | 1.19 | sig worse -> sig worse |
| combined | 5 | D1_concat_mlp | 569 | 996 | +6.85 | 0.0000 | +5.97 | 0.0000 | 0.0000 | 1.15 | sig worse -> sig worse |
| combined | 5 | A5_arima | 569 | 996 | +6.17 | 0.0000 | +5.97 | 0.0000 | 0.0000 | 1.04 | sig worse -> sig worse |
| combined | 5 | D2_gated_fusion | 569 | 996 | +7.22 | 0.0000 | +6.19 | 0.0000 | 0.0000 | 1.17 | sig worse -> sig worse |
| combined | 5 | C5_qwen3 | 569 | 996 | +8.78 | 0.0000 | +6.27 | 0.0000 | 0.0000 | 1.41 | sig worse -> sig worse |
| combined | 5 | C5_gteqwen2 | 569 | 996 | +8.92 | 0.0000 | +6.34 | 0.0000 | 0.0000 | 1.41 | sig worse -> sig worse |
| combined | 5 | C2_finbert_s1 | 569 | 996 | +8.19 | 0.0000 | +6.53 | 0.0000 | 0.0000 | 1.26 | sig worse -> sig worse |
| combined | 5 | C5_e5mistral | 569 | 996 | +9.07 | 0.0000 | +6.53 | 0.0000 | 0.0000 | 1.40 | sig worse -> sig worse |
| combined | 5 | C3_roberta_s1 | 569 | 996 | +8.66 | 0.0000 | +6.79 | 0.0000 | 0.0000 | 1.28 | sig worse -> sig worse |
| combined | 5 | C4_longformer | 569 | 996 | +9.01 | 0.0000 | +6.81 | 0.0000 | 0.0000 | 1.33 | sig worse -> sig worse |
| combined | 5 | C2_finbert_s3 | 569 | 996 | +8.89 | 0.0000 | +6.89 | 0.0000 | 0.0000 | 1.30 | sig worse -> sig worse |
| combined | 5 | C2_finbert_s2 | 569 | 996 | +8.73 | 0.0000 | +6.95 | 0.0000 | 0.0000 | 1.26 | sig worse -> sig worse |
| combined | 5 | C1_bert_s1 | 569 | 996 | +9.19 | 0.0000 | +7.16 | 0.0000 | 0.0000 | 1.29 | sig worse -> sig worse |
| combined | 5 | B2_tfidf_ridge | 569 | 996 | +9.47 | 0.0000 | +7.35 | 0.0000 | 0.0000 | 1.29 | sig worse -> sig worse |
| combined | 5 | C2_finbert_s4 | 569 | 996 | +9.49 | 0.0000 | +7.36 | 0.0000 | 0.0000 | 1.30 | sig worse -> sig worse |
| combined | 5 | C1_bert_s2 | 569 | 996 | +9.56 | 0.0000 | +7.37 | 0.0000 | 0.0000 | 1.30 | sig worse -> sig worse |
| combined | 10 | D3_gteqwen2 | 569 | 991 | +0.99 | 0.3232 | +0.95 | 0.3440 | 0.3440 | 1.05 | ns -> ns |
| combined | 10 | D3_qwen3 | 569 | 991 | +1.90 | 0.1146 | +1.76 | 0.0788 | 0.1576 | 1.09 | ns -> ns |
| combined | 10 | D3_e5mistral | 569 | 991 | +2.14 | 0.0972 | +2.05 | 0.0411 | 0.1232 | 1.06 | ns -> ns |
| combined | 10 | A5_arima | 569 | 991 | +4.12 | 0.0002 | +4.11 | 0.0000 | 0.0002 | 1.01 | sig worse -> sig worse |
| combined | 10 | A3_garch | 569 | 991 | +4.78 | 0.0000 | +4.17 | 0.0000 | 0.0002 | 1.16 | sig worse -> sig worse |
| combined | 10 | D1_concat_mlp | 569 | 991 | +4.74 | 0.0000 | +4.22 | 0.0000 | 0.0002 | 1.13 | sig worse -> sig worse |
| combined | 10 | A4_egarch | 569 | 991 | +5.46 | 0.0000 | +4.50 | 0.0000 | 0.0001 | 1.23 | sig worse -> sig worse |
| combined | 10 | D2_gated_fusion | 569 | 991 | +5.34 | 0.0000 | +4.73 | 0.0000 | 0.0000 | 1.14 | sig worse -> sig worse |
| combined | 10 | C2_finbert_s1 | 569 | 991 | +5.97 | 0.0000 | +4.85 | 0.0000 | 0.0000 | 1.24 | sig worse -> sig worse |
| combined | 10 | C1_bert_s1 | 569 | 991 | +6.05 | 0.0000 | +4.85 | 0.0000 | 0.0000 | 1.26 | sig worse -> sig worse |
| combined | 10 | C2_finbert_s4 | 569 | 991 | +6.65 | 0.0000 | +5.46 | 0.0000 | 0.0000 | 1.23 | sig worse -> sig worse |
| combined | 10 | C4_longformer | 569 | 991 | +6.80 | 0.0000 | +5.51 | 0.0000 | 0.0000 | 1.25 | sig worse -> sig worse |
| combined | 10 | C2_finbert_s3 | 569 | 991 | +6.50 | 0.0000 | +5.56 | 0.0000 | 0.0000 | 1.18 | sig worse -> sig worse |
| combined | 10 | C5_qwen3 | 569 | 991 | +7.63 | 0.0000 | +5.68 | 0.0000 | 0.0000 | 1.36 | sig worse -> sig worse |
| combined | 10 | C1_bert_s2 | 569 | 991 | +6.85 | 0.0000 | +5.77 | 0.0000 | 0.0000 | 1.20 | sig worse -> sig worse |
| combined | 10 | C2_finbert_s2 | 569 | 991 | +6.89 | 0.0000 | +5.78 | 0.0000 | 0.0000 | 1.21 | sig worse -> sig worse |
| combined | 10 | C5_gteqwen2 | 569 | 991 | +7.79 | 0.0000 | +5.81 | 0.0000 | 0.0000 | 1.36 | sig worse -> sig worse |
| combined | 10 | C5_e5mistral | 569 | 991 | +7.70 | 0.0000 | +5.84 | 0.0000 | 0.0000 | 1.33 | sig worse -> sig worse |
| combined | 10 | C3_roberta_s1 | 569 | 991 | +7.54 | 0.0000 | +6.20 | 0.0000 | 0.0000 | 1.23 | sig worse -> sig worse |
| combined | 10 | B2_tfidf_ridge | 569 | 991 | +8.06 | 0.0000 | +6.58 | 0.0000 | 0.0000 | 1.24 | sig worse -> sig worse |
| combined | 20 | D3_gteqwen2 | 568 | 981 | +0.24 | 0.8131 | +0.23 | 0.8183 | 0.8183 | 1.05 | ns -> ns |
| combined | 20 | D3_qwen3 | 568 | 981 | +1.93 | 0.1078 | +1.80 | 0.0731 | 0.1463 | 1.10 | ns -> ns |
| combined | 20 | A5_arima | 568 | 981 | +2.27 | 0.0705 | +2.25 | 0.0250 | 0.0812 | 1.03 | ns -> ns |
| combined | 20 | D3_e5mistral | 568 | 981 | +2.42 | 0.0620 | +2.33 | 0.0203 | 0.0812 | 1.06 | ns -> ns |
| combined | 20 | A3_garch | 568 | 981 | +3.49 | 0.0025 | +3.03 | 0.0025 | 0.0126 | 1.17 | sig worse -> sig worse |
| combined | 20 | A4_egarch | 568 | 981 | +4.18 | 0.0002 | +3.52 | 0.0005 | 0.0028 | 1.21 | sig worse -> sig worse |
| combined | 20 | D1_concat_mlp | 568 | 981 | +3.92 | 0.0006 | +3.72 | 0.0002 | 0.0015 | 1.07 | sig worse -> sig worse |
| combined | 20 | C2_finbert_s4 | 568 | 981 | +4.98 | 0.0000 | +4.48 | 0.0000 | 0.0001 | 1.13 | sig worse -> sig worse |
| combined | 20 | C2_finbert_s1 | 568 | 981 | +5.25 | 0.0000 | +4.68 | 0.0000 | 0.0000 | 1.14 | sig worse -> sig worse |
| combined | 20 | C4_longformer | 568 | 981 | +5.36 | 0.0000 | +4.69 | 0.0000 | 0.0000 | 1.17 | sig worse -> sig worse |
| combined | 20 | C2_finbert_s3 | 568 | 981 | +5.35 | 0.0000 | +4.75 | 0.0000 | 0.0000 | 1.15 | sig worse -> sig worse |
| combined | 20 | C2_finbert_s2 | 568 | 981 | +5.50 | 0.0000 | +4.93 | 0.0000 | 0.0000 | 1.14 | sig worse -> sig worse |
| combined | 20 | C1_bert_s2 | 568 | 981 | +5.67 | 0.0000 | +5.00 | 0.0000 | 0.0000 | 1.16 | sig worse -> sig worse |
| combined | 20 | C3_roberta_s1 | 568 | 981 | +5.88 | 0.0000 | +5.14 | 0.0000 | 0.0000 | 1.17 | sig worse -> sig worse |
| combined | 20 | C5_qwen3 | 568 | 981 | +6.38 | 0.0000 | +5.25 | 0.0000 | 0.0000 | 1.24 | sig worse -> sig worse |
| combined | 20 | C1_bert_s1 | 568 | 981 | +6.12 | 0.0000 | +5.29 | 0.0000 | 0.0000 | 1.18 | sig worse -> sig worse |
| combined | 20 | D2_gated_fusion | 568 | 981 | +5.65 | 0.0000 | +5.33 | 0.0000 | 0.0000 | 1.08 | sig worse -> sig worse |
| combined | 20 | C5_gteqwen2 | 568 | 981 | +6.61 | 0.0000 | +5.37 | 0.0000 | 0.0000 | 1.26 | sig worse -> sig worse |
| combined | 20 | C5_e5mistral | 568 | 981 | +6.52 | 0.0000 | +5.41 | 0.0000 | 0.0000 | 1.23 | sig worse -> sig worse |
| combined | 20 | B2_tfidf_ridge | 568 | 981 | +6.64 | 0.0000 | +5.87 | 0.0000 | 0.0000 | 1.15 | sig worse -> sig worse |
| event_driven | 5 | D3_gteqwen2 | 569 | 996 | +2.24 | 0.0252 | +1.99 | 0.0467 | 0.0467 | 1.13 | sig worse -> sig worse |
| event_driven | 5 | D3_qwen3 | 569 | 996 | +3.73 | 0.0004 | +3.44 | 0.0006 | 0.0013 | 1.09 | sig worse -> sig worse |
| event_driven | 5 | D3_e5mistral | 569 | 996 | +3.92 | 0.0003 | +3.59 | 0.0004 | 0.0011 | 1.10 | sig worse -> sig worse |
| event_driven | 5 | A3_garch | 569 | 996 | +4.33 | 0.0001 | +4.08 | 0.0001 | 0.0002 | 1.07 | sig worse -> sig worse |
| event_driven | 5 | A4_egarch | 569 | 996 | +5.75 | 0.0000 | +4.65 | 0.0000 | 0.0000 | 1.24 | sig worse -> sig worse |
| event_driven | 5 | C2_finbert_s1 | 569 | 996 | +7.18 | 0.0000 | +5.29 | 0.0000 | 0.0000 | 1.36 | sig worse -> sig worse |
| event_driven | 5 | D2_gated_fusion | 569 | 996 | +6.47 | 0.0000 | +5.40 | 0.0000 | 0.0000 | 1.20 | sig worse -> sig worse |
| event_driven | 5 | A5_arima | 569 | 996 | +5.48 | 0.0000 | +5.43 | 0.0000 | 0.0000 | 1.01 | sig worse -> sig worse |
| event_driven | 5 | C2_finbert_s3 | 569 | 996 | +7.43 | 0.0000 | +5.53 | 0.0000 | 0.0000 | 1.35 | sig worse -> sig worse |
| event_driven | 5 | C3_roberta_s1 | 569 | 996 | +7.47 | 0.0000 | +5.60 | 0.0000 | 0.0000 | 1.34 | sig worse -> sig worse |
| event_driven | 5 | C5_e5mistral | 569 | 996 | +8.38 | 0.0000 | +5.65 | 0.0000 | 0.0000 | 1.49 | sig worse -> sig worse |
| event_driven | 5 | C5_qwen3 | 569 | 996 | +9.20 | 0.0000 | +5.87 | 0.0000 | 0.0000 | 1.58 | sig worse -> sig worse |
| event_driven | 5 | C4_longformer | 569 | 996 | +7.87 | 0.0000 | +5.97 | 0.0000 | 0.0000 | 1.32 | sig worse -> sig worse |
| event_driven | 5 | D1_concat_mlp | 569 | 996 | +6.98 | 0.0000 | +6.06 | 0.0000 | 0.0000 | 1.16 | sig worse -> sig worse |
| event_driven | 5 | C5_gteqwen2 | 569 | 996 | +9.57 | 0.0000 | +6.09 | 0.0000 | 0.0000 | 1.58 | sig worse -> sig worse |
| event_driven | 5 | C1_bert_s1 | 569 | 996 | +8.13 | 0.0000 | +6.11 | 0.0000 | 0.0000 | 1.34 | sig worse -> sig worse |
| event_driven | 5 | C2_finbert_s4 | 569 | 996 | +8.08 | 0.0000 | +6.16 | 0.0000 | 0.0000 | 1.32 | sig worse -> sig worse |
| event_driven | 5 | C1_bert_s2 | 569 | 996 | +8.71 | 0.0000 | +6.52 | 0.0000 | 0.0000 | 1.34 | sig worse -> sig worse |
| event_driven | 5 | C2_finbert_s2 | 569 | 996 | +8.54 | 0.0000 | +6.58 | 0.0000 | 0.0000 | 1.30 | sig worse -> sig worse |
| event_driven | 5 | B2_tfidf_ridge | 569 | 996 | +9.54 | 0.0000 | +7.08 | 0.0000 | 0.0000 | 1.35 | sig worse -> sig worse |
| event_driven | 10 | D3_gteqwen2 | 569 | 991 | +2.11 | 0.0360 | +2.01 | 0.0447 | 0.0531 | 1.06 | sig worse -> ns |
| event_driven | 10 | D3_e5mistral | 569 | 991 | +2.51 | 0.0360 | +2.37 | 0.0179 | 0.0531 | 1.07 | sig worse -> ns |
| event_driven | 10 | D3_qwen3 | 569 | 991 | +2.52 | 0.0360 | +2.38 | 0.0177 | 0.0531 | 1.07 | sig worse -> ns |
| event_driven | 10 | A3_garch | 569 | 991 | +3.37 | 0.0031 | +3.23 | 0.0013 | 0.0052 | 1.05 | sig worse -> sig worse |
| event_driven | 10 | A5_arima | 569 | 991 | +3.75 | 0.0009 | +3.86 | 0.0001 | 0.0006 | 0.98 | sig worse -> sig worse |
| event_driven | 10 | A4_egarch | 569 | 991 | +5.08 | 0.0000 | +4.23 | 0.0000 | 0.0002 | 1.21 | sig worse -> sig worse |
| event_driven | 10 | C2_finbert_s3 | 569 | 991 | +6.00 | 0.0000 | +4.86 | 0.0000 | 0.0000 | 1.25 | sig worse -> sig worse |
| event_driven | 10 | C2_finbert_s4 | 569 | 991 | +6.29 | 0.0000 | +5.12 | 0.0000 | 0.0000 | 1.24 | sig worse -> sig worse |
| event_driven | 10 | D2_gated_fusion | 569 | 991 | +5.73 | 0.0000 | +5.13 | 0.0000 | 0.0000 | 1.13 | sig worse -> sig worse |
| event_driven | 10 | C5_e5mistral | 569 | 991 | +7.44 | 0.0000 | +5.19 | 0.0000 | 0.0000 | 1.45 | sig worse -> sig worse |
| event_driven | 10 | C4_longformer | 569 | 991 | +6.51 | 0.0000 | +5.30 | 0.0000 | 0.0000 | 1.24 | sig worse -> sig worse |
| event_driven | 10 | C1_bert_s1 | 569 | 991 | +6.83 | 0.0000 | +5.33 | 0.0000 | 0.0000 | 1.29 | sig worse -> sig worse |
| event_driven | 10 | C5_qwen3 | 569 | 991 | +8.66 | 0.0000 | +5.56 | 0.0000 | 0.0000 | 1.57 | sig worse -> sig worse |
| event_driven | 10 | D1_concat_mlp | 569 | 991 | +6.43 | 0.0000 | +5.63 | 0.0000 | 0.0000 | 1.15 | sig worse -> sig worse |
| event_driven | 10 | C2_finbert_s1 | 569 | 991 | +7.07 | 0.0000 | +5.66 | 0.0000 | 0.0000 | 1.26 | sig worse -> sig worse |
| event_driven | 10 | C3_roberta_s1 | 569 | 991 | +7.11 | 0.0000 | +5.68 | 0.0000 | 0.0000 | 1.26 | sig worse -> sig worse |
| event_driven | 10 | C2_finbert_s2 | 569 | 991 | +7.29 | 0.0000 | +5.73 | 0.0000 | 0.0000 | 1.28 | sig worse -> sig worse |
| event_driven | 10 | C5_gteqwen2 | 569 | 991 | +9.02 | 0.0000 | +5.81 | 0.0000 | 0.0000 | 1.57 | sig worse -> sig worse |
| event_driven | 10 | C1_bert_s2 | 569 | 991 | +7.31 | 0.0000 | +5.84 | 0.0000 | 0.0000 | 1.27 | sig worse -> sig worse |
| event_driven | 10 | B2_tfidf_ridge | 569 | 991 | +8.18 | 0.0000 | +6.39 | 0.0000 | 0.0000 | 1.29 | sig worse -> sig worse |
| event_driven | 20 | D3_gteqwen2 | 568 | 981 | +1.16 | 0.2447 | +1.12 | 0.2643 | 0.2643 | 1.06 | ns -> ns |
| event_driven | 20 | D3_qwen3 | 568 | 981 | +1.80 | 0.2035 | +1.75 | 0.0803 | 0.2219 | 1.05 | ns -> ns |
| event_driven | 20 | D3_e5mistral | 568 | 981 | +1.83 | 0.2035 | +1.79 | 0.0740 | 0.2219 | 1.04 | ns -> ns |
| event_driven | 20 | A5_arima | 568 | 981 | +2.20 | 0.1119 | +2.21 | 0.0277 | 0.1108 | 1.02 | ns -> ns |
| event_driven | 20 | A3_garch | 568 | 981 | +2.58 | 0.0502 | +2.38 | 0.0177 | 0.0885 | 1.11 | ns -> ns |
| event_driven | 20 | A4_egarch | 568 | 981 | +3.96 | 0.0005 | +3.37 | 0.0008 | 0.0049 | 1.20 | sig worse -> sig worse |
| event_driven | 20 | D2_gated_fusion | 568 | 981 | +4.30 | 0.0001 | +4.06 | 0.0001 | 0.0004 | 1.08 | sig worse -> sig worse |
| event_driven | 20 | C2_finbert_s2 | 568 | 981 | +5.14 | 0.0000 | +4.50 | 0.0000 | 0.0001 | 1.17 | sig worse -> sig worse |
| event_driven | 20 | C2_finbert_s1 | 568 | 981 | +5.34 | 0.0000 | +4.52 | 0.0000 | 0.0001 | 1.21 | sig worse -> sig worse |
| event_driven | 20 | C1_bert_s1 | 568 | 981 | +5.28 | 0.0000 | +4.58 | 0.0000 | 0.0001 | 1.17 | sig worse -> sig worse |
| event_driven | 20 | C2_finbert_s3 | 568 | 981 | +5.04 | 0.0000 | +4.58 | 0.0000 | 0.0001 | 1.12 | sig worse -> sig worse |
| event_driven | 20 | D1_concat_mlp | 568 | 981 | +5.07 | 0.0000 | +4.72 | 0.0000 | 0.0000 | 1.09 | sig worse -> sig worse |
| event_driven | 20 | C4_longformer | 568 | 981 | +5.73 | 0.0000 | +4.95 | 0.0000 | 0.0000 | 1.18 | sig worse -> sig worse |
| event_driven | 20 | C1_bert_s2 | 568 | 981 | +5.82 | 0.0000 | +5.05 | 0.0000 | 0.0000 | 1.18 | sig worse -> sig worse |
| event_driven | 20 | C5_e5mistral | 568 | 981 | +6.62 | 0.0000 | +5.06 | 0.0000 | 0.0000 | 1.33 | sig worse -> sig worse |
| event_driven | 20 | C3_roberta_s1 | 568 | 981 | +5.84 | 0.0000 | +5.07 | 0.0000 | 0.0000 | 1.17 | sig worse -> sig worse |
| event_driven | 20 | C2_finbert_s4 | 568 | 981 | +6.06 | 0.0000 | +5.15 | 0.0000 | 0.0000 | 1.20 | sig worse -> sig worse |
| event_driven | 20 | C5_qwen3 | 568 | 981 | +7.70 | 0.0000 | +5.42 | 0.0000 | 0.0000 | 1.45 | sig worse -> sig worse |
| event_driven | 20 | C5_gteqwen2 | 568 | 981 | +8.77 | 0.0000 | +5.94 | 0.0000 | 0.0000 | 1.51 | sig worse -> sig worse |
| event_driven | 20 | B2_tfidf_ridge | 568 | 981 | +7.01 | 0.0000 | +5.96 | 0.0000 | 0.0000 | 1.20 | sig worse -> sig worse |
| long_form | 5 | D1_concat_mlp | 568 | 809 | +1.61 | 0.1141 | +1.31 | 0.1898 | 0.2016 | 1.23 | ns -> ns |
| long_form | 5 | D3_gteqwen2 | 568 | 809 | +1.91 | 0.1141 | +1.64 | 0.1008 | 0.2016 | 1.17 | ns -> ns |
| long_form | 5 | A4_egarch | 568 | 809 | +2.74 | 0.0188 | +2.73 | 0.0066 | 0.0242 | 1.01 | sig worse -> sig worse |
| long_form | 5 | D3_e5mistral | 568 | 809 | +3.19 | 0.0060 | +2.76 | 0.0060 | 0.0242 | 1.16 | sig worse -> sig worse |
| long_form | 5 | D2_gated_fusion | 568 | 809 | +3.53 | 0.0022 | +2.90 | 0.0039 | 0.0197 | 1.23 | sig worse -> sig worse |
| long_form | 5 | D3_qwen3 | 568 | 809 | +3.81 | 0.0009 | +3.20 | 0.0014 | 0.0086 | 1.20 | sig worse -> sig worse |
| long_form | 5 | A3_garch | 568 | 809 | +5.22 | 0.0000 | +3.71 | 0.0002 | 0.0016 | 1.41 | sig worse -> sig worse |
| long_form | 5 | C5_e5mistral | 568 | 809 | +5.24 | 0.0000 | +4.16 | 0.0000 | 0.0003 | 1.27 | sig worse -> sig worse |
| long_form | 5 | C5_qwen3 | 568 | 809 | +5.92 | 0.0000 | +4.51 | 0.0000 | 0.0001 | 1.32 | sig worse -> sig worse |
| long_form | 5 | C4_longformer | 568 | 809 | +6.06 | 0.0000 | +5.16 | 0.0000 | 0.0000 | 1.18 | sig worse -> sig worse |
| long_form | 5 | C2_finbert_s1 | 568 | 809 | +6.68 | 0.0000 | +5.24 | 0.0000 | 0.0000 | 1.28 | sig worse -> sig worse |
| long_form | 5 | C5_gteqwen2 | 568 | 809 | +6.71 | 0.0000 | +5.24 | 0.0000 | 0.0000 | 1.29 | sig worse -> sig worse |
| long_form | 5 | A5_arima | 568 | 809 | +5.84 | 0.0000 | +5.45 | 0.0000 | 0.0000 | 1.08 | sig worse -> sig worse |
| long_form | 5 | C2_finbert_s3 | 568 | 809 | +6.68 | 0.0000 | +5.62 | 0.0000 | 0.0000 | 1.19 | sig worse -> sig worse |
| long_form | 5 | B2_tfidf_ridge | 568 | 809 | +6.35 | 0.0000 | +5.74 | 0.0000 | 0.0000 | 1.11 | sig worse -> sig worse |
| long_form | 5 | C1_bert_s2 | 568 | 809 | +7.21 | 0.0000 | +5.92 | 0.0000 | 0.0000 | 1.22 | sig worse -> sig worse |
| long_form | 5 | C2_finbert_s4 | 568 | 809 | +7.07 | 0.0000 | +5.99 | 0.0000 | 0.0000 | 1.19 | sig worse -> sig worse |
| long_form | 5 | C1_bert_s1 | 568 | 809 | +7.64 | 0.0000 | +6.16 | 0.0000 | 0.0000 | 1.25 | sig worse -> sig worse |
| long_form | 5 | C2_finbert_s2 | 568 | 809 | +7.63 | 0.0000 | +6.18 | 0.0000 | 0.0000 | 1.24 | sig worse -> sig worse |
| long_form | 5 | C3_roberta_s1 | 568 | 809 | +7.68 | 0.0000 | +6.23 | 0.0000 | 0.0000 | 1.24 | sig worse -> sig worse |
| long_form | 10 | D3_gteqwen2 | 568 | 803 | +0.77 | 0.4388 | +0.69 | 0.4879 | 0.4879 | 1.13 | ns -> ns |
| long_form | 10 | D2_gated_fusion | 568 | 803 | +1.86 | 0.1521 | +1.61 | 0.1070 | 0.2150 | 1.17 | ns -> ns |
| long_form | 10 | D3_e5mistral | 568 | 803 | +1.96 | 0.1521 | +1.80 | 0.0717 | 0.2150 | 1.10 | ns -> ns |
| long_form | 10 | D1_concat_mlp | 568 | 803 | +2.62 | 0.0535 | +2.26 | 0.0241 | 0.1085 | 1.17 | ns -> ns |
| long_form | 10 | D3_qwen3 | 568 | 803 | +2.62 | 0.0535 | +2.30 | 0.0217 | 0.1085 | 1.15 | ns -> ns |
| long_form | 10 | A4_egarch | 568 | 803 | +2.39 | 0.0692 | +2.42 | 0.0160 | 0.0960 | 1.00 | ns -> ns |
| long_form | 10 | C1_bert_s2 | 568 | 803 | +4.42 | 0.0001 | +3.81 | 0.0002 | 0.0012 | 1.17 | sig worse -> sig worse |
| long_form | 10 | A3_garch | 568 | 803 | +4.73 | 0.0000 | +3.84 | 0.0001 | 0.0012 | 1.25 | sig worse -> sig worse |
| long_form | 10 | C4_longformer | 568 | 803 | +4.56 | 0.0001 | +3.85 | 0.0001 | 0.0012 | 1.20 | sig worse -> sig worse |
| long_form | 10 | A5_arima | 568 | 803 | +4.21 | 0.0002 | +3.99 | 0.0001 | 0.0007 | 1.07 | sig worse -> sig worse |
| long_form | 10 | C1_bert_s1 | 568 | 803 | +4.66 | 0.0000 | +4.06 | 0.0001 | 0.0006 | 1.16 | sig worse -> sig worse |
| long_form | 10 | C2_finbert_s4 | 568 | 803 | +4.89 | 0.0000 | +4.24 | 0.0000 | 0.0003 | 1.17 | sig worse -> sig worse |
| long_form | 10 | C2_finbert_s2 | 568 | 803 | +5.11 | 0.0000 | +4.47 | 0.0000 | 0.0001 | 1.16 | sig worse -> sig worse |
| long_form | 10 | C3_roberta_s1 | 568 | 803 | +5.61 | 0.0000 | +4.60 | 0.0000 | 0.0001 | 1.24 | sig worse -> sig worse |
| long_form | 10 | B2_tfidf_ridge | 568 | 803 | +4.92 | 0.0000 | +4.62 | 0.0000 | 0.0001 | 1.08 | sig worse -> sig worse |
| long_form | 10 | C2_finbert_s1 | 568 | 803 | +5.53 | 0.0000 | +4.74 | 0.0000 | 0.0000 | 1.18 | sig worse -> sig worse |
| long_form | 10 | C2_finbert_s3 | 568 | 803 | +5.50 | 0.0000 | +4.82 | 0.0000 | 0.0000 | 1.15 | sig worse -> sig worse |
| long_form | 10 | C5_e5mistral | 568 | 803 | +6.96 | 0.0000 | +5.07 | 0.0000 | 0.0000 | 1.39 | sig worse -> sig worse |
| long_form | 10 | C5_qwen3 | 568 | 803 | +7.08 | 0.0000 | +5.11 | 0.0000 | 0.0000 | 1.40 | sig worse -> sig worse |
| long_form | 10 | C5_gteqwen2 | 568 | 803 | +7.31 | 0.0000 | +5.53 | 0.0000 | 0.0000 | 1.34 | sig worse -> sig worse |
| long_form | 20 | D3_gteqwen2 | 567 | 794 | +0.24 | 1.0000 | +0.22 | 0.8243 | 1.0000 | 1.10 | ns -> ns |
| long_form | 20 | D3_e5mistral | 567 | 794 | +0.63 | 1.0000 | +0.60 | 0.5520 | 1.0000 | 1.08 | ns -> ns |
| long_form | 20 | D3_qwen3 | 567 | 794 | +1.26 | 0.6201 | +1.15 | 0.2510 | 0.7530 | 1.13 | ns -> ns |
| long_form | 20 | D2_gated_fusion | 567 | 794 | +2.03 | 0.1725 | +1.91 | 0.0563 | 0.2252 | 1.09 | ns -> ns |
| long_form | 20 | D1_concat_mlp | 567 | 794 | +2.44 | 0.0755 | +2.25 | 0.0252 | 0.1258 | 1.11 | ns -> ns |
| long_form | 20 | C4_longformer | 567 | 794 | +2.92 | 0.0250 | +2.68 | 0.0076 | 0.0457 | 1.12 | sig worse -> sig worse |
| long_form | 20 | C2_finbert_s4 | 567 | 794 | +2.86 | 0.0261 | +2.77 | 0.0057 | 0.0402 | 1.06 | sig worse -> sig worse |
| long_form | 20 | C2_finbert_s2 | 567 | 794 | +3.12 | 0.0171 | +2.89 | 0.0040 | 0.0323 | 1.11 | sig worse -> sig worse |
| long_form | 20 | C2_finbert_s1 | 567 | 794 | +3.32 | 0.0095 | +3.00 | 0.0029 | 0.0258 | 1.14 | sig worse -> sig worse |
| long_form | 20 | A5_arima | 567 | 794 | +3.10 | 0.0171 | +3.03 | 0.0026 | 0.0258 | 1.05 | sig worse -> sig worse |
| long_form | 20 | C1_bert_s2 | 567 | 794 | +3.46 | 0.0068 | +3.18 | 0.0015 | 0.0168 | 1.11 | sig worse -> sig worse |
| long_form | 20 | C2_finbert_s3 | 567 | 794 | +3.41 | 0.0076 | +3.26 | 0.0012 | 0.0141 | 1.07 | sig worse -> sig worse |
| long_form | 20 | C1_bert_s1 | 567 | 794 | +3.66 | 0.0041 | +3.32 | 0.0010 | 0.0127 | 1.13 | sig worse -> sig worse |
| long_form | 20 | C3_roberta_s1 | 567 | 794 | +3.87 | 0.0019 | +3.47 | 0.0006 | 0.0077 | 1.14 | sig worse -> sig worse |
| long_form | 20 | A4_egarch | 567 | 794 | +3.61 | 0.0044 | +3.54 | 0.0004 | 0.0067 | 1.05 | sig worse -> sig worse |
| long_form | 20 | B2_tfidf_ridge | 567 | 794 | +3.62 | 0.0044 | +3.55 | 0.0004 | 0.0067 | 1.05 | sig worse -> sig worse |
| long_form | 20 | A3_garch | 567 | 794 | +4.27 | 0.0004 | +3.88 | 0.0001 | 0.0020 | 1.13 | sig worse -> sig worse |
| long_form | 20 | C5_qwen3 | 567 | 794 | +6.79 | 0.0000 | +5.31 | 0.0000 | 0.0000 | 1.31 | sig worse -> sig worse |
| long_form | 20 | C5_e5mistral | 567 | 794 | +6.76 | 0.0000 | +5.37 | 0.0000 | 0.0000 | 1.29 | sig worse -> sig worse |
| long_form | 20 | C5_gteqwen2 | 567 | 794 | +6.67 | 0.0000 | +5.47 | 0.0000 | 0.0000 | 1.25 | sig worse -> sig worse |

## Bottom line

- Adding the firm clustering dimension changes **21** verdict(s) across the 318 committed inference cells (a: 5, b: 13, c: 3).
- (a) M1 grid: genuine cells 29/69 (day) -> **24/69** (two-way).
- (b) firm-identity control: survivors 8/69 (day) -> **5/69** (two-way); text-HURTS 29 -> 19.
- (c) the '0/180 challengers beat A2' headline: **0/180** significantly better under two-way clustering (0/180 even at raw p<.05); significantly worse 155 -> 152.
- Risk direction is as pre-registered in REVIEW_ROUND2_GAPS.md P1-4: two-way SEs are (weakly) wider, so any movement is TOWARD the null — the near-null headline cannot be an artifact of ignoring within-firm dependence; the flipped cells listed above must be quoted with the two-way (weaker) verdict in the paper.