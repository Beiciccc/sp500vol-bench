# P0-3 — Within-date permutation placebo + date-mean-text control (cross-sectional vs regime-timing decomposition)

## RESTATED vs ORIGINAL

| | ORIGINAL (forecast_combination_grid.csv) | RESTATED (this table) |
|---|---|---|
| Placebo design | whole-sample permutation of the text forecast — destroys which-firm AND when information simultaneously | (a) WITHIN-DATE permutation (keeps all date/regime info, kills which-firm); (b) DATE-MEAN-TEXT combiner (pure when-signal, zero cross-sectional content) |
| Inference | observation-order HAC DM (reviewer-verified ~2x inflated) | day-clustered DM: daily-mean loss differentials, HAC lag=h-1 over DAYS |
| What it can say | 'the increment is not a procedural artifact' | of each genuine increment, what fraction is cross-sectional (which-firm) vs regime-timing (when): median regime-timing share = 0.00, median cross-sectional share = 1.00, median within-date-placebo survival = 0.00 |

**Cells:** 69 total, 38 genuine (per the original grid's placebo-confirmed flag). SANITY: real rel% reproduces the grid in 69/69 cells.

**Verdicts over the 38 genuine cells:** cross-sectional = 33, mixed = 1, regime-timing = 4.

Reading guide: `wd_placebo_*` = mean over seeds 1000-1004 of the within-date permutation (a); if the increment DIES here (rel%~0, |DM|<2) the signal is cross-sectional. `datemean_*` = the day-mean-text combiner (b); the fraction `frac_regime_timing = datemean_rel / real_rel` (clipped to [0,1]) is the share of the increment a pure when-signal already delivers. All DM stats are day-clustered (negative = text-augmented combiner better).


## long_form
| model | h | n_days | real rel% | real DM(cl) | wd-placebo rel% | wd-placebo DM(cl) | date-mean rel% | date-mean DM(cl) | frac regime | frac cross-sec | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_bow_ridge | 5 | 809 | +1.65 | -3.83 | -0.07 | +1.53 | +0.18 | -2.67 | 0.11 | 0.89 | cross-sectional |
| B1_bow_ridge | 10 | 803 | +1.44 | -4.15 | -0.78 | +3.24 | -3.62 | +5.44 | 0.00 | 1.00 | cross-sectional |
| B1_bow_ridge | 20 | 794 | +2.99 | -5.45 | -1.61 | +5.73 | -8.16 | +7.98 | 0.00 | 1.00 | cross-sectional |
| B2_tfidf_ridge | 5 | 809 | +3.33 | -5.39 | -0.03 | -0.28 | -0.49 | +4.34 | 0.00 | 1.00 | cross-sectional |
| B2_tfidf_ridge | 10 | 803 | +3.48 | -8.89 | -0.64 | +5.75 | -6.24 | +8.22 | 0.00 | 1.00 | cross-sectional |
| B2_tfidf_ridge | 20 | 794 | +5.92 | -9.04 | -2.17 | +7.14 | -21.61 | +9.40 | 0.00 | 1.00 | n/a (not genuine) |
| B3_lm_linear | 5 | 809 | +0.49 | -2.58 | -0.13 | -2.32 | -0.35 | -0.63 | 0.00 | 1.00 | n/a (not genuine) |
| B3_lm_linear | 10 | 803 | +1.79 | -4.62 | +0.07 | -2.10 | +0.91 | -3.79 | 0.51 | 0.49 | regime-timing |
| B3_lm_linear | 20 | 794 | +3.48 | -4.69 | -0.30 | +3.25 | -2.08 | +6.18 | 0.00 | 1.00 | cross-sectional |
| B4_lm_features | 5 | 809 | +0.11 | +1.02 | -0.14 | -0.64 | -1.42 | +1.36 | 0.00 | 1.00 | n/a (not genuine) |
| B4_lm_features | 10 | 803 | -0.92 | +3.32 | -1.81 | +2.78 | -12.34 | +6.61 |  |  | n/a (not genuine) |
| B4_lm_features | 20 | 794 | -1.92 | +3.38 | -5.58 | +3.66 | -43.61 | +10.00 |  |  | n/a (not genuine) |
| C1_bert_s1 | 5 | 809 | +0.12 | -2.03 | -0.08 | -1.03 | -0.31 | -0.77 | 0.00 | 1.00 | n/a (not genuine) |
| C1_bert_s1 | 10 | 803 | -1.28 | +2.35 | -0.29 | +1.24 | -4.68 | +4.20 |  |  | n/a (not genuine) |
| C1_bert_s1 | 20 | 794 | +2.95 | -7.42 | -1.97 | +5.98 | -12.60 | +7.57 | 0.00 | 1.00 | cross-sectional |
| C2_finbert_s1 | 5 | 809 | +0.56 | -0.84 | -0.09 | +0.10 | -0.91 | +1.26 | 0.00 | 1.00 | cross-sectional |
| C2_finbert_s1 | 10 | 803 | +4.56 | -6.46 | +0.01 | -2.59 | +0.04 | -4.80 | 0.01 | 0.99 | cross-sectional |
| C2_finbert_s1 | 20 | 794 | -0.08 | +0.88 | -0.08 | -0.98 | -1.83 | +3.13 |  |  | n/a (not genuine) |
| C2_finbert_s2 | 5 | 809 | +1.73 | -4.30 | -0.03 | -0.81 | +0.07 | -1.66 | 0.04 | 0.96 | cross-sectional |
| C2_finbert_s2 | 10 | 803 | +0.24 | -6.58 | -0.56 | +4.11 | -3.14 | +6.33 | 0.00 | 1.00 | cross-sectional |
| C2_finbert_s2 | 20 | 794 | -5.19 | -0.06 | -4.47 | -0.47 | -17.09 | +4.22 |  |  | n/a (not genuine) |
| C2_finbert_s3 | 5 | 809 | +2.39 | -5.43 | -0.03 | +0.78 | -0.10 | +2.98 | 0.00 | 1.00 | cross-sectional |
| C2_finbert_s3 | 10 | 803 | +3.03 | -5.13 | -0.13 | +3.55 | -0.91 | +4.32 | 0.00 | 1.00 | cross-sectional |
| C2_finbert_s3 | 20 | 794 | -0.76 | -1.77 | +1.66 | +0.88 | +6.00 | +2.76 |  |  | n/a (not genuine) |
| C2_finbert_s4 | 5 | 809 | -1.09 | +3.36 | -1.40 | +2.06 | -3.93 | +4.22 |  |  | n/a (not genuine) |
| C2_finbert_s4 | 10 | 803 | +2.28 | -6.02 | -0.49 | +4.50 | -1.73 | +5.34 | 0.00 | 1.00 | cross-sectional |
| C2_finbert_s4 | 20 | 794 | +3.45 | -8.82 | -2.19 | +6.57 | -15.34 | +8.09 | 0.00 | 1.00 | cross-sectional |
| C3_roberta_s1 | 5 | 809 | +0.67 | -5.37 | -0.61 | +3.60 | -2.17 | +5.33 | 0.00 | 1.00 | cross-sectional |
| C3_roberta_s1 | 10 | 803 | +1.98 | -5.18 | -0.01 | +3.32 | +0.09 | -3.29 | 0.04 | 0.96 | cross-sectional |
| C3_roberta_s1 | 20 | 794 | +0.24 | -1.07 | -0.09 | -0.65 | -1.35 | +2.00 | 0.00 | 1.00 | cross-sectional |
| C4_longformer | 5 | 809 | +1.53 | -5.99 | -0.13 | +3.87 | -0.76 | +5.28 | 0.00 | 1.00 | cross-sectional |
| C4_longformer | 10 | 803 | -2.61 | +10.55 | -3.59 | +7.35 | -12.40 | +8.70 |  |  | n/a (not genuine) |
| C4_longformer | 20 | 794 | -9.34 | +3.46 | -7.58 | +3.41 | -19.04 | +5.67 |  |  | n/a (not genuine) |
| C6_llmtext | 5 | 809 | +1.79 | -6.31 | -0.03 | +1.10 | -0.40 | +5.36 | 0.00 | 1.00 | cross-sectional |
| C6_llmtext | 10 | 803 | +2.25 | -7.92 | -0.36 | +6.46 | -2.67 | +7.25 | 0.00 | 1.00 | cross-sectional |
| C6_llmtext | 20 | 794 | +0.27 | -3.23 | -0.31 | +4.29 | -2.79 | +5.80 | 0.00 | 1.00 | n/a (not genuine) |
| D1_concat_mlp | 5 | 809 | -0.01 | +0.10 | +0.17 | +1.59 | +0.72 | +1.57 |  |  | n/a (not genuine) |
| D1_concat_mlp | 10 | 803 | +0.18 | -0.67 | +0.11 | +1.93 | +0.49 | +1.58 | 1.00 | 0.00 | n/a (not genuine) |
| D1_concat_mlp | 20 | 794 | -0.08 | -0.63 | -0.23 | +0.23 | -0.90 | +1.28 |  |  | n/a (not genuine) |
| D2_gated_fusion | 5 | 809 | -0.12 | -0.29 | +0.15 | +1.54 | +0.83 | +1.52 |  |  | n/a (not genuine) |
| D2_gated_fusion | 10 | 803 | +0.13 | +0.96 | +0.08 | +1.44 | +0.34 | +1.00 | 1.00 | 0.00 | n/a (not genuine) |
| D2_gated_fusion | 20 | 794 | -6.00 | +4.27 | -1.12 | +1.79 | -2.32 | +2.56 |  |  | n/a (not genuine) |
| D4_llmfused | 5 | 809 | +0.15 | -0.75 | +0.05 | +1.99 | +1.13 | +2.68 | 1.00 | 0.00 | regime-timing |
| D4_llmfused | 10 | 803 | -0.15 | -0.43 | +0.23 | +0.73 | +0.89 | +0.15 |  |  | n/a (not genuine) |
| D4_llmfused | 20 | 794 | +0.60 | -3.58 | -0.29 | +2.06 | -0.97 | +3.04 | 0.00 | 1.00 | cross-sectional |

## event_driven
| model | h | n_days | real rel% | real DM(cl) | wd-placebo rel% | wd-placebo DM(cl) | date-mean rel% | date-mean DM(cl) | frac regime | frac cross-sec | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_bow_ridge | 5 | 996 | +1.33 | -3.35 | -0.03 | +1.23 | -0.38 | +2.08 | 0.00 | 1.00 | cross-sectional |
| B1_bow_ridge | 10 | 991 | +1.23 | -3.25 | -0.03 | +1.07 | -0.63 | +1.96 | 0.00 | 1.00 | cross-sectional |
| B1_bow_ridge | 20 | 981 | +1.53 | -3.10 | -0.01 | +0.92 | -0.29 | +1.48 | 0.00 | 1.00 | cross-sectional |
| B2_tfidf_ridge | 5 | 996 | +1.21 | -2.76 | -0.05 | +1.60 | -0.38 | +1.84 | 0.00 | 1.00 | cross-sectional |
| B2_tfidf_ridge | 10 | 991 | +1.35 | -2.97 | -0.02 | +0.78 | -0.05 | +0.58 | 0.00 | 1.00 | cross-sectional |
| B2_tfidf_ridge | 20 | 981 | +1.84 | -3.11 | +0.00 | -0.29 | -0.31 | +1.26 | 0.00 | 1.00 | cross-sectional |
| B3_lm_linear | 5 | 996 | +0.25 | -2.43 | +0.04 | -0.58 | +0.33 | -0.28 | 1.00 | 0.00 | regime-timing |
| B3_lm_linear | 10 | 991 | +0.20 | -1.23 | +0.00 | -0.01 | -0.10 | +0.25 | 0.00 | 1.00 | cross-sectional |
| B3_lm_linear | 20 | 981 | +0.26 | -1.12 | -0.01 | +0.33 | -0.53 | +1.08 | 0.00 | 1.00 | n/a (not genuine) |
| B4_lm_features | 5 | 996 | +0.18 | -0.99 | +0.02 | -0.21 | +0.38 | -0.31 | 1.00 | 0.00 | regime-timing |
| B4_lm_features | 10 | 991 | +0.08 | -2.10 | -0.01 | +0.47 | -0.03 | +0.68 | 0.00 | 1.00 | cross-sectional |
| B4_lm_features | 20 | 981 | +0.25 | -2.01 | -0.00 | +0.43 | -0.13 | +0.75 | 0.00 | 1.00 | cross-sectional |
| C2_finbert_s1 | 5 | 996 | +2.14 | -4.95 | -0.10 | +2.82 | -0.89 | +3.00 | 0.00 | 1.00 | n/a (not genuine) |
| C2_finbert_s1 | 10 | 991 | +2.10 | -5.52 | -0.14 | +1.39 | -0.87 | +1.75 | 0.00 | 1.00 | cross-sectional |
| C2_finbert_s1 | 20 | 981 | +0.92 | -0.50 | -0.64 | +2.79 | -7.21 | +3.96 | 0.00 | 1.00 | n/a (not genuine) |
| C6_llmtext | 5 | 996 | +1.21 | -5.04 | +0.04 | -1.54 | +0.39 | -1.44 | 0.32 | 0.68 | mixed |
| C6_llmtext | 10 | 991 | +1.00 | -3.76 | +0.01 | -1.21 | +0.05 | -2.24 | 0.05 | 0.95 | cross-sectional |
| C6_llmtext | 20 | 981 | +0.66 | -1.98 | -0.02 | +0.75 | -0.53 | +2.08 | 0.00 | 1.00 | n/a (not genuine) |
| D2_gated_fusion | 5 | 996 | +0.51 | -0.69 | +0.10 | -2.01 | +0.25 | -3.40 | 0.49 | 0.51 | n/a (not genuine) |
| D2_gated_fusion | 10 | 991 | +0.23 | -0.56 | -0.01 | +0.86 | -0.15 | +3.97 | 0.00 | 1.00 | n/a (not genuine) |
| D2_gated_fusion | 20 | 981 | -2.87 | +3.00 | -1.11 | +4.22 | -2.89 | +4.19 |  |  | n/a (not genuine) |
| D4_llmfused | 5 | 996 | -0.01 | +3.37 | +0.06 | -0.23 | +0.48 | -1.11 |  |  | n/a (not genuine) |
| D4_llmfused | 10 | 991 | -0.01 | +0.66 | +0.05 | -2.68 | +0.10 | -3.56 |  |  | n/a (not genuine) |
| D4_llmfused | 20 | 981 | -0.35 | +4.69 | -0.88 | +4.21 | -2.42 | +4.50 |  |  | n/a (not genuine) |

## Verdict
- Of the 38 genuine cells: **33 cross-sectional, 4 regime-timing, 1 mixed** (rule: regime if date-mean share >= 0.5 or within-date placebo survival >= 0.5; cross-sectional if both <= 0.25).
- Median decomposition of a genuine increment: **100% cross-sectional (which-firm) vs 0% regime-timing (when)**; the within-date placebo retains a median 0% of the real increment.
- Interpretation: a LOW regime share differentiates the finding from FinText/VIX-style aggregate-regime tracking; a HIGH regime share would mean the 'text signal' is mostly a calendar effect.