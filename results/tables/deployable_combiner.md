# ROW 7 — Deployable expanding combiner over the FULL grid (incl. C6/D4/C5 cells)

## RESTATED vs BEFORE

| | BEFORE (committed rolling_robustness) | RESTATED (this table) |
|---|---|---|
| grid | 36 cells (6-model subset, **no C6/D4**) | **75 cells** = 69-cell primary grid + C5_qwen3 extension |
| text basis | seed2026 only | seed-ensemble primary (mean over seeds 2026/2027/2028 where multi-seed) |
| per-quarter DM | observation-order HAC | day-clustered (HAC lag h-1 DAYS); legacy obs-order kept as a comparability column |
| deployable statistic | per-quarter counts only | + POOLED pseudo-OOS deployable path (16 quarter blocks concatenated), day-clustered DM, Holm, 5-seed permutation placebo |
| genuine cells, fixed val-frozen scheme | (m1_ensemble_primary: 38/69) | **36/75** (36/69 on the primary grid) |
| genuine cells, EXPANDING deployable scheme | never reported for C6 | **7/75** (6/69 on the primary grid) |

All QLIKE losses in this table are in **VOLATILITY units** (q(y, f), the convention of `m1_ensemble_primary`'s `vol_*` columns); the variance-unit sensitivity of the same grid lives in `m1_variance_unit` / `variance_unit_cascade`.

Schemes: **fixed** = `fc.log_combo` weights fit once on the 2020-21 validation window, frozen on the whole 2022Q1-2025Q4 test span (the committed primary; SANITY GATE 2 confirms machine-precision reproduction). **expanding** = for each test quarter q the weights are refit on ALL filings with `filing_time_utc` strictly before q's first calendar day — the 2010-2019 train split + validation + earlier test quarters, i.e. the full pre-q filing history (the identical pool convention as the committed `rolling_robustness.csv`, whose prose also under-described it as val+test); the 16 quarter blocks concatenate into ONE deployable pseudo-OOS forecast path. No look-ahead enters any weight applied to a quarter. Boundary caveat (same convention as the committed rolling_robustness.csv, kept for comparability): a filing whose effective day lies within h trading days of q_start has a label window ending after q_start, so the earliest few training labels per boundary are not yet fully realised at refit time; this affects only the training pool, never the evaluation rows.

Inference: pooled comparisons use the day-clustered DM (`clustered_dm.dm_test_clustered`, HAC lag = h-1 days, HLN). For the EXPANDING scheme the pooled DM is read in the Giacomini-White finite-estimation-window sense — it compares forecasting *methods including their recursive estimation scheme* — which is the appropriate frame once weights are re-estimated recursively (the fixed-scheme comparison remains a standard frozen-weight DM). Per-quarter clustered DM at h=20 sits on ~60 filing days with HAC lag 19 and is fragile; the POOLED statistics are the primary deployable evidence, the per-quarter `sig_q/16` counts are descriptive.

## PRE-DECLARED Holm families (declared before any result below was inspected)

- **F-DEPLOY-FIXED**: the 75 pooled day-clustered two-sided DM p-values of f_U vs f_R under the fixed scheme, one per grid cell; Holm within this family.
- **F-DEPLOY-EXP**: the 75 pooled day-clustered two-sided DM p-values of the expanding deployable path, one per grid cell; Holm within this family.
- `genuine` per scheme = pooled clustered DM < 0 AND Holm < .05 AND |placebo DM| < 2 (placebo = text forecast permuted on fit and application rows, 5 seeds, identical machinery per scheme).
- The family spans all 75 cells (69 primary + 6 C5_qwen3 extension cells) — slightly *more* conservative for the primary cells than the committed 69-cell convention.

## HEADLINE — the event-driven C6 residual under deployable weights

| disc | model | h | seeds | FIXED sig_q | FIXED mean rel% | FIXED pooled rel% | FIXED DM | FIXED Holm | EXP sig_q | EXP mean rel% | EXP pooled rel% | EXP DM | EXP Holm | EXP placebo | gap pooled rel% (exp-fixed) | genuine FIXED | genuine EXP | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | C6_llmtext | 5 | 2026 | 3/16 | +1.22 | +1.21 | -5.04 | 0.000 | 3/16 | +1.20 | +1.19 | -4.64 | 0.000 | -0.39 | -0.02 | YES | YES | SURVIVES-DEPLOY |
| event_driven | C6_llmtext | 10 | 2026 | 3/16 | +0.99 | +1.00 | -3.76 | 0.007 | 3/16 | +0.97 | +0.98 | -3.44 | 0.033 | -0.60 | -0.02 | YES | YES | SURVIVES-DEPLOY |
| event_driven | C6_llmtext | 20 | 2026 | 1/16 | +0.66 | +0.66 | -1.98 | 0.657 | 1/16 | +0.54 | +0.50 | -1.56 | 1.000 | +1.32 | -0.16 | no | no | null-null |
| event_driven | D4_llmfused | 5 | 2026 | 0/16 | -0.01 | -0.01 | +3.37 | 0.026 | 1/16 | +0.02 | +0.03 | -0.92 | 1.000 | +0.61 | +0.04 | no | no | null-null |
| event_driven | D4_llmfused | 10 | 2026 | 0/16 | -0.01 | -0.01 | +0.66 | 1.000 | 0/16 | -0.03 | -0.03 | +1.07 | 1.000 | -0.09 | -0.02 | no | no | null-null |
| event_driven | D4_llmfused | 20 | 2026 | 0/16 | -0.34 | -0.35 | +4.69 | 0.000 | 1/16 | -0.02 | +0.00 | -0.27 | 1.000 | -0.25 | +0.36 | no | no | null-null |
| event_driven | C5_qwen3 | 5 | 2026+2027+2028 | 0/16 | -0.12 | -0.13 | +2.83 | 0.109 | 0/16 | -0.45 | -0.45 | +7.03 | 0.000 | -1.28 | -0.32 | no | no | null-null |
| event_driven | C5_qwen3 | 10 | 2026+2027+2028 | 0/16 | -0.13 | -0.13 | +3.04 | 0.060 | 0/16 | -0.20 | -0.20 | +6.61 | 0.000 | +0.59 | -0.07 | no | no | null-null |
| event_driven | C5_qwen3 | 20 | 2026+2027+2028 | 0/16 | -0.47 | -0.45 | +3.40 | 0.025 | 2/16 | +0.07 | +0.07 | -4.18 | 0.002 | +0.42 | +0.52 | no | YES | GAINED-ON-DEPLOY |
| long_form | C6_llmtext | 5 | 2026 | 6/16 | +1.80 | +1.79 | -6.31 | 0.000 | 4/16 | +2.02 | +2.04 | -4.71 | 0.000 | +0.84 | +0.25 | YES | YES | SURVIVES-DEPLOY |
| long_form | C6_llmtext | 10 | 2026 | 11/16 | +2.18 | +2.25 | -7.92 | 0.000 | 9/16 | +2.00 | +2.03 | -5.64 | 0.000 | +0.53 | -0.22 | YES | YES | SURVIVES-DEPLOY |
| long_form | C6_llmtext | 20 | 2026 | 2/16 | +0.19 | +0.27 | -3.23 | 0.037 | 1/16 | -0.06 | -0.04 | -0.61 | 1.000 | +0.11 | -0.32 | YES | no | LOST-ON-DEPLOY |

Event-driven C6_llmtext: fixed-scheme genuine in 2/3 horizons; under the deployable expanding scheme **2/3** remain genuine (h5: pooled rel +1.19%, DM -4.64, Holm 0.000, h10: pooled rel +0.98%, DM -3.44, Holm 0.033).


## Full grid — per-cell deployable results

| disc | model | h | seeds | FIXED sig_q | FIXED mean rel% | FIXED pooled rel% | FIXED DM | FIXED Holm | EXP sig_q | EXP mean rel% | EXP pooled rel% | EXP DM | EXP Holm | EXP placebo | gap pooled rel% | genuine FIXED | genuine EXP | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | 2026 | 0/16 | +1.36 | +1.33 | -3.35 | 0.028 | 0/16 | -3.76 | -3.91 | +4.67 | 0.000 | +1.15 | -5.24 | YES | no | LOST-ON-DEPLOY |
| event_driven | B1_bow_ridge | 10 | 2026 | 2/16 | +1.24 | +1.23 | -3.25 | 0.037 | 0/16 | -4.42 | -4.84 | +3.92 | 0.006 | +0.76 | -6.06 | YES | no | LOST-ON-DEPLOY |
| event_driven | B1_bow_ridge | 20 | 2026 | 0/16 | +1.48 | +1.53 | -3.10 | 0.055 | 0/16 | -5.32 | -6.20 | +3.20 | 0.075 | +1.12 | -7.73 | no | no | null-null |
| event_driven | B2_tfidf_ridge | 5 | 2026 | 0/16 | +1.27 | +1.21 | -2.76 | 0.130 | 0/16 | -4.17 | -4.48 | +4.85 | 0.000 | +0.78 | -5.69 | no | no | null-null |
| event_driven | B2_tfidf_ridge | 10 | 2026 | 2/16 | +1.40 | +1.35 | -2.97 | 0.074 | 0/16 | -4.62 | -5.28 | +3.74 | 0.011 | +0.26 | -6.64 | no | no | null-null |
| event_driven | B2_tfidf_ridge | 20 | 2026 | 0/16 | +1.84 | +1.84 | -3.11 | 0.055 | 0/16 | -5.21 | -6.47 | +2.87 | 0.206 | +1.25 | -8.31 | no | no | null-null |
| event_driven | B3_lm_linear | 5 | 2026 | 0/16 | +0.24 | +0.25 | -2.43 | 0.273 | 0/16 | -0.15 | -0.16 | +5.30 | 0.000 | +1.44 | -0.41 | no | no | null-null |
| event_driven | B3_lm_linear | 10 | 2026 | 0/16 | +0.17 | +0.20 | -1.23 | 1.000 | 0/16 | +0.01 | +0.01 | -0.93 | 1.000 | +0.52 | -0.19 | no | no | null-null |
| event_driven | B3_lm_linear | 20 | 2026 | 0/16 | +0.19 | +0.26 | -1.12 | 1.000 | 2/16 | +0.01 | +0.01 | -0.57 | 1.000 | -0.16 | -0.25 | no | no | null-null |
| event_driven | B4_lm_features | 5 | 2026 | 0/16 | +0.19 | +0.18 | -0.99 | 1.000 | 0/16 | -0.11 | -0.11 | +3.50 | 0.027 | +0.05 | -0.29 | no | no | null-null |
| event_driven | B4_lm_features | 10 | 2026 | 0/16 | +0.08 | +0.08 | -2.10 | 0.604 | 0/16 | +0.16 | +0.18 | -0.50 | 1.000 | +0.10 | +0.09 | no | no | null-null |
| event_driven | B4_lm_features | 20 | 2026 | 1/16 | +0.20 | +0.25 | -2.01 | 0.657 | 3/16 | +0.12 | +0.10 | -1.16 | 1.000 | +0.22 | -0.15 | no | no | null-null |
| event_driven | C2_finbert_s1 | 5 | 2026+2027+2028 | 4/16 | +2.64 | +2.57 | -5.92 | 0.000 | 1/16 | +1.38 | +1.08 | +0.70 | 1.000 | -0.13 | -1.49 | YES | no | LOST-ON-DEPLOY |
| event_driven | C2_finbert_s1 | 10 | 2026+2027+2028 | 6/16 | +2.44 | +2.42 | -5.68 | 0.000 | 3/16 | -4.24 | -5.44 | +2.38 | 0.732 | +0.34 | -7.86 | YES | no | LOST-ON-DEPLOY |
| event_driven | C2_finbert_s1 | 20 | 2026+2027+2028 | 3/16 | +1.60 | +1.58 | -1.94 | 0.657 | 2/16 | +2.36 | +1.56 | -1.03 | 1.000 | -0.76 | -0.02 | no | no | null-null |
| event_driven | C5_qwen3 | 5 | 2026+2027+2028 | 0/16 | -0.12 | -0.13 | +2.83 | 0.109 | 0/16 | -0.45 | -0.45 | +7.03 | 0.000 | -1.28 | -0.32 | no | no | null-null |
| event_driven | C5_qwen3 | 10 | 2026+2027+2028 | 0/16 | -0.13 | -0.13 | +3.04 | 0.060 | 0/16 | -0.20 | -0.20 | +6.61 | 0.000 | +0.59 | -0.07 | no | no | null-null |
| event_driven | C5_qwen3 | 20 | 2026+2027+2028 | 0/16 | -0.47 | -0.45 | +3.40 | 0.025 | 2/16 | +0.07 | +0.07 | -4.18 | 0.002 | +0.42 | +0.52 | no | YES | GAINED-ON-DEPLOY |
| event_driven | C6_llmtext | 5 | 2026 | 3/16 | +1.22 | +1.21 | -5.04 | 0.000 | 3/16 | +1.20 | +1.19 | -4.64 | 0.000 | -0.39 | -0.02 | YES | YES | SURVIVES-DEPLOY |
| event_driven | C6_llmtext | 10 | 2026 | 3/16 | +0.99 | +1.00 | -3.76 | 0.007 | 3/16 | +0.97 | +0.98 | -3.44 | 0.033 | -0.60 | -0.02 | YES | YES | SURVIVES-DEPLOY |
| event_driven | C6_llmtext | 20 | 2026 | 1/16 | +0.66 | +0.66 | -1.98 | 0.657 | 1/16 | +0.54 | +0.50 | -1.56 | 1.000 | +1.32 | -0.16 | no | no | null-null |
| event_driven | D2_gated_fusion | 5 | 2026+2027+2028 | 0/16 | +0.60 | +0.56 | -1.70 | 0.988 | 0/16 | -0.07 | -0.22 | +2.65 | 0.377 | +0.79 | -0.78 | no | no | null-null |
| event_driven | D2_gated_fusion | 10 | 2026+2027+2028 | 0/16 | +0.20 | +0.18 | -0.42 | 1.000 | 0/16 | -2.26 | -2.43 | +4.81 | 0.000 | -0.55 | -2.61 | no | no | null-null |
| event_driven | D2_gated_fusion | 20 | 2026+2027+2028 | 0/16 | -3.00 | -2.76 | +3.55 | 0.016 | 2/16 | -0.36 | -1.83 | +0.50 | 1.000 | +0.59 | +0.93 | no | no | null-null |
| event_driven | D4_llmfused | 5 | 2026 | 0/16 | -0.01 | -0.01 | +3.37 | 0.026 | 1/16 | +0.02 | +0.03 | -0.92 | 1.000 | +0.61 | +0.04 | no | no | null-null |
| event_driven | D4_llmfused | 10 | 2026 | 0/16 | -0.01 | -0.01 | +0.66 | 1.000 | 0/16 | -0.03 | -0.03 | +1.07 | 1.000 | -0.09 | -0.02 | no | no | null-null |
| event_driven | D4_llmfused | 20 | 2026 | 0/16 | -0.34 | -0.35 | +4.69 | 0.000 | 1/16 | -0.02 | +0.00 | -0.27 | 1.000 | -0.25 | +0.36 | no | no | null-null |
| long_form | B1_bow_ridge | 5 | 2026 | 6/16 | +1.62 | +1.65 | -3.83 | 0.006 | 0/16 | -16.28 | -17.44 | +4.23 | 0.002 | -0.52 | -19.09 | YES | no | LOST-ON-DEPLOY |
| long_form | B1_bow_ridge | 10 | 2026 | 4/16 | +1.40 | +1.44 | -4.15 | 0.002 | 0/16 | -11.74 | -15.06 | +3.02 | 0.131 | +0.77 | -16.51 | YES | no | LOST-ON-DEPLOY |
| long_form | B1_bow_ridge | 20 | 2026 | 5/16 | +2.98 | +2.99 | -5.45 | 0.000 | 0/16 | -10.26 | -13.46 | +2.29 | 0.864 | -0.00 | -16.44 | YES | no | LOST-ON-DEPLOY |
| long_form | B2_tfidf_ridge | 5 | 2026 | 5/16 | +3.34 | +3.33 | -5.39 | 0.000 | 1/16 | -3.24 | -4.47 | +3.02 | 0.131 | -0.22 | -7.80 | YES | no | LOST-ON-DEPLOY |
| long_form | B2_tfidf_ridge | 10 | 2026 | 10/16 | +3.48 | +3.48 | -8.89 | 0.000 | 3/16 | -1.46 | -4.12 | +1.11 | 1.000 | +0.42 | -7.60 | YES | no | LOST-ON-DEPLOY |
| long_form | B2_tfidf_ridge | 20 | 2026 | 10/16 | +5.90 | +5.92 | -9.04 | 0.000 | 1/16 | +0.05 | -3.06 | +0.68 | 1.000 | -0.84 | -8.98 | YES | no | LOST-ON-DEPLOY |
| long_form | B3_lm_linear | 5 | 2026 | 3/16 | +0.53 | +0.49 | -2.58 | 0.189 | 1/16 | +0.54 | +0.54 | -2.30 | 0.859 | -0.27 | +0.04 | no | no | null-null |
| long_form | B3_lm_linear | 10 | 2026 | 3/16 | +1.74 | +1.79 | -4.62 | 0.000 | 2/16 | +1.06 | +1.16 | -3.19 | 0.078 | +0.19 | -0.64 | YES | no | LOST-ON-DEPLOY |
| long_form | B3_lm_linear | 20 | 2026 | 2/16 | +3.26 | +3.48 | -4.69 | 0.000 | 1/16 | +1.63 | +1.92 | -2.60 | 0.419 | -0.77 | -1.56 | YES | no | LOST-ON-DEPLOY |
| long_form | B4_lm_features | 5 | 2026 | 1/16 | +0.15 | +0.11 | +1.02 | 1.000 | 0/16 | -0.15 | -0.30 | +2.37 | 0.741 | +0.48 | -0.41 | no | no | null-null |
| long_form | B4_lm_features | 10 | 2026 | 0/16 | -0.93 | -0.92 | +3.32 | 0.030 | 1/16 | +0.38 | +0.10 | -1.54 | 1.000 | +0.14 | +1.02 | no | no | null-null |
| long_form | B4_lm_features | 20 | 2026 | 0/16 | -1.91 | -1.92 | +3.38 | 0.026 | 0/16 | +0.83 | +0.54 | -1.20 | 1.000 | -0.59 | +2.46 | no | no | null-null |
| long_form | C1_bert_s1 | 5 | 2026+2027+2028 | 4/16 | +3.39 | +3.42 | -3.25 | 0.037 | 1/16 | -7.19 | -8.45 | +4.08 | 0.003 | +0.14 | -11.87 | YES | no | LOST-ON-DEPLOY |
| long_form | C1_bert_s1 | 10 | 2026+2027+2028 | 1/16 | -0.74 | -0.85 | +3.06 | 0.060 | 1/16 | -0.74 | +0.96 | -1.05 | 1.000 | -0.03 | +1.81 | no | no | null-null |
| long_form | C1_bert_s1 | 20 | 2026+2027+2028 | 5/16 | +2.74 | +2.70 | -6.96 | 0.000 | 1/16 | -2.24 | -4.42 | +1.02 | 1.000 | -0.75 | -7.12 | YES | no | LOST-ON-DEPLOY |
| long_form | C2_finbert_s1 | 5 | 2026+2027+2028 | 2/16 | +1.93 | +1.90 | -4.53 | 0.000 | 3/16 | +8.09 | +8.58 | -5.50 | 0.000 | -0.75 | +6.68 | YES | YES | SURVIVES-DEPLOY |
| long_form | C2_finbert_s1 | 10 | 2026+2027+2028 | 6/16 | +2.60 | +2.62 | -6.67 | 0.000 | 1/16 | -6.71 | -10.05 | +2.18 | 1.000 | +0.32 | -12.67 | YES | no | LOST-ON-DEPLOY |
| long_form | C2_finbert_s1 | 20 | 2026+2027+2028 | 3/16 | -0.05 | -0.59 | -0.40 | 1.000 | 1/16 | -4.99 | -3.04 | +1.09 | 1.000 | +0.08 | -2.45 | no | no | null-null |
| long_form | C2_finbert_s2 | 5 | 2026+2027+2028 | 4/16 | +1.24 | +1.21 | -5.57 | 0.000 | 1/16 | -3.45 | -4.82 | +3.75 | 0.011 | -0.43 | -6.02 | YES | no | LOST-ON-DEPLOY |
| long_form | C2_finbert_s2 | 10 | 2026+2027+2028 | 12/16 | +0.48 | +0.48 | -7.59 | 0.000 | 2/16 | -0.54 | -2.36 | -0.50 | 1.000 | -0.34 | -2.85 | YES | no | LOST-ON-DEPLOY |
| long_form | C2_finbert_s2 | 20 | 2026+2027+2028 | 4/16 | +1.93 | +1.68 | -4.89 | 0.000 | 0/16 | +1.76 | -0.16 | -0.40 | 1.000 | -0.26 | -1.84 | YES | no | LOST-ON-DEPLOY |
| long_form | C2_finbert_s3 | 5 | 2026+2027+2028 | 5/16 | +2.78 | +2.90 | -5.36 | 0.000 | 1/16 | -3.27 | -4.35 | +3.56 | 0.022 | -0.50 | -7.25 | YES | no | LOST-ON-DEPLOY |
| long_form | C2_finbert_s3 | 10 | 2026+2027+2028 | 7/16 | +2.41 | +2.31 | -5.26 | 0.000 | 0/16 | -15.23 | -19.82 | +3.82 | 0.008 | +0.09 | -22.13 | YES | no | LOST-ON-DEPLOY |
| long_form | C2_finbert_s3 | 20 | 2026+2027+2028 | 2/16 | -2.88 | -3.86 | +1.68 | 0.988 | 1/16 | +0.98 | -0.25 | -0.12 | 1.000 | +0.09 | +3.61 | no | no | null-null |
| long_form | C2_finbert_s4 | 5 | 2026+2027+2028 | 6/16 | +1.53 | +1.46 | -4.78 | 0.000 | 0/16 | -12.02 | -14.21 | +5.94 | 0.000 | +0.33 | -15.67 | YES | no | LOST-ON-DEPLOY |
| long_form | C2_finbert_s4 | 10 | 2026+2027+2028 | 10/16 | +0.38 | +0.36 | -6.82 | 0.000 | 1/16 | -10.47 | -13.94 | +2.09 | 1.000 | +0.21 | -14.30 | YES | no | LOST-ON-DEPLOY |
| long_form | C2_finbert_s4 | 20 | 2026+2027+2028 | 9/16 | +3.14 | +3.08 | -8.34 | 0.000 | 1/16 | -1.42 | -4.28 | +0.75 | 1.000 | -0.05 | -7.35 | YES | no | LOST-ON-DEPLOY |
| long_form | C3_roberta_s1 | 5 | 2026+2027+2028 | 5/16 | +0.29 | +0.30 | -5.13 | 0.000 | 1/16 | -2.55 | -3.93 | +2.65 | 0.377 | -0.96 | -4.22 | YES | no | LOST-ON-DEPLOY |
| long_form | C3_roberta_s1 | 10 | 2026+2027+2028 | 5/16 | +1.89 | +1.84 | -5.06 | 0.000 | 3/16 | +3.38 | +3.32 | -2.80 | 0.252 | +0.01 | +1.48 | YES | no | LOST-ON-DEPLOY |
| long_form | C3_roberta_s1 | 20 | 2026+2027+2028 | 6/16 | +0.02 | +0.02 | -3.90 | 0.004 | 0/16 | -13.09 | -17.26 | +2.25 | 0.937 | -0.10 | -17.28 | YES | no | LOST-ON-DEPLOY |
| long_form | C4_longformer | 5 | 2026+2027+2028 | 6/16 | +1.43 | +1.47 | -6.18 | 0.000 | 4/16 | +2.32 | +2.00 | -1.48 | 1.000 | -0.39 | +0.52 | YES | no | LOST-ON-DEPLOY |
| long_form | C4_longformer | 10 | 2026+2027+2028 | 0/16 | -2.83 | -2.95 | +10.32 | 0.000 | 3/16 | -1.85 | -3.01 | -0.54 | 1.000 | -0.14 | -0.06 | no | no | null-null |
| long_form | C4_longformer | 20 | 2026+2027+2028 | 4/16 | +1.07 | +0.91 | -4.38 | 0.001 | 2/16 | -1.03 | -3.16 | -0.01 | 1.000 | +0.26 | -4.07 | YES | no | LOST-ON-DEPLOY |
| long_form | C5_qwen3 | 5 | 2026+2027+2028 | 0/16 | -0.85 | -1.02 | +2.64 | 0.170 | 0/16 | -0.24 | -0.27 | +2.40 | 0.720 | +0.33 | +0.75 | no | no | null-null |
| long_form | C5_qwen3 | 10 | 2026+2027+2028 | 0/16 | -2.48 | -3.10 | +3.62 | 0.012 | 0/16 | -0.12 | -0.14 | +0.51 | 1.000 | +0.66 | +2.96 | no | no | null-null |
| long_form | C5_qwen3 | 20 | 2026+2027+2028 | 0/16 | -5.97 | -6.39 | +5.38 | 0.000 | 2/16 | +0.34 | +0.69 | -0.99 | 1.000 | -0.86 | +7.08 | no | no | null-null |
| long_form | C6_llmtext | 5 | 2026 | 6/16 | +1.80 | +1.79 | -6.31 | 0.000 | 4/16 | +2.02 | +2.04 | -4.71 | 0.000 | +0.84 | +0.25 | YES | YES | SURVIVES-DEPLOY |
| long_form | C6_llmtext | 10 | 2026 | 11/16 | +2.18 | +2.25 | -7.92 | 0.000 | 9/16 | +2.00 | +2.03 | -5.64 | 0.000 | +0.53 | -0.22 | YES | YES | SURVIVES-DEPLOY |
| long_form | C6_llmtext | 20 | 2026 | 2/16 | +0.19 | +0.27 | -3.23 | 0.037 | 1/16 | -0.06 | -0.04 | -0.61 | 1.000 | +0.11 | -0.32 | YES | no | LOST-ON-DEPLOY |
| long_form | D1_concat_mlp | 5 | 2026+2027+2028 | 0/16 | -1.04 | -1.04 | +2.66 | 0.169 | 1/16 | +4.99 | +6.20 | -4.03 | 0.004 | +0.81 | +7.24 | no | YES | GAINED-ON-DEPLOY |
| long_form | D1_concat_mlp | 10 | 2026+2027+2028 | 0/16 | -0.50 | -0.46 | +3.54 | 0.016 | 1/16 | -1.09 | -1.05 | +0.14 | 1.000 | -0.35 | -0.59 | no | no | null-null |
| long_form | D1_concat_mlp | 20 | 2026+2027+2028 | 5/16 | +0.11 | +0.12 | -5.57 | 0.000 | 0/16 | -4.23 | -6.37 | +1.65 | 1.000 | +0.37 | -6.49 | YES | no | LOST-ON-DEPLOY |
| long_form | D2_gated_fusion | 5 | 2026+2027+2028 | 3/16 | +0.18 | +0.19 | -2.11 | 0.604 | 1/16 | +0.70 | +0.79 | +1.61 | 1.000 | +0.67 | +0.60 | no | no | null-null |
| long_form | D2_gated_fusion | 10 | 2026+2027+2028 | 0/16 | -0.02 | -0.02 | +2.02 | 0.657 | 0/16 | +0.50 | +1.57 | -1.03 | 1.000 | -0.31 | +1.59 | no | no | null-null |
| long_form | D2_gated_fusion | 20 | 2026+2027+2028 | 0/16 | -2.51 | -2.27 | +5.56 | 0.000 | 1/16 | -2.73 | -5.36 | +0.93 | 1.000 | +0.19 | -3.09 | no | no | null-null |
| long_form | D4_llmfused | 5 | 2026 | 3/16 | +0.17 | +0.15 | -0.75 | 1.000 | 0/16 | +0.13 | +0.14 | -0.31 | 1.000 | +0.52 | -0.01 | no | no | null-null |
| long_form | D4_llmfused | 10 | 2026 | 0/16 | -0.15 | -0.15 | -0.43 | 1.000 | 0/16 | -0.17 | -0.19 | -0.49 | 1.000 | +1.52 | -0.05 | no | no | null-null |
| long_form | D4_llmfused | 20 | 2026 | 1/16 | +0.58 | +0.60 | -3.58 | 0.014 | 1/16 | +0.69 | +0.67 | -2.79 | 0.252 | -0.65 | +0.06 | YES | no | LOST-ON-DEPLOY |

## Fixed-vs-expanding gap summary

| disc | mean gap pooled rel% (exp-fixed) | mean gap per-quarter rel% | mean gap sig_q |
|---|---|---|---|
| event_driven | -1.940 | -1.647 | -0.04 |
| long_form | -3.933 | -2.962 | -2.65 |

- LOST-ON-DEPLOY (31): long_form/B1_bow_ridge/h5; long_form/B1_bow_ridge/h10; long_form/B1_bow_ridge/h20; long_form/B2_tfidf_ridge/h5; long_form/B2_tfidf_ridge/h10; long_form/B2_tfidf_ridge/h20; long_form/B3_lm_linear/h10; long_form/B3_lm_linear/h20; long_form/C1_bert_s1/h5; long_form/C1_bert_s1/h20; long_form/C2_finbert_s1/h10; long_form/C2_finbert_s2/h5; long_form/C2_finbert_s2/h10; long_form/C2_finbert_s2/h20; long_form/C2_finbert_s3/h5; long_form/C2_finbert_s3/h10; long_form/C2_finbert_s4/h5; long_form/C2_finbert_s4/h10; long_form/C2_finbert_s4/h20; long_form/C3_roberta_s1/h5; long_form/C3_roberta_s1/h10; long_form/C3_roberta_s1/h20; long_form/C4_longformer/h5; long_form/C4_longformer/h20; long_form/C6_llmtext/h20; long_form/D1_concat_mlp/h20; long_form/D4_llmfused/h20; event_driven/B1_bow_ridge/h5; event_driven/B1_bow_ridge/h10; event_driven/C2_finbert_s1/h5; event_driven/C2_finbert_s1/h10
- GAINED-ON-DEPLOY (2): long_form/D1_concat_mlp/h5; event_driven/C5_qwen3/h20
- SURVIVES-DEPLOY (5): long_form/C2_finbert_s1/h5; long_form/C6_llmtext/h5; long_form/C6_llmtext/h10; event_driven/C6_llmtext/h5; event_driven/C6_llmtext/h10

## SANITY

**GATE 1 (committed table: `results/tables/rolling_robustness.csv`)** — the 36 overlapping (disc, model, h) cells were recomputed on the exact committed code path (seed2026 text, obs-order per-quarter DM, moving-block CI) and compared row-by-row (1152 rows x 6 columns) with NaN-aware machine-precision equality (atol 1e-12; bitwise identity is unattainable across runs — BLAS reduction order injects ~1e-15 noise into lstsq): rel_impr_pct: exact, dm_stat: exact, dm_p: exact, ci_lo: exact, ci_hi: exact, n: exact -> **PASS** (joined 1152/1152 rows; max abs diff over all float columns = 3.6e-15).

**GATE 2 (committed table: `results/tables/m1_ensemble_primary.csv`)** — the fixed val-frozen scheme on the ensemble basis reproduces the committed 69-cell primary to machine precision (atol 1e-12, DM stat included in the pass criterion): max|dQLIKE_R|=8.3e-17, max|dQLIKE_U|=9.7e-17, max|dg_log|=9.7e-17, max|dDM_clu|=8.9e-16 over 69/69 cells -> **PASS**.

- All test rows fall inside the 16-quarter span (rows outside span: 0); the expanding path covers every span row (no unfit quarters).
- No subsampling anywhere; the per-quarter long panel is in `deployable_combiner_quarters.csv`.
- The one committed anecdote reproduces on the new basis: long_form/B2_tfidf_ridge/h5 BEFORE fixed 12/16 -> expanding 3/16 (legacy obs-order, seed2026); see the per-cell table for the restated counts on the ensemble basis.

## Verdict

- Fixed val-frozen scheme (Holm within F-DEPLOY-FIXED): **36/75** genuine cells (36/69 primary).
- EXPANDING deployable scheme (Holm within F-DEPLOY-EXP): **7/75** genuine cells (6/69 primary); 31 lost on deploy, 2 gained, 5 survive.
- Event-driven C6 residual: 2/3 genuine fixed -> **2/3 genuine under deployable expanding weights** (see HEADLINE).
- Mean pooled-rel gap (expanding - fixed) across all cells: **-3.216pp**.