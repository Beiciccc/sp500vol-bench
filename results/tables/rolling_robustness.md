# R1 — Rolling-window period-robustness of the M1 text increment

Trained-model forecasts are FIXED; only the log-space nested combiner (`fc.log_combo`, weights fit on val or on strictly-earlier filings) is rolled across the 16 test quarters **2022Q1..2025Q4**. Per quarter we report the incremental-text QLIKE improvement `rel_impr_pct = 100*mean(qR-qU)/mean(qR)` (positive => text helps), DM `dm_test(qU,qR,h)` (NEGATIVE stat => text better), and a moving-block CI of `mean(qR-qU)`. Two schemes: **expanding** (combiner refit each quarter on all earlier filings, pseudo-OOS) vs **fixed** (combiner frozen on the COVID 2020-2021 val window — the current M1).

`sig_q/16` = quarters where text significantly helps (DM<0, p<.05). `slope` = OLS slope of per-quarter `rel_impr_pct` on quarter index 0..15 (negative => the increment is DECAYING over 2022-2025).

## Per-cell robustness (disclosure x model x horizon)

| disclosure | model | h | fixed sig_q/16 | fixed mean rel% | fixed slope | exp sig_q/16 | exp mean rel% | exp slope | exp-fixed mean rel% |
|---|---|---|---|---|---|---|---|---|---|
| event_driven | B2_tfidf_ridge | 5 | 8/16 | +1.27 | +0.015 | 1/16 | -4.17 | +0.889 | -5.43 |
| event_driven | B2_tfidf_ridge | 10 | 9/16 | +1.40 | +0.012 | 1/16 | -4.62 | +1.160 | -6.03 |
| event_driven | B2_tfidf_ridge | 20 | 9/16 | +1.84 | -0.052 | 2/16 | -5.21 | +1.540 | -7.05 |
| event_driven | C2_finbert_s1 | 5 | 13/16 | +2.22 | -0.083 | 9/16 | +2.45 | +0.720 | +0.23 |
| event_driven | C2_finbert_s1 | 10 | 13/16 | +2.16 | -0.019 | 6/16 | -2.89 | +1.714 | -5.05 |
| event_driven | C2_finbert_s1 | 20 | 6/16 | +0.88 | -0.376 | 6/16 | +1.60 | +1.031 | +0.71 |
| event_driven | C2_finbert_s2 | 5 | 8/16 | +1.73 | +0.054 | 6/16 | +0.36 | +0.822 | -1.37 |
| event_driven | C2_finbert_s2 | 10 | 13/16 | +2.06 | +0.035 | 1/16 | -9.87 | +2.474 | -11.93 |
| event_driven | C2_finbert_s2 | 20 | 10/16 | +0.74 | -0.082 | 5/16 | -5.42 | +2.004 | -6.16 |
| event_driven | C4_longformer | 5 | 11/16 | +1.48 | +0.138 | 2/16 | -10.51 | +2.953 | -11.99 |
| event_driven | C4_longformer | 10 | 9/16 | +0.89 | +0.027 | 7/16 | +0.82 | +1.269 | -0.07 |
| event_driven | C4_longformer | 20 | 0/16 | -2.06 | +0.016 | 4/16 | -11.03 | +3.205 | -8.96 |
| event_driven | C5_qwen3 | 5 | 1/16 | -0.12 | -0.044 | 0/16 | -0.46 | -0.025 | -0.35 |
| event_driven | C5_qwen3 | 10 | 1/16 | -0.13 | -0.021 | 0/16 | -0.21 | -0.005 | -0.08 |
| event_driven | C5_qwen3 | 20 | 0/16 | -0.44 | -0.012 | 10/16 | +0.10 | +0.011 | +0.54 |
| event_driven | D2_gated_fusion | 5 | 6/16 | +0.54 | -0.046 | 3/16 | +0.46 | -0.050 | -0.09 |
| event_driven | D2_gated_fusion | 10 | 6/16 | +0.27 | +0.043 | 0/16 | -3.53 | +0.426 | -3.80 |
| event_driven | D2_gated_fusion | 20 | 2/16 | -3.21 | +0.297 | 11/16 | +3.59 | -0.023 | +6.79 |
| long_form | B2_tfidf_ridge | 5 | 12/16 | +3.34 | +0.299 | 3/16 | -3.24 | +1.734 | -6.58 |
| long_form | B2_tfidf_ridge | 10 | 14/16 | +3.48 | +0.234 | 4/16 | -1.46 | +2.106 | -4.94 |
| long_form | B2_tfidf_ridge | 20 | 16/16 | +5.90 | +0.107 | 6/16 | +0.05 | +2.521 | -5.85 |
| long_form | C2_finbert_s1 | 5 | 6/16 | +0.55 | -0.023 | 10/16 | +8.43 | -0.325 | +7.88 |
| long_form | C2_finbert_s1 | 10 | 12/16 | +4.60 | +0.196 | 3/16 | -8.84 | +2.227 | -13.44 |
| long_form | C2_finbert_s1 | 20 | 0/16 | -0.09 | +0.002 | 7/16 | -0.04 | +0.135 | +0.05 |
| long_form | C2_finbert_s2 | 5 | 10/16 | +1.83 | +0.171 | 3/16 | +1.23 | +0.678 | -0.60 |
| long_form | C2_finbert_s2 | 10 | 8/16 | +0.24 | -0.006 | 3/16 | +1.59 | +0.646 | +1.36 |
| long_form | C2_finbert_s2 | 20 | 0/16 | -5.72 | +0.258 | 5/16 | -0.07 | +0.065 | +5.65 |
| long_form | C4_longformer | 5 | 11/16 | +1.49 | +0.077 | 4/16 | -0.45 | +0.824 | -1.95 |
| long_form | C4_longformer | 10 | 0/16 | -2.52 | -0.091 | 3/16 | -3.24 | +0.858 | -0.71 |
| long_form | C4_longformer | 20 | 0/16 | -9.71 | +0.405 | 2/16 | -0.13 | +0.019 | +9.59 |
| long_form | C5_qwen3 | 5 | 1/16 | -0.86 | -0.092 | 5/16 | -0.24 | +0.074 | +0.63 |
| long_form | C5_qwen3 | 10 | 3/16 | -2.51 | +0.011 | 1/16 | -0.13 | +0.025 | +2.37 |
| long_form | C5_qwen3 | 20 | 3/16 | -6.24 | +0.288 | 4/16 | +0.32 | -0.054 | +6.57 |
| long_form | D2_gated_fusion | 5 | 3/16 | -0.10 | +0.051 | 0/16 | -1.68 | +0.034 | -1.58 |
| long_form | D2_gated_fusion | 10 | 4/16 | +0.20 | +0.086 | 5/16 | +1.01 | +0.007 | +0.81 |
| long_form | D2_gated_fusion | 20 | 1/16 | -6.23 | +0.074 | 5/16 | -5.67 | +0.847 | +0.56 |

## Grand pooled summary across models (disclosure x horizon x scheme)

| disclosure | h | scheme | pooled mean rel% | n_sig quarter-cells | n quarter-cells |
|---|---|---|---|---|---|
| event_driven | 5 | expanding | -1.979 | 21 | 96 |
| event_driven | 5 | fixed | +1.186 | 47 | 96 |
| event_driven | 10 | expanding | -3.384 | 15 | 96 |
| event_driven | 10 | fixed | +1.108 | 51 | 96 |
| event_driven | 20 | expanding | -2.729 | 38 | 96 |
| event_driven | 20 | fixed | -0.374 | 27 | 96 |
| long_form | 5 | expanding | +0.676 | 25 | 96 |
| long_form | 5 | fixed | +1.041 | 43 | 96 |
| long_form | 10 | expanding | -1.845 | 19 | 96 |
| long_form | 10 | fixed | +0.581 | 41 | 96 |
| long_form | 20 | expanding | -0.923 | 29 | 96 |
| long_form | 20 | fixed | -3.684 | 20 | 96 |

## SANITY — pooled-fixed reproduces static M1 (within ~0.2pp)

| disclosure | model | h | pooled-fixed rel% | static M1 rel% | abs diff (pp) |
|---|---|---|---|---|---|
| event_driven | B2_tfidf_ridge | 5 | +1.207 | +1.207 | 0.0000 |
| event_driven | B2_tfidf_ridge | 10 | +1.355 | +1.355 | 0.0000 |
| event_driven | B2_tfidf_ridge | 20 | +1.840 | +1.840 | 0.0000 |
| event_driven | C2_finbert_s1 | 5 | +2.135 | +2.135 | 0.0000 |
| event_driven | C2_finbert_s1 | 10 | +2.104 | +2.104 | 0.0000 |
| event_driven | C2_finbert_s1 | 20 | +0.920 | +0.920 | 0.0000 |
| event_driven | C2_finbert_s2 | 5 | +1.694 | +nan | nan |
| event_driven | C2_finbert_s2 | 10 | +1.978 | +nan | nan |
| event_driven | C2_finbert_s2 | 20 | +0.752 | +nan | nan |
| event_driven | C4_longformer | 5 | +1.463 | +nan | nan |
| event_driven | C4_longformer | 10 | +0.887 | +nan | nan |
| event_driven | C4_longformer | 20 | -2.001 | +nan | nan |
| event_driven | C5_qwen3 | 5 | -0.125 | +nan | nan |
| event_driven | C5_qwen3 | 10 | -0.128 | +nan | nan |
| event_driven | C5_qwen3 | 20 | -0.421 | +nan | nan |
| event_driven | D2_gated_fusion | 5 | +0.509 | +0.509 | 0.0000 |
| event_driven | D2_gated_fusion | 10 | +0.231 | +0.231 | 0.0000 |
| event_driven | D2_gated_fusion | 20 | -2.869 | -2.869 | 0.0000 |
| long_form | B2_tfidf_ridge | 5 | +3.331 | +3.331 | 0.0000 |
| long_form | B2_tfidf_ridge | 10 | +3.482 | +3.482 | 0.0000 |
| long_form | B2_tfidf_ridge | 20 | +5.920 | +5.920 | 0.0000 |
| long_form | C2_finbert_s1 | 5 | +0.562 | +0.562 | 0.0000 |
| long_form | C2_finbert_s1 | 10 | +4.558 | +4.558 | 0.0000 |
| long_form | C2_finbert_s1 | 20 | -0.084 | -0.084 | 0.0000 |
| long_form | C2_finbert_s2 | 5 | +1.734 | +1.734 | 0.0000 |
| long_form | C2_finbert_s2 | 10 | +0.244 | +0.244 | 0.0000 |
| long_form | C2_finbert_s2 | 20 | -5.190 | -5.190 | 0.0000 |
| long_form | C4_longformer | 5 | +1.526 | +1.526 | 0.0000 |
| long_form | C4_longformer | 10 | -2.608 | -2.608 | 0.0000 |
| long_form | C4_longformer | 20 | -9.341 | -9.341 | 0.0000 |
| long_form | C5_qwen3 | 5 | -1.035 | +nan | nan |
| long_form | C5_qwen3 | 10 | -3.135 | +nan | nan |
| long_form | C5_qwen3 | 20 | -6.647 | +nan | nan |
| long_form | D2_gated_fusion | 5 | -0.117 | -0.117 | 0.0000 |
| long_form | D2_gated_fusion | 10 | +0.135 | +0.135 | 0.0000 |
| long_form | D2_gated_fusion | 20 | -6.001 | -6.001 | 0.0000 |

## Verdict — period-robustness

- Cells analysed: **36** (disclosure x model x horizon).
- Median significant-help quarters: **fixed 6/16**, **expanding 4/16**.
- Decaying trend (OLS slope<0): fixed **13/36**, expanding **6/36** cells.
- Updating the combiner vs freezing it on COVID-val: mean(exp-fixed rel%) = **-1.674pp** across cells (freezing is no worse / better).
- SANITY: pooled-fixed reproduces static M1 exactly — median abs diff **0.0000pp**, max **0.0000pp**; 24/24 cells (with a static reference) within 0.2pp (12 cells have no static M1 reference: C5_qwen3 + event_driven C2_finbert_s2/C4_longformer were not in the M1 SETS).