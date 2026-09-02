# E4 — Stronger price baselines (A6_shar, A6_harq) vs A2 HAR-RV

All fits replicate A2 conventions (pooled log-space OLS per horizon, eps=1e-12, Duan smearing). Returns from the canonical CRSP store (`market/crsp/market_returns.parquet`, log1p(DlyRet)) which exactly reproduces the aligned features; OHLCV adj_close only correlates ~0.78 and was not used. Rows with unreliable ticker-recycled joins dropped via `ret_match_ok`. Both variants use the BPQ (2016) insanity filter (any forecast outside the train-sample RV range is replaced by the train mean, per horizon); without it log-space HARQ diverges on ~0.4% extreme test points where rv_1=0 meets extreme quarticity (long_form h=5 test QLIKE 2.67 vs 0.56, 28 obs). It essentially never binds for SHAR.

## Row drops (per disclosure)

| disclosure   |   n_rows |   drop_ret_match |   drop_feat_window |   n_kept |   kept_pct |
|:-------------|---------:|-----------------:|-------------------:|---------:|-----------:|
| long_form    |    94237 |             9042 |                  0 |    85195 |      90.41 |
| event_driven |   333192 |            28178 |                  0 |   305014 |      91.54 |
| combined     |   427429 |            37220 |                  0 |   390209 |      91.29 |

## Test QLIKE (variance-unit, matching metrics.json) — A6 variants vs A2

DM on the common (kept) test sample; negative DM = variant better than A2. † p<0.05, ‡ p<0.01.

| disc | h | n | A2 (full) | A2 (common) | SHAR | ΔQ% | DM_q | DM_se | HARQ | ΔQ% | DM_q | DM_se |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long_form | 5 | 7550 | 0.5564 | 0.5635 | 0.5568 | +1.18 | -3.735‡ | -1.12 | 0.8804 | -56.25 | 0.957 | 1.319 |
| long_form | 10 | 7167 | 0.4618 | 0.4226 | 0.4106 | +2.84 | -6.485‡ | -2.737‡ | 0.5646 | -33.60 | 0.89 | 0.966 |
| long_form | 20 | 7097 | 0.2970 | 0.3076 | 0.2948 | +4.17 | -8.015‡ | -3.156‡ | 0.4031 | -31.03 | 0.841 | 0.514 |
| event_driven | 5 | 23855 | 0.5902 | 0.5992 | 0.5927 | +1.09 | -5.755‡ | -3.976‡ | 0.6186 | -3.24 | 0.764 | -2.157† |
| event_driven | 10 | 22785 | 0.4408 | 0.4576 | 0.4454 | +2.65 | -8.374‡ | -7.001‡ | 0.4569 | +0.15 | -0.062 | -5.394‡ |
| event_driven | 20 | 22318 | 0.3185 | 0.3327 | 0.3192 | +4.08 | -11.725‡ | -8.258‡ | 0.3263 | +1.93 | -0.887 | -5.66‡ |
| combined | 5 | 31405 | 0.5823 | 0.5908 | 0.5841 | +1.14 | -6.773‡ | -3.818‡ | 0.6601 | -11.73 | 1.308 | -1.323 |
| combined | 10 | 29952 | 0.4448 | 0.4497 | 0.4373 | +2.74 | -9.792‡ | -7.106‡ | 0.4700 | -4.52 | 0.855 | -3.576‡ |
| combined | 20 | 29415 | 0.3151 | 0.3286 | 0.3151 | +4.12 | -13.504‡ | -8.359‡ | 0.3361 | -2.27 | 0.473 | -4.13‡ |

## M1 incremental value of text — A2 vs A6_shar as the price reference

fc.log_combo (val-fitted recalibrated reference f_R vs +text f_U), QLIKE in VOL units (fc convention), same common sample for both references. rel_impr% > 0 with DM < 0 = text still adds. † p<0.05, ‡ p<0.01.

| disc | text model | h | n_test | A2-ref impr% | DM | A6_shar-ref impr% | DM | survives? |
|---|---|---|---|---|---|---|---|---|
| long_form | B2_tfidf_ridge | 5 | 7550 | +3.39 | -11.179‡ | +3.53 | -11.308‡ | YES |
| long_form | B2_tfidf_ridge | 10 | 7167 | +3.63 | -15.212‡ | +3.78 | -15.457‡ | YES |
| long_form | B2_tfidf_ridge | 20 | 7097 | +5.27 | -19.104‡ | +5.29 | -19.596‡ | YES |
| long_form | C2_finbert_s1 | 5 | 7550 | +0.49 | -5.507‡ | +0.55 | -5.339‡ | YES |
| long_form | C2_finbert_s1 | 10 | 7167 | +4.41 | -12.456‡ | +4.49 | -12.452‡ | YES |
| long_form | C2_finbert_s1 | 20 | 7097 | -0.33 | 4.963‡ | -0.30 | 4.918‡ | no |
| long_form | C5_qwen3 | 5 | 7550 | -1.03 | 3.908‡ | -0.94 | 3.663‡ | no |
| long_form | C5_qwen3 | 10 | 7167 | -3.08 | 4.235‡ | -2.89 | 4.094‡ | no |
| long_form | C5_qwen3 | 20 | 7097 | -6.14 | 4.188‡ | -5.98 | 4.134‡ | no |
| event_driven | B2_tfidf_ridge | 5 | 23855 | +1.17 | -6.756‡ | +1.18 | -6.797‡ | YES |
| event_driven | B2_tfidf_ridge | 10 | 22785 | +1.27 | -6.783‡ | +1.29 | -6.926‡ | YES |
| event_driven | B2_tfidf_ridge | 20 | 22318 | +1.68 | -7.934‡ | +1.69 | -8.035‡ | YES |
| event_driven | C2_finbert_s1 | 5 | 23855 | +2.01 | -14.235‡ | +2.02 | -14.152‡ | YES |
| event_driven | C2_finbert_s1 | 10 | 22785 | +1.92 | -13.369‡ | +1.92 | -13.402‡ | YES |
| event_driven | C2_finbert_s1 | 20 | 22318 | +0.75 | -1.684 | +0.78 | -1.764 | weak |
| event_driven | C5_qwen3 | 5 | 23855 | -0.14 | 2.67‡ | -0.15 | 2.778‡ | no |
| event_driven | C5_qwen3 | 10 | 22785 | -0.15 | 3.408‡ | -0.16 | 3.49‡ | no |
| event_driven | C5_qwen3 | 20 | 22318 | -0.52 | 6.51‡ | -0.53 | 6.538‡ | no |
