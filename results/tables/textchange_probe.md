# E3 — Textual-change probe (Lazy Prices): change vs level in long-form disclosure

B6_textchange = Ridge on [change_score(=1-TFIDF-cosine vs previous same-form filing), jaccard_change(1-Jaccard on word 5-shingles, bottom-k minhash k=256), log length ratio, 10-K dummy]; IDF fit on train docs only; gap<550d; B2 log-target/retransform conventions; splits inherited from A2. DM sign: negative = first argument better.

**Prev-filing match:** 29991/31601 unique long-form docs (94.9%) matched to a previous same-form filing (row-level imputed fraction in the run: 4.7%; imputed with train means {'change_score': 0.079, 'jaccard_change': 0.386, 'log_len_ratio': 0.0115}).
Match rate by year: 2010:50%, 2011:98%, 2012:98%, 2013:98%, 2014:98%, 2015:98%, 2016:97%, 2017:97%, 2018:97%, 2019:97%, 2020:98%, 2021:98%, 2022:98%, 2023:98%, 2024:98%, 2025:98%

**Change-feature distribution (matched docs):** change_score mean=0.078 sd=0.053 p10=0.025 p90=0.147; jaccard_change mean=0.370; median gap 98d.

## (1) Standalone test QLIKE (vol-unit) — B6 change vs B2 full-text level
| h | n_test | QLIKE HAR raw | QLIKE B2 | QLIKE B6 | DM B6 vs B2 | p |
|---|---|---|---|---|---|---|
| 5 | 7951 | 0.1226 | 0.2085 | 0.2347 | +8.67 | 0.0000 |
| 10 | 7933 | 0.0893 | 0.1646 | 0.1890 | +10.13 | 0.0000 |
| 20 | 7902 | 0.0646 | 0.1284 | 0.1451 | +8.33 | 0.0000 |

## (2) M1 incremental value over recalibrated HAR (log combo, val-frozen)
| h | QLIKE f_R | QLIKE f_U(B6) | g_B6 | DM | p | rel% | QLIKE f_U(B2) | g_B2 | DM(B2) | p(B2) | rel%(B2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.1209 | 0.1238 | -1.485 | +6.41 | 0.0000 | -2.41 | 0.1168 | +0.237 | -11.34 | 0.0000 | +3.33 |
| 10 | 0.0873 | 0.0904 | -2.101 | +5.34 | 0.0000 | -3.58 | 0.0843 | +0.170 | -15.30 | 0.0000 | +3.48 |
| 20 | 0.0701 | 0.0749 | -3.065 | +5.99 | 0.0000 | -6.85 | 0.0659 | +0.204 | -19.62 | 0.0000 | +5.92 |

## (3) KEY — does CHANGE add increment BEYOND the full-text LEVEL?
Joint val-frozen log combiner [1, log fHAR, log fB2, log fB6] vs level-only f_U(HAR+B2).
| h | QLIKE level-only f_U | QLIKE joint | g_level | g_change | DM joint vs level | p | rel% |
|---|---|---|---|---|---|---|---|
| 5 | 0.1168 | 0.1195 | +0.281 | -1.676 | +5.23 | 0.0000 | -2.26 |
| 10 | 0.0843 | 0.0867 | +0.218 | -2.226 | +4.06 | 0.0001 | -2.92 |
| 20 | 0.0659 | 0.0700 | +0.252 | -3.181 | +4.99 | 0.0000 | -6.14 |

## (4) Lazy Prices sign check — corr(change_score, future realised vol), matched docs
| h | Pearson (test) | Spearman (test) | Spearman p | Pearson (all splits) |
|---|---|---|---|---|
| 5 | +0.0415 | +0.0161 | 1.55e-01 | +0.0308 |
| 10 | +0.0504 | +0.0335 | 3.06e-03 | +0.0400 |
| 20 | +0.0847 | +0.0748 | 3.83e-11 | +0.0492 |
