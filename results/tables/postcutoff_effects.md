# ROW 8 — Post-training-cutoff effect sizes + 95% CIs for the 8-K (C6) residual

## RESTATED vs BEFORE

| | BEFORE (llm_contamination.md sec.3) | RESTATED (this table) |
|---|---|---|
| post-cutoff evidence | significance COUNTS only (5/6 Holm fulltext-vs-HAR, 3/6 beyond-identity); point rel% with NO uncertainty | per-cell EFFECT SIZES with day-block moving-bootstrap 95% CIs of the mean daily loss differential, pre and post side-by-side, both references |
| pre-vs-post comparison | eyeballed ("post > pre in 6/6 cells") | post-minus-pre difference of the mean daily differential with a bootstrap 95% CI per cell |
| references | same | same two: (a) single recalibrated HAR; (b) identity-augmented [1, log fHAR, log f_datefirm] (same-model date+ticker control) |

**Protocol (no refit, no look-ahead):** combiner weights are the ORIGINAL full-validation fit, frozen; ONLY the test evaluation is stratified at 2024-07-01 (approx. Qwen3 training-data boundary; caveat in llm_contamination.md applies). QLIKE in vol units; day-clustered DM (daily-mean loss differentials over effective_trading_day, HAC lag = h-1 days). Bootstrap: committed `clustered_dm.mbb_ci_daily` — moving-block bootstrap over DAYS of the daily-mean loss differential d = QLIKE(U) - QLIKE(R), block length h days, B=2000, seed 2026 (committed defaults reused; the brief's 1000-draw fallback applies only when no committed bootstrap exists). Negative d = text arm better; the rel%-scaled CI divides by the SUBSAMPLE daily-mean reference QLIKE (fixed denominator), so it is equal-weighted per day: it brackets the DAY-weighted point (rel%_daily), NOT the observation-weighted rel% column, and the two can differ materially where within-day observation counts covary with the loss differential (e.g. long_form h5 pre: obs +1.25% vs daily +3.08%). No subsampling anywhere: all C6 test observations are used.

## PRE-DECLARED Holm family

One family, declared before any results: the **24 stratified cells** (2 disclosures x 3 horizons x 2 references x 2 strata), Holm within it — identical to the committed llm_contamination cutoff block (sanity gate C asserts the Holm column reproduces it exactly). Full-sample rows are context only; their Holm values are inherited unchanged from the committed llm_contamination families (18-cell variant block / 6-cell joint block) — no new inference is run on them. `**` = clustered DM<0 and Holm p<.05.

## 1. Full-sample context rows (sanity anchor; CIs new)

| disc | h | ref | n_obs | n_days | rel% | d_daily x1e-4 [95% CI] | DM(clu) | p raw | p Holm* |
|---|---|---|---|---|---|---|---|---|---|
| long_form | 5 | fulltext_vs_har | 7951 | 809 | +1.79%** | -30.41 [-40.63, -21.77] | -6.31 | 4.66e-10 | 0.0000 |
| long_form | 5 | beyond_identity | 7951 | 809 | +1.50%** | -25.81 [-35.34, -17.83] | -5.83 | 8.14e-09 | 0.0000 |
| long_form | 10 | fulltext_vs_har | 7933 | 803 | +2.25%** | -26.66 [-33.28, -20.20] | -7.92 | 8.18e-15 | 0.0000 |
| long_form | 10 | beyond_identity | 7933 | 803 | +1.95%** | -22.66 [-28.95, -16.63] | -7.13 | 2.28e-12 | 0.0000 |
| long_form | 20 | fulltext_vs_har | 7902 | 794 | +0.27%** | -3.09 [-4.86, -1.25] | -3.23 | 1.28e-03 | 0.0115 |
| long_form | 20 | beyond_identity | 7902 | 794 | +0.17%** | -2.03 [-3.55, -0.47] | -2.51 | 1.23e-02 | 0.0247 |
| event_driven | 5 | fulltext_vs_har | 25109 | 996 | +1.21%** | -11.42 [-15.68, -6.85] | -5.04 | 5.42e-07 | 0.0000 |
| event_driven | 5 | beyond_identity | 25109 | 996 | +0.77%** | -7.47 [-11.28, -3.32] | -3.79 | 1.57e-04 | 0.0006 |
| event_driven | 10 | fulltext_vs_har | 25001 | 991 | +1.00%** | -6.72 [-10.07, -3.13] | -3.76 | 1.81e-04 | 0.0020 |
| event_driven | 10 | beyond_identity | 25001 | 991 | +0.68%** | -5.09 [-8.24, -1.38] | -2.85 | 4.41e-03 | 0.0132 |
| event_driven | 20 | fulltext_vs_har | 24732 | 981 | +0.66% | -3.28 [-6.29, -0.07] | -1.98 | 4.77e-02 | 0.3340 |
| event_driven | 20 | beyond_identity | 24732 | 981 | +0.63%** | -3.39 [-6.34, -0.12] | -2.01 | 4.46e-02 | 0.0446 |

*Holm inherited from the committed llm_contamination families (context only).

## 2. Pre/post stratified effect sizes (24 cells, the pre-declared family)

| disc | h | ref | stratum | n_obs | n_days | rel% | d_daily x1e-4 [95% CI] | rel%_daily [95% CI] | DM(clu) | p raw | p Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| long_form | 5 | fulltext_vs_har | pre | 4957 | 498 | +1.25%** | -33.64 [-48.00, -22.24] | +3.08% [+2.03, +4.39] | -4.95 | 1.00e-06 | 0.0000 |
| long_form | 5 | fulltext_vs_har | post | 2994 | 311 | +2.56%** | -25.22 [-37.48, -13.15] | +1.94% [+1.01, +2.88] | -4.01 | 7.74e-05 | 0.0013 |
| long_form | 5 | beyond_identity | pre | 4957 | 498 | +1.03%** | -28.68 [-41.89, -18.30] | +2.64% [+1.69, +3.86] | -4.61 | 5.06e-06 | 0.0001 |
| long_form | 5 | beyond_identity | post | 2994 | 311 | +2.16%** | -21.21 [-32.69, -9.69] | +1.64% [+0.75, +2.53] | -3.62 | 3.43e-04 | 0.0055 |
| long_form | 10 | fulltext_vs_har | pre | 4956 | 497 | +1.71%** | -25.00 [-32.89, -18.08] | +3.18% [+2.30, +4.19] | -6.43 | 2.94e-10 | 0.0000 |
| long_form | 10 | fulltext_vs_har | post | 2977 | 306 | +3.09%** | -29.36 [-41.00, -16.48] | +3.05% [+1.71, +4.26] | -4.63 | 5.32e-06 | 0.0001 |
| long_form | 10 | beyond_identity | pre | 4956 | 497 | +1.44%** | -21.31 [-28.85, -14.52] | +2.75% [+1.87, +3.72] | -5.61 | 3.43e-08 | 0.0000 |
| long_form | 10 | beyond_identity | post | 2977 | 306 | +2.73%** | -24.86 [-35.50, -13.04] | +2.62% [+1.37, +3.74] | -4.32 | 2.11e-05 | 0.0004 |
| long_form | 20 | fulltext_vs_har | pre | 4949 | 497 | +0.11% | -1.37 [-3.19, +0.37] | +0.21% [-0.06, +0.48] | -1.51 | 1.30e-01 | 0.5217 |
| long_form | 20 | fulltext_vs_har | post | 2953 | 297 | +0.53%** | -5.95 [-9.44, -2.28] | +0.77% [+0.29, +1.22] | -3.04 | 2.54e-03 | 0.0355 |
| long_form | 20 | beyond_identity | pre | 4949 | 497 | +0.05% | -0.76 [-2.38, +0.78] | +0.12% [-0.12, +0.36] | -0.94 | 3.46e-01 | 0.9822 |
| long_form | 20 | beyond_identity | post | 2953 | 297 | +0.35% | -4.16 [-7.12, -1.07] | +0.55% [+0.14, +0.94] | -2.50 | 1.28e-02 | 0.1280 |
| event_driven | 5 | fulltext_vs_har | pre | 15804 | 625 | +0.87% | -7.44 [-12.54, -2.27] | +0.65% [+0.20, +1.09] | -2.78 | 5.57e-03 | 0.0669 |
| event_driven | 5 | fulltext_vs_har | post | 9305 | 372 | +1.71%** | -18.04 [-25.86, -10.50] | +1.28% [+0.75, +1.84] | -4.52 | 8.21e-06 | 0.0002 |
| event_driven | 5 | beyond_identity | pre | 15804 | 625 | +0.55% | -5.05 [-9.20, -0.88] | +0.44% [+0.08, +0.80] | -2.28 | 2.30e-02 | 0.1837 |
| event_driven | 5 | beyond_identity | post | 9305 | 372 | +1.07%** | -11.51 [-18.46, -3.78] | +0.83% [+0.27, +1.32] | -3.09 | 2.16e-03 | 0.0324 |
| event_driven | 10 | fulltext_vs_har | pre | 15786 | 625 | +0.70% | -4.00 [-7.35, -0.49] | +0.50% [+0.06, +0.92] | -2.25 | 2.49e-02 | 0.1837 |
| event_driven | 10 | fulltext_vs_har | post | 9215 | 367 | +1.40%** | -11.30 [-17.97, -3.61] | +1.10% [+0.35, +1.74] | -3.03 | 2.63e-03 | 0.0355 |
| event_driven | 10 | beyond_identity | pre | 15786 | 625 | +0.47% | -3.30 [-6.24, -0.32] | +0.42% [+0.04, +0.78] | -2.13 | 3.33e-02 | 0.1996 |
| event_driven | 10 | beyond_identity | post | 9215 | 367 | +0.98% | -8.08 [-14.93, +1.03] | +0.79% [-0.10, +1.46] | -1.99 | 4.76e-02 | 0.2381 |
| event_driven | 20 | fulltext_vs_har | pre | 15751 | 625 | +0.51% | -2.81 [-4.90, -0.73] | +0.48% [+0.12, +0.83] | -2.46 | 1.40e-02 | 0.1280 |
| event_driven | 20 | fulltext_vs_har | post | 8981 | 357 | +0.86% | -4.10 [-10.96, +4.26] | +0.52% [-0.54, +1.40] | -0.96 | 3.36e-01 | 0.9822 |
| event_driven | 20 | beyond_identity | pre | 15751 | 625 | +0.46% | -2.85 [-4.87, -0.84] | +0.48% [+0.14, +0.83] | -2.60 | 9.47e-03 | 0.1042 |
| event_driven | 20 | beyond_identity | post | 8981 | 357 | +0.86% | -4.28 [-11.27, +4.33] | +0.55% [-0.55, +1.44] | -0.98 | 3.27e-01 | 0.9822 |

## 3. Pre vs post side-by-side (per cell x reference)

`diff` = post mean daily d minus pre mean daily d (negative = LARGER post-cutoff improvement), bootstrap 95% CI from independent day-block draws of each stratum (same block scheme, one rng seeded 2026 per cell). Each rel% cell shows the obs-weighted point first, then in parentheses the DAY-weighted point with the CI that brackets it (the CI is day-weighted, so it brackets the daily point, not necessarily the obs-weighted one).

| disc | h | ref | pre rel% (daily [95% CI]) | post rel% (daily [95% CI]) | diff x1e-4 [95% CI] | post verdict |
|---|---|---|---|---|---|---|
| long_form | 5 | fulltext_vs_har | +1.25% (+3.08% [+2.03, +4.39]) | +2.56% (+1.94% [+1.01, +2.88]) | +8.42 [-7.92, +27.01] | Holm-sig; point est larger than pre |
| long_form | 5 | beyond_identity | +1.03% (+2.64% [+1.69, +3.86]) | +2.16% (+1.64% [+0.75, +2.53]) | +7.47 [-7.65, +24.76] | Holm-sig; point est larger than pre |
| long_form | 10 | fulltext_vs_har | +1.71% (+3.18% [+2.30, +4.19]) | +3.09% (+3.05% [+1.71, +4.26]) | -4.36 [-17.90, +10.61] | Holm-sig; point est larger than pre |
| long_form | 10 | beyond_identity | +1.44% (+2.75% [+1.87, +3.72]) | +2.73% (+2.62% [+1.37, +3.74]) | -3.54 [-16.12, +10.22] | Holm-sig; point est larger than pre |
| long_form | 20 | fulltext_vs_har | +0.11% (+0.21% [-0.06, +0.48]) | +0.53% (+0.77% [+0.29, +1.22]) | -4.58 [-8.49, -0.38] | Holm-sig; point est larger than pre |
| long_form | 20 | beyond_identity | +0.05% (+0.12% [-0.12, +0.36]) | +0.35% (+0.55% [+0.14, +0.94]) | -3.40 [-6.81, +0.27] | raw-sig; point est larger than pre |
| event_driven | 5 | fulltext_vs_har | +0.87% (+0.65% [+0.20, +1.09]) | +1.71% (+1.28% [+0.75, +1.84]) | -10.60 [-20.54, -1.41] | Holm-sig; point est larger than pre |
| event_driven | 5 | beyond_identity | +0.55% (+0.44% [+0.08, +0.80]) | +1.07% (+0.83% [+0.27, +1.32]) | -6.46 [-15.27, +2.12] | Holm-sig; point est larger than pre |
| event_driven | 10 | fulltext_vs_har | +0.70% (+0.50% [+0.06, +0.92]) | +1.40% (+1.10% [+0.35, +1.74]) | -7.30 [-14.51, +1.37] | Holm-sig; point est larger than pre |
| event_driven | 10 | beyond_identity | +0.47% (+0.42% [+0.04, +0.78]) | +0.98% (+0.79% [-0.10, +1.46]) | -4.78 [-11.94, +4.24] | raw-sig; point est larger than pre |
| event_driven | 20 | fulltext_vs_har | +0.51% (+0.48% [+0.12, +0.83]) | +0.86% (+0.52% [-0.54, +1.40]) | -1.29 [-8.55, +7.33] | n.s.; point est larger than pre |
| event_driven | 20 | beyond_identity | +0.46% (+0.48% [+0.14, +0.83]) | +0.86% (+0.55% [-0.55, +1.44]) | -1.43 [-8.74, +7.21] | n.s.; point est larger than pre |

## Bottom line (honest headline)
- **The post-cutoff residual effect size HOLDS UP — no cell shows significant shrinkage.** The post-minus-pre difference CI indicates a significantly SMALLER post-cutoff effect in 0/12 cells; the 2 cell(s) where the difference IS significant go the other way (larger post-cutoff effect: long_form h20 (fulltext_vs_har), event_driven h5 (fulltext_vs_har)).
- **vs single recalibrated HAR (post-cutoff):** Holm-significant in 5/6 cells (raw 5/6); the 95% day-block CI of the mean daily differential excludes zero in 5/6.
- **vs identity-augmented reference (post-cutoff):** Holm-significant in 3/6 (raw 5/6); CI excludes zero in 4/6 — the beyond-identity residual is estimated with positive point effect in 6/6 post-cutoff cells but with wider intervals on the 297-372-day post subsamples; Holm-failing post-cutoff cells: long_form h20, event_driven h10, event_driven h20.
- **Weighting nuance (disclosed, not hidden):** obs-weighted rel% is larger post-cutoff in 12/12 cells, but the day-equal-weighted mean daily differential is larger (more negative) post-cutoff in 10/12 — the divergent cells (long_form h5, both references) have higher post-cutoff volatility inflating the obs-weighted rel% denominator mix. Under NEITHER weighting does the effect shrink significantly anywhere.
- **Honest claim:** "no evidence the residual weakens after the training cutoff; where the pre/post difference is individually significant (2/12 cells) it strengthens" — NOT "the residual grows" (most differences are individually insignificant).
- Post-cutoff subsamples are 297-372 days (2953-9305 obs) — CIs are correspondingly wider than full-sample; effect sizes remain economically small (post-cutoff rel% max +3.09%).

## SANITY (gate status: PASS)

Anchor table: `results/tables/llm_contamination.csv` (committed). All comparisons machine-precision (tolerance 1e-12; CSV float64 round-trip is lossless).

- Gate A_fulltext_vs_har_full: 6 rows, max|d n_test|=0.00e+00, max|d n_days|=0.00e+00, max|d rel_pct|=2.22e-16, max|d dm_clu|=4.44e-16, max|d p_raw|=6.25e-17: **PASS**
- Gate B_beyond_identity_full: 6 rows, max|d n_test|=0.00e+00, max|d n_days|=0.00e+00, max|d rel_pct|=2.22e-16, max|d dm_clu|=0.00e+00, max|d p_raw|=9.36e-17: **PASS**
- Gate C_stratified_incl_holm: 24 rows, max|d n_test|=0.00e+00, max|d n_days|=0.00e+00, max|d rel_pct|=9.02e-17, max|d dm_clu|=2.22e-16, max|d p_raw|=8.67e-17, max|d p_holm|=9.53e-17: **PASS**

Gate A is the brief-mandated gate (full-sample fulltext rows == llm_contamination.csv). Gates B/C additionally pin the identity-reference rows and the entire stratified block incl. Holm.