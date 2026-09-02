# LLM contamination controls — C6 date-only / date+firm vs full-text (P0-1, round-2)

## RESTATED vs BEFORE

| | BEFORE | RESTATED (this table) |
|---|---|---|
| citability | round-1 FATAL contamination finding claimed fixed; C6_dateonly/C6_datefirm runs existed but produced ZERO committed tables — the circulated numbers ("date-only carries no increment; date+ticker reproduces 30-80%") were citable nowhere | full M1 protocol (single recalibrated-HAR reference, val-fit/test-apply, day-clustered DM, raw p + Holm within block) on all three C6 arms, 18 cells, committed |
| date-only increment | claimed "no increment" | significant positive in 0/6 cells (Holm) — see grid |
| identity-memory share | claimed 30-80% | measured datefirm/fulltext reproduction fraction 31%-77% across the 3 cells with a well-identified denominator (fulltext rel% >= 1% and Holm-sig); full range 31%-405% over all 5 Holm-sig fulltext cells — the 405% is long_form h20, where datefirm ALONE (+1.11%) exceeds the tiny fulltext increment (+0.27%) |
| text-beyond-identity | untested | joint reference [1, log fHAR, log f_datefirm]: fulltext still adds in 6/6 cells (Holm) |
| era contamination | untested | test split at 2024-07-01 (approx. Qwen3 training-data boundary): post-cutoff fulltext-vs-HAR increment significant in 5/6 cells (Holm; 5/6 raw); post-cutoff beyond-identity in 3/6 (Holm; 5/6 raw) |

Protocol identical to crossfamily_llm / m1_clustered: log-space combiner weights fit on validation ONLY, frozen to test; QLIKE in vol units; day-clustered DM (daily-mean loss differentials over effective_trading_day, HAC lag = h-1 days). `**` = clustered DM<0 and Holm p<.05 (Holm within block). rel% > 0 = arm lowers QLIKE vs the reference.

## 1. The three arms vs the single recalibrated-HAR reference (18 cells)

`repro_frac` = rel%(arm) / rel%(fulltext), same cell — the share of the full-text increment that survives deleting the text. dateonly = prompt has the filing DATE only (no text, no ticker); datefirm = date + TICKER (no text) — anything datefirm reproduces is identity/era memory, not filing content.

| disc | h | arm | n_test | n_days | QLIKE(R) | QLIKE(U) | rel% | DM(clu) | p raw | p Holm | g_text | repro_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long_form | 5 | fulltext | 7951 | 809 | 0.1209 | 0.1187 | +1.79%** | -6.31 | 4.66e-10 | 0.0000 | +0.254 | - |
| long_form | 5 | datefirm | 7951 | 809 | 0.1209 | 0.1202 | +0.55%** | -4.52 | 6.97e-06 | 0.0001 | +0.160 | 0.31 |
| long_form | 5 | dateonly | 7951 | 809 | 0.1209 | 0.1209 | -0.00% | -0.93 | 3.50e-01 | 1.0000 | -0.099 | -0.00 |
| long_form | 10 | fulltext | 7933 | 803 | 0.0873 | 0.0853 | +2.25%** | -7.92 | 8.18e-15 | 0.0000 | +0.333 | - |
| long_form | 10 | datefirm | 7933 | 803 | 0.0873 | 0.0866 | +0.83%** | -5.72 | 1.46e-08 | 0.0000 | +0.239 | 0.37 |
| long_form | 10 | dateonly | 7933 | 803 | 0.0873 | 0.0877 | -0.49% | +3.46 | 5.64e-04 | 0.0056 | -0.145 | -0.22 |
| long_form | 20 | fulltext | 7902 | 794 | 0.0701 | 0.0699 | +0.27%** | -3.23 | 1.28e-03 | 0.0115 | +0.078 | - |
| long_form | 20 | datefirm | 7902 | 794 | 0.0701 | 0.0693 | +1.11%** | -5.82 | 8.73e-09 | 0.0000 | +0.111 | 4.05 |
| long_form | 20 | dateonly | 7902 | 794 | 0.0701 | 0.0712 | -1.65% | +4.34 | 1.58e-05 | 0.0002 | -0.104 | -5.99 |
| event_driven | 5 | fulltext | 25109 | 996 | 0.1265 | 0.1250 | +1.21%** | -5.04 | 5.42e-07 | 0.0000 | +0.264 | - |
| event_driven | 5 | datefirm | 25109 | 996 | 0.1265 | 0.1253 | +0.94%** | -2.78 | 5.48e-03 | 0.0439 | +0.368 | 0.77 |
| event_driven | 5 | dateonly | 25109 | 996 | 0.1265 | 0.1265 | -0.00% | +0.95 | 3.41e-01 | 1.0000 | -0.105 | -0.00 |
| event_driven | 10 | fulltext | 25001 | 991 | 0.0883 | 0.0874 | +1.00%** | -3.76 | 1.81e-04 | 0.0020 | +0.281 | - |
| event_driven | 10 | datefirm | 25001 | 991 | 0.0883 | 0.0876 | +0.71% | -1.24 | 2.17e-01 | 1.0000 | +0.320 | 0.72 |
| event_driven | 10 | dateonly | 25001 | 991 | 0.0883 | 0.0886 | -0.39% | +1.64 | 1.00e-01 | 0.6025 | -2.432 | -0.39 |
| event_driven | 20 | fulltext | 24732 | 981 | 0.0645 | 0.0641 | +0.66% | -1.98 | 4.77e-02 | 0.3340 | +0.245 | - |
| event_driven | 20 | datefirm | 24732 | 981 | 0.0645 | 0.0645 | +0.01% | +0.12 | 9.06e-01 | 1.0000 | +0.316 | 0.01 |
| event_driven | 20 | dateonly | 24732 | 981 | 0.0645 | 0.0645 | +0.00% | -0.69 | 4.92e-01 | 1.0000 | -0.071 | 0.01 |

## 2. Text beyond identity — joint reference [1, log fHAR, log f_datefirm] (+ log f_fulltext) (6 cells)

The same-model identity control: the reference already contains everything the SAME LLM produces from date+ticker alone, so any residual fulltext increment must come from the filing text.

| disc | h | n_test | n_days | QLIKE(R') | QLIKE(U') | rel% | DM(clu) | p raw | p Holm | g_text |
|---|---|---|---|---|---|---|---|---|---|---|
| long_form | 5 | 7951 | 809 | 0.1202 | 0.1184 | +1.50%** | -5.83 | 8.14e-09 | 0.0000 | +0.241 |
| long_form | 10 | 7933 | 803 | 0.0866 | 0.0849 | +1.95%** | -7.13 | 2.28e-12 | 0.0000 | +0.321 |
| long_form | 20 | 7902 | 794 | 0.0693 | 0.0692 | +0.17%** | -2.51 | 1.23e-02 | 0.0247 | +0.070 |
| event_driven | 5 | 25109 | 996 | 0.1253 | 0.1244 | +0.77%** | -3.79 | 1.57e-04 | 0.0006 | +0.233 |
| event_driven | 10 | 25001 | 991 | 0.0876 | 0.0870 | +0.68%** | -2.85 | 4.41e-03 | 0.0132 | +0.245 |
| event_driven | 20 | 24732 | 981 | 0.0645 | 0.0641 | +0.63%** | -2.01 | 4.46e-02 | 0.0446 | +0.214 |

## 3. Cutoff-date stratification — test filings split at 2024-07-01 (12 cells x 2 arms)

Combiner weights are the FULL-validation fit, frozen; only the test evaluation is stratified. "post" filings postdate the approximate Qwen3 training-data era, so their outcomes cannot be memorized.

| disc | h | arm | stratum | n_test | n_days | rel% | DM(clu) | p raw | p Holm |
|---|---|---|---|---|---|---|---|---|---|
| long_form | 5 | fulltext_vs_har | pre | 4957 | 498 | +1.25%** | -4.95 | 1.00e-06 | 0.0000 |
| long_form | 5 | beyond_identity | pre | 4957 | 498 | +1.03%** | -4.61 | 5.06e-06 | 0.0001 |
| long_form | 5 | fulltext_vs_har | post | 2994 | 311 | +2.56%** | -4.01 | 7.74e-05 | 0.0013 |
| long_form | 5 | beyond_identity | post | 2994 | 311 | +2.16%** | -3.62 | 3.43e-04 | 0.0055 |
| long_form | 10 | fulltext_vs_har | pre | 4956 | 497 | +1.71%** | -6.43 | 2.94e-10 | 0.0000 |
| long_form | 10 | beyond_identity | pre | 4956 | 497 | +1.44%** | -5.61 | 3.43e-08 | 0.0000 |
| long_form | 10 | fulltext_vs_har | post | 2977 | 306 | +3.09%** | -4.63 | 5.32e-06 | 0.0001 |
| long_form | 10 | beyond_identity | post | 2977 | 306 | +2.73%** | -4.32 | 2.11e-05 | 0.0004 |
| long_form | 20 | fulltext_vs_har | pre | 4949 | 497 | +0.11% | -1.51 | 1.30e-01 | 0.5217 |
| long_form | 20 | beyond_identity | pre | 4949 | 497 | +0.05% | -0.94 | 3.46e-01 | 0.9822 |
| long_form | 20 | fulltext_vs_har | post | 2953 | 297 | +0.53%** | -3.04 | 2.54e-03 | 0.0355 |
| long_form | 20 | beyond_identity | post | 2953 | 297 | +0.35% | -2.50 | 1.28e-02 | 0.1280 |
| event_driven | 5 | fulltext_vs_har | pre | 15804 | 625 | +0.87% | -2.78 | 5.57e-03 | 0.0669 |
| event_driven | 5 | beyond_identity | pre | 15804 | 625 | +0.55% | -2.28 | 2.30e-02 | 0.1837 |
| event_driven | 5 | fulltext_vs_har | post | 9305 | 372 | +1.71%** | -4.52 | 8.21e-06 | 0.0002 |
| event_driven | 5 | beyond_identity | post | 9305 | 372 | +1.07%** | -3.09 | 2.16e-03 | 0.0324 |
| event_driven | 10 | fulltext_vs_har | pre | 15786 | 625 | +0.70% | -2.25 | 2.49e-02 | 0.1837 |
| event_driven | 10 | beyond_identity | pre | 15786 | 625 | +0.47% | -2.13 | 3.33e-02 | 0.1996 |
| event_driven | 10 | fulltext_vs_har | post | 9215 | 367 | +1.40%** | -3.03 | 2.63e-03 | 0.0355 |
| event_driven | 10 | beyond_identity | post | 9215 | 367 | +0.98% | -1.99 | 4.76e-02 | 0.2381 |
| event_driven | 20 | fulltext_vs_har | pre | 15751 | 625 | +0.51% | -2.46 | 1.40e-02 | 0.1280 |
| event_driven | 20 | beyond_identity | pre | 15751 | 625 | +0.46% | -2.60 | 9.47e-03 | 0.1042 |
| event_driven | 20 | fulltext_vs_har | post | 8981 | 357 | +0.86% | -0.96 | 3.36e-01 | 0.9822 |
| event_driven | 20 | beyond_identity | post | 8981 | 357 | +0.86% | -0.98 | 3.27e-01 | 0.9822 |

## Bottom line
- **Date-only carries no increment** (0/6 Holm-significant positive; long_form h10/h20 date-only is significantly WORSE than the recalibrated HAR alone) — the increment is not an artefact of prompting per se or of era information in the date.
- **Date+ticker (identity memory, zero filing text) reproduces 31%-77% of the fulltext increment** in the 3 well-identified cells (fulltext rel% >= 1%, Holm-sig), and at long_form h20 identity memory alone (+1.11%) EXCEEDS the fulltext increment (+0.27%): a large share of the headline C6 rel% is firm-identity/era memory, not filing content. The raw C6-vs-HAR rel% must never be quoted as a text effect without this control.
- **But text is not reducible to identity**: with the same-model datefirm forecast IN the reference, fulltext still adds in 6/6 cells (Holm), retaining 63%-96% of the uncontrolled rel% per cell.
- **Era contamination cannot explain the residual**: on post-2024-07-01 filings (outcomes past the approximate Qwen3 training cutoff, unmemorizable), the fulltext-vs-HAR increment is Holm-significant in 5/6 cells and point estimates are LARGER post-cutoff than pre-cutoff in 6/6 cells; beyond-identity post-cutoff is Holm-significant in 3/6 (raw 5/6, positive point estimates 6/6). The one Holm-failure of fulltext-vs-HAR post-cutoff (event_driven h20) is the cell that is already non-significant on the full test set.
- Caveat: the 2024-07-01 boundary is an approximation of the Qwen3 training-data era (Qwen3 released 2025-04; its data cutoff is not publicly dated more precisely). The stratification is conservative in the sense that misplacing the boundary EARLIER only moves memorizable filings into the post stratum, which would bias the post-cutoff increment UP; the pre/post pattern observed (post > pre in 6/6 cells) is inconsistent with memorized-outcome leakage driving the increment.


## Sanity — fulltext rows vs committed crossfamily_llm.csv (qwen3_32b)
max|d rel%|=1.99e-14, max|d DM|=4.44e-16, max|d p|=6.25e-17, n_test mismatches=0 over 6 cells: **PASS**.
