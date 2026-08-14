# G3 — return-join drop bias diagnosis

**HEADLINE: the ~8% dropped rows ARE systematically different — the reviewer's low-vol allegation is confirmed on direction (dropped filings mean label vol 0.2074 vs kept 0.2704, -23.3%, Cohen's d=-0.397 on log-vol = small-to-moderate effect) — BUT the headline QLIKE/DM verdicts are UNCHANGED because they NEVER used the return join.** QLIKE is scored against `label_realised_vol`, which requires no return match; `ret_match_ok` only ever gated the economic tests, so a vol-tilted drop cannot touch the QLIKE/DM tables.

- Kept/dropped: **393,845 / 37,400** of 431,245 (**91.33%** kept). Reviewer's ~393,845/37,400 (~92%) target confirmed.


## 1. Are dropped rows systematically different?

### 1a. Volatility level (reviewer alleges dropped = lower-vol)

| group | mean label vol | median label vol | n |
|---|---|---|---|
| ret_match_ok=True | 0.2704 | 0.2249 | 393,845 |
| ret_match_ok=False | 0.2074 | 0.1811 | 37,400 |

Dropped-minus-kept mean = -0.0631 (-23.3% vs kept). Mann-Whitney p=0.00e+00; Welch-t(log-vol) p=0.00e+00; Cohen's d(log-vol)=-0.397. Direction: dropped rows are **LOWER-vol** (reviewer's low-vol allegation confirmed on direction). Effect size small-to-moderate (|d|=0.40), and — decisively — this bias lives ONLY in the economic-test sample, never in the QLIKE/DM tables.

### 1b. Form mix (8-K / 10-Q / 10-K)

| form | kept share | dropped share |
|---|---|---|
| 8-K | 0.7817 | 0.7571 |
| 10-Q | 0.1645 | 0.1746 |
| 10-K | 0.0538 | 0.0683 |

chi2=176.9, p=3.87e-39 (dof=2). Form composition of dropped rows differs from kept but modestly (see shares).

### 1c. Filing year (is the drop concentrated in a period?)

| year | drop rate | n |
|---|---|---|
| 2010 | 0.0762 | 25,393 |
| 2011 | 0.0744 | 28,129 |
| 2012 | 0.0995 | 28,580 |
| 2013 | 0.0997 | 27,735 |
| 2014 | 0.1045 | 27,831 |
| 2015 | 0.0950 | 28,528 |
| 2016 | 0.0910 | 28,016 |
| 2017 | 0.1066 | 27,201 |
| 2018 | 0.0876 | 26,764 |
| 2019 | 0.0990 | 26,210 |
| 2020 | 0.0590 | 28,276 |
| 2021 | 0.0786 | 26,138 |
| 2022 | 0.0723 | 24,640 |
| 2023 | 0.0892 | 24,639 |
| 2024 | 0.0818 | 24,649 |
| 2025 | 0.0753 | 24,700 |

Spearman(drop-rate, year)=-0.338, p=0.200. No significant time trend — ticker-recycling/dual-class mismatches are broadly spread, not a single-period artifact.

### 1d. n_days coverage

| group | mean n_days | median n_days |
|---|---|---|
| ret_match_ok=True | 11.53 | 10.0 |
| ret_match_ok=False | 12.97 | 10.0 |

Mann-Whitney p=0.00e+00; dropped-minus-kept mean n_days=+1.443.


## 2. IMPACT: headline QLIKE/DM verdicts, ALL rows vs kept-only

The QLIKE/DM headline tables score predictions against `label_realised_vol` (present in every predictions.parquet, no return join). `ret_match_ok` is a column of `_realized_returns.parquet` that only the ECONOMIC tests read. To prove the drop is inert for the headline, we replicate the exact M1 log-space nested combination (`fc.log_combo`): fit restricted (HAR-only recalibration) and unrestricted (HAR+text) on VAL, score QLIKE on TEST, then run clustered DM with lossA=U(HAR+text), lossB=R(HAR-only), so **dm_q<0 = text genuinely improves** (matches m1_clustered). We refit on ALL rows and on `ret_match_ok=True` rows.

| cell | subset | n_obs | n_days | rel impr % | dm_q clust | p_q clust | genuine? |
|---|---|---|---|---|---|---|---|
| long_form|C2_finbert_s1|h10 | ALL_rows | 7,933 | 803 | +4.558 | -6.461 | 1.80e-10 | YES |
| long_form|C2_finbert_s1|h10 | ret_match_ok_only | 7,167 | 766 | +4.413 | -5.912 | 5.08e-09 | YES |
| long_form|B2_tfidf_ridge|h20 | ALL_rows | 7,902 | 794 | +5.920 | -9.042 | 1.17e-18 | YES |
| long_form|B2_tfidf_ridge|h20 | ret_match_ok_only | 7,097 | 765 | +5.270 | -9.320 | 1.22e-19 | YES |
| event_driven|C6_llmtext|h5 | ALL_rows | 25,109 | 996 | +1.214 | -5.044 | 5.42e-07 | YES |
| event_driven|C6_llmtext|h5 | ret_match_ok_only | 23,855 | 996 | +1.202 | -4.800 | 1.83e-06 | YES |

Every cell keeps the SAME sign and the SAME genuine verdict on ALL rows vs kept-only; rel-improvement and dm_q shift only marginally. The ~8% dropped from the economic tests do **not** drive the QLIKE/DM conclusions.
