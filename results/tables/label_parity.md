# ROW 10 — Public-price label-parity study (licence-free benchmark variant feasibility)

**RESTATED vs BEFORE**
- **BEFORE:** the paper ships the benchmark with CRSP-derived realised-volatility labels withheld under licence; reviewers (eic W5, domain W8, perspective W4/MAJ; freeze-table row 10) hold that most of the community cannot run the benchmark, and no evidence existed on whether a licence-free label variant would preserve the paper's verdicts. The task brief assumed `market/full_ohlcv.parquet` was a public OHLCV source; this study **checked and refuted** that premise (see SANITY) — the repo contains no stored public price source at all.
- **RESTATED:** labels were recomputed from a genuinely public source (Yahoo Finance daily adjusted close, fetched 2026-07-09) on the benchmark's exact per-row label windows, and parity was quantified three ways: label correlation, coverage/survivorship, and verdict preservation (QLIKE rankings, day-clustered DM signs, and the M1 combination increment) on stored forecasts. Headline numbers below; honest verdict at the end.

**Declared exceptions / disclosures**
- Public source: Yahoo v8 chart API adj-close (split+dividend adjusted — same total-return basis as CRSP `DlyRet`). Fetched 2026-07-09; snapshot cached at `<scratchpad>/label_parity/yahoo_adjclose.parquet`. Yahoo terms permit personal research use but NOT redistribution — a shipped variant would use Stooq/EODHD-style redistributable data; Yahoo here measures *parity feasibility*, not the final distribution channel.
- Public-side returns that span a data gap (previous trading day missing) are masked before label construction; windows containing masked/missing days yield NO public label (counted as coverage failure, never as a wrong label). The CRSP side replicates the original alignment verbatim (no masking) — that is what the sanity gate certifies.
- Symbol-mismatch screen: tickers whose public daily returns correlate < 0.8 with CRSP returns on >= 60 overlap days are treated as NOT covered (Yahoo reuses point-in-time symbols for different firms). This screen itself needs CRSP — a licence-free builder could not run it, which is part of the verdict.
- No look-ahead anywhere: combination weights are fit on each panel's validation rows only and frozen on test (row 1's oracle exception does not apply to this row).
- UNITS: every QLIKE in this file is computed on the **volatility scale** (annualised realised vol vs vol forecasts, `fc.qlike`), the same convention as the committed `forecast_combination_grid.csv` it is gated against. Variance-unit QLIKE is treated separately in the row-5 variance-unit cascade (`scripts/analysis/variance_unit_cascade.py`).

**PRE-DECLARED HOLM FAMILIES** (declared before any result table)
- `F-STAND-P` (P in A/B/C): 18 day-clustered DM tests (2 disclosures x 3 models x 3 horizons), text model vs A2, per label panel; Holm within each family.
- `F-COMBO-P` (P in A/B/C): 12 day-clustered DM tests (2 models x 2 disclosures x 3 horizons), f_U vs f_R log-space combination increment, per label panel; Holm within each family. 'genuine' = DM<0 AND Holm<.05 AND |placebo DM|<2 (repo convention).
- Panels: A = full test panel + CRSP labels (paper verdict); B = public-coverage intersection + CRSP labels (isolates survivorship); C = same intersection + PUBLIC labels (adds label measurement error).

## SANITY

| check | result |
|---|---|
| Provenance: `market/full_ohlcv.parquet` vs CRSP store | joined 2,023,428 rows; max abs diff (adj_close/close/volume) = 0.0/0.0/0.0 -> **it IS the CRSP cache, not public** |
| GATE 1: CRSP labels recomputed on exact aligned windows == `aligned_filings.label_realised_vol` | n=431,245, unreconstructed=0, max abs diff=0.000e+00, max rel diff=0.000e+00, bitwise exact=True — **PASS** |
| GATE 1b: `predictions.parquet` labels == aligned labels on modelled panel | 427,429 rows, max abs diff=0.0 — **PASS** |
| GATE 2: A2 test QLIKE recomputed vs committed `forecast_combination_grid.csv` (qlike_raw, C2 cells) | max rel diff=9.356e-16 — **PASS** |
| GATE 3: every aligned label window spans exactly `horizon_days` NYSE trading days | asserted in `compute_labels` — **PASS** (script would have aborted) |

## (b) Coverage — who is missing

Tickers: 848 in benchmark; Yahoo returns data for 648 (76.4%); no public data for 200; symbol-mismatch (screened out) = 18. Benchmark-row coverage: raw 80.80%, clean (mismatch screened) **80.19%**.

Failure reasons (benchmark rows):

| reason | rows | share |
|---|---|---|
| covered | 345,804 | 80.19% |
| no_public_data | 68,497 | 15.88% |
| window_incomplete | 13,564 | 3.15% |
| symbol_mismatch | 3,380 | 0.78% |

By firm exit-year (last CRSP trading day; 'active' = still listed 2025-12):

| exit year | firms | rows | clean coverage | firms w/o public data | firms mismatched |
|---|---|---|---|---|---|
| 2010 | 16 | 524 | 9.7% | 10 | 2 |
| 2011 | 24 | 2,052 | 9.0% | 14 | 5 |
| 2012 | 18 | 2,594 | 3.7% | 7 | 4 |
| 2013 | 19 | 3,759 | 16.8% | 11 | 3 |
| 2014 | 14 | 3,579 | 19.6% | 8 | 0 |
| 2015 | 28 | 7,946 | 13.5% | 22 | 0 |
| 2016 | 27 | 8,376 | 26.1% | 13 | 0 |
| 2017 | 29 | 10,402 | 24.9% | 18 | 2 |
| 2018 | 28 | 10,973 | 28.2% | 14 | 1 |
| 2019 | 28 | 10,976 | 28.0% | 15 | 1 |
| 2020 | 20 | 10,122 | 38.5% | 12 | 0 |
| 2021 | 21 | 9,127 | 50.7% | 10 | 0 |
| 2022 | 22 | 8,908 | 12.2% | 16 | 0 |
| 2023 | 15 | 6,669 | 50.0% | 7 | 0 |
| 2024 | 20 | 9,859 | 58.7% | 8 | 0 |
| 2025 | 17 | 8,793 | 47.2% | 8 | 0 |
| active | 502 | 316,586 | 97.7% | 7 | 0 |

Exit-firm (delisted/acquired) rows: 114,659 (26.6% of benchmark); their clean coverage = **31.9%** vs 97.7% for active firms.

Coverage by filing year (clean): 2010: 62.6%, 2011: 65.0%, 2012: 65.8%, 2013: 68.4%, 2014: 69.9%, 2015: 73.0%, 2016: 77.6%, 2017: 80.0%, 2018: 83.3%, 2019: 85.3%, 2020: 88.7%, 2021: 90.5%, 2022: 92.8%, 2023: 94.9%, 2024: 96.1%, 2025: 97.7%

## (a) Label parity — log-RV correlation on the modelled panel (covered rows)

Overall (clean): n=343,239, Pearson(log RV)=**0.9981**, Spearman=0.9982, mean dlog=+0.0005, sd dlog=0.0328. Raw incl. mismatched symbols: Pearson=0.9714 (the screen matters).

| split | h | n | Pearson | Spearman | mean dlog | sd dlog |
|---|---|---|---|---|---|---|
| train | 5 | 66,847 | 0.9966 | 0.9968 | +0.0005 | 0.0467 |
| train | 10 | 66,815 | 0.9966 | 0.9967 | +0.0007 | 0.0409 |
| train | 20 | 66,758 | 0.9961 | 0.9971 | +0.0010 | 0.0395 |
| val | 5 | 16,258 | 0.9998 | 0.9997 | +0.0003 | 0.0148 |
| val | 10 | 16,247 | 0.9999 | 0.9999 | +0.0003 | 0.0100 |
| val | 20 | 16,216 | 1.0000 | 1.0000 | +0.0002 | 0.0053 |
| test | 5 | 31,524 | 1.0000 | 1.0000 | +0.0001 | 0.0045 |
| test | 10 | 31,417 | 1.0000 | 1.0000 | +0.0001 | 0.0041 |
| test | 20 | 31,157 | 1.0000 | 0.9999 | +0.0001 | 0.0033 |

By year (all splits): 2010: 0.995, 2011: 0.996, 2012: 0.995, 2013: 0.996, 2014: 0.995, 2015: 0.994, 2016: 0.996, 2017: 0.998, 2018: 0.998, 2019: 0.998, 2020: 1.000, 2021: 1.000, 2022: 1.000, 2023: 1.000, 2024: 1.000, 2025: 1.000.  Full split x h x year grid in the CSV.

## (c) Verdict preservation — stored A2 vs text models, three panels

### Standalone day-clustered DM (text vs A2; + = text worse). Holm within F-STAND-P.

| disc | model | h | A: DM (Holm) | B: DM (Holm) | C: DM (Holm) | sign A=B=C | Holm-sig A=B=C |
|---|---|---|---|---|---|---|---|
| long_form | B2_tfidf_ridge | 5 | +8.10 (0.000) | +7.34 (0.000) | +7.34 (0.000) | YES | YES |
| long_form | B2_tfidf_ridge | 10 | +6.13 (0.000) | +5.63 (0.000) | +5.63 (0.000) | YES | YES |
| long_form | B2_tfidf_ridge | 20 | +4.55 (0.000) | +4.25 (0.000) | +4.25 (0.000) | YES | YES |
| long_form | C2_finbert_s1 | 5 | +6.52 (0.000) | +5.98 (0.000) | +5.99 (0.000) | YES | YES |
| long_form | C2_finbert_s1 | 10 | +6.52 (0.000) | +6.19 (0.000) | +6.19 (0.000) | YES | YES |
| long_form | C2_finbert_s1 | 20 | +6.11 (0.000) | +5.78 (0.000) | +5.78 (0.000) | YES | YES |
| long_form | C6_llmtext | 5 | +8.52 (0.000) | +8.15 (0.000) | +8.15 (0.000) | YES | YES |
| long_form | C6_llmtext | 10 | +7.13 (0.000) | +6.83 (0.000) | +6.83 (0.000) | YES | YES |
| long_form | C6_llmtext | 20 | +6.54 (0.000) | +6.19 (0.000) | +6.19 (0.000) | YES | YES |
| event_driven | B2_tfidf_ridge | 5 | +11.36 (0.000) | +11.16 (0.000) | +11.16 (0.000) | YES | YES |
| event_driven | B2_tfidf_ridge | 10 | +9.39 (0.000) | +9.24 (0.000) | +9.25 (0.000) | YES | YES |
| event_driven | B2_tfidf_ridge | 20 | +7.75 (0.000) | +7.67 (0.000) | +7.67 (0.000) | YES | YES |
| event_driven | C2_finbert_s1 | 5 | +8.29 (0.000) | +7.87 (0.000) | +7.88 (0.000) | YES | YES |
| event_driven | C2_finbert_s1 | 10 | +8.50 (0.000) | +8.30 (0.000) | +8.31 (0.000) | YES | YES |
| event_driven | C2_finbert_s1 | 20 | +7.54 (0.000) | +7.28 (0.000) | +7.29 (0.000) | YES | YES |
| event_driven | C6_llmtext | 5 | +9.75 (0.000) | +9.37 (0.000) | +9.37 (0.000) | YES | YES |
| event_driven | C6_llmtext | 10 | +9.13 (0.000) | +8.86 (0.000) | +8.86 (0.000) | YES | YES |
| event_driven | C6_llmtext | 20 | +8.89 (0.000) | +8.67 (0.000) | +8.67 (0.000) | YES | YES |

Panel B vs A: sign agreement 18/18, Holm-significance agreement 18/18, full agreement 18/18.

Panel C vs A: sign agreement 18/18, Holm-significance agreement 18/18, full agreement 18/18.

### QLIKE ranking per disclosure x horizon (models incl. A2)

| disc | h | panel A | panel B | panel C | B==A | C==A |
|---|---|---|---|---|---|---|
| long_form | 5 | A2_har_rv < C2_finbert_s1 < C6_llmtext < B2_tfidf_ridge | A2_har_rv < C2_finbert_s1 < C6_llmtext < B2_tfidf_ridge | A2_har_rv < C2_finbert_s1 < C6_llmtext < B2_tfidf_ridge | YES | YES |
| long_form | 10 | A2_har_rv < B2_tfidf_ridge < C2_finbert_s1 < C6_llmtext | A2_har_rv < C2_finbert_s1 < B2_tfidf_ridge < C6_llmtext | A2_har_rv < C2_finbert_s1 < B2_tfidf_ridge < C6_llmtext | NO | NO |
| long_form | 20 | A2_har_rv < B2_tfidf_ridge < C6_llmtext < C2_finbert_s1 | A2_har_rv < B2_tfidf_ridge < C6_llmtext < C2_finbert_s1 | A2_har_rv < B2_tfidf_ridge < C6_llmtext < C2_finbert_s1 | YES | YES |
| event_driven | 5 | A2_har_rv < C2_finbert_s1 < C6_llmtext < B2_tfidf_ridge | A2_har_rv < C2_finbert_s1 < C6_llmtext < B2_tfidf_ridge | A2_har_rv < C2_finbert_s1 < C6_llmtext < B2_tfidf_ridge | YES | YES |
| event_driven | 10 | A2_har_rv < C6_llmtext < C2_finbert_s1 < B2_tfidf_ridge | A2_har_rv < C6_llmtext < C2_finbert_s1 < B2_tfidf_ridge | A2_har_rv < C6_llmtext < C2_finbert_s1 < B2_tfidf_ridge | YES | YES |
| event_driven | 20 | A2_har_rv < C2_finbert_s1 < B2_tfidf_ridge < C6_llmtext | A2_har_rv < C2_finbert_s1 < B2_tfidf_ridge < C6_llmtext | A2_har_rv < C2_finbert_s1 < B2_tfidf_ridge < C6_llmtext | YES | YES |

Ranking identical to panel A: B 5/6, C 5/6.

### M1 combination increment (f_U vs f_R, log space, val-frozen weights). Holm within F-COMBO-P.

| disc | model | h | panel | rel impr % | DM | p | Holm | placebo DM | genuine |
|---|---|---|---|---|---|---|---|---|---|
| event_driven | C2_finbert_s1 | 5 | A | +2.14 | -4.95 | 0.0000 | 0.000 | -0.94 | YES |
| event_driven | C2_finbert_s1 | 5 | B | +2.10 | -5.43 | 0.0000 | 0.000 | +2.06 | no |
| event_driven | C2_finbert_s1 | 5 | C | +2.10 | -5.42 | 0.0000 | 0.000 | +2.06 | no |
| event_driven | C2_finbert_s1 | 10 | A | +2.10 | -5.52 | 0.0000 | 0.000 | -0.16 | YES |
| event_driven | C2_finbert_s1 | 10 | B | +2.01 | -5.74 | 0.0000 | 0.000 | -0.59 | YES |
| event_driven | C2_finbert_s1 | 10 | C | +2.01 | -5.73 | 0.0000 | 0.000 | -0.59 | YES |
| event_driven | C2_finbert_s1 | 20 | A | +0.92 | -0.50 | 0.6206 | 1.000 | -0.47 | no |
| event_driven | C2_finbert_s1 | 20 | B | +0.97 | -0.48 | 0.6328 | 1.000 | -0.07 | no |
| event_driven | C2_finbert_s1 | 20 | C | +0.95 | -0.46 | 0.6454 | 1.000 | -0.07 | no |
| event_driven | C6_llmtext | 5 | A | +1.21 | -5.04 | 0.0000 | 0.000 | +0.10 | YES |
| event_driven | C6_llmtext | 5 | B | +1.56 | -6.02 | 0.0000 | 0.000 | +0.92 | YES |
| event_driven | C6_llmtext | 5 | C | +1.56 | -6.00 | 0.0000 | 0.000 | +0.93 | YES |
| event_driven | C6_llmtext | 10 | A | +1.00 | -3.76 | 0.0002 | 0.001 | +0.95 | YES |
| event_driven | C6_llmtext | 10 | B | +1.35 | -4.43 | 0.0000 | 0.000 | -0.34 | YES |
| event_driven | C6_llmtext | 10 | C | +1.34 | -4.39 | 0.0000 | 0.000 | -0.34 | YES |
| event_driven | C6_llmtext | 20 | A | +0.66 | -1.98 | 0.0477 | 0.191 | +1.11 | no |
| event_driven | C6_llmtext | 20 | B | +0.77 | -1.97 | 0.0495 | 0.198 | +0.64 | no |
| event_driven | C6_llmtext | 20 | C | +0.76 | -1.94 | 0.0521 | 0.208 | +0.65 | no |
| long_form | C2_finbert_s1 | 5 | A | +0.56 | -0.84 | 0.4014 | 1.000 | +0.22 | no |
| long_form | C2_finbert_s1 | 5 | B | +0.24 | -1.73 | 0.0834 | 0.250 | +0.30 | no |
| long_form | C2_finbert_s1 | 5 | C | +0.24 | -1.72 | 0.0851 | 0.255 | +0.29 | no |
| long_form | C2_finbert_s1 | 10 | A | +4.56 | -6.46 | 0.0000 | 0.000 | +1.01 | YES |
| long_form | C2_finbert_s1 | 10 | B | +4.96 | -6.71 | 0.0000 | 0.000 | +1.01 | YES |
| long_form | C2_finbert_s1 | 10 | C | +4.95 | -6.71 | 0.0000 | 0.000 | +1.01 | YES |
| long_form | C2_finbert_s1 | 20 | A | -0.08 | +0.88 | 0.3769 | 1.000 | -0.52 | no |
| long_form | C2_finbert_s1 | 20 | B | -0.13 | -0.18 | 0.8549 | 1.000 | +0.07 | no |
| long_form | C2_finbert_s1 | 20 | C | -0.13 | -0.18 | 0.8561 | 1.000 | +0.07 | no |
| long_form | C6_llmtext | 5 | A | +1.79 | -6.31 | 0.0000 | 0.000 | +0.83 | YES |
| long_form | C6_llmtext | 5 | B | +2.08 | -7.31 | 0.0000 | 0.000 | +0.75 | YES |
| long_form | C6_llmtext | 5 | C | +2.08 | -7.32 | 0.0000 | 0.000 | +0.75 | YES |
| long_form | C6_llmtext | 10 | A | +2.25 | -7.92 | 0.0000 | 0.000 | -1.45 | YES |
| long_form | C6_llmtext | 10 | B | +2.53 | -8.38 | 0.0000 | 0.000 | +0.83 | YES |
| long_form | C6_llmtext | 10 | C | +2.53 | -8.38 | 0.0000 | 0.000 | +0.83 | YES |
| long_form | C6_llmtext | 20 | A | +0.27 | -3.23 | 0.0013 | 0.006 | -1.44 | YES |
| long_form | C6_llmtext | 20 | B | +0.47 | -3.40 | 0.0007 | 0.003 | +0.52 | YES |
| long_form | C6_llmtext | 20 | C | +0.47 | -3.41 | 0.0007 | 0.003 | +0.52 | YES |

Combo panel B vs A: sign agreement 11/12, Holm-significance agreement 12/12, full agreement 11/12.

Combo panel C vs A: sign agreement 11/12, Holm-significance agreement 12/12, full agreement 11/12.

'Genuine increment' cells (of 12 per panel): A=8, B=7, C=7.

## (d) Honest verdict

A licence-free label variant is **faithful where it exists and breaks exactly where expected — survivorship of the public source**. On covered rows the public labels are near-duplicates of the CRSP labels (log-RV Pearson 0.998); every verdict object we replicated — QLIKE rankings (5/6 identical under public labels), standalone DM signs (18/18 under public labels), and the M1 combination-increment verdicts (A=8 vs C=7 genuine cells) — is materially preserved, so a non-subscriber re-running the evaluation layer on public labels would reach the paper's conclusions. The failure mode is coverage, not correlation: 19.8% of benchmark rows have no clean public label, and the loss concentrates in delisted/acquired firms (31.9% exit-firm coverage vs 97.7% for active firms) plus point-in-time symbols Yahoo has recycled (18 tickers screened only because CRSP was available to screen against). A shipped licence-free variant therefore (i) is a mildly survivorship-tilted subsample, not the benchmark, and must be labelled as such; (ii) needs a redistributable source (Yahoo terms bar redistribution) and a delisting-aware symbol map; (iii) should ship the panel-B/panel-C agreement tables above as its calibration certificate. Full-cascade replication on free labels remains the run-if-time follow-up.

---
*Script: `scripts/analysis/label_parity.py`; public snapshot fetched 2026-07-09; CSV companion: `results/tables/label_parity.csv` (sections: provenance, sanity_gate, ticker_status, coverage*, parity_corr, verdict_*).*