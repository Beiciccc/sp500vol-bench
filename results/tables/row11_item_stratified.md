# Round-3 ROW 11 — 8-K item-code stratification of the surviving event-driven residual

## RESTATED vs BEFORE

| | BEFORE (crossfamily_llm.md / crossfamily_llama70.md) | RESTATED (this table) |
|---|---|---|
| unit of analysis | ONE pooled 8-K firm-ID residual per (family, horizon) | the SAME residual decomposed by 8-K item code (6 disjoint groups) |
| open question | is the increment just the Item-2.02 earnings number? (unanswered) | share of the residual in 2.02 vs narrative items, and whether the earnings-free (narrative_ALL) residual survives |
| combiner / reference / DM | val-fit test-frozen, firm-identity-augmented HAR, day-clustered DM (unchanged) | identical; test residuals PARTITIONED by item code, nothing refit per stratum |
| multiplicity | Holm across families | Holm WITHIN each family over 6 item-groups x 3 horizons = 18 disjoint tests |

## Method & disclosures

- **Panel** event_driven (8-K only, all forms verified == 8-K). **Families** Qwen3-32B (`C6_llmtext`) and matched-class Llama-3.1-70B-AWQ (`C6_llmtext_llama70`).
- **Reference** firm-identity-augmented recalibrated HAR (`R=exp(a+b·logHAR+c·logFirmMeanRV)`, `U=+d·logText`; firm-mean map and both OLS fits estimated on the FULL validation split, applied frozen to test). The single recalibrated-HAR increment (`fc.log_combo`) is carried as `rel_har`/`dm_har`.
- **Item grouping** (earnings-CONCEDING priority): a filing is labelled by the first code present in order 2.02 -> 5.02 -> 7.01 -> 8.01 -> 5.07 -> other. Because Item 2.02 is captured FIRST, the five non-2.02 groups contain NO earnings number, so a surviving increment there cannot be number-parroting. `item_subtype` read directly from predictions.parquet (0 nulls) — no re-join to aligned_filings, row set stays bit-identical to the committed M1 merge.
- **DM** day-clustered on `effective_trading_day`, HAC lag = h-1 DAYS; NEGATIVE stat = text (U) better. **Holm** within each family over the 18 disjoint item-group x horizon cells (ALL and narrative_ALL are derived pooled rows, raw p only, excluded from Holm to avoid double counting).
- **QLIKE unit** every QLIKE (and each `rel%`, `DM`, and `share%` derived from it) is **volatility-unit** (label and forecasts in realised-vol / sigma units via `fc.qlike`) — the SAME convention as the committed crossfamily anchor; there is no variance-unit column in this table.

## Qwen3-32B (C6_llmtext) — item-group x horizon (firm-ID reference)

`**` = clustered DM<0 & Holm p<.05. rel% > 0 = text lowers volatility-unit QLIKE vs the firm-ID reference. share% = signed share of the pooled absolute QLIKE reduction carried by that group.

| item group | kind | h | n_test | n_days | % filings | rel% (firm-ID) | DM(clu) | raw p | Holm p | share% | rel% (vs HAR) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ALL | pooled | 5 | 25109 | 996 | 100.0 | +0.45 | -5.26 | 0.0000 | — | +100.0 | +1.21 |
| 2.02_earnings | earnings | 5 | 8348 | 821 | 33.2 | +0.99** | -5.48 | 0.0000 | 1.016e-06 | +64.7 | +2.89 |
| 5.02_leadership | narrative | 5 | 4706 | 978 | 18.7 | +0.18 | -1.33 | 0.1837 | 1 | +8.2 | +0.37 |
| 7.01_regFD | narrative | 5 | 3343 | 932 | 13.3 | +0.38 | -1.48 | 0.1384 | 0.9689 | +11.2 | +0.71 |
| 8.01_other_events | narrative | 5 | 4951 | 989 | 19.7 | +0.30 | -1.74 | 0.0821 | 0.6564 | +13.9 | +0.86 |
| 5.07_shareholder_vote | procedural | 5 | 1426 | 406 | 5.7 | -0.02 | +0.52 | 0.6019 | 1 | -0.3 | +0.20 |
| other_narrative | narrative | 5 | 2335 | 884 | 9.3 | +0.11 | -0.29 | 0.7686 | 1 | +2.3 | +0.06 |
| narrative_ALL | narrative | 5 | 16761 | 996 | 66.8 | +0.22 | -2.96 | 0.0031 | — | +35.3 | +0.52 |
| ALL | pooled | 10 | 25001 | 991 | 100.0 | +0.25 | -5.16 | 0.0000 | — | +100.0 | +1.00 |
| 2.02_earnings | earnings | 10 | 8331 | 817 | 33.3 | +0.33 | -2.82 | 0.0050 | 0.0646 | +35.3 | +1.63 |
| 5.02_leadership | narrative | 10 | 4681 | 973 | 18.7 | +0.13 | -1.87 | 0.0613 | 0.5518 | +11.0 | +0.41 |
| 7.01_regFD | narrative | 10 | 3319 | 927 | 13.3 | +0.32** | -3.56 | 0.0004 | 0.006255 | +18.1 | +0.84 |
| 8.01_other_events | narrative | 10 | 4920 | 984 | 19.7 | +0.28 | -2.89 | 0.0040 | 0.05574 | +22.6 | +1.16 |
| 5.07_shareholder_vote | procedural | 10 | 1424 | 405 | 5.7 | +0.22 | -1.93 | 0.0546 | 0.5457 | +5.1 | +1.05 |
| other_narrative | narrative | 10 | 2326 | 882 | 9.3 | +0.19 | -1.32 | 0.1875 | 1 | +8.0 | +0.46 |
| narrative_ALL | narrative | 10 | 16670 | 991 | 66.7 | +0.22 | -4.54 | 0.0000 | — | +64.7 | +0.77 |
| ALL | pooled | 20 | 24732 | 981 | 100.0 | +0.20 | -3.79 | 0.0002 | — | +100.0 | +0.66 |
| 2.02_earnings | earnings | 20 | 8305 | 811 | 33.6 | +0.25** | -3.70 | 0.0002 | 0.003856 | +36.3 | +1.40 |
| 5.02_leadership | narrative | 20 | 4610 | 963 | 18.6 | +0.07 | -1.11 | 0.2655 | 1 | +6.9 | -0.13 |
| 7.01_regFD | narrative | 20 | 3269 | 917 | 13.2 | +0.28** | -3.33 | 0.0009 | 0.01336 | +20.2 | +0.69 |
| 8.01_other_events | narrative | 20 | 4842 | 974 | 19.6 | +0.24 | -2.72 | 0.0066 | 0.07314 | +24.9 | +1.07 |
| 5.07_shareholder_vote | procedural | 20 | 1415 | 403 | 5.7 | -0.18 | +1.00 | 0.3192 | 1 | -5.0 | -1.98 |
| other_narrative | narrative | 20 | 2291 | 872 | 9.3 | +0.30 | -2.75 | 0.0060 | 0.07223 | +16.6 | +0.86 |
| narrative_ALL | narrative | 20 | 16427 | 981 | 66.4 | +0.17 | -3.61 | 0.0003 | — | +63.7 | +0.38 |

## Llama-3.1-70B-AWQ (C6_llmtext_llama70) — item-group x horizon (firm-ID reference)

`**` = clustered DM<0 & Holm p<.05. rel% > 0 = text lowers volatility-unit QLIKE vs the firm-ID reference. share% = signed share of the pooled absolute QLIKE reduction carried by that group.

| item group | kind | h | n_test | n_days | % filings | rel% (firm-ID) | DM(clu) | raw p | Holm p | share% | rel% (vs HAR) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ALL | pooled | 5 | 25109 | 996 | 100.0 | +0.83 | -2.58 | 0.0100 | — | +100.0 | +1.39 |
| 2.02_earnings | earnings | 5 | 8348 | 821 | 33.2 | +2.82** | -5.17 | 0.0000 | 5.307e-06 | +99.3 | +4.84 |
| 5.02_leadership | narrative | 5 | 4706 | 978 | 18.7 | +0.13 | -0.25 | 0.8056 | 1 | +3.3 | -0.18 |
| 7.01_regFD | narrative | 5 | 3343 | 932 | 13.3 | +0.29 | -0.79 | 0.4287 | 1 | +4.7 | +0.29 |
| 8.01_other_events | narrative | 5 | 4951 | 989 | 19.7 | +0.01 | +0.10 | 0.9223 | 1 | +0.2 | +0.45 |
| 5.07_shareholder_vote | procedural | 5 | 1426 | 406 | 5.7 | -0.68 | +1.08 | 0.2814 | 1 | -5.1 | -1.06 |
| other_narrative | narrative | 5 | 2335 | 884 | 9.3 | -0.21 | +0.66 | 0.5066 | 1 | -2.4 | -0.55 |
| narrative_ALL | narrative | 5 | 16761 | 996 | 66.8 | +0.01 | -0.19 | 0.8460 | — | +0.7 | -0.03 |
| ALL | pooled | 10 | 25001 | 991 | 100.0 | +0.64 | -2.00 | 0.0460 | — | +100.0 | +1.17 |
| 2.02_earnings | earnings | 10 | 8331 | 817 | 33.3 | +1.30 | -1.60 | 0.1110 | 1 | +55.6 | +3.10 |
| 5.02_leadership | narrative | 10 | 4681 | 973 | 18.7 | +0.11 | -0.14 | 0.8889 | 1 | +3.6 | -0.30 |
| 7.01_regFD | narrative | 10 | 3319 | 927 | 13.3 | +0.85 | -2.52 | 0.0120 | 0.204 | +18.9 | +1.17 |
| 8.01_other_events | narrative | 10 | 4920 | 984 | 19.7 | +0.31 | -0.84 | 0.4029 | 1 | +10.1 | +0.82 |
| 5.07_shareholder_vote | procedural | 10 | 1424 | 405 | 5.7 | +1.05 | -0.10 | 0.9215 | 1 | +9.4 | +1.23 |
| other_narrative | narrative | 10 | 2326 | 882 | 9.3 | +0.14 | -0.14 | 0.8885 | 1 | +2.4 | -0.06 |
| narrative_ALL | narrative | 10 | 16670 | 991 | 66.7 | +0.39 | -1.21 | 0.2252 | — | +44.4 | +0.46 |
| ALL | pooled | 20 | 24732 | 981 | 100.0 | +0.39 | -1.34 | 0.1812 | — | +100.0 | +0.70 |
| 2.02_earnings | earnings | 20 | 8305 | 811 | 33.6 | -0.35 | +1.00 | 0.3180 | 1 | -25.6 | +0.38 |
| 5.02_leadership | narrative | 20 | 4610 | 963 | 18.6 | +0.33 | -0.74 | 0.4581 | 1 | +17.7 | -0.06 |
| 7.01_regFD | narrative | 20 | 3269 | 917 | 13.2 | +0.69 | -2.02 | 0.0436 | 0.6984 | +25.3 | +1.28 |
| 8.01_other_events | narrative | 20 | 4842 | 974 | 19.6 | +0.51 | -1.61 | 0.1087 | 1 | +26.8 | +0.53 |
| 5.07_shareholder_vote | procedural | 20 | 1415 | 403 | 5.7 | +2.99 | -2.00 | 0.0464 | 0.6984 | +42.1 | +4.94 |
| other_narrative | narrative | 20 | 2291 | 872 | 9.3 | +0.49 | -0.52 | 0.6061 | 1 | +13.7 | +0.42 |
| narrative_ALL | narrative | 20 | 16427 | 981 | 66.4 | +0.68 | -2.47 | 0.0137 | — | +125.6 | +0.83 |

## Where the pooled residual lives — Item 2.02 (earnings) vs narrative (summed over horizons)

| family | 2.02 share of filings | 2.02 share of residual | narrative share of residual | narrative_ALL survives (h with +,DM<0,p<.05) |
|---|--:|--:|--:|--:|
| Qwen3-32B (C6_llmtext) | 33% | +54% | +46% | 3/3 |
| Llama-3.1-70B-AWQ (C6_llmtext_llama70) | 33% | +69% | +31% | 1/3 |

## HEADLINE (honest)

- **Qwen3-32B (C6_llmtext)** — MIXED — narrative items carry 46% of the pooled residual and the earnings-free narrative_ALL residual is significant in 3/3 horizons (2/15 narrative partition cells survive Holm). Partly event reading, partly the earnings number.
- **Llama-3.1-70B-AWQ (C6_llmtext_llama70)** — MIXED — narrative items carry 31% of the pooled residual and the earnings-free narrative_ALL residual is significant in 1/3 horizons (0/15 narrative partition cells survive Holm). Partly event reading, partly the earnings number.

## SANITY

Pooled ALL cell reproduces the committed crossfamily anchor to machine precision (rtol 1e-12) on ['rel_har', 'dm_har', 'p_har', 'rel_firm', 'dm_firm', 'p_firm', 'g_text', 'n_test', 'n_days']:
- **qwen3_32b** vs `results/tables/crossfamily_llm.csv`: PASS (h5 rel_firm=+0.4481%, dm_firm=-5.2599, n_test=25109, n_days=996).
- **llama70_awq** vs `results/tables/crossfamily_llama70.csv`: PASS (h5 rel_firm=+0.8321%, dm_firm=-2.5807, n_test=25109, n_days=996).
