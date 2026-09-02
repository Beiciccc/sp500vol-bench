# Cross-family table REPAIR — standalone quality + mode-collapse diagnostics (P0-3a)

## RESTATED vs BEFORE

| | BEFORE (crossfamily_llm.md) | RESTATED (this table) |
|---|---|---|
| headline | "the prompted-LLM residual is **family-specific**" | "the increment **does not replicate in the one additional family tested (Yi-1.5-34B)**; at n=2 with the second family capability-floored at the task, family-specificity is **unidentified** (capability-confounded)" |
| standalone quality | absent (hidden confound) | per-cell test QLIKE (vol + variance unit), R^2, prediction sd, n_unique / modal-share of round(pred,2) |
| Yi capability floor | invisible | variance-unit QLIKE 5.25-8.19 vs Qwen 0.93-1.32; event_driven h=5 mode-collapse: Yi 73.6% of test predictions at 0.15 vs Qwen 49.3% at 0.22 |
| Yi long_form rows | in the main table | **demoted: 4K-context-truncated** (Yi ctx 4K < ~6K-token excerpt cap; Qwen ran 8K) — cite only event_driven (median 8-K ~930 tokens, context not binding) for the family claim |

M1 columns (rel% / day-clustered DM vs single recalibrated HAR and vs the firm-identity-augmented reference) are carried over UNCHANGED from crossfamily_llm.csv; `**` = clustered DM<0, p<.05. Standalone columns are computed on the TEST split of each run's predictions.parquet; the variance-unit QLIKE is cross-checked against the stored metrics.json (sanity column). `combined` M1 cells were not part of the original cross-family grid (blank).

| disc | family | h | n_test | QLIKE(vol) | QLIKE(var) | R^2 | pred sd | n_uniq(2dp) | mode@2dp | mode share% | ctx flag | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|
| event_driven | qwen3_32b | 5 | 25109 | 0.1967 | 1.177 | -0.121 | 0.0323 | 10 | 0.22 | 49.3 | - | +1.21%** | -5.04 | +0.45%** | -5.26 |
| event_driven | qwen3_32b | 10 | 25001 | 0.1914 | 1.214 | -0.244 | 0.0243 | 19 | 0.20 | 50.3 | - | +1.00%** | -3.76 | +0.25%** | -5.16 |
| event_driven | qwen3_32b | 20 | 24732 | 0.2097 | 1.316 | -0.476 | 0.0214 | 16 | 0.18 | 49.2 | - | +0.66%** | -1.98 | +0.20%** | -3.79 |
| event_driven | yi_34b | 5 | 25109 | 0.7974 | 8.188 | -0.696 | 0.0405 | 18 | 0.15 | 73.6 | - | +0.37% | -0.60 | +0.22% | -1.00 |
| event_driven | yi_34b | 10 | 25001 | 0.7242 | 7.605 | -0.717 | 0.0473 | 23 | 0.17 | 58.6 | - | +0.07% | +0.60 | +0.07% | -0.13 |
| event_driven | yi_34b | 20 | 24732 | 0.7025 | 7.873 | -0.813 | 0.0560 | 25 | 0.19 | 55.9 | - | -0.62% | +2.54 | -0.07% | +1.58 |
| long_form | qwen3_32b | 5 | 7951 | 0.2072 | 1.226 | -0.174 | 0.0274 | 8 | 0.18 | 64.4 | - | +1.79%** | -6.31 | -0.14% | +5.16 |
| long_form | qwen3_32b | 10 | 7933 | 0.1778 | 1.095 | -0.201 | 0.0171 | 13 | 0.20 | 76.8 | - | +2.25%** | -7.92 | -0.17% | +6.83 |
| long_form | qwen3_32b | 20 | 7902 | 0.1601 | 0.926 | -0.345 | 0.0239 | 16 | 0.22 | 45.5 | - | +0.27%** | -3.23 | +0.02%** | -3.79 |
| long_form | yi_34b | 5 | 7951 | 0.6978 | 6.204 | -0.791 | 0.0336 | 18 | 0.15 | 56.6 | 4K-TRUNCATED | -0.64% | +2.56 | -0.21% | +1.83 |
| long_form | yi_34b | 10 | 7933 | 0.6562 | 5.851 | -0.766 | 0.0431 | 23 | 0.18 | 28.6 | 4K-TRUNCATED | -2.71% | +4.27 | -1.61% | +4.38 |
| long_form | yi_34b | 20 | 7902 | 0.6102 | 5.246 | -1.066 | 0.0564 | 26 | 0.21 | 26.1 | 4K-TRUNCATED | -9.86% | +5.68 | -6.05% | +5.55 |
| combined | qwen3_32b | 5 | 33060 | 0.1992 | 1.188 | -0.133 | 0.0315 | 10 | 0.18 | 48.3 | - | - | - | - | - |
| combined | qwen3_32b | 10 | 32934 | 0.1881 | 1.186 | -0.233 | 0.0228 | 20 | 0.20 | 56.7 | - | - | - | - | - |
| combined | qwen3_32b | 20 | 32634 | 0.1977 | 1.221 | -0.447 | 0.0238 | 19 | 0.18 | 44.7 | - | - | - | - | - |
| combined | yi_34b | 5 | 33060 | 0.7735 | 7.711 | -0.717 | 0.0389 | 22 | 0.15 | 69.5 | PARTIAL(4K) | - | - | - | - |
| combined | yi_34b | 10 | 32934 | 0.7078 | 7.182 | -0.730 | 0.0464 | 24 | 0.17 | 48.5 | PARTIAL(4K) | - | - | - | - |
| combined | yi_34b | 20 | 32634 | 0.6802 | 7.237 | -0.865 | 0.0565 | 26 | 0.19 | 46.4 | PARTIAL(4K) | - | - | - | - |

## Honest reading (replaces the "family-specific" headline)

- **The Qwen3-32B increment does not replicate in Yi-1.5-34B** on event_driven, the only disclosure where the comparison is context-clean (all Yi rel% ~0, no cell significant).
- **But the comparison is capability-confounded at n=2**: Yi's standalone forecasts are broken at the task (variance-unit QLIKE 5.25-8.19 vs Qwen 0.93-1.32; R^2 -1.07--0.70 vs Qwen -0.48--0.12; up to 73.6% of Yi test predictions collapse onto a single rounded value). A model that cannot produce a calibrated standalone forecast cannot be evidence that the *signal* is family-specific — only that it fails to replicate in this family.
- **Correct claim for the paper**: "the prompted-LLM increment does not replicate in a second model family; because the second family is capability-floored at the task, the test is confounded and family-specificity remains unidentified at n=2." Do NOT write "family-specific".
- **Yi long_form (and the long-form subset of combined) is additionally 4K-context-truncated** and must not be cited at all for the family claim; the citable cells are the six event_driven rows.

Sanity: recomputed variance-unit QLIKE matches metrics.json within 1e-3 relative in 18/18 cells.
