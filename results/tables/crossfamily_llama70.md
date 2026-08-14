# Cross-family replication, round-3 row 4 — matched-class Llama-3.1-70B (AWQ-INT4) on the 8-K channel

## RESTATED vs BEFORE

| | BEFORE (crossfamily_llm.md / crossfamily_standalone.md) | RESTATED (this table) |
|---|---|---|
| replication gate | Yi-1.5-34B + Phi-4-14B: smaller, older, mode-collapsed — "does not replicate" was capability-confounded, family-specificity unidentified at n=2 (round-3 DA-CRITICAL #4) | matched-class, different-lineage Meta-Llama-3.1-70B-Instruct (int4 AWQ), identical manifest/prompts/protocol, on the context-clean 8-K channel |
| verdict on the residual | uninformative gate | DIRECTIONALLY REPLICATES, significance attenuated. The matched-class Llama-3.1-70B reproduces the SIGN of the Qwen 8-K residual vs HAR+firmID in 3/3 horizons with point estimates larger than Qwen's (+0.83/+0.64/+0.39% vs +0.45/+0.25/+0.20%), clustered DM<0 in 3/3 and raw p<.05 in 2/3; but after the pre-declared Holm(6) only 1/3 vs-single-HAR cells survive (min firmID Holm p=0.05001, 0/3 firmID cells <.05). The residual is NOT Qwen-specific — a healthy matched-class family recovers same-sign, same-or-larger increments — but it is not fully Holm-robust in the second family either. |
| 70B standalone health | n/a | variance-unit QLIKE 1.12-2.10 vs Qwen 1.18-1.32, Yi 7.60-8.19; max modal share 51.2% vs Yi 73.6% |

## Disclosures

- **Quantisation**: llama70 = hugging-quants/Meta-Llama-3.1-70B-Instruct-**AWQ-INT4** (weight-only int4). All other families ran full/bf16 weights.
- **long_form was NOT run for llama70** (GPU budget; 8-K is the citable channel per the round-3 panel — long-form was context-confounded for Yi anyway). Its cells are therefore absent, exactly as phi4's are.
- **The C6_llmtext_llama70_full_combined_seed2026 run is a relabelled duplicate of the event_driven panel** (verified in-script: same 117,407 rows, all 8-K, predictions bit-identical) — with no long-form forecasts a "combined" pass degenerates to the 8-K subset. It is excluded; it carries no combined-disclosure information.
- parse_fail_rate = 0.0 and clipped_rate = 0.0 for llama70 (config.json stats); Yi clipped 0.79%.
- Holm is applied within the pre-declared family of the 6 NEW llama70 tests (3 horizons x {vs HAR, vs HAR+firmID}); qwen/yi/phi p-values are raw, carried unchanged from the committed tables.

## Table — M1 increment (log-space, combiner val-fit test-frozen, day-clustered DM) + standalone health

rel% is on **volatility-unit** QLIKE (the convention of the committed crossfamily_llm.csv anchor cells); the QLIKE(var) health column is **variance-unit** (Patton-robust convention). rel% > 0 = text lowers QLIKE vs the reference; `**` = clustered DM<0, raw p<.05.

| disc | family | h | n_test | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) | Holm p (firmID) | QLIKE(var) | R^2 | pred sd | n_uniq(2dp) | mode share% | parse_ok% | flag |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| long_form | qwen3_32b | 5 | 7951 | +1.79%** | -6.31 | -0.14% | +5.16 | - | 1.226 | -0.174 | 0.0274 | 8 | 64.4 | 100.0 | - |
| long_form | qwen3_32b | 10 | 7933 | +2.25%** | -7.92 | -0.17% | +6.83 | - | 1.095 | -0.201 | 0.0171 | 13 | 76.8 | 100.0 | - |
| long_form | qwen3_32b | 20 | 7902 | +0.27%** | -3.23 | +0.02%** | -3.79 | - | 0.926 | -0.345 | 0.0239 | 16 | 45.5 | 100.0 | - |
| long_form | yi_34b | 5 | 7951 | -0.64% | +2.56 | -0.21% | +1.83 | - | 6.204 | -0.791 | 0.0336 | 18 | 56.6 | 100.0 | 4K-TRUNCATED |
| long_form | yi_34b | 10 | 7933 | -2.71% | +4.27 | -1.61% | +4.38 | - | 5.851 | -0.766 | 0.0431 | 23 | 28.6 | 100.0 | 4K-TRUNCATED |
| long_form | yi_34b | 20 | 7902 | -9.86% | +5.68 | -6.05% | +5.55 | - | 5.246 | -1.066 | 0.0564 | 26 | 26.1 | 100.0 | 4K-TRUNCATED |
| event_driven | qwen3_32b | 5 | 25109 | +1.21%** | -5.04 | +0.45%** | -5.26 | - | 1.177 | -0.121 | 0.0323 | 10 | 49.3 | 100.0 | - |
| event_driven | qwen3_32b | 10 | 25001 | +1.00%** | -3.76 | +0.25%** | -5.16 | - | 1.214 | -0.244 | 0.0243 | 19 | 50.3 | 100.0 | - |
| event_driven | qwen3_32b | 20 | 24732 | +0.66%** | -1.98 | +0.20%** | -3.79 | - | 1.316 | -0.476 | 0.0214 | 16 | 49.2 | 100.0 | - |
| event_driven | yi_34b | 5 | 25109 | +0.37% | -0.60 | +0.22% | -1.00 | - | 8.188 | -0.696 | 0.0405 | 18 | 73.6 | 100.0 | - |
| event_driven | yi_34b | 10 | 25001 | +0.07% | +0.60 | +0.07% | -0.13 | - | 7.605 | -0.717 | 0.0473 | 23 | 58.6 | 100.0 | - |
| event_driven | yi_34b | 20 | 24732 | -0.62% | +2.54 | -0.07% | +1.58 | - | 7.873 | -0.813 | 0.0560 | 25 | 55.9 | 100.0 | - |
| event_driven | phi4_14b | 5 | 25109 | +0.38% | -1.96 | +0.20% | -1.68 | - | 0.677 | -0.122 | 0.0477 | 6 | 72.9 | 100.0 | - |
| event_driven | phi4_14b | 10 | 25001 | +0.18% | -0.38 | +0.14% | -1.16 | - | 0.578 | -0.084 | 0.0460 | 8 | 75.1 | 100.0 | - |
| event_driven | phi4_14b | 20 | 24732 | -0.12% | +1.22 | +0.00% | -0.07 | - | 0.533 | -0.254 | 0.0665 | 8 | 61.3 | 100.0 | - |
| event_driven | llama70_awq | 5 | 25109 | +1.39%** | -3.22 | +0.83%** | -2.58 | 0.05001 | 1.122 | -0.112 | 0.0873 | 10 | 51.2 | 100.0 | AWQ-INT4 |
| event_driven | llama70_awq | 10 | 25001 | +1.17%** | -2.35 | +0.64%** | -2.00 | 0.1379 | 1.403 | -0.215 | 0.0718 | 15 | 40.4 | 100.0 | AWQ-INT4 |
| event_driven | llama70_awq | 20 | 24732 | +0.70% | -0.90 | +0.39% | -1.34 | 0.3623 | 2.100 | -0.545 | 0.0643 | 16 | 44.9 | 100.0 | AWQ-INT4 |

## HEADLINE (honest)

**DIRECTIONALLY REPLICATES, significance attenuated.** The matched-class Llama-3.1-70B reproduces the SIGN of the Qwen 8-K residual vs HAR+firmID in 3/3 horizons with point estimates larger than Qwen's (+0.83/+0.64/+0.39% vs +0.45/+0.25/+0.20%), clustered DM<0 in 3/3 and raw p<.05 in 2/3; but after the pre-declared Holm(6) only 1/3 vs-single-HAR cells survive (min firmID Holm p=0.05001, 0/3 firmID cells <.05). The residual is NOT Qwen-specific — a healthy matched-class family recovers same-sign, same-or-larger increments — but it is not fully Holm-robust in the second family either.

- llama70 event_driven vs HAR+firmID: rel% = +0.83 / +0.64 / +0.39 (h=5/10/20), clustered DM = -2.58 / -2.00 / -1.34; Qwen benchmark was +0.45 / +0.25 / +0.20.
- vs single recalibrated HAR: rel% = +1.39 / +1.17 / +0.70, DM = -3.22 / -2.35 / -0.90; 1/3 cells survive Holm(6).
- The attenuated DM stats are NOT a power artefact of the panel: llama70 is scored on the identical test panel and day set as Qwen (same n_test, same n_days = 996/991/981). The increment is larger in mean but noisier — llama70's forecasts are far more dispersed (pred sd 0.064-0.087 vs Qwen 0.021-0.032), inflating the loss-differential variance.
- Health check: llama70 is a HEALTHY forecaster by the Yi/Phi criteria: variance-unit QLIKE 1.12-2.10 (Qwen 1.18-1.32, Yi 7.60-8.19, capability floor), R^2 -0.55--0.11 (Qwen -0.48--0.12), modal share max 51.2% (Yi up to 73.6%), pred sd 0.064-0.087, parse_ok 100%.

## SANITY

- G1 PASS: all 15 committed crossfamily_llm.csv M1 cells (qwen/yi/phi) reproduced to machine precision (rtol 1e-12) on columns ['n_test', 'n_days', 'rel_har', 'dm_har', 'p_har', 'rel_firm', 'dm_firm', 'p_firm', 'g_text'].
- G2 PASS: all 12 committed crossfamily_standalone.csv long_form/event_driven diagnostic cells (qwen/yi) reproduced to machine precision on columns ['qlike_vol', 'qlike_var', 'r2', 'pred_sd', 'n_unique_2dp', 'mode_val_2dp', 'mode_share_pct'].
- G3 PASS: recomputed variance-unit QLIKE matches stored metrics.json within 1e-3 relative in 18/18 cells (including all llama70 cells).
- G4 PASS: llama70 combined run verified as the relabelled ED duplicate (117,407/117,407 predictions identical, 8-K only) and excluded.
