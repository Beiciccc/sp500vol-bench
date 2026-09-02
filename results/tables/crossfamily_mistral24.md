# Prereg B1 — third family: Mistral-Small-24B-Instruct-2501 (bf16), 3-seed ensemble, 8-K channel

## Disclosures

- **Precision**: mistral24 ran **bf16, NO quantization** (unlike the llama70 replication arm's AWQ-INT4). Protocol otherwise byte-identical to C6/llama70: same manifest/prompt/guided-JSON/clip[0.03,3.0]/retry stack (scripts/experiments/e1_llm_forecast; scripts/experiments/row16_mistral24_ensemble/launch.sh).
- **Seed semantics**: temperature-0 decoding; `--seed` is NOT plumbed into vLLM's sampler (run_inference.py forwards it only to the mock path) — seeds 2026/2027/2028 differ only through vLLM/TP2 kernel non-determinism. This is a **reproducibility-jitter ensemble**, not a stochastic-decoding one (scripts/experiments/row16_mistral24_ensemble/launch.sh, TEMPERATURE PROTOCOL block; identical to the llama70 arm, row15).
- **Tokenizer caveat (fix_mistral_regex)**: vLLM's mistral-common tokenizer backend lacks `.is_fast` and crashes vLLM's tokenizer check, so the runs load an **hfview** snapshot (tekken.json/params.json removed) that forces the transformers FAST tokenizer (launch.sh HFVIEW override; config.json `llm: .../mistral24_hfview`). The transformers fast tokenizer emits its known Mistral tokenizer-regex warning; disclosed as a tokenizer-regex caveat. It is internally consistent: all three seeds (and hence the ensemble) used the identical hfview tokenizer, so it cannot differentiate the seeds or the cross-seed comparison.
- **Ensemble construction**: per-observation ARITHMETIC mean of prediction_realised_vol across the three seeds, inner-joined 1:1 on (ticker, accession, horizon_days), VAL+TEST only — identical convention to the C-model seed-ensemble primary (m1_ensemble_primary.ensemble_text) and to C6_llmtext_llama70ens (ens run config.json documents this); verified row-wise by sanity gate G5 (rtol 1e-06).
- **Multiplicity**: Holm is applied within each family's OWN pre-declared 6-test set (3 horizons x {vs HAR, vs HAR+firmID}) — mistral24_bf16 and mistral24_ens3 each get their own Holm(6); the llama70 anchor rows carry their committed Holm(6) values unchanged; the qwen primary's Holm(6) is computed here on its committed raw p's (the committed crossfamily_llm.csv carries raw p only). Families are parallel, never pooled.
- parse_fail_rate = 0.0 and clipped_rate = 0.0 in all three mistral seed configs (verified in-script: {'2026': (0.0, 0.0), '2027': (0.0, 0.0), '2028': (0.0, 0.0)}); the ens run dir carries no parse stats (its predictions are means of already-parsed seed forecasts) — its parse/clip cells are '-'.
- The single-seed mistral24_bf16 rows are retained side by side; the ensemble rows do not replace them (prereg B0/B1 convention).

## Table — M1 increment (log-space, combiner val-fit test-frozen, day-clustered DM) + standalone health

rel% is on **volatility-unit** QLIKE (committed-anchor convention); the QLIKE(var) health column is **variance-unit** (Patton-robust). rel% > 0 = text lowers QLIKE vs the reference; `**` = clustered DM<0, raw p<.05.

| family | h | n_test | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) | Holm p (HAR) | Holm p (firmID) | QLIKE(var) | R^2 | pred sd | n_uniq(2dp) | mode share% | parse_ok% | flag |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| qwen3_32b | 5 | 25109 | +1.21%** | -5.04 | +0.45%** | -5.26 | 2.167e-06 | 1.059e-06 | 1.177 | -0.121 | 0.0323 | 10 | 49.3 | 100.0 | - |
| qwen3_32b | 10 | 25001 | +1.00%** | -3.76 | +0.25%** | -5.16 | 0.000472 | 1.472e-06 | 1.214 | -0.244 | 0.0243 | 19 | 50.3 | 100.0 | - |
| qwen3_32b | 20 | 24732 | +0.66%** | -1.98 | +0.20%** | -3.79 | 0.04772 | 0.000472 | 1.316 | -0.476 | 0.0214 | 16 | 49.2 | 100.0 | - |
| llama70_awq | 5 | 25109 | +1.39%** | -3.22 | +0.83%** | -2.58 | 0.007934 | 0.05001 | 1.122 | -0.112 | 0.0873 | 10 | 51.2 | 100.0 | AWQ-INT4 |
| llama70_awq | 10 | 25001 | +1.17%** | -2.35 | +0.64%** | -2.00 | 0.0763 | 0.1379 | 1.403 | -0.215 | 0.0718 | 15 | 40.4 | 100.0 | AWQ-INT4 |
| llama70_awq | 20 | 24732 | +0.70% | -0.90 | +0.39% | -1.34 | 0.3698 | 0.3623 | 2.100 | -0.545 | 0.0643 | 16 | 44.9 | 100.0 | AWQ-INT4 |
| llama70_awq_ens3 | 5 | 25109 | +1.39%** | -3.22 | +0.84%** | -2.59 | 0.007829 | 0.04855 | - | - | - | - | - | - | AWQ-INT4-ens3 |
| llama70_awq_ens3 | 10 | 25001 | +1.16%** | -2.34 | +0.64%** | -2.00 | 0.07701 | 0.137 | - | - | - | - | - | - | AWQ-INT4-ens3 |
| llama70_awq_ens3 | 20 | 24732 | +0.69% | -0.88 | +0.38% | -1.35 | 0.3783 | 0.3545 | - | - | - | - | - | - | AWQ-INT4-ens3 |
| mistral24_bf16 | 5 | 25109 | +0.27% | -0.36 | -0.14% | +0.93 | 1 | 1 | 0.691 | -0.186 | 0.0640 | 10 | 71.5 | 100.0 | bf16 |
| mistral24_bf16 | 10 | 25001 | +0.14% | -0.41 | -0.46% | +1.00 | 1 | 1 | 0.537 | -0.079 | 0.0423 | 17 | 70.6 | 100.0 | bf16 |
| mistral24_bf16 | 20 | 24732 | +0.14% | -0.29 | -0.37% | +0.81 | 1 | 1 | 0.422 | -0.035 | 0.0288 | 18 | 88.6 | 100.0 | bf16 |
| mistral24_ens3 | 5 | 25109 | +0.31% | -0.46 | -0.10% | +0.83 | 1 | 1 | 0.689 | -0.183 | 0.0635 | 25 | 70.5 | - | bf16-ens3 |
| mistral24_ens3 | 10 | 25001 | +0.15% | -0.44 | -0.45% | +0.96 | 1 | 1 | 0.537 | -0.078 | 0.0420 | 40 | 69.3 | - | bf16-ens3 |
| mistral24_ens3 | 20 | 24732 | +0.13% | -0.27 | -0.38% | +0.82 | 1 | 1 | 0.421 | -0.034 | 0.0284 | 38 | 87.9 | - | bf16-ens3 |

## VERDICT (pre-registered ladder, mistral24_ens3 rows)

**Does NOT replicate.** The 3-seed Mistral-Small-24B (bf16) ensemble shows no positive increment over the firm-identity-augmented reference in any horizon.

- B1 family STRONG pass (>=2/3 horizons Holm<.05 & DM<0 vs single recalibrated HAR, within the mistral24_ens3 Holm(6)): FAIL (0/3)
- Info (not a prereg branch input) — single-seed mistral24_bf16 ladder: DOES NOT REPLICATE (0/3 firmID Holm<.05, 0/3 raw firmID p<.05, 0/3 rel_firm>0, 0/3 vs HAR after Holm).
- Health check (same formula as crossfamily_llama70.py): mistral24_ens3 is NOT clearly healthy by the Yi/Phi criteria — read the columns: variance-unit QLIKE 0.42-0.69 (Qwen 1.18-1.32, llama70 1.12-2.10, Yi 7.60-8.19 = capability floor) — the LOWEST of all families, NOT QLIKE-floored; but modal share reaches 87.9% (Yi's collapse benchmark was 73.6%, llama70 max 51.2%, Qwen max 50.3%): forecasts are heavily concentrated at 0.30-0.35 with pred sd 0.028-0.063, R^2 -0.18--0.03, parse_ok 100%. A mode-concentrated (near-constant) forecaster carries little firm-specific text signal by construction — reported as-is; the prereg draws no capability exemption for the third family.

## ACROSS-FAMILY RULE (prereg §B1, quoted verbatim)

> **Across-family claim rule (pre-declared)**:
> - Family STRONG pass: >=2/3 horizons Holm<.05 and DM<0 vs the single recalibrated HAR (within-family Holm(6));
> - Family WEAK pass: attains DIRECTIONALLY REPLICATES or above on the B0 ladder;
> - Paper wording: >=2/3 families STRONG -> "replicates across families";
>   >=2/3 families >=WEAK (including primary) -> "sign-robust across families, significance attenuated";
>   otherwise -> "does not replicate beyond the primary family" (the residual paragraph is downgraded accordingly;
>   per the established rule in FACTS.md, it must not be written as a family-specific proof).

F = {Qwen3-32B primary (single seed), Llama-3.1-70B-AWQ ens3, Mistral-24B ens3}, all on the 8-K (event_driven) channel.

| family | STRONG (>=2/3 Holm<.05 & DM<0 vs HAR, own Holm(6)) | ladder tier | >=WEAK |
|---|---|---|---|
| qwen3_32b (primary, single seed) | PASS (3/3) | REPLICATES | yes |
| llama70_awq_ens3 (committed crossfamily_llama70_ens.csv) | FAIL (1/3) | DIRECTIONALLY REPLICATES | yes |
| mistral24_ens3 (this table) | FAIL (0/3) | DOES NOT REPLICATE | no |

**FIRED: BRANCH 2 — "sign-robust across families, significance attenuated" (2/3 families >=WEAK incl. primary; 1/3 STRONG)**

## SANITY

- G1'' PASS: all 6 committed crossfamily_llama70_ens.csv rows (llama70_awq single-seed AND llama70_awq_ens3), recomputed on this exact code path, reproduced to machine precision (rtol 1e-12) on columns ['n_test', 'n_days', 'rel_har', 'dm_har', 'p_har', 'rel_firm', 'dm_firm', 'p_firm', 'g_text']; the ens Holm(6) columns re-derive identically.
- G1q PASS: the 3 committed qwen3_32b event_driven M1 rows (crossfamily_llm.csv) reproduced to machine precision (rtol 1e-12) — the primary family's cells are anchored before the across-family rule reads them.
- G5 PASS: mistral24ens prediction == row-wise ARITHMETIC mean of the three seed predictions on all 117407 rows (rtol 1e-06; max relative deviation 0.000e+00); merge on ['ticker', 'accession', 'horizon_days'] verified 1:1 across the four prediction files.
- G3'' PASS: recomputed variance-unit QLIKE matches stored metrics.json within 1e-3 relative in 6/6 mistral cells (single + ens).
