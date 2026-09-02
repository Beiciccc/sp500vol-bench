# Prereg B2 (prereg-rfa v1.3) — fourth cross-family probe: unsloth/gemma-3-27b-it (bf16), 3-seed ensemble, 8-K channel

## Disclosures

- **Model / precision**: unsloth/gemma-3-27b-it, **bf16, NO quantization** (matched-class to Qwen3-32B; also removes the 70B arm's int4 confound), vLLM TP=2 on 2xA100-40G. Weights resolved OFFLINE from the local HF cache (mirror-prefetched; box has no HF egress). Protocol otherwise byte-identical to C6/llama70/mistral24: same manifest/prompt/guided-JSON/clip[0.03,3.0]/retry stack (scripts/experiments/e1_llm_forecast).
- **Chat-template adaptation (disclosed protocol delta, cf. B1's Mistral tokenizer caveat)**: Gemma's chat template has NO system role. The committed C6 prompt is a [system, user] pair; here the system message is folded VERBATIM into the user turn as prefix + a blank line (the same convention Gemma's own template applies to system content). Text byte-identical otherwise; the fold is recorded in each run's config.json (system_fold=True) and is internally consistent across pilot, all three seeds and the ensemble, so it cannot differentiate seeds or the cross-family comparison.
- **Seed semantics**: temperature-0 decoding; `--seed` is NOT plumbed into vLLM's sampler — seeds 2026/2027/2028 differ only through vLLM/TP2 kernel non-determinism. This is a **reproducibility-jitter ensemble**, not a stochastic-decoding one (identical to the llama70/mistral24 arms).
- **Registered pilot gate (prereg v1.3, ran BEFORE the full pass)**: 2000 validation documents (deterministic FIRST-2,000 by canonical sort, seed 2026, single pass), health formula verbatim; result healthy=True, max QLIKE(var)=3.660, max modal share=45.2% (results/tables/crossfamily_gemma27_pilot.json). The pilot touched NO test rows and computed NO increment statistics.
- **Multiplicity (registered v1.3 delta from B1)**: Holm is applied within the 3 horizons PER REFERENCE — one Holm(3) family vs the single recalibrated HAR and one vs HAR+firmID (prereg v1.3: "Holm(3) per reference"); the branch rules below consume these. Pooled Holm(6) values (B1's convention) are reported as info-only columns p_*_holm6 and are NOT decision-bearing. Anchor rows carry their own committed Holm(6) values unchanged; families are parallel, never pooled.
- **Ensemble construction**: per-observation ARITHMETIC mean of prediction_realised_vol across the three seeds, inner-joined 1:1 on (ticker, accession, horizon_days), VAL+TEST only — identical convention to C6_llmtext_llama70ens / C6_llmtext_mistral24ens; verified row-wise by sanity gate G5 (rtol 1e-06).
- parse/clip stats per seed config.json (parse_fail_rate, clipped_rate): {'2026': (0.0, 0.0), '2027': (0.0, 0.0), '2028': (0.0, 0.0)}; the ens run dir carries no parse stats (its predictions are means of already-parsed seed forecasts).
- The single-seed rows are retained side by side; the ensemble rows are the primary basis (prereg v1.3: the ensemble is the primary basis, single seeds serve as robustness).

## Table — M1 increment (log-space, combiner val-fit test-frozen, day-clustered DM) + standalone health

rel% is on **volatility-unit** QLIKE (committed-anchor convention); the QLIKE(var) health column is **variance-unit** (Patton-robust). rel% > 0 = text lowers QLIKE vs the reference; `**` = clustered DM<0, raw p<.05. For the gemma27 rows 'Holm p' = the REGISTERED Holm(3) per reference; anchor rows carry their committed Holm(6).

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
| gemma27_bf16 | 5 | 25109 | +1.83%** | -3.05 | +1.07%** | -1.97 | 0.006951 | 0.1486 | 0.900 | -0.015 | 0.0383 | 13 | 71.5 | 100.0 | bf16 |
| gemma27_bf16 | 10 | 25001 | +2.58%** | -2.64 | +1.49% | -1.63 | 0.01692 | 0.2082 | 0.572 | +0.057 | 0.0454 | 18 | 71.3 | 100.0 | bf16 |
| gemma27_bf16 | 20 | 24732 | +2.95% | -1.59 | +1.25% | -0.54 | 0.1127 | 0.5877 | 0.378 | +0.083 | 0.0520 | 19 | 71.3 | 100.0 | bf16 |
| gemma27_ens3 | 5 | 25109 | +1.84%** | -3.07 | +1.08%** | -1.98 | 0.00654 | 0.1435 | 0.900 | -0.015 | 0.0383 | 20 | 71.4 | - | bf16-ens3 |
| gemma27_ens3 | 10 | 25001 | +2.59%** | -2.65 | +1.50% | -1.64 | 0.01616 | 0.2032 | 0.572 | +0.058 | 0.0454 | 21 | 71.2 | - | bf16-ens3 |
| gemma27_ens3 | 20 | 24732 | +2.97% | -1.60 | +1.26% | -0.55 | 0.1099 | 0.5827 | 0.377 | +0.083 | 0.0520 | 23 | 71.2 | - | bf16-ens3 |

## VERDICT (ladder shown for completeness — NON-INFERENTIAL, health failed)

**DIRECTIONALLY REPLICATES, significance attenuated.** The 3-seed unsloth/gemma-3-27b-it (bf16) ensemble reproduces the SIGN of the Qwen 8-K residual vs HAR+firmID in 3/3 horizons (+1.08/+1.50/+1.26% vs Qwen's +0.45/+0.25/+0.20%), clustered DM<0 in 3/3 and raw p<.05 in 1/3; but after the registered Holm only 2/3 vs-single-HAR cells survive (min firmID Holm p=0.14347, 0/3 firmID cells <.05).

- Family STRONG readout (B1 formula on the registered Holm(3) vs HAR: >=2/3 horizons Holm<.05 & DM<0): PASS (2/3)
- Info (not a branch input) — single-seed gemma27_bf16 ladder: DIRECTIONALLY REPLICATES (0/3 firmID Holm<.05, 1/3 raw firmID p<.05, 3/3 rel_firm>0, 2/3 vs HAR after Holm).
- Health check (full run, same registered formula): gemma27_ens3 is NOT healthy — instrument-dead (branch d): variance-unit QLIKE 0.38-0.90 (gate <4; Qwen 1.18-1.32, Yi 7.60-8.19 = capability floor), max modal share 71.4% (gate <60%; Yi 73.6%, Mistral-24B 89.5-92.9% = the instrument-dead precedent), pred sd 0.038-0.052, R^2 -0.01-+0.08.

## REPLICATION DECISION (prereg v1.3 §B2, quoted verbatim)

> **Replication decision rule (pre-declared, verbatim requirement from the internal adversarial dry-run; all branches enter the paper)**: health (full run, same formula) is a precondition;
> - **(a) Holm-robust replication** ⇔ vs the firm-identity reference, >=2/3 horizons satisfy clustered DM<0 and Holm(3)<.05 → the residual wording is upgraded to "family-robust (two healthy families Holm-significant)", and "only partly family-robust" is deleted from the abstract;
> - **(b) Directional replication** ⇔ 3/3 DM<0 but fewer than 2 meet the Holm threshold → same tier as llama70, "three healthy probes share the same sign" enters the main text, the Holm-robustness wording is unchanged;
> - **(c) No replication** ⇔ healthy but neither (a) nor (b) → the residual is downgraded to "Qwen-conditional" in the abstract + 06 + 07 (the fix costs points, and we commit to executing it);
> - **(d) instrument-dead** ⇔ the health formula fails (pilot or full run) → tabled per the Mistral precedent, no inference.

**FIRED: BRANCH (d) — instrument-dead: the full-run health formula FAILED (healthy <=> max variance-unit QLIKE < 4 AND max modal share of round(pred,2) < 60% (Yi/Phi criteria, verbatim the crossfamily_mistral24.py / crossfamily_llama70.py health formula)); max QLIKE(var)=0.90, max modal share=71.4%.**

Consequence (registered): tabled per the Mistral precedent, NO inference drawn; the probe-denominator sentence updates and the pilot/full health failure is reported as-is in Stress Tests.

## SANITY

- G1'' PASS: all 6 committed crossfamily_llama70_ens.csv rows reproduced to machine precision (rtol 1e-12) on columns ['n_test', 'n_days', 'rel_har', 'dm_har', 'p_har', 'rel_firm', 'dm_firm', 'p_firm', 'g_text'].
- G1q PASS: the 3 committed qwen3_32b event_driven M1 rows (crossfamily_llm.csv) reproduced to machine precision (rtol 1e-12).
- G5 PASS: gemma27 ensemble == row-wise ARITHMETIC mean of the three seed predictions on all 117407 rows (rtol 1e-06; max relative deviation 0.000e+00); merge on ['ticker', 'accession', 'horizon_days'] verified 1:1.
- G3'' PASS: recomputed variance-unit QLIKE matches stored metrics.json within 1e-3 relative in 6/6 gemma27 cells (single + ens).
- Committed anchors (qwen / llama70 / mistral24) carried unchanged; the llama70_ens3 anchor pin (n_rep_har=1, DIRECTIONALLY REPLICATES) verified in-script: yes.
