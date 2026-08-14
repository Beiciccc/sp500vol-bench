# Prereg B0 — 3-seed Llama-3.1-70B (AWQ-INT4) ensemble rescoring, 8-K channel

## Disclosures

- **Ensemble semantics**: temperature-0 decoding; seeds 2026/2027/2028 were NOT passed to vLLM's sampler — they differ only through vLLM/AWQ-INT4/TP2 kernel non-determinism. This is a **reproducibility-jitter ensemble**, not a stochastic-decoding one (scripts/experiments/row15_llama70_ensemble/launch.sh).
- **Ensemble construction**: per-row ARITHMETIC mean exp(mean(log(pred_seed))) across the three seeds, verified row-wise by sanity gate G5 (rtol 1e-06, 1:1 merge on ['ticker', 'accession', 'horizon_days']).
- Holm is applied within the NEW pre-declared family of the 6 ensemble tests (3 horizons x {vs HAR, vs HAR+firmID}); the single-seed rows carry their own committed Holm(6) values unchanged (crossfamily_llama70.csv). The two families are parallel, not pooled.
- The single-seed rows are retained side by side; the ensemble rows do not replace them (prereg B0).

## Table — M1 increment (log-space, combiner val-fit test-frozen, day-clustered DM)

rel% > 0 = text lowers QLIKE vs the reference; `**` = clustered DM<0, raw p<.05.

| family | h | n_test | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) | Holm p (HAR) | Holm p (firmID) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| llama70_awq | 5 | 25109 | +1.39%** | -3.22 | +0.83%** | -2.58 | 0.007934 | 0.05001 |
| llama70_awq | 10 | 25001 | +1.17%** | -2.35 | +0.64%** | -2.00 | 0.0763 | 0.1379 |
| llama70_awq | 20 | 24732 | +0.70% | -0.90 | +0.39% | -1.34 | 0.3698 | 0.3623 |
| llama70_awq_ens3 | 5 | 25109 | +1.39%** | -3.22 | +0.84%** | -2.59 | 0.007829 | 0.04855 |
| llama70_awq_ens3 | 10 | 25001 | +1.16%** | -2.34 | +0.64%** | -2.00 | 0.07701 | 0.137 |
| llama70_awq_ens3 | 20 | 24732 | +0.69% | -0.88 | +0.38% | -1.35 | 0.3783 | 0.3545 |

## VERDICT (pre-registered ladder, ens rows)

**DIRECTIONALLY REPLICATES, significance attenuated.** The 3-seed Llama-3.1-70B ensemble reproduces the SIGN of the Qwen 8-K residual vs HAR+firmID in 3/3 horizons (+0.84/+0.64/+0.38% vs Qwen's +0.45/+0.25/+0.20%), clustered DM<0 in 3/3 and raw p<.05 in 2/3; but after the pre-declared Holm(6) only 1/3 vs-single-HAR cells survive (min firmID Holm p=0.04855, 1/3 firmID cells <.05).

- B1 family STRONG pass (>=2/3 horizons Holm<.05 & DM<0 vs single recalibrated HAR, within the ens Holm(6)): FAIL (1/3)

## SANITY

- G1' PASS: all 3 committed single-seed llama70 M1 rows reproduced to machine precision (rtol 1e-12) on columns ['n_test', 'n_days', 'rel_har', 'dm_har', 'p_har', 'rel_firm', 'dm_firm', 'p_firm', 'g_text'].
- G5 PASS: ensemble prediction == mean(seed predictions) (arithmetic; prereg v1.2) on all rows (rtol 1e-06); merge on ['ticker', 'accession', 'horizon_days'] verified 1:1 across the four prediction files.
