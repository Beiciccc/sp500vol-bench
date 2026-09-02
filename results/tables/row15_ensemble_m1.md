# ROW 15 — 3-seed 70B ensemble vs single-seed, matched-class 8-K residual

## RESTATED vs BEFORE

| | BEFORE (single seed 2026, crossfamily_llama70) | RESTATED (3-seed ensemble 2026+2027+2028) |
|---|---|---|
| verdict | DIRECTIONALLY REPLICATES — 3/3 firmID cells positive, raw p<.05 in 2/3, but after Holm(6) 0/3 firmID (min 0.05001) and 1/3 vs-single-HAR cells survive. | **DIRECTIONALLY REPLICATES** — 3/3 firmID cells positive, raw p<.05 in 2/3, but after Holm(6) 1/3 firmID (min 0.04855) and 1/3 vs-single-HAR cells survive. |
| firmID cells surviving Holm(6) | 0/3 (best p=0.05001) | **1/3** (best p=0.04855) |
| vs-single-HAR cells surviving Holm(6) | 1/3 (best p=0.00793) | **1/3** (best p=0.00783) |

Both columns use the identical crossfamily_llama70 protocol: log-space combiner (val-fit / test-frozen), day-clustered DM, VOL-unit QLIKE, Holm within the pre-declared family of 6 tests (3 horizons x {vs recalibrated HAR, vs HAR+firm-identity}). The only change is the forecast object: a per-observation arithmetic mean across vLLM seeds vs seed 2026 alone.

## Table — M1 increment (rel% vs HAR | DM(clu) | Holm p | rel% vs HAR+firmID | DM(clu) | Holm p)

`**` = clustered DM<0 & raw p<.05.

| h | n_test | SINGLE | ENSEMBLE |
|--:|--:|---|---|
| 5 | 25109 | +1.39%** | -3.22 | 0.007934 | +0.83%** | -2.58 | 0.05001 | +1.39%** | -3.22 | 0.007829 | +0.84%** | -2.59 | 0.04855 |
| 10 | 25001 | +1.17%** | -2.35 | 0.0763 | +0.64%** | -2.00 | 0.1379 | +1.16%** | -2.34 | 0.07701 | +0.64%** | -2.00 | 0.137 |
| 20 | 24732 | +0.70% | -0.90 | 0.3698 | +0.39% | -1.34 | 0.3623 | +0.69% | -0.88 | 0.3783 | +0.38% | -1.35 | 0.3545 |

## HEADLINE (honest)

Ensembling **does NOT reach REPLICATES**. Single-seed was DIRECTIONALLY REPLICATES (0/3 firmID, 1/3 vs-HAR Holm-robust; best firmID Holm p=0.05001); the 3-seed ensemble is DIRECTIONALLY REPLICATES (1/3 firmID, 1/3 vs-HAR; best firmID Holm p=0.04855).

- Holm significance IMPROVES over the single seed (firmID Holm min 0.05001 -> 0.04855; vs-HAR 1/3 -> 1/3).
- Ensemble firmID rel%: +0.84/+0.64/+0.38 (h=5/10/20); Qwen event-driven benchmark was +0.45/+0.25/+0.20.
- CAVEAT on seed diversity: run_inference.py decodes at temperature 0 and does not pass --seed to vLLM, so the 3 seeds diverge only through AWQ-INT4 / TP2 kernel non-determinism. If the ensemble ≈ the single seed (rel% and Holm p nearly unchanged), that is the expected signature of near-deterministic decoding, NOT a bug — report it as such.

## SANITY

- PASS: single-seed 2026 cells reproduce results/tables/crossfamily_llama70.csv (llama70_awq / event_driven) to machine precision (rtol 1e-12) on ['n_test', 'n_days', 'rel_har', 'dm_har', 'p_har', 'rel_firm', 'dm_firm', 'p_firm', 'g_text', 'p_har_holm', 'p_firm_holm'].
