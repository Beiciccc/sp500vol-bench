# Row 3 — tuned challengers (val-selected lr) vs archived fixed recipe: does tuning rescue the null? (DA CRITICAL #1)

## RESTATED vs BEFORE

| quantity | BEFORE (archived fixed recipe: lr=8e-5, max_epochs=15, es_patience=3) | RESTATED (row-3 val-tuned) |
|---|---|---|
| standalone: tuned significantly better than archived (clustered DM, Holm-9) | — | **2/9** better, 5/9 WORSE |
| M1 genuine cells vs single recalibrated HAR (Holm-9 + placebo) | 3/9 | **1/9** |
| M1 genuine cells vs HAR+firm-identity reference (Holm-9 + placebo) | 0/9 | **1/9** |
| null cells OVERTURNED by tuning (vs HAR) | — | 0 |
| genuine cells DESTROYED by tuning (vs HAR) | — | 2 (long_form C2_finbert_s1 h10, event_driven C2_finbert_s1 h10) |
| null cells OVERTURNED by tuning (vs firm-ID ref) | — | 1 (event_driven C2_finbert_s1 h5) |
| genuine cells DESTROYED by tuning (vs firm-ID ref) | — | 0 |

Verdicts here use the PRE-DECLARED Holm family = the 9 tuned cells per comparison (archived arm re-Holmed on the same 9-cell family for symmetry; the committed 69-cell verdict from m1_clustered.csv is quoted in the grid below as `gen69`). 'genuine' = clustered DM<0, Holm<.05, AND |label-shuffle placebo DM|<2 (seeds 1000-1004, both references). Vol-unit QLIKE; combiner weights val-fit, test-frozen (fc.log_combo); firm-identity reference = val-window firm mean spec of crossfamily_llm.py.

## Selected configs (grid audit: row3_tuning_grid.csv)

| arm | grid (3 lrs, max_epochs=5, es_patience=1) | SELECTED | pooled val QLIKE | archived fixed recipe |
|---|---|---|---|---|
| C2_finbert_s1 long_form | lr ∈ {5e-06, 1e-05, 2e-05} | **lr=2e-05** → C2t_finbert_s1_full_long_form_seed2026 | 3.540 (min of 4.671, 4.113, 3.540) | lr=8e-5, max_epochs=15, es_patience=3 |
| C2_finbert_s1 event_driven | lr ∈ {5e-06, 1e-05, 2e-05} | **lr=5e-06** → C2t_finbert_s1_full_event_driven_seed2026 | 4.175 (min of 4.175, 4.444, 4.672) | lr=8e-5, max_epochs=15, es_patience=3 |
| D2_gated_fusion long_form | lr ∈ {5e-06, 1e-05, 2e-05} | **lr=5e-06** → D2t_gated_fusion_full_long_form_seed2026 | 1.437 (min of 1.437, 1.654, 1.998) | lr=8e-5, max_epochs=15, es_patience=3 |

## 1. Standalone: tuned vs archived (test, vol-unit QLIKE / OOS R2; clustered DM tuned−archived, negative = tuned better)

| arm | h | QLIKE arch | QLIKE tuned | R2 arch | R2 tuned | DM(clu) | p | Holm-9 | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| long_form C2_finbert_s1 | 5 | 0.1676 | 0.1754 | -0.048 | -0.075 | +5.26 | 1.9e-07 | 0.0000 | tuned WORSE |
| long_form C2_finbert_s1 | 10 | 0.1647 | 0.1519 | -0.108 | -0.131 | -4.77 | 2.2e-06 | 0.0000 | tuned BETTER |
| long_form C2_finbert_s1 | 20 | 0.1663 | 0.1726 | -0.401 | -0.424 | +5.22 | 2.3e-07 | 0.0000 | tuned WORSE |
| event_driven C2_finbert_s1 | 5 | 0.1719 | 0.2417 | -0.024 | -0.242 | +12.83 | 5.8e-35 | 0.0000 | tuned WORSE |
| event_driven C2_finbert_s1 | 10 | 0.1935 | 0.1436 | -0.203 | -0.087 | -7.66 | 4.5e-14 | 0.0000 | tuned BETTER |
| event_driven C2_finbert_s1 | 20 | 0.1480 | 0.1482 | -0.182 | -0.267 | -0.31 | 7.5e-01 | 0.7546 | ns |
| long_form D2_gated_fusion | 5 | 0.1461 | 0.1384 | +0.089 | +0.068 | -1.40 | 1.6e-01 | 0.3235 | ns |
| long_form D2_gated_fusion | 10 | 0.1131 | 0.1274 | +0.071 | -0.036 | +3.56 | 3.9e-04 | 0.0012 | tuned WORSE |
| long_form D2_gated_fusion | 20 | 0.0832 | 0.1295 | +0.131 | -0.469 | +4.04 | 5.9e-05 | 0.0002 | tuned WORSE |

## 2. M1 increment vs single recalibrated HAR (clustered DM<0 = text helps)

| arm | h | rel% arch | DM arch | Holm-9 arch | gen69 | rel% tuned | DM tuned | p tuned | Holm-9 tuned | placebo tuned | verdict arch → tuned |
|---|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|---|
| long_form C2_finbert_s1 | 5 | +0.56 | -0.84 | 1.0000 | n | -0.95 | +3.50 | 4.9e-04 | 0.0039 | -0.85 | null → **sig-WORSE** |
| long_form C2_finbert_s1 | 10 | +4.56 | -6.46 | 0.0000 | Y | -0.50 | +2.58 | 1.0e-02 | 0.0710 | -0.28 | genuine → **null** |
| long_form C2_finbert_s1 | 20 | -0.08 | +0.88 | 1.0000 | n | -0.85 | +0.71 | 4.8e-01 | 0.9533 | -0.12 | null → **null** |
| event_driven C2_finbert_s1 | 5 | +2.14 | -4.95 | 0.0000 | Y | +1.01 | -3.92 | 9.6e-05 | 0.0009 | -0.41 | genuine → **genuine** |
| event_driven C2_finbert_s1 | 10 | +2.10 | -5.52 | 0.0000 | Y | +0.80 | -2.23 | 2.6e-02 | 0.1293 | -0.07 | genuine → **null** |
| event_driven C2_finbert_s1 | 20 | +0.92 | -0.50 | 1.0000 | n | +0.38 | -2.35 | 1.9e-02 | 0.1135 | +1.34 | null → **null** |
| long_form D2_gated_fusion | 5 | -0.12 | -0.29 | 1.0000 | n | +0.14 | -0.07 | 9.5e-01 | 0.9533 | +0.70 | null → **null** |
| long_form D2_gated_fusion | 10 | +0.13 | +0.96 | 1.0000 | n | -0.05 | +1.07 | 2.9e-01 | 0.8584 | +0.80 | null → **null** |
| long_form D2_gated_fusion | 20 | -6.00 | +4.27 | 0.0001 | n | +0.18 | -2.17 | 3.0e-02 | 0.1293 | -0.16 | sig-WORSE → **null** |

## 3. M1 increment vs firm-identity-augmented reference

| arm | h | rel% arch | DM arch | Holm-9 arch | rel% tuned | DM tuned | p tuned | Holm-9 tuned | placebo tuned | verdict arch → tuned |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| long_form C2_finbert_s1 | 5 | -0.57 | +1.57 | 0.5818 | -3.67 | +3.16 | 1.7e-03 | 0.0149 | +0.08 | null → **sig-WORSE** |
| long_form C2_finbert_s1 | 10 | -0.47 | +4.51 | 0.0001 | -0.85 | +1.57 | 1.2e-01 | 0.5859 | -0.05 | sig-WORSE → **null** |
| long_form C2_finbert_s1 | 20 | -0.52 | +0.27 | 0.8441 | -0.81 | +1.30 | 1.9e-01 | 0.7707 | -0.04 | null → **null** |
| event_driven C2_finbert_s1 | 5 | -0.31 | +4.91 | 0.0000 | +0.25 | -2.98 | 2.9e-03 | 0.0232 | -0.32 | sig-WORSE → **genuine** |
| event_driven C2_finbert_s1 | 10 | -1.22 | +6.54 | 0.0000 | -0.09 | +2.34 | 2.0e-02 | 0.1370 | -0.23 | sig-WORSE → **null** |
| event_driven C2_finbert_s1 | 20 | +0.66 | -1.14 | 0.8153 | +0.04 | -2.02 | 4.4e-02 | 0.2623 | +1.59 | null → **null** |
| long_form D2_gated_fusion | 5 | -0.42 | +0.80 | 0.8441 | +0.04 | +1.05 | 3.0e-01 | 0.7707 | +0.28 | null → **null** |
| long_form D2_gated_fusion | 10 | -0.45 | +1.27 | 0.8153 | -0.17 | +1.30 | 1.9e-01 | 0.7707 | +0.16 | null → **null** |
| long_form D2_gated_fusion | 20 | -10.12 | +3.06 | 0.0138 | +0.03 | -0.42 | 6.8e-01 | 0.7707 | +0.08 | sig-WORSE → **null** |

## SANITY

- **G1 PASS** — the 9 archived C2/D2 M1-vs-HAR cells recomputed here reproduce the committed `m1_clustered.csv` (n_obs, n_days, qlike_R/U, rel%, g_log, clustered DM, p, placebo) to machine precision; max |diff| = 2.22e-16. BASIS: `m1_clustered.csv` is the seed-2026 SINGLE-SEED table — the same basis as the archived counterparts and the tuned runs (all seed2026). `m1_ensemble_primary.csv` is 3-seed-ensemble for C2/D2 (n_seeds=3) and is therefore NOT the comparison basis; its seed-2026 columns (s26_*) were cross-asserted equal to `m1_clustered.csv` for these rows.
- **G2 PASS** — the firm-identity-reference machinery reproduces the committed `crossfamily_llm.csv` qwen3_32b rows (rel/DM/p vs both references) to machine precision; max |diff| = 4.44e-16.
- **G3 PASS** — standalone OOS R2 computed here reproduces each tuned run's committed `metrics.json` test rows to machine precision; max |diff| = 0.00e+00. (metrics.json QLIKE is variance-unit; this table is vol-unit by convention.)

## HEADLINE (honest)

**Validation tuning does NOT rescue the challengers — the null survives tuning.** (1) The val-QLIKE-selected configs do not even reliably improve the challengers themselves: tuned is significantly WORSE than the archived fixed recipe in 5/9 standalone cells and better in 2/9 (Holm-9) — validation selection transfers poorly to test. (2) Vs the single recalibrated HAR, tuning yields 0 newly-genuine cell(s) and DESTROYS 2 previously genuine cell(s) (long_form C2_finbert_s1 h10, event_driven C2_finbert_s1 h10); genuine count 3/9 → 1/9. (3) Vs the firm-identity reference the only movement is event_driven C2_finbert_s1 h5 (0/9 → 1/9) — an isolated cell, not a systematic rescue. DA CRITICAL #1 is answered: the fixed-recipe nulls are not an artifact of under-tuning; giving the challengers a validation-tuned arm reshuffles isolated cells but produces no consistent text increment, and the previously-reported increments are themselves fragile to the training recipe.
