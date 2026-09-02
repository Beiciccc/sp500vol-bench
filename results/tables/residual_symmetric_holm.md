# ROW 6 — Symmetric Holm for the 8-K residual + two-way (firm x day) clustering on the SEED-ENSEMBLE basis

UNITS: every DM statistic in this table (dm_day/dm_2way and all Holm columns) is on **VOL-unit QLIKE** losses — the committed primary (m1_ensemble_primary `vol_*` columns, firm_identity_* `_q_` columns). No variance-unit (RV^2) QLIKE appears here; see variance_unit_cascade for that restatement.

## RESTATED vs BEFORE

| quantity | BEFORE (committed) | RESTATED (this table) |
|---|---|---|
| residual multiplicity | event-driven C6 cells vs firm-identity quoted at raw clustered p (firm_identity_control.csv); never Holm-corrected in a pre-declared family symmetric with the null claims | Holm in TWO pre-declared families: 12-cell C6-symmetric family and the 69-cell firm-identity grid family, both day-clustered primary + two-way variant |
| two-way clustering basis | seed2026 only (twoway_cluster.csv) | seed-ensemble primary: M1 grid genuine 38/69 (day) -> **27/69** (two-way); firm-identity survivors 8/69 (day) -> **5/69** (two-way) |
| M1 grid genuine, seed2026 (context) | 29/69 day -> 24/69 two-way (twoway_cluster.csv) | unchanged (gate G1) |

## PRE-DECLARED HOLM FAMILIES (declared in the script header BEFORE any result was computed)

* **FAMILY-1 'C6 symmetric' (12 cells):** C6_llmtext x {long_form, event_driven} x h in {5,10,20} x reference in {single recalibrated HAR, firm-identity (HAR + firm-mean-val-RV)}. Day-clustered DM p, Holm over 12; two-way variant reported alongside. C6 is single-seed, so this family is basis-invariant (asserted, gate G6).
* **FAMILY-2 '69-cell firm-identity grid':** all 69 text-model x disclosure x horizon cells vs the firm-identity reference — the SAME family the null claims use (committed holm_p of firm_identity_control.csv / firm_identity_ensemble.csv; reproduced by gates G2/G5). Reported on both bases + two-way variant.
* Survival rule: clustered DM < 0 AND Holm(p) < .05 within the family (panel-A cells additionally need |placebo DM| < 2; the firm-ref panel has no placebo, matching the committed convention).

## (a) Does the event-driven C6 residual survive symmetric Holm?

| residual cell | raw clustered p (day) | FAMILY-1 Holm (12) | survives? | FAMILY-2 Holm (69, s26) | survives? | FAMILY-2 Holm (69, ensemble) | survives? |
|---|---|---|---|---|---|---|---|
| event_driven C6 h5 vs firm-identity | 7.49e-07 | 0.0000 | YES | 0.0000 | YES | 0.0000 | YES |
| event_driven C6 h10 vs firm-identity | 1.04e-05 | 0.0001 | YES | 0.0006 | YES | 0.0005 | YES |
| event_driven C6 h20 vs firm-identity | 3.62e-04 | 0.0011 | YES | 0.0141 | YES | 0.0134 | YES |

Two-way variant of the same families:
| residual cell | p_2way | FAMILY-1 Holm-2way (12) | survives? | FAMILY-2 Holm-2way (69, s26) | survives? | FAMILY-2 Holm-2way (69, ensemble) | survives? |
|---|---|---|---|---|---|---|---|
| event_driven C6 h5 vs firm-identity | 7.36e-06 | 0.0001 | YES | 0.0005 | YES | 0.0005 | YES |
| event_driven C6 h10 vs firm-identity | 6.45e-05 | 0.0004 | YES | 0.0037 | YES | 0.0036 | YES |
| event_driven C6 h20 vs firm-identity | 1.33e-03 | 0.0040 | YES | 0.0598 | no | 0.0571 | no |

### FAMILY-1 full 12-cell table (basis-invariant; single-seed C6)

| disc | h | reference | dm_day | p_day | fam12 Holm | survives | dm_2way | p_2way | fam12 Holm-2way | survives |
|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | 5 | single-HAR | -5.04 | 5.42e-07 | 0.0000 | YES | -4.56 | 6.16e-06 | 0.0001 | YES |
| event_driven | 5 | firm-identity | -4.98 | 7.49e-07 | 0.0000 | YES | -4.52 | 7.36e-06 | 0.0001 | YES |
| event_driven | 10 | single-HAR | -3.76 | 1.81e-04 | 0.0007 | YES | -3.49 | 5.24e-04 | 0.0026 | YES |
| event_driven | 10 | firm-identity | -4.43 | 1.04e-05 | 0.0001 | YES | -4.03 | 6.45e-05 | 0.0004 | YES |
| event_driven | 20 | single-HAR | -1.98 | 4.77e-02 | 0.0477 | YES | -1.82 | 6.92e-02 | 0.0692 | no |
| event_driven | 20 | firm-identity | -3.58 | 3.62e-04 | 0.0011 | YES | -3.23 | 1.33e-03 | 0.0040 | YES |
| long_form | 5 | single-HAR | -6.31 | 4.66e-10 | 0.0000 | YES | -5.92 | 5.67e-09 | 0.0000 | YES |
| long_form | 5 | firm-identity | +4.94 | 9.60e-07 | 0.0000 | no | +4.37 | 1.48e-05 | 0.0001 | no |
| long_form | 10 | single-HAR | -7.92 | 8.18e-15 | 0.0000 | YES | -7.59 | 1.34e-13 | 0.0000 | YES |
| long_form | 10 | firm-identity | +5.79 | 1.03e-08 | 0.0000 | no | +5.64 | 2.62e-08 | 0.0000 | no |
| long_form | 20 | single-HAR | -3.23 | 1.28e-03 | 0.0026 | YES | -2.76 | 6.02e-03 | 0.0120 | YES |
| long_form | 20 | firm-identity | -4.15 | 3.70e-05 | 0.0002 | YES | -3.37 | 8.10e-04 | 0.0032 | YES |

## (b) Two-way (firm x day) CGM clustering on the SEED-ENSEMBLE basis

Identical CGM machinery as the committed twoway_cluster.csv (scripts/analysis/twoway_dm.py); only the text-forecast basis changes to the declared primary (per-observation 3-seed mean; single-seed models unchanged).

| grid | basis | genuine/survives (day, Holm<.05) | (two-way, Holm<.05) | hurts day->2way | day/2way verdict flips | guard hits |
|---|---|---|---|---|---|---|
| (i) M1 69-cell, single-HAR ref | seed2026 (twoway_cluster.csv) | 29/69 | 24/69 | 8->5 | 5 | 0 |
| (i) M1 69-cell, single-HAR ref | **seed-ensemble (primary)** | **38/69** | **27/69** | 8->5 | 11 | 0 |
| (ii) firm-identity ref | seed2026 (twoway_cluster.csv) | 8/69 | 5/69 | 29->19 | 3 | 0 |
| (ii) firm-identity ref | **seed-ensemble (primary)** | **8/69** | **5/69** | 35->20 | 3 | 0 |

### (iii) Residual cells, two-way verdicts (ensemble basis = seed2026 basis for C6, gate G6)

| cell | reference | dm_day | Holm(day,69) | dm_2way | p_2way | Holm(2way,69) | verdict day -> two-way |
|---|---|---|---|---|---|---|---|
| event_driven C6 h5 | firm-identity | -4.98 | 0.0000 | -4.52 | 0.0000 | 0.0005 | survives -> survives |
| event_driven C6 h10 | firm-identity | -4.43 | 0.0005 | -4.03 | 0.0001 | 0.0036 | survives -> survives |
| event_driven C6 h20 | firm-identity | -3.58 | 0.0134 | -3.23 | 0.0013 | 0.0571 | survives -> ns |
| event_driven C6 h5 | single-HAR | -5.04 | 0.0000 | -4.56 | 0.0000 | 0.0003 | genuine -> genuine |
| event_driven C6 h10 | single-HAR | -3.76 | 0.0065 | -3.49 | 0.0005 | 0.0231 | genuine -> genuine |
| event_driven C6 h20 | single-HAR | -1.98 | 0.6569 | -1.82 | 0.0692 | 1.0000 | ns -> ns |

### Ensemble-basis verdict flips (day -> two-way)

| panel | disc | model | h | dm_day | Holm(day) | dm_2way | p_2way | Holm(2way) | flip |
|---|---|---|---|---|---|---|---|---|---|
| single-HAR | event_driven | B1_bow_ridge | 5 | -3.35 | 0.0252 | -3.07 | 0.0023 | 0.0794 | YES -> no |
| single-HAR | event_driven | B1_bow_ridge | 10 | -3.25 | 0.0332 | -2.81 | 0.0052 | 0.1605 | YES -> no |
| single-HAR | event_driven | B1_bow_ridge | 20 | -3.10 | 0.0489 | -2.53 | 0.0116 | 0.3028 | YES -> no |
| single-HAR | event_driven | B2_tfidf_ridge | 20 | -3.11 | 0.0489 | -2.06 | 0.0398 | 0.7567 | YES -> no |
| single-HAR | long_form | B3_lm_linear | 20 | -4.69 | 0.0001 | -2.87 | 0.0043 | 0.1376 | YES -> no |
| single-HAR | long_form | C1_bert_s1 | 5 | -3.25 | 0.0332 | -2.62 | 0.0090 | 0.2516 | YES -> no |
| single-HAR | long_form | C2_finbert_s1 | 5 | -4.53 | 0.0003 | -3.07 | 0.0023 | 0.0794 | YES -> no |
| single-HAR | long_form | C2_finbert_s2 | 20 | -4.89 | 0.0001 | -3.08 | 0.0022 | 0.0775 | YES -> no |
| single-HAR | long_form | C3_roberta_s1 | 20 | -3.90 | 0.0040 | -3.04 | 0.0025 | 0.0812 | YES -> no |
| single-HAR | long_form | C4_longformer | 20 | -4.38 | 0.0005 | -3.19 | 0.0015 | 0.0553 | YES -> no |
| single-HAR | long_form | C6_llmtext | 20 | -3.23 | 0.0332 | -2.76 | 0.0060 | 0.1806 | YES -> no |
| firm-identity | event_driven | B1_bow_ridge | 5 | -3.50 | 0.0175 | -3.15 | 0.0017 | 0.0706 | YES -> no |
| firm-identity | event_driven | C6_llmtext | 20 | -3.58 | 0.0134 | -3.23 | 0.0013 | 0.0571 | YES -> no |
| firm-identity | long_form | B3_lm_linear | 10 | -3.63 | 0.0116 | -2.56 | 0.0108 | 0.3340 | YES -> no |

## SANITY

All gates are HARD assertions at machine precision (np.allclose rtol=1e-9, atol=1e-12); the script aborts before writing any output if one fails. All PASS:

* G1(twoway_cluster.csv panel a) dm_day: max|diff|=8.88e-16
* G1(twoway_cluster.csv panel a) p_day: max|diff|=1.11e-16
* G1(twoway_cluster.csv panel a) holm_day: max|diff|=1.11e-16
* G1(twoway_cluster.csv panel a) dm_2way: max|diff|=4.44e-16
* G1(twoway_cluster.csv panel a) p_2way: max|diff|=9.89e-17
* G1(twoway_cluster.csv panel a) holm_2way: max|diff|=7.98e-17
* G1(twoway_cluster.csv panel a) placebo_dm_2way: max|diff|=2.22e-16
* G2(twoway_cluster.csv panel b) dm_day: max|diff|=4.44e-16
* G2(twoway_cluster.csv panel b) p_day: max|diff|=9.71e-17
* G2(twoway_cluster.csv panel b) holm_day: max|diff|=9.78e-17
* G2(twoway_cluster.csv panel b) dm_2way: max|diff|=4.44e-16
* G2(twoway_cluster.csv panel b) p_2way: max|diff|=1.11e-16
* G2(twoway_cluster.csv panel b) holm_2way: max|diff|=9.71e-17
* G3(m1_clustered.csv) placebo_dm_day: max|diff|=2.22e-16
* G3 genuine flags: identical
* G4(m1_ensemble_primary.csv) dm_day: max|diff|=8.88e-16
* G4(m1_ensemble_primary.csv) p_day: max|diff|=9.71e-17
* G4(m1_ensemble_primary.csv) holm_day: max|diff|=1.11e-16
* G4(m1_ensemble_primary.csv) placebo_dm_day: max|diff|=1.11e-16
* G4 genuine flags: identical
* G5(firm_identity_ensemble.csv) dm_day: max|diff|=4.44e-16
* G5(firm_identity_ensemble.csv) p_day: max|diff|=9.71e-17
* G5(firm_identity_ensemble.csv) holm_day: max|diff|=1.11e-16
* G6 single-seed dm_day: max|diff|=0.00e+00 over 72 cells
* G6 single-seed p_day: max|diff|=0.00e+00 over 72 cells
* G6 single-seed dm_2way: max|diff|=0.00e+00 over 72 cells
* G6 single-seed p_2way: max|diff|=0.00e+00 over 72 cells

Gate targets (committed tables): `twoway_cluster.csv` (G1 panel a / G2 panel b: dm_day, p_day, holm_day, dm_2way, p_2way, holm_2way, placebo_dm_2way), `m1_clustered.csv` (G3: placebo_dm_clust, genuine_clust), `m1_ensemble_primary.csv` (G4: vol_dm_q_clu, vol_p_q_clu, vol_dmq_holm_clu, vol_placebo_dm_clu, genuine_ens_vol), `firm_identity_ensemble.csv` (G5: dm_q_clustered, p_q_clustered, holm_p), plus G6: every single-seed cell (incl. all C6 cells) identical between the seed2026 and ensemble bases.

No look-ahead: all reference/combiner weights are fit on the validation split only and applied frozen to test (fc.log_combo / mrf.log_ols_frozen).

## Bottom line

* FAMILY-1 (12-cell symmetric C6 family): the event-driven residual survives Holm at h5/h10/h20 = YES/YES/YES (two-way variant: YES/YES/YES).
* FAMILY-2 (69-cell firm-identity grid, ensemble basis): survives at h5/h10/h20 = YES/YES/YES day-clustered; YES/YES/no two-way.
* Ensemble-basis two-way headline: M1 grid 38/69 (day) -> 27/69 (two-way); firm-identity survivors 8/69 -> 5/69. Movement is toward the null, consistent with the seed2026 two-way robustness (wider SEs by construction).