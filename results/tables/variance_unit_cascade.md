# ROW 5 — Variance-unit QLIKE restatement of the FULL cascade + residual + standalone 180

## RESTATED vs BEFORE

BEFORE = the committed vol-unit q(y, f) cascade on the seed-ensemble basis (m1_ensemble_primary / firm_identity_ensemble / maximal_reference_ensemble / control_intersection_ensemble; standalone headline scored on squared error in dm_pairwise_clustered, QLIKE-vs-A2 only obs-level in dm_qlike_all_vs_A2.csv). RESTATED = the SAME seed-ensemble forecasts and the SAME val-fit frozen log-space references (unit-free; no look-ahead), evaluated with Patton-proxy-robust variance-unit QLIKE q(y^2, f^2), day-clustered DM (HAC lag = h-1 days).

| quantity | BEFORE (vol-unit / SE) | RESTATED (variance-unit) |
|---|---|---|
| (M1 primary) genuine cells | 38/69 | **20/69** (committed m1_variance_unit, gate-verified) |
| (a) firm-identity survivors raw / Holm | 15/69 / 8/69 | **12/69 / 3/69** |
| (b) maximal-pool survivors raw / Holm | 26/69 / 9/69 | **11/69 / 2/69** |
| (c) FULL AND (primary & maximal & firm), raw / Holm | 4 / 0 | **2 / 0** |
| (c) strictest (placebo-gated genuine & both Holm controls) | 0 | **0** |
| (c) maximal-vs-firm Holm survivor overlap | 0 (disjoint) | **0 (disjoint)** |
| (e) standalone vs A2, better raw / Holm (of 180) | SE: 0 / 0; vol-QLIKE: 0 / 0 | **var-QLIKE: 7 / 7** (all 7 = GARCH-family price baselines; text/fusion: 0 / 0 of 153) |
| (e) standalone significantly WORSE (Holm) | SE: 155/180; vol-QLIKE: 161/180 | **var-QLIKE: 153/180** |

## Pre-declared Holm families (declared BEFORE any result below was read)

1. **F-firm**: the 69-cell firm-identity grid, variance-unit clustered p-values, Holm across the 69 cells (mirrors the committed vol-unit family).
2. **F-max**: the 69-cell maximal-pool grid, same convention.
3. **F-primary**: the committed m1_variance_unit family (69 cells; reused as-is, gate-verified — not re-tested).
4. **F-standalone**: for each loss, Holm WITHIN each (disclosure, horizon) group over the 20 vs-A2 challengers (9 groups x 20 = 180) — identical to the committed dm_pairwise_clustered convention (weaker than a 180-wide family, i.e. conservative for the '0 better' headline; anti-conservative for 'worse' counts, so worse counts are descriptive).
5. **The residual is NOT granted its own family**: it is read out of F-firm (symmetric treatment; the dedicated symmetric-multiplicity re-analysis is row 6).

No new combiner/reference weights were fit: all forecasts are the committed val-fit-frozen objects; only the evaluation loss changes. No subsampling anywhere.

## (a)+(b) Vol-vs-var side-by-side — Holm survivors of either unit

All 69-cell detail is in variance_unit_cascade.csv; this table lists every cell that survives Holm in EITHER unit under EITHER control.

| control | cell | seeds | rel% vol | DM vol | Holm vol | rel% var | DM var | p var | Holm var | vol->var |
|---|---|---|---|---|---|---|---|---|---|---|
| firm | event_driven/B1_bow_ridge/h5 | 2026 | +0.46 | -3.50 | 0.017 | +0.57 | -2.13 | 0.0334 | 0.937 | **LOST** |
| firm | event_driven/B1_bow_ridge/h10 | 2026 | +0.25 | -4.71 | 0.000 | +0.27 | -2.73 | 0.0065 | 0.258 | **LOST** |
| firm | event_driven/B2_tfidf_ridge/h10 | 2026 | +0.28 | -4.28 | 0.001 | +0.35 | -3.53 | 0.0004 | 0.022 | KEEPS |
| firm | event_driven/B3_lm_linear/h5 | 2026 | +0.11 | -3.00 | 0.069 | +0.17 | -3.35 | 0.0008 | 0.041 | **GAINED** |
| firm | event_driven/C6_llmtext/h5 | 2026 | +0.52 | -4.98 | 0.000 | +0.67 | -3.43 | 0.0006 | 0.032 | KEEPS |
| firm | event_driven/C6_llmtext/h10 | 2026 | +0.24 | -4.43 | 0.001 | +0.25 | -2.62 | 0.0089 | 0.329 | **LOST** |
| firm | event_driven/C6_llmtext/h20 | 2026 | +0.21 | -3.58 | 0.013 | +0.18 | -2.17 | 0.0305 | 0.885 | **LOST** |
| firm | long_form/B3_lm_linear/h10 | 2026 | +0.26 | -3.63 | 0.012 | +0.09 | -1.56 | 0.1200 | 1.000 | **LOST** |
| firm | long_form/C6_llmtext/h20 | 2026 | +0.08 | -4.15 | 0.002 | +0.05 | -3.05 | 0.0024 | 0.108 | **LOST** |
| maximal | event_driven/C2_finbert_s1/h5 | 2026+2027+2028 | +0.77 | -2.41 | 0.608 | +1.81 | -5.29 | 0.0000 | 0.000 | **GAINED** |
| maximal | long_form/B2_tfidf_ridge/h5 | 2026 | +1.67 | -3.58 | 0.021 | +0.59 | +0.12 | 0.9035 | 1.000 | **LOST** |
| maximal | long_form/B2_tfidf_ridge/h10 | 2026 | +1.95 | -5.76 | 0.000 | +1.10 | -2.26 | 0.0242 | 1.000 | **LOST** |
| maximal | long_form/B2_tfidf_ridge/h20 | 2026 | +4.01 | -5.24 | 0.000 | +2.45 | -3.41 | 0.0007 | 0.046 | KEEPS |
| maximal | long_form/C2_finbert_s2/h5 | 2026+2027+2028 | +0.22 | -3.38 | 0.042 | -0.03 | +1.35 | 0.1765 | 1.000 | **LOST** |
| maximal | long_form/C2_finbert_s4/h20 | 2026+2027+2028 | +1.66 | -4.45 | 0.001 | +1.03 | -2.81 | 0.0051 | 0.310 | **LOST** |
| maximal | long_form/C4_longformer/h5 | 2026+2027+2028 | +0.47 | -3.91 | 0.006 | +0.57 | -3.30 | 0.0010 | 0.065 | **LOST** |
| maximal | long_form/C6_llmtext/h5 | 2026 | +1.05 | -4.86 | 0.000 | +0.37 | -2.02 | 0.0436 | 1.000 | **LOST** |
| maximal | long_form/C6_llmtext/h10 | 2026 | +1.07 | -4.66 | 0.000 | +0.23 | -1.61 | 0.1077 | 1.000 | **LOST** |
| maximal | long_form/D1_concat_mlp/h20 | 2026+2027+2028 | +0.00 | -5.10 | 0.000 | +0.00 | -2.74 | 0.0064 | 0.375 | **LOST** |

- (a) firm-identity: **12/69 raw, 3/69 Holm** in variance units (vol-unit: 15/8); convention-dependent Holm verdicts: 7/69.
- (b) maximal pool: **11/69 raw, 2/69 Holm** (vol-unit: 26/9); convention-dependent: 9/69.

## (c) Control intersection in variance units

| quantity | vol-unit (committed) | variance-unit (this run) |
|---|---|---|
| primary marginal raw / Holm | 46 / 38 | **38 / 21** |
| maximal raw / Holm | 26 / 9 | **11 / 2** |
| firm raw / Holm | 15 / 8 | **12 / 3** |
| FULL AND raw / Holm | 4 / 0 | **2 / 0** |
| strictest genuine AND | 0 | **0** |
| survivor-set overlap (maximal vs firm, Holm) | 0 (disjoint) | **0 (disjoint)** |

- Variance-unit Holm survivor sets: maximal = {event_driven/C2_finbert_s1/h5, long_form/B2_tfidf_ridge/h20}; firm = {event_driven/B2_tfidf_ridge/h10, event_driven/B3_lm_linear/h5, event_driven/C6_llmtext/h5}.
- Full-AND cells at raw p (variance-unit): long_form/D4_llmfused/h20; event_driven/B3_lm_linear/h5.
- Verdict: the two headline properties of the vol-unit cascade — Holm AND = 0 (HOLDS) and survivor-set disjointness (HOLDS) — under the proxy-robust convention.

## (d) The 8-K residual (event_driven C6_llmtext vs firm-identity reference)

Committed vol-unit values (firm_identity_ensemble.csv): h5 +0.52% (cluDM -4.98); h10 +0.24% (cluDM -4.43); h20 +0.21% (cluDM -3.58) — all clustered-significant, raw AND Holm (the paper's rounded +0.45/+0.25/+0.20 refers to the same cells across the identity-spec battery).

| h | n_test | n_days | rel% vol | DM vol | Holm vol | rel% var | DM var | p var | Holm var (F-firm) | daily-dQv 95% CI | survives raw | survives Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 23855 | 996 | +0.52 | -4.98 | 0.0000 | **+0.67** | **-3.43** | 0.0006 | 0.0319 | [-0.00453, -0.00105] | YES | YES |
| 10 | 22785 | 991 | +0.24 | -4.43 | 0.0005 | **+0.25** | **-2.62** | 0.0089 | 0.3294 | [-0.00125, -0.00015] | YES | no |
| 20 | 22318 | 981 | +0.21 | -3.58 | 0.0134 | **+0.18** | **-2.17** | 0.0305 | 0.8847 | [-0.00081, -0.00003] | YES | no |

**Residual verdict under the convention change:** 3/3 horizons survive at raw clustered p, 1/3 under Holm within the pre-declared 69-cell F-firm family (vol-unit: 3/3 raw, 3/3 Holm).

## (e) Standalone 180 vs A2 under day-clustered QLIKE

Basis inspection of the committed QLIKE table `dm_qlike_all_vs_A2.csv`: 57 rows, columns ['disc', 'model', 'h', 'DM', 'p', 'holm', 'bh'], max |DM| = 39.2 (obs-order HAC inflation), no n_days/cluster column, seed2026-only universe (scripts/analysis/qlike_dm.py) -> OBS-LEVEL. It is therefore NOT a valid day-clustered QLIKE grid; the 180 comparisons (dm_pairwise_clustered universe: seed-ensembled challengers, inner-joined per disclosure) are recomputed below. Full per-comparison detail: variance_unit_standalone180.csv (extra output file, disclosed here).

| loss | better than A2, raw p<.05 | better, Holm<.05 | significantly WORSE, Holm | n |
|---|---|---|---|---|
| SE (committed reference, gate G3) | 0 | 0 | 155 | 180 |
| QLIKE vol-unit (new) | 0 | 0 | 161 | 180 |
| **QLIKE variance-unit (new)** | **7** | **7** | **153** | 180 |

Comparisons better than A2 at raw p (either QLIKE unit):
| disclosure | h | challenger | DM se | DM qlike-vol | Holm | DM qlike-var | Holm |
|---|---|---|---|---|---|---|---|
| combined | 5 | A3_garch | +6.08 | +3.11 | 0.0019 | -4.34 | 0.0000 |
| combined | 10 | A3_garch | +4.78 | +1.50 | 0.1426 | -3.59 | 0.0010 |
| combined | 20 | A3_garch | +3.49 | +0.11 | 0.9770 | -3.61 | 0.0013 |
| event_driven | 5 | A3_garch | +4.33 | +2.15 | 0.0322 | -4.71 | 0.0000 |
| event_driven | 10 | A3_garch | +3.37 | +0.79 | 0.4308 | -3.91 | 0.0003 |
| event_driven | 20 | A3_garch | +2.58 | -0.30 | 0.9567 | -3.91 | 0.0004 |
| long_form | 5 | A4_egarch | +2.74 | +2.46 | 0.0141 | -2.92 | 0.0107 |

**Standalone verdict:** in variance-unit QLIKE, 7/180 comparisons beat A2 under Holm — but ALL 7 winners are GARCH-family PRICE baselines (A3_garch; A4_egarch); every text / text-fusion challenger stays at 0/153 better (Holm; 0/153 raw). The TEXT-standalone null (no text model beats HAR standalone) is convention-ROBUST; what the convention change flips is the intra-price ranking HAR-vs-GARCH, which is orthogonal to the paper's text claim. 153/180 significantly worse (descriptive; see family note 4).

## SANITY

- **G1 (named table: results/tables/m1_variance_unit.csv) PASS** — the full 69-cell variance-unit primary grid recomputed end-to-end (ensemble text -> log-space combiner -> q(y^2,f^2) -> day-clustered DM -> Holm) reproduces the committed rel%/DM/p/Holm columns: max|d rel| = 4.44e-16, max|d DM| = 1.78e-15, max|d p| = 9.97e-17, max|d Holm| = 9.71e-17 (vol-unit columns too: max|d rel| = 2.22e-16, max|d DM| = 8.88e-16).
- **G2 (named tables: firm_identity_ensemble.csv, maximal_reference_ensemble.csv) PASS** — the vol-unit legs of both cascade grids, recomputed in this script, match the committed tables on all 69 cells: firm max|d| = 4.44e-16, maximal max|d| = 1.78e-15.
- **G3 (named table: dm_pairwise_clustered.csv) PASS** — the day-clustered SE leg of the standalone universe matches the committed table on all 180 comparisons: max|d DM| = 1.78e-15, max|d p| = 9.70e-17, max|d Holm| = 9.47e-17.
- All gates enforced in scripts/analysis/variance_unit_cascade.py; the script aborts before writing any output if a gate fails.
- No look-ahead: every reference/combiner weight is the committed val-fit object, frozen on test; this script fits nothing.

## Bottom line

- The cascade's null architecture is convention-robust: in variance units the controls tighten rather than loosen — firm-identity survivors 15/8 -> **12/3** (raw/Holm), maximal-pool 26/9 -> **11/2**, full Holm AND stays **0/69**, survivor sets remain disjoint.
- The 8-K residual is **partially convention-dependent**: 3/3 horizons at raw clustered p and 1/3 under Holm in variance units (vol-unit 3/3 and 3/3).
- Standalone: the TEXT null is convention-robust — 0/153 text/fusion comparisons beat HAR in variance-unit QLIKE (Holm; 0 raw). The 7/180 total Holm winners are all GARCH-family price baselines (an intra-price HAR-vs-GARCH re-ranking under q(y^2,f^2), not a text result). The committed QLIKE table was obs-level and is superseded by variance_unit_standalone180.csv.