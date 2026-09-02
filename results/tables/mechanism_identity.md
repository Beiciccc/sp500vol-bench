# Pre-registered analysis C — is the identity control's size predictable from the baseline's entity encoding?

> prereg: configs/prereg_mechanism_and_labels.md §C, tag prereg-cd-v1.0 (committed before any statistic here). Single-shot.

## The 16 points

| panel | cell | x = entity R² of reference val preds | y = identity-control gain (rel%) | n_val | n_entities |
|---|---|---|---|---|---|
| SEC | long_form/h5 | 0.3695 | +0.173 | 3734 | 526 |
| SEC | long_form/h10 | 0.3788 | +1.700 | 3609 | 527 |
| SEC | long_form/h20 | 0.3649 | -1.391 | 3627 | 529 |
| SEC | event_driven/h5 | 0.3070 | +1.977 | 13489 | 532 |
| SEC | event_driven/h10 | 0.3080 | +1.502 | 13131 | 532 |
| SEC | event_driven/h20 | 0.3032 | +0.124 | 13101 | 532 |
| Yelp | chrono/h1m | 0.9499 | -10.581 | 38292 | 6648 |
| Yelp | chrono/h3m | 0.9424 | -9.484 | 51859 | 7407 |
| MAEC | primary/r_ar/h3 | 0.9802 | +14.967 | 333 | 317 |
| MAEC | primary/r_har/h3 | 0.9951 | +5.266 | 333 | 317 |
| MAEC | primary/r_ar/h7 | 0.9762 | +20.824 | 333 | 317 |
| MAEC | primary/r_har/h7 | 0.9949 | +6.633 | 333 | 317 |
| MAEC | primary/r_ar/h15 | 0.9746 | +16.855 | 333 | 317 |
| MAEC | primary/r_har/h15 | 0.9940 | +5.922 | 333 | 317 |
| MAEC | primary/r_ar/h30 | 0.9809 | +11.527 | 333 | 317 |
| MAEC | primary/r_har/h30 | 0.9921 | +5.241 | 333 | 317 |

## Pre-registered statistics

- **Spearman rho (16 points)** = **+0.5853** (prediction: rho < 0)
- Permutation p (one-sided, 10,000 draws, seed 2026, add-one): **global = 0.9908**, **within-panel = 0.0259**; conservative (prose) = **0.9908**
- Median-split Fisher exact (median x = 0.9623; table [[hi: y<=0 0, y>0 8], [lo: y<=0 3, y>0 5]]): one-sided (predicted direction) p = **1.0000**, two-sided p = 0.2000

## Verdict — fired branch: **FALSIFIED**

Falsification branch fired: rho >= 0 or both permutation p-values > .10 — the mechanism claim does not hold; the paper retains descriptive wording and does not escalate it; this result enters FACTS and the main text (one sentence of honest disclosure).

## Disclosures

- **Non-independence**: the 16 cells come from **3 panels** (SEC 6, Yelp 2, MAEC 8); cells share references, entities and overlapping outcome windows within a panel, so they are not independent draws. Per the prereg this is met with the double-reported permutation (within-panel permutation preserves the panel structure under the null) and the prose takes the conservative p.
- **x definition**: R² of OLS of the reference's log-space VALIDATION predictions on entity dummies (computed as the exact one-way-ANOVA between-group variance share), within the same panel/horizon/reference as y. SEC: log f_R val fit (f_R = exp OLS[1, log A2-HAR], val-fit on the 5-price-model inner-join panel — the committed firm-control code path); Yelp: log of the clipped recalibrated-AR val fit (yelp_protocol log_ols_frozen applied to val); MAEC: stored val predictions of the fit-stage reference halves (labels are already v = log vol).
- **No test labels**: every parquet is read with a split=='val' predicate pushdown (test rows never materialised); the MAEC read omits the label column entirely; _assert_val_only() guards every frame. Val labels enter only the val-side recalibration fits (prereg boundary: 'C reads only frozen predictions and val labels'). y is taken from frozen artifacts only — nothing on the y side was recomputed from raw predictions.
- **Sources, y**: SEC 6 cells = results/tables/firm_identity_ensemble.csv column rel_impr_firmMeanOnly_vs_fR (verified unique within each (disc, h) across all 69 rows — max spread exactly 0.0); Yelp 2 cells = results/tables/yelp_cascade.csv row 4 (entity-mean control, chronological) at full CSV precision — the generating protocol_results.json (tfidf primary) is no longer on disk, and the cascade CSV is the frozen per-cell artifact of the same pipeline (yelp_cascade_table.py); MAEC 8 cells = results/second_domain/maec/protocol_tfidf_primary.json horizons[h][ref].entity.delta_rel_pct (STPEV expanding control, field names verified in maec_protocol.py).
- **Sources, x**: SEC = results/runs/{A2_har_rv,A6_shar,A3_garch,A4_egarch,A5_arima}_full_{disc}_seed2026/predictions.parquet val rows; Yelp = results/second_domain/preds/preds_ar_ridge.parquet val rows (∩ preds_tfidf_chrono row set, no rows dropped); MAEC = results/second_domain/maec/preds/preds_r_{ar,har}_primary.parquet val rows.
- **Permutation conventions**: 10,000 draws each scheme from a single np.random.default_rng(2026) stream (global draws first, then within-panel), one-sided in the registered direction (rho < 0), add-one p = (1 + #{rho_perm <= rho_obs}) / (N + 1).
- **Mechanical caveat on x**: entities with few val rows fit their dummies near-perfectly, so x levels are panel-size dependent (n_val/n_entities per cell in the table); the statistic is rank-based and the within-panel permutation is unaffected.

## SANITY (anchors reproduced)

- SEC anchor PASS: 69-cell mean +0.522793% prints as +0.52%, firm_beats_fR = 53/69 (committed md: '53/69 (mean +0.52%)')
- SEC per-(disc,h) uniqueness PASS: max within-cell spread of rel_impr_firmMeanOnly_vs_fR across models = 0.0 (machine exact)
- Yelp anchor PASS: row-4 entity-mean control = -10.581141% / -9.483860% (md prints −10.58 / −9.48)
- MAEC anchor PASS: all 8 cells' row3 rel_pct (4 dp) and identity_share (2 dp) recomputed from the protocol json match the committed maec_audit.csv

Generated 2026-07-15 16:14:45 in 1.1s by scripts/analysis/mechanism_identity_predictability.py; outputs: mechanism_identity.csv/.md + writing/paper/figures/mechanism_identity.pdf