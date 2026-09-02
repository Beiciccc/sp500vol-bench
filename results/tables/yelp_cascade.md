# Yelp second-domain cascade table [REAL]

> Yelp Open Dataset — business-month rating forecasting

| # | arm | reference | h=1m MSE | Δ rel% | DM p | h=3m MSE | Δ rel% | DM p |
|---|---|---|---|---|---|---|---|---|
| 1 | naive pooled-split text (random 80/20; field design) | pooled mean | 0.4259 | +28.43 | n/a | 0.2981 | +36.64 | n/a |
| 2 | chronological text-alone (log-recalibrated) | recalibrated AR f_R | 0.4924 | -24.70 | 0.0000 | 0.3537 | -35.29 | 0.0000 |
| 3 | AR + text combiner f_U | recalibrated AR f_R | 0.3935 | +0.35 | 0.0000 | 0.2592 | +0.87 | 0.0000 |
| 4 | AR + entity-mean (identity control, zero text) | recalibrated AR f_R | 0.4367 | -10.58 | 0.0000 | 0.2863 | -9.48 | 0.0000 |
| 5 | AR + entity-mean + text (RESIDUAL text increment) | AR + entity-mean f_Re | 0.4351 | +0.37 | 0.0000 | 0.2845 | +0.61 | 0.0000 |

**Row notes.** row 1 (1m): identity share (entity-mean-only, zero text) = 139% of the apparent gain · row 1 (3m): identity share (entity-mean-only, zero text) = 148% of the apparent gain · row 2 (1m): DM +34.88 · row 2 (3m): DM +33.35 · row 3 (1m): DM -5.35; two-way p=0.0000 · row 3 (3m): DM -6.09; two-way p=0.0000 · row 4 (1m): DM +14.74; chrono identity share -3062% of row-3 gain · row 4 (3m): DM +7.40; chrono identity share -1089% of row-3 gain · row 5 (1m): DM -12.65; two-way p=0.0000; placebo mean DM +1.31 (mean p=0.363) · row 5 (3m): DM -20.54; two-way p=0.0000; placebo mean DM +0.34 (mean p=0.457)

**Table note (pre-registered).** Loss = squared error on stars (MSE); inference = month-clustered DM (HAC lag = h-1 months, HLN, t(n_months-1)); robustness = business x month two-way CGM. Combiner weights are validation-fit and test-frozen; entity means use train+val observed monthly stars only. MDE (80% power, 5% size, signal-injection methodology with a DISCLOSED oracle entity-orthogonal injection): h=1m: AR stage 0.18%, entity stage 0.08%; h=3m: AR stage 0.40%, entity stage 0.08%. Boundary: h=1m 0 val rows with outcome windows crossing the test start (embargo=off); h=3m 7909 val rows with outcome windows crossing the test start (embargo=off).

## SANITY (gates auto-filled from the numbers)

| gate | verdict | detail |
|---|---|---|
| G1 panel shape | PASS | 8,474 entities, 407,385 events; val>=100 & test>=30 per horizon: True |
| G2 AR baseline sane | PASS | h=1: b=1.162, AR recal 0.3949 vs global 0.6726 / last 0.6509; h=3: b=1.149, AR recal 0.2615 vs global 0.5441 / last 0.5643 |
| G3 naive arm credits text | PASS | h=1: +28.4%; h=3: +36.6% |
| G4 label-shuffle placebos clean (pre-registered primary) | PASS | max |mean DM| = 1.31 (threshold 2.0); all mean p > .05: True |
| G4b within-month text-swap (diagnostic, fully disclosed) | PASS | h=1 row3_swap: mean DM -0.21, mean p 0.459; h=1 row5_swap: mean DM -0.16, mean p 0.463; h=3 row3_swap: mean DM +1.08, mean p 0.193; h=3 row5_swap: mean DM +1.40, mean p 0.047 (BORDERLINE) |
| G5 MDE <= naive apparent gain | PASS | h=1: MDE 0.18% vs naive gain 28.4%; h=3: MDE 0.40% vs naive gain 36.6% |

## HONEST HEADLINE (auto-filled)

Under the field-standard pooled random split the text model appears to cut MSE by 28.4% (row 1, h=1m); a zero-text business-mean predictor recovers 139% of that apparent gain under the same split. Under the chronological protocol, text alone loses to the recalibrated AR baseline (-24.7%, row 2); with the entity-mean identity control in the reference, the residual text increment is +0.37% (month-clustered DM -12.65, p=0.0000; two-way p=0.0000; label-shuffle placebo mean DM +1.31, mean p=0.363; MDE 0.18% at 80% power).