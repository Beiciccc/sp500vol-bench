# Yelp second-domain cascade table [SYNTHETIC]

> SYNTHETIC FIXTURE — MACHINERY VALIDATION ONLY (known injected DGP; never citable as a Yelp result)

| # | arm | reference | h=1m MSE | Δ rel% | DM p | h=3m MSE | Δ rel% | DM p |
|---|---|---|---|---|---|---|---|---|
| 1 | naive pooled-split text (random 80/20; field design) | pooled mean | 0.1656 | +50.03 | n/a | 0.0875 | +64.95 | n/a |
| 2 | chronological text-alone (log-recalibrated) | recalibrated AR f_R | 0.1787 | -21.58 | 0.0000 | 0.1007 | -23.88 | 0.0000 |
| 3 | AR + text combiner f_U | recalibrated AR f_R | 0.1439 | +2.06 | 0.0000 | 0.0775 | +4.68 | 0.0000 |
| 4 | AR + entity-mean (identity control, zero text) | recalibrated AR f_R | 0.1475 | -0.35 | 0.6417 | 0.0776 | +4.45 | 0.0002 |
| 5 | AR + entity-mean + text (RESIDUAL text increment) | AR + entity-mean f_Re | 0.1471 | +0.27 | 0.0004 | 0.0777 | -0.05 | 0.0034 |

**Row notes.** row 1 (1m): identity share (entity-mean-only, zero text) = 107% of the apparent gain · row 1 (3m): identity share (entity-mean-only, zero text) = 106% of the apparent gain · row 2 (1m): DM +13.44 · row 2 (3m): DM +12.36 · row 3 (1m): DM -5.65; two-way p=0.0000 · row 3 (3m): DM -8.86; two-way p=0.0000 · row 4 (1m): DM +0.47; chrono identity share -17% of row-3 gain · row 4 (3m): DM -4.12; chrono identity share 95% of row-3 gain · row 5 (1m): DM -3.85; two-way p=0.0005; placebo mean DM +1.72 (mean p=0.300) · row 5 (3m): DM +3.09; two-way p=0.0036; placebo mean DM +0.34 (mean p=0.298)

**Table note (pre-registered).** Loss = squared error on stars (MSE); inference = month-clustered DM (HAC lag = h-1 months, HLN, t(n_months-1)); robustness = business x month two-way CGM. Combiner weights are validation-fit and test-frozen; entity means use train+val observed monthly stars only. MDE (80% power, 5% size, signal-injection methodology with a DISCLOSED oracle entity-orthogonal injection): h=1m: AR stage 1.01%, entity stage 0.20%; h=3m: AR stage 1.41%, entity stage 0.04%. Boundary: h=1m 0 val rows with outcome windows crossing the test start (embargo=off); h=3m 356 val rows with outcome windows crossing the test start (embargo=off).

## SANITY (gates auto-filled from the numbers)

| gate | verdict | detail |
|---|---|---|
| G1 panel shape | PASS | 200 entities, 19,171 events; val>=100 & test>=30 per horizon: True |
| G2 AR baseline sane | PASS | h=1: b=1.044, AR recal 0.1469 vs global 0.3264 / last 0.2242; h=3: b=1.035, AR recal 0.0813 vs global 0.2517 / last 0.1695 |
| G3 naive arm credits text | PASS | h=1: +50.0%; h=3: +64.9% |
| G4 placebos clean | PASS | max |mean DM| across horizons/stages/placebos = 1.72 (threshold 2.0); all mean p > .05: True |
| G5 MDE <= naive apparent gain | PASS | h=1: MDE 1.01% vs naive gain 50.0%; h=3: MDE 1.41% vs naive gain 64.9% |

## RECOVERY — machinery validation against the KNOWN injected structure

| check | verdict | detail |
|---|---|---|
| R1 combiner recalibration sane (b in [0.5,1.5]; near-neutral on an already-calibrated baseline) | PASS | h=1: b=1.044, MSE 0.1462->0.1469; h=3: b=1.035, MSE 0.0812->0.0813 |
| R2 text-alone loses chronologically (h=1) | PASS | -21.58%, DM +13.44, p=0.0000 |
| R3 entity-mean absorbs the entity effect | PASS | naive identity share >= 106% at every horizon (>=50 required); chronological absorption (row-4 sig. + chrono share >=50%) at h in [3]; h=1: row4 -0.35% (p=0.6417), chrono share -17%; h=3: row4 +4.45% (p=0.0002), chrono share 95% |
| R4 residual text effect recovered (h=1 band 0.25-1.10x text-oracle; other horizons no invented residual) | PASS | h=1: residual +0.27% vs text-oracle +0.41% (recovered 68%, p=0.0004; fixture arm extracted 9% of the DGP-injected signal); h=3: residual -0.05% vs text-oracle +0.22% (recovered -22%, p=0.0034; fixture arm extracted 29% of the DGP-injected signal) |
| R5 placebos kill the increment | PASS | label-shuffle + within-month swap, both stages, all horizons |
| R6 injection machinery (oracle, disclosed): calibration converges, injected signal detected at the AR stage | PASS | h=1: converged=True, adaptive 3.06% (genuine injection kappa=+0.0053) detect AR=True; entity-stage transmission detect=True (deployed loading kappa_ent=+0.0011, reported not gated); h=3: converged=True, adaptive 5.68% (genuine injection kappa=+0.0055) detect AR=True; entity-stage transmission detect=False (deployed loading kappa_ent=-0.0002, reported not gated) |

## HONEST HEADLINE (auto-filled)

Under the field-standard pooled random split the text model appears to cut MSE by 50.0% (row 1, h=1m); a zero-text business-mean predictor recovers 107% of that apparent gain under the same split. Under the chronological protocol, text alone loses to the recalibrated AR baseline (-21.6%, row 2); with the entity-mean identity control in the reference, the residual text increment is +0.27% (month-clustered DM -3.85, p=0.0004; two-way p=0.0005; label-shuffle placebo mean DM +1.72, mean p=0.300; MDE 1.01% at 80% power).