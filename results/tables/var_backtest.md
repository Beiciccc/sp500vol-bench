# R2 — VaR backtest: economic value of the text increment for downside risk

Three h-day vol forecasts per (disclosure, model, horizon): **rawHAR** (raw A2 HAR, known to under-forecast vol), **fR** (recalibrated price-only, log-space, frozen on val) and **fU** (fR + text). Left-tail VaR_alpha = mu_hat + Phi^-1(alpha)*sigma_h, sigma_h = sigma_ann*sqrt(h/252). Realized R_h = fwd_logret (ret_match_ok only). DM on the Gonzalez-Rivera tick loss vs fR: **negative = better than fR** (HAC lag h-1).

**Return-join sanity:** kept (ret_match_ok) = **544632** filing×cell rows across all cells; dropped = **47136** (8.0% dropped).


## Violation rates vs nominal alpha=0.01 (pooled-mean mu)

A well-calibrated forecast gives viol_rate near alpha. rawHAR under-forecasts vol so it should OVER-violate (viol_rate > alpha) more than fR.

| disclosure | model | h | viol rawHAR | viol fR | viol fU | Kupiec p (fR) | Kupiec p (fU) |
|---|---|---|---|---|---|---|---|
| event_driven | B2_tfidf_ridge | 5 | 0.0320 | 0.0333 | 0.0332 | 0.000 | 0.000 |
| event_driven | B2_tfidf_ridge | 10 | 0.0291 | 0.0258 | 0.0251 | 0.000 | 0.000 |
| event_driven | B2_tfidf_ridge | 20 | 0.0247 | 0.0161 | 0.0165 | 0.000 | 0.000 |
| event_driven | C2_finbert_s1 | 5 | 0.0320 | 0.0333 | 0.0319 | 0.000 | 0.000 |
| event_driven | C2_finbert_s1 | 10 | 0.0291 | 0.0258 | 0.0252 | 0.000 | 0.000 |
| event_driven | C2_finbert_s1 | 20 | 0.0247 | 0.0161 | 0.0154 | 0.000 | 0.000 |
| event_driven | C2_finbert_s2 | 5 | 0.0320 | 0.0333 | 0.0312 | 0.000 | 0.000 |
| event_driven | C2_finbert_s2 | 10 | 0.0291 | 0.0258 | 0.0256 | 0.000 | 0.000 |
| event_driven | C2_finbert_s2 | 20 | 0.0247 | 0.0161 | 0.0158 | 0.000 | 0.000 |
| event_driven | C4_longformer | 5 | 0.0320 | 0.0333 | 0.0336 | 0.000 | 0.000 |
| event_driven | C4_longformer | 10 | 0.0291 | 0.0258 | 0.0253 | 0.000 | 0.000 |
| event_driven | C4_longformer | 20 | 0.0247 | 0.0161 | 0.0160 | 0.000 | 0.000 |
| event_driven | C5_qwen3 | 5 | 0.0320 | 0.0333 | 0.0333 | 0.000 | 0.000 |
| event_driven | C5_qwen3 | 10 | 0.0291 | 0.0258 | 0.0259 | 0.000 | 0.000 |
| event_driven | C5_qwen3 | 20 | 0.0247 | 0.0161 | 0.0166 | 0.000 | 0.000 |
| event_driven | D2_gated_fusion | 5 | 0.0320 | 0.0333 | 0.0332 | 0.000 | 0.000 |
| event_driven | D2_gated_fusion | 10 | 0.0291 | 0.0258 | 0.0258 | 0.000 | 0.000 |
| event_driven | D2_gated_fusion | 20 | 0.0247 | 0.0161 | 0.0180 | 0.000 | 0.000 |
| long_form | B2_tfidf_ridge | 5 | 0.0317 | 0.0261 | 0.0264 | 0.000 | 0.000 |
| long_form | B2_tfidf_ridge | 10 | 0.0280 | 0.0177 | 0.0183 | 0.000 | 0.000 |
| long_form | B2_tfidf_ridge | 20 | 0.0223 | 0.0116 | 0.0114 | 0.199 | 0.242 |
| long_form | C2_finbert_s1 | 5 | 0.0317 | 0.0261 | 0.0257 | 0.000 | 0.000 |
| long_form | C2_finbert_s1 | 10 | 0.0280 | 0.0177 | 0.0191 | 0.000 | 0.000 |
| long_form | C2_finbert_s1 | 20 | 0.0223 | 0.0116 | 0.0116 | 0.199 | 0.199 |
| long_form | C2_finbert_s2 | 5 | 0.0317 | 0.0261 | 0.0265 | 0.000 | 0.000 |
| long_form | C2_finbert_s2 | 10 | 0.0280 | 0.0177 | 0.0179 | 0.000 | 0.000 |
| long_form | C2_finbert_s2 | 20 | 0.0223 | 0.0116 | 0.0117 | 0.199 | 0.162 |
| long_form | C4_longformer | 5 | 0.0317 | 0.0261 | 0.0261 | 0.000 | 0.000 |
| long_form | C4_longformer | 10 | 0.0280 | 0.0177 | 0.0177 | 0.000 | 0.000 |
| long_form | C4_longformer | 20 | 0.0223 | 0.0116 | 0.0117 | 0.199 | 0.162 |
| long_form | C5_qwen3 | 5 | 0.0317 | 0.0261 | 0.0261 | 0.000 | 0.000 |
| long_form | C5_qwen3 | 10 | 0.0280 | 0.0177 | 0.0181 | 0.000 | 0.000 |
| long_form | C5_qwen3 | 20 | 0.0223 | 0.0116 | 0.0099 | 0.199 | 0.908 |
| long_form | D2_gated_fusion | 5 | 0.0317 | 0.0261 | 0.0260 | 0.000 | 0.000 |
| long_form | D2_gated_fusion | 10 | 0.0280 | 0.0177 | 0.0184 | 0.000 | 0.000 |
| long_form | D2_gated_fusion | 20 | 0.0223 | 0.0116 | 0.0130 | 0.199 | 0.016 |

## Violation rates vs nominal alpha=0.05 (pooled-mean mu)

A well-calibrated forecast gives viol_rate near alpha. rawHAR under-forecasts vol so it should OVER-violate (viol_rate > alpha) more than fR.

| disclosure | model | h | viol rawHAR | viol fR | viol fU | Kupiec p (fR) | Kupiec p (fU) |
|---|---|---|---|---|---|---|---|
| event_driven | B2_tfidf_ridge | 5 | 0.0751 | 0.0781 | 0.0758 | 0.000 | 0.000 |
| event_driven | B2_tfidf_ridge | 10 | 0.0725 | 0.0653 | 0.0649 | 0.000 | 0.000 |
| event_driven | B2_tfidf_ridge | 20 | 0.0662 | 0.0508 | 0.0501 | 0.600 | 0.924 |
| event_driven | C2_finbert_s1 | 5 | 0.0751 | 0.0781 | 0.0751 | 0.000 | 0.000 |
| event_driven | C2_finbert_s1 | 10 | 0.0725 | 0.0653 | 0.0639 | 0.000 | 0.000 |
| event_driven | C2_finbert_s1 | 20 | 0.0662 | 0.0508 | 0.0474 | 0.600 | 0.068 |
| event_driven | C2_finbert_s2 | 5 | 0.0751 | 0.0781 | 0.0755 | 0.000 | 0.000 |
| event_driven | C2_finbert_s2 | 10 | 0.0725 | 0.0653 | 0.0660 | 0.000 | 0.000 |
| event_driven | C2_finbert_s2 | 20 | 0.0662 | 0.0508 | 0.0495 | 0.600 | 0.714 |
| event_driven | C4_longformer | 5 | 0.0751 | 0.0781 | 0.0777 | 0.000 | 0.000 |
| event_driven | C4_longformer | 10 | 0.0725 | 0.0653 | 0.0657 | 0.000 | 0.000 |
| event_driven | C4_longformer | 20 | 0.0662 | 0.0508 | 0.0509 | 0.600 | 0.518 |
| event_driven | C5_qwen3 | 5 | 0.0751 | 0.0781 | 0.0777 | 0.000 | 0.000 |
| event_driven | C5_qwen3 | 10 | 0.0725 | 0.0653 | 0.0658 | 0.000 | 0.000 |
| event_driven | C5_qwen3 | 20 | 0.0662 | 0.0508 | 0.0514 | 0.600 | 0.326 |
| event_driven | D2_gated_fusion | 5 | 0.0751 | 0.0781 | 0.0776 | 0.000 | 0.000 |
| event_driven | D2_gated_fusion | 10 | 0.0725 | 0.0653 | 0.0663 | 0.000 | 0.000 |
| event_driven | D2_gated_fusion | 20 | 0.0662 | 0.0508 | 0.0538 | 0.600 | 0.011 |
| long_form | B2_tfidf_ridge | 5 | 0.0759 | 0.0653 | 0.0668 | 0.000 | 0.000 |
| long_form | B2_tfidf_ridge | 10 | 0.0756 | 0.0546 | 0.0550 | 0.081 | 0.057 |
| long_form | B2_tfidf_ridge | 20 | 0.0640 | 0.0368 | 0.0375 | 0.000 | 0.000 |
| long_form | C2_finbert_s1 | 5 | 0.0759 | 0.0653 | 0.0634 | 0.000 | 0.000 |
| long_form | C2_finbert_s1 | 10 | 0.0756 | 0.0546 | 0.0572 | 0.081 | 0.006 |
| long_form | C2_finbert_s1 | 20 | 0.0640 | 0.0368 | 0.0366 | 0.000 | 0.000 |
| long_form | C2_finbert_s2 | 5 | 0.0759 | 0.0653 | 0.0652 | 0.000 | 0.000 |
| long_form | C2_finbert_s2 | 10 | 0.0756 | 0.0546 | 0.0540 | 0.081 | 0.125 |
| long_form | C2_finbert_s2 | 20 | 0.0640 | 0.0368 | 0.0380 | 0.000 | 0.000 |
| long_form | C4_longformer | 5 | 0.0759 | 0.0653 | 0.0640 | 0.000 | 0.000 |
| long_form | C4_longformer | 10 | 0.0756 | 0.0546 | 0.0550 | 0.081 | 0.057 |
| long_form | C4_longformer | 20 | 0.0640 | 0.0368 | 0.0355 | 0.000 | 0.000 |
| long_form | C5_qwen3 | 5 | 0.0759 | 0.0653 | 0.0633 | 0.000 | 0.000 |
| long_form | C5_qwen3 | 10 | 0.0756 | 0.0546 | 0.0529 | 0.081 | 0.267 |
| long_form | C5_qwen3 | 20 | 0.0640 | 0.0368 | 0.0351 | 0.000 | 0.000 |
| long_form | D2_gated_fusion | 5 | 0.0759 | 0.0653 | 0.0656 | 0.000 | 0.000 |
| long_form | D2_gated_fusion | 10 | 0.0756 | 0.0546 | 0.0548 | 0.081 | 0.064 |
| long_form | D2_gated_fusion | 20 | 0.0640 | 0.0368 | 0.0404 | 0.000 | 0.000 |

**Over-violation sanity:** rawHAR viol_rate > alpha in 100% of cells vs fR in 92%; rawHAR over-violates by MORE than fR in 83% of cells (expected: rawHAR under-forecasts vol => over-violates).


## Does text improve VaR? DM on tick loss, fU vs fR (pooled-mean mu)

negative DM = text (fU) LOWERS tick loss vs recalibrated HAR (fR). 'closer' = fU violation rate is nearer nominal than fR.

| disclosure | h | alpha | model | DM(fU vs fR) | p | text better? | viol closer? |
|---|---|---|---|---|---|---|---|
| event_driven | 5 | 0.01 | B2_tfidf_ridge | -1.77 | 0.077 | yes | yes |
| event_driven | 5 | 0.01 | C2_finbert_s1 | -7.76 | 0.000 | yes* | yes |
| event_driven | 5 | 0.01 | C2_finbert_s2 | -4.54 | 0.000 | yes* | yes |
| event_driven | 5 | 0.01 | C4_longformer | -3.86 | 0.000 | yes* | no |
| event_driven | 5 | 0.01 | C5_qwen3 | +3.43 | 0.001 | no | yes |
| event_driven | 5 | 0.01 | D2_gated_fusion | -0.69 | 0.488 | yes | yes |
| event_driven | 5 | 0.05 | B2_tfidf_ridge | -2.11 | 0.035 | yes* | yes |
| event_driven | 5 | 0.05 | C2_finbert_s1 | -5.74 | 0.000 | yes* | yes |
| event_driven | 5 | 0.05 | C2_finbert_s2 | -3.53 | 0.000 | yes* | yes |
| event_driven | 5 | 0.05 | C4_longformer | -3.29 | 0.001 | yes* | yes |
| event_driven | 5 | 0.05 | C5_qwen3 | +3.29 | 0.001 | no | yes |
| event_driven | 5 | 0.05 | D2_gated_fusion | -1.00 | 0.316 | yes | yes |
| event_driven | 10 | 0.01 | B2_tfidf_ridge | -1.76 | 0.078 | yes | yes |
| event_driven | 10 | 0.01 | C2_finbert_s1 | -4.23 | 0.000 | yes* | yes |
| event_driven | 10 | 0.01 | C2_finbert_s2 | -2.40 | 0.016 | yes* | yes |
| event_driven | 10 | 0.01 | C4_longformer | -4.40 | 0.000 | yes* | yes |
| event_driven | 10 | 0.01 | C5_qwen3 | +2.12 | 0.034 | no | no |
| event_driven | 10 | 0.01 | D2_gated_fusion | +2.82 | 0.005 | no | no |
| event_driven | 10 | 0.05 | B2_tfidf_ridge | -2.50 | 0.012 | yes* | yes |
| event_driven | 10 | 0.05 | C2_finbert_s1 | -4.01 | 0.000 | yes* | yes |
| event_driven | 10 | 0.05 | C2_finbert_s2 | -4.11 | 0.000 | yes* | no |
| event_driven | 10 | 0.05 | C4_longformer | -3.60 | 0.000 | yes* | no |
| event_driven | 10 | 0.05 | C5_qwen3 | +2.66 | 0.008 | no | no |
| event_driven | 10 | 0.05 | D2_gated_fusion | +0.38 | 0.706 | no | no |
| event_driven | 20 | 0.01 | B2_tfidf_ridge | -2.06 | 0.040 | yes* | no |
| event_driven | 20 | 0.01 | C2_finbert_s1 | -2.21 | 0.027 | yes* | yes |
| event_driven | 20 | 0.01 | C2_finbert_s2 | -3.54 | 0.000 | yes* | yes |
| event_driven | 20 | 0.01 | C4_longformer | +3.56 | 0.000 | no | yes |
| event_driven | 20 | 0.01 | C5_qwen3 | +3.62 | 0.000 | no | no |
| event_driven | 20 | 0.01 | D2_gated_fusion | +1.19 | 0.233 | no | no |
| event_driven | 20 | 0.05 | B2_tfidf_ridge | -2.21 | 0.027 | yes* | yes |
| event_driven | 20 | 0.05 | C2_finbert_s1 | -0.81 | 0.417 | yes | no |
| event_driven | 20 | 0.05 | C2_finbert_s2 | -1.39 | 0.165 | yes | yes |
| event_driven | 20 | 0.05 | C4_longformer | +5.64 | 0.000 | no | no |
| event_driven | 20 | 0.05 | C5_qwen3 | +4.48 | 0.000 | no | no |
| event_driven | 20 | 0.05 | D2_gated_fusion | -0.60 | 0.551 | yes | no |
| long_form | 5 | 0.01 | B2_tfidf_ridge | -0.59 | 0.558 | yes | no |
| long_form | 5 | 0.01 | C2_finbert_s1 | -4.69 | 0.000 | yes* | yes |
| long_form | 5 | 0.01 | C2_finbert_s2 | -0.30 | 0.767 | yes | no |
| long_form | 5 | 0.01 | C4_longformer | -0.69 | 0.491 | yes | no |
| long_form | 5 | 0.01 | C5_qwen3 | -0.42 | 0.673 | yes | no |
| long_form | 5 | 0.01 | D2_gated_fusion | +4.01 | 0.000 | no | yes |
| long_form | 5 | 0.05 | B2_tfidf_ridge | -2.98 | 0.003 | yes* | no |
| long_form | 5 | 0.05 | C2_finbert_s1 | -2.21 | 0.027 | yes* | yes |
| long_form | 5 | 0.05 | C2_finbert_s2 | -1.31 | 0.189 | yes | yes |
| long_form | 5 | 0.05 | C4_longformer | -2.24 | 0.025 | yes* | yes |
| long_form | 5 | 0.05 | C5_qwen3 | -1.31 | 0.191 | yes | yes |
| long_form | 5 | 0.05 | D2_gated_fusion | +3.01 | 0.003 | no | no |
| long_form | 10 | 0.01 | B2_tfidf_ridge | -0.86 | 0.389 | yes | no |
| long_form | 10 | 0.01 | C2_finbert_s1 | -0.35 | 0.729 | yes | no |
| long_form | 10 | 0.01 | C2_finbert_s2 | -0.28 | 0.780 | yes | no |
| long_form | 10 | 0.01 | C4_longformer | +0.73 | 0.463 | no | no |
| long_form | 10 | 0.01 | C5_qwen3 | -1.15 | 0.251 | yes | no |
| long_form | 10 | 0.01 | D2_gated_fusion | -0.51 | 0.613 | yes | no |
| long_form | 10 | 0.05 | B2_tfidf_ridge | -3.21 | 0.001 | yes* | no |
| long_form | 10 | 0.05 | C2_finbert_s1 | -2.98 | 0.003 | yes* | no |
| long_form | 10 | 0.05 | C2_finbert_s2 | -1.09 | 0.275 | yes | yes |
| long_form | 10 | 0.05 | C4_longformer | +3.88 | 0.000 | no | no |
| long_form | 10 | 0.05 | C5_qwen3 | -2.11 | 0.035 | yes* | yes |
| long_form | 10 | 0.05 | D2_gated_fusion | -0.64 | 0.525 | yes | no |
| long_form | 20 | 0.01 | B2_tfidf_ridge | -2.99 | 0.003 | yes* | yes |
| long_form | 20 | 0.01 | C2_finbert_s1 | -0.07 | 0.940 | yes | no |
| long_form | 20 | 0.01 | C2_finbert_s2 | -1.58 | 0.114 | yes | no |
| long_form | 20 | 0.01 | C4_longformer | -1.95 | 0.051 | yes | no |
| long_form | 20 | 0.01 | C5_qwen3 | -2.46 | 0.014 | yes* | yes |
| long_form | 20 | 0.01 | D2_gated_fusion | -1.44 | 0.149 | yes | no |
| long_form | 20 | 0.05 | B2_tfidf_ridge | -6.18 | 0.000 | yes* | yes |
| long_form | 20 | 0.05 | C2_finbert_s1 | +0.83 | 0.404 | no | no |
| long_form | 20 | 0.05 | C2_finbert_s2 | +0.17 | 0.867 | no | yes |
| long_form | 20 | 0.05 | C4_longformer | -1.44 | 0.151 | yes | no |
| long_form | 20 | 0.05 | C5_qwen3 | -2.01 | 0.045 | yes* | no |
| long_form | 20 | 0.05 | D2_gated_fusion | -1.40 | 0.160 | yes | yes |

## Tally (pooled-mu, 72 disclosure×model×horizon×alpha cells)

- Text LOWERS tick loss (DM<0) in **55/72** cells; significantly (p<.05) in **29**; significantly WORSE in **12**.

- Text moves the violation rate CLOSER to nominal in **35/72** cells.

- Recalibration (fR vs rawHAR) improves tick loss in **60/72** cells (rawHAR worse than fR), significantly in **54**.


## Caveats

- **Overlapping windows:** h-day return windows from filings < h trading days apart overlap, inducing serial dependence the Christoffersen independence test does not model; its p-values are indicative only. Kupiec UC and tick loss are unaffected; DM uses HAC lag h-1.

- Gaussian VaR assumes a Gaussian return given the vol forecast; fat left tails will raise violations for all three forecasts alike, so the fU-vs-fR comparison stays fair.
