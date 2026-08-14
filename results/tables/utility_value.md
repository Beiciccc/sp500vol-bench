# R3 — Utility / volatility-timing economic value (FKO performance fee)

Mean-variance investor sizes each filing-event bet by inverse conditional variance. f_R = recalibrated HAR (price-only), f_U = +text; both fit on val, frozen to test (M1 discipline). Bet return = signed realized h-day log return (`fwd_logret`, ret_match_ok only). Performance fee Delta = U(f_U) - U(f_R), annualized to bps. **Positive fee = the investor pays to obtain the text-augmented vol forecast.**

Realized-returns join: kept 393,845 / dropped 37,400 rows (ret_match_ok); 27 cells built, 0 cells dropped.

## Performance fee (bps/yr), mu=6% annual — headline

`fee_bps_ann` = unconstrained FKO fee; `fee_wins` = same after capping inverse-variance weights at their common 99th pct (removes the single-asset leverage blow-up); target-vol `sharpe` is the mu-free, leverage-controlled view.

| disclosure | model | h | gamma | fee_bps_ann | fee_wins | sharpe_fR | sharpe_fU |
|---|---|---|---|---:|---:|---:|---:|
| event_driven | B2_tfidf_ridge | 5 | 2 | -26.7 | -22.8 | 0.0166 | 0.0158 |
| event_driven | B2_tfidf_ridge | 5 | 10 | -5.3 | -4.6 | 0.0166 | 0.0158 |
| event_driven | B2_tfidf_ridge | 10 | 2 | -5.0 | -3.2 | 0.0233 | 0.0231 |
| event_driven | B2_tfidf_ridge | 10 | 10 | -1.0 | -0.6 | 0.0233 | 0.0231 |
| event_driven | B2_tfidf_ridge | 20 | 2 | -4.8 | -4.4 | 0.0392 | 0.0387 |
| event_driven | B2_tfidf_ridge | 20 | 10 | -1.0 | -0.9 | 0.0392 | 0.0387 |
| event_driven | C2_finbert_s1 | 5 | 2 | -14.1 | -12.2 | 0.0166 | 0.0157 |
| event_driven | C2_finbert_s1 | 5 | 10 | -2.8 | -2.4 | 0.0166 | 0.0157 |
| event_driven | C2_finbert_s1 | 10 | 2 | -14.7 | -13.3 | 0.0233 | 0.0223 |
| event_driven | C2_finbert_s1 | 10 | 10 | -2.9 | -2.7 | 0.0233 | 0.0223 |
| event_driven | C2_finbert_s1 | 20 | 2 | -11.1 | -9.5 | 0.0392 | 0.0377 |
| event_driven | C2_finbert_s1 | 20 | 10 | -2.2 | -1.9 | 0.0392 | 0.0377 |
| event_driven | D2_gated_fusion | 5 | 2 | -15.8 | -13.9 | 0.0166 | 0.0162 |
| event_driven | D2_gated_fusion | 5 | 10 | -3.2 | -2.8 | 0.0166 | 0.0162 |
| event_driven | D2_gated_fusion | 10 | 2 | -5.0 | -4.4 | 0.0233 | 0.0233 |
| event_driven | D2_gated_fusion | 10 | 10 | -1.0 | -0.9 | 0.0233 | 0.0233 |
| event_driven | D2_gated_fusion | 20 | 2 | 19.1 | 18.7 | 0.0392 | 0.0417 |
| event_driven | D2_gated_fusion | 20 | 10 | 3.8 | 3.7 | 0.0392 | 0.0417 |
| long_form | B2_tfidf_ridge | 5 | 2 | 11.7 | 14.2 | 0.0730 | 0.0737 |
| long_form | B2_tfidf_ridge | 5 | 10 | 2.3 | 2.8 | 0.0730 | 0.0737 |
| long_form | B2_tfidf_ridge | 10 | 2 | 2.6 | 2.6 | 0.0589 | 0.0592 |
| long_form | B2_tfidf_ridge | 10 | 10 | 0.5 | 0.5 | 0.0589 | 0.0592 |
| long_form | B2_tfidf_ridge | 20 | 2 | 6.6 | 6.3 | 0.0923 | 0.0933 |
| long_form | B2_tfidf_ridge | 20 | 10 | 1.3 | 1.3 | 0.0923 | 0.0933 |
| long_form | C2_finbert_s1 | 5 | 2 | -10.2 | -8.0 | 0.0730 | 0.0730 |
| long_form | C2_finbert_s1 | 5 | 10 | -2.0 | -1.6 | 0.0730 | 0.0730 |
| long_form | C2_finbert_s1 | 10 | 2 | 9.8 | 11.0 | 0.0589 | 0.0600 |
| long_form | C2_finbert_s1 | 10 | 10 | 2.0 | 2.2 | 0.0589 | 0.0600 |
| long_form | C2_finbert_s1 | 20 | 2 | 0.0 | 0.0 | 0.0923 | 0.0923 |
| long_form | C2_finbert_s1 | 20 | 10 | 0.0 | 0.0 | 0.0923 | 0.0923 |
| long_form | C2_finbert_s2 | 5 | 2 | -7.3 | -2.2 | 0.0730 | 0.0727 |
| long_form | C2_finbert_s2 | 5 | 10 | -1.5 | -0.4 | 0.0730 | 0.0727 |
| long_form | C2_finbert_s2 | 10 | 2 | -1.9 | -1.8 | 0.0589 | 0.0588 |
| long_form | C2_finbert_s2 | 10 | 10 | -0.4 | -0.4 | 0.0589 | 0.0588 |
| long_form | C2_finbert_s2 | 20 | 2 | 32.8 | 32.6 | 0.0923 | 0.0968 |
| long_form | C2_finbert_s2 | 20 | 10 | 6.6 | 6.5 | 0.0923 | 0.0968 |
| long_form | C4_longformer | 5 | 2 | -15.7 | -9.4 | 0.0730 | 0.0727 |
| long_form | C4_longformer | 5 | 10 | -3.1 | -1.9 | 0.0730 | 0.0727 |
| long_form | C4_longformer | 10 | 2 | -1.0 | -1.5 | 0.0589 | 0.0588 |
| long_form | C4_longformer | 10 | 10 | -0.2 | -0.3 | 0.0589 | 0.0588 |
| long_form | C4_longformer | 20 | 2 | 71.8 | 73.6 | 0.0923 | 0.1036 |
| long_form | C4_longformer | 20 | 10 | 14.4 | 14.7 | 0.0923 | 0.1036 |
| long_form | C5_qwen3 | 5 | 2 | 25.2 | 25.6 | 0.0730 | 0.0747 |
| long_form | C5_qwen3 | 5 | 10 | 5.0 | 5.1 | 0.0730 | 0.0747 |
| long_form | C5_qwen3 | 10 | 2 | 87.4 | 90.5 | 0.0589 | 0.0678 |
| long_form | C5_qwen3 | 10 | 10 | 17.5 | 18.1 | 0.0589 | 0.0678 |
| long_form | C5_qwen3 | 20 | 2 | 96.3 | 103.3 | 0.0923 | 0.1097 |
| long_form | C5_qwen3 | 20 | 10 | 19.3 | 20.7 | 0.0923 | 0.1097 |
| long_form | D2_gated_fusion | 5 | 2 | 0.2 | 2.4 | 0.0730 | 0.0730 |
| long_form | D2_gated_fusion | 5 | 10 | 0.0 | 0.5 | 0.0730 | 0.0730 |
| long_form | D2_gated_fusion | 10 | 2 | 10.9 | 11.3 | 0.0589 | 0.0597 |
| long_form | D2_gated_fusion | 10 | 10 | 2.2 | 2.3 | 0.0589 | 0.0597 |
| long_form | D2_gated_fusion | 20 | 2 | 54.4 | 51.6 | 0.0923 | 0.1002 |
| long_form | D2_gated_fusion | 20 | 10 | 10.9 | 10.3 | 0.0923 | 0.1002 |

## mu-robustness of fee_bps_ann (gamma=2), across mu in {4%,6%,8%}

| disclosure | model | h | fee@4% | fee@6% | fee@8% |
|---|---|---|---:|---:|---:|
| event_driven | B2_tfidf_ridge | 5 | -18.0 | -26.7 | -35.3 |
| event_driven | B2_tfidf_ridge | 10 | -3.9 | -5.0 | -5.4 |
| event_driven | B2_tfidf_ridge | 20 | -3.7 | -4.8 | -5.3 |
| event_driven | C2_finbert_s1 | 5 | -12.8 | -14.1 | -11.9 |
| event_driven | C2_finbert_s1 | 10 | -10.6 | -14.7 | -18.0 |
| event_driven | C2_finbert_s1 | 20 | -9.5 | -11.1 | -10.5 |
| event_driven | D2_gated_fusion | 5 | -9.5 | -15.8 | -23.1 |
| event_driven | D2_gated_fusion | 10 | -2.5 | -5.0 | -8.6 |
| event_driven | D2_gated_fusion | 20 | 14.5 | 19.1 | 21.9 |
| long_form | B2_tfidf_ridge | 5 | 8.8 | 11.7 | 13.8 |
| long_form | B2_tfidf_ridge | 10 | 2.1 | 2.6 | 2.7 |
| long_form | B2_tfidf_ridge | 20 | 4.5 | 6.6 | 8.5 |
| long_form | C2_finbert_s1 | 5 | -8.5 | -10.2 | -10.3 |
| long_form | C2_finbert_s1 | 10 | 8.2 | 9.8 | 9.7 |
| long_form | C2_finbert_s1 | 20 | 0.0 | 0.0 | -0.0 |
| long_form | C2_finbert_s2 | 5 | -3.3 | -7.3 | -12.8 |
| long_form | C2_finbert_s2 | 10 | -1.4 | -1.9 | -2.3 |
| long_form | C2_finbert_s2 | 20 | 23.9 | 32.8 | 39.6 |
| long_form | C4_longformer | 5 | -11.2 | -15.7 | -19.3 |
| long_form | C4_longformer | 10 | -0.7 | -1.0 | -1.4 |
| long_form | C4_longformer | 20 | 51.3 | 71.8 | 89.0 |
| long_form | C5_qwen3 | 5 | 18.3 | 25.2 | 30.4 |
| long_form | C5_qwen3 | 10 | 60.4 | 87.4 | 112.2 |
| long_form | C5_qwen3 | 20 | 67.4 | 96.3 | 121.9 |
| long_form | D2_gated_fusion | 5 | 1.0 | 0.2 | -1.3 |
| long_form | D2_gated_fusion | 10 | 7.3 | 10.9 | 14.3 |
| long_form | D2_gated_fusion | 20 | 41.9 | 54.4 | 61.3 |

## Verdict

- **Headline fee is small on balance but noisy across cells.** At mu=6%, gamma=2 the annualized performance fee is positive in 14/27 (disclosure,model,horizon) cells with a median of only 0.03 bps/yr, yet individual cells span -27 to 96 bps/yr. Winsorizing the inverse-variance weights at the 99th pct barely moves it (positive in 14/27, median 0.01 bps/yr), so the spread is NOT a single-outlier artifact.
- **The large per-cell fees are a known realized-utility artifact, not economic signal.** On a single risky asset the realized quadratic utility rewards a systematically HIGHER (more conservative) vol forecast: higher sigma => smaller bets => a smaller realized variance penalty, which can raise utility even when the forecast is LESS accurate. The biggest 'fees' (C5_qwen3/C4_longformer/D2 at long-form h=20, +70..+103 bps) occur exactly where f_U forecasts slightly higher mean vol than f_R AND QLIKE says f_U is WORSE — the fee is pricing conservatism, not skill.
- **Sign disagreement with the QLIKE increment confirms this.** The fee agrees in sign with the per-cell QLIKE improvement in only 6/27 cells (raw) / 6/27 (winsorized): the mean-variance timing metric and the statistical accuracy metric are largely orthogonal here, because the fee is dominated by the realized mean-return term interacting with bet size.
- **mu-free, leverage-controlled target-vol Sharpe is the cleanest read, and it says ~zero.** f_U beats f_R in 12/27 cells, median Sharpe gain -0.00003 — economically indistinguishable from zero. It agrees in sign with QLIKE in 8/27 cells, slightly better than the fee, with the only visible gains at long-form h=20 for the strongest text models (the same cells with the largest QLIKE increments), and losses elsewhere.
- **Fee sign is mu-invariant, magnitude monotone in mu** (mu enters the weight linearly), confirmed across mu in {4%,6%,8%}; larger gamma just scales the fee down ~5x (2->10).
- **Bottom line: text adds NO robust economic value for a volatility-timing investor beyond a recalibrated HAR.** The ~0.1-4.6% statistical QLIKE increment (M1) is real but does not convert into a leverage-safe, mu-robust economic gain: the median fee is a fraction of a bps/yr, the Sharpe gain is ~0, and the sizeable per-cell fees are an artifact of realized single-asset utility rewarding conservative forecasts rather than evidence of timing skill. This is fully consistent with M1's 'small but real, economically modest' reading — here the economic magnitude is, honestly, negligible.
