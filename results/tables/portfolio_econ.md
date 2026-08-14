# P2 — Portfolio-level economic test of the incremental text signal

## RESTATED-vs-ORIGINAL

- **ORIGINAL (single-asset FKO fee):** the per-filing FKO performance fee was diagnosed *artifact-ridden* — it mechanically rewards conservative (larger) vol forecasts because a smaller single-name position from a higher sigma shrinks realized return variance regardless of forecast accuracy, so any systematic scale gap between f_U and f_R surfaces as a 'fee' with no genuine information content.
- **RESTATED (this table, portfolio-level):** the increment's real economic claim is *cross-sectional* — each day it should re-rank firms by risk. We build a long-only inverse-variance portfolio over ALL live filing signals and ask whether the text-augmented sigma (f_U) instead of the recalibrated-HAR sigma (f_R) raises the portfolio Sharpe. Weights fit on VALIDATION, frozen to TEST; non-overlapping h-day holding; day-block bootstrap CI on the Sharpe difference.


## Verdict
**The cross-sectional increment translates into at most a MARGINAL, NON-ROBUST portfolio-level improvement — Sharpe rises in a majority of cells but only 1/18 clear a day-block bootstrap CI (about the multiplicity false-positive floor), and the median gain is economically negligible (ΔSharpe=+0.003).** Across 18 disc×model×h cells: f_U Sharpe > f_R Sharpe with a bootstrap CI excluding zero in **1** cells (any-sign significant: 1); median ΔSharpe = +0.003. Portfolio FKO fee positive in 10/18 cells (γ=2), 11/18 (γ=10).


**SANITY:** f_R-portfolio annualized Sharpe range = [-0.104, +0.945] (target plausible 0–2 band). Returns matched on ret_match_ok only; 37400 of 431245 return rows dropped for match failure.


## Portfolio economics (per disc × model × horizon)

| disc | model | h | n_periods | avg_names | Sharpe f_R | Sharpe f_U | ΔSharpe | ΔSh CI lo | ΔSh CI hi | sig | rvol f_R | rvol f_U | tgt vol f_R | FKO γ2 (ann) | FKO γ10 (ann) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | B2_tfidf_ridge | 5 | 200 | 101.8 | +0.329 | +0.336 | +0.006 | -0.022 | +0.038 | no | 0.125 | 0.124 | 0.184 | +0.0005 | +0.0006 |
| event_driven | B2_tfidf_ridge | 10 | 100 | 174.6 | +0.125 | +0.151 | +0.026 | +0.004 | +0.047 | YES | 0.120 | 0.119 | 0.109 | +0.0030 | +0.0030 |
| event_driven | B2_tfidf_ridge | 20 | 50 | 282.8 | +0.340 | +0.352 | +0.012 | -0.012 | +0.045 | no | 0.113 | 0.112 | 0.076 | +0.0011 | +0.0011 |
| event_driven | C2_finbert_s1 | 5 | 200 | 101.8 | +0.329 | +0.331 | +0.002 | -0.028 | +0.031 | no | 0.125 | 0.124 | 0.184 | -0.0000 | +0.0000 |
| event_driven | C2_finbert_s1 | 10 | 100 | 174.6 | +0.125 | +0.121 | -0.004 | -0.028 | +0.021 | no | 0.120 | 0.120 | 0.109 | -0.0005 | -0.0005 |
| event_driven | C2_finbert_s1 | 20 | 50 | 282.8 | +0.340 | +0.335 | -0.005 | -0.043 | +0.060 | no | 0.113 | 0.111 | 0.076 | -0.0011 | -0.0010 |
| event_driven | C6_llmtext | 5 | 200 | 101.8 | +0.329 | +0.324 | -0.005 | -0.024 | +0.012 | no | 0.125 | 0.125 | 0.184 | -0.0006 | -0.0006 |
| event_driven | C6_llmtext | 10 | 100 | 174.6 | +0.125 | +0.118 | -0.007 | -0.020 | +0.006 | no | 0.120 | 0.121 | 0.109 | -0.0009 | -0.0009 |
| event_driven | C6_llmtext | 20 | 50 | 282.8 | +0.340 | +0.344 | +0.004 | -0.003 | +0.014 | no | 0.113 | 0.113 | 0.076 | +0.0005 | +0.0005 |
| long_form | B2_tfidf_ridge | 5 | 159 | 47.4 | +0.945 | +0.966 | +0.021 | -0.036 | +0.080 | no | 0.135 | 0.133 | 0.494 | +0.0011 | +0.0012 |
| long_form | B2_tfidf_ridge | 10 | 77 | 92.9 | +0.207 | +0.223 | +0.016 | -0.011 | +0.048 | no | 0.127 | 0.126 | 0.244 | +0.0018 | +0.0018 |
| long_form | B2_tfidf_ridge | 20 | 39 | 181.4 | -0.104 | -0.089 | +0.015 | +nan | +nan | no | 0.132 | 0.131 | 0.201 | +0.0021 | +0.0021 |
| long_form | C2_finbert_s1 | 5 | 159 | 47.4 | +0.945 | +0.941 | -0.004 | -0.020 | +0.012 | no | 0.135 | 0.135 | 0.494 | -0.0008 | -0.0008 |
| long_form | C2_finbert_s1 | 10 | 77 | 92.9 | +0.207 | +0.235 | +0.028 | -0.020 | +0.077 | no | 0.127 | 0.124 | 0.244 | +0.0032 | +0.0032 |
| long_form | C2_finbert_s1 | 20 | 39 | 181.4 | -0.104 | -0.104 | +0.001 | +nan | +nan | no | 0.132 | 0.132 | 0.201 | +0.0001 | +0.0001 |
| long_form | C6_llmtext | 5 | 159 | 47.4 | +0.945 | +0.962 | +0.017 | -0.018 | +0.053 | no | 0.135 | 0.134 | 0.494 | +0.0017 | +0.0018 |
| long_form | C6_llmtext | 10 | 77 | 92.9 | +0.207 | +0.207 | -0.001 | -0.011 | +0.016 | no | 0.127 | 0.127 | 0.244 | -0.0001 | -0.0001 |
| long_form | C6_llmtext | 20 | 39 | 181.4 | -0.104 | -0.104 | -0.000 | +nan | +nan | no | 0.132 | 0.132 | 0.201 | -0.0000 | -0.0000 |

## Notes
- **Portfolio:** long-only, w_i ∝ 1/sigma_i^2 normalized across live names each rebalance day. A firm is *live* if its most-recent filing's signal falls in the trailing h trading days. Realized payoff = that filing's h-day forward log return (fwd_logret, ret_match_ok only).
- **Non-overlapping** h-day holding periods ⇒ independent portfolio returns; the day-block bootstrap uses blocks of h periods.
- **Sharpe** annualized by √(252/h) on non-overlapping h-day log returns. **Target vol** = inverse-variance-implied √(Σ w_i² σ_i²), annualized. It is a single-name-scaled idealization; realized portfolio vol is typically LOWER than this target when many names are held (diversification across 47–280 firms dominates residual cross-firm correlation), and approaches/exceeds it only for the most concentrated, short-horizon books.
- **FKO fee** solved at the portfolio level (mean quadratic utility), annualized; positive ⇒ the f_U portfolio is preferred at that risk aversion.
