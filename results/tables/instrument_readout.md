# T6-15 — What the surviving instrument emits (C6, event_driven, h=5, 25,109 test rows)

DIAGNOSTIC OF FROZEN FORECASTS. No clustering, no Holm, no placebo; this enters no pre-declared family and is not a survivor claim.

- distinct emitted values: **10**
- two modal values carry **92.6%** of forecasts (0.22 at 49.3%, 0.18 at 43.2%)
- the 0.22/0.18 flag separates mean realised volatility in **10 of 10** deciles of the HAR-RV forecast
- gap widens from +0.0071 in the lowest decile to +0.0744 in the highest; strictly increasing across deciles: **False**

| HAR-RV decile | n(high) | n(low) | mean RV, high | mean RV, low | difference |
|---|---|---|---|---|---|
| 1 | 1,067 | 1,324 | 0.1919 | 0.1849 | +0.0071 |
| 2 | 1,144 | 1,259 | 0.2092 | 0.1986 | +0.0106 |
| 3 | 1,188 | 1,197 | 0.2240 | 0.2077 | +0.0163 |
| 4 | 1,173 | 1,214 | 0.2416 | 0.2256 | +0.0160 |
| 5 | 1,225 | 1,157 | 0.2634 | 0.2437 | +0.0197 |
| 6 | 1,219 | 1,139 | 0.2680 | 0.2471 | +0.0209 |
| 7 | 1,337 | 991 | 0.2959 | 0.2713 | +0.0246 |
| 8 | 1,333 | 976 | 0.3205 | 0.2915 | +0.0289 |
| 9 | 1,293 | 953 | 0.3694 | 0.3220 | +0.0474 |
| 10 | 1,410 | 644 | 0.4534 | 0.3790 | +0.0744 |
