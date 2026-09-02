# Recalibration slope of Eq. (1), log space vs level space

Primary arm C2_finbert_s1, validation split, the rows FAMILY 3 of forecast_combination.md is computed on.

| disclosure | h | n val | b (log space, Eq. 1) | b (level space, FAMILY 3) |
|---|---|---|---|---|
| long_form | 5 | 3956 | 1.111 | 1.495 |
| long_form | 10 | 3950 | 1.132 | 1.562 |
| long_form | 20 | 3943 | 1.017 | 1.334 |
| event_driven | 5 | 14213 | 1.167 | 1.508 |
| event_driven | 10 | 14196 | 1.142 | 1.447 |
| event_driven | 20 | 14156 | 1.042 | 1.269 |

mean_b_log= 1.1019
range_b_log= 1.0172 1.1671
mean_b_level= 1.4357
range_b_level= 1.2691 1.5615
