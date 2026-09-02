# TASK A3 — Long-doc strategy sweep (H3)

Does sophisticated long-document handling beat S1 truncation? Test QLIKE (mean±std over 3 seeds) and R² per model×horizon. Within each family, DM-test each strategy vs S1 truncation on seed-ensembled per-obs loss. **QLIKE-DM is the PRIMARY test** (matches the paper's primary loss); SE-DM is a secondary robustness.

Strategies: S1=truncation, S2=chunk-mean, S3=chunk-attention, S4=hierarchical, S5=long-context (Longformer).
DM sign: **positive stat = strategy WORSE than S1** (higher loss); negative = better. `*` = p<0.05.

## long_form

### FinBERT

| strategy | h | QLIKE (mean±std) | R² (mean±std) | QLIKE-DM vs S1 (primary) | SE-DM vs S1 |
|---|--:|--:|--:|:--|:--|
| S1_truncation | 5 | 1.3428±0.3318 | -0.1997±0.1185 | — (baseline) | — (baseline) |
| S1_truncation | 10 | 1.2440±0.1776 | -0.1925±0.0609 | — (baseline) | — (baseline) |
| S1_truncation | 20 | 0.7262±0.2018 | -0.1951±0.1473 | — (baseline) | — (baseline) |
| S2_chunk_mean | 5 | 1.4870±0.0562 | -0.2526±0.0228 | +16.649, p=0.000* | +14.875, p=0.000* |
| S2_chunk_mean | 10 | 1.1952±0.2612 | -0.1874±0.0933 | -6.067, p=0.000* | -1.508, p=0.132 |
| S2_chunk_mean | 20 | 0.5720±0.1283 | -0.0630±0.1286 | -4.107, p=0.000* | -1.609, p=0.108 |
| S3_chunk_attn | 5 | 1.2430±0.0504 | -0.1524±0.0105 | -2.038, p=0.042* | -5.288, p=0.000* |
| S3_chunk_attn | 10 | 1.2546±0.1243 | -0.1888±0.0222 | -4.033, p=0.000* | -3.611, p=0.000* |
| S3_chunk_attn | 20 | 0.6731±0.1599 | -0.1100±0.1011 | +4.254, p=0.000* | +2.662, p=0.008* |
| S4_hierarchical | 5 | 1.4320±0.3491 | -0.2217±0.0914 | +4.242, p=0.000* | +3.718, p=0.000* |
| S4_hierarchical | 10 | 1.0943±0.2628 | -0.1452±0.0577 | -12.593, p=0.000* | -7.377, p=0.000* |
| S4_hierarchical | 20 | 0.6032±0.2053 | -0.0538±0.1283 | -4.980, p=0.000* | -5.355, p=0.000* |
| S5_long_context | 5 | 1.2351±0.2865 | -0.1467±0.0940 | -4.103, p=0.000* | -6.165, p=0.000* |
| S5_long_context | 10 | 1.0637±0.0254 | -0.1129±0.0144 | -9.693, p=0.000* | -8.986, p=0.000* |
| S5_long_context | 20 | 0.6051±0.1026 | -0.0814±0.1041 | -3.020, p=0.003* | -0.707, p=0.480 |

### BERT-base

| strategy | h | QLIKE (mean±std) | R² (mean±std) | QLIKE-DM vs S1 (primary) | SE-DM vs S1 |
|---|--:|--:|--:|:--|:--|
| S1_truncation | 5 | 1.6444±0.2140 | -0.2725±0.0951 | — (baseline) | — (baseline) |
| S1_truncation | 10 | 1.1968±0.3635 | -0.2112±0.0925 | — (baseline) | — (baseline) |
| S1_truncation | 20 | 0.6995±0.1162 | -0.1454±0.0622 | — (baseline) | — (baseline) |
| S2_chunk_mean | 5 | 1.4557±0.1759 | -0.2424±0.0770 | -8.765, p=0.000* | -4.390, p=0.000* |
| S2_chunk_mean | 10 | 1.0372±0.2343 | -0.1397±0.0923 | -4.676, p=0.000* | -4.309, p=0.000* |
| S2_chunk_mean | 20 | 0.6905±0.0481 | -0.1156±0.0503 | +0.940, p=0.347 | -2.021, p=0.043* |

## event_driven

_Short-doc subset: truncation loses little information, so long-doc strategy is expected to matter less here._

### FinBERT

| strategy | h | QLIKE (mean±std) | R² (mean±std) | QLIKE-DM vs S1 (primary) | SE-DM vs S1 |
|---|--:|--:|--:|:--|:--|
| S1_truncation | 5 | 0.9763±0.1443 | -0.0318±0.0463 | — (baseline) | — (baseline) |
| S1_truncation | 10 | 1.1358±0.0765 | -0.1559±0.0357 | — (baseline) | — (baseline) |
| S1_truncation | 20 | 0.7024±0.1284 | -0.0836±0.0736 | — (baseline) | — (baseline) |
| S2_chunk_mean | 5 | 1.4556±0.2166 | -0.1706±0.0514 | +38.228, p=0.000* | +31.650, p=0.000* |
| S2_chunk_mean | 10 | 1.1665±0.1811 | -0.1606±0.0534 | -0.417, p=0.677 | +1.151, p=0.250 |
| S2_chunk_mean | 20 | 0.7287±0.1844 | -0.1153±0.0843 | +0.608, p=0.543 | +0.595, p=0.552 |
| S3_chunk_attn | 5 | 1.1119±0.0551 | -0.0587±0.0133 | +17.281, p=0.000* | +12.556, p=0.000* |
| S3_chunk_attn | 10 | 0.8271±0.1709 | -0.0326±0.0442 | -30.482, p=0.000* | -23.094, p=0.000* |
| S3_chunk_attn | 20 | 0.9199±0.1839 | -0.2274±0.0387 | +14.837, p=0.000* | +7.895, p=0.000* |
| S4_hierarchical | 5 | 1.3306±0.3283 | -0.1207±0.0895 | +29.965, p=0.000* | +23.985, p=0.000* |
| S4_hierarchical | 10 | 0.9871±0.0544 | -0.0943±0.0264 | -20.476, p=0.000* | -16.702, p=0.000* |
| S4_hierarchical | 20 | 0.8596±0.1968 | -0.1760±0.1118 | +17.792, p=0.000* | +16.332, p=0.000* |
| S5_long_context | 5 | 1.2275±0.1727 | -0.0879±0.0304 | +23.630, p=0.000* | +18.385, p=0.000* |
| S5_long_context | 10 | 0.9733±0.1339 | -0.0975±0.0575 | -19.068, p=0.000* | -14.609, p=0.000* |
| S5_long_context | 20 | 0.8356±0.0596 | -0.1530±0.0492 | +17.268, p=0.000* | +14.624, p=0.000* |

### BERT-base

| strategy | h | QLIKE (mean±std) | R² (mean±std) | QLIKE-DM vs S1 (primary) | SE-DM vs S1 |
|---|--:|--:|--:|:--|:--|
| S1_truncation | 5 | 1.2640±0.0393 | -0.1227±0.0118 | — (baseline) | — (baseline) |
| S1_truncation | 10 | 1.0992±0.2230 | -0.1346±0.0762 | — (baseline) | — (baseline) |
| S1_truncation | 20 | 0.7859±0.1641 | -0.1270±0.0771 | — (baseline) | — (baseline) |
| S2_chunk_mean | 5 | 1.5037±0.2390 | -0.1582±0.0519 | +18.015, p=0.000* | +7.595, p=0.000* |
| S2_chunk_mean | 10 | 1.2292±0.2198 | -0.1773±0.0706 | +15.394, p=0.000* | +10.862, p=0.000* |
| S2_chunk_mean | 20 | 0.9014±0.1322 | -0.1765±0.0614 | +13.901, p=0.000* | +12.621, p=0.000* |

## combined

_Short-doc subset: truncation loses little information, so long-doc strategy is expected to matter less here._

### FinBERT

| strategy | h | QLIKE (mean±std) | R² (mean±std) | QLIKE-DM vs S1 (primary) | SE-DM vs S1 |
|---|--:|--:|--:|:--|:--|
| S1_truncation | 5 | 1.3786±0.1740 | -0.1341±0.0327 | — (baseline) | — (baseline) |
| S1_truncation | 10 | 0.8203±0.1365 | -0.0434±0.0557 | — (baseline) | — (baseline) |
| S1_truncation | 20 | 0.8455±0.2226 | -0.1506±0.0978 | — (baseline) | — (baseline) |
| S2_chunk_mean | 5 | 1.5138±0.1573 | -0.1568±0.0313 | +8.214, p=0.000* | +5.258, p=0.000* |
| S2_chunk_mean | 10 | 1.1488±0.2860 | -0.1555±0.0973 | +31.544, p=0.000* | +24.394, p=0.000* |
| S2_chunk_mean | 20 | 0.8834±0.1719 | -0.1871±0.0821 | +7.045, p=0.000* | +6.918, p=0.000* |
| S3_chunk_attn | 5 | 1.4810±0.2622 | -0.1612±0.0689 | +12.267, p=0.000* | +12.023, p=0.000* |
| S3_chunk_attn | 10 | 1.2171±0.1166 | -0.1582±0.0615 | +26.940, p=0.000* | +20.474, p=0.000* |
| S3_chunk_attn | 20 | 0.9043±0.1450 | -0.1747±0.0619 | +5.102, p=0.000* | +1.918, p=0.055 |
| S4_hierarchical | 5 | 1.6165±0.2421 | -0.2220±0.0564 | +25.472, p=0.000* | +25.070, p=0.000* |
| S4_hierarchical | 10 | 1.0603±0.1078 | -0.1231±0.0341 | +23.594, p=0.000* | +18.675, p=0.000* |
| S4_hierarchical | 20 | 0.7423±0.1217 | -0.1109±0.0810 | -10.197, p=0.000* | -6.514, p=0.000* |
| S5_long_context | 5 | 1.5939±0.1596 | -0.2092±0.0281 | +9.190, p=0.000* | +15.917, p=0.000* |
| S5_long_context | 10 | 1.0328±0.1162 | -0.1157±0.0407 | +24.775, p=0.000* | +20.119, p=0.000* |
| S5_long_context | 20 | 0.7616±0.1444 | -0.1104±0.0826 | -7.543, p=0.000* | -5.415, p=0.000* |

### BERT-base

| strategy | h | QLIKE (mean±std) | R² (mean±std) | QLIKE-DM vs S1 (primary) | SE-DM vs S1 |
|---|--:|--:|--:|:--|:--|
| S1_truncation | 5 | 1.5902±0.3126 | -0.1887±0.0833 | — (baseline) | — (baseline) |
| S1_truncation | 10 | 0.7889±0.1663 | -0.0372±0.0557 | — (baseline) | — (baseline) |
| S1_truncation | 20 | 0.9343±0.0833 | -0.2130±0.0134 | — (baseline) | — (baseline) |
| S2_chunk_mean | 5 | 1.6921±0.0498 | -0.2171±0.0197 | +16.716, p=0.000* | +15.152, p=0.000* |
| S2_chunk_mean | 10 | 1.1525±0.1777 | -0.1584±0.0608 | +29.592, p=0.000* | +23.277, p=0.000* |
| S2_chunk_mean | 20 | 0.8803±0.0187 | -0.1663±0.0111 | -6.019, p=0.000* | -7.673, p=0.000* |

## VERDICT

- **long_form**: significant improvement over S1 at: S2_chunk_mean@h10 (QLIKE-DM -6.07 p=0.000); S2_chunk_mean@h20 (QLIKE-DM -4.11 p=0.000); S3_chunk_attn@h5 (QLIKE-DM -2.04 p=0.042); S3_chunk_attn@h10 (QLIKE-DM -4.03 p=0.000); S4_hierarchical@h10 (QLIKE-DM -12.59 p=0.000); S4_hierarchical@h20 (QLIKE-DM -4.98 p=0.000); S5_long_context@h5 (QLIKE-DM -4.10 p=0.000); S5_long_context@h10 (QLIKE-DM -9.69 p=0.000); S5_long_context@h20 (QLIKE-DM -3.02 p=0.003)
- **event_driven**: significant improvement over S1 at: S3_chunk_attn@h10 (QLIKE-DM -30.48 p=0.000); S4_hierarchical@h10 (QLIKE-DM -20.48 p=0.000); S5_long_context@h10 (QLIKE-DM -19.07 p=0.000)
- **combined**: significant improvement over S1 at: S4_hierarchical@h20 (QLIKE-DM -10.20 p=0.000); S5_long_context@h20 (QLIKE-DM -7.54 p=0.000)

**Overall (H3):** Mixed, with **one consistent winner on the PRIMARY QLIKE loss**. On long_form, **S5 long-context (Longformer) significantly beats S1 truncation at all three horizons** (QLIKE-DM −4.10/−9.69/−3.02, all p<0.05); the chunk strategies are horizon-inconsistent — each of S2/S3/S4 beats S1 at two horizons but is significantly WORSE at the third (S2 at h5, S3 at h20, S4 at h5). The strategy ranking is loss-metric- dependent: the SE-DM column differs (e.g. S5 only ties on SE at h20, and BERT-base S2's SE all-3 win drops to a 2/3 tie on QLIKE), so both columns are reported. Caveats that blunt the S5 win: QLIKE cross-seed std is wide (often ≥0.2) overlapping S1, and Longformer costs ~20× S1's compute (254.7 vs 12.7 GPU-h) for a within-text gain that still loses to HAR outright. So long-context genuinely extracts more than truncation on the primary loss, but the improvement is modest, expensive, and does not lift text above the price baseline.
