# Stratified DM vs A2_har_rv (squared-error, test). Positive=worse than HAR-RV.


## long_form
| model | period | h5 | h10 | h20 |
|---|---|---|---|---|
| C2_finbert_s1 | 2022 | +13.03* | +15.70* | +16.98* |
| C2_finbert_s1 | 2023 | +4.04* | +4.90* | +7.77* |
| C2_finbert_s1 | 24-25 | +3.54* | +6.61* | +8.48* |
| C4_longformer | 2022 | +14.93* | +14.70* | +14.56* |
| C4_longformer | 2023 | +4.14* | +4.40* | +6.58* |
| C4_longformer | 24-25 | +5.23* | +7.96* | +6.55* |
| D2_gated_fusion | 2022 | +13.76* | +14.54* | +10.14* |
| D2_gated_fusion | 2023 | +1.74 ns | +2.59* | +4.79* |
| D2_gated_fusion | 24-25 | +4.10* | +4.18* | +2.52* |

## event_driven
| model | period | h5 | h10 | h20 |
|---|---|---|---|---|
| C2_finbert_s1 | 2022 | +19.39* | +24.39* | +23.45* |
| C2_finbert_s1 | 2023 | +7.82* | +13.72* | +13.80* |
| C2_finbert_s1 | 24-25 | +9.49* | +13.00* | +12.53* |
| D2_gated_fusion | 2022 | +18.31* | +22.32* | +13.97* |
| D2_gated_fusion | 2023 | +8.12* | +12.06* | +6.03* |
| D2_gated_fusion | 24-25 | +7.08* | +11.23* | +5.19* |