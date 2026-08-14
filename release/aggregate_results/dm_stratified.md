# Stratified DM vs A2_har_rv (squared-error, test). Positive=worse than HAR-RV.


## long_form
| model | period | h5 | h10 | h20 |
|---|---|---|---|---|
| C2_finbert_s1 | 2022 | +16.76* | +14.55* | +20.65* |
| C2_finbert_s1 | 2023 | +7.30* | +5.11* | +9.43* |
| C2_finbert_s1 | 24-25 | +8.27* | +6.76* | +10.31* |
| C4_longformer | 2022 | +11.60* | +12.05* | +13.86* |
| C4_longformer | 2023 | +3.36* | +4.12* | +5.54* |
| C4_longformer | 24-25 | +6.11* | +6.62* | +7.84* |
| D2_gated_fusion | 2022 | +0.06 ns | +14.83* | +14.93* |
| D2_gated_fusion | 2023 | +0.89 ns | +5.69* | +6.43* |
| D2_gated_fusion | 24-25 | -1.71 ns | +7.63* | +7.31* |

## event_driven
| model | period | h5 | h10 | h20 |
|---|---|---|---|---|
| C2_finbert_s1 | 2022 | +17.32* | +20.09* | +20.27* |
| C2_finbert_s1 | 2023 | +10.20* | +12.27* | +14.53* |
| C2_finbert_s1 | 24-25 | +11.47* | +11.24* | +11.62* |
| D2_gated_fusion | 2022 | +13.53* | +15.71* | +24.53* |
| D2_gated_fusion | 2023 | +6.75* | +9.58* | +16.53* |
| D2_gated_fusion | 24-25 | +7.87* | +6.09* | +14.52* |