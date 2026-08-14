# E3 Stratified DM vs A2_har_rv: exiting (delisted/acquired) vs surviving firms (long-form test).
Tests whether disclosure text beats HAR-RV on the distressed/exiting names where text should help most.

| stratum | model | n_filings | h5 | h10 | h20 |
|---|---|---|---|---|---|
| exiting | C2_finbert_s1 | 822 | +7.80* | +6.62* | +8.94* |
| survivor | C2_finbert_s1 | 7106 | +15.66* | +13.15* | +16.97* |
| exiting | C4_longformer | 822 | +5.07* | +5.84* | +7.10* |
| survivor | C4_longformer | 7106 | +11.32* | +11.75* | +12.81* |
| exiting | D2_gated_fusion | 822 | -2.36* | +6.22* | +6.71* |
| survivor | D2_gated_fusion | 7106 | +0.37 ns | +14.37* | +12.87* |