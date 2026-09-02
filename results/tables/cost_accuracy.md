# Cost vs Accuracy — Efficiency Frontier (Task A4)

Cost = total GPU-hours summed across all runs of that model (C/D: 3 seeds x 3 disclosures = 9 runs; per-run = mean). Accuracy = best test QLIKE on **long_form** (min over horizons 5/10/20; C/D QLIKE averaged across the 3 seeds per horizon first). A/B are CPU / seed-invariant baselines reported at ~0 GPU-hours to anchor the cheap end (their CPU wall-clock seconds noted separately).

**Total GPU-hours across all C runs = 566.7; all D runs = 23.6; C+D = 590.4.**

| Rank | Model | Block | GPU-h total | GPU-h/run | QLIKE h5 | QLIKE h10 | QLIKE h20 | Best QLIKE | Pareto |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A3_garch | A | 0.00 | 0.000 | 0.4829 | 0.3750 | 0.2686 | 0.2686 | **YES** |
| 2 | A5_arima | A | 0.00 | 0.000 | 0.5317 | 0.4067 | 0.2732 | 0.2732 | no |
| 3 | A2_har_rv | A | 0.00 | 0.000 | 0.5564 | 0.4618 | 0.2970 | 0.2970 | no |
| 4 | A4_egarch | A | 0.00 | 0.000 | 0.4631 | 0.3541 | 0.3180 | 0.3180 | no |
| 5 | A1_hv | A | 0.00 | 0.000 | 1.1458 | 0.5600 | 0.3219 | 0.3219 | no |
| 6 | D3_gteqwen2 | D | 0.20 | 0.022 | 0.6698 | 0.5321 | 0.3270 | 0.3270 | no |
| 7 | D3_e5mistral | D | 0.21 | 0.023 | 0.7127 | 0.5688 | 0.3400 | 0.3400 | no |
| 8 | D3_qwen3 | D | 0.20 | 0.023 | 0.7486 | 0.6232 | 0.3594 | 0.3594 | no |
| 9 | C5_gteqwen2 | C | 5.72 | 0.635 | 0.6538 | 0.5714 | 0.3916 | 0.3916 | no |
| 10 | C5_qwen3 | C | 4.44 | 0.494 | 0.6866 | 0.6045 | 0.3940 | 0.3940 | no |
| 11 | D1_concat_mlp | D | 11.43 | 1.270 | 0.6408 | 0.6445 | 0.4374 | 0.4374 | no |
| 12 | C5_e5mistral | C | 8.94 | 0.993 | 0.8467 | 0.6480 | 0.4375 | 0.4375 | no |
| 13 | D2_gated_fusion | D | 11.60 | 1.288 | 0.7964 | 0.6022 | 0.4410 | 0.4410 | no |
| 14 | C2_finbert_s2 | C | 71.27 | 7.919 | 1.4870 | 1.1952 | 0.5720 | 0.5720 | no |
| 15 | C2_finbert_s4 | C | 62.43 | 6.936 | 1.4320 | 1.0943 | 0.6032 | 0.6032 | no |
| 16 | C4_longformer | C | 254.72 | 28.302 | 1.2351 | 1.0637 | 0.6051 | 0.6051 | no |
| 17 | C2_finbert_s3 | C | 67.80 | 7.533 | 1.2430 | 1.2546 | 0.6731 | 0.6731 | no |
| 18 | C1_bert_s2 | C | 51.73 | 5.748 | 1.4557 | 1.0372 | 0.6905 | 0.6905 | no |
| 19 | B3_lm_linear | B | 0.00 | 0.000 | 1.2937 | 1.0649 | 0.6918 | 0.6918 | no |
| 20 | C1_bert_s1 | C | 13.71 | 1.523 | 1.6444 | 1.1968 | 0.6995 | 0.6995 | no |
| 21 | B2_tfidf_ridge | B | 0.00 | 0.000 | 1.2371 | 1.0522 | 0.7083 | 0.7083 | no |
| 22 | B4_lm_features | B | 0.00 | 0.000 | 1.4166 | 1.1223 | 0.7231 | 0.7231 | no |
| 23 | C2_finbert_s1 | C | 12.65 | 1.406 | 1.3428 | 1.2440 | 0.7262 | 0.7262 | no |
| 24 | C3_roberta_s1 | C | 13.31 | 1.479 | 1.6303 | 1.0863 | 0.7518 | 0.7518 | no |
| 25 | B1_bow_ridge | B | 0.00 | 0.000 | 2.2844 | 1.5916 | 1.0668 | 1.0668 | no |

### CPU baselines — recorded wall-clock seconds (reported at ~0 GPU-h)
| Model | Block | CPU seconds (sum over 3 disclosures) |
|---|---|---|
| A3_garch | A | 212.1 |
| A5_arima | A | 2862.6 |
| A2_har_rv | A | 3.9 |
| A4_egarch | A | 203.9 |
| A1_hv | A | 3.2 |
| B3_lm_linear | B | 799.1 |
| B2_tfidf_ridge | B | 4474.1 |
| B4_lm_features | B | 1565.8 |
| B1_bow_ridge | B | 2310.2 |
