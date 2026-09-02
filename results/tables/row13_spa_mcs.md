# Row 13 — Hansen SPA + Model Confidence Set over the standalone leaderboard

## RESTATED vs BEFORE

| quantity | BEFORE (pairwise clustered DM, dm_pairwise_clustered.md) | RESTATED (this table: joint SPA/MCS) |
|---|---|---|
| multiplicity scope | family-wise Holm across the pairs actually run | joint over the ENTIRE alternative set (White/Hansen data-snooping) |
| headline statistic | # challengers with clustered DM<0, Holm<.05 vs A2 = **0/180** | SPA consistent-p that HAR is not inferior + the 90% MCS membership |
| pure-text models in the top tier | 0/180 pairwise-beat | **0 entries in any 90% MCS across 18 loss-panels** |

## What this adds

Pairwise DM (even day-clustered + Holm) only controls error across the pairs you chose to test. A referee's standing objection is *aggregate* data-snooping: with a whole leaderboard of alternatives, the best-looking challenger is selected post hoc. The Hansen (2005) SPA and the Hansen-Lunde-Nason (2011) MCS answer exactly that — SPA gives one p-value for “no alternative beats the HAR benchmark” after accounting for the full set; the MCS returns the set of models that cannot be statistically separated, so we can read off whether ANY text/fusion model belongs in the top tier.

**Spec.** Common inner-joined sample; multi-seed neural models seed-ensembled; losses block-aggregated to one value per model per `effective_trading_day` (matching the day-clustered DM primary); stationary bootstrap (Politis-Romano) over DAYS with expected block length = horizon, B=2000, seed 2026; MCS size 0.10 (90% set), T_max variant. Losses: QLIKE (vol-unit) and squared error, both reported. Engine: transparent numpy implementation; `arch` cross-check ran and is reported.

## HEADLINE (honest)

*(MCS membership means “cannot be statistically separated from the best model”, NOT “beats HAR”. SPA rejection means some model beats HAR.)*

- **No pure disclosure-text model — none of the B or C blocks, including every LLM elicitation (C6 qwen, C6 llama70) — enters the 90% MCS in ANY of the 18 loss×disclosure×horizon panels.** The text family is jointly excluded from the top predictive tier by the aggregate MCS, closing the post-hoc-selection loophole the pairwise DM leaves open.

- SPA benchmarking HAR against the **entire text+fusion block (B/C/D)** never rejects: consistent p in [1.000, 1.000] across all 18 panels (large p = HAR not beaten by text/fusion as a class) — the direct aggregate-power test of the thesis.

- Some price+text **fusion** models (D block — D3 embedding fusions, D1/D2) do enter the 90% MCS in 12/18 panels (24 slots). This is not a text win: fusion models embed the HAR/RV price signal, so they can *tie* the price tier without beating it (SPA never rejects in their favour). No PURE-text model ever ties — the survivor set is always price models ± price-carrying fusion.

- The full-set SPA rejects HAR in 9/18 panels; in **9/9** the beating model is a **price** model (VIX-augmented HARX, GARCH, HARQ, semivariance — all known price results) and in 0/9 a text/fusion model. HAR-RV's only genuine competitors are other price models, never disclosure text.

## Panel table

`spa_p(cons)` = Hansen consistent p, H0 “HAR not inferior to the best of ALL alternatives” (large = HAR not beaten). `spa_tf_p` = same but vs the text/fusion block only. MCS90 columns count survivors by block (price / text / fusion) out of the totals present.

| disc | h | loss | n_obs | n_days | K | spa_p(low/cons/up) | best challenger (t) | spa_tf_p(cons) | tf best (t) | MCS90 price/text/fusion | arch tf-in-MCS |
|---|--:|---|--:|--:|--:|---|---|--:|---|---|--:|
| long_form | 5 | QLIKE (vol-unit) | 7550 | 792 | 26 | 0.155/0.385/0.858 | A7_harx_vix (+1.00) | 1.000 | D1_concat_mlp (-3.60) | 5/0/1 (of 8/12/6) | 1 |
| long_form | 5 | squared error (vol^2) | 7550 | 792 | 26 | 0.148/0.462/0.778 | A7_harx_vix (+1.05) | 1.000 | D1_concat_mlp (-1.75) | 4/0/2 (of 8/12/6) | 2 |
| long_form | 10 | QLIKE (vol-unit) | 7167 | 766 | 26 | 0.184/0.350/0.744 | A7_harx_vix (+1.08) | 1.000 | D3_gteqwen2 (-2.49) | 7/0/4 (of 8/12/6) | 4 |
| long_form | 10 | squared error (vol^2) | 7167 | 766 | 26 | 0.117/0.343/0.610 | A7_harx_vix (+1.22) | 1.000 | D3_gteqwen2 (-1.19) | 4/0/1 (of 8/12/6) | 2 |
| long_form | 20 | QLIKE (vol-unit) | 7097 | 765 | 26 | 0.090/0.150/0.278 | A7_harx_vix (+1.61) | 1.000 | D3_gteqwen2 (-1.62) | 7/0/4 (of 8/12/6) | 4 |
| long_form | 20 | squared error (vol^2) | 7097 | 765 | 26 | 0.065/0.116/0.203 | A7_harx_vix (+1.78) | 1.000 | D3_gteqwen2 (-0.51) | 4/0/3 (of 8/12/6) | 3 |
| event_driven | 5 | QLIKE (vol-unit) | 23855 | 996 | 27 | 0.018/0.021/0.042 | A7_harx_vix (+2.58) | 1.000 | D3_gteqwen2 (-3.12) | 1/0/0 (of 8/13/6) | 0 |
| event_driven | 5 | squared error (vol^2) | 23855 | 996 | 27 | 0.074/0.074/0.182 | A7_harx_vix (+1.94) | 1.000 | D3_gteqwen2 (-2.32) | 2/0/0 (of 8/13/6) | 0 |
| event_driven | 10 | QLIKE (vol-unit) | 22785 | 991 | 27 | 0.016/0.021/0.037 | A6_shar (+2.60) | 1.000 | D3_gteqwen2 (-3.27) | 6/0/1 (of 8/13/6) | 1 |
| event_driven | 10 | squared error (vol^2) | 22785 | 991 | 27 | 0.014/0.014/0.018 | A6_harq (+2.84) | 1.000 | D3_gteqwen2 (-2.41) | 2/0/0 (of 8/13/6) | 0 |
| event_driven | 20 | QLIKE (vol-unit) | 22318 | 981 | 27 | 0.002/0.002/0.003 | A6_shar (+3.58) | 1.000 | D3_gteqwen2 (-2.21) | 6/0/2 (of 8/13/6) | 2 |
| event_driven | 20 | squared error (vol^2) | 22318 | 981 | 27 | 0.005/0.005/0.005 | A6_harq (+3.22) | 1.000 | D3_gteqwen2 (-1.46) | 3/0/0 (of 8/13/6) | 0 |
| combined | 5 | QLIKE (vol-unit) | 31405 | 996 | 26 | 0.030/0.030/0.070 | A7_harx_vix (+2.33) | 1.000 | D3_gteqwen2 (-4.09) | 1/0/0 (of 8/12/6) | 0 |
| combined | 5 | squared error (vol^2) | 31405 | 996 | 26 | 0.080/0.080/0.181 | A7_harx_vix (+1.90) | 1.000 | D3_gteqwen2 (-2.73) | 3/0/0 (of 8/12/6) | 0 |
| combined | 10 | QLIKE (vol-unit) | 29952 | 991 | 26 | 0.021/0.027/0.051 | A6_shar (+2.42) | 1.000 | D3_gteqwen2 (-2.51) | 6/0/2 (of 8/12/6) | 2 |
| combined | 10 | squared error (vol^2) | 29952 | 991 | 26 | 0.046/0.052/0.082 | A6_harq (+2.18) | 1.000 | D3_gteqwen2 (-1.31) | 3/0/1 (of 8/12/6) | 1 |
| combined | 20 | QLIKE (vol-unit) | 29415 | 981 | 26 | 0.002/0.003/0.003 | A6_shar (+3.52) | 1.000 | D3_gteqwen2 (-1.41) | 6/0/2 (of 8/12/6) | 2 |
| combined | 20 | squared error (vol^2) | 29415 | 981 | 26 | 0.013/0.015/0.019 | A6_harq (+2.69) | 1.000 | D3_gteqwen2 (-0.52) | 3/0/1 (of 8/12/6) | 1 |

## 90% Model Confidence Set membership (who survives)

- **long_form h5 [QLIKE (vol-unit)]** (n_days=792): A2_har_rv; A4_egarch; A6_harq; A6_shar; A7_harx_vix; D1_concat_mlp
- **long_form h5 [squared error (vol^2)]** (n_days=792): A2_har_rv; A6_harq; A6_shar; A7_harx_vix; D1_concat_mlp; D3_gteqwen2
- **long_form h10 [QLIKE (vol-unit)]** (n_days=766): A2_har_rv; A3_garch; A4_egarch; A5_arima; A6_harq; A6_shar; A7_harx_vix; D2_gated_fusion; D3_e5mistral; D3_gteqwen2; D3_qwen3
- **long_form h10 [squared error (vol^2)]** (n_days=766): A2_har_rv; A6_harq; A6_shar; A7_harx_vix; D3_gteqwen2
- **long_form h20 [QLIKE (vol-unit)]** (n_days=765): A2_har_rv; A3_garch; A4_egarch; A5_arima; A6_harq; A6_shar; A7_harx_vix; D2_gated_fusion; D3_e5mistral; D3_gteqwen2; D3_qwen3
- **long_form h20 [squared error (vol^2)]** (n_days=765): A2_har_rv; A6_harq; A6_shar; A7_harx_vix; D3_e5mistral; D3_gteqwen2; D3_qwen3
- **event_driven h5 [QLIKE (vol-unit)]** (n_days=996): A7_harx_vix
- **event_driven h5 [squared error (vol^2)]** (n_days=996): A6_harq; A7_harx_vix
- **event_driven h10 [QLIKE (vol-unit)]** (n_days=991): A2_har_rv; A3_garch; A4_egarch; A6_harq; A6_shar; A7_harx_vix; D3_gteqwen2
- **event_driven h10 [squared error (vol^2)]** (n_days=991): A6_harq; A7_harx_vix
- **event_driven h20 [QLIKE (vol-unit)]** (n_days=981): A2_har_rv; A3_garch; A4_egarch; A6_harq; A6_shar; A7_harx_vix; D3_gteqwen2; D3_qwen3
- **event_driven h20 [squared error (vol^2)]** (n_days=981): A6_harq; A6_shar; A7_harx_vix
- **combined h5 [QLIKE (vol-unit)]** (n_days=996): A7_harx_vix
- **combined h5 [squared error (vol^2)]** (n_days=996): A6_harq; A6_shar; A7_harx_vix
- **combined h10 [QLIKE (vol-unit)]** (n_days=991): A2_har_rv; A3_garch; A4_egarch; A6_harq; A6_shar; A7_harx_vix; D3_gteqwen2; D3_qwen3
- **combined h10 [squared error (vol^2)]** (n_days=991): A6_harq; A6_shar; A7_harx_vix; D3_gteqwen2
- **combined h20 [QLIKE (vol-unit)]** (n_days=981): A2_har_rv; A3_garch; A4_egarch; A6_harq; A6_shar; A7_harx_vix; D3_gteqwen2; D3_qwen3
- **combined h20 [squared error (vol^2)]** (n_days=981): A6_harq; A6_shar; A7_harx_vix; D3_gteqwen2

## SANITY

- **G1 PASS** — per-model mean TEST QLIKE reproduces seed_aggregate.csv `qlike_mean` (variance-unit Patton, own test split, seed-averaged) to machine precision (rtol 1e-09) for anchors B2_tfidf_ridge/long_form/h5, C2_finbert_s1/long_form/h5, C2_finbert_s1/event_driven/h10, D2_gated_fusion/combined/h20.
- **G2 PASS** — the day-clustered SE Diebold-Mariano vs A2 reproduces dm_pairwise_clustered.csv `dm_clust`, `n_obs`, `n_days` to machine precision (rtol 1e-09) for anchors long_form/h5/A3_garch, long_form/h5/C2_finbert_s1, event_driven/h10/C2_finbert_s1, combined/h20/B2_tfidf_ridge — i.e. the exact seed-ensemble + inner-join + daily-block loss machinery fed to SPA/MCS is verified against the committed leaderboard.
- The SPA/MCS panels necessarily use the COMMON inner-joined sample and vol-unit QLIKE / squared error (per-obj means differ from the own-sample leaderboard by construction); the gate verifies the loss/clustering code on the leaderboard's own convention.
