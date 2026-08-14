# PREREG D — range-based (Parkinson primary / GK restated) 69-cell cascade

Pre-registered in configs/prereg_mechanism_and_labels.md §D (tag prereg-cd-v1.0), single-shot. Text-arm predictions FROZEN (3-seed ensemble, the declared primary object); combiner/recalibration + firm-mean refit on validation with the NEW labels; A2 + A6_shar refit on range-based features+labels via the committed fitting code; A3/A4/A5 frozen label-free return-based (recalibrated). Day-clustered DM, Holm within each pre-declared 69-cell family, placebo gate, per-cell MDE + injection recovery — machinery verbatim from the committed cascade (G1-gated below).

## OLD vs NEW cascade (Holm<.05 detections per stage, 69 cells)

| stage | committed (close-to-close) | **Parkinson (primary)** | GK (restated) |
|---|---|---|---|
| primary: text over recalibrated HAR | 38/69 | **21/69** | 23/69 |
| firm-identity-augmented reference | 8/69 | **7/69** | 8/69 |
| maximal 5-price pool | 9/69 | **15/69** | 15/69 |
| **full conjunction (primary AND firm AND pool)** | 0/69 | **1/69** | 2/69 |
| placebo-gated genuine (primary stage) | 38/69 | **19/69** | 21/69 |

## MDE — old vs new (per-cell, 80% power, 5% two-sided; prereg branch (a) needs median shrink >= 30%)

| | committed | **Parkinson** | GK |
|---|---|---|---|
| median MDE_rel% | 0.823 | **0.913** | 1.018 |
| IQR | [0.37, 1.27] | [0.60, 1.44] | [0.77, 1.83] |
| median shrink vs committed | — | **-10.9%** | -23.6% |

## Injection recovery — old vs new (Holm-detected /69 per pre-declared (stage, level) family)

| level | committed HAR/firm/pool/all3 | **Parkinson** HAR/firm/pool/all3 | GK HAR/firm/pool/all3 |
|---|---|---|---|
| 0.3% | 12/7/6/2 | **7/1/4/1** | 4/2/3/2 |
| 0.5% | 20/11/12/6 | **11/4/9/3** | 8/5/6/4 |
| 1.0% | 41/20/19/13 | **26/16/20/11** | 19/14/15/6 |

## FIRED BRANCH: **(c)**

conjunction > 0 under Parkinson: text survives primary AND firm-identity AND maximal-pool jointly under Holm — honest reversal; residual chapter must be rewritten (prereg (c)).

Branch-(d) reference-ordering check (A2 rank among the 5 single recalibrated price references, per disc x h): original labels mean rank 3.67 (rank-1 in 0/6), Parkinson mean rank 1.50 (rank-1 in 3/6). HAR remains the strong reference — (d) does not fire.

## Per-cell detail — Parkinson (primary object)

| disc | model | h | n_test | rel% old->PK (primary) | DM old->PK | detect old H/F/P | detect PK H/F/P | placebo PK | MDE old->PK |
|---|---|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | 25109 | +1.33->+1.44 | -3.35->-3.32 | Y/Y/. | Y/Y/. | +1.13 | 0.80->0.83 |
| event_driven | B1_bow_ridge | 10 | 25001 | +1.23->+0.83 | -3.25->-1.39 | Y/Y/. | ./Y/. | +0.02 | 0.82->1.00 |
| event_driven | B1_bow_ridge | 20 | 24732 | +1.53->+0.95 | -3.10->-1.19 | Y/./. | ././. | +0.87 | 1.22->1.33 |
| event_driven | B2_tfidf_ridge | 5 | 25109 | +1.21->+1.35 | -2.76->-2.04 | ././. | ./Y/. | +0.98 | 0.76->0.95 |
| event_driven | B2_tfidf_ridge | 10 | 25001 | +1.35->+1.11 | -2.97->-1.46 | ./Y/. | ./Y/. | +0.40 | 0.87->1.14 |
| event_driven | B2_tfidf_ridge | 20 | 24732 | +1.84->+1.23 | -3.11->-1.07 | Y/./. | ././. | +0.36 | 1.36->1.60 |
| event_driven | B3_lm_linear | 5 | 25109 | +0.25->+0.43 | -2.43->-1.83 | ././. | ././. | -0.22 | 0.20->0.34 |
| event_driven | B3_lm_linear | 10 | 25001 | +0.20->+0.23 | -1.23->-0.60 | ././. | ././. | +0.43 | 0.36->0.60 |
| event_driven | B3_lm_linear | 20 | 24732 | +0.26->+0.20 | -1.12->-0.24 | ././. | ././. | +0.79 | 0.55->0.91 |
| event_driven | B4_lm_features | 5 | 25109 | +0.18->+0.32 | -0.99->-2.86 | ././. | ./Y/. | -0.03 | 0.18->0.19 |
| event_driven | B4_lm_features | 10 | 25001 | +0.08->+0.10 | -2.10->-2.99 | ././. | ././. | +0.03 | 0.08->0.08 |
| event_driven | B4_lm_features | 20 | 24732 | +0.25->+0.27 | -2.01->-1.36 | ././. | ././. | -0.55 | 0.37->0.56 |
| event_driven | C2_finbert_s1 | 5 | 25109 | +2.57->+2.51 | -5.92->-4.73 | Y/./. | Y/Y/Y | +0.29 | 0.86->1.04 |
| event_driven | C2_finbert_s1 | 10 | 25001 | +2.42->+2.29 | -5.68->-3.60 | Y/./. | Y/./. | +0.54 | 0.93->1.38 |
| event_driven | C2_finbert_s1 | 20 | 24732 | +1.58->-1.45 | -1.94->+0.80 | ././. | ././. | -0.12 | 2.63->3.26 |
| event_driven | C6_llmtext | 5 | 25109 | +1.21->+1.41 | -5.04->-3.67 | Y/Y/. | Y/Y/. | -0.47 | 0.50->0.69 |
| event_driven | C6_llmtext | 10 | 25001 | +1.00->+0.84 | -3.76->-1.54 | Y/Y/. | ././. | +2.02 | 0.56->0.81 |
| event_driven | C6_llmtext | 20 | 24732 | +0.66->+0.49 | -1.98->-0.45 | ./Y/. | ././. | +0.60 | 0.70->1.04 |
| event_driven | D2_gated_fusion | 5 | 25109 | +0.56->-0.06 | -1.70->+0.86 | ././. | ././. | +0.74 | 0.37->0.50 |
| event_driven | D2_gated_fusion | 10 | 25001 | +0.18->-1.09 | -0.42->+4.22 | ././. | ././. | +0.49 | 0.33->0.72 |
| event_driven | D2_gated_fusion | 20 | 24732 | -2.76->-0.17 | +3.55->+1.84 | ././. | ././. | -0.26 | 2.25->0.27 |
| event_driven | D4_llmfused | 5 | 25109 | -0.01->-0.35 | +3.37->+2.62 | ././. | ././. | +0.49 | 0.01->0.44 |
| event_driven | D4_llmfused | 10 | 25001 | -0.01->-0.31 | +0.66->+2.64 | ././. | ././. | +0.14 | 0.04->0.38 |
| event_driven | D4_llmfused | 20 | 24732 | -0.35->-0.09 | +4.69->+1.54 | ././. | ././. | -0.32 | 0.20->0.22 |
| long_form | B1_bow_ridge | 5 | 7951 | +1.65->+1.85 | -3.83->-3.11 | Y/./. | ././. | -0.18 | 1.11->1.19 |
| long_form | B1_bow_ridge | 10 | 7932 | +1.44->+1.17 | -4.15->-2.70 | Y/./. | ././. | +1.25 | 0.81->0.79 |
| long_form | B1_bow_ridge | 20 | 7902 | +2.99->+2.82 | -5.45->-3.10 | Y/./. | ././. | -0.07 | 1.38->1.72 |
| long_form | B2_tfidf_ridge | 5 | 7951 | +3.33->+4.10 | -5.39->-5.34 | Y/./Y | Y/./Y | -0.84 | 1.67->2.10 |
| long_form | B2_tfidf_ridge | 10 | 7932 | +3.48->+3.23 | -8.89->-6.19 | Y/./Y | Y/./Y | +2.27 | 1.25->1.44 |
| long_form | B2_tfidf_ridge | 20 | 7902 | +5.92->+5.13 | -9.04->-4.95 | Y/./Y | Y/./Y | +0.86 | 1.86->2.35 |
| long_form | B3_lm_linear | 5 | 7951 | +0.49->+0.96 | -2.58->-1.96 | ././. | ././. | +1.59 | 1.70->1.43 |
| long_form | B3_lm_linear | 10 | 7932 | +1.79->+1.76 | -4.62->-2.84 | Y/Y/. | ././. | +0.79 | 1.70->1.38 |
| long_form | B3_lm_linear | 20 | 7902 | +3.48->+3.07 | -4.69->-2.09 | Y/./. | ././. | -1.35 | 1.87->1.82 |
| long_form | B4_lm_features | 5 | 7951 | +0.11->+0.39 | +1.02->+0.71 | ././. | ././. | +1.44 | 0.21->0.34 |
| long_form | B4_lm_features | 10 | 7932 | -0.92->-1.73 | +3.32->+2.88 | ././. | ././. | -0.07 | 0.57->0.82 |
| long_form | B4_lm_features | 20 | 7902 | -1.92->-2.57 | +3.38->+1.76 | ././. | ././. | -3.58 | 0.70->1.05 |
| long_form | C1_bert_s1 | 5 | 7951 | +3.42->+3.87 | -3.25->-2.73 | Y/./. | ././. | +1.73 | 2.99->4.06 |
| long_form | C1_bert_s1 | 10 | 7932 | -0.85->-0.11 | +3.06->+0.63 | ././. | ././. | +1.14 | 0.84->0.91 |
| long_form | C1_bert_s1 | 20 | 7902 | +2.70->+2.24 | -6.96->-3.67 | Y/./. | Y/./Y | +0.97 | 1.27->1.58 |
| long_form | C2_finbert_s1 | 5 | 7951 | +1.90->+1.33 | -4.53->-2.06 | Y/./. | ././. | -0.12 | 0.95->1.26 |
| long_form | C2_finbert_s1 | 10 | 7932 | +2.62->+2.67 | -6.67->-4.23 | Y/./. | Y/./Y | -0.73 | 1.27->1.71 |
| long_form | C2_finbert_s1 | 20 | 7902 | -0.59->-3.92 | -0.40->+1.77 | ././. | ././. | +1.29 | 2.82->4.02 |
| long_form | C2_finbert_s2 | 5 | 7951 | +1.21->+1.31 | -5.57->-6.05 | Y/./Y | Y/./Y | +1.38 | 0.59->0.68 |
| long_form | C2_finbert_s2 | 10 | 7932 | +0.48->-0.18 | -7.59->+6.12 | Y/./. | ././. | +1.02 | 0.26->0.13 |
| long_form | C2_finbert_s2 | 20 | 7902 | +1.68->+0.47 | -4.89->-0.97 | Y/./. | ././. | +0.00 | 0.93->0.89 |
| long_form | C2_finbert_s3 | 5 | 7951 | +2.90->+2.02 | -5.36->-4.13 | Y/./. | Y/./Y | +0.94 | 1.35->1.71 |
| long_form | C2_finbert_s3 | 10 | 7932 | +2.31->+1.15 | -5.26->-3.47 | Y/./. | Y/./Y | +0.38 | 2.36->3.00 |
| long_form | C2_finbert_s3 | 20 | 7902 | -3.86->-8.19 | +1.68->+3.71 | ././. | ././. | -1.83 | 3.65->4.31 |
| long_form | C2_finbert_s4 | 5 | 7951 | +1.46->+1.66 | -4.78->-6.53 | Y/./. | Y/./Y | +0.44 | 1.07->0.89 |
| long_form | C2_finbert_s4 | 10 | 7932 | +0.36->+0.09 | -6.82->-4.76 | Y/./. | Y/./. | -0.29 | 0.22->0.08 |
| long_form | C2_finbert_s4 | 20 | 7902 | +3.08->+2.64 | -8.34->-5.41 | Y/./Y | Y/./Y | +0.90 | 1.31->1.63 |
| long_form | C3_roberta_s1 | 5 | 7951 | +0.30->+0.24 | -5.13->-3.64 | Y/./. | Y/./. | +0.07 | 0.14->0.19 |
| long_form | C3_roberta_s1 | 10 | 7932 | +1.84->+1.56 | -5.06->-3.34 | Y/./. | Y/./. | -0.73 | 1.11->1.23 |
| long_form | C3_roberta_s1 | 20 | 7902 | +0.02->+0.06 | -3.90->-1.24 | Y/./. | ././. | +0.21 | 0.02->0.18 |
| long_form | C4_longformer | 5 | 7951 | +1.47->+1.09 | -6.18->-3.96 | Y/./Y | Y/./Y | -0.49 | 0.69->0.89 |
| long_form | C4_longformer | 10 | 7932 | -2.95->-2.25 | +10.32->+7.45 | ././. | ././. | -0.15 | 0.96->1.06 |
| long_form | C4_longformer | 20 | 7902 | +0.91->+0.13 | -4.38->-0.51 | Y/./. | ././. | +0.84 | 0.63->0.77 |
| long_form | C6_llmtext | 5 | 7951 | +1.79->+1.48 | -6.31->-5.03 | Y/./Y | Y/./Y | +1.95 | 1.11->0.87 |
| long_form | C6_llmtext | 10 | 7932 | +2.25->+1.46 | -7.92->-5.24 | Y/./Y | Y/./Y | -2.50 | 1.07->0.84 |
| long_form | C6_llmtext | 20 | 7902 | +0.27->+0.04 | -3.23->-3.33 | Y/Y/. | Y/./. | -0.66 | 0.37->0.05 |
| long_form | D1_concat_mlp | 5 | 7951 | -1.04->+0.01 | +2.66->-1.86 | ././. | ././. | +1.47 | 1.21->0.10 |
| long_form | D1_concat_mlp | 10 | 7932 | -0.46->-0.66 | +3.54->+3.74 | ././. | ././Y | -1.06 | 0.62->0.84 |
| long_form | D1_concat_mlp | 20 | 7902 | +0.12->-1.39 | -5.57->+3.66 | Y/./Y | ././. | -0.63 | 0.09->2.84 |
| long_form | D2_gated_fusion | 5 | 7951 | +0.19->-0.37 | -2.11->+2.45 | ././. | ././. | +1.20 | 0.29->0.89 |
| long_form | D2_gated_fusion | 10 | 7932 | -0.02->-2.04 | +2.02->+6.56 | ././. | ././. | -0.95 | 0.03->1.40 |
| long_form | D2_gated_fusion | 20 | 7902 | -2.27->-0.22 | +5.56->+0.78 | ././. | ././. | -0.13 | 3.04->0.37 |
| long_form | D4_llmfused | 5 | 7951 | +0.15->-0.80 | -0.75->+3.57 | ././. | ././. | +0.95 | 0.32->1.35 |
| long_form | D4_llmfused | 10 | 7932 | -0.15->-1.09 | -0.43->+3.51 | ././. | ././. | +0.12 | 0.84->1.94 |
| long_form | D4_llmfused | 20 | 7902 | +0.60->-0.89 | -3.58->+2.95 | Y/./. | ././. | -0.45 | 0.73->2.46 |

## Disclosures

1. **Frozen text predictions (FIRST LIMITATION-FEEDER).** Every text arm was trained and tuned against the close-to-close RV target; its predictions are reused frozen and only the log-space recalibration/combiner weights are refit on validation under the new labels. The readout is therefore CONSERVATIVE for the text side: a text arm optimised directly for range-based targets could do better. All range-based null readings must carry this caveat.
2. **Estimator formulas.** Parkinson: pk_i = ln(H_i/L_i)^2/(4 ln 2); Garman-Klass (standard): gk_i = 0.5 ln(H_i/L_i)^2 - (2 ln 2 - 1) ln(C_i/O_i)^2. Labels sqrt(252/H * sum est_i) over the SAME H-trading-day windows as the committed labels; features sqrt(252/w * trailing-w-valid-day sums) ending at feature_window_end — the exact alignment.py conventions (verified to <1e-8, see SANITY). GK day values can be negative; window sums are clipped at 0 (label clip count: 0; feature clips: {'gk_1d_clipped': 0, 'gk_5d_clipped': 0, 'gk_22d_clipped': 0}).
3. **Feature-side consistency.** A2_har_rv refit (HARRV class: train split, per horizon, log OLS + Duan smearing) on range-based rv_1d/5d/22d + range-based label; A6_shar refit (stronger_baselines conventions incl. the BPQ insanity filter) with rv_5/rv_22 converted — its signed daily semivols RS-/RS+ remain return-based (a range estimator has no sign decomposition; disclosed, not a free choice). A3_garch/A4_egarch/A5_arima are label-free return-based forecasters with no aligned-panel RV features: frozen, recalibrated on val — identical to the committed combination-time treatment. The firm-identity reference term is the firm's own VAL-split mean of the NEW label.
4. **Refit machinery validated**: on the ORIGINAL features/labels the refit code reproduces the stored A2/A6_shar prediction parquets (max abs diffs in SANITY). Same seeds, same placebo permutations, same Holm families as the committed cascade. Single-shot: this table was written once; the script refuses to overwrite it.

## SANITY

| gate | result |
|---|---|
| G1 cascade path reproduces committed tables (original labels) | **PASS** — max abs diff 2.01e-10 over primary/firm/pool/MDE/placebo; counts {'primary_holm': [38, 38], 'firm_holm': [8, 8], 'pool_holm': [9, 9], 'conjunction': [0, 0], 'genuine': [38, 38]}; injection counts match: True |
| G1 refit machinery reproduces stored A2/A6_shar runs | max abs diff 1.22e-14 (tol 1e-06) |
| G2 rank correlation new-vs-old labels (gate > 0.8) | h5: PK 0.8452 / GK 0.7811, h10: PK 0.8921 / GK 0.8499, h20: PK 0.9215 / GK 0.8949 — PK PASS, GK BELOW GATE (disclosed) |
| G3 leakage asserts | PASS (fwe < label_window_start; label window strictly after effective day; feature days <= fwe by construction); combiner/reference weights val-only by construction (log_ols_frozen / log_combo) |
| G4 placebo (primary stage, 5 permutation seeds) | |placebo DM|<2 in 65/69 Parkinson cells (committed convention) |
| Coverage (labels parquet, 431245 panel rows) | PK label lost 1, GK label lost 1; usable rows old/PK/GK = 427429/427440/427440 |
| Cascade row losses (PK) | long_form A2 refit rows lost: 1/94237; long_form A6_shar refit rows lost: 0/85195; long_form 5-price panel rows lost: 0/85195; event_driven A2 refit rows lost: 0/333192; event_driven A6_shar refit rows lost: 0/305014; event_driven 5-price panel rows lost: 0/305014 |
| Cascade row losses (GK) | long_form A2 refit rows lost: 1/94237; long_form A6_shar refit rows lost: 0/85195; long_form 5-price panel rows lost: 0/85195; event_driven A2 refit rows lost: 0/333192; event_driven A6_shar refit rows lost: 0/305014; event_driven 5-price panel rows lost: 0/305014 |
| Per-cell n_test drift (PK vs committed) | max 1 obs, total 15 obs across 69 cells |
