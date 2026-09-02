# PREREG H — licence-free public-price variant: three-panel full cascade + standalone leaderboard

Pre-registered in configs/prereg_public_variant.md (tag prereg-h-v1.0), single-shot. Panels: **A** = full panel + CRSP labels (committed anchor, G1-reproduced); **B** = public-coverage rows + CRSP labels (survivorship isolated); **C** = public-coverage rows + PUBLIC labels/features (the shippable variant). Text-arm predictions FROZEN (3-seed ensemble, the declared primary object); combiner/recalibration + firm-mean refit on validation with each panel's own rows/labels; panel C refits A2 + A6_shar on public features+labels via the committed fitting code, A3/A4/A5 frozen label-free (val-recalibrated). Day-clustered DM, Holm within each pre-declared family, placebo gate, per-cell MDE + injection recovery — machinery verbatim from the committed cascade (G1-gated below).

**THE TRADE, priced openly (recomputed from this refetch):** coverage: train 72.6% / val 89.3% / test 95.2% of modelled rows; benchmark-row clean coverage 79.83%; exit-firm rows 31.5% vs active 97.3%.

## FIRED BRANCH: **(c)**

VERDICT FLIP — conjunction > 0 (B=1, C=1). Honest report; the variant still ships; the flip itself becomes the finding, located by the A-vs-B-vs-C decomposition (prereg (c)).

**Branch operationalisation (fixed before any B/C statistic was read):** (c) fires iff conjunction>0 on B or C OR any text/fusion arm (any challenger outside {A3_garch, A4_egarch, A5_arima}) becomes a Holm standalone winner on B or C; (a) fires iff conjunction stays 0/69 on both panels AND all 180 per-comparison standalone Holm verdicts match panel A on B and on C; (b) otherwise. Panel A's committed standalone verdicts already contain price-arm winners (7/180, all GARCH-family), so 'a standalone winner appears' is read against the paper's verdict object: the TEXT standalone null.

## A-vs-B-vs-C decomposition (A-B = survivorship; B-C = label source)

Cascade cells: 69 per panel. coverage: train 72.6% / val 89.3% / test 95.2% of modelled rows; benchmark-row clean coverage 79.83%; exit-firm rows 31.5% vs active 97.3%.

| object | A (committed anchor) | **B (covered, CRSP labels)** | **C (covered, public labels)** | A-B (survivorship) | B-C (label source) |
|---|---|---|---|---|---|
| primary: text over recalibrated HAR (Holm) | 38/69 | **35/69** | **36/69** | +3 | -1 |
| firm-identity-augmented reference (Holm) | 8/69 | **9/69** | **9/69** | -1 | +0 |
| maximal 5-price pool (Holm) | 9/69 | **11/69** | **13/69** | -2 | -2 |
| **full conjunction (primary AND firm AND pool)** | 0/69 | **1/69** | **1/69** | -1 | +0 |
| placebo-gated genuine (primary stage) | 38/69 | **32/69** | **31/69** | +6 | +1 |
| median MDE_rel% | 0.823 | **0.762** | **0.782** | +0.061 | -0.019 |
| standalone better-than-A2 (Holm, of 180) | 7 | **7** | **7** | +0 | +0 |
| ... of which TEXT/fusion arms | 0/153 | **0/153** | **0/153** |  |  |
| standalone significantly WORSE (Holm) | 153 | 153 | 153 |  |  |

## Standalone leaderboard — day-clustered variance-unit QLIKE DM vs A2 (committed convention; Holm within each (disclosure, horizon) group)

Universe: the dm_pairwise_clustered.csv 180 comparisons (20 seed-ensembled challengers x 3 disclosures x 3 horizons). coverage: train 72.6% / val 89.3% / test 95.2% of modelled rows; benchmark-row clean coverage 79.83%; exit-firm rows 31.5% vs active 97.3%.

| panel | better raw | better Holm | text/fusion better Holm | worse Holm | Holm winners |
|---|---|---|---|---|---|
| A (committed) | 7 | 7 | 0/153 | 153 | combined/h10/A3_garch; combined/h20/A3_garch; combined/h5/A3_garch; event_driven/h10/A3_garch; event_driven/h20/A3_garch; event_driven/h5/A3_garch; long_form/h5/A4_egarch |
| B | 7 | 7 | 0/153 | 153 | combined/h10/A3_garch; combined/h20/A3_garch; combined/h5/A3_garch; event_driven/h10/A3_garch; event_driven/h20/A3_garch; event_driven/h5/A3_garch; long_form/h5/A4_egarch |
| C | 7 | 7 | 0/153 | 153 | combined/h10/A3_garch; combined/h20/A3_garch; combined/h5/A3_garch; event_driven/h10/A3_garch; event_driven/h20/A3_garch; event_driven/h5/A3_garch; long_form/h5/A4_egarch |

Per-comparison Holm-verdict flips vs panel A: **B 0/180, C 0/180**.

## Injection recovery — Holm-detected /69 per pre-declared (stage, level) family

| level | A (committed) HAR/firm/pool/all3 | **B** HAR/firm/pool/all3 | **C** HAR/firm/pool/all3 |
|---|---|---|---|
| 0.3% | 12/7/6/2 | **12/5/7/2** | **13/6/7/2** |
| 0.5% | 20/11/12/6 | **18/12/12/6** | **20/10/11/4** |
| 1.0% | 41/20/19/13 | **40/21/21/15** | **38/20/18/13** |

## MDE — per panel (80% power, 5% two-sided)

| | A (committed) | B | C |
|---|---|---|---|
| median MDE_rel% | 0.823 | 0.762 | 0.782 |
| IQR | [0.37, 1.27] | [0.34, 1.19] | [0.34, 1.19] |

Reference-ordering diagnostic (A2 rank among the 5 single recalibrated price references, per disc x h): A mean rank 3.67 (rank-1 in 0/6); B mean rank 4.17 (rank-1 in 0/6); C mean rank 4.00 (rank-1 in 0/6).

## Per-cell detail — panels B and C vs the committed panel A

| disc | model | h | n_test A->B->C | rel% A->B->C (primary) | DM A->B->C | detect A H/F/P | detect B H/F/P | detect C H/F/P | placebo B/C | MDE A->B->C |
|---|---|---|---|---|---|---|---|---|---|---|
| event_driven | B1_bow_ridge | 5 | 25109->23878->23874 | +1.33->+1.41->+1.42 | -3.35->-3.91->-3.93 | Y/Y/. | Y/Y/. | Y/Y/. | +0.59/+0.20 | 0.80->0.74->0.74 |
| event_driven | B1_bow_ridge | 10 | 25001->23788->23784 | +1.23->+1.36->+1.36 | -3.25->-4.01->-3.99 | Y/Y/. | Y/Y/. | Y/Y/. | -0.06/+0.44 | 0.82->0.74->0.74 |
| event_driven | B1_bow_ridge | 20 | 24732->23551->23547 | +1.53->+1.66->+1.68 | -3.10->-3.34->-3.34 | Y/./. | Y/./. | Y/./. | -0.42/+1.11 | 1.22->1.18->1.18 |
| event_driven | B2_tfidf_ridge | 5 | 25109->23878->23874 | +1.21->+1.14->+1.15 | -2.76->-3.02->-3.00 | ././. | ././. | ././. | +0.57/+0.21 | 0.76->0.65->0.65 |
| event_driven | B2_tfidf_ridge | 10 | 25001->23788->23784 | +1.35->+1.29->+1.28 | -2.97->-2.96->-2.88 | ./Y/. | ./Y/. | ./Y/. | +0.19/-0.05 | 0.87->0.78->0.79 |
| event_driven | B2_tfidf_ridge | 20 | 24732->23551->23547 | +1.84->+1.79->+1.79 | -3.11->-2.76->-2.72 | Y/./. | ././. | ././. | +0.31/+0.81 | 1.36->1.39->1.40 |
| event_driven | B3_lm_linear | 5 | 25109->23878->23874 | +0.25->+0.21->+0.22 | -2.43->-2.00->-2.01 | ././. | ././. | ././. | +0.91/+0.66 | 0.20->0.20->0.20 |
| event_driven | B3_lm_linear | 10 | 25001->23788->23784 | +0.20->+0.18->+0.18 | -1.23->-1.02->-1.00 | ././. | ././. | ././. | +1.34/-0.05 | 0.36->0.34->0.34 |
| event_driven | B3_lm_linear | 20 | 24732->23551->23547 | +0.26->+0.27->+0.27 | -1.12->-1.20->-1.18 | ././. | ././. | ././. | +0.20/+0.02 | 0.55->0.57->0.57 |
| event_driven | B4_lm_features | 5 | 25109->23878->23874 | +0.18->+0.10->+0.10 | -0.99->-0.34->-0.37 | ././. | ././. | ././. | -0.55/-0.54 | 0.18->0.11->0.12 |
| event_driven | B4_lm_features | 10 | 25001->23788->23784 | +0.08->-0.02->-0.02 | -2.10->+1.99->+2.07 | ././. | ././. | ././. | +0.80/+0.73 | 0.08->0.02->0.02 |
| event_driven | B4_lm_features | 20 | 24732->23551->23547 | +0.25->+0.22->+0.22 | -2.01->-1.94->-1.94 | ././. | ././. | ././. | +0.32/+0.47 | 0.37->0.37->0.37 |
| event_driven | C2_finbert_s1 | 5 | 25109->23878->23874 | +2.57->+2.56->+2.54 | -5.92->-6.24->-6.19 | Y/./. | Y/./. | Y/./. | +1.19/+2.03 | 0.86->0.85->0.85 |
| event_driven | C2_finbert_s1 | 10 | 25001->23788->23784 | +2.42->+2.46->+2.45 | -5.68->-6.38->-6.34 | Y/./. | Y/./. | Y/./. | +0.07/+0.04 | 0.93->0.88->0.88 |
| event_driven | C2_finbert_s1 | 20 | 24732->23551->23547 | +1.58->+1.64->+1.61 | -1.94->-1.93->-1.88 | ././. | ././. | ././. | -0.28/-0.13 | 2.63->2.90->2.91 |
| event_driven | C6_llmtext | 5 | 25109->23878->23874 | +1.21->+1.55->+1.55 | -5.04->-6.01->-5.96 | Y/Y/. | Y/Y/. | Y/Y/. | -0.61/-0.17 | 0.50->0.56->0.56 |
| event_driven | C6_llmtext | 10 | 25001->23788->23784 | +1.00->+1.34->+1.33 | -3.76->-4.44->-4.34 | Y/Y/. | Y/Y/. | Y/Y/. | -0.08/+0.54 | 0.56->0.68->0.69 |
| event_driven | C6_llmtext | 20 | 24732->23551->23547 | +0.66->+0.77->+0.76 | -1.98->-1.98->-1.92 | ./Y/. | ./Y/. | ./Y/. | +0.03/+0.50 | 0.70->0.85->0.86 |
| event_driven | D2_gated_fusion | 5 | 25109->23878->23874 | +0.56->+0.46->+0.47 | -1.70->-1.75->-1.47 | ././. | ././. | ././. | -3.28/-2.67 | 0.37->0.29->0.33 |
| event_driven | D2_gated_fusion | 10 | 25001->23788->23784 | +0.18->+0.18->+0.06 | -0.42->+0.03->+0.80 | ././. | ././. | ././. | +0.20/+0.14 | 0.33->0.37->0.56 |
| event_driven | D2_gated_fusion | 20 | 24732->23551->23547 | -2.76->-2.76->-2.37 | +3.55->+3.43->+3.34 | ././. | ././. | ././. | +0.83/+0.81 | 2.25->2.38->2.10 |
| event_driven | D4_llmfused | 5 | 25109->23878->23874 | -0.01->-0.06->-0.12 | +3.37->+3.69->+3.35 | ././. | ././. | ././. | -1.26/-1.41 | 0.01->0.05->0.12 |
| event_driven | D4_llmfused | 10 | 25001->23788->23784 | -0.01->-0.02->-0.03 | +0.66->+1.44->+1.35 | ././. | ././. | ././. | +0.44/+0.06 | 0.04->0.07->0.11 |
| event_driven | D4_llmfused | 20 | 24732->23551->23547 | -0.35->-0.35->-0.31 | +4.69->+4.42->+4.27 | ././. | ././. | ././. | +1.11/+0.74 | 0.20->0.20->0.18 |
| long_form | B1_bow_ridge | 5 | 7951->7565->7564 | +1.65->+1.28->+1.26 | -3.83->-4.08->-3.89 | Y/./. | Y/./. | Y/./. | -0.28/-0.25 | 1.11->0.80->0.81 |
| long_form | B1_bow_ridge | 10 | 7933->7548->7547 | +1.44->+0.97->+0.97 | -4.15->-4.22->-4.12 | Y/./. | Y/./. | Y/./. | +0.06/+0.89 | 0.81->0.56->0.57 |
| long_form | B1_bow_ridge | 20 | 7902->7526->7525 | +2.99->+2.30->+2.32 | -5.45->-5.35->-5.29 | Y/./. | Y/./. | Y/./. | +2.36/+2.30 | 1.38->1.08->1.11 |
| long_form | B2_tfidf_ridge | 5 | 7951->7565->7564 | +3.33->+3.24->+3.20 | -5.39->-4.85->-4.80 | Y/./Y | Y/./. | Y/./Y | +0.45/+0.25 | 1.67->1.66->1.67 |
| long_form | B2_tfidf_ridge | 10 | 7933->7548->7547 | +3.48->+3.52->+3.54 | -8.89->-8.41->-8.25 | Y/./Y | Y/./Y | Y/./Y | +0.60/+1.51 | 1.25->1.25->1.29 |
| long_form | B2_tfidf_ridge | 20 | 7902->7526->7525 | +5.92->+5.59->+5.64 | -9.04->-8.95->-8.78 | Y/./Y | Y/./Y | Y/./Y | -1.71/-2.38 | 1.86->1.68->1.73 |
| long_form | B3_lm_linear | 5 | 7951->7565->7564 | +0.49->+0.53->+0.51 | -2.58->-2.96->-3.01 | ././. | ././. | ././. | +1.11/+1.10 | 1.70->1.86->1.86 |
| long_form | B3_lm_linear | 10 | 7933->7548->7547 | +1.79->+2.09->+2.04 | -4.62->-5.03->-5.06 | Y/Y/. | Y/Y/Y | Y/Y/Y | +1.69/+1.62 | 1.70->1.91->1.91 |
| long_form | B3_lm_linear | 20 | 7902->7526->7525 | +3.48->+3.91->+3.87 | -4.69->-4.65->-4.69 | Y/./. | Y/Y/. | Y/Y/. | -0.16/-0.52 | 1.87->2.01->2.01 |
| long_form | B4_lm_features | 5 | 7951->7565->7564 | +0.11->+0.15->+0.15 | +1.02->+0.48->+0.50 | ././. | ././. | ././. | +0.87/+1.15 | 0.21->0.24->0.23 |
| long_form | B4_lm_features | 10 | 7933->7548->7547 | -0.92->-0.65->-0.66 | +3.32->+3.70->+3.67 | ././. | ././. | ././. | +1.21/+1.64 | 0.57->0.38->0.39 |
| long_form | B4_lm_features | 20 | 7902->7526->7525 | -1.92->-1.32->-1.30 | +3.38->+3.32->+3.29 | ././. | ././. | ././. | +1.23/+0.69 | 0.70->0.47->0.47 |
| long_form | C1_bert_s1 | 5 | 7951->7565->7564 | +3.42->+3.86->+3.79 | -3.25->-3.61->-3.48 | Y/./. | Y/./. | Y/./. | -0.12/+0.43 | 2.99->2.80->2.82 |
| long_form | C1_bert_s1 | 10 | 7933->7548->7547 | -0.85->-1.19->-1.15 | +3.06->+3.09->+2.98 | ././. | ././. | ././. | +0.17/-0.03 | 0.84->1.05->1.05 |
| long_form | C1_bert_s1 | 20 | 7902->7526->7525 | +2.70->+2.44->+2.46 | -6.96->-6.85->-6.77 | Y/./. | Y/./. | Y/./. | -0.65/-1.04 | 1.27->1.10->1.12 |
| long_form | C2_finbert_s1 | 5 | 7951->7565->7564 | +1.90->+1.88->+1.85 | -4.53->-5.25->-5.07 | Y/./. | Y/./Y | Y/./Y | +0.64/+0.46 | 0.95->0.85->0.85 |
| long_form | C2_finbert_s1 | 10 | 7933->7548->7547 | +2.62->+2.76->+2.74 | -6.67->-6.81->-6.70 | Y/./. | Y/./Y | Y/./Y | +0.38/+0.26 | 1.27->1.23->1.25 |
| long_form | C2_finbert_s1 | 20 | 7902->7526->7525 | -0.59->-0.57->-0.63 | -0.40->-0.21->-0.14 | ././. | ././. | ././. | +1.16/+0.82 | 2.82->2.60->2.62 |
| long_form | C2_finbert_s2 | 5 | 7951->7565->7564 | +1.21->+0.95->+0.94 | -5.57->-6.24->-6.16 | Y/./Y | Y/./. | Y/./. | -0.24/-0.46 | 0.59->0.40->0.40 |
| long_form | C2_finbert_s2 | 10 | 7933->7548->7547 | +0.48->+0.26->+0.29 | -7.59->-7.80->-7.76 | Y/./. | Y/./. | Y/./. | -0.29/+0.31 | 0.26->0.13->0.14 |
| long_form | C2_finbert_s2 | 20 | 7902->7526->7525 | +1.68->+1.46->+1.46 | -4.89->-4.63->-4.54 | Y/./. | Y/./. | Y/./. | +1.24/+0.90 | 0.93->0.76->0.77 |
| long_form | C2_finbert_s3 | 5 | 7951->7565->7564 | +2.90->+2.96->+2.91 | -5.36->-5.49->-5.38 | Y/./. | Y/./. | Y/./. | +0.35/+0.43 | 1.35->1.26->1.26 |
| long_form | C2_finbert_s3 | 10 | 7933->7548->7547 | +2.31->+2.38->+2.32 | -5.26->-5.14->-5.07 | Y/./. | Y/./. | Y/./. | +0.76/+0.68 | 2.36->2.35->2.37 |
| long_form | C2_finbert_s3 | 20 | 7902->7526->7525 | -3.86->-3.81->-3.93 | +1.68->+1.91->+1.99 | ././. | ././. | ././. | +3.72/+3.59 | 3.65->3.60->3.63 |
| long_form | C2_finbert_s4 | 5 | 7951->7565->7564 | +1.46->+1.41->+1.41 | -4.78->-5.08->-5.01 | Y/./. | Y/./Y | Y/./Y | +0.51/+0.60 | 1.07->0.91->0.92 |
| long_form | C2_finbert_s4 | 10 | 7933->7548->7547 | +0.36->+0.23->+0.23 | -6.82->-6.68->-6.65 | Y/./. | Y/./. | Y/./. | -0.27/+1.03 | 0.22->0.13->0.13 |
| long_form | C2_finbert_s4 | 20 | 7902->7526->7525 | +3.08->+2.58->+2.58 | -8.34->-8.33->-8.28 | Y/./Y | Y/./Y | Y/./Y | -1.12/-1.28 | 1.31->1.04->1.05 |
| long_form | C3_roberta_s1 | 5 | 7951->7565->7564 | +0.30->+0.13->+0.12 | -5.13->-6.34->-6.22 | Y/./. | Y/./. | Y/./. | +0.34/+0.19 | 0.14->0.05->0.05 |
| long_form | C3_roberta_s1 | 10 | 7933->7548->7547 | +1.84->+1.90->+1.91 | -5.06->-6.08->-5.91 | Y/./. | Y/./Y | Y/./Y | +0.75/+0.61 | 1.11->0.96->0.98 |
| long_form | C3_roberta_s1 | 20 | 7902->7526->7525 | +0.02->-0.28->-0.29 | -3.90->+3.64->+3.50 | Y/./. | ././. | ././. | +0.50/-0.02 | 0.02->0.27->0.28 |
| long_form | C4_longformer | 5 | 7951->7565->7564 | +1.47->+1.40->+1.40 | -6.18->-6.45->-6.31 | Y/./Y | Y/./Y | Y/./Y | +0.94/+0.51 | 0.69->0.61->0.62 |
| long_form | C4_longformer | 10 | 7933->7548->7547 | -2.95->-3.64->-3.62 | +10.32->+10.26->+10.21 | ././. | ././. | ././. | -0.43/+0.73 | 0.96->1.19->1.19 |
| long_form | C4_longformer | 20 | 7902->7526->7525 | +0.91->+0.63->+0.62 | -4.38->-4.02->-3.91 | Y/./. | Y/./. | Y/./. | +2.12/+1.94 | 0.63->0.46->0.46 |
| long_form | C6_llmtext | 5 | 7951->7565->7564 | +1.79->+2.09->+2.04 | -6.31->-7.28->-7.20 | Y/./Y | Y/./Y | Y/./Y | +0.76/+0.29 | 1.11->0.96->0.95 |
| long_form | C6_llmtext | 10 | 7933->7548->7547 | +2.25->+2.54->+2.50 | -7.92->-8.37->-8.24 | Y/./Y | Y/./Y | Y/./Y | -1.63/-2.19 | 1.07->1.16->1.16 |
| long_form | C6_llmtext | 20 | 7902->7526->7525 | +0.27->+0.47->+0.47 | -3.23->-3.41->-3.44 | Y/Y/. | Y/Y/. | Y/Y/. | -0.85/-0.82 | 0.37->0.55->0.54 |
| long_form | D1_concat_mlp | 5 | 7951->7565->7564 | -1.04->-1.16->-0.98 | +2.66->+2.78->+2.41 | ././. | ././. | ././. | +0.64/+0.48 | 1.21->1.35->1.33 |
| long_form | D1_concat_mlp | 10 | 7933->7548->7547 | -0.46->-0.79->-0.35 | +3.54->+3.66->+2.80 | ././. | ././. | ././. | +0.29/+0.12 | 0.62->1.01->0.78 |
| long_form | D1_concat_mlp | 20 | 7902->7526->7525 | +0.12->-0.46->+0.34 | -5.57->+5.50->-4.51 | Y/./Y | ././. | Y/./. | -1.95/-1.29 | 0.09->0.31->0.31 |
| long_form | D2_gated_fusion | 5 | 7951->7565->7564 | +0.19->+0.16->+0.15 | -2.11->-2.18->-1.80 | ././. | ././. | ././. | +0.56/+0.60 | 0.29->0.27->0.32 |
| long_form | D2_gated_fusion | 10 | 7933->7548->7547 | -0.02->+0.15->-0.17 | +2.02->-1.98->+2.41 | ././. | ././. | ././. | +1.04/+0.59 | 0.03->0.20->0.18 |
| long_form | D2_gated_fusion | 20 | 7902->7526->7525 | -2.27->-2.81->-2.42 | +5.56->+5.56->+5.27 | ././. | ././. | ././. | -1.14/-0.71 | 3.04->3.56->3.53 |
| long_form | D4_llmfused | 5 | 7951->7565->7564 | +0.15->+0.15->-0.03 | -0.75->-0.63->+0.60 | ././. | ././. | ././. | +0.61/+0.57 | 0.32->0.32->0.06 |
| long_form | D4_llmfused | 10 | 7933->7548->7547 | -0.15->-0.06->-0.14 | -0.43->-0.71->-0.59 | ././. | ././. | ././. | +1.08/+0.69 | 0.84->0.53->0.86 |
| long_form | D4_llmfused | 20 | 7902->7526->7525 | +0.60->+0.07->+0.18 | -3.58->-3.72->-3.62 | Y/./. | Y/./. | Y/./Y | -2.40/-2.22 | 0.73->0.08->0.22 |

Primary-stage per-cell detection flips: A->B 3/69, B->C 1/69 (composition detail above; verdict objects in the decomposition table).

## Refetch-drift disclosure (prereg G3; committed label_parity vs this refetch)

| quantity | committed (2026-07-09 fetch) | this refetch | drift | flag threshold |
|---|---|---|---|---|
| clean row coverage | 80.19% | 79.83% | -0.353pp | 0.5pp (ok) |
| Pearson log-RV (covered modelled rows) | 0.998148 | 0.998141 | -0.000008 | 0.001 (ok) |
| tickers with Yahoo data | 648 | 645 | -3 | — |
| symbol-mismatch screened | 18 | 18 | added: none; dropped: none | — |
| parity n (covered modelled rows) | 343,239 | 341,728 | -1,511 | — |

## Disclosures

1. **Frozen text predictions (LIMITATION-FEEDER).** Every text arm was trained and tuned against the CRSP close-to-close RV target on the FULL panel; predictions are reused frozen and only the log-space recalibration/combiner weights are refit per panel. Panel-C readings are therefore conservative for the text side.
2. **Yahoo terms bar redistribution.** The release artifact is the rebuild pipeline + fetch script (scripts/analysis/public_variant_labels.py), never the data; the price cache lives only in the session scratchpad. The symbol-mismatch screen itself needs CRSP — a licence-free builder cannot run it (inherited label_parity caveat; part of the variant's honest labelling).
3. **A-block treatment (prereg).** Panel C: A2_har_rv refit (committed HARRV class: train split, per horizon, log OLS + Duan smearing) on public pub_1d/5d/22d + public label; A6_shar refit (stronger_baselines conventions incl. the BPQ insanity filter) with RS-/RS+ rebuilt from public signed daily returns at the feature-window end; A3/A4/A5 are label-free return-based forecasters — frozen, recalibrated on val inside the committed combination machinery. Panel B freezes ALL stored forecasts and only filters rows. The firm-identity reference term is the firm's own VAL-split mean of the panel's OWN label.
4. **Row losses (counted, reconciled to the coverage table).** Panel B: long_form A2 panel rows filtered (not covered): 18347/94237; long_form 5-price panel rows filtered (not covered): 16693/85195; event_driven A2 panel rows filtered (not covered): 67354/333192; event_driven 5-price panel rows filtered (not covered): 62085/305014. Panel C: long_form A2 refit rows lost (label/feat): 18362/94237; long_form A6_shar refit rows lost (label/feat/RS): 16706/85195; long_form 5-price panel rows lost: 16706/85195; event_driven A2 refit rows lost (label/feat): 67482/333192; event_driven A6_shar refit rows lost (label/feat/RS): 62203/305014; event_driven 5-price panel rows lost: 62203/305014. The label-verification gate is covered-rows-only on the public side: rows without a clean public label are counted (never scored); the CRSP-side reconstruction is machine-precision on ALL rows (labels build, gate L1).
5. **Extra output file disclosed:** results/tables/public_variant_leaderboard.csv (per-comparison leaderboard detail for the three panels' 180-comparison universe with flip flags).
6. **Single-shot:** this table was written once; the script refuses to overwrite it. Same seeds, same placebo permutations, same Holm families as the committed cascade.

## SANITY

| gate | result |
|---|---|
| G1 cascade path reproduces committed tables (panel A, original labels) | **PASS** — max abs diff 1.78e-15 over primary/firm/pool/MDE/placebo; counts {'primary_holm': [38, 38], 'firm_holm': [8, 8], 'pool_holm': [9, 9], 'conjunction': [0, 0], 'genuine': [38, 38]}; injection counts match: True |
| G1 refit machinery reproduces stored A2/A6_shar runs | max abs diff 3.55e-15 (tol 1e-06) |
| G1 leaderboard reproduces variance_unit_standalone180.csv (panel A) | max abs diff 1.78e-15; verdict mismatches 0/180 — **PASS** |
| L1 CRSP label reconstruction (labels build) | n=431245, unreconstructed=0, max abs diff 0.0e+00 — **PASS** |
| L1b feature windows (CRSP side, TickerSeries machinery) | max abs diff 6.2e-13 (tol 1e-8) — **PASS** |
| L1c one-mask consistency + L2 A2-QLIKE anchor | PASS (exact); PASS (aborts otherwise; label_parity.gate2_qlike) |
| G2 public-label parity (gate >= 0.99) | Pearson(logRV)=0.998141 on 341,728 covered modelled rows — **PASS** |
| G3 coverage reconciliation | parquet covered rows == coverage-table covered rows (assert); drift table above; FLAGGED=False |
| G4 placebo (panel B) | |placebo DM|<2 in 64/69 cells (committed convention) |
| G4 placebo (panel C) | |placebo DM|<2 in 62/69 cells (committed convention) |
| Per-cell n_test totals | A 955,526 -> B 909,321 -> C 909,180 obs across 69 cells (max per-cell loss A->B 1231, B->C 4) |
