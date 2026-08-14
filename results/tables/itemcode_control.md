# Prereg IC — item-code / earnings-window control for the 8-K residual (Qwen3-32B, event_driven)

## Disclosures

- Object: C6_llmtext (Qwen3-32B, **single seed 2026** — prereg correction: C6 entered with near-deterministic single-seed decoding, no 3-seed ensemble), event_driven, 3 horizons.
- `has_202 = 1[item_subtype contains "2.02"]` (string contains; item_subtype is a comma-separated item list carried on predictions.parquet, 0% null). Dummies enter the log-space combiner **LINEARLY** (not log-transformed) in BOTH the restricted and unrestricted design matrices: R = [1, L(fh), L(fid), has202]; U = R + [L(ft)]. Val-fit, test-frozen, exp back-transform; QLIKE; day-clustered DM (HAC lag h-1, HLN).
- Holm within the pre-declared 3-test family (3 horizons x this one augmented reference).
- Secondary spec (REPORT ONLY, not in the decision): dummies for the top-8 individual items by (train+val)-row frequency, parsed from item_subtype by comma (membership, not substring), each entering linearly in both R and U. Frequencies counted on the A2_har_rv event_driven panel's train+val rows (258350 rows; C6 predictions carry no train rows); A2 and C6 item_subtype verified identical on every merged val/test key. Top-8: 9.01 (198257), 2.02 (75451), 8.01 (65381), 7.01 (59125), 5.02 (45774), 1.01 (25961), 5.07 (17958), 2.03 (14423).

## Table — residual over the firm-identity reference, plain vs augmented

rel% > 0 = text lowers QLIKE vs the reference; DM<0 = text helps; `**` = DM<0 & raw p<.05. The plain firmID columns are the sanity anchor (committed crossfamily_llm.csv values, reproduced).

| h | n_test | test has_202 frac | rel% firmID | DM | rel% firmID+has202 | DM | raw p | Holm(3) p | rel% firmID+top8 (secondary) | DM | raw p |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 | 25109 | 0.332 | +0.45%** | -5.26 | +0.21%** | -5.28 | 1.569e-07 | 3.138e-07 | +0.23%** | -5.37 | 1.002e-07 |
| 10 | 25001 | 0.333 | +0.25%** | -5.16 | +0.18%** | -5.53 | 4.15e-08 | 1.245e-07 | +0.19%** | -5.85 | 6.612e-09 |
| 20 | 24732 | 0.336 | +0.20%** | -3.79 | +0.20%** | -3.65 | 0.0002743 | 0.0002743 | +0.29%** | -3.93 | 8.987e-05 |

## VERDICT (pre-registered)

**NOT an earnings-window artefact.** The 8-K residual survives the has_202-augmented firm-identity reference in 3/3 horizons (DM<0 & Holm<.05, pre-declared Holm(3), day-clustered).

## SANITY

- ANCHOR PASS: the committed plain firm-identity numbers for qwen event_driven (crossfamily_llm.csv, columns ['n_test', 'rel_firm', 'dm_firm', 'p_firm']) reproduced on this code path to machine precision (rtol 1e-12) in 3/3 horizons.
- item_subtype: 0% null on the C6 side; A2 vs C6 identical on all 117407 merged val/test keys (1:1 merge verified).
