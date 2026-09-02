# Control-intersection table — headline: **0/69 cells clear maximal + firm jointly under Holm**

Under Holm, **0/69** cells clear the full AND (clustered-genuine AND maximal AND firm-identity). Under raw p<.05, **4** scattered weak cells clear it — with no coherent model/disclosure pattern.

## Counts: per-control marginals and intersections (raw p<.05 vs Holm<.05)

| Control / intersection | raw p<.05 | Holm<.05 |
|---|---|---|
| clustered-genuine (m1_clustered) | 39 | 29 |
| maximal price pool | 23 | 8 |
| firm-identity | 14 | 8 |
| within-date cross-sectional [extra] | 33 | 33 |
| seed-ensemble genuine [extra] | 29 | 29 |
| INTERSECTION: maximal AND firm | 4 | 0 |
| INTERSECTION: clustered-genuine AND maximal AND firm (full) | 4 | 0 |
| INTERSECTION: 5-way (+within-date x-sec +seed-ensemble) | 1 | 0 |

_Extra controls (within-date cross-sectional, seed-ensemble) are single-basis; their two columns repeat._

## Full AND survivors

**Holm basis: 0/69.**
No cell survives clustered-genuine AND maximal AND firm-identity jointly under Holm.

**Raw p<.05 basis: 4/69** scattered weak cells:

| cell (disc/model/h) | rel% maximal | rel% firm |
|---|---|---|
| long_form/B3_lm_linear/h=10 | 0.641 | 0.261 |
| long_form/C6_llmtext/h=20 | 0.233 | 0.079 |
| long_form/D4_llmfused/h=20 | 0.365 | 0.083 |
| event_driven/B4_lm_features/h=10 | 0.084 | 0.130 |

These span disciplines ['event_driven', 'long_form'], models ['B3_lm_linear', 'B4_lm_features', 'C6_llmtext', 'D4_llmfused'], horizons ['10', '20'] — scattered across cells with **no coherent model/disclosure pattern** (no single model or disclosure type dominates; they do not cluster on any horizon).

## Disjoint survivor sets (Holm basis) — substantiates "disjoint"

**Maximal survivors (Holm, n=8):** long_form/B2_tfidf_ridge/h=10, long_form/B2_tfidf_ridge/h=20, long_form/B2_tfidf_ridge/h=5, long_form/C2_finbert_s4/h=10, long_form/C2_finbert_s4/h=20, long_form/C4_longformer/h=5, long_form/C6_llmtext/h=10, long_form/C6_llmtext/h=5

**Firm-identity survivors (Holm, n=8):** event_driven/B1_bow_ridge/h=10, event_driven/B1_bow_ridge/h=5, event_driven/B2_tfidf_ridge/h=10, event_driven/C6_llmtext/h=10, event_driven/C6_llmtext/h=20, event_driven/C6_llmtext/h=5, long_form/B3_lm_linear/h=10, long_form/C6_llmtext/h=20

**Overlap (cells in BOTH):** 0 — **EMPTY (the two survivor sets are disjoint)**

## Sanity — per-control marginals match source tables

| check | value |
|---|---|
| m1_clustered genuine (Holm) == 29 | 29 |
| m1_clustered genuine (raw) | 39 |
| maximal Holm == 8 | 8 |
| maximal raw | 23 |
| firm Holm == 8 | 8 |
| firm raw | 14 |
| within-date cross-sectional == 33 | 33 |
| seed-ensemble genuine == 29 | 29 |
