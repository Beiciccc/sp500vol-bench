# C5x INPUT-PARITY — prompting vs same-lineage embeddings on identical excerpts (long_form, P0-3b)

## RESTATED vs BEFORE

| | BEFORE | RESTATED (this table) |
|---|---|---|
| citable table for the abstract's "prompting > same-lineage embeddings" sentence | **NONE** (C5x_qwen3exc run existed only in results/runs/ + config_fingerprints rows) | full M1 comparison committed here |
| basis discipline | - | C5_qwen3 uses the declared PRIMARY 3-seed per-observation ensemble (m1_ensemble_primary loader); C5x/C6 are single-run by design; seed2026 C5 row shown as a check |
| inference | - | day-clustered DM vs single recalibrated HAR, raw p AND Holm (one family, 12 cells), 5-seed label-shuffle placebo |

C5x_qwen3exc = Qwen3-Embedding-8B run on the BYTE-IDENTICAL curated excerpts fed to the C6 prompt (10-K 1A/7/7A else head-truncation; ridge head on log target with Duan smearing) — it isolates input curation from the prompting mechanism within the same Qwen3 lineage. Combiner weights are val-fit and frozen to test throughout; QLIKE in vol units; `genuine` = clustered DM<0, Holm<.05, |placebo DM|<2.

| model | h | seeds | n_test | n_days | QLIKE alone (vol) | QLIKE alone (var) | R^2 alone | pred sd | n_uniq(2dp) | QLIKE(R) | QLIKE(U) | rel% | g_log | DM(clu) | p raw | Holm | placebo DM | genuine |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| C5_qwen3 (ens, PRIMARY) | 5 | 2026+2027+2028 | 7951 | 809 | 0.1540 | 0.682 | -0.013 | 0.0240 | 15 | 0.1209 | 0.1221 | -1.02 | +0.572 | +2.64 | 0.0085 | 0.016 | +0.47 | no |
| C5_qwen3 (ens, PRIMARY) | 10 | 2026+2027+2028 | 7933 | 803 | 0.1207 | 0.596 | -0.007 | 0.0249 | 15 | 0.0873 | 0.0900 | -3.10 | +1.122 | +3.62 | 0.0003 | 0.001 | +0.45 | no |
| C5_qwen3 (ens, PRIMARY) | 20 | 2026+2027+2028 | 7902 | 794 | 0.0937 | 0.389 | -0.004 | 0.0249 | 15 | 0.0701 | 0.0746 | -6.39 | +1.795 | +5.38 | 0.0000 | 0.000 | +0.42 | no |
| C5_qwen3 (seed2026) | 5 | 2026 | 7951 | 809 | 0.1548 | 0.650 | -0.021 | 0.0250 | 16 | 0.1209 | 0.1221 | -1.03 | +0.573 | +2.66 | 0.0080 | 0.016 | +0.47 | no |
| C5_qwen3 (seed2026) | 10 | 2026 | 7933 | 803 | 0.1241 | 0.561 | -0.032 | 0.0261 | 17 | 0.0873 | 0.0900 | -3.13 | +1.145 | +3.70 | 0.0002 | 0.001 | +0.45 | no |
| C5_qwen3 (seed2026) | 20 | 2026 | 7902 | 794 | 0.0943 | 0.388 | -0.011 | 0.0255 | 16 | 0.0701 | 0.0747 | -6.65 | +1.789 | +5.45 | 0.0000 | 0.000 | +0.42 | no |
| C5x_qwen3exc (input-parity) | 5 | 2026 | 7951 | 809 | 0.1584 | 0.848 | -0.007 | 0.0422 | 32 | 0.1209 | 0.1213 | -0.32 | -0.027 | +5.66 | 0.0000 | 0.000 | +0.32 | no |
| C5x_qwen3exc (input-parity) | 10 | 2026 | 7933 | 803 | 0.1288 | 0.781 | -0.020 | 0.0409 | 29 | 0.0873 | 0.0906 | -3.74 | -0.155 | +8.60 | 0.0000 | 0.000 | -0.58 | no |
| C5x_qwen3exc (input-parity) | 20 | 2026 | 7902 | 794 | 0.0972 | 0.498 | +0.036 | 0.0540 | 38 | 0.0701 | 0.0730 | -4.12 | -0.101 | +9.31 | 0.0000 | 0.000 | +1.58 | no |
| C6_llmtext (prompted) | 5 | 2026 | 7951 | 809 | 0.2072 | 1.226 | -0.174 | 0.0274 | 8 | 0.1209 | 0.1187 | +1.79 | +0.254 | -6.31 | 0.0000 | 0.000 | +0.83 | YES |
| C6_llmtext (prompted) | 10 | 2026 | 7933 | 803 | 0.1778 | 1.095 | -0.201 | 0.0171 | 13 | 0.0873 | 0.0853 | +2.25 | +0.333 | -7.92 | 0.0000 | 0.000 | -1.45 | YES |
| C6_llmtext (prompted) | 20 | 2026 | 7902 | 794 | 0.1601 | 0.926 | -0.345 | 0.0239 | 16 | 0.0701 | 0.0699 | +0.27 | +0.078 | -3.23 | 0.0013 | 0.004 | -1.44 | YES |

## Reading

- **Prompting (C6_llmtext)**: M1 increment +0.27% to +2.25% across horizons, genuine in 3/3 cells.
- **Same-lineage embeddings on the SAME excerpts (C5x_qwen3exc)**: -4.12% to -0.32%, genuine in 0/3 cells.
- **Standard-input embeddings (C5_qwen3, primary ensemble)**: -6.39% to -1.02%, genuine in 0/3 cells.
- Input parity means any C6-vs-C5x gap is attributable to the elicitation mechanism (prompting vs embedding pooling), not to excerpt curation. This is the citable basis for the abstract's prompting-vs-embedding sentence; cite the exact rel% and Holm p from this table, and keep the claim limited to long_form (the only C5x cell run).
- Caveat for the paper: this compares mechanisms at parity of INPUT, not of parameter count (32B decoder vs 8B embedder + ridge); say "same-lineage" not "same-size".
