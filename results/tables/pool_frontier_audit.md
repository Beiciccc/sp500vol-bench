# B4 — Price-frontier completeness audit (REAL)

**Charge (R6 quant_finance #8):** the "maximal" price pool has five members (HAR, SHAR,
GARCH, EGARCH, ARIMA) while HARQ and HAR-X(VIX) are archived, computed, and absent —
HARQ especially, since correcting RV measurement error is exactly what a daily
close-to-close proxy needs. The paper dismissed HARQ in one parenthetical
("log-space-unstable, disclosed").

**Why this needed measuring, not asserting.** A stronger pool makes text's job harder
(conservative). An *unstable* member makes a val-fitted pool worse out of sample, which
would credit text for the pool's own breakage — reverse reference-shopping.

## Pool quality (val-fit, test-frozen, day-clustered DM; 6 panels)

| pool | mean test QLIKE | panels clustered-sig. better than pool5 | max abs log weight |
|---|---|---|---|
| pool6 +HAR-X | **0.0887** | **4 of 6** | 1.81 |
| pool7 (all) | 0.0892 | 3 of 6 | 1.81 |
| pool5 (paper) | 0.0916 | — | 1.82 |
| pool6 +HARQ | 0.0918 | 3 of 6 | 1.82 |

**No weight explosion anywhere** (max |log weight| ~1.8 in every pool, including the ones
containing HARQ). The paper's "log-space-unstable" dismissal of HARQ is *not evidenced*
in the pool context: HARQ is mixed on value (better in 3 of 6 panels, mean QLIKE a hair
worse), not unstable. HAR-X(VIX) is simply better and belongs in the pool.

## What it does to the cascade (69-cell grid, same basis for both)

| reference | raw | Holm |
|---|---|---|
| pool5 (paper's five) | 23 of 69 | **8 of 69** |
| pool7 (all seven) | 17 of 69 | **3 of 69** |

**Completing the frontier strengthens the null**: Holm survivors 8 -> 3.

*Basis note:* this recomputation is on the seed-2026 basis (`fc.load` default), where the
paper's committed pool5 count is 9 of 69 on the seed-ensemble primary. The pool5 -> pool7
comparison is internally consistent (same basis, same code path), so the delta 8 -> 3 is
the valid claim; the headline restatement must be recomputed on the ensemble basis before
it enters the paper.

Sources: `pool_frontier_audit.csv` (per-panel pool quality + fitted weights),
`pool_frontier_cascade.csv` (per-cell 69-grid, both references).
