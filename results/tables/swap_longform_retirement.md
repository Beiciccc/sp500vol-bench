# E-lf (long-form matched swap) — retirement record (not a statistics table)

All three registered arms retired at their respective pre-registered gates, in every case before any E-lf statistic was produced; the pre-declared readout
(median retention) was never produced, and swap_longform.csv does not exist.

- **C2 FinBERT-S1: artifact loss** (v1.2, registered 2026-07-16, prior to any E-lf statistic): the three horizon
  checkpoints existed only on the box and were destroyed by a disk cleanup; a retrained stand-in would violate this section's zero-training principle.
- **B2 TF-IDF: recipe refit reproducibility gate fired** — max|Δpred|/|pred| = **1.402e+00** (35,635
  val+test rows, threshold 1e-8). The root cause is identical to that of the anon B2 retirement (anon_arm.md Disclosures): the committed
  June (env × cache) pair cannot be reconstructed + text-store lineage drift (with CV alpha entirely identical, the vocabulary sets differ,
  idf max|diff| 7.5).
- **C5 qwen3 frozen-embedding MLP head: recipe rebuild reproducibility gate fired** — max rel diff **1.062e-05**
  (94,237 rows, all splits, threshold 1e-8; v1.3 verbatim: "the 1e-8 reproducibility gate depends on same-model GPU determinism; if the gate fires then
  the C5 arm is truthfully retired"). The rebuilt head has been archived as forensic evidence (box ELF_swap_C5_qwen3.../rebuilt_heads; locally
  /path/to/data-root/sp500vol-data/forensics/elf_forensics.tgz).
- Evidence: logs/elf2_b2.log, logs/elf2_c5.log (inside elf_forensics.tgz), FAIL_elf2 sentinel
  (2026-07-16).
- The v1.0 reconciliation-paragraph obligation is discharged jointly by this record + the already committed ED matched-swap and C-anon tables:
  the reconciliation lies in the anon-vs-swap gap (identity share point estimate 0.51 vs swap-implied 0.71) — anonymization only
  strips identity strings, matched-swap destroys level-alignment, and the gap is exactly the difference between the two channels; no E-lf
  statistic is required.
- This file is consumed as the substitute prerequisite for anon_score.py --channel lf (prereg-ea v1.5).

---

**Correction (2026-07-27).** This record's interpretive phrase "swap-implied 0.71" follows the registered wording of its era and is superseded: the swap's own design (RV-matched partners preserve a level shortcut) makes the *retention* the identity share, 0.29, with 0.71 the alignment-dependent content share. No number above is altered. See the paper's Discussion and FACTS.md L495.
