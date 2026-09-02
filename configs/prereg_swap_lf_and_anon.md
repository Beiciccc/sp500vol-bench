# Pre-registration: E-lf long-form matched-firm swap + C-anon entity-anonymisation arm (prereg-ea-v1.0)

Date: 2026-07-15. Committed and tagged prior to any statistic from either analysis. Commitments (family convention): all branch
results enter the paper regardless of direction; single-shot discipline; revisions precede the corresponding statistics and are recorded; no results-conditional adjustment of specification.

## E-lf — long-form matched-firm swap (the direct answer to ml rank-1)

**Motivation (verbatim from the internal review simulation)**: the tension between the current draft's swap evidence (ED: kills 84–93% of the residual = content;
median retention 31% across the 38 HAR-genuine cells) and the title's identity narrative is unreconciled; "a critic who wants to kill this
paper will use the authors' own swap numbers to do it". This experiment extends the swap to **per-model readouts on the
long-form panel**, answering directly: how much of the long-form HAR-genuine increment survives after correspondence is destroyed.

**Design**: the mechanism reuses committed `scripts/analysis/matched_firm_swap.py` verbatim (within-day validation-period
RV matching, level preserved, correspondence destroyed; matching rule, random seed 2026 and number of redraws exactly as in the committed
version). New: after the long-form documents are swapped, re-inference (no retraining) on the **frozen** long-form models: C2 FinBERT-S1,
C5 (frozen embed + fitted head), B2 TF-IDF (CPU).
[v1.2 revision (2026-07-16, prior to any E-lf statistic): C2's three horizon checkpoints are physically lost
(they existed only on the box and were removed by a disk clean-up; the .pt files were never held locally). Retraining a stand-in would violate this section's zero-training principle, so the **C2 arm is pre-registered
as "artefact lost, not executed"**, the readout coverage is downgraded to the two arms B2 + C5 and this is disclosed truthfully in the paper (the deep fine-tuning arm is absent);
if the file for C5's fitted regression head is missing, it is deterministically rebuilt on the original train split following the recipe and put through a 1e-8
reproduction gate against the committed predictions (the same mechanism as B2) -- a rebuild is a reproduction, not a training degree of freedom. [v1.3 one-word correction (prior to any E-lf statistic,
following the v1.1 typo precedent): the first draft wrongly wrote "ridge head"; the committed C5 recipe is in fact an MLP VolatilityHead (hidden 128,
AdamW 1e-4, early stopping patience 3, env.json on file). The mechanism is unchanged; the 1e-8 reproduction gate depends on same-model GPU determinism,
and if the gate fires the C5 arm is retired truthfully.]] Every long-form HAR-genuine cell reports
retention = post-swap increment / original increment, day-clustered DM.
**Pre-declared readouts**: median and quartiles of retention over the long-form genuine cells; tabulated alongside the committed ED / all-cell readouts.
**Branches**: (a) retention low (median <50%) → the long-form increment is mainly content (consistent with the ED residual),
the "who spoke" wording is limited by the existing bracket framing, and the swap tension is explicitly reconciled in one paragraph of the main text;
(b) retention high (median ≥50%) → the level channel dominates and the identity narrative gets direct support;
(c) mixed → reported cell by cell, truthfully. **All three branches must be written into the "reconciliation paragraph" of the main text** -- the purpose of this experiment is to give that paragraph
numbers to write, whatever the direction.
**Gates**: G1 machine-precision reproduction of the committed matched_firm_swap table (same code path); G2 the level-preservation
assertion on the post-swap inputs (the paired RV difference distribution is identical to committed); G3 the no-retraining assertion (checkpoint hashes unchanged).

## C-anon — entity-anonymisation arm (bound → estimate)

**Motivation**: ac/ml -- "the central quantity of the title (identity share) is never point-estimated; the reference interval only gives bounds". The
masked-unmasked increment difference = a direct point estimate of the identity share, complementary to the matched-swap (the swap destroys
correspondence and preserves level; anon destroys identity cues and preserves all content).

**Design**: NER masking (company names / tickers / executive personal names / product names / CIK; spaCy en_core_web_lg + a rule table,
masking rate and samples disclosed) applied to the **event-driven panel** (the channel where the residual lives; long-form as a stretch, only if ED completes).
Three arms are re-run: C6 prompted (masked excerpts, verbatim the same protocol and the same specification as the original C6 -- **v1.1 revision: the actual specification of committed
C6 is Qwen3-32B bf16, TP=2 (config.json on file); the first draft's "AWQ single card" was a typo; bit-for-bit G1 comparison
requires the committed weights by construction, and quantisation differences would contaminate the masked/unmasked ratio, so both masked and control use
bf16 TP=2**; single seed 2026),
C2 FinBERT-S1 (masked retraining, rung-2 recipe = fixed recipe, seed 2026, the same method as the original arm),
B2 TF-IDF (CPU). Each arm vs the two references (single recalibrated HAR / firm-identity) M1 increment, day-clustered DM,
with a pre-declared Holm(6) per arm (3 horizons × 2 refs).
**Pre-declared point estimate**: identity share^anon = 1 − (masked increment / unmasked increment), cell by cell + aggregated;
triangulated against the reference-interval bound and the swap retention.
**Branches**: (a) masked increment ≈ 0 (share→1) → identity dominates and the title gets point-estimate support;
(b) most of the masked increment survives and is still absorbed by the firmID reference → a "genuinely firm-stable content" narrative,
and the title wording is softened (the fix costs us points; committed to executing it); (c) the masked increment survives and is **no longer** absorbed by firmID →
the masking itself has broken the alignment of the firm-stable channel -- reported truthfully as a methodological finding.
**Gates**: G1 the unmasked control re-run is bit-for-bit identical to the committed predictions (pipeline invariance); G2 masking-quality spot check
(100 documents checked by hand against the rules, miss rate disclosed); G3 the excerpt builder's truncation statistics on masked text are comparable to the original.
**Scope disclosure**: 2 GPU box; ED first; long-form a stretch; C5x not in this round (GPU budget, pre-registered as not done).
**v1.1 pre-scoring operationalisation addendum (prior to any anon statistic)**: branch thresholds share median ≥0.75 → (a),
≤0.50 and the masked increment still absorbed by firmID → (b), otherwise (c) / mixed cell by cell; aggregation = median (consistent with the swap
retention convention); cells with a non-positive unmasked increment have share recorded as n/a and are not dropped; G1 is reported for GPU arms as
exact-match rate + max|diff|, and anything short of bit-for-bit identity must be explicitly recorded with --record-g1-deviation before it may enter scoring
(bf16 batched inference is not bit-deterministic, REVIEW_BLINDSPOTS on file).

## v1.4 revision (2026-07-16, prior to any long-form-anon statistic): LF stretch operationalisation — B2-only

v1.0's execution gate "long-form as a stretch, only if ED completes" **is hereby explicitly revised** into a scoring-order gate: the ED anon scoring table
(`anon_arm.{csv,md}`) must land on disk before any LF statistic; LF mask construction and the running of the two B2-lf arms may proceed in parallel with ED C6
inference -- because this section seals off every LF decision degree of freedom in advance and reads no ED anon statistic (ED scoring
has not yet fired, the anon_arm table does not yet exist; the *_smoke products currently in results/anon/ are non-scoring engineering artefacts). v1.4
is committed + tagged before any non-smoke LF run. The decisions below are **based solely on committed table numbers**.

**Arm set = the single arm B2-lf** (TF-IDF ridge, long-form; masked = retrained with the fixed recipe on the anonymised store, seed 2026,
the same estimator as the ED B2 arm). **Estimation target**: the identity share of the committed main-table B2-lf increment —
vol_rel_impr_pct = +3.3305/+3.4823/+5.9199% (h=5/10/20, m1_ensemble_primary; B2 single seed,
i.e. the scoring denominator) — of which h=20 is the headline cell behind the main text's "up to 5.9%".

**Contraction of the share estimation surface (constructive, citing committed numbers)**: B2-lf's unmasked increment against the firm-identity
reference is −0.615/−3.892/−8.089% (h=5/10/20, committed firm_identity_control.csv) -- all non-positive,
so by the v1.1 n/a rule the 3 firmID-side cells' share is **n/a prior to execution**; share median = **the median over the defined
cells (the 3 HAR cells)** (with 3 cells that is the h=10 cell; with an even number of cells take the mean of the middle two values -- registered generally).
**Domain clarification of the v1.1 n/a rule (the same meaning in both channels, both prior to scoring)**: n/a acts only on share, not on the tests --
an **executed** cell with share=n/a still keeps its day-clustered DM and stays inside that arm's Holm(6) family (3 horizons ×
2 references, v1.0 original text); only arms with status=not-executed (C2/C5/C6-lf) carry no test.

**Quantitative operationalisation of the branches (used in both channels, both prior to either table firing; the scoring script matches this text verbatim and is committed with this tag)**:
- **(a)** ⇔ the median share over defined cells is ≥0.75 **and** the masked arm is Holm-significantly positive against the HAR reference in 0/3 cells;
- **(b)** ⇔ the median is ≤0.50 **and** the masked increment is still absorbed by firmID, absorption = the masked arm is Holm-significantly positive against the firmID
  reference in ≤1/3 cells;
- **(c)** ⇔ everything else (including a median falling in (0.50,0.75), or firmID Holm significance ≥2/3 = absorption broken), mixed, cell by cell.
- LF degeneracy registered truthfully: B2-lf's **unmasked** increment is already fully absorbed by firmID (committed numbers cited above),
  so (b)'s absorption clause is expected to be true for LF and has weak discriminating power -- the LF (b)/(c) decision in fact rests on the median and the
  firmID-Holm count, and the paper's wording is weakened accordingly.

**Exclusions (all prior to any statistic)**:
- **C2-lf**: the artefact-lost precedent (v1.2) + a marginal retraining cost of ~20–30 GPU-h, not executed.
- **C5-lf (HAR side): constructive n/a**. Committed m1_multiseed.csv, long-form C5_qwen3 seed-2026
  rel_impr_pct = −1.0347/−3.1346/−6.6467 (primary citation, on the same basis as the single-seed specification at scoring time); deployable_combiner
  FIXED mean rel% = −0.85/−2.48/−5.97 corroborates (3-seed basis). All negative → the HAR-side share cells are empty prior to execution.
- **C5-lf (firmID side)**: that increment is not in the committed tables → constructive n/a is not invoked, and it is excluded on GPU
  budget / scope like C6-lf; and whether or not it is executed cannot change the branch decision (the median is taken over HAR cells only). The code path is kept, not executed.
- **C6-lf: premise correction + budget exclusion**. The earlier internal working assumption that "C6 was never run on long-form" **is wrong** -- committed
  `C6_llmtext_full_long_form_seed2026` really exists (11,907/11,907 long-form documents, full coverage, parse_fail 0).
  So the C6-lf masked arm is **well defined** and cheaper than ED C6 (11,907 vs 39,322 documents); the reason for exclusion can only be, and is only,
  GPU budget and scope (the ED channel is where the residual lives), and it must not be written up as "infeasible".
- No text-model recipe is retrained (the B2 fixed-recipe retraining is the same method as ED, not tuning).

**Products and ordering**: sister table `results/tables/anon_arm_lf.{csv,md}`, write-once, in a separate file from the ED table
(the single-shot guard operates per file; the ED table fires first). **Triangulation column** (operationalised): = the document-swap retention of the E-lf **B2 arm** at the same
horizon, taken from committed `swap_longform.csv`; if that cell is absent, or any of E-lf G1–G3
fails, it is recorded as n/a with the reason noted. Hard precondition for LF scoring: both `swap_longform.csv` and `anon_arm.csv` already exist.
**Gate mapping**: **G1 (CPU arm, no deviation escape hatch)** = the control reproduction passes the **1e-8 full-panel reproduction gate**
(following the CPU convention already registered in E-lf, "the same mechanism as B2"; v1.1's exact-match rate + --record-g1-deviation
procedure applies to GPU arms only) -- anything above 1e-8 is a pipeline-invariance failure and **the arm is retired truthfully** (reported as G1-fail, no share estimate);
the deviation-recording route must not be taken. G2 = mask_stats_lf.json + a 100-document audit spot check under the same protocol as ED. G3 (C6 truncation
statistics) is n/a with no C6-lf arm, stated in the md.
**Engineering registration**: anon_mask_build.py gains --panel lf (outputs renamed *_lf) and --batch-docs atomic
sharded resumption -- pure engineering, no statistical effect; default 0 = the original single-shot path, and the path already run for ED is unaffected.

## v1.5 revision (2026-07-16, prior to the LF table landing on disk; E-lf has been retired in full at its own gates, and its statistics were never produced)

**Record of fact**: all three E-lf arms were retired under already-registered rules -- C2 artefact lost (v1.2); B2 rebuild reproduction gate
max rel diff 1.402e+00 (the gate of v1.2's "the same mechanism as B2"); C5 rebuild reproduction gate 1.062e-05 > 1e-8
(v1.3: "if the gate fires the C5 arm is retired truthfully"). swap_longform.csv will never exist; the E-lf pre-declared readouts cannot be produced.
Retirement record = `results/tables/swap_longform_retirement.md` (not a statistics table, gate-firing numbers + diagnostic chain).

**The reconciliation-paragraph obligation stands**: v1.0's "all three branches must be written into the reconciliation paragraph of the main text" is instead discharged by the retirement facts + the already-committed
ED matched-swap and C-anon tables (the gap between anon share 0.51 and the 0.71 implied by the ED swap = the decomposition of the nominal-identity
vs level-alignment channels) -- introducing no new statistic.

**Disposition of the LF table precondition**: v1.4's hard precondition "swap_longform.csv already exists" cannot be satisfied because of the retirement. Given that the LF table,
under the existing rulings, contains no statistic at all (1 row g1-fail + 3 rows not-executed, decision undefined, and the triangulation column was already
registered as "absent = n/a with the reason noted"), the precondition is revised to: "swap_longform.csv **or** swap_longform_retirement.md
exists"; the triangulation column's n/a reason = E-lf retired at its own gates. This revision changes no cell value in the table
(all of them are determined by already-committed gate records).

## v1.6 revision (2026-07-16, prior to any share confidence-interval statistic): day-block bootstrap CI for the identity share

**Motivation**: verbatim from the R12 skeptic ("Bootstrap confidence intervals on the per-horizon identity shares
(0.51/0.56/0.71)"). Pure uncertainty quantification, no new decision, no branch -- the branch decision is permanently locked by the anon_arm table (already fired),
and this analysis may not and cannot alter it.

**Design**: a **day-block bootstrap** on the committed ED anon predictions (ctrl/masked, six runs, commit a721b8b):
resampling unit = effective trading day (the same unit as day-clustered DM), each cell resampling days with replacement within its own test panel,
B = 2000, seed 2026; combiner weights follow committed (fitted on val, frozen on test) and are **not refitted**;
each draw recomputes the unmasked/masked M1 increments (the same machinery as anon_score) and share = 1 − masked/unmasked.
**Registered readouts**: the **percentile 95% CI** for the 3 C6 HAR-side cell shares, the 3 C2 HAR-side cell shares, the median of each arm, and the pooled median over the six cells;
if in a given draw the unmasked increment is ≤0 → that draw's share is undefined (following the v1.1 n/a rule),
the CI is taken over the defined draws and the **undefined proportion is disclosed** (for cells where that proportion is >20%, the CI is annotated as unstable).
**Products**: `results/tables/anon_share_ci.{csv,md}`, write-once, single-shot. **Prose rule**: wherever the paper cites a point estimate,
attach the CI (the abstract may be exempted subject to space); the CI changes no locked branch wording.

## Boundaries

Neither analysis touches test-label selection; E-lf trains nothing; C-anon's FinBERT retraining uses a fixed recipe (not tuning).
With prereg-cd (D running) it shares the box but not the cards: D takes the CPU cores, E-lf/C-anon take the GPU.
