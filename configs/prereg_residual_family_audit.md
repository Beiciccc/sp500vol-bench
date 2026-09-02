# Pre-registration: family-robustness audit of the 8-K residual + item-code control + omnibus test (R9 queue B/D)

Date: 2026-07-15. This document was submitted and tagged (`prereg-rfa-v1.0`) **before** any of the statistics below were computed.
Motivation: 4/4 rank-3 concern (the only positive result is stuck on the p=.049 borderline cell); the quantitative objection:
"the paper catches the literature's shortcut and misses its own" (the cost of not running the item-code control).
Commitment: **the results of all branches go into the paper regardless of direction; the specification must not be chosen conditional on results.**

## Frozen inputs (already on disk, not read by any of the tests below at the time this document was submitted)

- `results/runs/C6_llmtext_llama70_full_event_driven_seed2026/predictions.parquet` (single seed, already in the committed table)
- `results/runs/C6_llmtext_llama70_s2027_full_event_driven_seed2026/predictions.parquet`
- `results/runs/C6_llmtext_llama70_s2028_full_event_driven_seed2026/predictions.parquet`
- `results/runs/C6_llmtext_llama70ens_full_event_driven_seed2026/predictions.parquet` (3-seed ensemble)
- Control anchor: `results/tables/crossfamily_llama70.csv` (committed, single-seed Holm(6) readings)

## B0 -- Llama-70B three-seed ensemble rescoring (zero GPU)

**Test batch**: verbatim identical to the M1 block of `scripts/analysis/crossfamily_llama70.py` -- log-space combiner
fitted on val, test frozen; references (a) single recalibrated HAR (A2), (b) firm-identity augmented reference (val-window firm mean spec);
day-clustered DM, HAC lag h−1, HLN correction. **Holm is applied to the new pre-declared 6-test family**
(the ensemble's 3 horizons × 2 references), parallel to the single-seed Holm(6), not merged.

**Sanity gates (any failure aborts, no table produced)**:
- G1′: the single-seed llama70 rows are recomputed through the same code path and agree with the committed `crossfamily_llama70.csv` to machine precision;
- G5 (v1.2 revision): the ensemble predictions equal, row by row, the **arithmetic mean** of the three seeds' predictions (`mean(pred_seed)`, rtol 1e-6);
  if they are not equal, inspect the ensemble's generating script first before deciding, and **do not** continue on "approximately holds".
  [Revision record 2026-07-15: v1.1 wrongly wrote log-space mean (mis-copied from the HPO seed_validation spec); after the gate
  aborted as prescribed, the generating script was checked, confirming that the frozen artifact's spec is the arithmetic mean -- row15 launch.sh deliberately aligns with the paper's
  seed-ensemble primary `m1_ensemble_primary.ensemble_text` convention and discloses it in config.json.
  For cross-family comparison against the C model primary baseline, the arithmetic mean is the consistent spec. The revision happened before any
  M1/Holm statistic on any ensemble row was computed (G5 runs before the M1 block; nothing downstream was touched at the abort).]

**Pre-declared decision ladder (ensemble rows)** -- reusing the current script's graded wording, thresholds unchanged:
- REPLICATES: vs the firmID reference, 3/3 horizons Holm<.05 and DM<0;
- DIRECTIONALLY REPLICATES: 3/3 rel_firm>0 and (≥2/3 raw p<.05 firmID or ≥1/3 Holm vs HAR);
- DOES NOT REPLICATE: 0/3 positive signs and 0/3 raw significant;
- otherwise: PARTIAL/MIXED, reported as the numbers stand.
Single-seed rows are **kept** in the table, with the ensemble rows alongside; replacing the single-seed rows with the ensemble rows and restating history is forbidden.

## B1 -- Third family (this document is to be revised after the model is verified, then launched)

Family set F = {Qwen3-32B (primary, single seed, disclosed), Llama-3.1-70B-AWQ (3-seed ensemble), third family (3-seed ensemble)}.
The third family must not be of the Qwen/Llama/Yi/Phi lineage (Qwen2.5-72B is of the same stock as the primary, **not eligible**;
Llama-3.3 is of the same family as the replication arm, **not eligible**).

**Third family selected (revised after verification, 2026-07-15, before any predictions were generated)**:
`mistralai/Mistral-Small-24B-Instruct-2501` (Mistral lineage; apache-2.0, not gated, directly downloadable from the mirror).
Serving plan: vLLM offline batch, bf16, TP2, `--max-model-len 8192`, with the
prompt / guided-JSON / clip[0.03,3.0] / retry protocol verbatim identical to C6/llama70 (`scripts/experiments/e1_llm_forecast`).
**The semantics of "3 seeds" are exactly the same as in the llama70 replication arm**: at temperature=0, seeds 2026/2027/2028 differ only through
kernel non-determinism (reproducibility-jitter ensemble, disclosed in launch.sh); the paper's wording follows
that spec, and it must not be called a random-decoding ensemble. Fallback (used only if the Mistral download/load fails, requires a further revision):
`google/gemma-2-27b-it` (gated=manual, requires an already-approved token; the chat template has no system role, needs a shim).
The test batch is exactly the same as B0; each family gets its own pre-declared Holm(6).

**Cross-family claim rules (pre-declared)**:
- Family STRONG pass: ≥2/3 horizons Holm<.05 and DM<0 vs the single recalibrated HAR (within-family Holm(6));
- Family WEAK pass: reaches DIRECTIONALLY REPLICATES or above on the B0 ladder;
- Paper wording: ≥2/3 families STRONG -> "replicates across families";
  ≥2/3 families ≥WEAK (including the primary) -> "sign-robust across families, significance attenuated";
  otherwise -> "does not replicate beyond the primary family" (the residual paragraph is downgraded accordingly;
  by the rules established in FACTS.md, it must not be written as a family-specific proof).

## IC -- item-code / earnings-window control (zero GPU; data = `item_subtype` carried in predictions.parquet, 0% missing)

**Primary spec**: add one binary term to the log-space combiner of the firm-identity reference,
`has_202 = 1[item_subtype contains "2.02"]` (the log transform `L(x)=log(clip(x+ε))` does not apply to a dummy --
the dummy **enters the design matrix linearly and directly**, with no log). Fitted on val, test frozen, same method as the current combiner.
**Secondary spec** (reported only, not part of the decision): add binary indicators for the top-8 items by train+val frequency.
Target: C6_llmtext (Qwen3-32B, **single seed** -- C6 enters with near-deterministic decoding at a single seed, disclosed in the paper,
with no 3-seed ensemble; this line was corrected before any IC statistic was computed, reason for the correction: the submitted draft wrongly wrote ensemble),
event-driven, 3 horizons. **Pre-declared Holm(3) family** (3 horizons × 1 augmented reference).

**Decision**:
- Under the firmID+has_202 reference the residual has ≥2/3 horizons DM<0 and Holm<.05 -> "not an earnings-window artefact";
- otherwise -> the paper is rewritten: the 8-K residual is (in part) an earnings-window effect, the residual paragraph is downgraded, and the abstract's
  "what survives" sentence is weakened accordingly. **Both branches are committed to the main text.**

## D -- cross-cell omnibus joint test + power calibration (zero GPU)

- Statistic: the daily loss differences of the 69-cell primary family (seed-ensemble baseline, vs the single recalibrated HAR)
  (QLIKE(f_R) − QLIKE(f_U)); first take the within-cell same-day mean over (day, cell), then take the cross-cell mean over day,
  giving a single daily series; apply day-clustered DM to it (HAC lag = max(h)−1 days, HLN).
  Pre-declared subfamilies: long-form cells, event-driven cells, all 69 cells, 3 omnibus p-values in total, Holm(3).
- Secondary (reported only): run SPA/MCS once over the reference set.
- Power calibration: using the existing signal-injection pipeline, estimate the detection rate of this omnibus on the {0.1, 0.2, 0.3, 0.5, 1.0}% firm-orthogonal
  injection grid, and report the MDE corresponding to 80% power.
- Decision language (pre-declared): omnibus does not reject and MDE ≤ 0.3% -> "a power-endorsed bound";
  omnibus rejects -> written up consistently with the detectable≠attributable≠bankable trichotomy
  (what is detected is a systematic cross-cell micro-increment; attribution and realisability are unchanged); insufficient power -> report the MDE honestly, do not upgrade the wording.

## Boundary with the running HPO

T1c seed retraining and the single test evaluation (the correction to the tuning-artefact sentence) are governed by `configs/hpo_arm.yaml`
(tag `hpo-prereg-v1.0`) and are not part of this document; this document adds no new reads of the test split --
all the tests above touch only the existing test predictions in the existing predictions.parquet and train no new model (except B1;
B1 only generates new test predictions, follows exactly the same established protocol as C6, and makes no selection based on test performance).


## v1.3 revision (2026-07-16, prior to any B2 statistic): B2 = fourth cross-family probe (Gemma) + 70B zero-content probe

**Motivation**: ml "a second healthy non-Qwen cross-family probe …
with the forecaster-health screen and the replication decision rule pre-declared — this is the
single experiment that moves the residual from 'directional' to real or kills it, and either
outcome raises my P"; skeptic "would remove my strongest live objection and add ~0.10 to my P";
R12 major #2 (headline point estimate single seed, single family).

**B2 design**: the mechanism reuses committed `crossfamily_mistral24.py` verbatim (3 reproducibility-jitter seeds + arithmetic
ensemble, ED panel, same excerpts, same prompt verbatim, G1''/G1q/G5/G3'' machine-precision gates), changing only the model:
**Gemma-3-27B-it, bf16, TP=2** (Google family, matched-class against Qwen3-32B, not quantised -- which incidentally
removes one of the 70B's int4 confounds). Weights via ModelScope (the established practice as the box has no HF egress).
**Degradation sequence (advanced only on hard failure, never advanced because of results)**: download unavailable or pilot health failure -> GLM-4-32B
(Zhipu family) under the same protocol; at most two pilots, and all pilots go into the paper's Stress Tests narrative whether they live or die.

**Pilot health gate (before the full run, pre-registered)**: val slice of ~2,000 documents, seed 2026, single pass; health formula =
verbatim the committed Yi/Phi criterion: **max variance-unit QLIKE < 4 and max modal share (round(pred,2)) <
60%** (pred_sd, R² reported as diagnostic columns). Pilot fails -> that model is instrument-dead-at-pilot, reported honestly,
no inference drawn, and the probe denominator sentence is updated to "of five probes" and flagged pilot-gated; pilot passes -> full run released
(3 seeds × 39,322 ED documents + ensemble).

**Replication decision rule (pre-declared, as ml worded it; all branches go into the paper)**: health (same formula on the full run) is a precondition;
- **(a) Holm-robust replication** ⇔ vs the firm-identity reference, ≥2/3 horizons satisfy clustered DM<0 and
  Holm(3)<.05 -> the residual wording is upgraded to "family-robust (two healthy families Holm-significant)", and the abstract drops
  "only partly family-robust";
- **(b) directional replication** ⇔ 3/3 DM<0 but fewer than 2 meeting Holm -> same tier as llama70, "three healthy probes agree in sign"
  goes into the main text, the Holm-robustness wording unchanged;
- **(c) does not replicate** ⇔ healthy but neither (a) nor (b) -> the residual is downgraded to "Qwen-conditional" in the abstract + 06 + 07
  (a fix that costs us points; committed to execute);
- **(d) instrument-dead** ⇔ the health formula fails (pilot or full run) -> entered in the table following the Mistral precedent, no inference.
Readings in the same form as B1: rel% vs firmID (primary) and vs HAR, per-h clustered DM, Holm(3) per reference,
STRONG/WEAK/NONE verbatim following the committed decision formula; the ensemble is the primary basis, single seeds as robustness.
**Products**: `results/tables/crossfamily_gemma27.{csv,md}` (named after the actual model if degraded),
write-once single shot.

**Piggyback: 70B zero-content probe (quant's words: "date+ticker term inside its reference, reconciling
the Table 6 149/103% probe cell with the replication claim")**: on the already committed
llama70-ens panel, run the same date+ticker zero-content prompt (verbatim the same template as the C6 contamination arm)
through llama70 (int4, at the same precision as the committed run it is compared against, for internal consistency), single seed; readings =
probe rel% alongside fulltext rel% + text-beyond-identity (whether fulltext still adds under the joint f_datefirm
reference) -- **descriptive, no branches**, for use in the reconciling sentence between the Table 6 probe cell and the replication claim.
Product: `results/tables/crossfamily_llama70_probe.{csv,md}`, single shot.

**Boundary**: no model is retrained; zero changes to prompts; the committed readings of C6/llama70 are not recomputed (anchor gates
follow the G1'' convention); timestamp = this tag (the author is advised to additionally deposit this document on OSF).
