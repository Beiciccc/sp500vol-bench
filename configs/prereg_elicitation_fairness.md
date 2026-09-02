# Pre-registration: M2 — elicitation-protocol fairness test (prereg-ef-v1.0)

Date: 2026-07-17. Committed and tagged before any M2 statistic. Commitments (family convention): all branches enter the paper regardless of direction;
single-shot discipline; revisions precede statistics; the criteria are not conditioned on results.

## Motivation

"If 4 of the 5 instruction-tuned families collapse under this elicitation, the most economical explanation is that the **prompt protocol is fragile and was developed on
Qwen3-32B** -- the sieve may be selecting for **elicitation-protocol fit**, and the family it fits is precisely the one
the pipeline was built around. C5x controls for curation, **not for elicitation-protocol tuning**." (R14 ac, major #2)

This experiment tests that confound directly: **give every already-dead family a fair chance at prompt adaptation** and see whether health recovers.

## Design

**Arm set (2 families)**: Mistral-Small-24B (bf16) and Gemma-3-27B-it (bf16).
**Rationale for the selection (prior to any statistic, citing only already-committed facts)**: both are bf16 (same precision as primary, removing
llama70's int4 confound), both are 24–27B (matched-class against 32B), and both sets of weights are already on the box; Yi-1.5-34B and
Phi-4-14B are excluded -- their failure is already explained by the committed capability-floor readings (Yi variance QLIKE
7.60–8.19 = capability floor; Phi 14B far smaller than the matched class), and the GPU budget is limited; disclosed as such.

**prompt variants (K=3, contents frozen in this file before any run)**:
- **V0** = the committed C6 prompt verbatim (baseline, i.e. the one used for the existing instrument-dead readings);
- **V1 = few-shot**: V0 + 2 **validation-set** examples (fixed: the 1st and 2nd documents of the canonical sort of the val panel,
  together with their true RV labels); the examples are **verbatim identical** for all families;
- **V2 = format-hardened**: V0 + an explicit numeric-range hint ("annualised RV, typically 0.05--1.50")
  + a restatement of the JSON schema + one re-ask on parse failure (same mechanism as the committed retry).
The V1/V2 texts are fixed in this tag; **they must not be modified because of any run result**. Across families they are verbatim identical except for the chat-template fold
(Gemma has no system role; following the fold already disclosed in prereg-rfa v1.3).

**Selection rule (val only, test never read)**: each family × each variant is run once over **2,000 val documents**
(deterministic: the first 2,000 of the canonical sort, the same set as the v1.3 pilot), computing the committed health formula
(max variance-unit QLIKE < 4 **and** max modal share(round(pred,2)) < 60%).
**For each family select the variant with the lowest val modal share** (ties broken in the order V0 → V1 → V2). If that variant is **healthy**
→ release to the full run (3 jitter seeds × 39,322 ED documents + arithmetic ensemble, verbatim the same protocol as B1/B2);
if **all three** variants of that family are unhealthy → that family is registered as elicitation-robust instrument-dead and does not go to the full run.
**test is touched exactly once, and only by families released as healthy.**

## Branch commitments (all pre-registered, all enter the paper)

- **(a) healthy after adaptation and a Holm-robust replication** (≥2/3 horizons, vs firm-identity, clustered DM<0 and
  Holm(3)<.05) → **the AC's confound is confirmed**: the earlier "instrument dead" did have a prompt-protocol component; the residual wording
  is upgraded in 06/07 to family-robust (abstract untouched, see the v1.1 revision); and it is stated honestly that this conclusion is driven by prompt adaptation
  rather than by model capability.
- **(b) healthy after adaptation, directional replication** (3/3 DM<0, Holm<2) → tabulated at the same level as llama70;
  "three healthy probes, same sign"; the Holm-robustness wording is unchanged; the confound partly holds (health can be restored by prompt).
- **(c) healthy after adaptation but no replication** → **the residual is downgraded to Qwen-conditional** (**06 + 07**; fixing it costs marks,
  committed to be carried out); the confound holds and is unfavourable to this paper -- reported honestly.
  [v1.1 revision (2026-07-17, prior to any M2 statistic -- the pilot has not been run, no reading exists): the locus of the consequence
  is narrowed from "abstract + 06 + 07" to "**06 + 07**". **The reason is an external deadline, not a result**: the abstract of an external manuscript freezes on
  2026-07-21, after which substantive changes may, under that manuscript's revision rules, cause rejection; M2 lands between 07-21 and the full-text deadline 07-28. To make (c) executable without violating the freeze, the abstract **already carries
  the most conservative reading now** -- adding "only partly family-robust" (that qualifier is true under all of branches
  (a)(b)(c)(d): under (a) it is merely an understatement, under (b)(c)(d) it is exact), so the downgrade in (c) can fall entirely in 06/07 with no change to the abstract.
  Symmetrically, **the "delete only partly family-robust from the abstract" registered under branch (a) is likewise narrowed to leaving the abstract untouched** -- the upgrade
  is expressed only in 06/07. This revision does not change the decision conditions or the statistical criteria of any branch.]
- **(d) all three variants unhealthy (in both families)** → **the AC's confound is falsified**: fair prompt adaptation cannot revive them,
  so the health sieve is not selecting for elicitation-protocol fit but is measuring a capability floor; the main text gains a direct answer to that challenge,
  and the capability-floor claim is upgraded to "tested under prompt adaptation".
- Mixed (one family (a)/(b)/(c), the other (d)) → honest family by family, with the wording taken from the **more conservative** side.

## v1.1 revision (2026-07-18, prior to any M2 statistic -- the pilot has not been run, no reading exists): hardware change TP=2 → TP=1

The originally planned box (2×A100-40G) is unavailable; the actual box is a **single A100-80GB card**, hence **TP=2 → TP=1** (both 24B/27B
bf16 fit in 80GB, no tensor parallelism needed). This is an **external hardware constraint, not result-driven**; the revision precedes the pilot, and no
statistic exists. Three things are adjusted accordingly:

1. **Inference at TP=1**; every artefact, pilot json, sentinel and log records the actual TP (`tp_effective`).
2. **G-E1 is downgraded from a reproduction gate to a TP-invariance diagnostic** (honest reason): the committed crossfamily readings were produced at
   TP=2, and bf16 batched inference is **not bit-deterministic** (the paper already discloses that repeat-decode is only 94–97% bit-identical), so
   V0@TP=1 could never have reproduced committed bit for bit -- not even at TP=2. G-E1 therefore becomes: the val
   health columns of V0@TP=1 are **reported side by side with the differences** against committed (TP=2), and the script continues (does not abort); **an alarm is raised loudly only when a family's healthy/dead
   verdict flips between TP=2-committed and TP=1-V0** -- a flip would mean the committed
   instrument-dead verdict carries a TP qualifier and must be reported honestly (unfavourable to this paper, but reported anyway).
3. **Pre-registered handling of the TP confound**: M2's core estimand (the within-family V0/V1/V2 contrast, judging "whether fair adaptation can
   revive health") is unaffected with TP held constant -- TP only affects the anchoring to committed. Branch verdicts and statistical criteria
   **do not change because of TP**. The paper discloses TP=2→1 honestly along with the TP-invariance diagnostic result; if the diagnostic PASSes (no verdict flips,
   as expected, since the 60% modal-share gate is far from Mistral 87.9%/Gemma 71.4%), it incidentally proves that the committed
   instrument-dead verdict is not a TP artefact -- which is favourable to the paper.

This revision does not change any variant text (the V0/V1/V2 hashes are unchanged), any branch condition, any health formula, or any val-only
discipline.

## Gates

- **G-E1**: the V0 re-run reproduces the committed readings to machine precision (same code path, same weights, same seed) -- if it does not,
  this experiment aborts and pipeline drift is reported (old readings must not be overwritten with new numbers).
- **G-E2**: val-only assertion -- during the pilot, reading any test row is a fatal error (code-level assertion + log).
- **G-E3**: the variant-text hashes agree with the values fixed in this tag (guards against changing the prompt mid-run).
- **G-E4**: re-check of the health formula on the full run (same as v1.3, on the ensemble basis).

## Artefacts and boundaries

`results/tables/elicitation_fairness.{csv,md}` (single-shot guard): the val health columns for each family × each variant,
the selected variant, the full-run readings (if released), and the branch verdict. **No model is retrained; the combiner is not changed; the committed readings of C6/llama70
are not touched**. GPU budget: val pilot 6 × ~0.5h ≈ 3h; ≈ 1.5 box-days per released family for the full run.
Timestamp = this tag (an additional OSF deposit is recommended).


## v1.2 record (2026-07-18, **after the pilot statistics were produced** -- so this is a record + deviation registration, not a modification)

The single-shot pilot has been run (6 cells, 2000 val documents, TP=1). Execution exposed one **selection rule defect**, put through an adversarial check
(two readings, adversarial and methodological, both judging that "the substantive conclusion is robust, but the script's literal branch label is
self-serving and must be corrected"). This section records the ruling + full disclosure; **the conclusion was not selected in the favourable direction**.

### Facts (val health cell by cell; health gate = max variance QLIKE<4 and max modal<60%)

- **Mistral24**: V0 dead (qlike 4.34, modal 57.7%), V1 dead (qlike 49.8), V2 dead (qlike 5.03,
  modal 34.2%) -- **all three variants dead, no val-healthy variant**.
- **Gemma27**: **V0 HEALTHY (qlike 3.69, modal 45.4%<60%)**, V1 dead (qlike 10.3), V2 dead
  (qlike 6.03) -- V0 healthy, the adapted variants V1/V2 dead.

### selection rule defect (registered as a deviation)

The rule "select the variant with the lowest val modal; if healthy → release; if all three variants are unhealthy → instrument-dead" **fails for Gemma**:
the lowest modal is V2 (40.8%) but V2 is **QLIKE-dead** (6.03>4); the only healthy variant, V0, does **not** minimise modal.
"Lowest modal" and "healthy" part company, so the literal rule **returns an undefined gap** -- clause 1 selects the dead V2, and the premise of clause 2,
"all three variants unhealthy", is **false** (V0 is healthy). The literal (d) triggered by the script (both families all unhealthy) is **factually wrong
for Gemma** and must not be adopted.

### Ruling: the purposive/mixed reading (governed by the prereg's "mixed" clause: honest family by family, taking the more conservative wording)

The estimand is "**whether fair adaptation (V1 few-shot / V2 format-hardened) can restore health**", not "whether there exists any
val-healthy variant". Family by family:
- **Mistral24 = truly literal (d)**: no val-healthy variant, adaptation ineffective.
- **Gemma27 = not literally (d)**: there is a val-healthy variant V0, but -- (i) **V0 = the committed original C6 prompt**,
  whose committed full run (crossfamily_gemma27) **collapses on test at 71.4% (instrument-dead)**;
  val health has **never** been the instrument-dead criterion (that verdict is a full-run/test property), so a val-healthy V0
  and the committed instrument-dead verdict **never contradict each other**; (ii) **the adapted V1/V2 are worse than V0**
  (qlike 10.3/6.03 vs 3.69) -- adaptation **hurts**, it does not help.
- **Joint conclusion** (holds under every honest reading): fair prompt adaptation cannot restore full-run health in either family →
  what the health sieve measures is a **capability floor**, not Qwen-elicitation-protocol fit → **R14 AC major #2 falsified**.

### deviation registration (two items)

1. **Gemma V0 was not re-run at full scale under the registered 3-seed protocol**: the committed crossfamily_gemma27
   (single run / TP=2, test 71.4% dead) is accepted instead. Reason: V0 = the committed prompt verbatim, so a re-run would only produce the same
   test collapse, and the pilot has already shown V1/V2 to be val-dead (no new variant qualifies for release); zero new test budget (no val/test
   double-dip).
2. **The defect in the selection criterion "lowest modal"** as above, registered; it does not change the conclusion (the adapted variants of both families are dead).

### Mandatory disclosure (required by the check; goes into the md and into the main text, verbatim)

(a) Gemma V0 val-HEALTHY verbatim (qlike 3.69, modal 45.4%); **never write or imply "all Gemma variants are
val-unhealthy"**; (b) the selection defect (the lowest-modal V2 parts company with the healthy V0); (c) the route is purposive/mixed,
not literal all-dead; (d) instrument-dead is a committed full-run/test property, and a val-healthy V0 does not contradict it;
(e) adaptation hurts, it does not help (V1 10.3 / V2 6.03 vs V0 3.69); (f) Mistral is truly literal (d); (g) the TP diagnostic has zero flips
(Mistral Δmodal 0.05%, Gemma 0.20% → the committed verdict is not a TP artefact); (h) the Gemma V0 deviation
as above.

### TP-invariance diagnostic (registered in v1.1)

V0@TP1 vs committed@TP2 verdicts show **zero flips**: Mistral (dead↔dead, Δqlike 0.011, Δmodal 0.05%),
Gemma (healthy↔healthy, Δqlike 0.028, Δmodal 0.20%). → the committed instrument-dead verdict is
**stable under TP=2→TP=1 and not a TP artefact**; the box2/TP=1 choice is verified harmless (incidental positive evidence).
