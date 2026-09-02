# HPO challenger arm: final experiment specification (pre-registration draft v1.0)

**Skeleton: plan 1 (compute-first, highest judge scores on all three items 0.82/0.85/0.88); grafting on plan 0's top-2 seed verification, manifest firewall, pre-registered exclusion list and free B-family grid, plus plan 2's Track-B combination steelman, val-fit/val-select time split, quantified coverage bound and degradation ladder; and fixing item by item every defect listed in the three verdicts.**

---

## 1) One-sentence positioning

Use a **pre-registered, ASHA-tuned, 3-seed, dual-track selection** challenger arm to turn R5 ml_methods concern #5 ("the null is merely an artefact of under-optimisation") from a blocking hole into a positive evidence sentence in the paper -- whether the challenger wins or loses after tuning, the paper gets a stronger headline (expected ΔP +0.03~0.05), while incidentally closing R3's two CRITICALs, "loss misalignment" and "Holm family not auditable".

---

## 2) Scope and search space

### 2.1 Models included (for each spectrum stratum, the best val representative under the fixed recipe + items raised in internal review)

| # | Model | Panel | Target parameterisation (stratum) | Search budget | Coverage bound (95% confidence) |
|---|------|------|----------------------|---------|------------------|
| T1a | C2 FinBERT-S1 | long-form | levels (log-RV) | 32 Sobol configurations | top-9% |
| T1b | C2 FinBERT-S1 | long-form | **firm-demeaned** (Δln RV vs firm training mean) | 16 configurations (warm-started from the levels winner) | top-17% |
| T1c | C2 FinBERT-S1 | event-driven | levels | 24 configurations | top-12% |
| T1d | C2 FinBERT-S1 | event-driven | firm-demeaned | 12 configurations | top-22% |
| T2 | C5 frozen embedder heads (e5-mistral, gte-qwen2, qwen3 -- all three) | both panels | levels | ridge closed-form full grid + MLP 48 random | grid exhaustive |
| T3 | D2 gated fusion + D1 concat-MLP | both panels | levels | 48/32 random respectively | top-6%/9% |
| T4 | C4 Longformer-4096 | long-form only | levels | **16 configurations × {h=5, h=20} searched independently**, h=10 inherits the h=5 winner | top-17% |
| T5 | B1/B2 TF-IDF/BoW ridge (including delta-text variants) | both panels | levels | CPU full grid (α 13 points × n-gram {1,1-2} × min_df {5,20,100} × sublinear-tf) | exhaustive |

**Fixing "Longformer is a token gesture" (named jointly by two judges):** T4 goes from 12 configurations / h=5 only / 50% subsample, upgraded to 16 configurations, the full training set, and **h=20 (the existing -32.5% disaster cell, the place where tuning headroom is most credible) searched independently**. batch is realised via gradient accumulation to give an equivalent batch (equivalence declared, fixing the confound of batch 128 under 40GB VRAM).

**Fixing "S1 inclusion is circular" (plan 0 defect 1):** the pre-registration declares that input strategies S1-S4 are benchmark-defining dimensions and not hyperparameters; **plus one cheap patch** -- once the FinBERT long-form winning configuration is frozen, run a single-seed val re-check of S2/S3/S4 under that configuration (≈14 A100-h); if some S* overtakes S1 on val under the tuned configuration, that S* replaces it and enters the 3-seed retrain (switching rule pre-registered, depends on val only). The S re-check for event-driven is triggered conditionally, only if long-form flips.

**Exclusion list (pre-registered, one sentence of reasoning each):** C1/C3 are dominated by FinBERT in every cell at the same stratum and the same input (C3 long-form is the pre-declared additional run for the case where it leads at D9); B3/B4 dictionary methods have no substantive hyperparameters (the linear head is folded into the B-family grid); C6 prompted Qwen3-32B has no gradient-trained parameters, its prompt space is unbounded and is not searched, the existing elicitation-robustness arm is cited as compensation, and C6 is the arm that wins its cell, so nobody asked us to make it stronger; D4's trainable parts are covered by D1/D2.

### 2.2 Search space (written into `configs/hpo_arm.yaml`, zero changes after commit)

**8 dimensions shared by T1/T4:**
- lr: log-uniform [5e-6, 1e-4] (brackets the fixed recipe's 8e-5 and the 1e-5 commonly cited in the literature)
- head LR multiplier: {1, 10}
- weight decay: log-uniform [1e-4, 0.3]
- head: {linear, MLP-256-GELU}; head dropout: {0.0, 0.1, 0.3}
- **layer freezing: {full fine-tune, top 4 layers only, frozen-encoder linear probe}** (a direct answer to "regularisation/freezing can fix the overfitting")
- effective batch: {32, 128} (Longformer via gradient accumulation)
- **training objective: {MSE-on-log-RV, QLIKE-aligned (variance scale, softplus output, ε=1e-8 floor, gradient clipping)}** -- takes R3's CRITICAL "trained with squared error, evaluated with QLIKE" and solves it inside the search space
- epochs is not a search dimension: ≤15, val-fit QLIKE early stopping (patience 3), the same mechanism as the fixed recipe

**T2 C5 heads:** ridge α ∈ 10^{-3..5} 15-point log grid (closed form, CPU); MLP: depth {1,2} × width {128,256,512} × dropout {0,0.1,0.3} × lr log-U[1e-4,1e-2] × wd {0,1e-4,1e-2} × objective {MSE, QLIKE}.
**T3 D2/D1:** the table above minus layer freezing, D2 adds gate hidden layer {64,128,256}, gate initialisation {price-prior, uniform}, text-branch L2 {0,1e-4,1e-2}.

**Quantified coverage bound written into the paper (plan 2's best idea):** "if a configuration exists that can beat HAR, with 95% confidence it occupies less than 9% of the declared space (FinBERT long-form)" -- honestly noting that this bound is a sampling-level bound, with ASHA promotion risk mitigated by the deepened rungs (see §3).

---

## 3) Selection rules and leakage prevention

### 3.1 ASHA (fixing the rung-0 noise defect named jointly by two verdicts)

- Algorithm: fixed seed (sampler seed 2026) Sobol/random sampling + deterministic ASHA. **No BO/TPE** -- zero adaptive experimenter degrees of freedom, the search itself can be pre-registered.
- **Rungs changed to {2, 5, 15} epochs (Longformer {2, 5, 10}), η=3** (the original plan's {1,4,12}/η=4 would systematically kill slow-starting configurations such as low-lr full fine-tuning and frozen encoders -- exactly the region where the standing objection places the winner; the judges confirm this change costs about +40% of T1 search compute, affordable, see §5).
- Promotion metric: the **vol-unit val-fit QLIKE** at that horizon (consistent with the paper's primary convention; variance-unit is recorded only, never used for selection).
- event-driven rung-0 uses a pre-declared 30% temporally stratified subsample (the subsample only affects who dies early, never the winner's final training data).
- Selection granularity: per (model × panel × stratum × horizon) -- strictly more generous to the challenger, and the pre-registration states explicitly that "generosity favours the challenger".

### 3.2 Three-level val split + top-2 seed verification (plan 2 + plan 0 grafted, each fixing the other's defect)

- **val-fit (2020-01~2021-06):** in-rung promotion + early stopping.
- **val-select (2021-07~2021-12):** final configuration ranking -- out-of-sample with respect to search-time early stopping, sealing off the counter-attack "config selection and early stopping double-use the same val".
- **top-2 seed verification (fixing "val-select is only 6 months, single-seed noise dominates"):** the top 2 configurations by val-select ranking are each retrained with the full 3 seeds (early stopping uses the full val window per the main protocol, reusing the existing pipeline byte for byte; the mild information reuse of val-select here is disclosed honestly -- the real firewall is at test), winner = **whichever has the lower val-select QLIKE for the 3-seed ensemble**. Seed noise is averaged out by the ensemble, a direct answer to the "48 lottery tickets" attack. Longformer and the demeaned strata take top-1 (declared reason: compute).
- Tie-break rule pre-registered: lower lr first, then higher weight decay.
- **Diagnostics (must be reported):** val-select→test rank correlation over all max-rung trials, the generalisation gap of winner vs median-config, the winner's per-val-year ranking (disclosing the known properties of 2020-21, which includes COVID).

### 3.3 Dual-track selection (plan 2's core asset, grafted on almost for free)

The same trial pool, two mechanical argmins, both pre-registered, both reported:
- **Track A (standalone):** argmin of the model's standalone QLIKE on val-select → guards the 0/180 standalone null.
- **Track B (combination steelman), applied only to FinBERT-levels on both panels and to D2:** for each max-rung trial, using the existing f_Ufirm design ([1, log f_A2, log firm-mean-val-RV, log f_text], weights fitted on val-fit), argmin over val-select of QLIKE(f_Ufirm) − QLIKE(f_Rfirm) -- **optimising the challenger directly against the firm-identity-augmented reference**. If it still fails the identity+pool gate, the bound on "a tuned challenger could have won" is as tight as it gets. If the Track B winner is the same as Track A's, the retrains are merged (the budget is counted as an upper bound assuming no merge).

### 3.4 Leakage-prevention boundary

- The data manifest consumed by the HPO harness **physically excludes** the features and labels of test 2022-25, and every trial records the manifest hash (plan 0: "we could not possibly have peeked" becomes verifiable infrastructure rather than a verbal claim).
- Conditional retraining rule (val-only, pre-registered): Longformer and the demeaned strata get a full 3-seed retrain only if tuned val beats the fixed recipe's val; otherwise report "no val headroom, fixed recipe retained".
- **Pre-registration artefacts (fixing the "self-signed git tag" defect):** `configs/hpo_arm.yaml` + the design document are, before any test evaluation, (a) committed with a dated git tag and the hash printed in the paper, and (b) **simultaneously registered on OSF** -- an external timestamp, no longer "the author holds both the clock and the prior".
- **QLIKE loss engineering risk (named in two verdicts):** D0-D1 are set aside specifically as harness-build days, the QLIKE loss carries unit tests (numerical stability, correctness of the variance→log-space conversion into the recalibration pipeline), with a 2-trial end-to-end smoke test first; pre-registered fallback -- if >30% of a cell's QLIKE-loss trials are NaN/divergent, that cell's loss dimension collapses back to MSE and this is reported honestly (a val-only decision, not a fork).

---

## 4) Seeds and how this enters the protocol

- **Search phase:** single seed 2026 (pre-registered, never swapped post hoc; standard practice in the ASHA literature, seed noise is absorbed by the top-2 verification and the retrains).
- **Final runs:** each winning configuration is retrained from scratch with 3 seeds (2026/2027/2028), **primary = seed-ensemble (mean of the three seeds' log-forecasts)**, word-for-word identical to the definition in each of the A2/C/D arms; per-seed goes to the appendix.
- **Zero new code paths into the protocol:** what the tuned models produce is just a new `results/runs/` directory, fed straight into the existing `forecast_combination.py` leakage-safe log-space recalibration → reference interval (single recalibrated HAR upper bound / firm-identity lower bound) → maximal price pool → label-shuffle + within-date placebo → day-clustered DM (HAC lag h−1, HLN) → Holm.
- **Placebo hard gate (especially for Track B):** the Track-B winner is "selected against the combination objective" and therefore has the strongest incentive to learn the identity shortcut -- its seed-ensemble must pass both placebos before it is eligible for any win count; failure is recorded as an artefact, written into the pre-registration as binding. This is the paper's own methodology imposed symmetrically on the arm most favourable to text.
- **Multiplicity:** the original 180/69 family is frozen unchanged and the denominator does not change; the tuned arm forms its own pre-registered Holm family (the full cell enumeration table goes into the yaml: standalone ≈51 items + cascade ≈51 items + Track-B ≤12 items); a joint primary+tuned Holm is additionally reported as a sensitivity.
- The existing single-seed tuned arm (lr + early stopping) is demoted to a **pilot**, and the pre-registration openly discloses that its null result has already touched test, together with an argument about the direction of the bias: a prior null only lowers our motivation to run more HPO, it does not shape the space towards the null; the new space strictly contains the pilot space and adds every dimension the objections named (freezing, decay, head, QLIKE objective). Hiding it is what would be the real forking path.

---

## 5) Compute schedule

**Two anchors side by side (fixing "single-point estimate with no audit margin"):** upper bound = the task-spec convention (FinBERT-LF 0.2 A100-h/horizon-epoch, full run 1.5 h/horizon/seed; ED ×2.5; Longformer ×5); lower bound = measured cost.json (FinBERT-LF full run 1.14 h/seed, ED 1.53, Longformer 18.7, D2 0.6), about 1/3 of the upper bound. **Budget against the upper bound, read the slack off the lower bound.**

| Item | A100-h (upper bound) | Notes |
|---|---|---|
| T1a FinBERT-LF levels search (32 configurations, rungs 2/5/15) | 79 | ≈131 epoch-units/horizon ×3×0.2 |
| T1b LF demeaned search (16) | 39 | |
| T1c FinBERT-ED levels search (24, rung0 30% subsample) | 98 | |
| T1d ED demeaned search (12) | 48 | |
| T4 Longformer search (16 × {h5, h20}, rungs 2/5/10) | 114 | h10 inherits the h5 winner |
| T2 C5 MLP heads (3 embedders × 2 panels × 48) | 30 | explicit budget, includes audit margin for ASHA-free full-epoch training |
| T3 D2+D1 heads (2×2 panels) | 40 | |
| T5 B family + C5 ridge full grid | 0 (CPU) | saturates the box cgroup quota, in parallel with the GPUs |
| top-2 3-seed verification retrains: FinBERT LF 27 + ED 66 + D2/D1/C5 14 | 107 | |
| Conditional retrains (val-gated): demeaned LF+ED 47 + Longformer 60 | 107 | triggered only if there is val headroom |
| Track-B extra retrains (happens only if the winner differs from Track-A's, ≤4 cells) | 60 | |
| S2-S4 val re-check (LF, winning configuration, single seed) | 14 | ED re-check triggered conditionally |
| test single evaluation + cascade/DM/Holm/placebo recomputation | 15 | +CPU |
| **Total** | **≈750 (upper bound) / ≈260 (measured convention)** | available 4×A100×12 days ≈ 1150 h → **utilisation 65% (worst case) / ~23% (normal)** |

**Schedule (D0 = 7/13; the box is occupied by Yelp for the preceding 1 day):**
- **D0–D1 (7/13–14):** in parallel with wrapping up Yelp -- harness build (extending run_matrix into an asynchronous ASHA scheduler) + QLIKE loss unit tests + 2-trial end-to-end smoke test; the pre-registration yaml lands as a git tag + OSF anonymised registration. **Explicitly budgets 1.5 engineering days, fixing the defect "the schedule assumes the code is written correctly first time".**
- **D1.5–D4:** T1a/T1b search (2 GPUs) ‖ T5/C5-ridge CPU grid ‖ T2/T3 heads slotted into the gaps.
- **D3–D6:** T1c/T1d (2 GPUs) + T4 (2 GPUs).
- **D6–D8:** top-2 seed verification retrains + S re-check + conditional retrains + Track-B incremental retrains.
- **D8 (7/21, abstract deadline):** test single evaluation + cascade recomputation -- **the verdict lands on the day of the abstract**, so the abstract uses dual-version wording (see §7).
- **D9–D10:** tables and integration into the paper, **numbers frozen 7/23**; on D9 the C3 RoBERTa additional run is triggered only if it leads.
- **D11–D15 (7/24–28):** ≥4 days of buffer, able to absorb a full rerun of any one arm or a rewrite after a crash.

**Pre-registered degradation ladder (wall clock only, never results; degrading does not constitute a fork):** ① cut the Longformer h=20 search, migrate from the h=5 winner instead; ② cut the ED demeaned stratum, keep demeaned for LF only; ③ cut D1, let D2 represent fusion.

---

## 6) Where this lands in the paper

1. **Add a forward reference at the end of the §4 sentence "comparability, not per-model tuning":** "…complemented by a pre-registered, externally time-stamped HPO arm (§X) that tests whether this design choice manufactures the null."
2. **New main-text table (~0.4 page) "Pre-registered tuned-challenger arm":** rows = tuned (model × panel × stratum), columns = fixed-recipe QLIKE / Track-A tuned / Track-B tuned, **val-headroom (tuned val − fixed val)**, vs HAR standalone verdict, cascade (recalibrated-HAR / identity / maximal-pool) verdict, placebo, coverage bound. Every cell enters the table whether it wins or loses.
3. **Headline rewrite:** "0/180 under the comparability recipe **and 0/N under a pre-registered ASHA-tuned, multi-seed arm — including challengers selected directly against the identity-augmented combination objective**" (if the null holds); the cascade sentence (38/69→0/69) notes that tuned Track-B is likewise 0/N; contribution 3 ("neither scale nor domain adaptation lifts text") switches to citing the tuned FinBERT/Longformer/C5 cells.
4. **Add one subordinate clause to the abstract**, nothing else changes.
5. **A separate sentence for the tuned firm-demeaned FinBERT cell in the identity-confound subsection** -- it is the only cell that answers both W1 (under-tuned) and W2 (levels-only objective), and it is reported whether or not it passes the identity reference.
6. **If val-headroom ≤ 0:** that is itself an evidence sentence -- "validation headroom is ≤0 in k/N cells: the fixed recipe was not the binding constraint."
7. **Appendix X:** the full yaml text + git hash + OSF link, the full ASHA trajectories, the full trial table (config/val-fit/val-select/test), val→test rank correlation, per-seed table, pilot disclosure, Holm family enumeration table (incidentally closing R3's "family not auditable" CRITICAL).

---

## 7) Contingency plans

- **(A) tuned still wins 0 (a priori most likely: 5/9 pilot cells were worse, and the overfitting diagnostics imply tuning mainly adds regularisation and shrinks towards HAR):** the null is upgraded from "an artefact of the recipe" to "insensitive to a pre-registered search over lr × decay × freezing × head × objective, and the combination-steelman track fails too", R5 concern #5 is closed, ΔP +0.03~0.05 is realised.
- **(B) tuned beats the single recalibrated HAR but dies at the identity+pool gate:** not a crash -- this is a live re-enactment of the paper's 38/69→0/69 main line, and "even the gain of a challenger optimised against the strong reference is reproduced by identity" is the strongest new evidence for the shortcut argument, headline unchanged.
- **(C) the Track-B winner passes identity+pool and both placebos:** the headline null dies, the paper does not -- the benchmark (contribution 1) and the reference-interval protocol (contribution 2) results are unaffected; the narrative shifts to "the protocol adjudicates a real effect for the first time, and it only surfaces when tuned against the strong reference", which incidentally answers a further standing objection (that the cascade has no sensitivity demonstration), and is structurally identical to the existing C6 8-K bounded-residual narrative.
- **Schedule insurance:** the test verdict lands on D8 (7/21), ≥7 days before the full-text deadline; **both abstract versions (null-holds / survivor-found) are pre-written before D8** -- the one unpublishable situation ("(C) is discovered only after the abstract has locked in the null framing") is excluded by construction.
- **The only genuinely bad case is results arriving after the 7/23 number freeze:** controlled by the 65% upper-bound utilisation + the degradation ladder + the 4-day buffer, not carried by the experimental design.

---

## 8) Explicitly what we will not do (forking-paths prevention, all written into the pre-registration)

1. **Do not search the C6 prompt space** (unbounded fork; cite the existing elicitation-robustness arm).
2. **Do not re-search the full S-strategy matrix** -- only the val re-check under the winning configuration (switching rule pre-registered).
3. **Do not change the data windows, panel definitions, splits or evaluation metrics** -- they are benchmark definitions.
4. **Do not widen the search space post hoc** -- the space was deliberately made generous at declaration time (bracketing 8e-5 and 1e-5, including freezing and the QLIKE objective), and changes not at all thereafter; deviations forced by crashes/OOM are logged item by item and reported as they are.
5. **Do not use adaptive search algorithms** (BO/TPE) -- fixed-seed sampling + deterministic promotion, the search itself has no experimenter degrees of freedom.
6. **test is touched only once**, the test evaluation of each final-run cell is irrevocable and may not be dropped post hoc; the harness physically excludes test (manifest hash auditable per trial).
7. **Do not retroactively change the original Holm family** -- the original 69/180 denominator is frozen; the tuned arm forms its own family; do not cherry-pick single points from the new arm to tell a story (all cells enter the table).
8. **Do not hide the pilot** -- the single-seed tuned arm and its test null are openly disclosed together with an argument about the direction of the bias.
9. **Do not exempt the Track-B winner from any gate** -- placebo + identity reference are imposed symmetrically and hard.
10. **Both selection tracks are reported** -- there is no degree of freedom to "pick the track that favours us after the fact".

---

Judge scores: [{"angle": "critic-first", "acc": 0.8, "feas": 0.75, "safe": 0.85}, {"angle": "compute-first", "acc": 0.82, "feas": 0.85, "safe": 0.88}, {"angle": "inference-first", "acc": 0.8, "feas": 0.72, "safe": 0.88}]
