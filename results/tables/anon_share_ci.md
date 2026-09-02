**UNCERTAINTY QUANTIFICATION ONLY — branch adjudication is locked by anon_arm.{csv,md} (fired) and is not altered here. prereg-ea v1.6, day-block bootstrap, B=2000, seed 2026.**

# Identity share — day-block bootstrap 95% CI (prereg-ea v1.6)

Registered at tag `prereg-ea-v1.6` BEFORE any share-CI statistic; computed once on 2026-07-16T15:37:19+00:00.

- **B = 2000**, **seed = 2026**; percentile interval [2.5%, 97.5%].
- **Resampling unit = effective trading day** (the day-clustered DM unit): each cell independently resamples its own test days with replacement; val is untouched.
- **Combiner weights frozen**: committed val-fit, test-frozen legs; a draw reweights test days only, so per-observation (qR, qU_unmasked, qU_masked) are fixed and each draw statistic is a ratio of day-sum aggregates — identical machinery to the anon_score cell statistic (obs-mean normalisers cancel).
- **Undefined rule (v1.1)**: a draw with unmasked increment <= 0 has no defined share; CIs use defined draws only; the undefined fraction is disclosed per readout and > 20% is flagged UNSTABLE.
- **Median readouts**: per draw, the median over the DEFINED cell shares of that draw (the point-estimate convention); a draw with zero defined cells is undefined for the median.
- **Gate**: recomputed point estimates (rel_har unmasked/masked, share) reproduce the fired anon_arm.csv at machine precision (rtol 1e-12) in all 6 cells — PASSED before any draw.
- B2 exited at G1 and does not participate (see anon_arm.md); firmID-side cells are outside the registered readout set.

| readout | arm | h | point share | 95% CI | undefined % | flag |
|---|---|---|--:|---|--:|---|
| cell | c6 | 5 | +0.508 | [+0.398, +0.624] | 0.0% |  |
| cell | c6 | 10 | +0.561 | [+0.430, +0.695] | 0.0% |  |
| cell | c6 | 20 | +0.709 | [+0.567, +0.838] | 0.0% |  |
| cell | c2 | 5 | +0.024 | [-0.135, +0.186] | 0.0% |  |
| cell | c2 | 10 | +0.510 | [+0.447, +0.570] | 0.0% |  |
| cell | c2 | 20 | -0.389 | [-4.936, +0.134] | 2.8% |  |
| arm_median | c6 | 5/10/20 | +0.561 | [+0.468, +0.687] | 0.0% |  |
| arm_median | c2 | 5/10/20 | +0.024 | [-0.131, +0.258] | 0.0% |  |
| pooled_median | c6+c2 | 5/10/20 | +0.509 | [+0.436, +0.560] | 0.0% |  |

## Prose rule (registered)

- Wherever the paper cites a share point estimate, attach the CI (the abstract may be exempted for space); CIs change NO locked branch wording.

Runtime: cell prep 0.7s, bootstrap 0.1s (9 readouts).

Single-shot: this file is written once; re-running refuses while it exists.
