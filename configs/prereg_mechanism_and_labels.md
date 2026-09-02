# Pre-registration: C mechanism test (shortcut size is predictable) + D range-based label robustness (prereg-cd-v1.0)

Date: 2026-07-15. This file is committed and tagged before any statistic of either analysis. Commitments (family convention):
**all branch results enter the paper regardless of direction; the specification must not be adjusted conditional on results; single-shot discipline; revisions must precede the corresponding statistic and be recorded.**

## C — Mechanism test: the sign and size of the identity control are a function of "how much of the entity the baseline encodes"

**Target of the claim upgrade**: the current draft's "the shortcut's size is a property of the panel and its baseline, not a
constant" is a post-hoc description; C turns it into **a diagnostic with a predicted quantity**.

**Predicted quantity y (per cell)**: the relative gain of the zero-text identity control over the reference
y = 100·[QLIKE(f_R) − QLIKE(f_Re)]/QLIKE(f_R) (f_Re = reference + entity-mean term; positive = the control helps the reference).
All taken from **frozen artefacts**: SEC 6 cells (2 channels × 3 horizons, the zero-text-vs-f_R rows of
firm_identity_ensemble), Yelp 2 cells (chronological, the entity-mean row of yelp_cascade, negative),
MAEC 8 cells (4 horizons × 2 references, the row4/f_Re readings from the protocol json) = **16 points**.

**Predictor x (per cell, the only new computation)**: the entity-encoding degree of the baseline prediction
x = R²[regress f_R's log predictions on entity dummies, on val] (within panel, same horizon, same reference).
Touching test labels is forbidden; x uses only val predictions and entity ids.

**Pre-declared statistics**: (i) Spearman rank correlation ρ(x, y) over the 16 points, prediction **ρ < 0** (the more entity the baseline encodes,
the less useful and even harmful the control); permutation p (cell labels permuted 10,000 times, seed 2026); (ii) sign test: the proportion of
y≤0 among cells with x above the median vs those below (Fisher exact); (iii) disclosure: panel clustering (the 16 points come from 3
panels, the points are not independent -- permutation reported both ways, within-panel permutation + global permutation, wording takes the more conservative).
**Falsification line**: ρ ≥ 0 or both permutation p >.10 → the mechanism claim does not hold, the paper keeps the descriptive wording, no upgrade;
this result likewise enters FACTS and the main text (one sentence of honest disclosure).
**Holding line**: ρ < 0 and conservative permutation p < .05 → upgrade the Discussion paragraph + one 16-point scatter (or a compact table),
wording ceiling: "the audit's identity term is a *predictable* correction: its sign and size track how
much of the entity the baseline already encodes". **No** claim of causality or universality (the objection pre-loaded by the ac
-- "3 panels = 3 points is too thin" -- is met head-on with 16 cells + panel-clustered permutation, with an added sentence in Limitations).

## D — range-based RV label robustness (Parkinson / Garman–Klass)

**Data**: `/path/to/data-root/sp500vol-data/market/full_ohlcv.parquet` (verified = CRSP DlyHigh/DlyLow,
100% coverage of the panel window). Labels: within the forward window, Parkinson σ̂²_P = (1/(4n·ln2))Σ ln(H/L)²,
Garman–Klass standard form, annualise per current convention (√(252/H)·RMS form aligned with volatility.py);
**Parkinson is primary, GK in the same table**. Price-side consistency: the past-RV features of the A-block references (A2 HAR etc.)
are **converted to the same estimator in step** (feature windows all end before the filing, the current anti-leakage audit runs as usual); the text arm is **not retrained**
-- its predictions are label-independent frozen artefacts, the combiner/recalibration is refitted on val against the new labels (log-space
recalibration absorbs the scale difference); this "predictions frozen, labels swapped" design is **disclosed as the primary Limitation**
(the text arm was optimised against a close-to-close target, so the reading is conservative towards the text side).

**Scope**: full-chain recomputation of the 69-cell grid (primary → firm-identity → maximal pool → conjunction),
day-clustered DM + per-family Holm + placebo gate + **per-cell MDE and injection recovery rate** (the mechanism reuses the current
signal_injection pipeline verbatim). Output `results/tables/rangebased_cascade.{csv,md}`, single-shot.

**Branch commitments**:
- **(a) null holds + MDE shrinks** (expected): conjunction still 0/69 and the median MDE materially below the current
  0.82% (shrinkage ≥30%) → wording upgrade: "under a ~5× lower-variance label proxy the near-null
  persists with materially smaller MDEs -- the 'selection device' part is upgraded to evidence of absence
  at the observed effect sizes"; qf's rank-1 (noise proxy) is retired.
- **(b) null holds + MDE does not shrink**: report as is, the qualifying sentence unchanged.
- **(c) text is Holm-significant against the identity reference**: honest reversal -- the residual chapter is rewritten, the abstract adjusted accordingly (nothing hidden).
- **(d) the reference-side ranking changes greatly** (HAR no longer strong) → first check the label construction (G2-style magnitude gate: the rank correlation between the Parkinson and
  close-to-close labels must be >0.8, otherwise halt and debug); no table may be shipped with a known fault.

**Sanity gates**: G1 recomputation with the current labels reproduces the committed 38/69 to machine precision (same code path); G2 rank correlation between new and old labels
>0.8 (per horizon); G3 leakage assertions the same set as the current audit; G4 placebo runs as usual.

## Boundaries

C only reads frozen predictions and val labels; D touches no model training; neither oversteps into the other or into the existing pre-registrations.
