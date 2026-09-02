# Action A — money experiment: split-based entity controls are blind to pretrained priors

**Claim under test.** The field's standard fix for entity leakage is an entity-disjoint
split (group-wise CV): no entity in both train and test, so a fitted model cannot memorise
an entity's mean and call it text understanding. It cannot work on a zero-shot prompted
arm, for a mechanical reason: that model never sees our training split at all — its
knowledge of a business was acquired from a pretraining corpus no split of OUR panel can
touch. Splitting is a statement about our data; the model's entity prior is a statement
about its data.

**Design.** Identical audit, identical universe and rows, ONE thing changed: the split rule.
The zero-shot predictions are literally the same vector under both splits (that is the
point); the fitted arms are refit under each. The zero-content identity probe (business
name + city + categories + month, no review text) is pure identity by construction, so it
is the instrument that detects whether the split removed identity.
Prediction was recorded in the script docstring before running, with its falsifier.

**Universe.** 38,399 test rows max per cell; 1,926 businesses with zero-shot
70B predictions. Month-clustered DM (HAC lag h-1 months, HLN) and business x month two-way
CGM, both ported from the paper's own protocol code. DM < 0 = the arm helps.

## Result

| arm | A: shared-entity (chronological) | B: entity-disjoint (**the standard fix**) |
|---|---|---|
| **h = 1 month** | | |
| 70B identity probe (zero-content) | **+0.159%** (DM -5.26, p=0.0000) | **+0.197%** (DM -2.77, p=0.0075) |
| 70B prompted (zero-shot) | +0.195% (DM -1.99, p=0.0524) | **+0.184%** (DM -2.59, p=0.0121) |
| TF-IDF (fitted) | **+0.291%** (DM -3.42, p=0.0013) | **+0.220%** (DM -2.13, p=0.0374) |
| **h = 3 months** | | |
| 70B identity probe (zero-content) | **+0.278%** (DM -2.95, p=0.0051) | **+0.371%** (DM -3.43, p=0.0011) |
| 70B prompted (zero-shot) | **+0.225%** (DM -4.47, p=0.0001) | +0.280% (DM -1.50, p=0.1386) |
| TF-IDF (fitted) | **+0.534%** (DM -2.62, p=0.0119) | +0.152% (DM -0.86, p=0.3943) |

| entity-mean control: share of test entities it has a mean for | **83.4%** | **0.0%** |
|---|---|---|

## Reading

**The fix works on the fitted arm.** TF-IDF at h=3 goes from significant (+0.534%, p=.012)
to nothing (+0.152%, p=.394). The entity-mean control is mechanically dead: 0% of test
entities have a train/val mean to learn.

**The fix does nothing to the zero-shot probe.** A forecast with *no review text at all* —
only the business's name, city, categories and the month — stays significant at BOTH
horizons under the entity-disjoint split (p=.0075, p=.0011), and its point estimate rises
(+0.159->+0.197, +0.278->+0.371). Its identity channel was never in our training split to
remove.

**Consequence for benchmark builders.** Running group-wise CV and reporting 'entity leakage
controlled' is, for any prompted foundation model, a claim the design cannot support. The
cheap detector is an arm-matched partial-input probe: give the model the identity fields
and no content, and read what is left.

**Honest scope.** Absolute gains are small (<0.6%) in a domain where text adds little; the
claim is about the CONTRAST (fix kills fitted, spares pretrained), which is significant and
directional at both horizons, not about the magnitudes. One domain, one prompted family.
The claim is existence-and-mechanism, not a law about how large the blind spot is.

Source: `scripts/experiments/second_domain/yelp_entity_disjoint.py` -> `yelp_entity_disjoint.csv`.
