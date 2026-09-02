# Pre-registration: M1 -- cascade audit on Kogan's original corpus (prereg-kc-v1.0)

Date: 2026-07-17. Committed and tagged before any M1 statistic. Commitments (family convention): all branches enter the paper regardless of direction;
single-shot discipline; revisions precede statistics. This experiment **does not use** any data on /path/to/data-root and occupies no GPU (CPU-only);
it runs in parallel with prereg-ef (M2, which occupies the box).

## Motivation

"The only route that would move me is to expand the protocol from 'our instance + 2 demos' to **applying the audit at the scale of the field's published results**
-- run the cascade on several published disclosure-NLP results that have public code and report the survival rate."

## Important scope correction (prior to any statistic, based on checkable facts)

**The "N independent published results" the AC literally asks for do not exist in this field.** Reconnaissance (2026-07-17, repo-by-repo check) confirms:
the **public evidence base for disclosure-text→volatility is only three corpora** -- (i) the Kogan/FIN10K 10-K corpus,
(ii) the MDRM/EC earnings-call corpus, (iii) MAEC. HTML, NumHTML, VolTAGE, KeFVP, ECHO-GL are **all**
built on top of MDRM (by each repo's own README), so "auditing N models" = N models run on 1 corpus =
**pseudo-replication**, whose survival-rate denominator is fake.
**The paper therefore argues by census rather than by sampling**: of the three corpora, every one whose data can be legally obtained is run through the full cascade;
for those that cannot be obtained, the reason is disclosed publicly. **MDRM has already been ruled cite-only by `prereg_maec_audit.md` §9** (text and audio
bundled in split volumes, no licence, no longer distributed) -- that ruling came first and is not changed by this experiment. MAEC has been audited (FACTS §13g).
**This experiment adds the third: the Kogan corpus.**

## Relation to the existing Kogan section (key distinction, to prevent self-confusion)

The committed `kogan_dissolve.md` is **transplanting Kogan's evaluation DESIGN onto our panel**; FACTS has already
bound its honest reading ("the published-style design yields no transferable positive on a modern panel";
it **must not** be claimed that a published positive result was reproduced). **This experiment is the opposite: running our cascade on Kogan's own corpus** --
if it succeeds, that is the genuine external audit of "reproduce a published positive result, then have the protocol reprice it" (the same type as MAEC).
The two coexist: the former answers "does the design transfer to a modern panel", the latter answers "what happens when the protocol is applied to a published result on its original corpus".

## Data (all public, verified reachable 2026-07-17)

The Kogan et al. (2009) 10-K corpus `http://www.cs.cmu.edu/~ark/10K/`: for each year (1996–2006) it provides
`meta.txt` (key | filing date yyyymmdd | EDGAR URL | company | **CIK**), `tok.tgz` (tokenised text),
`logvol.+12.txt` (**forward 12-month log volatility = the label**), `logvol.-12.txt` (**past 12-month log volatility
= the price baseline needed for recalibration**). No licence terms, citation only is required. **Self-contained: no WRDS/CRSP/GPU needed.**
G-K0: SHA-256 of the six files recorded in the artefacts; row-count and key-space consistency assertions.

## Cascade (rung by rung; the mechanism reuses the same-type ladder of the committed `maec_protocol.py`)

- **L0 published convention**: Kogan's convention -- text features (TF-IDF) + `logvol.-12` as control,
  regressed on `logvol.+12`, **naive obs-level inference**, their annual OOS split (train ≤ y, test = y+1).
  Reading = the MSE improvement rate of the text arm vs the `logvol.-12`-only arm (the quantity they report).
- **L1 recalibrated baseline**: the baseline becomes a **recalibrated** `logvol.-12` (OLS intercept + slope,
  fitted on the training years and frozen for the test year) -- the first practice of this paper's protocol.
- **L2 firm-identity reference**: the reference additionally gets the **training-period mean log volatility of the same CIK** (zero text terms).
- **L3 clustered inference**: clustered by **filing date** (the shock-sharing unit), HAC + HLN, replacing the naive obs-t.
- **L4 Holm (pre-declared family)**: Holm within the family of L3's per-year p-values.
- **L5 conjunction**: survival requires L1∧L2∧L4 to hold simultaneously.
- **placebo**: label permutation (5 seeds), |DM|<2 as the gate, the same form as the main protocol.

## Branch commitments (all pre-registered, all enter the paper)

- **(a) Reproduce then dissolve** (expected): L0 reproduces a positive text effect of the published magnitude, and L1–L5 dissolve it rung by rung to non-survival
  → the census claim holds: "of the three corpora we ran every one whose data we could obtain; apparent gain reproduced k/k, survival 0/k";
  the existing Kogan section in 07 is **replaced in place** by this reading (pages self-funded).
- **(b) Text survives the whole cascade on the Kogan corpus** → **the protocol has certified a real published positive result**: this is exactly
  the **real-world positive control** R11/R14 repeatedly asked for (proof that the protocol does not only kill); report it faithfully and on that basis
  soften the generality of the "near-null" wording -- `the size of the shortcut is a property of the panel and the baseline, not a constant`
  (directly absorbed by the existing FACTS §11/§13g framework); **this is good news for the paper, write it as such**.
- **(c) L0 cannot reproduce the published positive result** (e.g. because their exact SVR hyper-parameters / feature convention were not used) → report the reproduction failure itself,
  and infer nothing further; in the census table that corpus is marked "published reading not reproduced", with the reason made public.
- **(d) Data acquisition fails** (dead links etc.) → register it faithfully as not executed, and the census drops to 2/3 corpora.

## Gates

- **G-K1**: a consistency check of the L0 reading against the order of magnitude reported in Kogan et al. (2009) (same sign, same order of magnitude);
  an inconsistency triggers branch (c), and parameters **must not** be tuned to make it consistent (single-shot discipline).
- **G-K2**: no-look-ahead assertion -- from L1 on, every fit uses training years only; L0's naive convention is reproduced as-is and annotated
  (consistent with the look-ahead disclosure convention of kogan_dissolve.md).
- **G-K3**: report CIK coverage and cross-year recurrence rate (the precondition for the firm-identity reference).

## Artefacts

`results/tables/kogan_corpus_audit.{csv,md}` (single-shot guard) + `scripts/analysis/kogan_corpus_audit.py`
+ the fetch script (the data is not redistributed, only the pipeline). CPU-only, an estimated 2–3 hours of machine time.
Timestamp = this tag (an additional OSF record is recommended).


## v1.1 record (2026-07-17, **after the statistics were produced** -- so this is a record, not a modification)

The single shot has been fired (`kogan_corpus_audit.{csv,md}`). Execution exposed the two prereg defects below; **both readings of both
are reported unconditionally, neither selected on the result**; this section records the rulings and their reasons for a reader to check.

**(1) Misattribution of the split (a factual error in the prereg; the rule itself is unambiguous)**. This file calls `train ≤ y, test = y+1`
"**their** annual OOS split" -- **it is not theirs**. Kogan et al. §6 explicitly use a **5-year rolling window**,
test 2001–2006 (their Table 4 varies the window length over 1/2/5 years and **never** uses an expanding window). The prereg's *rule* is executable and
unambiguous; only the *attribution* is wrong. Running only the prereg rule leaves G-K1 unanswerable (an expanding-window reading cannot be matched against their
rolling-window published number); running only the published convention violates the already-tagged prereg. **Both arms were therefore declared in the script before any statistic and
both reported unconditionally**; G-K1's comparison target is `L0_pub` (the only one commensurable with their Table 2).

**(2) Silence on L2 self-inclusion (a branch-deciding ambiguity)**. The prereg says only "the **training-period mean** of the same CIK", and does not state whether a training row's
own label may enter its own CIK mean. **Ruling: `loo` (leave-one-out) is primary**, on grounds that are structural,
not result-driven:
- the prereg specifies that L2 **strengthens** the reference (it is the firm-identity control the text must beat). Under `incl` the fitted coefficient is
  pushed towards 1.0 (the feature **is partly the label**), the reference overfits, and its **test MSE is worse than L1 in 10/10** -- that rung **weakens**
  the reference, mechanically **inflates** the text gain and can manufacture survival out of nothing. A rung that weakens the reference cannot be the rung the prereg describes.
- the committed template points the same way: the entity-mean control (STPEV) in `maec_protocol.py` is a `shift(1)` PIT expanding
  prior mean -- the current row's label is excluded by construction; a self-inclusive fixed mean is already demoted to robustness in that template.
- the clinching evidence: in test 1997 (train = the single year 1996) 99.1% of training rows are the only filing for their CIK, so the "firm mean" is that row's own
  label, and the fitted β = **+1.000**.
- **Honest declaration**: `incl` gives **(a)** (text dissolves, **favourable to this paper**), `loo` gives **(b)** (text survives,
  **unfavourable to this paper**). The ruling takes the one unfavourable to this paper; all `incl` readings are recorded in the artefacts as-is, **but no incl L2
  number may be cited in the main text**.

**(3) The magnitude reading of G-K1 (annotated faithfully, the gate not changed)**. The gate says "same sign, same order of magnitude". Measured L0_pub **+9.54%**
vs published **+1.21%**: same sign, ratio **7.91×**, within the "≤10×" operationalisation but **not comfortably**; under the stricter
"same power of ten" reading it fails the gate and triggers (c). **Both readings are written out in the artefacts**; the supporting evidence for the PASS verdict is
**per-year structural agreement** (4 of the 6 test years have the same sign; 2001 worst and 2004 best agree on both sides -- their Sarbanes-Oxley
pattern), i.e. the same effect at a different magnitude rather than another effect.

**(4) The transportability limit of the placebo (a methodological finding, must enter the main text as a qualification)**. Kogan's **joint-arm** convention
(text and the volatility control in the same regression) means that "permuting the text" does **not** collapse that arm onto the reference (unlike maec's combiner convention),
so the transplanted |DM|<2 gate confounds **text signal** with **functional-form difference** under this design: in 50 draws the permuted text
still significantly beat the reference **13 times**. Therefore, among the 6/10 un-gated survivals, **1997 and 2005 cannot be attributed to text**, while
1998/2000/2003 clearly can → **3/10 are placebo-gated survivals**. When the main text cites (b) it must give
this qualification and the number 3/10 alongside it, and must not be written as "text survives on the Kogan corpus".
