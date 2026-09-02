# Pre-registration: H public-price variant, full cascade (prereg-h-v1.0)

Date: 2026-07-16. Committed and tagged before any H statistic. Commitments (family convention):
all branch results enter the paper regardless of direction; single-shot discipline; revisions
precede statistics. Upstream: label_parity (ROW 10, committed) has already supplied the
verdict-preservation three-panel prior (standalone 18/18, combo genuine A=8 vs C=7) and the trade
quantification (clean coverage 80.19%, exit firm-row coverage 31.9%, test split coverage 95.4%).

## Design

**Public source**: Yahoo Finance v8 daily adjusted close (same fetcher and same total-return
convention as label_parity Stage 1; the cache has been lost and must be re-fetched, **fetch-drift
disclosure**: the re-fetched correlation/coverage are tabulated alongside the committed
label_parity table, and drift exceeding 0.5pp of coverage or 0.001 of correlation must be
conspicuously flagged in the artefacts). Yahoo data must not be redistributed -- the release object
is the **rebuild pipeline + fetch script**, not the data itself (consistent with label_parity.md).

**Three panels (the cascade version of label_parity Stage 5)**:
- **A** = full panel + CRSP labels (anchor, G1 reproduction gate);
- **B** = publicly covered rows ∩ panel + CRSP labels (isolates the survivor effect);
- **C** = publicly covered rows + public labels/features (= the releasable variant itself).
The A−B difference = survivor loss; the B−C difference = label-source noise.

**Scope**: standalone verdicts for all leaderboard arms (day-clustered DM vs A2, variance-unit,
same method as committed) + the full 69-cell cascade chain (primary → firm-identity → maximal pool
→ conjunction, text predictions frozen, combiner refit on val, machinery = an est-hook extension of
rangebased_cascade) + per-cell MDE and injection recovery (verbatim). A-block: A2/A6 refit on public
features+labels (A6's RS± sign decomposition rebuilt from public returns); A3/A4/A5 frozen + val
recalibration (range-based precedent).

**Gates**: G1 panel A reproduces the committed cascade to machine precision (counts + stats);
G2 covered-row label Pearson ≥0.99 (prior 0.998; below that, abort and check the fetch);
G3 coverage reconciliation (80.19% ± fetch drift, decomposed per split and per exit-status and
tabulated alongside the committed parity table); G4 placebo run as usual.

**Branch commitments**:
- **(a) verdict preserved** (expected): the standalone verdicts and the conjunction (Holm=0) on
  panels B and C agree with A → the licence-free variant is promoted to a **formal release object**:
  12_reproducibility is rewritten from "withheld" to "shipped variant + quantified survivor cost"
  (train loses 27%, test loses 4.6%, priced openly); the one sentence in 07 is upgraded to full
  cascade numbers; contribution 1 gains a variant entry.
- **(b) composition moves, verdict preserved** (range-based precedent): the same upgrade as (a),
  with the composition difference tabulated faithfully.
- **(c) verdict flips** (conjunction>0 or a standalone winner appears): report honestly; the variant
  ships regardless, with the flip itself as a finding about survivor/label-source effects, and the
  A vs B vs C decomposition locating the source.
- **Deliberate exclusions**: no second public source beyond Yahoo (Stooq etc. left for the future);
  long-form/event-driven documents are not processed separately (same pipeline); no text model is
  retrained.

**Outputs**: `results/tables/public_variant_cascade.{csv,md}` (single-shot guard) +
`scripts/analysis/public_variant_{labels,cascade}.py` + the fetch script. Entirely local CPU (≤5 cores).
