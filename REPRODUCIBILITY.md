# Reproducibility

This repository is the reproducibility artifact for the study. It is written so
that a reader who does not trust the reported result can check it, and so that a
reader without a WRDS/CRSP subscription can still get further than "take our
word for it".

## What you can verify without any licence

Everything in this section runs on what ships here.

**The fingerprint chain.** Every production run's configuration is published
twice: the original SHA-256 and the fingerprint of the shipped preimage. Some
preimages had absolute local paths neutralised before release, and each such row
names the exact fields that moved.

```bash
uv run python scripts/analysis/config_fingerprints.py --verify-preimages
```

This re-hashes all 240 configurations using only files inside this repository.

**The test suite.**

```bash
uv sync
uv run pytest tests/ -m "not slow"
```

33 modules. The point-in-time alignment tests are the ones that matter: any
look-ahead in label construction would invalidate every downstream number, so
they are the first thing to run and the first thing to distrust.

**The evidence tables.** `results/tables/` holds the aggregate tables every
reported number is read from. They are summary statistics, so they carry no
licensed value. Each analysis script under `scripts/analysis/` names the table
it reads and aborts if that table no longer matches the numbers it recorded — a
stale artefact cannot quietly survive an upstream change.

**The pre-registration records.** `configs/prereg_*.md` fix each audit family's
membership and decision rule before the statistics inside it were inspected.
Where a record was amended, the amendment carries its own date and its reason.
Read these before the results: they are what makes the negative result a finding
rather than a failure to look hard enough.

**The generations.** `release/raw_generations/` carries all 608,221 prompted-model
outputs as the model emitted them, beside the prompt templates and decoding
configuration. The parsing from raw string to numeric forecast is therefore
auditable rather than asserted.

## What needs a licence, and how to rebuild it

The price side comes from CRSP via WRDS and cannot be redistributed. No per-row
CRSP-derived value is in this repository: no daily returns, no realised-volatility
labels, no RV features.

To rebuild that layer:

1. `scripts/ingest_wrds.py` — CRSP export to membership, point-in-time links and the returns store
2. `scripts/build_dataset.py` — EDGAR fetch and parse, aligned to filing-by-horizon rows

`release/accession_index.csv` is the exact EDGAR pull list, so step 2 is
reproducible by anyone; step 1 is the only step behind a subscription.

## What the numbers rest on

| Item | Choice | Where |
|---|---|---|
| Forecast object | per-observation mean over three training seeds | `configs/`, seeds 2026–2028 |
| Loss | QLIKE; volatility units on the combination grid, variance units on the standalone leaderboard | `src/sp500vol/evaluation/` |
| Test | Diebold–Mariano with the Harvey–Leybourne–Newbold small-sample adjustment | `src/sp500vol/evaluation/dm_test.py` |
| Dependence | loss differentials averaged within each trading day, then tested on the daily series | `scripts/analysis/clustered_dm.py` |
| Multiplicity | Holm within each of 15 families declared in advance, never pooled | `results/tables/holm_families.md` |
| Placebo | a genuine cell must also fail to reward a permuted copy of the text forecast | `scripts/analysis/` |
| Power | a stated minimum detectable effect beside every null, plus an oracle-injection calibration | `results/tables/signal_injection_power.md` |

Numbers computed off that basis carry an inline tag wherever they are reported.

## Known limits of this artifact

- Model checkpoints are not published. Weights for the fine-tuned encoders run to
  several gigabytes per arm and are not needed to check any reported number; the
  committed predictions and the aggregate tables are.
- Three arms were retired at their reproduction gates because the earliest run
  metadata logged no package versions, so a June-era environment could not be
  reconstructed. Those retirements are on record; the statistics they would have
  produced do not exist and are not reported.
- The evidence covers one market and one regime: large-cap US equities, test
  years 2022–2025. Two external panels are used to test portability, and both
  are described in the tables rather than shipped, for the same licensing reason.

## Compute

Training ran on a single 4×A100-40G node. Total recorded GPU time across all
arms is 590.4 hours, of which one long-context arm consumed 254.7. Price and
classical-text arms are CPU-only and deterministic. Per-arm cost is in
`results/tables/cost_accuracy.csv`.
