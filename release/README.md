# SP500Vol-Bench: public reproducibility bundle

This directory is the auditable index for SP500Vol-Bench. It carries the filing
list, the split rule, the run fingerprints and the aggregate evidence, so that the
pipeline and every reported test can be checked **without** redistributing
licensed market data. Rebuilding the study sample itself additionally needs a CRSP
subscription, for the universe and links as well as for the returns.
Two CRSP-derived crosswalks that an earlier staging carried —
`membership_intervals.csv` and `cik_links_pit.csv` — have been withdrawn to
`crsp_restricted/`, which is **not** part of this public release. They hold CRSP
PERMNOs and CRSP dates, and the CRSP subscription agreement (cl. 1.2, 1.3, 2.1
and 5.3) and the WRDS Terms of Use (s. 2) do not permit publishing them; see
`crsp_restricted/README.md`. `scripts/ingest_wrds.py` rebuilds both from a CRSP
subscription.

## Contents
| File | What | Public? |
|---|---|---|
| `accession_index.csv` | 144,129 filings: accession, CIK, ticker, form, timestamps, **split** | ✅ (SEC + own mappings) |
| `split_definition.md` | chronological split rule + counts | ✅ |
| `config_hashes.csv` | two-digest index over 240 runs: original fingerprint, released-preimage fingerprint, and the fields sanitised | ✅ |
| `run_configs/` | the 240 `config.json` preimages themselves; re-hash with `config_fingerprints.py --verify-preimages` | ✅ |
| `raw_generations/` | 1,221 parquet shards, 608,221 prompted-LLM generations (SEC text in, model JSON out; no CRSP value) | ✅ |
| `aggregate_results/` | DM / QLIKE / encompassing / stratified statistics, p-values, Holm-BH, SEs (summary statistics only, no per-row CRSP values) | ✅ |
| `DATA_CARD.md` | datasheet | ✅ |

## How to regenerate the full dataset (with a CRSP subscription)
```
scripts/ingest_wrds.py          # CRSP zips -> membership + CIK links + returns
scripts/build_dataset.py        # EDGAR fetch + parse + PIT align -> aligned_filings
scripts/train.py --model <id> --dataset full --disclosure <subset>
scripts/evaluate.py / dm_vs_baseline.py / compare_runs.py
```
The filing text itself is fetched from SEC EDGAR by `accession` (the index here
is the exact pull list). RV labels are computed from CRSP total returns by the
labelling code; they are reproducible but not redistributable.

## Released aggregate results (licence-safe)
`aggregate_results/` holds the Diebold–Mariano and forecast-encompassing
statistics, p-values, Holm/BH-corrected values, stratified breakdowns and the
gate-cushioning table underlying every claim in the report. These are **summary
statistics only (no per-row CRSP-derived value**) and are releasable as
research results, consistent with AEA/RFS restricted-data policy. Any reader can
check every reported test value against them.

## Withheld under the CRSP/WRDS licence
- **Per-row CRSP-derived values are NOT released**: realised-volatility labels,
  RV features (`feature_rv_*`), and model predictions. The raw
  `results/runs/*/predictions.parquet` files embed the RV label in plain form, so
  they are licensed data and are deliberately excluded from this bundle.
- Per-row **loss-differential series** could in principle be released in a
  **non-invertible form** (loss differentials alone, with no predictions, labels or
  features from which the CRSP label could be reconstructed), but only subject
  to **written confirmation from the institutional CRSP licensing administrator
  and CRSP/Morningstar**. Until then the aggregate results above are the
  reproducibility artifact and are sufficient to audit every claim.
- Regeneration: any holder of a CRSP/WRDS subscription can reproduce the labels,
  predictions and the full per-row pipeline from the scripts above.

## Attribution (required by CRSP / WRDS terms)
- Volatility labels: "Calculated (or Derived) based on data from CRSP © 2026
  Center for Research in Security Prices, LLC."
- "Wharton Research Data Services (WRDS) was used in preparing this work. This
  service and the data available thereon constitute valuable intellectual
  property and trade secrets of WRDS and/or its third-party suppliers."

## Artefact availability
This bundle ships in full in the public repository that carries the work's
supporting materials, together with `scripts/`, `src/`, `configs/`, `tests/` and the seven frozen
analysis records (`configs/prereg_*.md`), so every number can be traced without
reference to any external repository.
