# SP500Vol-Bench

**Do long-form (10-K/10-Q) and event-driven (8-K) SEC disclosures add incremental
predictive value for short-horizon realised volatility, beyond a strong price
baseline and beyond knowing *which firm* filed?**

This repository is the benchmark, the code and the committed evidence tables
behind that question. The headline answer is a negative with a small, bounded
exception, and the apparatus is built so that the negative can be checked rather
than taken on trust.

---

## What the study finds

| Rung | Result |
|---|---|
| Standalone, against HAR-RV | **0 of 180** comparisons favour a text challenger; 155 are significantly worse |
| Combined with a *recalibrated* HAR reference | **38 of 69** cells show a placebo-confirmed increment |
| …against a reference that also knows the firm's own mean volatility | falls to **8 of 69** |
| …against a pool of five price models | falls to **9 of 69** |
| …against both at once | **0 of 69** |
| Economic value of what survives | significant in **1 of 18** portfolio cells |

The one durable positive is a prompted large language model reading 8-K current
reports at the five-day horizon. An anonymisation arm prices roughly half of the
surviving increment as firm identity rather than filing content.

The ordering the project argues for is that *detectable*, *attributable* and
*realisable* are three different claims, and that conflating them is how
weak-baseline studies overclaim.

## The evaluation protocol

Every headline count is computed on one declared basis:

- **Forecast object** — per-observation mean over three training seeds, not a single run
- **Test** — Diebold–Mariano with the Harvey–Leybourne–Newbold small-sample adjustment
- **Clustering** — loss differentials averaged within each trading day, then tested on the daily series (filings sharing a day share one market shock)
- **Multiplicity** — Holm's step-down within each of 15 families declared in advance, never pooled across them
- **Placebo gate** — a genuine cell must also fail to reward a permuted copy of the text forecast
- **Loss** — QLIKE, volatility units for the combination grid and variance units for the standalone leaderboard, never compared across the two

Numbers computed off that basis carry an inline tag wherever they appear.

## Data licence — read before using this repository

The price side of the benchmark is **licensed from WRDS/CRSP and is not
redistributable**. This repository therefore ships **no per-row CRSP-derived
value**: no daily returns, no realised-volatility labels, no RV features.

What ships instead is `release/`, engineered so a third party can reconstruct
the exact study sample and audit every reported test without receiving a single
licensed value:

| Artefact | Contents |
|---|---|
| `accession_index.csv` | all 144,129 benchmark filings with accession, CIK, ticker, form, timestamp, effective trading day and split — doubles as the exact EDGAR pull list |
| `membership_intervals.csv` | the 914 survivorship-free point-in-time S&P 500 membership intervals |
| `cik_links_pit.csv` | 30,100 point-in-time PERMNO→CIK link windows, including 31 audited successor-CIK overrides |
| `split_definition.md` | the chronological split rule, re-derivable from the effective trading day alone |
| `run_configs/`, `config_hashes.csv` | 240 production-run configurations and their SHA-256 chain, so the fingerprint check is executable rather than asserted |
| `raw_generations/` | 608,221 prompted-model generations with the prompt templates and decoding configuration that produced them |
| `aggregate_results/` | observation-level comparison tables from the earlier inference vintage, retained as the version those numbers were quoted from |

`results/tables/` carries the aggregate evidence tables the analysis is read
from — summary statistics only, again with no per-row licensed value.

To rebuild the licensed layer you need a WRDS/CRSP subscription and
`scripts/ingest_wrds.py`; everything downstream of that is deterministic.

## Layout

```
├── src/sp500vol/          the importable package
│   ├── data/              EDGAR fetch, iXBRL parse, point-in-time alignment
│   ├── models/            price (A), classical text (B), neural text (C), fusion (D)
│   ├── training/          trainer, checkpoint resume, losses
│   └── evaluation/        QLIKE, Diebold-Mariano, block bootstrap
├── scripts/
│   ├── ingest_wrds.py     CRSP export -> membership, PIT links, returns store
│   ├── build_dataset.py   EDGAR fetch and parse -> aligned filing-horizon rows
│   └── analysis/          the audit layer: clustered DM, forecast combination,
│                          firm-identity control, anonymisation, signal injection
├── configs/               one YAML per arm, plus the pre-registration records
├── release/               the licence-safe public bundle described above
├── results/tables/        committed aggregate evidence tables
└── tests/                 33 modules; point-in-time alignment is the critical one
```

## The model matrix

| Block | Arms |
|---|---|
| A — price | naive HV, HAR-RV, GARCH, EGARCH, ARIMA, SHAR and HARQ, HAR-X with point-in-time VIX |
| B — classical text | bag-of-words and TF–IDF ridge, Loughran–McDonald dictionary and engineered features |
| C — neural text | BERT, FinBERT, RoBERTa, Longformer-4096, three frozen 7–8B embedders, a prompted 32B decoder |
| D — fusion | concatenation MLP, gated fusion, price-plus-embedding, price-plus-prompted-forecast |

Long documents are handled five ways — truncation, chunk-mean pooling, learned
chunk attention, a hierarchical encoder, and native 4,096-token context — so the
comparison is between representation strategies rather than between one arbitrary
choice and a baseline.

## Reproducing

```bash
uv sync
cp .env.example .env      # set EDGAR_USER_AGENT="Your Name your.email@example.com"
uv run pytest tests/ -m "not slow"
```

The pre-registration records under `configs/` fix each audit family's design
before the statistics inside it were inspected. Where a record was amended, the
amendment states its date and its reason.

Experiments are complete and frozen; this repository is a record of them, not an
active training harness.

## Ethics and scope

- Sources: SEC EDGAR (US federal public disclosure) and CRSP market data via an institutional WRDS licence
- No human subjects and no personal data
- The models here are research artefacts and are **not financial advice**
- The evidence covers one market and one regime: large-cap US equities, test years 2022–2025

## Licence

- Code: MIT, see `LICENSE`
- Derived benchmark metadata in `release/`: CC-BY-4.0
- Raw EDGAR filings are US government public-domain and are not redistributed here; the accession index is the pull list
- CRSP-derived values are **not** included under any licence
