# SP500Vol-Bench

**A Point-in-Time Benchmark and Identity-Controlled Audit of Disclosure-Based Volatility Forecasting**

**Author**: Kun Zhang (MSc Advanced Computer Science, University of Leeds)
**Supervisor**: Dr Chunwei Xia
**Programme**: COMP5200M MSc Advanced Computer Science dissertation

---

> ### What this repository is, and what it is not
>
> This is a **curated, sanitised mirror**, not the working repository. It was
> rebuilt so that it could be made public without disclosing the author's machine
> paths, storage layout, rented GPU hosts and personal contact details, and that
> rebuild cost it its history: **it carries a single squashed commit and no git
> tags**, and the commit is dated after the dissertation was submitted because
> that is when the mirror was last rebuilt.
>
> The dissertation's Appendix A describes the working repository, which carries
> 489 commits from 16 June 2026 onward and 26 tags, 21 of them pinning
> pre-registrations. Those tags are listed in `release/tag_manifest.txt` with the
> commit each pins and its date. **In this mirror those commits do not resolve**,
> so the manifest here is a record of the tagged history rather than an index into
> it, and a reader cannot check a registration out and diff it. The working
> repository is private because its history carries the strings above; the author
> will grant read access to examiners on request.
>
> Everything else the appendix describes — the release bundle, the code, the test
> suite, the evidence tables, the seven registered analysis records and the
> report's own LaTeX sources — is here and is complete.

---

## What this is

When a language model appears to beat a price-based benchmark at forecasting
stock-market volatility, the gain may reflect not what a filing said but which
firm filed it. This project builds the two instruments needed to tell those
apart.

1. **SP500Vol-Bench** — a survivorship-free, point-in-time benchmark linking
   144,129 SEC filings of S&P 500 constituents over 2010–2025 to the volatility
   their filers subsequently realised, under a no-look-ahead protocol.
2. **An identity-controlled audit** — a protocol that credits text only for the
   out-of-sample loss it removes *beyond a recalibrated price reference*, with
   day-clustered inference, placebo gates and power calibration.

The audit is the point. A text model can look useful for three quite different
reasons — it read the filing, it recognised the filer, or the baseline it was
measured against was weak — and only the third is easy to spot.

## What it found

Across models from word counts to a prompted 32-billion-parameter language
model, the audit returns a **calibrated near-null**.

- No challenger beats the price baseline standalone under squared error or
  volatility-unit QLIKE: **0 of 180** comparisons. Seven of 180 favour a
  challenger in variance units, and all seven are price models.
- Apparent gains against a single recalibrated reference (**38 of 69** cells)
  collapse to **8** against a reference that knows only *who* is filing, to
  **9** against a pool of five price models, and to **0** against both.
- A zero-text term carrying each firm's average volatility — reading no text at
  all — beats the recalibrated price baseline in four of six channel–horizon
  comparisons on the shared rows.
- An anonymisation arm prices the 8-K channel's identity share at a pooled
  median of **0.51**: roughly half of what survives the price reference is who
  filed rather than what was filed.
- What remains is a bounded residual — prompted readings of 8-K filings retain
  **0.2–0.5 per cent** over the firm-identity reference — with no measurable
  portfolio value.

The verdict is an ordering rather than a number: on this panel disclosure text
is detectable, only partially attributable to content, and not realisable as
economic value. The audit travels — on an external earnings-call corpus it
reprices a published gain to nothing detectable, while on consumer reviews a
content signal survives the same controls.

## What is in this repository

This repository carries the redistributable part of the work: the release
bundle, the regeneration pipeline, the test suite, the aggregate evidence tables
and the registered analysis records.

| Path | Contents |
|---|---|
| `release/accession_index.csv` | The 144,129-filing benchmark index: accession number, CIK, ticker, form, filing timestamp, effective trading day, split assignment |
| `release/split_definition.md` | The chronological split rule and its filing counts |
| `release/DATA_CARD.md` | Datasheet: motivation, composition, collection, uses, exclusions, and the CRSP/WRDS attribution the licence requires |
| `release/config_hashes.csv`, `release/run_configs/` | The fingerprint chain: one row and one config per production run, 240 runs carrying 233 distinct SHA-256 digests |
| `release/aggregate_results/` | Twelve observation-level comparison and stratification tables |
| `release/raw_generations/` | 1,221 parquet shards holding 608,221 prompted-model generations, with the prompt templates and decoding configuration behind them |
| `release/tag_manifest.txt` | All 26 repository tags with the commit each pins and its date |
| `results/tables/` | The working evidence set every number in the report is read from |
| `configs/prereg_*.md` | Seven registered analysis records, pre-declared design against what was reported |
| `src/`, `scripts/`, `tests/` | The regeneration pipeline and its 186 tests |

### What is not here, and why

CRSP data reaches this project under a University of Leeds subscription to
Wharton Research Data Services, which permits academic research use but bars
redistribution of the raw data and of per-row quantities derived from it.
Withheld are the daily return series, the realised-volatility labels, the price
features, the per-row model predictions, and two CRSP-derived crosswalks (the
914 survivorship-free membership intervals and the 30,100 point-in-time
PERMNO→CIK link windows).

Those rows are exactly regenerable from the released scripts by any licence
holder. A **licence-free variant** rebuilds the price side from public sources
at 72.6 / 89.3 / 95.2 per cent coverage of training, validation and test rows,
so the benchmark can be exercised end to end with no subscription at all.

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
# 1. Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync deps and create .venv
uv sync

# 3. Configure the EDGAR User-Agent (required by SEC)
cp .env.example .env
# Edit .env: set EDGAR_USER_AGENT="Your Name your.email@example.com"
# Optional: redirect bulk data to another disk via SP500VOL_DATA_ROOT
# (see .env.example) — the full corpus is ~50-100 GB.

# 4. Verify the install
uv run pytest tests/ -q -m "not slow and not gpu and not network"
```

## Typical workflows

```bash
make spike        # prototype on 5 firms x 1 year — run this first
make data         # full ingestion: EDGAR + market data + alignment + labels
make train MODEL=C2_finbert_s3
make ablation AB=AB1
make tables       # regenerate evidence tables from results/
make figures
make dissertation
```

## Repository layout

```
.
├── configs/            # YAML configs, and the seven registered analysis records
│   ├── data/           # EDGAR fetch, universe, splits
│   ├── models/         # one file per configured arm; A6, A7, C6 and D4 are script-driven
│   ├── ablations/      # AB configs
│   ├── compute/        # hardware profiles
│   └── prereg_*.md     # the seven registered analysis records
├── data/universe/      # scaffolding only; the CRSP-derived crosswalks are withheld
├── release/            # the licence-safe released core, inventoried above
├── results/tables/     # the working evidence set
├── scripts/            # CLI entry points and the analysis layer
├── src/sp500vol/
│   ├── data/           # EDGAR fetcher, parser, point-in-time alignment, calendar
│   ├── features/       # returns, realised volatility, Loughran-McDonald dictionary
│   ├── models/         # price / classical_text / neural_text / fusion
│   ├── training/       # trainer, checkpoint resume, losses
│   ├── evaluation/     # MAE/RMSE/QLIKE, Diebold-Mariano, block bootstrap
│   ├── pipelines/      # end-to-end orchestration
│   └── utils/          # seeds, environment capture, cost tracking
├── tests/              # 186 tests in 33 modules
└── writing/dissertation/supporting/   # frozen sources the report's provenance comments cite
```

## Model families

| Block | Arms |
|---|---|
| A. Price-only | A1 HV, A2 HAR-RV, A3 GARCH, A4 EGARCH, A5 ARIMA, A6 SHAR and HARQ, A7 HAR-X with point-in-time log VIX |
| B. Classical text | B1 BoW+Ridge, B2 TF-IDF+Ridge, B3 L-M+Linear, B4 L-M+features |
| C. Neural text | C1 BERT-base, C2 FinBERT (S1 truncation / S2 chunk-mean / S3 chunk-attention / S4 hierarchical), C3 RoBERTa-base, C4 Longformer-base (4096), C5 frozen decoder-LLM embedding probe (three lineages), C6 a numeric forecast elicited directly from instruction-tuned Qwen3-32B |
| D. Fusion | D1 Concat-MLP, D2 gated fusion, D3 LLM-embedding gated fusion, D4 price lags supplied to C6 inside the prompt |

Long-document handling (S1–S5) is a design axis within Block C rather than a
separate contribution: the audit's finding is that the choice among them does
not survive an identity control.

## Tests and continuous integration

`.github/workflows/ci.yml` runs on every push to `main`. The first job installs
the locked dependency set and runs the fast tests on Python 3.11 under
`not slow and not gpu and not network`. The second holds the library and its
tests to the full lint rule set (`ruff check` and `ruff format --check` over
`src/` and `tests/`), checks `scripts/` for the errors that would make an
analysis script *wrong* rather than untidy (`ruff check --select F821,F811,E9`),
and runs `mypy` over `src/sp500vol/`.

Two things about that arrangement are worth stating plainly. The `mypy` step is
invoked with a trailing `|| true`, so type checking is **advisory and not
enforced**; `make lint` runs the same checker without the escape. And the
project has a single author and no merge commits, so although the workflow
declares a pull-request trigger, no second-party review ever took place. What
CI supplies is a regression check on one person's pushes.

## Provenance

- Every number printed in the dissertation carries a same-line `% src:` comment
  in the LaTeX source naming the file it was read from.
- Every production run is fingerprinted: `config_fingerprints.py
  --verify-preimages` re-hashes all 240 released preimages against the published
  index using only files in this repository.
- Seven registered analysis records sit under `configs/`, one per audit family,
  each pinned by a git tag listed in `release/tag_manifest.txt`. One arm — the
  tuned challenger — carries an OpenTimestamps proof anchored outside the
  repository.
- Evidence tables are generated, not typed: their generators abort when a source
  table drifts from its recorded values.
- **Why the analysis scripts talk about "reviewers" and "review rounds".** Every
  analysis in the audit layer carries, in its docstring, the objection it was
  written to answer and which round of review raised it. Those rounds were
  adversarial self-review: panels run against the author's own work to find what
  it had got wrong, before anyone else could. They were not external peer review,
  and the dissertation describes no such process because none took place. The
  round dossiers themselves are not published — they are working documents about
  a companion manuscript rather than about this study — so a filename such as
  `REVIEW_ROUND3_FRESH_PANEL.md` in a docstring will not resolve here. What
  survives is the useful half: for each analysis, a record of the criticism that
  caused it to exist.

## Design decisions

- **Date range**: 2010–2025. Train 2010–2019, validate 2020–2021 (the COVID
  volatility regime), test 2022–2025.
- **Universe**: S&P 500 with survivorship-free, point-in-time membership from
  CRSP, PERMNO→CIK linked through CRSP/Compustat-Merged. 914 membership
  intervals; the linking handles M&A successions, dual-class tickers and
  renames.
- **Market data**: CRSP daily total return, split- and dividend-adjusted;
  `log_return = log1p(DlyRet)`. No external price-API fallback chain.
- **8-K handling**: combined into a single event-driven channel.
- **Point-in-time alignment**: any look-ahead in label construction invalidates
  the study. See `tests/data/test_alignment.py`.

## Ethics

- Data is public or licensed research data: SEC EDGAR (US federal public
  disclosure) and CRSP via WRDS under an institutional licence.
- No human subjects and no personal data.
- The models are research artefacts and **not financial advice**. The audit's
  own finding is that the measured text signal has no realisable portfolio
  value.
- The full assessment is Appendix B of the dissertation, whose source is here at
  `writing/dissertation/appendices/B_ethics.tex`.

## Licence

- **Code**: MIT (see `LICENSE`).
- **Released benchmark artefacts** (`release/`): the accession index, split
  definition, datasheet, fingerprint chain and aggregate tables are derived
  works of this project and are released under MIT alongside the code. The
  underlying EDGAR filings are US government public-domain material.
- **Not covered**: CRSP-derived values, which are not redistributed here. Their
  reuse is governed by the user's own CRSP/WRDS licence, and the attribution
  those terms require is reproduced in `release/DATA_CARD.md`.
