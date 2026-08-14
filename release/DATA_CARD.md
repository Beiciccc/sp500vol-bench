# Data Card: SP500Vol-Bench (public index release)

A datasheet (Gebru et al. style) for the **publicly redistributable** portion of
SP500Vol-Bench. The benchmark links U.S. regulatory disclosures to forward
realised volatility under a survivorship-free, point-in-time design.

## Motivation
Created to test, under leakage-free and survivorship-free conditions, whether
regulatory disclosure text adds incremental predictive value for equity
volatility beyond a strong price-history baseline (HAR-RV). Built for a peer-reviewed conference submission.

## Composition (this public release)
- **`accession_index.csv`**: 144,129 filings with `accession`, `cik`, `ticker`,
  `form` (10-K/10-Q/8-K), `filing_time_utc`, `effective_trading_day`, `split`.
  Disclosure text is **not** included (fetch from SEC EDGAR by accession; the
  index gives the exact pull list).
- **`membership_intervals.csv`**: 914 S&P 500 membership intervals
  (`ticker`, `permno`, `cik`, `member_from`, `member_to`), survivorship-free.
- **`cik_links_pit.csv`**: 30,100 point-in-time PERMNO→CIK link windows,
  including the audited successor-CIK overrides that recover filings from
  acquired/delisted firms.
- **`split_definition.md`**: the chronological split rule + counts.
- **`config_hashes.csv`**: SHA-256 fingerprints of every run configuration (one row per run).

## What is NOT in this release (and why)
Realised-volatility **labels** and the **daily return series** are derived from
CRSP/WRDS data, whose licence prohibits redistribution. They are therefore not
included. Everything needed to **regenerate** them with a CRSP subscription is
provided (universe, links, splits, and the labelling code). Release of
**model predictions / loss differentials** (model outputs, not CRSP data) is
pending a licence determination; see `README.md`.

## Collection & preprocessing
- Universe + returns: WRDS CRSP (survivorship-free, CIZ 2.0).
- Disclosure text: SEC EDGAR (fair-access rate limits respected).
- PERMNO→CIK: CRSP/Compustat-Merged point-in-time link windows + 23 audited
  manual override firms (31 link windows) for M&A successions.
- Labels: forward realised volatility over 5/10/20 trading days from CRSP total
  returns; aligned to the effective trading day with no look-ahead.

## Recommended uses
Benchmarking disclosure-text vs price-only volatility models under strong
baselines; studying survivorship/selection bias in financial NLP; evaluation-
protocol research. **Not** investment advice; a research artefact only.

## Distribution & licence
Public files here are derived from public sources (SEC EDGAR) and the project's
own audited mappings, plus aggregate (summary-statistic) test results, released
under a permissive licence. Per-row CRSP-derived quantities (RV labels, RV
features, model predictions) are excluded per the WRDS/CRSP licence.

## Attribution (required by CRSP / WRDS terms)
- Volatility labels: "Calculated (or Derived) based on data from CRSP © 2026
  Center for Research in Security Prices, LLC."
- "Wharton Research Data Services (WRDS) was used in preparing this work. This
  service and the data available thereon constitute valuable intellectual
  property and trade secrets of WRDS and/or its third-party suppliers."

## Maintenance
Maintained by the project author; regeneration is fully scripted (see
`README.md`). Versioned by git commit SHA.
