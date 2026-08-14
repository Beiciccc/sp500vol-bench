# S&P 500 Time-Varying Membership

> **Main universe: point-in-time S&P 500, 2010-2025.** This directory stores the
> membership + CIK-link artifacts used by SP500Vol-Bench, built from WRDS CRSP.

Membership tables use columns `(ticker, cik, member_from, member_to, source)`
plus a `permno` column carried for provenance. `member_to` is empty/NaT for
current members.

## Artifacts (in-repo)

- `sp500_membership.parquet` — point-in-time S&P 500 membership intervals.
- `crsp_cik_links.parquet` — point-in-time PERMNO→CIK link table (from CCM).

The bulk daily-returns store lives off-repo under
`$SP500VOL_DATA_ROOT/market/crsp/` (built by the same script).

## Source & Provenance

Built from WRDS:

- **CRSP S&P 500 Index Constituents** (CIZ): survivorship-free point-in-time
  membership (`MbrStartDt`/`MbrEndDt`) with daily prices/returns (`DlyRet`).
- **CRSP/Compustat-Merged (CCM)** linking table: PERMNO→CIK with link windows,
  resolved point-in-time (a PERMNO can map to different CIKs across M&A
  boundaries, e.g. Baker Hughes Inc→Co).

CRSP `DlyRet` is the split/dividend-adjusted *total* return, so realised-vol
labels use `log_return = log1p(DlyRet)` directly — no price differencing, no
yfinance/Tiingo.

## Regenerate

```bash
python scripts/ingest_wrds.py \
    --wrds-dir "/path/to/WRDS S&P500 2010-2025"
```

Reads the WRDS zips (CRSP constituents, CCM, CRSP names, Compustat company,
CRSP daily) and emits `sp500_membership.parquet` + `crsp_cik_links.parquet`
(in-repo) and the daily-returns store (off-repo under `$SP500VOL_DATA_ROOT`).

## Coverage (2010-2025)

- 914 membership intervals, 832 distinct PERMNO, 884 distinct tickers.
- Year-end active members: 2010=500, 2013=499, 2016=504, 2019=503, 2022=502,
  2025=503 (≈500; endpoint counts vary slightly with CRSP membership dates).
- Survivorship-free: removed/delisted members retain full price history.
- Returns store: 2,023,421 rows, 871 tickers, `log_return` NaN-free.

## CIK Resolution

PERMNO→CIK is a **point-in-time** join on CCM link windows
(`LINKTYPE ∈ {LC,LU,LS,LX}`, `LINKPRIM ∈ {P,C,J}`), so it handles:

- M&A successions (different CIK per period, e.g. Baker Hughes Inc `0000808362`
  → Co `0001701605`) — never a flat dict.
- Dual-class / rename continuations where the old GVKEY's CIK is blank (carried
  across via GVKEY- and PERMNO-level backfill: GOOG↔GOOGL,
  Alcoa→Arconic→Howmet `0000004281`, MetroPCS→T-Mobile `0001283699`).
- Audited manual overrides for known historical registrant successions where
  CCM's 2026 company header points a 2010-2025 PERMNO at a successor CIK
  (e.g. CBS/PARA `0000813828`, MYL `0000069499`→`0001623613`, XRX
  `0000108772`→`0001770450`, CCE `0000804055`→`0001491675`, old DOW
  `0000029915`) plus Dr Pepper Snapple (PERMNO 92618) → `0001418135`
  (blank in CCM).

Coverage: 832/833 real PERMNOs resolve to a CIK; the unresolved row is DVMT, a
1-trading-day spin-off artifact with no usable label window.

## Filing Full Text

Filings are fetched from EDGAR by the CRSP-resolved CIK; WRDS does not carry the
disclosure text. See `scripts/build_dataset.py`. The CRSP availability of the
universe can be pre-checked with `scripts/preflight_full_ingestion.py`.
