# Split definition (chronological, no look-ahead)

Assignment is by each filing's **effective trading day** (after-hours filings roll
forward to the next trading day), day-inclusive:

- **train**: effective trading day `<= 2019-12-31`
- **val**: `2020-01-01 .. 2021-12-31` (contains the COVID volatility regime)
- **test**: `>= 2022-01-01`

Filing counts by disclosure subset and split:

| subset | train | val | test |
|---|---:|---:|---:|
| long_form (10-K / 10-Q) | 19,668 | 3,963 | 7,970 |
| event_driven (8-K) | 73,088 | 14,266 | 25,174 |

The `split` column in `accession_index.csv` is fully reproducible from
`effective_trading_day` alone, so any third party can re-derive the exact
train/validation/test partition without access to CRSP data.
