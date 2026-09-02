# T6-16 — Train/test filer overlap (released public index only)

Counts are DISTINCT FILER CIKs, not PERMNOs, tickers or membership intervals. Source: `release/accession_index.csv`.

| stratum | test filings | from an already-read filer | % | distinct CIKs | also in train | % |
|---|---|---|---|---|---|---|
| all test filings | 33,144 | 29,597 | 89.3 | 558 | 463 | 83.0 |
| 8-K | 25,174 | 22,596 | 89.8 | 558 | 463 | 83.0 |
| long-form (10-K and 10-Q) | 7,970 | 7,001 | 87.8 | 557 | 463 | 83.1 |
