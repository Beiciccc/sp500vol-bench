# Data directory

This directory is **gitignored** for raw / interim / market / processed data
(too large; reproducible from the pipeline). Only `sample/` and `.gitkeep`
markers are committed.

## Layout

```
data/
├── raw/                 # raw EDGAR HTML / iXBRL (~50-80 GB at full scale)
│   ├── 10-K/{cik}/{accession}.html
│   ├── 10-Q/{cik}/{accession}.html
│   └── 8-K/{cik}/{accession}.html
├── interim/             # parsed text + metadata JSON per filing
├── market/              # OHLCV by ticker (parquet)
├── processed/           # final aligned dataset (parquet, partitioned by year)
│   ├── sample/          # output of build_dataset --config configs/data/sample.yaml
│   └── full/            # output of build_dataset --config configs/data/full.yaml
├── universe/            # time-varying S&P 500 membership table
│   └── sp500_membership.parquet
└── sample/              # 5 firms × 1 year (small enough to commit)
    └── README.md
```

## Regeneration

Raw EDGAR data is not redistributed (SEC public domain but bulky). To regenerate:

```bash
cp .env.example .env       # fill in EDGAR_USER_AGENT
make data-sample            # ~30 min, ~200 MB
# OR
make data                   # ~2 weeks, ~50-80 GB
```

## Sizes (rough estimates)

| Path | Sample | Full (SP500Vol-Bench) |
|---|---|---|
| `data/raw/` | ~200 MB | ~50-80 GB |
| `data/interim/` | ~50 MB | ~10-15 GB |
| `data/market/` | ~5 MB | ~500 MB |
| `data/processed/` | ~20 MB | ~5-10 GB |

## Provenance

Every parquet file in `data/processed/` carries a `_meta.json` sidecar with:
- Source EDGAR accession numbers
- Fetch timestamps
- Pipeline config hash
- Git SHA of the build
