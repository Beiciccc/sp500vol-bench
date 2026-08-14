# Rows excluded from this release

8 generation(s) were dropped because their `raw_output`
contains non-English text. The model emitted it inside its own
chain of thought; the submission package is English-only, and
rewriting the string would mean the column is no longer the raw
output. The rows are therefore removed rather than edited, and
listed here.

Affected arm(s): raw_elic_think

| shard | column | rows dropped | rows in shard |
|---|---|---|---|
| `raw_elic_think/part-00001.parquet` | `raw_output` | 2 | 500 |
| `raw_elic_think/part-00002.parquet` | `raw_output` | 2 | 500 |
| `raw_elic_think/part-00003.parquet` | `raw_output` | 1 | 500 |
| `raw_elic_think/part-00005.parquet` | `raw_output` | 1 | 500 |
| `raw_elic_think/part-00007.parquet` | `raw_output` | 2 | 500 |

The parsed forecasts for these filings still enter every
aggregate table in `../aggregate_results/`; only the verbatim
generation strings are withheld.
