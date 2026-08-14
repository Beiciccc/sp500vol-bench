# Raw prompted-LLM generations

1221 parquet shards, 608,221 generations, covering every
prompted-LLM arm in the paper: the C6 primary (long-form and
event-driven), the Llama-70B date+ticker contamination rider,
the elicitation-sensitivity arms (paraphrase, think, two repeats),
and the cross-family probes (Yi-1.5-34B, Phi-4-14B, Mistral-24B,
Gemma-3-27B, Llama-3.1-70B) with their seed replicates.

The D4 fused arm consumes the C6 generations above and produces none of
its own. The Qwen date-only and date+ticker probe generations were not
retained; their results survive in the released aggregate evidence tables.

Schema: `text_path` (data-root-relative path to the SEC filing, and
the join key into `../accession_index.csv`), `variant`, `model_name`,
`raw_output` (the model's verbatim JSON string), the three parsed
horizon forecasts `vol_5d/10d/20d`, `parse_ok`, `retry_used`,
`prompt_chars`, `excerpt_source`.

No CRSP-derived value appears in any column: prompts consume SEC
filing text only, so these generations are releasable in full.
