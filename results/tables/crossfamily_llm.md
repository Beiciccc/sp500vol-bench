# Cross-family replication of the prompted-LLM increment (P2)

Identical manifest/prompts/protocol; day-clustered DM. rel% > 0 = text lowers QLIKE vs the reference; `**` = clustered DM<0, p<.05. Yi-1.5-34B has a 4K context (vs Qwen3's 8K): binding for long-form excerpts, NOT binding for 8-K (median ~930 tokens) — so the event_driven non-replication cannot be a context artefact.

**HEADLINE: the prompted-LLM increment does NOT replicate across families.** Yi-34B shows no significant positive increment in any cell (long_form negative everywhere; event_driven ~0), while Qwen3-32B was positive-significant in 6/6 vs single-HAR. The prompted-LLM residual is family-specific, further supporting the near-null verdict.

| disc | family | h | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) |
|---|---|--:|--:|--:|--:|--:|
| long_form | qwen3_32b | 5 | +1.79%** | -6.31 | -0.14% | +5.16 |
| long_form | qwen3_32b | 10 | +2.25%** | -7.92 | -0.17% | +6.83 |
| long_form | qwen3_32b | 20 | +0.27%** | -3.23 | +0.02%** | -3.79 |
| long_form | yi_34b | 5 | -0.64% | +2.56 | -0.21% | +1.83 |
| long_form | yi_34b | 10 | -2.71% | +4.27 | -1.61% | +4.38 |
| long_form | yi_34b | 20 | -9.86% | +5.68 | -6.05% | +5.55 |
| event_driven | qwen3_32b | 5 | +1.21%** | -5.04 | +0.45%** | -5.26 |
| event_driven | qwen3_32b | 10 | +1.00%** | -3.76 | +0.25%** | -5.16 |
| event_driven | qwen3_32b | 20 | +0.66%** | -1.98 | +0.20%** | -3.79 |
| event_driven | yi_34b | 5 | +0.37% | -0.60 | +0.22% | -1.00 |
| event_driven | yi_34b | 10 | +0.07% | +0.60 | +0.07% | -0.13 |
| event_driven | yi_34b | 20 | -0.62% | +2.54 | -0.07% | +1.58 |
| event_driven | phi4_14b | 5 | +0.38% | -1.96 | +0.20% | -1.68 |
| event_driven | phi4_14b | 10 | +0.18% | -0.38 | +0.14% | -1.16 |
| event_driven | phi4_14b | 20 | -0.12% | +1.22 | +0.00% | -0.07 |
