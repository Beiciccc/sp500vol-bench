# Yelp second domain — multi-arm cascade (REAL, 3 text arms + identity probe)

8,474 businesses, 407,385 business-months, 2005-2022; train<=2016/val 2017/test 2018-21.
Combiner = AR-ridge + text vs recalibrated-AR reference; residual = combiner vs AR + entity-mean
(zero-text business identity control); month-clustered + two-way DM; 20-seed label-shuffle placebo.

| arm | h | text-alone vs recal AR | AR+text combiner | identity-ctrl residual | residual DM (2-way p) | placebo mean DM (p) |
|---|---|---|---|---|---|---|
| TF-IDF ridge | 1m | -24.7% | +0.346% | +0.365% | -12.65 (2.3e-11) | +0.98 (0.310) |
| TF-IDF ridge | 3m | -35.3% | +0.871% | +0.612% | -20.54 (3.4e-17) | +0.17 (0.520) |
| Qwen3-Emb-8B | 1m | -19.9% | +0.677% | +0.376% | -16.57 (4.4e-15) | +0.76 (0.367) |
| Qwen3-Emb-8B | 3m | -28.6% | +1.513% | +0.786% | -13.97 (1.1e-14) | +0.84 (0.470) |
| Llama-70B prompt | 1m | -25.5% | +0.163% | +0.025% | +0.50 (6.4e-01) | +0.86 (0.346) |
| Llama-70B prompt | 3m | -38.5% | +0.241% | +0.176% | -2.30 (8.4e-02) | +1.01 (0.261) |
| 70B probe (0-text) | 1m | -62.8% | +0.243% | -0.056% | +5.65 (7.4e-04) | +1.19 (0.163) |
| 70B probe (0-text) | 3m | -96.2% | +0.248% | +0.057% | -2.82 (3.4e-02) | +0.45 (0.109) |

## Reading
- All three real text arms LOSE text-alone to the recalibrated AR baseline (-20 to -38%);
  the zero-content probe loses hardest (-63/-96%), as it has no review text.
- The prompted-LLM (Llama-70B) apparent combiner gain is essentially IDENTITY: the zero-content
  probe (business name+city+categories+month) reproduces 149%/103% of it, and after the entity-mean
  control its residual is +0.025%/+0.176% (h=1 DM +0.50 n.s.; h=3 marginal p_2way .084).
- The CONTENT arms (TF-IDF, frozen Qwen3 embeddings) leave a small, placebo-clean, strongly
  significant identity-controlled residual (+0.37-0.79%, two-way p 1e-11..1e-17): consumer review
  TEXT carries genuine forward-rating signal beyond the business's own rating history.
- Cross-domain contrast: SEC's surviving residual is the PROMPTED-LLM reading event 8-Ks (content);
  Yelp's surviving residual is the CLASSICAL/embedding arms (content), while the prompted LLM's gain
  is identity. Same protocol, domain-specific split of identity vs content — a measurement instrument,
  not a null-manufacturing device.
