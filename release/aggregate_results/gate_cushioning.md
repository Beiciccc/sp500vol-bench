# Gate behavioural evidence (G3): validation-period cushioning, long-form

D2 gated fusion measurably cushions the COVID (2020-21) regime collapse that sinks text-only models -- the behavioural signature of a gate that down-weights the uninformative text branch. Direct internal gate-weight readout needs a GPU forward pass and is deferred to the next rental (local Torch environment is broken).

| model | val R2 h5 | val R2 h10 | val R2 h20 |
|---|---|---|---|
| C1_bert_s1 | -0.396 | -0.407 | -0.400 |
| C2_finbert_s1 | -0.340 | -0.277 | -0.354 |
| C3_roberta_s1 | -0.335 | -0.331 | -0.353 |
| C4_longformer | -0.249 | -0.315 | -0.367 |
| D1_concat_mlp | +0.000 | -0.052 | -0.195 |
| D2_gated_fusion | +0.108 | -0.081 | -0.188 |
| A2_har_rv | +0.077 | +0.032 | -0.074 |

**Cushioning (D2 minus text-only mean val R2):** h5 +0.438, h10 +0.251, h20 +0.180. D2 also significantly beats naive concatenation D1 at long-form h5 (DM -4.58), isolating the gating mechanism from mere feature concatenation.