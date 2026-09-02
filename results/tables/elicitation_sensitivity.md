# Elicitation sensitivity of the C6 prompted-LLM increment

## [1] Repeat-decode determinism (identical config, n=4000)

| horizon | exact-equal | Spearman | mean rel diff |
|---|--:|--:|--:|
| vol_5d | 0.970 | 0.9639 | 0.59% |
| vol_10d | 0.945 | 0.9538 | 0.68% |
| vol_20d | 0.936 | 0.9244 | 1.08% |

temp-0 batched vLLM decoding is near- but not bit-deterministic: 94-97% of forecasts identical across repeats, rank agreement rho>0.92.

## [2] Cross-template agreement vs baseline (vol_10d Spearman)

| arm | n | Spearman | parse_ok |
|---|--:|--:|--:|
| para1 | 4000 | 0.578 | 1.000 |
| para2 | 4000 | 0.413 | 1.000 |
| think | 4000 | 0.391 | 1.000 |

Individual-level forecasts are strongly prompt-dependent (rho 0.39-0.58).

## [3] Per-arm M1 increment (vs recalibrated HAR, day-clustered DM)

| disc | arm | h5 | h10 | h20 |
|---|---|--:|--:|--:|
| long_form | base_rep1 | +0.20% | +4.20%** | +0.44% |
| long_form | rep2 | +0.21%** | +4.21%** | +0.58% |
| long_form | para1 | +2.32%** | +5.95% | +3.14%** |
| long_form | para2 | -0.44% | -2.16% | -3.77% |
| long_form | think | -0.15% | +1.55% | +1.32% |
| event_driven | base_rep1 | +0.82%** | +0.45% | +0.10% |
| event_driven | rep2 | +0.89%** | +0.62%** | +0.17% |
| event_driven | para1 | +1.13%** | +1.58%** | +1.58%** |
| event_driven | para2 | +0.81%** | +0.44% | -0.16% |
| event_driven | think | +0.63%** | +0.39% | -0.59% |

**Verdict:** the event-driven h5 residual is DIRECTION-STABLE across all five arms (positive everywhere, significant in most), while the long-form increment FLIPS SIGN under paraphrase (para1 positive vs para2 negative) and is not rescued by thinking mode. The residual 8-K signal is elicitation-robust; the long-form apparent signal is prompt-fragile — consistent with it failing the firm-identity and maximal-price controls. "The near-null is just a bad prompt" is answered: no tested elicitation produces a robust long-form gain.
