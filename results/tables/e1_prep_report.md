# E1-PREP report — generative-LLM forecaster (C6_llmtext / D4_llmfused), GPU-ready

Date: 2026-07-02. Status: **READY** — all code written, mocked end-to-end validated,
manifest built. The run starts the moment a GPU box is up (see
`scripts/experiments/e1_llm_forecast/README_BOX.md`).

## Deliverables

| File | Role |
|---|---|
| `scripts/experiments/e1_llm_forecast/prompt.py` | Prompt templates (c6_text / d4_fused), excerpt policy, JSON schema, robust output parser |
| `scripts/experiments/e1_llm_forecast/run_inference.py` | `build-manifest` (local) + `run` (box): vLLM offline batch, temp=0, max_tokens 120, guided-JSON when available, regex parse + 1 retry pass, checkpoint every 500, resumable, `--pilot N` / `--limit N` / `--mock` |
| `scripts/experiments/e1_llm_forecast/postprocess.py` | `build-runs` → standard run dirs per NEW-MODEL convention; `pilot-eval` → go/no-go readout |
| `scripts/experiments/e1_llm_forecast/README_BOX.md` | Exact box setup: rig, HF mirror, install, pilot→gate→full commands, ETA, disk |
| `scripts/experiments/e1_llm_forecast/go_no_go.md` | 4 pilot gates + evaluation snippet + prompt-iteration playbook |
| `results/e1_llm_forecast/manifest_valtest.parquet` | 51,229 val+test filings (built, 133MB) — ship with the 3.3GB text cache |

## Design facts (all verified against the repo, not assumed)

* **Splits are constant per filing across horizons** (0 / 142,855 filings mixed in A2
  combined) → one manifest row and ONE LLM call per filing covers h=5/10/20.
* **Manifest = A2 combined val+test**: 51,229 filings = 18,169 val + 33,060 test;
  39,322 8-K + 8,934 10-Q + 2,973 10-K; 0 missing joins to aligned_filings.
  combined = exact union of event_driven + long_form, so a single inference pass
  serves all three disclosure run dirs.
* **No train inference needed**: the LLM is never fit; M1 (`fc.log_combo`) uses val
  (weights) + test (evaluation) only. Run dirs therefore contain val+test rows only
  (documented in each config.json); metrics.json has 6 rows (val,test × 5/10/20).
* **Metrics convention reproduced exactly**: recomputed B2 combined test h=5 →
  n=33,060, mae 0.121541, rmse 0.190456, r2 −0.202111, qlike(variance-unit) 1.521285
  — byte-identical to the stored metrics.json row. postprocess uses this exact code.
* **Excerpt policy** (cap ~6k prompt tokens): 8-K full text (median ~600 tok);
  10-K item_1a+item_7+item_7a and 10-Q part_i_item_2+part_ii_item_1a from
  sections_json **only if combined length ≥ 2,000 chars** — many sections_json entries
  are TOC stubs (10-K median section value = 70 chars) — else head-truncate full text.
* **Prompt volume**: ~132M prompt tokens per variant (measured from manifest token
  counts, capped) → ETA ~18–30h both variants on a 48GB card with Qwen3-32B-AWQ,
  ~8–12h with Qwen3-14B bf16 (see README_BOX.md table).

## Mocked end-to-end validation (50 filings, both variants, run on this Mac)

Mock generator: `vol = clip(rv_22d · exp(N(0,0.3)))`, 4% deliberate garbage outputs.

1. `run --mock --limit 50` → 100 generations; **retry path exercised: 4/100 garbage
   first-pass, all recovered → parse_ok 100%**; checkpoint part-files written;
   immediate re-run detects `resume: 100 pairs done / nothing to do` (resumability OK).
2. `postprocess build-runs` → 6 run dirs (C6_llmtext, D4_llmfused × 3 disclosures) in a
   scratch out-root. **Predictions schema = existing B2 run exactly** (column order and
   dtypes, 0 mismatches); metrics.json = 6 rows as specced; clip rate 0.
3. **fc.log_combo consumability against A2 (combined, pooled horizons)** — joined on
   (ticker, accession, horizon_days), 100% key & split agreement with A2:

   | run (MOCK) | n_val | n_test | g_text | QLIKE(vol) f_R | f_U | raw HAR |
   |---|---|---|---|---|---|---|
   | C6_llmtext | 51 | 99 | +0.121 | 0.0554 | 0.0550 | 0.0550 |
   | D4_llmfused | 51 | 99 | −0.183 | 0.0554 | 0.0619 | 0.0550 |

   Numbers are **mock-only sanity** (mock ≈ noisy rv_22d, so no genuine text signal is
   expected); the point is the pipeline runs unmodified through the M1 combiner.
4. `run --pilot 60 --mock` + `postprocess pilot-eval` → gate readout prints all four
   gates per variant with a final GO flag (mock passes trivially).

## Guard rails

* Predictions clipped to **[0.03, 3.0] annualized**, clip-rate reported in config.json.
* Parse failures after retry: default **fill with feature_rv_22d** (shrink-to-price,
  counted in config.json `stats.parse_fail_*`); `--on-missing drop` available.
  Gate 1 (≥95% parse) keeps the fill ≤5%.
* Qwen3 thinking mode explicitly disabled (`enable_thinking=False`) — else the 120-token
  budget is consumed by `<think>`.

## Open decisions (need a call before/at pilot time)

1. **Model final pick**: Qwen/Qwen3-32B-AWQ (primary) vs Qwen/Qwen3-14B (2–2.5× faster,
   possibly enough) — decide after pilot; pilot both if box time allows (~1h extra).
   Note the task-spec names ("-Instruct") don't exist for Qwen3; runner handles this.
2. **c6_text soft-fail waiver**: if text-only fails gates 3/4 but d4_fused passes,
   proceed with both? (go_no_go.md recommends yes with documented waiver, since M1 is
   the endpoint and the M1 memory says text-alone losing is expected.)
3. **Fill policy for the paper**: rv22-fill (keeps N aligned, slightly price-contaminates
   C6 on ≤5% rows) vs drop (clean text-only claim, unbalanced panel). Default rv22-fill;
   flag in the writeup either way.
4. **Seeds**: decoding is deterministic-INTENT (temperature=0, single decode pass), but
   NOT bit-reproducible — vLLM batched bf16 decoding is nondeterministic across batch
   compositions, and the retry pass samples at temp 0.2 — so C6/D4 are SINGLE-DECODE
   runs, not verified seed-invariant; only seed2026 dirs are written, and aggregation
   scripts treat them as single-run. [Corrected 2026-07-05; original wrongly claimed
   "temperature=0 ⇒ seed-invariant". Repeat-decode agreement is unmeasured pending P1-2a.]
5. **metrics.json has 6 rows, not 9** (no train split by design) — any downstream script
   that hard-codes 9 rows must special-case C6/D4.

## Not done here (needs the GPU box)

Real pilot (500 filings), go/no-go decision, full 102,458-generation run, and the real
`build-runs` into `results/runs/` — mock artifacts were kept OUT of `results/runs/`
(scratch dir only) so no fake run dirs pollute the matrix.
