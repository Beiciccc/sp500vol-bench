# E1 — Generative-LLM vol forecaster (C6_llmtext + D4_llmfused): GPU-box runbook

Everything below is ready to execute the moment a rented GPU box is up. No training —
zero-shot batch inference over **51,229 val+test filings x 2 prompt variants =
102,458 generations** (one generation covers all three horizons 5/10/20d).

## 0. Rig

| | |
|---|---|
| Recommended | **single 48GB card** — L20 / RTX A6000 / A6000 Ada / A100-40G(ok) / A100-80G |
| Primary model | `Qwen/Qwen3-32B-AWQ` (int4 AWQ, ~19GB weights — fits 48GB with vLLM at 8k ctx) |
| Fallback model | `Qwen/Qwen3-14B` (bf16, ~30GB — use if AWQ kernel issues, or for a 2x faster run) |
| CUDA | >= 12.1, driver >= 535 |
| Disk | **~150GB**: models ~50GB (32B-AWQ + 14B fallback + HF cache), text cache 3.3GB, manifest 0.14GB, raw outputs <1GB, venv/torch ~15GB, headroom |
| RAM | >= 64GB recommended (text streaming is chunked; peak resident prompts ~2-3GB) |

Note on names: the task spec says "Qwen3-32B-Instruct-AWQ" — the actual HF repo id is
`Qwen/Qwen3-32B-AWQ` (Qwen3 has no separate -Instruct variant; the runner disables
thinking mode via `enable_thinking=False`, which yields instruct behaviour). Fallback
"Qwen3-14B-Instruct" = `Qwen/Qwen3-14B`.

## 1. Ship these files to the box

```
scripts/experiments/e1_llm_forecast/        # this directory (prompt.py, run_inference.py)
results/e1_llm_forecast/manifest_valtest.parquet          # 133MB, built locally already
/path/to/data-root/sp500vol-data/processed/_text_cache/filing_texts.parquet   # 3.3GB
```

If the manifest needs rebuilding: `.venv/bin/python scripts/experiments/e1_llm_forecast/run_inference.py build-manifest`
(run on the Mac — it needs `results/runs/A2_har_rv_full_combined_seed2026/` and the aligned parquet).

On the box, place the cache under any root and export it:

```bash
export SP500VOL_DATA_ROOT=/data/sp500vol-data     # cache at $ROOT/processed/_text_cache/filing_texts.parquet
mkdir -p /data/sp500vol-data/processed/_text_cache
# rsync/scp filing_texts.parquet + manifest_valtest.parquet + this dir
```

## 2. Environment (use the HF mirror if huggingface.co is unreachable)

```bash
export HF_ENDPOINT=https://hf-mirror.com          # HF mirror fallback
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv ~/e1venv --python 3.11 && source ~/e1venv/bin/activate
uv pip install "vllm>=0.8" pandas pyarrow scipy "transformers>=4.51"   # 4.51+ for Qwen3
# pre-download the model (resumable):
uv pip install "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-32B-AWQ --local-dir-use-symlinks False
```

Sanity: `python -c "import vllm; print(vllm.__version__)"`.

## 3. Sequence: pilot -> go/no-go -> full

Run everything inside `tmux`. The runner **checkpoints a parquet part-file every 500
filings and resumes automatically** (skips already-done `(text_path, variant)` pairs),
so crashes/preemptions only cost the current chunk.

### 3.1 Pilot (500 test filings, both variants; ~30-60 min)

```bash
cd <shipped repo dir>
python scripts/experiments/e1_llm_forecast/run_inference.py run \
    --manifest results/e1_llm_forecast/manifest_valtest.parquet \
    --model Qwen/Qwen3-32B-AWQ --variant both \
    --pilot 500 --out-dir /data/e1_raw_pilot --checkpoint-every 250
```

### 3.2 Go/no-go

Copy `/data/e1_raw_pilot` back to the Mac (a few MB) and run the gate there
(it needs `results/runs/A1_hv_*`):

```bash
.venv/bin/python scripts/experiments/e1_llm_forecast/postprocess.py pilot-eval \
    --raw-dir <copied e1_raw_pilot> --manifest results/e1_llm_forecast/manifest_valtest.parquet
```

Gates and iteration guidance: **`go_no_go.md`** in this directory. Do NOT start the
full run until both variants print `"GO": true` (or a documented prompt iteration
decision is made).

### 3.3 Full run (both variants, all 51,229 filings)

```bash
python scripts/experiments/e1_llm_forecast/run_inference.py run \
    --manifest results/e1_llm_forecast/manifest_valtest.parquet \
    --model Qwen/Qwen3-32B-AWQ --variant both \
    --out-dir /data/e1_raw_full --checkpoint-every 500
```

Optional split across 2 boxes: `--subset event_driven` (39,322 8-Ks, short prompts) on
one, `--subset long_form` (11,907 10-K/Qs, long prompts) on the other; merge the two
`part-*.parquet` sets into one dir afterwards (filenames may collide — rename parts
from one box, e.g. `for f in part-*; do mv $f lf-$f; done`; postprocess globs `part-*`
so keep the prefixchange consistent: use `--out-dir` subdirs and copy with new names).

### 3.4 ETA

Prompt volume measured from the manifest: **~132M prompt tokens per variant**
(8-K median ~600 tok; 10-K/Q capped at ~6k tok), output ~120 tok x 102k generations.
vLLM offline throughput (prefill-dominated), rough planning numbers:

| Setup | tok/s (prompt) | per variant | both variants |
|---|---|---|---|
| Qwen3-32B-AWQ, A6000/L20 48G | ~2.5-4k | ~9-15h | **~18-30h** |
| Qwen3-32B-AWQ, A100-80G | ~5-8k | ~5-7h | ~10-14h |
| Qwen3-14B bf16, 48G | ~6-10k | ~4-6h | ~8-12h |

The runner prints live gen/s and ETA every checkpoint — recalibrate from the first
30 min. Budget 1.5x for safety before renting by the hour.

## 4. Bring results home and integrate

Copy `/data/e1_raw_full` (<1GB) back to the Mac, then:

```bash
.venv/bin/python scripts/experiments/e1_llm_forecast/postprocess.py build-runs \
    --raw-dir <copied e1_raw_full>          # writes results/runs/{C6_llmtext,D4_llmfused}_full_{long_form,event_driven,combined}_seed2026
```

This emits standard run dirs (schema verified identical to B2; predictions are
**val+test only** — no train rows, documented in each config.json; downstream M1
`forecast_combination.log_combo` needs only val+test). Then run the M1 incremental-value
evaluation against A2 as for any other text model.

## 5. Knobs

* `--max-model-len 8192` (default) — prompts capped at ~6k tokens by `prompt.py`.
* `--max-tokens 120` — JSON needs ~40; headroom for stray prose.
* `--tp 2` if you end up on 2 smaller cards.
* Guided JSON decoding is used automatically when the installed vLLM exposes
  `GuidedDecodingParams`; otherwise plain decoding + robust regex parse + one retry
  pass (both paths are implemented and tested with the mock).
* Kill/restart at any time; resume is automatic from `--out-dir`.
