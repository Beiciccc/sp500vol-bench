#!/usr/bin/env bash
# =============================================================================
# ROW 16 — 3-seed ensemble of the THIRD-family Mistral-Small-24B-Instruct-2501
# (bf16, NO quantization) on the 8-K (event_driven) panel.
#
# Pre-registered in configs/prereg_residual_family_audit.md §B1 (prereg-rfa-v1.0):
# family set F = {Qwen3-32B primary (single seed), Llama-3.1-70B-AWQ (3-seed,
# row15), THIS (3-seed)}. Protocol byte-identical to C6/llama70: same prompt /
# guided-JSON / clip[0.03,3.0] / retry stack in scripts/experiments/e1_llm_forecast.
#
# ADAPTED FROM scripts/experiments/row15_llama70_ensemble/launch.sh.
# Deliberate deviations from row15 (everything else copied verbatim):
#   1. MODEL    -> mistralai/Mistral-Small-24B-Instruct-2501, bf16, no quant.
#                  ~47 GB weights = ~23.5 GB/GPU at TP2; an A100-80GB pair keeps
#                  ample KV headroom at the default gpu_memory_utilization 0.92.
#   2. SEEDS    -> all three (2026 2027 2028) are runnable: unlike llama70, NO
#                  seed pre-exists. Suffix convention mirrors llama70 exactly:
#                  seed 2026 -> _mistral24 (no _s), others -> _mistral24_s<seed>.
#   3. MODES    -> first positional arg:
#                    single    inference + postprocess build-runs for the seeds
#                              in $SEEDS_TO_RUN on the GPU pair $GPUS (both env)
#                    ensemble  CPU-only 3-seed ensemble build -> _ROW16_DONE
#                  row15 was one monolithic serial run; row16 seeds run
#                  CONCURRENTLY on different pairs under an orchestrator, so the
#                  per-seed step and the ensemble step must be separable.
#   4. CUDA pin -> single mode exports CUDA_VISIBLE_DEVICES="$GPUS". row15 never
#                  pinned devices (sole tenant; vLLM took GPUs 0,1 by default and
#                  $GPUS was only the watchdog's `nvidia-smi -i` argument). With
#                  two pairs active at once pinning is mandatory. nvidia-smi -i
#                  keeps using PHYSICAL indices, so the watchdog is unchanged.
#   5. clean_shm-> age-filtered (only /dev/shm entries older than 15 min).
#                  row15's blanket `rm -f /dev/shm/psm_* /dev/shm/sem.*` is
#                  unsafe under pair-concurrency: unlinking a just-created
#                  psm_*/sem.* of the OTHER pair's vLLM before its spawn workers
#                  attach kills that run. Old segments (leaked, or belonging to
#                  long-running healthy processes whose workers attached long
#                  ago) are safe to unlink.
#   6. markers  -> per-seed _ROW16_S<seed>_DONE written by single mode; overall
#                  _ROW16_DONE / _ROW16_FAILED written ONLY by ensemble mode (a
#                  concurrent per-seed invocation cannot know overall status, so
#                  row15's launch-time rm of the overall markers moved there too).
#   7. ensemble -> requires ALL THREE seed run dirs (row15 tolerated >=2 because
#                  seed 2026 pre-existed); aborts with _ROW16_FAILED otherwise.
#
# ENSEMBLE CONVENTION (unchanged from row15 / the declared primary): per-row
# ARITHMETIC mean of prediction_realised_vol, inner join on (ticker, accession,
# horizon_days) — identical to m1_ensemble_primary.ensemble_text and to the
# on-disk C6_llmtext_llama70ens (see its config.json). NOTE: prereg §B0 gate G5
# words the check as a LOG-space mean; the committed llama70ens is ARITHMETIC.
# We copy row15 faithfully so both families share one convention — reconcile G5
# in the prereg/analysis layer, NOT by forking the convention here.
#
# TEMPERATURE PROTOCOL UNCHANGED (run_inference.py: temperature=0 first pass,
# 0.2 retry pass on parse failures, guided JSON when available; --seed is NOT
# plumbed into vLLM). Seeds 2026/2027/2028 therefore differ only through
# vLLM/TP2 kernel non-determinism: a REPRODUCIBILITY-JITTER ensemble, not a
# stochastic-decoding one. prereg §B1 pins exactly this semantics — do NOT add
# sampling changes here.
#
# GPU-HOUR ESTIMATE (honest): 24B bf16 at TP2 should undercut the 70B AWQ-INT4
# (6-7 h/seed); expect roughly 3-5 h/seed on the same ~39k-filing event_driven
# panel, plus ~5 min model load per (re)start. Runs are RESUMABLE
# (checkpoint-every 500): a watchdog kill/retry resumes from the last
# part-*.parquet, not from scratch.
#
# USAGE (on the box; any cwd — the script cd's to $REPO):
#   GPUS="0,1" SEEDS_TO_RUN="2026" bash scripts/experiments/row16_mistral24_ensemble/launch.sh single
#   bash scripts/experiments/row16_mistral24_ensemble/launch.sh ensemble
# =============================================================================
set -uo pipefail

MODE="${1:-}"

# ---- paths / env (all overridable) — identical to row15 ---------------------
REPO="${REPO:-/root/rivermind-data/repo}"
PY="${PY:-/root/rivermind-data/venvs/main/bin/python}"
export SP500VOL_DATA_ROOT="${SP500VOL_DATA_ROOT:-/root/rivermind-data/sp500vol-data}"
export HF_HOME="${HF_HOME:-/root/rivermind-data/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1        # box has no direct egress; resolve from cache ONLY
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn          # CRITICAL: fork deadlocks w/ TP2

MODEL="mistralai/Mistral-Small-24B-Instruct-2501"
# SNAP_OVERRIDE: bypass hub completeness check (consolidated.safetensors deliberately excluded)
SNAP=$(ls -d ${HF_HOME:-/root/rivermind-data/hf}/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/* 2>/dev/null | head -1)
[ -n "$SNAP" ] && MODEL="$SNAP"
# HFVIEW override: snapshot minus tekken.json/params.json so transformers resolves the
# FAST tokenizer (the mistral-common backend lacks .is_fast and crashes vLLM's check)
[ -d /root/rivermind-data/hf/mistral24_hfview ] && MODEL=/root/rivermind-data/hf/mistral24_hfview
MANIFEST="results/e1_llm_forecast/manifest_valtest.parquet"
SEEDS_TO_RUN="${SEEDS_TO_RUN:-2026 2027 2028}"      # all three: no seed pre-exists
ALL_SEEDS="2026 2027 2028"

STATE="results/experiments/row16_mistral24_ensemble"
LOGDIR="$STATE/logs"

# ---- watchdog tunables — identical to row15 ----------------------------------
GPUS="${GPUS:-0,1}"                                 # the TP2 pair (PHYSICAL indices)
GPU_BUSY_MIB="${GPU_BUSY_MIB:-500}"                 # >this MiB used == run is alive
HANG_SECS="${HANG_SECS:-480}"                       # 8 min of idle GPU == hung
POLL_SECS="${POLL_SECS:-30}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"

cd "$REPO" || { echo "FATAL: cannot cd to REPO=$REPO"; exit 1; }
mkdir -p "$STATE" "$LOGDIR"

# gpus= tag disambiguates interleaved lines when two pairs append concurrently
log() { echo "[row16 $(date -u +%H:%M:%S) gpus=$GPUS] $*" | tee -a "$LOGDIR/launch.log"; }

# _mistral24 for seed 2026 (no _s), _mistral24_s<seed> otherwise — mirrors llama70
suffix_of() { if [ "$1" = "2026" ]; then echo "_mistral24"; else echo "_mistral24_s$1"; fi; }
rundir_of() { echo "results/runs/C6_llmtext$(suffix_of "$1")_full_event_driven_seed2026"; }

# age-filtered: never unlink shm a concurrently STARTING vLLM just created (dev. 5)
clean_shm() {
  find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.*' \) -mmin +15 -delete 2>/dev/null || true
}

gpu_mem_max() {                                     # max MiB used across the TP pair
  local v
  v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null \
      | sort -n | tail -1)
  v="${v//[^0-9]/}"
  echo "${v:-0}"
}

kill_tree() {                                       # kill the run's whole process group
  local pid="$1"
  kill -TERM -"$pid" 2>/dev/null || true
  sleep 10
  kill -KILL -"$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  clean_shm
}

wait_gpu_free() {                                   # let VRAM drain before a retry
  local t=0
  while [ "$(gpu_mem_max)" -gt "$GPU_BUSY_MIB" ] && [ "$t" -lt 120 ]; do sleep 10; t=$((t+10)); done
}

# watchdog: return 0 if the child exited on its own (caller checks its rc),
#           return 1 if we had to kill it for hanging (GPU idle > HANG_SECS).
watchdog() {
  local pid="$1" seed="$2" zero_since=0 now used
  while kill -0 "$pid" 2>/dev/null; do
    used="$(gpu_mem_max)"; now="$(date +%s)"
    if [ "$used" -gt "$GPU_BUSY_MIB" ]; then
      zero_since=0
    else
      [ "$zero_since" -eq 0 ] && zero_since="$now"
      if [ $((now - zero_since)) -gt "$HANG_SECS" ]; then
        log "seed=$seed GPU idle (<=${GPU_BUSY_MIB} MiB) for >$((HANG_SECS/60)) min -> killing pgid=$pid"
        kill_tree "$pid"
        return 1
      fi
    fi
    sleep "$POLL_SECS"
  done
  return 0
}

# run one seed with watchdog + retries. return 0 on a clean inference, 1 if skipped.
run_seed() {
  local seed="$1"
  local outdir="results/e1_llm_forecast/raw$(suffix_of "$seed")"
  local attempt pid rc
  for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
    clean_shm
    log "seed=$seed attempt=$attempt/$MAX_ATTEMPTS -> $outdir"
    # setsid: own process group so the watchdog can kill the vLLM spawn workers too
    setsid "$PY" scripts/experiments/e1_llm_forecast/run_inference.py run \
        --manifest "$MANIFEST" \
        --model "$MODEL" \
        --variant c6_text \
        --subset event_driven \
        --tp 2 \
        --max-model-len 8192 \
        --seed "$seed" \
        --checkpoint-every 500 \
        --out-dir "$outdir" \
        > "$LOGDIR/s${seed}_attempt${attempt}.log" 2>&1 &
    pid=$!
    if watchdog "$pid" "$seed"; then
      wait "$pid"; rc=$?
      if [ "$rc" -eq 0 ]; then
        log "seed=$seed inference OK (rc=0)"
        return 0
      fi
      log "seed=$seed inference exited rc=$rc (attempt $attempt) — see s${seed}_attempt${attempt}.log"
    else
      log "seed=$seed watchdog killed a hung run (attempt $attempt)"
    fi
    wait_gpu_free
  done
  log "seed=$seed FAILED after $MAX_ATTEMPTS attempts — skipping"
  return 1
}

# ---- mode: single — inference + build-runs for $SEEDS_TO_RUN on $GPUS --------
mode_single() {
  # row16 pair-concurrency: pin the pair (row15 relied on being the sole tenant).
  # $GPUS stays in PHYSICAL indices, so the nvidia-smi -i watchdog is unaffected.
  export CUDA_VISIBLE_DEVICES="$GPUS"

  local seed rundir
  declare -a OK_SEEDS=()
  local ANY_SKIP=0
  for seed in $SEEDS_TO_RUN; do
    rundir="$(rundir_of "$seed")"
    if [ -f "$STATE/_ROW16_S${seed}_DONE" ] && [ -f "$rundir/predictions.parquet" ]; then
      log "seed=$seed already DONE ($rundir exists) — skipping"
      OK_SEEDS+=("$seed"); continue
    fi
    if run_seed "$seed"; then
      log "seed=$seed postprocess build-runs (suffix $(suffix_of "$seed"))"
      if "$PY" scripts/experiments/e1_llm_forecast/postprocess.py build-runs \
            --raw-dir "results/e1_llm_forecast/raw$(suffix_of "$seed")" \
            --model-suffix "$(suffix_of "$seed")" >> "$LOGDIR/s${seed}_postprocess.log" 2>&1 \
         && [ -f "$rundir/predictions.parquet" ]; then
        touch "$STATE/_ROW16_S${seed}_DONE"
        OK_SEEDS+=("$seed")
        log "seed=$seed marker _ROW16_S${seed}_DONE written"
      else
        log "seed=$seed postprocess FAILED — see s${seed}_postprocess.log"
        ANY_SKIP=1
      fi
    else
      ANY_SKIP=1
    fi
  done

  if [ "$ANY_SKIP" -eq 0 ]; then
    log "single mode COMPLETE — seeds ok: ${OK_SEEDS[*]:-none}"
  else
    log "single mode FAILED — seeds ok: ${OK_SEEDS[*]:-none}"
    exit 1
  fi
}

# ---- mode: ensemble — CPU-only 3-seed build (row15's tail, all-3 required) ---
mode_ensemble() {
  rm -f "$STATE/_ROW16_DONE" "$STATE/_ROW16_FAILED"  # recompute overall status

  # hard gate: ALL THREE seed run dirs must exist (stricter than row15's >=2)
  local seed rd missing=0
  for seed in $ALL_SEEDS; do
    rd="$(rundir_of "$seed")"
    if [ ! -f "$rd/predictions.parquet" ]; then
      log "ensemble: MISSING $rd/predictions.parquet"
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    touch "$STATE/_ROW16_FAILED"
    log "ROW16 FAILED — ensemble aborted, seed run dir(s) missing"
    exit 1
  fi

  # 3-seed ensemble build (per-observation ARITHMETIC mean — row15 code path,
  # names swapped llama70 -> mistral24, guard tightened to all-3 seeds).
  log "building 3-seed ensemble run dir C6_llmtext_mistral24ens_full_event_driven_seed2026"
  "$PY" - "$ALL_SEEDS" <<'PY' >> "$LOGDIR/ensemble_build.log" 2>&1
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "scripts/experiments/e1_llm_forecast")
import postprocess as pp  # reuse metrics_rows / PRED_COLS / qlike / clip

KEY = ["ticker", "accession", "horizon_days"]
DISC = "event_driven"
seeds = sys.argv[1].split()

def run_of(seed):
    base = "C6_llmtext_mistral24" if seed == "2026" else f"C6_llmtext_mistral24_s{seed}"
    return Path(f"results/runs/{base}_full_{DISC}_seed2026/predictions.parquet")

frames, used = [], []
base_df = None
for s in seeds:
    p = run_of(s)
    if not p.exists():
        print(f"[skip] seed {s}: {p} absent"); continue
    d = pd.read_parquet(p)
    if s == "2026":
        base_df = d.copy()                       # carries all PRED_COLS metadata
    frames.append(d[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": f"f{s}"}))
    used.append(s)

if base_df is None or len(used) < 3:
    print(f"FATAL: need seed 2026 base + ALL 3 seeds; have {used}"); sys.exit(3)

ens = frames[0]
for f in frames[1:]:
    ens = ens.merge(f, on=KEY, how="inner")      # inner: only obs present in every seed
ens["prediction_realised_vol"] = ens[[f"f{s}" for s in used]].mean(axis=1)
ens["prediction_realised_vol"] = ens["prediction_realised_vol"].clip(pp.CLIP_LO, pp.CLIP_HI)

out = base_df.drop(columns=["prediction_realised_vol"]).merge(
    ens[KEY + ["prediction_realised_vol"]], on=KEY, how="inner")
model_id = "C6_llmtext_mistral24ens"
run_id = f"{model_id}_full_{DISC}_seed2026"
out["model_id"], out["run_id"] = model_id, run_id
out = out[pp.PRED_COLS].reset_index(drop=True)

rd = Path(f"results/runs/{run_id}")
rd.mkdir(parents=True, exist_ok=True)
out.to_parquet(rd / "predictions.parquet", index=False)
(rd / "metrics.json").write_text(json.dumps(pp.metrics_rows(out, DISC), indent=2))
(rd / "config.json").write_text(json.dumps({
    "model_id": model_id,
    "note": ("3-seed ensemble of the third-family Mistral-Small-24B-Instruct-2501 "
             "(bf16, no quantization) 8-K forecaster (prereg §B1). Per-observation "
             "ARITHMETIC MEAN of prediction_realised_vol across vLLM seeds "
             f"{'+'.join(used)}, inner-joined on (ticker,accession,horizon_days) — "
             "identical convention to the C-model seed-ensemble primary "
             "(m1_ensemble_primary.ensemble_text) and to C6_llmtext_llama70ens. "
             "VAL+TEST only."),
    "llm": "mistralai/Mistral-Small-24B-Instruct-2501",
    "seeds_used": used,
    "clip_range": [pp.CLIP_LO, pp.CLIP_HI],
    "stats": {"n_rows": int(len(out)),
              "n_filings": int(out["text_path"].nunique()),
              "n_seeds": len(used)},
}, indent=2))
print(f"wrote {rd} rows={len(out)} filings={out['text_path'].nunique()} seeds={used}")
PY
  local ens_rc=$?

  local ENS_PRED="results/runs/C6_llmtext_mistral24ens_full_event_driven_seed2026/predictions.parquet"
  if [ "$ens_rc" -eq 0 ] && [ -f "$ENS_PRED" ]; then
    touch "$STATE/_ROW16_DONE"
    log "ROW16 COMPLETE — ensemble at $ENS_PRED"
    log "NEXT (local, after rsync-back): crossfamily rescoring per prereg §B1 (same battery as §B0)"
  else
    touch "$STATE/_ROW16_FAILED"
    log "ROW16 FAILED — ensemble_rc=$ens_rc; ensemble_present=$([ -f "$ENS_PRED" ] && echo yes || echo no)"
    exit 1
  fi
}

# ---- mode dispatch -----------------------------------------------------------
case "$MODE" in
  single)   mode_single ;;
  ensemble) mode_ensemble ;;
  *)
    echo "usage: launch.sh {single|ensemble}"
    echo "  single    env: GPUS=\"0,1\" SEEDS_TO_RUN=\"2026 2027 2028\" (defaults shown)"
    echo "  ensemble  CPU-only; requires all 3 seed run dirs"
    exit 2
    ;;
esac
