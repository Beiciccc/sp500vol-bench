#!/usr/bin/env bash
# =============================================================================
# ROW 15 — multi-seed ensemble of the matched-class Llama-3.1-70B (AWQ-INT4) on
# the 8-K (event_driven) panel: the only upside GPU line.
#
# WHY: the single-seed 70B firm-identity residual is "directionally replicates,
# significance attenuated" — its best firmID cell sits at Holm p = 0.05001 and
# only 1/3 vs-single-HAR cells survive the pre-declared Holm(6). This gives the
# 70B the SAME 3-seed averaging the neural C-models get as declared primary, and
# tests whether ensembling nudges "directionally replicates" -> "replicates".
#
# WHAT IT DOES (box-side; run ON the GPU box, from the repo root):
#   for seed in {2027, 2028} (2026 already exists as C6_llmtext_llama70):
#     1. clean /dev/shm/psm_* /dev/shm/sem.* (fork/AWQ shm leaks bit us)
#     2. run_inference.py run (c6_text, event_driven, TP2, len 8192, --seed s)
#        under a WATCHDOG that kills+retries a run whose GPU stays ~idle >8 min
#        (max 2 attempts/seed; skip the seed and continue if still hung)
#     3. postprocess.py build-runs --model-suffix _llama70_s<seed>
#   then build the per-observation 3-seed ensemble (2026+2027+2028) and write
#   run dir results/runs/C6_llmtext_llama70ens_full_event_driven_seed2026.
#
#   VLLM_WORKER_MULTIPROC_METHOD=spawn is exported (fork deadlocks with AWQ+TP2).
#
# GPU-HOUR ESTIMATE (honest): each 8-K seed ~6-7 h at TP2 (≈39k val+test filings,
#   c6_text only) -> 2 seeds ≈ 13 GPU-h wall (≈26 GPU-device-h). Add ~5-10 min
#   model-load per (re)start. Runs are RESUMABLE (checkpoint-every 500), so a
#   watchdog kill/retry resumes from the last part-*.parquet, not from scratch.
#
# Markers (under $STATE): _ROW15_S2027_DONE / _ROW15_S2028_DONE / _ROW15_DONE
#   on full success; _ROW15_FAILED if any seed was skipped after 2 hung attempts
#   (the 3-seed ensemble cannot be completed). Re-launching is idempotent: a seed
#   whose _DONE marker + run dir already exist is skipped.
#
# CAVEAT YOU MUST KNOW BEFORE SPENDING THE GPU-HOURS (see the delivery note):
#   run_inference.py's VllmGenerator does NOT pass --seed to vLLM and decodes at
#   temperature 0. Seeds 2027/2028 therefore differ from 2026 ONLY through vLLM /
#   AWQ-INT4 / TP2 kernel non-determinism (all-reduce ordering, batch-dependent
#   int4 dequant), not through the sampler. That is a real but SMALL variation
#   source, so this ensemble is a reproducibility-jitter ensemble, not a
#   stochastic-decoding one. If you want genuine seed diversity, plumb --seed into
#   LLM(seed=...) and set a small temperature FIRST (a one-line change I did not
#   make, since it edits a SANITY-gated shared script outside this deliverable).
# =============================================================================
set -uo pipefail

# ---- paths / env (all overridable) -----------------------------------------
REPO="${REPO:-/root/rivermind-data/repo}"
PY="${PY:-/root/rivermind-data/venvs/main/bin/python}"
export SP500VOL_DATA_ROOT="${SP500VOL_DATA_ROOT:-/root/rivermind-data/sp500vol-data}"
export HF_HOME="${HF_HOME:-/root/rivermind-data/hf}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn          # CRITICAL: fork deadlocks w/ AWQ+TP2

MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
MANIFEST="results/e1_llm_forecast/manifest_valtest.parquet"
SEEDS_TO_RUN="2027 2028"                            # 2026 already exists
ALL_SEEDS="2026 2027 2028"

STATE="results/experiments/row15_llama70_ensemble"
LOGDIR="$STATE/logs"

# ---- watchdog tunables ------------------------------------------------------
GPUS="${GPUS:-0,1}"                                 # the TP2 pair to watch
GPU_BUSY_MIB="${GPU_BUSY_MIB:-500}"                 # >this MiB used == run is alive
HANG_SECS="${HANG_SECS:-480}"                       # 8 min of idle GPU == hung
POLL_SECS="${POLL_SECS:-30}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"

cd "$REPO" || { echo "FATAL: cannot cd to REPO=$REPO"; exit 1; }
mkdir -p "$STATE" "$LOGDIR"
rm -f "$STATE/_ROW15_DONE" "$STATE/_ROW15_FAILED"   # recompute overall status each launch

log() { echo "[row15 $(date -u +%H:%M:%S)] $*" | tee -a "$LOGDIR/launch.log"; }

clean_shm() { rm -f /dev/shm/psm_* /dev/shm/sem.* 2>/dev/null || true; }

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
  local outdir="results/e1_llm_forecast/raw_llama70_s${seed}"
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

# ---- per-seed loop ----------------------------------------------------------
declare -a OK_SEEDS=()
ANY_SKIP=0
for seed in $SEEDS_TO_RUN; do
  rundir="results/runs/C6_llmtext_llama70_s${seed}_full_event_driven_seed2026"
  if [ -f "$STATE/_ROW15_S${seed}_DONE" ] && [ -f "$rundir/predictions.parquet" ]; then
    log "seed=$seed already DONE ($rundir exists) — skipping"
    OK_SEEDS+=("$seed"); continue
  fi
  if run_seed "$seed"; then
    log "seed=$seed postprocess build-runs (suffix _llama70_s${seed})"
    if "$PY" scripts/experiments/e1_llm_forecast/postprocess.py build-runs \
          --raw-dir "results/e1_llm_forecast/raw_llama70_s${seed}" \
          --model-suffix "_llama70_s${seed}" >> "$LOGDIR/s${seed}_postprocess.log" 2>&1 \
       && [ -f "$rundir/predictions.parquet" ]; then
      touch "$STATE/_ROW15_S${seed}_DONE"
      OK_SEEDS+=("$seed")
      log "seed=$seed marker _ROW15_S${seed}_DONE written"
    else
      log "seed=$seed postprocess FAILED — see s${seed}_postprocess.log"
      ANY_SKIP=1
    fi
  else
    ANY_SKIP=1
  fi
done

# ---- 3-seed ensemble build (per-observation mean, Qwen declared-primary spec) ----
# Averages prediction_realised_vol across the seeds whose run dir exists, inner-joined
# on (ticker, accession, horizon_days) — identical convention to
# m1_ensemble_primary.ensemble_text. Reuses postprocess.metrics_rows / PRED_COLS.
log "building 3-seed ensemble run dir C6_llmtext_llama70ens_full_event_driven_seed2026"
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
    base = "C6_llmtext_llama70" if seed == "2026" else f"C6_llmtext_llama70_s{seed}"
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

if base_df is None or len(used) < 2:
    print(f"FATAL: need seed 2026 base + >=2 seeds; have {used}"); sys.exit(3)

ens = frames[0]
for f in frames[1:]:
    ens = ens.merge(f, on=KEY, how="inner")      # inner: only obs present in every seed
ens["prediction_realised_vol"] = ens[[f"f{s}" for s in used]].mean(axis=1)
ens["prediction_realised_vol"] = ens["prediction_realised_vol"].clip(pp.CLIP_LO, pp.CLIP_HI)

out = base_df.drop(columns=["prediction_realised_vol"]).merge(
    ens[KEY + ["prediction_realised_vol"]], on=KEY, how="inner")
model_id = "C6_llmtext_llama70ens"
run_id = f"{model_id}_full_{DISC}_seed2026"
out["model_id"], out["run_id"] = model_id, run_id
out = out[pp.PRED_COLS].reset_index(drop=True)

rd = Path(f"results/runs/{run_id}")
rd.mkdir(parents=True, exist_ok=True)
out.to_parquet(rd / "predictions.parquet", index=False)
(rd / "metrics.json").write_text(json.dumps(pp.metrics_rows(out, DISC), indent=2))
(rd / "config.json").write_text(json.dumps({
    "model_id": model_id,
    "note": ("3-seed ensemble of the matched-class Llama-3.1-70B (AWQ-INT4) 8-K "
             "forecaster. Per-observation ARITHMETIC MEAN of prediction_realised_vol "
             f"across vLLM seeds {'+'.join(used)}, inner-joined on (ticker,accession,"
             "horizon_days) — identical convention to the C-model seed-ensemble "
             "primary (m1_ensemble_primary.ensemble_text). VAL+TEST only."),
    "llm": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    "seeds_used": used,
    "clip_range": [pp.CLIP_LO, pp.CLIP_HI],
    "stats": {"n_rows": int(len(out)),
              "n_filings": int(out["text_path"].nunique()),
              "n_seeds": len(used)},
}, indent=2))
print(f"wrote {rd} rows={len(out)} filings={out['text_path'].nunique()} seeds={used}")
PY
ens_rc=$?

# ---- overall status ---------------------------------------------------------
ENS_DIR="results/runs/C6_llmtext_llama70ens_full_event_driven_seed2026/predictions.parquet"
if [ "$ANY_SKIP" -eq 0 ] && [ "$ens_rc" -eq 0 ] && [ -f "$ENS_DIR" ]; then
  touch "$STATE/_ROW15_DONE"
  log "ROW15 COMPLETE — seeds ok: ${OK_SEEDS[*]:-none}; ensemble at $ENS_DIR"
  log "NEXT (local, after rsync-back): .venv/bin/python scripts/analysis/row15_ensemble_m1.py"
else
  touch "$STATE/_ROW15_FAILED"
  log "ROW15 FAILED — ok seeds: ${OK_SEEDS[*]:-none}; ensemble_rc=$ens_rc; ensemble_present=$([ -f "$ENS_DIR" ] && echo yes || echo no)"
  exit 1
fi
