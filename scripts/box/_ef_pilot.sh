#!/usr/bin/env bash
# M2 elicitation-fairness — the val PILOT chain on the run box.
#   prereg: configs/prereg_elicitation_fairness.md (tag prereg-ef-v1.0)
#   amendment PENDING (prereg-ef v1.1): the box is a SINGLE A100-80GB, not 2xA100-40GB,
#   so the pilot runs at TP=1. The v1.0 text says TP=2. The effective TP is passed
#   explicitly here and recorded in every artifact the pilot writes.
#
# Runs the registered 2 families x 3 variants over the SAME 2,000 val docs. Each cell is
# its OWN python3 process so vLLM tears the engine down and frees the 80GB between cells
# (two 24-27B bf16 models never share the card).
#
# Waits for: DONE_dl_mistral, DONE_dl_gemma, DONE_setup.
# Writes:    DONE_ef_pilot on success, FAIL_ef_pilot on any failure.
# Logs:      $REPO/logs/ef_pilot_<family>_<variant>.log, ef_pilot_assemble.log
#
# Resumable: a cell whose shard already exists is skipped; the script never overwrites a
# registered output (the single-shot guard exits 3, which is treated as "already done").
#
# usage:  bash scripts/box/_ef_pilot.sh            # TP=1 (default)
#         TP=2 bash scripts/box/_ef_pilot.sh       # only if the box ever has 2 GPUs again
#         WAIT_TIMEOUT=7200 bash scripts/box/_ef_pilot.sh
set -uo pipefail

ROOT=${ROOT:-/root/gpu-data}     # overridable ONLY so the chain is testable off-box
REPO=${REPO:-$ROOT/repo}
LOGS=$REPO/logs
PY=${PY:-python3}                      # system python3 — NO venv on this box
TP=${TP:-1}                            # effective tensor-parallel size
WAIT_TIMEOUT=${WAIT_TIMEOUT:-43200}    # 12h to wait out the model downloads
POLL=${POLL:-30}

EF=scripts/analysis/elicitation_fairness.py
SIDECAR=results/e1_llm_forecast/ef_pilot_sidecar.json
PILOT_JSON=results/tables/elicitation_fairness_pilot.json
SHARD_DIR=results/e1_llm_forecast/ef_pilot_shards

export SP500VOL_DATA_ROOT=$ROOT/sp500vol-data
export HF_HOME=$ROOT/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOGS"
cd "$REPO" || { echo "FATAL: repo tree not staged at $REPO"; exit 1; }

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

fail() {
    say "FAIL: $*"
    { echo "failed_utc=$(date -u +%FT%TZ)"
      echo "reason=$*"
      echo "tp=$TP"; } > "$ROOT/FAIL_ef_pilot"
    exit 1
}

# ----------------------------------------------------------- wait for the sentinels
wait_for() {
    local name=$1 path=$ROOT/$1 waited=0
    if [ -e "$path" ]; then say "sentinel $name: already present"; return 0; fi
    say "waiting for sentinel $name (timeout ${WAIT_TIMEOUT}s) ..."
    while [ ! -e "$path" ]; do
        # bail out early if the producing step announced its own failure
        for f in "$ROOT"/FAIL_dl_* "$ROOT"/FAIL_setup; do
            [ -e "$f" ] && fail "upstream failure sentinel present: $(basename "$f")"
        done
        sleep "$POLL"; waited=$((waited + POLL))
        if [ "$waited" -ge "$WAIT_TIMEOUT" ]; then
            fail "timed out after ${WAIT_TIMEOUT}s waiting for $name"
        fi
        if [ $((waited % 600)) -eq 0 ]; then say "  ... still waiting for $name (${waited}s)"; fi
    done
    say "sentinel $name: present"
}

say "EF pilot chain starting (TP=$TP, repo=$REPO)"
rm -f "$ROOT/FAIL_ef_pilot"
wait_for DONE_setup
wait_for DONE_dl_mistral
wait_for DONE_dl_gemma

# ----------------------------------------------------------- preflight
[ -f "$EF" ] || fail "$EF not staged (scp the scripts first)"
[ -f "$SIDECAR" ] || fail "$SIDECAR not staged.
  The box slice cannot rebuild the registered pilot on its own: the committed C6 prompt
  renders filing_date (an EDGAR metadata field, NOT derivable from filing_time_utc), and
  the box panel holds 14,266 ED val docs vs the committed manifest's 14,213 — so
  head(2,000) here would select a DIFFERENT set from the registered v1.3 pilot.
  Emit it LOCALLY and scp it:
    .venv/bin/python scripts/analysis/elicitation_fairness.py --emit-sidecar
    scp $SIDECAR box:$REPO/$SIDECAR"
[ -f "$SP500VOL_DATA_ROOT/processed/full/aligned_ed_val.parquet" ] \
    || fail "aligned_ed_val.parquet not under $SP500VOL_DATA_ROOT/processed/full/"
[ -f "$SP500VOL_DATA_ROOT/processed/_text_cache/filing_texts_ed_val.parquet" ] \
    || fail "filing_texts_ed_val.parquet not under $SP500VOL_DATA_ROOT/processed/_text_cache/"

say "preflight OK — sidecar + val slice + text cache all present"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/  GPU /' || say "  (nvidia-smi unavailable)"

if [ -f "$PILOT_JSON" ]; then
    say "$PILOT_JSON already exists — the pilot is done (single-shot guard). Nothing to do."
    touch "$ROOT/DONE_ef_pilot"; exit 0
fi

# ----------------------------------------------------------- the 2 x 3 grid
mkdir -p "$SHARD_DIR"
for fam in mistral24 gemma27; do
    for v in V0 V1 V2; do
        shard=$SHARD_DIR/${fam}_${v}.json
        log=$LOGS/ef_pilot_${fam}_${v}.log
        if [ -f "$shard" ]; then
            say "$fam/$v: shard exists -> skip"
            continue
        fi
        say "$fam/$v: running (TP=$TP) -> $log"
        # one process per cell: vLLM frees the GPU on exit before the next model loads
        $PY "$EF" --pilot --family "$fam" --variant "$v" --tp "$TP" > "$log" 2>&1
        rc=$?
        if [ $rc -eq 3 ]; then
            say "$fam/$v: write-once guard (shard already present) -> treating as done"
        elif [ $rc -ne 0 ]; then
            tail -n 40 "$log"
            fail "$fam/$v exited $rc (see $log)"
        fi
        [ -f "$shard" ] || fail "$fam/$v exited 0 but wrote no shard (see $log)"
        grep -E "HEALTH_PASS|HEALTH_FAIL" "$log" | tail -1 | sed "s/^/  [$fam\/$v] /"
        # make sure the card is actually free before the next 27B lands on it
        sleep 5
    done
done

# ----------------------------------------------------------- assemble
say "assembling the six shards -> $PILOT_JSON"
$PY "$EF" --pilot --assemble --tp "$TP" > "$LOGS/ef_pilot_assemble.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    tail -n 60 "$LOGS/ef_pilot_assemble.log"
    fail "assemble exited $rc (see $LOGS/ef_pilot_assemble.log)"
fi
[ -f "$PILOT_JSON" ] || fail "assemble exited 0 but wrote no $PILOT_JSON"

# ----------------------------------------------------------- report
sed -n '/EF PILOT SUMMARY/,$p' "$LOGS/ef_pilot_assemble.log"
if grep -q "TP VERDICT FLIP" "$LOGS/ef_pilot_assemble.log"; then
    say "!! TP VERDICT FLIP recorded — the committed instrument-dead judgement is"
    say "!! TP-confounded for at least one family. See tp_invariance in $PILOT_JSON."
fi

{ echo "done_utc=$(date -u +%FT%TZ)"
  echo "tp=$TP"
  echo "pilot_json=$PILOT_JSON"; } > "$ROOT/DONE_ef_pilot"
say "DONE — wrote $ROOT/DONE_ef_pilot"
say "Next: --full is BLOCKED until the full panel is staged (it exits 4 with a clear"
say "message by design); scp $PILOT_JSON back and decide the branch."
exit 0
