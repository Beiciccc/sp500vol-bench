#!/usr/bin/env bash
# ROW 2B launch — firm-demeaned C2 FinBERT S1 retraining (GPU).
# Deployed to /root/rivermind-data/_row2_gpu.sh on the box. Fire when a GPU frees up:
#   CUDA_VISIBLE_DEVICES=0 nohup bash /root/rivermind-data/_row2_gpu.sh \
#       > /root/rivermind-data/logs/row2_gpu.out 2>&1 &
# Markers: /root/rivermind-data/_ROW2_C2dm_<disclosure>_DONE per run,
#          /root/rivermind-data/_ROW2_DONE when both runs finished,
#          /root/rivermind-data/_ROW2_FAILED if any run failed.
set -uo pipefail

export SP500VOL_DATA_ROOT="${SP500VOL_DATA_ROOT:-/root/rivermind-data/sp500vol-data}"
export HF_HOME="${HF_HOME:-/root/rivermind-data/hf}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO=/root/rivermind-data/repo
PY=/root/rivermind-data/venvs/main/bin/python
LOGDIR=/root/rivermind-data/logs
MARK=/root/rivermind-data
mkdir -p "$LOGDIR"
cd "$REPO"

# clear stale outcome markers from a previous attempt (per-run markers are
# re-touched below when a finished run dir is found, so clearing is safe)
rm -f "$MARK/_ROW2_DONE" "$MARK/_ROW2_FAILED" \
      "$MARK"/_ROW2_C2dm_long_form_DONE "$MARK"/_ROW2_C2dm_event_driven_DONE

fail=0
for disc in long_form event_driven; do
    run_dir="$REPO/results/runs/C2dm_finbert_s1_full_${disc}_seed2026"
    if [[ -f "$run_dir/predictions.parquet" && -f "$run_dir/metrics.json" ]]; then
        echo "[row2] SKIP ${disc} — already done ($run_dir)"
        touch "$MARK/_ROW2_C2dm_${disc}_DONE"
        continue
    fi
    echo "[row2] $(date -u +%FT%TZ) start C2dm_finbert_s1 ${disc} on GPU ${CUDA_VISIBLE_DEVICES}"
    "$PY" scripts/experiments/row2_demeaned/train_demeaned.py \
        --model C2_finbert_s1 --dataset full --disclosure "$disc" --seed 2026 \
        >> "$LOGDIR/row2_gpu_${disc}.log" 2>&1
    rc=$?
    if [[ $rc -eq 0 && -f "$run_dir/predictions.parquet" ]]; then
        echo "[row2] $(date -u +%FT%TZ) done ${disc}"
        touch "$MARK/_ROW2_C2dm_${disc}_DONE"
    else
        echo "[row2] $(date -u +%FT%TZ) FAILED ${disc} (rc=$rc) — see $LOGDIR/row2_gpu_${disc}.log"
        fail=1
    fi
done

if [[ $fail -eq 0 ]]; then
    touch "$MARK/_ROW2_DONE"
    echo "[row2] ALL DONE — marker $MARK/_ROW2_DONE"
else
    touch "$MARK/_ROW2_FAILED"
    echo "[row2] finished with failures — marker $MARK/_ROW2_FAILED (_ROW2_DONE NOT written)"
fi
exit $fail
