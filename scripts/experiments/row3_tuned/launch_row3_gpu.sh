#!/usr/bin/env bash
# ROW 3 — validation-tuned challenger arm: box-side GPU launch script.
# Deployed as /root/gpu-data/_row3_gpu.sh (committable copy lives at
# scripts/experiments/row3_tuned/launch_row3_gpu.sh).
#
# Usage (fire when the GPUs free up):
#   nohup bash /root/gpu-data/_row3_gpu.sh > /root/gpu-data/_row3.log 2>&1 &
#   GPUS="0,1" nohup bash /root/gpu-data/_row3_gpu.sh > /root/gpu-data/_row3.log 2>&1 &
#
# GPUS (default "0") lists physical GPU ids. The 9-config grid is split across
# them by shard (config i goes to worker i % n); each worker runs its configs
# SEQUENTIALLY on its own GPU. After every shard finishes, the val-QLIKE
# selection runs once and writes the C2t_/D2t_ run dirs + the tuning-audit CSV.
# Writes /root/gpu-data/_ROW3_DONE on success, _ROW3_FAILED on failure.
# Grid runs are resumable: rerunning this script skips finished configs.
set -uo pipefail

export SP500VOL_DATA_ROOT=/root/gpu-data/sp500vol-data
export HF_HOME=/root/gpu-data/hf
export HF_ENDPOINT=https://hf-mirror.com

REPO=/root/gpu-data/repo
PY=/root/gpu-data/venvs/main/bin/python
TUNE=scripts/experiments/row3_tuned/tune_challengers.py
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"  # honor an inherited GPU mask
ROW3_ARGS="${ROW3_ARGS:-}"   # extra flags for BOTH stages (e.g. --seed 2026)

rm -f /root/gpu-data/_ROW3_DONE /root/gpu-data/_ROW3_FAILED
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
cd "$REPO"

echo "[row3] $(date -u +%FT%TZ) train: 9 configs across ${N} GPU(s): ${GPUS}"
pids=()
for i in "${!GPU_ARR[@]}"; do
  g="${GPU_ARR[$i]}"
  CUDA_VISIBLE_DEVICES="$g" "$PY" "$TUNE" --stage train \
      --shard "$i" --num-shards "$N" $ROW3_ARGS \
      > "/root/gpu-data/_row3_train_gpu${g}.log" 2>&1 &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[row3] shard $i (gpu ${GPU_ARR[$i]}) FAILED — see _row3_train_gpu${GPU_ARR[$i]}.log"
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  touch /root/gpu-data/_ROW3_FAILED
  exit 1
fi

echo "[row3] $(date -u +%FT%TZ) select: best val-QLIKE per (model, disclosure)"
if ! "$PY" "$TUNE" --stage select $ROW3_ARGS \
    > /root/gpu-data/_row3_select.log 2>&1; then
  echo "[row3] SELECT FAILED — see _row3_select.log"
  touch /root/gpu-data/_ROW3_FAILED
  exit 1
fi

cat /root/gpu-data/_row3_select.log
touch /root/gpu-data/_ROW3_DONE
echo "[row3] $(date -u +%FT%TZ) DONE — tuned runs in results/runs/{C2t_finbert_s1,D2t_gated_fusion}_full_*_seed2026, audit CSV in results/tables/row3_tuning_grid.csv"
