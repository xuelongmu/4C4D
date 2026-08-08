#!/usr/bin/env bash
# Fast regression smoke test for training-loop changes.
#
# Runs 700 iterations at res 4 on the Xuelong posefix scene with the 8-camera
# split (held out: 4,6), exercising densification (from iter 200), the neural
# opacity decay path (from iter 500), SH degree increases, and held-out
# evaluation. Prints a one-line metrics summary parsed from the log.
#
# Usage: scripts/smoke_test.sh <run_name> [gpu_index]
# Run from the repo (or worktree) root whose code should be tested.
set -euo pipefail

RUN_NAME=${1:?usage: smoke_test.sh <run_name> [gpu]}
GPU=${2:-0}
STAMP=$(date +%H%M%S)
OUT_DIR="${RUN_NAME}-${STAMP}"
LOG="/tmp/smoke_${RUN_NAME}_${STAMP}.log"

START=$(date +%s)
CUDA_VISIBLE_DEVICES="$GPU" micromamba run -n 4c4d python -u train.py \
  --config configs/custom/xuelong_posefix_smoke700.yaml \
  --output_dir "$OUT_DIR" \
  --res 4 \
  --training_view 0,1,2,3,5,7,8,9 \
  --test_iterations 700 \
  > "$LOG" 2>&1 || { echo "SMOKE FAILED (exit $?) — tail of $LOG:"; tail -20 "$LOG"; exit 1; }
END=$(date +%s)

TRAIN_LINE=$(grep 'Evaluating train' "$LOG" | tail -1 || true)
TEST_LINE=$(grep 'Evaluating test' "$LOG" | tail -1 || true)
GS_NUM=$(grep -o 'gs_num[^,]*' "$LOG" | tail -1 || true)
echo "SMOKE OK  wall=$((END-START))s  ${GS_NUM}"
echo "  ${TRAIN_LINE}"
echo "  ${TEST_LINE}"
echo "  log: $LOG"
