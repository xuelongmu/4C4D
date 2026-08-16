#!/usr/bin/env bash
# Launch a detached full-length A/B training run (7,500 iters, res 2).
#
# The unit of an A/B is two invocations of this script, one per GPU, pointed at
# two worktrees holding the commits under comparison. Both sides get the same
# config, split and seed; only the code (or the extra flags) differ. See
# docs/EXPERIMENT_METHODOLOGY.md.
#
# The config path is resolved to an absolute path *before* changing into the
# worktree, so both sides train against the config the launcher saw rather than
# each worktree's own copy of it.
#
# On completion the log is copied to <run_dir>/train.log, which is what
# scripts/build_experiment_report.py reads. (It cannot be written there during
# training: train.py refuses to start if the run directory already exists.)
#
# Usage:
#   scripts/ab_launch.sh <worktree_dir> <output_name> <gpu> [extra train.py args...]
#
# Environment:
#   FOURC4D_PYTHON      interpreter to use            (default: python)
#   FOURC4D_AB_CONFIG   config path, relative to cwd
#                       (default: configs/custom/xuelong_posefix_production.yaml)
#   FOURC4D_AB_VIEWS    training views                (default: 0,1,2,3,5,7,8,9)
#   FOURC4D_AB_LOGDIR   scratch dir for the live log  (default: /tmp)
#
# NOTE: keep the calling shell alive ~10 s after launch (e.g. `&& sleep 10`)
# or the nohup child dies with the session.
set -euo pipefail

# --- detached child: train, then preserve the log next to the run ------------
if [[ "${FOURC4D_AB_CHILD:-}" == 1 ]]; then
  status=0
  CUDA_VISIBLE_DEVICES="$AB_GPU" "$AB_PY" -u train.py \
    --config "$AB_CFG" \
    --output_dir "$AB_OUT" \
    --res 2 \
    --training_view "$AB_VIEWS" \
    "$@" > "$AB_LOG" 2>&1 || status=$?
  if [[ -d "$AB_RUN_DIR" ]]; then
    cp -f "$AB_LOG" "$AB_RUN_DIR/train.log"
  fi
  exit "$status"
fi

# --- launcher ----------------------------------------------------------------
WT=${1:?usage: ab_launch.sh <worktree_dir> <output_name> <gpu> [extra args...]}
OUT=${2:?output name}
GPU=${3:?gpu index}
EXTRA=("${@:4}")

SELF=$(readlink -f "$0")
PY=${FOURC4D_PYTHON:-python}
VIEWS=${FOURC4D_AB_VIEWS:-0,1,2,3,5,7,8,9}
LOGDIR=${FOURC4D_AB_LOGDIR:-/tmp}
CFG=$(readlink -f "${FOURC4D_AB_CONFIG:-configs/custom/xuelong_posefix_production.yaml}")

[[ -f "$CFG" ]] || { echo "config not found: $CFG" >&2; exit 1; }
[[ -d "$WT" ]]  || { echo "worktree not found: $WT" >&2; exit 1; }

MODEL_PATH=$("$PY" - "$CFG" <<'PY'
import sys
from omegaconf import OmegaConf
print(OmegaConf.load(sys.argv[1]).ModelParams.model_path)
PY
)
RUN_DIR="$MODEL_PATH/$OUT"
[[ -e "$RUN_DIR" ]] && { echo "run dir already exists: $RUN_DIR" >&2; exit 1; }

LOG="$LOGDIR/${OUT}.log"

cd "$WT"
FOURC4D_AB_CHILD=1 AB_GPU="$GPU" AB_PY="$PY" AB_CFG="$CFG" AB_OUT="$OUT" \
  AB_VIEWS="$VIEWS" AB_LOG="$LOG" AB_RUN_DIR="$RUN_DIR" \
  nohup "$SELF" ${EXTRA[@]+"${EXTRA[@]}"} > /dev/null 2>&1 &

echo "launched pid $!"
echo "  worktree : $WT"
echo "  config   : $CFG"
echo "  run dir  : $RUN_DIR"
echo "  live log : $LOG  (copied to \$run_dir/train.log at exit)"
