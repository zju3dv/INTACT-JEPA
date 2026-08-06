#!/usr/bin/env bash
set -euo pipefail

SEED=${1:?usage: train_multitask.sh SEED [lewm|inverse|goal_only|displacement|waypoint] OUTPUT_CACHE RUN_DIR}
VARIANT=${2:-displacement}
OUTPUT_CACHE=${3:?usage: train_multitask.sh SEED VARIANT OUTPUT_CACHE RUN_DIR}
RUN_DIR=${4:?usage: train_multitask.sh SEED VARIANT OUTPUT_CACHE RUN_DIR}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "$SEED" in
  0|42|3072) ;;
  *) echo "canonical training seeds are 0, 42, and 3072" >&2; exit 2 ;;
esac

case "$VARIANT" in
  lewm|inverse|goal_only|displacement|waypoint) ;;
  goal) VARIANT=displacement ;;
  *) echo "variant must be lewm, inverse, goal_only, displacement, or waypoint" >&2; exit 2 ;;
esac

if [[ "${INTACT_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python "$REPO_ROOT/scripts/preflight_check.py" train-multitask \
    --train-seed "$SEED"
fi

torchrun --standalone --nproc_per_node=4 train_multitask.py \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --epochs 5 \
  --batch-size 256 \
  --output-prefix "${VARIANT}_multitask" \
  --output-cache "$OUTPUT_CACHE" \
  --run-dir "$RUN_DIR"
