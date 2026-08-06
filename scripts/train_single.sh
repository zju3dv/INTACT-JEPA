#!/usr/bin/env bash
set -euo pipefail

TASK=${1:?usage: train_single.sh TASK SEED [lewm|inverse|goal_only|displacement|waypoint]}
SEED=${2:?usage: train_single.sh TASK SEED [lewm|inverse|goal_only|displacement|waypoint]}
VARIANT=${3:-displacement}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "$SEED" in
  0|42|3072) ;;
  *) echo "canonical training seeds are 0, 42, and 3072" >&2; exit 2 ;;
esac

case "$TASK" in
  pusht) DATA=pusht ;;
  cube) DATA=ogb ;;
  reacher) DATA=dmc ;;
  tworoom) DATA=tworoom ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac

case "$VARIANT" in
  lewm) CONFIG=lewm ;;
  inverse) CONFIG=intact_inverse ;;
  goal_only) CONFIG=intact_goal_only ;;
  displacement|goal) CONFIG=intact_goal; VARIANT=displacement ;;
  waypoint) CONFIG=intact_waypoint ;;
  *) echo "variant must be lewm, inverse, goal_only, displacement, or waypoint" >&2; exit 2 ;;
esac

if [[ "${INTACT_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python "$REPO_ROOT/scripts/preflight_check.py" train-single \
    --task "$TASK" --train-seed "$SEED"
fi

python train.py --config-name="$CONFIG" \
  data="$DATA" \
  seed="$SEED" \
  output_model_name="${VARIANT}_${TASK}_s${SEED}" \
  trainer.max_epochs=1
