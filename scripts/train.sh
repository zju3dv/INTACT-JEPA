#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train.sh [goal|waypoint] [TASK] [OPTIONS] [HYDRA_OVERRIDES...]

TASK: pusht | cube | reacher | tworoom

Options:
  --run-name NAME       Checkpoint/log directory name (recommended for formal runs)
  --output-home PATH    Dataset-independent output/cache root
  --smoke               One batch, batch size 2, no validation workers
  -h, --help            Show this help

Examples:
  CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh goal pusht --smoke
  CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh goal pusht \
    --run-name intact_goal_pusht_s3072_e1 seed=3072 trainer.max_epochs=1

All unrecognized arguments are forwarded to Hydra.
EOF
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-goal}
TASK=${2:-pusht}
if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi
shift $(( $# >= 2 ? 2 : $# ))

case "$MODE" in
  goal|goal_displacement) CONFIG=intact_goal; MODE_LABEL=goal ;;
  waypoint) CONFIG=intact_waypoint; MODE_LABEL=waypoint ;;
  *) echo "Mode must be goal or waypoint" >&2; usage >&2; exit 2 ;;
esac

case "$TASK" in
  pusht) DATA=pusht ;;
  cube) DATA=ogb ;;
  reacher) DATA=dmc ;;
  tworoom) DATA=tworoom ;;
  *) echo "Task must be pusht, cube, reacher, or tworoom" >&2; usage >&2; exit 2 ;;
esac

RUN_NAME=""
OUTPUT_HOME=""
SMOKE=false
OVERRIDES=()
while (( $# )); do
  case "$1" in
    --run-name)
      (( $# >= 2 )) || { echo "--run-name requires a value" >&2; exit 2; }
      RUN_NAME=$2
      shift 2
      ;;
    --output-home)
      (( $# >= 2 )) || { echo "--output-home requires a value" >&2; exit 2; }
      OUTPUT_HOME=$2
      shift 2
      ;;
    --smoke)
      SMOKE=true
      shift
      ;;
    output_model_name=*)
      [[ -z "$RUN_NAME" ]] && RUN_NAME=${1#output_model_name=}
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      OVERRIDES+=("$@")
      break
      ;;
    *)
      OVERRIDES+=("$1")
      shift
      ;;
  esac
done

SEED=3072
EPOCHS=1
for override in "${OVERRIDES[@]}"; do
  case "$override" in
    seed=*) SEED=${override#seed=} ;;
    trainer.max_epochs=*) EPOCHS=${override#trainer.max_epochs=} ;;
  esac
done
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="intact_${MODE_LABEL}_${TASK}_s${SEED}_e${EPOCHS}"
  [[ "$SMOKE" == true ]] && RUN_NAME="${RUN_NAME}_smoke"
fi

[[ -n "$OUTPUT_HOME" ]] && export INTACT_OUTPUT_HOME=$OUTPUT_HOME
source "$ROOT/scripts/fleet_env.sh"
export STABLEWM_HOME="${INTACT_OUTPUT_HOME:-$STABLEWM_HOME}"

if [[ "$SMOKE" == true ]]; then
  OVERRIDES+=(
    trainer.max_epochs=1
    +trainer.limit_train_batches=1
    trainer.limit_val_batches=0
    loader.batch_size=2
    loader.num_workers=0
    loader.persistent_workers=false
    '~loader.prefetch_factor'
  )
fi

LOG_DIR="$STABLEWM_HOME/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
mkdir -p "$LOG_DIR"
if [[ -e "$LOG_FILE" ]]; then
  echo "Log already exists: $LOG_FILE" >&2
  echo "Choose a different --run-name; existing runs are never overwritten." >&2
  exit 1
fi

echo "INTACT_TRAIN_LAUNCH mode=$MODE_LABEL task=$TASK run=$RUN_NAME"
echo "Python: $INTACT_PYTHON"
echo "Data: ${LOCAL_DATASET_DIR:-$STABLEWM_HOME}"
echo "Output: $STABLEWM_HOME/checkpoints/$RUN_NAME"
echo "Log: $LOG_FILE"

cd "$ROOT"
set +e
"$INTACT_PYTHON" train.py \
  --config-name="$CONFIG" \
  data="$DATA" \
  "${OVERRIDES[@]}" \
  output_model_name="$RUN_NAME" \
  2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
if (( status != 0 )); then
  echo "Training failed with exit code $status. See $LOG_FILE" >&2
  exit "$status"
fi
echo "Training complete: $STABLEWM_HOME/checkpoints/$RUN_NAME"
