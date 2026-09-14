#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_multitask.sh [OPTIONS]

Options:
  --run-name NAME       Shared run/log name
  --output-home PATH    Dataset-independent output/cache root
  --smoke               Four GPUs, one optimizer step, batch size 2/task
  --epochs N            Override the paper default (5)
  --max-steps N         Limit optimizer steps per epoch
  --batch-size N        Per-task batch size (paper default: 256)
  --num-workers N       DataLoader workers per rank
  --config PATH         Alternate multitask YAML
  -h, --help            Show this help

Examples:
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multitask.sh --smoke
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multitask.sh \
    --run-name intact_multitask_goal_s3072_e5
EOF
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_NAME=""
OUTPUT_HOME=""
SMOKE=false
ARGS=()
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
    --output-model-name)
      (( $# >= 2 )) || { echo "--output-model-name requires a value" >&2; exit 2; }
      RUN_NAME=$2
      shift 2
      ;;
    --output-model-name=*)
      RUN_NAME=${1#--output-model-name=}
      shift
      ;;
    --smoke)
      SMOKE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

[[ -n "$OUTPUT_HOME" ]] && export INTACT_OUTPUT_HOME=$OUTPUT_HOME
source "$ROOT/scripts/fleet_env.sh"
export STABLEWM_HOME="${INTACT_OUTPUT_HOME:-$STABLEWM_HOME}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
  if (( ${#visible_devices[@]} != 4 )); then
    echo "Multitask training requires exactly four visible GPUs; got: $CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
fi

if [[ "$SMOKE" == true ]]; then
  ARGS+=(--epochs 1 --max-steps 1 --batch-size 2 --num-workers 0)
fi
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME=intact_multitask_goal_s3072_e5
  [[ "$SMOKE" == true ]] && RUN_NAME=${RUN_NAME}_smoke
fi
ARGS+=(--output-model-name "$RUN_NAME")

LOG_DIR="$STABLEWM_HOME/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
mkdir -p "$LOG_DIR"
if [[ -e "$LOG_FILE" ]]; then
  echo "Log already exists: $LOG_FILE" >&2
  echo "Choose a different --run-name; existing runs are never overwritten." >&2
  exit 1
fi

echo "INTACT_MULTITASK_LAUNCH run=$RUN_NAME tasks=pusht,cube,reacher,tworoom"
echo "Python: $INTACT_PYTHON"
echo "Data: ${LOCAL_DATASET_DIR:-$STABLEWM_HOME}"
echo "Output: $STABLEWM_HOME/checkpoints/$RUN_NAME"
echo "Log: $LOG_FILE"

cd "$ROOT"
set +e
"$INTACT_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train_multitask.py \
  "${ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
if (( status != 0 )); then
  echo "Multitask training failed with exit code $status. See $LOG_FILE" >&2
  exit "$status"
fi
echo "Multitask training complete: $STABLEWM_HOME/checkpoints/$RUN_NAME"
