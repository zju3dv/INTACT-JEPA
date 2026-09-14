#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 CELL TASK TRAIN_SEED [EVAL_SEED] [NUM_EVAL] [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CELL=$1
TASK=$2
TRAIN_SEED=$3
EVAL_SEED=${4:-0}
NUM_EVAL=${5:-100}
OVERRIDES=("${@:6}")

case "$TASK" in pusht|cube|reacher|tworoom) ;; *) echo "Unknown task: $TASK" >&2; exit 2 ;; esac
case "$TRAIN_SEED" in 0|42|3072) ;; *) echo "Unknown train seed: $TRAIN_SEED" >&2; exit 2 ;; esac
case "$CELL" in
  lewm) PREFIX=recovery_lewm ;;
  inverse_only) PREFIX=recovery_inverse_only ;;
  waypoint_intent) PREFIX=recovery_query_only ;;
  goal_intent) PREFIX=recovery_delta_query_only ;;
  waypoint_intact) PREFIX=recovery_full ;;
  goal_intact) PREFIX=recovery_delta_full ;;
  *) echo "Unknown cell: $CELL" >&2; exit 2 ;;
esac

case "$CELL" in
  waypoint_intent|waypoint_intact)
    export INVERSE_QUERY_CONDITION_SOURCE=waypoint
    ;;
  *)
    export INVERSE_QUERY_CONDITION_SOURCE=goal
    ;;
esac

source "$ROOT/scripts/fleet_env.sh"
export STABLEWM_HOME="${PAPER_CHECKPOINT_HOME:-$STABLEWM_HOME}"
POLICY="${PREFIX}_${TASK}_s${TRAIN_SEED}/weights_epoch_5.pt"
"$INTACT_PYTHON" "$ROOT/scripts/verify_paper_checkpoints.py" \
  --root "$STABLEWM_HOME" --cell "$CELL" --seed "$TRAIN_SEED"

cd "$ROOT/paper_runtime"
export PYTHONPATH="$ROOT/paper_runtime${PYTHONPATH:+:$PYTHONPATH}"
COMMON=(
  --config-name="$TASK"
  policy="$POLICY"
  seed="$EVAL_SEED"
  eval.num_eval="$NUM_EVAL"
  solver=cem
  solver.batch_size=1
)

if [[ "$CELL" == lewm ]]; then
  exec "$INTACT_PYTHON" eval.py \
    "${COMMON[@]}" \
    solver.num_samples=300 \
    solver.n_steps=30 \
    solver.topk=30 \
    ++eval.actor_warmstart=false \
    "${OVERRIDES[@]}"
fi

exec "$INTACT_PYTHON" eval.py \
  "${COMMON[@]}" \
  solver._target_=prior_only_solver.PriorOnlySolver \
  ++eval.actor_warmstart=true \
  "${OVERRIDES[@]}"
