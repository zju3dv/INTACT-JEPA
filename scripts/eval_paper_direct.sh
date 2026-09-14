#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 TASK TRAIN_SEED [EVAL_SEED] [NUM_EVAL] [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK=$1
TRAIN_SEED=$2
EVAL_SEED=${3:-0}
NUM_EVAL=${4:-100}
shift $(( $# >= 4 ? 4 : $# ))

case "$TASK" in pusht|cube|reacher|tworoom) ;; *) echo "Unknown task: $TASK" >&2; exit 2 ;; esac
case "$TRAIN_SEED" in 0|42|3072) ;; *) echo "Unknown train seed: $TRAIN_SEED" >&2; exit 2 ;; esac

source "$ROOT/scripts/fleet_env.sh"
export STABLEWM_HOME="${PAPER_CHECKPOINT_HOME:-$STABLEWM_HOME}"
POLICY="recovery_delta_full_${TASK}_s${TRAIN_SEED}/weights_epoch_5.pt"
"$INTACT_PYTHON" "$ROOT/scripts/verify_paper_checkpoints.py" \
  --root "$STABLEWM_HOME" --seed "$TRAIN_SEED"

cd "$ROOT/paper_runtime"
export PYTHONPATH="$ROOT/paper_runtime${PYTHONPATH:+:$PYTHONPATH}"
exec "$INTACT_PYTHON" eval.py \
  --config-name="$TASK" \
  policy="$POLICY" \
  seed="$EVAL_SEED" \
  eval.num_eval="$NUM_EVAL" \
  solver=cem \
  solver._target_=prior_only_solver.PriorOnlySolver \
  solver.batch_size=1 \
  ++eval.actor_warmstart=true \
  "$@"
