#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  cat >&2 <<'EOF'
Usage: scripts/eval.sh MODE TASK CHECKPOINT [SEED] [NUM_EVAL] [HYDRA_OVERRIDES...]

MODE:
  direct       Search-free INTACT controller
  cem          Actor-disabled CEM baseline
  guarded_a    Direct plan plus bounded CEM verification (128x3, sigma=0.25)

All actor-backed modes use causal continuation history: rows[t-5:t] at the
dataset start state, raw-zero left padding only before the episode begins, and
executed actions thereafter. Official LeWM is the default scoring benchmark.
EOF
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=$1
TASK=$2
CHECKPOINT=$3
SEED=${4:-0}
NUM_EVAL=${5:-100}
shift $(( $# >= 5 ? 5 : $# ))

case "$MODE" in
  direct)
    SOLVER=direct
    CONTRACT_MODE=direct
    ;;
  cem)
    SOLVER=pure_cem
    CONTRACT_MODE=pure_cem
    ;;
  guarded_a)
    SOLVER=guarded_a
    CONTRACT_MODE=guarded_a
    ;;
  *)
    echo "MODE must be direct, cem, or guarded_a" >&2
    exit 2
    ;;
esac
case "$TASK" in
  pusht|cube|reacher|tworoom) ;;
  *) echo "TASK must be pusht, cube, reacher, or tworoom" >&2; exit 2 ;;
esac
case "$SEED" in
  0|1|42) ;;
  *) echo "Canonical evaluation seeds are 0, 1, and 42" >&2; exit 2 ;;
esac

source "$ROOT/scripts/fleet_env.sh"
export STABLEWM_HOME="${INTACT_OUTPUT_HOME:-$STABLEWM_HOME}"

cd "$ROOT"
exec "$INTACT_PYTHON" eval.py \
  --config-name="$TASK" \
  solver="$SOLVER" \
  policy="$CHECKPOINT" \
  seed="$SEED" \
  eval.num_eval="$NUM_EVAL" \
  eval.protocol=official \
  eval.inference_mode="$CONTRACT_MODE" \
  cache_dir="$LOCAL_DATASET_DIR" \
  sdpa_backend="${INTACT_EVAL_SDPA_BACKEND:-math}" \
  output.filename="${TASK}_${MODE}_s${SEED}_n${NUM_EVAL}_results.txt" \
  "$@"
