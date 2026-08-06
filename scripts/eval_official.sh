#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: eval_official.sh MODE TASK POLICY [SEED] [NUM_EVAL] [HYDRA_OVERRIDES...]}
TASK=${2:?usage: eval_official.sh MODE TASK POLICY [SEED] [NUM_EVAL] [HYDRA_OVERRIDES...]}
POLICY=${3:?usage: eval_official.sh MODE TASK POLICY [SEED] [NUM_EVAL] [HYDRA_OVERRIDES...]}
SEED=${4:-0}
NUM_EVAL=${5:-100}
shift $(( $# >= 5 ? 5 : $# ))
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "$MODE" in
  direct|pure_cem|actor_cem|guarded_a) ;;
  *) echo "MODE must be direct, pure_cem, actor_cem, or guarded_a" >&2; exit 2 ;;
esac
case "$SEED" in
  0|1|42) ;;
  *) echo "Canonical evaluation seeds are 0, 1, and 42" >&2; exit 2 ;;
esac

if [[ "${INTACT_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  "$ROOT/scripts/eval_direct.sh" --preflight-only "$TASK" "$POLICY" "$SEED"
fi

exec python "$ROOT/eval.py" \
  --config-name="$TASK" \
  solver="$MODE" \
  policy="$POLICY" \
  seed="$SEED" \
  eval.num_eval="$NUM_EVAL" \
  eval.actor_warmstart=$([[ "$MODE" == pure_cem ]] && echo false || echo true) \
  output.filename="${TASK}_${MODE}_s${SEED}_n${NUM_EVAL}.txt" \
  "$@"
