#!/usr/bin/env bash
set -euo pipefail

PREFLIGHT_ONLY=0
if [[ "${1:-}" == "--preflight-only" ]]; then
  PREFLIGHT_ONLY=1
  shift
fi
TASK=${1:?usage: eval_direct.sh TASK POLICY [SEED] [NUM_EVAL]}
POLICY=${2:?usage: eval_direct.sh TASK POLICY [SEED] [NUM_EVAL]}
SEED=${3:-0}
NUM_EVAL=${4:-100}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "$SEED" in
  0|1|42) ;;
  *) echo "canonical evaluation seeds are 0, 1, and 42" >&2; exit 2 ;;
esac

if [[ "${INTACT_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python "$REPO_ROOT/scripts/preflight_check.py" eval-official \
    --task "$TASK" --policy "$POLICY" --eval-seed "$SEED"
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  exit 0
fi

python eval.py --config-name="$TASK" \
  solver=direct \
  policy="$POLICY" \
  seed="$SEED" \
  eval.num_eval="$NUM_EVAL" \
  eval.actor_warmstart=true
