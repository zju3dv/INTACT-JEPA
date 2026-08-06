#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  direct|pure_cem|actor_cem|guarded_a) MODE=$1; shift ;;
  *) MODE=direct ;;
esac
TASK=${1:?usage: eval_clear_v051.sh [MODE] TASK POLICY DATASET CLEAR_ROOT UPSTREAM_ROOT [SEED] [OUTPUT]}
POLICY=${2:?usage: eval_clear_v051.sh [MODE] TASK POLICY DATASET CLEAR_ROOT UPSTREAM_ROOT [SEED] [OUTPUT]}
DATASET=${3:?usage: eval_clear_v051.sh [MODE] TASK POLICY DATASET CLEAR_ROOT UPSTREAM_ROOT [SEED] [OUTPUT]}
CLEAR_ROOT=${4:?usage: eval_clear_v051.sh [MODE] TASK POLICY DATASET CLEAR_ROOT UPSTREAM_ROOT [SEED] [OUTPUT]}
UPSTREAM_ROOT=${5:?usage: eval_clear_v051.sh [MODE] TASK POLICY DATASET CLEAR_ROOT UPSTREAM_ROOT [SEED] [OUTPUT]}
SEED=${6:-0}
OUTPUT=${7:-outputs/clear_v051_${MODE}_${TASK}_seed${SEED}.json}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "$SEED" in
  0|1|42) ;;
  *) echo "CLEAR v0.5.1 canonical Moderate seeds are 0, 1, and 42" >&2; exit 2 ;;
esac

test -n "${STABLEWM_HOME:-}" || {
  echo "STABLEWM_HOME must point to the cache containing checkpoints/" >&2
  exit 2
}
MANIFEST="$CLEAR_ROOT/manifests/v0.5/$TASK/moderate-seed${SEED}-n100.json"
test -f "$MANIFEST" || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
test -f "$DATASET" || { echo "missing dataset: $DATASET" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"

case "$MODE" in
  direct)
    MODE_ARGS=(--inference-mode direct --actor-warmstart on --direct-target-mode goal)
    ;;
  pure_cem)
    MODE_ARGS=(--inference-mode cem --actor-warmstart off --num-samples 300 --n-steps 30 --topk 30)
    ;;
  actor_cem)
    MODE_ARGS=(--inference-mode cem --actor-warmstart on --num-samples 300 --n-steps 30 --topk 30)
    ;;
  guarded_a)
    MODE_ARGS=(--inference-mode cem --cem-variant guarded-a --actor-warmstart on --num-samples 128 --n-steps 3 --topk 16)
    ;;
  *) echo "MODE must be direct, pure_cem, actor_cem, or guarded_a" >&2; exit 2 ;;
esac

if [[ "${INTACT_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python "$REPO_ROOT/scripts/preflight_check.py" eval-clear \
    --task "$TASK" --policy "$POLICY" \
    --dataset-path "$DATASET" \
    --clear-root "$CLEAR_ROOT" --upstream-root "$UPSTREAM_ROOT" \
    --eval-seed "$SEED"
fi

export PYTHONPATH="$CLEAR_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

python -m clear_lewm.cli evaluate \
  --manifest "$MANIFEST" \
  --policy "$POLICY" \
  --policy-label "INTACT ${MODE} corrected previous-action" \
  --output "$OUTPUT" \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path "$DATASET" \
  --upstream-dir "$UPSTREAM_ROOT" \
  --runtime-dir "$REPO_ROOT" \
  --policy-seed "$SEED" \
  --solver-batch-size 1 \
  --cpu-threads 1 \
  --matmul-precision highest \
  --strict-checkpoint \
  --allow-modified-stable-worldmodel \
  "${MODE_ARGS[@]}"
