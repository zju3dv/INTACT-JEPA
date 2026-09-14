#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/scripts/fleet_env.sh"

paths=(
  datasets/pusht_expert_train.lance
  datasets/ogbench/cube_single_expert.h5
  datasets/reacher.h5
  datasets/tworoom.h5
)

test -x "$INTACT_PYTHON" || {
  echo "Python is not executable: $INTACT_PYTHON" >&2
  exit 1
}
for path in "${paths[@]}"; do
  test -e "$LOCAL_DATASET_DIR/$path" || {
    echo "Missing: $LOCAL_DATASET_DIR/$path" >&2
    exit 1
  }
done

"$INTACT_PYTHON" - <<'PY'
import hydra
import lightning
import stable_pretraining
import stable_worldmodel
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible GPUs: {torch.cuda.device_count()}")
PY

echo "Host: $(hostname -s)"
echo "Python: $INTACT_PYTHON"
echo "Dataset cache: $LOCAL_DATASET_DIR"
echo "Output/cache root: $STABLEWM_HOME"
echo "Fleet preflight: OK"
