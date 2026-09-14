#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=${INTACT_HF_REPO:-INTACT-JEPA/INTACT}
REVISION=${INTACT_CHECKPOINT_REVISION:-paper-e5-goal-v1}
BASE_URL=${INTACT_CHECKPOINT_BASE_URL:-https://huggingface.co/$REPO/resolve/$REVISION}
MANIFEST="$ROOT/checkpoints/PAPER_E5_MATRIX_MANIFEST.json"
CHECKSUMS_SHA256=538349b2a29c98ca5ec17f3ec288faefb137687768bc8f7a9550f566b7d07ff2
FIRST=${1:-0}

case "$FIRST" in
  0|42|3072|all)
    CELL=goal_intact
    SELECTOR=$FIRST
    CACHE_ARG=${2:-}
    ;;
  *)
    CELL=$FIRST
    SELECTOR=${2:-all}
    CACHE_ARG=${3:-}
    ;;
esac

case "$CELL" in
  lewm|inverse_only|waypoint_intent|goal_intent|waypoint_intact|goal_intact)
    CELLS=("$CELL")
    ;;
  matrix)
    CELLS=(lewm inverse_only waypoint_intent goal_intent waypoint_intact goal_intact)
    ;;
  *)
    echo "Unknown cell: $CELL" >&2
    echo "Cells: lewm inverse_only waypoint_intent goal_intent waypoint_intact goal_intact matrix" >&2
    exit 2
    ;;
esac

case "$SELECTOR" in
  0|42|3072) SEEDS=("$SELECTOR") ;;
  all) SEEDS=(0 42 3072) ;;
  *) echo "Unknown seed selector: $SELECTOR" >&2; exit 2 ;;
esac

if [[ -n "$CACHE_ARG" ]]; then
  CACHE_ROOT=$CACHE_ARG
elif [[ -n ${STABLEWM_HOME:-} ]]; then
  CACHE_ROOT=$STABLEWM_HOME
else
  source "$ROOT/scripts/fleet_env.sh"
  CACHE_ROOT=$STABLEWM_HOME
fi

if [[ -z "${INTACT_PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    INTACT_PYTHON="$ROOT/.venv/bin/python"
  else
    INTACT_PYTHON=$(command -v python3 || true)
  fi
fi
[[ -n "$INTACT_PYTHON" && -x "$INTACT_PYTHON" ]] || {
  echo "Python is not executable; set INTACT_PYTHON." >&2
  exit 1
}

command -v jq >/dev/null || { echo "jq is required." >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required." >&2; exit 1; }
mkdir -p "$CACHE_ROOT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

download_asset() {
  local name=$1
  local target="$TMP/$name"
  local download_url="${BASE_URL%/}/$name"
  echo "Downloading $name from $REPO@$REVISION..."
  local fallback
  fallback="$TMP/aria2-$name"
  if [[ ${INTACT_USE_ARIA2:-0} == 1 ]] && command -v aria2c >/dev/null; then
    aria2c --continue=true --allow-overwrite=true --auto-file-renaming=false \
      --file-allocation=none --max-connection-per-server=16 --split=16 \
      --min-split-size=1M --max-tries=20 --retry-wait=1 --timeout=30 \
      --console-log-level=warn --summary-interval=10 \
      --dir="$TMP" --out="aria2-$name" "$download_url"
  else
    curl --fail --location --continue-at - --retry 20 --retry-all-errors \
      --retry-delay 1 \
      --output "$fallback" "$download_url"
  fi
  [[ -s "$fallback" ]] || { echo "Empty download: $name" >&2; exit 1; }
  mv "$fallback" "$target"
}

download_asset SHA256SUMS
actual_checksums_sha256=$(sha256sum "$TMP/SHA256SUMS" | awk '{print $1}')
[[ "$actual_checksums_sha256" == "$CHECKSUMS_SHA256" ]] || {
  echo "SHA256SUMS integrity mismatch: $actual_checksums_sha256" >&2
  exit 1
}
for cell in "${CELLS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    asset=$(jq -r --arg cell "$cell" --argjson seed "$seed" \
      '.cells[] | select(.id == $cell) | .training_seeds[] | select(.seed == $seed) | .asset' \
      "$MANIFEST")
    [[ -n "$asset" && "$asset" != null ]] || {
      echo "No release asset for cell=$cell seed=$seed" >&2
      exit 1
    }
    download_asset "$asset"
    expected=$(awk -v name="$asset" '$2 == name {print $1}' "$TMP/SHA256SUMS")
    [[ -n "$expected" ]] || { echo "No checksum for $asset" >&2; exit 1; }
    actual=$(sha256sum "$TMP/$asset" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || {
      echo "Archive SHA256 mismatch for $asset" >&2
      exit 1
    }
    tar -xzf "$TMP/$asset" -C "$CACHE_ROOT"
  done
done

for cell in "${CELLS[@]}"; do
  verify_args=(--cell "$cell")
  for seed in "${SEEDS[@]}"; do
    verify_args+=(--seed "$seed")
  done
  "$INTACT_PYTHON" "$ROOT/scripts/verify_paper_checkpoints.py" \
    --root "$CACHE_ROOT" "${verify_args[@]}"
done
echo "Paper checkpoints installed under $CACHE_ROOT/checkpoints"
