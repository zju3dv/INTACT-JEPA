#!/usr/bin/env bash

# Source this file before training or evaluation. Machine-specific paths belong
# in the caller's environment or an ignored .env file, never in the repository.
_INTACT_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_INTACT_REPO_ROOT=$(cd "$_INTACT_SCRIPT_DIR/.." && pwd)

# Values already exported by the caller take precedence over .env.
declare -A _INTACT_SAVED_ENV=()
for _INTACT_NAME in \
  INTACT_PYTHON LOCAL_DATASET_DIR STABLEWM_HOME INTACT_OUTPUT_HOME MUJOCO_GL; do
  if [[ -v "$_INTACT_NAME" ]]; then
    _INTACT_SAVED_ENV["$_INTACT_NAME"]=${!_INTACT_NAME}
  fi
done
if [[ -f "$_INTACT_REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$_INTACT_REPO_ROOT/.env"
  set +a
fi
for _INTACT_NAME in "${!_INTACT_SAVED_ENV[@]}"; do
  printf -v "$_INTACT_NAME" '%s' "${_INTACT_SAVED_ENV[$_INTACT_NAME]}"
  export "$_INTACT_NAME"
done

if [[ -z "${INTACT_PYTHON:-}" ]]; then
  if [[ -x "$_INTACT_REPO_ROOT/.venv/bin/python" ]]; then
    export INTACT_PYTHON="$_INTACT_REPO_ROOT/.venv/bin/python"
  else
    INTACT_PYTHON=$(command -v python3 || true)
    export INTACT_PYTHON
  fi
fi

export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-${STABLEWM_HOME:-$HOME/.cache/stable-worldmodel}}"
export STABLEWM_HOME="${INTACT_OUTPUT_HOME:-${STABLEWM_HOME:-$LOCAL_DATASET_DIR}}"
export INTACT_OUTPUT_HOME="${INTACT_OUTPUT_HOME:-$STABLEWM_HOME}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

unset _INTACT_NAME _INTACT_SCRIPT_DIR _INTACT_REPO_ROOT _INTACT_SAVED_ENV
