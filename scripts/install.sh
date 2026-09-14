#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${INTACT_VENV:-$ROOT/.venv}
BACKEND=${1:-cu124}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-300}

case "$BACKEND" in
  cu124|cpu) ;;
  *)
    echo "Usage: $0 [cu124|cpu]" >&2
    exit 2
    ;;
esac

if command -v uv >/dev/null 2>&1; then
  uv venv --allow-existing --seed --python "$PYTHON_VERSION" "$VENV"
  pip_install() {
    uv pip install --python "$VENV/bin/python" "$@"
  }
  pip_check() { uv pip check --python "$VENV/bin/python"; }
else
  PYTHON_BIN=${PYTHON_BIN:-$(command -v "python$PYTHON_VERSION" || true)}
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python $PYTHON_VERSION or uv is required." >&2
    exit 2
  fi
  "$PYTHON_BIN" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
  pip_install() {
    "$VENV/bin/python" -m pip install "$@"
  }
  pip_check() {
    "$VENV/bin/python" -m pip check
  }
fi

if [[ "$BACKEND" == cu124 ]]; then
  pip_install \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
else
  pip_install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu
fi

if [[ "$BACKEND" == cu124 ]]; then
  pip_install -r "$ROOT/requirements-cu124.lock" \
    --extra-index-url https://download.pytorch.org/whl/cu124
else
  pip_install -r "$ROOT/requirements.txt"
fi
pip_check
"$VENV/bin/python" "$ROOT/scripts/verify_install.py"

cat <<EOF

INTACT environment is ready.

  source "$VENV/bin/activate"
  cp "$ROOT/.env.example" "$ROOT/.env"
  # Edit .env, then run the verification commands in docs/INSTALL.md.
EOF
