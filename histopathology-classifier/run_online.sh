#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${HOME}/.cache/histopathology-classifier/models"
VENV_DIR="${APP_DIR}/.venv"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required. Run this script in a GitHub Codespace." >&2
  exit 1
fi

PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.11, 3.12, or 3.13 is required." >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"
echo "Downloading trained models from the private models-v1 release..."
gh release download models-v1 \
  --repo adi-issa/AIML_P2 \
  --dir "$MODEL_DIR" \
  --pattern '*.keras' \
  --skip-existing

printf '%s\n' \
  '515589e9330ef4ad8250724dd9574da437f0b131a5c1cb06576b9987139e14e3  resnet50v2_Tuning.keras' \
  '3a20ca66c183cd11b33a4f4eeb190c33a707fc088365ee37ca8d852792642f55  densenet169_Tuning.keras' \
  | (cd "$MODEL_DIR" && sha256sum --check)
echo "Both model checksums match."

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install -r "$APP_DIR/requirements.txt"

export HISTOPATHOLOGY_MODEL_DIR="$MODEL_DIR"
exec "$VENV_DIR/bin/python" -m streamlit run "$APP_DIR/app.py" \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
