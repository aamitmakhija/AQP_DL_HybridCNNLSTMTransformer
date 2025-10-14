#!/usr/bin/env bash
# a_scripts/01_make_data.sh — ingestion only
set -euo pipefail

# pick python (prefer venv)
if [[ -n "${VIRTUAL_ENV-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "ERROR: No python found. Activate your venv or install python." >&2
  exit 1
fi

# config (default if not provided)
export CONFIG="${CONFIG:-configs/default.yaml}"

echo "[01_make_data.sh] Using CONFIG=${CONFIG}"
echo "[01_make_data.sh] CWD=$(pwd)"
echo "[01_make_data.sh] Python: $("$PY" -V) at ${PY}"

# --- ingestion only ---
echo "[1/3] b_ingestion/01_csv2parquet_safe.py"
"$PY" b_ingestion/01_csv2parquet_safe.py

echo "[2/3] b_ingestion/02_build_stream.py"
"$PY" b_ingestion/02_build_stream.py

echo "[3/3] b_ingestion/03_inspect_dataset.py"
"$PY" b_ingestion/03_inspect_dataset.py

echo "********************************01_make_data.sh] DONE********************************"