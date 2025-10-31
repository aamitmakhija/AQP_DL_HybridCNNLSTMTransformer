#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Pick a Python interpreter (prefer venv)
# -----------------------------
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

# -----------------------------
# Config handling
# -----------------------------
export CONFIG="${CONFIG:-configs/default.yaml}"

echo "[03_modelprep] Using CONFIG=${CONFIG}"
echo "[03_modelprep] CWD=$(pwd)"
echo "[03_modelprep] Python: $("$PY" -V) at ${PY}"

# -----------------------------
# Paths derived from CONFIG (handles comma-separated overlays)
# -----------------------------
read -r ART_DIR SEQ_OUT_DIR_REL FEATS_REL < <("$PY" - <<'PY'
import os, yaml

def deep_update(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

cfg = yaml.safe_load(open("configs/default.yaml"))
cfg_env = os.environ.get("CONFIG","")
for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
    with open(p, "r") as f:
        deep_update(cfg, yaml.safe_load(f) or {})

art = cfg["paths"]["artifacts_dir"]
seq_rel = cfg["sequence"]["out_dir"]
feats_rel = cfg["paths"].get("features_scaled_dir", "features_scaled")

print(art)
print(seq_rel)
print(feats_rel)
PY
)

# Make seq dir absolute if needed
if [[ "${SEQ_OUT_DIR_REL}" = /* ]]; then
  SEQ_DIR="${SEQ_OUT_DIR_REL}"
else
  SEQ_DIR="${ART_DIR}/${SEQ_OUT_DIR_REL}"
fi

# Resolve features_scaled_dir (absolute or relative to artifacts_dir)
if [[ "${FEATS_REL}" = /* ]]; then
  SCALED_DIR="${FEATS_REL}"
else
  SCALED_DIR="${ART_DIR}/${FEATS_REL}"
fi

echo "[03_modelprep] artifacts_dir=${ART_DIR}"
echo "[03_modelprep] seq_dir=${SEQ_DIR}"
echo "[03_modelprep] scaled_dir=${SCALED_DIR}"

# -----------------------------
# Ensure scaled features exist (trust dataprep/scaler as the source of truth)
# -----------------------------
NEED_SCALE=0
for SPL in train val test; do
  [[ -f "${SCALED_DIR}/${SPL}.parquet" ]] || NEED_SCALE=1
done

if [[ "${NEED_SCALE}" -eq 1 ]]; then
  echo "[03_modelprep] Scaled features not found — running per-station scaler"
  "$PY" -m c_dataprep.03b_scale_features_per_station
else
  echo "[03_modelprep] Scaled features present — skipping scaling step"
fi

# -----------------------------
# Build sequence windows (resilient module selection)
# -----------------------------
echo "[03_modelprep] Building sequence windows"

run_py () {  # helper: run python file with CONFIG env intact
  local f="$1"
  echo "[03_modelprep] -> ${PY} ${f}"
  "$PY" "${f}"
}

if "$PY" - <<'PY'
import importlib, sys
try:
    importlib.import_module("d_modelprep.CA_make_windows")
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
then
  "$PY" -m d_modelprep.CA_make_windows
elif [[ -f "d_modelprep/CA_make_windows.py" ]]; then
  run_py "d_modelprep/CA_make_windows.py"
elif [[ -f "d_modelprep/CA_make_windows_multi.py" ]]; then
  run_py "d_modelprep/CA_make_windows_multi.py"
else
  CAND="$(ls d_modelprep/*make_windows*.py 2>/dev/null | head -n 1 || true)"
  if [[ -n "${CAND}" ]]; then
    echo "[03_modelprep] Falling back to ${CAND}"
    run_py "${CAND}"
  else
    echo "ERROR: No window-maker found (expected one of:
  - d_modelprep/CA_make_windows.py
  - d_modelprep/CA_make_windows_multi.py
  - d_modelprep/*make_windows*.py )" >&2
    exit 1
  fi
fi

# -----------------------------
# DO NOT touch the feature lock here
# (feature_list.json is authored by the scaler/dataprep step)
# -----------------------------
if [[ -f "${ART_DIR}/features_locked/feature_list.json" ]]; then
  echo "[03_modelprep] Using existing feature lock at ${ART_DIR}/features_locked/feature_list.json"
else
  echo "[03_modelprep] WARNING: feature lock missing; ensure dataprep/scaler ran successfully." >&2
fi

# -----------------------------
# Quick window checks
# -----------------------------
echo "[03_modelprep] Checking sequence windows"
"$PY" -m d_modelprep.CB_check_windows

echo "******************************** [03_modelprep] ******************************** DONE"