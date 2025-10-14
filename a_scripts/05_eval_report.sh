#!/usr/bin/env bash
# a_scripts/05_eval_report.sh — run eval report with the same CONFIG logic as train
# Usage:
#   ./a_scripts/05_eval_report.sh
#   CONFIG="configs/default.yaml,configs/keep.yaml" ./a_scripts/05_eval_report.sh

set -euo pipefail
trap 'echo "[eval_report] FAILED on line $LINENO" >&2' ERR

# ---------- locate repo root & cd ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# ---------- pick python (prefer python3 on mac/linux) ----------
if [[ -n "${VIRTUAL_ENV-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "ERROR: No python found. Activate your venv or install Python." >&2
  exit 1
fi

# ---------- quick dependency sanity ----------
if ! "${PY}" -c 'import yaml' 2>/dev/null; then
  echo "ERROR: PyYAML not installed in ${PY}. Try: pip install pyyaml" >&2
  exit 1
fi

# ---------- config ----------
export CONFIG="${CONFIG:-configs/default.yaml}"

echo "[eval_report] Using CONFIG=${CONFIG}"
echo "[eval_report] CWD=$(pwd)"
echo "[eval_report] Python: $("$PY" -V) at ${PY}"

# ---------- resolve key paths from YAML (same approach as other scripts) ----------
ART_DIR="$("$PY" - <<'PY'
import os, yaml
cfgs=[s.strip() for s in os.environ.get("CONFIG","configs/default.yaml").split(",") if s.strip()]
merged={}
for p in cfgs:
    with open(p,"r") as f:
        y=yaml.safe_load(f) or {}
    for k,v in y.get("paths",{}).items():
        merged.setdefault("paths",{})[k]=v
print(merged.get("paths",{}).get("artifacts_dir","experiments/artifacts"))
PY
)"

MODEL_DIR_REL="$("$PY" - <<'PY'
import os, yaml
cfgs=[s.strip() for s in os.environ.get("CONFIG","configs/default.yaml").split(",") if s.strip()]
merged={}
for p in cfgs:
    with open(p,"r") as f:
        y=yaml.safe_load(f) or {}
    for k,v in y.get("dl",{}).items():
        merged.setdefault("dl",{})[k]=v
print(merged.get("dl",{}).get("model_dir","models"))
PY
)"

FEATS_DIR_REL="$("$PY" - <<'PY'
import os, yaml
cfgs=[s.strip() for s in os.environ.get("CONFIG","configs/default.yaml").split(",") if s.strip()]
merged={}
for p in cfgs:
    with open(p,"r") as f:
        y=yaml.safe_load(f) or {}
    for k,v in y.get("paths",{}).items():
        merged.setdefault("paths",{})[k]=v
print(merged.get("paths",{}).get("features_scaled_dir","features_scaled"))
PY
)"

# Make model/features dirs absolute if needed
MODEL_DIR="${MODEL_DIR_REL}"
[[ "${MODEL_DIR}" = /* ]] || MODEL_DIR="${ART_DIR}/${MODEL_DIR_REL}"
FEATS_DIR="${FEATS_DIR_REL}"
[[ "${FEATS_DIR}" = /* ]] || FEATS_DIR="${ART_DIR}/${FEATS_DIR_REL}"

echo "[eval_report] artifacts_dir=${ART_DIR}"
echo "[eval_report] model_dir=${MODEL_DIR}"
echo "[eval_report] features_scaled_dir=${FEATS_DIR}"

# ---------- preflight ----------
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "WARNING: ${MODEL_DIR} does not exist; eval will proceed but may find no checkpoints."
fi
if [[ ! -f "${FEATS_DIR}/test.parquet" ]]; then
  echo "ERROR: ${FEATS_DIR}/test.parquet not found. Run modelprep/dataprep first." >&2
  exit 1
fi

# ---------- run ----------
echo "[eval_report] Launching: ${PY} -m e_training.eval_report"
CONFIG="${CONFIG}" "${PY}" -m e_training.eval_report

echo
echo "******************************** eval report done ********************************"