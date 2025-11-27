#!/usr/bin/env bash
# Train the CNN+Transformer hybrid model (portable: no readarray/jq)
# Usage:
#   ./a_scripts/04_train_hybrid.sh
#   CONFIG="configs/default.yaml,configs/cpu.yaml" ./a_scripts/04_train_hybrid.sh
set -euo pipefail
trap 'echo "[train_hybrid] FAILED on line $LINENO" >&2' ERR

# -------- pick python (prefer venv) --------
if [ -n "${VIRTUAL_ENV-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "ERROR: No python found. Activate your venv or install Python." >&2
  exit 1
fi

# -------- config --------
export CONFIG="${CONFIG:-configs/default.yaml}"
echo "[train_hybrid] Using CONFIG=${CONFIG}"
echo "[train_hybrid] CWD=$(pwd)"
echo "[train_hybrid] Python: $("$PY" -V) at ${PY}"

# -------- resolve key paths from YAML via Python --------
ART_DIR="$("$PY" - <<'PY'
import os, yaml, sys
merged={}
for p in [s.strip() for s in os.environ.get("CONFIG","configs/default.yaml").split(",") if s.strip()]:
    with open(p,"r") as f:
        y=yaml.safe_load(f) or {}
    # deep merge only "paths"
    if "paths" in y:
        merged.setdefault("paths",{}).update(y["paths"])
print(merged.get("paths",{}).get("artifacts_dir","experiments/artifacts"))
PY
)"

SEQ_DIR_REL="$("$PY" - <<'PY'
import os, yaml, sys
merged={}
for p in [s.strip() for s in os.environ.get("CONFIG","configs/default.yaml").split(",") if s.strip()]:
    with open(p,"r") as f:
        y=yaml.safe_load(f) or {}
    if "sequence" in y:
        merged.setdefault("sequence",{}).update(y["sequence"])
print(merged.get("sequence",{}).get("out_dir","seq"))
PY
)"

# absolute seq dir
case "${SEQ_DIR_REL}" in
  /*) SEQ_DIR="${SEQ_DIR_REL}" ;;
  *)  SEQ_DIR="${ART_DIR}/${SEQ_DIR_REL}" ;;
esac

echo "[train_hybrid] artifacts_dir=${ART_DIR}"
echo "[train_hybrid] seq_dir=${SEQ_DIR}"

# -------- ensure feature lock exists (optional convenience) --------
LOCK="${ART_DIR}/features_locked/feature_list.json"
if [ ! -f "${LOCK}" ]; then
  echo "[train_hybrid] feature lock missing — trying to generate via a_scripts/02_dataprep.sh"
  CONFIG="${CONFIG}" bash a_scripts/02_dataprep.sh >/dev/null 2>&1 || true
fi

# -------- pre-flight --------
if [ ! -f "${ART_DIR}/features_locked/feature_list.json" ]; then
  echo "Missing ${ART_DIR}/features_locked/feature_list.json. Run scaling/lock step first." >&2
  exit 1
fi
# require at least one train/val shard anywhere under h=*
ls -1 "${SEQ_DIR}"/train/h=*/shard_*.npz >/dev/null 2>&1 || { echo "No train windows under ${SEQ_DIR}/train/h=*/shard_*.npz"; exit 1; }
ls -1 "${SEQ_DIR}"/val/h=*/shard_*.npz   >/dev/null 2>&1 || { echo "No val windows under ${SEQ_DIR}/val/h=*/shard_*.npz";   exit 1; }

# quieten matplotlib font spam on macOS (optional)
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:::matplotlib.font_manager}"

# -------- train --------
echo "[train_hybrid] Launching: ${PY} -m e_training.train_hybrid"
exec "${PY}" -m e_training.train_hybrid
echo "******************************** train hybrid done ********************************"