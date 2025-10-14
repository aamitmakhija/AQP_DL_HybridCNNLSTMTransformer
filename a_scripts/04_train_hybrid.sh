# sequence > baseline >train_hybrid  > eval_report

#!/usr/bin/env bash
# Train the CNN+Transformer hybrid model
# Usage:
#   ./a_scripts/04_train_hybrid.sh
#   CONFIG="configs/default.yaml,configs/keep.yaml" ./a_scripts/04_train_hybrid.sh
set -euo pipefail

# -------- pick a python interpreter (prefer venv) --------
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

LOCK="experiments/artifacts/features_locked/feature_list.json"
if [[ ! -f "$LOCK" ]]; then
  echo "[train_hybrid] no feature lock found — generating one"
  # reuse the same heredoc as above or call the dataprep script that writes it
  CONFIG="${CONFIG}" bash a_scripts/02_dataprep.sh >/dev/null 2>&1 || true
fi

# -------- config --------
export CONFIG="${CONFIG:-configs/default.yaml}"

echo "[train_hybrid] Using CONFIG=${CONFIG}"
echo "[train_hybrid] CWD=$(pwd)"
echo "[train_hybrid] Python: $("$PY" -V) at ${PY}"

# -------- resolve paths from YAML (py-only, no jq) --------
ART_DIR="$("$PY" - <<'PY'
import os, sys, yaml
cfgs=[s.strip() for s in os.environ.get("CONFIG","configs/default.yaml").split(",") if s.strip()]
merged={}
for p in cfgs:
    with open(p,"r") as f: merged.update(yaml.safe_load(f) or {})
try:
    print(merged["paths"]["artifacts_dir"])
except KeyError as e:
    print(f"Missing YAML key: {e}", file=sys.stderr); sys.exit(1)
PY
)"

SEQ_DIR_REL="$("$PY" - <<'PY'
import os, sys, yaml
cfgs=[s.strip() for s in os.environ.get("CONFIG","configs/default.yaml").split(",") if s.strip()]
merged={}
for p in cfgs:
    with open(p,"r") as f: merged.update(yaml.safe_load(f) or {})
try:
    print(merged["sequence"]["out_dir"])
except KeyError as e:
    print(f"Missing YAML key: {e}", file=sys.stderr); sys.exit(1)
PY
)"

# absolute seq dir
if [[ "${SEQ_DIR_REL}" = /* ]]; then
  SEQ_DIR="${SEQ_DIR_REL}"
else
  SEQ_DIR="${ART_DIR}/${SEQ_DIR_REL}"
fi

echo "[train_hybrid] artifacts_dir=${ART_DIR}"
echo "[train_hybrid] seq_dir=${SEQ_DIR}"

# -------- pre-flight --------
if [[ ! -f "${ART_DIR}/features_locked/feature_list.json" ]]; then
  echo "Missing ${ART_DIR}/features_locked/feature_list.json. Run scaling/lock step first." >&2
  exit 1
fi

# must have train/val at least one horizon
if ! ls -1 "${SEQ_DIR}"/train/h=*/shard_*.npz >/dev/null 2>&1; then
  echo "No train windows found under ${SEQ_DIR}/train/h=*/shard_*.npz. Run a_scripts/03_modelprep.sh." >&2
  exit 1
fi
if ! ls -1 "${SEQ_DIR}"/val/h=*/shard_*.npz >/dev/null 2>&1; then
  echo "No val windows found under ${SEQ_DIR}/val/h=*/shard_*.npz. Run a_scripts/03_modelprep.sh." >&2
  exit 1
fi

# Optional: reduce noisy font warnings on macOS
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:::matplotlib.font_manager}"

# -------- train --------
echo "[train_hybrid] Launching: python -m e_training.train_hybrid"
exec "$PY" -m e_training.train_hybrid

print ("******************************** train hybrid done ********************************")