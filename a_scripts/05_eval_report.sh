#!/usr/bin/env bash
# Run eval report (portable: no readarray/jq)
# Usage:
#   ./a_scripts/05_eval_report.sh
#   CONFIG="configs/default.yaml,configs/cpu.yaml" ./a_scripts/05_eval_report.sh
set -euo pipefail
trap 'echo "[eval_report] FAILED on line $LINENO" >&2' ERR

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
echo "[eval_report] Using CONFIG=${CONFIG}"
echo "[eval_report] CWD=$(pwd)"
echo "[eval_report] Python: $("$PY" -V) at ${PY}"

# -------- resolve paths via Python --------
ART_DIR="$("$PY" - <<'PY'
import os, yaml
merged={}
for p in [s.strip() for s in os.environ.get("CONFIG","configs/default.yaml").split(",") if s.strip()]:
    with open(p,"r") as f:
        y=yaml.safe_load(f) or {}
    if "paths" in y:
        merged.setdefault("paths",{}).update(y["paths"])
print(merged.get("paths",{}).get("artifacts_dir","experiments/artifacts"))
PY
)"

MODEL_DIR_REL="$("$PY" - <<'PY'
import os, yaml
merged={}
for p in [s.strip() for s in os.environ.get("CONFIG","configs/default.yaml").split(",") if s.strip()]:
    with open(p,"r") as f:
        y=yaml.safe_load(f) or {}
    if "dl" in y:
        merged.setdefault("dl",{}).update(y["dl"])
print(merged.get("dl",{}).get("model_dir","models"))
PY
)"

FEATS_DIR_REL="$("$PY" - <<'PY'
import os, yaml
merged={}
for p in [s.strip() for s in os.environ.get("CONFIG","configs/default.yaml").split(",") if s.strip()]:
    with open(p,"r") as f:
        y=yaml.safe_load(f) or {}
    if "paths" in y:
        merged.setdefault("paths",{}).update(y["paths"])
print(merged.get("paths",{}).get("features_scaled_dir","features_scaled"))
PY
)"

# absolutize dirs
MODEL_DIR="${MODEL_DIR_REL}"; case "${MODEL_DIR}" in /*) ;; *) MODEL_DIR="${ART_DIR}/${MODEL_DIR_REL}";; esac
FEATS_DIR="${FEATS_DIR_REL}"; case "${FEATS_DIR}" in /*) ;; *) FEATS_DIR="${ART_DIR}/${FEATS_DIR_REL}";; esac

echo "[eval_report] artifacts_dir=${ART_DIR}"
echo "[eval_report] model_dir=${MODEL_DIR}"
echo "[eval_report] features_scaled_dir=${FEATS_DIR}"

# -------- preflight --------
if [ ! -d "${MODEL_DIR}" ]; then
  echo "WARNING: ${MODEL_DIR} does not exist; eval will proceed but may find no checkpoints."
fi
# accept parquet/feather/csv depending on your config; check the most common
if [ ! -f "${FEATS_DIR}/test.parquet" ] && [ ! -f "${FEATS_DIR}/test.feather" ] && [ ! -f "${FEATS_DIR}/test.csv" ]; then
  echo "ERROR: scaled test split missing in ${FEATS_DIR} (parquet/feather/csv). Run dataprep first." >&2
  exit 1
fi

# -------- run --------
echo "[eval_report] Launching: ${PY} -m e_training.eval_report"
CONFIG="${CONFIG}" "${PY}" -m e_training.eval_report

# -------- optional post-inspect (top errors for a station) --------
if [ -n "${STATION_ID-}" ]; then
  echo "[post] Inspecting station_id=${STATION_ID}"
  CONFIG="${CONFIG}" "${PY}" - <<'PY'
import os, pandas as pd, sys
sid = os.environ.get("STATION_ID")
p = "experiments/artifacts/reports/preds_H1.csv"
try:
    df = pd.read_csv(p)
except FileNotFoundError:
    print(f"[post] {p} not found (set eval.save_preds: true in CONFIG).")
    sys.exit(0)
if "station_id" in df.columns:
    sub = df[df["station_id"].astype(str) == str(sid)].assign(abs_err=lambda x: (x.y_pred - x.y_true).abs())
    out = "experiments/artifacts/reports/residuals_{}_H1_top10.csv".format(sid)
    sub.sort_values("abs_err", ascending=False).head(10).to_csv(out, index=False)
    print(f"[post] Wrote {out}")
else:
    print("[post] preds_H1.csv has no station_id column.")
PY
fi

echo
echo "******************************** eval report done ********************************"