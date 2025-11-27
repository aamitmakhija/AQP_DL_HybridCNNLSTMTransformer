#!/usr/bin/env bash
# a_scripts/02_dataprep.sh — features → splits → (optional) impute → feature_eng → scale → checks/plots
set -euo pipefail
# make repo root importable for "common", "e_training", etc.
export PYTHONPATH="$(pwd):${PYTHONPATH}"
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

export CONFIG="${CONFIG:-configs/default.yaml}"
echo "[02_dataprep.sh] Using CONFIG=${CONFIG}"
echo "[02_dataprep.sh] CWD=$(pwd)"
echo "[02_dataprep.sh] Python: $("$PY" -V) at ${PY}"

# ---------- core dataprep ----------
# 1) Engineer base features from dataset_stream (or splits if they already exist)
echo "[1/10] c_dataprep/02_engineer_features.py"
"$PY" c_dataprep/02_engineer_features.py

# --- Deduplicate features once before splitting ---
"$PY" c_dataprep/01b_dedupe_features.py

echo "[1.5/10] c_dataprep/01c_dedupe_splits.py"
$PY c_dataprep/01c_dedupe_splits.py

# 2) Now that features exist, create time splits
echo "[2/10] c_dataprep/01_make_splits.py"
"$PY" c_dataprep/01_make_splits.py

# 3) Optional imputation (script may no-op per YAML)
echo "[3/10] c_dataprep/02c_impute_past_only.py"
"$PY" c_dataprep/02c_impute_past_only.py || true

# 4) Feature enrichment that overwrites the active features file (STRICT YAML)
echo "[4/10] c_dataprep/03_feature_eng.py"
"$PY" c_dataprep/03_feature_eng.py

# 5) Scale per YAML (per-station/global)
echo "[5/10] c_dataprep/03b_scale_features_per_station.py"
"$PY" c_dataprep/03b_scale_features_per_station.py

# ---------- reports / views ----------
echo "[6/10] c_dataprep/04_check_prep.py"
"$PY" c_dataprep/04_check_prep.py

echo "[7/10] c_dataprep/05_check_leakage.py"
"$PY" c_dataprep/05_check_leakage.py

echo "[8/10] c_dataprep/06_check_duplicates.py"
"$PY" c_dataprep/06_check_duplicates.py

echo "[9/10] c_dataprep/07_quick_stats.py"
"$PY" c_dataprep/07_quick_stats.py

# 10) Plots (don’t fail the run if inverse scaling or target isn’t present)
echo "[10/10] c_dataprep/08_plot_distributions.py"
"$PY" c_dataprep/08_plot_distributions.py --inverse || true

echo "******************************** data prep done ********************************"