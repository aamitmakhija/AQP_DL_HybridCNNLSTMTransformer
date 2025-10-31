#!/usr/bin/env bash
# a_scripts/run_all.sh
# # End-to-end pipeline runner


# Prefer GPU (MPS) but don’t crash on unsupported ops
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Tune CPU threading for Apple Silicon; 6–8 is a good starting point
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}

# If you ever use OpenBLAS/NumExpr in your stack, these help too (harmless otherwise):
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-6}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-6}

# CONFIG="configs/default.yaml,configs/keep.yaml" bash a_scripts/run_all.sh
set -euo pipefail



# Pretty banners
banner() { printf '\n******** %s ********\n' "$*"; }

# Helpful error message if anything fails
trap 'echo "[run_all] ERROR on line $LINENO. Exiting."; exit 1' ERR

# Forward CONFIG if caller set it; else default
export CONFIG="${CONFIG:-configs/default.yaml}"
echo "[run_all] CONFIG=${CONFIG}"

# 0) Clean previous run artifacts (seq, features_scaled[_ps], models, reports, etc.)
banner "00_clean"
bash a_scripts/00_clean.sh

# 1) Raw → parquet + dataset stream
banner "01_make_data"
bash a_scripts/01_make_data.sh

# 2) Splits, feature engineering, scaling, checks, lock
banner "02_dataprep"
bash a_scripts/02_dataprep.sh

# 3) Sequence windowing (uses feature lock), window checks
banner "03_modelprep"
bash a_scripts/03_modelprep.sh

# 4) Train hybrid model (reads seq windows)
banner "04_train_hybrid"
bash a_scripts/04_train_hybrid.sh

# 5) Evaluate and write report
banner "05_eval_report"
bash a_scripts/05_eval_report.sh

banner "ALL DONE"