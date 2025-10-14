#!/usr/bin/env bash
# a_scripts/02_dataprep.sh — splits → features → per-station scaling → checks
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

echo "[dataprep] Using CONFIG=${CONFIG}"
echo "[dataprep] CWD=$(pwd)"
echo "[dataprep] Python: $("$PY" -V) at ${PY}"

echo "[1/6] c_dataprep/01_make_splits.py"
"$PY" -m c_dataprep.01_make_splits

echo "[2/6] c_dataprep/02_engineer_features.py"
"$PY" -m c_dataprep.02_engineer_features

echo "[3/6] c_dataprep/03b_scale_features_per_station.py"
"$PY" -m c_dataprep.03b_scale_features_per_station

echo "[4/6] c_dataprep/04_check_prep.py"
"$PY" -m c_dataprep.04_check_prep

echo "[5/6] c_dataprep/05_check_leakage.py"
"$PY" -m c_dataprep.05_check_leakage

echo "[6/6] c_dataprep/06_check_duplicates.py"
"$PY" -m c_dataprep.06_check_duplicates

echo "[dataprep] DONE"

# --- ensure feature lock exists (works for both global & per-station scaling) ---
"$PY" - <<'PY'
import os, json, yaml
from pathlib import Path
import pandas as pd

# load default + overlays if CONFIG is set
cfg = yaml.safe_load(open("configs/default.yaml"))
cfg_env = os.environ.get("CONFIG","")
for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
    with open(p) as f:
        o = yaml.safe_load(f) or {}
    def upd(d,s):
        for k,v in (s or {}).items():
            if isinstance(v,dict) and isinstance(d.get(k),dict): upd(d[k],v)
            else: d[k]=v
    upd(cfg,o)

art = Path(cfg["paths"]["artifacts_dir"])
# prefer per-station scaler, then global
candidates = [
    art / cfg["paths"].get("features_scaled_dir","features_scaled") / "train.parquet",
    art / "features_scaled_ps" / "train.parquet",
    art / "features_scaled" / "train.parquet",
]
train_pq = next((p for p in candidates if p.exists()), None)
if not train_pq:
    raise SystemExit("[lock] No scaled train parquet found; skipping lock write")

df = pd.read_parquet(train_pq)

id_col   = cfg.get("data",{}).get("id_col","station_id")
time_col = cfg.get("data",{}).get("time_col","Datetime")
target   = cfg.get("data",{}).get("target","PM25_Concentration")

X_cols = [c for c in df.columns if c not in {id_col, time_col, target}]

out_dir = art / "features_locked"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "feature_list.json"
json.dump(
    {"time_col": time_col, "target_col": target, "id_cols": [id_col], "X_cols_ordered": X_cols},
    open(out_path, "w"),
    indent=2
)
print(f"[lock] wrote {out_path} with {len(X_cols)} features")
print("******************************** data prep done ********************************")
PY