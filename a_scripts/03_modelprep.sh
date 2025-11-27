# a_scripts/03_modelprep.sh
#!/usr/bin/env bash
set -euo pipefail

# ------- python picker -------
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
export PYTHONUNBUFFERED=1
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[03_modelprep] Using CONFIG=${CONFIG}"
echo "[03_modelprep] CWD=$(pwd)"
echo "[03_modelprep] Python: $("$PY" -V) at ${PY}"

# ------- read key paths from YAML (deep-merge + coalesce blanks) -------
eval "$("$PY" - <<'PY'
import os, sys, yaml, shlex
from copy import deepcopy

def deep_update(dst, src):
    if not isinstance(dst, dict): dst = {}
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def ry(p):
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}

def coalesce(v, default):
    if v is None: return default
    if isinstance(v, str) and not v.strip(): return default
    return v

cfg = ry("configs/default.yaml")
cfg_env = os.environ.get("CONFIG","")
if cfg_env:
    base = deepcopy(cfg)
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        base = deep_update(base, ry(p))
    cfg = base

art      = coalesce(cfg.get("paths",{}).get("artifacts_dir"), "experiments/artifacts")
seq_rel  = coalesce(cfg.get("sequence",{}).get("out_dir"), "seq")
feats_rel= coalesce(cfg.get("paths",{}).get("features_scaled_dir"), "features_scaled")

out_fmt = (cfg.get("scaling",{}) or {}).get("output_format") or (cfg.get("output",{}) or {}).get("format") or "parquet"
out_fmt = str(out_fmt).strip().lower()
normalized = False
if out_fmt not in {"parquet","feather","csv"}:
    out_fmt = "parquet"; normalized = True

print(f"ART_DIR={shlex.quote(art)}")
print(f"SEQ_OUT_DIR_REL={shlex.quote(seq_rel)}")
print(f"FEATS_REL={shlex.quote(feats_rel)}")
print(f"OUT_FMT={shlex.quote(out_fmt)}")
print(f"NORMALIZED_FMT={'1' if normalized else '0'}")
PY
)"

# read back the NORMALIZED flag into a variable (already set by eval)
if [[ "${NORMALIZED_FMT}" == "1" ]]; then
  echo "[03_modelprep] WARN: split format was invalid in overlays; normalized to '${OUT_FMT}'"
fi

# abs/rel resolve
SEQ_DIR="${SEQ_OUT_DIR_REL}"; [[ "${SEQ_DIR}" != /* ]] && SEQ_DIR="${ART_DIR}/${SEQ_OUT_DIR_REL}"
SCALED_DIR="${FEATS_REL}";    [[ "${SCALED_DIR}" != /* ]] && SCALED_DIR="${ART_DIR}/${FEATS_REL}"

# ensure dirs exist
mkdir -p "${ART_DIR}" "${SEQ_DIR}" "${SCALED_DIR}"

echo "[03_modelprep] artifacts_dir=${ART_DIR}"
echo "[03_modelprep] seq_dir=${SEQ_DIR}"
echo "[03_modelprep] scaled_dir=${SCALED_DIR}"
echo "[03_modelprep] split format=${OUT_FMT}"

# ------- ensure scaled splits exist -------
case "${OUT_FMT}" in
  parquet) EXT="parquet" ;;
  feather) EXT="feather" ;;
  csv)     EXT="csv" ;;
  *) echo "Unsupported split format after normalization: ${OUT_FMT}" >&2; exit 2 ;;
esac

need_scale=0
for spl in train val test; do
  [[ -f "${SCALED_DIR}/${spl}.${EXT}" ]] || need_scale=1
done

if (( need_scale )); then
  echo "[03_modelprep] Scaled splits missing → running scaler"
  "$PY" c_dataprep/03b_scale_features_per_station.py
else
  echo "[03_modelprep] Scaled splits present — skipping scaling"
fi

# ------- build sequence windows (prefer multi-horizon) -------
echo "[03_modelprep] Building sequence windows"
if [[ -f "d_modelprep/CA_make_windows_multi.py" ]]; then
  "$PY" d_modelprep/CA_make_windows_multi.py
elif [[ -f "d_modelprep/CA_make_windows.py" ]]; then
  "$PY" d_modelprep/CA_make_windows.py
else
  cand="$(ls d_modelprep/*make_windows*.py 2>/dev/null | head -n 1 || true)"
  [[ -n "${cand}" ]] || { echo "ERROR: No window-maker found in d_modelprep/"; exit 3; }
  echo "[03_modelprep] Falling back to ${cand}"
  "$PY" "${cand}"
fi

# ------- feature lock presence (informational) -------
if [[ -f "${ART_DIR}/features_locked/feature_list.json" ]]; then
  echo "[03_modelprep] Using feature lock: ${ART_DIR}/features_locked/feature_list.json"
else
  echo "[03_modelprep] WARNING: feature lock missing; ensure dataprep/scaler produced it" >&2
fi

# ------- quick window checks -------
echo "[03_modelprep] Checking sequence windows"
"$PY" d_modelprep/CB_check_windows.py --max-shards 2

echo "******************************** [03_modelprep] ******************************** DONE"