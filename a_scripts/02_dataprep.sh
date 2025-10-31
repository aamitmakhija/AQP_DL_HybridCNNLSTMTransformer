# --- (optional) only create lock if missing; do NOT overwrite ---
LOCK_PATH="$(python - <<'PY'
import os, yaml
from pathlib import Path
cfg = yaml.safe_load(open("configs/default.yaml"))
cfg_env = os.environ.get("CONFIG","")
def upd(d,s):
    for k,v in (s or {}).items():
        if isinstance(v,dict) and isinstance(d.get(k),dict): upd(d[k],v)
        else: d[k]=v
for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
    upd(cfg, yaml.safe_load(open(p)) or {})
art = Path(cfg["paths"]["artifacts_dir"])
print(art / "features_locked" / "feature_list.json")
PY
)"
if [[ ! -f "$LOCK_PATH" ]]; then
  echo "[lock] not found; run scaler again or create manually."; exit 1
fi
echo "******************************** data prep done ********************************"