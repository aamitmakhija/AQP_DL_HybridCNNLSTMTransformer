#!/usr/bin/env python3
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from copy import deepcopy

# ---------- config ----------
def deep_update(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

cfg_files = (sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml").split(",")
cfg = yaml.safe_load(open("configs/default.yaml"))
for f in cfg_files:
    f = f.strip()
    if not f or f == "configs/default.yaml":
        continue
    with open(f, "r") as fh:
        deep_update(cfg, yaml.safe_load(fh) or {})

ART_DIR   = Path(cfg["paths"]["artifacts_dir"])
FEAT_DIR  = ART_DIR / cfg["paths"]["features_dir"]
FEAT_FILE = FEAT_DIR / cfg["features"]["output"]["features_file"]

id_col   = cfg["data"]["id_col"]
time_col = cfg["data"]["time_col"]
target   = cfg["data"]["target"]

imp_cfg    = (cfg.get("missing", {}) or {}).get("impute", {}) or {}
ENABLED    = bool(imp_cfg.get("enabled", True))
FFILL_LIM  = int(imp_cfg.get("ffill_limit", 6))
ROLL_WIN   = int(imp_cfg.get("rolling_window", 24))
USE_MEDIAN = bool(imp_cfg.get("median_fallback", True))

if not ENABLED:
    print("[impute] disabled via config; exiting.")
    sys.exit(0)

print("[impute] loading features:", FEAT_FILE)
df = pq.read_table(FEAT_FILE).to_pandas()

# ---------- column selection ----------
# Only impute numeric *feature* columns. Never touch id/time/target (labels).
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
if id_col in df.columns:
    df[id_col] = df[id_col].astype(str)

df = df.sort_values([id_col, time_col])

exclude = {id_col, time_col, target}
num_cols = [
    c for c in df.columns
    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
]

if not num_cols:
    print("[impute] no numeric feature columns detected; nothing to do.")
    sys.exit(0)

def impute_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(time_col).copy()

    # 1) short forward-fill (bounded; past-only)
    g[num_cols] = g[num_cols].ffill(limit=FFILL_LIM)

    # 2) past-only rolling mean (shift(1) ensures strictly prior info)
    past_mean = (
        g[num_cols]
        .rolling(window=ROLL_WIN, min_periods=1)
        .mean()
        .shift(1)
    )
    g[num_cols] = g[num_cols].combine_first(past_mean)

    # 3) optional fallback: past-only expanding median
    if USE_MEDIAN:
        past_median = g[num_cols].expanding(min_periods=1).median().shift(1)
        g[num_cols] = g[num_cols].combine_first(past_median)

    return g

print(f"[impute] applying past-only imputation: ffill_limit={FFILL_LIM}, rolling_window={ROLL_WIN}, median_fallback={USE_MEDIAN}")
# Prefer include_groups=False when available (pandas >=2.2)
gb = df.groupby(id_col, group_keys=False, sort=False)
try:
    df = gb.apply(impute_group, include_groups=False)
except TypeError:
    df = gb.apply(impute_group)

OUT_FILE = FEAT_DIR / "dataset_features_imputed.parquet"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT_FILE)
print("[impute] wrote:", OUT_FILE)

# Make the imputed file the active features file for later steps (scaling/windowing)
ACTIVE_LINK = FEAT_DIR / cfg["features"]["output"]["features_file"]
try:
    if ACTIVE_LINK.exists() or ACTIVE_LINK.is_symlink():
        ACTIVE_LINK.unlink()
    ACTIVE_LINK.symlink_to(OUT_FILE.name)  # relative symlink
    print("[impute] activated features via symlink →", ACTIVE_LINK, "→", OUT_FILE.name)
except Exception as e:
    print(f"[impute] symlink failed ({e}); falling back to overwrite of {ACTIVE_LINK}")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ACTIVE_LINK)
    print("[impute] activated features by writing:", ACTIVE_LINK)