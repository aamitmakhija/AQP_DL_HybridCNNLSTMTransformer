# 2_dataprep/06_check_duplicates.py
from __future__ import annotations
from pathlib import Path
import os
import json
import pandas as pd
import yaml
from copy import deepcopy
from typing import Dict

# --- minimal config loader (supports CONFIG overlays) ---
def _deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> Dict:
    base = yaml.safe_load(open("configs/default.yaml"))
    cfg_env = os.environ.get("CONFIG")
    if not cfg_env:
        return base
    merged = deepcopy(base)
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        overlay = yaml.safe_load(open(p)) or {}
        _deep_update(merged, overlay)
    return merged

cfg = _load_cfg()
ART = Path(cfg["paths"]["artifacts_dir"])
SPLITS_DIR = ART / "splits"
FEATS_DIR = ART / "features"
SCALED_DIR = ART / "features_scaled"

TIME_COL = cfg["data"]["time_col"]
ID_COL = cfg["data"].get("id_col", "station_id")

def _dup_report(df: pd.DataFrame, name: str):
    if df.empty:
        print(f"[{name}] empty")
        return
    if not {ID_COL, TIME_COL}.issubset(df.columns):
        print(f"[{name}] missing key columns; has: {sorted(df.columns)}")
        return
    total = len(df)
    dups_mask = df.duplicated(subset=[ID_COL, TIME_COL], keep=False)
    n_dups = int(dups_mask.sum())
    n_dup_groups = df.loc[dups_mask, [ID_COL, TIME_COL]].drop_duplicates().shape[0]
    print(f"[{name}] rows={total:,}  dup_rows={n_dups:,}  dup_key_groups={n_dup_groups:,}")
    if n_dups:
        top = (df.loc[dups_mask, [ID_COL, TIME_COL]]
                 .groupby(ID_COL).size().sort_values(ascending=False).head(10))
        print("  top stations by dup rows:")
        for sid, cnt in top.items():
            print(f"    - {sid}: {cnt}")

def main():
    # 1) splits
    for split in ["train", "val", "test"]:
        p = SPLITS_DIR / f"{split}.parquet"
        df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        _dup_report(df, f"splits/{split}")

    # 2) features (full concatenated features)
    p_feats = FEATS_DIR / "dataset_features.parquet"
    feats = pd.read_parquet(p_feats) if p_feats.exists() else pd.DataFrame()
    _dup_report(feats, "features/dataset_features")

    # 3) scaled splits
    for split in ["train", "val", "test"]:
        p = SCALED_DIR / f"{split}.parquet"
        df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        _dup_report(df, f"features_scaled/{split}")

    # Optional: tiny JSON summary for CI/automation
    summary = {}
    def collect(df, key):
        if df.empty or not {ID_COL, TIME_COL}.issubset(df.columns):
            summary[key] = {"rows": int(len(df)), "dup_rows": None, "dup_key_groups": None}
        else:
            mask = df.duplicated(subset=[ID_COL, TIME_COL], keep=False)
            summary[key] = {
                "rows": int(len(df)),
                "dup_rows": int(mask.sum()),
                "dup_key_groups": int(df.loc[mask, [ID_COL, TIME_COL]].drop_duplicates().shape[0]),
            }

    for split in ["train", "val", "test"]:
        p = SPLITS_DIR / f"{split}.parquet"
        collect(pd.read_parquet(p) if p.exists() else pd.DataFrame(), f"splits/{split}")
    collect(feats, "features/dataset_features")
    for split in ["train", "val", "test"]:
        p = SCALED_DIR / f"{split}.parquet"
        collect(pd.read_parquet(p) if p.exists() else pd.DataFrame(), f"features_scaled/{split}")

    out = ART / "duplicates_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] wrote {out}")

if __name__ == "__main__":
    main()