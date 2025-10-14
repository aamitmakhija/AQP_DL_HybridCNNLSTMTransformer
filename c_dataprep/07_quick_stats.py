# 2_dataprep/07_quick_stats.py
from __future__ import annotations
import os
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict
import pandas as pd
import yaml

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
SCALED = ART / "features_scaled"
LOCKED = ART / "features_locked" / "feature_list.json"
OUT = ART / "quick_stats.json"

def _summ(df: pd.DataFrame, cols: list[str], tag: str):
    recs = {}
    for c in cols:
        if c in df.columns:
            s = df[c].dropna()
            recs[c] = {
                "rows": len(s),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            }
    return {tag: recs}

def main():
    # Load locked schema
    if not LOCKED.exists():
        raise FileNotFoundError(f"{LOCKED} missing — run scale_features first.")
    schema = json.load(open(LOCKED))
    target = schema["target_col"]
    # pick first 5 features just for overview
    feats = schema["X_cols_ordered"][:5]

    # Read splits
    train = pd.read_parquet(SCALED / "train.parquet")
    val   = pd.read_parquet(SCALED / "val.parquet")
    test  = pd.read_parquet(SCALED / "test.parquet")

    stats = {}
    stats.update(_summ(train, [target] + feats, "train"))
    stats.update(_summ(val,   [target] + feats, "val"))
    stats.update(_summ(test,  [target] + feats, "test"))

    with open(OUT, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] wrote {OUT}")

    # print summary to console
    for split, recs in stats.items():
        print(f"\n=== {split.upper()} ===")
        for c, m in recs.items():
            print(f"{c:25s} rows={m['rows']:7d} mean={m['mean']:.3f} std={m['std']:.3f} "
                  f"min={m['min']:.3f} max={m['max']:.3f}")

if __name__ == "__main__":
    main()