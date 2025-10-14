from __future__ import annotations
from common.config_loader import load_cfg, require, make_abs
from pathlib import Path
import pandas as pd
import json
import yaml

CFG = yaml.safe_load(open("configs/default.yaml"))
ART = Path(CFG["paths"]["artifacts_dir"])
ID_COL = CFG["data"].get("id_col", "station_id")
TIME_COL = CFG["data"]["time_col"]

def _exists(p: Path): 
    return p.exists()

def _shape_info(p: Path):
    if not p.exists():
        return "missing"
    df = pd.read_parquet(p)
    st = df[ID_COL].nunique() if ID_COL in df else 0
    return f"rows={len(df):7d}  stations={st}"

def main():
    splits = ART / "splits"
    feats  = ART / "features" / "dataset_features.parquet"
    scaled = ART / "features_scaled"

    print("=== SPLITS CHECK ===")
    for name in ["train", "val", "test"]:
        p = splits / f"{name}.parquet"
        print(f"{name:<5} {_shape_info(p)}")

    print("\n=== FEATURES CHECK ===")
    if feats.exists():
        df = pd.read_parquet(feats)
        stn = df[ID_COL].nunique() if ID_COL in df else 0
        print(f"rows={len(df)}  stations={stn}  cols={len(df.columns)}")
        if TIME_COL in df:
            print(f"time range: {df[TIME_COL].min()} → {df[TIME_COL].max()}")
        num_cols_sample = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:10]
        print("sample numeric cols:", num_cols_sample)
    else:
        print("features parquet not found; run engineer_features.py")

    print("\n=== SCALED FEATURES CHECK ===")
    if (scaled / "train.parquet").exists():
        for name in ["train", "val", "test"]:
            print(f"scaled-{name:<5} {_shape_info(scaled / f'{name}.parquet')}")
        meta = json.load(open(scaled / "scaler.json"))
        print(f"scaler mode={meta['mode']} numeric_cols={len(meta['scaled_numeric_cols'])}")
    else:
        print("scaled features not found; run scale_features.py")

if __name__ == "__main__":
    main()
