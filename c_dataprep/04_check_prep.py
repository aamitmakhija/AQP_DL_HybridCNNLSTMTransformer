# checks/check_prep.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import json

import pandas as pd

from common.config_loader import load_cfg

# --------- small IO helpers ----------
def _read_df(path: Path, fmt: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "feather":
        return pd.read_feather(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported format: {fmt}")

def _shape_line(df: pd.DataFrame, id_col: str) -> str:
    if df.empty:
        return "missing"
    st = df[id_col].nunique() if id_col in df.columns else 0
    return f"rows={len(df):,}  stations={st}"

def main():
    cfg = load_cfg()

    art_dir      = Path(cfg["paths"]["artifacts_dir"])
    splits_dir   = art_dir / cfg["paths"].get("splits_dir", "splits")
    features_dir = art_dir / cfg["paths"].get("features_dir", "features")
    scaled_dir   = art_dir / cfg["paths"].get("features_scaled_dir", "features_scaled_ps")

    id_col   = cfg["data"].get("id_col", "station_id")
    time_col = cfg["data"]["time_col"]

    # split filenames & format
    split_names: Dict[str, str] = cfg.get("output", {}).get("split_filenames", {}) or {}
    in_fmt = cfg.get("output", {}).get("format", "parquet")

    # features output naming
    feats_cfg = cfg.get("features", {}).get("output", {})
    feats_file = feats_cfg.get("features_file", "dataset_features.parquet")
    feats_fmt  = feats_cfg.get("format", "parquet")

    # scaler meta filename (new scaler writes scaler_params.json)
    scaler_meta_candidates = ["scaler_params.json", "scaler.json"]

    # -------- SPLITS CHECK --------
    print("=== SPLITS CHECK ===")
    for name in ("train", "val", "test"):
        fn = split_names.get(name, f"{name}.parquet" if in_fmt == "parquet" else f"{name}.{in_fmt}")
        p  = splits_dir / fn
        df = _read_df(p, in_fmt)
        print(f"{name:<5} {_shape_line(df, id_col)}")

    # -------- FEATURES CHECK --------
    print("\n=== FEATURES CHECK ===")
    feats_path = features_dir / feats_file
    feats = _read_df(feats_path, feats_fmt)
    if feats.empty:
        print("features file not found or empty; run engineer_features.py")
    else:
        stn = feats[id_col].nunique() if id_col in feats.columns else 0
        print(f"rows={len(feats):,}  stations={stn}  cols={len(feats.columns)}")
        if time_col in feats.columns:
            tmin = pd.to_datetime(feats[time_col], errors="coerce").min()
            tmax = pd.to_datetime(feats[time_col], errors="coerce").max()
            print(f"time range: {tmin} → {tmax}")
        # sample some numeric columns
        num_cols_sample = [c for c in feats.columns if pd.api.types.is_numeric_dtype(feats[c])][:10]
        print(f"sample numeric cols: {num_cols_sample}")

    # -------- SCALED FEATURES CHECK --------
    print("\n=== SCALED FEATURES CHECK ===")
    # scaled splits assumed to mirror input split names & out format from scaling.output_format (fallback to in_fmt)
    out_fmt = cfg.get("scaling", {}).get("output_format", in_fmt)

    any_scaled = False
    for name in ("train", "val", "test"):
        fn = split_names.get(name, f"{name}.parquet" if out_fmt == "parquet" else f"{name}.{out_fmt}")
        p  = scaled_dir / fn
        df = _read_df(p, out_fmt)
        if not df.empty:
            any_scaled = True
        print(f"scaled-{name:<5} {_shape_line(df, id_col)}")

    # scaler meta
    meta_path = None
    for cand in scaler_meta_candidates:
        p = scaled_dir / cand
        if p.exists():
            meta_path = p
            break
    if meta_path:
        meta = json.loads(meta_path.read_text())
        mode = meta.get("mode", "<unknown>")
        cols = meta.get("columns") or meta.get("scaled_numeric_cols") or []
        print(f"scaler mode={mode} numeric_cols={len(cols)}")
    else:
        if any_scaled:
            print("scaler meta not found (looked for scaler_params.json / scaler.json)")
        else:
            print("scaled features not found; run scale_features_per_station.py")

if __name__ == "__main__":
    main()