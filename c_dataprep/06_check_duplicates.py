# 2_dataprep/06_check_duplicates.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import pandas as pd

from common.config_loader import load_cfg  # use the unified loader

# ---------- IO helpers ----------
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

def _dup_report(df: pd.DataFrame, name: str, id_col: str, time_col: str):
    if df.empty:
        print(f"[{name}] empty or missing")
        return
    if not {id_col, time_col}.issubset(df.columns):
        print(f"[{name}] missing key columns; has keys={sorted(set(df.columns) & {id_col, time_col})}")
        return
    total = len(df)
    mask = df.duplicated(subset=[id_col, time_col], keep=False)
    n_dups = int(mask.sum())
    n_groups = df.loc[mask, [id_col, time_col]].drop_duplicates().shape[0]
    print(f"[{name}] rows={total:,}  dup_rows={n_dups:,}  dup_key_groups={n_groups:,}")
    if n_dups:
        top = (df.loc[mask, [id_col, time_col]]
               .groupby(id_col).size().sort_values(ascending=False).head(10))
        print("  top stations by dup rows:")
        for sid, cnt in top.items():
            print(f"    - {sid}: {cnt}")

def main():
    cfg = load_cfg()

    # Paths (all config-driven)
    art_dir      = Path(cfg["paths"]["artifacts_dir"])
    splits_dir   = art_dir / cfg["paths"].get("splits_dir", "splits")
    features_dir = art_dir / cfg["paths"].get("features_dir", "features")
    scaled_dir   = art_dir / cfg["paths"].get("features_scaled_dir", "features_scaled_ps")

    # Columns
    id_col   = cfg["data"].get("id_col", "station_id")
    time_col = cfg["data"]["time_col"]

    # Formats & filenames
    split_fmt  = cfg.get("output", {}).get("format", "parquet")
    split_names: Dict[str, str] = cfg.get("output", {}).get("split_filenames", {}) or {}
    def split_path(name: str) -> Path:
        # default name if not provided in config
        default = f"{name}.{split_fmt}"
        return splits_dir / (split_names.get(name, default))

    feats_out_cfg = cfg.get("features", {}).get("output", {})
    feats_fmt  = feats_out_cfg.get("format", "parquet")
    feats_file = feats_out_cfg.get("features_file", "dataset_features.parquet")
    feats_path = features_dir / feats_file

    scaled_fmt = cfg.get("scaling", {}).get("output_format", split_fmt)
    def scaled_path(name: str) -> Path:
        default = f"{name}.{scaled_fmt}"
        return scaled_dir / (split_names.get(name, default))

    # Output summary filename (configurable, with safe default)
    reports_cfg = cfg.get("reports", {})
    summary_out = art_dir / reports_cfg.get("duplicates_summary", "duplicates_summary.json")

    # -------- run reports --------
    # 1) raw splits
    for name in ("train", "val", "test"):
        df = _read_df(split_path(name), split_fmt)
        _dup_report(df, f"splits/{name}", id_col, time_col)

    # 2) engineered features
    feats = _read_df(feats_path, feats_fmt)
    _dup_report(feats, "features/dataset_features", id_col, time_col)

    # 3) scaled splits
    for name in ("train", "val", "test"):
        df = _read_df(scaled_path(name), scaled_fmt)
        _dup_report(df, f"features_scaled/{name}", id_col, time_col)

    # -------- JSON summary (for CI/automation) --------
    summary: Dict[str, Dict[str, int | None]] = {}

    def collect(df: pd.DataFrame, key: str):
        if df.empty or not {id_col, time_col}.issubset(df.columns):
            summary[key] = {"rows": int(len(df)), "dup_rows": None, "dup_key_groups": None}
            return
        mask = df.duplicated(subset=[id_col, time_col], keep=False)
        summary[key] = {
            "rows": int(len(df)),
            "dup_rows": int(mask.sum()),
            "dup_key_groups": int(df.loc[mask, [id_col, time_col]].drop_duplicates().shape[0]),
        }

    for name in ("train", "val", "test"):
        collect(_read_df(split_path(name), split_fmt), f"splits/{name}")
    collect(feats, "features/dataset_features")
    for name in ("train", "val", "test"):
        collect(_read_df(scaled_path(name), scaled_fmt), f"features_scaled/{name}")

    summary_out.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] wrote {summary_out}")

if __name__ == "__main__":
    main()