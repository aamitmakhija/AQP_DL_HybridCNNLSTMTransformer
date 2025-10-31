# c_dataprep/01_make_splits.py
from __future__ import annotations
from pathlib import Path
import json
import os
import pandas as pd

from common.config_loader import load_cfg


def _write_df(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "feather":
        # feather doesn't support non-default indices well; reset first
        df.reset_index(drop=True).to_feather(path)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise SystemExit(f"Unsupported output.format: {fmt}")


def main():
    cfg = load_cfg()

    # --- config fields ---
    id_col   = cfg["data"].get("id_col", "station_id")
    time_col = cfg["data"]["time_col"]

    art_dir   = Path(cfg["paths"]["artifacts_dir"])
    feats_dir = art_dir / cfg["paths"].get("features_dir", "features")
    splits_dir = art_dir / cfg["paths"].get("splits_dir", "splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    # engineered features file produced by 02_engineer_features.py
    feats_file = cfg.get("features", {}).get("output", {}).get("features_file", "dataset_features.parquet")
    feats_path = feats_dir / feats_file
    if not feats_path.exists():
        raise SystemExit(f"Engineered features not found: {feats_path} "
                         f"(did you run 02_engineer_features.py?)")

    out_fmt = cfg.get("output", {}).get("format", "parquet")
    split_names = cfg.get("output", {}).get("split_filenames", {}) or {}
    fn_train  = split_names.get("train",  "train.parquet")
    fn_val    = split_names.get("val",    "val.parquet")
    fn_test   = split_names.get("test",   "test.parquet")
    fn_sum    = split_names.get("summary","split_summary.json")

    train_end = pd.to_datetime(cfg["split"]["train_end"])
    val_end   = pd.to_datetime(cfg["split"]["val_end"])
    drop_years = set(cfg.get("split", {}).get("drop_years", []) or [])

    # --- load engineered features ---
    df = pd.read_parquet(feats_path)
    if time_col not in df.columns:
        raise SystemExit(f"Column '{time_col}' missing in {feats_path}")
    if id_col not in df.columns:
        raise SystemExit(f"Column '{id_col}' missing in {feats_path}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # optional station keep-list (from ingestion coverage step)
    keep_txt = art_dir / "stations_keep.txt"
    if keep_txt.exists():
        keep_ids = {s.strip() for s in keep_txt.read_text().splitlines() if s.strip()}
        # input may be numeric; normalize to string for comparison, then cast back
        sid_is_numeric = pd.api.types.is_integer_dtype(df[id_col]) or pd.api.types.is_float_dtype(df[id_col])
        df["_sid_str"] = df[id_col].astype(str)
        before = len(df)
        df = df[df["_sid_str"].isin(keep_ids)].drop(columns=["_sid_str"])
        if sid_is_numeric:
            # retain original dtype
            df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype(df[id_col].dtype)
        print(f"[filter] stations_keep.txt applied: rows {before:,} -> {len(df):,}")

    if drop_years:
        before = len(df)
        df = df[~df[time_col].dt.year.isin(drop_years)].copy()
        print(f"[filter] drop_years={sorted(drop_years)}: rows {before:,} -> {len(df):,}")

    # --- splits ---
    train = df[df[time_col] <= train_end].copy()
    val   = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test  = df[df[time_col] > val_end].copy()

    _write_df(train, splits_dir / fn_train, out_fmt)
    _write_df(val,   splits_dir / fn_val,   out_fmt)
    _write_df(test,  splits_dir / fn_test,  out_fmt)

    summary = {
        "cutoffs": {"train_end": str(train_end), "val_end": str(val_end)},
        "rows":    {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "stations":{
            "train": int(train[id_col].nunique()) if len(train) else 0,
            "val":   int(val[id_col].nunique()) if len(val) else 0,
            "test":  int(test[id_col].nunique()) if len(test) else 0,
        },
        "time_range": {
            "train": [str(train[time_col].min()), str(train[time_col].max())] if len(train) else None,
            "val":   [str(val[time_col].min()), str(val[time_col].max())]     if len(val)   else None,
            "test":  [str(test[time_col].min()), str(test[time_col].max())]   if len(test)  else None,
        },
        "source": str(feats_path),
    }
    (splits_dir / fn_sum).write_text(json.dumps(summary, indent=2))

    print(f"[config] CONFIG={os.environ.get('CONFIG','<env not set>')}")
    print(f"[input]  engineered features: {feats_path}")
    print(f"[split]  train_end={train_end}  val_end={val_end}  time_col={time_col}")
    print(f"[OK] wrote train rows={len(train):8d} → {splits_dir / fn_train}")
    print(f"[OK] wrote val   rows={len(val):8d} → {splits_dir / fn_val}")
    print(f"[OK] wrote test  rows={len(test):8d} → {splits_dir / fn_test}")
    print(f"[OK] wrote split summary → {splits_dir / fn_sum}")


if __name__ == "__main__":
    main()