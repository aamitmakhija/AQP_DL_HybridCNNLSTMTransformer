# dataprep/make_splits.py
from __future__ import annotations
from pathlib import Path
import json
import os
from copy import deepcopy
from typing import Dict
import yaml
import pandas as pd
import pyarrow.dataset as ds

from common.config_loader import load_cfg, require, make_abs
cfg = load_cfg()


def _deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def main():
    cfg = load_cfg()
    time_col = cfg["data"]["time_col"]
    id_col   = cfg["data"].get("id_col", "station_id")  # <- use config, not hard-coded
    art_dir = Path(cfg["paths"]["artifacts_dir"])
    ds_dir = art_dir / "dataset_stream"
    out_dir = art_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    # read full dataset
    dset = ds.dataset(str(ds_dir), format="parquet")
    df = dset.to_table().to_pandas()

    # harmonize time + drop 1970
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].dt.year != 1970].copy()

    # split by dates
    train_end = pd.to_datetime(cfg["split"]["train_end"])
    val_end   = pd.to_datetime(cfg["split"]["val_end"])

    train = df[df[time_col] <= train_end]
    val   = df[(df[time_col] > train_end) & (df[time_col] <= val_end)]
    test  = df[df[time_col] > val_end]

    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    summary = {
        "rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "stations": {
            "train": int(train[id_col].nunique()) if id_col in train else 0,
            "val":   int(val[id_col].nunique()) if id_col in val else 0,
            "test":  int(test[id_col].nunique()) if id_col in test else 0,
        },
        "time_range": {
            "train": [str(train[time_col].min()), str(train[time_col].max())] if len(train) else None,
            "val":   [str(val[time_col].min()), str(val[time_col].max())] if len(val) else None,
            "test":  [str(test[time_col].min()), str(test[time_col].max())] if len(test) else None,
        },
        "cutoffs": {"train_end": str(train_end), "val_end": str(val_end)},
    }
    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[config] CONFIG={os.environ.get('CONFIG','configs/default.yaml')}")
    print(f"[input]  {ds_dir}")
    print(f"[split]  train_end={train_end}  val_end={val_end}  time_col={time_col}")
    print(f"[OK] wrote train rows={len(train):8d} → {out_dir/'train.parquet'}")
    print(f"[OK] wrote val   rows={len(val):8d} → {out_dir/'val.parquet'}")
    print(f"[OK] wrote test  rows={len(test):8d} → {out_dir/'test.parquet'}")
    print(f"[OK] wrote split summary → {out_dir/'split_summary.json'}")

if __name__ == "__main__":
    main()