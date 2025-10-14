from __future__ import annotations
from common.config_loader import load_cfg, require, make_abs
# dataprep/engineer_features.py

import os, json
from pathlib import Path
from typing import Dict, List
from copy import deepcopy

import yaml
import pandas as pd
import pyarrow.dataset as ds

# ---------- config ----------
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

# ---------- helpers ----------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def _pick_pm25_col(cols: List[str], cfg: Dict) -> str | None:
    cand = set([c.lower() for c in cfg["data"]["headers"]["airquality"]["pm25"]])
    fallbacks = ["pm25_concentration", "pm25", "pm2.5"]
    for c in cols:
        if c.lower() in cand or c.lower() in fallbacks:
            return c
    return None

# ---------- feature builder ----------
def _feature_block(df: pd.DataFrame, cfg: Dict, horizons: List[int], roll_hours: List[int]) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    id_col   = cfg["data"].get("id_col", "station_id")
    pm25_col = _pick_pm25_col(df.columns.tolist(), cfg)

    met_cols = [c for c in [
        "temperature","pressure","humidity","wind_speed","wind_direction","weather",
        "up_temperature","bottom_temperature","wind_level"
    ] if c in df.columns]

    base_cols = [id_col, time_col] + ([pm25_col] if pm25_col else []) + met_cols
    df = df[base_cols].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([id_col, time_col])

    def _per_station(g: pd.DataFrame) -> pd.DataFrame:
        sid = g.name  # group key
        g = g.sort_values(time_col).reset_index(drop=True).copy()

        key_block = pd.DataFrame({
            id_col: sid,
            time_col: g[time_col]
        })
        frames = [key_block]

        if pm25_col and pm25_col in g.columns:
            frames.append(pd.concat(
                {f"{pm25_col}_lag{h}h": g[pm25_col].shift(h) for h in horizons},
                axis=1
            ))

        num_cols = g.select_dtypes(include="number").columns.tolist()
        if num_cols:
            frames.append(g[num_cols])
            for w in roll_hours:
                roll = g[num_cols].rolling(window=w, min_periods=max(1, w // 2))
                frames.append(roll.mean().add_suffix(f"_roll{w}h_mean"))
                frames.append(roll.std().add_suffix(f"_roll{w}h_std"))

        return pd.concat(frames, axis=1)

    out = (
        df.groupby(id_col, group_keys=False)
          .apply(_per_station, include_groups=False)
          .reset_index(drop=True)
          .copy()
    )
    return out

# ---------- main ----------
def engineer():
    cfg = _load_cfg()
    art_dir = Path(cfg["paths"]["artifacts_dir"])
    ds_dir  = art_dir / "dataset_stream"
    splits  = art_dir / "splits"
    out_dir = art_dir / "features"
    _ensure_dir(out_dir)

    time_col = cfg["data"]["time_col"]
    id_col   = cfg["data"].get("id_col", "station_id")
    roll_default = int(cfg.get("missing", {}).get("rolling_mean_hours", 24))
    horizons = [1, 3, 6, 12, 24]
    roll_windows = [6, 12, roll_default, 48, 72]

    train = _read_parquet(splits / "train.parquet")
    val   = _read_parquet(splits / "val.parquet")
    test  = _read_parquet(splits / "test.parquet")

    if train.empty and not ds_dir.exists():
        raise FileNotFoundError("No splits and no dataset_stream found. Run ingestion first.")

    if train.empty:
        dset = ds.dataset(str(ds_dir), format="parquet")
        base_df = dset.to_table().to_pandas()
    else:
        base_df = pd.concat([train, val, test], ignore_index=True)

    base_df[time_col] = pd.to_datetime(base_df[time_col], errors="coerce")
    base_df = base_df[base_df[time_col].dt.year != 1970].copy()
    base_df = base_df.sort_values([id_col, time_col])

    feats = _feature_block(base_df, cfg, horizons=horizons, roll_hours=roll_windows)

    # Print split date ranges
    print("\n=== SPLIT DATE RANGES (from features) ===")
    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        if df.empty:
            print(f"[{split_name}] missing or empty")
            continue
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        print(f"[{split_name}] rows={len(df):7d} start={df[time_col].min()} end={df[time_col].max()}")

    feats.to_parquet(out_dir / "dataset_features.parquet", index=False)

    manifest = {
        "rows": int(len(feats)),
        "stations": int(feats[id_col].nunique()) if id_col in feats else 0,
        "cols": feats.columns.tolist(),
        "horizons": horizons,
        "roll_windows": roll_windows,
        "time_min": str(feats[time_col].min()) if len(feats) else None,
        "time_max": str(feats[time_col].max()) if len(feats) else None,
    }
    with open(out_dir / "features_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] wrote features → {out_dir/'dataset_features.parquet'}")
    print(f"[OK] wrote manifest → {out_dir/'features_manifest.json'}")

if __name__ == "__main__":
    engineer()
