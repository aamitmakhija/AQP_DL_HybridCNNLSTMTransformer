from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List

import pandas as pd
import numpy as np
import pyarrow.dataset as ds

from common.config_loader import load_cfg


# ---------- helpers ----------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def _pick_pm25_col(cols: List[str], aliases: List[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None

def _write_df(df: pd.DataFrame, out_fmt: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if out_fmt == "parquet":
        df.to_parquet(path, index=False)
    elif out_fmt == "feather":
        df.reset_index(drop=True).to_feather(path)
    elif out_fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise SystemExit(f"Unsupported features.output.format: {out_fmt}")

def _add_time_features(df: pd.DataFrame, time_col: str, enable: bool) -> pd.DataFrame:
    if not enable or time_col not in df.columns or df.empty:
        return df
    out = df.copy()
    # hour/day
    hr = out[time_col].dt.hour
    dow = out[time_col].dt.dayofweek
    # cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * hr / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hr / 24.0)
    out["dow_sin"]  = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0)
    return out


# ---------- feature builder ----------
def _feature_block(
    df: pd.DataFrame,
    cfg: Dict,
    horizons: List[int],
    roll_windows: List[int],
    add_time_feats: bool,
) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    id_col   = cfg["data"].get("id_col", "station_id")

    base_feats: List[str]   = cfg["data"].get("features", [])
    pm25_aliases: List[str] = cfg["data"]["headers"]["airquality"]["pm25"]
    pm25_col = _pick_pm25_col(df.columns.tolist(), pm25_aliases)

    # subset to id, time, pm25 (if available), and configured features that exist
    take = [c for c in base_feats if c in df.columns]
    base_cols = [id_col, time_col] + ([pm25_col] if pm25_col else []) + take
    base_cols = [c for c in base_cols if c in df.columns]

    if id_col not in df.columns:
        raise KeyError(f"{id_col} missing in base_df. Columns: {df.columns.tolist()}")

    df = df[base_cols].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([id_col, time_col])

    # informative note if PM2.5 not found
    if pm25_col is None and horizons:
        print("[warn] PM2.5 column not found by aliases; skipping lag features")

    groups = []
    for sid, g in df.groupby(id_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        base = pd.DataFrame({id_col: sid, time_col: g[time_col]})

        frames = [base]

        # lags for PM2.5 only (if present)
        if pm25_col and pm25_col in g.columns and horizons:
            lag_block = pd.concat(
                {f"{pm25_col}_lag{h}h": g[pm25_col].shift(h) for h in horizons},
                axis=1,
            )
            lag_block.columns = lag_block.columns.get_level_values(0)
            frames.append(lag_block)

        # rolling stats for numeric base features only (past-only window including current)
        num_cols = g[[c for c in take if c in g.columns]].select_dtypes(include="number").columns.tolist()
        if num_cols and roll_windows:
            frames.append(g[num_cols])
            for w in roll_windows:
                roll = g[num_cols].rolling(window=int(w), min_periods=max(1, int(w)//2))
                frames.append(roll.mean().add_suffix(f"_roll{w}h_mean"))
                frames.append(roll.std().add_suffix(f"_roll{w}h_std"))

        merged = pd.concat(frames, axis=1)

        # optional time features (same timestamp alignment)
        if add_time_feats:
            merged = _add_time_features(merged, time_col, enable=True)

        groups.append(merged)

    out = pd.concat(groups, ignore_index=True)

    # ensure unique columns (in case of accidental dupes)
    out.columns = [
        "_".join(map(str, c)) if isinstance(c, tuple) else str(c)
        for c in out.columns
    ]
    out = out.loc[:, ~out.columns.duplicated()].copy()

    return out


# ---------- main ----------
def engineer():
    cfg = load_cfg()

    art_dir  = Path(cfg["paths"]["artifacts_dir"])
    ds_dir   = art_dir / cfg["paths"].get("dataset_stream_dir", "dataset_stream")
    splits   = art_dir / cfg["paths"].get("splits_dir", "splits")
    feats_dirname = cfg["paths"].get("features_dir", "features")
    out_dir  = art_dir / feats_dirname
    _ensure_dir(out_dir)

    time_col = cfg["data"]["time_col"]
    id_col   = cfg["data"].get("id_col", "station_id")

    horizons: List[int] = list(cfg.get("lags", {}).get("hours", [1, 3, 6, 12, 24]))
    roll_default = int(cfg.get("missing", {}).get("rolling_mean_hours", 24))
    roll_windows: List[int] = list(cfg.get("features", {}).get("rolling_windows", [6, 12, roll_default, 48, 72]))

    # optional time features
    time_cfg = (cfg.get("features", {}).get("time_features", {}) or {})
    add_time_feats = bool(time_cfg.get("cyclical", True))

    out_cfg = cfg.get("features", {}).get("output", {})
    out_fmt      = out_cfg.get("format", "parquet")
    out_features = out_cfg.get("features_file", "dataset_features.parquet")
    out_manifest = out_cfg.get("manifest_file", "features_manifest.json")

    # load splits if present; else fall back to whole stream
    train = _read_parquet(splits / cfg.get("output", {}).get("split_filenames", {}).get("train", "train.parquet"))
    val   = _read_parquet(splits / cfg.get("output", {}).get("split_filenames", {}).get("val",   "val.parquet"))
    test  = _read_parquet(splits / cfg.get("output", {}).get("split_filenames", {}).get("test",  "test.parquet"))

    if train.empty and not ds_dir.exists():
        raise FileNotFoundError("No splits and no dataset_stream found. Run ingestion first.")

    if train.empty:
        base_df = ds.dataset(str(ds_dir), format="parquet").to_table().to_pandas()
    else:
        base_df = pd.concat([train, val, test], ignore_index=True)

    # optional station keep-list
    keep_txt = art_dir / "stations_keep.txt"
    if keep_txt.exists() and id_col in base_df.columns:
        keep_ids = {s.strip() for s in keep_txt.read_text().splitlines() if s.strip()}
        before = len(base_df)
        base_df = base_df[base_df[id_col].astype(str).isin(keep_ids)].copy()
        print(f"[filter] stations_keep.txt applied: rows {before:,} -> {len(base_df):,}")

    # time normalization + drop_years
    base_df[time_col] = pd.to_datetime(base_df[time_col], errors="coerce")
    drop_years = set(cfg.get("split", {}).get("drop_years", []))
    if drop_years:
        before = len(base_df)
        base_df = base_df[~base_df[time_col].dt.year.isin(drop_years)].copy()
        print(f"[filter] drop_years={sorted(drop_years)}: rows {before:,} -> {len(base_df):,}")

    base_df = base_df.sort_values([id_col, time_col]).reset_index(drop=True)
    feats = _feature_block(
        base_df, cfg,
        horizons=horizons,
        roll_windows=roll_windows,
        add_time_feats=add_time_feats,
    )

    # drop configured high-missing features
    drop_cfg_raw = (cfg.get("missing") or {}).get("drop_features", None)
    drop_list: List[str] = []
    if isinstance(drop_cfg_raw, (list, tuple, set)): drop_list = [str(c) for c in drop_cfg_raw]
    elif isinstance(drop_cfg_raw, str):              drop_list = [drop_cfg_raw]

    if drop_list:
        existing = [c for c in drop_list if c in feats.columns]
        if existing:
            print(f"[features] dropping columns from config missing.drop_features: {existing}")
            feats = feats.drop(columns=existing)

    # persist
    _write_df(feats, out_fmt, out_dir / out_features)

    manifest = {
        "rows": int(len(feats)),
        "stations": int(feats[id_col].nunique()) if id_col in feats else 0,
        "cols": feats.columns.tolist(),
        "horizons": horizons,
        "roll_windows": roll_windows,
        "time_min": str(feats[time_col].min()) if len(feats) else None,
        "time_max": str(feats[time_col].max()) if len(feats) else None,
        "dropped_from_config": drop_list,
        "time_features": {"cyclical": bool(add_time_feats)},
    }
    (out_dir / out_manifest).write_text(json.dumps(manifest, indent=2))

    print("\n=== SPLIT DATE RANGES (from features inputs) ===")
    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        if df.empty:
            print(f"[{split_name}] missing or empty")
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            print(f"[{split_name}] rows={len(df):7d} start={df[time_col].min()} end={df[time_col].max()}")

    print(f"[OK] wrote features → {out_dir/out_features}")
    print(f"[OK] wrote manifest → {out_dir/out_manifest}")


if __name__ == "__main__":
    engineer()