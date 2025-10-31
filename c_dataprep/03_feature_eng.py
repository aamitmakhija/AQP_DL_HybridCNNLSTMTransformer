# c_dataprep/02_engineer_features.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Iterable
import numpy as np
import pandas as pd

from common.config_loader import load_cfg  # overlay-aware

def _numeric_cols(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    ex = set(exclude)
    return [c for c in df.columns if c not in ex and pd.api.types.is_numeric_dtype(df[c])]

def _create_target_lags(df: pd.DataFrame, id_col: str, y_col: str, lags: List[int]) -> pd.DataFrame:
    if not lags: return df
    out = df.copy()
    for h in sorted(set(int(x) for x in lags)):
        out[f"{y_col}_lag{h}h"] = out.groupby(id_col, sort=False)[y_col].shift(h)
    return out

def _create_rolling(df: pd.DataFrame, id_col: str, cols: List[str], windows: List[int]) -> pd.DataFrame:
    """Rolling stats are *left-closed* past-only (exclude current row)."""
    if not cols or not windows: return df
    out = df.copy()
    g = out.groupby(id_col, sort=False)
    for c in cols:
        s = g[c]
        for w in sorted(set(int(x) for x in windows)):
            r = s.shift(1).rolling(window=w, min_periods=1)  # shift(1) to exclude current time
            out[f"{c}_roll{w}h_mean"] = r.mean().reset_index(level=0, drop=True)
            out[f"{c}_roll{w}h_std"]  = r.std(ddof=0).reset_index(level=0, drop=True)
    return out

def _add_time_feats(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out[time_col], errors="coerce")
    hour = t.dt.hour.astype("Int16")
    dow  = t.dt.dayofweek.astype("Int16")
    out["hour_sin"] = np.sin(2*np.pi*hour/24.0)
    out["hour_cos"] = np.cos(2*np.pi*hour/24.0)
    out["dow_sin"]  = np.sin(2*np.pi*dow/7.0)
    out["dow_cos"]  = np.cos(2*np.pi*dow/7.0)
    return out

def main():
    cfg: Dict = load_cfg()
    art_dir = Path(cfg["paths"]["artifacts_dir"])
    features_dir = art_dir / cfg["paths"].get("features_dir", "features")
    features_dir.mkdir(parents=True, exist_ok=True)

    # Base features file produced by earlier steps (imputed, etc.)
    in_path  = features_dir / cfg.get("features", {}).get("output", {}).get("features_file", "dataset_features.parquet")
    out_path = in_path  # overwrite in-place (matches your logs)
    manifest_path = features_dir / cfg.get("features", {}).get("output", {}).get("manifest_file", "features_manifest.json")

    id_col   = cfg["data"].get("id_col", "station_id")
    time_col = cfg["data"]["time_col"]
    y_col    = cfg["data"].get("target", "PM25_Concentration")

    lags = list(cfg.get("lags", {}).get("hours", []) or [])
    roll_windows = list(cfg.get("features", {}).get("rolling_windows", []) or [])
    roll_cols = list(cfg["data"].get("features", []) or [])  # base set to roll

    print(f"[features] loading → {in_path}")
    df = pd.read_parquet(in_path)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df.sort_values([id_col, time_col], inplace=True, kind="mergesort")

    # --- Feature engineering (past-only) ---
    before_cols = set(df.columns)

    # 1) target lags
    df = _create_target_lags(df, id_col=id_col, y_col=y_col, lags=lags)

    # 2) rolling stats (exclude current row to avoid leakage)
    if roll_cols and roll_windows:
        # guard against accidentally rolling the target without shift
        safe_roll_cols = [c for c in roll_cols if c != y_col]
        df = _create_rolling(df, id_col=id_col, cols=safe_roll_cols, windows=roll_windows)

    # 3) cyclical time features
    df = _add_time_feats(df, time_col=time_col)

    added = [c for c in df.columns if c not in before_cols]
    print(f"[features] added {len(added)} columns: sample={added[:8]}")

    # --- Persist engineered features ---
    df.to_parquet(out_path, index=False)
    print(f"[OK] wrote features → {out_path}")

    # --- Build manifest/lock (numeric X only; exclude id/time/target) ---
    exclude_cols = {id_col, time_col, y_col}
    X_cols = _numeric_cols(df, exclude=exclude_cols)
    manifest = {
        "time_col": time_col,
        "target_col": y_col,
        "id_cols": [id_col],
        "X_cols_ordered": X_cols,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[OK] wrote manifest → {manifest_path}  (|X|={len(X_cols)})")

if __name__ == "__main__":
    main()