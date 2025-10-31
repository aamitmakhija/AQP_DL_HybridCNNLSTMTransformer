# 2_dataprep/03_scale_features.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set

import numpy as np
import pandas as pd

from common.config_loader import load_cfg  # overlay-aware


# ----------------------- small helpers -----------------------

def _read_df(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "feather":
        return pd.read_feather(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported input format: {fmt}")

def _write_df(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "feather":
        df.to_feather(path)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise SystemExit(f"Unsupported output format: {fmt}")

def _is_identifier(col: str) -> bool:
    c = str(col).lower()
    return (
        c in {"id", "station", "station_id", "district_id", "city_id"} or
        c.endswith("_id")
    )

def _numeric_cols(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    """
    Numeric, non-excluded, and not identifier-like.
    """
    ex = set(exclude)
    cols = []
    for c in df.columns:
        if c in ex:
            continue
        if _is_identifier(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def _split_paths(cfg: Dict, base_dir: Path) -> Tuple[Dict[str, Path], str]:
    """
    Resolve input split filenames and format from config.
    """
    names: Dict[str, str] = cfg.get("output", {}).get("split_filenames", {}) or {}
    in_fmt: str = cfg.get("output", {}).get("format", "parquet")
    paths = {
        "train": base_dir / names.get("train", f"train.{in_fmt}"),
        "val":   base_dir / names.get("val",   f"val.{in_fmt}"),
        "test":  base_dir / names.get("test",  f"test.{in_fmt}"),
    }
    return paths, in_fmt

def _out_fmt(cfg: Dict, fallback: str) -> str:
    return cfg.get("scaling", {}).get("output_format", fallback)

def _safe_div(num: np.ndarray, denom: np.ndarray, eps: float) -> np.ndarray:
    return num / np.where(np.abs(denom) < eps, eps, denom)


# ---------------------- parameter fitting ----------------------

def _fit_params_standard(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    return {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in cols}

def _fit_params_minmax(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    return {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in cols}

def _fit_params_robust(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    med = df[cols].median()
    out = {}
    for c in cols:
        out[c] = {"median": float(med[c]), "iqr": float(q3[c] - q1[c])}
    return out

def _apply_standard(g: pd.DataFrame, cols: List[str], params: Dict[str, Dict[str, float]], eps: float):
    for c in cols:
        p = params.get(c, {})
        std = p.get("std", 1.0)
        mean = p.get("mean", 0.0)
        g[c] = (g[c] - mean) / (std if abs(std) >= eps else eps)
    return g

def _apply_minmax(g: pd.DataFrame, cols: List[str], params: Dict[str, Dict[str, float]], eps: float):
    for c in cols:
        p = params.get(c, {})
        lo = p.get("min", 0.0)
        hi = p.get("max", 1.0)
        denom = (hi - lo) if abs(hi - lo) >= eps else eps
        g[c] = (g[c] - lo) / denom
    return g

def _apply_robust(g: pd.DataFrame, cols: List[str], params: Dict[str, Dict[str, float]], eps: float):
    for c in cols:
        p = params.get(c, {})
        med = p.get("median", 0.0)
        iqr = p.get("iqr", 1.0)
        g[c] = (g[c] - med) / (iqr if abs(iqr) >= eps else eps)
    return g


# ---------------------- feature lock helpers ----------------------

def _intersect_columns(*dfs: pd.DataFrame) -> Set[str]:
    """
    Columns present in ALL provided DataFrames.
    """
    common: Set[str] = set(dfs[0].columns)
    for d in dfs[1:]:
        common &= set(d.columns)
    return common

def _build_feature_lock(
    train_s: pd.DataFrame,
    val_s: pd.DataFrame,
    test_s: pd.DataFrame,
    id_col: str,
    time_col: str,
    y_col: str,
    exclude_cols: Iterable[str],
) -> Dict:
    """
    Build lock from the scaled splits using the intersection of columns across
    splits, minus exclusions, keeping TRAIN's column order; only numeric dtypes.
    Also guards against identifier-like columns accidentally entering X.
    """
    exclude = set(exclude_cols) | {id_col, time_col, y_col}
    common = _intersect_columns(train_s, val_s, test_s)
    ordered = [c for c in train_s.columns if c in common]
    X_cols_ordered = [
        c for c in ordered
        if c not in exclude
        and not _is_identifier(c)                      # ← guard
        and pd.api.types.is_numeric_dtype(train_s[c])
    ]
    return {
        "time_col": time_col,
        "target_col": y_col,
        "id_cols": [id_col],
        "X_cols_ordered": X_cols_ordered,
    }


# -------------------------- main flow ---------------------------

def main():
    cfg = load_cfg()

    # Paths
    art_dir = Path(cfg["paths"]["artifacts_dir"])
    splits_dir = art_dir / cfg["paths"].get("splits_dir", "splits")
    scaled_dir = art_dir / cfg["paths"].get("features_scaled_dir", "features_scaled_ps")

    split_paths, in_fmt = _split_paths(cfg, splits_dir)
    out_fmt = _out_fmt(cfg, in_fmt)

    # Columns
    id_col = cfg["data"].get("id_col", "station_id")
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"].get("target", "PM25_Concentration")

    # Scaling config
    s_cfg = cfg.get("scaling", {})
    s_type = s_cfg.get("type", "standard").lower()       # standard|minmax|robust
    per_station = bool(s_cfg.get("per_station", True))   # default True

    # Exclusions: scaling.exclude_columns ∪ missing.drop_features ∪ {id, time} ∪ (target)
    exclude_cols = set(s_cfg.get("exclude_columns", []) or [])
    exclude_cols |= set(cfg.get("missing", {}).get("drop_features", []) or {})
    exclude_cols |= {id_col, time_col}
    eps = float(s_cfg.get("epsilon", 1e-8))

    # Read splits
    train = _read_df(split_paths["train"], in_fmt)
    val   = _read_df(split_paths["val"],   in_fmt)
    test  = _read_df(split_paths["test"],  in_fmt)

    # Coerce dtypes
    if time_col in train.columns:
        train[time_col] = pd.to_datetime(train[time_col], errors="coerce")
    if time_col in val.columns:
        val[time_col] = pd.to_datetime(val[time_col], errors="coerce")
    if time_col in test.columns:
        test[time_col] = pd.to_datetime(test[time_col], errors="coerce")
    if id_col in train.columns:
        train[id_col] = train[id_col].astype(str)
    if id_col in val.columns:
        val[id_col] = val[id_col].astype(str)
    if id_col in test.columns:
        test[id_col] = test[id_col].astype(str)

    # What to scale (don’t scale target unless explicitly removed from exclude)
    if target_col:
        exclude_cols.add(target_col)

    # Fit on TRAIN only
    num_cols = _numeric_cols(train, exclude=exclude_cols)
    if not num_cols:
        raise SystemExit("No numeric feature columns found to scale (after exclusions).")

    # ------------ Fit parameters on train (global and/or per-station) ------------
    def _fit(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if s_type == "standard":
            return _fit_params_standard(df, num_cols)
        if s_type == "minmax":
            return _fit_params_minmax(df, num_cols)
        if s_type == "robust":
            return _fit_params_robust(df, num_cols)
        raise SystemExit(f"Unknown scaling.type: {s_type}")

    # Global params (useful as fallback)
    global_params = _fit(train)

    # Per-station params (fit only on that station’s TRAIN slice)
    station_params: Dict[str, Dict[str, Dict[str, float]]] = {}
    if per_station and id_col in train.columns:
        for sid, g in train.groupby(id_col):
            station_params[str(sid)] = _fit(g)

    # ------------ Apply transform to each split ------------
    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if per_station and id_col in out.columns and station_params:
            # apply per-station if available, else fallback to global params
            for sid, g in out.groupby(id_col):
                params = station_params.get(str(sid), global_params)
                if s_type == "standard":
                    out.loc[g.index, num_cols] = _apply_standard(g[num_cols].copy(), num_cols, params, eps)
                elif s_type == "minmax":
                    out.loc[g.index, num_cols] = _apply_minmax(g[num_cols].copy(), num_cols, params, eps)
                else:  # robust
                    out.loc[g.index, num_cols] = _apply_robust(g[num_cols].copy(), num_cols, params, eps)
        else:
            # global scaling
            g = out[num_cols].copy()
            if s_type == "standard":
                out[num_cols] = _apply_standard(g, num_cols, global_params, eps)
            elif s_type == "minmax":
                out[num_cols] = _apply_minmax(g, num_cols, global_params, eps)
            else:
                out[num_cols] = _apply_robust(g, num_cols, global_params, eps)
        return out

    train_s = _apply(train)
    val_s   = _apply(val)
    test_s  = _apply(test)

    # ------------ Persist scaled splits ------------
    _write_df(train_s, scaled_dir / f"train.{out_fmt}", out_fmt)
    _write_df(val_s,   scaled_dir / f"val.{out_fmt}",   out_fmt)
    _write_df(test_s,  scaled_dir / f"test.{out_fmt}",  out_fmt)

    # ------------ Scaler meta (single canonical file) ------------
    mode_scope = f"{s_type}_{'per_station' if per_station else 'global'}"
    meta = {
        "mode": mode_scope,
        "type": s_type,
        "per_station": per_station,
        "id_col": id_col,
        "time_col": time_col,
        "exclude_columns": sorted(list(exclude_cols)),
        "scaled_numeric_cols": num_cols,
        "global_params": global_params,                          # always included (fallback reference)
        "station_params": station_params if per_station else {}, # empty when global
    }
    (scaled_dir / "scaler_params.json").write_text(json.dumps(meta, indent=2))

    # ------------ Feature lock (from SCALED splits) ------------
    lock = _build_feature_lock(
        train_s=train_s,
        val_s=val_s,
        test_s=test_s,
        id_col=id_col,
        time_col=time_col,
        y_col=target_col,
        exclude_cols=exclude_cols,  # honors scaling.exclude_columns + missing.drop_features + id/time/target
    )
    lock_dir = art_dir / "features_locked"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "feature_list.json").write_text(json.dumps(lock, indent=2))

    # ------------ Console summary ------------
    print(f"[OK] wrote scaled splits → {scaled_dir}")
    print(f"  train: rows={len(train_s):,}  stations={train_s[id_col].nunique() if id_col in train_s else 0}")
    print(f"  val:   rows={len(val_s):,}    stations={val_s[id_col].nunique() if id_col in val_s else 0}")
    print(f"  test:  rows={len(test_s):,}   stations={test_s[id_col].nunique() if id_col in test_s else 0}")
    print(f"[OK] scaler meta → {scaled_dir/'scaler_params.json'}  (mode={mode_scope})")
    print(f"[lock] wrote {lock_dir/'feature_list.json'} with {len(lock['X_cols_ordered'])} features")


if __name__ == "__main__":
    main()