#!/usr/bin/env python3
# c_dataprep/03b_scale_features_per_station.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Iterable, Set
import json
import numpy as np
import pandas as pd

# ensure repo root is importable
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.config_loader import load_cfg  # noqa: E402


# ---------- utils ----------
def _req(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise KeyError("configs: missing " + ".".join(path))
        cur = cur[k]
    return cur

def _under(root: Path, p: str) -> Path:
    P = Path(p)
    return P if P.is_absolute() else (root / P)

def _read_df(path: Path, fmt: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    f = fmt.lower()
    if f == "parquet": return pd.read_parquet(path)
    if f == "feather": return pd.read_feather(path)
    if f == "csv":     return pd.read_csv(path)
    raise SystemExit(f"Unsupported input format: {fmt!r}")

def _write_df(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = fmt.lower()
    if f == "parquet":   df.to_parquet(path, index=False)
    elif f == "feather": df.reset_index(drop=True).to_feather(path)
    elif f == "csv":     df.to_csv(path, index=False)
    else:                raise SystemExit(f"Unsupported output format: {fmt!r}")

def _is_identifier(col: str) -> bool:
    c = str(col).lower()
    return (c in {"id", "station", "station_id", "district_id", "city_id"} or c.endswith("_id"))

def _numeric_cols(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    ex = set(exclude)
    return [c for c in df.columns if c not in ex and not _is_identifier(c) and pd.api.types.is_numeric_dtype(df[c])]

def _intersect_columns(*dfs: pd.DataFrame) -> Set[str]:
    non_empty = [d for d in dfs if not d.empty]
    if not non_empty: return set()
    common: Set[str] = set(non_empty[0].columns)
    for d in non_empty[1:]: common &= set(d.columns)
    return common

def _norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


# ---------- scalers ----------
def _fit_params_standard(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    return {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in cols}

def _fit_params_minmax(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    return {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in cols}

def _fit_params_robust(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    q1 = df[cols].quantile(0.25); q3 = df[cols].quantile(0.75); med = df[cols].median()
    return {c: {"median": float(med[c]), "iqr": float(q3[c]-q1[c])} for c in cols}

def _apply_block(block: pd.DataFrame, cols: List[str], params: Dict[str, Dict[str, float]], eps: float, mode: str) -> pd.DataFrame:
    out = block.copy()
    if mode == "standard":
        for c in cols:
            p = params[c]; denom = p["std"] if abs(p["std"]) >= eps else eps
            out[c] = (out[c] - p["mean"]) / denom
    elif mode == "minmax":
        for c in cols:
            p = params[c]; denom = (p["max"] - p["min"]) if abs(p["max"]-p["min"]) >= eps else eps
            out[c] = (out[c] - p["min"]) / denom
    else:  # robust
        for c in cols:
            p = params[c]; denom = p["iqr"] if abs(p["iqr"]) >= eps else eps
            out[c] = (out[c] - p["median"]) / denom
    return out


# ---------- feature lock ----------
def _build_feature_lock(train_s: pd.DataFrame, val_s: pd.DataFrame, test_s: pd.DataFrame,
                        id_col: str, time_col: str, y_col: str, exclude_cols: Iterable[str]) -> Dict[str, Any]:
    exclude = set(exclude_cols) | {id_col, time_col, y_col}
    common = _intersect_columns(train_s, val_s, test_s)
    ordered = [c for c in train_s.columns if c in common] if not train_s.empty else sorted(common)
    X_cols_ordered = [c for c in ordered if c not in exclude and not _is_identifier(c) and
                      (not train_s.empty and pd.api.types.is_numeric_dtype(train_s[c]))]
    return {"time_col": time_col, "target_col": y_col, "id_cols": [id_col], "X_cols_ordered": X_cols_ordered}


# ---------- main ----------
def main():
    cfg = load_cfg()

    # paths
    art_dir    = _under(Path("."), _req(cfg, ["paths", "artifacts_dir"]))
    splits_dir = _under(art_dir, _req(cfg, ["paths", "splits_dir"]))
    scaled_dir = _under(art_dir, _req(cfg, ["paths", "features_scaled_dir"]))
    lock_dir   = _under(art_dir, _req(cfg, ["paths", "features_locked_dir"]))
    scaled_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)

    # formats
    in_fmt  = str(_req(cfg, ["output", "format"])).lower()            # input split format
    out_fmt = str(_req(cfg, ["scaling", "output_format"])).lower()    # output scaled splits format

    # split filenames (tolerant)
    names_cfg = (cfg.get("output", {}) or {}).get("split_filenames", {}) or {}
    def _fname(key: str) -> str: return str(names_cfg.get(key, f"{key}.{in_fmt}"))
    p_train = splits_dir / _fname("train")
    p_val   = splits_dir / _fname("val")
    p_test  = splits_dir / _fname("test")

    # columns
    id_col     = str(_req(cfg, ["data", "id_col"]))
    time_col   = str(_req(cfg, ["data", "time_col"]))
    target_col = str(_req(cfg, ["data", "target"]))

    # scaling config
    s_cfg       = _req(cfg, ["scaling"])
    mode        = str(_req(s_cfg, ["type"])).lower()                  # standard|minmax|robust
    per_station = bool(_req(s_cfg, ["per_station"]))
    eps         = float(_req(s_cfg, ["epsilon"]))

    excl_from_yaml = list((s_cfg.get("exclude_columns") or []))
    drop_feats_cfg = cfg.get("missing", {}).get("drop_features", [])
    drop_feats     = list(drop_feats_cfg if isinstance(drop_feats_cfg, (list, tuple, set)) else ([drop_feats_cfg] if drop_feats_cfg else []))
    exclude_cols   = set(excl_from_yaml) | set(drop_feats) | {id_col, time_col, target_col}

    # read splits
    train = _read_df(p_train, in_fmt)
    val   = _read_df(p_val,   in_fmt)
    test  = _read_df(p_test,  in_fmt)
    if train.empty:
        raise SystemExit(f"TRAIN split not found or empty at {p_train}")

    for df in (train, val, test):
        if not df.empty:
            if id_col in df.columns:   df[id_col] = _norm_id(df[id_col])
            if time_col in df.columns: df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # numeric cols from TRAIN only
    num_cols = _numeric_cols(train, exclude=exclude_cols)
    if not num_cols:
        raise SystemExit("No numeric feature columns to scale after exclusions.")

    # fit params
    def _fit(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if mode == "standard": return _fit_params_standard(df, num_cols)
        if mode == "minmax":   return _fit_params_minmax(df, num_cols)
        if mode == "robust":   return _fit_params_robust(df, num_cols)
        raise SystemExit(f"Unknown scaling.type: {mode}")

    global_params = _fit(train)
    station_params: Dict[str, Dict[str, Dict[str, float]]] = {}
    if per_station and id_col in train.columns:
        for sid, g in train.groupby(id_col, sort=False):
            station_params[str(sid)] = _fit(g)

    # apply
    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        out = df.copy()
        if per_station and id_col in out.columns and station_params:
            for sid, g in out.groupby(id_col, sort=False):
                params = station_params.get(str(sid), global_params)
                out.loc[g.index, num_cols] = _apply_block(g[num_cols], num_cols, params, eps, mode)
        else:
            out[num_cols] = _apply_block(out[num_cols], num_cols, global_params, eps, mode)
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out

    train_s, val_s, test_s = _apply(train), _apply(val), _apply(test)

    # write scaled splits
    _write_df(train_s, scaled_dir / f"train.{out_fmt}", out_fmt)
    _write_df(val_s,   scaled_dir / f"val.{out_fmt}",   out_fmt)
    _write_df(test_s,  scaled_dir / f"test.{out_fmt}",  out_fmt)

    # scaler meta
    meta = {
        "mode": f"{mode}_{'per_station' if per_station else 'global'}",
        "type": mode,
        "per_station": per_station,
        "id_col": id_col,
        "time_col": time_col,
        "exclude_columns": sorted(list(exclude_cols)),
        "scaled_numeric_cols": num_cols,
        "global_params": global_params,
        "station_params": station_params if per_station else {},
    }
    (scaled_dir / "scaler_params.json").write_text(json.dumps(meta, indent=2))

    # feature lock
    lock_name = str(_req(cfg, ["features", "locked_manifest"]))
    lock_obj  = _build_feature_lock(train_s, val_s, test_s, id_col, time_col, target_col, exclude_cols)
    (lock_dir / lock_name).write_text(json.dumps(lock_obj, indent=2))

    # console summary
    n_train_st = train_s[id_col].nunique() if id_col in train_s else 0
    n_val_st   = (val_s[id_col].nunique() if (not val_s.empty and id_col in val_s) else 0)
    n_test_st  = (test_s[id_col].nunique() if (not test_s.empty and id_col in test_s) else 0)
    print(f"[OK] wrote scaled splits → {scaled_dir}")
    print(f"  train: rows={len(train_s):,}  stations={n_train_st}")
    print(f"  val:   rows={len(val_s):,}    stations={n_val_st}")
    print(f"  test:  rows={len(test_s):,}   stations={n_test_st}")
    print(f"[OK] scaler meta → {scaled_dir/'scaler_params.json'}")
    print(f"[lock] wrote {(lock_dir/lock_name)} with {len(lock_obj['X_cols_ordered'])} features")


if __name__ == "__main__":
    main()