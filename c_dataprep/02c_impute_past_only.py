#!/usr/bin/env python3
# c_dataprep/02c_impute_past_only.py  (STRICT YAML, format-aware)
from __future__ import annotations
from pathlib import Path
import os, sys
from typing import Any, Dict, List

import pandas as pd

from common.config_loader import load_cfg

# ---------- tiny utils ----------
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

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_df(path: Path, fmt: str) -> pd.DataFrame:
    f = (fmt or "").lower()
    if f == "parquet": return pd.read_parquet(path)
    if f == "feather": return pd.read_feather(path)
    if f == "csv":     return pd.read_csv(path)
    raise SystemExit(f"Unsupported features.output.format: {fmt!r}")

def _write_df(df: pd.DataFrame, path: Path, fmt: str) -> None:
    _ensure_dir(path.parent)
    f = (fmt or "").lower()
    if f == "parquet": df.to_parquet(path, index=False)
    elif f == "feather": df.reset_index(drop=True).to_feather(path)
    elif f == "csv": df.to_csv(path, index=False)
    else: raise SystemExit(f"Unsupported features.output.format: {fmt!r}")

# ---------- main ----------
def main() -> None:
    cfg = load_cfg()

    # paths (strict)
    art_dir  = _under(Path("."), _req(cfg, ["paths", "artifacts_dir"]))
    feat_dir = _under(art_dir, _req(cfg, ["paths", "features_dir"]))

    # schema (strict)
    id_col   = _req(cfg, ["data", "id_col"])
    time_col = _req(cfg, ["data", "time_col"])
    target   = _req(cfg, ["data", "target"])

    # IO names + format (strict)
    out_cfg        = _req(cfg, ["features", "output"])
    io_fmt         = _req(out_cfg, ["format"])                 # parquet|feather|csv
    in_name        = _req(out_cfg, ["features_file"])
    out_name       = _req(out_cfg, ["features_file_imputed"])
    activate       = bool(_req(out_cfg, ["activate_imputed"])) # True/False
    activate_mode  = str(_req(out_cfg, ["activate_mode"]))     # symlink|overwrite

    # impute params (strict)
    imp_cfg    = _req(cfg, ["missing", "impute"])
    enabled    = bool(_req(imp_cfg, ["enabled"]))
    ffill_lim  = int(_req(imp_cfg, ["ffill_limit"]))
    roll_win   = int(_req(imp_cfg, ["rolling_window"]))
    use_median = bool(_req(imp_cfg, ["median_fallback"]))

    if not enabled:
        print("[impute] disabled via YAML; exiting.")
        sys.exit(0)

    in_path  = feat_dir / in_name
    out_path = feat_dir / out_name
    if not in_path.exists():
        raise FileNotFoundError(f"features not found: {in_path}")

    print(f"[config] CONFIG={os.environ.get('CONFIG','<unset>')}")
    print(f"[impute] loading: {in_path}  (format={io_fmt})")

    df = _read_df(in_path, io_fmt)

    # guards
    if time_col not in df.columns or id_col not in df.columns:
        raise SystemExit(f"columns missing: expected [{id_col},{time_col}] in {in_path}")

    # canonical order
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    # normalize station_id ONLY where not null; preserve real NaNs
    sid = df[id_col]
    m = sid.notna()
    df[id_col] = sid.astype("object")               # keep NaN as NaN
    df.loc[m, id_col] = sid.loc[m].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df.sort_values([id_col, time_col], inplace=True)

    # numeric feature columns only (exclude id/time/target)
    exclude = {id_col, time_col, target}
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        print("[impute] no numeric features; nothing to do.")
        sys.exit(0)

    # per-station, past-only imputation
    def _impute_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).copy()
        # 1) short gaps
        g[num_cols] = g[num_cols].ffill(limit=ffill_lim)
        # 2) past rolling mean (shifted)
        past_mean = g[num_cols].rolling(window=roll_win, min_periods=1).mean().shift(1)
        g[num_cols] = g[num_cols].combine_first(past_mean)
        # 3) expanding median fallback (shifted)
        if use_median:
            past_med = g[num_cols].expanding(min_periods=1).median().shift(1)
            g[num_cols] = g[num_cols].combine_first(past_med)
        return g

    try:
        df = df.groupby(id_col, group_keys=False, sort=False).apply(_impute_group, include_groups=False)
    except TypeError:
        df = df.groupby(id_col, group_keys=False, sort=False).apply(_impute_group)

    _write_df(df, out_path, io_fmt)
    print("[impute] wrote:", out_path)

    # activation strictly per YAML (with safe fallback)
    if activate:
        active_file = feat_dir / in_name
        try:
            if active_file.exists() or active_file.is_symlink():
                active_file.unlink()
            if activate_mode.lower() == "symlink" and active_file.name != out_path.name:
                rel = Path(os.path.relpath(out_path, start=feat_dir))
                active_file.symlink_to(rel)
                print("[impute] activated via symlink →", active_file, "→", rel)
            elif activate_mode.lower() == "overwrite" or active_file.name == out_path.name:
                _write_df(df, active_file, io_fmt)
                print("[impute] activated by overwrite:", active_file)
            else:
                raise ValueError("features.output.activate_mode must be 'symlink' or 'overwrite'")
        except Exception as e:
            print(f"[impute] activation failed ({e}); falling back to overwrite.")
            _write_df(df, active_file, io_fmt)
            print("[impute] activated by overwrite:", active_file)

if __name__ == "__main__":
    main()