#!/usr/bin/env python3
# c_dataprep/02_engineer_features.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# Ensure repo root on sys.path so "common" is importable when run as a script
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.config_loader import load_cfg


# ---------------- helpers ----------------

def _req(d: dict, path: List[str], name: str | None = None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            dotted = ".".join(path)
            raise KeyError(f"configs: missing {dotted}{' ('+name+')' if name else ''}")
        cur = cur[k]
    return cur

def _under(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (root / p)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_df(path: Path, fmt: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    f = (fmt or "").lower()
    if f == "parquet": return pd.read_parquet(path)
    if f == "feather": return pd.read_feather(path)
    if f == "csv":     return pd.read_csv(path)
    raise SystemExit(f"Unsupported input format: {fmt!r}")

def _write_df(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = (fmt or "").lower()
    if f == "parquet": df.to_parquet(path, index=False)
    elif f == "feather": df.reset_index(drop=True).to_feather(path)
    elif f == "csv": df.to_csv(path, index=False)
    else: raise SystemExit(f"Unsupported features.output.format: {fmt!r}")

def _norm_sid(s: str) -> str:
    return str(s).strip().removesuffix(".0")

def _pick(cols: Iterable[str], cand: Iterable[str]) -> Optional[str]:
    s = {c.lower(): c for c in cols}
    for a in cand:
        if a.lower() in s:
            return s[a.lower()]
    return None

def _add_time_features(df: pd.DataFrame, time_col: str, enable: bool) -> pd.DataFrame:
    if not enable or time_col not in df.columns or df.empty:
        return df
    out = df.copy()
    t = pd.to_datetime(out[time_col], errors="coerce")
    hour = t.dt.hour
    dow  = t.dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"]  = np.sin(2 * np.pi * dow  / 7.0)
    out["dow_cos"]  = np.cos(2 * np.pi * dow  / 7.0)
    return out


# ---------------- feature builders ----------------

def _create_target_lags(df: pd.DataFrame, id_col: str, y_col: str, lags: List[int]) -> pd.DataFrame:
    if not lags or y_col not in df.columns:
        return df
    out = df.copy()
    g = out.groupby(id_col, sort=False)[y_col]
    for h in sorted({int(x) for x in lags}):
        out[f"{y_col}_lag{h}h"] = g.shift(h)
    return out

def _create_rolling(df: pd.DataFrame, id_col: str, cols: List[str], windows: List[int]) -> pd.DataFrame:
    if not cols or not windows:
        return df
    take = [c for c in cols if c in df.columns]
    if not take:
        return df
    out = df.copy()
    g = out.groupby(id_col, sort=False)
    for c in take:
        if not pd.api.types.is_numeric_dtype(out[c]):
            continue
        s = g[c]
        for w in sorted({int(x) for x in windows}):
            r = s.shift(1).rolling(window=w, min_periods=1)
            out[f"{c}_roll{w}h_mean"] = r.mean().reset_index(level=0, drop=True)
            out[f"{c}_roll{w}h_std"]  = r.std(ddof=0).reset_index(level=0, drop=True)
    return out


# ---------------- main ----------------

def engineer():
    cfg: Dict = load_cfg()

    # Required paths (strict)
    art_rel    = _req(cfg, ["paths", "artifacts_dir"], "paths.artifacts_dir")
    feats_rel  = _req(cfg, ["paths", "features_dir"], "paths.features_dir")
    splits_rel = _req(cfg, ["paths", "splits_dir"], "paths.splits_dir")
    ds_rel     = _req(cfg, ["paths", "dataset_stream_dir"], "paths.dataset_stream_dir")

    # Resolve roots
    cwd       = Path(".")
    art_dir   = _under(cwd, art_rel)
    feats_dir = _under(art_dir, feats_rel)
    splits    = _under(art_dir, splits_rel)
    ds_dir    = _under(art_dir, ds_rel)
    _ensure_dir(feats_dir)

    # Schema keys (strict)
    id_col   = _req(cfg, ["data", "id_col"], "data.id_col")
    time_col = _req(cfg, ["data", "time_col"], "data.time_col")
    y_col    = _req(cfg, ["data", "target"], "data.target")

    # PM2.5 aliases (tolerant fallback to target)
    H_aq = ((cfg.get("data") or {}).get("headers") or {}).get("airquality", {}) or {}
    pm25_aliases = list(H_aq.get("pm25", [])) or [y_col]

    # Feature config (strict containers; content can be empty lists)
    data_feats   = list((_req(cfg, ["data", "features"], "data.features")) or [])
    lags_hours   = list((cfg.get("lags", {}) or {}).get("hours", []) or [])
    roll_windows = list((cfg.get("features", {}) or {}).get("rolling_windows", []) or [])
    time_feat_cfg  = (cfg.get("features", {}).get("time_features", {}) or {})
    use_time_feats = bool(time_feat_cfg.get("cyclical", True))

    # IO config (strict)
    out_cfg       = _req(cfg, ["features", "output"], "features.output")
    io_format     = _req(out_cfg, ["format"], "features.output.format")      # parquet|feather|csv
    out_features  = _req(out_cfg, ["features_file"], "features.output.features_file")
    out_manifest  = _req(out_cfg, ["manifest_file"], "features.output.manifest_file")

    # Split IO (strict)
    split_out_cfg = _req(cfg, ["output"], "output")
    split_fmt     = _req(split_out_cfg, ["format"], "output.format")
    split_names   = split_out_cfg.get("split_filenames", {}) or {}
    fn_train      = split_names.get("train", f"train.{split_fmt}")
    fn_val        = split_names.get("val",   f"val.{split_fmt}")
    fn_test       = split_names.get("test",  f"test.{split_fmt}")

    # Load splits or fallback to dataset_stream
    train = _read_df(_under(splits, fn_train), split_fmt)
    val   = _read_df(_under(splits, fn_val),   split_fmt)
    test  = _read_df(_under(splits, fn_test),  split_fmt)

    if train.empty and not ds_dir.exists():
        raise FileNotFoundError("No splits found and dataset_stream dir is missing. Run ingestion first.")

    base_df = (ds.dataset(str(ds_dir), format="parquet").to_table().to_pandas()
               if train.empty else pd.concat([train, val, test], ignore_index=True))

    # Optional station keep-list
    keep_txt_name = (cfg.get("reports", {}) or {}).get("stations_keep_txt")
    if keep_txt_name and id_col in base_df.columns:
        keep_txt = _under(art_dir, keep_txt_name)
        if keep_txt.exists():
            keep_ids = {
                _norm_sid(s)
                for s in keep_txt.read_text().splitlines()
                if s.strip() and not s.strip().startswith("#")
            }
            before = len(base_df)
            base_df = base_df[base_df[id_col].astype(str).map(_norm_sid).isin(keep_ids)].copy()
            print(f"[filter] keep-list applied: rows {before:,} -> {len(base_df):,}")

    # Time normalization + optional drop_years
    base_df[time_col] = pd.to_datetime(base_df[time_col], errors="coerce")
    drop_years = set((cfg.get("split", {}) or {}).get("drop_years", []) or [])
    if drop_years:
        before = len(base_df)
        base_df = base_df[~base_df[time_col].dt.year.isin(drop_years)].copy()
        print(f"[filter] drop_years={sorted(drop_years)}: rows {before:,} -> {len(base_df):,}")

    # Canonical order
    base_df = base_df.sort_values([id_col, time_col]).reset_index(drop=True)

    # Build features
    df = base_df

    # 1) target lags (if requested) – resolve actual target col by alias if needed
    if lags_hours:
        y_eff = y_col if y_col in df.columns else _pick(df.columns, pm25_aliases)
        if y_eff:
            df = _create_target_lags(df, id_col=id_col, y_col=y_eff, lags=lags_hours)
        else:
            print(f"[warn] target '{y_col}' not found and no alias matched; skipping lags")

    # 2) rolling stats for configured base numeric features (excluding the target to avoid leakage duplication)
    roll_cols = [c for c in data_feats if c in df.columns and c != y_col]
    df = _create_rolling(df, id_col=id_col, cols=roll_cols, windows=roll_windows)

    # 3) cyclical time features
    df = _add_time_features(df, time_col=time_col, enable=use_time_feats)

    # Drop configured high-missing features if any
    drop_cfg_raw = (cfg.get("missing") or {}).get("drop_features", None)
    if drop_cfg_raw:
        drop_list = [str(c) for c in (drop_cfg_raw if isinstance(drop_cfg_raw, (list, tuple, set)) else [drop_cfg_raw])]
        existing = [c for c in drop_list if c in df.columns]
        if existing:
            print(f"[features] dropping columns from config missing.drop_features: {existing}")
            df = df.drop(columns=existing)

    # Persist engineered features
    feats_path    = _under(feats_dir, out_features)
    manifest_path = _under(feats_dir, out_manifest)
    _write_df(df, feats_path, io_format)
    print(f"[OK] wrote features → {feats_path}")

    # Manifest
    exclude_cols = {id_col, time_col, y_col}
    X_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    manifest = {
        "rows": int(len(df)),
        "stations": int(df[id_col].nunique()) if id_col in df else 0,
        "time_min": str(df[time_col].min()) if len(df) else None,
        "time_max": str(df[time_col].max()) if len(df) else None,
        "time_col": time_col,
        "target_col": y_col,
        "id_cols": [id_col],
        "X_cols_ordered": X_cols,
        "lags_hours": lags_hours,
        "rolling_windows": roll_windows,
        "time_features": {"cyclical": bool(use_time_feats)},
        "dropped_from_config": [c for c in (drop_cfg_raw or [])] if isinstance(drop_cfg_raw, (list, tuple, set)) else ([drop_cfg_raw] if drop_cfg_raw else []),
        "source": {
            "splits_used": {
                "train": str(_under(splits, fn_train)),
                "val":   str(_under(splits, fn_val)),
                "test":  str(_under(splits, fn_test)),
                "format": split_fmt,
                "fallback_dataset_stream": (train.empty),
                "dataset_stream_dir": str(ds_dir),
            }
        },
        "output_format": io_format,
        "output_file": str(feats_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[OK] wrote manifest → {manifest_path}")

    # Quick ranges from splits (if present)
    print("\n=== SPLIT DATE RANGES (from features inputs) ===")
    for name, d in [("train", train), ("val", val), ("test", test)]:
        if d.empty:
            print(f"[{name}] missing or empty")
        else:
            d = d.copy()
            d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
            print(f"[{name}] rows={len(d):7d} start={d[time_col].min()} end={d[time_col].max()}")


if __name__ == "__main__":
    engineer()