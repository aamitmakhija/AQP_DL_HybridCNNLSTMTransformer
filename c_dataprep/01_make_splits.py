#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import os, json
from copy import deepcopy

import yaml
import pandas as pd

# --- local config loader (no external deps) ---
def _deep_update(dst: dict, src: dict | None) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_cfg() -> dict:
    base = yaml.safe_load(open("configs/default.yaml")) or {}
    cfg_env = os.environ.get("CONFIG", "")
    if not cfg_env:
        return base
    merged = deepcopy(base)
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        overlay = yaml.safe_load(open(p)) or {}
        _deep_update(merged, overlay)
    return merged
# ------------------------------------------------

def _req(d: dict, path: list[str], name: str | None = None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            dotted = ".".join(path)
            raise KeyError(f"configs: missing {dotted}{' ('+name+')' if name else ''}")
        cur = cur[k]
    return cur

def _under(root: Path, maybe: str) -> Path:
    p = Path(maybe)
    return p if p.is_absolute() else (root / p)

def _read_df(path: Path, fmt: str) -> pd.DataFrame:
    f = (fmt or "").lower()
    if f == "parquet": return pd.read_parquet(path)
    if f == "feather": return pd.read_feather(path)
    if f == "csv":     return pd.read_csv(path)
    raise SystemExit(f"Unsupported features.output.format: {fmt!r}")

def _write_df(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = (fmt or "").lower()
    if f == "parquet": df.to_parquet(path, index=False)
    elif f == "feather": df.reset_index(drop=True).to_feather(path)
    elif f == "csv": df.to_csv(path, index=False)
    else: raise SystemExit(f"Unsupported output.format: {fmt!r}")

def _norm_station_id(s: str) -> str:
    return str(s).strip().removesuffix(".0")

def main():
    cfg = load_cfg()

    art_rel   = _req(cfg, ["paths", "artifacts_dir"], "artifacts dir")
    feats_rel = _req(cfg, ["paths", "features_dir"], "features dir")
    splits_rel= _req(cfg, ["paths", "splits_dir"], "splits dir")

    feats_out_cfg = _req(cfg, ["features", "output"], "features.output")
    feats_file    = _req(feats_out_cfg, ["features_file"], "features.output.features_file")
    feats_fmt     = _req(feats_out_cfg, ["format"], "features.output.format")

    train_end = pd.to_datetime(_req(cfg, ["split", "train_end"], "split.train_end"))
    val_end   = pd.to_datetime(_req(cfg, ["split", "val_end"], "split.val_end"))
    if pd.isna(train_end) or pd.isna(val_end):
        raise SystemExit("split.train_end / split.val_end must be valid dates")
    if train_end > val_end:
        raise SystemExit(f"split boundaries invalid: train_end ({train_end}) > val_end ({val_end})")

    id_col   = _req(cfg, ["data", "id_col"], "data.id_col")
    time_col = _req(cfg, ["data", "time_col"], "data.time_col")

    out_cfg   = _req(cfg, ["output"], "output")
    out_fmt   = _req(out_cfg, ["format"], "output.format")
    names     = out_cfg.get("split_filenames", {}) or {}
    fn_train  = names.get("train",   f"train.{out_fmt}")
    fn_val    = names.get("val",     f"val.{out_fmt}")
    fn_test   = names.get("test",    f"test.{out_fmt}")
    fn_sum    = names.get("summary", "split_summary.json")

    drop_years = set((cfg.get("split", {}) or {}).get("drop_years", []) or [])
    keep_txt_name = (cfg.get("reports", {}) or {}).get("stations_keep_txt")

    cwd = Path(".")
    art_dir   = _under(cwd, art_rel)
    feats_dir = _under(art_dir, feats_rel) if not Path(feats_rel).is_absolute() else Path(feats_rel)
    splits_dir= _under(art_dir, splits_rel) if not Path(splits_rel).is_absolute() else Path(splits_rel)
    splits_dir.mkdir(parents=True, exist_ok=True)

    feats_path = feats_dir / feats_file
    if not feats_path.exists():
        raise SystemExit(f"Engineered features not found: {feats_path}")

    df = _read_df(feats_path, feats_fmt)

    missing = [c for c in (id_col, time_col) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {feats_path}: {missing}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    if keep_txt_name:
        keep_txt = art_dir / keep_txt_name
        if keep_txt.exists():
            keep_ids = {
                _norm_station_id(s)
                for s in keep_txt.read_text().splitlines()
                if s.strip() and not s.strip().startswith("#")
            }
            df["_sid_str"] = df[id_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            before = len(df)
            df = df[df["_sid_str"].isin(keep_ids)].drop(columns=["_sid_str"])
            print(f"[filter] keep-list applied: rows {before:,} -> {len(df):,}")
        else:
            print(f"[info] keep-list not found (skipping): {keep_txt}")

    if drop_years:
        before = len(df)
        df = df[~df[time_col].dt.year.isin(drop_years)].copy()
        print(f"[filter] drop_years={sorted(drop_years)}: rows {before:,} -> {len(df):,}")

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
            "val":   [str(val[time_col].min()),   str(val[time_col].max())]   if len(val)   else None,
            "test":  [str(test[time_col].min()),  str(test[time_col].max())]  if len(test)  else None,
        },
        "source_features": {"path": str(feats_path), "format": feats_fmt},
        "output_format": out_fmt,
        "filenames": {"train": fn_train, "val": fn_val, "test": fn_test, "summary": fn_sum},
    }
    (splits_dir / fn_sum).write_text(json.dumps(summary, indent=2))

    print(f"[config] CONFIG={os.environ.get('CONFIG','<env not set>')}")
    print(f"[input]  features: {feats_path} ({feats_fmt})")
    print(f"[split]  train_end={train_end}  val_end={val_end}  time_col={time_col}")
    print(f"[OK] wrote train rows={len(train):8d} → {splits_dir / fn_train}")
    print(f"[OK] wrote val   rows={len(val):8d} → {splits_dir / fn_val}")
    print(f"[OK] wrote test  rows={len(test):8d} → {splits_dir / fn_test}")
    print(f"[OK] wrote split summary → {splits_dir / fn_sum}")

if __name__ == "__main__":
    main()