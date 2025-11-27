#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
import numpy as np

from common.config_loader import load_cfg

def _req(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise KeyError("configs: missing " + ".".join(path))
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
    raise SystemExit(f"Unsupported format: {fmt!r}")

def _resolve_col(df: pd.DataFrame, primary: str, aliases: List[str], extra_fallbacks: Optional[List[str]] = None) -> str:
    by_lower = {c.lower(): c for c in df.columns}
    tried = []
    for cand in [primary] + list(aliases or []) + list(extra_fallbacks or []):
        if cand is None: continue
        k = str(cand).lower()
        tried.append(cand)
        if k in by_lower:
            return by_lower[k]
    sample = sorted(df.columns[:12].tolist())
    raise SystemExit(f"Required column '{primary}' not found. Tried aliases={tried}. Sample columns={sample}")

def _manifest_cols(art_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    feats_dir = _under(art_dir, _req(cfg, ["paths", "features_dir"]))
    out_cfg   = _req(cfg, ["features", "output"])
    man_file  = out_cfg.get("manifest_file")
    if man_file:
        q = feats_dir / man_file
        if q.exists():
            try:
                m = json.loads(q.read_text())
                if "time_col" in m and "id_cols" in m and m["id_cols"]:
                    return {"time_col": m["time_col"], "id_col": m["id_cols"][0], "target_col": m.get("target_col")}
            except Exception:
                pass
    locked_dir = _under(art_dir, cfg.get("paths", {}).get("features_locked_dir", "features_locked"))
    locked_name = cfg.get("features", {}).get("locked_manifest", "feature_list.json")
    q = locked_dir / locked_name
    if q.exists():
        try:
            m = json.loads(q.read_text())
            if m.get("time_col") and (m.get("id_cols") or m.get("id")):
                ids = m.get("id_cols") or m.get("id")
                return {"time_col": m["time_col"], "id_col": ids[0], "target_col": m.get("target_col")}
        except Exception:
            pass
    return {}

def _recompute_rolling_past(g: pd.Series, window: int) -> pd.Series:
    return g.shift(1).rolling(window=window, min_periods=1).mean()

def main() -> None:
    cfg = load_cfg()

    art_dir   = _under(Path("."), _req(cfg, ["paths", "artifacts_dir"]))
    feats_dir = _under(art_dir, _req(cfg, ["paths", "features_dir"]))
    out_cfg   = _req(cfg, ["features", "output"])
    fmt       = _req(out_cfg, ["format"])
    feats_file = _req(out_cfg, ["features_file"])
    feats_path = feats_dir / feats_file

    if not feats_path.exists():
        raise SystemExit(f"features file missing: {feats_path}")
    df = _read_df(feats_path, fmt)

    manifest = _manifest_cols(art_dir, cfg)
    target = manifest.get("target_col") or _req(cfg, ["data", "target"])

    if manifest.get("id_col") and manifest.get("time_col"):
        id_col = manifest["id_col"]; time_col = manifest["time_col"]
        if id_col not in df.columns or time_col not in df.columns:
            hdr = _req(cfg, ["data", "headers"])
            id_primary = _req(cfg, ["data", "id_col"])
            time_primary = _req(cfg, ["data", "time_col"])
            id_aliases = list(hdr.get("id", []))
            time_aliases = list(hdr.get("time", []))
            add_fallbacks = []
            lowers = {str(x).lower() for x in id_aliases + [id_primary]}
            if "id" not in lowers: add_fallbacks.append("id")
            if "ID".lower() not in lowers: add_fallbacks.append("ID")
            id_col   = _resolve_col(df, id_primary, id_aliases, add_fallbacks)
            time_col = _resolve_col(df, time_primary, time_aliases)
    else:
        hdr = _req(cfg, ["data", "headers"])
        id_primary = _req(cfg, ["data", "id_col"])
        time_primary = _req(cfg, ["data", "time_col"])
        id_aliases = list(hdr.get("id", []))
        time_aliases = list(hdr.get("time", []))
        add_fallbacks = []
        lowers = {str(x).lower() for x in id_aliases + [id_primary]}
        if "id" not in lowers: add_fallbacks.append("id")
        if "ID".lower() not in lowers: add_fallbacks.append("ID")
        id_col   = _resolve_col(df, id_primary, id_aliases, add_fallbacks)
        time_col = _resolve_col(df, time_primary, time_aliases)

    # canonical order
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df.sort_values([id_col, time_col], inplace=True)

    # Monotonic
    for sid, g in df.groupby(id_col, sort=False):
        t = g[time_col]
        if not t.is_monotonic_increasing:
            raise SystemExit(f"[ERR] timestamps not monotonic for station {sid}")
    print("[OK] timestamps are monotonic per station")

    # Lags check
    lag_hours = []
    if "lags" in cfg and isinstance(cfg["lags"], dict):
        if cfg["lags"].get("enabled") and cfg["lags"].get("hours"):
            lag_hours = list(cfg["lags"]["hours"])
    base = df.groupby(id_col, sort=False)[target]
    for h in lag_hours:
        col = f"{target}_lag{int(h)}h"
        if col in df.columns:
            exp = base.shift(h)
            mask = exp.notna() & df[col].notna()
            if mask.any():
                if not np.allclose(exp[mask].to_numpy(dtype=float), df.loc[mask, col].to_numpy(dtype=float)):
                    raise SystemExit(f"[ERR] {col} mismatch against shifted {target}")
            print(f"[OK] {col} matches past-only shift")

    # Rolling recompute spot-check (first numeric feature except target)
    roll_windows = list(cfg.get("features", {}).get("rolling_windows", []))
    num_cols = [c for c in df.columns if c not in {id_col, time_col, target} and pd.api.types.is_numeric_dtype(df[c])]
    if roll_windows and num_cols:
        c0 = num_cols[0]
        w  = int(roll_windows[0])
        exp = df.groupby(id_col, sort=False)[c0].pipe(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        # we only check mean if present
        col_mean = f"{c0}_roll{w}h_mean"
        if col_mean in df.columns:
            mask = df[col_mean].notna() & exp.notna()
            if mask.any():
                if not np.allclose(df.loc[mask, col_mean].to_numpy(dtype=float), exp[mask].to_numpy(dtype=float)):
                    raise SystemExit(f"[ERR] {col_mean} mismatch against recomputed past-only rolling mean")
            print("[OK] rolling stats recompute matches (spot-checked)")

    # Split boundary checks
    split_fmt = cfg.get("output", {}).get("format", "parquet")
    sfiles = cfg.get("output", {}).get("split_filenames", {}) or {}
    splits_dir = _under(art_dir, cfg.get("paths", {}).get("splits_dir", "splits"))
    train_p = splits_dir / sfiles.get("train", f"train.{split_fmt}")
    val_p   = splits_dir / sfiles.get("val",   f"val.{split_fmt}")
    test_p  = splits_dir / sfiles.get("test",  f"test.{split_fmt}")

    te = pd.to_datetime(_req(cfg, ["split", "train_end"]))
    ve = pd.to_datetime(_req(cfg, ["split", "val_end"]))

    t_train = pd.to_datetime(_read_df(train_p, split_fmt)[time_col], errors="coerce")
    t_val   = pd.to_datetime(_read_df(val_p,   split_fmt)[time_col], errors="coerce")
    t_test  = pd.to_datetime(_read_df(test_p,  split_fmt)[time_col], errors="coerce")

    if not (t_train.max() <= te): raise SystemExit("[ERR] train split exceeds train_end")
    print("[OK] train split ≤ train_end")
    if not ((t_val.min() > te) and (t_val.max() <= ve)): raise SystemExit("[ERR] val split bounds invalid")
    print("[OK] val split within (train_end, val_end]")
    if not (t_test.min() > ve): raise SystemExit("[ERR] test split not after val_end")
    print("[OK] test split > val_end")
    print("[done] leakage checks complete.")

if __name__ == "__main__":
    main()