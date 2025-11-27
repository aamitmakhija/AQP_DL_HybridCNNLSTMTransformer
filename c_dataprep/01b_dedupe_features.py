#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import os, json
from copy import deepcopy
from typing import Any, Dict, List

import yaml
import pandas as pd

# --- local config loader (no external deps) ---
def _deep_update(dst: Dict, src: Dict | None) -> Dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_cfg() -> Dict:
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
    else: raise SystemExit(f"Unsupported features.output.format: {fmt!r}")

def _first_present(cols: List[str], cands: List[str]) -> str | None:
    for c in cands:
        if c in cols: return c
    return None

def _aliases(cfg: Dict[str, Any], key: str, fallback: List[str]) -> List[str]:
    hdrs = (cfg.get("data", {}).get("headers", {}) or {})
    al = hdrs.get(key)
    return [str(x) for x in al] if isinstance(al, list) and al else fallback

def _norm_id_preserve_na(s: pd.Series) -> pd.Series:
    if s.empty: return s
    out = s.copy()
    m = out.notna()
    out.loc[m] = (
        out.loc[m].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    return out

def main() -> None:
    cfg = load_cfg()
    art_dir   = _under(Path("."), _req(cfg, ["paths", "artifacts_dir"]))
    feats_dir = _under(art_dir,   _req(cfg, ["paths", "features_dir"]))
    out_cfg   = _req(cfg, ["features", "output"])
    fmt       = _req(out_cfg, ["format"])
    fname     = _req(out_cfg, ["features_file"])
    fpath     = feats_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"features not found: {fpath}")

    df = _read_df(fpath, fmt)
    input_rows = int(len(df))

    id_can   = str(_req(cfg, ["data", "id_col"]))
    time_can = str(_req(cfg, ["data", "time_col"]))

    id_alias   = _aliases(cfg, "id",   [id_can, "station_id", "Station ID", "STATION_ID", "station", "id", "ID"])
    time_alias = _aliases(cfg, "time", [time_can, "Datetime", "Time", "timestamp", "time", "TIME"])

    id_res   = _first_present(df.columns.tolist(), id_alias)
    time_res = _first_present(df.columns.tolist(), time_alias)
    if id_res is None:  raise KeyError(f"Could not resolve id column. Tried {id_alias}.")
    if time_res is None:raise KeyError(f"Could not resolve time column. Tried {time_alias}.")

    df[time_res] = pd.to_datetime(df[time_res], errors="coerce")
    df[id_res]   = _norm_id_preserve_na(df[id_res])

    if id_res != id_can:
        if id_can in df.columns: df.drop(columns=[id_can], inplace=True)
        df.rename(columns={id_res: id_can}, inplace=True)
    if time_res != time_can:
        if time_can in df.columns: df.drop(columns=[time_can], inplace=True)
        df.rename(columns={time_res: time_can}, inplace=True)

    rows_after_key_clean = int(len(df))

    df.sort_values([id_can, time_can], inplace=True, kind="mergesort")
    before = len(df)
    df.drop_duplicates(subset=[id_can, time_can], keep="first", inplace=True)
    after = len(df)
    dropped = int(before - after)

    _write_df(df, fpath, fmt)
    print(json.dumps({
        "input_rows": input_rows,
        "rows_after_key_clean": rows_after_key_clean,
        "rows_after_dedup": after,
        "dropped_total": dropped,
        "id_col_resolved": id_can,
        "time_col_resolved": time_can
    }, indent=2))
    print(f"[dedupe] overwrote → {fpath}")

if __name__ == "__main__":
    main()