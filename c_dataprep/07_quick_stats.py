#!/usr/bin/env python3
# c_dataprep/07_quick_stats.py  (manifest + alias aware; hard-prefer primary IDs)
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from common.config_loader import load_cfg

def _req(d: Dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise KeyError("configs: missing " + ".".join(path))
        cur = cur[k]
    return cur

def _read(path: Path, fmt: str) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    f = fmt.lower()
    if f=="parquet": return pd.read_parquet(path)
    if f=="feather": return pd.read_feather(path)
    if f=="csv":     return pd.read_csv(path)
    raise SystemExit(f"Unsupported fmt: {fmt}")

def _manifest_cols(art: Path, cfg: dict) -> dict:
    feats_dir = art / cfg["paths"].get("features_dir","features")
    man_name  = (cfg.get("features",{}) or {}).get("output",{}).get("manifest_file")
    if man_name:
        p = feats_dir / man_name
        if p.exists():
            try: return json.loads(p.read_text())
            except Exception: pass
    return {}

def _resolve_id_time(df: pd.DataFrame, cfg: Dict[str, Any], manifest: dict) -> Tuple[Optional[str], Optional[str]]:
    if df.empty: return None, None
    man_id   = (manifest.get("id_cols") or [None])[0]
    man_time = manifest.get("time_col")

    id_primary   = str(_req(cfg, ["data", "id_col"]))
    time_primary = str(_req(cfg, ["data", "time_col"]))
    headers      = cfg.get("data", {}).get("headers", {}) or {}
    id_aliases   = list(headers.get("id", []))
    time_aliases = list(headers.get("time", []))

    by_lower = {c.lower(): c for c in df.columns}

    def pick(primary: str, aliases: list[str]) -> Optional[str]:
        if primary.lower() in by_lower: return by_lower[primary.lower()]
        for cand in aliases or []:
            k = str(cand).lower()
            if k in by_lower: return by_lower[k]
        for k in ("id","ID"):
            if k.lower() in by_lower and primary.lower() not in by_lower:
                return by_lower[k.lower()]
        return None

    id_col   = man_id   if man_id   in df.columns else pick(id_primary, id_aliases)
    time_col = man_time if man_time in df.columns else pick(time_primary, time_aliases)
    return id_col, time_col

def main():
    cfg   = load_cfg()
    art   = Path(_req(cfg, ["paths", "artifacts_dir"]))
    splits= art / _req(cfg, ["paths", "splits_dir"])
    featsd= art / _req(cfg, ["paths", "features_dir"])

    out_cfg   = _req(cfg, ["output"])
    split_fmt = _req(out_cfg, ["format"]).lower()
    names     = out_cfg.get("split_filenames", {}) or {}
    p_train   = splits / names.get("train", f"train.{split_fmt}")
    p_val     = splits / names.get("val",   f"val.{split_fmt}")
    p_test    = splits / names.get("test",  f"test.{split_fmt}")

    f_out     = _req(cfg, ["features", "output"])
    feats_fmt = _req(f_out, ["format"]).lower()
    feats_file= _req(f_out, ["features_file"])

    train, val, test = (_read(p_train, split_fmt), _read(p_val, split_fmt), _read(p_test, split_fmt))
    F = _read(featsd/feats_file, feats_fmt)

    manifest = _manifest_cols(art, cfg)

    def brief(df: pd.DataFrame):
        if df.empty: return {"rows":0,"stations":0,"tmin":None,"tmax":None}
        id_col, time_col = _resolve_id_time(df, cfg, manifest)
        d = df.copy()
        if time_col in d.columns:
            d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
        return {
            "rows": int(len(d)),
            "stations": (int(d[id_col].nunique()) if id_col in d.columns else None),
            "tmin": (str(d[time_col].min()) if time_col in d.columns else None),
            "tmax": (str(d[time_col].max()) if time_col in d.columns else None),
        }

    target = (manifest.get("target_col") or (cfg.get("data",{}) or {}).get("target"))
    summary = {
        "splits": {"train": brief(train), "val": brief(val), "test": brief(test)},
        "features": {
            "rows": int(len(F)),
            "cols": list(F.columns),
            "target_in_features": (bool(target) and target in F.columns),
        }
    }

    out_json = art / ((cfg.get("reports",{}) or {}).get("quick_stats_json","quick_stats.json"))
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {out_json}")

if __name__ == "__main__":
    main()