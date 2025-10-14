from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, List
from copy import deepcopy
import yaml
import pandas as pd

def _deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> Dict:
    base = yaml.safe_load(open("configs/default.yaml"))
    cfg_env = os.environ.get("CONFIG")
    if not cfg_env:
        return base
    merged = deepcopy(base)
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        overlay = yaml.safe_load(open(p)) or {}
        _deep_update(merged, overlay)
    return merged

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _num_cols(df: pd.DataFrame, exclude: set[str]) -> List[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def main():
    cfg = _load_cfg()
    art = Path(cfg["paths"]["artifacts_dir"])
    time_col = cfg["data"]["time_col"]
    id_col   = cfg["data"].get("id_col","station_id")

    splits = art / "splits"
    out    = art / "features_scaled_ps"
    _ensure_dir(out)

    frames = {n: pd.read_parquet(splits/f"{n}.parquet") for n in ("train","val","test")}
    for n, df in frames.items():
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df[id_col]   = df[id_col].astype(str)

    # fit params on train per-station
    train = frames["train"]
    exclude = {id_col, time_col}
    nums = _num_cols(train, exclude=exclude)
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for sid, g in train.groupby(id_col):
        d = {}
        for c in nums:
            s = g[c]
            m = float(s.mean())
            sd = float(s.std(ddof=0)) or 1.0
            d[c] = {"mean": m, "std": sd}
        stats[str(sid)] = d

    # apply to each split
    scaled = {}
    for n, df in frames.items():
        out_df = df.copy()
        for sid, g in df.groupby(id_col):
            key = str(sid)
            if key not in stats:
                # unseen station → skip scaling
                out_df.loc[g.index, nums] = g[nums]
                continue
            p = stats[key]
            for c in nums:
                if c in g:
                    out_df.loc[g.index, c] = (g[c] - p[c]["mean"]) / (p[c]["std"] or 1.0)
        scaled[n] = out_df
        out_df.to_parquet(out/f"{n}.parquet", index=False)

    with open(out/"scaler_per_station.json","w") as f:
        json.dump({"mode":"standard_per_station","id_col":id_col,"time_col":time_col,"params":stats}, f, indent=2)

    print(f"[OK] wrote per-station scaled splits → {out}")
    for n, df in scaled.items():
        print(f"  {n}: rows={len(df):,} stations={df[id_col].nunique()}")

if __name__ == "__main__":
    main()