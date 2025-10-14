# src/csv2parquet_safe.py
from __future__ import annotations
import os
from pathlib import Path
from copy import deepcopy
import yaml
import pandas as pd

# ---------- config helpers (minimal, preserves your structure) ----------
def _deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> dict:
    """
    Load configs/default.yaml, then merge any overlays from CONFIG if present.
    Accepts CONFIG as a single path or comma-separated list of YAML files.
    """
    base = yaml.safe_load(open("configs/default.yaml"))
    cfg_env = os.environ.get("CONFIG")
    if not cfg_env:
        return base
    merged = deepcopy(base)
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        with open(p, "r") as f:
            overlay = yaml.safe_load(f) or {}
        _deep_update(merged, overlay)
    return merged

cfg = _load_cfg()

ART_DIR = Path(cfg["paths"]["artifacts_dir"])
DATA_DIR = Path(cfg["paths"]["data_dir"])
FILES = cfg["data_files"]

ART_DIR.mkdir(parents=True, exist_ok=True)

def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        # fast path
        return pd.read_csv(
            path,
            encoding="utf-8",
            engine="c",
            na_values=["", "NA", "NaN", "null", "NULL"],
            keep_default_na=True,
            low_memory=False,
        )
    except Exception:
        # fallback path
        return pd.read_csv(
            path,
            encoding="utf-8",
            engine="python",
            na_values=["", "NA", "NaN", "null", "NULL"],
            keep_default_na=True,
        )

def main():
    print(f"[csv2parquet] CONFIG={os.environ.get('CONFIG', 'configs/default.yaml')}")
    print(f"[csv2parquet] DATA_DIR={DATA_DIR}  →  ART_DIR={ART_DIR}")

    written = []
    skipped = []
    for key, fname in FILES.items():
        src = DATA_DIR / fname
        if not src.exists():
            print(f"[WARN] {src} missing; skipping")
            skipped.append(key)
            continue
        df = _read_csv_safe(src)
        out = ART_DIR / f"{key}.parquet"
        df.to_parquet(out, index=False)
        written.append(f"{key}.parquet")

    if written:
        print(f"[OK] wrote {', '.join(written)} in {ART_DIR}")
    if skipped:
        print(f"[INFO] skipped (not found): {', '.join(skipped)}")

if __name__ == "__main__":
    main()