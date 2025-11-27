#!/usr/bin/env python3
# c_dataprep/01c_dedupe_splits.py
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List

import pandas as pd

# Ensure repo root on sys.path (so "common" is importable when run as a script)
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.config_loader import load_cfg  # noqa: E402


# ---------------- core utils ----------------

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

def _read(path: Path, fmt: str) -> pd.DataFrame:
    f = fmt.lower()
    if f == "parquet": return pd.read_parquet(path)
    if f == "feather": return pd.read_feather(path)
    if f == "csv":     return pd.read_csv(path)
    raise SystemExit(f"Unsupported format: {fmt!r}")

def _write(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = fmt.lower()
    if f == "parquet":   df.to_parquet(path, index=False)
    elif f == "feather": df.reset_index(drop=True).to_feather(path)
    elif f == "csv":     df.to_csv(path, index=False)
    else:                raise SystemExit(f"Unsupported format: {fmt!r}")

def _norm_id_preserve_na(s: pd.Series) -> pd.Series:
    out = s.copy()
    m = out.notna()
    out.loc[m] = out.loc[m].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return out

def _dedupe_one(p: Path, fmt: str, id_col: str, time_col: str) -> Dict[str, int]:
    df = _read(p, fmt)
    if df.empty or id_col not in df.columns or time_col not in df.columns:
        return {"rows": int(len(df)), "dropped": 0}
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[id_col]   = _norm_id_preserve_na(df[id_col])
    before = len(df)
    # stable deterministic order before dropping duplicates
    df.sort_values([id_col, time_col], inplace=True, kind="mergesort")
    df.drop_duplicates(subset=[id_col, time_col], keep="first", inplace=True)
    _write(df, p, fmt)
    return {"rows": int(len(df)), "dropped": int(before - len(df))}


# ---------------- main ----------------

def main() -> None:
    cfg = load_cfg()

    art_dir   = _under(Path("."), _req(cfg, ["paths", "artifacts_dir"]))
    splits_dir = _under(art_dir, _req(cfg, ["paths", "splits_dir"]))

    fmt      = str(_req(cfg, ["output", "format"]))  # parquet|feather|csv
    id_col   = str(_req(cfg, ["data", "id_col"]))
    time_col = str(_req(cfg, ["data", "time_col"]))

    # Use configured split filenames if present; otherwise default to {key}.{fmt}
    names_cfg = (cfg.get("output", {}) or {}).get("split_filenames", {}) or {}
    def _fname(key: str) -> str:
        return str(names_cfg.get(key, f"{key}.{fmt}"))

    paths = {k: splits_dir / _fname(k) for k in ("train", "val", "test")}

    # Sanity: if a file is missing, surface a helpful error listing what exists
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        existing = sorted([p.name for p in splits_dir.glob("*")])
        raise FileNotFoundError(
            "Split file(s) missing: "
            + ", ".join(f"{k} -> {paths[k].name}" for k in missing)
            + f".\nLooked in: {splits_dir}\nExisting: {existing}"
        )

    stats = {k: _dedupe_one(p, fmt, id_col, time_col) for k, p in paths.items()}
    print(json.dumps({"deduped_splits": stats}, indent=2))

if __name__ == "__main__":
    main()