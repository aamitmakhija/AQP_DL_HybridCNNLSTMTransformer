# 2_dataprep/07_quick_stats.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import pandas as pd

from common.config_loader import load_cfg  # unified, overlay-aware

def _read_df(path: Path, fmt: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "feather":
        return pd.read_feather(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported format: {fmt}")

def _summ(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in cols:
        if c in df.columns:
            s = df[c].dropna()
            out[c] = {
                "rows": int(s.shape[0]),
                "mean": float(s.mean()),
                "std": float(s.std()),      # ddof=1 (pandas default)
                "min": float(s.min()) if not s.empty else float("nan"),
                "max": float(s.max()) if not s.empty else float("nan"),
            }
    return out

def main():
    cfg = load_cfg()

    # ---- paths / filenames from config ----
    art_dir       = Path(cfg["paths"]["artifacts_dir"])
    scaled_dir    = art_dir / cfg["paths"].get("features_scaled_dir", "features_scaled_ps")
    features_dir  = art_dir / cfg["paths"].get("features_dir", "features")
    locked_dir    = art_dir / cfg["paths"].get("features_locked_dir", "features_locked")

    # lockfile naming
    locked_file   = cfg.get("features", {}).get("locked_manifest", "feature_list.json")
    lock_path     = locked_dir / locked_file

    # split I/O
    split_fmt     = cfg.get("scaling", {}).get("output_format",
                     cfg.get("output", {}).get("format", "parquet"))
    split_names   = cfg.get("output", {}).get("split_filenames", {}) or {}
    def split_path(name: str) -> Path:
        default = f"{name}.{split_fmt}"
        return scaled_dir / split_names.get(name, default)

    # report config
    reports_cfg   = cfg.get("reports", {})
    out_name      = reports_cfg.get("quick_stats", "quick_stats.json")
    out_path      = art_dir / out_name
    topk          = int(reports_cfg.get("quick_stats_topk", 5))

    # ---- read locked schema ----
    if not lock_path.exists():
        raise FileNotFoundError(f"{lock_path} missing — run feature locking / scaling first.")
    locked = json.loads(lock_path.read_text())
    target = locked.get("target_col")
    x_cols = locked.get("X_cols_ordered", [])
    if not isinstance(x_cols, list):
        x_cols = []

    # choose columns to summarize
    cols_to_report = [c for c in [target] if c] + x_cols[:topk]

    # ---- read splits ----
    train = _read_df(split_path("train"), split_fmt)
    val   = _read_df(split_path("val"),   split_fmt)
    test  = _read_df(split_path("test"),  split_fmt)

    stats = {
        "train": _summ(train, cols_to_report),
        "val":   _summ(val,   cols_to_report),
        "test":  _summ(test,  cols_to_report),
        "meta": {
            "columns": cols_to_report,
            "split_format": split_fmt,
            "topk": topk,
            "lock_file": str(lock_path),
        }
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"[OK] wrote {out_path}")

    # console
    for split in ("train", "val", "test"):
        print(f"\n=== {split.upper()} ===")
        for c, m in stats[split].items():
            print(f"{c:25s} rows={m['rows']:7d} mean={m['mean']:.3f} std={m['std']:.3f} "
                  f"min={m['min']:.3f} max={m['max']:.3f}")

if __name__ == "__main__":
    main()