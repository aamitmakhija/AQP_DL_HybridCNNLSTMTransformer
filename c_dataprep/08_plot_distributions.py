# 2_dataprep/08_plot_distributions.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.config_loader import load_cfg  # overlay-aware unified loader


# ---------------------------- IO helpers ----------------------------

def _read_df(path: Path, fmt: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "feather":
        return pd.read_feather(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported split format: {fmt}")


def _split_paths(cfg: Dict, scaled_dir: Path) -> Tuple[Path, Path, Path, str]:
    """
    Resolve train/val/test paths from config:
      - filenames via output.split_filenames
      - format via scaling.output_format (fallback to output.format)
    """
    split_names: Dict[str, str] = cfg.get("output", {}).get("split_filenames", {}) or {}
    split_fmt: str = cfg.get("scaling", {}).get("output_format",
                        cfg.get("output", {}).get("format", "parquet"))

    def _p(name: str) -> Path:
        default = f"{name}.{split_fmt}"
        return scaled_dir / split_names.get(name, default)

    return _p("train"), _p("val"), _p("test"), split_fmt


def _load_lock_target(cfg: Dict, features_locked_dir: Path) -> str | None:
    """
    Prefer target from config (data.target). Else use features.locked_manifest if present.
    """
    if cfg.get("data", {}).get("target"):
        return cfg["data"]["target"]

    locked_manifest = cfg.get("features", {}).get("locked_manifest", "feature_list.json")
    lock_path = features_locked_dir / locked_manifest
    if lock_path.exists():
        try:
            meta = json.loads(lock_path.read_text())
            return meta.get("target_col")
        except Exception:
            return None
    return None


def _load_scaler(cfg: Dict, scaled_dir: Path) -> Dict:
    """
    Load scaler meta if available and standardized.
    Expect one of:
      - scaling.meta_file (e.g., 'scaler_params.json')
      - fallback candidates: 'scaler_params.json', 'scaler.json'
    The function returns {} if unavailable or unreadable.
    """
    meta_file = cfg.get("scaling", {}).get("meta_file")
    candidates = [meta_file] if meta_file else ["scaler_params.json", "scaler.json"]
    for name in candidates:
        p = scaled_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return {}
    return {}


# -------------------------- math / plotting --------------------------

def _kde(x: np.ndarray, grid: np.ndarray, bw: float | None = None) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid)
    if bw is None:
        std = np.std(x, ddof=1)
        bw = 1.06 * std * (x.size ** (-1 / 5)) if std > 0 else 1.0
    diffs = (grid.reshape(-1, 1) - x.reshape(1, -1)) / bw
    dens = np.exp(-0.5 * diffs ** 2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return dens


def _nice_bins(x: np.ndarray, bins: int | None) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.linspace(-1, 1, 11)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = x.min(), x.max() + 1e-6
    return np.linspace(lo, hi, (bins or 60))


def plot_overlay_hist_kde(train: pd.Series, val: pd.Series, test: pd.Series,
                          outpath: Path, title: str, bins: int | None):
    train = train.to_numpy()
    val = val.to_numpy()
    test = test.to_numpy()
    grid = _nice_bins(np.concatenate([train, val, test]), bins)
    plt.figure(figsize=(9, 5.5))
    for arr, lab in [(train, "train"), (val, "val"), (test, "test")]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        hist_vals, edges = np.histogram(arr, bins=grid, density=True)
        mids = 0.5 * (edges[1:] + edges[:-1])
        plt.step(mids, hist_vals, where="mid", alpha=0.6, label=f"{lab} hist")
        kde_vals = _kde(arr, mids)
        plt.plot(mids, kde_vals, alpha=0.9, label=f"{lab} kde")
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_split_hist_kde(series: pd.Series, outpath: Path, title: str, bins: int | None):
    arr = series.to_numpy()
    arr = arr[np.isfinite(arr)]
    grid = _nice_bins(arr, bins)
    hist_vals, edges = np.histogram(arr, bins=grid, density=True)
    mids = 0.5 * (edges[1:] + edges[:-1])
    kde_vals = _kde(arr, mids)
    plt.figure(figsize=(8, 5))
    if mids.size > 1:
        plt.bar(mids, hist_vals, width=(mids[1] - mids[0]), alpha=0.4, align="center")
    else:
        plt.bar(mids, hist_vals, alpha=0.4, align="center")
    plt.plot(mids, kde_vals, linewidth=2)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


# -------------------------- inverse scaling --------------------------

def _can_inverse_target_globally(scaler_meta: Dict, target_col: str) -> Tuple[bool, Dict]:
    """
    We only inverse-transform if the scaler is clearly GLOBAL (not per-station),
    and contains parameters for the target column directly.
      Expected shapes supported:
        - {"mode":"standard_global","params":{"<col>":{"mean":..,"std":..}}}
        - {"mode":"minmax_global","params":{"<col>":{"min":..,"max":..}}}
    Returns (ok, params_for_target).
    """
    if not scaler_meta or "params" not in scaler_meta or not isinstance(scaler_meta["params"], dict):
        return False, {}
    params = scaler_meta["params"].get(target_col)
    if not isinstance(params, dict):
        return False, {}
    mode = scaler_meta.get("mode", "")
    if mode in ("standard_global", "minmax_global", "robust_global"):
        return True, params
    # Heuristic: if there's no id_col and params are per-column, treat it as global
    if "id_col" not in scaler_meta and all(isinstance(v, dict) for v in scaler_meta["params"].values()):
        return True, params
    return False, {}


def _inverse_target_series(s: pd.Series, scaler_meta: Dict, target_col: str) -> Tuple[pd.Series, str]:
    ok, p = _can_inverse_target_globally(scaler_meta, target_col)
    if not ok:
        return s, f"{target_col} (scaled)"
    mode = scaler_meta.get("mode", "")
    if mode.startswith("standard"):
        std = float(p.get("std", 1.0)) or 1.0
        mean = float(p.get("mean", 0.0))
        return (s * std + mean), f"{target_col} (original units)"
    if mode.startswith("minmax"):
        denom = (float(p.get("max", 1.0)) - float(p.get("min", 0.0))) or 1.0
        return (s * denom + float(p.get("min", 0.0))), f"{target_col} (original units)"
    # add more modes if you support them
    return s, f"{target_col} (scaled)"


# ------------------------------ main ------------------------------

def main():
    cfg = load_cfg()

    # CLI: allow overrides, but default to config-derived values
    ap = argparse.ArgumentParser(description="Plot target hist/KDE for train/val/test (config-driven).")
    ap.add_argument("--target", default=None, help="Target column; defaults to data.target or locked manifest.")
    ap.add_argument("--bins", type=int, default=None, help="Histogram bins; default 60.")
    ap.add_argument("--inverse", action="store_true",
                    help="Inverse-transform target using scaler meta if global scaler is available.")
    ap.add_argument("--outdir", default=None, help="Override output plots dir (defaults to reports.plots_dir).")
    args = ap.parse_args()

    # Paths from config
    art_dir = Path(cfg["paths"]["artifacts_dir"])
    scaled_dir = art_dir / cfg["paths"].get("features_scaled_dir", "features_scaled_ps")
    features_locked_dir = art_dir / cfg["paths"].get("features_locked_dir", "features_locked")
    plots_dir = art_dir / cfg.get("reports", {}).get("plots_dir", "plots") if args.outdir is None else Path(args.outdir)

    # Resolve split files & format
    train_path, val_path, test_path, split_fmt = _split_paths(cfg, scaled_dir)

    # Resolve target
    target = args.target or _load_lock_target(cfg, features_locked_dir) or cfg.get("data", {}).get("target")
    if not target:
        raise SystemExit("Target column not provided and could not be inferred from config or lock file. "
                         "Pass --target or set data.target in your YAML.")

    # Load data
    train = _read_df(train_path, split_fmt)
    val = _read_df(val_path, split_fmt)
    test = _read_df(test_path, split_fmt)

    # Ensure target exists
    for split_name, df in (("train", train), ("val", val), ("test", test)):
        if target not in df.columns:
            raise KeyError(f"Target '{target}' not found in {split_name} split: {scaled_dir}")

    # Optional inverse transform (only if scaler is clearly global)
    label_suffix = f"{target} (scaled)"
    if args.inverse:
        scaler_meta = _load_scaler(cfg, scaled_dir)
        s_train, label_suffix = _inverse_target_series(train[target], scaler_meta, target)
        s_val,   _            = _inverse_target_series(val[target],   scaler_meta, target)
        s_test,  _            = _inverse_target_series(test[target],  scaler_meta, target)
    else:
        s_train, s_val, s_test = train[target], val[target], test[target]

    bins = args.bins or 60

    # 1) Overlay plot
    plot_overlay_hist_kde(
        s_train, s_val, s_test,
        outpath=plots_dir / "target_overlay_hist_kde.png",
        title=f"Target distribution — {label_suffix}",
        bins=bins,
    )

    # 2) Per-split plots
    plot_split_hist_kde(s_train, plots_dir / "target_train_hist_kde.png",
                        title=f"Train — {label_suffix}", bins=bins)
    plot_split_hist_kde(s_val,   plots_dir / "target_val_hist_kde.png",
                        title=f"Val — {label_suffix}",   bins=bins)
    plot_split_hist_kde(s_test,  plots_dir / "target_test_hist_kde.png",
                        title=f"Test — {label_suffix}",  bins=bins)

    print(f"[OK] wrote plots → {plots_dir}")


if __name__ == "__main__":
    main()