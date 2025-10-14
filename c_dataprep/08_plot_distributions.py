# 2_dataprep/08_plot_distributions.py
from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import Dict, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# ---------- tiny config loader (respects CONFIG overlays) ----------
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

# ---------- utils ----------
def _read_scaled(art_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fs = art_dir / "features_scaled"
    tr = pd.read_parquet(fs / "train.parquet")
    va = pd.read_parquet(fs / "val.parquet")
    te = pd.read_parquet(fs / "test.parquet")
    return tr, va, te

def _load_scaler(art_dir: Path) -> Dict:
    p = art_dir / "features_scaled" / "scaler.json"
    return json.loads(p.read_text()) if p.exists() else {}

def _infer_target(art_dir: Path, fallback: str) -> str:
    locked = art_dir / "features_locked" / "feature_list.json"
    if locked.exists():
        try:
            return json.loads(locked.read_text()).get("target_col", fallback) or fallback
        except Exception:
            return fallback
    return fallback

def _inverse_target_if_possible(df: pd.DataFrame, target: str, scaler: Dict) -> Tuple[pd.DataFrame, str]:
    if not scaler or scaler.get("mode", "none") == "none":
        return df, f"{target} (scaled)"
    params = scaler.get("params", {})
    if target not in params:
        return df, f"{target} (scaled)"
    mode = scaler.get("mode")
    df = df.copy()
    p = params[target]
    if mode == "standard":
        df[target] = df[target] * (p.get("std", 1.0) or 1.0) + p.get("mean", 0.0)
        label = f"{target} (original units)"
    elif mode == "minmax":
        denom = (p.get("max", 1.0) - p.get("min", 0.0)) or 1.0
        df[target] = df[target] * denom + p.get("min", 0.0)
        label = f"{target} (original units)"
    else:
        label = f"{target} (scaled)"
    return df, label

def _kde(x: np.ndarray, grid: np.ndarray, bw: float | None = None) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid)
    if bw is None:
        std = np.std(x, ddof=1)
        bw = 1.06 * std * (x.size ** (-1/5)) if std > 0 else 1.0
    diffs = (grid.reshape(-1, 1) - x.reshape(1, -1)) / bw
    dens = np.exp(-0.5 * diffs**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return dens

def _nice_bins(x: np.ndarray, bins: int | None) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.linspace(-1, 1, 11)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = x.min(), x.max() + 1e-6
    return np.linspace(lo, hi, (bins or 60))

# ---------- plotting ----------
def plot_overlay_hist_kde(train: pd.Series, val: pd.Series, test: pd.Series,
                          outpath: Path, title: str, bins: int | None):
    train = train.to_numpy(); val = val.to_numpy(); test = test.to_numpy()
    grid = _nice_bins(np.concatenate([train, val, test]), bins)
    plt.figure(figsize=(9, 5.5))
    for arr, lab in [(train, "train"), (val, "val"), (test, "test")]:
        hist_vals, edges = np.histogram(arr[np.isfinite(arr)], bins=grid, density=True)
        mids = 0.5 * (edges[1:] + edges[:-1])
        plt.step(mids, hist_vals, where="mid", alpha=0.6, label=f"{lab} hist")
        kde_vals = _kde(arr, mids)
        plt.plot(mids, kde_vals, alpha=0.9, label=f"{lab} kde")
    plt.title(title); plt.xlabel("value"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150); plt.close()

def plot_split_hist_kde(series: pd.Series, outpath: Path, title: str, bins: int | None):
    arr = series.to_numpy()
    grid = _nice_bins(arr, bins)
    hist_vals, edges = np.histogram(arr[np.isfinite(arr)], bins=grid, density=True)
    mids = 0.5 * (edges[1:] + edges[:-1]); kde_vals = _kde(arr, mids)
    plt.figure(figsize=(8, 5))
    plt.bar(mids, hist_vals, width=(mids[1]-mids[0]) if mids.size > 1 else 0.1, alpha=0.4, align="center")
    plt.plot(mids, kde_vals, linewidth=2)
    plt.title(title); plt.xlabel("value"); plt.ylabel("density"); plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150); plt.close()

# ---------- main ----------
def main():
    # parse with the same flags you had; defaults will be replaced by config if user leaves them at default
    ap = argparse.ArgumentParser(description="Plot target histograms/KDE for train/val/test.")
    ap.add_argument("--artifacts", default="experiments/artifacts", help="Artifacts root dir")
    ap.add_argument("--target", default="PM25_Concentration", help="Target column name")
    ap.add_argument("--bins", type=int, default=60, help="Number of histogram bins (approx)")
    ap.add_argument("--inverse", action="store_true",
                    help="Inverse target scaling using features_scaled/scaler.json when possible")
    ap.add_argument("--outdir", default="experiments/artifacts/plots", help="Where to save plots")
    args = ap.parse_args()

    cfg = _load_cfg()

    # replace hard-coded defaults with config-driven values when user didn’t override
    if args.artifacts == "experiments/artifacts":
        args.artifacts = cfg["paths"]["artifacts_dir"]
    # default target from locked schema if present (only when user didn’t override)
    if args.target == "PM25_Concentration":
        args.target = _infer_target(Path(args.artifacts), args.target)
    # default outdir under artifacts when user leaves default
    if args.outdir == "experiments/artifacts/plots":
        args.outdir = str(Path(args.artifacts) / "plots")

    art_dir = Path(args.artifacts)
    out_dir = Path(args.outdir)

    train, val, test = _read_scaled(art_dir)
    scaler = _load_scaler(art_dir) if args.inverse else {}

    # Ensure target exists
    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        if args.target not in df.columns:
            raise KeyError(f"Target '{args.target}' not found in features_scaled/{split_name}.parquet")

    # Optionally inverse-transform target back to original units
    label_suffix = f"{args.target} (scaled)"
    if args.inverse:
        train, label_suffix = _inverse_target_if_possible(train, args.target, scaler)
        val,   label_suffix = _inverse_target_if_possible(val, args.target, scaler)
        test,  label_suffix = _inverse_target_if_possible(test, args.target, scaler)

    # 1) Overlay comparison
    plot_overlay_hist_kde(
        train[args.target], val[args.target], test[args.target],
        outpath=out_dir / "target_overlay_hist_kde.png",
        title=f"Target distribution — {label_suffix}",
        bins=args.bins,
    )
    # 2) Per-split quick views
    plot_split_hist_kde(train[args.target], out_dir / "target_train_hist_kde.png",
                        title=f"Train — {label_suffix}", bins=args.bins)
    plot_split_hist_kde(val[args.target],   out_dir / "target_val_hist_kde.png",
                        title=f"Val — {label_suffix}",   bins=args.bins)
    plot_split_hist_kde(test[args.target],  out_dir / "target_test_hist_kde.png",
                        title=f"Test — {label_suffix}",  bins=args.bins)

    print(f"[OK] wrote plots → {out_dir}")

if __name__ == "__main__":
    main()