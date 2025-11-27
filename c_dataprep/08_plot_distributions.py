#!/usr/bin/env python3
# c_dataprep/08_plot_distributions.py  (tolerant, keeps CLI)
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.config_loader import load_cfg  # noqa: E402

def _req(d: Dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise KeyError("configs: missing " + ".".join(path))
        cur = cur[k]
    return cur

def _under(root: Path, maybe: str) -> Path:
    p = Path(maybe); return p if p.is_absolute() else (root / p)

def _read_df(path: Path, fmt: str) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"Missing split file: {path}")
    f = fmt.lower()
    if f=="parquet": return pd.read_parquet(path)
    if f=="feather": return pd.read_feather(path)
    if f=="csv":     return pd.read_csv(path)
    raise SystemExit(f"Unsupported split format: {fmt!r}")

def _split_paths(cfg: Dict[str, Any], scaled_dir: Path) -> Tuple[Path, Path, Path, str]:
    # tolerant names for scaled splits
    names = (cfg.get("output",{}) or {}).get("split_filenames") or {}
    split_fmt = str((cfg.get("scaling",{}) or {}).get("output_format","parquet")).lower()
    def _p(name: str) -> Path: return scaled_dir / str(names.get(name, f"{name}.{split_fmt}"))
    return _p("train"), _p("val"), _p("test"), split_fmt

def _load_lock_target(cfg: Dict[str, Any], features_locked_dir: Path) -> str | None:
    t = (cfg.get("data",{}) or {}).get("target")
    if t: return str(t)
    lock_name = (cfg.get("features",{}) or {}).get("locked_manifest") or "feature_list.json"
    p = features_locked_dir / lock_name
    if p.exists():
        try: return json.loads(p.read_text()).get("target_col")
        except Exception: return None
    return None

def _load_scaler_meta(cfg: Dict[str, Any], scaled_dir: Path) -> Dict:
    meta_file = (cfg.get("scaling",{}) or {}).get("meta_file") or "scaler_params.json"
    p = scaled_dir / meta_file
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: return {}
    return {}

def _kde(x: np.ndarray, grid: np.ndarray, bw: float | None = None) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0: return np.zeros_like(grid)
    if bw is None:
        std = np.std(x, ddof=1); bw = 1.06 * std * (x.size ** (-1/5)) if std>0 else 1.0
    diffs = (grid.reshape(-1,1) - x.reshape(1,-1)) / bw
    return np.exp(-0.5*diffs**2).mean(axis=1) / (bw*np.sqrt(2*np.pi))

def _nice_bins(x: np.ndarray, bins: int | None) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0: return np.linspace(-1,1,11)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo==hi:
        lo, hi = x.min(), x.max()+1e-6
    return np.linspace(lo, hi, (bins or 60))

def _extract_global(s_meta: Dict) -> tuple[str|None, Dict|None]:
    if isinstance(s_meta, dict) and s_meta.get("per_station") is False:
        st = str(s_meta.get("type","")).lower()
        params = s_meta.get("global_params") or s_meta.get("params")
        if st in {"standard","minmax","robust"} and isinstance(params, dict):
            return st, params
    mode = str(s_meta.get("mode","")).lower()
    if mode.endswith("_global"):
        st = ("standard" if "standard" in mode else
              "minmax"   if "minmax"   in mode else
              "robust"   if "robust"   in mode else None)
        if st and isinstance(s_meta.get("params"), dict): return st, s_meta["params"]
    return None, None

def _inverse_target(series: pd.Series, meta: Dict, target: str) -> tuple[pd.Series, str]:
    stype, params = _extract_global(meta)
    if not stype or target not in (params or {}): return series, f"{target} (scaled)"
    p = params[target] or {}
    if stype=="standard":
        std = float(p.get("std",1.0)) or 1.0; mean=float(p.get("mean",0.0))
        return series*std + mean, f"{target} (original units)"
    if stype=="minmax":
        lo=float(p.get("min",0.0)); hi=float(p.get("max",1.0))
        return series*((hi-lo) or 1.0)+lo, f"{target} (original units)"
    if stype=="robust":
        med=float(p.get("median",0.0)); iqr=float(p.get("iqr",1.0)) or 1.0
        return series*iqr + med, f"{target} (original units)"
    return series, f"{target} (scaled)"

def _plot_overlay(train: pd.Series, val: pd.Series, test: pd.Series, outpath: Path, title: str, bins: int | None):
    t, v, te = train.to_numpy(), val.to_numpy(), test.to_numpy()
    grid = _nice_bins(np.concatenate([t[np.isfinite(t)], v[np.isfinite(v)], te[np.isfinite(te)]]), bins)
    plt.figure(figsize=(9,5.5))
    for arr, lab in [(t,"train"),(v,"val"),(te,"test")]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0: continue
        hist, edges = np.histogram(arr, bins=grid, density=True)
        mids = 0.5*(edges[1:]+edges[:-1])
        plt.step(mids, hist, where="mid", alpha=0.6, label=f"{lab} hist")
        plt.plot(mids, _kde(arr, mids), alpha=0.9, label=f"{lab} kde")
    plt.title(title); plt.xlabel("value"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True); plt.savefig(outpath, dpi=150); plt.close()

def _plot_single(series: pd.Series, outpath: Path, title: str, bins: int | None):
    arr = series.to_numpy(); arr = arr[np.isfinite(arr)]
    grid = _nice_bins(arr, bins)
    hist, edges = np.histogram(arr, bins=grid, density=True)
    mids = 0.5*(edges[1:]+edges[:-1])
    plt.figure(figsize=(8,5))
    if mids.size > 1:
        plt.bar(mids, hist, width=(mids[1]-mids[0]), alpha=0.4, align="center")
    else:
        plt.bar(mids, hist, alpha=0.4, align="center")
    plt.plot(mids, _kde(arr, mids), linewidth=2)
    plt.title(title); plt.xlabel("value"); plt.ylabel("density"); plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True); plt.savefig(outpath, dpi=150); plt.close()

def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description="Plot target hist/KDE for train/val/test (config-driven).")
    ap.add_argument("--target", default=None, help="Override target column; else uses data.target/locked manifest.")
    ap.add_argument("--bins", type=int, default=None, help="Histogram bins (default 60).")
    ap.add_argument("--inverse", action="store_true", help="Inverse-transform target if GLOBAL scaler meta exists.")
    ap.add_argument("--outdir", default=None, help="Override output directory.")
    args = ap.parse_args()

    art_dir    = _under(Path("."), _req(cfg, ["paths", "artifacts_dir"]))
    scaled_dir = _under(art_dir,     _req(cfg, ["paths", "features_scaled_dir"]))
    lock_dir   = _under(art_dir,     _req(cfg, ["paths", "features_locked_dir"]))
    plots_dir  = Path(args.outdir) if args.outdir else (art_dir / (cfg.get("reports",{}) or {}).get("plots_dir","reports/plots"))

    train_p, val_p, test_p, split_fmt = _split_paths(cfg, scaled_dir)
    target = args.target or _load_lock_target(cfg, lock_dir)
    if not target: raise SystemExit("Target not provided. Set data.target in YAML or pass --target.")

    train, val, test = (_read_df(train_p, split_fmt), _read_df(val_p, split_fmt), _read_df(test_p, split_fmt))
    for nm, df in (("train",train),("val",val),("test",test)):
        if target not in df.columns:
            raise KeyError(f"Target '{target}' not found in {nm} split: {train_p.parent}")

    label = f"{target} (scaled)"
    if args.inverse:
        scaler_meta = _load_scaler_meta(cfg, scaled_dir)
        s_train, label = _inverse_target(train[target], scaler_meta, target)
        s_val,   _     = _inverse_target(val[target],   scaler_meta, target)
        s_test,  _     = _inverse_target(test[target],  scaler_meta, target)
    else:
        s_train, s_val, s_test = train[target], val[target], test[target]

    bins = args.bins or 60
    _plot_overlay(s_train, s_val, s_test, plots_dir/"target_overlay_hist_kde.png",
                  title=f"Target distribution — {label}", bins=bins)
    _plot_single(s_train, plots_dir/"target_train_hist_kde.png", f"Train — {label}", bins=bins)
    _plot_single(s_val,   plots_dir/"target_val_hist_kde.png",   f"Val — {label}",   bins=bins)
    _plot_single(s_test,  plots_dir/"target_test_hist_kde.png",  f"Test — {label}",  bins=bins)
    print(f"[OK] wrote plots → {plots_dir}")

if __name__ == "__main__":
    main()