# d_modelprep/CB_check_windows.py
#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys, time, argparse
from pathlib import Path
from typing import Dict, List, Iterable, Optional
import numpy as np

try:
    from common.config_loader import load_cfg
except Exception:
    import yaml
    from copy import deepcopy
    def _deep_update(dst: dict, src: dict | None) -> dict:
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict): _deep_update(dst[k], v)
            else: dst[k] = v
        return dst
    def _read_yaml(p: str | Path) -> dict:
        with Path(p).open("r") as f: return yaml.safe_load(f) or {}
    def load_cfg() -> dict:
        base = _read_yaml("configs/default.yaml")
        cfg_env = os.environ.get("CONFIG", "").strip()
        if cfg_env:
            merged = deepcopy(base)
            for path in [s.strip() for s in cfg_env.split(",") if s.strip()]:
                _deep_update(merged, _read_yaml(path))
            return merged
        if sys.platform == "darwin":
            cpu = Path("configs/cpu.yaml")
            if cpu.exists():
                merged = deepcopy(base); _deep_update(merged, _read_yaml(cpu)); return merged
        return base

def _req(d: dict, path: List[str], name: str | None = None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            dotted = ".".join(path); raise KeyError(f"configs: missing {dotted}{' ('+name+')' if name else ''}")
        cur = cur[k]
    return cur

def _bytes_str(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]; s = float(n); i = 0
    while s >= 1024 and i < len(units) - 1: s /= 1024.0; i += 1
    return f"{s:.2f} {units[i]}"

def _pick_horizons(man: dict, only: Iterable[int] | None) -> List[int]:
    hs = [int(h) for h in (man.get("horizons") or [])]
    if only:
        onlys = {int(x) for x in only}; hs = [h for h in hs if h in onlys]
    return hs

def _parse_only_horizons(arg: Optional[str]) -> Optional[set[int]]:
    if not arg: return None
    return {int(h.strip()) for h in arg.split(",") if h.strip()}

def _check_y_dtype(prev: Optional[str], arr: np.ndarray) -> str:
    d = str(arr.dtype)
    if prev is not None and d != prev: raise ValueError(f"Inconsistent y dtype across shards: {prev} vs {d}")
    return d

def _check_split(split_dir: Path, lookback: int, F_expected: Optional[int], max_shards: Optional[int]) -> Dict:
    shards = sorted(split_dir.glob("shard_*.npz"))
    if not shards:
        return {"checked_shards": 0, "num_shards": 0, "sampled_windows": 0, "y_min": float("nan"), "y_max": float("nan"),
                "size_str": _bytes_str(0), "dir": str(split_dir), "F_actual": None, "manifest_F": int(F_expected) if F_expected is not None else None,
                "warn_manifest_mismatch": False, "dtype_X": "<unknown>", "dtype_y": "<unknown>"}
    to_check = shards[:max_shards] if (max_shards and max_shards > 0) else shards
    sampled = 0; ymins: List[float] = []; ymaxs: List[float] = []
    total_bytes = 0; F_actual: Optional[int] = None; dtype_X: Optional[str] = None
    warn_manifest_mismatch = False; dtype_y: Optional[str] = None
    for s in to_check:
        with np.load(s, allow_pickle=False) as z:
            if "X" not in z or "y" not in z: raise KeyError(f"{s} missing 'X' or 'y'")
            X, y = z["X"], z["y"]; sid = z["sid"] if "sid" in z else None
        if X.ndim != 3: raise ValueError(f"Bad X rank in {s}: {X.shape} (expected 3D [N,T,F])")
        N, T, F = X.shape
        if T != lookback: raise ValueError(f"Bad X shape in {s}: {X.shape} (expected T={lookback} in [N,T,F])")
        if F_actual is None:
            F_actual = F
            if F_expected is not None and F_expected != F_actual: warn_manifest_mismatch = True
        elif F != F_actual: raise ValueError(f"Inconsistent feature dim across shards: {F_actual} vs {F} in {s}")
        if dtype_X is None: dtype_X = str(X.dtype)
        elif str(X.dtype) != dtype_X: raise ValueError(f"Inconsistent X dtype across shards: {dtype_X} vs {X.dtype} in {s}")
        yv = y.reshape(-1) if (y.ndim == 2 and y.shape[1] == 1) else (y if y.ndim == 1 else None)
        if yv is None: raise ValueError(f"Bad y shape in {s}: {y.shape} (expected [N] or [N,1])")
        if yv.shape[0] != N: raise ValueError(f"Length mismatch in {s}: X.N={N} vs y.N={yv.shape[0]}")
        if sid is not None and len(sid) != N: raise ValueError(f"Length mismatch in {s}: X.N={N} vs sid.N={len(sid)}")
        dtype_y = _check_y_dtype(dtype_y, yv)
        if N > 0 and np.isfinite(yv).any():
            ymins.append(float(np.nanmin(yv.astype(np.float64, copy=False))))
            ymaxs.append(float(np.nanmax(yv.astype(np.float64, copy=False))))
        sampled += int(N); total_bytes += s.stat().st_size
    return {"checked_shards": len(to_check), "num_shards": len(shards), "sampled_windows": sampled,
            "y_min": (float(min(ymins)) if ymins else float("nan")), "y_max": (float(max(ymaxs)) if ymaxs else float("nan")),
            "size_str": _bytes_str(total_bytes), "dir": str(split_dir), "F_actual": int(F_actual) if F_actual is not None else None,
            "manifest_F": int(F_expected) if F_expected is not None else None,
            "warn_manifest_mismatch": bool(warn_manifest_mismatch),
            "dtype_X": dtype_X or "<unknown>", "dtype_y": dtype_y or "<unknown>"}

def main():
    ap = argparse.ArgumentParser(description="Check window shards for shape/consistency.")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--only-splits", type=str, default=None)
    ap.add_argument("--only-horizons", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time(); cfg = load_cfg()
    art_dir = Path(_req(cfg, ["paths", "artifacts_dir"]))
    out_dir_cfg = _req(cfg, ["sequence", "out_dir"])
    seq_root = Path(out_dir_cfg) if Path(out_dir_cfg).is_absolute() else (art_dir / out_dir_cfg)

    seq_cfg = cfg.get("sequence", {}) or {}
    max_shards_cfg = int(seq_cfg.get("check_max_shards", 2))
    max_shards = args.max_shards if args.max_shards is not None else max_shards_cfg

    manifest_path = seq_root / "manifest.json"
    if not manifest_path.exists(): raise FileNotFoundError(f"{manifest_path} not found. Run CA_make_windows_multi.py first.")
    man = json.loads(manifest_path.read_text())

    splits = man.get("splits") or {}
    if not splits: raise ValueError(f"{manifest_path} has no 'splits'.")

    allowed_splits = {"train", "val", "test"}
    only_splits = None
    if args.only_splits:
        only_splits = {s.strip() for s in args.only_splits.split(",") if s.strip()}
        unknown = only_splits - allowed_splits
        if unknown: raise ValueError(f"--only-splits invalid: {sorted(unknown)}")

    only_horizons = _parse_only_horizons(args.only_horizons)

    print("=== WINDOW CHECKS ===")
    print(f"[info] manifest: {manifest_path}")
    print(f"[info] sampling up to {max_shards} shard(s) per (split,horizon)")

    if isinstance(splits.get("train", {}), dict) and "X_dim" not in splits.get("train", {}):
        horizons = [int(h) for h in (man.get("horizons") or [])]
        if only_horizons: horizons = [h for h in horizons if h in only_horizons]
        for split in ("train", "val", "test"):
            if only_splits and split not in only_splits: continue
            hmap = splits.get(split, {})
            for H in horizons:
                info = hmap.get(str(H)); 
                if not info: 
                    continue
                T = int(info["X_dim"]["T"]); F_expected = int(info["X_dim"]["F"])
                split_dir = Path(info["paths"]["dir"])
                t1 = time.time(); res = _check_split(split_dir, T, F_expected, max_shards); dt = time.time() - t1
                extra = (f" [warn: manifest F={res['manifest_F']} ≠ actual F={res['F_actual']}]"
                         if res["warn_manifest_mismatch"] else "")
                print(f"[{split}][H={H:>3}] shards={res['num_shards']:2d} checked={res['checked_shards']:2d} "
                      f"sampled_windows={res['sampled_windows']:,} y_range=[{res['y_min']:.3f},{res['y_max']:.3f}] "
                      f"X.dtype={res['dtype_X']} y.dtype={res['dtype_y']} size={res['size_str']} dir={res['dir']}{extra} ({dt:.2f}s)")
    else:
        for split, info in splits.items():
            if only_splits and split not in only_splits: continue
            if not info or "X_dim" not in info or "paths" not in info: continue
            T = int(info["X_dim"]["T"]); F_expected = int(info["X_dim"]["F"])
            split_dir = Path(info["paths"]["dir"])
            t1 = time.time(); res = _check_split(split_dir, T, F_expected, max_shards); dt = time.time() - t1
            extra = (f" [warn: manifest F={res['manifest_F']} ≠ actual F={res['F_actual']}]"
                     if res["warn_manifest_mismatch"] else "")
            print(f"[{split}] shards={res['num_shards']:2d} checked={res['checked_shards']:2d} "
                  f"sampled_windows={res['sampled_windows']:,} y_range=[{res['y_min']:.3f},{res['y_max']:.3f}] "
                  f"X.dtype={res['dtype_X']} y.dtype={res['dtype_y']} size={res['size_str']} dir={res['dir']}{extra} ({dt:.2f}s)")

    print(f"[done] total runtime {time.time() - t0:.2f}s")

if __name__ == "__main__": 
    main()