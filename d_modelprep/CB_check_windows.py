# d_modelprep/CB_check_windows.py
from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Dict, List
import numpy as np
import yaml
from copy import deepcopy

# ---------------- config ----------------
def _deep_update(dst: dict, src: dict | None) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> dict:
    base = yaml.safe_load(open("configs/default.yaml"))
    cfg_env = os.environ.get("CONFIG", "")
    if not cfg_env:
        return base
    merged = deepcopy(base)
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        with open(p, "r") as f:
            overlay = yaml.safe_load(f) or {}
        _deep_update(merged, overlay)
    return merged

# ---------------- utils ----------------
def _bytes_str(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n); i = 0
    while s >= 1024 and i < len(units) - 1:
        s /= 1024.0; i += 1
    return f"{s:.2f} {units[i]}"

# ---------------- checks ----------------
def _check_split(split_dir: Path, lookback: int, F_expected: int | None, max_shards: int) -> Dict:
    shards = sorted(split_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shards in {split_dir}")

    to_check = shards[:max_shards] if max_shards > 0 else shards

    sampled = 0
    ymins: List[float] = []
    ymaxs: List[float] = []
    total_bytes = 0
    F_actual: int | None = None
    warn_manifest_mismatch = False

    for s in to_check:
        with np.load(s, allow_pickle=False) as z:
            X, y = z["X"], z["y"]

        if X.ndim != 3:
            raise ValueError(f"Bad X rank in {s}: {X.shape} (expected 3D [N,T,F])")

        N, T, F = X.shape
        if T != lookback:
            raise ValueError(f"Bad X shape in {s}: {X.shape} (expected T={lookback} in [N,T,F])")

        if F_actual is None:
            F_actual = F
            if F_expected is not None and F_expected != F_actual:
                warn_manifest_mismatch = True
        else:
            if F != F_actual:
                raise ValueError(f"Inconsistent feature dim across shards: saw {F_actual} then {F} in {s}")

        if y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError(f"Bad y shape in {s}: {y.shape} (expected [N] or [N,1])")
            yv = y.reshape(-1)
        elif y.ndim == 1:
            yv = y
        else:
            raise ValueError(f"Bad y rank in {s}: {y.shape} (expected [N] or [N,1])")

        if yv.shape[0] != N:
            raise ValueError(f"Length mismatch in {s}: X.N={N} vs y.N={yv.shape[0]}")

        if N > 0:
            yv = yv.astype(np.float64, copy=False)
            if np.isfinite(yv).any():
                ymins.append(float(np.nanmin(yv)))
                ymaxs.append(float(np.nanmax(yv)))

        sampled += int(N)
        total_bytes += s.stat().st_size

    return {
        "checked_shards": len(to_check),
        "num_shards": len(shards),
        "sampled_windows": sampled,
        "y_min": (float(min(ymins)) if ymins else float("nan")),
        "y_max": (float(max(ymaxs)) if ymaxs else float("nan")),
        "size_str": _bytes_str(total_bytes),
        "dir": str(split_dir),
        "F_actual": int(F_actual) if F_actual is not None else None,
        "manifest_F": int(F_expected) if F_expected is not None else None,
        "warn_manifest_mismatch": bool(warn_manifest_mismatch),
    }

# ---------------- main ----------------
def main():
    t0 = time.time()
    cfg = _load_cfg()

    art_dir = Path(cfg["paths"]["artifacts_dir"])
    out_dir_cfg = cfg["sequence"]["out_dir"]
    seq_root = Path(out_dir_cfg) if Path(out_dir_cfg).is_absolute() else (art_dir / out_dir_cfg)

    max_shards = int(cfg.get("sequence", {}).get("check_max_shards", 2))

    manifest_path = seq_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run CA_make_windows_multi.py first.")

    man = json.loads(manifest_path.read_text())

    print("=== WINDOW CHECKS ===")
    print(f"[info] manifest: {manifest_path}")
    print(f"[info] sampling up to {max_shards} shard(s) per (split,horizon)")

    splits = man.get("splits", {})

    # Multi-horizon format: splits[split][str(H)]
    if isinstance(splits.get("train", {}), dict) and "X_dim" not in splits.get("train", {}):
        horizons = [int(h) for h in man.get("horizons", [])]
        for split in ("train", "val", "test"):
            hmap = splits.get(split, {})
            for H in horizons:
                info = hmap.get(str(H))
                if not info:
                    continue
                T = int(info["X_dim"]["T"])
                F_expected = int(info["X_dim"]["F"])
                split_dir = Path(info["paths"]["dir"])
                t1 = time.time()
                res = _check_split(split_dir, T, F_expected, max_shards)
                dt = time.time() - t1

                extra = ""
                if res["warn_manifest_mismatch"]:
                    extra = f" [warn: manifest F={res['manifest_F']} ≠ actual F={res['F_actual']}]"

                print(
                    f"[{split}][H={H:>3}] shards={res['num_shards']:2d} "
                    f"checked={res['checked_shards']:2d} "
                    f"sampled_windows={res['sampled_windows']:,} "
                    f"y_range=[{res['y_min']:.3f},{res['y_max']:.3f}] "
                    f"size={res['size_str']} "
                    f"dir={res['dir']}{extra} "
                    f"({dt:.2f}s)"
                )
    else:
        # Legacy single-horizon
        for split, info in splits.items():
            if not info or "X_dim" not in info or "paths" not in info:
                continue
            T = int(info["X_dim"]["T"])
            F_expected = int(info["X_dim"]["F"])
            split_dir = Path(info["paths"]["dir"])
            t1 = time.time()
            res = _check_split(split_dir, T, F_expected, max_shards)
            dt = time.time() - t1

            extra = ""
            if res["warn_manifest_mismatch"]:
                extra = f" [warn: manifest F={res['manifest_F']} ≠ actual F={res['F_actual']}]"

            print(
                f"[{split}] shards={res['num_shards']:2d} "
                f"checked={res['checked_shards']:2d} "
                f"sampled_windows={res['sampled_windows']:,} "
                f"y_range=[{res['y_min']:.3f},{res['y_max']:.3f}] "
                f"size={res['size_str']} "
                f"dir={res['dir']}{extra} "
                f"({dt:.2f}s)"
            )

    print(f"[done] total runtime {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()