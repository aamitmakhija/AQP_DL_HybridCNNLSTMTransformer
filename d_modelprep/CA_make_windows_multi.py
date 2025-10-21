# d_modelprep/CA_make_windows_multi.py
from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Set
from copy import deepcopy

import yaml
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------- config helpers ----------------
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
        with open(p, "r") as f:
            overlay = yaml.safe_load(f) or {}
        _deep_update(merged, overlay)
    return merged

def _require(cfg: Dict, path: List[str], name: str | None = None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            dotted = ".".join(path)
            raise KeyError(f"Missing config key: {dotted}{' ('+name+')' if name else ''}")
        cur = cur[k]
    return cur


# ---------------- io helpers ----------------
def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _load_lock(art_dir: Path, cfg: Dict) -> tuple[str, str | None, list[str], list[str]]:
    """
    Try to load the legacy feature lock. If missing, fall back to inferring from
    the current scaled features (per-station or global), using config for id/time/target.
    """
    lock_path = art_dir / "features_locked" / "feature_list.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        time_col   = lock.get("time_col")
        target_col = lock.get("target_col")
        id_cols    = lock.get("id_cols", [])
        X_cols     = lock.get("X_cols_ordered") or lock.get("X_cols") or []
        if not (time_col and id_cols and X_cols):
            raise KeyError(f"{lock_path} is missing required keys")
        return time_col, target_col, id_cols, X_cols

    # ---- Fallback: infer from scaled features on disk ----
    paths_cfg = cfg.get("paths", {})
    feats_rel = paths_cfg.get("features_scaled_dir", "features_scaled")
    scaled_dir = Path(feats_rel) if Path(feats_rel).is_absolute() else (art_dir / feats_rel)

    candidates = [
        scaled_dir / "train.parquet",
        art_dir / "features_scaled" / "train.parquet",
        art_dir / "features_scaled_ps" / "train.parquet",
    ]
    train_pq = next((p for p in candidates if p.exists()), None)
    if train_pq is None:
        raise FileNotFoundError(
            "Could not find a scaled train parquet to infer features.\n"
            f"Tried: {', '.join(str(p) for p in candidates)}"
        )

    df_head = pd.read_parquet(train_pq, columns=None)

    data_cfg   = cfg.get("data", {})
    id_col     = data_cfg.get("id_col") or data_cfg.get("station_col") or "station_id"
    time_col   = data_cfg.get("time_col") or "Datetime"
    target_col = data_cfg.get("target")   or "PM25_Concentration"

    exclude = {id_col, time_col, target_col}
    X_cols = [c for c in df_head.columns if c not in exclude]

    if not X_cols:
        raise RuntimeError("Inferred empty X_cols from scaled features; check your scaled parquet columns.")

    return time_col, target_col, [id_col], X_cols


# ---------------- windowing ----------------
def _sequence_iter(
    df: pd.DataFrame,
    X_cols: List[str],
    y_col: str,
    lookback: int,
    horizon: int,
    stride: int,
    dropna: bool,
) -> Iterator[Tuple[np.ndarray, float]]:
    # df must be sorted by time and from a single station

    # Coerce to numeric defensively
    X_df = df[X_cols].apply(pd.to_numeric, errors="coerce")
    X = X_df.to_numpy(dtype=float, copy=False)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float, copy=False)

    n = len(df)
    max_start = n - lookback - horizon + 1
    if max_start <= 0:
        return
    for start in range(0, max_start, stride):
        end = start + lookback
        t_idx = end - 1 + horizon
        x_seq = X[start:end, :]
        y_val = y[t_idx]
        # Drop windows with any non-finite values (NaN or Inf) in X or y
        if dropna and (not np.isfinite(x_seq).all() or not np.isfinite(y_val)):
            continue
        yield x_seq, float(y_val)

def _build_split_for_h(
    frame: pd.DataFrame,
    out_split_dir: Path,
    station_col: str,
    time_col: str,
    X_cols: List[str],
    y_col: str,
    lookback: int,
    horizon: int,
    stride: int,
    dropna: bool,
    shard_size: int,
    dtype: str,
) -> Dict:
    _ensure_dir(out_split_dir)
    shard_id = 0
    buf_X, buf_y, buf_sid = [], [], []
    np_dtype = np.dtype(dtype).name

    for sid, g in frame.groupby(station_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        for x_seq, y_val in _sequence_iter(g, X_cols, y_col, lookback, horizon, stride, dropna):
            buf_X.append(x_seq.astype(np_dtype, copy=False))
            buf_y.append(np.array(y_val, dtype=np_dtype))
            buf_sid.append(str(sid))  # NEW: track station id per window
            if len(buf_X) >= shard_size:
                shard_id += 1
                np.savez_compressed(
                    out_split_dir / f"shard_{shard_id:03d}.npz",
                    X=np.stack(buf_X, axis=0).astype(np_dtype, copy=False),
                    y=np.asarray(buf_y, dtype=np_dtype),
                    sid=np.asarray(buf_sid, dtype="U32"),  # NEW
                )
                buf_X.clear(); buf_y.clear(); buf_sid.clear()

    if buf_X:
        shard_id += 1
        np.savez_compressed(
            out_split_dir / f"shard_{shard_id:03d}.npz",
            X=np.stack(buf_X, axis=0).astype(np_dtype, copy=False),
            y=np.asarray(buf_y, dtype=np_dtype),
            sid=np.asarray(buf_sid, dtype="U32"),  # NEW
        )
        buf_X.clear(); buf_y.clear(); buf_sid.clear()

    shards = sorted([p.name for p in out_split_dir.glob("shard_*.npz")])
    shard_counts, total = [], 0
    for s in shards:
        with np.load(out_split_dir / s) as z:
            n = int(len(z["X"]))
        shard_counts.append({"shard": s, "windows": n})
        total += n

    man = {
        "lookback": lookback,
        "horizon": horizon,
        "stride": stride,
        "dropna": dropna,
        "shard_size": shard_size,
        "dtype": np_dtype,
        "num_shards": len(shards),
        "total_windows": total,
        "X_dim": {"T": lookback, "F": len(X_cols)},
        "y_dim": 1,
        "has_sid": True,  # NEW — signals that NPZ files include 'sid'
        "shards": shard_counts,
        "paths": {"dir": str(out_split_dir)},
    }
    with open(out_split_dir / "manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    return man


# ---------------- diagnostics ----------------
def _ordered_unique(xs: List[str]) -> List[str]:
    seen = set(); out = []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _read_split(base: Path, name: str, needed: Set[str], time_col: str) -> pd.DataFrame:
    p = base / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run dataprep/scale_features.py first")
    schema_names = list(pq.read_schema(str(p)).names)
    wanted = set(needed) | {time_col}
    cols = [c for c in schema_names if c in wanted]
    cols = _ordered_unique(cols)
    df = pd.read_parquet(p, columns=cols)
    if (df.columns == time_col).sum() > 1:
        first = df.loc[:, df.columns == time_col].iloc[:, 0]
        df = df.drop(columns=[time_col])
        df[time_col] = first
    return df

def _split_health_report(
    name: str,
    df: pd.DataFrame,
    station_col: str,
    time_col: str,
    y_col: str,
    X_cols: List[str],
    lookback: int,
    horizons: List[int],
    stride: int,
):
    if df.empty:
        print(f"[{name}] WARNING: empty split")
        return
    n_rows = len(df)
    n_stations = df[station_col].nunique(dropna=True)
    t = pd.to_datetime(df[time_col], errors="coerce")
    t_min, t_max = t.min(), t.max()
    y_nan = float(df[y_col].isna().mean())
    xs_sample = X_cols[: min(8, len(X_cols))]
    x_nan_map = {c: float(df[c].isna().mean()) for c in xs_sample if c in df.columns}
    print(f"[{name}] rows={n_rows:,}  stations={n_stations}  time=[{t_min} → {t_max}]")
    print(f"[{name}] NaN rate: y={y_nan:.3f}  X(sample)={{{', '.join(f'{k}:{v:.3f}' for k,v in x_nan_map.items())}}}")

    sizes = df.groupby(station_col, sort=False)[time_col].size().to_numpy()
    for H in horizons:
        ub = 0
        for s in sizes:
            ub += max(0, (int(s) - lookback - int(H) + 1) // max(1, stride))
        print(f"[{name}] Upper-bound windows (no NaN) H={H}: ~{ub:,}")


# ---------------- main ----------------
def main():
    t0 = time.time()
    cfg = _load_cfg()

    art_dir = Path(_require(cfg, ["paths", "artifacts_dir"]))
    feats_rel = _require(cfg, ["paths", "features_scaled_dir"])
    feats_dir = Path(feats_rel) if Path(feats_rel).is_absolute() else (art_dir / feats_rel)

    seq_cfg = _require(cfg, ["sequence"])
    out_dir_cfg = _require(seq_cfg, ["out_dir"])
    out_root = Path(out_dir_cfg) if Path(out_dir_cfg).is_absolute() else (art_dir / out_dir_cfg)
    _ensure_dir(out_root)

    data_id_col   = _require(cfg, ["data", "id_col"])
    data_time_col = _require(cfg, ["data", "time_col"])
    data_target   = _require(cfg, ["data", "target"])

    time_col_lock, target_col_lock, id_cols, X_cols = _load_lock(art_dir, cfg)

    if data_time_col != time_col_lock:
        print(f"[warn] data.time_col ({data_time_col}) != lock.time_col ({time_col_lock}); using lock")
    time_col    = time_col_lock
    y_col       = target_col_lock if target_col_lock else data_target
    station_col = data_id_col

    lookback   = int(_require(seq_cfg, ["lookback"]))
    raw_h      = _require(seq_cfg, ["horizon"])
    horizons   = list(raw_h) if isinstance(raw_h, (list, tuple)) else [int(raw_h)]
    horizons   = [int(h) for h in horizons]
    stride     = int(_require(seq_cfg, ["stride"]))
    dropna     = bool(seq_cfg.get("dropna", True))
    shard_size = int(_require(seq_cfg, ["shard_size"]))
    dtype      = str(seq_cfg.get("dtype", "float32"))
    dtype = np.dtype(dtype).name

    tr_name    = _require(seq_cfg, ["train_split"])
    va_name    = _require(seq_cfg, ["val_split"])
    te_name    = _require(seq_cfg, ["test_split"])

    needed = set(id_cols + [y_col] + X_cols + [time_col])
    train = _read_split(feats_dir, tr_name, needed, time_col)
    val   = _read_split(feats_dir, va_name, needed, time_col)
    test  = _read_split(feats_dir, te_name, needed, time_col)

    for df in (train, val, test):
        if station_col not in df.columns:
            raise KeyError(f"'{station_col}' not found in split; check data.id_col and feature lock.")
        df[station_col] = df[station_col].astype(str)
        df[time_col]    = pd.to_datetime(df[time_col], errors="coerce")

    non_feature = {station_col, time_col, y_col}
    X_cols = [c for c in X_cols if c not in non_feature and c in train.columns and c in val.columns and c in test.columns]
    X_cols = [c for c in X_cols if pd.api.types.is_numeric_dtype(train[c])]
    if not X_cols:
        raise RuntimeError("After sanitizing, X_cols is empty. Check your features/scaling step.")

    print("=== SPLIT HEALTH REPORT ===")
    _split_health_report("train", train, station_col, time_col, y_col, X_cols, lookback, horizons, stride)
    _split_health_report("val",   val,   station_col, time_col, y_col, X_cols, lookback, horizons, stride)
    _split_health_report("test",  test,  station_col, time_col, y_col, X_cols, lookback, horizons, stride)

    top = {"horizons": horizons, "dtype": dtype, "splits": {}}
    print("=== SEQUENCE WINDOWING ===")
    for split_name, frame in [("train", train), ("val", val), ("test", test)]:
        top["splits"][split_name] = {}
        for H in horizons:
            hs = time.time()
            split_dir = out_root / split_name / f"h={int(H)}"
            man = _build_split_for_h(
                frame, split_dir, station_col, time_col, X_cols, y_col,
                lookback, int(H), stride, dropna, shard_size, dtype=dtype
            )
            top["splits"][split_name][str(H)] = man
            he = time.time()
            print(f"[{split_name}][H={H}] shards={man['num_shards']:2d}  "
                  f"windows={man['total_windows']:,}  "
                  f"shape=[B,{man['X_dim']['T']},{man['X_dim']['F']}]  "
                  f"dtype={dtype}  time={he-hs:.2f}s  → {man['paths']['dir']}")

    with open(out_root / "manifest.json", "w") as f:
        json.dump(top, f, indent=2)

    print("=== DONE ===")
    print(f"[total runtime] {time.time() - t0:.2f}s")
    print(f"[manifest] {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()