# d_modelprep/CA_make_windows_multi.py
from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Set
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

rng = np.random.default_rng  # for reproducible subsampling

# ========== config loader ==========
try:
    from common.config_loader import load_cfg  # preferred if available
except Exception:
    # Minimal fallback if common loader isn't available
    import yaml
    from copy import deepcopy

    def _deep_update(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst

    def load_cfg() -> Dict:
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


# ========== helpers ==========
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def subsample_windows(
    X: np.ndarray,
    y: np.ndarray,
    sid: np.ndarray,
    times: np.ndarray | None,
    *,
    method: str,
    frac: float,
    seed: int,
    min_per_station: int = 0,
):
    """
    Subsample an already-built set of windows.
    - method='per_station_uniform' keeps station balance by sampling within each station.
    - method='time_head' or 'time_tail' keeps the first or last contiguous portion.
    """
    assert 0 < frac <= 1.0
    if frac == 1.0:
        return X, y, sid, times

    N = len(sid)
    idx_keep = []

    if method == "per_station_uniform":
        rnd = rng(seed)
        sid_series = pd.Series(sid)
        for station, idxs in sid_series.groupby(sid_series).groups.items():
            idxs = np.fromiter(sorted(idxs), dtype=int)
            k = max(min_per_station, int(np.ceil(frac * len(idxs))))
            k = min(k, len(idxs))
            if k < len(idxs):
                pick = rnd.choice(idxs, size=k, replace=False)
                pick.sort()
                idx_keep.append(pick)
            else:
                idx_keep.append(idxs)

    elif method in ("time_head", "time_tail"):
        order = np.arange(N)
        if method == "time_tail":
            order = order[::-1]
        k = int(np.ceil(frac * N))
        take = np.sort(order[:k])
        idx_keep.append(take)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    keep = np.sort(np.concatenate(idx_keep)) if len(idx_keep) > 1 else idx_keep[0]
    return X[keep], y[keep], sid[keep], (times[keep] if times is not None else None)


def _load_lock(art_dir: Path, cfg: Dict) -> tuple[str, str | None, list[str], list[str]]:
    """
    Load feature lock; if missing, infer from scaled train split.
    Returns: (time_col, target_col|None, id_cols, X_cols)
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

    # ---- Fallback: infer from scaled features on disk using config ----
    paths_cfg = cfg.get("paths", {}) or {}
    art_dir   = Path(paths_cfg.get("artifacts_dir", "experiments/artifacts"))

    scaled_rel = paths_cfg.get("features_scaled_dir", "features_scaled_ps")
    scaled_dir = Path(scaled_rel) if Path(scaled_rel).is_absolute() else (art_dir / scaled_rel)

    split_names = (cfg.get("output", {}) or {}).get("split_filenames", {}) or {}
    train_name  = split_names.get("train", "train.parquet")

    out_fmt = (cfg.get("scaling", {}) or {}).get("output_format",
              (cfg.get("output", {}) or {}).get("format", "parquet"))
    ext = {"parquet": "parquet", "feather": "feather", "csv": "csv"}.get(out_fmt, "parquet")
    base_stem = Path(train_name).stem
    train_path = scaled_dir / f"{base_stem}.{ext}"
    if not train_path.exists():
        raise FileNotFoundError(f"Scaled train split not found at {train_path}")

    if ext == "parquet":
        df_head = pd.read_parquet(train_path)
    elif ext == "feather":
        df_head = pd.read_feather(train_path)
    elif ext == "csv":
        df_head = pd.read_csv(train_path)
    else:
        raise ValueError(f"Unsupported scaling/output format: {out_fmt}")

    data_cfg   = cfg.get("data", {}) or {}
    id_col     = data_cfg.get("id_col") or data_cfg.get("station_col") or "station_id"
    time_col   = data_cfg.get("time_col") or "Datetime"
    target_col = data_cfg.get("target")   or "PM25_Concentration"

    exclude = {id_col, time_col, target_col}
    X_cols = [c for c in df_head.columns if c not in exclude]
    if not X_cols:
        raise RuntimeError("Inferred empty X_cols from scaled features; check your scaled split columns.")

    return time_col, target_col, [id_col], X_cols


def _sequence_iter(
    df: pd.DataFrame,
    X_cols: List[str],
    y_col: str,
    lookback: int,
    horizon: int,
    stride: int,
    dropna: bool,
    min_valid_ratio: float,
) -> Iterator[Tuple[np.ndarray, float]]:
    """
    Iterate windows from a single-station, time-sorted frame.

    Rules:
      - y must be finite.
      - If dropna is True: require mean(isfinite(X_window)) >= min_valid_ratio.
      - Remaining non-finites in X are replaced with 0.0.
    """
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

        if not np.isfinite(y_val):
            continue

        if dropna:
            valid_ratio = np.isfinite(x_seq).mean()
            if valid_ratio < float(min_valid_ratio):
                continue

        x_seq = np.nan_to_num(x_seq, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
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
    min_valid_ratio: float,
    shard_size: int,
    dtype: str,
    *,
    split_name: str,          # NEW
    sample_cfg: dict,         # NEW
) -> Dict:
    _ensure_dir(out_split_dir)
    np_dtype = np.dtype(dtype).name

    # 1) Build ALL windows for this (split, horizon)
    all_X, all_y, all_sid = [], [], []
    for sid, g in frame.groupby(station_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        for x_seq, y_val in _sequence_iter(
            g, X_cols, y_col, lookback, horizon, stride, dropna, min_valid_ratio
        ):
            all_X.append(np.asarray(x_seq, dtype=np_dtype))
            all_y.append(np.asarray(y_val, dtype=np_dtype))
            all_sid.append(str(sid))

    if len(all_X) == 0:
        man = {
            "lookback": lookback, "horizon": horizon, "stride": stride,
            "dropna": dropna, "min_valid_ratio": float(min_valid_ratio),
            "shard_size": shard_size, "dtype": np_dtype,
            "num_shards": 0, "total_windows": 0,
            "X_dim": {"T": lookback, "F": len(X_cols)},
            "y_dim": 1, "has_sid": True, "shards": [],
            "paths": {"dir": str(out_split_dir)},
        }
        with open(out_split_dir / "manifest.json", "w") as f:
            json.dump(man, f, indent=2)
        return man

    X = np.stack(all_X, axis=0).astype(np_dtype, copy=False)
    y = np.asarray(all_y, dtype=np_dtype)
    sid = np.asarray(all_sid, dtype="U32")
    times = None  # reserved for future (time-based sampling)

    # 2) Optional subsample (from YAML sequence.sample)
    if sample_cfg.get("enabled", False):
        frac_map = {
            "train": float(sample_cfg.get("train_frac", 1.0)),
            "val":   float(sample_cfg.get("val_frac",   1.0)),
            "test":  float(sample_cfg.get("test_frac",  1.0)),
        }
        frac = frac_map.get(split_name, 1.0)
        if 0.0 < frac < 1.0:
            X, y, sid, times = subsample_windows(
                X, y, sid, times,
                method=sample_cfg.get("method", "per_station_uniform"),
                frac=frac,
                seed=int(sample_cfg.get("seed", 42)),
                min_per_station=int(sample_cfg.get("min_windows_per_station", 0)),
            )
            print(f"[{split_name}][H={horizon}] subsampled to {len(sid)} windows (frac={frac})")

    # 3) Write shards AFTER subsampling
    shards_meta = []
    total = int(len(sid))
    shard_id = 0
    for start in range(0, total, shard_size):
        end = min(start + shard_size, total)
        shard_id += 1
        np.savez_compressed(
            out_split_dir / f"shard_{shard_id:03d}.npz",
            X=X[start:end],
            y=y[start:end],
            sid=sid[start:end],
        )
        shards_meta.append({"shard": f"shard_{shard_id:03d}.npz", "windows": int(end - start)})

    man = {
        "lookback": lookback,
        "horizon": horizon,
        "stride": stride,
        "dropna": dropna,
        "min_valid_ratio": float(min_valid_ratio),
        "shard_size": shard_size,
        "dtype": np_dtype,
        "num_shards": len(shards_meta),
        "total_windows": total,
        "X_dim": {"T": lookback, "F": len(X_cols)},
        "y_dim": 1,
        "has_sid": True,
        "shards": shards_meta,
        "paths": {"dir": str(out_split_dir)},
    }
    with open(out_split_dir / "manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    return man


def _ordered_unique(xs: List[str]) -> List[str]:
    seen = set(); out = []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


def _read_split(base: Path, name: str, needed: Set[str], time_col: str, out_fmt: str) -> pd.DataFrame:
    """
    Read a scaled split by name using the configured format and return only needed columns.
    """
    base_stem = Path(name).stem
    ext = {"parquet": "parquet", "feather": "feather", "csv": "csv"}.get(out_fmt, "parquet")
    p = base / f"{base_stem}.{ext}"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run the scaling step first")

    wanted_set = set(needed) | {time_col}

    if ext == "parquet":
        cols = None
        try:
            schema_cols = list(pq.read_schema(str(p)).names)
            cols = [c for c in schema_cols if c in wanted_set]
            cols = _ordered_unique(cols)
        except Exception:
            cols = None
        df = pd.read_parquet(p, columns=cols)
    elif ext == "feather":
        df = pd.read_feather(p)
        df = df[[c for c in df.columns if c in wanted_set]]
    elif ext == "csv":
        df = pd.read_csv(p, usecols=lambda c: c in wanted_set)
    else:
        raise ValueError(f"Unsupported format: {out_fmt}")

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
) -> None:
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


# ========== main ==========
def main():
    t0 = time.time()
    cfg = load_cfg()

    # Paths
    paths_cfg = cfg.get("paths", {}) or {}
    art_dir   = Path(paths_cfg.get("artifacts_dir", "experiments/artifacts"))

    scaled_rel = paths_cfg.get("features_scaled_dir", "features_scaled_ps")
    scaled_dir = Path(scaled_rel) if Path(scaled_rel).is_absolute() else (art_dir / scaled_rel)
    _ensure_dir(scaled_dir)

    seq_cfg = cfg.get("sequence", {}) or {}
    out_dir_cfg = seq_cfg.get("out_dir", "seq")
    out_root = Path(out_dir_cfg) if Path(out_dir_cfg).is_absolute() else (art_dir / out_dir_cfg)
    _ensure_dir(out_root)

    data_id_col   = (cfg.get("data", {}) or {}).get("id_col", "station_id")
    data_time_col = (cfg.get("data", {}) or {}).get("time_col", "Datetime")
    data_target   = (cfg.get("data", {}) or {}).get("target", "PM25_Concentration")

    # Lock or infer
    time_col_lock, target_col_lock, id_cols, X_cols = _load_lock(art_dir, cfg)
    if data_time_col != time_col_lock:
        print(f"[warn] data.time_col ({data_time_col}) != lock.time_col ({time_col_lock}); using lock")
    time_col    = time_col_lock
    y_col       = target_col_lock if target_col_lock else data_target
    station_col = data_id_col

    # Window params
    lookback   = int(seq_cfg.get("lookback", 168))
    raw_h      = seq_cfg.get("horizon", [1])
    horizons   = list(raw_h) if isinstance(raw_h, (list, tuple)) else [int(raw_h)]
    horizons   = [int(h) for h in horizons]
    stride     = int(seq_cfg.get("stride", 12))
    dropna     = bool(seq_cfg.get("dropna", True))
    min_valid_ratio = float(seq_cfg.get("min_valid_ratio", 0.8))
    shard_size = int(seq_cfg.get("shard_size", 15000))
    dtype      = np.dtype(str(seq_cfg.get("dtype", "float32"))).name

    # Split names
    tr_name    = seq_cfg.get("train_split", "train")
    va_name    = seq_cfg.get("val_split", "val")
    te_name    = seq_cfg.get("test_split", "test")

    # Format for reading scaled splits
    out_fmt = (cfg.get("scaling", {}) or {}).get("output_format",
              (cfg.get("output", {}) or {}).get("format", "parquet"))

    # Read scaled splits (column-pruned)
    needed = set(id_cols + [y_col] + X_cols + [time_col])
    train = _read_split(scaled_dir, tr_name, needed, time_col, out_fmt)
    val   = _read_split(scaled_dir, va_name, needed, time_col, out_fmt)
    test  = _read_split(scaled_dir, te_name, needed, time_col, out_fmt)

    for df in (train, val, test):
        if station_col not in df.columns:
            raise KeyError(f"'{station_col}' not found in split; check data.id_col and the scaling step.")
        df[station_col] = df[station_col].astype(str)
        df[time_col]    = pd.to_datetime(df[time_col], errors="coerce")

    non_feature = {station_col, time_col, y_col}
    X_cols = [c for c in X_cols if c not in non_feature and c in train.columns and c in val.columns and c in test.columns]
    X_cols = [c for c in X_cols if pd.api.types.is_numeric_dtype(train[c])]
    if not X_cols:
        raise RuntimeError("After sanitizing, X_cols is empty. Check your features/scaling step.")

    # Health report
    print("=== SPLIT HEALTH REPORT ===")
    _split_health_report("train", train, station_col, time_col, y_col, X_cols, lookback, horizons, stride)
    _split_health_report("val",   val,   station_col, time_col, y_col, X_cols, lookback, horizons, stride)
    _split_health_report("test",  test,  station_col, time_col, y_col, X_cols, lookback, horizons, stride)

    # Windowing
    sample_cfg = (cfg.get("sequence", {}) or {}).get("sample", {})  # NEW
    top = {"horizons": horizons, "dtype": dtype, "splits": {}}
    print("=== SEQUENCE WINDOWING ===")
    for split_name, frame in [("train", train), ("val", val), ("test", test)]:
        top["splits"][split_name] = {}
        for H in horizons:
            hs = time.time()
            split_dir = out_root / split_name / f"h={int(H)}"
            man = _build_split_for_h(
                frame, split_dir, station_col, time_col, X_cols, y_col,
                lookback, int(H), stride, dropna, min_valid_ratio, shard_size, dtype=dtype,
                split_name=split_name,          # pass split name
                sample_cfg=sample_cfg,          # pass YAML sampling cfg
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