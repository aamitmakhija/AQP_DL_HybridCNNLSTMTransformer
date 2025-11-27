# d_modelprep/CA_make_windows_multi.py
from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

rng = np.random.default_rng

# ---------------- config loader (strict, overlay-friendly) ----------------
try:
    from common.config_loader import load_cfg
except Exception:
    import yaml
    from copy import deepcopy

    def _deep_update(dst: dict, src: dict | None) -> dict:
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst

    def load_cfg() -> Dict:
        base = yaml.safe_load(open("configs/default.yaml")) or {}
        cfg_env = os.environ.get("CONFIG", "")
        if not cfg_env:
            return base
        merged = deepcopy(base)
        for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
            with open(p, "r") as f:
                _deep_update(merged, yaml.safe_load(f) or {})
        return merged

# ---------------- small utils ----------------
def _req(d: dict, path: List[str], name: str | None = None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            dotted = ".".join(path)
            raise KeyError(f"configs: missing {dotted}{' ('+name+')' if name else ''}")
        cur = cur[k]
    return cur

def _opt(d: dict, path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _under(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (root / p)

def _avoid_double_prefix(base: Path, maybe: Path) -> Path:
    """If 'maybe' already starts with 'base', return 'maybe' as-is."""
    b, m = base.resolve(), maybe.resolve()
    try:
        m.relative_to(b)
        return maybe  # already prefixed by base
    except Exception:
        return base / maybe

# ---------------- subsampling ----------------
def subsample_windows(X, y, sid, times, *, method, frac, seed, min_per_station=0):
    if not (0 < frac <= 1.0):
        raise ValueError("frac must be in (0,1].")
    if frac == 1.0:
        return X, y, sid, times

    N = len(sid)
    if method == "per_station_uniform":
        rnd = rng(seed)
        groups = pd.Series(sid).groupby(sid).groups
        picks = []
        for _, idxs in groups.items():
            idxs = np.fromiter(sorted(idxs), dtype=int)
            k = min(max(min_per_station, int(np.ceil(frac * len(idxs)))), len(idxs))
            take = rnd.choice(idxs, size=k, replace=False) if k < len(idxs) else idxs
            picks.append(np.sort(take))
        keep = np.sort(np.concatenate(picks)) if picks else np.arange(0, 0, dtype=int)
    elif method in ("time_head", "time_tail"):
        order = np.arange(N)
        if method == "time_tail":
            order = order[::-1]
        keep = np.sort(order[: int(np.ceil(frac * N))])
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return X[keep], y[keep], sid[keep], (times[keep] if times is not None else None)

# ---------------- lock I/O ----------------
def _resolve_features_locked_dir(art_dir: Path, cfg: Dict) -> Path:
    locked_rel = _req(cfg, ["paths", "features_locked_dir"])
    return _under(art_dir, locked_rel)

def _load_lock(art_dir: Path, cfg: Dict) -> tuple[str, str | None, list[str], list[str]]:
    lf = _opt(cfg, ["features", "lock_file"], None)
    if lf:
        p = Path(lf)
        if not p.is_absolute() and not p.exists():
            fld = _opt(cfg, ["paths", "features_locked_dir"], None)
            if fld:
                p2 = _under(art_dir, fld) / Path(lf).name
                if p2.exists():
                    p = p2
        if not p.exists():
            raise FileNotFoundError(f"feature lock not found at {lf}")
        lock = json.loads(p.read_text())
    else:
        lock_dir = _resolve_features_locked_dir(art_dir, cfg)
        lock_file = _req(cfg, ["features", "locked_manifest"])
        p = lock_dir / lock_file
        if not p.exists():
            raise FileNotFoundError(f"feature lock not found at {p}")
        lock = json.loads(p.read_text())

    for k in ("time_col", "id_cols"):
        if k not in lock:
            raise KeyError(f"feature lock missing '{k}'")
    X_cols = lock.get("X_cols_ordered") or lock.get("X_cols")
    if not X_cols:
        raise KeyError("feature lock missing 'X_cols_ordered' (or 'X_cols')")
    return lock["time_col"], lock.get("target_col"), list(lock["id_cols"]), list(X_cols)

# ---------------- windowing core ----------------
def _sequence_iter(
    df: pd.DataFrame,
    X_cols: List[str],
    y_col: str,
    lookback: int,
    horizon: int,
    stride: int,
    dropna: bool,
    min_valid_ratio: float,
    nan_fill_value: float,
) -> Iterator[Tuple[np.ndarray, float]]:
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
        if dropna and float(np.isfinite(x_seq).mean()) < float(min_valid_ratio):
            continue
        x_seq = np.nan_to_num(x_seq, copy=False, nan=nan_fill_value, posinf=nan_fill_value, neginf=nan_fill_value)
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
    split_name: str,
    sample_cfg: dict,
    expected_X: List[str] | None = None,
    strict_features: bool = True,
    nan_fill_value: float = 0.0,
) -> Dict:
    _ensure_dir(out_split_dir)
    np_dtype = np.dtype(dtype).name

    # parity with lock
    if expected_X is not None:
        same_len = (len(X_cols) == len(expected_X))
        same_order = same_len and all(a == b for a, b in zip(X_cols, expected_X))
        if not (same_len and same_order):
            msg = (f"[feature-parity] mismatch @ {split_name} H={horizon}: "
                   f"lock_C={len(expected_X)} vs window_C={len(X_cols)}")
            if strict_features:
                raise RuntimeError(msg)
            else:
                print("[warn]", msg)

    # build all windows
    all_X, all_y, all_sid = [], [], []
    for sid, g in frame.groupby(station_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        for x_seq, y_val in _sequence_iter(g, X_cols, y_col, lookback, horizon, stride, dropna, min_valid_ratio, nan_fill_value):
            all_X.append(np.asarray(x_seq, dtype=np_dtype))
            all_y.append(np.asarray(y_val, dtype=np_dtype))
            all_sid.append(str(sid))

    if not all_X:
        man = {
            "lookback": lookback, "horizon": horizon, "stride": stride, "dropna": dropna,
            "min_valid_ratio": float(min_valid_ratio), "shard_size": shard_size, "dtype": np_dtype,
            "num_shards": 0, "total_windows": 0,
            "X_dim": {"T": lookback, "F": len(X_cols)}, "X_cols": list(X_cols),
            "lock_X_cols": (list(expected_X) if expected_X is not None else None),
            "y_dim": 1, "has_sid": True, "shards": [], "paths": {"dir": str(out_split_dir)},
        }
        (out_split_dir / "manifest.json").write_text(json.dumps(man, indent=2))
        (out_split_dir / "features_used.json").write_text(
            json.dumps({"X_cols": list(X_cols), "lock_X_cols": (list(expected_X) if expected_X else None)}, indent=2)
        )
        return man

    X = np.stack(all_X, axis=0).astype(np_dtype, copy=False)
    y = np.asarray(all_y, dtype=np_dtype)
    sid = np.asarray(all_sid, dtype="U32")
    times = None  # reserved

    # optional subsample
    if sample_cfg.get("enabled", False):
        frac_map = {
            "train": float(sample_cfg["train_frac"]),
            "val":   float(sample_cfg["val_frac"]),
            "test":  float(sample_cfg["test_frac"]),
        }
        X, y, sid, times = subsample_windows(
            X, y, sid, times,
            method=str(sample_cfg["method"]),
            frac=frac_map.get(split_name, 1.0),
            seed=int(sample_cfg["seed"]),
            min_per_station=int(sample_cfg["min_windows_per_station"]),
        )
        print(f"[{split_name}][H={horizon}] subsampled to {len(sid)} windows")

    # write shards
    shards_meta, total = [], int(len(sid))
    for i, start in enumerate(range(0, total, shard_size), start=1):
        end = min(start + shard_size, total)
        np.savez_compressed(out_split_dir / f"shard_{i:03d}.npz",
                            X=X[start:end], y=y[start:end], sid=sid[start:end])
        shards_meta.append({"shard": f"shard_{i:03d}.npz", "windows": int(end - start)})

    man = {
        "lookback": lookback, "horizon": horizon, "stride": stride, "dropna": dropna,
        "min_valid_ratio": float(min_valid_ratio), "shard_size": shard_size, "dtype": np_dtype,
        "num_shards": len(shards_meta), "total_windows": total,
        "X_dim": {"T": lookback, "F": len(X_cols)}, "X_cols": list(X_cols),
        "lock_X_cols": (list(expected_X) if expected_X is not None else None),
        "y_dim": 1, "has_sid": True, "shards": shards_meta,
        "paths": {"dir": str(out_split_dir)},
    }
    (out_split_dir / "manifest.json").write_text(json.dumps(man, indent=2))
    (out_split_dir / "features_used.json").write_text(
        json.dumps({"X_cols": list(X_cols), "lock_X_cols": (list(expected_X) if expected_X else None)}, indent=2)
    )
    return man

# ---------------- split I/O ----------------
def _ordered_unique(xs: List[str]) -> List[str]:
    seen = set(); out = []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _read_split(base: Path, name: str, needed: Set[str], time_col: str, out_fmt: str) -> pd.DataFrame:
    base_stem = Path(name).stem
    p = base / f"{base_stem}.{out_fmt}"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run the scaling step first")

    wanted = set(needed) | {time_col}
    if out_fmt == "parquet":
        try:
            schema_cols = list(pq.read_schema(str(p)).names)
            cols = _ordered_unique([c for c in schema_cols if c in wanted])
        except Exception:
            cols = None
        df = pd.read_parquet(p, columns=cols)
    elif out_fmt == "feather":
        df = pd.read_feather(p)
        df = df[[c for c in df.columns if c in wanted]]
    elif out_fmt == "csv":
        df = pd.read_csv(p, usecols=lambda c: c in wanted)
    else:
        raise SystemExit(f"Unsupported split format: {out_fmt!r}")

    if (df.columns == time_col).sum() > 1:
        first = df.loc[:, df.columns == time_col].iloc[:, 0]
        df = df.drop(columns=[time_col])
        df[time_col] = first
    return df

def _split_health_report(name, df, station_col, time_col, y_col, X_cols, lookback, horizons, stride) -> None:
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
    cfg = load_cfg()

    # paths
    art_dir = _under(Path("."), _req(cfg, ["paths", "artifacts_dir"]))
    scaled_dir_rel = _req(cfg, ["paths", "features_scaled_dir"])
    out_dir_rel = _req(cfg, ["sequence", "out_dir"])

    scaled_dir = _under(art_dir, scaled_dir_rel)
    # avoid double prefix if sequence.out_dir already contains artifacts_dir
    out_root_candidate = Path(out_dir_rel)
    out_root = (out_root_candidate if out_root_candidate.is_absolute()
                else _avoid_double_prefix(art_dir, out_root_candidate))
    _ensure_dir(scaled_dir)
    _ensure_dir(out_root)

    # schema + lock
    data_time_col = _req(cfg, ["data", "time_col"])
    data_target   = _req(cfg, ["data", "target"])
    time_col_lock, target_col_lock, id_cols_from_lock, lock_X_cols = _load_lock(art_dir, cfg)
    if data_time_col != time_col_lock:
        print(f"[warn] data.time_col ({data_time_col}) != lock.time_col ({time_col_lock}); using lock")
    time_col = time_col_lock
    y_col = (target_col_lock or data_target)
    station_col = id_cols_from_lock[0]

    # sequence params
    seq_cfg = _req(cfg, ["sequence"])
    rq = lambda k: _req(seq_cfg, [k])
    lookback = int(rq("lookback"))
    raw_h = rq("horizon")
    horizons = [int(h) for h in (raw_h if isinstance(raw_h, (list, tuple)) else [raw_h])]
    stride = int(rq("stride"))
    dropna = bool(rq("dropna"))
    min_valid_ratio = float(rq("min_valid_ratio"))
    shard_size = int(rq("shard_size"))
    dtype = np.dtype(str(rq("dtype"))).name
    nan_fill_value = float(rq("nan_fill_value"))
    strict_features = bool(rq("strict_features"))

    # split names + formats
    tr_name = str(rq("train_split"))
    va_name = str(rq("val_split"))
    te_name = str(rq("test_split"))

    if "scaling" in cfg and isinstance(cfg["scaling"], dict) and cfg["scaling"].get("output_format"):
        out_fmt = str(cfg["scaling"]["output_format"]).lower()
    else:
        out_fmt = str(_req(cfg, ["output", "format"])).lower()

    if out_fmt not in ("parquet", "feather", "csv"):
        raise SystemExit(f"Unsupported split format: {out_fmt!r}")

    # read scaled splits (column-pruned)
    needed = set(id_cols_from_lock + [y_col] + lock_X_cols + [time_col])
    read = lambda name: _read_split(scaled_dir, name, needed, time_col, out_fmt)
    train, val, test = read(tr_name), read(va_name), read(te_name)

    for df in (train, val, test):
        if station_col not in df.columns:
            raise KeyError(f"'{station_col}' not found in split; check feature lock id_cols and scaling step.")
        df[station_col] = df[station_col].astype(str)
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # sanitize X
    non_feature = {station_col, time_col, y_col}
    X_cols = [c for c in lock_X_cols if c not in non_feature and c in train.columns and c in val.columns and c in test.columns]
    X_cols = [c for c in X_cols if pd.api.types.is_numeric_dtype(train[c])]
    if not X_cols:
        raise RuntimeError("After sanitizing, X_cols is empty. Check features/scaling.")

    # health report
    print("=== SPLIT HEALTH REPORT ===")
    _split_health_report("train", train, station_col, time_col, y_col, X_cols, lookback, horizons, stride)
    _split_health_report("val",   val,   station_col, time_col, y_col, X_cols, lookback, horizons, stride)
    _split_health_report("test",  test,  station_col, time_col, y_col, X_cols, lookback, horizons, stride)

    # windowing
    top = {"horizons": horizons, "dtype": dtype, "splits": {}}
    sample_cfg = (_req(seq_cfg, ["sample"])
                  if (isinstance(seq_cfg.get("sample"), dict) and seq_cfg["sample"].get("enabled"))
                  else {"enabled": False})

    print("=== SEQUENCE WINDOWING ===")
    for split_name, frame in (("train", train), ("val", val), ("test", test)):
        top["splits"][split_name] = {}
        for H in horizons:
            t1 = time.time()
            split_dir = out_root / split_name / f"h={int(H)}"
            man = _build_split_for_h(
                frame, split_dir, station_col, time_col, X_cols, y_col,
                lookback, int(H), stride, dropna, min_valid_ratio, shard_size, dtype=dtype,
                split_name=split_name, sample_cfg=sample_cfg,
                expected_X=lock_X_cols, strict_features=strict_features, nan_fill_value=nan_fill_value
            )
            top["splits"][split_name][str(H)] = man
            dt = time.time() - t1
            print(f"[{split_name}][H={H}] shards={man['num_shards']:2d}  windows={man['total_windows']:,}  "
                  f"shape=[B,{man['X_dim']['T']},{man['X_dim']['F']}]  dtype={dtype}  time={dt:.2f}s  → {man['paths']['dir']}")

    with open(out_root / "manifest.json", "w") as f:
        json.dump(top, f, indent=2)

    print("=== DONE ===")
    print(f"[total runtime] {time.time() - t0:.2f}s")
    print(f"[manifest] {out_root / 'manifest.json'}")

if __name__ == "__main__":
    main()