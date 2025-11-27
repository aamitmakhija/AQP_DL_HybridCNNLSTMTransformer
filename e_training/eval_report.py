#!/usr/bin/env python3
# e_training/eval_report.py
from __future__ import annotations

import os, json, inspect
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Iterable, Optional
from copy import deepcopy

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from e_training.metrics import METRICS
from e_training.models.factory import build_hybrid  # impl selected via YAML or train-time meta

# =========================
# Config helpers
# =========================
def _deep_update(dst: Dict, src: Dict | None):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> Dict:
    base = yaml.safe_load(open("configs/default.yaml")) or {}
    cfg_env = os.environ.get("CONFIG")
    if not cfg_env:
        return base
    merged = deepcopy(base)
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        overlay = yaml.safe_load(open(p)) or {}
        _deep_update(merged, overlay)
    return merged

def _req(cfg: Dict, path: List[str]) -> Any:
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise KeyError(f"configs: missing {'.'.join(path)}")
        cur = cur[k]
    return cur

def _opt(cfg: Dict, path: List[str], default: Any = None) -> Any:
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _resolve_device(pref: str | None) -> str:
    s = (pref or "auto").lower()
    if s == "auto":
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return s

def _resolve_torch_dtype(name: str | None) -> torch.dtype:
    if name is None: return torch.float32
    s = str(name).lower()
    if s in ("float16","fp16","half"):   return torch.float16
    if s in ("bfloat16","bf16"):         return torch.bfloat16
    if s in ("float32","fp32","single"): return torch.float32
    if s in ("float64","fp64","double"): return torch.float64
    raise ValueError(f"Unknown float dtype: {name}")

# =========================
# Feature lock
# =========================
def _resolve_locked_dir(art_dir: Path, cfg: Dict) -> Path:
    rel = _req(cfg, ["paths", "features_locked_dir"])
    return Path(rel) if Path(rel).is_absolute() else (art_dir / rel)

def _load_feature_lock(art_dir: Path, cfg: Dict) -> Tuple[str, str | None, List[str], List[str]]:
    features_cfg = (cfg.get("features", {}) or {})
    lock_file_opt = features_cfg.get("lock_file")

    if lock_file_opt:
        p = Path(lock_file_opt)
        if not p.is_absolute():
            p = Path(lock_file_opt)
        if not p.exists():
            fld = _opt(cfg, ["paths", "features_locked_dir"], None)
            if fld:
                p2 = (Path(fld) if Path(fld).is_absolute() else (art_dir / fld)) / Path(lock_file_opt).name
                if p2.exists():
                    p = p2
        if not p.exists():
            raise FileNotFoundError(f"Feature lock not found at {lock_file_opt}")
        lock = json.loads(p.read_text())
    else:
        lock_dir = _resolve_locked_dir(art_dir, cfg)
        manifest = features_cfg.get("locked_manifest")
        if not manifest:
            raise KeyError("configs: missing features.locked_manifest (or features.lock_file)")
        p = lock_dir / manifest
        if not p.exists():
            raise FileNotFoundError(f"Feature lock not found at {p}")
        lock = json.loads(p.read_text())

    for k in ("time_col", "id_cols"):
        if k not in lock:
            raise KeyError("feature lock missing required key: " + k)
    X_cols = lock.get("X_cols_ordered") or lock.get("X_cols")
    if not isinstance(X_cols, list) or not X_cols:
        raise KeyError("feature lock missing 'X_cols_ordered' (or 'X_cols')")

    time_col   = lock["time_col"]
    target_col = lock.get("target_col")
    id_cols    = list(lock["id_cols"])
    return time_col, target_col, id_cols, list(X_cols)

# =========================
# I/O helpers
# =========================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _ordered_unique(xs: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _read_split(base: Path, name: str, needed: Set[str], time_col: str, out_fmt: str) -> pd.DataFrame:
    stem = Path(name).stem
    if out_fmt not in ("parquet", "feather", "csv"):
        raise ValueError("configs: output/scaling format must be one of: parquet|feather|csv")

    p = base / f"{stem}.{out_fmt}"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run the scaling step")

    wanted = set(needed) | {time_col}
    if out_fmt == "parquet":
        try:
            import pyarrow.parquet as pq
            cols = [c for c in list(pq.read_schema(str(p)).names) if c in wanted]
            cols = _ordered_unique(cols)
        except Exception:
            cols = None
        df = pd.read_parquet(p, columns=cols)
    elif out_fmt == "feather":
        df = pd.read_feather(p)
        df = df[[c for c in df.columns if c in wanted]]
    else:
        df = pd.read_csv(p, usecols=lambda c: c in wanted)

    if (df.columns == time_col).sum() > 1:
        first = df.loc[:, df.columns == time_col].iloc[:, 0]
        df = df.drop(columns=[time_col])
        df[time_col] = first
    return df

# =========================
# Feature sanitization
# =========================
def _sanitize_X_cols(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    station_col: str,
    time_col: str,
    y_col: str,
    X_cols_lock: List[str],
    exclude_cols: Set[str],
) -> List[str]:
    non_feature = {station_col, time_col, y_col} | set(exclude_cols)
    x = [c for c in X_cols_lock if c not in non_feature
         and c in tr_df.columns and c in va_df.columns and c in te_df.columns]
    x = [c for c in x if pd.api.types.is_numeric_dtype(tr_df[c])]
    if not x:
        raise RuntimeError("After sanitizing, X_cols is empty. Check scaling/exclusions.")
    return x

# =========================
# Windowing (test/rebuild)
# =========================
def _make_windows_per_station(
    df: pd.DataFrame,
    station_col: str,
    time_col: str,
    X_cols: List[str],
    y_col: str,
    lookback: int,
    horizon: int,
    stride: int,
    min_valid_ratio: float | None,
    nan_fill_value: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    all_X, all_y, stations = [], [], []
    for sid, g in df.groupby(station_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        X = g[X_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=False)
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy(float, copy=False)
        n = len(g)
        max_start = n - lookback - horizon + 1
        if max_start <= 0:
            continue
        for start in range(0, max_start, stride):
            end = start + lookback
            tgt = end - 1 + horizon
            x_seq = X[start:end, :]
            y_val = y[tgt]
            if not np.isfinite(y_val):
                continue
            if min_valid_ratio is None:
                if not np.isfinite(x_seq).all():
                    continue
            else:
                if np.isfinite(x_seq).mean() < float(min_valid_ratio):
                    continue
                if nan_fill_value is not None:
                    x_seq = np.where(np.isfinite(x_seq), x_seq, float(nan_fill_value))
                else:
                    x_seq = np.nan_to_num(x_seq, copy=False)
            all_X.append(x_seq)
            all_y.append(y_val)
            stations.append(str(sid))
    if not all_X:
        return np.empty((0, lookback, len(X_cols))), np.empty((0,), float), []
    return np.stack(all_X, axis=0), np.asarray(all_y, float), stations

# =========================
# Diagnostics
# =========================
def _upper_bound_windows(df: pd.DataFrame, station_col: str, time_col: str,
                         lookback: int, horizon: int, stride: int) -> int:
    sizes = df.groupby(station_col, sort=False)[time_col].size().to_numpy()
    ub = 0
    for s in sizes:
        ub += max(0, (int(s) - lookback - int(horizon) + 1) // max(1, stride))
    return ub

def _print_split_diag(name: str, df: pd.DataFrame, station_col: str, time_col: str,
                      y_col: str, X_cols: list[str], lookback: int,
                      horizons: list[int], stride: int):
    if df.empty:
        print(f"[{name}] WARNING: empty split"); return
    n_rows = len(df)
    n_stations = df[station_col].nunique(dropna=True)
    t = pd.to_datetime(df[time_col], errors="coerce")
    t_min, t_max = t.min(), t.max()
    y_nan = float(df[y_col].isna().mean())
    xs_sample = X_cols[: min(8, len(X_cols))]
    x_nan_map = {c: float(df[c].isna().mean()) for c in xs_sample if c in df.columns}
    print(f"[{name}] rows={n_rows:,}  stations={n_stations}  time=[{t_min} → {t_max}]")
    print(f"[{name}] NaN rate: y={y_nan:.3f}  X(sample)={{{', '.join(f'{k}:{v:.3f}' for k,v in x_nan_map.items())}}}")
    for H in horizons:
        ub = _upper_bound_windows(df, station_col, time_col, lookback, int(H), stride)
        print(f"[{name}] Upper-bound windows (no NaN) H={H}: ~{ub:,}")

# =========================
# Checkpoint helpers
# =========================
def _find_checkpoint(model_root: Path, H: int) -> Path | None:
    exact = model_root / f"hybrid_h{int(H)}.pt"
    if exact.exists():
        return exact
    cands = list(model_root.rglob("*.pt"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def _load_train_time_params(model_root: Path, H: int) -> dict | None:
    p = model_root / f"hybrid_h{int(H)}_train_summary.json"
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text())
        return meta.get("params", None)
    except Exception:
        return None

def _filter_state_dict(sd: dict, *, drop_prefixes: tuple[str, ...]) -> dict:
    return {k: v for k, v in sd.items() if not any(k.startswith(px) for px in drop_prefixes)}

def _copy_overlap(dst_tensor: torch.Tensor, src_tensor: torch.Tensor) -> None:
    if dst_tensor.shape == src_tensor.shape:
        dst_tensor.copy_(src_tensor); return
    dshape, sshape = list(dst_tensor.shape), list(src_tensor.shape)
    dims = min(len(dshape), len(sshape))
    slices = tuple(slice(0, min(dshape[i], sshape[i])) for i in range(dims))
    try:
        dst_tensor[slices].copy_(src_tensor[slices])
    except Exception:
        pass

def load_state_dict_forgiving(model: nn.Module, state_dict: dict,
                              drop_prefixes: tuple[str, ...] = ("sid_emb",)) -> tuple[list[str], list[str], list[str]]:
    sd = _filter_state_dict(state_dict, drop_prefixes=drop_prefixes)
    model_sd = model.state_dict()
    loaded_ok, skipped_mm, missing = [], [], []
    for k, v in sd.items():
        if k in model_sd:
            try:
                _copy_overlap(model_sd[k], v)
                loaded_ok.append(k)
            except Exception:
                skipped_mm.append(k)
    for k in model_sd.keys():
        if k not in sd:
            missing.append(k)
    model.load_state_dict(model_sd, strict=False)
    return loaded_ok, skipped_mm, missing

# =========================
# AMP helpers
# =========================
def _supports_sid_kw(m: nn.Module) -> bool:
    try:
        sig = inspect.signature(m.forward)
        return any(p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name in ("sid","sid_idx")
                   for p in sig.parameters.values())
    except Exception:
        # fallback to bytecode varnames check
        try:
            return ("sid" in str(m.forward.__code__.co_varnames)) or ("sid_idx" in str(m.forward.__code__.co_varnames))
        except Exception:
            return False

# =========================
# Prediction (AMP, safe)
# =========================
@torch.no_grad()
def _predict(model: nn.Module,
             X: np.ndarray,
             device: torch.device,
             batch_size: int,
             sid_idx: np.ndarray | None,
             sanitize_cfg: Dict[str, Any],
             *,
             amp_enabled: bool,
             amp_device: str,
             amp_dtype: torch.dtype,
             horizon: int) -> np.ndarray:
    preds: List[np.ndarray] = []
    N = len(X)
    bad_total = 0
    use_sid_kw = _supports_sid_kw(model)

    clamp_cfg = sanitize_cfg.get("clamp", None) or {}
    cmin = clamp_cfg.get("min", None)
    cmax = clamp_cfg.get("max", None)
    nan_fill_value = sanitize_cfg.get("nan_fill_value", None)

    for i in range(0, N, batch_size):
        xb_np = X[i:i+batch_size]
        bad_total += int(np.size(xb_np) - np.isfinite(xb_np).sum())
        if nan_fill_value is not None:
            xb_np = np.where(np.isfinite(xb_np), xb_np, float(nan_fill_value))
        else:
            xb_np = np.nan_to_num(xb_np, copy=False)
        if cmin is not None or cmax is not None:
            np.clip(xb_np,
                    cmin if cmin is not None else -np.inf,
                    cmax if cmax is not None else  np.inf,
                    out=xb_np)

        xb = torch.from_numpy(xb_np).to(device, dtype=torch.float32)
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled):
            if sid_idx is not None:
                sb = torch.from_numpy(sid_idx[i:i+batch_size]).to(device, dtype=torch.long)
                out = model(xb, sid_idx=sb) if use_sid_kw else model(xb, sb)
            else:
                out = model(xb)

        pb_t = out[int(horizon)] if isinstance(out, dict) else out
        preds.append(pb_t.detach().cpu().view(-1).numpy())

    if bad_total > 0:
        print(f"[predict] sanitized {bad_total} non-finite value(s)")
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), float)

# =========================
# MAIN
# =========================
def main():
    cfg = _load_cfg()

    # paths & device
    art_rel  = _req(cfg, ["paths", "artifacts_dir"])
    art_dir  = Path(art_rel) if Path(art_rel).is_absolute() else Path(art_rel)

    dl_cfg   = _req(cfg, ["dl"])
    device_s = _resolve_device(_opt(dl_cfg, ["device"], "auto"))
    device   = torch.device(device_s)
    batch    = int(_opt(dl_cfg, ["batch_size"], 64))

    # AMP (eval-scoped)
    eval_cfg = _req(cfg, ["eval"])
    amp_cfg  = _opt(eval_cfg, ["amp"], {}) or {}
    amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type in ("cuda","mps")
    amp_dtype   = _resolve_torch_dtype(amp_cfg.get("dtype", "fp16"))
    amp_device  = device.type

    model_rel = _opt(dl_cfg, ["model_dir"], "models")
    model_root = Path(model_rel) if Path(model_rel).is_absolute() else (art_dir / model_rel)
    model_root.mkdir(parents=True, exist_ok=True)

    # locked features + data keys
    time_col, target_lock, id_cols_lock, X_cols_lock = _load_feature_lock(art_dir, cfg)
    data_cfg    = _req(cfg, ["data"])
    station_col = _req(data_cfg, ["id_col"])
    y_col       = target_lock if target_lock else _req(data_cfg, ["target"])

    # features dir + splits + format
    feats_rel = _req(cfg, ["paths", "features_scaled_dir"])
    feats_dir = Path(feats_rel) if Path(feats_rel).is_absolute() else (art_dir / feats_rel)

    seq_cfg   = _req(cfg, ["sequence"])
    lookback  = int(_req(seq_cfg, ["lookback"]))
    horizon_v = _req(seq_cfg, ["horizon"])
    horizons  = list(horizon_v) if isinstance(horizon_v, (list, tuple)) else [int(horizon_v)]
    horizons  = [int(h) for h in horizons]
    stride    = int(_req(seq_cfg, ["stride"]))
    seq_out_dir_rel = _req(seq_cfg, ["out_dir"])
    seq_out_dir = Path(seq_out_dir_rel) if Path(seq_out_dir_rel).is_absolute() else (art_dir / seq_out_dir_rel)

    tr_name = _req(seq_cfg, ["train_split"])
    va_name = _req(seq_cfg, ["val_split"])
    te_name = _req(seq_cfg, ["test_split"])

    out_fmt = _opt(_opt(cfg, ["scaling"], {}), ["output_format"],
                   _opt(_opt(cfg, ["output"], {}), ["format"], "parquet"))

    # read splits
    needed = set([station_col, time_col, y_col] + X_cols_lock)
    tr_df = _read_split(feats_dir, tr_name, needed, time_col, out_fmt)
    va_df = _read_split(feats_dir, va_name, needed, time_col, out_fmt)
    te_df = _read_split(feats_dir, te_name, needed, time_col, out_fmt)

    for df in (tr_df, va_df, te_df):
        if station_col not in df.columns:
            raise KeyError(f"'{station_col}' not found in split")
        df[station_col] = df[station_col].astype(str)
        df[time_col]    = pd.to_datetime(df[time_col], errors="coerce")

    exclude_cols_cfg = set((_opt(cfg, ["scaling", "exclude_columns"], []) or []))
    drop_feats_cfg   = set((_opt(cfg, ["missing", "drop_features"], []) or []))
    exclude_all      = exclude_cols_cfg | drop_feats_cfg

    X_cols = _sanitize_X_cols(tr_df, va_df, te_df, station_col, time_col, y_col, X_cols_lock, exclude_all)
    print(f"\n[eval] Lock proposes {len(X_cols)} feature(s): {X_cols}\n")

    _print_split_diag("test", te_df, station_col, time_col, y_col, X_cols, lookback, horizons, stride)

    # eval settings
    want_per_station = bool(_opt(eval_cfg, ["per_station"], True))
    save_preds       = bool(_opt(eval_cfg, ["save_preds"], True))
    min_valid_eval   = _opt(eval_cfg, ["min_valid_ratio"], None)
    sanitize_cfg     = _opt(eval_cfg, ["sanitize"], {}) or {}

    # reports output
    reports_rel = _opt(cfg, ["paths", "reports_dir"], None)
    reports_dir = (art_dir / "reports") if reports_rel is None else \
                  (Path(reports_rel) if Path(reports_rel).is_absolute() else (art_dir / reports_rel))
    _ensure_dir(reports_dir)

    report_filename              = _opt(eval_cfg, ["report_filename"], "eval_report.json")
    preds_filename_pattern       = _opt(eval_cfg, ["preds_filename_pattern"], "preds_H{H}.csv")
    per_station_filename_pattern = _opt(eval_cfg, ["per_station_filename_pattern"], "per_station_H{H}.csv")

    report = {"overall": {}, "per_station": {}, "horizons": [int(h) for h in horizons]}

    # YAML model block (may be overridden by train-time params)
    mh = _req(cfg, ["model", "hybrid"])
    # base knobs (kept for rebuild if meta missing)
    yaml_defaults = dict(
        cnn_channels    = list(_req(mh, ["cnn_channels"])),
        cnn_kernels     = list(_req(mh, ["cnn_kernels"])),
        cnn_dropout     = float(_req(mh, ["cnn_dropout"])),
        d_model         = int(_req(mh, ["d_model"])),
        nhead           = int(_req(mh, ["nhead"])),
        num_layers      = int(_req(mh, ["num_layers"])),
        ff_mult         = int(_req(mh, ["ff_mult"])),
        attn_dropout    = float(_req(mh, ["attn_dropout"])),
        ffn_dropout     = float(_req(mh, ["ffn_dropout"])),
        pool            = str(_req(mh, ["pool"])),
        posenc          = str(_req(mh, ["posenc"])),
        ln_eps          = float(_opt(mh, ["ln_eps"], 1e-6)),
        station_embed_dim = int(_opt(mh, ["station_embed_dim"], 0)),
        use_station_embedding = bool(_opt(mh, ["use_station_embedding"], _opt(mh, ["station_embed_dim"], 0) or 0) ),
        n_stations      = _opt(mh, ["n_stations"], None),
        use_lstm        = bool(_opt(mh, ["use_lstm"], False)),
        lstm_hidden     = _opt(mh, ["lstm_hidden"], None),
        lstm_layers     = int(_opt(mh, ["lstm_layers"], 1)),
        lstm_dropout    = float(_opt(mh, ["lstm_dropout"], 0.0)),
        impl            = str(_opt(mh, ["impl"], "legacy_hybrid_encoder")),
    )

    print()
    for H in horizons:
        print(f"[eval] Horizon H={H}")

        ckpt = _find_checkpoint(model_root, int(H))
        if ckpt is None:
            print(f"[warn] no checkpoint under {model_root}; skipping H={H}")
            report["overall"][str(H)] = {m: float("nan") for m in ("rmse","mae","smape","r2")}
            report["per_station"][str(H)] = {}
            continue

        # prefer NPZ shards built during modelprep
        npz_dir = (seq_out_dir / "test" / f"h={int(H)}")
        shards = sorted(npz_dir.glob("shard_*.npz"))

        X = y = stations = None
        stations_have_ids = False

        if shards:
            X_list, y_list, sid_list = [], [], []
            for s in shards:
                with np.load(s, allow_pickle=False) as z:
                    X_list.append(z["X"].astype(np.float32, copy=False))
                    y_arr = z["y"]; y_list.append(y_arr if y_arr.ndim == 1 else y_arr.squeeze(-1))
                    if "sid" in z.files:
                        sid = z["sid"]
                        # normalize to ints where possible
                        if sid.dtype.kind in "i":
                            sid_list.extend([int(x) for x in sid.tolist()])
                        else:
                            sid_list.extend([int(str(x)) if str(x).isdigit() else str(x) for x in sid.tolist()])
            X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, lookback, len(X_cols)), np.float32)
            y = np.concatenate(y_list, axis=0) if y_list else np.empty((0,), np.float32)
            stations = sid_list if sid_list else []
            stations_have_ids = bool(stations)
            print(f"[windows] loaded from shards: {len(X)} @ {npz_dir} "
                  f"{'(with station_id)' if stations_have_ids else '(no station_id)'}")
        else:
            nan_fill_value = _opt(seq_cfg, ["nan_fill_value"], None)
            X, y, stations = _make_windows_per_station(
                te_df, station_col, time_col, X_cols, y_col,
                lookback=lookback, horizon=int(H), stride=stride,
                min_valid_ratio=(float(min_valid_eval) if min_valid_eval is not None else None),
                nan_fill_value=(float(nan_fill_value) if nan_fill_value is not None else None),
            )
            stations_have_ids = len(stations) > 0
            print(f"[windows] rebuilt test windows: {len(X)}  (stations: {len(set(stations))})")
            if len(X) == 0:
                print(f"[windows] no windows and no shards at {npz_dir}")
                report["overall"][str(H)] = {m: float("nan") for m in ("rmse","mae","smape","r2")}
                report["per_station"][str(H)] = {}
                continue

        in_dim = int(X.shape[2]) if (isinstance(X, np.ndarray) and X.ndim == 3) else int(len(X_cols))
        print(f"[eval] Effective input channels (C) = {in_dim}")

        # train-time params (if present)
        train_time = _load_train_time_params(model_root, int(H))
        mh_effective = dict(yaml_defaults)  # start with YAML
        if train_time:
            print(f"[eval] loaded train-time params from hybrid_h{H}_train_summary.json")
            # accept top-level fields and nested "model_hybrid_block"
            if isinstance(train_time, dict):
                for k, v in train_time.items():
                    if k in mh_effective and v is not None:
                        mh_effective[k] = v
                if isinstance(train_time.get("model_hybrid_block"), dict):
                    for k, v in train_time["model_hybrid_block"].items():
                        mh_effective[k] = v

        # station embedding resolution
        eval_station_embed_dim = int(mh_effective.get("station_embed_dim", 0) or 0)
        eval_use_station = bool(
            (mh_effective.get("use_station_embedding", False) or eval_station_embed_dim > 0)
            and stations_have_ids
        )
        eval_n_stations = mh_effective.get("n_stations", None)
        if eval_use_station and eval_n_stations is None and stations_have_ids:
            try:
                sid_numeric = [int(s) for s in stations]
                max_sid = max(sid_numeric) if len(sid_numeric) > 0 else -1
                eval_n_stations = int(max_sid + 1) if max_sid >= 0 else None
            except Exception:
                eval_n_stations = None

        sid_idx_np: np.ndarray | None = None
        if eval_use_station and stations_have_ids and eval_n_stations is not None:
            sid_idx_np = np.asarray([int(s) for s in stations], dtype=np.int64)
        else:
            eval_use_station = False
            eval_station_embed_dim = 0
            eval_n_stations = None
            sid_idx_np = None

        # -------- build model via factory (select implementation) --------
        model = build_hybrid(
            mh_effective,
            input_dim=in_dim,
            has_sid=eval_use_station,
            n_stations=eval_n_stations,
            horizon=int(H),
        ).to(device)
        model.eval()

        # load checkpoint
        print(f"[load] {ckpt}")
        state = torch.load(ckpt, map_location=device)
        state_dict = state["state_dict"] if (isinstance(state, dict) and "state_dict" in state) else state
        try:
            model.load_state_dict(state_dict, strict=True)
            print("[load] strict load OK")
        except RuntimeError as e_strict:
            print(f"[load/warn] strict load failed: {e_strict}")
            drop_prefixes = tuple([]) if eval_use_station else ("sid_emb",)
            ok, mm, miss = load_state_dict_forgiving(model, state_dict, drop_prefixes=drop_prefixes)
            print(f"[load/info] forgiving load: ok={len(ok)}  partial/mismatch={len(mm)}  missing={len(miss)}")
            if len(ok) == 0:
                raise RuntimeError("Non-strict load failed even after forgiving adapter.")

        # predict (AMP-enabled)
        preds = _predict(
            model, X, device, batch, sid_idx_np if eval_use_station else None, sanitize_cfg,
            amp_enabled=amp_enabled, amp_device=amp_device, amp_dtype=amp_dtype, horizon=int(H)
        )

        # metrics
        mask = np.isfinite(preds) & np.isfinite(y)
        n_all = len(y); n_ok = int(mask.sum()); n_drop = n_all - n_ok
        if n_drop > 0:
            print(f"[metrics] dropping {n_drop}/{n_all} non-finite pairs before scoring")
        yy = y[mask]; pp = preds[mask]
        overall = {k: METRICS[k](yy, pp) for k in METRICS} if n_ok > 0 else {k: float("nan") for k in METRICS}
        report["overall"][str(H)] = overall
        print(f"[overall@H={H}] " + "  ".join(f"{k}={v:.4f}" for k, v in overall.items()))

        # per-station
        if want_per_station and stations_have_ids and len(stations) == len(y):
            ps: Dict[str, Dict[str, float]] = {}
            stations_np = np.asarray(stations)
            st_mask_all = stations_np[mask]
            for sid in np.unique(st_mask_all):
                m = (st_mask_all == sid)
                if m.any():
                    ps[str(sid)] = {k: METRICS[k](yy[m], pp[m]) for k in METRICS}
            report["per_station"][str(H)] = ps
            print(f"[per-station@H={H}] computed for {len(ps)} station(s)")

            df_ps = (pd.DataFrame.from_dict(ps, orient="index")
                     .rename_axis("station_id").reset_index())
            out_ps = reports_dir / per_station_filename_pattern.format(H=int(H))
            df_ps.to_csv(out_ps, index=False)
            print(f"[save] per-station metrics → {out_ps} ({len(df_ps)} rows)")
        else:
            if want_per_station and not stations_have_ids:
                print("[per-station] skipped (no station ids)")
            elif want_per_station and len(stations or []) != len(y):
                print("[per-station] skipped (station ids length mismatch)")
            report["per_station"][str(H)] = {}

        if save_preds:
            out_csv = reports_dir / preds_filename_pattern.format(H=int(H))
            df_out = pd.DataFrame({"y_true": y.astype(float), "y_pred": preds.astype(float)})
            if stations_have_ids:
                df_out.insert(0, "station_id", stations)
            df_out.to_csv(out_csv, index=False)
            print(f"[save] wrote {out_csv}  ({len(df_out)} rows)")

    # write report JSON
    out_path = reports_dir / report_filename
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[OK] wrote {out_path}")
    for H in horizons:
        m = report["overall"].get(str(H), {})
        if m:
            print(f"[summary@H={H}] " + "  ".join(f"{k}={m.get(k, np.nan):.4f}" for k in ("rmse","mae","smape","r2")))

if __name__ == "__main__":
    main()