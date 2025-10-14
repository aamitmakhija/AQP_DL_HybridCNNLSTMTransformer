# e_training/eval_report.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, List, Tuple
from copy import deepcopy

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# unified metrics (single source of truth)
from e_training.metrics import METRICS
# reuse the SAME model class as training
from e_training.train_hybrid import HybridEncoder


# ---------------- config helpers ----------------
def _deep_update(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> Dict:
    """
    Load configs/default.yaml, then merge any overlays in CONFIG env (comma-separated).
    Mirrors other scripts so eval and train stay consistent.
    """
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

def _resolve_device(pref: str) -> str:
    pref = str(pref).lower()
    if pref == "auto":
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return pref


# ---------------- feature lock ----------------
def _load_feature_lock(art_dir: Path) -> Tuple[str, str | None, List[str], List[str]]:
    lock_path = art_dir / "features_locked" / "feature_list.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"{lock_path} missing — run scaling/lock step first")
    lock = json.loads(lock_path.read_text())
    for k in ["time_col", "id_cols", "X_cols_ordered"]:
        if k not in lock:
            raise KeyError(f"feature_list.json missing '{k}'")
    time_col   = lock["time_col"]
    target_col = lock.get("target_col")
    id_cols    = lock["id_cols"]
    X_cols     = lock["X_cols_ordered"]
    return time_col, target_col, id_cols, X_cols


# ---------------- io helpers ----------------
def _ordered_unique(xs: List[str]) -> List[str]:
    seen = set(); out = []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _read_split(base: Path, name: str, needed: set[str], time_col: str) -> pd.DataFrame:
    p = base / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run dataprep/scale_features.py first")
    # Column-pruned read; also guard if time_col duplicated
    df = pd.read_parquet(p, columns=None)
    wanted = _ordered_unique([c for c in df.columns if c in (needed | {time_col})])
    df = df[wanted]
    if (df.columns == time_col).sum() > 1:
        first = df.loc[:, df.columns == time_col].iloc[:, 0]
        df = df.drop(columns=[time_col])
        df[time_col] = first
    return df


# ---------------- feature sanitization (mirror training/windowing) ----------------
def _sanitize_X_cols(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    station_col: str,
    time_col: str,
    y_col: str,
    X_cols_lock: List[str],
) -> List[str]:
    non_feature = {station_col, time_col, y_col}
    # keep only those present in all splits and not id/time/target
    x = [c for c in X_cols_lock if c not in non_feature
         and c in tr_df.columns and c in va_df.columns and c in te_df.columns]
    # keep only numeric (using train dtypes; coercion will handle remainder at read-time)
    x = [c for c in x if pd.api.types.is_numeric_dtype(tr_df[c])]
    if not x:
        raise RuntimeError("After sanitizing, X_cols is empty. Check your features/scaling step.")
    return x


# ---------------- windowing on test ----------------
def _make_windows_per_station(
    df: pd.DataFrame,
    station_col: str,
    time_col: str,
    X_cols: List[str],
    y_col: str,
    lookback: int,
    horizon: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    all_X, all_y, stations = [], [], []
    for sid, g in df.groupby(station_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)

        # numeric & non-finite guard
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
            # strict: drop any non-finite values
            if not (np.isfinite(x_seq).all() and np.isfinite(y_val)):
                continue
            all_X.append(x_seq)
            all_y.append(y_val)
            stations.append(str(sid))
    if not all_X:
        return np.empty((0, lookback, len(X_cols))), np.empty((0,), float), []
    return np.stack(all_X, axis=0), np.asarray(all_y, float), stations


# ---------------- diagnostics ----------------
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
    print(f"[{name}] NaN rate: y]={y_nan:.3f}  X(sample)={{{', '.join(f'{k}:{v:.3f}' for k,v in x_nan_map.items())}}}")
    for H in horizons:
        ub = _upper_bound_windows(df, station_col, time_col, lookback, int(H), stride)
        print(f"[{name}] Upper-bound windows (no NaN) H={H}: ~{ub:,}")


# ---------------- checkpoint finder ----------------
def _find_checkpoint(model_root: Path, H: int) -> Path | None:
    exact = model_root / f"hybrid_h{int(H)}.pt"
    if exact.exists():
        return exact
    cands = list(model_root.rglob("*.pt"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

@torch.no_grad()
def _predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """
    Safe prediction: sanitize batches (replace NaN/Inf, clamp) to avoid NaN explosions.
    """
    preds: List[np.ndarray] = []
    N = len(X)
    bad_total = 0
    for i in range(0, N, batch_size):
        xb_np = X[i:i+batch_size]
        # Count non-finite before sanitizing (for debug)
        bad = np.size(xb_np) - np.isfinite(xb_np).sum()
        bad_total += int(bad)
        # Replace non-finite and clamp
        xb_np = np.nan_to_num(xb_np, copy=False)     # NaN->0, ±Inf->±large finite
        np.clip(xb_np, -1e6, 1e6, out=xb_np)         # wide clamp just in case
    
        xb = torch.from_numpy(xb_np).to(device, dtype=torch.float32)
        pb = model(xb).detach().cpu().numpy()
        preds.append(pb)
    if bad_total > 0:
        print(f"[predict] sanitized {bad_total} non-finite value(s) across all batches")
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), float)


# ---------------- main ----------------
def main():
    cfg = _load_cfg()

    # paths & device
    art_dir   = Path(_require(cfg, ["paths", "artifacts_dir"]))
    dl_cfg    = _require(cfg, ["dl"])
    device    = torch.device(_resolve_device(dl_cfg.get("device", "auto")))
    batch     = int(dl_cfg.get("batch_size", 64))

    seq_cfg   = _require(cfg, ["sequence"])
    lookback  = int(_require(seq_cfg, ["lookback"]))
    horizon_v = _require(seq_cfg, ["horizon"])
    horizons  = list(horizon_v) if isinstance(horizon_v, (list, tuple)) else [int(horizon_v)]
    stride    = int(_require(seq_cfg, ["stride"]))
    seq_out_dir = _require(seq_cfg, ["out_dir"])

    model_dir_cfg = dl_cfg.get("model_dir", "models")
    model_root = Path(model_dir_cfg) if Path(model_dir_cfg).is_absolute() else (art_dir / model_dir_cfg)
    model_root.mkdir(parents=True, exist_ok=True)

    # locked features
    time_col, target_lock, id_cols_lock, X_cols_lock = _load_feature_lock(art_dir)
    data_cfg    = _require(cfg, ["data"])
    station_col = data_cfg["id_col"]
    y_col       = target_lock if target_lock else data_cfg["target"]

    feats_rel = _require(cfg, ["paths", "features_scaled_dir"])
    feats_dir = Path(feats_rel) if Path(feats_rel).is_absolute() else (art_dir / feats_rel)

    # Read splits for sanitization parity with training
    needed = set([station_col, time_col, y_col] + X_cols_lock)
    tr_df = _read_split(feats_dir, _require(seq_cfg, ["train_split"]), needed, time_col)
    va_df = _read_split(feats_dir, _require(seq_cfg, ["val_split"]),   needed, time_col)
    te_df = _read_split(feats_dir, _require(seq_cfg, ["test_split"]),  needed, time_col)

    # Ensure types like training
    for df in (tr_df, va_df, te_df):
        if station_col not in df.columns:
            raise KeyError(f"'{station_col}' not found in split; check data.id_col and feature lock.")
        df[station_col] = df[station_col].astype(str)
        df[time_col]    = pd.to_datetime(df[time_col], errors="coerce")

    # Sanitize X cols to match windowing/training
    X_cols = _sanitize_X_cols(tr_df, va_df, te_df, station_col, time_col, y_col, X_cols_lock)
    print(f"\n[eval] Using {len(X_cols)} feature(s): {X_cols}\n")

    # Diagnostics on test
    _print_split_diag("test", te_df, station_col, time_col, y_col, X_cols, lookback, horizons, stride)

    eval_cfg         = _require(cfg, ["eval"])
    want_per_station = bool(eval_cfg.get("per_station", True))
    save_preds       = bool(eval_cfg.get("save_preds", False))

    reports_dir = art_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {"overall": {}, "per_station": {}, "horizons": [int(h) for h in horizons]}

    # --- model hyperparams ONLY from YAML (keep parity with training) ---
    mh = _require(cfg, ["model", "hybrid"])
    cnn_channels = list(_require(mh, ["cnn_channels"]))
    cnn_kernels  = list(_require(mh, ["cnn_kernels"]))
    cnn_dropout  = float(_require(mh, ["cnn_dropout"]))
    d_model      = int(_require(mh, ["d_model"]))
    nhead        = int(_require(mh, ["nhead"]))
    num_layers   = int(_require(mh, ["num_layers"]))
    ff_mult      = int(_require(mh, ["ff_mult"]))
    attn_dropout = float(_require(mh, ["attn_dropout"]))
    ffn_dropout  = float(_require(mh, ["ffn_dropout"]))
    pool         = str(mh.get("pool", "gap"))
    posenc       = str(mh.get("posenc", "sin"))
    ln_eps       = float(mh.get("ln_eps", 1e-5))  # optional stabilization, present if added in training

    print()
    for H in horizons:
        print(f"[eval] Horizon H={H}")

        # Build model with input_dim=len(X_cols) (must match training input feature count)
        try:
            model = HybridEncoder(
                in_dim=len(X_cols),
                cnn_channels=cnn_channels,
                cnn_kernels=cnn_kernels,
                cnn_dropout=cnn_dropout,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                pool=pool,
                posenc=posenc,
                ln_eps=ln_eps,  # only effective if HybridEncoder supports it
            ).to(device)
        except TypeError:
            # If training HybridEncoder has no ln_eps arg, rebuild without it
            model = HybridEncoder(
                in_dim=len(X_cols),
                cnn_channels=cnn_channels,
                cnn_kernels=cnn_kernels,
                cnn_dropout=cnn_dropout,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                pool=pool,
                posenc=posenc,
            ).to(device)
        model.eval()

        ckpt = _find_checkpoint(model_root, int(H))
        if ckpt is None:
            print(f"[warn] no checkpoint found under {model_root}; skipping H={H}")
            report["overall"][str(H)] = {m: float("nan") for m in ("rmse","mae","smape","r2")}
            report["per_station"][str(H)] = {}
            continue

        print(f"[load] {ckpt}")
        state = torch.load(ckpt, map_location=device)
        # support raw state_dict or {"state_dict":...}
        state_dict = state["state_dict"] if (isinstance(state, dict) and "state_dict" in state) else state

        # strict load first; if shape mismatch (e.g., different in_dim), print warning and retry non-strict
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"[load/warn] strict load failed: {e}")
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
            except Exception as e2:
                raise RuntimeError(f"Non-strict load also failed: {e2}") from e2
            if missing or unexpected:
                print(f"[load/info] missing={missing}  unexpected={unexpected}")

        # Rebuild test windows for this horizon (primary path)
        X, y, stations = _make_windows_per_station(
            te_df, station_col, time_col, X_cols, y_col,
            lookback=lookback, horizon=int(H), stride=stride
        )
        print(f"[windows] rebuilt test windows: {len(X)}  (stations: {len(set(stations))})")

        # Fallback: use prebuilt NPZ shards if rebuild yields 0
        used_fallback = False
        if len(X) == 0:
            npz_dir = (art_dir / seq_out_dir) / "test" / f"h={int(H)}"
            shards = sorted(npz_dir.glob("shard_*.npz"))
            if shards:
                X_list, y_list = [], []
                for s in shards:
                    with np.load(s, allow_pickle=False) as z:
                        X_list.append(z["X"].astype(np.float32, copy=False))
                        y_arr = z["y"]
                        y_list.append(y_arr if y_arr.ndim == 1 else y_arr.squeeze(-1))
                X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, lookback, len(X_cols)), np.float32)
                y = np.concatenate(y_list, axis=0) if y_list else np.empty((0,), np.float32)
                stations = []  # shards don’t carry station ids
                used_fallback = True
                print(f"[windows/fallback] loaded from shards: {len(X)} windows @ {npz_dir}")
            else:
                print(f"[windows] no windows and no shards at {npz_dir}")
                report["overall"][str(H)] = {m: float("nan") for m in ("rmse","mae","smape","r2")}
                report["per_station"][str(H)] = {}
                continue

        preds = _predict(model, X, device, batch)

        # Finite-mask both arrays for metrics
        mask = np.isfinite(preds) & np.isfinite(y)
        n_all = len(y)
        n_ok = int(mask.sum())
        n_drop = n_all - n_ok
        if n_drop > 0:
            print(f"[metrics] dropping {n_drop}/{n_all} non-finite pairs before scoring")
        yy = y[mask]
        pp = preds[mask]

        overall = {k: METRICS[k](yy, pp) for k in METRICS} if n_ok > 0 else {k: float("nan") for k in METRICS}
        report["overall"][str(H)] = overall
        print(f"[overall@H={H}] " + "  ".join(f"{k}={v:.4f}" for k, v in overall.items()))

        # Per-station metrics only if we have station ids (not available in shard fallback)
        if want_per_station and not used_fallback:
            ps: Dict[str, Dict[str, float]] = {}
            y_np, p_np = np.asarray(y), np.asarray(preds)
            stations_np = np.asarray(stations)
            for sid in np.unique(stations_np):
                m = (stations_np == sid)
                m = m & np.isfinite(y_np) & np.isfinite(p_np)
                if m.any():
                    ps[str(sid)] = {k: METRICS[k](y_np[m], p_np[m]) for k in METRICS}
            report["per_station"][str(H)] = ps
            print(f"[per-station@H={H}] computed for {len(ps)} station(s)")
        elif want_per_station and used_fallback:
            report["per_station"][str(H)] = {}
            print("[per-station] skipped (shard fallback has no station ids)")

        if save_preds:
            out_csv = reports_dir / f"preds_H{int(H)}.csv"
            df_out = pd.DataFrame({"y_true": y.astype(float), "y_pred": preds.astype(float)})
            if stations:
                df_out.insert(0, "station_id", stations)
            df_out.to_csv(out_csv, index=False)
            print(f"[save] wrote {out_csv}  ({len(df_out)} rows)")

    out_path = reports_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[OK] wrote {out_path}")
    for H in horizons:
        m = report["overall"].get(str(H), {})
        if m:
            print(f"[summary@H={H}] " + "  ".join(f"{k}={m.get(k, np.nan):.4f}" for k in ("rmse","mae","smape","r2")))


if __name__ == "__main__":
    main()