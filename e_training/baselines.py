# e_training/baselines.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import os, json, time
from copy import deepcopy

import yaml
import numpy as np
import pandas as pd

# ---------------- config helpers ----------------
def _deep_update(dst: Dict, src: Dict) -> Dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> Dict[str, Any]:
    base = yaml.safe_load(open("configs/default.yaml")) or {}
    cfg_env = os.environ.get("CONFIG", "").strip()
    if not cfg_env:
        return base
    merged = deepcopy(base)
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        ov = yaml.safe_load(open(p)) or {}
        if isinstance(ov, dict):
            _deep_update(merged, ov)
    return merged

def _require(cfg: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing config key: {'.'.join(path)}")
        cur = cur[k]
    return cur

def _opt(cfg: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# ---------------- metric helpers ----------------
def _metrics() -> Dict[str, Any]:
    from e_training.metrics import METRICS
    return METRICS

# ---------------- paths & formats ----------------
def _resolve_paths_and_fmt(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path, str, Dict[str, str]]:
    art = Path(_require(cfg, ["paths", "artifacts_dir"]))
    # features_scaled_dir
    fs_rel = _require(cfg, ["paths", "features_scaled_dir"])
    fs_dir = Path(fs_rel) if Path(fs_rel).is_absolute() else (art / fs_rel)
    # reports_dir (fallback to artifacts_dir/reports)
    reports_rel = _opt(cfg, ["paths", "reports_dir"], None)
    reports_dir = (Path(reports_rel) if (reports_rel and Path(reports_rel).is_absolute())
                   else (art / (reports_rel or "reports")))

    split_names: Dict[str, str] = (cfg.get("output", {}) or {}).get("split_filenames", {}) or {}
    out_fmt: str = (cfg.get("scaling", {}) or {}).get(
        "output_format",
        (cfg.get("output", {}) or {}).get("format", "parquet"),
    )
    out_fmt = (out_fmt or "parquet").lower()
    if out_fmt not in {"parquet", "feather", "csv"}:
        out_fmt = "parquet"
    return art, fs_dir, reports_dir, out_fmt, split_names

def _split_path(fs_dir: Path, name: str, split_names: Dict[str, str], fmt: str) -> Path:
    fname = split_names.get(name, f"{name}.{fmt}")
    p = fs_dir / fname
    return p if p.suffix else p.with_suffix("." + fmt)

# ---------------- data io ----------------
def _read_split_generic(path: Path, fmt: str, id_col: str, tcol: str, ycol: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scaling first.")
    want = {id_col, tcol, ycol}
    if fmt == "parquet":
        try:
            import pyarrow.parquet as pq
            schema_cols = [c for c in pq.read_schema(str(path)).names if c in want]
        except Exception:
            schema_cols = list(want)
        df = pd.read_parquet(path, columns=schema_cols)
    elif fmt == "feather":
        df = pd.read_feather(path)
        df = df[[c for c in df.columns if c in want]]
    elif fmt == "csv":
        df = pd.read_csv(path, usecols=lambda c: c in want)
    else:
        raise SystemExit(f"Unsupported scaled split format: {fmt}")

    missing = want - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}; available={sorted(df.columns)}")

    df[tcol] = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize(None)
    df[id_col] = df[id_col].astype(str)

    df = df.sort_values([id_col, tcol]).drop_duplicates([id_col, tcol], keep="last").reset_index(drop=True)
    return df

# ---------------- lag builder ----------------
def _make_lags(df: pd.DataFrame, id_col: str, ycol: str, L: int) -> pd.DataFrame:
    g = df.groupby(id_col, sort=False)[ycol]
    out = [df]
    for h in range(1, L + 1):
        out.append(g.shift(h).rename(f"{ycol}_lag{h}"))
    return pd.concat(out, axis=1)

# ---------------- baselines ----------------
def _persistence_and_seasonal(
    df: pd.DataFrame,
    id_col: str,
    ycol: str,
    horizons: List[int],
    seasonal_period: int,
) -> Dict[str, Dict[str, Dict[str, float] | Dict[str, str]]]:
    mets = _metrics()
    out_p, out_s = {}, {}
    for H in horizons:
        ytp, ypp, yts, yps = [], [], [], []
        for _, g in df.groupby(id_col, sort=False):
            s = g[ycol].to_numpy(dtype=np.float64, copy=False)
            if s.size == 0:
                continue
            tgt = np.roll(s, -H); tgt[-H:] = np.nan

            m = np.isfinite(tgt) & np.isfinite(s)
            if m.any():
                ytp.append(tgt[m]); ypp.append(s[m])

            seas = np.roll(s, seasonal_period)
            seas[:seasonal_period] = np.nan
            m2 = np.isfinite(tgt) & np.isfinite(seas)
            if m2.any():
                yts.append(tgt[m2]); yps.append(seas[m2])

        def pack(y_true, y_pred):
            if not y_true:
                return {"note": "no_valid_rows", "n": 0}
            yt = np.concatenate(y_true)
            yp = np.concatenate(y_pred)
            return {
                "rmse": float(mets["rmse"](yt, yp)),
                "mae":  float(mets["mae"](yt, yp)),
                "smape":float(mets["smape"](yt, yp)),
                "r2":   float(mets["r2"](yt, yp)),
                "n":    int(yt.size),
            }

        out_p[str(H)] = pack(ytp, ypp)
        out_s[str(H)] = pack(yts, yps)

    return {"persistence": out_p, "seasonal": out_s}

def _ridge_direct(
    trainval: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    ycol: str,
    H: int,
    lookback: int,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return np.array([]), np.array([]), "sklearn_not_available"

    trv = _make_lags(trainval, id_col, ycol, lookback)
    te  = _make_lags(test_df,  id_col, ycol, lookback)

    trv["y_tgt"] = trv.groupby(id_col, sort=False)[ycol].shift(-H)
    te["y_tgt"]  = te .groupby(id_col, sort=False)[ycol].shift(-H)

    lag_cols = [f"{ycol}_lag{h}" for h in range(1, lookback + 1)]
    needed = lag_cols + ["y_tgt"]

    trv2 = trv.dropna(subset=needed)
    te2  = te .dropna(subset=needed)
    if trv2.empty or te2.empty:
        return np.array([]), np.array([]), "insufficient_rows"

    Xtr = trv2[lag_cols].to_numpy(dtype=np.float64, copy=False)
    ytr = trv2["y_tgt"].to_numpy(dtype=np.float64, copy=False)
    Xte = te2[lag_cols].to_numpy(dtype=np.float64, copy=False)
    yte = te2["y_tgt"].to_numpy(dtype=np.float64, copy=False)

    ss = StandardScaler().fit(Xtr)
    Xtr = ss.transform(Xtr)
    Xte = ss.transform(Xte)

    model = Ridge(alpha=float(alpha))
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)

    return yte, yhat, None

# ---------------- main ----------------
def main():
    t0 = time.time()
    cfg = _load_cfg()

    # Paths, formats, names
    art, fs_dir, reports_dir, fmt, split_names = _resolve_paths_and_fmt(cfg)
    baselines_dir = reports_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    # Schema
    id_col = _require(cfg, ["data", "id_col"])
    tcol   = _require(cfg, ["data", "time_col"])
    ycol   = _require(cfg, ["data", "target"])

    # Sequence params
    seq = _require(cfg, ["sequence"])
    raw_h = seq.get("horizons", seq.get("horizon", [1]))
    horizons = [int(h) for h in (raw_h if isinstance(raw_h, (list, tuple)) else [raw_h])]
    lookback = int(_require(seq, ["lookback"]))

    # Baseline knobs
    seasonal_period = int(_opt(cfg, ["baselines", "seasonal_period"], 168))
    ridge_alpha     = float(_opt(cfg, ["baselines", "ridge_alpha"], 1.0))
    save_pred       = bool(_opt(cfg, ["baselines", "save_pred_samples"], False))

    # Splits
    tr_name = _require(seq, ["train_split"])
    va_name = _require(seq, ["val_split"])
    te_name = seq.get("test_split", "test")

    tr_path = _split_path(fs_dir, tr_name, split_names, fmt)
    va_path = _split_path(fs_dir, va_name, split_names, fmt)
    te_path = _split_path(fs_dir, te_name, split_names, fmt)

    train = _read_split_generic(tr_path, fmt, id_col, tcol, ycol)
    val   = _read_split_generic(va_path, fmt, id_col, tcol, ycol)
    test  = _read_split_generic(te_path, fmt, id_col, tcol, ycol)

    # Persistence & Seasonal (TEST)
    ps = _persistence_and_seasonal(test, id_col, ycol, horizons, seasonal_period)

    # Ridge (TRAIN+VAL -> TEST)
    mets = _metrics()
    res_ridge: Dict[str, Dict[str, float] | Dict[str, str]] = {}
    counts_ridge: Dict[str, int] = {}
    trainval = pd.concat([train, val], axis=0, ignore_index=True)

    if save_pred:
        (baselines_dir / "samples").mkdir(parents=True, exist_ok=True)

    for H in horizons:
        yref, yhat, err = _ridge_direct(trainval, test, id_col, ycol, H, lookback, alpha=ridge_alpha)
        if err or yref.size == 0:
            res_ridge[str(H)] = {"note": err or "empty", "n": 0}
            counts_ridge[str(H)] = 0
        else:
            res_ridge[str(H)] = {
                "rmse": float(mets["rmse"](yref, yhat)),
                "mae":  float(mets["mae"](yref, yhat)),
                "smape":float(mets["smape"](yref, yhat)),
                "r2":   float(mets["r2"](yref, yhat)),
                "n":    int(yref.size),
            }
            counts_ridge[str(H)] = int(yref.size)
            if save_pred:
                samp = min(5000, yref.size)
                dfp = pd.DataFrame({"y_true": yref[:samp], "y_pred": yhat[:samp]})
                dfp.to_csv(baselines_dir / "samples" / f"ridge_samples_H{H}.csv", index=False)

    # Pack & write
    out = {
        "config": {
            "horizons": horizons,
            "lookback": lookback,
            "seasonal_period": seasonal_period,
            "ridge_alpha": ridge_alpha,
            "id_col": id_col,
            "time_col": tcol,
            "target_col": ycol,
            "features_scaled_dir": str(fs_dir),
            "format": fmt,
            "split_names": split_names,
            "splits": {"train": str(tr_path), "val": str(va_path), "test": str(te_path)},
        },
        "persistence": ps["persistence"],
        "seasonal168": ps["seasonal"],
        "ridge": res_ridge,
        "runtime_sec": round(time.time() - t0, 3),
    }

    out_json = baselines_dir / "baselines.json"
    out_csv  = baselines_dir / "baselines_summary.csv"

    out_json.write_text(json.dumps(out, indent=2))
    # compact CSV summary for quick compare
    rows = []
    for name, block in (("persistence", out["persistence"]),
                        ("seasonal168", out["seasonal168"]),
                        ("ridge", out["ridge"])):
        for H, m in block.items():
            rows.append({"baseline": name, "H": int(H), **{k: v for k, v in m.items() if k != "note"}})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[OK] baselines → {out_json}  (took {out['runtime_sec']}s)")

if __name__ == "__main__":
    main()