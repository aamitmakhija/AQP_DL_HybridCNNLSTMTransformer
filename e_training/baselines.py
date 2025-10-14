# e_training/baselines.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Tuple
import os, json, time
from copy import deepcopy

import yaml
import numpy as np
import pandas as pd

# ---------------- config helpers ----------------
def _deep_update(dst: Dict, src: Dict) -> Dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> Dict[str, Any]:
    # Support CONFIG="a.yaml,b.yaml"
    base = yaml.safe_load(open("configs/default.yaml")) or {}
    cfg_env = os.environ.get("CONFIG", "").strip()
    if not cfg_env:
        return base
    merged = deepcopy(base)
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        overlay = yaml.safe_load(open(p)) or {}
        _deep_update(merged, overlay)
    return merged

def _require(cfg: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing config key: {'.'.join(path)}")
        cur = cur[k]
    return cur

# ---------------- metric helpers ----------------
def _metrics() -> Dict[str, Any]:
    # Use your existing metrics registry
    from e_training.metrics import METRICS
    return METRICS

# ---------------- data io ----------------
def _resolve_paths(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    art = Path(_require(cfg, ["paths", "artifacts_dir"]))
    fs_rel = _require(cfg, ["paths", "features_scaled_dir"])
    fs_dir = Path(fs_rel) if Path(fs_rel).is_absolute() else (art / fs_rel)
    return art, fs_dir

def _read_split(fs_dir: Path, name: str, id_col: str, tcol: str, ycol: str) -> pd.DataFrame:
    p = fs_dir / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run scaling first.")
    df = pd.read_parquet(p, columns=[id_col, tcol, ycol])
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df[id_col] = df[id_col].astype(str)
    return df.sort_values([id_col, tcol]).reset_index(drop=True)


def mk_lags(df: pd.DataFrame, id_col: str, ycol: str, L: int) -> pd.DataFrame:
    """
    Build lag features 1..L for ycol grouped by id_col.
    Returns a new DataFrame with lag columns.
    """
    out = [df]  # keep original columns
    for lag in range(1, L + 1):
        out.append(df.groupby(id_col, group_keys=False)[ycol].shift(lag).rename(f"{ycol}_lag{lag}"))
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
    out_pers: Dict[str, Dict[str, float] | Dict[str, str]] = {}
    out_seas: Dict[str, Dict[str, float] | Dict[str, str]] = {}

    for H in horizons:
        # Keep *separate* stacks for persistence and seasonal
        y_true_p, y_pred_p = [], []
        y_true_s, y_pred_s = [], []

        for _, g in df.groupby(id_col, sort=False):
            s = g[ycol].astype(float)

            tgt  = s.shift(-H)                 # forecast target at horizon H
            pers = s                           # persistence predictor
            seas = s.shift(seasonal_period)    # seasonal-naive predictor

            m_p = tgt.notna() & pers.notna()
            if m_p.any():
                y_true_p.append(tgt[m_p].to_numpy())
                y_pred_p.append(pers[m_p].to_numpy())

            m_s = tgt.notna() & seas.notna()
            if m_s.any():
                y_true_s.append(tgt[m_s].to_numpy())
                y_pred_s.append(seas[m_s].to_numpy())

        # Compute metrics for persistence
        if y_true_p:
            y_all_p = np.concatenate(y_true_p)
            y_hat_p = np.concatenate(y_pred_p)
            out_pers[str(H)] = {
                "rmse": float(mets["rmse"](y_all_p, y_hat_p)),
                "mae":  float(mets["mae"](y_all_p, y_hat_p)),
                "smape":float(mets["smape"](y_all_p, y_hat_p)),
                "r2":   float(mets["r2"](y_all_p, y_hat_p)),
            }
        else:
            out_pers[str(H)] = {"note": "no valid rows for persistence"}

        # Compute metrics for seasonal
        if y_true_s:
            y_all_s = np.concatenate(y_true_s)
            y_hat_s = np.concatenate(y_pred_s)
            out_seas[str(H)] = {
                "rmse": float(mets["rmse"](y_all_s, y_hat_s)),
                "mae":  float(mets["mae"](y_all_s, y_hat_s)),
                "smape":float(mets["smape"](y_all_s, y_hat_s)),
                "r2":   float(mets["r2"](y_all_s, y_hat_s)),
            }
        else:
            out_seas[str(H)] = {"note": "no valid rows for seasonal"}

    return {"persistence": out_pers, "seasonal": out_seas}

def _make_lags(df: pd.DataFrame, id_col: str, ycol: str, L: int) -> pd.DataFrame:
    """
    Add lag features 1..L for ycol grouped by id_col.
    Builds all lag columns efficiently to avoid fragmentation warnings.
    """
    groups = df.groupby(id_col, sort=False)[ycol]
    lagged = [
        groups.shift(h).rename(f"{ycol}_lag{h}")
        for h in range(1, L + 1)
    ]
    return pd.concat([df] + lagged, axis=1)

def _ridge_direct(
    trainval: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    ycol: str,
    H: int,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, str | None]:
    """
    Direct ridge baseline for horizon H using lags 1..lookback of ycol (per-station).
    Returns: (y_true, y_pred, err_msg_or_None)
    """
    try:
        from sklearn.linear_model import Ridge
    except Exception:
        return np.array([]), np.array([]), "sklearn_not_available"

    # Build lag features efficiently (your vectorized helper)
    trv = _make_lags(trainval, id_col, ycol, lookback).copy()
    te  = _make_lags(test_df,  id_col, ycol, lookback).copy()

    # Create direct target y_{t+H}
    trv["y_tgt"] = trv.groupby(id_col, sort=False)[ycol].shift(-H)
    te["y_tgt"]  = te .groupby(id_col, sort=False)[ycol].shift(-H)

    lag_cols = [f"{ycol}_lag{h}" for h in range(1, lookback + 1)]

    # Keep rows with complete features and target
    trv2 = trv.dropna(subset=lag_cols + ["y_tgt"])
    te2  = te .dropna(subset=lag_cols + ["y_tgt"])

    if trv2.empty or te2.empty:
        return np.array([]), np.array([]), "insufficient_rows"

    Xtr = trv2[lag_cols].to_numpy(dtype=float)
    ytr = trv2["y_tgt"].to_numpy(dtype=float)

    Xte = te2[lag_cols].to_numpy(dtype=float)
    yte = te2["y_tgt"].to_numpy(dtype=float)

    model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)

    return yte, yhat, None

# ---------------- main ----------------
def main():
    t0 = time.time()
    cfg = _load_cfg()

    # Paths and schema
    art, fs_dir = _resolve_paths(cfg)
    id_col = _require(cfg, ["data", "id_col"])
    tcol   = _require(cfg, ["data", "time_col"])
    ycol   = _require(cfg, ["data", "target"])

    seq = _require(cfg, ["sequence"])
    # horizons can be list under 'horizon' or 'horizons'
    if "horizons" in seq:
        horizons = [int(h) for h in seq["horizons"]]
    else:
        raw_h = seq.get("horizon", [1])
        horizons = [int(h) for h in (raw_h if isinstance(raw_h, (list, tuple)) else [raw_h])]
    lookback = int(_require(seq, ["lookback"]))

    seasonal_period = int(cfg.get("baselines", {}).get("seasonal_period", 168))  # configurable; default weekly (24*7)

    tr_name = _require(seq, ["train_split"])
    va_name = _require(seq, ["val_split"])
    te_name = seq.get("test_split", "test")

    # Load
    train = _read_split(fs_dir, tr_name, id_col, tcol, ycol)
    val   = _read_split(fs_dir, va_name, id_col, tcol, ycol)
    test  = _read_split(fs_dir, te_name, id_col, tcol, ycol)

    # Persistence & Seasonal on TEST
    ps = _persistence_and_seasonal(test, id_col, ycol, horizons, seasonal_period)

    # Ridge on TRAIN+VAL -> TEST for each H
    mets = _metrics()
    res_ridge: Dict[str, Dict[str, float] | Dict[str, str]] = {}
    trainval = pd.concat([train, val], axis=0, ignore_index=True)

    for H in horizons:
        yref, yhat, err = _ridge_direct(trainval, test, id_col, ycol, H, lookback)
        if err or yref.size == 0:
            res_ridge[str(H)] = {"note": err or "empty"}  # type: ignore[assignment]
        else:
            res_ridge[str(H)] = {
                "rmse": float(mets["rmse"](yref, yhat)),
                "mae":  float(mets["mae"](yref, yhat)),
                "smape":float(mets["smape"](yref, yhat)),
                "r2":   float(mets["r2"](yref, yhat)),
            }

    out = {
        "config": {
            "horizons": horizons,
            "lookback": lookback,
            "seasonal_period": seasonal_period,
            "id_col": id_col,
            "time_col": tcol,
            "target_col": ycol,
            "features_scaled_dir": str(fs_dir),
        },
        "persistence": ps["persistence"],
        "seasonal168": ps["seasonal"],   # name kept for continuity; period is configurable
        "ridge": res_ridge,
        "runtime_sec": round(time.time() - t0, 3),
    }

    out_path = art / "models" / "baselines.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[OK] baselines → {out_path}  (took {out['runtime_sec']}s)")

if __name__ == "__main__":
    main()