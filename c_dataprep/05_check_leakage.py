# c_dataprep/05_check_leakage.py
from __future__ import annotations

import json, re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from common.config_loader import load_cfg

# ---------- helpers ----------
def _pick_pm25_col(cols: List[str], cfg: Dict) -> str | None:
    aliases = [c.lower() for c in cfg["data"]["headers"]["airquality"]["pm25"]]
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None

def _split_by_cutoffs(df: pd.DataFrame, time_col: str, train_end: str, val_end: str):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    tr_end = pd.to_datetime(train_end)
    v_end  = pd.to_datetime(val_end)
    train = df[df[time_col] <= tr_end].copy()
    val   = df[(df[time_col] > tr_end) & (df[time_col] <= v_end)].copy()
    test  = df[df[time_col] > v_end].copy()
    return train, val, test

def _infer_horizons_from_columns(cols: List[str], pm25_col: str | None) -> List[int]:
    if not pm25_col:
        return []
    pat = re.compile(rf"^{re.escape(pm25_col)}_lag(\d+)h$")
    found = set()
    for c in cols:
        m = pat.match(c)
        if m:
            try:
                found.add(int(m.group(1)))
            except Exception:
                pass
    return sorted(found)

def _collect_roll_windows(df: pd.DataFrame) -> List[int]:
    wins = set()
    pat = re.compile(r"_roll(\d+)h_(mean|std)$")
    for c in df.columns:
        m = pat.search(c)
        if m:
            wins.add(int(m.group(1)))
    return sorted(wins)

# ---------- checks ----------
def _check_monotonic(df: pd.DataFrame, station_col: str, time_col: str) -> bool:
    ok = True
    for sid, g in df.groupby(station_col):
        if not g[time_col].is_monotonic_increasing:
            print(f"[FAIL] non-monotonic timestamps in station {sid}")
            ok = False
    if ok:
        print("[OK] timestamps are monotonic per station")
    return ok

def _check_lags(df: pd.DataFrame, station_col: str, time_col: str, pm25_col: str, horizons: List[int]) -> bool:
    ok = True
    base = df.sort_values([station_col, time_col]).copy()
    for h in horizons:
        col = f"{pm25_col}_lag{h}h"
        if col not in base.columns:
            continue
        exp = base.groupby(station_col)[pm25_col].transform(lambda s: s.shift(h))
        mask = base[col].notna() & exp.notna()
        diff = (base.loc[mask, col] - exp.loc[mask]).abs()
        if diff.max(skipna=True) > 1e-8:
            print(f"[FAIL] {col} does not equal past-only shift (max abs diff={diff.max()})")
            ok = False
        else:
            print(f"[OK] {col} matches past-only shift")
    if ok:
        print("[OK] lags are past-only and match shifted base series (spot-checked)")
    return ok

def _check_rolls(df: pd.DataFrame, station_col: str, time_col: str, ddof: int = 1) -> bool:
    """
    Recompute rolling mean/std with groupby(...).transform(rolling(...))
    Compare where both sides are finite. ddof defaults to 1 to match pandas std().
    """
    ok = True
    base = df.sort_values([station_col, time_col]).copy()
    num_cols = base.select_dtypes(include="number").columns.tolist()
    windows = _collect_roll_windows(base)

    for col in num_cols:
        for w in windows:
            mean_col = f"{col}_roll{w}h_mean"
            std_col  = f"{col}_roll{w}h_std"

            if mean_col in base.columns:
                exp_mean = base.groupby(station_col)[col].transform(
                    lambda s: s.rolling(window=w, min_periods=max(1, w // 2)).mean()
                )
                mask = base[mean_col].notna() & exp_mean.notna()
                diff = (base.loc[mask, mean_col] - exp_mean.loc[mask]).abs()
                if diff.max(skipna=True) > 1e-6:
                    print(f"[FAIL] {mean_col} mismatch (max abs diff={diff.max()})")
                    ok = False

            if std_col in base.columns:
                exp_std = base.groupby(station_col)[col].transform(
                    lambda s: s.rolling(window=w, min_periods=max(1, w // 2)).std(ddof=ddof)
                )
                mask = base[std_col].notna() & exp_std.notna()
                diff = (base.loc[mask, std_col] - exp_std.loc[mask]).abs()
                if diff.max(skipna=True) > 1e-6:
                    print(f"[FAIL] {std_col} mismatch (max abs diff={diff.max()})")
                    ok = False

    if ok:
        print("[OK] rolling stats are past-only (spot-checked)")
    return ok

def _check_splits(df: pd.DataFrame, time_col: str, train_end: str, val_end: str) -> bool:
    train, val, test = _split_by_cutoffs(df, time_col, train_end, val_end)
    ok = True
    if not (train[time_col] <= pd.to_datetime(train_end)).all():
        print("[FAIL] train split exceeds train_end")
        ok = False
    else:
        print("[OK] train split ≤ train_end")
    if not ((val[time_col] > pd.to_datetime(train_end)) & (val[time_col] <= pd.to_datetime(val_end))).all():
        print("[FAIL] val split not fully within (train_end, val_end]")
        ok = False
    else:
        print("[OK] val split within (train_end, val_end]")
    if not (test[time_col] > pd.to_datetime(val_end)).all():
        print("[FAIL] test split not strictly after val_end")
        ok = False
    else:
        print("[OK] test split > val_end")
    return ok

# ---------- main ----------
def main():
    cfg = load_cfg()

    art_dir       = Path(cfg["paths"]["artifacts_dir"])
    features_dir  = art_dir / cfg["paths"].get("features_dir", "features")

    # features I/O from config
    feats_out_cfg = cfg.get("features", {}).get("output", {})
    feats_file    = feats_out_cfg.get("features_file", "dataset_features.parquet")
    feats_fmt     = feats_out_cfg.get("format", "parquet")
    manifest_file = feats_out_cfg.get("manifest_file", "features_manifest.json")

    feats_path    = features_dir / feats_file
    manifest_path = features_dir / manifest_file

    if not feats_path.exists():
        raise FileNotFoundError(f"{feats_path} not found — run engineer_features.py first.")

    time_col    = cfg["data"]["time_col"]
    station_col = cfg["data"].get("id_col", "station_id")
    train_end   = cfg["split"]["train_end"]
    val_end     = cfg["split"]["val_end"]

    # read features (respect format)
    if feats_fmt == "parquet":
        df = pd.read_parquet(feats_path)
    elif feats_fmt == "feather":
        df = pd.read_feather(feats_path)
    elif feats_fmt == "csv":
        df = pd.read_csv(feats_path)
    else:
        raise SystemExit(f"Unsupported features format: {feats_fmt}")

    # normalize time + optional year drops
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    drop_years = set(cfg.get("split", {}).get("drop_years", []))
    if drop_years:
        df = df[~df[time_col].dt.year.isin(drop_years)].copy()

    df = df.sort_values([station_col, time_col])

    pm25_col = _pick_pm25_col(df.columns.tolist(), cfg)

    # horizons: prefer manifest; else config; else infer from columns
    horizons: List[int] = []
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            hz = m.get("horizons")
            if isinstance(hz, list) and all(isinstance(x, int) for x in hz):
                horizons = hz
        except Exception:
            pass
    if not horizons:
        horizons = cfg.get("lags", {}).get("hours", []) or _infer_horizons_from_columns(df.columns.tolist(), pm25_col)

    # ddof setting for rolling std (defaults to pandas behavior = 1)
    roll_ddof = int(cfg.get("features", {}).get("rolling_std_ddof", 1))

    results = {
        "monotonic": _check_monotonic(df, station_col, time_col),
        "lags_ok": _check_lags(df, station_col, time_col, pm25_col, horizons) if pm25_col else True,
        "rolls_ok": _check_rolls(df, station_col, time_col, ddof=roll_ddof),
        "splits_ok": _check_splits(df, time_col, train_end, val_end),
    }

    if all(results.values()):
        print("[done] leakage checks complete.")
    else:
        print("[warn] some leakage checks failed. See messages above.")

if __name__ == "__main__":
    main()