# src/inspect_dataset.py
from __future__ import annotations

import os
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Iterable, Optional

import yaml
import numpy as np
import pandas as pd
import pyarrow.dataset as ds


# ---------------------------- config ----------------------------

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> dict:
    cfg_env = os.environ.get("CONFIG")
    if not cfg_env:
        raise SystemExit("CONFIG is required (comma-separated YAML paths).")
    merged: dict = {}
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        with open(p, "r") as f:
            overlay = yaml.safe_load(f) or {}
        merged = deepcopy(overlay) if not merged else _deep_update(merged, overlay)
    return merged

cfg = _load_cfg()

ART_DIR = Path(cfg["paths"]["artifacts_dir"])
STREAM_DIR = ART_DIR / cfg["paths"]["dataset_stream_dir"]

# Partition/prefix & output filenames
PART_PREFIX = cfg["output"]["partition_prefix"]         # e.g. "station="
PER_ST_FILE = cfg["output"].get("per_station_filename", "data.parquet")  # not used directly, kept for parity

reports_cfg: Dict = cfg.get("reports", {})
OUT_MISSING_CSV = ART_DIR / reports_cfg.get("station_missingness_csv", "station_missingness.csv")
OUT_KEEP_TXT    = ART_DIR / reports_cfg.get("stations_keep_txt", "stations_keep.txt")
OUT_EXCL_TXT    = ART_DIR / reports_cfg.get("stations_exclude_txt", "stations_exclude.txt")
OUT_REPORT_JSON = ART_DIR / reports_cfg.get("dataset_report_json", "dataset_report.json")

# Coverage rules
cov_cfg: Dict = cfg.get("coverage", {})
CORE_MET_COLS: List[str] = cov_cfg.get("core_columns", ["temperature","humidity","pressure","wind_speed","wind_direction"])
FORECAST_COLS: List[str] = cov_cfg.get("forecast_columns", ["up_temperature","bottom_temperature","wind_level"])
THRESHOLD_CORE: float = float(cov_cfg.get("threshold_core", 0.70))  ## missing value threshold

# Preferred ID/time aliases from config
H = cfg["data"]["headers"]
ID_ALIASES: List[str] = H.get("id", ["station_id"])
TIME_ALIASES: List[str] = H.get("time", ["Datetime"])

# (Optional) extra presence check list
POLLUTANT = [
    "PM25_Concentration","PM10_Concentration","NO2_Concentration",
    "CO_Concentration","O3_Concentration","SO2_Concentration",
]
REPORT_COLS_CHECK = list(set(POLLUTANT + CORE_MET_COLS + FORECAST_COLS +
                             ["weather","time_forecast","id","city_id","district_id"]))


# ---------------------------- helpers ----------------------------

def _station_dirs(root: Path, prefix: str) -> List[Path]:
    return sorted([p for p in root.glob(f"{prefix}*") if p.is_dir()], key=lambda p: p.name)

def _sid_from_dir(d: Path, prefix: str) -> str:
    name = d.name
    return name.split("=", 1)[1] if name.startswith(prefix) and "=" in name else name

def _dataset_schema_cols(st_dir: Path) -> List[str]:
    S = ds.dataset(str(st_dir), format="parquet")
    return list(S.schema.names)

def _count_rows(st_dir: Path) -> int:
    S = ds.dataset(str(st_dir), format="parquet")
    try:
        return int(S.count_rows())
    except Exception:
        return int(S.to_table(columns=[]).num_rows)

def _pick(cols: Iterable[str], cand: Iterable[str]) -> Optional[str]:
    s = set(cols)
    for c in cand:
        if c in s:
            return c
    return None

def _read_columns(st_dir: Path, cols: List[str]) -> pd.DataFrame:
    S = ds.dataset(str(st_dir), format="parquet")
    have = [c for c in cols if c in S.schema.names]
    tbl = S.to_table(columns=have) if have else S.to_table(columns=[])
    pdf = tbl.to_pandas()

    id_col = _pick(pdf.columns, ID_ALIASES) or "station_id"
    time_col = _pick(pdf.columns, TIME_ALIASES) or "Datetime"
    if id_col in pdf.columns:
        pdf[id_col] = pdf[id_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    if time_col in pdf.columns:
        pdf[time_col] = pd.to_datetime(pdf[time_col], errors="coerce")
    return pdf

def _coverage(non_null: int, total: int) -> float:
    return (non_null / total) if total else 0.0

def _fmt_pct(x: float) -> float:
    return round(float(x), 3)


# ---------------------------- main ----------------------------

def main():
    if not STREAM_DIR.exists():
        raise FileNotFoundError(f"{STREAM_DIR} not found. Build per-station dataset first.")

    part_dirs = _station_dirs(STREAM_DIR, PART_PREFIX)
    files_count = len(part_dirs)

    # union & intersection of columns
    union_cols: set = set()
    inter_cols: set = set()
    schema_by_station: Dict[Path, set] = {}
    for i, st_dir in enumerate(part_dirs):
        cols = set(_dataset_schema_cols(st_dir))
        schema_by_station[st_dir] = cols
        union_cols |= cols
        inter_cols = cols if i == 0 else (inter_cols & cols)

    rows: List[Dict] = []
    for st_dir in part_dirs:
        sid = _sid_from_dir(st_dir, PART_PREFIX)
        schema_cols = schema_by_station[st_dir]

        need_cols = list({*ID_ALIASES, *TIME_ALIASES, *CORE_MET_COLS, *FORECAST_COLS})
        pdf = _read_columns(st_dir, need_cols)
        rowcount = _count_rows(st_dir)

        rec: Dict = {"station_id": sid, "rowcount": int(rowcount)}

        # per-column coverage
        for col in CORE_MET_COLS + FORECAST_COLS:
            if col in pdf.columns:
                nn = int(pdf[col].notna().sum())
                cov = _coverage(nn, rowcount)
            else:
                nn, cov = 0, 0.0
            rec[f"{col}_non_null"] = nn
            rec[f"{col}_coverage"] = _fmt_pct(cov)

        core_covs = [rec.get(f"{c}_coverage", 0.0) for c in CORE_MET_COLS if c in schema_cols]
        overall_core = float(np.mean(core_covs)) if core_covs else 0.0
        rec["overall_core_coverage"] = _fmt_pct(overall_core)

        f_covs = [rec.get(f"{c}_coverage", 0.0) for c in FORECAST_COLS if c in schema_cols]
        rec["overall_forecast_coverage"] = _fmt_pct(float(np.mean(f_covs)) if f_covs else 0.0)

        rec["flag_exclude"] = bool(overall_core < THRESHOLD_CORE)
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    keep = df.loc[~df["flag_exclude"], "station_id"].tolist()
    excl = df.loc[df["flag_exclude"], "station_id"].tolist()

    ART_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_MISSING_CSV, index=False)
    Path(OUT_KEEP_TXT).write_text("\n".join(keep) + ("\n" if keep else ""))
    Path(OUT_EXCL_TXT).write_text("\n".join(excl) + ("\n" if excl else ""))

    report = {
        "files": files_count,
        "stations": len(part_dirs),
        "total_rows_sum_metadata": int(sum(_count_rows(d) for d in part_dirs)),
        "union_columns": sorted(list(union_cols)),
        "intersection_columns": sorted(list(inter_cols)),
        "coverage_threshold_core": THRESHOLD_CORE,
        "kept_stations": len(keep),
        "excluded_stations": len(excl),
        "mean_core_coverage": float(df["overall_core_coverage"].mean()) if not df.empty else 0.0,
        "mean_forecast_coverage": float(df["overall_forecast_coverage"].mean()) if not df.empty else 0.0,
        "columns_check_presence": {c: (c in union_cols) for c in REPORT_COLS_CHECK},
    }

    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== COVERAGE SUMMARY ===")
    print(f"Files: {files_count}")
    print(f"Stations: {len(part_dirs)}")
    print(f"Coverage threshold (CORE mean): {int(THRESHOLD_CORE * 100)}%")
    print(f"Keep: {len(keep)}  |  Exclude: {len(excl)}")
    if not df.empty:
        have_core = [c for c in CORE_MET_COLS if f"{c}_coverage" in df.columns]
        if have_core:
            core_line = ", ".join([f"{c}={df[f'{c}_coverage'].mean():.2f}" for c in have_core])
            print("\nMean CORE coverage across stations:")
            print(f"  {core_line}")

    print("\nOutputs:")
    print(f"  - station_missingness_csv: {OUT_MISSING_CSV}")
    print(f"  - stations_keep_txt: {OUT_KEEP_TXT}")
    print(f"  - stations_exclude_txt: {OUT_EXCL_TXT}")
    print(f"  - dataset_report_json: {OUT_REPORT_JSON}")


if __name__ == "__main__":
    main()