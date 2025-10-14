# src/test.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# ---------------------------- config ----------------------------

ART_DIR = Path("experiments/artifacts")
DATASET_DIR = ART_DIR / "dataset_stream"  # partitioned output (station=<id>/)
OUT_MISSING_CSV = ART_DIR / "station_missingness.csv"
OUT_KEEP_TXT = ART_DIR / "stations_keep.txt"
OUT_EXCL_TXT = ART_DIR / "stations_exclude.txt"
OUT_REPORT_JSON = ART_DIR / "dataset_report.json"

# Coverage rules
CORE_MET_COLS = ["temperature", "humidity", "pressure", "wind_speed", "wind_direction", "weather"]
FORECAST_COLS = ["up_temperature", "bottom_temperature", "wind_level"]  # optional
THRESHOLD_CORE = 0.70  # keep station if MEAN(core coverage) >= 70%

# Canonical identifiers we *prefer* to read if present in files
ID_COLS = ["station_id", "Datetime"]

POLLUTANT = [
    "PM25_Concentration",
    "PM10_Concentration",
    "NO2_Concentration",
    "CO_Concentration",
    "O3_Concentration",
    "SO2_Concentration",
]

# For quick union/intersection reporting
REPORT_COLS_CHECK = (
    POLLUTANT
    + CORE_MET_COLS
    + FORECAST_COLS
    + ["weather", "time_forecast", "id", "city_id", "district_id"]
)

# ---------------------------- helpers ----------------------------

def _station_dirs(root: Path) -> List[Path]:
    """Return sorted list of station partition directories like station=1001/"""
    return sorted([p for p in root.glob("station=*") if p.is_dir()], key=lambda p: p.name)

def _sid_from_dir(d: Path) -> str:
    """Extract station id from 'station=XXXX' directory name."""
    name = d.name
    return name.split("=", 1)[1] if name.startswith("station=") else name

def _dataset_schema_cols(st_dir: Path) -> List[str]:
    """Read dataset schema column names for a single station without loading all data."""
    S = ds.dataset(str(st_dir), format="parquet")
    return list(S.schema.names)

def _count_rows(st_dir: Path) -> int:
    """Fast, schema-safe row count for a partition directory."""
    S = ds.dataset(str(st_dir), format="parquet")
    try:
        # Arrow fast path using metadata/statistics when available
        return int(S.count_rows())
    except Exception:
        # Fallback: zero-column table still returns the correct row count
        return int(S.to_table(columns=[]).num_rows)

def _read_columns(st_dir: Path, cols: List[str]) -> pd.DataFrame:
    """
    Read selected columns from a single station partition to pandas,
    requesting only columns that actually exist (avoid Arrow errors).
    """
    S = ds.dataset(str(st_dir), format="parquet")
    have = [c for c in cols if c in S.schema.names]
    # Zero-column table is allowed; useful when only counting rows
    tbl = S.to_table(columns=have) if have else S.to_table(columns=[])
    pdf = tbl.to_pandas()

    # Normalize types of key columns if present
    if "station_id" in pdf.columns:
        pdf["station_id"] = (
            pdf["station_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        )
    if "Datetime" in pdf.columns:
        pdf["Datetime"] = pd.to_datetime(pdf["Datetime"], errors="coerce")

    return pdf

def _coverage(non_null: int, total: int) -> float:
    return (non_null / total) if total else 0.0

def _fmt_pct(x: float) -> float:
    return round(float(x), 3)

# ---------------------------- main logic ----------------------------

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"{DATASET_DIR} not found. Build first (e.g., ./scripts/02_dataprep.sh to create dataset_stream)."
        )

    part_dirs = _station_dirs(DATASET_DIR)
    files_count = len(part_dirs)

    # --- union & intersection of columns (station-level schemas)
    union_cols: set = set()
    inter_cols: set = set()
    # cache schema per station so we don't re-read it repeatedly
    schema_by_station: Dict[Path, set] = {}
    for i, st_dir in enumerate(part_dirs):
        cols = set(_dataset_schema_cols(st_dir))
        schema_by_station[st_dir] = cols
        union_cols |= cols
        inter_cols = cols if i == 0 else (inter_cols & cols)

    # --- per-station missingness / coverage
    rows: List[Dict] = []
    for st_dir in part_dirs:
        sid = _sid_from_dir(st_dir)
        schema_cols = schema_by_station[st_dir]

        # Read only what we need to compute coverage
        need_cols = list({*ID_COLS, *CORE_MET_COLS, *FORECAST_COLS})
        pdf = _read_columns(st_dir, need_cols)

        # Rowcount = all rows in partition (regardless of core cols)
        # Use fast counter to be robust even if ID_COLS aren't present in files
        rowcount = _count_rows(st_dir)

        rec: Dict = {"station_id": sid, "rowcount": int(rowcount)}

        # Per-column coverage
        for col in CORE_MET_COLS + FORECAST_COLS:
            if col in pdf.columns:
                nn = int(pdf[col].notna().sum())
                cov = _coverage(nn, rowcount)
            else:
                nn, cov = 0, 0.0
            rec[f"{col}_non_null"] = nn
            rec[f"{col}_coverage"] = _fmt_pct(cov)

        # Core-only gating: mean across core MET columns that exist in this partition's schema
        core_covs = [rec.get(f"{c}_coverage", 0.0) for c in CORE_MET_COLS if c in schema_cols]
        overall_core = np.mean(core_covs) if core_covs else 0.0
        rec["overall_core_coverage"] = _fmt_pct(overall_core)

        # Forecast rollup (FYI only)
        f_covs = [rec.get(f"{c}_coverage", 0.0) for c in FORECAST_COLS if c in schema_cols]
        rec["overall_forecast_coverage"] = _fmt_pct(np.mean(f_covs) if f_covs else 0.0)

        # Decision: gate only by core coverage
        rec["flag_exclude"] = bool(overall_core < THRESHOLD_CORE)

        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)

    # Keep/exclude lists
    keep = df.loc[~df["flag_exclude"], "station_id"].tolist()
    excl = df.loc[df["flag_exclude"], "station_id"].tolist()

    # --- write outputs
    ART_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_MISSING_CSV, index=False)
    Path(OUT_KEEP_TXT).write_text("\n".join(keep) + ("\n" if keep else ""))
    Path(OUT_EXCL_TXT).write_text("\n".join(excl) + ("\n" if excl else ""))

    # High-level report JSON
    report = {
        "files": files_count,
        "stations": len(part_dirs),
        # FIX: don't try to project 'station_id' (partition key) from file contents
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

    # Make timestamps JSON-safe (just in case)
    def _default(o):
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        return str(o)

    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_default)

    # --- console summary (concise)
    print("\n=== COVERAGE SUMMARY ===")
    print(f"Files: {files_count}")
    print(f"Stations: {len(part_dirs)}")
    print(f"Coverage threshold (CORE mean): {int(THRESHOLD_CORE * 100)}%")
    print(f"Keep: {len(keep)}  |  Exclude: {len(excl)}")

    # Per-column mean coverage (for quick feel)
    core_means = {c: float(df.get(f"{c}_coverage", pd.Series(dtype=float)).mean()) for c in CORE_MET_COLS}
    f_means = {c: float(df.get(f"{c}_coverage", pd.Series(dtype=float)).mean()) for c in FORECAST_COLS}

    have_core = [c for c in CORE_MET_COLS if f"{c}_coverage" in df.columns]
    if have_core:
        core_line = ", ".join([f"{c}={core_means[c]:.2f}" for c in have_core])
        print("\nMean CORE coverage across stations:")
        print(f"  {core_line}")

    have_f = [c for c in FORECAST_COLS if f"{c}_coverage" in df.columns]
    if have_f:
        f_line = ", ".join([f"{c}={f_means[c]:.2f}" for c in have_f])
        print("\nMean FORECAST coverage across stations (FYI):")
        print(f"  {f_line}")

    print("\nOutputs:")
    print(f"  - station_missingness_csv: {OUT_MISSING_CSV}")
    print(f"  - stations_keep_txt: {OUT_KEEP_TXT}")
    print(f"  - stations_exclude_txt: {OUT_EXCL_TXT}")
    print(f"  - dataset_report_json: {OUT_REPORT_JSON}")

if __name__ == "__main__":
    main()