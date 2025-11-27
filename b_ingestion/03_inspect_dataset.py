#!/usr/bin/env python3
# b_ingestion/03_inspect_dataset.py
from __future__ import annotations

import sys, json
from pathlib import Path
from typing import Dict, List, Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# Ensure repo root on sys.path (so "common" is importable)
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.config_loader import load_cfg  # overlay-aware, shared


# ---------------------------- helpers ----------------------------

def _req(d: dict, path: List[str]):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise SystemExit("configs: missing " + ".".join(path))
        cur = cur[k]
    return cur

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

def _read_columns(st_dir: Path, cols: List[str], id_aliases: List[str], time_aliases: List[str]) -> pd.DataFrame:
    S = ds.dataset(str(st_dir), format="parquet")
    have = [c for c in cols if c in S.schema.names]
    tbl = S.to_table(columns=have) if have else S.to_table(columns=[])
    pdf = tbl.to_pandas()

    id_col = _pick(pdf.columns, id_aliases)
    time_col = _pick(pdf.columns, time_aliases)
    if id_col:
        pdf[id_col] = pdf[id_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    if time_col:
        pdf[time_col] = pd.to_datetime(pdf[time_col], errors="coerce")
    return pdf

def _coverage(non_null: int, total: int) -> float:
    return (non_null / total) if total else 0.0

def _fmt3(x: float) -> float:
    return round(float(x), 3)

def _resolve_out(art_dir: Path, val: Optional[str], default_name: str) -> Path:
    """
    Robust resolution for report outputs:
    - Empty / None -> artifacts_dir/default_name
    - Ends with '/' -> treat as directory under artifacts_dir
    - Relative path -> artifacts_dir/val
    - Absolute path -> val
    """
    if not val or not str(val).strip():
        return art_dir / default_name
    val_s = str(val)
    p = Path(val_s)
    if val_s.endswith("/"):
        return (art_dir / p) / default_name if not p.is_absolute() else p / default_name
    # If caller passed a directory-like token (no suffix) that already exists, still allow file default inside it
    if p.suffix == "" and (p.is_dir() or val_s in (".", "./")):
        return (art_dir / p / default_name) if not p.is_absolute() else (p / default_name)
    return p if p.is_absolute() else (art_dir / p)


# ---------------------------- main ----------------------------

def main():
    cfg: Dict = load_cfg()

    # Paths (strict)
    art_dir        = Path(_req(cfg, ["paths", "artifacts_dir"]))
    stream_dirname = _req(cfg, ["paths", "dataset_stream_dir"])
    stream_dir     = art_dir / stream_dirname

    # Output partition conventions (strict)
    part_prefix    = _req(cfg, ["output", "partition_prefix"])

    # Reports (tolerant + safe)
    reports_cfg       = cfg.get("reports", {}) or {}
    out_missing_csv   = _resolve_out(art_dir, reports_cfg.get("station_missingness_csv"), "station_missingness.csv")
    out_keep_txt      = _resolve_out(art_dir, reports_cfg.get("stations_keep_txt"), "stations_keep.txt")
    out_excl_txt      = _resolve_out(art_dir, reports_cfg.get("stations_exclude_txt"), "stations_exclude.txt")
    out_report_json   = _resolve_out(art_dir, reports_cfg.get("dataset_report_json"), "dataset_report.json")

    # Ensure parent dirs exist
    for p in (out_missing_csv, out_keep_txt, out_excl_txt, out_report_json):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Coverage config (tolerant)
    cov_cfg = cfg.get("coverage", {}) or {}
    core_cols: List[str]       = list(cov_cfg.get("core_columns", ["temperature","humidity","pressure","wind_speed","wind_direction"]))
    forecast_cols: List[str]   = list(cov_cfg.get("forecast_columns", []))
    pollutant_cols: List[str]  = list(cov_cfg.get("pollutant_columns", ["PM25_Concentration","PM10_Concentration","NO2_Concentration","CO_Concentration","O3_Concentration","SO2_Concentration"]))
    threshold_core: float      = float(cov_cfg.get("threshold_core", 0.50))

    # Header aliases (tolerant)
    H = (cfg.get("data") or {}).get("headers", {}) or {}
    id_aliases   = list(H.get("id", []))
    time_aliases = list(H.get("time", []))
    if not id_aliases:
        id_aliases = [ (cfg.get("data") or {}).get("id_col", "station_id") ]
    if not time_aliases:
        time_aliases = [ (cfg.get("data") or {}).get("time_col", "Datetime") ]

    # Optional alias lists for presence checks
    wf_future_aliases = list((H.get("weatherforecast", {}) or {}).get("future", []))
    met_geo_aliases   = list((H.get("meteorology", {})     or {}).get("geo", []))

    # Columns to report presence on
    report_cols_check = sorted(set(pollutant_cols + core_cols + forecast_cols + ["weather"] + wf_future_aliases + met_geo_aliases + id_aliases))

    if not stream_dir.exists():
        raise FileNotFoundError(f"{stream_dir} not found. Build per-station dataset first.")

    part_dirs = _station_dirs(stream_dir, part_prefix)
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
        sid = _sid_from_dir(st_dir, part_prefix)
        schema_cols = schema_by_station[st_dir]

        need_cols = list({*id_aliases, *time_aliases, *core_cols, *forecast_cols})
        pdf = _read_columns(st_dir, need_cols, id_aliases, time_aliases)
        rowcount = _count_rows(st_dir)

        rec: Dict = {"station_id": sid, "rowcount": int(rowcount)}

        # per-column coverage for configured sets
        for col in core_cols + forecast_cols:
            if col in pdf.columns:
                nn = int(pdf[col].notna().sum())
                cov = _coverage(nn, rowcount)
            else:
                nn, cov = 0, 0.0
            rec[f"{col}_non_null"] = nn
            rec[f"{col}_coverage"] = _fmt3(cov)

        # mean coverages
        core_covs = [rec.get(f"{c}_coverage", 0.0) for c in core_cols if c in schema_cols]
        overall_core = float(np.mean(core_covs)) if core_covs else 0.0
        rec["overall_core_coverage"] = _fmt3(overall_core)

        f_covs = [rec.get(f"{c}_coverage", 0.0) for c in forecast_cols if c in schema_cols]
        rec["overall_forecast_coverage"] = _fmt3(float(np.mean(f_covs)) if f_covs else 0.0)

        rec["flag_exclude"] = bool(overall_core < threshold_core)
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    keep = df.loc[~df["flag_exclude"], "station_id"].tolist()
    excl = df.loc[df["flag_exclude"], "station_id"].tolist()

    df.to_csv(out_missing_csv, index=False)
    Path(out_keep_txt).write_text("\n".join(keep) + ("\n" if keep else ""))
    Path(out_excl_txt).write_text("\n".join(excl) + ("\n" if excl else ""))

    report = {
        "files": files_count,
        "stations": len(part_dirs),
        "total_rows_sum_metadata": int(sum(_count_rows(d) for d in part_dirs)),
        "union_columns": sorted(list(union_cols)),
        "intersection_columns": sorted(list(inter_cols)),
        "coverage_threshold_core": threshold_core,
        "kept_stations": len(keep),
        "excluded_stations": len(excl),
        "mean_core_coverage": float(df["overall_core_coverage"].mean()) if not df.empty else 0.0,
        "mean_forecast_coverage": float(df["overall_forecast_coverage"].mean()) if not df.empty else 0.0,
        "columns_check_presence": {c: (c in union_cols) for c in report_cols_check},
    }

    with open(out_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== COVERAGE SUMMARY ===")
    print(f"Files: {files_count}")
    print(f"Stations: {len(part_dirs)}")
    print(f"Coverage threshold (CORE mean): {int(float(threshold_core) * 100)}%")
    print(f"Keep: {len(keep)}  |  Exclude: {len(excl)}")
    if not df.empty and core_cols:
        have_core = [c for c in core_cols if f"{c}_coverage" in df.columns]
        if have_core:
            core_line = ", ".join([f"{c}={df[f'{c}_coverage'].mean():.2f}" for c in have_core])
            print("\nMean CORE coverage across stations:")
            print(f"  {core_line}")

    print("\nOutputs:")
    print(f"  - station_missingness_csv: {out_missing_csv}")
    print(f"  - stations_keep_txt: {out_keep_txt}")
    print(f"  - stations_exclude_txt: {out_excl_txt}")
    print(f"  - dataset_report_json: {out_report_json}")


if __name__ == "__main__":
    main()