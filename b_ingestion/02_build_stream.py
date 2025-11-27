#!/usr/bin/env python3
# b_ingestion/02_build_stream.py
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

# Ensure repo root is importable (parent of this file)
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.config_loader import load_cfg  # overlay-aware


# ======================== small helpers ========================

def _req(d: dict, path: List[str]):
    """Strictly require a nested key path; exit with a clear message if missing."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise SystemExit("configs: missing " + ".".join(path))
        cur = cur[k]
    return cur

def _opt_list(d: dict, path: List[str], fallback: List[str] | None = None) -> List[str]:
    """Get optional list from config; fallback to [] or provided default."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return list(fallback or [])
        cur = cur[k]
    v = cur
    if v is None:
        return list(fallback or [])
    if isinstance(v, list):
        return v
    return [v]

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ext_from_format(fmt: str) -> str:
    f = str(fmt).lower()
    if f in ("parquet", "feather", "csv"):
        return f
    raise SystemExit(f"Unsupported output.format '{fmt}'")

def _pick(cols: Iterable[str], cand: Iterable[str]) -> Optional[str]:
    s = set(cols)
    for c in cand:
        if c in s:
            return c
    return None

def _norm_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def _field_type(dset: ds.Dataset, name: str) -> pa.DataType:
    return dset.schema.field(name).type

def _maybe_time_filter(dset: ds.Dataset, field: str, tmin, tmax):
    t = _field_type(dset, field)
    if pa.types.is_timestamp(t) or pa.types.is_date(t):
        f = ds.field(field)
        if pa.types.is_timestamp(t):
            unit = t.unit  # 's'/'ms'/'us'/'ns'
            tmin_pa = pa.scalar(pd.Timestamp(tmin).to_pydatetime(), type=pa.timestamp(unit))
            tmax_pa = pa.scalar(pd.Timestamp(tmax).to_pydatetime(), type=pa.timestamp(unit))
        else:
            tmin_pa = pa.scalar(pd.Timestamp(tmin).date(), type=t)
            tmax_pa = pa.scalar(pd.Timestamp(tmax).date(), type=t)
        return (f >= tmin_pa) & (f <= tmax_pa)
    return None

def _eq_string_field(field: str, value: str):
    return ds.field(field).cast(pa.string()) == str(value)

def _isin_string_field(field: str, values: List[str]):
    return ds.field(field).cast(pa.string()).isin([str(v) for v in values])

def _load_small_table(path: Path | None, key_renames: Dict[str, str]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    cols_wanted = [c for c in key_renames.keys() if c is not None]
    tbl = ds.dataset(path).to_table(columns=list(dict.fromkeys(cols_wanted)))  # dedup columns
    pdf = tbl.to_pandas().rename(columns={k: v for k, v in key_renames.items() if k is not None})
    for k in key_renames.values():
        if k in pdf.columns:
            pdf[k] = _norm_id_series(pdf[k])
    key_cols = [c for c in key_renames.values() if c in pdf.columns]
    if key_cols:
        pdf = pdf.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    return pdf

def _coalesce_station_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key = ["station_id", "Datetime"]
    num_cols = df.select_dtypes(include="number").columns.difference(key)
    agg = {c: "mean" for c in num_cols}
    for c in df.columns.difference(num_cols.union(key)):
        agg[c] = "first"
    return df.sort_values(key).groupby(key, as_index=False).agg(agg)

def _split_by_geo_membership(df: pd.DataFrame, geo_col: str,
                             district_ids: set[str], city_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or geo_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    g = df[geo_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    dist = df.loc[g.isin(district_ids)].copy()
    city = df.loc[g.isin(city_ids)].copy()
    if not dist.empty:
        dist.rename(columns={geo_col: "district_id"}, inplace=True)
    if not city.empty:
        city.rename(columns={geo_col: "city_id"}, inplace=True)
    return dist, city

def _merge_new_cols(left: pd.DataFrame, right: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if right.empty:
        return left
    r_non_keys = [c for c in right.columns if c not in keys]
    r_take = [c for c in r_non_keys if c not in left.columns]
    if not r_take:
        return left
    return left.merge(right[keys + r_take], on=keys, how="left")

def _is_within(root: Path, path: Path) -> bool:
    try:
        r = root.resolve()
        p = path.resolve()
    except FileNotFoundError:
        r = root.resolve()
        p = path.parent.resolve()
    return r == p or str(p).startswith(str(r))


# ============================ main ============================

def build_stream():
    cfg: Dict = load_cfg()

    # --- artifacts naming (Option B) ---
    out_cfg   = _req(cfg, ["output"])
    fmt       = _req(out_cfg, ["format"])
    ext       = out_cfg.get("extension") or _ext_from_format(fmt)

    # Prefer new location; tolerate legacy top-level as fallback
    out_naming = (out_cfg.get("naming") or {}).get("pattern")
    legacy_naming = (cfg.get("naming") or {}).get("pattern")
    pattern = out_naming or legacy_naming or "{key}.{ext}"

    partition_prefix = _req(out_cfg, ["partition_prefix"])
    per_station_file = _req(out_cfg, ["per_station_filename"])
    subdir           = out_cfg.get("subdir")  # optional

    # --- strict paths + output dirs ---
    art_dir            = Path(_req(cfg, ["paths", "artifacts_dir"]))
    dataset_stream_dir = _req(cfg, ["paths", "dataset_stream_dir"])
    out_dir            = art_dir / dataset_stream_dir

    # Where csv2parquet wrote artifacts (optionally nested)
    parquet_dir = art_dir / subdir if subdir else art_dir

    def _artifact_name_for(key: str) -> str:
        # pattern may contain {key} and {ext}
        name = pattern.format(key=key, ext=ext)
        if not ext and name.endswith("."):
            name = name[:-1]
        return name

    # Logical inputs (present only if declared in YAML)
    data_keys = _req(cfg, ["data_files"])
    logical = {k: None for k in ("air", "met", "station", "district", "city", "forecast")}
    for k in logical.keys():
        if k in data_keys:
            logical[k] = parquet_dir / _artifact_name_for(k)

    pq_air = logical["air"]
    pq_met = logical["met"]
    pq_sta = logical["station"]
    pq_dis = logical["district"]
    pq_cit = logical["city"]
    pq_wf  = logical["forecast"]

    # Headers (tolerant for optional sections)
    H = cfg.get("data", {}).get("headers", {}) or {}

    # Required: AIR
    if pq_air is None or not pq_air.exists():
        raise SystemExit("Airquality parquet not found. Run csv2parquet first with data_files.air configured.")
    ds_air = ds.dataset(pq_air)

    # Optional: MET, WF
    ds_met = ds.dataset(pq_met) if pq_met and pq_met.exists() else None
    ds_wf  = ds.dataset(pq_wf)  if pq_wf  and pq_wf.exists()  else None

    # Header picks with safe fallbacks
    air_station_candidates = _opt_list(H, ["airquality", "station"], ["station_id", "Station ID", "station", "STATION_ID"])
    air_time_candidates    = _opt_list(H, ["airquality", "time"],    ["Datetime", "Time", "timestamp"])
    air_station_src = _pick(ds_air.schema.names, air_station_candidates)
    air_time_src    = _pick(ds_air.schema.names, air_time_candidates)
    if air_station_src is None or air_time_src is None:
        raise SystemExit("Could not detect station/time columns in AIR (check data.headers.airquality).")

    met_geo_src  = None
    met_time_src = None
    if ds_met:
        met_geo_src  = _pick(ds_met.schema.names, _opt_list(H, ["meteorology", "geo"],  ["ID", "geo_id", "id"]))
        met_time_src = _pick(ds_met.schema.names, _opt_list(H, ["meteorology", "time"], ["Time", "Datetime", "timestamp"]))

    wf_geo_src = wf_future_src = wf_ftime_src = None
    if ds_wf:
        wf_geo_src    = _pick(ds_wf.schema.names, _opt_list(H, ["weatherforecast", "geo"],   ["ID", "geo_id", "id"]))
        wf_future_src = _pick(ds_wf.schema.names, _opt_list(H, ["weatherforecast", "future"],["Future Time", "future_time", "time_future"]))
        wf_ftime_src  = _pick(ds_wf.schema.names, _opt_list(H, ["weatherforecast", "ftime"], ["Forecast Time", "time_forecast", "ftime"]))

    # Small dimension tables (guard for missing header sections)
    station_section  = H.get("station", {}) if isinstance(H.get("station"), dict) else {}
    district_section = H.get("district", {}) if isinstance(H.get("district"), dict) else {}
    city_section     = H.get("city", {}) if isinstance(H.get("city"), dict) else {}

    sta = _load_small_table(
        pq_sta,
        {
            _pick(ds.dataset(pq_sta).schema.names, _opt_list({"station": station_section}, ["station", "station"], ["station_id", "Station ID", "station"]))
            if pq_sta else None: "station_id",
            _pick(ds.dataset(pq_sta).schema.names, _opt_list({"station": station_section}, ["station", "district"], ["district_id", "District ID", "district"]))
            if pq_sta else None: "district_id",
        }
    ) if pq_sta and pq_sta.exists() else pd.DataFrame()

    dis = _load_small_table(
        pq_dis,
        {
            _pick(ds.dataset(pq_dis).schema.names, _opt_list({"district": district_section}, ["district", "district"], ["district_id", "District ID", "district"]))
            if pq_dis else None: "district_id",
            _pick(ds.dataset(pq_dis).schema.names, _opt_list({"district": district_section}, ["district", "city"], ["city_id", "City ID", "city"]))
            if pq_dis else None: "city_id",
        }
    ) if pq_dis and pq_dis.exists() else pd.DataFrame()

    cit = _load_small_table(
        pq_cit,
        {
            _pick(ds.dataset(pq_cit).schema.names, _opt_list({"city": city_section}, ["city", "city"], ["city_id", "City ID", "city"]))
            if pq_cit else None: "city_id",
        }
    ) if pq_cit and pq_cit.exists() else pd.DataFrame()

    # Membership sets for GEO mapping
    district_ids: set[str] = set()
    city_ids: set[str] = set()
    if not sta.empty and "district_id" in sta:
        district_ids |= set(sta["district_id"].dropna().astype(str))
    if not dis.empty and "district_id" in dis:
        district_ids |= set(dis["district_id"].dropna().astype(str))
    if not dis.empty and "city_id" in dis:
        city_ids |= set(dis["city_id"].dropna().astype(str))
    if not cit.empty and "city_id" in cit:
        city_ids |= set(cit["city_id"].dropna().astype(str))

    # Station/time range from AIR
    air_minmax = ds_air.to_table(columns=[air_station_src, air_time_src]).to_pandas()
    air_minmax[air_station_src] = _norm_id_series(air_minmax[air_station_src])
    tmin = _to_dt(air_minmax[air_time_src]).min()
    tmax = _to_dt(air_minmax[air_time_src]).max()
    station_ids_all = air_minmax[air_station_src].dropna().astype(str).unique().tolist()
    del air_minmax

    # station_scope
    scope = cfg.get("station_scope", {}) or {}
    mode = str(scope.get("mode", "all")).lower()
    if mode == "min_rows":
        tbl = ds_air.to_table(columns=[air_station_src, air_time_src]).to_pandas()
        tbl[air_station_src] = _norm_id_series(tbl[air_station_src])
        tbl[air_time_src] = _to_dt(tbl[air_time_src]).dt.floor("h")
        train_end = pd.to_datetime(_req(cfg, ["split", "train_end"]))
        counts = tbl.loc[tbl[air_time_src] <= train_end, air_station_src].value_counts()
        thr = int(scope.get("min_rows", 0))
        keep = set(counts[counts >= thr].index.astype(str))
        station_ids = [s for s in station_ids_all if s in keep]
        print(f"[build] station_scope=min_rows≥{thr} kept={len(station_ids)} of {len(station_ids_all)}")
    elif mode in ("ids", "filter"):
        raw_ids = scope.get("station_ids", []) or []
        scope_ids = {str(x).strip() for x in raw_ids if str(x).strip()}
        station_ids = [s for s in station_ids_all if s in scope_ids]
        print(f"[build] station_scope={mode} requested={len(scope_ids)} kept={len(station_ids)}")
    elif mode == "all":
        station_ids = station_ids_all
        print("[build] station_scope=all")
    else:
        raise SystemExit(f"Unknown station_scope.mode '{mode}'")

    if not station_ids:
        raise SystemExit("station_scope filtered all stations; nothing to build.")
    print(f"[build] stations={len(station_ids)}  time=[{tmin} → {tmax}]")

    # Fresh output
    if out_dir.exists():
        if _is_within(art_dir, out_dir) and out_dir.name == Path(dataset_stream_dir).name:
            shutil.rmtree(out_dir)
        else:
            raise RuntimeError(f"Refusing to delete unsafe path: {out_dir}")
    _ensure_dir(out_dir)

    # ----- loaders -----

    def _load_air_station(
        ds_air: ds.Dataset,
        sid: str,
        station_src: str,
        time_src: str,
        tmin,
        tmax,
        dedupe_air: bool = False,
    ) -> pd.DataFrame:
        flt_time = _maybe_time_filter(ds_air, time_src, tmin, tmax)
        flt = _eq_string_field(station_src, sid)
        if flt_time is not None:
            flt = flt & flt_time
        cols = [station_src, time_src] + [c for c in ds_air.schema.names if c not in {station_src, time_src}]
        tbl = ds_air.to_table(filter=flt, columns=cols)
        if tbl.num_rows == 0:
            return pd.DataFrame()
        pdf = tbl.to_pandas().rename(columns={station_src: "station_id", time_src: "Datetime"})
        pdf["station_id"] = _norm_id_series(pdf["station_id"])
        pdf["Datetime"] = _to_dt(pdf["Datetime"]).dt.floor("h")
        if flt_time is None:
            pdf = pdf.loc[(pdf["Datetime"] >= pd.to_datetime(tmin)) & (pdf["Datetime"] <= pd.to_datetime(tmax))]
        pdf = pdf.sort_values(["station_id", "Datetime"]).reset_index(drop=True)
        if dedupe_air:
            pdf = pdf.drop_duplicates(subset=["station_id", "Datetime"], keep="last")
        return pdf

    def _load_met_slice(
        ds_met: Optional[ds.Dataset],
        geo_ids: List[str],
        met_geo_src: Optional[str],
        met_time_src: Optional[str],
        tmin,
        tmax,
    ) -> pd.DataFrame:
        if ds_met is None or met_time_src is None:
            return pd.DataFrame()
        flt_time = _maybe_time_filter(ds_met, met_time_src, tmin, tmax)
        flt = flt_time
        if met_geo_src and geo_ids:
            geo_f = _isin_string_field(met_geo_src, geo_ids)
            flt = geo_f if flt is None else (geo_f & flt)
        cols = [met_time_src] + ([met_geo_src] if met_geo_src else [])
        try:
            tbl = ds_met.to_table(filter=flt, columns=list(dict.fromkeys(cols))) if flt is not None else ds_met.to_table(columns=list(dict.fromkeys(cols)))
        except Exception:
            tbl = ds_met.to_table(filter=flt) if flt is not None else ds_met.to_table()
        if tbl.num_rows == 0:
            return pd.DataFrame()
        pdf = tbl.to_pandas().rename(columns={met_time_src: "Datetime"})
        pdf["Datetime"] = _to_dt(pdf["Datetime"]).dt.floor("h")
        if met_geo_src and met_geo_src in pdf.columns:
            pdf["geo_id"] = _norm_id_series(pdf[met_geo_src])
        return pdf

    def _load_wf_slice(
        ds_wf: Optional[ds.Dataset],
        geo_ids: List[str],
        tmin,
        tmax,
        wf_geo_src: Optional[str],
        wf_future_src: Optional[str],
        wf_ftime_src: Optional[str],
    ) -> pd.DataFrame:
        if ds_wf is None:
            return pd.DataFrame()
        flt = None
        if wf_future_src:
            flt = _maybe_time_filter(ds_wf, wf_future_src, tmin, tmax)
        if wf_geo_src and geo_ids:
            geo_f = _isin_string_field(wf_geo_src, geo_ids)
            flt = geo_f if flt is None else (geo_f & flt)
        cols = []
        if wf_future_src: cols.append(wf_future_src)
        if wf_geo_src:    cols.append(wf_geo_src)
        try:
            tbl = ds_wf.to_table(filter=flt, columns=list(dict.fromkeys(cols))) if flt is not None else ds_wf.to_table(columns=list(dict.fromkeys(cols)))
        except Exception:
            tbl = ds_wf.to_table(filter=flt) if flt is not None else ds_wf.to_table()
        if tbl.num_rows == 0:
            return pd.DataFrame()
        pdf = tbl.to_pandas()
        if wf_geo_src and wf_geo_src in pdf.columns:
            pdf[wf_geo_src] = _norm_id_series(pdf[wf_geo_src])
        return pdf

    # ----- per-station build -----
    for i, sid in enumerate(station_ids, 1):
        print(f"[{i}/{len(station_ids)}] station={sid} …")

        a = _load_air_station(ds_air, sid, air_station_src, air_time_src, tmin, tmax, dedupe_air=False)
        if a.empty:
            print(f"[warn][station={sid}] no air rows, skipping")
            continue

        # attach district/city via lookups (if available)
        if pq_sta and pq_sta.exists():
            sta_df = _load_small_table(
                pq_sta,
                {
                    _pick(ds.dataset(pq_sta).schema.names, _opt_list({"station": station_section}, ["station", "station"], ["station_id", "Station ID", "station"])): "station_id",
                    _pick(ds.dataset(pq_sta).schema.names, _opt_list({"station": station_section}, ["station", "district"], ["district_id", "District ID", "district"])): "district_id",
                }
            )
            if not sta_df.empty:
                a = a.merge(
                    sta_df[["station_id", "district_id"]].drop_duplicates(["station_id", "district_id"]),
                    on="station_id", how="left"
                )
        if pq_dis and pq_dis.exists():
            dis_df = _load_small_table(
                pq_dis,
                {
                    _pick(ds.dataset(pq_dis).schema.names, _opt_list({"district": district_section}, ["district", "district"], ["district_id", "District ID", "district"])): "district_id",
                    _pick(ds.dataset(pq_dis).schema.names, _opt_list({"district": district_section}, ["district", "city"], ["city_id", "City ID", "city"])): "city_id",
                }
            )
            if not dis_df.empty:
                a = a.merge(
                    dis_df[["district_id", "city_id"]].drop_duplicates(["district_id", "city_id"]),
                    on="district_id", how="left"
                )

        # MET join
        if ds_met and met_time_src:
            geo_ids: List[str] = []
            if "district_id" in a.columns:
                geo_ids.extend(a["district_id"].dropna().astype(str).tolist())
            if "city_id" in a.columns:
                geo_ids.extend(a["city_id"].dropna().astype(str).tolist())
            geo_ids = list({g for g in geo_ids if g})

            met_pdf = _load_met_slice(ds_met, geo_ids, met_geo_src, met_time_src, tmin, tmax)
            if not met_pdf.empty:
                if met_geo_src and "geo_id" in met_pdf.columns:
                    met_dist, met_city = _split_by_geo_membership(met_pdf, "geo_id", district_ids, city_ids)
                    if not met_dist.empty:
                        met_dist = met_dist.sort_values(["district_id","Datetime"]).drop_duplicates(["district_id","Datetime"], keep="last")
                    if not met_city.empty:
                        met_city = met_city.sort_values(["city_id","Datetime"]).drop_duplicates(["city_id","Datetime"], keep="last")
                    if not met_dist.empty and "district_id" in a.columns:
                        a = _merge_new_cols(a, met_dist, keys=["district_id","Datetime"])
                    if not met_city.empty and "city_id" in a.columns:
                        a = _merge_new_cols(a, met_city,  keys=["city_id","Datetime"])
                else:
                    met_trim = met_pdf.drop(columns=[c for c in ["geo_id"] if c in met_pdf.columns])
                    a = _merge_new_cols(a, met_trim, keys=["Datetime"])

        # WF join
        if ds_wf:
            geo_ids = []
            if "district_id" in a.columns:
                geo_ids += a["district_id"].dropna().astype(str).tolist()
            if "city_id" in a.columns:
                geo_ids += a["city_id"].dropna().astype(str).tolist()
            geo_ids = list({g for g in geo_ids if g})

            wf_pdf = _load_wf_slice(ds_wf, geo_ids, tmin, tmax, wf_geo_src, wf_future_src, wf_ftime_src)
            if not wf_pdf.empty:
                # choose future or fallback to any time-like column named in headers; finally allow "Datetime"
                candidates = []
                if wf_future_src: candidates.append(wf_future_src)
                candidates += _opt_list(H, ["weatherforecast", "future"], [])
                if "Datetime" not in candidates:
                    candidates.append("Datetime")
                fut_col = next((c for c in candidates if c in wf_pdf.columns), None)

                if fut_col:
                    if fut_col != "Datetime":
                        wf_pdf = wf_pdf.rename(columns={fut_col: "Datetime"})
                    wf_pdf["Datetime"] = _to_dt(wf_pdf["Datetime"]).dt.floor("h")

                    if wf_geo_src and wf_geo_src in wf_pdf.columns:
                        wf_pdf[wf_geo_src] = _norm_id_series(wf_pdf[wf_geo_src])
                        wf_dist, wf_city = _split_by_geo_membership(wf_pdf, wf_geo_src, district_ids, city_ids)
                        if not wf_dist.empty:
                            wf_dist = wf_dist.sort_values(["district_id","Datetime"]).drop_duplicates(["district_id","Datetime"], keep="last")
                        if not wf_city.empty:
                            wf_city = wf_city.sort_values(["city_id","Datetime"]).drop_duplicates(["city_id","Datetime"], keep="last")
                        if not wf_dist.empty and "district_id" in a.columns:
                            a = _merge_new_cols(a, wf_dist, keys=["district_id","Datetime"])
                        if not wf_city.empty and "city_id" in a.columns:
                            a = _merge_new_cols(a, wf_city,  keys=["city_id","Datetime"])
                    else:
                        drop_cols = [c for c in (wf_ftime_src, wf_geo_src, wf_future_src) if c and c in wf_pdf.columns]
                        wf_trim = wf_pdf.drop(columns=drop_cols, errors="ignore")
                        a = _merge_new_cols(a, wf_trim, keys=["Datetime"])

        # final coalesce + write
        a = _coalesce_station_time(a)
        st_dir = out_dir / f"{partition_prefix}{sid}"
        st_dir.mkdir(parents=True, exist_ok=True)
        a.to_parquet(st_dir / per_station_file, index=False)

    print(f"[OK] wrote partitioned dataset → {out_dir}")
    print("Read via pyarrow.dataset: ds.dataset(str(out_dir))")


if __name__ == "__main__":
    build_stream()