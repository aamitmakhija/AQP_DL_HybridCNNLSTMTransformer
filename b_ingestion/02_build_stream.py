# src/data_build_stream.py
from __future__ import annotations

import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Tuple

import yaml
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


# ======================== config ========================

def _deep_update(dst, src):
    """Recursively update dict dst with dict src."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> Dict:
    """Load base config + optional overlays from CONFIG env (comma-separated)."""
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

CFG_PATH = os.environ.get("CONFIG", "configs/default.yaml")
print(f"[config] using {CFG_PATH}")
cfg: Dict = _load_cfg()

ART_DIR = Path(cfg["paths"]["artifacts_dir"])

# paths to parquet artifacts
PQ_AIR = ART_DIR / "air.parquet"
PQ_MET = ART_DIR / "met.parquet"
PQ_STA = ART_DIR / "station.parquet"
PQ_DIS = ART_DIR / "district.parquet"
PQ_CIT = ART_DIR / "city.parquet"
PQ_WF  = ART_DIR / "forecast.parquet"

# headers dictionary from config
H = cfg["data"]["headers"]

# output directory for the per-station dataset
OUT_DIR = ART_DIR / "dataset_stream"  # partitioned output (one file per station)


# ======================== helpers ========================

def _pick(cols: Iterable[str], cand: Iterable[str]) -> Optional[str]:
    s = set(cols)
    for c in cand:
        if c in s:
            return c
    return None

def _norm_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"\.0$", "", regex=True)
         .str.strip()
    )

def _read_dataset_required(path: Path) -> ds.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.csv2parquet_safe")
    return ds.dataset(path)

def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def _field_type(dset: ds.Dataset, name: str) -> pa.DataType:
    return dset.schema.field(name).type

def _maybe_time_filter(dset: ds.Dataset, field: str, tmin, tmax):
    """Return an Arrow filter if the field is temporal, else None (filter later in pandas)."""
    t = _field_type(dset, field)
    if pa.types.is_timestamp(t) or pa.types.is_date(t):
        f = ds.field(field)
        if pa.types.is_timestamp(t):
            unit = t.unit  # 's'/'ms'/'us'/'ns'
            tmin_pa = pa.scalar(pd.Timestamp(tmin).to_datetime64(), type=pa.timestamp(unit))
            tmax_pa = pa.scalar(pd.Timestamp(tmax).to_datetime64(), type=pa.timestamp(unit))
        else:
            tmin_pa = pa.scalar(pd.Timestamp(tmin).date(), type=t)
            tmax_pa = pa.scalar(pd.Timestamp(tmax).date(), type=t)
        return (f >= tmin_pa) & (f <= tmax_pa)
    return None

def _eq_string_field(field: str, value: str):
    return ds.field(field).cast(pa.string()) == str(value)

def _isin_string_field(field: str, values: List[str]):
    return ds.field(field).cast(pa.string()).isin([str(v) for v in values])

def _load_small_table(path: Path, key_renames: Dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    tbl = ds.dataset(path).to_table()
    pdf = tbl.to_pandas()
    pdf = pdf.rename(columns=key_renames)
    for k in key_renames.values():
        if k in pdf.columns:
            pdf[k] = _norm_id_series(pdf[k])
    return pdf

def _coalesce_station_time(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 1 row per (station_id, Datetime). Numeric -> mean; non-numeric -> first."""
    if df.empty:
        return df
    key = ["station_id", "Datetime"]
    num_cols = df.select_dtypes(include="number").columns.difference(key)
    agg: Dict[str, str] = {c: "mean" for c in num_cols}
    for c in df.columns.difference(num_cols.union(key)):
        agg[c] = "first"
    out = (
        df.sort_values(key)
          .groupby(key, as_index=False)
          .agg(agg)
    )
    return out

def _split_by_geo_membership(df: pd.DataFrame, geo_col: str,
                             district_ids: set[str], city_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if geo_col not in df.columns or (not district_ids and not city_ids) or df.empty:
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
    """
    Bring only new (non-key) columns from `right` that don't already exist in `left`.
    Prevents *_x/_y suffix explosions and MergeError due to duplicate names.
    """
    if right.empty:
        return left
    r_non_keys = [c for c in right.columns if c not in keys]
    r_take = [c for c in r_non_keys if c not in left.columns]
    if not r_take:
        return left
    return left.merge(right[keys + r_take], on=keys, how="left")


# ================= loaders (arrow -> pandas slices) =================

def _air_time_range(ds_air: ds.Dataset, air_time_src: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    tbl = ds_air.to_table(columns=[air_time_src])
    pdf = tbl.to_pandas()
    t = _to_dt(pdf[air_time_src])
    return t.min(), t.max()

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
    pdf = tbl.to_pandas()
    pdf = pdf.rename(columns={station_src: "station_id", time_src: "Datetime"})
    pdf["station_id"] = _norm_id_series(pdf["station_id"])
    pdf["Datetime"] = _to_dt(pdf["Datetime"]).dt.floor("h")  # normalize to hourly
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
    met_time_src: str,
    tmin,
    tmax,
) -> pd.DataFrame:
    if ds_met is None:
        return pd.DataFrame()
    flt_time = _maybe_time_filter(ds_met, met_time_src, tmin, tmax)
    flt = flt_time
    if met_geo_src and geo_ids:
        geo_f = _isin_string_field(met_geo_src, geo_ids)
        flt = geo_f if flt is None else (geo_f & flt)
    tbl = ds_met.to_table(filter=flt) if flt is not None else ds_met.to_table()
    if tbl.num_rows == 0:
        return pd.DataFrame()
    pdf = tbl.to_pandas()
    pdf = pdf.rename(columns={met_time_src: "Datetime"})
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
    tbl = ds_wf.to_table(filter=flt) if flt is not None else ds_wf.to_table()
    if tbl.num_rows == 0:
        return pd.DataFrame()
    pdf = tbl.to_pandas()
    if wf_geo_src and wf_geo_src in pdf.columns:
        pdf[wf_geo_src] = _norm_id_series(pdf[wf_geo_src])
    return pdf


# ============================ main ============================

def build_stream():
    # Datasets
    ds_air = _read_dataset_required(PQ_AIR)
    ds_met = _read_dataset_required(PQ_MET) if PQ_MET.exists() else None
    ds_wf  = _read_dataset_required(PQ_WF)  if PQ_WF.exists() else None

    # Header picks
    air_station_src = _pick(ds_air.schema.names, H["airquality"]["station"])
    air_time_src    = _pick(ds_air.schema.names, H["airquality"]["time"])
    if air_station_src is None or air_time_src is None:
        raise KeyError("Could not detect station/time columns in air data.")

    met_geo_src  = _pick(ds_met.schema.names, H["meteorology"]["geo"]) if ds_met else None
    met_time_src = _pick(ds_met.schema.names, H["meteorology"]["time"]) if ds_met else None

    wf_geo_src    = _pick(ds_wf.schema.names, H["weatherforecast"]["geo"]) if ds_wf else None
    wf_future_src = _pick(ds_wf.schema.names, H["weatherforecast"]["future"]) if ds_wf else None
    wf_ftime_src  = _pick(ds_wf.schema.names, H["weatherforecast"]["ftime"]) if ds_wf else None

    # Small dimension tables
    sta = _load_small_table(PQ_STA, {
        _pick(ds.dataset(PQ_STA).schema.names, H["station"]["station"])  or "station_id": "station_id",
        _pick(ds.dataset(PQ_STA).schema.names, H["station"]["district"]) or "district_id": "district_id",
    }) if PQ_STA.exists() else pd.DataFrame()

    dis = _load_small_table(PQ_DIS, {
        _pick(ds.dataset(PQ_DIS).schema.names, H["district"]["district"]) or "district_id": "district_id",
        _pick(ds.dataset(PQ_DIS).schema.names, H["district"]["city"])     or "city_id": "city_id",
    }) if PQ_DIS.exists() else pd.DataFrame()

    cit = _load_small_table(PQ_CIT, {
        _pick(ds.dataset(PQ_CIT).schema.names, H["city"]["city"]) or "city_id": "city_id",
    }) if PQ_CIT.exists() else pd.DataFrame()

    # Membership sets
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

    # Stations and time range from air
    air_minmax = ds_air.to_table(columns=[air_station_src, air_time_src])
    air_df = air_minmax.to_pandas()
    air_df[air_station_src] = _norm_id_series(air_df[air_station_src])
    tmin, tmax = _air_time_range(ds_air, air_time_src)
    station_ids_all = air_df[air_station_src].dropna().astype(str).unique().tolist()
    del air_minmax, air_df

    # station_scope filter
    scope_mode = cfg.get("station_scope", {}).get("mode", "all")
    scope_ids = [str(x) for x in cfg.get("station_scope", {}).get("station_ids", [])]
    if scope_mode == "filter" and scope_ids:
        station_ids = [s for s in station_ids_all if s in set(scope_ids)]
    else:
        station_ids = station_ids_all

    print(f"[build] stations={len(station_ids)}  time=[{tmin} → {tmax}]")

    # Fresh output directory
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, sid in enumerate(station_ids, 1):
        print(f"[{i}/{len(station_ids)}] station={sid} …")

        # 1) air slice (backbone)
        a = _load_air_station(ds_air, sid, air_station_src, air_time_src, tmin, tmax, dedupe_air=False)
        if a.empty:
            print(f"[warn][station={sid}] no air rows, skipping")
            continue

        # 2) attach district/city
        if not sta.empty:
            a = a.merge(sta[["station_id", "district_id"]], on="station_id", how="left")
        if not dis.empty:
            a = a.merge(dis[["district_id", "city_id"]], on="district_id", how="left")
        if not cit.empty and "city_id" in a.columns:
            a = a.merge(cit[["city_id"]], on="city_id", how="left")

        # 3) MET join
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
                        met_dist = (met_dist.sort_values(["district_id","Datetime"])
                                           .drop_duplicates(["district_id","Datetime"], keep="last"))
                    if not met_city.empty:
                        met_city = (met_city.sort_values(["city_id","Datetime"])
                                           .drop_duplicates(["city_id","Datetime"], keep="last"))

                    if not met_dist.empty and "district_id" in a.columns:
                        a = _merge_new_cols(a, met_dist, keys=["district_id","Datetime"])
                    if not met_city.empty and "city_id" in a.columns:
                        a = _merge_new_cols(a, met_city,  keys=["city_id","Datetime"])
                else:
                    met_trim = met_pdf.drop(columns=[c for c in ["geo_id"] if c in met_pdf.columns])
                    a = _merge_new_cols(a, met_trim, keys=["Datetime"])

        # 4) WF join
        if ds_wf:
            geo_ids = []
            if "district_id" in a.columns:
                geo_ids += a["district_id"].dropna().astype(str).tolist()
            if "city_id" in a.columns:
                geo_ids += a["city_id"].dropna().astype(str).tolist()
            geo_ids = list({g for g in geo_ids if g})

            wf_pdf = _load_wf_slice(ds_wf, geo_ids, tmin, tmax, wf_geo_src, wf_future_src, wf_ftime_src)
            if not wf_pdf.empty:
                # choose future-time column → normalize to "Datetime" (hour)
                candidates = []
                if wf_future_src: candidates.append(wf_future_src)
                for c in H["weatherforecast"]["future"]:
                    if c not in candidates:
                        candidates.append(c)
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
                            wf_dist = (wf_dist.sort_values(["district_id","Datetime"])
                                               .drop_duplicates(["district_id","Datetime"], keep="last"))
                        if not wf_city.empty:
                            wf_city = (wf_city.sort_values(["city_id","Datetime"])
                                               .drop_duplicates(["city_id","Datetime"], keep="last"))

                        if not wf_dist.empty and "district_id" in a.columns:
                            a = _merge_new_cols(a, wf_dist, keys=["district_id","Datetime"])
                        if not wf_city.empty and "city_id" in a.columns:
                            a = _merge_new_cols(a, wf_city,  keys=["city_id","Datetime"])
                    else:
                        drop_cols = [c for c in (wf_ftime_src, wf_geo_src, wf_future_src) if c and c in wf_pdf.columns]
                        wf_trim = wf_pdf.drop(columns=drop_cols, errors="ignore")
                        a = _merge_new_cols(a, wf_trim, keys=["Datetime"])

        # Final coalesce (defensive)
        a = _coalesce_station_time(a)

        # Write one parquet per station
        st_dir = OUT_DIR / f"station={sid}"
        st_dir.mkdir(parents=True, exist_ok=True)
        a.to_parquet(st_dir / "data.parquet", index=False)

    print(f"[OK] wrote partitioned dataset to {OUT_DIR}")
    print("You can read it with: ds.dataset(str(OUT_DIR))  or  pandas via pyarrow.dataset to_table().to_pandas()")


# ============================ CLI ============================

if __name__ == "__main__":
    build_stream()