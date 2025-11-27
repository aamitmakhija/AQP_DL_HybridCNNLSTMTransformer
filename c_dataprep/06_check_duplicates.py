#!/usr/bin/env python3
# c_dataprep/06_check_duplicates.py  (YAML-strict, manifest/alias-aware, fallback split names)
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
from common.config_loader import load_cfg

# ---------- utils ----------
def _req(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise KeyError("configs: missing " + ".".join(path))
        cur = cur[k]
    return cur

def _under(root: Path, maybe: str) -> Path:
    p = Path(maybe); return p if p.is_absolute() else (root / p)

def _read_df(path: Path, fmt: str) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    f = fmt.lower()
    if f=="parquet": return pd.read_parquet(path)
    if f=="feather": return pd.read_feather(path)
    if f=="csv":     return pd.read_csv(path)
    raise SystemExit(f"Unsupported format: {fmt!r}")

def _norm_station_id_series(s: pd.Series, enable: bool) -> pd.Series:
    if not enable or s.empty: return s
    out = s.astype("object").copy()
    m = pd.notna(out)
    out.loc[m] = (
        pd.Series(out.loc[m], dtype="object").astype(str)
        .str.strip().str.replace(r"\.0$", "", regex=True)
    )
    return out

def _manifest_cols(art_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    feats_dir = _under(art_dir, _req(cfg, ["paths", "features_dir"]))
    out_cfg   = _req(cfg, ["features", "output"])
    man_file  = out_cfg.get("manifest_file")
    if man_file:
        p = feats_dir / man_file
        if p.exists():
            try:
                m = json.loads(p.read_text())
                ids = m.get("id_cols") or m.get("id")
                if m.get("time_col") and ids:
                    return {"time_col": m["time_col"], "id_col": ids[0]}
            except Exception:
                pass
    return {}

def _resolve_id_time(df: pd.DataFrame, cfg: Dict[str, Any], art_dir: Path) -> tuple[Optional[str], Optional[str]]:
    if df.empty: return None, None
    man = _manifest_cols(art_dir, cfg)
    man_id, man_time = man.get("id_col"), man.get("time_col")

    id_primary   = str(_req(cfg, ["data", "id_col"]))
    time_primary = str(_req(cfg, ["data", "time_col"]))
    headers      = (cfg.get("data", {}) or {}).get("headers", {}) or {}
    id_aliases   = list(headers.get("id", []))
    time_aliases = list(headers.get("time", []))

    by_lower = {c.lower(): c for c in df.columns}
    def pick(primary: str, aliases: List[str]) -> Optional[str]:
        if primary.lower() in by_lower: return by_lower[primary.lower()]
        for cand in aliases or []:
            k = str(cand).lower()
            if k in by_lower: return by_lower[k]
        if primary.lower() not in by_lower:
            for k in ("id","ID"):
                if k.lower() in by_lower: return by_lower[k.lower()]
        return None

    id_col   = man_id   if man_id   in df.columns else pick(id_primary, id_aliases)
    time_col = man_time if man_time in df.columns else pick(time_primary, time_aliases)
    return id_col, time_col

def _dup_report(df: pd.DataFrame, name: str, id_col: str, time_col: str,
                normalize_ids: bool, csv_out_dir: Path | None = None, csv_rows_limit: int = 100_000) -> None:
    if df.empty: print(f"[{name}] empty or missing"); return
    if {id_col, time_col} - set(df.columns):
        have = sorted(set(df.columns) & {id_col, time_col})
        print(f"[{name}] missing key columns; has={have}"); return

    dfx = df[[id_col, time_col]].copy()
    dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce")
    dfx[id_col]   = _norm_station_id_series(dfx[id_col], normalize_ids)

    total = len(dfx)
    mask = dfx.duplicated(subset=[id_col, time_col], keep=False)
    n_dups = int(mask.sum())
    if n_dups == 0:
        print(f"[{name}] rows={total:,}  dup_rows=0  dup_key_groups=0"); return

    dup_pairs = dfx.loc[mask, [id_col, time_col]]
    n_groups = dup_pairs.drop_duplicates().shape[0]
    print(f"[{name}] rows={total:,}  dup_rows={n_dups:,}  dup_key_groups={n_groups:,}")

    if csv_out_dir is not None:
        csv_out_dir.mkdir(parents=True, exist_ok=True)
        joined = df.loc[mask].copy()
        if len(joined) > csv_rows_limit: joined = joined.head(csv_rows_limit)
        out_path = csv_out_dir / f"{name.replace('/','_')}_duplicates.csv"
        joined.to_csv(out_path, index=False)
        print(f"  [csv] wrote duplicate rows sample → {out_path} (rows={len(joined):,})")

# ---------- main ----------
def main():
    cfg = load_cfg()

    # paths
    art_dir      = _under(Path("."), _req(cfg, ["paths", "artifacts_dir"]))
    splits_dir   = _under(art_dir,     _req(cfg, ["paths", "splits_dir"]))
    features_dir = _under(art_dir,     _req(cfg, ["paths", "features_dir"]))
    scaled_dir   = _under(art_dir,     _req(cfg, ["paths", "features_scaled_dir"]))

    # split IO (fallback-friendly)
    out_cfg    = _req(cfg, ["output"])
    split_fmt  = str(_req(out_cfg, ["format"])).lower()
    names      = (out_cfg.get("split_filenames") or
                  {"train": f"train.{split_fmt}", "val": f"val.{split_fmt}", "test": f"test.{split_fmt}"})
    p_train    = splits_dir / str(names["train"])
    p_val      = splits_dir / str(names["val"])
    p_test     = splits_dir / str(names["test"])

    # features IO
    feats_out  = _req(cfg, ["features", "output"])
    feats_fmt  = str(_req(feats_out, ["format"])).lower()
    feats_file = str(_req(feats_out, ["features_file"]))
    feats_path = features_dir / feats_file

    # scaled splits IO (reuse names; use configured scaling.output_format)
    scaled_fmt = str(_req(cfg, ["scaling", "output_format"])).lower()
    p_train_s  = scaled_dir / str(names["train"])
    p_val_s    = scaled_dir / str(names["val"])
    p_test_s   = scaled_dir / str(names["test"])

    # reporting
    reports_cfg = cfg.get("reports", {}) or {}
    summary_out = art_dir / reports_cfg.get("duplicates_summary", "duplicates_summary.json")
    write_csv   = bool(reports_cfg.get("write_duplicates_csv", False))
    csv_dir     = (art_dir / reports_cfg.get("duplicates_csv_dir", "duplicates_csv")) if write_csv else None
    csv_limit   = int(reports_cfg.get("duplicates_csv_max_rows", 100_000))
    normalize_ids = bool(cfg.get("checks", {}).get("normalize_ids", True))

    # run
    for name, pth, fmt in [
        ("splits/train", p_train, split_fmt),
        ("splits/val",   p_val,   split_fmt),
        ("splits/test",  p_test,  split_fmt),
    ]:
        df = _read_df(pth, fmt)
        id_col, time_col = _resolve_id_time(df, cfg, art_dir)
        if id_col and time_col:
            _dup_report(df, name, id_col, time_col, normalize_ids, csv_dir, csv_limit)
        else:
            print(f"[{name}] unable to resolve id/time columns; skipping")

    feats = _read_df(feats_path, feats_fmt)
    id_col, time_col = _resolve_id_time(feats, cfg, art_dir)
    if id_col and time_col:
        _dup_report(feats, "features/dataset_features", id_col, time_col, normalize_ids, csv_dir, csv_limit)
    else:
        print("[features/dataset_features] unable to resolve id/time columns; skipping")

    for name, pth, fmt in [
        ("features_scaled/train", p_train_s, scaled_fmt),
        ("features_scaled/val",   p_val_s,   scaled_fmt),
        ("features_scaled/test",  p_test_s,  scaled_fmt),
    ]:
        df = _read_df(pth, fmt)
        id_col, time_col = _resolve_id_time(df, cfg, art_dir)
        if id_col and time_col:
            _dup_report(df, name, id_col, time_col, normalize_ids, csv_dir, csv_limit)
        else:
            print(f"[{name}] unable to resolve id/time columns; skipping")

    # JSON summary
    summary: Dict[str, Dict[str, int | None]] = {}
    def collect(df: pd.DataFrame, key: str, fmt: str):
        idc, tc = _resolve_id_time(df, cfg, art_dir)
        if df.empty or not idc or not tc:
            summary[key] = {"rows": int(len(df)), "dup_rows": None, "dup_key_groups": None}; return
        dfx = df[[idc, tc]].copy()
        dfx[tc] = pd.to_datetime(dfx[tc], errors="coerce")
        dfx[idc] = _norm_station_id_series(dfx[idc], normalize_ids)
        mask = dfx.duplicated(subset=[idc, tc], keep=False)
        summary[key] = {"rows": int(len(dfx)),
                        "dup_rows": int(mask.sum()),
                        "dup_key_groups": int(dfx.loc[mask, [idc, tc]].drop_duplicates().shape[0])}

    collect(_read_df(p_train,  split_fmt), "splits/train", split_fmt)
    collect(_read_df(p_val,    split_fmt), "splits/val",   split_fmt)
    collect(_read_df(p_test,   split_fmt), "splits/test",  split_fmt)
    collect(feats, "features/dataset_features", feats_fmt)
    collect(_read_df(p_train_s, scaled_fmt), "features_scaled/train", scaled_fmt)
    collect(_read_df(p_val_s,   scaled_fmt), "features_scaled/val",   scaled_fmt)
    collect(_read_df(p_test_s,  scaled_fmt), "features_scaled/test",  scaled_fmt)

    summary_out.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] wrote {summary_out}")

if __name__ == "__main__":
    main()