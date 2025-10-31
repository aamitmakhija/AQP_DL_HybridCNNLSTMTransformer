# ingestion/make_keep_config.py (no hard-coding)
from __future__ import annotations
import argparse, json, os, yaml
from pathlib import Path
from copy import deepcopy

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
        raise SystemExit("CONFIG env var is not set.")
    merged: dict = {}
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        with open(p, "r") as f:
            overlay = yaml.safe_load(f) or {}
        merged = deepcopy(overlay) if not merged else _deep_update(merged, overlay)
    return merged

def read_keep_list(p: Path) -> list[str]:
    if not p.exists():
        raise FileNotFoundError(f"keep-list not found: {p}")
    ids = [line.strip() for line in p.read_text().splitlines()
           if line.strip() and not line.startswith("#")]
    if not ids:
        raise ValueError(f"keep-list is empty: {p}")
    return ids

def main():
    cfg = _load_cfg()
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    reports_cfg = cfg.get("reports", {})
    keep_file = artifacts_dir / reports_cfg.get("stations_keep_txt", "stations_keep.txt")
    out_file  = Path(cfg.get("paths", {}).get("keep_config", "configs/keep.yaml"))

    ap = argparse.ArgumentParser(description="Generate keep.yaml from keep list")
    ap.add_argument("--keep-list", type=Path, default=keep_file)
    ap.add_argument("--out-config", type=Path, default=out_file)
    args = ap.parse_args()

    keep_ids = read_keep_list(args.keep_list)
    cfg_overlay = {"station_scope": {"mode": "filter", "station_ids": keep_ids}}

    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_config, "w") as f:
        yaml.safe_dump(cfg_overlay, f, sort_keys=False)

    print(f"[keep] wrote {args.out_config} with {len(keep_ids)} stations")
    print(json.dumps({"out_config": str(args.out_config), "count": len(keep_ids)}))

if __name__ == "__main__":
    main()