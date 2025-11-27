# ingestion/make_keep_config.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
from common.config_loader import load_cfg  # overlay-aware


# ----------------------- strict helpers -----------------------

def _req(d: Dict, path: List[str]):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise SystemExit("configs: missing " + ".".join(path))
        cur = cur[k]
    return cur

def _norm_station_id(s: str) -> str:
    return str(s).strip().replace("\n", "").removesuffix(".0")

def _dedup_stable(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _read_keep_list(p: Path) -> List[str]:
    if not p.exists():
        raise SystemExit(f"keep-list not found: {p}")
    ids: List[str] = []
    for line in p.read_text().splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        ids.append(_norm_station_id(t))
    ids = _dedup_stable(ids)
    if not ids:
        raise SystemExit(f"keep-list is empty: {p}")
    return ids


# ----------------------------- main -----------------------------

def main():
    cfg = load_cfg()  # respects CONFIG overlays

    # Required config keys (no hardcoded fallbacks)
    art_dir      = Path(_req(cfg, ["paths", "artifacts_dir"]))
    keep_txt_rel = _req(cfg, ["reports", "stations_keep_txt"])   # e.g., "stations_keep.txt"
    out_keep_yml = Path(_req(cfg, ["paths", "keep_config"]))     # e.g., "configs/keep.yaml"

    default_keep_file = art_dir / keep_txt_rel

    ap = argparse.ArgumentParser(description="Generate keep.yaml from stations_keep.txt")
    ap.add_argument("--keep-list", type=Path, default=default_keep_file, help="Path to stations_keep.txt")
    ap.add_argument("--out-config", type=Path, default=out_keep_yml, help="Destination keep.yaml")
    args = ap.parse_args()

    keep_ids = _read_keep_list(args.keep_list)

    overlay = {"station_scope": {"mode": "filter", "station_ids": keep_ids}}

    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_config, "w") as f:
        yaml.safe_dump(overlay, f, sort_keys=False)

    print(f"[keep] wrote {args.out_config} with {len(keep_ids)} stations")
    print(json.dumps({"out_config": str(args.out_config), "count": len(keep_ids)}))


if __name__ == "__main__":
    main()