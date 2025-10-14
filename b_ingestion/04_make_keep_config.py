# ingestion/make_keep_config.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml

DEF_KEEP = Path("experiments/artifacts/stations_keep.txt")
DEF_OUT  = Path("configs/keep.yaml")

def read_keep_list(p: Path) -> list[str]:
    if not p.exists():
        raise FileNotFoundError(f"keep-list not found: {p}")
    ids = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(str(s))
    if not ids:
        raise ValueError(f"keep-list is empty: {p}")
    return ids

def main():
    ap = argparse.ArgumentParser(description="Create overlay config with station keep-list")
    # new args
    ap.add_argument("--keep-list", type=Path, default=DEF_KEEP,
                    help=f"path to stations_keep.txt (default: {DEF_KEEP})")
    ap.add_argument("--out-config", type=Path, default=DEF_OUT,
                    help=f"where to write overlay YAML (default: {DEF_OUT})")
    # legacy arg (ignored, for compatibility)
    ap.add_argument("--threshold", type=float, default=None,
                    help="ignored; kept for backward compatibility")
    args = ap.parse_args()

    keep_ids = read_keep_list(args.keep_list)

    cfg_overlay = {
        "station_scope": {
            "mode": "filter",
            "station_ids": keep_ids,
        }
    }

    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_config, "w") as f:
        yaml.safe_dump(cfg_overlay, f, sort_keys=False)

    print(f"[keep] wrote {args.out_config} with {len(keep_ids)} stations")
    # small machine-readable echo if you ever want to parse
    print(json.dumps({"out_config": str(args.out_config), "count": len(keep_ids)}))

if __name__ == "__main__":
    main()