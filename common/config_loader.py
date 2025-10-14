# common/config_loader.py
from __future__ import annotations
import os
from pathlib import Path
from copy import deepcopy
from typing import Dict, List
import yaml

def _deep_update(dst: Dict, src: Dict) -> Dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_cfg() -> Dict:
    """
    Load configs/default.yaml, then merge any overlays listed in CONFIG
    (comma-separated). Missing overlay files are ignored (but warned),
    so callers don’t need to special-case.
    """
    base_path = Path("configs/default.yaml")
    if not base_path.exists():
        raise FileNotFoundError(f"{base_path} not found")
    base = yaml.safe_load(base_path.read_text()) or {}

    cfg_env = os.environ.get("CONFIG", "").strip()
    if not cfg_env:
        return base

    merged = deepcopy(base)
    for raw in cfg_env.split(","):
        p = raw.strip()
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            print(f"[config_loader] WARN overlay not found: {p}")
            continue
        overlay = yaml.safe_load(path.read_text()) or {}
        _deep_update(merged, overlay)
    return merged

def require(cfg: Dict, path: List[str], *, name: str | None = None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            dotted = ".".join(path)
            raise KeyError(f"Missing config key: {dotted}{' ('+name+')' if name else ''}")
        cur = cur[k]
    return cur

def make_abs(root: Path | str, maybe_rel: Path | str) -> Path:
    root = Path(root)
    p = Path(maybe_rel)
    return p if p.is_absolute() else (root / p)