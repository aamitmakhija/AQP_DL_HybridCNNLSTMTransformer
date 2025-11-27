# common/config_loader.py
from __future__ import annotations

import os
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict

import yaml

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any] | None) -> Dict[str, Any]:
    """Recursively merge src into dst (src wins)."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_cfg() -> Dict[str, Any]:
    """
    Load configs/default.yaml, then deep-merge any overlays in the CONFIG
    env var (comma-separated paths, relative or absolute). Missing overlays
    are warned and skipped.
    """
    base = _read_yaml(Path("configs/default.yaml"))

    cfg_env = os.environ.get("CONFIG", "").strip()
    if not cfg_env:
        return base

    merged = deepcopy(base)
    for raw in cfg_env.split(","):
        p = raw.strip()
        if not p:
            continue
        path = Path(p)
        if not path.is_absolute():
            path = Path(p)  # keep relative to CWD
        if not path.exists():
            print(f"[config_loader] WARN overlay not found: {p}")
            continue
        overlay = _read_yaml(path)
        _deep_update(merged, overlay)
    return merged


def require(cfg: Dict[str, Any], path: list[str], *, name: str | None = None) -> Any:
    """Traverse cfg by keys in `path`; raise if any segment is missing."""
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            dotted = ".".join(path)
            raise KeyError(f"Missing config key: {dotted}{(' (' + name + ')') if name else ''}")
        cur = cur[k]
    return cur


def make_abs(root: Path | str, maybe_rel: Path | str) -> Path:
    """Join `maybe_rel` to `root` if not already absolute."""
    root = Path(root)
    p = Path(maybe_rel)
    return p if p.is_absolute() else (root / p)