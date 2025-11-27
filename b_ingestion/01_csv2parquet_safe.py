# src/csv2parquet_safe.py
from __future__ import annotations
import os
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Union

import yaml
import pandas as pd


# ---------- config helpers ----------
def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_cfg() -> dict:
    """
    Load configuration strictly from CONFIG env var (comma-separated YAML paths).
    """
    cfg_env = os.environ.get("CONFIG")
    if not cfg_env:
        raise SystemExit(
            "CONFIG env var is not set. Provide comma-separated YAML files, "
            "e.g. CONFIG='configs/default.yaml,configs/keep.yaml'"
        )
    merged: Dict[str, Any] = {}
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        with open(p, "r") as f:
            overlay = yaml.safe_load(f) or {}
        if not merged:
            merged = deepcopy(overlay)
        else:
            _deep_update(merged, overlay)
    return merged


def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- IO helpers ----------
def _read_csv_with_engines(
    path: Path,
    engines: Iterable[str] | None,
    common_kwargs: Dict[str, Any],
    per_file_kwargs: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Try engines in order; if none provided, let pandas choose."""
    kwargs = {**common_kwargs, **(per_file_kwargs or {})}
    if not engines:
        return pd.read_csv(path, **kwargs)

    last_err: Exception | None = None
    for eng in engines:
        try:
            return pd.read_csv(path, engine=eng, **kwargs)
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def _build_output_name(
    key: str,
    pattern: str,
    ext: str | None,
    per_file_override: str | None = None,
) -> str:
    """Name comes from per-file override or global pattern."""
    if per_file_override:
        return per_file_override
    e = (ext or "").strip()
    name = pattern.format(key=key, ext=e)
    if e == "" and name.endswith("."):
        name = name[:-1]
    return name


def _write_frame(df: pd.DataFrame, out_path: Path, fmt: str, writer_kwargs: Dict[str, Any]) -> None:
    """Generic writer calling DataFrame.to_{fmt}()."""
    method_name = f"to_{fmt}"
    if not hasattr(df, method_name):
        raise ValueError(f"Unsupported output format '{fmt}'")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if "index" not in writer_kwargs:
        writer_kwargs = {**writer_kwargs, "index": False}

    getattr(df, method_name)(out_path.as_posix(), **writer_kwargs)


# ---------- main ----------
def main():
    cfg = _load_cfg()
    paths_cfg: Dict[str, Any] = cfg.get("paths", {})
    data_files_cfg: Dict[str, Any] = cfg.get("data_files", {})
    output_cfg: Dict[str, Any] = cfg.get("output", {})
    read_csv_cfg: Dict[str, Any] = cfg.get("read_csv", {})

    data_dir = paths_cfg.get("data_dir")
    artifacts_dir = paths_cfg.get("artifacts_dir")
    if not data_dir or not artifacts_dir:
        raise SystemExit("paths.data_dir and paths.artifacts_dir must be set in config.")

    DATA_DIR = Path(data_dir)
    ART_DIR = _ensure_dir(artifacts_dir)

    engine_preference: List[str] | None = read_csv_cfg.get("engine_preference")
    common_read_kwargs: Dict[str, Any] = read_csv_cfg.get("kwargs", {})

    fmt: str = output_cfg.get("format")
    if not fmt:
        raise SystemExit("output.format must be set (e.g., parquet, feather, csv).")

    ext: str | None = output_cfg.get("extension")
    naming_pattern: str = output_cfg.get("naming", {}).get("pattern", "{key}.{ext}")
    writer_kwargs: Dict[str, Any] = output_cfg.get("writer_kwargs", {})
    subdir: str | None = output_cfg.get("subdir")

    target_dir = ART_DIR / subdir if subdir else ART_DIR
    _ensure_dir(target_dir)

    print(f"[csv2parquet] CONFIG={os.environ.get('CONFIG')}")
    print(f"[csv2parquet] DATA_DIR={DATA_DIR}  →  ART_DIR={target_dir}")

    written, skipped, errors = [], [], []

    for key, spec in data_files_cfg.items():
        if isinstance(spec, str):
            src_rel = spec
            per_file_read_kwargs = None
            per_file_output_name = None
        elif isinstance(spec, dict):
            src_rel = spec.get("path")
            _rc = spec.get("read_csv", {})
            per_file_read_kwargs = _rc.get("kwargs", _rc if isinstance(_rc, dict) else {})
            per_file_output_name = spec.get("output_name")
        else:
            print(f"[WARN] data_files.{key} malformed; skipping")
            skipped.append(key)
            continue

        if not src_rel:
            print(f"[WARN] data_files.{key} missing 'path'; skipping")
            skipped.append(key)
            continue

        src = DATA_DIR / src_rel
        if not src.exists():
            print(f"[WARN] {src} missing; skipping")
            skipped.append(key)
            continue

        try:
            df = _read_csv_with_engines(
                src,
                engines=engine_preference,
                common_kwargs=common_read_kwargs,
                per_file_kwargs=per_file_read_kwargs,
            )
            out_name = _build_output_name(
                key=key,
                pattern=naming_pattern,
                ext=ext,
                per_file_override=per_file_output_name,
            )
            out_path = target_dir / out_name
            _write_frame(df, out_path, fmt=fmt, writer_kwargs=writer_kwargs)
            written.append(out_name)
        except Exception as e:
            errors.append(f"{key}: {e.__class__.__name__}: {e}")
            continue

    if written:
        print(f"[OK] wrote {', '.join(written)} in {target_dir}")
    if skipped:
        print(f"[INFO] skipped (not found or misconfigured): {', '.join(skipped)}")
    if errors:
        print(f"[ERR] failures ({len(errors)}):")
        for msg in errors:
            print("  -", msg)


if __name__ == "__main__":
    main()