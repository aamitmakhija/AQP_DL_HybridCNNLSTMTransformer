# d_modelprep/CC_seq_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict, Union
import os, sys, time, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info

# ---------------- config loader (YAML, respects CONFIG overlays) ----------------
try:
    from common.config_loader import load_cfg  # preferred
except Exception:
    import yaml
    from copy import deepcopy
    def _deep_update(dst: dict, src: dict | None) -> dict:
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst
    def _read_yaml(p: str | Path) -> dict:
        with Path(p).open("r") as f:
            return yaml.safe_load(f) or {}
    def load_cfg() -> dict:
        base = _read_yaml("configs/default.yaml")
        cfg_env = os.environ.get("CONFIG", "").strip()
        if cfg_env:
            merged = deepcopy(base)
            for path in [s.strip() for s in cfg_env.split(",") if s.strip()]:
                _deep_update(merged, _read_yaml(path))
            return merged
        if sys.platform == "darwin":
            cpu = Path("configs/cpu.yaml")
            if cpu.exists():
                merged = deepcopy(base)
                _deep_update(merged, _read_yaml(cpu))
                return merged
        return base

# ---------------- small utils ----------------
def _req(d: dict, path: Iterable[str]) -> object:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise KeyError("configs: missing " + ".".join(path))
        cur = cur[k]
    return cur

def _opt(d: dict, path: Iterable[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _resolve_seq_root(cfg: dict) -> Path:
    art = Path(_req(cfg, ["paths", "artifacts_dir"]))
    out_raw = Path(_req(cfg, ["sequence", "out_dir"]))
    if out_raw.is_absolute():
        return out_raw
    art_parts, raw_parts = Path(art).parts, out_raw.parts
    if len(raw_parts) >= len(art_parts) and tuple(raw_parts[:len(art_parts)]) == art_parts:
        return out_raw
    return art / out_raw

def _read_manifest(seq_root: Path) -> dict | None:
    p = seq_root / "manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

def _detect_horizons(cfg: dict, seq_root: Path) -> List[int]:
    raw_h = _opt(cfg, ["sequence", "horizon"], None)
    if raw_h is not None:
        return [int(h) for h in (raw_h if isinstance(raw_h, (list, tuple)) else [raw_h])]
    man = _read_manifest(seq_root)
    if man and isinstance(man.get("horizons"), list) and man["horizons"]:
        return [int(h) for h in man["horizons"]]
    hs = []
    for d in (seq_root / "train").glob("h=*"):
        try: hs.append(int(str(d.name).split("=", 1)[1]))
        except Exception: pass
    return sorted(set(hs))

def _path_has_shards(p: Path) -> bool:
    return any(p.glob("shard_*.npz"))

def _resolve_split_dir_for_h(seq_root: Path, split_label: str, H: int) -> Path:
    base = seq_root / split_label
    if "h=" in split_label:
        return base
    if _path_has_shards(base):
        return base
    cand = base / f"h={H}"
    if _path_has_shards(cand) or cand.exists():
        return cand
    return cand

def _derive_seed(cfg: dict) -> int:
    env = os.environ.get("SEED")
    if env is not None and str(env).strip() != "":
        return int(env)
    yml = _opt(cfg, ["train", "seed"], None)
    if yml is not None:
        return int(yml)
    return int(np.random.SeedSequence().entropy) & 0xFFFFFFFF

def _worker_init_fn(worker_id: int):
    info = get_worker_info()
    if info is None:
        return
    base = int(np.random.SeedSequence().entropy) & 0xFFFFFFFF
    np.random.seed(base + worker_id)

def _auto_pin(pin_memory_cfg: bool) -> bool:
    if pin_memory_cfg:
        return True
    try:
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except Exception:
        return False

def _float_dtype(cfg: dict):
    dt = str(_opt(cfg, ["train", "float_dtype"], "float32")).lower()
    return torch.float16 if dt in ("fp16", "float16", "half") else torch.float32

# ---------------- Dataset ----------------
class NpzSeqDataset(Dataset):
    """
    NPZ shard format:
      - X: [N, T, F] float
      - y: [N] or [N, H] float
      - sid: [N] optional
    """
    def __init__(self, root: Path | str, include_sid: bool = False, enforce_float32: bool = True):
        self.root = Path(root)
        self.include_sid = bool(include_sid)
        self.enforce_float32 = bool(enforce_float32)
        self.files: List[Path] = sorted(self.root.glob("shard_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz in {self.root}")
        with np.load(self.files[0], mmap_mode="r", allow_pickle=False) as z0:
            if "X" not in z0 or "y" not in z0:
                raise KeyError(f"{self.files[0]} missing 'X' or 'y'")
            X0, y0 = z0["X"], z0["y"]
            if X0.ndim != 3:
                raise ValueError(f"X rank must be 3, got {X0.shape}")
            self.T, self.F = int(X0.shape[1]), int(X0.shape[2])
            self.H = 1 if y0.ndim == 1 else (int(y0.shape[1]) if y0.ndim == 2 else None)
            if self.H is None:
                raise ValueError(f"y rank must be 1 or 2, got {y0.shape}")
        self._index: List[Tuple[int, int]] = []
        self._sizes: List[int] = []
        for i, f in enumerate(self.files):
            with np.load(f, mmap_mode="r", allow_pickle=False) as z:
                n = int(z["X"].shape[0])
            if n == 0: continue
            self._sizes.append(n)
            self._index.extend((i, j) for j in range(n))
        if not self._index:
            raise RuntimeError(f"All shards in {self.root} are empty")
        self._cache: Optional[Tuple[int, np.lib.npyio.NpzFile]] = None
        print(f"[Dataset] {self.root} → {len(self._index):,} samples (T={self.T}, F={self.F}, H={self.H}, shards={len(self.files)})")

    def __len__(self) -> int: return len(self._index)

    def __del__(self):
        try:
            if self._cache is not None:
                _, h = self._cache
                if hasattr(h, "close"):
                    h.close()
        except Exception:
            pass

    def _get_file(self, i: int):
        if self._cache is not None and self._cache[0] == i:
            return self._cache[1]
        try:
            if self._cache is not None:
                _, h = self._cache
                if hasattr(h, "close"):
                    h.close()
        except Exception:
            pass
        handle = np.load(self.files[i], mmap_mode="r", allow_pickle=False)
        self._cache = (i, handle)
        return handle

    def __getitem__(self, idx: int):
        fi, row = self._index[idx]
        z = self._get_file(fi)
        X = z["X"][row]       # [T,F]
        y_raw = z["y"][row]   # scalar | [H] | [1,H]
        y = np.asarray(y_raw)
        if y.ndim == 0: y = y[None]
        elif y.ndim == 2: y = y.reshape(-1)
        if y.ndim != 1: y = y.reshape(-1)
        if y.shape[0] != self.H:
            y = y[: self.H] if y.shape[0] > self.H else np.pad(y, (0, self.H - y.shape[0]), mode="edge")
        x_t = torch.as_tensor(X, dtype=torch.float32) if self.enforce_float32 else torch.from_numpy(X)
        y_t = torch.as_tensor(y, dtype=torch.float32) if self.enforce_float32 else torch.from_numpy(y)
        if self.include_sid:
            sid = ""
            if "sid" in z.files:
                sv = z["sid"][row]
                sid = sv.decode("utf-8") if isinstance(sv, (bytes, bytearray)) else str(sv)
            return x_t, y_t, sid
        return x_t, y_t

# ---------------- Sampler ----------------
class GroupedBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, seed: int = 42):
        self.ds, self.bs, self.shuffle, self.seed, self._epoch = dataset, int(batch_size), bool(shuffle), int(seed), 0
    def set_epoch(self, epoch: int): self._epoch = int(epoch)
    def __iter__(self):
        idxs = np.arange(len(self.ds))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch); rng.shuffle(idxs)
        for i in range(0, len(idxs), self.bs): yield idxs[i:i+self.bs].tolist()
    def __len__(self): return (len(self.ds) + self.bs - 1) // self.bs

def _collate_xy_sid(batch):
    xs, ys, sids = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(sids)

# ---------------- Core builders ----------------
def _build_pair(
    split_dir: Path, batch_size: int, num_workers: int, pin_memory: bool,
    persistent_workers: bool, include_sid: bool, shuffle: bool, seed: int, drop_last: bool,
    float_dtype: torch.dtype = torch.float32
) -> Tuple[DataLoader, int, int, int]:
    if not split_dir.exists() or not any(split_dir.glob("shard_*.npz")):
        raise FileNotFoundError(f"No shards found under {split_dir}. Run windowing first.")
    ds = NpzSeqDataset(split_dir, include_sid=include_sid, enforce_float32=(float_dtype==torch.float32))
    sampler = GroupedBatchSampler(ds, batch_size, shuffle=shuffle, seed=seed)
    collate = _collate_xy_sid if include_sid else None
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=_auto_pin(pin_memory),
        persistent_workers=(persistent_workers and num_workers > 0),
        collate_fn=collate,
        drop_last=drop_last,
        worker_init_fn=_worker_init_fn,
        prefetch_factor=(2 if num_workers > 0 else None)
    )
    return loader, ds.F, ds.H, ds.T

# ---------------- Public API ----------------
def make_loaders_for_horizon(cfg: dict, H: int, *, include_sid: bool = False):
    """Build train/val loaders for a specific horizon H from YAML."""
    seq_root = _resolve_seq_root(cfg)
    tr_label = str(_req(cfg, ["sequence", "train_split"]))
    va_label = str(_req(cfg, ["sequence", "val_split"]))
    tr_dir   = _resolve_split_dir_for_h(seq_root, tr_label, H)
    va_dir   = _resolve_split_dir_for_h(seq_root, va_label, H)

    batch_size  = int(_opt(cfg, ["train", "batch_size"], 64))
    num_workers = int(_opt(cfg, ["train", "num_workers"], 0))
    pin_memory  = bool(_opt(cfg, ["train", "pin_memory"], False))
    persistent  = bool(_opt(cfg, ["train", "persistent_workers"], False))
    seed        = _derive_seed(cfg)
    drop_last   = bool(_opt(cfg, ["train", "drop_last"], False))
    float_dtype = _float_dtype(cfg)

    t0 = time.time()
    tr_loader, F, H_out, T = _build_pair(tr_dir, batch_size, num_workers, pin_memory, persistent,
                                         include_sid, True,  seed, drop_last, float_dtype=float_dtype)
    va_loader, _,     _, _ = _build_pair(va_dir, batch_size, num_workers, pin_memory, persistent,
                                         include_sid, False, seed, False, float_dtype=float_dtype)
    print(f"[make_loaders_for_horizon] H={H} → {tr_dir} | {va_dir} (bs={batch_size}) built in {time.time()-t0:.2f}s")
    return tr_loader, va_loader, F, H_out, T

def make_loaders_from_cfg(cfg: dict, *, include_sid: bool = False) -> Union[
    Tuple[DataLoader, DataLoader, int, int, int],
    Dict[int, Tuple[DataLoader, DataLoader, int, int, int]]
]:
    """
    If multiple horizons are configured/available, returns {H: (tr, va, F, H, T)}.
    If exactly one horizon, returns a single tuple for convenience.
    """
    seq_root = _resolve_seq_root(cfg)
    horizons = _detect_horizons(cfg, seq_root)
    if not horizons:
        raise FileNotFoundError(f"No horizons found under {seq_root}. Run windowing step first.")
    if len(horizons) == 1:
        H = horizons[0]
        return make_loaders_for_horizon(cfg, H, include_sid=include_sid)
    out: Dict[int, Tuple[DataLoader, DataLoader, int, int, int]] = {}
    for H in horizons:
        out[H] = make_loaders_for_horizon(cfg, H, include_sid=include_sid)
    return out

def make_test_loader_for_horizon(cfg: dict, H: int, *, include_sid: bool = True):
    seq_root = _resolve_seq_root(cfg)
    te_label = str(_req(cfg, ["sequence", "test_split"]))
    te_dir   = _resolve_split_dir_for_h(seq_root, te_label, H)

    batch_size  = int(_opt(cfg, ["eval", "batch_size"], _opt(cfg, ["train", "batch_size"], 32)))
    num_workers = int(_opt(cfg, ["eval", "num_workers"], _opt(cfg, ["train", "num_workers"], 0)))
    pin_memory  = bool(_opt(cfg, ["eval", "pin_memory"], _opt(cfg, ["train", "pin_memory"], False)))
    persistent  = bool(_opt(cfg, ["eval", "persistent_workers"], _opt(cfg, ["train", "persistent_workers"], False)))
    float_dtype = _float_dtype(cfg)

    if not te_dir.exists() or not any(te_dir.glob("shard_*.npz")):
        raise FileNotFoundError(f"No test shards under {te_dir}.")
    ds = NpzSeqDataset(te_dir, include_sid=include_sid, enforce_float32=(float_dtype==torch.float32))
    sampler = GroupedBatchSampler(ds, batch_size, shuffle=False, seed=_derive_seed(cfg))
    collate = _collate_xy_sid if include_sid else None
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=_auto_pin(pin_memory),
        persistent_workers=(persistent and num_workers > 0),
        collate_fn=collate,
        worker_init_fn=_worker_init_fn,
        prefetch_factor=(2 if num_workers > 0 else None)
    )
    print(f"[make_test_loader_for_horizon] H={H} → {te_dir} (bs={batch_size})")
    return loader, ds.F, ds.H, ds.T

def make_test_loader_from_cfg(cfg: dict, *, include_sid: bool = True) -> Union[
    Tuple[DataLoader, int, int, int],
    Dict[int, Tuple[DataLoader, int, int, int]]
]:
    """
    If multiple horizons are configured/available, returns {H: (loader, F, H, T)}.
    If exactly one horizon, returns a single tuple for convenience.
    """
    seq_root = _resolve_seq_root(cfg)
    horizons = _detect_horizons(cfg, seq_root)
    if not horizons:
        raise FileNotFoundError(f"No horizons found under {seq_root}. Run windowing step first.")
    if len(horizons) == 1:
        H = horizons[0]
        return make_test_loader_for_horizon(cfg, H, include_sid=include_sid)
    out: Dict[int, Tuple[DataLoader, int, int, int]] = {}
    for H in horizons:
        out[H] = make_test_loader_for_horizon(cfg, H, include_sid=include_sid)
    return out

# ---------------- Convenience ----------------
def build_train_val(*, include_sid: bool = False):
    cfg = load_cfg()
    return make_loaders_from_cfg(cfg, include_sid=include_sid)

def build_test(*, include_sid: bool = True):
    cfg = load_cfg()
    return make_test_loader_from_cfg(cfg, include_sid=include_sid)

# ---------------- Debug entrypoint ----------------
if __name__ == "__main__":
    cfg = load_cfg()
    loaders = make_loaders_from_cfg(cfg, include_sid=False)
    if isinstance(loaders, dict):
        for H, (tr, va, F, Hout, T) in loaders.items():
            print(f"[debug] H={H} → F={F} H={Hout} T={T} | batches=({len(tr)}, {len(va)})")
    else:
        tr, va, F, Hout, T = loaders
        print(f"[debug] F={F} H={Hout} T={T} | batches=({len(tr)}, {len(va)})")
    tests = make_test_loader_from_cfg(cfg, include_sid=True)
    if isinstance(tests, dict):
        for H, (te, F2, H2, T2) in tests.items():
            print(f"[debug test] H={H} → F={F2} H={H2} T={T2} | batches={len(te)}")
    else:
        te, F2, H2, T2 = tests
        print(f"[debug test] F={F2} H={H2} T={T2} | batches={len(te)}")