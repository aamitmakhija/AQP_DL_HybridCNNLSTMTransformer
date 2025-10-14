# d_modelprep/CC_seq_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# ---------------- Dataset ----------------
class NpzSeqDataset(Dataset):
    """
    Memory-friendly loader for NPZ shards with keys:
      - X: [N, T, F]
      - y: [N] or [N, H]
    Returns:
      x: torch.float32 [T, F]
      y: torch.float32 [H]  (H==1 for single-horizon)
    """
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.files: List[Path] = sorted(self.root.glob("shard_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz in {self.root}")

        # Probe first shard for dimensions
        with np.load(self.files[0], mmap_mode="r") as z0:
            X0 = z0["X"]
            y0 = z0["y"]
            self.T = int(X0.shape[1])
            self.F = int(X0.shape[2])
            self.H = 1 if y0.ndim == 1 else int(y0.shape[1])

        # Build global index (file_idx, row_idx) and keep shard sizes
        self._index: List[Tuple[int, int]] = []
        self._sizes: List[int] = []
        for i, f in enumerate(self.files):
            with np.load(f, mmap_mode="r") as z:
                n = int(z["X"].shape[0])
            self._sizes.append(n)
            self._index.extend((i, j) for j in range(n))

        self._cache: Tuple[int, np.lib.npyio.NpzFile] | None = None

        print(f"[Dataset] {self.root} → {len(self._index):,} samples "
              f"(T={self.T}, F={self.F}, H={self.H}, shards={len(self.files)})")

    def __len__(self) -> int:
        return len(self._index)

    def _get_file(self, i: int):
        if self._cache is not None and self._cache[0] == i:
            return self._cache[1]
        if self._cache is not None:
            self._cache[1].close()
        handle = np.load(self.files[i], mmap_mode="r")
        self._cache = (i, handle)
        return handle

    def __getitem__(self, idx: int):
        fi, row = self._index[idx]
        z = self._get_file(fi)

        # X is always [T, F]
        x = z["X"][row]                       # numpy [T, F]
        # y can be scalar (if saved as [N] and row selection returns np.float64),
        # or 1D [H], or 2D [1, H]. Normalize to [H].
        y_raw = z["y"][row]                   # could be scalar / [H]
        y_np = np.asarray(y_raw)
        if y_np.ndim == 0:                    # scalar → [1]
            y_np = y_np[None]
        elif y_np.ndim == 2:                  # [1, H] → [H]
            y_np = y_np.reshape(-1)

        # Safety: ensure length matches probed H (pad/trim if inconsistent)
        if y_np.ndim != 1:
            y_np = y_np.reshape(-1)
        if y_np.shape[0] != self.H:
            if y_np.shape[0] > self.H:
                y_np = y_np[: self.H]
            else:
                y_np = np.pad(y_np, (0, self.H - y_np.shape[0]), mode="edge")

        x_t = torch.from_numpy(x).float()     # [T, F]
        y_t = torch.from_numpy(y_np).float()  # [H]
        return x_t, y_t


# ---------------- Sampler ----------------
class GroupedBatchSampler(Sampler[List[int]]):
    """Cache-friendly sampler yielding contiguous index chunks."""
    def __init__(self, dataset: NpzSeqDataset, batch_size: int, shuffle: bool = True):
        self.ds = dataset
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        idxs = np.arange(len(self.ds))
        if self.shuffle:
            np.random.shuffle(idxs)
        for i in range(0, len(idxs), self.bs):
            yield idxs[i : i + self.bs].tolist()

    def __len__(self):
        return (len(self.ds) + self.bs - 1) // self.bs


# ---------------- Loader builder ----------------
def make_loaders(
    seq_root: Path | str,
    train_split: str,
    val_split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
):
    root = Path(seq_root)
    t0 = time.time()

    tr_ds = NpzSeqDataset(root / train_split)
    va_ds = NpzSeqDataset(root / val_split)

    train_loader = DataLoader(
        tr_ds,
        batch_sampler=GroupedBatchSampler(tr_ds, batch_size, shuffle=True),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    val_loader = DataLoader(
        va_ds,
        batch_sampler=GroupedBatchSampler(va_ds, batch_size, shuffle=False),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    dt = time.time() - t0
    print(
        f"[make_loaders] built in {dt:.2f}s | "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} bs={batch_size}"
    )
    return train_loader, val_loader, tr_ds.F, tr_ds.H, tr_ds.T


# ---------------- Debug entrypoint ----------------
if __name__ == "__main__":
    # Minimal smoke test; adjust splits if needed.
    seq_dir = Path("experiments/artifacts/seq")
    tr, va, F, H, T = make_loaders(
        seq_dir, "train/h=1", "val/h=1",
        batch_size=32, num_workers=0, pin_memory=False, persistent_workers=False
    )
    for i, (x, y) in enumerate(tr):
        print(f"[batch {i}] X={tuple(x.shape)} y={tuple(y.shape)}  (T={T}, F={F}, H={H})")
        if i == 1:
            break