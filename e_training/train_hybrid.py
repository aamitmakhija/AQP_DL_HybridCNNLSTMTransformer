# e_training/train_hybrid.py
from __future__ import annotations
import os, json, math, random
from pathlib import Path
from typing import Dict, Tuple, List
from copy import deepcopy

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

# use shared engine utilities (no duplicates here)
from e_training.engine import train_one_epoch, evaluate, EarlyStopper

# ================= config helpers =================
def _deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> Dict:
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

def _require(cfg: Dict, path: List[str], name: str | None = None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            dotted = ".".join(path)
            raise KeyError(f"Missing config key: {dotted}{' ('+name+')' if name else ''}")
        cur = cur[k]
    return cur

# ================= utils =================
def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def _resolve_device(pref: str) -> str:
    pref = str(pref).lower()
    if pref == "auto":
        # Prefer CUDA if present; otherwise **CPU** (skip MPS by default due to instability)
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if pref == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return pref

def _auto_bool(x, *, default: bool) -> bool:
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.strip().lower() in {"1","true","yes","y","on"}
    if x is None or str(x).lower() == "auto": return default
    return bool(x)

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# ================= FAST NPZ DATASET (per-shard RAM cache) =================

class NpzSeqDataset(Dataset):
    """
    Expects shards at <seq_root>/<split>/h=<H>/shard_*.npz
    Each npz contains:
      X: [N, T, F] float32
      y: [N] or [N,1] float32
    We load ONE shard entirely into RAM at a time to avoid per-sample zip decompression.
    """
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.files = sorted(self.root.glob("shard_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz shards found in {self.root}")

        # Inspect first shard
        with np.load(self.files[0], allow_pickle=False) as z0:
            X0, y0 = z0["X"], z0["y"]
            self.T, self.F = int(X0.shape[1]), int(X0.shape[2])
            self.H = 1 if y0.ndim == 1 else int(y0.shape[1])

        # Build global index via offsets
        self._sizes: List[int] = []
        self._offsets: List[int] = []
        total = 0
        for f in self.files:
            with np.load(f, allow_pickle=False) as z:
                n = int(z["X"].shape[0])
            self._sizes.append(n)
            self._offsets.append(total)
            total += n
        self._length = total

        # In-RAM cache for current shard arrays
        self._cache_idx: int | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def __len__(self) -> int:
        return self._length

    def _load_shard(self, shard_idx: int):
        if self._cache_idx == shard_idx:
            return
        # Load and make owned, C-contiguous float32 copies (avoids MPS allocator crashes)
        with np.load(self.files[shard_idx], allow_pickle=False) as z:
            X = z["X"].astype(np.float32, copy=True, order="C")
            y = z["y"].astype(np.float32, copy=True, order="C")
        self._X = X
        self._y = y if y.ndim == 1 else y.squeeze(-1)
        self._cache_idx = shard_idx

    def _locate(self, global_idx: int) -> tuple[int, int]:
        # Few shards → linear scan is fine
        for s_idx, off in enumerate(self._offsets):
            if global_idx < off + self._sizes[s_idx]:
                return s_idx, global_idx - off
        raise IndexError(global_idx)
    # e_training/train_hybrid.py  (inside NpzSeqDataset)
        
    def __getitem__(self, idx: int):
        s_idx, row = self._locate(idx)
        self._load_shard(s_idx)
        x = torch.from_numpy(self._X[row]).clone()                 # [T, F], own storage
        y = torch.as_tensor(self._y[row], dtype=torch.float32).clone()  # 0-D scalar, own storage
        return x, y

class PerShardBatchSampler(Sampler[List[int]]):
    """
    Yields batches that never cross shard boundaries (maximizes cache hits).
    """
    def __init__(self, ds: NpzSeqDataset, batch_size: int, shuffle: bool = True):
        self.ds = ds
        self.bs = int(batch_size)
        self.shuffle = shuffle
        # Precompute per-shard ranges
        self._ranges: List[np.ndarray] = []
        for off, n in zip(ds._offsets, ds._sizes):
            idxs = np.arange(off, off + n)
            if self.shuffle:
                np.random.shuffle(idxs)
            self._ranges.append(idxs)

    def __iter__(self):
        if self.shuffle:
            for r in self._ranges:
                np.random.shuffle(r)
        for r in self._ranges:
            n = len(r)
            for i in range(0, n, self.bs):
                yield r[i:i + self.bs].tolist()

    def __len__(self) -> int:
        total = 0
        for r in self._ranges:
            total += (len(r) + self.bs - 1) // self.bs
        return total

def _make_loader(root: Path, batch_size: int, num_workers: int,
                 pin_memory: bool, persistent_workers: bool,
                 train: bool) -> tuple[DataLoader, int, int]:
    ds = NpzSeqDataset(root)
    sampler = PerShardBatchSampler(ds, batch_size, shuffle=train)
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,                 # keep 0 on macOS
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    return loader, ds.F, ds.T

# ================= model: CNN -> PosEnc -> Transformer(encoder) -> GAP/CLS -> MLP =================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor):  # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        cnn_channels: List[int],
        cnn_kernels: List[int],
        cnn_dropout: float,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_mult: int,
        attn_dropout: float,
        ffn_dropout: float,
        pool: str = "gap",
        posenc: str = "sin",
    ):
        super().__init__()
        assert len(cnn_channels) == len(cnn_kernels), "cnn_channels and cnn_kernels length mismatch"
        layers = []
        c_in = in_dim
        for c_out, k in zip(cnn_channels, cnn_kernels):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k//2),
                nn.ReLU(),
                nn.Dropout(cnn_dropout),
            ]
            c_in = c_out
        self.cnn = nn.Sequential(*layers) if layers else nn.Identity()
        proj_in = c_in if layers else in_dim
        self.proj = nn.Linear(proj_in, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(ff_mult * d_model),
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.pool = pool  # "gap" or "cls"
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pool == "cls" else None
        self.posenc = SinusoidalPositionalEncoding(d_model) if posenc == "sin" else nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor):  # x: [B, T, F]
        # CNN over time: transpose to [B, F, T]
        x = x.transpose(1, 2)                     # [B, F, T]
        x = self.cnn(x)                            # [B, C, T]
        x = x.transpose(1, 2)                      # [B, T, C]
        x = self.proj(x)                           # [B, T, D]

        if self.pool == "cls":
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)  # [B,1,D]
            x = torch.cat([cls, x], dim=1)          # [B,1+T,D]

        x = self.posenc(x) if isinstance(self.posenc, SinusoidalPositionalEncoding) else x
        x = self.attn_dropout(x)
        x = self.encoder(x)                         # [B, T(±1), D]

        if self.pool == "gap":
            z = x.mean(dim=1)                       # [B, D]
        else:
            z = x[:, 0, :]                          # [B, D] (CLS)
        yhat = self.head(z).squeeze(-1)             # [B]
        return yhat

# ================= main =================
def main():
    cfg = _load_cfg()

    # --- global
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)

    # --- paths
    art_dir = Path(_require(cfg, ["paths", "artifacts_dir"]))
    seq_cfg = _require(cfg, ["sequence"])
    seq_dir_cfg = _require(seq_cfg, ["out_dir"])
    seq_dir = Path(seq_dir_cfg) if Path(seq_dir_cfg).is_absolute() else (art_dir / seq_dir_cfg)

    horizons = list(_require(seq_cfg, ["horizon"])) if isinstance(_require(seq_cfg, ["horizon"]), list) else [int(_require(seq_cfg, ["horizon"]))]
    train_split = _require(seq_cfg, ["train_split"])
    val_split   = _require(seq_cfg, ["val_split"])

    # --- dl/runtime
    dl = _require(cfg, ["dl"])
    device = _resolve_device(dl.get("device", "auto"))
    batch_size = int(dl.get("batch_size", 64))
    num_workers = int(dl.get("num_workers", 0))
    pin_memory = _auto_bool(dl.get("pin_memory", False), default=torch.cuda.is_available())
    persistent_workers = _auto_bool(dl.get("persistent_workers", False), default=False)
    epochs = int(dl.get("epochs", 50))
    grad_clip = float(dl.get("grad_clip", 0.0))
    patience = int(dl.get("patience", 10))

    # --- model hyperparams
    mh = _require(cfg, ["model", "hybrid"])
    cnn_channels = list(mh.get("cnn_channels", [64, 64]))
    cnn_kernels  = list(mh.get("cnn_kernels",  [5, 3]))
    cnn_dropout  = float(mh.get("cnn_dropout", 0.1))
    d_model      = int(mh.get("d_model", 128))
    nhead        = int(mh.get("nhead", 4))
    num_layers   = int(mh.get("num_layers", 3))
    ff_mult      = int(mh.get("ff_mult", 2))
    attn_dropout = float(mh.get("attn_dropout", 0.1))
    ffn_dropout  = float(mh.get("ffn_dropout", 0.1))
    pool         = str(mh.get("pool", "gap"))
    posenc       = str(mh.get("posenc", "sin"))

    # --- optimizer
    opt_name = str(dl.get("optimizer", "adamw")).lower()
    lr = float(dl.get("lr", 1e-3))
    weight_decay = float(dl.get("weight_decay", 1e-4))

    # --- outputs
    model_dir_cfg = dl.get("model_dir", "models")
    model_dir = Path(model_dir_cfg) if Path(model_dir_cfg).is_absolute() else (art_dir / model_dir_cfg)
    _ensure_dir(model_dir)

    print("[DL] Training Hybrid CNN→Transformer (encoder-only) per horizon", horizons)
    print(f"[DL] device={device}  bs={batch_size}  workers={num_workers}  pin_memory={pin_memory}")

    for H in horizons:
        htag = f"h={int(H)}"
        tr_root = seq_dir / train_split / htag
        va_root = seq_dir / val_split / htag

        # loaders (per-shard batching to hit RAM cache)
        train_loader, input_dim, _ = _make_loader(tr_root, batch_size, num_workers, pin_memory, persistent_workers, train=True)
        val_loader,   _,         _ = _make_loader(va_root, batch_size, num_workers, pin_memory, persistent_workers, train=False)

        # model
        model = HybridEncoder(
            in_dim=input_dim,
            cnn_channels=cnn_channels,
            cnn_kernels=cnn_kernels,
            cnn_dropout=cnn_dropout,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            pool=pool,
            posenc=posenc,
        ).to(device)

        if opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_ep = 0
        history = []
        stopper = EarlyStopper(patience=patience, enabled=(patience is not None and patience > 0))

        for ep in range(1, epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, torch.device(device), optimizer, criterion, grad_clip)
            va_loss, _, _ = evaluate(model, val_loader, torch.device(device), criterion)
            history.append({"epoch": ep, "train_mse": tr_loss, "val_mse": va_loss})
            print(f"[H={H:>3}] ep {ep:03d}  train_mse={tr_loss:.5f}  val_mse={va_loss:.5f}")

            if va_loss + 1e-12 < best_val:
                best_val, best_ep = va_loss, ep
                torch.save(model.state_dict(), model_dir / f"hybrid_h{int(H)}.pt")

            if stopper.update(va_loss):
                print(f"[H={H}] Early stopping at ep={ep} (best ep={best_ep}, val_mse={best_val:.5f})")
                break

            print("[model/hybrid]", {
                "cnn_channels": cnn_channels,
                "cnn_kernels":  cnn_kernels,
                "cnn_dropout":  cnn_dropout,
                "d_model":      d_model,
                "nhead":        nhead,
                "num_layers":   num_layers,
                "ff_mult":      ff_mult,
                "attn_dropout": attn_dropout,
                "ffn_dropout":  ffn_dropout,
                "pool":         pool,
                "posenc":       posenc,
            })

        # save run summary
        summary = {
            "horizon": int(H),
            "best_epoch": int(best_ep),
            "best_val_mse": float(best_val),
            "history": history,
            "params": {
                "device": device, "batch_size": batch_size, "num_workers": num_workers,
                "pin_memory": pin_memory, "epochs": epochs, "patience": patience,
                "optimizer": opt_name, "lr": lr, "weight_decay": weight_decay,
                "cnn_channels": cnn_channels, "cnn_kernels": cnn_kernels, "cnn_dropout": cnn_dropout,
                "d_model": d_model, "nhead": nhead, "num_layers": num_layers, "ff_mult": ff_mult,
                "attn_dropout": attn_dropout, "ffn_dropout": ffn_dropout, "pool": pool, "posenc": posenc,
            },
        }
        with open(model_dir / f"hybrid_h{int(H)}_train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"[OK] checkpoints → {model_dir}")

if __name__ == "__main__":
    main()