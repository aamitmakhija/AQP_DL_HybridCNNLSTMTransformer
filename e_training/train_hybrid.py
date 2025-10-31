#!/usr/bin/env python3
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
import torch.optim as optim

from e_training.losses import LossWrapper, LossConfig

# ============================================================
# CONFIG HELPERS
# ============================================================
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

def _require(cfg: Dict, path: List[str]):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError("Missing config key: " + ".".join(path))
        cur = cur[k]
    return cur

# ============================================================
# UTILITIES
# ============================================================
def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _auto_bool(x, *, default: bool = False) -> bool:
    if isinstance(x, bool): return x
    if x is None: return default
    if isinstance(x, str): return x.strip().lower() in {"1","true","yes","y","on"}
    return bool(x)

def pick_device(pref: str | None) -> Tuple[torch.device, str]:
    """Select device: CUDA → MPS → CPU unless forced."""
    name = (pref or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available(): return torch.device("cuda"), "cuda"
        if torch.backends.mps.is_available(): return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"
    if name in {"cuda","gpu"} and torch.cuda.is_available(): return torch.device("cuda"), "cuda"
    if name == "mps" and torch.backends.mps.is_available(): return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"

class EarlyStopper:
    def __init__(self, patience: int = 10, enabled: bool = True, min_delta: float = 0.0):
        self.patience = int(patience)
        self.enabled = bool(enabled) and self.patience > 0
        self.best = float("inf")
        self.count = 0
        self.min_delta = float(min_delta)

    def update(self, value: float) -> bool:
        if not self.enabled: return False
        if value < self.best - self.min_delta:
            self.best = value
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience

# ============================================================
# NPZ DATASET + SAMPLER
# ============================================================
class NpzSeqDataset(Dataset):
    """
    Expects shards at <seq_root>/shard_*.npz
    Each .npz: X [N,T,F] float32, y [N] or [N,1] float32
    Optional: sid [N] int64 station ids (if present, we return it)
    Loads one shard fully into RAM to avoid per-sample zip cost.
    """
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.files = sorted(self.root.glob("shard_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz shards in {self.root}")

        with np.load(self.files[0], allow_pickle=False) as z0:
            X0, y0 = z0["X"], z0["y"]
            self.T, self.F = int(X0.shape[1]), int(X0.shape[2])
            self.H = 1 if y0.ndim == 1 else int(y0.shape[1])
            self.has_sid = "sid" in z0

        self._sizes, self._offsets = [], []
        total = 0
        for f in self.files:
            with np.load(f, allow_pickle=False) as z:
                n = int(z["X"].shape[0])
            self._sizes.append(n)
            self._offsets.append(total)
            total += n
        self._length = total

        self._cache_idx: int | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._sid: np.ndarray | None = None

    def __len__(self) -> int:
        return self._length

    def _load_shard(self, shard_idx: int):
        if self._cache_idx == shard_idx:
            return
        with np.load(self.files[shard_idx], allow_pickle=False) as z:
            X = z["X"].astype(np.float32, copy=True, order="C")
            y = z["y"].astype(np.float32, copy=True, order="C")
            sid = z["sid"].astype(np.int64, copy=True, order="C") if "sid" in z else None
        self._X = X
        self._y = y.squeeze(-1) if y.ndim > 1 else y
        self._sid = sid
        self._cache_idx = shard_idx

    def _locate(self, global_idx: int) -> tuple[int, int]:
        for s_idx, off in enumerate(self._offsets):
            if global_idx < off + self._sizes[s_idx]:
                return s_idx, global_idx - off
        raise IndexError(global_idx)

    def __getitem__(self, idx: int):
        s_idx, row = self._locate(idx)
        self._load_shard(s_idx)
        x = torch.from_numpy(self._X[row]).clone()                 # [T,F]
        y = torch.tensor(self._y[row], dtype=torch.float32).clone()# []
        if self._sid is not None:
            sid = torch.tensor(self._sid[row], dtype=torch.long)
            return x, y, sid
        return x, y

class PerShardBatchSampler(Sampler[List[int]]):
    """Yield batches that never cross shard boundaries (cache-friendly)."""
    def __init__(self, ds: NpzSeqDataset, batch_size: int, shuffle: bool = True):
        self.ds = ds
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)
        self._ranges: List[np.ndarray] = []
        for off, n in zip(ds._offsets, ds._sizes):
            idxs = np.arange(off, off + n)
            if self.shuffle: np.random.shuffle(idxs)
            self._ranges.append(idxs)

    def __iter__(self):
        if self.shuffle:
            for r in self._ranges:
                np.random.shuffle(r)
        for r in self._ranges:
            n = len(r)
            for i in range(0, n, self.bs):
                yield r[i:i+self.bs].tolist()

    def __len__(self) -> int:
        return sum((len(r) + self.bs - 1)//self.bs for r in self._ranges)

def _make_loader(root: Path,
                 batch_size: int,
                 num_workers: int,
                 pin_memory: bool,
                 persistent_workers: bool,
                 train: bool,
                 prefetch_factor: int = 2) -> tuple[DataLoader, int, int, bool]:
    ds = NpzSeqDataset(root)
    sampler = PerShardBatchSampler(ds, batch_size, shuffle=train)
    kwargs = dict(
        batch_sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers and num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(ds, **kwargs)
    return loader, ds.F, ds.T, ds.has_sid

# ============================================================
# MODEL
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor):  # x: [B,T,D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class HybridEncoder(nn.Module):
    """
    CNN → Transformer encoder → pooling (gap|gmp|max|cls) → (optional station emb) → MLP head
    Outputs a single value per sequence (seq2one).
    """
    def __init__(self,
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
                 n_stations: int | None = None,
                 station_embed_dim: int = 0):
        super().__init__()
        assert len(cnn_channels) == len(cnn_kernels), "cnn_channels and cnn_kernels length mismatch"
        layers, c_in = [], in_dim
        for c_out, k in zip(cnn_channels, cnn_kernels):
            layers += [nn.Conv1d(c_in, c_out, kernel_size=k, padding=k//2),
                       nn.ReLU(),
                       nn.Dropout(cnn_dropout)]
            c_in = c_out
        self.cnn = nn.Sequential(*layers) if layers else nn.Identity()
        self.proj = nn.Linear(c_in, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=int(ff_mult*d_model),
            dropout=ffn_dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # ---- pooling options ----
        pool = (pool or "gap").lower()
        if pool not in {"gap", "gmp", "cls"}:
            raise ValueError(f"Unsupported pool='{pool}' (choose 'gap'|'gmp'|'cls')")
        self.pool = pool
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model)) if self.pool == "cls" else None

        self.posenc = SinusoidalPositionalEncoding(d_model) if posenc == "sin" else nn.Identity()

        self.use_station = station_embed_dim > 0 and (n_stations is not None) and (n_stations > 0)
        if self.use_station:
            self.sid_emb = nn.Embedding(n_stations, station_embed_dim)
            head_in = d_model + station_embed_dim
        else:
            self.sid_emb = None
            head_in = d_model

        self.head = nn.Sequential(
            nn.Linear(head_in, head_in), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(head_in, 1),
        )

    def forward(self, x: torch.Tensor, sid: torch.Tensor | None = None):  # x: [B,T,F]
        x = self.cnn(x.transpose(1,2)).transpose(1,2)  # [B,T,C]
        x = self.proj(x)                                # [B,T,D]
        if self.pool == "cls":
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)              # [B,1+T,D]
        x = self.attn_dropout(self.posenc(x))
        x = self.encoder(x)                             # [B,T,D] or [B,1+T,D]

        if self.pool == "gap":
            z = x.mean(dim=1)                           # [B,D]
        elif self.pool == "gmp":
            z, _ = x.max(dim=1)                         # [B,D]
        else:  # 'cls'
            z = x[:, 0, :]                              # [B,D]

        if self.use_station and sid is not None:
            z = torch.cat([z, self.sid_emb(sid)], dim=-1)
        return self.head(z).squeeze(-1)

# ============================================================
# MODEL SUMMARY (printed + saved)
# ============================================================
def _summarize_model(model: nn.Module, *, input_dim: int, horizons: List[int]) -> str:
    """Return a human-readable summary of the hybrid model."""
    lines = []
    lines.append("=== HYBRID MODEL SUMMARY ===")
    lines.append(f"Input features: {input_dim}")
    # CNN
    try:
        lines.append(f"CNN: {len(model.cnn)} layer(s)")
        lines.append(repr(model.cnn))
    except Exception:
        lines.append("CNN: <unavailable>")
    # Transformer
    try:
        n_layers = getattr(model.encoder, "num_layers", None)
        if n_layers is None and hasattr(model.encoder, "layers"):
            n_layers = len(model.encoder.layers)
        if hasattr(model.encoder, "layers") and len(model.encoder.layers) > 0:
            nhead = model.encoder.layers[0].self_attn.num_heads
        else:
            nhead = "?"
        lines.append(f"Transformer: {n_layers} layer(s), {nhead}-head attention")
    except Exception:
        lines.append("Transformer: <unavailable>")
    # PosEnc
    posenc_on = not isinstance(getattr(model, "posenc", nn.Identity()), nn.Identity)
    lines.append(f"Positional Encoding: {'Enabled' if posenc_on else 'Disabled'}")
    # Pool
    pool_mode = getattr(model, "pool", "gap")
    lines.append(f"Pooling: {pool_mode}")
    # Station embedding
    sid_used = getattr(model, "use_station", False)
    if sid_used:
        lines.append(f"Station embedding: dim={model.sid_emb.embedding_dim}, n={model.sid_emb.num_embeddings}")
    else:
        lines.append("Station embedding: disabled")
    # Head(s)
    try:
        head = model.head
        lines.append("Head (seq2one):")
        lines.append(repr(head))
    except Exception:
        lines.append("Head: <unavailable>")
    lines.append(f"Horizons: {horizons}")
    lines.append("=====================================")
    return "\n".join(lines)

# ============================================================
# TRAIN / EVAL (AMP)
# ============================================================
def _unpack(batch):
    # Supports (X,y) or (X,y,sid)
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    X, y = batch
    return X, y, None

def train_one_epoch_amp(model, loader, device, optimizer, criterion,
                        grad_clip, amp_enabled, amp_device, amp_dtype) -> float:
    model.train(True)
    total, n = 0.0, 0
    for batch in loader:
        X, y, sid = _unpack(batch)
        X = X.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False).float()
        sid = sid.to(device) if sid is not None else None

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled):
            preds = model(X, sid)            # model outputs ORIGINAL scale
            loss = criterion(preds, y)       # transform/clipping handled inside loss
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)

@torch.no_grad()
def evaluate_amp(model, loader, device, criterion, amp_enabled, amp_device, amp_dtype) -> float:
    model.train(False)
    total, n = 0.0, 0
    for batch in loader:
        X, y, sid = _unpack(batch)
        X = X.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False).float()
        sid = sid.to(device) if sid is not None else None
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled):
            preds = model(X, sid)
            loss = criterion(preds, y)
        bs = y.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)

# ============================================================
# MAIN
# ============================================================
def main():
    cfg = _load_cfg()
    _seed_everything(int(cfg.get("seed", 42)))
    torch.set_float32_matmul_precision("high")

    # Paths
    art_dir = Path(_require(cfg, ["paths", "artifacts_dir"]))
    seq_cfg = _require(cfg, ["sequence"])
    seq_out = _require(seq_cfg, ["out_dir"])
    seq_dir = Path(seq_out) if Path(seq_out).is_absolute() else (art_dir / seq_out)
    horizons = seq_cfg["horizon"] if isinstance(seq_cfg["horizon"], list) else [int(seq_cfg["horizon"])]
    train_split, val_split = seq_cfg["train_split"], seq_cfg["val_split"]

    # DL/runtime
    dl = _require(cfg, ["dl"])
    dev, dev_name = pick_device(dl.get("device"))
    batch_size = int(dl.get("batch_size", 64))
    num_workers = int(dl.get("num_workers", 0))
    pin_memory = _auto_bool(dl.get("pin_memory", False), default=False)
    persistent_workers = _auto_bool(dl.get("persistent_workers", False), default=False)
    prefetch = int(dl.get("prefetch_factor", 2))
    epochs = int(dl.get("epochs", 50))
    grad_clip = float(dl.get("grad_clip", 0.0))
    patience = int(dl.get("patience", 10))

    # AMP
    amp_cfg = dl.get("amp", {}) or {}
    amp_enabled = bool(amp_cfg.get("enabled", False)) and dev.type in ("cuda", "mps")
    amp_dtype = {"bf16": torch.bfloat16}.get(str(amp_cfg.get("dtype", "fp16")).lower(), torch.float16)
    amp_device = dev.type  # "cuda" | "mps" | "cpu"

    # Model hp
    mh = _require(cfg, ["model", "hybrid"])
    cnn_channels = list(mh.get("cnn_channels", [64, 64]))
    cnn_kernels  = list(mh.get("cnn_kernels",  [5, 3]))
    cnn_dropout  = float(mh.get("cnn_dropout", 0.05))
    d_model      = int(mh.get("d_model", 128))
    nhead        = int(mh.get("nhead", 4))
    num_layers   = int(mh.get("num_layers", 3))
    ff_mult      = int(mh.get("ff_mult", 2))
    attn_dropout = float(mh.get("attn_dropout", 0.05))
    ffn_dropout  = float(mh.get("ffn_dropout", 0.05))
    pool         = str(mh.get("pool", "gap"))
    posenc       = str(mh.get("posenc", "sin"))

    # Optional station embedding
    station_embed_dim = int(mh.get("station_embed_dim", 0))
    n_stations_cfg = mh.get("n_stations", None)
    n_stations = int(n_stations_cfg) if n_stations_cfg is not None else None

    # Optimizer + scheduler
    opt_name = str(dl.get("optimizer", "adamw")).lower()
    lr = float(dl.get("lr", 5e-4))
    weight_decay = float(dl.get("weight_decay", 1e-4))
    lrs_cfg = dl.get("lr_scheduler", {}) or {}
    use_plateau = str(lrs_cfg.get("name", "")).lower() == "reduce_on_plateau"
    plateau_factor = float(lrs_cfg.get("factor", 0.5))
    plateau_patience = int(lrs_cfg.get("patience", 4))
    plateau_min_lr = float(lrs_cfg.get("min_lr", 1e-5))

    # Loss (wired to YAML)
    loss_cfg = LossConfig(
        name=str(dl.get("loss", "huber")),
        huber_delta=float(dl.get("huber_delta", 1.0)),
        target_transform=str(dl.get("target_transform", "none")),
        clip_target_max=dl.get("clip_target_max", None),
    )
    criterion = LossWrapper(loss_cfg).to(dev)

    # Outputs
    model_dir_cfg = dl.get("model_dir", "models")
    model_dir = Path(model_dir_cfg) if Path(model_dir_cfg).is_absolute() else (art_dir / model_dir_cfg)
    _ensure_dir(model_dir)

    print(f"[DL] device={dev_name}  bs={batch_size}  workers={num_workers}  pin_memory={pin_memory}  prefetch={prefetch}")
    print(f"[DL] AMP: enabled={amp_enabled} dtype={amp_dtype} device={amp_device}")

    for H in horizons:
        htag = f"h={int(H)}"
        tr_root = seq_dir / train_split / htag
        va_root = seq_dir / val_split / htag

        # Loaders
        train_loader, input_dim, _, train_has_sid = _make_loader(
            tr_root, batch_size, num_workers, pin_memory, persistent_workers,
            train=True, prefetch_factor=prefetch
        )
        val_loader,   _,        _, val_has_sid   = _make_loader(
            va_root, batch_size, num_workers, pin_memory, persistent_workers,
            train=False, prefetch_factor=prefetch
        )
        has_sid = train_has_sid and val_has_sid

        # Infer n_stations if needed and sids exist
        inferred_n_stations = None
        if station_embed_dim > 0 and n_stations is None and has_sid:
            seen_max = -1
            for b in train_loader:
                if len(b) == 3:
                    seen_max = max(seen_max, int(b[2].max().item()))
            inferred_n_stations = seen_max + 1 if seen_max >= 0 else None

        # Model
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
            n_stations=(n_stations or inferred_n_stations or 0),
            station_embed_dim=station_embed_dim if has_sid else 0,
        ).to(dev)

        # --- Print & save a model summary for this horizon ---
        summary_txt = _summarize_model(model, input_dim=input_dim, horizons=[H])
        print(summary_txt)
        with open(model_dir / f"model_summary_H{int(H)}.txt", "w") as fsum:
            fsum.write(summary_txt + "\n")

        # Optimizer / scheduler
        opt_cls = optim.AdamW if opt_name == "adamw" else optim.Adam
        optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None
        if use_plateau:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=plateau_factor,
                patience=plateau_patience, min_lr=plateau_min_lr
            )

        stopper = EarlyStopper(patience=patience)

        best_val, best_ep = float("inf"), 0
        history: List[Dict[str, float]] = []

        for ep in range(1, epochs + 1):
            tr_loss = train_one_epoch_amp(
                model, train_loader, dev, optimizer, criterion,
                grad_clip, amp_enabled, amp_device, amp_dtype
            )
            va_loss = evaluate_amp(
                model, val_loader, dev, criterion,
                amp_enabled, amp_device, amp_dtype
            )
            history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
            print(f"[H={H}] ep {ep:03d}  train={tr_loss:.6f}  val={va_loss:.6f}")

            if scheduler is not None:
                scheduler.step(va_loss)

            if va_loss + 1e-12 < best_val:
                best_val, best_ep = va_loss, ep
                torch.save(model.state_dict(), model_dir / f"hybrid_h{int(H)}.pt")

            if stopper.update(va_loss):
                print(f"[H={H}] Early stopping at ep={ep} (best ep={best_ep}, val={best_val:.6f})")
                break

        # Summary
        with open(model_dir / f"hybrid_h{int(H)}_train_summary.json", "w") as f:
            json.dump({
                "horizon": int(H),
                "best_epoch": int(best_ep),
                "best_val_loss": float(best_val),
                "history": history,
                "params": {
                    "device": dev_name, "batch_size": batch_size, "num_workers": num_workers,
                    "pin_memory": pin_memory, "epochs": epochs, "patience": patience,
                    "optimizer": opt_name, "lr": lr, "weight_decay": weight_decay,
                    "loss": loss_cfg.__dict__,
                    "cnn_channels": cnn_channels, "cnn_kernels": cnn_kernels, "cnn_dropout": cnn_dropout,
                    "d_model": d_model, "nhead": nhead, "num_layers": num_layers, "ff_mult": ff_mult,
                    "attn_dropout": attn_dropout, "ffn_dropout": ffn_dropout, "pool": pool, "posenc": posenc,
                    "amp": {"enabled": amp_enabled, "dtype": str(amp_dtype), "device": amp_device},
                    "station_embed_dim": station_embed_dim,
                    "n_stations": (n_stations or inferred_n_stations),
                },
            }, f, indent=2)

    print(f"[OK] checkpoints → {model_dir}")

if __name__ == "__main__":
    main()