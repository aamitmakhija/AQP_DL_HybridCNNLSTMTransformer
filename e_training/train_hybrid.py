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
from e_training.models.factory import build_hybrid  # choose impl via YAML (impl:)

# ------------------------ config ------------------------

def _deep_update(dst: Dict, src: Dict) -> Dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_cfg() -> Dict:
    base = yaml.safe_load(open("configs/default.yaml")) or {}
    cfg_env = os.environ.get("CONFIG", "")
    if not cfg_env:
        return base
    merged = deepcopy(base)
    for p in [s.strip() for s in cfg_env.split(",") if s.strip()]:
        overlay = yaml.safe_load(open(p)) or {}
        _deep_update(merged, overlay)
    return merged

def _require(cfg: Dict, path: List[str]):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError("Missing config key: " + ".".join(path))
        cur = cur[k]
    return cur

# ------------------------ utils ------------------------

def _seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _auto_bool(x, default=False) -> bool:
    if isinstance(x, bool): return x
    if x is None: return default
    if isinstance(x, str): return x.strip().lower() in {"1","true","yes","y","on"}
    return bool(x)

def pick_device(pref: str | None) -> Tuple[torch.device, str]:
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
        self.best = float("inf"); self.count = 0; self.min_delta = float(min_delta)
    def update(self, value: float) -> bool:
        if not self.enabled: return False
        if value < self.best - self.min_delta:
            self.best = value; self.count = 0; return False
        self.count += 1
        return self.count >= self.patience

# ------------------------ data ------------------------

class NpzSeqDataset(Dataset):
    """
    <seq_root>/shard_*.npz with keys: X [N,T,F] float32, y [N] or [N,1] float32, optional sid [N] int64
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
        self._sizes, self._offsets, total = [], [], 0
        for f in self.files:
            with np.load(f, allow_pickle=False) as z:
                n = int(z["X"].shape[0])
            self._sizes.append(n); self._offsets.append(total); total += n
        self._length = total
        self._cache_idx = None; self._X = None; self._y = None; self._sid = None

    def __len__(self): return self._length

    def _load_shard(self, shard_idx: int):
        if self._cache_idx == shard_idx: return
        with np.load(self.files[shard_idx], allow_pickle=False) as z:
            X = z["X"].astype(np.float32, copy=True, order="C")
            y = z["y"].astype(np.float32, copy=True, order="C")
            sid = z["sid"].astype(np.int64, copy=True, order="C") if "sid" in z else None
        self._X = X; self._y = y.squeeze(-1) if y.ndim > 1 else y; self._sid = sid; self._cache_idx = shard_idx

    def _locate(self, global_idx: int) -> tuple[int,int]:
        for s_idx, off in enumerate(self._offsets):
            if global_idx < off + self._sizes[s_idx]: return s_idx, global_idx - off
        raise IndexError(global_idx)

    def __getitem__(self, idx: int):
        s_idx, row = self._locate(idx)
        self._load_shard(s_idx)
        x = torch.from_numpy(self._X[row]).clone()                 # [T,F]
        y = torch.tensor(self._y[row], dtype=torch.float32).clone()
        if self._sid is not None:
            sid = torch.tensor(self._sid[row], dtype=torch.long)
            return x, y, sid
        return x, y

class PerShardBatchSampler(Sampler[List[int]]):
    def __init__(self, ds: NpzSeqDataset, batch_size: int, shuffle: bool = True):
        self.ds = ds; self.bs = int(batch_size); self.shuffle = bool(shuffle)
        self._ranges = []
        for off, n in zip(ds._offsets, ds._sizes):
            idxs = np.arange(off, off + n)
            if self.shuffle: np.random.shuffle(idxs)
            self._ranges.append(idxs)
    def __iter__(self):
        if self.shuffle:
            for r in self._ranges: np.random.shuffle(r)
        for r in self._ranges:
            for i in range(0, len(r), self.bs):
                yield r[i:i+self.bs].tolist()
    def __len__(self): return sum((len(r) + self.bs - 1)//self.bs for r in self._ranges)

def _make_loader(root: Path, bs: int, nw: int, pin: bool, persist: bool, train: bool, prefetch: int = 2):
    ds = NpzSeqDataset(root)
    sampler = PerShardBatchSampler(ds, bs, shuffle=train)
    kw = dict(batch_sampler=sampler, num_workers=int(nw), pin_memory=bool(pin),
              persistent_workers=bool(persist and nw > 0))
    if nw > 0: kw["prefetch_factor"] = int(prefetch)
    loader = DataLoader(ds, **kw)
    return loader, ds.F, ds.T, ds.has_sid

# ------------------------ summary (introspects any impl) ------------------------

def _summarize_model(m: nn.Module, *, input_dim:int, horizons:List[int]) -> str:
    def has_attr(name): return hasattr(m, name)
    def val(name, default="?"): return getattr(m, name, default)
    lines = [
        "=== HYBRID MODEL SUMMARY ===",
        f"Input features: {input_dim}",
        f"CNN: {len(getattr(m,'cnn',[])) if has_attr('cnn') else '?'} layer(s)",
        f"LSTM: {'enabled' if bool(getattr(m,'use_lstm', False)) else 'disabled'}",
        f"Transformer layers: {getattr(getattr(m,'encoder', None),'num_layers', '?')}",
        f"Positional Encoding: {'enabled' if not isinstance(getattr(m,'posenc',nn.Identity()), nn.Identity) else 'disabled'}",
        f"Pooling: {val('pool','gap')}",
        f"Station embedding: {'enabled' if bool(getattr(m,'use_station_emb', False)) else 'disabled'}",
        f"Horizons: {horizons}",
        "====================================="
    ]
    return "\n".join(lines)

# ------------------------ schedulers ------------------------

class WarmupCosine:
    """Warmup + cosine schedule."""
    def __init__(self, optimizer, total_epochs:int, warmup_epochs:int=0, min_lr:float=1e-6):
        self.optimizer = optimizer
        self.total = max(1, int(total_epochs))
        self.warmup = max(0, int(warmup_epochs))
        self.min_lr = float(min_lr)
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.step_num += 1
        t = self.step_num
        if t <= self.warmup and self.warmup > 0:
            scale = t / self.warmup
        else:
            tt = max(1, self.total - self.warmup)
            k = min(tt, max(0, t - self.warmup))
            scale = 0.5 * (1.0 + math.cos(math.pi * k / tt))
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base * scale)

# ------------------------ train/eval (AMP + accum) ------------------------

def _unpack(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 3: return batch[0], batch[1], batch[2]
    X, y = batch; return X, y, None

# --- replace both helpers ---

def train_one_epoch_amp(model, loader, device, optimizer, criterion,
                        grad_clip, amp_enabled, amp_device, amp_dtype,
                        grad_accum_steps:int=1, head_key:int|None=None) -> float:
    model.train(True); total = 0.0; n = 0
    optimizer.zero_grad(set_to_none=True); step = 0
    for batch in loader:
        X, y, sid = _unpack(batch)
        X = X.to(device); y = y.to(device).float()
        sid = sid.to(device) if sid is not None else None
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled):
            out = model(X, sid_idx=sid) if sid is not None else model(X)
            preds = out[head_key] if (head_key is not None and isinstance(out, dict)) else out
            loss = criterion(preds, y)
        (loss / max(1, grad_accum_steps)).backward()
        step += 1
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if step % max(1, grad_accum_steps) == 0:
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        bs = y.size(0); total += float(loss.item()) * bs; n += bs
    if step % max(1, grad_accum_steps) != 0:
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return total / max(1, n)

@torch.no_grad()
def evaluate_amp(model, loader, device, criterion, amp_enabled, amp_device, amp_dtype,
                 head_key:int|None=None) -> float:
    model.train(False); total = 0.0; n = 0
    for batch in loader:
        X, y, sid = _unpack(batch)
        X = X.to(device); y = y.to(device).float()
        sid = sid.to(device) if sid is not None else None
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled):
            out = model(X, sid_idx=sid) if sid is not None else model(X)
            preds = out[head_key] if (head_key is not None and isinstance(out, dict)) else out
            loss = criterion(preds, y)
        bs = y.size(0); total += float(loss.item()) * bs; n += bs
    return total / max(1, n)

# ------------------------ main ------------------------

def main():
    cfg = _load_cfg()
    _seed_everything(int(cfg.get("seed", 42)))
    torch.set_float32_matmul_precision("high")

    # paths / sequence
    art_dir = Path(_require(cfg, ["paths", "artifacts_dir"]))
    seq_cfg = _require(cfg, ["sequence"])
    seq_dir = Path(_require(seq_cfg, ["out_dir"]));  seq_dir = seq_dir if seq_dir.is_absolute() else (art_dir / seq_dir)
    horizons = seq_cfg["horizon"] if isinstance(seq_cfg["horizon"], list) else [int(seq_cfg["horizon"])]
    train_split, val_split = seq_cfg["train_split"], seq_cfg["val_split"]

    # runtime
    dl = _require(cfg, ["dl"])
    dev, dev_name = pick_device(dl.get("device"))
    bs = int(dl.get("batch_size", 64))
    nw = int(dl.get("num_workers", 0))
    pin = _auto_bool(dl.get("pin_memory", False))
    persist = _auto_bool(dl.get("persistent_workers", False))
    prefetch = int(dl.get("prefetch_factor", 2))
    epochs = int(dl.get("epochs", 50))
    grad_clip = float(dl.get("grad_clip", 0.0))
    patience = int(dl.get("patience", 10))
    grad_accum_steps = int(dl.get("grad_accum_steps", 1))

    # AMP
    amp_cfg = dl.get("amp", {}) or {}
    amp_enabled = bool(amp_cfg.get("enabled", False)) and dev.type in ("cuda","mps")
    amp_dtype = {"bf16": torch.bfloat16}.get(str(amp_cfg.get("dtype", "fp16")).lower(), torch.float16)
    amp_device = dev.type

    # hp (read from YAML; factory will interpret)
    mh = _require(cfg, ["model", "hybrid"])
    station_embed_dim = int(mh.get("station_embed_dim", 0))
    n_stations_cfg = mh.get("n_stations", None)
    n_stations = int(n_stations_cfg) if n_stations_cfg is not None else None
    impl_name = str(mh.get("impl", "legacy_hybrid_encoder"))

    # optim
    opt_name = str(dl.get("optimizer", "adamw")).lower()
    lr = float(dl.get("lr", 5e-4)); weight_decay = float(dl.get("weight_decay", 1e-4))

    # lr schedulers
    lrs_cfg = dl.get("lr_scheduler", {}) or {}
    scheduler_name = str(lrs_cfg.get("name","")).lower()
    use_plateau = scheduler_name == "reduce_on_plateau"
    use_cosine  = scheduler_name == "cosine_warmup"
    plateau_factor  = float(lrs_cfg.get("factor", 0.5))
    plateau_patience= int(lrs_cfg.get("patience", 4))
    plateau_min_lr  = float(lrs_cfg.get("min_lr", 1e-5))
    warmup_epochs   = int(lrs_cfg.get("warmup_epochs", 0))
    cosine_min_lr   = float(lrs_cfg.get("min_lr", 1e-6))

    # loss
    loss_cfg = LossConfig(
        name=str(dl.get("loss", "huber")),
        huber_delta=float(dl.get("huber_delta", 1.0)),
        target_transform=str(dl.get("target_transform", "none")),
        clip_target_max=dl.get("clip_target_max", None),
    )
    criterion = LossWrapper(loss_cfg).to(dev)

    # outputs
    model_dir_cfg = dl.get("model_dir", "models")
    model_dir = Path(model_dir_cfg) if Path(model_dir_cfg).is_absolute() else (art_dir / model_dir_cfg)
    _ensure_dir(model_dir)

    print(f"[DL] device={dev_name} bs={bs} workers={nw} pin_memory={pin} prefetch={prefetch} grad_accum_steps={grad_accum_steps}")
    print(f"[DL] AMP: enabled={amp_enabled} dtype={amp_dtype} device={amp_device}")
    if use_plateau:
        print(f"[LR] ReduceLROnPlateau factor={plateau_factor} patience={plateau_patience} min_lr={plateau_min_lr}")
    elif use_cosine:
        print(f"[LR] Cosine w/ warmup: warmup_epochs={warmup_epochs} min_lr={cosine_min_lr}")
    else:
        print("[LR] no scheduler")
    print(f"[Model] impl={impl_name}")

    for H in (horizons if isinstance(horizons, list) else [horizons]):
        htag = f"h={int(H)}"
        tr_root = seq_dir / train_split / htag
        va_root = seq_dir / val_split / htag

        train_loader, input_dim, _, train_has_sid = _make_loader(tr_root, bs, nw, pin, persist, train=True, prefetch=prefetch)
        val_loader,   _,        _, val_has_sid   = _make_loader(va_root, bs, nw, pin, persist, train=False, prefetch=prefetch)
        has_sid = train_has_sid and val_has_sid

        inferred_n_stations = None
        if station_embed_dim > 0 and n_stations is None and has_sid:
            seen_max = -1
            for b in train_loader:
                if isinstance(b, (list,tuple)) and len(b) == 3:
                    seen_max = max(seen_max, int(b[2].max().item()))
            inferred_n_stations = seen_max + 1 if seen_max >= 0 else None

        # -------- build model via factory (selects implementation) --------
        model = build_hybrid(
            mh,
            input_dim=input_dim,
            has_sid=has_sid,
            n_stations=(n_stations or inferred_n_stations),
            horizon=int(H),
        ).to(dev)

        summary_txt = _summarize_model(model, input_dim=input_dim, horizons=[H])
        print(summary_txt)
        (model_dir / f"model_summary_H{int(H)}.txt").write_text(summary_txt + "\n")

        opt_cls = optim.AdamW if opt_name == "adamw" else optim.Adam
        optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

        if use_plateau:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=plateau_factor,
                patience=plateau_patience, min_lr=plateau_min_lr
            )
        elif use_cosine:
            scheduler = WarmupCosine(optimizer, total_epochs=epochs, warmup_epochs=warmup_epochs, min_lr=cosine_min_lr)
        else:
            scheduler = None

        stopper = EarlyStopper(patience=patience)
        best_val, best_ep = float("inf"), 0
        history: List[Dict[str, float]] = []

        head_key = int(H)  # select the correct head if model returns a dict

        for ep in range(1, int(epochs) + 1):
            tr_loss = train_one_epoch_amp(
                model, train_loader, dev, optimizer, criterion,
                grad_clip, amp_enabled, amp_device, amp_dtype,
                grad_accum_steps=grad_accum_steps, head_key=head_key
            )
            va_loss = evaluate_amp(
                model, val_loader, dev, criterion, amp_enabled, amp_device, amp_dtype,
                head_key=head_key
            )
            history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
            print(f"[H={H}] ep {ep:03d}  train={tr_loss:.6f}  val={va_loss:.6f}")

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            elif isinstance(scheduler, WarmupCosine):
                scheduler.step()

            if va_loss + 1e-12 < best_val:
                best_val, best_ep = va_loss, ep
                torch.save(model.state_dict(), model_dir / f"hybrid_h{int(H)}.pt")

            if stopper.update(va_loss):
                print(f"[H={H}] Early stopping at ep={ep} (best ep={best_ep}, val={best_val:.6f})")
                break

        (model_dir / f"hybrid_h{int(H)}_train_summary.json").write_text(json.dumps({
            "horizon": int(H),
            "best_epoch": int(best_ep),
            "best_val_loss": float(best_val),
            "history": history,
            "params": {
                "device": dev_name, "batch_size": bs, "num_workers": nw, "pin_memory": pin,
                "epochs": epochs, "patience": patience, "optimizer": opt_name, "lr": lr,
                "weight_decay": weight_decay,
                "loss": {
                    "name": str(dl.get("loss", "huber")),
                    "huber_delta": float(dl.get("huber_delta", 1.0)),
                    "target_transform": str(dl.get("target_transform", "none")),
                    "clip_target_max": dl.get("clip_target_max", None),
                },
                "amp": {"enabled": amp_enabled, "dtype": str(amp_dtype), "device": amp_device},
                "impl": impl_name,
                "n_stations": (n_stations or inferred_n_stations),
                "grad_accum_steps": grad_accum_steps,
                "lr_scheduler": (
                    {"name":"reduce_on_plateau","factor":plateau_factor,"patience":plateau_patience,"min_lr":plateau_min_lr}
                    if use_plateau else
                    {"name":"cosine_warmup","warmup_epochs":warmup_epochs,"min_lr":cosine_min_lr}
                    if use_cosine else {"name":"none"}
                ),
                # also persist the raw model.hybrid block for reproducibility
                "model_hybrid_block": mh,
            },
        }, indent=2))

    print(f"[OK] checkpoints → {model_dir}")

if __name__ == "__main__":
    main()