# e_training/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# --------------------------- utils ---------------------------

def _unpack(batch):
    return (batch[0], batch[1]) if not (isinstance(batch, (list, tuple)) and len(batch) == 3) else (batch[0], batch[1])

def _sanitize_batch(
    Xb: torch.Tensor,
    yb: torch.Tensor,
    clamp: Optional[float] = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if not torch.isfinite(Xb).all() or not torch.isfinite(yb).all():
        Xb = torch.nan_to_num(Xb)
        yb = torch.nan_to_num(yb)
    if clamp and clamp > 0:
        Xb = torch.clamp(Xb, -clamp, clamp)
        yb = torch.clamp(yb, -clamp, clamp)
    ok = bool(torch.isfinite(Xb).all() and torch.isfinite(yb).all())
    return Xb, yb, ok

def _as_1d(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 2 and y.shape[1] == 1:
        return y.view(-1)
    if y.ndim == 1:
        return y
    return y  # possibly [B,H]

def _align_outputs_and_targets(
    model_out: torch.Tensor | Dict[int, torch.Tensor],
    yb: torch.Tensor,
    horizons: Optional[Iterable[int]] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    yb = _as_1d(yb)
    target_map: Dict[int, torch.Tensor] = {}
    if yb.ndim == 1:
        target_map[1] = yb
    elif yb.ndim == 2:
        for i in range(yb.shape[1]):
            target_map[i + 1] = yb[:, i]
    else:
        raise ValueError(f"Unsupported target shape: {tuple(yb.shape)}")

    if isinstance(model_out, torch.Tensor):
        pred_map = {1: model_out.view(-1)}
    elif isinstance(model_out, dict):
        pred_map = {int(h): p.view(-1) for h, p in model_out.items()}
    else:
        raise TypeError("model forward must return Tensor or Dict[int, Tensor]")

    keys = set(pred_map) & set(target_map)
    if horizons is not None:
        keys &= set(int(h) for h in horizons)
    if not keys:
        raise RuntimeError("No overlapping horizons between predictions and targets.")

    pred_map = {h: pred_map[h] for h in sorted(keys)}
    target_map = {h: target_map[h] for h in sorted(keys)}
    return pred_map, target_map

# ----------------------- early stopper -----------------------

@dataclass
class EarlyStopper:
    patience: int = 0
    best: float = float("inf")
    steps: int = 0
    enabled: bool = True
    eps: float = 1e-8

    def update(self, value: float) -> bool:
        if not self.enabled or self.patience is None or self.patience <= 0:
            return False
        if value < (self.best - self.eps):
            self.best = value
            self.steps = 0
            return False
        self.steps += 1
        return self.steps >= self.patience

# --------------------- AMP helpers (CUDA/MPS) ----------------

def _amp_context(enabled: bool, device: torch.device, dtype: torch.dtype | None = None):
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    devtype = device.type
    if devtype not in {"cuda", "mps"}:
        return torch.autocast(device_type="cpu", enabled=False)
    dt = dtype or (torch.bfloat16 if devtype == "cuda" else torch.float16)
    return torch.autocast(device_type=devtype, dtype=dt, enabled=True)

def _make_scaler(enabled: bool, device: torch.device):
    # GradScaler is CUDA-only; use no-scaler on MPS/CPU
    return torch.cuda.amp.GradScaler(enabled=(enabled and device.type == "cuda"))

# ------------------------ train / eval ------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    *,
    grad_clip: Optional[float] = None,
    max_steps: Optional[int] = None,
    clamp: Optional[float] = 10.0,
    amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    grad_accum_steps: int = 1,
    loss_reduction: str = "mean_over_horizons",
    horizons: Optional[Iterable[int]] = None,
) -> float:
    model.train()
    total_loss, seen, steps = 0.0, 0, 0
    scaler = _make_scaler(amp, device)
    optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        Xb, yb = _unpack(batch)
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        Xb, yb, ok = _sanitize_batch(Xb, yb, clamp=clamp)
        if not ok:
            continue

        with _amp_context(amp, device, amp_dtype):
            out = model(Xb)
            preds, targets = _align_outputs_and_targets(out, yb, horizons=horizons)
            losses = [criterion(preds[h], targets[h]) for h in preds.keys()]
            loss = torch.stack(losses).sum() if loss_reduction == "sum_over_horizons" else torch.stack(losses).mean()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            do_step = ((steps + 1) % max(1, grad_accum_steps) == 0)
            if do_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if (steps + 1) % max(1, grad_accum_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        bs = yb.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        seen += bs
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    # flush leftover grads if accumulation didn't trigger a step (CUDA scaler path)
    if scaler.is_enabled() and (steps % max(1, grad_accum_steps) != 0):
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return total_loss / max(1, seen)

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    *,
    y_collect_limit: Optional[int] = None,
    clamp: Optional[float] = 10.0,
    amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    loss_reduction: str = "mean_over_horizons",
    horizons: Optional[Iterable[int]] = None,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    total_loss, seen = 0.0, 0
    ys, ps, collected = [], [], 0

    for batch in loader:
        Xb, yb = _unpack(batch)
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        Xb, yb, ok = _sanitize_batch(Xb, yb, clamp=clamp)
        if not ok:
            continue

        with _amp_context(amp, device, amp_dtype):
            out = model(Xb)
            preds, targets = _align_outputs_and_targets(out, yb, horizons=horizons)
            losses = [criterion(preds[h], targets[h]) for h in preds.keys()]
            loss = torch.stack(losses).sum() if loss_reduction == "sum_over_horizons" else torch.stack(losses).mean()

        bs = yb.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        seen += bs

        if y_collect_limit is None or collected < y_collect_limit:
            h0 = sorted(preds.keys())[0]
            y_np = targets[h0].detach().cpu().numpy()
            p_np = preds[h0].detach().cpu().numpy()
            if y_collect_limit is not None and collected + len(y_np) > y_collect_limit:
                keep = y_collect_limit - collected
                y_np = y_np[:keep]; p_np = p_np[:keep]
            ys.append(y_np.astype(np.float64, copy=False))
            ps.append(p_np.astype(np.float64, copy=False))
            collected += len(y_np)

    y_all = np.concatenate(ys) if ys else None
    p_all = np.concatenate(ps) if ps else None
    return total_loss / max(1, seen), y_all, p_all

# ----------------------- checkpointing -----------------------

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
):
    ckpt = {
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    torch.save(ckpt, path)