# e_training/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


# --------------------------- utils ---------------------------

def _sanitize_batch(
    Xb: torch.Tensor,
    yb: torch.Tensor,
    clamp: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Replace NaN/±Inf with finite values and clamp magnitudes.
    Returns (Xb_sanitized, yb_sanitized, ok_flag).
    """
    # Fast path: if already finite, just (optionally) clamp
    if torch.isfinite(Xb).all() and torch.isfinite(yb).all():
        if clamp and clamp > 0:
            Xb = torch.clamp(Xb, -clamp, clamp)
        return Xb, yb, True

    # Replace non-finites
    Xb = torch.nan_to_num(Xb)  # NaN->0, +Inf->max, -Inf->min
    yb = torch.nan_to_num(yb)

    # Clamp extremes to a reasonable range
    if clamp and clamp > 0:
        Xb = torch.clamp(Xb, -clamp, clamp)
        yb = torch.clamp(yb, -clamp, clamp)

    ok = bool(torch.isfinite(Xb).all() and torch.isfinite(yb).all())
    return Xb, yb, ok


# ----------------------- early stopper -----------------------

@dataclass
class EarlyStopper:
    patience: int = 0                 # 0/None disables early stopping
    best: float = float("inf")
    steps: int = 0
    enabled: bool = True

    def update(self, value: float) -> bool:
        if not self.enabled or self.patience is None or self.patience <= 0:
            return False
        if value < self.best - 1e-12:
            self.best = value
            self.steps = 0
            return False
        self.steps += 1
        return self.steps > self.patience


# ------------------------ train / eval ------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    grad_clip: Optional[float] = None,
    max_steps: Optional[int] = None,
    clamp: float = 10.0,         # defensive clamp range for inputs/targets
) -> float:
    """
    Train for one epoch. Returns mean loss over *examples* (not batches).
    Shapes:
      - model(X) -> [B] or [B,1]; will be flattened to [B]
      - y -> [B] or [B,1]; will be flattened to [B]
    """
    model.train()
    total_loss = 0.0
    seen = 0
    steps = 0

    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).view(-1)

        # Defensive: sanitize batch to avoid NaN/Inf explosions
        Xb, yb, ok = _sanitize_batch(Xb, yb, clamp=clamp)
        if not ok:
            # If still not finite after sanitization, skip this batch
            continue

        optimizer.zero_grad(set_to_none=True)
        preds = model(Xb).view(-1)       # [B]
        loss = criterion(preds, yb)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = yb.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        seen += bs
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    return total_loss / max(1, seen)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    y_collect_limit: Optional[int] = None,
    clamp: float = 10.0,         # keep eval numerically stable too
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Evaluate; returns (mean_loss, y_concat_or_None, pred_concat_or_None).
    If y_collect_limit is set, caps the concatenated arrays to that size.
    """
    model.eval()
    total_loss = 0.0
    seen = 0

    ys, ps = [], []
    collected = 0

    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).view(-1)

        # Defensive: sanitize batch
        Xb, yb, ok = _sanitize_batch(Xb, yb, clamp=clamp)
        if not ok:
            continue

        preds = model(Xb).view(-1)
        loss = criterion(preds, yb)

        bs = yb.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        seen += bs

        if y_collect_limit is None or collected < y_collect_limit:
            y_np = yb.detach().cpu().numpy()
            p_np = preds.detach().cpu().numpy()
            if y_collect_limit is not None and collected + len(y_np) > y_collect_limit:
                keep = y_collect_limit - collected
                y_np = y_np[:keep]
                p_np = p_np[:keep]
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
    """
    Save a simple checkpoint with epoch, val_loss, model & optimizer states.
    """
    ckpt = {
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    torch.save(ckpt, path)