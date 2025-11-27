# e_training/losses.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn

TargetTransform = Literal["none", "log1p"]
LossName        = Literal["mse", "huber"]
Reduction       = Literal["mean", "sum", "none"]


@dataclass
class LossConfig:
    name: LossName = "huber"
    huber_delta: float = 1.0
    target_transform: TargetTransform = "none"
    clip_target_max: Optional[float] = None   # clip upper bound (original scale)
    clip_pred_only: bool = False              # if True, clip only predictions
    reduction: Reduction = "mean"
    log1p_eps: float = 0.0                    # floor before log1p (e.g., 1e-8)


class _TargetTransform(nn.Module):
    def __init__(self, method: TargetTransform = "none", eps: float = 0.0):
        super().__init__()
        self.method = method
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "none":
            return x
        if self.method == "log1p":
            # enforce non-negativity with small floor if provided
            return torch.log1p(torch.clamp_min(x, self.eps if self.eps > 0.0 else 0.0))
        raise ValueError(f"Unknown target_transform={self.method}")


class LossWrapper(nn.Module):
    """
    Wraps a base loss with optional clipping and a shared target transform.
    Model should emit ORIGINAL-scale predictions.
    """
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.name == "mse":
            self.base = nn.MSELoss(reduction=cfg.reduction)
        elif cfg.name == "huber":
            self.base = nn.HuberLoss(delta=cfg.huber_delta, reduction=cfg.reduction)
        else:
            raise ValueError(f"Unknown loss name: {cfg.name}")
        self.txfm = _TargetTransform(cfg.target_transform, eps=cfg.log1p_eps)

    def _maybe_clip(self, y_hat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        maxv = self.cfg.clip_target_max
        if maxv is None:
            return y_hat, y
        maxv = float(maxv)
        if self.cfg.clip_pred_only:
            return torch.clamp_max(y_hat, maxv), y
        return torch.clamp_max(y_hat, maxv), torch.clamp_max(y, maxv)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # clip in ORIGINAL scale first
        y_hat, y = self._maybe_clip(y_hat, y)
        # apply the SAME transform to both (keeps symmetry)
        y_hat_t = self.txfm(y_hat)
        y_t     = self.txfm(y)
        return self.base(y_hat_t, y_t)