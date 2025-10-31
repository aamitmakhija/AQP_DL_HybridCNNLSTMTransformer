# e_training/losses.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn

TargetTransform = Literal["none", "log1p"]

@dataclass
class LossConfig:
    name: Literal["mse", "huber"] = "huber"
    huber_delta: float = 1.0
    target_transform: TargetTransform = "none"
    clip_target_max: Optional[float] = None  # clip both y and y_hat to this max (original scale)

class _TargetTransform(nn.Module):
    def __init__(self, method: TargetTransform = "none"):
        super().__init__()
        self.method = method

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.method == "none":
            return y
        if self.method == "log1p":
            # stabilize negatives (shouldn't happen for PM2.5, but guard anyway)
            return torch.log1p(torch.clamp_min(y, 0.0))
        raise ValueError(f"Unknown target_transform={self.method}")

class LossWrapper(nn.Module):
    """
    Applies optional clipping/transform consistently to y and y_hat
    before computing loss. Model should output ORIGINAL scale.
    """
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.name == "mse":
            self.base = nn.MSELoss()
        elif cfg.name == "huber":
            self.base = nn.HuberLoss(delta=cfg.huber_delta)
        else:
            raise ValueError(f"Unknown loss name: {cfg.name}")

        self.txfm = _TargetTransform(cfg.target_transform)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Optional clipping in ORIGINAL scale
        if self.cfg.clip_target_max is not None:
            maxv = float(self.cfg.clip_target_max)
            y_hat = torch.clamp_max(y_hat, maxv)
            y = torch.clamp_max(y, maxv)
        # Optional shared transform
        y_hat_t = self.txfm(y_hat)
        y_t     = self.txfm(y)
        return self.base(y_hat_t, y_t)