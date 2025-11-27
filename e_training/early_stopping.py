# e_training/early_stopping.py
from __future__ import annotations
import math, os, torch

class EarlyStopper:
    """
    Early stopping with optional cooldown and on-best checkpoint.
    mode: 'min' or 'max'
    eps: minimum improvement
    patience: epochs without improvement before stop (>= patience)
    """
    def __init__(
        self,
        patience: int,
        mode: str = "min",
        eps: float = 1e-8,
        cooldown: int = 0,
        checkpoint_path: str | None = None,
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.mode = mode
        self.eps = float(eps)
        self.cooldown = int(cooldown)
        self.ckpt = checkpoint_path

        self.best: float | None = None
        self.bad_epochs = 0
        self.best_epoch: int | None = None
        self.cooldown_counter = 0
        self.stop_triggered = False

    def _improved(self, v: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return v < self.best - self.eps
        return v > self.best + self.eps

    def step(self, value: float, model: torch.nn.Module | None = None, epoch: int | None = None) -> bool:
        if not math.isfinite(value):
            return False  # ignore NaN/Inf

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        if self._improved(value):
            self.best = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            self.cooldown_counter = self.cooldown
            if self.ckpt and model is not None:
                os.makedirs(os.path.dirname(self.ckpt), exist_ok=True)
                torch.save(model.state_dict(), self.ckpt)
            return False

        self.bad_epochs += 1
        self.stop_triggered = self.bad_epochs >= self.patience
        return self.stop_triggered

    def reset(self):
        self.best = None
        self.best_epoch = None
        self.bad_epochs = 0
        self.cooldown_counter = 0
        self.stop_triggered = False

    @property
    def state(self) -> dict:
        return {
            "best": self.best,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "cooldown_left": self.cooldown_counter,
            "stop_triggered": self.stop_triggered,
        }