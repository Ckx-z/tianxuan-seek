"""v3 调度器 — chem warm-up + 学习率调度。"""
from __future__ import annotations

import math
import torch.optim as optim


class ChemWarmupScheduler:
    """化学正则化 warm-up: 前 warmup_frac=0, ramp_frac 线性增, hold_frac 恒定。"""

    def __init__(self, max_epochs: int, warmup_frac: float = 0.5,
                 ramp_frac: float = 0.3, hold_frac: float = 0.2,
                 lambda_max: float = 0.005):
        self.warmup_end = int(max_epochs * warmup_frac)
        self.ramp_end = int(max_epochs * (warmup_frac + ramp_frac))
        self.lambda_max = lambda_max

    def get_lambda(self, epoch: int) -> float:
        if epoch < self.warmup_end:
            return 0.0
        if epoch >= self.ramp_end:
            return self.lambda_max
        progress = (epoch - self.warmup_end) / (self.ramp_end - self.warmup_end)
        return self.lambda_max * progress


def build_optimizer(model_params, lr: float = 0.001,
                    weight_decay: float = 0.0001,
                    optimizer_name: str = "adamw") -> optim.Optimizer:
    if optimizer_name == "adamw":
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)


def build_lr_scheduler(optimizer: optim.Optimizer, max_epochs: int,
                       warmup_epochs: int = 5,
                       scheduler_name: str = "cosine"
                       ) -> optim.lr_scheduler.LRScheduler:
    if scheduler_name == "cosine":
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )
    return optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
