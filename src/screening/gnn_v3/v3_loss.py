"""v3 损失函数 — Focal Loss + 条件 CE + 化学正则化 + 对称性正则化。

L_total = L_film (Focal, x film_quality_weight)
        + cond_lambda * sum(L_cond_i)
        + chem_lambda * L_chem  (warm-up)
        + sym_lambda * L_sym
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: torch.Tensor | None = None) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        fl = alpha_t * (1 - pt) ** self.gamma * ce
        if sample_weights is not None:
            fl = fl * sample_weights
        return fl.mean()


def condition_losses(cond_logits: dict[str, torch.Tensor],
                     cond_labels: dict[str, torch.Tensor],
                     cond_masks: dict[str, torch.Tensor] | None = None
                     ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_task = {}
    for task, logits in cond_logits.items():
        labels = cond_labels[task]
        mask = cond_masks[task] if cond_masks else torch.ones_like(labels)
        loss = F.cross_entropy(logits, labels, reduction="none")
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        per_task[task] = loss
    total = sum(per_task.values())
    return total, per_task


def symmetry_loss(ald_attn: torch.Tensor, amine_attn: torch.Tensor) -> torch.Tensor:
    ald_avg = ald_attn.mean(dim=0)
    amine_avg = amine_attn.mean(dim=0)
    diff = ald_avg - amine_avg.t()
    return (diff ** 2).mean()


class V3Loss(nn.Module):
    """v3 总损失函数。"""

    def __init__(self, focal_alpha: float = 0.75, focal_gamma: float = 2.0,
                 cond_lambda: float = 0.1, sym_lambda: float = 0.01,
                 use_film_quality_weight: bool = True):
        super().__init__()
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.cond_lambda = cond_lambda
        self.sym_lambda = sym_lambda
        self.use_film_quality_weight = use_film_quality_weight

    def forward(self, film_logits: torch.Tensor, film_labels: torch.Tensor,
                cond_logits: dict[str, torch.Tensor],
                cond_labels: dict[str, torch.Tensor],
                cond_masks: dict[str, torch.Tensor] | None = None,
                quality_weights: torch.Tensor | None = None,
                ald_attn: torch.Tensor | None = None,
                amine_attn: torch.Tensor | None = None,
                chem_penalty: torch.Tensor | None = None,
                chem_lambda: float = 0.0
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        qw = quality_weights if self.use_film_quality_weight else None
        l_film = self.focal(film_logits, film_labels, qw)
        l_cond, per_cond = condition_losses(cond_logits, cond_labels, cond_masks)
        total = l_film + self.cond_lambda * l_cond

        components = {"film": l_film, "cond": l_cond}
        components.update({f"cond_{k}": v for k, v in per_cond.items()})

        if ald_attn is not None and amine_attn is not None:
            l_sym = symmetry_loss(ald_attn, amine_attn)
            total = total + self.sym_lambda * l_sym
            components["sym"] = l_sym

        if chem_penalty is not None and chem_lambda > 0:
            l_chem = chem_lambda * chem_penalty
            total = total + l_chem
            components["chem"] = l_chem

        components["total"] = total
        return total, components
