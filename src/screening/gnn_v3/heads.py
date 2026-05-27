"""v3 预测头 — FilmHead (成膜二分类) + ConditionHead (5 条件辅助任务)。

输入: ea', eb', e_pair 三个图级向量
输出: film_logit [1], condition_logits {task: logits}
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FilmHead(nn.Module):
    """成膜预测头 — [ea' || eb' || ea'⊙eb' || e_pair] → MLP → 1。

    Args:
        hidden_dim: 输入向量维度
        dropout: dropout 率
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        in_dim = hidden_dim * 4
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, ea: torch.Tensor, eb: torch.Tensor,
                e_pair: torch.Tensor) -> torch.Tensor:
        h = torch.cat([ea, eb, ea * eb, e_pair], dim=-1)
        h = self.norm(h)
        return self.mlp(h).squeeze(-1)


class ConditionHead(nn.Module):
    """条件辅助预测头 — 5 个独立分类任务。

    每个任务有独立的投影层 (512→128) 和分类器 (128→num_classes)。
    缺失标签在 loss 中 mask，heads 不做特殊处理。

    Args:
        hidden_dim: 输入向量维度
        tasks: {task_name: num_classes}
        dropout: dropout 率
    """

    def __init__(self, hidden_dim: int = 256,
                 tasks: dict[str, int] | None = None,
                 dropout: float = 0.1):
        super().__init__()
        self.tasks = tasks or {
            "synthesis_route": 5,
            "interface_type": 3,
            "catalyst": 5,
            "solvent": 7,
            "temperature": 5,
        }
        in_dim = hidden_dim * 2

        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for name in self.tasks
        })

        self.classifiers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )
            for name, num_classes in self.tasks.items()
        })

    def forward(self, ea: torch.Tensor, eb: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        h = torch.cat([ea, eb], dim=-1)
        return {
            name: self.classifiers[name](self.projections[name](h))
            for name in self.tasks
        }
