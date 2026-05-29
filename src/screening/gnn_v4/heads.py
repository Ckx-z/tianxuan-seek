"""v4 预测头 — 只有 FilmHead，去掉 ConditionHead。

hidden_dim=128, 更小的 MLP。
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

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        in_dim = hidden_dim * 4
        self.norm = nn.LayerNorm(in_dim)
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
