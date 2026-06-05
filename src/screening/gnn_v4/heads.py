"""v4 预测头 — 只有 FilmHead，去掉 ConditionHead，支持 3D 分支。

hidden_dim=128, 更小的 MLP。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FilmHead(nn.Module):
    """成膜预测头 — [ea' || eb' || ea'⊙eb' || e_pair || emb_3d || rule_vec] → MLP → 1。

    Args:
        hidden_dim: 输入向量维度 (2D GNN 部分)
        use_3d: 是否拼接 3D 分支嵌入
        dim_3d: 3D 分支嵌入维度
        use_rules: 是否拼接规则命中向量 (方案 B)
        dim_rules: 规则向量维度
        dropout: dropout 率
    """

    def __init__(self, hidden_dim: int = 128, use_3d: bool = False,
                 dim_3d: int = 64, use_rules: bool = False, dim_rules: int = 23,
                 dropout: float = 0.25):
        super().__init__()
        in_dim = hidden_dim * 4
        if use_3d:
            in_dim += dim_3d
        if use_rules:
            in_dim += dim_rules
        self.use_rules = use_rules
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
                e_pair: torch.Tensor,
                emb_3d: torch.Tensor | None = None,
                rule_vec: torch.Tensor | None = None) -> torch.Tensor:
        parts = [ea, eb, ea * eb, e_pair]
        if emb_3d is not None:
            parts.append(emb_3d)
        if rule_vec is not None and self.use_rules:
            parts.append(rule_vec)
        h = torch.cat(parts, dim=-1)
        h = self.norm(h)
        return self.mlp(h).squeeze(-1)
