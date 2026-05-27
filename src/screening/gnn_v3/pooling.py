"""v3 注意力池化 — 多头 Attention Pooling + e_pair 构造。

输入: 交叉注意力后的醛/胺原子嵌入
输出: ea' [256], eb' [256], e_pair [256] 三个图级向量
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionPool(nn.Module):
    """多头注意力池化 — 多个 query 学不同原子重要性模式。

    Args:
        hidden_dim: 输入维度
        num_queries: 注意力头数
    """

    def __init__(self, hidden_dim: int = 256, num_queries: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.xavier_uniform_(self.query.unsqueeze(0))
        self.query = nn.Parameter(self.query.squeeze(0))
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.compress = nn.Sequential(
            nn.Linear(num_queries * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.key(x)  # [N, hidden_dim]
        scores = torch.einsum('qd,nd->qn', self.query, k) * self.scale
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(attn, x)
        pooled = pooled.reshape(1, -1)
        return self.compress(pooled).squeeze(0)


class PairPooling(nn.Module):
    """醛胺配对池化 — 构造 ea', eb', e_pair。

    Args:
        hidden_dim: 输入/输出维度
        num_queries: 单体池化注意力头数
    """

    def __init__(self, hidden_dim: int = 256, num_queries: int = 4):
        super().__init__()
        self.ald_pool = MultiHeadAttentionPool(hidden_dim, num_queries)
        self.amine_pool = MultiHeadAttentionPool(hidden_dim, num_queries)

        # e_pair: [ea' || eb' || ea'⊙eb' || pair_cross] → 256
        self.pair_compress = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, ald_emb: torch.Tensor, amine_emb: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ea = self.ald_pool(ald_emb)
        eb = self.amine_pool(amine_emb)

        # 原子对交互矩阵 → 醛→胺加权交互
        pair_matrix = ald_emb @ amine_emb.t()           # [N_ald, N_amine]
        pair_attn = F.softmax(pair_matrix, dim=-1)       # 每醛原子关注胺原子
        pair_cross = pair_attn @ amine_emb               # [N_ald, 256]
        pair_cross = pair_cross.mean(dim=0)              # [256]

        e_pair = self.pair_compress(
            torch.cat([ea, eb, ea * eb, pair_cross])
        )
        return ea, eb, e_pair
