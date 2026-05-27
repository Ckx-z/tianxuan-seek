"""v3 GNN 编码器 — GIN+GINE x5 + JK-Net (mean) + 残差 + LayerNorm。

输入: PyG Data (x, edge_index, edge_attr)
输出: JK-Net mean 聚合后的原子嵌入 [N, hidden_dim] + 每层 hidden_states
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from torch_geometric.data import Data


class GINELayer(nn.Module):
    """单层 GINE: 边特征经 MLP 映射后加法式融入消息传递, eps=0。"""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(self.node_mlp, edge_dim=hidden_dim, eps=0.0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        edge_emb = self.edge_mlp(edge_attr)
        h = self.conv(x, edge_index, edge_emb)
        h = self.norm(h)
        h = self.dropout(h)
        return h


class GINEncoder(nn.Module):
    """GIN+GINE x5 + JK-Net (mean) + 残差 + LayerNorm。

    Args:
        in_dim: 输入原子特征维度 (37)
        hidden_dim: 隐藏层维度 (256)
        num_layers: GIN 层数 (5)
        dropout: dropout 率
    """

    def __init__(self, in_dim: int = 37, hidden_dim: int = 256,
                 num_layers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GINELayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, data: Data) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.input_proj(x)
        hidden_states = []

        for layer in self.layers:
            h_new = layer(h, edge_index, edge_attr)
            h = h + h_new
            hidden_states.append(h)

        stacked = torch.stack(hidden_states, dim=0)
        pooled = stacked.mean(dim=0)
        pooled = self.output_norm(pooled)

        return pooled, hidden_states


class SiameseEncoder(nn.Module):
    """Siamese 共享编码器 — 醛和胺用同一套 GIN 权重，角色标记在 featurizer 中区分。"""

    def __init__(self, in_dim: int = 37, hidden_dim: int = 256,
                 num_layers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim, num_layers, dropout)

    def forward(self, ald_data: Data, amine_data: Data
                ) -> tuple[torch.Tensor, torch.Tensor]:
        ald_emb, _ = self.encoder(ald_data)
        amine_emb, _ = self.encoder(amine_data)
        return ald_emb, amine_emb
