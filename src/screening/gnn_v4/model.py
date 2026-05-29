"""v4 模型顶层组装 — encoder -> attention -> pooling -> FilmHead。

与 v3 的区别:
- 更小 (hidden_dim=128, layers=3)
- 无 ConditionHead
- 支持 batch_size > 1
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.screening.gnn_v4.encoder import SiameseEncoder
from src.screening.gnn_v4.attention import CrossGraphAttention
from src.screening.gnn_v4.pooling import PairPooling
from src.screening.gnn_v4.heads import FilmHead


class V4Model(nn.Module):
    """v4 成膜预测模型 — 缩小版 GIN+GINE + Cross-Attention + FilmHead。"""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}

        enc_cfg = cfg.get("encoder", {})
        attn_cfg = cfg.get("attention", {})
        pool_cfg = cfg.get("pooling", {})
        head_cfg = cfg.get("heads", {})
        film_cfg = head_cfg.get("film_head", {})

        hidden_dim = enc_cfg.get("hidden_dim", 128)

        self.encoder = SiameseEncoder(
            in_dim=enc_cfg.get("in_dim", 37),
            hidden_dim=hidden_dim,
            num_layers=enc_cfg.get("num_layers", 3),
            dropout=enc_cfg.get("dropout", 0.15),
        )
        self.attention = CrossGraphAttention(
            hidden_dim=attn_cfg.get("hidden_dim", hidden_dim),
            num_heads=attn_cfg.get("num_heads", 4),
            dropout=attn_cfg.get("dropout", 0.15),
        )
        self.pooling = PairPooling(
            hidden_dim=pool_cfg.get("hidden_dim", hidden_dim),
            num_queries=pool_cfg.get("num_queries", 4),
        )
        self.film_head = FilmHead(
            hidden_dim=hidden_dim,
            dropout=film_cfg.get("dropout", 0.25),
        )

    def forward(self, ald_data: Data, amine_data: Data,
                ald_batch: torch.Tensor, amine_batch: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        ald_emb, amine_emb = self.encoder(ald_data, amine_data)
        ald_emb, amine_emb = self.attention(ald_emb, amine_emb)
        ea, eb, e_pair = self.pooling(ald_emb, amine_emb,
                                       ald_batch, amine_batch, batch_size)
        return self.film_head(ea, eb, e_pair)

    def predict_single(self, ald_data: Data, amine_data: Data) -> torch.Tensor:
        """单样本推理，用于筛选。"""
        with torch.no_grad():
            ald_emb, amine_emb = self.encoder(ald_data, amine_data)
            ald_emb, amine_emb = self.attention(ald_emb, amine_emb)
            # 单样本: batch_size=1
            ald_batch = torch.zeros(ald_emb.shape[0], dtype=torch.long,
                                    device=ald_emb.device)
            amine_batch = torch.zeros(amine_emb.shape[0], dtype=torch.long,
                                      device=amine_emb.device)
            ea, eb, e_pair = self.pooling(ald_emb, amine_emb,
                                           ald_batch, amine_batch, 1)
            return self.film_head(ea, eb, e_pair)
