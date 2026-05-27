"""v3 模型顶层组装 — 串联 encoder -> attention -> pooling -> heads。

Usage:
    model = V3Model(cfg)
    film_logit, cond_logits = model(ald_data, amine_data)
    film_logit = model.predict(ald_data, amine_data)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.screening.gnn_v3.encoder import SiameseEncoder
from src.screening.gnn_v3.attention import CrossGraphAttention
from src.screening.gnn_v3.pooling import PairPooling
from src.screening.gnn_v3.heads import FilmHead, ConditionHead


class V3Model(nn.Module):
    """v3 成膜预测模型 — GIN+GINE + Cross-Attention + 多头预测。

    Args:
        cfg: 配置字典 (来自 model_v3.yaml) 或默认值
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}

        enc_cfg = cfg.get("encoder", {})
        attn_cfg = cfg.get("attention", {})
        pool_cfg = cfg.get("pooling", {})
        head_cfg = cfg.get("heads", {})
        film_cfg = head_cfg.get("film_head", {})
        cond_cfg = head_cfg.get("condition_head", {})

        hidden_dim = enc_cfg.get("hidden_dim", 256)

        self.encoder = SiameseEncoder(
            in_dim=enc_cfg.get("in_dim", 37),
            hidden_dim=hidden_dim,
            num_layers=enc_cfg.get("num_layers", 5),
            dropout=enc_cfg.get("dropout", 0.1),
        )
        self.attention = CrossGraphAttention(
            hidden_dim=attn_cfg.get("hidden_dim", hidden_dim),
            num_heads=attn_cfg.get("num_heads", 4),
            dropout=attn_cfg.get("dropout", 0.1),
        )
        self.pooling = PairPooling(
            hidden_dim=pool_cfg.get("hidden_dim", hidden_dim),
            num_queries=pool_cfg.get("num_queries", 4),
        )
        self.film_head = FilmHead(
            hidden_dim=hidden_dim,
            dropout=film_cfg.get("dropout", 0.2),
        )
        self.condition_head = ConditionHead(
            hidden_dim=hidden_dim,
            tasks=cond_cfg.get("tasks"),
            dropout=cond_cfg.get("dropout", 0.1),
        )

    def forward(self, ald_data: Data, amine_data: Data,
                return_attn: bool = False
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | \
                   tuple[torch.Tensor, dict[str, torch.Tensor],
                         tuple[torch.Tensor, torch.Tensor]]:
        ald_emb, amine_emb = self.encoder(ald_data, amine_data)
        attn_result = self.attention(ald_emb, amine_emb, return_attn=return_attn)
        if return_attn:
            ald_emb, amine_emb, attn_weights = attn_result
        else:
            ald_emb, amine_emb = attn_result
        ea, eb, e_pair = self.pooling(ald_emb, amine_emb)
        film_logit = self.film_head(ea, eb, e_pair)
        cond_logits = self.condition_head(ea, eb)
        if return_attn:
            return film_logit, cond_logits, attn_weights
        return film_logit, cond_logits

    def predict(self, ald_data: Data, amine_data: Data) -> torch.Tensor:
        with torch.no_grad():
            ald_emb, amine_emb = self.encoder(ald_data, amine_data)
            ald_emb, amine_emb = self.attention(ald_emb, amine_emb)
            ea, eb, e_pair = self.pooling(ald_emb, amine_emb)
            return self.film_head(ea, eb, e_pair)
