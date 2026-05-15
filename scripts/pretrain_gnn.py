"""MolCLR 分子图对比学习预训练。

用全部 1044 个唯一 SMILES 做自监督对比学习，
让 GNN 编码器学会通用分子表示，再微调到成膜任务。

核心思路:
  - 同一分子的两个增强视图 → 嵌入靠近
  - 不同分子的嵌入 → 推远
  - NT-Xent 损失 + 节点/边 Dropout 增强
"""
import argparse
import os
import sys
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn import MoleculeEncoder, smiles_to_graph, collate_graphs
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("molclr")

DEVICE = "cpu"
HIDDEN = 256
PROJECTION_DIM = 128
EPOCHS = 300
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-5
TEMPERATURE = 0.07
NODE_DROPOUT = 0.15
EDGE_DROPOUT = 0.15


class MolCLR(nn.Module):
    """分子图对比学习框架。

    Encoder → Projection → L2 normalize → NT-Xent loss
    """

    def __init__(self, encoder: MoleculeEncoder, hidden: int = 256,
                 proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, data):
        h = self.encoder(data)
        z = self.projection(h)
        return F.normalize(z, dim=-1)


def augment_graph(data, node_p: float = 0.15, edge_p: float = 0.15):
    """对分子图做数据增强，返回两个增强视图。

    Aug 1: node feature dropout (随机遮罩原子特征)
    Aug 2: edge dropout (随机删除边)
    """
    x = data.x.clone()
    edge_index = data.edge_index.clone()

    # Aug 1: node feature dropout
    mask1 = torch.rand(x.shape, device=x.device) > node_p
    x1 = x * mask1.float()

    # Aug 2: edge dropout
    n_edges = edge_index.size(1)
    keep_mask = torch.rand(n_edges, device=edge_index.device) > edge_p
    edge_index2 = edge_index[:, keep_mask]
    if edge_index2.size(1) == 0:
        edge_index2 = edge_index[:, :1]  # 保留至少一条边

    from torch_geometric.data import Data
    g1 = Data(x=x1, edge_index=edge_index)
    g2 = Data(x=x, edge_index=edge_index2)
    return g1, g2


def load_pretrain_smiles() -> list[str]:
    """从所有可用源收集唯一 SMILES。"""
    import json

    import pandas as pd
    from rdkit import Chem

    smiles_set = set()

    # 来源 1: label_metadata.csv
    meta_path = "data/processed/label_metadata.csv"
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path, encoding="utf-8-sig")
        for col in ["aldehyde_smiles", "amine_smiles"]:
            for s in meta[col].dropna():
                mol = Chem.MolFromSmiles(s)
                if mol:
                    smiles_set.add(Chem.MolToSmiles(mol, isomericSmiles=True))

    # 来源 2: monomer_smiles_cache.json
    cache_path = "data/processed/monomer_smiles_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        for v in cache.values():
            if isinstance(v, str) and v:
                mol = Chem.MolFromSmiles(v)
                if mol:
                    smiles_set.add(Chem.MolToSmiles(mol, isomericSmiles=True))

    # 过滤过小分子 (单个原子/离子)
    result = []
    for s in sorted(smiles_set):
        mol = Chem.MolFromSmiles(s)
        if mol and mol.GetNumAtoms() >= 3:
            result.append(s)
    return result


def pretrain_molclr(smiles_list: list[str], model_dir: str):
    """MolCLR 对比学习预训练主循环。"""
    # 转图
    logger.info(f"将 {len(smiles_list)} 个 SMILES 转为图...")
    graphs = []
    for s in smiles_list:
        g = smiles_to_graph(s)
        if g is not None:
            graphs.append(g)
    logger.info(f"有效图: {len(graphs)}")

    encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
    model = MolCLR(encoder, hidden=HIDDEN, proj_dim=PROJECTION_DIM)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n = len(graphs)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"开始预训练: {EPOCHS} epochs, batch={BATCH_SIZE}, {n_batches} batches/epoch")

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = np.random.permutation(n)
        epoch_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            batch_graphs = [graphs[i] for i in batch_idx]

            # 为每个分子生成两个增强视图
            g1_list, g2_list = [], []
            for g in batch_graphs:
                g1, g2 = augment_graph(g, NODE_DROPOUT, EDGE_DROPOUT)
                g1_list.append(g1)
                g2_list.append(g2)

            batch_g1 = collate_graphs(g1_list)
            batch_g2 = collate_graphs(g2_list)

            # 拼接: [batch, batch] — 前半是视图1, 后半是视图2
            combined = collate_graphs(g1_list + g2_list)
            z = model(combined)  # [2*batch, proj_dim]

            # 分离
            z1, z2 = z[:len(g1_list)], z[len(g1_list):]

            # 余弦相似度矩阵 [batch, batch]
            sim = torch.mm(z1, z2.t()) / TEMPERATURE

            # 标签: 对角线是正对
            labels = torch.arange(len(g1_list), device=sim.device)

            # 对称 NT-Xent
            loss_z1 = F.cross_entropy(sim, labels)
            loss_z2 = F.cross_entropy(sim.t(), labels)
            loss = (loss_z1 + loss_z2) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % 30 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d}/{EPOCHS}: loss={avg_loss:.4f}")

        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            best_state = deepcopy(encoder.state_dict())

    # 保存
    encoder_path = os.path.join(model_dir, "gnn_encoder_pretrained.pt")
    torch.save(best_state, encoder_path)
    logger.info(f"预训练编码器已保存: {encoder_path}")
    logger.info(f"最佳 loss: {best_loss:.4f}")
    return best_state


def main():
    parser = argparse.ArgumentParser(description="MolCLR 分子图对比学习预训练")
    parser.add_argument("--model-dir", default="models/v2.0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    global EPOCHS, BATCH_SIZE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("=== MolCLR 对比学习预训练 ===")
    logger.info(f"超参: HIDDEN={HIDDEN}, PROJ={PROJECTION_DIM}, "
                f"TEMP={TEMPERATURE}, node_p={NODE_DROPOUT}, edge_p={EDGE_DROPOUT}")

    smiles_list = load_pretrain_smiles()
    logger.info(f"收集到 {len(smiles_list)} 个唯一 SMILES (≥3 原子)")

    pretrain_molclr(smiles_list, args.model_dir)

    print("\n" + "=" * 50)
    print("  MolCLR 预训练完成")
    print(f"  预训练数据: {len(smiles_list)} SMILES")
    print(f"  编码器: models/v2.0/gnn_encoder_pretrained.pt")
    print("=" * 50)


if __name__ == "__main__":
    main()
