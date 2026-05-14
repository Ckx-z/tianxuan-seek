"""GNN 分子编码器 — 3 层 GIN 将分子图编码为 256 维向量。

用于 Phase 6 阶段 1: GNN 特征 vs Morgan 指纹对比。
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdchem


# ── 原子特征 ───────────────────────────────────────────
_ATOMIC_NUMS = list(range(1, 119))  # 1..118

def _onehot(val: int, categories: list[int]) -> list[float]:
    return [1.0 if val == c else 0.0 for c in categories]


def _atom_features(atom: rdchem.Atom) -> list[float]:
    """提取单原子特征向量 (~44 维)。"""
    feats = []
    # 原子序数: one-hot 分桶 (常见元素 + other)
    an = atom.GetAtomicNum()
    common = [6, 7, 8, 9, 16, 17, 35, 53]  # C, N, O, F, S, Cl, Br, I
    for c in common:
        feats.append(1.0 if an == c else 0.0)
    feats.append(1.0 if an not in common else 0.0)  # other

    # 度
    deg = atom.GetDegree()
    feats.extend(_onehot(deg, [0, 1, 2, 3, 4]))

    # 形式电荷
    fc = atom.GetFormalCharge()
    feats.extend(_onehot(fc, [-1, 0, 1]))

    # 杂化
    hyb = atom.GetHybridization()
    hyb_map = {Chem.HybridizationType.SP: 0, Chem.HybridizationType.SP2: 1,
               Chem.HybridizationType.SP3: 2, Chem.HybridizationType.SP3D: 3}
    hyb_val = hyb_map.get(hyb, 4)
    feats.extend(_onehot(hyb_val, [0, 1, 2, 3, 4]))

    # 芳香性 / 环内
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.IsInRing()))

    # 氢原子数
    nh = atom.GetTotalNumHs()
    feats.extend(_onehot(nh, [0, 1, 2, 3]))

    # 手性
    chiral = atom.GetChiralTag()
    feats.append(1.0 if chiral in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                   Chem.ChiralType.CHI_TETRAHEDRAL_CCW) else 0.0)

    return feats


def _bond_features(bond: rdchem.Bond) -> list[float]:
    """提取单键特征向量 (5 维)。"""
    bt = bond.GetBondType()
    return [
        float(bt == Chem.BondType.SINGLE),
        float(bt == Chem.BondType.DOUBLE),
        float(bt == Chem.BondType.TRIPLE),
        float(bt == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
    ]


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """SMILES → PyG Data (分子图)。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    atom_feats = [_atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([bf, bf])

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 5), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ── GNN 编码器 ──────────────────────────────────────────
class MoleculeEncoder(nn.Module):
    """3 层 GCN 编码器，输出固定维度分子向量。"""

    def __init__(self, node_dim: int = 29, edge_dim: int = 5,
                 hidden: int = 256, num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(hidden, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(1, num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.node_emb(data.x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                torch.zeros(data.x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)


# ── 配对预测器 ──────────────────────────────────────────
class PairPredictor(nn.Module):
    """Siamese GNN + 交互特征 + MLP → 成膜概率。"""

    def __init__(self, encoder: MoleculeEncoder, hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        # aldehyde_vec + amine_vec + diff + hadamard → 4 * hidden
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, ald_data: Data, am_data: Data) -> torch.Tensor:
        ald_vec = self.encoder(ald_data)
        am_vec = self.encoder(am_data)
        combined = torch.cat([ald_vec, am_vec, ald_vec - am_vec, ald_vec * am_vec], dim=-1)
        return self.mlp(combined).squeeze(-1)


# ── 反应条件增强预测器 ──────────────────────────────────
class CondPairPredictor(nn.Module):
    """Siamese GNN + 反应条件特征 + MLP → 成膜概率。

    在分子嵌入基础上融合反应条件（溶剂、温度、催化剂等），
    让模型同时学习分子结构和实验条件对成膜的影响。
    """

    def __init__(self, encoder: MoleculeEncoder, hidden: int = 256,
                 cond_dim: int = 53):
        super().__init__()
        self.encoder = encoder
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # ald + amine + diff + hadamard + cond_emb → 4*hidden + 64
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden + 64, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, ald_data: Data, am_data: Data,
                conditions: torch.Tensor) -> torch.Tensor:
        ald_vec = self.encoder(ald_data)
        am_vec = self.encoder(am_data)
        cond_vec = self.cond_emb(conditions)
        combined = torch.cat([
            ald_vec, am_vec,
            ald_vec - am_vec, ald_vec * am_vec,
            cond_vec,
        ], dim=-1)
        return self.mlp(combined).squeeze(-1)


# ── 多任务预测器 ────────────────────────────────────────
class MultiTaskPredictor(nn.Module):
    """Siamese GNN + 反应条件 → 多任务预测 (成膜 + 结晶度)。

    共享分子编码器和条件嵌入，任务特定 MLP 头各司其职。
    多任务训练让编码器学到更通用的分子表示。
    """

    def __init__(self, encoder: MoleculeEncoder, hidden: int = 256,
                 cond_dim: int = 53, num_crystallinity: int = 3):
        super().__init__()
        self.encoder = encoder
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        shared_dim = 4 * hidden + 64  # ald+amine+diff+hadamard+cond
        self.shared = nn.Sequential(
            nn.Linear(shared_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # 成膜头 (二分类)
        self.film_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        # 结晶度头 (多分类)
        self.cryst_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_crystallinity),
        )

    def forward(self, ald_data: Data, am_data: Data,
                conditions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ald_vec = self.encoder(ald_data)
        am_vec = self.encoder(am_data)
        cond_vec = self.cond_emb(conditions)
        combined = torch.cat([
            ald_vec, am_vec,
            ald_vec - am_vec, ald_vec * am_vec,
            cond_vec,
        ], dim=-1)
        shared_feat = self.shared(combined)
        film_logits = self.film_head(shared_feat).squeeze(-1)
        cryst_logits = self.cryst_head(shared_feat)
        return film_logits, cryst_logits


# ── 批量图构造 ──────────────────────────────────────────
def collate_graphs(graphs: list[Data]) -> Data:
    """将多个图批量合并为一个大图 (batch 维度)。"""
    from torch_geometric.data import Batch
    return Batch.from_data_list(graphs)


# ── 嵌入提取 ────────────────────────────────────────────
def extract_monomer_embeddings(
    encoder: MoleculeEncoder,
    smiles_list: list[str],
    device: str = "cpu",
) -> np.ndarray:
    """对一批 SMILES 提取 256 维 GNN 嵌入。"""
    encoder.eval()
    embeddings = []
    with torch.no_grad():
        for smi in smiles_list:
            g = smiles_to_graph(smi)
            if g is None:
                embeddings.append(np.zeros(256, dtype=np.float32))
            else:
                g = g.to(device)
                emb = encoder(g).cpu().numpy()
                embeddings.append(emb.ravel())
    return np.array(embeddings, dtype=np.float32)
