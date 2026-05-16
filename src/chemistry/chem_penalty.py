r"""化学惩罚项 — 将筛选硬规则连续化为可微损失，注入训练。

设计原则:
  1. 每条规则定义连续化「违反度」v ∈ [0, 1]
  2. 仅惩罚预测为正 (pred > 0.5) 的样本 — 语义是「不能对违规自信判正」
  3. 惩罚强度由规则化学重要性加权: w_sym > w_para > w_rings > w_substituent

L_total = L_focal + \lambda \cdot \sum_i w_i \cdot v_i \cdot \mathbb{1}[pred > 0.5]
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, CanonicalRankAtoms

_ALD_SMARTS = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
_AM_SMARTS = Chem.MolFromSmarts("[NH2][c]")
_BENZENE_SMARTS = Chem.MolFromSmarts("c1ccccc1")
_HALOGENS = {9, 17, 35, 53}

# ── 规则权重 (化学重要性排序) ──
DEFAULT_WEIGHTS = {
    "symmetry": 0.5,       # 对称性是最重要的 2D 约束
    "para": 0.4,            # C2 对位是几何硬要求
    "rings": 0.3,           # 芳环过多 → 空间位阻
    "substituent": 0.3,     # 取代基过多 (>4 非卤素) → 位阻+电子干扰
}


def _topology(n_ald: int, n_am: int) -> str:
    if n_ald >= 3 or n_am >= 3:
        return "C3"
    if n_ald >= 2 or n_am >= 2:
        return "C2"
    return "C1"


def _violation_symmetry(mol: Chem.Mol, n_ald: int, n_am: int, topo: str) -> float:
    """不对称违反度: 不对称 C2/C3 → 1.0, 对称 → 0.0"""
    if topo not in ("C2", "C3"):
        return 0.0
    if n_ald >= 2:
        matches = mol.GetSubstructMatches(_ALD_SMARTS)
        reactive = [m[0] for m in matches]
    elif n_am >= 2:
        matches = mol.GetSubstructMatches(_AM_SMARTS)
        reactive = [m[0] for m in matches]
    else:
        return 1.0  # <2 反应位点，严重违规
    if len(reactive) < 2:
        return 1.0
    ranks = CanonicalRankAtoms(mol, breakTies=False)
    reactive_ranks = [ranks[a] for a in reactive]
    if all(r == reactive_ranks[0] for r in reactive_ranks[1:]):
        return 0.0
    return 1.0


def _violation_para(mol: Chem.Mol, n_ald: int, n_am: int, topo: str) -> float:
    """C2 非对位违反度: 同环间位/邻位 → 1.0, 对位 → 0.0"""
    if topo != "C2":
        return 0.0
    if n_ald >= 2:
        matches = mol.GetSubstructMatches(_ALD_SMARTS)
        reactive_atoms = [m[2] for m in matches]
    elif n_am >= 2:
        matches = mol.GetSubstructMatches(_AM_SMARTS)
        reactive_atoms = [m[1] for m in matches]
    else:
        return 1.0
    if len(reactive_atoms) < 2:
        return 1.0

    rings = mol.GetSubstructMatches(_BENZENE_SMARTS)
    for ring in rings:
        ring_set = set(ring)
        on_ring = [a for a in reactive_atoms if a in ring_set]
        if len(on_ring) < 2:
            continue
        for i in range(len(on_ring)):
            for j in range(i + 1, len(on_ring)):
                path = Chem.GetShortestPath(mol, on_ring[i], on_ring[j])
                ring_bonds = sum(
                    1 for k in range(len(path) - 1)
                    if path[k] in ring_set and path[k + 1] in ring_set
                )
                if ring_bonds == 3:
                    return 0.0
                elif ring_bonds == 2:   # 间位
                    return 0.8
                elif ring_bonds == 1:   # 邻位
                    return 1.0
    return 0.0  # 不在同环 → 通过


def _violation_rings(mol: Chem.Mol) -> float:
    """芳环过多违反度: 连续化, >4 环部分归一化到 [0, 1]"""
    n = Descriptors.NumAromaticRings(mol)
    if n <= 4:
        return 0.0
    return min(1.0, (n - 4) / 6.0)


def _violation_substituent(mol: Chem.Mol) -> float:
    """非卤素过取代替反度: 每多 1 个非卤素取代 +0.25, 上限 1.0"""
    rings = mol.GetSubstructMatches(_BENZENE_SMARTS)
    max_excess = 0
    for ring in rings:
        ring_set = set(ring)
        n_sub, n_nonhalo = 0, 0
        for aidx in ring:
            atom = mol.GetAtomWithIdx(aidx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in ring_set:
                    n_sub += 1
                    an = nbr.GetAtomicNum()
                    if an not in _HALOGENS and an != 1:
                        n_nonhalo += 1
        if n_sub > 4:
            max_excess = max(max_excess, n_nonhalo)
    return min(1.0, max_excess / 4.0)


def compute_pair_violations(ald_smi: str, am_smi: str) -> Dict[str, float]:
    """计算一对单体的化学规则违反度, 返回 {rule_name: violation_score}。

    每个 score ∈ [0, 1], 0 = 完全遵守, 1 = 严重违反。
    """
    ald = Chem.MolFromSmiles(ald_smi)
    am = Chem.MolFromSmiles(am_smi)
    if ald is None or am is None:
        return {"invalid": 1.0}

    n_ald = len(ald.GetSubstructMatches(_ALD_SMARTS, uniquify=True))
    n_am_ald = len(ald.GetSubstructMatches(_AM_SMARTS, uniquify=True))
    n_ald_am = len(am.GetSubstructMatches(_ALD_SMARTS, uniquify=True))
    n_am = len(am.GetSubstructMatches(_AM_SMARTS, uniquify=True))

    ald_topo = _topology(n_ald, n_am_ald)
    am_topo = _topology(n_ald_am, n_am)

    return {
        "symmetry": max(
            _violation_symmetry(ald, n_ald, n_am_ald, ald_topo),
            _violation_symmetry(am, n_ald_am, n_am, am_topo),
        ),
        "para": max(
            _violation_para(ald, n_ald, n_am_ald, ald_topo),
            _violation_para(am, n_ald_am, n_am, am_topo),
        ),
        "rings": max(_violation_rings(ald), _violation_rings(am)),
        "substituent": max(_violation_substituent(ald), _violation_substituent(am)),
    }


def chem_penalty_loss(preds: torch.Tensor, ald_smiles: List[str],
                      am_smiles: List[str],
                      weights: Dict[str, float] | None = None,
                      threshold: float = 0.5) -> torch.Tensor:
    """计算批量的化学惩罚损失。

    Args:
        preds: 模型预测概率 (sigmoid 后), shape (N,)
        ald_smiles: 醛 SMILES 列表
        am_smiles: 胺 SMILES 列表
        weights: 规则权重, 默认 DEFAULT_WEIGHTS
        threshold: 仅惩罚 pred > threshold 的样本

    Returns:
        标量惩罚损失 (已对 batch 取平均, 无违规时为 0)
    """
    w = weights or DEFAULT_WEIGHTS
    violations = []
    for ald, am in zip(ald_smiles, am_smiles):
        v = compute_pair_violations(ald, am)
        total_v = sum(w.get(k, 0.0) * v.get(k, 0.0) for k in w)
        violations.append(total_v)

    v_tensor = torch.tensor(violations, dtype=torch.float32, device=preds.device)
    # 仅惩罚 pred > threshold 的样本
    mask = (preds > threshold).float()
    penalty = (mask * v_tensor).mean()
    return penalty


# ── 缓存: 预计算训练集所有配对的违反度 ──

class ViolationCache:
    """预计算训练集违反度, 避免每 epoch 重复 RDKit 计算。"""

    def __init__(self, ald_smiles_list: List[str], am_smiles_list: List[str],
                 weights: Dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.scores: List[float] = []
        for ald, am in zip(ald_smiles_list, am_smiles_list):
            v = compute_pair_violations(ald, am)
            total = sum(self.weights.get(k, 0.0) * v.get(k, 0.0) for k in self.weights)
            self.scores.append(total)

    def to_tensor(self, indices: List[int], device: str = "cpu") -> torch.Tensor:
        return torch.tensor([self.scores[i] for i in indices],
                           dtype=torch.float32, device=device)

    def mean_violation(self) -> float:
        return float(np.mean(self.scores))

    def violation_summary(self) -> Dict[str, float]:
        arr = np.array(self.scores)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "nonzero_frac": float((arr > 0).mean()),
        }
