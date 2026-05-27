r"""化学惩罚项 — 3 条化学硬事实规则，将化学先验注入训练。

仅保留几乎无例外的化学硬事实: 苯环必需 / 官能团对称 / C2 对位。
其余规则（取代基、溶解性、刚性、共轭、位阻）交给 GNN 自己学。

设计原则:
  1. 每条规则定义连续化「违反度」v in [0, 1]
  2. 只罚假阳性: penalty = violation * prob（通过 prob 回传梯度）
  3. lambda_chem=0.005
  4. Warm-up: 前 50% epochs lambda=0, 50%-80% 线性增加, 最后 20% 恒定
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, CanonicalRankAtoms

_ALD_SMARTS = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
_AM_SMARTS = Chem.MolFromSmarts("[NH2][c]")
_BENZENE_SMARTS = Chem.MolFromSmarts("c1ccccc1")
_HALOGENS = {9, 17, 35, 53}

DEFAULT_WEIGHTS = {
    "phenyl": 0.5, "symmetry": 0.5, "para": 0.4,
}


# ── 辅助 ──────────────────────────────────────────────────

def _topology(n_ald: int, n_am: int) -> str:
    if n_ald >= 3 or n_am >= 3:
        return "C3"
    if n_ald >= 2 or n_am >= 2:
        return "C2"
    return "C1"


# ── 单体级 (3 条) ────────────────────────────────────────

def _violation_phenyl(mol: Chem.Mol) -> float:
    n = len(mol.GetSubstructMatches(_BENZENE_SMARTS))
    return 1.0 - min(n, 1.0)


def _violation_symmetry(mol: Chem.Mol, n_ald: int, n_am: int, topo: str) -> float:
    if topo not in ("C2", "C3"):
        return 0.0
    if n_ald >= 2:
        matches = mol.GetSubstructMatches(_ALD_SMARTS)
        reactive = [m[0] for m in matches]
    elif n_am >= 2:
        matches = mol.GetSubstructMatches(_AM_SMARTS)
        reactive = [m[0] for m in matches]
    else:
        return 1.0
    if len(reactive) < 2:
        return 1.0
    ranks = CanonicalRankAtoms(mol, breakTies=False)
    reactive_ranks = [ranks[a] for a in reactive]
    if all(r == reactive_ranks[0] for r in reactive_ranks[1:]):
        return 0.0
    return 1.0


def _violation_para(mol: Chem.Mol, n_ald: int, n_am: int, topo: str) -> float:
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
                elif ring_bonds == 2:
                    return 0.8
                elif ring_bonds == 1:
                    return 1.0
    return 0.0


# ── 主接口 ────────────────────────────────────────────────

def compute_pair_violations(ald_smi: str, am_smi: str) -> Dict[str, float]:
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
        "phenyl": max(_violation_phenyl(ald), _violation_phenyl(am)),
        "symmetry": max(
            _violation_symmetry(ald, n_ald, n_am_ald, ald_topo),
            _violation_symmetry(am, n_ald_am, n_am, am_topo),
        ),
        "para": max(
            _violation_para(ald, n_ald, n_am_ald, ald_topo),
            _violation_para(am, n_ald_am, n_am, am_topo),
        ),
    }


def chem_penalty_loss(preds: torch.Tensor, ald_smiles: List[str],
                      am_smiles: List[str],
                      weights: Dict[str, float] | None = None,
                      threshold: float = 0.5) -> torch.Tensor:
    w = weights or DEFAULT_WEIGHTS
    violations = []
    for ald, am in zip(ald_smiles, am_smiles):
        v = compute_pair_violations(ald, am)
        total_v = sum(w.get(k, 0.0) * v.get(k, 0.0) for k in w)
        violations.append(total_v)

    v_tensor = torch.tensor(violations, dtype=torch.float32, device=preds.device)
    mask = (preds > threshold).float()
    return (mask * v_tensor).mean()


class ViolationCache:
    """预计算训练集违反度, 避免每 epoch 重复 RDKit 计算。"""

    def __init__(self, ald_smiles_list: List[str], am_smiles_list: List[str],
                 weights: Dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.scores: List[float] = []
        self.details: List[Dict[str, float]] = []
        for ald, am in zip(ald_smiles_list, am_smiles_list):
            v = compute_pair_violations(ald, am)
            total = sum(self.weights.get(k, 0.0) * v.get(k, 0.0) for k in self.weights)
            self.scores.append(total)
            self.details.append(v)

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

    def rule_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for rule in DEFAULT_WEIGHTS:
            vals = [d.get(rule, 0.0) for d in self.details]
            arr = np.array(vals)
            summary[rule] = {
                "mean": float(np.mean(arr)),
                "nonzero_frac": float((arr > 0).mean()),
            }
        return summary
