"""氟检测与虚拟氟化模块。

- FluorineDetector: SMARTS 模式检测 F 原子、CF3 基团
- virtual_fluorination: 在芳香环上将 H 替换为 F（启发式虚拟氟化）
"""
from typing import Any, Dict, Optional

from rdkit import Chem
from rdkit.Chem import AllChem


class FluorineDetector:
    """氟检测器 — SMARTS 模式 + 描述符。"""

    F_SMARTS = "[F]"
    CF3_SMARTS = "[F][C](F)(F)"

    def __init__(self):
        self._f_pat = Chem.MolFromSmarts(self.F_SMARTS)
        self._cf3_pat = Chem.MolFromSmarts(self.CF3_SMARTS)

    def has_fluorine(self, mol: Chem.Mol) -> bool:
        """检测是否含氟原子。"""
        return mol.HasSubstructMatch(self._f_pat)

    def count_fluorine(self, mol: Chem.Mol) -> int:
        """统计氟原子数量。"""
        return len(mol.GetSubstructMatches(self._f_pat, uniquify=True))

    def has_cf3(self, mol: Chem.Mol) -> bool:
        """检测是否含三氟甲基 (-CF3)。"""
        return mol.HasSubstructMatch(self._cf3_pat)

    def get_fluorine_info(self, mol: Chem.Mol) -> Dict[str, Any]:
        """返回氟相关信息汇总。"""
        return {
            "has_f": self.has_fluorine(mol),
            "n_f": self.count_fluorine(mol),
            "has_cf3": self.has_cf3(mol),
        }


def virtual_fluorination(smiles: str, n_f: int = 1, max_attempts: int = 10) -> Optional[str]:
    """虚拟氟化：在芳香环上替换氢为氟。

    启发式算法（不代表真实化学可行性）：
    1. 找到 [cH]（芳香碳连一个氢）的位点
    2. 优先选择位阻最小（邻位取代最少）的位点
    3. 替换 H → F 并 Sanitize

    Args:
        smiles: 输入 SMILES
        n_f: 添加氟原子数（默认 1）
        max_attempts: 最大尝试次数

    Returns:
        氟化后的 Canonical SMILES，失败返回 None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        mol = Chem.AddHs(mol)
    except Exception:
        return None

    # 找到芳香碳连氢位点 [cH]
    ch_pat = Chem.MolFromSmarts("[cH]")
    if ch_pat is None:
        return None

    matches = mol.GetSubstructMatches(ch_pat)
    if len(matches) < n_f:
        # 没有足够 [cH] 位点 → 尝试脂肪链上的 H
        ch_pat = Chem.MolFromSmarts("[CH1,CH2,CH3]")
        if ch_pat is None:
            return None
        matches = mol.GetSubstructMatches(ch_pat)
        if len(matches) < n_f:
            return None

    # 按位阻排序：计算每个候选 C 周围非 H 原子数
    def _steric_score(idx: int) -> int:
        atom = mol.GetAtomWithIdx(idx)
        n_neighbors = len([n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1])
        return n_neighbors

    matches_sorted = sorted(matches, key=lambda m: _steric_score(m[0]))

    # 逐个替换
    added = 0
    for match in matches_sorted:
        if added >= n_f:
            break
        c_idx = match[0]
        c_atom = mol.GetAtomWithIdx(c_idx)
        # 找到该碳上的氢原子
        h_idx = None
        for n in c_atom.GetNeighbors():
            if n.GetAtomicNum() == 1:
                h_idx = n.GetIdx()
                break
        if h_idx is None:
            continue
        h_atom = mol.GetAtomWithIdx(h_idx)
        h_atom.SetAtomicNum(9)  # H(1) → F(9)
        # 清除同位素信息
        h_atom.SetIsotope(0)
        added += 1

    if added == 0:
        return None

    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
