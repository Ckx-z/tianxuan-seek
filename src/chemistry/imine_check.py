"""亚胺键单体检测器 — 通过 SMARTS 模式检测醛基和伯胺基。

用于判定单体是否能参与席夫碱（Schiff base）亚胺键形成：
- 醛基单体：[CX3H1](=O)[#6] — 含 -CHO 的芳香/脂肪醛
- 胺基单体：[NH2]        — 伯胺（覆盖芳香胺和脂肪胺）
"""
from typing import Any, Dict

from rdkit import Chem


class ImineChecker:
    """SMARTS 亚胺键单体检测器。"""

    # 醛基：三价碳连双键氧和一个非氢重原子
    ALDEHYDE_SMARTS = "[CX3H1](=O)[#6]"
    # 伯胺：sp3 氮连两个氢（覆盖芳香胺中 N 连两个 H 的情形）
    AMINE_SMARTS = "[NH2]"
    # 亚胺键（验证用）：C=N 键连两个非氢碳
    IMINE_SMARTS = "[CX3](=[NX2])[#6]"

    def __init__(self):
        self._aldehyde_pat = Chem.MolFromSmarts(self.ALDEHYDE_SMARTS)
        self._amine_pat = Chem.MolFromSmarts(self.AMINE_SMARTS)
        self._imine_pat = Chem.MolFromSmarts(self.IMINE_SMARTS)

    def is_aldehyde(self, mol: Chem.Mol) -> bool:
        """检测是否含有醛基 (-CHO)。"""
        return mol.HasSubstructMatch(self._aldehyde_pat)

    def is_amine(self, mol: Chem.Mol) -> bool:
        """检测是否含有伯胺基 (-NH2)。"""
        return mol.HasSubstructMatch(self._amine_pat)

    def is_imine_capable(self, mol: Chem.Mol) -> bool:
        """检测是否能参与亚胺键形成（含醛基或伯胺基）。"""
        return self.is_aldehyde(mol) or self.is_amine(mol)

    def count_aldehyde_groups(self, mol: Chem.Mol) -> int:
        """统计醛基数量（互斥匹配）。"""
        return len(mol.GetSubstructMatches(self._aldehyde_pat, uniquify=True))

    def count_amine_groups(self, mol: Chem.Mol) -> int:
        """统计伯胺基数量（互斥匹配）。"""
        return len(mol.GetSubstructMatches(self._amine_pat, uniquify=True))

    def count_imine_bonds(self, mol: Chem.Mol) -> int:
        """统计已存在的亚胺键数量。"""
        return len(mol.GetSubstructMatches(self._imine_pat, uniquify=True))


def classify_monomer(mol: Chem.Mol) -> Dict[str, Any]:
    """对单体进行综合分类。

    Returns:
        {
            'is_aldehyde': bool,
            'is_amine': bool,
            'n_aldehyde': int,
            'n_amine': int,
            'monomer_type': 'aldehyde' | 'amine' | 'aldehyde-amine' | 'other',
            'is_imine_capable': bool,
        }
    """
    checker = ImineChecker()
    is_ald = checker.is_aldehyde(mol)
    is_am = checker.is_amine(mol)
    n_ald = checker.count_aldehyde_groups(mol) if is_ald else 0
    n_am = checker.count_amine_groups(mol) if is_am else 0

    if is_ald and is_am:
        mtype = "aldehyde-amine"
    elif is_ald:
        mtype = "aldehyde"
    elif is_am:
        mtype = "amine"
    else:
        mtype = "other"

    return {
        "is_aldehyde": is_ald,
        "is_amine": is_am,
        "n_aldehyde": n_ald,
        "n_amine": n_am,
        "monomer_type": mtype,
        "is_imine_capable": is_ald or is_am,
    }
