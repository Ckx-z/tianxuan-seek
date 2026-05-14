"""连接基团类型检测 + RDKit 机理描述符。

区分 COF 中苯链接(优越) vs 炔基链接(劣化)的化学信号，
为排序损失约束提供标签, 为模型提供廉价机理特征。
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# 炔基 SMARTS 模式
_ACETYLENE_SMARTS = Chem.MolFromSmarts("[C]#[C]")


def has_acetylene(mol: Chem.Mol) -> bool:
    """分子是否含 C≡C 三键。"""
    return mol.HasSubstructMatch(_ACETYLENE_SMARTS)


def count_acetylene(mol: Chem.Mol) -> int:
    """分子中 C≡C 三键的数量。"""
    return len(mol.GetSubstructMatches(_ACETYLENE_SMARTS))


def classify_linker_type(mol: Chem.Mol) -> str:
    """将单体核心连接基团分类: benzene / acetylene / mixed / other。"""
    has_ac = has_acetylene(mol)
    ri = mol.GetRingInfo()
    n_arom_rings = sum(
        1 for ring in ri.AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    )
    if has_ac and n_arom_rings > 0:
        return "mixed"
    elif has_ac:
        return "acetylene"
    elif n_arom_rings > 0:
        return "benzene"
    else:
        return "other"


def pair_linker_type(ald_mol: Chem.Mol, am_mol: Chem.Mol) -> str:
    """判定单体对的链接类型: benzene / acetylene / mixed。

    benzene  = 两个单体都不含 C≡C (纯芳香链接)
    acetylene = 至少一个单体含 C≡C
    mixed    = 仅一个单体含 C≡C (保留, 当前未细分)
    """
    ald_ac = has_acetylene(ald_mol)
    am_ac = has_acetylene(am_mol)
    if ald_ac and am_ac:
        return "acetylene"
    elif ald_ac or am_ac:
        return "mixed"
    else:
        return "benzene"


def compute_monomer_descriptors(mol: Chem.Mol) -> dict:
    """计算单体的 RDKit 机理描述符 (8 维)。"""
    n_atoms = mol.GetNumAtoms()
    n_heavy = mol.GetNumHeavyAtoms()

    # 芳香性
    n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    aromatic_frac = n_aromatic / max(n_atoms, 1)

    # 刚性: 环原子占比
    ri = mol.GetRingInfo()
    ring_atoms: set[int] = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    ring_frac = len(ring_atoms) / max(n_atoms, 1)

    # 芳香环数
    n_arom_rings = sum(
        1 for ring in ri.AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    )

    # 可旋转键
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # 炔基含量
    n_acet = count_acetylene(mol)

    return {
        "has_acetylene": float(n_acet > 0),
        "n_acetylene": float(n_acet),
        "aromatic_frac": aromatic_frac,
        "ring_frac": ring_frac,
        "n_rotatable": float(n_rot),
        "n_aromatic_rings": float(n_arom_rings),
        "n_heavy": float(n_heavy),
        "mw": Descriptors.MolWt(mol),
    }


def compute_pair_descriptors(ald_mol: Chem.Mol, am_mol: Chem.Mol) -> dict:
    """计算单体对的聚合机理描述符。"""
    ald_d = compute_monomer_descriptors(ald_mol)
    am_d = compute_monomer_descriptors(am_mol)
    return {
        **{f"ald_{k}": v for k, v in ald_d.items()},
        **{f"am_{k}": v for k, v in am_d.items()},
        "total_acetylene": ald_d["n_acetylene"] + am_d["n_acetylene"],
        "any_acetylene": float(
            ald_d["has_acetylene"] > 0 or am_d["has_acetylene"] > 0
        ),
        "linker_type": pair_linker_type(ald_mol, am_mol),
    }


def compute_pair_descriptor_vector(ald_mol: Chem.Mol, am_mol: Chem.Mol) -> np.ndarray:
    """返回单体对的数值描述符向量 (18 维), 用于拼入模型输入。

    维度: ald_8 + am_8 + total_acetylene + any_acetylene = 18
    """
    d = compute_pair_descriptors(ald_mol, am_mol)
    keys = [
        "ald_has_acetylene", "ald_n_acetylene", "ald_aromatic_frac",
        "ald_ring_frac", "ald_n_rotatable", "ald_n_aromatic_rings",
        "ald_n_heavy", "ald_mw",
        "am_has_acetylene", "am_n_acetylene", "am_aromatic_frac",
        "am_ring_frac", "am_n_rotatable", "am_n_aromatic_rings",
        "am_n_heavy", "am_mw",
        "total_acetylene", "any_acetylene",
    ]
    return np.array([d[k] for k in keys], dtype=np.float32)
