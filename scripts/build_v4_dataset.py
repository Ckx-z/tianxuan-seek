"""清洗 + 化学规则负样本构建 v4 训练集。

路线 B:
1. 丢弃全负文献（597篇，1607个假负样本）
2. 处理标签冲突（61个配对标签矛盾）
3. 用化学规则生成确定性负样本（~1500个）
4. 去重：化学规则负样本不与已有正/负样本重复
5. 输出 v4_train.csv

新增 5 种边界化学规则负样本:
- flex_rigid_mismatch: 柔性-刚性失配
- c1_c3_boundary: C1+C3 临界
- nonplanar: 平面性不足
- weak_nucleophile: 弱亲核胺
- excess_fluoro: 过量氟取代

Usage:
  python scripts/build_v4_dataset.py
  python scripts/build_v4_dataset.py --target-neg 1500 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.negative_sampler import (
    ReplacementPool,
    SyntheticPair,
    MonomerInfo,
    build_replacement_pool,
    generate_synthetic_pairs,
    compute_monomer_info,
)
from src.chemistry.conformer import compute_3d_descriptors, DESCRIPTOR_NAMES
from src.chemistry.dimer import compute_dimer_3d, DIMER_DESCRIPTOR_NAMES
from src.utils.logger import setup_logger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = setup_logger("build_v4")


def _canon(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else ""


# ── 5 种边界化学属性检测 ──

_ANILINE_PAT = Chem.MolFromSmarts("[NH2][c]")
_PYRIDINE_AMINE_PAT = Chem.MolFromSmarts("[NH2][n]")
_SULFONIC_PAT = Chem.MolFromSmarts("[S](=O)(=O)[OH]")
_CARBOXYL_PAT = Chem.MolFromSmarts("[CX3](=O)[OH]")
_PEG_PAT = Chem.MolFromSmarts("[OD2]-[CX4]-[CX4]-[OD2]")
_NITRO_PAT = Chem.MolFromSmarts("[N+](=O)[O-]")
_NITRILE_PAT = Chem.MolFromSmarts("[C]#[N]")
_SULFONE_PAT = Chem.MolFromSmarts("[S](=O)(=O)")
_ALD_PAT = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
_BIPHENYL_PAT = Chem.MolFromSmarts("c1ccccc1-c2ccccc2")


def _detect_rigid(info: MonomerInfo) -> bool:
    """刚性: 芳环≥3 且 可旋转键≤1。"""
    if info.n_rings < 3:
        return False
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False
    return rdMolDescriptors.CalcNumRotatableBonds(mol) <= 1


def _is_rigid_pair(ald_info: MonomerInfo, am_info: MonomerInfo) -> tuple[bool, str]:
    """配对级刚性检测: 总芳环在[4,8]外或两个都≥4时为刚性。

    规则:
      - total < 4: 太简单 (不是刚性，但训练时也应包含少量)
      - total > 8: 太刚性
      - 一个 ≥ 4 且另一个 ≤ 3: 大+小失衡 (不是刚性，好配对)
      - 两个都 ≥ 4: 刚性
      - 4 ≤ total ≤ 8 且都 ≤ 3: 正常 (不是刚性)
      - 任何单体 > 7: 过大不参与

    Returns:
        (is_rigid, reason)
    """
    ra = ald_info.n_rings
    rm = am_info.n_rings
    if ra > 7 or rm > 7:
        return False, "oversize"
    total = ra + rm
    if total < 4:
        return False, f"too_simple(t={total})"
    if total > 8:
        return True, f"too_rigid(ra={ra},rm={rm},t={total})"
    if (ra >= 4 and rm <= 3) or (rm >= 4 and ra <= 3):
        return False, f"big+small(ra={ra},rm={rm})"
    if ra >= 4 and rm >= 4:
        return True, f"rigid(ra={ra},rm={rm},t={total})"
    return False, f"normal(t={total})"


def _detect_flexible(info: MonomerInfo) -> bool:
    """柔性: 含连续≥3 个 sp3 碳的脂肪链。"""
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False
    sp3_c = {a.GetIdx() for a in mol.GetAtoms()
             if a.GetAtomicNum() == 6
             and a.GetHybridization() == Chem.HybridizationType.SP3}
    visited = set()
    for aidx in sp3_c:
        if aidx in visited:
            continue
        stack = [aidx]
        chain = set()
        while stack:
            a = stack.pop()
            if a in chain:
                continue
            chain.add(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() in sp3_c:
                    stack.append(nb.GetIdx())
            visited.add(a)
        if len(chain) >= 3:
            return True
    return False


def _detect_sp3_bridge(info: MonomerInfo) -> bool:
    """含 sp3 碳连接 2+ 芳环 → 破坏平面性。"""
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        aromatic_nb = sum(1 for nb in atom.GetNeighbors() if nb.GetIsAromatic())
        if aromatic_nb >= 2:
            return True
    return False


def _detect_weak_amine(info: MonomerInfo) -> bool:
    """弱亲核胺: NH2 直接连芳环 (苯胺/吡啶胺)。"""
    if info.monomer_type != "amine":
        return False
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False
    if mol.HasSubstructMatch(_ANILINE_PAT):
        return True
    if mol.HasSubstructMatch(_PYRIDINE_AMINE_PAT):
        return True
    return False


def _detect_perfluoro(info: MonomerInfo) -> bool:
    """过量氟: 单个芳环上 ≥4 个 F 取代。"""
    if not info.has_fluorine:
        return False
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) < 6:
            continue
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        f_on_ring = 0
        for aidx in ring:
            atom = mol.GetAtomWithIdx(aidx)
            for nb in atom.GetNeighbors():
                if nb.GetAtomicNum() == 9:
                    f_on_ring += 1
        if f_on_ring >= 4:
            return True
    return False


def _detect_steric_block(info: MonomerInfo) -> bool:
    """位阻阻断: 反应位点邻位有大位阻基团 (非 H/卤素, 原子量 >30)。

    检测醛基/胺基连接碳的邻位 (芳环上距离 2 键) 是否有 bulky 取代基。
    """
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False

    # 找到反应基团连接在芳环上的碳
    reactive_carbons = []
    if info.monomer_type == "aldehyde":
        matches = mol.GetSubstructMatches(_ALD_PAT)
        reactive_carbons = [m[2] for m in matches]  # m[2]=连接碳
    elif info.monomer_type == "amine":
        matches = mol.GetSubstructMatches(_ANILINE_PAT)
        reactive_carbons = [m[1] for m in matches]  # m[1]=连接碳

    for rc in reactive_carbons:
        atom = mol.GetAtomWithIdx(rc)
        if not atom.GetIsAromatic():
            continue
        for ring in mol.GetRingInfo().AtomRings():
            if rc not in ring:
                continue
            for other in ring:
                if other == rc:
                    continue
                path = Chem.GetShortestPath(mol, rc, other)
                if len(path) == 3:  # ortho: rc-a-b
                    ortho_atom = mol.GetAtomWithIdx(other)
                    for nb in ortho_atom.GetNeighbors():
                        an = nb.GetAtomicNum()
                        if nb.GetIdx() not in ring and an not in (1, 9, 17, 35, 53):
                            if an > 1:
                                return True
    return False


def _detect_charge_interference(info: MonomerInfo) -> bool:
    """电荷干扰: 含磺酸(-SO3H)或羧酸(-COOH)基团。

    在界面聚合条件下 (pH~4-5), 这些基团会质子化胺, 阻止亲核进攻。
    """
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False
    if mol.HasSubstructMatch(_SULFONIC_PAT):
        return True
    if mol.HasSubstructMatch(_CARBOXYL_PAT):
        return True
    return False


def _detect_peg_chain(info: MonomerInfo) -> bool:
    """PEG 链检测: -O-C-C-O- 重复单元 (柔性冠醚类)。"""
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False
    return len(mol.GetSubstructMatches(_PEG_PAT)) >= 2


def _detect_twisted_biaryl(info: MonomerInfo) -> bool:
    """扭曲联芳: 联苯类且邻位有 ≥2 个非 H 取代基，阻止旋转共面。"""
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False
    matches = mol.GetSubstructMatches(_BIPHENYL_PAT)
    if not matches:
        return False
    for match in matches:
        ring1 = set(match[:6])
        ring2 = set(match[6:])
        # 找联芳键
        for a1 in ring1:
            for a2 in ring2:
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                # 检查邻位取代
                ortho_sub = 0
                for ring, center in [(ring1, a1), (ring2, a2)]:
                    for other in ring:
                        if other == center:
                            continue
                        path = Chem.GetShortestPath(mol, center, other)
                        if len(path) == 3:
                            oa = mol.GetAtomWithIdx(other)
                            for nb in oa.GetNeighbors():
                                if nb.GetIdx() not in ring and nb.GetAtomicNum() > 1:
                                    ortho_sub += 1
                if ortho_sub >= 2:
                    return True
    return False


def _generate_frequency_decoys(
    positive_pairs: list[tuple[str, str, str, str]],
    pool: ReplacementPool,
    existing_pairs: set[tuple[str, str]],
    target_count: int = 120,
    seed: int = 42,
) -> list[dict]:
    """频率诱饵: 高频单体 × 错配伙伴 → 反例。

    防止模型学到"高频单体 → 成膜"的捷径。
    策略: Top-N 高频醛 × 单胺基/错配胺 + Top-N 高频胺 × 单醛基/错配醛。
    """
    import random
    random.seed(seed)

    # 统计单体频率
    ald_freq: dict[str, int] = {}
    am_freq: dict[str, int] = {}
    for ald, am, _, _ in positive_pairs:
        ald_freq[ald] = ald_freq.get(ald, 0) + 1
        am_freq[am] = am_freq.get(am, 0) + 1

    top_n = 5
    top_alds = sorted(ald_freq.items(), key=lambda x: -x[1])[:top_n]
    top_ams = sorted(am_freq.items(), key=lambda x: -x[1])[:top_n]
    logger.info(f"频率诱饵 — Top {top_n} 高频醛: {[(s[:25], c) for s, c in top_alds]}")
    logger.info(f"频率诱饵 — Top {top_n} 高频胺: {[(s[:25], c) for s, c in top_ams]}")

    # 收集无效配对伙伴
    invalid_amines: list[tuple[str, MonomerInfo]] = []  # (smi, info)
    invalid_aldehydes: list[tuple[str, MonomerInfo]] = []

    for m in pool.all_monomers:
        info = pool.lookup.get(m.canonical_smiles) or compute_monomer_info(
            m.canonical_smiles, monomer_type=m.monomer_type, source="pool")
        if info is None:
            continue
        if info.monomer_type == "amine":
            # 无效胺: n_amine < 2 或 n_amine >= 4 或 弱亲核
            n_am = info.n_am or 0
            is_weak = _detect_weak_amine_extended(info)
            if n_am < 2 or n_am >= 4 or is_weak:
                if info.canonical_smiles not in {s for s, _ in top_ams}:
                    invalid_amines.append((info.canonical_smiles, info))
        elif info.monomer_type == "aldehyde":
            n_ald_eff = info.n_ald or 0
            if n_ald_eff < 2 or n_ald_eff >= 4:
                if info.canonical_smiles not in {s for s, _ in top_alds}:
                    invalid_aldehydes.append((info.canonical_smiles, info))

    logger.info(f"  无效胺池: {len(invalid_amines)}, 无效醛池: {len(invalid_aldehydes)}")

    seen = set(existing_pairs)
    decoys = []
    n_per_top = target_count // (top_n * 2)  # 每对高频单体配~12个错配伙伴

    for ald_smi, _ in top_alds:
        ald_info = pool.lookup.get(ald_smi)
        if ald_info is None:
            continue
        random.shuffle(invalid_amines)
        count = 0
        for am_smi, am_info in invalid_amines[:n_per_top * 2]:
            if count >= n_per_top:
                break
            if (ald_smi, am_smi) in seen:
                continue
            seen.add((ald_smi, am_smi))
            decoys.append({
                "paper_id": "freq_decoy_ald",
                "group_id": "0",
                "source_db": "freq_decoy",
                "aldehyde_smiles": ald_smi,
                "amine_smiles": am_smi,
                "aldehyde_smiles_source": "pool",
                "amine_smiles_source": "pool",
                "aldehyde_name": "",
                "amine_name": "",
                "stoichiometry": "",
                "solvent_raw": "",
                "temperature_raw": "",
                "catalyst_raw": "",
                "synthesis_route_raw": "",
                "interface_type_raw": "",
                "solvent_label": "",
                "temperature_bin": "",
                "catalyst_label": "",
                "synthesis_route_label": "",
                "interface_type_label": "",
                "is_film": "0",
                "film_quality": "unknown",
                "quality_weight": "1.0",
                "has_fluorine_monomer": str(int(
                    ald_info.has_fluorine or am_info.has_fluorine)),
                "has_n_heterocycle": "0",
                "confidence": "high",
            })
            count += 1

    for am_smi, _ in top_ams:
        am_info = pool.lookup.get(am_smi)
        if am_info is None:
            continue
        random.shuffle(invalid_aldehydes)
        count = 0
        for ald_smi, ald_info in invalid_aldehydes[:n_per_top * 2]:
            if count >= n_per_top:
                break
            if (ald_smi, am_smi) in seen:
                continue
            seen.add((ald_smi, am_smi))
            decoys.append({
                "paper_id": "freq_decoy_am",
                "group_id": "0",
                "source_db": "freq_decoy",
                "aldehyde_smiles": ald_smi,
                "amine_smiles": am_smi,
                "aldehyde_smiles_source": "pool",
                "amine_smiles_source": "pool",
                "aldehyde_name": "",
                "amine_name": "",
                "stoichiometry": "",
                "solvent_raw": "",
                "temperature_raw": "",
                "catalyst_raw": "",
                "synthesis_route_raw": "",
                "interface_type_raw": "",
                "solvent_label": "",
                "temperature_bin": "",
                "catalyst_label": "",
                "synthesis_route_label": "",
                "interface_type_label": "",
                "is_film": "0",
                "film_quality": "unknown",
                "quality_weight": "1.0",
                "has_fluorine_monomer": str(int(
                    ald_info.has_fluorine or am_info.has_fluorine)),
                "has_n_heterocycle": "0",
                "confidence": "high",
            })
            count += 1

    logger.info(f"频率诱饵生成: {len(decoys)}")
    return decoys


def _generate_rigid_pair_negatives(
    positive_pairs: list[tuple[str, str, str, str]],
    pool,
    existing_pairs: set[tuple[str, str]],
    target_count: int = 200,
    seed: int = 42,
) -> list[dict]:
    """合成刚性×刚性配对负样本 — 让 GNN 学到"双高芳环 → 不成膜"。

    规则:
      - ra+rm ≥ 4 (总芳环≥4)
      - ra ≤ 7, rm ≤ 7 (单体不超过 7 环)
      - 非"大+小"失衡例外 (即不出现 ra>4∧rm≤3 或 rm>4∧ra≤3)

    策略:
      1. 复用正样本对中两侧均刚性的对 (高置信) → 直接标 0
      2. 从池中合成 刚性醛 × 刚性胺 配对 → 标 0

    Args:
        positive_pairs: list of (ald_smi, am_smi, ald_type, am_type)
        pool: MonomerPool
        existing_pairs: 已存在的 (ald, am) 集合, 用于去重
        target_count: 目标生成数量
    """
    import random
    random.seed(seed)

    from src.chemistry.negative_sampler import compute_monomer_info

    decoys: list[dict] = []

    # ── 来源 A: 复用正样本中双刚性的对 ──
    reused = 0
    for ald_smi, am_smi, ald_type, am_type in positive_pairs:
        ald_info = pool.lookup.get(ald_smi)
        am_info = pool.lookup.get(am_smi)
        if ald_info is None or am_info is None:
            continue
        is_rigid, reason = _is_rigid_pair(ald_info, am_info)
        if is_rigid and (ald_smi, am_smi) not in existing_pairs:
            decoys.append({
                "paper_id": "rigid_pair_rule",
                "group_id": "0",
                "source_db": "rigid_pair_rule",
                "aldehyde_smiles": ald_smi,
                "amine_smiles": am_smi,
                "aldehyde_smiles_source": "pool",
                "amine_smiles_source": "pool",
                "aldehyde_name": "",
                "amine_name": "",
                "stoichiometry": "",
                "solvent_raw": "",
                "temperature_raw": "",
                "catalyst_raw": "",
                "synthesis_route_raw": "",
                "interface_type_raw": "",
                "solvent_label": "",
                "temperature_bin": "",
                "catalyst_label": "",
                "synthesis_route_label": "",
                "interface_type_label": "",
                "is_film": "0",
                "film_quality": "unknown",
                "quality_weight": "1.0",
                "has_fluorine_monomer": str(int(ald_info.has_fluorine or am_info.has_fluorine)),
                "has_n_heterocycle": "0",
                "confidence": "high",
            })
            reused += 1
    logger.info(f"刚性诱饵 复用正样本: {reused}")

    # ── 来源 B: 池中合成 刚性醛×刚性胺 配对 (芳环≥4) ──
    rigid_alds = [m for m in pool.all_monomers
                  if m.monomer_type == "aldehyde" and 4 <= m.n_rings <= 7]
    rigid_ams = [m for m in pool.all_monomers
                 if m.monomer_type == "amine" and 4 <= m.n_rings <= 7]
    logger.info(f"刚性池: 醛={len(rigid_alds)}, 胺={len(rigid_ams)}")

    random.shuffle(rigid_alds)
    random.shuffle(rigid_ams)
    seen = set(existing_pairs)
    for d in decoys:
        seen.add((d["aldehyde_smiles"], d["amine_smiles"]))
    synth = 0
    for ald_m in rigid_alds:
        if synth >= target_count - reused:
            break
        for am_m in rigid_ams:
            is_rigid, reason = _is_rigid_pair(ald_m, am_m)
            if not is_rigid:
                continue
            key = (ald_m.canonical_smiles, am_m.canonical_smiles)
            if key in seen:
                continue
            seen.add(key)
            decoys.append({
                "paper_id": "rigid_pair_rule",
                "group_id": "0",
                "source_db": "rigid_pair_rule",
                "aldehyde_smiles": ald_m.canonical_smiles,
                "amine_smiles": am_m.canonical_smiles,
                "aldehyde_smiles_source": "pool",
                "amine_smiles_source": "pool",
                "aldehyde_name": "",
                "amine_name": "",
                "stoichiometry": "",
                "solvent_raw": "",
                "temperature_raw": "",
                "catalyst_raw": "",
                "synthesis_route_raw": "",
                "interface_type_raw": "",
                "solvent_label": "",
                "temperature_bin": "",
                "catalyst_label": "",
                "synthesis_route_label": "",
                "interface_type_label": "",
                "is_film": "0",
                "film_quality": "unknown",
                "quality_weight": "1.0",
                "has_fluorine_monomer": str(int(ald_m.has_fluorine or am_m.has_fluorine)),
                "has_n_heterocycle": "0",
                "confidence": "high",
            })
            synth += 1
            if synth >= target_count - reused:
                break
    logger.info(f"刚性诱饵 池合成: {synth}, 总计: {len(decoys)}")
    return decoys


def _detect_weak_amine_extended(info: MonomerInfo) -> bool:
    """弱亲核胺 (扩展): 苯胺/吡啶胺; 或 -NO2/-CN/-SO2- 邻接 NH2。

    吸电子基团降低胺的亲核性，不利于亚胺键形成。
    """
    if info.monomer_type != "amine":
        return False
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False
    # 原始检测: 苯胺/吡啶胺
    if mol.HasSubstructMatch(_ANILINE_PAT) or mol.HasSubstructMatch(_PYRIDINE_AMINE_PAT):
        return True
    # 扩展: NH2 邻接吸电子基团
    am_matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[NH2]"))
    ewg = []
    if mol.HasSubstructMatch(_NITRO_PAT):
        ewg.extend([m[0] for m in mol.GetSubstructMatches(_NITRO_PAT)])
    if mol.HasSubstructMatch(_NITRILE_PAT):
        ewg.extend([m[0] for m in mol.GetSubstructMatches(_NITRILE_PAT)])
    if mol.HasSubstructMatch(_SULFONE_PAT):
        ewg.extend([m[0] for m in mol.GetSubstructMatches(_SULFONE_PAT)])
    for n_idx in [m[0] for m in am_matches]:
        for ew in ewg:
            path = Chem.GetShortestPath(mol, n_idx, ew)
            if path is not None and len(path) <= 5:  # 5 键内
                return True
    return False


def _get_monomer_violation_profile(info: MonomerInfo) -> dict[str, bool]:
    """计算单体的完整违规画像 — 所有适用策略的违规状态。

    返回 {strategy_name: is_violated}，用于分层负样本生成。
    """
    profile = {}

    # 对称性 (C2/C3 单体)
    if info.topology in ("C2", "C3"):
        profile["symmetry"] = not info.is_symmetric
        if info.topology == "C2":
            profile["nonpara"] = not info.is_para

    # 多环 (>8)
    profile["multiring"] = info.n_rings > 8

    # 过取代 (非卤素多余取代 >0)
    profile["oversub"] = info.max_nonhalo_extra > 0

    # 新增策略 (monomer_type 感知)
    profile["steric_block"] = _detect_steric_block(info)
    profile["charge_interference"] = _detect_charge_interference(info)

    # 非平面 (sp3 桥接 + 扭曲联芳)
    profile["nonplanar"] = _detect_sp3_bridge(info) or _detect_twisted_biaryl(info)

    # 弱亲核 (仅胺, 扩展检测)
    if info.monomer_type == "amine":
        profile["weak_nucleophile"] = _detect_weak_amine_extended(info)

    # 刚柔失配 (刚性单体 + PEG 柔性)
    profile["flex_rigid_extreme"] = (
        _detect_rigid(info) or _detect_flexible(info) or _detect_peg_chain(info)
    )

    # 过量氟
    profile["excess_fluoro"] = _detect_perfluoro(info)

    # 清理 None/False
    return {k: v for k, v in profile.items() if v}




def load_and_clean(csv_path: str) -> tuple[list[dict], list[dict], list[dict]]:
    """加载 v3_train.csv 并清洗。

    Returns:
        (kept_rows, discarded_rows, conflict_rows)
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 按文献分组
    paper_data: dict[str, list[dict]] = {}
    for r in rows:
        paper_data.setdefault(r["paper_id"], []).append(r)

    # 分类文献
    all_pos_papers = {}
    all_neg_papers = {}
    mixed_papers = {}
    for pid, prows in paper_data.items():
        has_pos = any(r["is_film"] == "1" for r in prows)
        has_neg = any(r["is_film"] == "0" for r in prows)
        if has_pos and has_neg:
            mixed_papers[pid] = prows
        elif has_pos:
            all_pos_papers[pid] = prows
        else:
            all_neg_papers[pid] = prows

    logger.info(
        f"文献分类: 全正={len(all_pos_papers)}({sum(len(v) for v in all_pos_papers.values())}样本), "
        f"混合={len(mixed_papers)}({sum(len(v) for v in mixed_papers.values())}样本), "
        f"全负={len(all_neg_papers)}({sum(len(v) for v in all_neg_papers.values())}样本)"
    )

    # 保留全正 + 混合，丢弃全负
    kept = []
    discarded = []
    for pid in all_pos_papers:
        kept.extend(all_pos_papers[pid])
    for pid in mixed_papers:
        kept.extend(mixed_papers[pid])
    for pid in all_neg_papers:
        discarded.extend(all_neg_papers[pid])

    # 处理标签冲突：同一配对不同标签 → 保留正样本（保守策略，避免假负）
    pair_labels: dict[tuple, set] = {}
    for r in kept:
        key = (r["aldehyde_smiles"], r["amine_smiles"])
        pair_labels.setdefault(key, set()).add(r["is_film"])

    conflict_pairs = {k: v for k, v in pair_labels.items() if len(v) > 1}
    conflict_rows = []
    if conflict_pairs:
        logger.info(f"标签冲突配对: {len(conflict_pairs)}")
        # 对冲突配对，保留正样本标签行，丢弃负样本标签行
        cleaned = []
        for r in kept:
            key = (r["aldehyde_smiles"], r["amine_smiles"])
            if key in conflict_pairs and r["is_film"] == "0":
                conflict_rows.append(r)
            else:
                cleaned.append(r)
        kept = cleaned
        logger.info(f"冲突处理: 丢弃 {len(conflict_rows)} 个冲突负样本行")

    pos_count = sum(1 for r in kept if r["is_film"] == "1")
    neg_count = sum(1 for r in kept if r["is_film"] == "0")
    logger.info(f"清洗后: {len(kept)} 样本 (正={pos_count}, 负={neg_count}, "
                f"正率={pos_count/len(kept)*100:.1f}%)")

    return kept, discarded, conflict_rows


def generate_chem_negatives(
    kept_rows: list[dict],
    pool_path: str,
    target_count: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """分 5 层生成化学规则负样本 — L1(880) 单违规 → L5(70) 五违规。

    L1 策略权重: symmetry=0.30, nonpara=0.20, steric_block=0.15,
      charge_interference=0.10, nonplanar=0.08, weak_nucleophile=0.07,
      flex_rigid_extreme=0.05, excess_fluoro=0.05。
    醛/胺替换平衡, 高层通过叠加违规数实现。
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    # 收集已有配对
    existing_pairs: set[tuple[str, str]] = set()
    positive_pairs_raw: list[tuple[str, str, str, str]] = []
    for r in kept_rows:
        ald = _canon(r["aldehyde_smiles"])
        am = _canon(r["amine_smiles"])
        if not ald or not am:
            continue
        existing_pairs.add((ald, am))
        if r["is_film"] == "1":
            positive_pairs_raw.append((ald, am, "aldehyde", "amine"))

    logger.info(f"已有配对: {len(existing_pairs)}, 正样本: {len(positive_pairs_raw)}")

    # 构建替换池
    extra_smiles = list({smi for pair in existing_pairs for smi in pair})
    pool = build_replacement_pool(pool_path, extra_smiles=extra_smiles)
    logger.info(f"替换池: {len(pool.all_monomers)} 个单体")

    # 分类所有单体 by violation profile
    monomer_vc: dict[str, int] = {}
    monomer_profile: dict[str, set[str]] = {}
    by_vc: dict[int, list[MonomerInfo]] = {}
    by_strat_all: dict[str, list[MonomerInfo]] = {}  # 所有违反该策略的单体

    for m in pool.all_monomers:
        profile = _get_monomer_violation_profile(m)
        vc = len(profile)
        monomer_vc[m.canonical_smiles] = vc
        monomer_profile[m.canonical_smiles] = set(profile.keys())
        by_vc.setdefault(vc, []).append(m)
        for strat in profile:
            by_strat_all.setdefault(strat, []).append(m)

    # 按 VC 升序排列每个策略列表 (L1 优先取低 VC 单体)
    for strat in by_strat_all:
        by_strat_all[strat].sort(key=lambda m: monomer_vc.get(m.canonical_smiles, 99))

    for vc in sorted(by_vc):
        logger.info(f"  VC={vc}: {len(by_vc[vc])} 个单体")
    for strat in sorted(by_strat_all):
        ald_c = sum(1 for m in by_strat_all[strat] if m.monomer_type == "aldehyde")
        am_c = sum(1 for m in by_strat_all[strat] if m.monomer_type == "amine")
        logger.info(f"  {strat}: {len(by_strat_all[strat])} (醛={ald_c}, 胺={am_c})")

    # Layer targets
    layer_targets = {1: 880, 2: 500, 3: 350, 4: 200, 5: 70}
    l1_weights = {
        "symmetry": 0.25, "nonpara": 0.15, "steric_block": 0.12,
        "charge_interference": 0.08, "nonplanar": 0.08,
        "weak_nucleophile": 0.07, "flex_rigid_extreme": 0.05, "excess_fluoro": 0.05,
        "oversub": 0.10, "multiring": 0.05,
    }
    # 权重重新分配: 无单体策略的权重按比例分配给有单体的策略
    active_weights = {s: w for s, w in l1_weights.items() if by_strat_all.get(s)}
    inactive_weights = {s: w for s, w in l1_weights.items() if not by_strat_all.get(s)}
    if inactive_weights:
        total_active = sum(active_weights.values())
        total_inactive = sum(inactive_weights.values())
        logger.info(f"  无单体策略: {list(inactive_weights.keys())}, 权重重新分配")
        l1_weights = {s: w + w / total_active * total_inactive
                      for s, w in active_weights.items()}

    def _pair_exists(ald_smi: str, am_smi: str, seen_set: set) -> bool:
        return (ald_smi, am_smi) in seen_set

    seen = set(existing_pairs)
    all_pairs: dict[int, list[SyntheticPair]] = {i: [] for i in range(1, 6)}
    max_per_strat_l1 = 600  # 每种策略最多生成 600 候选

    # ── Layer 1: 单策略违规, 醛/胺平衡 ──
    logger.info("=== 生成 L1: 单违规 (880) ===")
    for strat, weight in l1_weights.items():
        monomers = by_strat_all.get(strat, [])
        if not monomers:
            logger.warning(f"  策略 {strat}: 无可用单体!")
            continue
        alds = [m for m in monomers if m.monomer_type == "aldehyde"]
        ams = [m for m in monomers if m.monomer_type == "amine"]
        target_n = int(layer_targets[1] * weight)
        # 平衡醛/胺替换: 各一半
        n_ald = target_n // 2
        n_am = target_n - n_ald

        # 替换醛侧: 正样本胺 + 违规醛
        count_ald = 0
        random.shuffle(alds)
        random.shuffle(positive_pairs_raw)
        for rep_ald in alds:
            if count_ald >= min(n_ald, max_per_strat_l1):
                break
            for ald_smi, am_smi, _, am_type in positive_pairs_raw:
                key = (rep_ald.canonical_smiles, am_smi)
                if key in seen:
                    continue
                seen.add(key)
                am_info = pool.lookup.get(am_smi) or compute_monomer_info(
                    am_smi, monomer_type=am_type, source="train")
                if am_info is None:
                    continue
                all_pairs[1].append(SyntheticPair(
                    ald_smiles=rep_ald.canonical_smiles, am_smiles=am_smi,
                    ald_info=rep_ald, am_info=am_info,
                    strategy=strat, replaced="aldehyde",
                ))
                count_ald += 1
                if count_ald >= min(n_ald, max_per_strat_l1):
                    break

        # 替换胺侧: 正样本醛 + 违规胺
        count_am = 0
        random.shuffle(ams)
        random.shuffle(positive_pairs_raw)
        for rep_am in ams:
            if count_am >= min(n_am, max_per_strat_l1):
                break
            for ald_smi, am_smi, ald_type, _ in positive_pairs_raw:
                key = (ald_smi, rep_am.canonical_smiles)
                if key in seen:
                    continue
                seen.add(key)
                ald_info = pool.lookup.get(ald_smi) or compute_monomer_info(
                    ald_smi, monomer_type=ald_type, source="train")
                if ald_info is None:
                    continue
                all_pairs[1].append(SyntheticPair(
                    ald_smiles=ald_smi, am_smiles=rep_am.canonical_smiles,
                    ald_info=ald_info, am_info=rep_am,
                    strategy=strat, replaced="amine",
                ))
                count_am += 1
                if count_am >= min(n_am, max_per_strat_l1):
                    break

        logger.info(f"  {strat}: 醛={count_ald}, 胺={count_am} (目标醛={n_ald}, 胺={n_am})")

    # ── Layers 2–5: 多违规单体替换 ──
    for layer_n in range(2, 6):
        logger.info(f"=== 生成 L{layer_n}: {layer_n}违规 ({layer_targets[layer_n]}) ===")
        target_n = layer_targets[layer_n]
        monomers_n = by_vc.get(layer_n, [])
        if not monomers_n:
            logger.warning(f"  无 VC={layer_n} 单体, 尝试 VC≥{layer_n}")
            for vc in range(layer_n, 12):
                monomers_n.extend(by_vc.get(vc, []))
        alds_n = [m for m in monomers_n if m.monomer_type == "aldehyde"]
        ams_n = [m for m in monomers_n if m.monomer_type == "amine"]

        count_ald, count_am = 0, 0
        n_each = target_n // 2
        max_candidates = target_n * 3

        random.shuffle(alds_n)
        random.shuffle(positive_pairs_raw)
        for rep_ald in alds_n:
            if count_ald >= n_each or count_ald >= max_candidates:
                break
            for ald_smi, am_smi, _, am_type in positive_pairs_raw:
                key = (rep_ald.canonical_smiles, am_smi)
                if key in seen:
                    continue
                seen.add(key)
                am_info = pool.lookup.get(am_smi) or compute_monomer_info(
                    am_smi, monomer_type=am_type, source="train")
                if am_info is None:
                    continue
                all_pairs[layer_n].append(SyntheticPair(
                    ald_smiles=rep_ald.canonical_smiles, am_smiles=am_smi,
                    ald_info=rep_ald, am_info=am_info,
                    strategy=f"L{layer_n}_multi", replaced="aldehyde",
                ))
                count_ald += 1

        random.shuffle(ams_n)
        random.shuffle(positive_pairs_raw)
        for rep_am in ams_n:
            if count_am >= n_each or count_am >= max_candidates:
                break
            for ald_smi, am_smi, ald_type, _ in positive_pairs_raw:
                key = (ald_smi, rep_am.canonical_smiles)
                if key in seen:
                    continue
                seen.add(key)
                ald_info = pool.lookup.get(ald_smi) or compute_monomer_info(
                    ald_smi, monomer_type=ald_type, source="train")
                if ald_info is None:
                    continue
                all_pairs[layer_n].append(SyntheticPair(
                    ald_smiles=ald_smi, am_smiles=rep_am.canonical_smiles,
                    ald_info=ald_info, am_info=rep_am,
                    strategy=f"L{layer_n}_multi", replaced="amine",
                ))
                count_am += 1
        logger.info(f"  L{layer_n}: 醛={count_ald}, 胺={count_am}")

    # ── 汇总: 按层采样 ──
    final_pairs: list[SyntheticPair] = []
    layer_strat_counts: dict[int, Counter] = {}
    for layer_n in range(1, 6):
        candidates = all_pairs[layer_n]
        target = layer_targets[layer_n]
        random.shuffle(candidates)
        sampled = candidates[:target]
        final_pairs.extend(sampled)
        layer_strat_counts[layer_n] = Counter(sp.strategy for sp in sampled)
        logger.info(f"  L{layer_n}: 生成={len(candidates)}, 采样={len(sampled)}")
        logger.info(f"    策略分布: {dict(layer_strat_counts[layer_n])}")

    logger.info(f"总化学负样本: {len(final_pairs)}")

    # 转换为 CSV 行
    chem_neg_rows = []
    for sp in final_pairs:
        chem_neg_rows.append({
            "paper_id": f"chem_rule_{sp.strategy}",
            "group_id": "0",
            "source_db": f"chem_rule_{sp.strategy}",
            "aldehyde_smiles": sp.ald_smiles,
            "amine_smiles": sp.am_smiles,
            "aldehyde_smiles_source": "pool",
            "amine_smiles_source": "pool",
            "aldehyde_name": "",
            "amine_name": "",
            "stoichiometry": "",
            "solvent_raw": "",
            "temperature_raw": "",
            "catalyst_raw": "",
            "synthesis_route_raw": "",
            "interface_type_raw": "",
            "solvent_label": "",
            "temperature_bin": "",
            "catalyst_label": "",
            "synthesis_route_label": "",
            "interface_type_label": "",
            "is_film": "0",
            "film_quality": "unknown",
            "quality_weight": "1.0",
            "has_fluorine_monomer": str(int(sp.ald_info.has_fluorine or sp.am_info.has_fluorine)),
            "has_n_heterocycle": "0",
            "confidence": "high",
        })

    return chem_neg_rows


def main():
    parser = argparse.ArgumentParser(description="构建 v4 训练集")
    parser.add_argument("--input", type=str, default="data/processed/v3_train.csv")
    parser.add_argument("--output", type=str, default="data/processed/v4_train.csv")
    parser.add_argument("--pool", type=str, default="data/processed/merged_monomer_pool.csv")
    parser.add_argument("--target-neg", type=int, default=2000,
                        help="化学规则负样本目标数量 (分层: L1=880 L2=500 L3=350 L4=200 L5=70)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-3d", action="store_true",
                        help="跳过 3D 描述符计算 (节省时间, 训练用 --no-3d 时使用)")
    args = parser.parse_args()

    # Step 1: 清洗
    logger.info("=== Step 1: 清洗训练数据 ===")
    kept, discarded, conflicts = load_and_clean(args.input)

    # Step 2: 生成化学规则负样本
    logger.info("=== Step 2: 生成化学规则负样本 ===")
    chem_neg = generate_chem_negatives(kept, args.pool, args.target_neg, args.seed)

    # Step 2.5: 频率诱饵负样本
    logger.info("=== Step 2.5: 频率诱饵负样本 ===")
    # 重建 positive_pairs 列表供频率诱饵使用
    from src.chemistry.negative_sampler import build_replacement_pool
    extra_smiles_set = set()
    for r in kept:
        extra_smiles_set.add(_canon(r["aldehyde_smiles"]))
        extra_smiles_set.add(_canon(r["amine_smiles"]))
    decoy_pool = build_replacement_pool(args.pool, extra_smiles=list(extra_smiles_set))
    pos_pairs = [(_canon(r["aldehyde_smiles"]), _canon(r["amine_smiles"]),
                  "aldehyde", "amine")
                 for r in kept if r["is_film"] == "1"]
    existing_all = set()
    for r in kept:
        existing_all.add((_canon(r["aldehyde_smiles"]), _canon(r["amine_smiles"])))
    for cn in chem_neg:
        existing_all.add((_canon(cn["aldehyde_smiles"]), _canon(cn["amine_smiles"])))
    freq_decoys = _generate_frequency_decoys(
        pos_pairs, decoy_pool, existing_all, target_count=120, seed=args.seed)

    # Step 2.7: 刚性配对合成负样本 (让 GNN 学到"双高芳环→不成膜")
    logger.info("=== Step 2.7: 刚性配对合成负样本 ===")
    rigid_decoys = _generate_rigid_pair_negatives(
        pos_pairs, decoy_pool, existing_all, target_count=200, seed=args.seed)

    # Step 3: 合并
    logger.info("=== Step 3: 合并 ===")
    all_rows = kept + chem_neg + freq_decoys + rigid_decoys

    pos_count = sum(1 for r in all_rows if r["is_film"] == "1")
    neg_count = sum(1 for r in all_rows if r["is_film"] == "0")
    lit_pos = sum(1 for r in kept if r["is_film"] == "1")
    lit_neg = sum(1 for r in kept if r["is_film"] == "0")
    chem_neg_count = len(chem_neg) + len(freq_decoys) + len(rigid_decoys)

    logger.info(
        f"最终训练集: {len(all_rows)} 样本\n"
        f"  文献正样本: {lit_pos}\n"
        f"  文献负样本: {lit_neg}\n"
        f"  化学规则负样本: {chem_neg_count} (含频率诱饵 {len(freq_decoys)} + 刚性诱饵 {len(rigid_decoys)})\n"
        f"  总正: {pos_count} ({pos_count/len(all_rows)*100:.1f}%)\n"
        f"  总负: {neg_count} ({neg_count/len(all_rows)*100:.1f}%)\n"
        f"  正:负 = 1:{neg_count/pos_count:.1f}"
    )

    # Step 4: 3D 描述符 — 单体 + 二聚体
    if args.no_3d:
        logger.info("=== Step 4: 跳过 3D 描述符 (--no-3d) ===")
        for r in all_rows:
            for name in DESCRIPTOR_NAMES:
                r[f"ald_3d_{name}"] = "0.0"
                r[f"amine_3d_{name}"] = "0.0"
            for name in DIMER_DESCRIPTOR_NAMES:
                r[name] = "0.0"
    else:
        logger.info("=== Step 4: 计算 3D 描述符 (单体 + 二聚体) ===")

        # 单体 3D 缓存
        smiles_cache: dict[str, list[float] | None] = {}
        for r in all_rows:
            for key in ("aldehyde_smiles", "amine_smiles"):
                smi = r[key]
                if smi not in smiles_cache:
                    smiles_cache[smi] = compute_3d_descriptors(smi)
        n_ok = sum(1 for v in smiles_cache.values() if v is not None)
        logger.info(f"单体 3D: 成功={n_ok}, 失败={len(smiles_cache)-n_ok}, 唯一={len(smiles_cache)}")

        # 二聚体 3D 缓存
        dimer_cache: dict[tuple[str, str], list[float] | None] = {}
        for r in all_rows:
            key = (r["aldehyde_smiles"], r["amine_smiles"])
            if key not in dimer_cache:
                dimer_cache[key] = compute_dimer_3d(r["aldehyde_smiles"], r["amine_smiles"])
        n_dimer_ok = sum(1 for v in dimer_cache.values() if v is not None)
        logger.info(f"二聚体 3D: 成功={n_dimer_ok}, 失败={len(dimer_cache)-n_dimer_ok}")

        # 写入描述符到每行
        for r in all_rows:
            ald_desc = smiles_cache.get(r["aldehyde_smiles"])
            amine_desc = smiles_cache.get(r["amine_smiles"])
            for i, name in enumerate(DESCRIPTOR_NAMES):
                r[f"ald_3d_{name}"] = f"{ald_desc[i]:.6f}" if ald_desc else "0.0"
                r[f"amine_3d_{name}"] = f"{amine_desc[i]:.6f}" if amine_desc else "0.0"

            dimer_desc = dimer_cache.get((r["aldehyde_smiles"], r["amine_smiles"]))
            for i, name in enumerate(DIMER_DESCRIPTOR_NAMES):
                r[name] = f"{dimer_desc[i]:.6f}" if dimer_desc else "0.0"

    # 写入 CSV
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info(f"已写入: {args.output}")


if __name__ == "__main__":
    main()
