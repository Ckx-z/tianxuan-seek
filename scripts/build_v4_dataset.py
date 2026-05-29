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
from src.utils.logger import setup_logger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = setup_logger("build_v4")


def _canon(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else ""


# ── 5 种边界化学属性检测 ──

_ANILINE_PAT = Chem.MolFromSmarts("[NH2][c]")
_PYRIDINE_AMINE_PAT = Chem.MolFromSmarts("[NH2][n]")


def _detect_rigid(info: MonomerInfo) -> bool:
    """刚性: 芳环≥3 且 可旋转键≤1。"""
    if info.n_rings < 3:
        return False
    mol = Chem.MolFromSmiles(info.canonical_smiles)
    if mol is None:
        return False
    return rdMolDescriptors.CalcNumRotatableBonds(mol) <= 1


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
    target_count: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """用化学规则生成确定性负样本，与已有样本去重。"""
    import random
    random.seed(seed)
    np.random.seed(seed)

    # 收集已有配对（正+负），用于去重
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
    logger.info(
        f"替换池: {len(pool.all_monomers)} 个单体 "
        f"(对称={sum(1 for m in pool.all_monomers if m.is_symmetric)}, "
        f"多环>4={sum(1 for m in pool.all_monomers if m.n_rings > 4)})"
    )

    # 策略 1: 从正样本替换生成 (asymmetry/multiring/oversub/nonpara)
    # 每条正样本生成 2 个变体，增加生成量
    synth = generate_synthetic_pairs(
        positive_pairs_raw,
        pool,
        strategies=("asymmetry", "multiring", "oversub", "nonpara"),
        max_per_pair=2,
        multiring_max_rings=6,
        oversub_max_excess=2,
    )

    # 策略 2: 官能团不足配对 — 从单体池中选 C1 单体配对
    c1_alds = [m for m in pool.all_monomers
               if m.monomer_type == "aldehyde" and m.topology == "C1"]
    c1_amines = [m for m in pool.all_monomers
                 if m.monomer_type == "amine" and m.topology == "C1"]

    c1_pairs = []
    for ald in c1_alds[:50]:
        for am in c1_amines[:50]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                c1_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles,
                    am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am,
                    strategy="c1_insufficient_fg",
                    replaced="both",
                ))
    logger.info(f"C1 官能团不足配对: {len(c1_pairs)}")

    # 策略 3: 几何不兼容配对 (C3+C4, C4+C4)
    c3_mons = [m for m in pool.all_monomers if m.topology == "C3"]
    c4_mons = [m for m in pool.all_monomers if m.topology == "C4"]
    c3_alds = [m for m in c3_mons if m.monomer_type == "aldehyde"]
    c3_amines = [m for m in c3_mons if m.monomer_type == "amine"]
    c4_alds = [m for m in c4_mons if m.monomer_type == "aldehyde"]
    c4_amines = [m for m in c4_mons if m.monomer_type == "amine"]

    geo_pairs = []
    # C3+C4
    for ald in c3_alds[:20]:
        for am in c4_amines[:20]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                geo_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles,
                    am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am,
                    strategy="c3_c4_incompatible",
                    replaced="both",
                ))
    # C4+C4
    for ald in c4_alds[:20]:
        for am in c4_amines[:20]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                geo_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles,
                    am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am,
                    strategy="c4_c4_incompatible",
                    replaced="both",
                ))
    logger.info(f"几何不兼容配对: {len(geo_pairs)}")

    # ── 5 种边界化学负样本 ──

    # 预分类所有单体
    rigid_alds = [m for m in pool.all_monomers if m.monomer_type == "aldehyde" and _detect_rigid(m)]
    rigid_amines = [m for m in pool.all_monomers if m.monomer_type == "amine" and _detect_rigid(m)]
    flexible_alds = [m for m in pool.all_monomers if m.monomer_type == "aldehyde" and _detect_flexible(m)]
    flexible_amines = [m for m in pool.all_monomers if m.monomer_type == "amine" and _detect_flexible(m)]
    nonplanar_alds = [m for m in pool.all_monomers if m.monomer_type == "aldehyde" and _detect_sp3_bridge(m)]
    nonplanar_amines = [m for m in pool.all_monomers if m.monomer_type == "amine" and _detect_sp3_bridge(m)]
    weak_amines = [m for m in pool.all_monomers if _detect_weak_amine(m)]
    perfluoro_alds = [m for m in pool.all_monomers if m.monomer_type == "aldehyde" and _detect_perfluoro(m)]
    perfluoro_amines = [m for m in pool.all_monomers if m.monomer_type == "amine" and _detect_perfluoro(m)]

    boundary_pairs = []

    # 策略 4: 柔性-刚性失配 — 刚性醛+柔性胺 或 柔性醛+刚性胺
    for ald in rigid_alds[:15]:
        for am in flexible_amines[:15]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                boundary_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles, am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am, strategy="flex_rigid_mismatch", replaced="both"))
    for ald in flexible_alds[:15]:
        for am in rigid_amines[:15]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                boundary_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles, am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am, strategy="flex_rigid_mismatch", replaced="both"))
    logger.info(f"  柔性-刚性失配: {len(boundary_pairs)} (刚性醛={len(rigid_alds)}, 柔性胺={len(flexible_amines)}, 柔性醛={len(flexible_alds)}, 刚性胺={len(rigid_amines)})")

    # 策略 5: C1+C3 临界 — 刚好在 2D 网络形成边缘
    c1_boundary_alds = [m for m in pool.all_monomers if m.monomer_type == "aldehyde" and m.topology == "C1"]
    c3_boundary_alds = [m for m in pool.all_monomers if m.monomer_type == "aldehyde" and m.topology == "C3"]
    c1_boundary_amines = [m for m in pool.all_monomers if m.monomer_type == "amine" and m.topology == "C1"]
    c3_boundary_amines = [m for m in pool.all_monomers if m.monomer_type == "amine" and m.topology == "C3"]
    pre_boundary = len(boundary_pairs)
    for ald in c1_boundary_alds[:20]:
        for am in c3_boundary_amines[:20]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                boundary_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles, am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am, strategy="c1_c3_boundary", replaced="both"))
    for ald in c3_boundary_alds[:20]:
        for am in c1_boundary_amines[:20]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                boundary_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles, am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am, strategy="c1_c3_boundary", replaced="both"))
    logger.info(f"  C1+C3 临界: +{len(boundary_pairs)-pre_boundary}")

    # 策略 6: 平面性不足 — sp3 桥接单体 + 平面单体
    planar_alds = [m for m in pool.all_monomers if m.monomer_type == "aldehyde" and not _detect_sp3_bridge(m) and m.n_rings >= 1]
    planar_amines = [m for m in pool.all_monomers if m.monomer_type == "amine" and not _detect_sp3_bridge(m) and m.n_rings >= 1]
    pre_boundary = len(boundary_pairs)
    for ald in nonplanar_alds[:15]:
        for am in planar_amines[:15]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                boundary_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles, am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am, strategy="nonplanar", replaced="aldehyde"))
    for ald in planar_alds[:15]:
        for am in nonplanar_amines[:15]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                boundary_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles, am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am, strategy="nonplanar", replaced="amine"))
    logger.info(f"  平面性不足: +{len(boundary_pairs)-pre_boundary} (sp3桥接醛={len(nonplanar_alds)}, 桥接胺={len(nonplanar_amines)})")

    # 策略 7: 弱亲核胺 — 苯胺/吡啶胺 + 正常醛
    normal_alds = [m for m in pool.all_monomers if m.monomer_type == "aldehyde" and m.topology in ("C2", "C3")]
    pre_boundary = len(boundary_pairs)
    for ald in normal_alds[:20]:
        for am in weak_amines[:15]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                boundary_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles, am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am, strategy="weak_nucleophile", replaced="amine"))
    logger.info(f"  弱亲核胺: +{len(boundary_pairs)-pre_boundary} (弱胺={len(weak_amines)})")

    # 策略 8: 过量氟取代 — 全氟芳环单体 + 正常单体
    normal_amines = [m for m in pool.all_monomers if m.monomer_type == "amine" and m.topology in ("C2", "C3") and not _detect_perfluoro(m)]
    pre_boundary = len(boundary_pairs)
    for ald in perfluoro_alds[:15]:
        for am in normal_amines[:20]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                boundary_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles, am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am, strategy="excess_fluoro", replaced="aldehyde"))
    normal_alds_nof = [m for m in pool.all_monomers if m.monomer_type == "aldehyde" and m.topology in ("C2", "C3") and not _detect_perfluoro(m)]
    for ald in normal_alds_nof[:20]:
        for am in perfluoro_amines[:15]:
            if (ald.canonical_smiles, am.canonical_smiles) not in existing_pairs:
                boundary_pairs.append(SyntheticPair(
                    ald_smiles=ald.canonical_smiles, am_smiles=am.canonical_smiles,
                    ald_info=ald, am_info=am, strategy="excess_fluoro", replaced="amine"))
    logger.info(f"  过量氟取代: +{len(boundary_pairs)-pre_boundary} (全氟醛={len(perfluoro_alds)}, 全氟胺={len(perfluoro_amines)})")

    # 合并所有策略
    all_synth = synth + c1_pairs + geo_pairs + boundary_pairs

    # 去重：不与已有正/负样本重复
    deduped = []
    dup_with_existing = 0
    dup_internal = 0
    seen = set(existing_pairs)

    for sp in all_synth:
        key = (sp.ald_smiles, sp.am_smiles)
        if key in seen:
            if key in existing_pairs:
                dup_with_existing += 1
            else:
                dup_internal += 1
            continue
        seen.add(key)
        deduped.append(sp)

    logger.info(
        f"去重: 生成={len(all_synth)} → "
        f"撞已有={dup_with_existing}, 内部重复={dup_internal} → 保留={len(deduped)}"
    )
    logger.info(f"策略分布: {Counter(sp.strategy for sp in deduped)}")

    # 如果超过 target_count，按策略均衡采样
    if len(deduped) > target_count:
        strat_counts = Counter(sp.strategy for sp in deduped)
        logger.info(f"生成 {len(deduped)} > 目标 {target_count}，均衡采样")
        per_strat = max(1, target_count // len(strat_counts))
        by_strat: dict[str, list[SyntheticPair]] = {}
        for sp in deduped:
            by_strat.setdefault(sp.strategy, []).append(sp)
        sampled = []
        for strat, pairs in by_strat.items():
            n = min(per_strat, len(pairs))
            random.shuffle(pairs)
            sampled.extend(pairs[:n])
        # 补足
        if len(sampled) < target_count:
            remaining = [sp for sp in deduped if sp not in sampled]
            random.shuffle(remaining)
            sampled.extend(remaining[:target_count - len(sampled)])
        deduped = sampled[:target_count]

    # 转换为 CSV 行格式
    chem_neg_rows = []
    for sp in deduped:
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

    logger.info(f"最终化学规则负样本: {len(chem_neg_rows)}")
    return chem_neg_rows


def main():
    parser = argparse.ArgumentParser(description="构建 v4 训练集")
    parser.add_argument("--input", type=str, default="data/processed/v3_train.csv")
    parser.add_argument("--output", type=str, default="data/processed/v4_train.csv")
    parser.add_argument("--pool", type=str, default="data/processed/merged_monomer_pool.csv")
    parser.add_argument("--target-neg", type=int, default=1000,
                        help="化学规则负样本目标数量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Step 1: 清洗
    logger.info("=== Step 1: 清洗训练数据 ===")
    kept, discarded, conflicts = load_and_clean(args.input)

    # Step 2: 生成化学规则负样本
    logger.info("=== Step 2: 生成化学规则负样本 ===")
    chem_neg = generate_chem_negatives(kept, args.pool, args.target_neg, args.seed)

    # Step 3: 合并
    logger.info("=== Step 3: 合并 ===")
    all_rows = kept + chem_neg

    pos_count = sum(1 for r in all_rows if r["is_film"] == "1")
    neg_count = sum(1 for r in all_rows if r["is_film"] == "0")
    lit_pos = sum(1 for r in kept if r["is_film"] == "1")
    lit_neg = sum(1 for r in kept if r["is_film"] == "0")
    chem_neg_count = len(chem_neg)

    logger.info(
        f"最终训练集: {len(all_rows)} 样本\n"
        f"  文献正样本: {lit_pos}\n"
        f"  文献负样本: {lit_neg}\n"
        f"  化学规则负样本: {chem_neg_count}\n"
        f"  总正: {pos_count} ({pos_count/len(all_rows)*100:.1f}%)\n"
        f"  总负: {neg_count} ({neg_count/len(all_rows)*100:.1f}%)\n"
        f"  正:负 = 1:{neg_count/pos_count:.1f}"
    )

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
