"""路线 A 单体筛选 — 基于 LLM 提取的单体数据。

严格 2D COF 筛选：
  - 醛单体 ≥2 醛基 (C2/C3/C4 对称)
  - 胺单体 ≥2 伯胺基 (C2/C3/C4 对称)
  - C3+C2/C3+C3 → 六方 (hcb), C2+C2 → 四方 (sql)

配对策略（氟策略）：
  (1) F-醛 × 非F-胺  (2) 非F-醛 × F-胺
  (3) F-醛 × F-胺    (4) 非F-醛 × 非F-胺
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.logger().setLevel(RDLogger.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.features import FeatureEngineer
from src.chemistry.monomer import MonomerLibrary
from src.chemistry.imine_check import ImineChecker
from src.chemistry.fluorination import FluorineDetector
from src.chemistry.linker_analyzer import (
    is_functionally_symmetric, has_heterocycle, count_aromatic_rings,
)
from src.utils.logger import setup_logger

logger = setup_logger("screen_2d")

# ── Phase 2 硬规则常量 ──
MAX_AROMATIC_RINGS = 4       # 规则 1: 单体芳环数上限
HETEROCYCLE_PENALTY = 0.85   # 规则 3: 含杂环单体对的 margin 降权系数

# 官能团 SMARTS (用于类型感知对称性检测 + 对位检查)
_ALD_SMARTS = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
_AM_SMARTS = Chem.MolFromSmarts("[NH2][c]")          # 仅芳香伯胺
_BENZENE_SMARTS = Chem.MolFromSmarts("c1ccccc1")     # 苯环
_PROPARGYL_ETHER = Chem.MolFromSmarts("cOCC#C")      # 炔丙基醚


def _check_monomer_symmetry(mol: Chem.Mol, n_ald: int, n_am: int) -> bool:
    """检测单体在目标官能团类型上的对称性 (处理 aldehyde-amine 双类型)。"""
    from rdkit.Chem.AllChem import GetMorganFingerprint

    if n_ald >= 2:
        matches = mol.GetSubstructMatches(_ALD_SMARTS)
    elif n_am >= 2:
        matches = mol.GetSubstructMatches(_AM_SMARTS)
    else:
        return False

    reactive = [m[0] for m in matches]
    if len(reactive) < 2:
        return False

    Chem.GetSymmSSSR(mol)
    fps = []
    for aidx in reactive:
        try:
            fp = GetMorganFingerprint(mol, 2, fromAtoms=[aidx], useChirality=True)
            fps.append(fp)
        except Exception:
            return False

    for i in range(1, len(fps)):
        if fps[i] != fps[0]:
            return False
    return True


def _check_para_position(mol: Chem.Mol, n_ald: int, n_am: int, topo: str) -> bool:
    """规则 #4: C2 单体的两个反应基团必须在同一苯环的对位 (1,4)。

    若两基团分属不同苯环 (联苯连接臂)，放行。
    """
    if topo != "C2":
        return True

    if n_ald >= 2:
        matches = mol.GetSubstructMatches(_ALD_SMARTS)
        reactive_atoms = [m[2] for m in matches]  # m[2] = ring carbon
    elif n_am >= 2:
        matches = mol.GetSubstructMatches(_AM_SMARTS)
        reactive_atoms = [m[1] for m in matches]  # m[1] = ring carbon
    else:
        return False
    if len(reactive_atoms) < 2:
        return False

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
                    return True
                elif ring_bonds in (1, 2):
                    return False

    return True


def _has_propargyl_ether(mol: Chem.Mol) -> bool:
    return mol.HasSubstructMatch(_PROPARGYL_ETHER)


def _check_c2_substituents(mol: Chem.Mol, topo: str, n_ald: int, n_am: int) -> bool:
    """规则 #5: C2 单体苯环取代基 >4 时，多余取代基限卤素。"""
    if topo != "C2":
        return True

    if n_ald >= 2:
        reactive_smarts = _ALD_SMARTS
        idx = 2  # m[2] = ring C in [CX3H1](=O)[#6]
    elif n_am >= 2:
        reactive_smarts = _AM_SMARTS
        idx = 1  # m[1] = ring C in [NH2][c]
    else:
        return False

    matches = mol.GetSubstructMatches(reactive_smarts)
    rings = mol.GetSubstructMatches(_BENZENE_SMARTS)
    _HALOGENS = {9, 17, 35, 53}

    for ring in rings:
        ring_set = set(ring)
        n_sub = 0
        n_nonhalo_extra = 0
        for aidx in ring:
            atom = mol.GetAtomWithIdx(aidx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in ring_set:
                    n_sub += 1
                    if nbr.GetAtomicNum() not in _HALOGENS and nbr.GetAtomicNum() != 1:
                        n_nonhalo_extra += 1

        if n_sub > 4:
            reactive = {m[idx] for m in matches if m[idx] in ring_set}
            if n_nonhalo_extra > len(reactive):
                return False

    return True


def extract_2d_monomers(llm_path: str) -> pd.DataFrame:
    """从 LLM 数据提取 2D COF 可用单体 (≥2 官能团)，按 Canonical SMILES 去重。"""
    with open(llm_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 去重文献
    seen_lid = {}
    for r in records:
        lid = r.get("literature_id", "")
        if lid and lid not in seen_lid:
            seen_lid[lid] = r
    unique_records = list(seen_lid.values())

    # 聚合：按 Canonical SMILES
    smi_info = {}  # canonical_smiles → {names, type, fluorine, count, n_ald, n_am}
    imine_checker = ImineChecker()
    f_detector = FluorineDetector()

    for rec in unique_records:
        for m in rec.get("monomers", []):
            if not isinstance(m, dict):
                continue
            name = m.get("name", "").strip()
            smi = m.get("canonical_smiles") or m.get("smiles", "")
            if not name or not smi or smi.lower() in ("null", "none", ""):
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            can_smi = Chem.MolToSmiles(mol, canonical=True)
            mtype = m.get("monomer_type", "other")

            if can_smi not in smi_info:
                n_ald = imine_checker.count_aldehyde_groups(mol)
                n_am = imine_checker.count_amine_groups(mol)
                smi_info[can_smi] = {
                    "smiles": can_smi,
                    "names": set(),
                    "best_name": name,
                    "monomer_type": mtype,
                    "has_fluorine": f_detector.has_fluorine(mol),
                    "n_f_atoms": f_detector.count_fluorine(mol),
                    "has_cf3": f_detector.has_cf3(mol),
                    "n_aldehyde": n_ald,
                    "n_amine": n_am,
                    "n_papers": 1,
                    "has_heterocycle": has_heterocycle(mol),
                    "n_aromatic_rings": count_aromatic_rings(mol),
                }
            else:
                smi_info[can_smi]["n_papers"] += 1

            smi_info[can_smi]["names"].add(name)
            # 保留最短的非空名称
            cur_best = smi_info[can_smi]["best_name"]
            if len(name) > 0 and len(name) < len(cur_best):
                smi_info[can_smi]["best_name"] = name

    # 排除非 COF 单体（还原剂、催化剂、小分子杂质等）
    COF_EXCLUDE = {
        Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
        for s in ["NN", "NN.O", "NO", "NO.O", "CO", "CCO", "CC(=O)O"]
        if Chem.MolFromSmiles(s) is not None
    }

    # 过滤：2D COF 需要 ≥2 官能团 + 硬规则 1/2
    valid = []
    n_excluded_rings = 0     # 规则 1: 芳环数超标
    n_excluded_symmetry = 0  # 规则 2: 官能团不对称
    n_excluded_para = 0      # 规则 4: C2 非对位
    n_excluded_propargyl = 0  # 规则 5: 炔丙基醚
    n_excluded_c2sub = 0    # 规则 6: C2 取代基超标
    for can_smi, info in smi_info.items():
        if can_smi in COF_EXCLUDE:
            continue
        mtype = info["monomer_type"]
        n_ald = info["n_aldehyde"]
        n_am = info["n_amine"]

        # 醛类单体：≥2 醛基
        is_ald = mtype in ("aldehyde", "aldehyde-amine") and n_ald >= 2
        # 胺类单体：≥2 伯胺基
        is_am = mtype in ("amine", "aldehyde-amine") and n_am >= 2
        # aldehyde-amine 型可同时作为醛和胺
        is_dual = mtype == "aldehyde-amine" and n_ald >= 2 and n_am >= 2

        if not (is_ald or is_am):
            continue

        mol = Chem.MolFromSmiles(can_smi)
        if mol is None:
            continue

        # 硬规则 1: 芳环数 ≤ MAX_AROMATIC_RINGS
        if info["n_aromatic_rings"] > MAX_AROMATIC_RINGS:
            n_excluded_rings += 1
            continue

        # 硬规则 2: 官能团必须对称
        if not _check_monomer_symmetry(mol, n_ald, n_am):
            n_excluded_symmetry += 1
            continue

        # 计算拓扑 (规则 #4 需要)
        if is_ald:
            topo = "C3" if n_ald >= 3 else "C2" if n_ald >= 2 else "?"
        else:
            topo = "C4" if n_am >= 4 else "C3" if n_am >= 3 else "C2" if n_am >= 2 else "?"

        # 硬规则 4: C2 必须对位
        if not _check_para_position(mol, n_ald, n_am, topo):
            n_excluded_para += 1
            continue

        # 硬规则 5: 炔丙基醚排除
        if _has_propargyl_ether(mol):
            n_excluded_propargyl += 1
            continue

        # 硬规则 6: C2 取代基 >4 限卤素
        if not _check_c2_substituents(mol, topo, n_ald, n_am):
            n_excluded_c2sub += 1
            continue

        info["is_aldehyde"] = is_ald or is_dual
        info["is_amine"] = is_am or is_dual
        info["is_dual"] = is_dual
        info["name"] = info["best_name"]
        info["topology"] = topo
        valid.append(info)

    if n_excluded_rings > 0:
        logger.info(f"硬规则1 排除 (芳环数>{MAX_AROMATIC_RINGS}): {n_excluded_rings} 个单体")
    if n_excluded_symmetry > 0:
        logger.info(f"硬规则2 排除 (官能团不对称): {n_excluded_symmetry} 个单体")
    if n_excluded_para > 0:
        logger.info(f"硬规则4 排除 (C2非对位): {n_excluded_para} 个单体")
    if n_excluded_propargyl > 0:
        logger.info(f"硬规则5 排除 (炔丙基醚): {n_excluded_propargyl} 个单体")
    if n_excluded_c2sub > 0:
        logger.info(f"硬规则6 排除 (C2取代基超标): {n_excluded_c2sub} 个单体")

    df = pd.DataFrame(valid)
    if len(df) == 0:
        return df

    # 计算分子量和拓扑类型
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    df["mw"] = [Descriptors.MolWt(m) if m else 0 for m in mols]

    # 拓扑类型: C2/C3/C4 对称性
    def _topology(n_ald, n_am):
        if n_ald >= 3:
            return "C3"
        elif n_ald >= 2:
            return "C2"
        elif n_am >= 4:
            return "C4"
        elif n_am >= 3:
            return "C3"
        elif n_am >= 2:
            return "C2"
        return "?"

    df["topology"] = [(_topology(r["n_aldehyde"], r["n_amine"])
                        if r["is_aldehyde"] else
                        _topology(0, r["n_amine"]))
                      for _, r in df.iterrows()]

    logger.info(f"总唯一 SMILES: {len(smi_info)}, 2D可用: {len(df)}")
    return df.sort_values("n_papers", ascending=False)


def _topology_label(ald_topo, am_topo):
    """根据醛/胺对称性返回 2D COF 拓扑类型。"""
    pair = f"{ald_topo}+{am_topo}"
    mapping = {
        "C3+C2": "六方 (hcb)",
        "C3+C3": "六方 (hcb)",
        "C2+C3": "六方 (hcb)",
        "C2+C2": "四方 (sql)",
        "C4+C2": "四方 (sql)",
        "C2+C4": "四方 (sql)",
        "C4+C4": "四方 (sql)",
        # C4 胺（四面体构型如 tetrakis(aminophenyl)ethane）与 C3 醛配对
        # 倾向于形成 3D COF，非标准 2D 拓扑
        "C3+C4": "C3+C4 (非标准)",
        "C4+C3": "C4+C3 (非标准)",
    }
    label = mapping.get(pair, f"{ald_topo}+{am_topo}")
    return label


def _is_2d_topology(label: str) -> bool:
    """判断拓扑标签是否对应标准 2D COF。"""
    return label.startswith("六方") or label.startswith("四方")


def main():
    parser = argparse.ArgumentParser(description="路线 A 单体筛选 — 2D COF")
    parser.add_argument("--llm", default="data/processed/monomer_smiles_llm.json")
    parser.add_argument("--extra-monomers", default=None,
                       help="额外单体 CSV (如商业单体), columns: smiles,name,monomer_type,has_fluorine,n_f_atoms,n_aldehyde,n_amine")
    parser.add_argument("--model-dir", default="models/v1.0")
    parser.add_argument("--cache", default="data/processed/monomer_smiles_cache.json")
    parser.add_argument("--output", default="data/processed/route_a_top40_v1.csv")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    # 检查模型
    model_path = os.path.join(args.model_dir, "xgboost_model.pkl")
    if not os.path.exists(model_path):
        logger.error(f"模型不存在: {model_path}")
        sys.exit(1)

    # 1. 提取 2D 可用单体
    logger.info("提取 2D COF 可用单体...")
    monomers = extract_2d_monomers(args.llm)
    if len(monomers) == 0:
        logger.error("未提取到任何 2D 可用单体")
        sys.exit(1)

    # 1b. 合并额外单体 (如商业单体)
    if args.extra_monomers and os.path.exists(args.extra_monomers):
        extra = pd.read_csv(args.extra_monomers, encoding="utf-8-sig")
        extra_2d = extra[extra["n_aldehyde"] >= 2] if "n_aldehyde" in extra.columns else extra
        extra_2d = extra_2d[extra_2d["monomer_type"].isin(["aldehyde", "amine", "aldehyde-amine"])]

        exist_inchi = {}
        for smi in monomers["smiles"]:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    exist_inchi[MolToInchiKey(mol)] = smi
            except Exception:
                pass

        new_rows = []
        dup_n = 0
        for _, row in extra_2d.iterrows():
            smi = row["smiles"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                inchi = MolToInchiKey(mol)
            except Exception:
                inchi = smi
            if inchi in exist_inchi:
                dup_n += 1
                continue
            exist_inchi[inchi] = smi

            n_ald = int(row.get("n_aldehyde", 0))
            n_am = int(row.get("n_amine", 0))
            mtype = row["monomer_type"]

            # 硬规则 1: 芳环数 ≤ MAX_AROMATIC_RINGS
            n_arom = count_aromatic_rings(mol)
            if n_arom > MAX_AROMATIC_RINGS:
                continue

            # 硬规则 2: 官能团对称性
            if not _check_monomer_symmetry(mol, n_ald, n_am):
                continue

            is_ald = mtype in ("aldehyde", "aldehyde-amine") and n_ald >= 2
            is_am = mtype in ("amine", "aldehyde-amine") and n_am >= 2
            dual = mtype == "aldehyde-amine" and n_ald >= 2 and n_am >= 2

            def _topo(n_al, n_am_):
                if n_al >= 3: return "C3"
                elif n_al >= 2: return "C2"
                elif n_am_ >= 4: return "C4"
                elif n_am_ >= 3: return "C3"
                elif n_am_ >= 2: return "C2"
                return "?"

            topo = _topo(n_ald if is_ald else 0, n_am if is_am else 0)

            # 硬规则 4: C2 必须对位
            if not _check_para_position(mol, n_ald, n_am, topo):
                continue

            # 硬规则 5: 炔丙基醚排除
            if _has_propargyl_ether(mol):
                continue

            # 硬规则 6: C2 取代基 >4 限卤素
            if not _check_c2_substituents(mol, topo, n_ald, n_am):
                continue

            mw = Descriptors.MolWt(mol)

            new_rows.append({
                "smiles": smi,
                "best_name": row.get("name", "?"),
                "name": row.get("name", "?"),
                "monomer_type": mtype,
                "has_fluorine": bool(row.get("has_fluorine", False)),
                "n_f_atoms": int(row.get("n_f_atoms", 0)),
                "has_cf3": bool(row.get("has_cf3", False)),
                "n_aldehyde": n_ald,
                "n_amine": n_am,
                "n_papers": int(row.get("n_papers", 0)),
                "source": row.get("source", "extra"),
                "is_aldehyde": is_ald or dual,
                "is_amine": is_am or dual,
                "is_dual": dual,
                "mw": mw,
                "topology": topo,
                "has_heterocycle": has_heterocycle(mol),
                "n_aromatic_rings": n_arom,
            })

        if new_rows:
            extra_df = pd.DataFrame(new_rows)
            monomers = pd.concat([monomers, extra_df], ignore_index=True)
            logger.info(f"合并额外单体: {len(new_rows)} 新增, {dup_n} InChI重复")
            logger.info(f"合并后总单体: {len(monomers)}")

    # 2. 分组
    aldehydes = monomers[monomers["is_aldehyde"]]
    amines = monomers[monomers["is_amine"]]

    f_ald = aldehydes[aldehydes["has_fluorine"]]
    nf_ald = aldehydes[~aldehydes["has_fluorine"]]
    f_am = amines[amines["has_fluorine"]]
    nf_am = amines[~amines["has_fluorine"]]

    logger.info(
        f"2D单体: 醛={len(aldehydes)}, 胺={len(amines)} | "
        f"F-醛={len(f_ald)}, F-胺={len(f_am)}, "
        f"非F-醛={len(nf_ald)}, 非F-胺={len(nf_am)}"
    )

    # 3. 路线 A 配对
    pair_specs = [
        (f_ald, nf_am, "F-醛 × 非F-胺"),
        (nf_ald, f_am, "非F-醛 × F-胺"),
        (f_ald, f_am, "F-醛 × F-胺"),
        (nf_ald, nf_am, "非F-醛 × 非F-胺"),
    ]

    pairs = []
    for ald_df, am_df, pair_type in pair_specs:
        if len(ald_df) == 0 or len(am_df) == 0:
            logger.info(f"  跳过 {pair_type}: 数据不足")
            continue
        for _, ald in ald_df.iterrows():
            for _, am in am_df.iterrows():
                # 跳过相同 SMILES 的自配对
                if ald["smiles"] == am["smiles"]:
                    continue
                pairs.append({
                    "aldehyde": ald["name"],
                    "amine": am["name"],
                    "aldehyde_smiles": ald["smiles"],
                    "amine_smiles": am["smiles"],
                    "aldehyde_f": ald["has_fluorine"],
                    "amine_f": am["has_fluorine"],
                    "aldehyde_topo": ald.get("topology", "?"),
                    "amine_topo": am.get("topology", "?"),
                    "pair_type": pair_type,
                    "ald_has_heterocycle": ald.get("has_heterocycle", False),
                    "am_has_heterocycle": am.get("has_heterocycle", False),
                })
        n_pairs = len(ald_df) * len(am_df)
        logger.info(f"  {pair_type}: {len(ald_df)}×{len(am_df)}={n_pairs}")

    pairs_df = pd.DataFrame(pairs)
    logger.info(f"路线 A 总配对: {len(pairs_df)}")

    if len(pairs_df) == 0:
        logger.error("未生成任何配对")
        sys.exit(1)

    # 加载训练集配对 (aldehyde_smiles, amine_smiles)，严格排除已见组合

    meta_path = os.path.join(os.path.dirname(args.output), "label_metadata.csv")
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)
        train_pairs = set()
        for _, row in meta_df.iterrows():
            a_smi = row.get("aldehyde_smiles", "")
            b_smi = row.get("amine_smiles", "")
            if a_smi and b_smi and pd.notna(a_smi) and pd.notna(b_smi):
                try:
                    a_can = Chem.MolToSmiles(Chem.MolFromSmiles(a_smi), canonical=True)
                    b_can = Chem.MolToSmiles(Chem.MolFromSmiles(b_smi), canonical=True)
                    train_pairs.add((a_can, b_can))
                except Exception:
                    continue
        logger.info(f"训练集配对: {len(train_pairs)} 对")

        # 过滤：剔除训练集中已出现的配对
        def _in_train(row):
            try:
                a_can = Chem.MolToSmiles(Chem.MolFromSmiles(row["aldehyde_smiles"]), canonical=True)
                b_can = Chem.MolToSmiles(Chem.MolFromSmiles(row["amine_smiles"]), canonical=True)
                return (a_can, b_can) in train_pairs
            except Exception:
                return True  # 解析失败也排除

        before = len(pairs_df)
        pairs_df = pairs_df[~pairs_df.apply(_in_train, axis=1)]
        logger.info(f"剔除训练集配对: {before} → {len(pairs_df)}")
    else:
        logger.warning(f"训练集元数据不存在: {meta_path}，跳过训练集排除")

    if len(pairs_df) == 0:
        logger.error("剔除训练集后无剩余配对")
        sys.exit(1)

    # 4. 预测
    logger.info("预测成膜概率...")
    monomer_lib = MonomerLibrary(cache_path=args.cache, use_pubchem=False)
    feature_eng = FeatureEngineer(monomer_lib)

    import pickle

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaler = None
    sp = os.path.join(args.model_dir, "scaler.pkl")
    if os.path.exists(sp):
        with open(sp, "rb") as f:
            scaler = pickle.load(f)

    selected_features = None
    ip = os.path.join(args.model_dir, "model_info.json")
    if os.path.exists(ip):
        with open(ip, "r", encoding="utf-8") as f:
            info = json.load(f)
            sf = info.get("selected_features", [])
            if sf and len(sf) > 0:
                selected_features = np.array(sf)

    # 使用 XGBoost raw margin scores（更广分布，避免 sigmoid 压缩至全 ~0.999）
    margins = []
    for _, row in pairs_df.iterrows():
        ald_mol = Chem.MolFromSmiles(row["aldehyde_smiles"])
        am_mol = Chem.MolFromSmiles(row["amine_smiles"])
        if ald_mol is None or am_mol is None:
            margins.append(np.nan)
            continue
        try:
            feat = feature_eng.featurize_monomer_pair(ald_mol, am_mol)
            feat = feat.reshape(1, -1)
            if selected_features is not None and len(selected_features) > 0:
                feat = feat[:, selected_features]
            if scaler is not None:
                feat = scaler.transform(feat)
            margins.append(float(model.predict(feat, output_margin=True)[0]))
        except Exception:
            margins.append(np.nan)

    pairs_df["margin"] = margins

    # 用训练集 margin 分布做 isotonic 校准
    try:
        from sklearn.isotonic import IsotonicRegression

        X_path = os.path.join(os.path.dirname(args.output), "X_features.npz")
        y_path = os.path.join(os.path.dirname(args.output), "y_labels.npy")
        if os.path.exists(X_path) and os.path.exists(y_path):
            X_train = np.load(X_path)["X"]
            y_train = np.load(y_path)
            if selected_features is not None and len(selected_features) > 0:
                X_train = X_train[:, selected_features]
            if scaler is not None:
                X_train = scaler.transform(X_train)
            train_margins = model.predict(X_train, output_margin=True)

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(train_margins, y_train.astype(float))
            calibrated = iso.predict(pairs_df["margin"].fillna(-999).values)
            pairs_df["film_probability"] = np.clip(calibrated, 0.001, 0.999)
            logger.info(
                f"概率校准完成: margin范围=[{train_margins.min():.2f}, {train_margins.max():.2f}]"
            )
    except Exception as e:
        logger.warning(f"概率校准失败: {e}")
        pairs_df["film_probability"] = np.nan

    # 用 margin 排序（更有区分度），添加百分位得分
    valid_margin = pairs_df["margin"].notna()
    pairs_df["margin_score"] = np.nan
    # z-score 归一化，映射到 0-100
    m_mean = pairs_df.loc[valid_margin, "margin"].mean()
    m_std = pairs_df.loc[valid_margin, "margin"].std()
    pairs_df.loc[valid_margin, "margin_score"] = (
        50 + 10 * (pairs_df.loc[valid_margin, "margin"] - m_mean) / m_std
    )
    pairs_df["margin_score"] = pairs_df["margin_score"].clip(0, 100)
    logger.info(
        f"Margin分布: mean={m_mean:.2f}, std={m_std:.2f}, "
        f"range=[{pairs_df.loc[valid_margin, 'margin'].min():.2f}, "
        f"{pairs_df.loc[valid_margin, 'margin'].max():.2f}]"
    )
    valid = pairs_df.dropna(subset=["margin"])

    # 按 (aldehyde_inchi, amine_inchi) 去重（InChI Key 处理互变异构体），保留最高 margin

    def _inchi_key(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi
            try:
                return MolToInchiKey(mol)
            except Exception:
                return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return smi

    pair_dedup = {}
    for _, row in valid.iterrows():
        key = (_inchi_key(row["aldehyde_smiles"]), _inchi_key(row["amine_smiles"]))
        if key not in pair_dedup or row["margin"] > pair_dedup[key]["margin"]:
            pair_dedup[key] = row
    deduped = pd.DataFrame(pair_dedup.values())
    logger.info(f"去重: {len(valid)} → {len(deduped)} 对")

    ranked = deduped.sort_values("margin_score", ascending=False)

    # 硬规则 3: 含杂环单体对降权 (软惩罚, 非硬排除)
    n_hetero_pairs = (ranked["ald_has_heterocycle"] | ranked["am_has_heterocycle"]).sum()
    ranked["hetero_penalty"] = 1.0
    hetero_mask = ranked["ald_has_heterocycle"] | ranked["am_has_heterocycle"]
    ranked.loc[hetero_mask, "hetero_penalty"] = HETEROCYCLE_PENALTY
    # 双侧杂环叠加降权
    double_hetero = ranked["ald_has_heterocycle"] & ranked["am_has_heterocycle"]
    ranked.loc[double_hetero, "hetero_penalty"] = HETEROCYCLE_PENALTY ** 2
    ranked["adjusted_score"] = ranked["margin_score"] * ranked["hetero_penalty"]
    ranked = ranked.sort_values("adjusted_score", ascending=False)
    if n_hetero_pairs > 0:
        logger.info(
            f"硬规则3 杂环降权: {n_hetero_pairs}/{len(ranked)} 对 "
            f"(单侧×{HETEROCYCLE_PENALTY}, 双侧×{HETEROCYCLE_PENALTY**2:.4f})"
        )

    # 添加拓扑标签
    ranked["topology"] = [
        _topology_label(r["aldehyde_topo"], r["amine_topo"])
        for _, r in ranked.iterrows()
    ]

    # 仅保留标准 2D 拓扑
    std2d = ranked[ranked["topology"].apply(_is_2d_topology)]
    skipped = len(ranked) - len(std2d)
    if skipped > 0:
        logger.info(f"排除非标准拓扑 (C3+C4等): {skipped} 对")

    top = std2d.head(args.top).reset_index(drop=True)

    # 5. 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    top.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nTop {args.top} 已保存至: {args.output}")

    # 打印
    print("\n" + "=" * 88)
    print(f"  Route A Top {min(args.top, len(top))} 单体对 — 2D COF 成膜预测 (adjusted_score排序)")
    print("=" * 88)
    print(f"  {'#':3s}  {'调整分':7s}  {'原始分':7s}  {'醛单体':30s}  {'胺单体':28s}  {'拓扑':10s}")
    print("  " + "-" * 86)
    for i, (_, row) in enumerate(top.iterrows()):
        print(
            f"  [{i+1:2d}]  {row['adjusted_score']:6.1f}  "
            f"{row['margin_score']:6.1f}  "
            f"{str(row['aldehyde'])[:28]:28s}  {str(row['amine'])[:26]:26s}  "
            f"{row['topology']:10s}"
        )


if __name__ == "__main__":
    main()
