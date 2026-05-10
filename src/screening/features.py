"""特征工程模块 — 从 YAML 记录构建 ML 特征矩阵。

三层特征设计：
  Layer 1: 分子指纹 — Morgan ECFP4 (1024) + MACCS (167) × 2 单体
  Layer 2: 分子描述符 — MW, LogP, HBA/HBD, TPSA, F-count 等 × 2
  Layer 3: 配对特征 — F总数差、醛胺比、总MW 等
  Layer 4: 文本特征 — interface_type, synthesis_mode one-hot
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

from src.chemistry.fluorination import FluorineDetector
from src.chemistry.imine_check import ImineChecker
from src.chemistry.monomer import MonomerLibrary, extract_monomer_names
from src.utils.logger import setup_logger

logger = setup_logger("features")


class FeatureEngineer:
    """特征工程器 — 从 YAML 记录 + 单体分子构建特征矩阵。"""

    def __init__(self, monomer_lib: MonomerLibrary):
        self.monomer_lib = monomer_lib
        self.imine_checker = ImineChecker()
        self.f_detector = FluorineDetector()
        self._fp_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # -------------------------------------------------------------------
    # 单体级特征
    # -------------------------------------------------------------------

    def featurize_monomer(self, mol: Chem.Mol) -> Dict[str, Any]:
        """对单个单体生成 Morgan + MACCS 指纹和描述符。"""
        # Morgan ECFP4 (radius=2, 1024 bits)
        morgan = np.zeros(1024, dtype=np.float32)
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()
        _to_numpy(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), morgan)

        # MACCS (167 bits)
        maccs = np.zeros(167, dtype=np.float32)
        _to_numpy(MACCSkeys.GenMACCSKeys(mol), maccs)

        # 分子描述符
        desc = {
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "rot_bonds": Descriptors.NumRotatableBonds(mol),
            "arom_rings": Descriptors.NumAromaticRings(mol),
            "tpsa": Descriptors.TPSA(mol),
            "heteroatoms": Descriptors.NumHeteroatoms(mol),
            "fraction_csp3": Descriptors.FractionCSP3(mol),
            "f_count": self.f_detector.count_fluorine(mol),
            "has_cf3": 1 if self.f_detector.has_cf3(mol) else 0,
            "n_aldehyde": self.imine_checker.count_aldehyde_groups(mol),
            "n_amine": self.imine_checker.count_amine_groups(mol),
        }
        return {"morgan": morgan, "maccs": maccs, "descriptors": desc}

    def featurize_monomer_pair(self, mol1: Chem.Mol, mol2: Chem.Mol) -> np.ndarray:
        """对单体对构建联合特征向量。"""
        f1 = self.featurize_monomer(mol1)
        f2 = self.featurize_monomer(mol2)
        d1, d2 = f1["descriptors"], f2["descriptors"]

        # 配对级特征
        pair_f_total = d1["f_count"] + d2["f_count"]
        pair_f_diff = abs(d1["f_count"] - d2["f_count"])
        pair_mw = d1["mw"] + d2["mw"]
        pair_logp_diff = abs(d1["logp"] - d2["logp"])
        # 醛胺比：最小化功能基失衡（1:1 最优，失衡 → 低分）
        total_ald = d1["n_aldehyde"] + d2["n_aldehyde"]
        total_am = d1["n_amine"] + d2["n_amine"]
        pair_balance = min(total_ald, total_am) / max(total_ald, total_am, 1)
        pair_features = np.array([
            pair_f_total, pair_f_diff, pair_mw, pair_logp_diff, pair_balance,
            total_ald, total_am,
            d1["has_cf3"] or d2["has_cf3"],
            d1["f_count"], d2["f_count"],
        ], dtype=np.float32)

        return np.concatenate([
            f1["morgan"], f2["morgan"],
            f1["maccs"], f2["maccs"],
            _descriptors_to_array(d1),
            _descriptors_to_array(d2),
            pair_features,
        ])

    # -------------------------------------------------------------------
    # 特征矩阵构建
    # -------------------------------------------------------------------

    def build_feature_matrix(
        self, records: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
        """从文献记录构建完整特征矩阵 X 和标签 y。

        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,)
            feature_names: 特征名称列表
            metadata: 每条样本的元信息 [{literature_id, monomer1, monomer2, ...}]
        """
        X_list, y_list = [], []
        meta_list = []
        feature_names = None

        for rec in records:
            # 解析标签
            label = _extract_film_label(rec.get("film_crystallinity_fluorine", ""))
            if label is None:
                continue

            # 提取单体
            reagent = rec.get("reagent", "") or ""
            monomer_names = extract_monomer_names(reagent)
            if len(monomer_names) < 2:
                continue

            # 名称 → Mol
            mols = []
            for name in monomer_names:
                mol = self.monomer_lib.get_mol(name)
                if mol is not None:
                    mols.append((name, mol))

            # 分组
            aldehydes = [(n, m) for n, m in mols if self.imine_checker.is_aldehyde(m)]
            amines = [(n, m) for n, m in mols if self.imine_checker.is_amine(m)]

            if not aldehydes or not amines:
                continue

            # 生成所有醛-胺配对
            for ald_name, ald_mol in aldehydes:
                for am_name, am_mol in amines:
                    try:
                        feat = self.featurize_monomer_pair(ald_mol, am_mol)
                    except Exception:
                        continue
                    if feature_names is None:
                        feature_names = _make_feature_names(len(feat))
                    X_list.append(feat)
                    y_list.append(label)
                    meta_list.append({
                        "literature_id": rec.get("literature_id", ""),
                        "monomer_aldehyde": ald_name,
                        "monomer_amine": am_name,
                        "has_fluorine": self.f_detector.has_fluorine(ald_mol)
                        or self.f_detector.has_fluorine(am_mol),
                    })

        if not X_list:
            logger.error("未能构建任何有效样本")
            return np.array([]), np.array([]), [], []

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        logger.info(
            f"特征矩阵: X.shape={X.shape}, "
            f"正样本={int(y.sum())}, 负样本={int(len(y) - y.sum())}"
        )
        return X, y, feature_names, meta_list

    def build_from_llm_data(
        self, records: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
        """从 LLM 提取的 monomer_smiles_llm.json 记录直接构建特征矩阵。

        与 build_feature_matrix 不同，此方法使用 LLM 识别的：
        - monomer SMILES（无需名称解析）
        - monomer_type（aldehyde/amine 分类）
        - film_label（成膜标签）

        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,)
            feature_names: 特征名称列表
            metadata: 每条样本的元信息
        """
        X_list, y_list = [], []
        meta_list = []
        feature_names = None

        for rec in records:
            # 只保留 2D COF（is_2d_cof=True）
            if rec.get("is_2d_cof") is not True:
                continue

            # 标签：film_label 必须是 True/False
            film_label = rec.get("film_label")
            if film_label is None:
                continue
            label = 1 if film_label is True else 0

            # 获取有效单体
            monomers = [m for m in rec.get("monomers", []) if isinstance(m, dict)]
            valid_monomers = []
            for m in monomers:
                smi = m.get("canonical_smiles") or m.get("smiles", "")
                if not smi or smi.lower() in ("null", "none", ""):
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                valid_monomers.append({
                    "name": m.get("name", "?"),
                    "smiles": smi,
                    "mol": mol,
                    "type": m.get("monomer_type", "other"),
                })

            # 分组：醛 vs 胺
            aldehydes = [vm for vm in valid_monomers
                        if vm["type"] in ("aldehyde", "aldehyde-amine")]
            amines = [vm for vm in valid_monomers
                     if vm["type"] in ("amine", "aldehyde-amine")]

            if not aldehydes or not amines:
                continue

            # 生成所有醛-胺配对
            for ald in aldehydes:
                for am in amines:
                    # 跳过同一单体同时作为醛和胺的自配对（aldehyde-amine 型）
                    if ald["name"] == am["name"] and ald["smiles"] == am["smiles"]:
                        continue
                    try:
                        feat = self.featurize_monomer_pair(ald["mol"], am["mol"])
                    except Exception:
                        continue
                    if feature_names is None:
                        feature_names = _make_feature_names(len(feat))
                    X_list.append(feat)
                    y_list.append(label)
                    meta_list.append({
                        "literature_id": rec.get("literature_id", ""),
                        "monomer_aldehyde": ald["name"],
                        "monomer_amine": am["name"],
                        "aldehyde_smiles": ald["smiles"],
                        "amine_smiles": am["smiles"],
                        "has_fluorine": (self.f_detector.has_fluorine(ald["mol"])
                                        or self.f_detector.has_fluorine(am["mol"])),
                    })

        if not X_list:
            logger.error("未能从 LLM 数据构建任何有效样本")
            return np.array([]), np.array([]), [], []

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        logger.info(
            f"LLM特征矩阵: X.shape={X.shape}, "
            f"正样本={int(y.sum())}, 负样本={int(len(y) - y.sum())}"
        )
        return X, y, feature_names, meta_list
        """返回单个样本的特征维度（使用占位 Mol 计算）。"""
        placeholder = Chem.MolFromSmiles("C")  # 甲烷占位
        feat = self.featurize_monomer_pair(placeholder, placeholder)
        return len(feat)


# -----------------------------------------------------------------------
# 标签解析
# -----------------------------------------------------------------------

def _extract_film_label(text: str) -> Optional[int]:
    """从 film_crystallinity_fluorine 文本提取二元成膜标签。

    优先级从高到低：
    1. 强阳性关键词 → 1
    2. 强阴性关键词 → 0
    3. 弱阴性（粉末无成膜）→ 0
    4. 弱阳性（成膜无粉末）→ 1
    5. null/空 → None（排除）
    """
    if not text or text.strip().lower() in ("null", "none", "n/a", ""):
        return None

    t = text.lower()

    # ---- 强阳性 ----
    strong_pos = [
        "成膜性良好", "成膜性好", "成功制备.*膜",
        "连续薄膜", "自支撑膜", "free-standing film",
        "free standing film", "standalone film",
        "致密薄膜", "均匀薄膜", "无缺陷薄膜",
        "连续.*膜", "大面积.*膜", "自组装.*膜",
    ]
    for pat in strong_pos:
        if re.search(pat, t):
            return 1

    # ---- 强阴性 ----
    strong_neg = [
        "不成膜", "未成膜", "难以成膜", "无法成膜",
        "非成膜", "不能成膜", "难以.*成膜",
    ]
    for pat in strong_neg:
        if re.search(pat, t):
            return 0

    # ---- 弱信号判断 ----
    has_film = bool(re.search(r"成膜|薄膜|film|membrane|膜\b", t))
    has_powder = bool(re.search(r"粉末|powder|块状|bulk powder|颗粒|微晶粉末", t))
    has_no_film = bool(re.search(r"不成膜|非膜.*形态|没有.*膜|未.*成膜", t))

    if has_film and not has_no_film:
        return 1
    if has_powder and not has_film:
        return 0
    if has_no_film:
        return 0

    # 有结晶性描述但完全不提及成膜 → 弱阴性（多数 COF 是粉末）
    has_crystallinity = bool(re.search(
        r"结晶|crystal|多晶|polycrystal|单晶|无定形|amorphous|结晶度",
        t
    ))
    if has_crystallinity and not has_film:
        return 0

    return None


# -----------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------

def _to_numpy(bv, arr: np.ndarray):
    """RDKit ExplicitBitVect → numpy array（写入预分配数组）。"""
    bv.ToBitString()  # 触发惰性求值
    for i in range(len(arr)):
        arr[i] = 1.0 if bv.GetBit(i) else 0.0


def _descriptors_to_array(d: Dict[str, Any]) -> np.ndarray:
    """描述符 dict → 固定顺序 numpy array。"""
    keys = [
        "mw", "logp", "hba", "hbd", "rot_bonds", "arom_rings",
        "tpsa", "heteroatoms", "fraction_csp3", "f_count",
        "has_cf3", "n_aldehyde", "n_amine",
    ]
    return np.array([d.get(k, 0) for k in keys], dtype=np.float32)


def _make_feature_names(dim: int) -> List[str]:
    """生成特征名称列表（占位命名）。"""
    # Morgan1 (1024) + Morgan2 (1024) + MACCS1 (167) + MACCS2 (167)
    # + desc1 (13) + desc2 (13) + pair (9)
    names = []
    for i in range(1024):
        names.append(f"morgan1_{i}")
    for i in range(1024):
        names.append(f"morgan2_{i}")
    for i in range(167):
        names.append(f"maccs1_{i}")
    for i in range(167):
        names.append(f"maccs2_{i}")
    desc_keys = [
        "mw", "logp", "hba", "hbd", "rot_bonds", "arom_rings",
        "tpsa", "heteroatoms", "fraction_csp3", "f_count",
        "has_cf3", "n_aldehyde", "n_amine",
    ]
    for k in desc_keys:
        names.append(f"desc1_{k}")
    for k in desc_keys:
        names.append(f"desc2_{k}")
    pair_keys = [
        "pair_f_total", "pair_f_diff", "pair_mw",
        "pair_logp_diff", "pair_balance",
        "total_aldehyde", "total_amine",
        "either_has_cf3", "f1_count", "f2_count",
    ]
    for k in pair_keys:
        names.append(k)
    # 截断或填充至 dim
    while len(names) < dim:
        names.append(f"extra_{len(names)}")
    return names[:dim]
