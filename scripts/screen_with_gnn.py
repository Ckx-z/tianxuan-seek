"""Route A 筛选 — GNN+Bilinear 模型 + 全量单体池 + Phase 2 硬规则。

单体来源:
  - merged_monomer_pool.csv (LLM + 商业单体)
  - label_metadata_v4.csv 中的醛/胺 SMILES (含 group2/group3)

模型: GNN 编码器 + BilinearHead (end_to_end_a.pt, PR-AUC 0.731)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem.AllChem import GetMorganFingerprint
from collections import Counter

RDLogger.logger().setLevel(RDLogger.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.linker_analyzer import (
    has_acetylene, count_acetylene, has_heterocycle, count_aromatic_rings,
    compute_monomer_descriptors, compute_pair_descriptor_vector,
)
from src.chemistry.imine_check import ImineChecker
from src.chemistry.fluorination import FluorineDetector
from src.screening.gnn import MoleculeEncoder, smiles_to_graph, collate_graphs
from scripts.train_pair_predictor_a import EndToEndModel, BilinearHead
from src.utils.logger import setup_logger

logger = setup_logger("screen_gnn")

HIDDEN = 256
DEVICE = "cpu"
BATCH_SIZE = 64

# ── Phase 2 硬规则 ──
MAX_AROMATIC_RINGS = 4
HETEROCYCLE_PENALTY = 0.85

_ALD_SMARTS = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
_AM_SMARTS = Chem.MolFromSmarts("[NH2][c]")          # 仅芳香伯胺 (排除酰肼、脂肪胺、磺酰胺)
_BENZENE_SMARTS = Chem.MolFromSmarts("c1ccccc1")     # 苯环硬规则
_PROPARGYL_ETHER = Chem.MolFromSmarts("cOCC#C")      # 炔丙基醚 (醚键+炔基共存)

def _has_benzene_ring(mol: Chem.Mol) -> bool:
    """单体是否至少含一个苯环 (全碳六元芳香环)。"""
    return mol.HasSubstructMatch(_BENZENE_SMARTS)

def _count_aromatic_amines(mol: Chem.Mol) -> int:
    """统计芳香伯胺数量 ([NH2] 直接连接芳香碳)。"""
    return len(mol.GetSubstructMatches(_AM_SMARTS, uniquify=True))


def _canon(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else smi


def _check_monomer_symmetry(mol: Chem.Mol, n_ald: int, n_am: int) -> bool:
    """类型感知的官能团对称性检测 (CanonicalRankAtoms 全分子对称感知)。"""
    from rdkit.Chem import CanonicalRankAtoms

    if n_ald >= 2:
        matches = mol.GetSubstructMatches(_ALD_SMARTS)
    elif n_am >= 2:
        matches = mol.GetSubstructMatches(_AM_SMARTS)
    else:
        return False
    reactive = [m[0] for m in matches]
    if len(reactive) < 2:
        return False

    ranks = CanonicalRankAtoms(mol, breakTies=False)
    reactive_ranks = [ranks[a] for a in reactive]
    return all(r == reactive_ranks[0] for r in reactive_ranks[1:])


def _check_para_position(mol: Chem.Mol, n_ald: int, n_am: int, topo: str) -> bool:
    """规则 #4: C2 单体的两个反应基团必须在同一苯环的对位 (1,4)，不允许邻/间位。

    若两基团分属不同苯环 (如联苯连接臂)，放行——各自独立定位不受几何约束。
    """
    if topo != "C2":
        return True  # 仅检查 C2 单体

    if n_ald >= 2:
        matches = mol.GetSubstructMatches(_ALD_SMARTS)
        # m[2] = [#6] ring carbon the formyl is attached to (m[0] is formyl C, not in ring)
        reactive_atoms = [m[2] for m in matches]
    elif n_am >= 2:
        matches = mol.GetSubstructMatches(_AM_SMARTS)
        # m[1] = [c] ring carbon the NH2 is attached to (m[0] is N)
        reactive_atoms = [m[1] for m in matches]
    else:
        return False
    if len(reactive_atoms) < 2:
        return False

    rings = mol.GetSubstructMatches(_BENZENE_SMARTS)

    for ring in rings:
        ring_set = set(ring)
        on_ring = [a for a in reactive_atoms if a in ring_set]
        if len(on_ring) < 2:
            continue  # 两基团不在同一苯环，跳过

        # 同环 → 必须对位 (环内最短路径 = 3 键)
        for i in range(len(on_ring)):
            for j in range(i + 1, len(on_ring)):
                path = Chem.GetShortestPath(mol, on_ring[i], on_ring[j])
                ring_bonds = sum(
                    1 for k in range(len(path) - 1)
                    if path[k] in ring_set and path[k + 1] in ring_set
                )
                if ring_bonds == 3:  # para (1,4)
                    return True
                elif ring_bonds in (1, 2):  # ortho (1,2) / meta (1,3)
                    return False

    # 两基团不在同一苯环 → 通过 (联苯二醛/二胺等长连接臂)
    return True


def _has_propargyl_ether(mol: Chem.Mol) -> bool:
    """检测炔丙基醚 (醚键+炔基共存于同一取代基)。"""
    return mol.HasSubstructMatch(_PROPARGYL_ETHER)


def _check_c2_substituents(mol: Chem.Mol, topo: str, n_ald: int, n_am: int) -> bool:
    """规则 #5: C2 单体反应苯环上 >4 取代基时，多余取代基必须全部为卤素。

    对苯二甲醛衍生物常有 OH/OEt/F 等取代。若取代基数 >4，非卤素取代
    (如 OH, OMe, OEt, CH3) 会造成位阻过大或电子效应过于复杂。
    """
    if topo != "C2":
        return True

    if n_ald >= 2:
        reactive_smarts = _ALD_SMARTS
    elif n_am >= 2:
        reactive_smarts = _AM_SMARTS
    else:
        return False

    matches = mol.GetSubstructMatches(reactive_smarts)
    rings = mol.GetSubstructMatches(_BENZENE_SMARTS)

    _HALOGENS = {9, 17, 35, 53}  # F, Cl, Br, I
    _H = {1}

    for ring in rings:
        ring_set = set(ring)
        n_sub = 0
        n_nonhalo_extra = 0
        for aidx in ring:
            atom = mol.GetAtomWithIdx(aidx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in ring_set:
                    n_sub += 1
                    nbr_atomic = nbr.GetAtomicNum()
                    if nbr_atomic not in _HALOGENS and nbr_atomic not in _H:
                        n_nonhalo_extra += 1

        if n_sub > 4:
            reactive_ring_positions = set()
            for m in matches:
                if n_ald >= 2:
                    ring_c = m[2]  # [#6] atom
                else:
                    ring_c = m[1]  # [c] atom
                if ring_c in ring_set:
                    reactive_ring_positions.add(ring_c)

            # non-halogen substituents should only be the reactive groups
            n_reactive = len(reactive_ring_positions)
            if n_nonhalo_extra > n_reactive:
                return False

    return True


def _topology_label(ald_topo: str, am_topo: str) -> str:
    pair = f"{ald_topo}+{am_topo}"
    mapping = {
        "C3+C2": "六方 (hcb)", "C3+C3": "六方 (hcb)", "C2+C3": "六方 (hcb)",
        "C2+C2": "四方 (sql)", "C4+C2": "四方 (sql)", "C2+C4": "四方 (sql)",
        "C4+C4": "四方 (sql)",
        "C3+C4": "C3+C4 (非标准)", "C4+C3": "C4+C3 (非标准)",
    }
    return mapping.get(pair, f"{ald_topo}+{am_topo}")


def _is_2d_topology(label: str) -> bool:
    return label.startswith("六方") or label.startswith("四方")


def load_monomer_universe(pool_path: str, meta_path: str,
                        use_hard_rules: bool = True) -> pd.DataFrame:
    """加载全量单体池: merged_pool + training labels 中的单体。"""
    imine_checker = ImineChecker()
    f_detector = FluorineDetector()

    # 合并池
    pool = pd.read_csv(pool_path)
    all_smis = {}  # canonical_smiles → info

    for _, row in pool.iterrows():
        smi = str(row["smiles"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        if can in all_smis:
            all_smis[can]["n_papers"] = max(all_smis[can].get("n_papers", 0),
                                             int(row.get("n_papers", 0)))
            continue
        n_ald = int(row.get("n_aldehyde", 0))
        # 用 [NH2][c] 重新计算芳香胺数 (修复酰肼等误判)
        n_am = _count_aromatic_amines(mol)
        all_smis[can] = {
            "smiles": can,
            "name": str(row.get("best_name", row.get("name", "?"))),
            "monomer_type": str(row.get("monomer_type", "other")),
            "has_fluorine": bool(row.get("has_fluorine", False)),
            "n_f_atoms": int(row.get("n_f_atoms", 0)),
            "has_cf3": bool(row.get("has_cf3", False)),
            "n_aldehyde": n_ald,
            "n_amine": n_am,
            "n_papers": int(row.get("n_papers", 0)),
            "source": str(row.get("source", "pool")),
        }

    # 从训练集标签中补充缺失单体
    meta = pd.read_csv(meta_path, encoding="utf-8-sig")
    extra_count = 0
    for _, row in meta.iterrows():
        for col in ["aldehyde_smiles", "amine_smiles"]:
            smi = str(row.get(col, ""))
            if not smi or smi == "nan":
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol, canonical=True)
            if can in all_smis:
                continue
            n_ald = imine_checker.count_aldehyde_groups(mol)
            n_am = _count_aromatic_amines(mol)
            mtype = "aldehyde" if n_ald >= n_am else "amine"
            all_smis[can] = {
                "smiles": can,
                "name": can[:40],
                "monomer_type": mtype,
                "has_fluorine": f_detector.has_fluorine(mol),
                "n_f_atoms": f_detector.count_fluorine(mol),
                "has_cf3": f_detector.has_cf3(mol),
                "n_aldehyde": n_ald,
                "n_amine": n_am,
                "n_papers": 0,
                "source": "training_labels",
            }
            extra_count += 1

    logger.info(f"单体池: pool={len(pool)} + training_extra={extra_count} → unique={len(all_smis)}")

    # 应用规则过滤
    valid = []
    n_no_benzene, n_rings, n_sym, n_para, n_propargyl, n_c2sub, n_func = 0, 0, 0, 0, 0, 0, 0
    for can, info in all_smis.items():
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            continue

        mtype = info["monomer_type"]
        n_ald = info["n_aldehyde"]
        n_am = info["n_amine"]

        is_ald = mtype in ("aldehyde", "aldehyde-amine") and n_ald >= 2
        is_am = mtype in ("amine", "aldehyde-amine") and n_am >= 2
        is_dual = mtype == "aldehyde-amine" and n_ald >= 2 and n_am >= 2

        if not (is_ald or is_am):
            n_func += 1
            continue

        # 规则 0: 必须含苯环 (结构前提，chem_penalty 未覆盖)
        if not _has_benzene_ring(mol):
            n_no_benzene += 1
            continue

        # 规则 1: 芳环数 ≤ MAX_AROMATIC_RINGS (chem_penalty 覆盖)
        n_arom = count_aromatic_rings(mol)
        if use_hard_rules and n_arom > MAX_AROMATIC_RINGS:
            n_rings += 1
            continue

        # 规则 2: 对称性 (chem_penalty 覆盖)
        if use_hard_rules and not _check_monomer_symmetry(mol, n_ald, n_am):
            n_sym += 1
            continue

        # 计算拓扑标签 (规则 #4 需要)
        if is_ald:
            topo = "C3" if n_ald >= 3 else ("C2" if n_ald >= 2 else "?")
        else:
            if n_am >= 4:
                topo = "C4"
            elif n_am >= 3:
                topo = "C3"
            elif n_am >= 2:
                topo = "C2"
            else:
                topo = "?"

        # 规则 4: C2 必须对位 (chem_penalty 覆盖)
        if use_hard_rules and not _check_para_position(mol, n_ald, n_am, topo):
            n_para += 1
            continue

        # 规则 5: 炔丙基醚排除 (chem_penalty 未覆盖, 始终保留)
        if _has_propargyl_ether(mol):
            n_propargyl += 1
            continue

        # 规则 6: C2 取代基 >4 限卤素 (chem_penalty 覆盖)
        if use_hard_rules and not _check_c2_substituents(mol, topo, n_ald, n_am):
            n_c2sub += 1
            continue

        info["is_aldehyde"] = is_ald or is_dual
        info["is_amine"] = is_am or is_dual
        info["is_dual"] = is_dual
        info["topology"] = topo
        info["has_heterocycle"] = has_heterocycle(mol)
        info["n_aromatic_rings"] = n_arom
        info["mw"] = Descriptors.MolWt(mol)
        valid.append(info)

    logger.info(
        f"过滤: 官能团不足={n_func}, 无苯环={n_no_benzene}, "
        f"芳环>{MAX_AROMATIC_RINGS}={n_rings}, "
        f"不对称={n_sym}, 非对位C2={n_para}, "
        f"炔丙基醚={n_propargyl}, C2取代基超标={n_c2sub} → 可用={len(valid)}"
    )

    df = pd.DataFrame(valid)
    return df


def build_pairs(monomers: pd.DataFrame, meta_path: str) -> pd.DataFrame:
    """全量醛×胺配对, 排除训练集已见组合。"""
    aldehydes = monomers[monomers["is_aldehyde"]]
    amines = monomers[monomers["is_amine"]]

    f_ald = aldehydes[aldehydes["has_fluorine"]]
    nf_ald = aldehydes[~aldehydes["has_fluorine"]]
    f_am = amines[amines["has_fluorine"]]
    nf_am = amines[~amines["has_fluorine"]]

    logger.info(
        f"醛={len(aldehydes)} (F={len(f_ald)}, 非F={len(nf_ald)}), "
        f"胺={len(amines)} (F={len(f_am)}, 非F={len(nf_am)})"
    )

    pairs = []
    for _, ald in aldehydes.iterrows():
        for _, am in amines.iterrows():
            if ald["smiles"] == am["smiles"]:
                continue
            # 确定氟策略标签
            if ald["has_fluorine"] and not am["has_fluorine"]:
                ptype = "F-醛 × 非F-胺"
            elif not ald["has_fluorine"] and am["has_fluorine"]:
                ptype = "非F-醛 × F-胺"
            elif ald["has_fluorine"] and am["has_fluorine"]:
                ptype = "F-醛 × F-胺"
            else:
                ptype = "非F-醛 × 非F-胺"

            pairs.append({
                "aldehyde": ald["name"],
                "amine": am["name"],
                "aldehyde_smiles": ald["smiles"],
                "amine_smiles": am["smiles"],
                "aldehyde_f": ald["has_fluorine"],
                "amine_f": am["has_fluorine"],
                "aldehyde_topo": ald.get("topology", "?"),
                "amine_topo": am.get("topology", "?"),
                "pair_type": ptype,
                "ald_has_heterocycle": ald.get("has_heterocycle", False),
                "am_has_heterocycle": am.get("has_heterocycle", False),
            })

    pairs_df = pd.DataFrame(pairs)
    logger.info(f"全量配对: {len(pairs_df)}")

    # 排除训练集
    meta = pd.read_csv(meta_path, encoding="utf-8-sig")
    train_pairs = set()
    for _, row in meta.iterrows():
        try:
            a = _canon(str(row["aldehyde_smiles"]))
            b = _canon(str(row["amine_smiles"]))
            if a and b:
                train_pairs.add((a, b))
        except Exception:
            pass

    logger.info(f"训练集配对: {len(train_pairs)}")

    def _in_train(r):
        try:
            return (_canon(r["aldehyde_smiles"]), _canon(r["amine_smiles"])) in train_pairs
        except Exception:
            return True

    before = len(pairs_df)
    pairs_df = pairs_df[~pairs_df.apply(_in_train, axis=1)]
    logger.info(f"排除训练集: {before} → {len(pairs_df)}")

    return pairs_df


def score_with_gnn(pairs_df: pd.DataFrame, model, monomers: pd.DataFrame) -> pd.DataFrame:
    """GNN+Bilinear 推理 — 预计算单体嵌入, 仅 BilinearHead 参与配对。"""
    # 构建所有单体图
    smi_graph = {}
    for _, row in monomers.iterrows():
        smi = row["smiles"]
        if smi not in smi_graph:
            g = smiles_to_graph(smi)
            if g is not None:
                smi_graph[smi] = g

    # 构建 mol 缓存 (用于描述符)
    smi_mol = {}
    for smi in smi_graph:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi_mol[smi] = mol

    # 预计算所有单体 GNN 嵌入
    logger.info("  预计算单体 GNN 嵌入...")
    smi_embed = {}
    unique_smis = list(smi_graph.keys())
    for i in range(0, len(unique_smis), BATCH_SIZE):
        batch_smis = unique_smis[i:i + BATCH_SIZE]
        graphs = [smi_graph[s] for s in batch_smis]
        batch_data = collate_graphs(graphs)
        with torch.no_grad():
            emb = model.encoder(batch_data)
        for j, s in enumerate(batch_smis):
            smi_embed[s] = emb[j:j+1]

    # 批量推理: 只做 BilinearHead
    logger.info(f"  配对推理 ({len(pairs_df)} 对)...")
    logits = []
    n = len(pairs_df)
    for i in range(0, n, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n)
        batch = pairs_df.iloc[i:end]

        ea_list, eb_list, extras = [], [], []
        valid_indices = []

        for j, (_, row) in enumerate(batch.iterrows()):
            a_smi = row["aldehyde_smiles"]
            b_smi = row["amine_smiles"]
            if a_smi not in smi_embed or b_smi not in smi_embed:
                continue
            ea_list.append(smi_embed[a_smi])
            eb_list.append(smi_embed[b_smi])

            mol_a = smi_mol.get(a_smi)
            mol_b = smi_mol.get(b_smi)
            if mol_a is not None and mol_b is not None:
                vec = compute_pair_descriptor_vector(mol_a, mol_b)
            else:
                vec = np.zeros(26, dtype=np.float32)
            extras.append(vec)
            valid_indices.append(i + j)

        if not ea_list:
            continue

        ea = torch.cat(ea_list, dim=0)
        eb = torch.cat(eb_list, dim=0)
        extra_t = torch.tensor(np.stack(extras), dtype=torch.float)

        with torch.no_grad():
            batch_logits = model.head(ea, eb, extra=extra_t)
            logits.extend(zip(valid_indices, batch_logits.cpu().numpy().flatten()))

        if (i // BATCH_SIZE) % 50 == 0:
            logger.info(f"  GNN 推理: {end}/{n}")

    pairs_df["logit"] = np.nan
    for idx, val in logits:
        pairs_df.loc[idx, "logit"] = float(val)

    return pairs_df


def _compute_xgb_margins(pairs_df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """计算 XGBoost margin scores (Morgan 指纹 + 描述符 + 配对特征)。"""
    import pickle
    from src.screening.features import FeatureEngineer
    from src.chemistry.monomer import MonomerLibrary

    with open(os.path.join(model_dir, "xgboost_model.pkl"), "rb") as f:
        xgb = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, "model_info.json"), "r") as f:
        info = json.load(f)
    selected = np.array(info.get("selected_features", []))

    monomer_lib = MonomerLibrary(
        cache_path="data/processed/monomer_smiles_cache.json", use_pubchem=False,
    )
    fe = FeatureEngineer(monomer_lib)

    margins = []
    n = len(pairs_df)
    for i, (_, row) in enumerate(pairs_df.iterrows()):
        if i % 10000 == 0:
            logger.info(f"  XGBoost: {i}/{n}")
        a_smi = str(row.get("aldehyde_smiles", ""))
        b_smi = str(row.get("amine_smiles", ""))
        if not a_smi or not b_smi or a_smi == "nan" or b_smi == "nan":
            margins.append(np.nan)
            continue
        mol_a = Chem.MolFromSmiles(a_smi)
        mol_b = Chem.MolFromSmiles(b_smi)
        if mol_a is None or mol_b is None:
            margins.append(np.nan)
            continue
        try:
            feat = fe.featurize_monomer_pair(mol_a, mol_b).reshape(1, -1)
            if len(selected) > 0:
                feat = feat[:, selected]
            feat = scaler.transform(feat)
            margins.append(float(xgb.predict(feat, output_margin=True)[0]))
        except Exception:
            margins.append(np.nan)

    pairs_df["xgb_margin"] = margins
    logger.info(f"XGBoost 有效: {sum(1 for m in margins if not np.isnan(m))}/{n}")
    return pairs_df


def _select_top_stratified(
    ranked: pd.DataFrame, n_total: int = 40,
    c3_am_ratio: float = 0.40, c3_ald_ratio: float = 0.40,
) -> pd.DataFrame:
    """双模分层选取: 大胺小醛 (C3-胺) 40% + 大醛小胺 (C3-醛) 40% + 其余 20%。"""
    # 池 1: 大胺小醛 (any-aldehyde × C3-amine)
    dama_pool = ranked[ranked["amine_topo"] == "C3"]
    # 池 2: 大醛小胺 (C3-aldehyde × non-C3-amine)
    daan_pool = ranked[
        (ranked["aldehyde_topo"] == "C3") & (ranked["amine_topo"] != "C3")
    ]
    # 池 3: 其余 (C2×C2, C4, etc.)
    used_idx = set(dama_pool.index) | set(daan_pool.index)
    rest_pool = ranked[~ranked.index.isin(used_idx)]

    n_dama = int(n_total * c3_am_ratio)
    n_daan = int(n_total * c3_ald_ratio)
    n_rest = n_total - n_dama - n_daan

    dama_top = dama_pool.head(n_dama)
    daan_top = daan_pool.head(n_daan)
    rest_top = rest_pool.head(n_rest)

    # 各池不足时从其余池补充
    actual_dama = len(dama_top)
    actual_daan = len(daan_top)

    if actual_dama < n_dama:
        n_rest += n_dama - actual_dama
    if actual_daan < n_daan:
        n_rest += n_daan - actual_daan

    combined = pd.concat(
        [dama_top, daan_top, rest_pool.iloc[len(rest_top):len(rest_top) + max(0, n_rest)]],
        ignore_index=True,
    )
    # 补足 rest
    remaining = n_total - len(combined)
    if remaining > 0:
        extra = rest_pool.iloc[len(rest_top) + max(0, n_rest) : len(rest_top) + max(0, n_rest) + remaining]
        combined = pd.concat([combined, extra], ignore_index=True)

    combined = combined.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
    logger.info(
        f"分层: 大胺小醛(C3胺)={actual_dama}/{len(dama_pool)}, "
        f"大醛小胺(C3醛)={actual_daan}/{len(daan_pool)}, "
        f"其余={n_rest}/{len(rest_pool)} → 共{len(combined)}"
    )
    return combined


def main():
    parser = argparse.ArgumentParser(description="Route A GNN 筛选")
    parser.add_argument("--pool", default="data/processed/merged_monomer_pool.csv")
    parser.add_argument("--meta", default="data/processed/label_metadata_v4.csv")
    parser.add_argument("--model", default="models/v2.0/end_to_end_a.pt")
    parser.add_argument("--xgb-model", default="models/v1.0")
    parser.add_argument("--output", default="data/processed/route_a_gnn_top40.csv")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--no-hard-rules", action="store_true",
                        help="移除化学硬规则 (#1#2#4#6), 信任模型自行判断")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="输出文件后缀 (用于对比实验)")
    args = parser.parse_args()

    # 1. 加载全量单体
    hard_rules_enabled = not args.no_hard_rules
    tag = "no_hard" if args.no_hard_rules else "with_hard"
    logger.info(f"=== 1. 加载全量单体 (硬规则={'ON' if hard_rules_enabled else 'OFF'}) ===")
    monomers = load_monomer_universe(args.pool, args.meta,
                                     use_hard_rules=hard_rules_enabled)
    logger.info(f"可用单体: {len(monomers)} (醛={monomers['is_aldehyde'].sum()}, "
                f"胺={monomers['is_amine'].sum()})")

    # 2. 配对
    logger.info("=== 2. 全量配对 ===")
    pairs_df = build_pairs(monomers, args.meta)

    # 3. 加载 GNN 模型
    logger.info("=== 3. 加载 GNN 模型 ===")
    checkpoint = torch.load(args.model, map_location="cpu")
    encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.2)
    head = BilinearHead(hidden=HIDDEN, bilinear_rank=64, mlp_hidden=128, dropout=0.4, extra_dim=26)
    model = EndToEndModel(encoder, head)
    model.load_state_dict(checkpoint)
    model.eval()
    logger.info(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 4. GNN 推理
    logger.info("=== 4. GNN 推理 ===")
    pairs_df = score_with_gnn(pairs_df, model, monomers)

    valid = pairs_df.dropna(subset=["logit"])
    logger.info(f"有效预测: {len(valid)}/{len(pairs_df)}")

    # 5. XGBoost 推理
    logger.info("=== 5. XGBoost 推理 ===")
    valid = valid.dropna(subset=["aldehyde_smiles", "amine_smiles"]).copy()
    valid = _compute_xgb_margins(valid, args.xgb_model)
    valid = valid.dropna(subset=["xgb_margin"]).copy()
    logger.info(
        f"XGBoost margin: mean={valid['xgb_margin'].mean():.2f}, "
        f"min={valid['xgb_margin'].min():.2f}, max={valid['xgb_margin'].max():.2f}"
    )

    # 6. Ensemble: GNN (70%) + XGBoost (30%) − 分歧惩罚
    logger.info("=== 6. Ensemble (GNN 60% + XGBoost 40%) ===")
    DELTA = 0.10  # 分歧惩罚系数
    from sklearn.preprocessing import MinMaxScaler
    gnn_norm = MinMaxScaler().fit_transform(valid["logit"].values.reshape(-1, 1)).flatten()
    xgb_norm = MinMaxScaler().fit_transform(valid["xgb_margin"].values.reshape(-1, 1)).flatten()
    valid["gnn_norm"] = gnn_norm
    valid["xgb_norm"] = xgb_norm
    valid["divergence"] = np.abs(gnn_norm - xgb_norm)
    valid["ensemble_score"] = (
        0.60 * gnn_norm + 0.40 * xgb_norm - DELTA * valid["divergence"]
    )
    high_div = (valid["divergence"] > 0.5).sum()
    logger.info(
        f"Ensemble: GNN mean={gnn_norm.mean():.3f}, XGB mean={xgb_norm.mean():.3f}, "
        f"高分歧(>0.5)={high_div}/{len(valid)}"
    )

    # 7. C3 soft bonus (胺 + 醛)
    logger.info("=== 7. C3 bonus (胺 ×1.15, 醛 ×1.10) ===")
    c3_am_mask = valid["amine_topo"] == "C3"
    c3_ald_mask = valid["aldehyde_topo"] == "C3"
    valid["c3_bonus"] = 1.0
    valid.loc[c3_am_mask, "c3_bonus"] = 1.15  # 大胺小醛
    valid.loc[c3_ald_mask & ~c3_am_mask, "c3_bonus"] = 1.10  # 大醛小胺 (避免叠加)
    valid["margin_score"] = valid["ensemble_score"] * valid["c3_bonus"]
    logger.info(
        f"C3 胺受益: {c3_am_mask.sum()}, "
        f"C3 醛受益: {(c3_ald_mask & ~c3_am_mask).sum()}/{len(valid)} 对"
    )

    # 8. InChI 去重
    logger.info("=== 8. 去重 ===")
    pair_dedup = {}
    for _, row in valid.iterrows():
        try:
            k = (MolToInchiKey(Chem.MolFromSmiles(row["aldehyde_smiles"])),
                 MolToInchiKey(Chem.MolFromSmiles(row["amine_smiles"])))
        except Exception:
            k = (row["aldehyde_smiles"], row["amine_smiles"])
        if k not in pair_dedup or row["margin_score"] > pair_dedup[k]["margin_score"]:
            pair_dedup[k] = row
    deduped = pd.DataFrame(pair_dedup.values())
    logger.info(f"去重: {len(valid)} → {len(deduped)}")

    # 9. 规则 3: 杂环降权
    logger.info("=== 9. 规则3 杂环降权 ===")
    n_hetero = (deduped["ald_has_heterocycle"] | deduped["am_has_heterocycle"]).sum()
    deduped["hetero_penalty"] = 1.0
    mask = deduped["ald_has_heterocycle"] | deduped["am_has_heterocycle"]
    deduped.loc[mask, "hetero_penalty"] = HETEROCYCLE_PENALTY
    double = deduped["ald_has_heterocycle"] & deduped["am_has_heterocycle"]
    deduped.loc[double, "hetero_penalty"] = HETEROCYCLE_PENALTY ** 2
    deduped["adjusted_score"] = deduped["margin_score"] * deduped["hetero_penalty"]
    deduped = deduped.sort_values("adjusted_score", ascending=False)
    logger.info(f"受影响: {n_hetero}/{len(deduped)} 对")

    # 10. 拓扑过滤
    logger.info("=== 10. 拓扑过滤 ===")
    deduped["topology"] = [
        _topology_label(r["aldehyde_topo"], r["amine_topo"])
        for _, r in deduped.iterrows()
    ]
    std2d = deduped[deduped["topology"].apply(_is_2d_topology)]
    logger.info(f"标准2D拓扑: {len(std2d)} (排除 {len(deduped)-len(std2d)})")

    # 11. 分层选取 Top 40
    logger.info("=== 11. 分层选取 Top 40 (C3-胺 40%, C3-醛 40%) ===")
    top = _select_top_stratified(std2d, n_total=args.top, c3_am_ratio=0.40, c3_ald_ratio=0.40)

    # 保存
    output_path = args.output
    suffix = args.output_suffix or tag
    if suffix and suffix not in output_path:
        output_path = output_path.replace(".csv", f"_{suffix}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top.to_csv(output_path, index=False, encoding="utf-8-sig")
    deduped.to_csv(output_path.replace(".csv", "_full.csv"), index=False, encoding="utf-8-sig")

    print(f"\nTop {args.top} 已保存至: {output_path}")
    print(f"全量结果: {output_path.replace('.csv', '_full.csv')}")

    # 打印 Top 40
    print("\n" + "=" * 112)
    print(f"  Route A Top {args.top} — GNN+XGB 集成 + 四项硬规则 + C3 双模分层")
    print("=" * 112)
    print(f"  {'#':3s}  {'综合分':7s}  {'GNN':7s}  {'XGB':7s}  {'分歧':6s}  "
          f"{'醛(拓扑)':14s}  {'胺(拓扑)':14s}  {'拓扑':10s}  {'氟策略':14s}")
    print("  " + "-" * 110)
    for i, (_, row) in enumerate(top.iterrows()):
        ald_name = str(row['aldehyde'])[:11].encode("ascii", "replace").decode("ascii")
        am_name = str(row['amine'])[:11].encode("ascii", "replace").decode("ascii")
        ald_tag = f"{ald_name} {row['aldehyde_topo']}"
        am_tag = f"{am_name} {row['amine_topo']}"
        print(
            f"  [{i+1:2d}]  {row['adjusted_score']:6.3f}  "
            f"{row['gnn_norm']:6.3f}  {row['xgb_norm']:6.3f}  "
            f"{row['divergence']:5.2f}  "
            f"{ald_tag:14s}  {am_tag:14s}  "
            f"{row['topology']:10s}  {row['pair_type']:14s}"
        )

    # 统计
    print(f"\n--- 统计 ---")
    print(f"单体池: {len(monomers)} (醛={monomers['is_aldehyde'].sum()}, "
          f"胺={monomers['is_amine'].sum()})")
    print(f"配对: {len(pairs_df)}, 去重后: {len(deduped)}, 标准2D: {len(std2d)}")
    print(f"拓扑分布: {std2d['topology'].value_counts().to_dict()}")
    c3_am_in_top = (top["amine_topo"] == "C3").sum()
    c3_ald_in_top = (top["aldehyde_topo"] == "C3").sum()
    print(f"Top {args.top} 中 C3-胺: {c3_am_in_top}/{len(top)} ({100*c3_am_in_top/len(top):.0f}%), "
          f"C3-醛: {c3_ald_in_top}/{len(top)} ({100*c3_ald_in_top/len(top):.0f}%)")
    print(f"氟策略分布: {top['pair_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
