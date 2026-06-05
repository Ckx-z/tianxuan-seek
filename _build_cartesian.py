"""Step 1: 合并训练集 + 商业+LLM + target 单体池 → 笛卡尔积 → 排除 326 文献配对.

合并池子:
  醛池 = 训练集醛 (v4_train_3d_dimer.csv, n_ald>=2)
       U merged_monomer_pool 醛 (n_ald>=2)
       U target 醛 (无氟 3,3'-二苯基联苯二甲醛)
  胺池 = 训练集胺 (v4_train_3d_dimer.csv, n_am>=2)
       U merged_monomer_pool 胺 (n_am>=2)
       U target 胺 (TAPB)

去重 (canonical SMILES) -> 笛卡尔积 -> 排除 326 唯一正样本配对 -> 输出 CSV
"""
import os, sys, csv
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 复用 screen_v4.py 的辅助函数
from scripts.screen_v4 import (
    _canon_smiles, _count_valid_aldehydes, _count_valid_amines,
    _ALD_SMARTS, _AMINE_SMARTS, _HYDRAZIDE_SMARTS, _CARBOHYDRAZIDE_SMARTS,
    _AMIDE_NH2_SMARTS, _SULFONAMIDE_SMARTS, _ESTER_ALDEHYDE_SMARTS,
    _METALS,
)


def _has_metal(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return True
    return any(a.GetAtomicNum() in _METALS for a in mol.GetAtoms())


# 不合理元素: Se, Te, As, Hg, Cd, Pb 等
_BAD_ATOMIC_NUMS = {34, 52, 33, 80, 48, 82}

# 明显错误的 SMILES (LLM 提取错误)
_BAD_SMILES = {
    "O=CC=O",  # 乙二醛，被错误标为三甲酰基间苯三酚
    "O=CC=O.Oc1ccccc1",  # 乙二醛+苯酚，同样错误
}


def _is_valid_monomer_smiles(smi: str) -> bool:
    """检查 SMILES 是否为有效的单体分子。排除多片段/金属/不合理元素/已知错误。"""
    if not smi or smi == "nan":
        return False
    if smi in _BAD_SMILES:
        return False
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    frags = Chem.GetMolFrags(mol)
    if len(frags) > 1:
        return False
    if any(a.GetAtomicNum() in _METALS for a in mol.GetAtoms()):
        return False
    if any(a.GetAtomicNum() in _BAD_ATOMIC_NUMS for a in mol.GetAtoms()):
        return False
    return True


def is_amine_mol(mol):
    """是否为真正的伯胺单体 (不是酰肼/酰胺/磺酰胺)。"""
    if mol.HasSubstructMatch(_CARBOHYDRAZIDE_SMARTS):
        return False
    if mol.HasSubstructMatch(_HYDRAZIDE_SMARTS):
        return False
    if mol.HasSubstructMatch(_AMIDE_NH2_SMARTS):
        return False
    if mol.HasSubstructMatch(_SULFONAMIDE_SMARTS):
        return False
    if not mol.HasSubstructMatch(_AMINE_SMARTS):
        return False
    return _count_valid_amines(mol) > 0


def is_aldehyde_mol(mol):
    """是否为真正的醛单体 (不是酯羰基)。"""
    if mol.HasSubstructMatch(_ESTER_ALDEHYDE_SMARTS):
        return False
    if not mol.HasSubstructMatch(_ALD_SMARTS):
        return False
    return _count_valid_aldehydes(mol) > 0


# target 配对
TARGET_ALD = "O=Cc1cc(-c2ccccc2)c(C=O)cc1-c2ccccc2"
TARGET_AM = "Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1"
TARGET_ALD_NAME = "无氟版: 3,3'-二苯基-[1,1'-联苯]-4,4'-二甲醛"
TARGET_AM_NAME = "1,3,5-tris(4-aminophenyl)benzene (TAPB)"


def load_training_monomers(train_csv):
    """从 v4_train_3d_dimer.csv 提取训练集醛/胺 (n_ald>=1, n_am>=1, RDKit 可解析)。

    训练集字段没有 n_aldehyde/n_amine, 用 RDKit SMARTS 计算。
    训练集醛/胺强制进池 (不管 n_ald/n_am 是否 >=2):
      - 训练集是"成膜正样本", 涉及的醛/胺必须参与笛卡尔积
      - 否则这 326 配对中的 96 个不会被排除
      - n_ald=1/n_am=1 的单体配对会被 check_topology 软过滤, 不污染 Top 40
    """
    with open(train_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    alds = {}
    ams = {}
    for r in rows:
        ald_smi = r["aldehyde_smiles"].strip()
        am_smi = r["amine_smiles"].strip()
        ald_name = r.get("aldehyde_name", "") or ""
        am_name = r.get("amine_name", "") or ""
        has_f = r.get("has_fluorine_monomer", "") in ("1", "True", "true")

        # 醛 (n_ald>=1, 是真正的醛基)
        if ald_smi and _is_valid_monomer_smiles(ald_smi):
            ald_mol = Chem.MolFromSmiles(ald_smi)
            if ald_mol is not None and is_aldehyde_mol(ald_mol):
                n_ald = _count_valid_aldehydes(ald_mol)
                if n_ald >= 1:
                    canon = _canon_smiles(ald_smi)
                    alds[canon] = {
                        "smiles": ald_smi, "canon": canon,
                        "name": ald_name, "has_f": has_f,
                        "n_ald": n_ald, "source": "train",
                    }

        # 胺 (n_am>=1, 是真正的伯胺)
        if am_smi and _is_valid_monomer_smiles(am_smi):
            am_mol = Chem.MolFromSmiles(am_smi)
            if am_mol is not None and is_amine_mol(am_mol):
                n_am = _count_valid_amines(am_mol)
                if n_am >= 1:
                    canon = _canon_smiles(am_smi)
                    ams[canon] = {
                        "smiles": am_smi, "canon": canon,
                        "name": am_name, "has_f": has_f,
                        "n_am": n_am, "source": "train",
                    }
    return alds, ams


def load_commercial_monomers(pool_csv):
    """从 merged_monomer_pool.csv 提取醛/胺 (is_aldehyde/amine=True, n>=2)。"""
    with open(pool_csv, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    alds = {}
    ams = {}
    for r in rows:
        smi = r.get("smiles", "").strip()
        if not _is_valid_monomer_smiles(smi):
            continue

        n_ald = int(r.get("n_aldehyde", 0) or 0)
        n_am = int(r.get("n_amine", 0) or 0)
        is_ald = (r.get("is_aldehyde", "") == "True") or n_ald > 0
        is_am = (r.get("is_amine", "") == "True") or n_am > 0
        has_f = r.get("has_fluorine", "") == "True"
        name = r.get("best_name", "") or ""
        source = r.get("source", "unknown") or "unknown"

        if is_ald and n_ald >= 2:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and is_aldehyde_mol(mol):
                canon = _canon_smiles(smi)
                alds[canon] = {
                    "smiles": smi, "canon": canon, "name": name,
                    "has_f": has_f, "n_ald": n_ald, "source": source,
                }

        if is_am and n_am >= 2:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and is_amine_mol(mol):
                canon = _canon_smiles(smi)
                ams[canon] = {
                    "smiles": smi, "canon": canon, "name": name,
                    "has_f": has_f, "n_am": n_am, "source": source,
                }
    return alds, ams


def load_target_monomers():
    alds = {}
    ams = {}
    for label, smi, name, role in [
        ("ald", TARGET_ALD, TARGET_ALD_NAME, "ald"),
        ("am", TARGET_AM, TARGET_AM_NAME, "am"),
    ]:
        if not _is_valid_monomer_smiles(smi):
            continue
        mol = Chem.MolFromSmiles(smi)
        canon = _canon_smiles(smi)
        if role == "ald":
            n = _count_valid_aldehydes(mol)
            if n >= 2 and is_aldehyde_mol(mol):
                alds[canon] = {
                    "smiles": smi, "canon": canon, "name": name,
                    "has_f": False, "n_ald": n, "source": "target",
                }
        else:
            n = _count_valid_amines(mol)
            if n >= 2 and is_amine_mol(mol):
                ams[canon] = {
                    "smiles": smi, "canon": canon, "name": name,
                    "has_f": False, "n_am": n, "source": "target",
                }
    return alds, ams


def load_train_pairs(train_csv):
    """加载训练集 326 唯一正样本 (醛, 胺) 配对 (精确 SMILES 配对匹配)。"""
    pairs = set()
    with open(train_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r.get("is_film", 0)) != 1:
                continue
            ald_s = r["aldehyde_smiles"]
            am_s = r["amine_smiles"]
            if not ald_s or not am_s:
                continue
            ald_mol = Chem.MolFromSmiles(ald_s)
            am_mol = Chem.MolFromSmiles(am_s)
            if ald_mol is None or am_mol is None:
                continue
            pairs.add((_canon_smiles(ald_s), _canon_smiles(am_s)))
    return pairs


def main():
    out_csv = "data/processed/v4_cartesian_pairs.csv"

    print("=== Step 1.1: 加载训练集单体 (v4_train_3d_dimer.csv) ===")
    train_alds, train_ams = load_training_monomers("data/processed/v4_train_3d_dimer.csv")
    print("  醛:", len(train_alds))
    print("  胺:", len(train_ams))

    print("\n=== Step 1.2: 加载商业+LLM 单体 (merged_monomer_pool.csv) ===")
    com_alds, com_ams = load_commercial_monomers("data/processed/merged_monomer_pool.csv")
    print("  醛:", len(com_alds))
    print("  胺:", len(com_ams))

    print("\n=== Step 1.3: 加载 target 单体 ===")
    tgt_alds, tgt_ams = load_target_monomers()
    print("  醛:", len(tgt_alds))
    print("  胺:", len(tgt_ams))

    # 合并去重 (来源合并标注: train+commercial, train+llm 等)
    pool_alds = {}
    for d in (train_alds, com_alds, tgt_alds):
        for canon, info in d.items():
            if canon in pool_alds:
                # 合并 source
                existing_srcs = set(pool_alds[canon]["source"].split("+"))
                new_srcs = set(info["source"].split("+"))
                pool_alds[canon]["source"] = "+".join(sorted(existing_srcs | new_srcs))
            else:
                pool_alds[canon] = dict(info)

    pool_ams = {}
    for d in (train_ams, com_ams, tgt_ams):
        for canon, info in d.items():
            if canon in pool_ams:
                existing_srcs = set(pool_ams[canon]["source"].split("+"))
                new_srcs = set(info["source"].split("+"))
                pool_ams[canon]["source"] = "+".join(sorted(existing_srcs | new_srcs))
            else:
                pool_ams[canon] = dict(info)

    print("\n=== Step 1.4: 合并去重后单体池 ===")
    print("  醛总数:", len(pool_alds))
    src_count = {}
    for info in pool_alds.values():
        for s in info["source"].split("+"):
            src_count[s] = src_count.get(s, 0) + 1
    print("    按 source 拆分:", src_count)
    print("  胺总数:", len(pool_ams))
    src_count = {}
    for info in pool_ams.values():
        for s in info["source"].split("+"):
            src_count[s] = src_count.get(s, 0) + 1
    print("    按 source 拆分:", src_count)

    # 检查 target 是否在池中
    tgt_ald_canon = _canon_smiles(TARGET_ALD)
    tgt_am_canon = _canon_smiles(TARGET_AM)
    print("\n=== Step 1.5: Target 在池中? ===")
    print("  醛 {}: in_pool={}".format(TARGET_ALD[:30], tgt_ald_canon in pool_alds))
    print("  胺 {}: in_pool={}".format(TARGET_AM[:30], tgt_am_canon in pool_ams))

    # 训练集 326 唯一正样本配对
    print("\n=== Step 1.6: 加载训练集 326 唯一正样本配对 ===")
    train_pairs = load_train_pairs("data/processed/v4_train_3d_dimer.csv")
    print("  唯一正样本配对:", len(train_pairs))

    # 笛卡尔积
    print("\n=== Step 1.7: 笛卡尔积生成 ===")
    ald_list = list(pool_alds.values())
    am_list = list(pool_ams.values())
    total = len(ald_list) * len(am_list)
    print("  醛池: {}, 胺池: {}, 理论笛卡尔积: {}".format(
        len(ald_list), len(am_list), total))

    pairs = []
    n_excl_self = 0
    n_excl_train = 0
    for a in ald_list:
        for m in am_list:
            if a["canon"] == m["canon"]:
                n_excl_self += 1
                continue
            if (a["canon"], m["canon"]) in train_pairs:
                n_excl_train += 1
                continue
            pairs.append({
                "aldehyde_smiles": a["smiles"],
                "aldehyde_name": a["name"],
                "ald_n": a["n_ald"],
                "ald_has_f": a["has_f"],
                "ald_source": a["source"],
                "amine_smiles": m["smiles"],
                "amine_name": m["name"],
                "am_n": m["n_am"],
                "am_has_f": m["has_f"],
                "am_source": m["source"],
            })
    print("  排除 (醛==胺):", n_excl_self)
    print("  排除 (训练集配对):", n_excl_train)
    print("  剩余配对:", len(pairs))

    # 写出
    out_fields = [
        "aldehyde_smiles", "aldehyde_name", "ald_n", "ald_has_f", "ald_source",
        "amine_smiles", "amine_name", "am_n", "am_has_f", "am_source",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for p in pairs:
            w.writerow(p)
    print("\n=== 写出: {} ({} 配对) ===".format(out_csv, len(pairs)))


if __name__ == "__main__":
    main()
