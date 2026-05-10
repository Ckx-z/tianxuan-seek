"""路线 A 单体筛选 — 基于 LLM 提取的单体数据。

从 LLM 数据提取唯一单体 → 分组 → 生成配对 → 预测 → Top N
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.predict import MonomerScreener
from src.screening.features import FeatureEngineer
from src.chemistry.monomer import MonomerLibrary
from src.chemistry.imine_check import ImineChecker
from src.chemistry.fluorination import FluorineDetector
from src.utils.logger import setup_logger

logger = setup_logger("screen_llm")


def extract_monomers_from_llm(llm_path: str) -> pd.DataFrame:
    """从 LLM 提取的数据汇总唯一单体及其属性。"""
    with open(llm_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 去重
    seen = {}
    for r in records:
        lid = r.get("literature_id", "")
        if lid and lid not in seen:
            seen[lid] = r
    unique_records = list(seen.values())

    monomer_info = {}
    for rec in unique_records:
        for m in rec.get("monomers", []):
            if not isinstance(m, dict):
                continue
            name = m.get("name", "").strip()
            smi = m.get("canonical_smiles") or m.get("smiles", "")
            if not name or not smi or smi.lower() in ("null", "none", ""):
                continue

            # 按 Canonical SMILES 去重（化学同一性）
            key = smi
            if key not in monomer_info:
                monomer_info[key] = {
                    "name": name,
                    "smiles": smi,
                    "monomer_type": m.get("monomer_type", "other"),
                    "has_fluorine": m.get("has_fluorine", False),
                    "n_papers": 1,
                }
            else:
                monomer_info[key]["n_papers"] += 1
                # 保留出现次数更多的名称
                if len(name) < len(monomer_info[key]["name"]):
                    monomer_info[key]["name"] = name

    df = pd.DataFrame(monomer_info.values())
    if len(df) == 0:
        return df

    # 用 RDKit 补充化学属性
    from rdkit import Chem
    imine_checker = ImineChecker()
    f_detector = FluorineDetector()

    mols = []
    is_ald = []
    is_am = []
    has_f = []
    n_f = []
    has_cf3 = []
    mw = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        mols.append(mol is not None)
        is_ald.append(imine_checker.is_aldehyde(mol) if mol else False)
        is_am.append(imine_checker.is_amine(mol) if mol else False)
        has_f.append(f_detector.has_fluorine(mol) if mol else row["has_fluorine"])
        n_f.append(f_detector.count_fluorine(mol) if mol else 0)
        has_cf3.append(f_detector.has_cf3(mol) if mol else False)
        mw.append(Chem.Descriptors.MolWt(mol) if mol else 0)

    df["valid_mol"] = mols
    df["is_aldehyde"] = is_ald
    df["is_amine"] = is_am
    df["has_fluorine"] = has_f
    df["n_f_atoms"] = n_f
    df["has_cf3"] = has_cf3
    df["mw"] = mw

    # 只保留有效 Mol
    df = df[df["valid_mol"]].copy()
    df.drop(columns=["valid_mol"], inplace=True)

    logger.info(f"LLM 唯一单体: {len(df)} 个")
    return df.sort_values("n_papers", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="路线 A 单体筛选 (LLM 数据)")
    parser.add_argument("--llm", default="data/processed/monomer_smiles_llm.json")
    parser.add_argument("--model-dir", default="models/v1.0")
    parser.add_argument("--cache", default="data/processed/monomer_smiles_cache.json")
    parser.add_argument("--output", default="data/processed/route_a_top20.csv")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.model_dir, "logistic_model.pkl")):
        logger.error(f"模型不存在: {args.model_dir}/")
        sys.exit(1)

    # 1. 从 LLM 数据提取单体
    logger.info("从 LLM 数据提取唯一单体...")
    monomers = extract_monomers_from_llm(args.llm)
    if len(monomers) == 0:
        logger.error("未提取到任何单体")
        sys.exit(1)

    # 2. 分组统计
    n_ald = monomers["is_aldehyde"].sum()
    n_am = monomers["is_amine"].sum()
    n_f_ald = ((monomers["is_aldehyde"]) & (monomers["has_fluorine"])).sum()
    n_f_am = ((monomers["is_amine"]) & (monomers["has_fluorine"])).sum()
    logger.info(f"单体统计: 醛={n_ald}, 胺={n_am}, F-醛={n_f_ald}, F-胺={n_f_am}")

    # 3. 生成配对
    f_ald = monomers[(monomers["is_aldehyde"]) & (monomers["has_fluorine"])]
    nonf_ald = monomers[(monomers["is_aldehyde"]) & (~monomers["has_fluorine"])]
    f_am = monomers[(monomers["is_amine"]) & (monomers["has_fluorine"])]
    nonf_am = monomers[(monomers["is_amine"]) & (~monomers["has_fluorine"])]

    # 策略判断
    n_f_total = n_f_ald + n_f_am
    use_fluorine = n_f_total >= 6 and n_f_ald >= 2 and n_f_am >= 2
    if not use_fluorine:
        logger.warning(f"含氟单体不足(F总计={n_f_total},F-醛={n_f_ald},F-胺={n_f_am})，使用全量自由配对")
        # 全量配对
        aldehydes = monomers[monomers["is_aldehyde"]]
        amines = monomers[monomers["is_amine"]]
        pairs = []
        for _, ald in aldehydes.iterrows():
            for _, am in amines.iterrows():
                pairs.append({
                    "aldehyde": ald["name"],
                    "amine": am["name"],
                    "aldehyde_smiles": ald["smiles"],
                    "amine_smiles": am["smiles"],
                    "aldehyde_f": ald["has_fluorine"],
                    "amine_f": am["has_fluorine"],
                    "pair_type": "all-pairs",
                })
        pairs_df = pd.DataFrame(pairs)
        logger.info(f"全量配对: {len(aldehydes)}×{len(amines)}={len(pairs_df)}")
    else:
        logger.info("启用氟策略配对")
        pair_specs = [
            (f_ald, nonf_am, "F-aldehyde × nonF-amine"),
            (nonf_ald, f_am, "nonF-aldehyde × F-amine"),
            (f_ald, f_am, "F-aldehyde × F-amine"),
            (nonf_ald, nonf_am, "nonF-aldehyde × nonF-amine"),
        ]
        pairs = []
        for ald_df, am_df, pair_type in pair_specs:
            if len(ald_df) == 0 or len(am_df) == 0:
                continue
            for _, ald in ald_df.iterrows():
                for _, am in am_df.iterrows():
                    pairs.append({
                        "aldehyde": ald["name"],
                        "amine": am["name"],
                        "aldehyde_smiles": ald["smiles"],
                        "amine_smiles": am["smiles"],
                        "aldehyde_f": ald["has_fluorine"],
                        "amine_f": am["has_fluorine"],
                        "pair_type": pair_type,
                    })
            logger.info(f"  {pair_type}: {len(ald_df)}×{len(am_df)}={len(ald_df)*len(am_df)}")
        pairs_df = pd.DataFrame(pairs)
        logger.info(f"路线 A 总配对: {len(pairs_df)}")

    if len(pairs_df) == 0:
        logger.error("未生成任何配对")
        sys.exit(1)

    # 4. 特征 + 预测
    logger.info("预测成膜概率...")
    monomer_lib = MonomerLibrary(cache_path=args.cache, use_pubchem=False)
    feature_eng = FeatureEngineer(monomer_lib)

    # 加载模型和预处理
    import pickle
    from rdkit import Chem

    with open(os.path.join(args.model_dir, "logistic_model.pkl"), "rb") as f:
        model = pickle.load(f)
    scaler = None
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    selected_features = None
    info_path = os.path.join(args.model_dir, "model_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            sf = info.get("selected_features", [])
            if sf and len(sf) > 0:
                selected_features = np.array(sf)

    # 预测每对
    probs = []
    for _, row in pairs_df.iterrows():
        ald_mol = Chem.MolFromSmiles(row["aldehyde_smiles"])
        am_mol = Chem.MolFromSmiles(row["amine_smiles"])
        if ald_mol is None or am_mol is None:
            probs.append(float("nan"))
            continue
        try:
            feat = feature_eng.featurize_monomer_pair(ald_mol, am_mol)
            feat = feat.reshape(1, -1)
            if selected_features is not None and len(selected_features) > 0:
                feat = feat[:, selected_features]
            if scaler is not None:
                feat = scaler.transform(feat)
            probs.append(model.predict_proba(feat)[0, 1])
        except Exception:
            probs.append(float("nan"))

    pairs_df["film_probability"] = probs
    valid = pairs_df.dropna(subset=["film_probability"])
    ranked = valid.sort_values("film_probability", ascending=False)
    top = ranked.head(args.top).reset_index(drop=True)

    # 5. 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    top.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nTop {args.top} 已保存至: {args.output}")

    # 打印
    print("\n" + "=" * 75)
    print(f"  Route A Top {min(args.top, len(top))} 单体对 (成膜概率)")
    print("=" * 75)
    header = f"  {'#':3s}  {'概率':8s}  {'醛单体':35s}  {'胺单体':35s}  {'类型':25s}"
    print(header)
    print("  " + "-" * 72)
    for i, (_, row) in enumerate(top.iterrows()):
        print(
            f"  [{i+1:2d}]  {row['film_probability']:.4f}  "
            f"{row['aldehyde'][:33]:33s}  {row['amine'][:33]:33s}  "
            f"{row['pair_type'][:24]:24s}"
        )


if __name__ == "__main__":
    main()
