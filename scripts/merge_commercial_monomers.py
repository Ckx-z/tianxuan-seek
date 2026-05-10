"""将商业单体融入 2D COF 筛选管道。

读取 extract_commercial_monomers.py 的输出 CSV，用 ImineChecker 分类，
然后与 LLM 提取的 2D 单体合并，重新运行完整筛选流程。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.imine_check import ImineChecker
from src.chemistry.fluorination import FluorineDetector
from src.utils.logger import setup_logger

logger = setup_logger("merge_commercial")


def classify_commercial_monomers(csv_path: str) -> pd.DataFrame:
    """读取商业单体 CSV，用 ImineChecker 分类为 aldehyde/amine。"""
    from rdkit import Chem

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    logger.info(f"读取商业单体: {len(df)} 条")

    imine = ImineChecker()
    f_detector = FluorineDetector()

    rows = []
    for _, r in df.iterrows():
        smi = r.get("canonical_smiles") or r.get("smiles")
        if not smi or pd.isna(smi):
            continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        try:
            can_smi = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            can_smi = str(smi)

        n_ald = imine.count_aldehyde_groups(mol)
        n_am = imine.count_amine_groups(mol)

        # 单分类: aldehyde / amine / aldehyde-amine / other
        if n_ald >= 1 and n_am >= 1:
            mtype = "aldehyde-amine"
        elif n_ald >= 1:
            mtype = "aldehyde"
        elif n_am >= 1:
            mtype = "amine"
        else:
            mtype = "other"

        rows.append({
            "name": str(r.get("name", "?")),
            "smiles": can_smi,
            "monomer_type": mtype,
            "has_fluorine": f_detector.has_fluorine(mol),
            "n_f_atoms": f_detector.count_fluorine(mol),
            "has_cf3": f_detector.has_cf3(mol),
            "n_aldehyde": n_ald,
            "n_amine": n_am,
            "source": "commercial",
            "commercial_id": str(r.get("id", "")),
            "cas": str(r.get("cas", "")),
            "formula": str(r.get("formula", "")),
        })

    result = pd.DataFrame(rows)
    logger.info(
        f"分类: 醛 {int((result['monomer_type'] == 'aldehyde').sum())}, "
        f"胺 {int((result['monomer_type'] == 'amine').sum())}, "
        f"双功能 {int((result['monomer_type'] == 'aldehyde-amine').sum())}, "
        f"其他 {int((result['monomer_type'] == 'other').sum())}"
    )
    return result


def merge_with_llm_monomers(
    commercial_df: pd.DataFrame,
    llm_json_path: str,
    output_path: str,
) -> pd.DataFrame:
    """将商业单体与 LLM 提取的 2D 单体合并，按 Canonical SMILES 去重。"""
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey

    with open(llm_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 去重文献
    seen_lid = {}
    for r in records:
        lid = r.get("literature_id", "")
        if lid and lid not in seen_lid:
            seen_lid[lid] = r
    unique_records = list(seen_lid.values())

    # 从 LLM 数据提取单体 (同 extract_2d_monomers 逻辑)
    imine_checker = ImineChecker()
    f_detector = FluorineDetector()

    smi_info = {}
    for rec in unique_records:
        for m in rec.get("monomers", []):
            if not isinstance(m, dict):
                continue
            name = m.get("name", "").strip()
            smi = m.get("canonical_smiles") or m.get("smiles", "")
            if not name or not smi or str(smi).lower() in ("null", "none", ""):
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
                    "best_name": name,
                    "monomer_type": mtype,
                    "has_fluorine": f_detector.has_fluorine(mol),
                    "n_f_atoms": f_detector.count_fluorine(mol),
                    "has_cf3": f_detector.has_cf3(mol),
                    "n_aldehyde": n_ald,
                    "n_amine": n_am,
                    "n_papers": 1,
                    "source": "llm",
                }
            else:
                smi_info[can_smi]["n_papers"] += 1
                if len(name) > len(smi_info[can_smi]["best_name"]):
                    smi_info[can_smi]["best_name"] = name

    llm_df = pd.DataFrame(smi_info.values())
    logger.info(f"LLM 单体: {len(llm_df)} (去重)")

    # 合并商业单体 — 用 InChI Key 去重
    existing_inchi = {}
    for smi in llm_df["smiles"]:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                existing_inchi[MolToInchiKey(mol)] = smi
        except Exception:
            pass

    new_commercial = []
    dup_count = 0
    for _, row in commercial_df.iterrows():
        smi = row["smiles"]
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            inchi_key = MolToInchiKey(mol)
        except Exception:
            inchi_key = smi

        if inchi_key in existing_inchi:
            dup_count += 1
            continue

        existing_inchi[inchi_key] = smi
        new_commercial.append({
            "smiles": smi,
            "best_name": row["name"],
            "monomer_type": row["monomer_type"],
            "has_fluorine": row["has_fluorine"],
            "n_f_atoms": row["n_f_atoms"],
            "has_cf3": row["has_cf3"],
            "n_aldehyde": row["n_aldehyde"],
            "n_amine": row["n_amine"],
            "n_papers": 0,
            "source": "commercial",
            "commercial_id": row.get("commercial_id", ""),
            "cas": row.get("cas", ""),
            "formula": row.get("formula", ""),
        })

    logger.info(f"商业单体新增: {len(new_commercial)}, 重复 (InChI): {dup_count}")

    merged = pd.concat([llm_df, pd.DataFrame(new_commercial)], ignore_index=True)
    logger.info(f"合并后总单体: {len(merged)}")

    # 筛选 2D 可用 (≥2 官能团)
    merged["is_aldehyde"] = (merged["n_aldehyde"] >= 2) & (merged["monomer_type"].isin(["aldehyde", "aldehyde-amine"]))
    merged["is_amine"] = (merged["n_amine"] >= 2) & (merged["monomer_type"].isin(["amine", "aldehyde-amine"]))
    merged["is_2d"] = merged["is_aldehyde"] | merged["is_amine"]

    n_2d = merged["is_2d"].sum()
    n_ald_2d = merged["is_aldehyde"].sum()
    n_am_2d = merged["is_amine"].sum()
    logger.info(f"2D 可用: {n_2d} (醛 {n_ald_2d}, 胺 {n_am_2d})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"合并单体池已保存: {output_path}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="合并商业单体到筛选管道")
    parser.add_argument("--commercial", default="data/processed/commercial_monomers.csv")
    parser.add_argument("--llm", default="data/processed/monomer_smiles_llm.json")
    parser.add_argument("--output", default="data/processed/merged_monomer_pool.csv")
    args = parser.parse_args()

    # 1. 分类商业单体
    commercial_df = classify_commercial_monomers(args.commercial)
    commercial_classified = args.commercial.replace(".csv", "_classified.csv")
    commercial_df.to_csv(commercial_classified, index=False, encoding="utf-8-sig")
    logger.info(f"分类后商业单体已保存: {commercial_classified}")

    # 2. 合并
    merged = merge_with_llm_monomers(commercial_df, args.llm, args.output)

    # 3. 统计
    print("\n" + "=" * 60)
    print("  商业单体合并完成")
    print("=" * 60)
    print(f"  总单体:        {len(merged)}")
    print(f"  其中 LLM:      {int((merged['source'] == 'llm').sum())}")
    print(f"  其中 商业:     {int((merged['source'] == 'commercial').sum())}")
    print(f"  2D 可用:       {int(merged['is_2d'].sum())}")
    print(f"  2D 醛:         {int(merged['is_aldehyde'].sum())}")
    print(f"  2D 胺:         {int(merged['is_amine'].sum())}")
    print(f"  含氟:          {int(merged['has_fluorine'].sum())}")
    print(f"  输出:          {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
