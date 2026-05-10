"""从 LLM 提取的单体 SMILES 数据构建特征矩阵。

输入: data/processed/monomer_smiles_llm.json
输出: data/processed/X_features.npz
      data/processed/y_labels.npy
      data/processed/feature_names.json
      data/processed/label_metadata.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.features import FeatureEngineer
from src.chemistry.monomer import MonomerLibrary
from src.utils.logger import setup_logger

logger = setup_logger("build_features_llm")


def main():
    parser = argparse.ArgumentParser(description="从 LLM 单体数据构建特征矩阵")
    parser.add_argument("--input", default="data/processed/monomer_smiles_llm.json")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--cache", default="data/processed/monomer_smiles_cache.json")
    args = parser.parse_args()

    # 加载 LLM 数据
    logger.info(f"加载 LLM 数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 去重
    seen = {}
    for r in records:
        lid = r.get("literature_id", "")
        if lid and lid not in seen:
            seen[lid] = r
    unique = list(seen.values())
    logger.info(f"唯一文献: {len(unique)} (原始 {len(records)})")

    # 统计
    total_2d = sum(1 for r in unique if r.get("is_2d_cof") is True)
    film_pos = sum(1 for r in unique if r.get("film_label") is True)
    film_neg = sum(1 for r in unique if r.get("film_label") is False)
    logger.info(
        f"2D COF: {total_2d}, 成膜: {film_pos}, 未成膜: {film_neg}"
    )

    # 初始化特征工程器
    monomer_lib = MonomerLibrary(cache_path=args.cache, use_pubchem=False)
    feature_eng = FeatureEngineer(monomer_lib)

    # 构建特征矩阵（LLM 数据路径）
    logger.info("构建特征矩阵 (LLM 数据路径)...")
    X, y, feature_names, metadata = feature_eng.build_from_llm_data(unique)

    if X.size == 0:
        logger.error("未能生成任何有效样本")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # 保存
    np.savez_compressed(os.path.join(args.output, "X_features.npz"), X=X)
    np.save(os.path.join(args.output, "y_labels.npy"), y)

    with open(os.path.join(args.output, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False)

    meta_df = pd.DataFrame(metadata)
    meta_df["label"] = y
    meta_df["is_film_forming"] = meta_df["label"].map({1: "是", 0: "否"})
    meta_df.to_csv(
        os.path.join(args.output, "label_metadata.csv"),
        index=False, encoding="utf-8-sig",
    )

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    print("\n" + "=" * 60)
    print("  特征矩阵构建完成 (LLM 数据)")
    print("=" * 60)
    print(f"  总样本数:       {len(y)}")
    print(f"  特征维度:       {X.shape[1]}")
    print(f"  正样本(成膜):   {n_pos} ({100 * n_pos / len(y):.1f}%)")
    print(f"  负样本(不成膜): {n_neg} ({100 * n_neg / len(y):.1f}%)")
    print(f"  正负比:         {n_pos / max(n_neg, 1):.2f}:1")
    print(f"  输出目录:       {args.output}/")
    print("=" * 60)

    monomer_lib.flush_cache()


if __name__ == "__main__":
    main()
