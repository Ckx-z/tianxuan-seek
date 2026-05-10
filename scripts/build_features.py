"""构建特征矩阵脚本。

输入: data/fluorofilm.db（954 条文献记录）
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

from src.chemistry.monomer import MonomerLibrary
from src.screening.features import FeatureEngineer
from src.utils.db import get_all_records, init_db
from src.utils.logger import setup_logger

logger = setup_logger("build_features")


def main():
    parser = argparse.ArgumentParser(description="构建 COF 成膜预测特征矩阵")
    parser.add_argument("--db", default="data/fluorofilm.db", help="SQLite 数据库路径")
    parser.add_argument("--output", default="data/processed", help="输出目录")
    parser.add_argument("--cache", default="data/processed/monomer_smiles_cache.json",
                        help="单体 SMILES 缓存路径")
    parser.add_argument("--max-records", type=int, default=0,
                        help="限制处理记录数（0 表示全部）")
    parser.add_argument("--no-pubchem", action="store_true",
                        help="跳过 PubChem API 查询（仅用内置字典）")
    args = parser.parse_args()

    # 加载数据
    logger.info("加载文献数据库...")
    conn = init_db(args.db)
    records = get_all_records(conn)
    conn.close()
    logger.info(f"共 {len(records)} 条文献记录")

    if args.max_records > 0:
        records = records[:args.max_records]
        logger.info(f"限制前 {args.max_records} 条")

    # 初始化
    logger.info("初始化 MonomerLibrary（首次运行需查询 PubChem，较慢）...")
    monomer_lib = MonomerLibrary(cache_path=args.cache,
                                  use_pubchem=not args.no_pubchem)
    feature_eng = FeatureEngineer(monomer_lib)

    # 构建特征矩阵
    logger.info("开始构建特征矩阵...")
    X, y, feature_names, metadata = feature_eng.build_feature_matrix(records)

    if X.size == 0:
        logger.error("未能生成任何有效样本，请检查数据质量")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # 保存特征矩阵
    np.savez_compressed(os.path.join(args.output, "X_features.npz"), X=X)
    np.save(os.path.join(args.output, "y_labels.npy"), y)

    # 保存特征名称
    with open(os.path.join(args.output, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False)

    # 保存元数据
    meta_df = pd.DataFrame(metadata)
    meta_df["label"] = y
    meta_df["is_film_forming"] = meta_df["label"].map({1: "是", 0: "否"})
    meta_df.to_csv(os.path.join(args.output, "label_metadata.csv"), index=False, encoding="utf-8-sig")

    # 统计
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    ratio = n_pos / max(n_neg, 1)

    print("\n" + "=" * 60)
    print("  特征矩阵构建完成")
    print("=" * 60)
    print(f"  总样本数:       {len(y)}")
    print(f"  特征维度:       {X.shape[1]}")
    print(f"  正样本(成膜):   {n_pos} ({100*n_pos/len(y):.1f}%)")
    print(f"  负样本(不成膜): {n_neg} ({100*n_neg/len(y):.1f}%)")
    print(f"  正负比:         {ratio:.2f}:1")
    print(f"  输出目录:       {args.output}/")
    print("=" * 60)

    monomer_lib.flush_cache()


if __name__ == "__main__":
    main()
