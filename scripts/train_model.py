"""训练成膜预测模型脚本。

标准 ML 流程:
  加载特征矩阵 → 数据分割 → 特征预处理 → 超参调优 → 训练 → CV → 评估 → 保存
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.train import ModelTrainer, N_FOLDS, RANDOM_STATE
from src.utils.logger import setup_logger

logger = setup_logger("train_model")


def main():
    parser = argparse.ArgumentParser(description="训练 COF 成膜预测模型")
    parser.add_argument("--features", default="data/processed/X_features.npz")
    parser.add_argument("--labels", default="data/processed/y_labels.npy")
    parser.add_argument("--feature-names", default="data/processed/feature_names.json")
    parser.add_argument("--model-dir", default="models/v1.0")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--no-tune", action="store_true",
                        help="跳过超参调优（样本少时使用）")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    # 加载数据
    logger.info("加载特征矩阵...")
    data = np.load(args.features)
    X = data["X"]
    y = np.load(args.labels)

    logger.info(f"X.shape={X.shape}, y.shape={y.shape}")
    logger.info(f"正样本: {int(y.sum())}, 负样本: {int(len(y) - y.sum())}, "
                f"正负比: {y.sum() / max(len(y) - y.sum(), 1):.2f}:1")

    # 加载特征名称
    feature_names = []
    if os.path.exists(args.feature_names):
        with open(args.feature_names, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

    # 初始化训练器
    trainer = ModelTrainer(random_state=args.seed)
    trainer.feature_names = feature_names

    # 数据分割
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        X, y, test_size=args.test_size,
    )

    # 特征预处理（仅在训练集上 fit）
    X_train, X_test = trainer.preprocess_features(
        X_train, X_test,
        variance_threshold=0.0,
        correlation_threshold=0.95,
    )

    # 训练
    logger.info("训练模型...")
    trainer.train_all(X_train, y_train, tune=not args.no_tune)

    # 交叉验证 (在全部训练数据上)
    logger.info(f"{N_FOLDS} 折交叉验证...")
    cv_df = trainer.cross_validate_all(X_train, y_train)
    print("\n" + "=" * 65)
    print("  交叉验证结果 (PR-AUC 为主指标)")
    print("=" * 65)
    print(cv_df.to_string(index=False))

    # 测试集评估
    logger.info("测试集评估...")
    test_df = trainer.evaluate_all(X_test, y_test)
    print("\n" + "=" * 65)
    print("  测试集评估结果")
    print("=" * 65)
    print(test_df.to_string(index=False))

    # 特征重要性
    print("\n" + "=" * 65)
    print("  XGBoost 特征重要性 Top 15")
    print("=" * 65)
    fi_df = trainer.get_feature_importance("xgboost", top_n=15)
    for _, row in fi_df.iterrows():
        print(f"  {row['feature'][:55]:55s}  {row['importance']:.4f}")

    # 学习曲线
    os.makedirs(args.model_dir, exist_ok=True)
    lc_path = os.path.join(args.model_dir, "learning_curve.png")
    trainer.plot_learning_curve(
        X_train, y_train, model_name="xgboost", output_path=lc_path,
    )

    # 特征重要性图
    fi_path = os.path.join(args.model_dir, "feature_importance.png")
    trainer.plot_feature_importance(
        model_name="xgboost", top_n=30, output_path=fi_path,
    )

    # 保存模型
    trainer.save_models(args.model_dir)
    print(f"\n模型已保存至: {args.model_dir}/")


if __name__ == "__main__":
    main()
