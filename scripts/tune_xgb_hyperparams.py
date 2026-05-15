"""XGBoost 超参搜索 + 剪枝验证 — 687 清洁训练集 (G1+G1u+G3)。

搜索维度: max_depth, gamma, min_child_weight, subsample,
         colsample_bytree, learning_rate, n_estimators
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import randint, uniform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn import (MoleculeEncoder, extract_monomer_embeddings)
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("xgb_tune")

HIDDEN = 256


def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else smi


def main():
    # ── 加载数据 ──
    meta = pd.read_csv("data/processed/label_metadata_v3.csv", encoding="utf-8-sig")
    with open("data/processed/benchmark_pairs.json", encoding="utf-8") as f:
        bench = json.load(f)
    bench_pairs = set()
    for lk in ["positive", "negative"]:
        for p in bench[lk]:
            bench_pairs.add((_canon(p["ald_smi"]), _canon(p["am_smi"])))

    train_ald, train_am, train_y = [], [], []
    test_ald, test_am, test_y = [], [], []
    for _, row in meta.iterrows():
        a = _canon(str(row["aldehyde_smiles"]))
        b = _canon(str(row["amine_smiles"]))
        if not a or not b:
            continue
        if (a, b) in bench_pairs:
            continue
        src = str(row.get("source", ""))
        lbl = int(row["label"])
        if src == "group2":
            test_ald.append(a); test_am.append(b); test_y.append(lbl)
        else:
            train_ald.append(a); train_am.append(b); train_y.append(lbl)

    logger.info(f"训练集: {len(train_ald)} (正={sum(train_y)}, {sum(train_y)/len(train_y)*100:.1f}%)")
    logger.info(f"测试集 (G2): {len(test_ald)} (正={sum(test_y)}, {sum(test_y)/len(test_y)*100:.1f}%)")

    # ── GNN 嵌入 ──
    encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
    encoder.load_state_dict(torch.load(
        "models/v2.0/gnn_encoder_finetuned.pt", map_location="cpu"))
    encoder.eval()

    all_smis = set(train_ald + train_am + test_ald + test_am)
    emb_arr = extract_monomer_embeddings(encoder, list(all_smis))
    emb_dict = {s: emb_arr[i] for i, s in enumerate(all_smis)}

    def pair_feat(ald_s, am_s):
        X = []
        for a, b in zip(ald_s, am_s):
            ea = emb_dict.get(a, np.zeros(HIDDEN, dtype=np.float32))
            eb = emb_dict.get(b, np.zeros(HIDDEN, dtype=np.float32))
            X.append(np.concatenate([ea, eb, ea - eb, ea * eb]))
        return np.array(X, dtype=np.float32)

    X_tr = pair_feat(train_ald, train_am)
    y_tr = np.array(train_y)
    X_te = pair_feat(test_ald, test_am)
    y_te = np.array(test_y)

    # ── 基线 ──
    import xgboost as xgb
    sw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    baseline = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=sw, eval_metric="aucpr", random_state=42)
    baseline.fit(X_tr, y_tr)
    base_prob = baseline.predict_proba(X_te)[:, 1]
    base_pr = average_precision_score(y_te, base_prob)
    base_roc = roc_auc_score(y_te, base_prob)
    logger.info(f"基线 (max_depth=5, lr=0.05, n=200): "
                f"Test PR-AUC={base_pr:.4f}, ROC-AUC={base_roc:.4f}")

    # ── RandomizedSearchCV ──
    param_dist = {
        "max_depth": randint(3, 11),           # 3~10
        "gamma": uniform(0, 1.5),              # 0~1.5
        "min_child_weight": randint(1, 10),    # 1~9
        "subsample": uniform(0.6, 0.4),        # 0.6~1.0
        "colsample_bytree": uniform(0.5, 0.5), # 0.5~1.0
        "learning_rate": uniform(0.01, 0.2),   # 0.01~0.21
        "n_estimators": randint(100, 600),     # 100~599
        "reg_alpha": uniform(0, 1),            # L1 正则
        "reg_lambda": uniform(0.5, 2),         # L2 正则
    }
    model = xgb.XGBClassifier(
        scale_pos_weight=sw, eval_metric="aucpr",
        random_state=42, n_jobs=1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        model, param_dist, n_iter=100, cv=cv,
        scoring="average_precision", random_state=42,
        verbose=1, n_jobs=1)
    search.fit(X_tr, y_tr)

    # ── 结果 ──
    print("\n" + "=" * 72)
    print("  RandomizedSearchCV 超参调优结果")
    print("=" * 72)
    print(f"\n  最佳 PR-AUC (CV): {search.best_score_:.4f}")
    print(f"\n  最佳参数:")
    for k, v in sorted(search.best_params_.items()):
        print(f"    {k:25s} = {v}")

    # 评估最佳模型
    best_model = search.best_estimator_
    test_prob = best_model.predict_proba(X_te)[:, 1]
    test_hard = (test_prob >= 0.5).astype(int)

    print(f"\n  Group2 测试集 (n={len(y_te)}):")
    print(f"    PR-AUC:  {average_precision_score(y_te, test_prob):.4f}")
    print(f"    ROC-AUC: {roc_auc_score(y_te, test_prob):.4f}")
    print(f"    F1:      {f1_score(y_te, test_hard, zero_division=0):.4f}")

    # ── 关键: gamma 剪枝效应分析 ──
    print(f"\n  --- 剪枝(gamma)效应分析 ---")
    gamma_best = search.best_params_.get("gamma", 0)
    print(f"  最佳 gamma: {gamma_best:.4f}")

    # 固定其他参数，测试不同 gamma
    fix_params = {k: v for k, v in search.best_params_.items()}
    print(f"\n  {'gamma':8s} {'CV PR-AUC':12s} {'树平均叶子数':12s}")
    print(f"  {'-' * 40}")
    for g in [0, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0]:
        p = {**fix_params, "gamma": g}
        m = xgb.XGBClassifier(
            scale_pos_weight=sw, eval_metric="aucpr",
            random_state=42, **p)
        pr_scores, leaf_counts = [], []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, va_idx in skf.split(np.zeros(len(y_tr)), y_tr):
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            prob = m.predict_proba(X_tr[va_idx])[:, 1]
            pr_scores.append(average_precision_score(y_tr[va_idx], prob))
            # 计算平均叶子数
            leaves = []
            booster = m.get_booster()
            dump = booster.get_dump(dump_format="json")
            for tree_json in dump:
                tree = json.loads(tree_json)
                leaves.append(_count_leaves(tree))
            leaf_counts.append(np.mean(leaves))
        print(f"  {g:<8.3f} {np.mean(pr_scores):.4f} +/- {np.std(pr_scores):.4f}    {np.mean(leaf_counts):.1f}")

    # ── 对比总结 ──
    # 用 best params 做 5-fold CV 看训练集表现
    print(f"\n  --- 最终 5 折 CV (最佳参数) ---")
    pr_cv, roc_cv = [], []
    for tr_idx, va_idx in cv.split(np.zeros(len(y_tr)), y_tr):
        m = xgb.XGBClassifier(
            scale_pos_weight=(len(y_tr[tr_idx]) - y_tr[tr_idx].sum()) / max(y_tr[tr_idx].sum(), 1),
            eval_metric="aucpr", random_state=42, **search.best_params_)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        prob = m.predict_proba(X_tr[va_idx])[:, 1]
        pr_cv.append(average_precision_score(y_tr[va_idx], prob))
        roc_cv.append(roc_auc_score(y_tr[va_idx], prob))
    print(f"    PR-AUC:  {np.mean(pr_cv):.4f} +/- {np.std(pr_cv):.4f}")
    print(f"    ROC-AUC: {np.mean(roc_cv):.4f} +/- {np.std(roc_cv):.4f}")

    print(f"\n  对比:")
    print(f"    基线 (默认参数): PR-AUC={base_pr:.4f}")
    print(f"    调优后:          PR-AUC={np.mean(pr_cv):.4f}")
    print(f"    增益:            {np.mean(pr_cv) - base_pr:+.4f}")

    # 保存最佳参数
    best_params_path = "models/v2.0/xgb_best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, bool, type(None)))
                    else v for k, v in search.best_params_.items()},
                  f, indent=2, ensure_ascii=False)
    logger.info(f"最佳参数已保存: {best_params_path}")


def _count_leaves(tree):
    """递归统计一棵树的叶子节点数。"""
    if "leaf" in tree:
        return 1.0
    count = 0
    for child in tree.get("children", []):
        count += _count_leaves(child)
    return count if count > 0 else 1.0


if __name__ == "__main__":
    main()
