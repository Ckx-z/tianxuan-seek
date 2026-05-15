"""v4 标签 + 512-dim 简化模型 + 超参调优。

步骤:
  1. 应用 v4 标签规则
  2. 512-dim GNN 嵌入 (ald+am only)
  3. RandomizedSearchCV 调优 XGBoost
  4. v3 vs v4 对比 (各自最佳参数)
"""
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from rdkit import Chem
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import randint, uniform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn import MoleculeEncoder, extract_monomer_embeddings
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("v4_tune")

HIDDEN = 256


def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else smi


def label_v4(film_field) -> int:
    """标签规则 v4: 扩增否定模式。"""
    if pd.isna(film_field) or not str(film_field).strip():
        return 0
    ff = str(film_field).strip()

    # 否定层 (扩增)
    neg_exact = [
        "未成膜", "不成膜", "未提及成膜", "未制备成膜", "不能成膜",
        "未涉及成膜", "未涉及薄膜制备", "而不成膜", "未涉及膜制备",
        "而非薄膜", "而非成膜", "未成薄膜", "非成膜材料",
        "未提及是否成膜", "未讨论成膜", "未涉及膜性能",
        "不涉及成膜", "不涉及薄膜", "不涉及膜",
        "无成膜", "没有成膜", "无薄膜", "无膜形成",
        "不成薄膜", "难以成膜", "无法成膜",
    ]
    for p in neg_exact:
        try:
            if re.search(p, ff):
                return 0
        except re.error:
            if p in ff:
                return 0

    # L1 高置信
    l1_verbs = [
        "制备了", "形成了", "获得了", "得到了", "自支撑膜",
        "free-standing", "free standing", "self-standing",
        "self-supporting", "成功制备", "制备出",
        "制得.*膜", "成功制得", "成功合成.*膜",
    ]
    l1_nouns = ["膜", "薄膜", "film", "membrane"]
    has_verb = any(
        (re.search(v, ff) if any(c in v for c in ".*+()[]") else v in ff)
        for v in l1_verbs
    )
    has_noun = any(n in ff for n in l1_nouns)
    if has_verb and has_noun:
        return 1

    # L2 中置信
    l2 = ["成膜", "薄膜", "film formation", "membrane formation", "film", "membrane"]
    if any(kw in ff for kw in l2):
        return 1

    return 0


def build_512_features(emb_dict, ald_s, am_s):
    X = []
    for a, b in zip(ald_s, am_s):
        ea = emb_dict.get(a, np.zeros(HIDDEN, dtype=np.float32))
        eb = emb_dict.get(b, np.zeros(HIDDEN, dtype=np.float32))
        X.append(np.concatenate([ea, eb]))
    return np.array(X, dtype=np.float32)


def main():
    # ── 加载数据 ──
    meta = pd.read_csv("data/processed/label_metadata_v3.csv", encoding="utf-8-sig")
    with open("data/processed/benchmark_pairs.json", encoding="utf-8") as f:
        bench = json.load(f)
    bench_pairs = set()
    for lk in ["positive", "negative"]:
        for p in bench[lk]:
            bench_pairs.add((_canon(p["ald_smi"]), _canon(p["am_smi"])))

    # v4 标签
    v4_labels = np.array([label_v4(row["film_field"]) for _, row in meta.iterrows()])
    v3_labels = meta["label"].values
    logger.info(f"v4 翻转: {(v4_labels != v3_labels).sum()} 个 "
                f"({(v3_labels==1)&(v4_labels==0)}正→负, {(v3_labels==0)&(v4_labels==1)}负→正)")

    # 构建训练集 (v4标签)
    train_ald, train_am, train_y_v4, train_y_v3 = [], [], [], []
    for i, (_, row) in enumerate(meta.iterrows()):
        a = _canon(str(row["aldehyde_smiles"]))
        b = _canon(str(row["amine_smiles"]))
        if not a or not b:
            continue
        if (a, b) in bench_pairs:
            continue
        if str(row.get("source", "")) == "group2":
            continue
        train_ald.append(a)
        train_am.append(b)
        train_y_v4.append(int(v4_labels[i]))
        train_y_v3.append(int(v3_labels[i]))

    train_y_v4 = np.array(train_y_v4)
    train_y_v3 = np.array(train_y_v3)
    logger.info(f"训练集: {len(train_y_v4)} 样本")
    logger.info(f"  v3: {train_y_v3.sum()}+/{len(train_y_v3)-train_y_v3.sum()}- "
                f"({train_y_v3.mean()*100:.1f}%正)")
    logger.info(f"  v4: {train_y_v4.sum()}+/{len(train_y_v4)-train_y_v4.sum()}- "
                f"({train_y_v4.mean()*100:.1f}%正)")

    # GNN 嵌入
    encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
    encoder.load_state_dict(torch.load(
        "models/v2.0/gnn_encoder_finetuned.pt", map_location="cpu"))
    encoder.eval()

    all_smis = set(train_ald + train_am)
    emb_arr = extract_monomer_embeddings(encoder, list(all_smis))
    emb_dict = {s: emb_arr[i] for i, s in enumerate(all_smis)}

    X = build_512_features(emb_dict, train_ald, train_am)

    # ════════════════════════════════════════════════════════
    # 超参调优 (v4 标签, 512-dim)
    # ════════════════════════════════════════════════════════
    param_dist = {
        "max_depth": randint(3, 9),
        "gamma": uniform(0, 2.0),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "learning_rate": uniform(0.01, 0.25),
        "n_estimators": randint(100, 600),
        "reg_alpha": uniform(0, 1.5),
        "reg_lambda": uniform(0.5, 3),
    }
    sw_v4 = (len(train_y_v4) - train_y_v4.sum()) / max(train_y_v4.sum(), 1)
    base_model = xgb.XGBClassifier(
        scale_pos_weight=sw_v4, eval_metric="aucpr", random_state=42, n_jobs=1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base_model, param_dist, n_iter=80, cv=cv,
        scoring="average_precision", random_state=42,
        verbose=2, n_jobs=1)
    search.fit(X, train_y_v4)

    logger.info(f"v4 最佳 CV PR-AUC: {search.best_score_:.4f}")
    logger.info(f"v4 最佳参数:")
    for k, v in sorted(search.best_params_.items()):
        logger.info(f"  {k}: {v}")

    # ════════════════════════════════════════════════════════
    # 对比: v3 (旧参数) vs v3 (调优) vs v4 (旧参数) vs v4 (调优)
    # ════════════════════════════════════════════════════════
    old_params = {
        "colsample_bytree": 0.7929, "gamma": 1.41, "learning_rate": 0.125,
        "max_depth": 4, "min_child_weight": 3, "n_estimators": 479,
        "reg_alpha": 0.1285, "reg_lambda": 2.12, "subsample": 0.9283,
    }
    best_params_v4 = search.best_params_

    configs = [
        ("v3 标签 + 旧参数", train_y_v3, old_params),
        ("v4 标签 + 旧参数", train_y_v4, old_params),
        ("v4 标签 + 调优参数", train_y_v4, best_params_v4),
    ]

    print("\n" + "=" * 70)
    print("  标签修正 + 超参调优 5 折 CV 对比 (512-dim ald+am)")
    print("=" * 70)
    print(f"\n  {'配置':25s} {'PR-AUC':12s} {'ROC-AUC':12s} {'F1':10s}")
    print(f"  {'-' * 65}")

    for name, y_use, params in configs:
        pr_scores, roc_scores, f1_scores = [], [], []
        for tr_idx, va_idx in cv.split(np.zeros(len(y_use)), y_use):
            sw = (len(y_use[tr_idx]) - y_use[tr_idx].sum()) / max(y_use[tr_idx].sum(), 1)
            m = xgb.XGBClassifier(
                scale_pos_weight=sw, eval_metric="aucpr", random_state=42, **params)
            m.fit(X[tr_idx], y_use[tr_idx])
            prob = m.predict_proba(X[va_idx])[:, 1]
            hard = (prob >= 0.5).astype(int)
            pr_scores.append(average_precision_score(y_use[va_idx], prob))
            roc_scores.append(roc_auc_score(y_use[va_idx], prob))
            f1_scores.append(f1_score(y_use[va_idx], hard, zero_division=0))
        print(f"  {name:25s} {np.mean(pr_scores):.4f} +/- {np.std(pr_scores):.4f}  "
              f"{np.mean(roc_scores):.4f} +/- {np.std(roc_scores):.4f}  "
              f"{np.mean(f1_scores):.4f}")

    # 保存 v4 最佳参数
    out_params = {k: float(v) if isinstance(v, (np.floating,)) else v
                  for k, v in best_params_v4.items()}
    os.makedirs("models/v2.0", exist_ok=True)
    with open("models/v2.0/xgb_best_params_v4.json", "w", encoding="utf-8") as f:
        json.dump(out_params, f, indent=2, ensure_ascii=False)

    # 保存 v4 标签
    meta_v4 = meta.copy()
    meta_v4["label"] = v4_labels
    meta_v4["label_version"] = "v4"
    meta_v4.to_csv("data/processed/label_metadata_v4.csv", index=False, encoding="utf-8-sig")
    logger.info("v4 标签和参数已保存")


if __name__ == "__main__":
    main()
