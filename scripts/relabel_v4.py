"""标签规则 v4: 扩增否定模式，解决 ~143 个误标正样本。

与 v3 的关键差异:
  - 新增否定词: 未涉及成膜/未涉及薄膜/而非薄膜/未提及是否成膜
  - 更精确的正则: 未涉及.*(薄膜|成膜)
  - L1/L2 逻辑不变
"""
import json
import os
import re
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn import MoleculeEncoder, extract_monomer_embeddings
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("relabel_v4")

HIDDEN = 256


def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else smi


def label_v4(film_field) -> int:
    """标签规则 v4: 扩增否定模式。

    优先级:
      1. 空 → 0
      2. 明确否定 (未成膜/不成膜/未涉及成膜/而非薄膜/未提及是否成膜...) → 0
      3. L1 高置信: 制备动词 + 膜共现 → 1
      4. L2 中置信: 成膜/薄膜/film/membrane → 1 (否定已排除)
      5. 其余 → 0
    """
    if pd.isna(film_field) or not str(film_field).strip():
        return 0
    ff = str(film_field).strip()

    # ── 否定层 (扩增) ──
    neg_exact = [
        "未成膜", "不成膜", "未提及成膜", "未制备成膜", "不能成膜",
        "未涉及成膜", "未涉及薄膜制备", "而不成膜", "未涉及膜制备",
        "而非薄膜", "而非成膜", "未成薄膜", "非成膜材料",
        "未提及是否成膜", "未讨论成膜", "未涉及膜性能",
        "不涉及成膜", "不涉及薄膜", "不涉及膜",
        "无成膜", "没有成膜", "无薄膜", "无膜形成",
        "水凝胶.*非薄膜", "凝胶.*非膜",
        "不成薄膜", "难以成膜", "无法成膜",
    ]
    for p in neg_exact:
        try:
            if re.search(p, ff):
                return 0
        except re.error:
            if p in ff:
                return 0

    # ── L1 高置信: 制备动词 + 膜共现 ──
    l1_verbs = [
        "制备了", "形成了", "获得了", "得到了", "自支撑膜",
        "free-standing", "free standing", "self-standing",
        "self-supporting", "成功制备", "制备出", "合成了.*膜",
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

    # ── L2 中置信 ──
    l2 = ["成膜", "薄膜", "film formation", "membrane formation", "film", "membrane"]
    if any(kw in ff for kw in l2):
        return 1

    return 0


def main():
    # ── 加载数据 ──
    meta = pd.read_csv("data/processed/label_metadata_v3.csv", encoding="utf-8-sig")
    with open("data/processed/benchmark_pairs.json", encoding="utf-8") as f:
        bench = json.load(f)
    bench_pairs = set()
    for lk in ["positive", "negative"]:
        for p in bench[lk]:
            bench_pairs.add((_canon(p["ald_smi"]), _canon(p["am_smi"])))

    # 应用 v4 标签
    v4_labels = np.array([label_v4(row["film_field"]) for _, row in meta.iterrows()])

    # 统计变化
    old_labels = meta["label"].values
    flips = v4_labels != old_labels
    flip_pos_to_neg = ((old_labels == 1) & (v4_labels == 0)).sum()
    flip_neg_to_pos = ((old_labels == 0) & (v4_labels == 1)).sum()
    logger.info(f"全量标签翻转: {flips.sum()}/{len(meta)} ({flip_pos_to_neg}正→负, {flip_neg_to_pos}负→正)")

    # ── 构建训练集 (v4 标签) ──
    train_ald, train_am, train_y = [], [], []
    old_train_y = []
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
        train_y.append(int(v4_labels[i]))
        old_train_y.append(int(old_labels[i]))

    train_y = np.array(train_y)
    old_train_y = np.array(old_train_y)

    pos_rate = train_y.mean()
    logger.info(f"训练集 (v4): {len(train_y)} 样本, 正={train_y.sum()}, {pos_rate*100:.1f}%")
    logger.info(f"  v3→v4 翻转: {(train_y != old_train_y).sum()} 个")

    # ── GNN 嵌入 ──
    encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
    encoder.load_state_dict(torch.load(
        "models/v2.0/gnn_encoder_finetuned.pt", map_location="cpu"))
    encoder.eval()

    all_smis = set(train_ald + train_am)
    emb_arr = extract_monomer_embeddings(encoder, list(all_smis))
    emb_dict = {s: emb_arr[i] for i, s in enumerate(all_smis)}

    # 512-dim: ald+am only
    X = []
    for a, b in zip(train_ald, train_am):
        ea = emb_dict.get(a, np.zeros(HIDDEN, dtype=np.float32))
        eb = emb_dict.get(b, np.zeros(HIDDEN, dtype=np.float32))
        X.append(np.concatenate([ea, eb]))
    X = np.array(X, dtype=np.float32)

    # ── 5 折 CV 对比: v3 vs v4 标签 ──
    best_params = {
        "colsample_bytree": 0.7929, "gamma": 1.41, "learning_rate": 0.125,
        "max_depth": 4, "min_child_weight": 3, "n_estimators": 479,
        "reg_alpha": 0.1285, "reg_lambda": 2.12, "subsample": 0.9283,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # v3 baseline
    v3_pr, v3_roc = [], []
    for tr_idx, va_idx in cv.split(np.zeros(len(old_train_y)), old_train_y):
        m = xgb.XGBClassifier(
            scale_pos_weight=(len(old_train_y[tr_idx]) - old_train_y[tr_idx].sum()) / max(old_train_y[tr_idx].sum(), 1),
            eval_metric="aucpr", random_state=42, **best_params)
        m.fit(X[tr_idx], old_train_y[tr_idx])
        prob = m.predict_proba(X[va_idx])[:, 1]
        v3_pr.append(average_precision_score(old_train_y[va_idx], prob))
        v3_roc.append(roc_auc_score(old_train_y[va_idx], prob))

    # v4 labels
    v4_pr, v4_roc = [], []
    for tr_idx, va_idx in cv.split(np.zeros(len(train_y)), train_y):
        m = xgb.XGBClassifier(
            scale_pos_weight=(len(train_y[tr_idx]) - train_y[tr_idx].sum()) / max(train_y[tr_idx].sum(), 1),
            eval_metric="aucpr", random_state=42, **best_params)
        m.fit(X[tr_idx], train_y[tr_idx])
        prob = m.predict_proba(X[va_idx])[:, 1]
        v4_pr.append(average_precision_score(train_y[va_idx], prob))
        v4_roc.append(roc_auc_score(train_y[va_idx], prob))

    print("\n" + "=" * 65)
    print("  标签规则 v3 → v4: CV 对比 (512-dim ald+am)")
    print("=" * 65)
    print(f"\n  {'指标':15s} {'v3 标签':22s} {'v4 标签':22s} {'Δ':10s}")
    print(f"  {'-' * 65}")
    print(f"  {'PR-AUC':15s} {np.mean(v3_pr):.4f} +/- {np.std(v3_pr):.4f}   "
          f"{np.mean(v4_pr):.4f} +/- {np.std(v4_pr):.4f}   {np.mean(v4_pr)-np.mean(v3_pr):+.4f}")
    print(f"  {'ROC-AUC':15s} {np.mean(v3_roc):.4f} +/- {np.std(v3_roc):.4f}   "
          f"{np.mean(v4_roc):.4f} +/- {np.std(v4_roc):.4f}   {np.mean(v4_roc)-np.mean(v3_roc):+.4f}")

    print(f"\n  训练样本: {len(train_y)}")
    print(f"  v3 正负: {old_train_y.sum()}+/{len(old_train_y)-old_train_y.sum()}-")
    print(f"  v4 正负: {train_y.sum()}+/{len(train_y)-train_y.sum()}-")

    # ── 保存 v4 标签 ──
    meta_v4 = meta.copy()
    meta_v4["label"] = v4_labels
    meta_v4["label_version"] = "v4"
    out_path = "data/processed/label_metadata_v4.csv"
    meta_v4.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"v4 标签已保存: {out_path}")

    # ── 展示翻转样本 ──
    flip_mask = (v4_labels != old_labels) & np.array([
        str(meta.loc[i, "source"]) != "group2" for i in range(len(meta))
    ])
    if flip_mask.sum() > 0:
        print(f"\n  翻转样本示例 (前5, 训练集中):")
        for idx in np.where(flip_mask)[0][:5]:
            row = meta.iloc[idx]
            ff = str(row["film_field"])[:150]
            print(f"    [{row['source']}] v3→v4: {int(old_labels[idx])}→{int(v4_labels[idx])}")
            print(f"      {ff}")


if __name__ == "__main__":
    import xgboost as xgb

    main()
