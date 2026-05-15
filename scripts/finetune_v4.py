"""D1: v4 干净标签 + 512-dim 嵌入 + GNN 编码器重新微调。

关键变化 vs finetune_benchmark_holdout.py:
  1. 使用 v4 标签 (label_metadata_v4.csv)
  2. 排除 group2 (作为独立测试集)
  3. 微调后用 XGBoost CV 评估 (512-dim ald+am)
  4. 与旧编码器 (v3微调) 对比
"""
import json
import os
import sys
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from rdkit import Chem
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn import (MoleculeEncoder, PairPredictor,
                               smiles_to_graph, collate_graphs,
                               extract_monomer_embeddings)
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("finetune_v4")

DEVICE = "cpu"
HIDDEN = 256
EPOCHS = 200
EARLY_STOP = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32


def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else smi


def load_v4_data():
    """加载 v4 标签数据，排除 group2 和 benchmark pairs。"""
    meta = pd.read_csv("data/processed/label_metadata_v4.csv", encoding="utf-8-sig")
    with open("data/processed/benchmark_pairs.json", encoding="utf-8") as f:
        bench = json.load(f)

    bench_pairs = set()
    for lk in ["positive", "negative"]:
        for p in bench[lk]:
            bench_pairs.add((_canon(p["ald_smi"]), _canon(p["am_smi"])))

    train_data = []
    for _, row in meta.iterrows():
        a = _canon(str(row["aldehyde_smiles"]))
        b = _canon(str(row["amine_smiles"]))
        if not a or not b:
            continue
        if (a, b) in bench_pairs:
            continue
        src = str(row.get("source", ""))
        if src == "group2":
            continue
        train_data.append({
            "ald_smi": a, "am_smi": b,
            "label": int(row["label"]),
        })

    logger.info(f"v4 训练集: {len(train_data)} (正={sum(d['label'] for d in train_data)}, "
                f"{sum(d['label'] for d in train_data)/len(train_data)*100:.1f}%)")

    # 图缓存
    all_smis = set()
    for d in train_data:
        all_smis.add(d["ald_smi"])
        all_smis.add(d["am_smi"])
    graph_cache = {}
    for smi in all_smis:
        g = smiles_to_graph(smi)
        if g is not None:
            graph_cache[smi] = g

    ald_g, am_g, lbs, ald_s, am_s = [], [], [], [], []
    for d in train_data:
        ga = graph_cache.get(d["ald_smi"])
        gb = graph_cache.get(d["am_smi"])
        if ga is not None and gb is not None:
            ald_g.append(ga)
            am_g.append(gb)
            lbs.append(d["label"])
            ald_s.append(d["ald_smi"])
            am_s.append(d["am_smi"])

    return ald_g, am_g, np.array(lbs), ald_s, am_s


def build_512_features(emb_dict, ald_s, am_s):
    X = []
    for a, b in zip(ald_s, am_s):
        ea = emb_dict.get(a, np.zeros(HIDDEN, dtype=np.float32))
        eb = emb_dict.get(b, np.zeros(HIDDEN, dtype=np.float32))
        X.append(np.concatenate([ea, eb]))
    return np.array(X, dtype=np.float32)


def xgb_cv_eval(encoder, ald_s, am_s, labels, params, name=""):
    """用编码器提取 512-dim 嵌入，跑 XGBoost 5 折 CV。"""
    all_smis = list(set(ald_s + am_s))
    emb_arr = extract_monomer_embeddings(encoder, all_smis)
    emb_dict = {s: emb_arr[i] for i, s in enumerate(all_smis)}

    X = build_512_features(emb_dict, ald_s, am_s)
    y = np.array(labels)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_scores, roc_scores = [], []

    for tr_idx, va_idx in cv.split(np.zeros(len(y)), y):
        sw = (len(y[tr_idx]) - y[tr_idx].sum()) / max(y[tr_idx].sum(), 1)
        m = xgb.XGBClassifier(
            scale_pos_weight=sw, eval_metric="aucpr", random_state=42, **params)
        m.fit(X[tr_idx], y[tr_idx])
        prob = m.predict_proba(X[va_idx])[:, 1]
        pr_scores.append(average_precision_score(y[va_idx], prob))
        roc_scores.append(roc_auc_score(y[va_idx], prob))

    return np.mean(pr_scores), np.std(pr_scores), np.mean(roc_scores)


def main():
    # ── 加载 v4 数据 ──
    ald_g, am_g, labels, ald_s, am_s = load_v4_data()

    # ── 基线: 旧编码器 (v3 微调) on v4 数据 ──
    logger.info("\n=== 基线: 旧编码器 (v3微调) + v4标签 ===")
    old_encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
    old_encoder.load_state_dict(torch.load(
        "models/v2.0/gnn_encoder_finetuned.pt", map_location="cpu"))
    old_encoder.eval()

    old_params = {
        "colsample_bytree": 0.7929, "gamma": 1.41, "learning_rate": 0.125,
        "max_depth": 4, "min_child_weight": 3, "n_estimators": 479,
        "reg_alpha": 0.1285, "reg_lambda": 2.12, "subsample": 0.9283,
    }
    v4_params = {
        "colsample_bytree": 0.9849, "gamma": 0.0863, "learning_rate": 0.233,
        "max_depth": 8, "min_child_weight": 5, "n_estimators": 302,
        "reg_alpha": 1.379, "reg_lambda": 1.259, "subsample": 0.878,
    }

    old_pr, old_pr_std, old_roc = xgb_cv_eval(
        old_encoder, ald_s, am_s, labels, old_params, "old_enc+v4")

    # 也用 v4 调优参数试试
    old_pr_v4p, _, old_roc_v4p = xgb_cv_eval(
        old_encoder, ald_s, am_s, labels, v4_params, "old_enc+v4_params")

    logger.info(f"旧编码器+旧参数: PR-AUC={old_pr:.4f}±{old_pr_std:.4f}, ROC={old_roc:.4f}")
    logger.info(f"旧编码器+v4参数: PR-AUC={old_pr_v4p:.4f}, ROC={old_roc_v4p:.4f}")

    # ── 微调新编码器 (v4 labels) ──
    logger.info("\n=== 5 折 CV 微调 (v4 标签) ===")
    pretrained_path = "models/v2.0/gnn_encoder_cof_pretrained.pt"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_encoder_state = None
    best_pr = 0.0
    cv_prs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        t_ald = [ald_g[i] for i in tr_idx]
        t_am = [am_g[i] for i in tr_idx]
        t_y = labels[tr_idx]
        v_ald = [ald_g[i] for i in va_idx]
        v_am = [am_g[i] for i in va_idx]
        v_y = labels[va_idx]

        # 初始化编码器 (从预训练权重开始)
        encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
        if os.path.exists(pretrained_path):
            encoder.load_state_dict(
                torch.load(pretrained_path, map_location="cpu"), strict=False)

        model = PairPredictor(encoder, hidden=HIDDEN)
        opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sch = CosineAnnealingLR(opt, T_max=EPOCHS)

        best_loss = float("inf")
        best_state = None
        patience = 0
        pos_w = torch.tensor([(len(t_y) - t_y.sum()) / max(t_y.sum(), 1)]).float()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            n = len(t_ald)
            idx = np.random.permutation(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, BATCH_SIZE):
                bi = idx[start:start + BATCH_SIZE]
                ald_b = collate_graphs([t_ald[i] for i in bi])
                am_b = collate_graphs([t_am[i] for i in bi])
                y_b = torch.tensor(t_y[bi], dtype=torch.float)
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(ald_b, am_b), y_b, pos_weight=pos_w)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
            sch.step()

            if total_loss < best_loss - 1e-4:
                best_loss = total_loss
                best_state = deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= EARLY_STOP:
                break

        model.load_state_dict(best_state)
        encoder_fold = deepcopy(encoder).cpu()

        # XGBoost CV on this fold's encoder
        fold_pr, _, _ = xgb_cv_eval(
            encoder_fold, ald_s, am_s, labels, old_params,
            f"fold{fold+1}")
        cv_prs.append(fold_pr)

        logger.info(f"  Fold {fold+1}: End-to-End PR-AUC on val={evaluate_e2e(model, v_ald, v_am, v_y):.4f}, "
                    f"XGB PR-AUC={fold_pr:.4f}")

        if fold_pr > best_pr:
            best_pr = fold_pr
            best_encoder_state = deepcopy(encoder_fold.state_dict())

    # ── 保存最佳编码器 ──
    if best_encoder_state:
        encoder_path = "models/v2.0/gnn_encoder_finetuned_v4.pt"
        torch.save(best_encoder_state, encoder_path)
        logger.info(f"\nv4 微调编码器已保存: {encoder_path}")

        # 全量训练最终版本
        final_encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
        if os.path.exists(pretrained_path):
            final_encoder.load_state_dict(
                torch.load(pretrained_path, map_location="cpu"), strict=False)

        final_model = PairPredictor(final_encoder, hidden=HIDDEN)
        final_opt = AdamW(final_model.parameters(), lr=LR * 0.5, weight_decay=WEIGHT_DECAY)
        final_sch = CosineAnnealingLR(final_opt, T_max=EPOCHS // 2)
        pos_w = torch.tensor([(len(labels) - labels.sum()) / max(labels.sum(), 1)]).float()

        best_loss = float("inf")
        best_full = None
        patience = 0
        for epoch in range(1, EPOCHS // 2 + 1):
            final_model.train()
            n = len(ald_g)
            idx = np.random.permutation(n)
            total_loss = 0.0
            for start in range(0, n, BATCH_SIZE):
                bi = idx[start:start + BATCH_SIZE]
                ald_b = collate_graphs([ald_g[i] for i in bi])
                am_b = collate_graphs([am_g[i] for i in bi])
                y_b = torch.tensor(labels[bi], dtype=torch.float)
                final_opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    final_model(ald_b, am_b), y_b, pos_weight=pos_w)
                loss.backward()
                final_opt.step()
                total_loss += loss.item()
            final_sch.step()
            if total_loss < best_loss - 1e-4:
                best_loss = total_loss
                best_full = deepcopy(final_model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= EARLY_STOP:
                break

        if best_full:
            final_model.load_state_dict(best_full)

    # ── 最终对比 ──
    new_encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
    new_encoder.load_state_dict(best_encoder_state or torch.load(encoder_path))
    new_encoder.eval()

    new_pr, new_pr_std, new_roc = xgb_cv_eval(
        new_encoder, ald_s, am_s, labels, old_params, "new_enc+old_params")
    new_pr_v4p, _, new_roc_v4p = xgb_cv_eval(
        new_encoder, ald_s, am_s, labels, v4_params, "new_enc+v4_params")

    print("\n" + "=" * 65)
    print("  D1 结果: v4 标签 + 重微调 GNN 编码器")
    print("=" * 65)
    print(f"\n  {'编码器':20s} {'参数':15s} {'PR-AUC':12s} {'ROC-AUC':10s}")
    print(f"  {'-' * 55}")
    print(f"  {'v3 微调 (旧)':20s} {'旧参数':15s} {old_pr:.4f} ± {old_pr_std:.4f}  {old_roc:.4f}")
    print(f"  {'v3 微调 (旧)':20s} {'v4参数':15s} {old_pr_v4p:.4f}           {old_roc_v4p:.4f}")
    print(f"  {'v4 微调 (新)':20s} {'旧参数':15s} {new_pr:.4f} ± {new_pr_std:.4f}  {new_roc:.4f}")
    print(f"  {'v4 微调 (新)':20s} {'v4参数':15s} {new_pr_v4p:.4f}           {new_roc_v4p:.4f}")

    delta = new_pr - old_pr
    print(f"\n  v4 重微调增益: PR-AUC {delta:+.4f} (旧编码器 {old_pr:.4f} → 新编码器 {new_pr:.4f})")

    if delta > 0:
        print(f"\n  ✓ D1 成功! 新编码器用 v4 标签微调后 PR-AUC 提升 {delta:+.4f}")
    else:
        print(f"\n  ✗ D1 未提升。v4 标签不足以改善编码器表示。")


@torch.no_grad()
def evaluate_e2e(model, ald_g, am_g, labels):
    model.eval()
    probs = []
    for i in range(0, len(ald_g), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(ald_g))
        ald_b = collate_graphs(ald_g[i:end])
        am_b = collate_graphs(am_g[i:end])
        logits = model(ald_b, am_b)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)
    return average_precision_score(labels, probs)


if __name__ == "__main__":
    main()
