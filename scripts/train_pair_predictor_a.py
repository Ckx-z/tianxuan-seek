"""D2-A: 端到端配对分类器 (方案 A) — 微调 GNN + 双线性交互 + MLP 头。

与方案 B 的关键差异:
  1. GNN 编码器从 v4 权重初始化但参与训练 (方案 B 冻结)
  2. 底层 LR 1e-4, 分类头 LR 1e-3 (分层学习率)
  3. 更强正则化: Dropout 0.4, 更高 Weight Decay
  4. 直接在图数据上训练, 不做预计算嵌入

对比基准:
  XGBoost (512-dim):   0.671
  方案 B (冻结+交互头): 0.704
  方案 A (微调+交互头): 目标 ≥0.72
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
import torch.nn.functional as F
import xgboost as xgb
from rdkit import Chem
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn import (MoleculeEncoder, smiles_to_graph,
                               collate_graphs, extract_monomer_embeddings)
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("pair_pred_a")

HIDDEN = 256
DEVICE = "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else smi


class BilinearHead(nn.Module):
    """双线性交互 + MLP 分类头 (编码器无关, 可复用方案 B 权重)."""

    def __init__(self, hidden: int = 256, bilinear_rank: int = 64,
                 mlp_hidden: int = 128, dropout: float = 0.4):
        super().__init__()
        self.U = nn.Parameter(torch.randn(hidden, bilinear_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden, bilinear_rank) * 0.01)
        self.bilinear_bias = nn.Parameter(torch.zeros(bilinear_rank))

        input_dim = hidden * 4 + bilinear_rank
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.BatchNorm1d(mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        bilinear = (ea @ self.U) * (eb @ self.V) + self.bilinear_bias
        features = torch.cat([ea, eb, ea * eb, ea - eb, bilinear], dim=-1)
        return self.mlp(features).squeeze(-1)


class EndToEndModel(nn.Module):
    """方案 A: 可训练 GNN 编码器 + 双线性交互头."""

    def __init__(self, encoder: MoleculeEncoder, head: BilinearHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, ald_data, am_data):
        ea = self.encoder(ald_data)
        eb = self.encoder(am_data)
        return self.head(ea, eb)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()


@torch.no_grad()
def eval_graph_model(model, ald_g, am_g, labels, batch_size=64):
    model.eval()
    probs = []
    for i in range(0, len(ald_g), batch_size):
        end = min(i + batch_size, len(ald_g))
        ald_b = collate_graphs(ald_g[i:end])
        am_b = collate_graphs(am_g[i:end])
        logits = model(ald_b, am_b)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)
    return average_precision_score(labels, probs), roc_auc_score(labels, probs)


def main():
    # ── 加载数据 ──
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
        if str(row.get("source", "")) == "group2":
            continue
        train_data.append({"ald": a, "am": b, "label": int(row["label"])})

    labels = np.array([d["label"] for d in train_data])
    logger.info(f"训练集: {len(labels)} 样本, 正={labels.sum()} ({labels.mean()*100:.1f}%)")

    # ── 构建图缓存 ──
    all_smis = set()
    for d in train_data:
        all_smis.add(d["ald"]); all_smis.add(d["am"])
    graph_cache = {}
    for smi in all_smis:
        g = smiles_to_graph(smi)
        if g is not None:
            graph_cache[smi] = g
    logger.info(f"图缓存: {len(graph_cache)}/{len(all_smis)} 个单体成功建图")

    # 准备图列表
    ald_graphs, am_graphs = [], []
    for d in train_data:
        ga = graph_cache.get(d["ald"])
        gb = graph_cache.get(d["am"])
        if ga is not None and gb is not None:
            ald_graphs.append(ga)
            am_graphs.append(gb)
        else:
            ald_graphs.append(None)
            am_graphs.append(None)

    # ── XGBoost 基线 ──
    encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
    encoder.load_state_dict(torch.load(
        "models/v2.0/gnn_encoder_finetuned_v4.pt", map_location="cpu"))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    emb_arr = extract_monomer_embeddings(encoder, list(all_smis))
    emb_dict = {s: emb_arr[i] for i, s in enumerate(all_smis)}
    X_ea = np.array([emb_dict[d["ald"]] for d in train_data], dtype=np.float32)
    X_eb = np.array([emb_dict[d["am"]] for d in train_data], dtype=np.float32)
    X_xgb = np.concatenate([X_ea, X_eb], axis=1)

    v4_params = {
        "colsample_bytree": 0.9849, "gamma": 0.0863, "learning_rate": 0.233,
        "max_depth": 8, "min_child_weight": 5, "n_estimators": 302,
        "reg_alpha": 1.379, "reg_lambda": 1.259, "subsample": 0.878,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    xgb_prs = []
    for tr_idx, va_idx in cv.split(np.zeros(len(labels)), labels):
        sw = (len(labels[tr_idx]) - labels[tr_idx].sum()) / max(labels[tr_idx].sum(), 1)
        m = xgb.XGBClassifier(scale_pos_weight=sw, eval_metric="aucpr",
                              random_state=SEED, **v4_params)
        m.fit(X_xgb[tr_idx], labels[tr_idx])
        prob = m.predict_proba(X_xgb[va_idx])[:, 1]
        xgb_prs.append(average_precision_score(labels[va_idx], prob))

    # ── 方案 A: 微调 GNN + Bilinear Head ──
    logger.info("\n=== 方案 A: 微调 GNN + Bilinear Head ===")
    pretrained_path = "models/v2.0/gnn_encoder_finetuned_v4.pt"
    plan_b_path = "models/v2.0/pair_predictor_b.pt"

    epochs = 200
    early_stop = 40
    batch_size = 16
    lr_encoder = 1e-4
    lr_head = 1e-3
    weight_decay = 2e-4

    alpha = labels.sum() / len(labels)
    criterion = FocalLoss(alpha=1 - alpha, gamma=2.0)

    a_prs, a_rocs = [], []
    best_model_state = None
    best_val_pr = 0.0

    for fold, (tr_idx, va_idx) in enumerate(cv.split(np.zeros(len(labels)), labels)):
        t_ald = [ald_graphs[i] for i in tr_idx]
        t_am = [am_graphs[i] for i in tr_idx]
        t_y = labels[tr_idx]
        v_ald = [ald_graphs[i] for i in va_idx]
        v_am = [am_graphs[i] for i in va_idx]
        v_y = labels[va_idx]

        # 初始化编码器 (v4 权重)
        encoder_a = MoleculeEncoder(hidden=HIDDEN, dropout=0.2)
        if os.path.exists(pretrained_path):
            encoder_a.load_state_dict(
                torch.load(pretrained_path, map_location="cpu"), strict=False)

        # 初始化分类头 (方案 B 权重作为暖启动)
        head = BilinearHead(hidden=HIDDEN, bilinear_rank=64, mlp_hidden=128, dropout=0.4)
        if os.path.exists(plan_b_path):
            plan_b_state = torch.load(plan_b_path, map_location="cpu")
            head_state = {k: v for k, v in plan_b_state.items()
                          if k in head.state_dict()}
            head.load_state_dict(head_state, strict=False)
            logger.info(f"  Fold {fold+1}: 分类头从方案 B 权重暖启动")

        model = EndToEndModel(encoder_a, head)

        # 分层学习率
        opt = AdamW([
            {"params": model.encoder.parameters(), "lr": lr_encoder},
            {"params": model.head.parameters(), "lr": lr_head},
        ], weight_decay=weight_decay)
        sch = CosineAnnealingLR(opt, T_max=epochs)

        best_loss = float("inf")
        best_state = None
        patience = 0

        for ep in range(1, epochs + 1):
            model.train()
            n = len(t_ald)
            idx = np.random.permutation(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                bi = idx[start:start + batch_size]
                ald_b = collate_graphs([t_ald[i] for i in bi])
                am_b = collate_graphs([t_am[i] for i in bi])
                y_b = torch.tensor(t_y[bi], dtype=torch.float)
                opt.zero_grad()
                loss = criterion(model(ald_b, am_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                n_batches += 1
            sch.step()

            # 验证
            model.eval()
            with torch.no_grad():
                va_logits_list = []
                for i in range(0, len(v_ald), batch_size * 2):
                    end = min(i + batch_size * 2, len(v_ald))
                    ald_b = collate_graphs(v_ald[i:end])
                    am_b = collate_graphs(v_am[i:end])
                    va_logits_list.append(model(ald_b, am_b))
                va_logits = torch.cat(va_logits_list)
                va_loss = F.binary_cross_entropy_with_logits(
                    va_logits, torch.tensor(v_y, dtype=torch.float))

            if va_loss < best_loss - 1e-4:
                best_loss = va_loss.item()
                best_state = deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= early_stop:
                logger.info(f"  Fold {fold+1}: early stop @ epoch {ep}")
                break

        model.load_state_dict(best_state)
        val_pr, val_roc = eval_graph_model(model, v_ald, v_am, v_y)
        a_prs.append(val_pr)
        a_rocs.append(val_roc)
        logger.info(f"  Fold {fold+1}: PR-AUC={val_pr:.4f}, ROC-AUC={val_roc:.4f}")

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_model_state = deepcopy(best_state)

    # ── 加载方案 B 结果 (从保存的日志读取或重新计算) ──
    # 使用脚本中硬编码的方案 B 5折均值作为对比
    b_pr_mean, b_roc_mean = 0.7036, 0.8243

    # ── 结果对比 ──
    print("\n" + "=" * 65)
    print("  D2-A: 微调 GNN + Bilinear Head (方案 A)")
    print("=" * 65)
    print(f"\n  {'方法':35s} {'PR-AUC':14s} {'ROC-AUC':10s}")
    print(f"  {'-' * 55}")
    print(f"  {'XGBoost (512-dim)':35s} "
          f"{np.mean(xgb_prs):.4f} ± {np.std(xgb_prs):.4f}  —")
    print(f"  {'方案 B (冻结+交互头)':35s} "
          f"{b_pr_mean:.4f}              {b_roc_mean:.4f}")
    print(f"  {'方案 A (微调+交互头)':35s} "
          f"{np.mean(a_prs):.4f} ± {np.std(a_prs):.4f}  "
          f"{np.mean(a_rocs):.4f}")

    delta_b = np.mean(a_prs) - b_pr_mean
    delta_xgb = np.mean(a_prs) - np.mean(xgb_prs)
    print(f"\n  方案 A vs 方案 B: {delta_b:+.4f}")
    print(f"  方案 A vs XGBoost: {delta_xgb:+.4f}")

    if delta_b > 0.01:
        print(f"  [OK] GNN 微调带来额外增益 (+{delta_b:.4f}), 方案 A 有效!")
    elif delta_b > -0.01:
        print(f"  [~] GNN 微调无显著提升, 方案 B 或为最优")
    else:
        print(f"  [FAIL] 微调导致过拟合, 方案 B (冻结编码器) 更优")

    # ── 保存 ──
    if best_model_state:
        os.makedirs("models/v2.0", exist_ok=True)
        torch.save(best_model_state, "models/v2.0/end_to_end_a.pt")
        logger.info("方案 A 模型已保存: models/v2.0/end_to_end_a.pt")


if __name__ == "__main__":
    main()
