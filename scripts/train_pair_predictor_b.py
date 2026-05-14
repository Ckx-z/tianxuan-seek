"""D2-B: 端到端配对分类器 (方案 B) — 冻结 GNN + 双线性交互 + MLP 头。

核心假设: 冻结编码器 + 可学习交互头 ≥ XGBoost (0.671)
验证: 如果 PR-AUC > 0.67，说明神经网络交互头有信息增益，值得微调 GNN。
       如果 PR-AUC < 0.67，说明 XGBoost 已是天花板，需要方案 A (微调 GNN) 或 C (交叉图注意力)。

关键设计:
  1. 复用 v4 微调编码器 (FROZEN), 预计算所有单体嵌入
  2. 分解双线性: ea^T (U V^T) eb, rank=64
  3. 特征拼接: [ea, eb, ea⊙eb, ea-eb, bilinear] → 3 层 MLP → sigmoid
  4. Focal Loss (γ=2) 缓解正负不均衡
  5. 5 折 CV, 与 XGBoost 相同划分基准
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
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn import MoleculeEncoder, extract_monomer_embeddings
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("pair_pred_b")

HIDDEN = 256
DEVICE = "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else smi


class BilinearPairPredictor(nn.Module):
    """冻结编码器 + 双线性交互 + MLP 分类头 (方案 B)."""

    def __init__(self, hidden: int = 256, bilinear_rank: int = 64,
                 mlp_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        # Factorized bilinear: ea^T U V^T eb
        self.U = nn.Parameter(torch.randn(hidden, bilinear_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden, bilinear_rank) * 0.01)
        self.bilinear_bias = nn.Parameter(torch.zeros(bilinear_rank))

        # Input: ea(256) + eb(256) + hadamard(256) + diff(256) + bilinear(64) = 1088
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


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for ea_b, eb_b, y_b in loader:
        optimizer.zero_grad()
        logits = model(ea_b, eb_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for ea_b, eb_b, y_b in loader:
        logits = model(ea_b, eb_b)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_b.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return average_precision_score(labels, probs), roc_auc_score(labels, probs)


def main():
    # ── 加载数据 (v4 标签, 排除 group2 + benchmark) ──
    meta = pd.read_csv("data/processed/label_metadata_v4.csv", encoding="utf-8-sig")
    with open("data/processed/benchmark_pairs.json", encoding="utf-8") as f:
        bench = json.load(f)
    bench_pairs = set()
    for lk in ["positive", "negative"]:
        for p in bench[lk]:
            bench_pairs.add((_canon(p["ald_smi"]), _canon(p["am_smi"])))

    ald_smis, am_smis, labels = [], [], []
    for _, row in meta.iterrows():
        a = _canon(str(row["aldehyde_smiles"]))
        b = _canon(str(row["amine_smiles"]))
        if not a or not b:
            continue
        if (a, b) in bench_pairs:
            continue
        if str(row.get("source", "")) == "group2":
            continue
        ald_smis.append(a)
        am_smis.append(b)
        labels.append(int(row["label"]))

    labels = np.array(labels)
    logger.info(f"训练集: {len(labels)} 样本, 正={labels.sum()} ({labels.mean()*100:.1f}%)")

    # ── 预计算嵌入 (冻结 v4 编码器) ──
    encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
    encoder.load_state_dict(torch.load(
        "models/v2.0/gnn_encoder_finetuned_v4.pt", map_location="cpu"))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    all_smis = list(set(ald_smis + am_smis))
    logger.info(f"唯一单体: {len(all_smis)}")
    emb_arr = extract_monomer_embeddings(encoder, all_smis)
    emb_dict = {s: emb_arr[i] for i, s in enumerate(all_smis)}

    X_ea = np.array([emb_dict[s] for s in ald_smis], dtype=np.float32)
    X_eb = np.array([emb_dict[s] for s in am_smis], dtype=np.float32)

    # ── XGBoost 基线 (512-dim ald+am) ──
    v4_params = {
        "colsample_bytree": 0.9849, "gamma": 0.0863, "learning_rate": 0.233,
        "max_depth": 8, "min_child_weight": 5, "n_estimators": 302,
        "reg_alpha": 1.379, "reg_lambda": 1.259, "subsample": 0.878,
    }
    X_xgb = np.concatenate([X_ea, X_eb], axis=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    xgb_prs, xgb_rocs = [], []
    for tr_idx, va_idx in cv.split(np.zeros(len(labels)), labels):
        sw = (len(labels[tr_idx]) - labels[tr_idx].sum()) / max(labels[tr_idx].sum(), 1)
        m = xgb.XGBClassifier(scale_pos_weight=sw, eval_metric="aucpr",
                              random_state=SEED, **v4_params)
        m.fit(X_xgb[tr_idx], labels[tr_idx])
        prob = m.predict_proba(X_xgb[va_idx])[:, 1]
        xgb_prs.append(average_precision_score(labels[va_idx], prob))
        xgb_rocs.append(roc_auc_score(labels[va_idx], prob))

    # ── 方案 B: Bilinear + MLP 分类头 ──
    logger.info("\n=== 方案 B: Bilinear + MLP (冻结编码器) ===")
    pos_weight = (len(labels) - labels.sum()) / max(labels.sum(), 1)
    alpha = labels.sum() / len(labels)
    criterion = FocalLoss(alpha=1 - alpha, gamma=2.0)

    epochs = 300
    early_stop = 50
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-4

    b_prs, b_rocs = [], []
    best_model_state = None
    best_val_pr = 0.0

    for fold, (tr_idx, va_idx) in enumerate(cv.split(np.zeros(len(labels)), labels)):
        ea_tr = torch.tensor(X_ea[tr_idx])
        eb_tr = torch.tensor(X_eb[tr_idx])
        y_tr = torch.tensor(labels[tr_idx], dtype=torch.float)
        ea_va = torch.tensor(X_ea[va_idx])
        eb_va = torch.tensor(X_eb[va_idx])
        y_va = torch.tensor(labels[va_idx], dtype=torch.float)

        ds_tr = TensorDataset(ea_tr, eb_tr, y_tr)
        ds_va = TensorDataset(ea_va, eb_va, y_va)
        ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        ld_va = DataLoader(ds_va, batch_size=batch_size * 2)

        model = BilinearPairPredictor(hidden=HIDDEN, bilinear_rank=64,
                                      mlp_hidden=128, dropout=0.3)
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sch = CosineAnnealingLR(opt, T_max=epochs)

        best_loss = float("inf")
        best_state = None
        patience = 0

        for ep in range(1, epochs + 1):
            train_loss = train_one_epoch(model, ld_tr, opt, criterion)
            sch.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(ea_va, eb_va)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, y_va)
                val_pr, _ = evaluate_model(model, ld_va)

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss.item()
                best_state = deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= early_stop:
                logger.info(f"  Fold {fold+1}: early stop @ epoch {ep}")
                break

        model.load_state_dict(best_state)
        val_pr, val_roc = evaluate_model(model, ld_va)
        b_prs.append(val_pr)
        b_rocs.append(val_roc)
        logger.info(f"  Fold {fold+1}: PR-AUC={val_pr:.4f}, ROC-AUC={val_roc:.4f}")

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_model_state = deepcopy(best_state)

    # ── 结果对比 ──
    print("\n" + "=" * 65)
    print("  D2-B: 冻结编码器 + Bilinear + MLP (方案 B)")
    print("=" * 65)
    print(f"\n  {'方法':30s} {'PR-AUC':14s} {'ROC-AUC':10s}")
    print(f"  {'-' * 55}")
    print(f"  {'GNN → XGBoost (512-dim)':30s} "
          f"{np.mean(xgb_prs):.4f} ± {np.std(xgb_prs):.4f}  "
          f"{np.mean(xgb_rocs):.4f}")
    print(f"  {'GNN → Bilinear+MLP (方案B)':30s} "
          f"{np.mean(b_prs):.4f} ± {np.std(b_prs):.4f}  "
          f"{np.mean(b_rocs):.4f}")

    delta = np.mean(b_prs) - np.mean(xgb_prs)
    print(f"\n  ΔPR-AUC: {delta:+.4f}")
    if delta > 0.01:
        print(f"  [OK] 方案 B 优于 XGBoost! 交互头有信息增益，建议推进方案 A (微调 GNN)")
    elif delta > -0.02:
        print(f"  [~] 方案 B 与 XGBoost 持平，交互头无明显增益")
    else:
        print(f"  [FAIL] 方案 B 不如 XGBoost，神经网络头过拟合或表达不足")

    # ── 保存最佳模型 ──
    if best_model_state:
        os.makedirs("models/v2.0", exist_ok=True)
        torch.save(best_model_state, "models/v2.0/pair_predictor_b.pt")
        logger.info("方案 B 模型已保存: models/v2.0/pair_predictor_b.pt")


if __name__ == "__main__":
    main()
