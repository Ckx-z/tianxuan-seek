"""D2-A: 端到端配对分类器 (方案 A) — 渐进解冻 + 双线性交互 + 化学机理约束。

训练策略:
  1. Phase 1 (前 10 ep): 冻结 GNN 编码器, 仅训练 BilinearHead (含 18 维机理描述符)
  2. Phase 2 (后 790 ep): 解冻编码器, 分层 LR (encoder 1e-4, head 1e-3)
  3. 排序损失: 苯链接 > 含炔链接 (weight=0.05, 含炔正样本降权 0.2)
  4. 化学惩罚项 (方案 B): L_total = L_focal + λ * violation_score
  5. PR-AUC 每 5 ep 监控, 早停 patience=30

用法:
  python scripts/train_pair_predictor_a.py                          # 默认 λ=0.005
  python scripts/train_pair_predictor_a.py --lambda-chem 0          # 关闭化学正则化
  python scripts/train_pair_predictor_a.py --lambda-chem 0.01       # 更强惩罚
"""
import argparse
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
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.linker_analyzer import (
    has_acetylene, compute_monomer_descriptors, compute_pair_descriptor_vector)
from src.chemistry.chem_penalty import ViolationCache
from src.screening.gnn import (MoleculeEncoder, smiles_to_graph,
                               collate_graphs, extract_monomer_embeddings)
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
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
    """双线性交互 + MLP 分类头 (编码器无关, 可复用方案 B 权重).

    extra_dim > 0 时, MLP 输入额外拼接机理描述符, 让模型直接看到
    连接基团类型(苯/炔)、共轭程度、刚性等化学先验信号。
    """

    def __init__(self, hidden: int = 256, bilinear_rank: int = 64,
                 mlp_hidden: int = 128, dropout: float = 0.4,
                 extra_dim: int = 0):
        super().__init__()
        self.extra_dim = extra_dim
        self.U = nn.Parameter(torch.randn(hidden, bilinear_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden, bilinear_rank) * 0.01)
        self.bilinear_bias = nn.Parameter(torch.zeros(bilinear_rank))

        input_dim = hidden * 4 + bilinear_rank + extra_dim
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

    def forward(self, ea: torch.Tensor, eb: torch.Tensor,
                extra: torch.Tensor | None = None) -> torch.Tensor:
        bilinear = (ea @ self.U) * (eb @ self.V) + self.bilinear_bias
        features = torch.cat([ea, eb, ea * eb, ea - eb, bilinear], dim=-1)
        if extra is not None:
            features = torch.cat([features, extra], dim=-1)
        return self.mlp(features).squeeze(-1)


class EndToEndModel(nn.Module):
    """方案 A: 可训练 GNN 编码器 + 双线性交互头."""

    def __init__(self, encoder: MoleculeEncoder, head: BilinearHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, ald_data, am_data, extra_features=None):
        ea = self.encoder(ald_data)
        eb = self.encoder(am_data)
        return self.head(ea, eb, extra=extra_features)


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


def ranking_loss(logits, ald_acet, am_acet, labels=None, margin=0.05,
                 pos_acet_weight=1.0):
    """惩罚含炔对得分高于纯苯对的排序错误。

    化学先验: 苯环共轭稳定亚胺键 > 炔基刚性弱共轭。
    pos_acet_weight: 含炔正样本的排序损失权重 (默认 1.0, 无降权)。
    仅当批内同时存在苯对和含炔对时才生效。
    """
    benzene_mask = ~(ald_acet | am_acet)
    acetylene_mask = ald_acet | am_acet
    if not benzene_mask.any() or not acetylene_mask.any():
        return torch.tensor(0.0, device=logits.device)

    if labels is not None:
        pos_acet_mask = acetylene_mask & (labels == 1)
        w = torch.ones(len(logits), device=logits.device)
        w[pos_acet_mask] = pos_acet_weight
        b_mean = (logits[benzene_mask] * w[benzene_mask]).sum() / max(w[benzene_mask].sum(), torch.tensor(1.0, device=logits.device))
        a_mean = (logits[acetylene_mask] * w[acetylene_mask]).sum() / max(w[acetylene_mask].sum(), torch.tensor(1.0, device=logits.device))
    else:
        b_mean = logits[benzene_mask].mean()
        a_mean = logits[acetylene_mask].mean()
    return torch.clamp(margin + a_mean - b_mean, min=0)


@torch.no_grad()
def eval_graph_model(model, ald_g, am_g, labels, batch_size=64,
                     pair_extra=None):
    model.eval()
    probs = []
    for i in range(0, len(ald_g), batch_size):
        end = min(i + batch_size, len(ald_g))
        ald_b = collate_graphs(ald_g[i:end])
        am_b = collate_graphs(am_g[i:end])
        extra = None
        if pair_extra is not None:
            extra = torch.tensor(pair_extra[i:end], dtype=torch.float)
        logits = model(ald_b, am_b, extra_features=extra)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)
    return average_precision_score(labels, probs), roc_auc_score(labels, probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-chem", type=float, default=0.005,
                        help="化学惩罚项强度 (默认 0.005, 设 0 关闭)")
    args = parser.parse_args()

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

    # ── 化学违反度缓存 (方案 B) ──
    v_cache = None
    if args.lambda_chem > 0:
        ald_smis = [d["ald"] for d in train_data]
        am_smis = [d["am"] for d in train_data]
        v_cache = ViolationCache(ald_smis, am_smis)
        summary = v_cache.violation_summary()
        logger.info(f"化学违反度缓存: {len(v_cache.scores)} 对, "
                    f"均值={summary['mean']:.4f}, 非零率={summary['nonzero_frac']:.2%}")
        logger.info(f"  λ_chem={args.lambda_chem}, "
                    f"有效惩罚强度 ≈ λ × mean_viol × P(pred>0.5)")

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

    # ── 机理描述符缓存 (RDKit, 18 维/对) ──
    mol_cache = {}
    for smi in all_smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol_cache[smi] = mol
    pair_extra = np.zeros((len(train_data), 26), dtype=np.float32)
    ald_acet = np.zeros(len(train_data), dtype=bool)
    am_acet = np.zeros(len(train_data), dtype=bool)
    for i, d in enumerate(train_data):
        ald_mol = mol_cache.get(d["ald"])
        am_mol = mol_cache.get(d["am"])
        if ald_mol is not None and am_mol is not None:
            pair_extra[i] = compute_pair_descriptor_vector(ald_mol, am_mol)
            ald_acet[i] = has_acetylene(ald_mol)
            am_acet[i] = has_acetylene(am_mol)
    n_acet_pairs = (ald_acet | am_acet).sum()
    logger.info(f"机理描述符: 26 维/对, 含炔对={n_acet_pairs}/{len(train_data)} "
                f"({n_acet_pairs/len(train_data)*100:.1f}%)")

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
    cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=8, random_state=SEED)
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

    epochs = 800
    early_stop = 30
    freeze_epochs = 10
    batch_size = 16
    lr_encoder = 1e-5
    lr_head = 1e-3
    weight_decay = 2e-4
    ranking_weight = 0  # 苯>炔排序损失已关闭

    alpha = labels.sum() / len(labels)
    criterion = FocalLoss(alpha=1 - alpha, gamma=2.0)

    a_prs, a_rocs = [], []
    best_model_state = None
    best_val_pr = 0.0

    n_folds = cv.get_n_splits() if hasattr(cv, 'get_n_splits') else 64
    for fold, (tr_idx, va_idx) in enumerate(cv.split(np.zeros(len(labels)), labels)):
        rep = fold // 8 + 1
        fid = fold % 8 + 1
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
        head = BilinearHead(hidden=HIDDEN, bilinear_rank=64, mlp_hidden=128,
                              dropout=0.4, extra_dim=26)
        if os.path.exists(plan_b_path):
            plan_b_state = torch.load(plan_b_path, map_location="cpu")
            # 只暖启动 U/V/Bias (双线性部分), MLP 因 extra_dim 扩维需随机初始化
            head_state = {k: v for k, v in plan_b_state.items()
                          if k in head.state_dict()
                          and v.shape == head.state_dict()[k].shape}
            head.load_state_dict(head_state, strict=False)
            logger.info(f"  R{rep}F{fid}: 分类头从方案 B 权重暖启动")

        model = EndToEndModel(encoder_a, head)

        # 本 fold 的额外特征 + 炔基标志
        t_extra = pair_extra[tr_idx]
        v_extra = pair_extra[va_idx]
        t_ald_acet = ald_acet[tr_idx]
        t_am_acet = am_acet[tr_idx]

        # Phase 1: 冻结编码器，仅训练分类头
        for p in model.encoder.parameters():
            p.requires_grad = False
        opt = AdamW(model.head.parameters(), lr=lr_head,
                    weight_decay=weight_decay)
        sch = CosineAnnealingLR(opt, T_max=epochs)

        best_pr = 0.0
        best_state = None
        patience = 0
        encoder_unfrozen = False

        for ep in range(1, epochs + 1):
            # 渐进式解冻: freeze_epochs 后解冻编码器, 分层 LR
            if ep == freeze_epochs + 1:
                for p in model.encoder.parameters():
                    p.requires_grad = True
                opt = AdamW([
                    {"params": model.encoder.parameters(), "lr": lr_encoder},
                    {"params": model.head.parameters(), "lr": lr_head},
                ], weight_decay=weight_decay)
                sch = CosineAnnealingLR(opt, T_max=epochs)
                encoder_unfrozen = True
                logger.info(f"  R{rep}F{fid} ep {ep}: 编码器解冻 (lr_enc={lr_encoder})")

            model.train()
            n = len(t_ald)
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                bi = idx[start:start + batch_size]
                ald_b = collate_graphs([t_ald[i] for i in bi])
                am_b = collate_graphs([t_am[i] for i in bi])
                y_b = torch.tensor(t_y[bi], dtype=torch.float)
                extra_b = torch.tensor(t_extra[bi], dtype=torch.float)
                opt.zero_grad()
                logits = model(ald_b, am_b, extra_features=extra_b)
                focal = criterion(logits, y_b)
                rank = ranking_loss(logits,
                                    torch.tensor(t_ald_acet[bi]),
                                    torch.tensor(t_am_acet[bi]),
                                    labels=y_b)
                loss = focal + ranking_weight * rank
                if args.lambda_chem > 0 and v_cache is not None:
                    probs = torch.sigmoid(logits)
                    v_batch = v_cache.to_tensor(
                        [tr_idx[i] for i in bi], device=DEVICE)
                    mask = (probs > 0.5).float()
                    chem_pen = (mask * v_batch).mean()
                    loss = loss + args.lambda_chem * chem_pen
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

            # PR-AUC 早停 (每 5 ep 评估, 对齐 val_loss 计算)
            if ep % 5 == 0 or ep == 1:
                model.eval()
                va_pr, _ = eval_graph_model(model, v_ald, v_am, v_y,
                                            pair_extra=v_extra)
                tag = "[frozen]" if not encoder_unfrozen else "[finetune]"
                logger.info(f"  R{rep}F{fid} ep {ep:3d} {tag}: val_PR={va_pr:.4f}")

                if va_pr > best_pr + 1e-4:
                    best_pr = va_pr
                    best_state = deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                if patience >= early_stop:
                    logger.info(f"  R{rep}F{fid}: early stop @ epoch {ep}, best_PR={best_pr:.4f}")
                    break

        model.load_state_dict(best_state)
        val_pr, val_roc = eval_graph_model(model, v_ald, v_am, v_y,
                                            pair_extra=v_extra)
        a_prs.append(val_pr)
        a_rocs.append(val_roc)
        logger.info(f"  R{rep}F{fid}: PR-AUC={val_pr:.4f}, ROC-AUC={val_roc:.4f}")

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_model_state = deepcopy(best_state)

    # ── 加载方案 B 结果 (从保存的日志读取或重新计算) ──
    # 使用脚本中硬编码的方案 B 5折均值作为对比
    b_pr_mean, b_roc_mean = 0.7036, 0.8243

    # ── 结果对比 ──
    print("\n" + "=" * 65)
    print(f"  D2-A: 微调 GNN + Bilinear Head (方案 A) — 8×8 重复 CV ({len(a_prs)} 折)")
    print("=" * 65)
    print(f"\n  各折 PR-AUC: {' '.join(f'{p:.4f}' for p in a_prs)}")
    print(f"  {'方法':35s} {'PR-AUC':14s} {'ROC-AUC':10s}")
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
