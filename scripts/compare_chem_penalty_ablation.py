"""方案 B 化学惩罚项 λ 消融实验 — 3-fold CV 对比不同惩罚强度。

用法:
  python scripts/compare_chem_penalty_ablation.py
  python scripts/compare_chem_penalty_ablation.py --lambdas 0,0.005,0.01,0.05
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
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.linker_analyzer import (
    has_acetylene, compute_monomer_descriptors, compute_pair_descriptor_vector)
from src.chemistry.chem_penalty import ViolationCache
from src.screening.gnn import (MoleculeEncoder, smiles_to_graph,
                               collate_graphs, extract_monomer_embeddings)
from scripts.train_pair_predictor_a import (EndToEndModel, BilinearHead, FocalLoss)
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)
logger = setup_logger("chem_pen_abl")

HIDDEN = 256
DEVICE = "cpu"
SEED = 42
N_FOLDS = 3
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 64
RANKING_WEIGHT = 0.005

torch.manual_seed(SEED)
np.random.seed(SEED)


def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else ""


def load_data():
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

    all_smis = set()
    for d in train_data:
        all_smis.add(d["ald"]); all_smis.add(d["am"])
    graph_cache = {}
    for smi in all_smis:
        g = smiles_to_graph(smi)
        if g is not None:
            graph_cache[smi] = g

    mol_cache = {}
    for smi in all_smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol_cache[smi] = mol

    return train_data, labels, graph_cache, mol_cache


def train_fold(train_idx, val_idx, train_data, labels, graph_cache, mol_cache,
               violations, lambda_chem: float):
    """训练一折, 返回 best PR-AUC。violations 是全部样本的违反度列表。"""
    ald_graphs, am_graphs, descs, ys, viols = [], [], [], [], []
    kept_indices = []  # 记录保留样本在原数据中的位置
    for idx_in_full in train_idx:
        d = train_data[idx_in_full]
        ga = graph_cache.get(d["ald"])
        gb = graph_cache.get(d["am"])
        if ga is None or gb is None:
            continue
        ma = mol_cache.get(d["ald"])
        mb = mol_cache.get(d["am"])
        if ma is None or mb is None:
            continue
        ald_graphs.append(ga); am_graphs.append(gb)
        ys.append(d["label"])
        dp = compute_pair_descriptor_vector(ma, mb)
        descs.append(dp)
        viols.append(violations[idx_in_full])
        kept_indices.append(idx_in_full)

    # 炔基掩码 (排序损失用)
    ald_acet = np.array([has_acetylene(mol_cache[train_data[i]["ald"]])
                         for i in kept_indices], dtype=bool)
    am_acet = np.array([has_acetylene(mol_cache[train_data[i]["am"]])
                        for i in kept_indices], dtype=bool)

    encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
    head = BilinearHead(hidden=HIDDEN, bilinear_rank=64, mlp_hidden=128,
                        dropout=0.4, extra_dim=26)
    model = EndToEndModel(encoder, head).to(DEVICE)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    pretrained = "models/v2.0/gnn_encoder_finetuned_v4.pt"
    if os.path.exists(pretrained):
        state = torch.load(pretrained, map_location=DEVICE, weights_only=True)
        model.encoder.load_state_dict(state, strict=False)

    opt = AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=2e-4)

    viol_tensor = torch.tensor(viols, dtype=torch.float32, device=DEVICE)

    def ranking_loss(logits, a_acet, b_acet, labels_b=None, margin=0.05):
        benzene_mask = ~(a_acet | b_acet)
        acetylene_mask = a_acet | b_acet
        if not benzene_mask.any() or not acetylene_mask.any():
            return torch.tensor(0.0, device=logits.device)
        if labels_b is not None:
            pos_acet_mask = acetylene_mask & (labels_b == 1)
            w = torch.ones(len(logits), device=logits.device)
            w[pos_acet_mask] = 0.2
            b_mean = (logits[benzene_mask] * w[benzene_mask]).sum() / w[benzene_mask].sum().clamp(min=1)
            a_mean = (logits[acetylene_mask] * w[acetylene_mask]).sum() / w[acetylene_mask].sum().clamp(min=1)
        else:
            b_mean = logits[benzene_mask].mean()
            a_mean = logits[acetylene_mask].mean()
        return torch.clamp(margin + a_mean - b_mean, min=0)

    best_pr, patience = 0.0, 0

    for ep in range(MAX_EPOCHS):
        model.train()
        n = len(ald_graphs)
        perm = np.random.permutation(n)
        for j in range(0, n, BATCH_SIZE):
            idx = perm[j:j + BATCH_SIZE]
            ga_batch = collate_graphs([ald_graphs[k] for k in idx])
            gb_batch = collate_graphs([am_graphs[k] for k in idx])
            d_batch = torch.tensor(np.array([descs[k] for k in idx]), dtype=torch.float32)
            y_batch = torch.tensor(np.array([ys[k] for k in idx]), dtype=torch.float32)
            logits = model(ga_batch, gb_batch, d_batch)
            focal = criterion(logits, y_batch)
            rank = ranking_loss(logits,
                                torch.tensor(ald_acet[idx]),
                                torch.tensor(am_acet[idx]),
                                labels_b=y_batch)
            loss = focal + RANKING_WEIGHT * rank
            if lambda_chem > 0:
                probs = torch.sigmoid(logits)
                v_batch = viol_tensor[idx]
                mask = (probs > 0.5).float()
                chem_pen = (mask * v_batch).mean()
                loss = loss + lambda_chem * chem_pen
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for i in val_idx:
                d = train_data[i]
                ga = graph_cache.get(d["ald"])
                gb = graph_cache.get(d["am"])
                ma = mol_cache.get(d["ald"])
                mb = mol_cache.get(d["am"])
                if ga is None or gb is None or ma is None or mb is None:
                    continue
                dp = compute_pair_descriptor_vector(ma, mb)
                logit = model(collate_graphs([ga]), collate_graphs([gb]),
                              torch.tensor([dp], dtype=torch.float32))
                val_probs.append(torch.sigmoid(logit).item())
                val_labels.append(d["label"])

        if len(set(val_labels)) < 2:
            continue
        pr = average_precision_score(val_labels, val_probs)
        if pr > best_pr:
            best_pr = pr
            patience = 0
        else:
            patience += 1
        if patience >= PATIENCE:
            break

    return best_pr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=str, default="0,0.005,0.01,0.05",
                        help="逗号分隔的 λ 值列表")
    args = parser.parse_args()
    lambda_values = [float(x.strip()) for x in args.lambdas.split(",")]

    logger.info("加载数据...")
    train_data, labels, graph_cache, mol_cache = load_data()
    logger.info(f"训练集: {len(labels)} 样本, 正={labels.sum()} ({labels.mean()*100:.1f}%)")

    # ── 预计算化学违反度 ──
    ald_smis = [d["ald"] for d in train_data]
    am_smis = [d["am"] for d in train_data]
    v_cache = ViolationCache(ald_smis, am_smis)
    summary = v_cache.violation_summary()
    logger.info(f"违反度统计: mean={summary['mean']:.4f}, "
                f"median={summary['median']:.4f}, "
                f"nonzero={summary['nonzero_frac']:.2%}")

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_results = {}
    for lam in lambda_values:
        tag = f"λ={lam}"
        logger.info(f"\n{'='*50}")
        logger.info(f"=== {tag} ===")
        logger.info(f"{'='*50}")

        fold_prs = []
        for fi, (train_idx, val_idx) in enumerate(
                kf.split(np.zeros(len(labels)), labels)):
            pr = train_fold(train_idx, val_idx, train_data, labels,
                            graph_cache, mol_cache,
                            violations=v_cache.scores, lambda_chem=lam)
            fold_prs.append(pr)
            logger.info(f"  Fold {fi+1}: PR-AUC = {pr:.4f}")

        mean_pr = np.mean(fold_prs)
        std_pr = np.std(fold_prs)
        all_results[lam] = {"mean": mean_pr, "std": std_pr, "folds": fold_prs}
        logger.info(f"{tag} PR-AUC: {mean_pr:.4f} ± {std_pr:.4f}")

    # ── 对比表 ──
    print("\n" + "=" * 65)
    print("  方案 B λ 消融 — 化学惩罚项强度对比 (3-fold CV)")
    print("=" * 65)
    print(f"\n  {'λ':12s} {'PR-AUC':14s} {'Δ vs λ=0':12s} {'各折':30s}")
    print(f"  {'-' * 60}")
    baseline_mean = all_results[0.0]["mean"]
    for lam in lambda_values:
        r = all_results[lam]
        delta = r["mean"] - baseline_mean
        folds_str = " ".join(f"{p:.4f}" for p in r["folds"])
        print(f"  {lam:<12.4f} {r['mean']:.4f} ± {r['std']:.4f}  "
              f"{delta:+.4f}        {folds_str}")

    # ── 确定最优 λ ──
    best_lam = max(all_results, key=lambda l: all_results[l]["mean"])
    print(f"\n  最优 λ: {best_lam} (PR-AUC = {all_results[best_lam]['mean']:.4f})")

    if best_lam == 0.0:
        print("  [!] 化学惩罚项未带来提升, 数据可能已充分满足化学约束")
    else:
        delta = all_results[best_lam]["mean"] - baseline_mean
        print(f"  [OK] 化学惩罚项提升 PR-AUC +{delta:.4f}")


if __name__ == "__main__":
    main()
