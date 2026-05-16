"""方案 A 快速训练 — 单模型 (80/20) 用于筛选对比, 非 CV 评估。

用法:
  python scripts/train_pair_predictor_quick.py                        # λ=0.005
  python scripts/train_pair_predictor_quick.py --lambda-chem 0        # 基线
  python scripts/train_pair_predictor_quick.py --lambda-chem 0.01
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
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.linker_analyzer import (
    has_acetylene, compute_monomer_descriptors, compute_pair_descriptor_vector)
from src.chemistry.chem_penalty import ViolationCache
from src.screening.gnn import (MoleculeEncoder, smiles_to_graph,
                               collate_graphs, extract_monomer_embeddings)
from scripts.train_pair_predictor_a import (EndToEndModel, BilinearHead,
                                            FocalLoss, ranking_loss)
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)
logger = setup_logger("pair_quick")

HIDDEN = 256
DEVICE = "cpu"
SEED = 42
EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 16
RANKING_WEIGHT = 0.005

torch.manual_seed(SEED)
np.random.seed(SEED)


def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else smi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-chem", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", type=str, default="models/v2.0/end_to_end_a_lambda.pt")
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

    # ── 化学违反度缓存 ──
    v_cache = None
    if args.lambda_chem > 0:
        ald_smis = [d["ald"] for d in train_data]
        am_smis = [d["am"] for d in train_data]
        v_cache = ViolationCache(ald_smis, am_smis)
        summary = v_cache.violation_summary()
        logger.info(f"违反度: mean={summary['mean']:.4f}, nonzero={summary['nonzero_frac']:.2%}")

    # ── 构建缓存 ──
    all_smis = set()
    for d in train_data:
        all_smis.add(d["ald"]); all_smis.add(d["am"])
    graph_cache = {}
    for smi in all_smis:
        g = smiles_to_graph(smi)
        if g is not None:
            graph_cache[smi] = g
    logger.info(f"图缓存: {len(graph_cache)}/{len(all_smis)}")

    mol_cache = {}
    for smi in all_smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol_cache[smi] = mol

    # ── 80/20 分层分割 ──
    idx = np.arange(len(labels))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=SEED)
    logger.info(f"训练: {len(tr_idx)}, 验证: {len(va_idx)}")

    # 过滤有效样本
    def _valid_indices(indices):
        valid = []
        for i in indices:
            d = train_data[i]
            if (graph_cache.get(d["ald"]) is not None and
                    graph_cache.get(d["am"]) is not None and
                    mol_cache.get(d["ald"]) is not None and
                    mol_cache.get(d["am"]) is not None):
                valid.append(i)
        return valid

    tr_idx = _valid_indices(tr_idx)
    va_idx = _valid_indices(va_idx)

    t_ald = [graph_cache[train_data[i]["ald"]] for i in tr_idx]
    t_am = [graph_cache[train_data[i]["am"]] for i in tr_idx]
    t_y = labels[tr_idx]

    # 机理描述符
    t_extra = np.zeros((len(tr_idx), 26), dtype=np.float32)
    t_ald_acet = np.zeros(len(tr_idx), dtype=bool)
    t_am_acet = np.zeros(len(tr_idx), dtype=bool)
    for j, i in enumerate(tr_idx):
        ma = mol_cache[train_data[i]["ald"]]
        mb = mol_cache[train_data[i]["am"]]
        t_extra[j] = compute_pair_descriptor_vector(ma, mb)
        t_ald_acet[j] = has_acetylene(ma)
        t_am_acet[j] = has_acetylene(mb)

    # ── 构建模型 ──
    encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.2)
    pretrained_path = "models/v2.0/gnn_encoder_finetuned_v4.pt"
    if os.path.exists(pretrained_path):
        encoder.load_state_dict(
            torch.load(pretrained_path, map_location="cpu"), strict=False)
        logger.info("编码器: v4 预训练权重加载")

    head = BilinearHead(hidden=HIDDEN, bilinear_rank=64, mlp_hidden=128,
                        dropout=0.4, extra_dim=26)
    plan_b_path = "models/v2.0/pair_predictor_b.pt"
    if os.path.exists(plan_b_path):
        plan_b_state = torch.load(plan_b_path, map_location="cpu")
        head_state = {k: v for k, v in plan_b_state.items()
                      if k in head.state_dict()
                      and v.shape == head.state_dict()[k].shape}
        head.load_state_dict(head_state, strict=False)
        logger.info("分类头: 方案 B 权重暖启动")

    model = EndToEndModel(encoder, head)
    alpha = labels.sum() / len(labels)
    criterion = FocalLoss(alpha=1 - alpha, gamma=2.0)

    opt = AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=2e-4)
    sch = CosineAnnealingLR(opt, T_max=args.epochs)

    # ── 训练 ──
    best_pr, best_state, patience = 0.0, None, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        n = len(t_ald)
        perm = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            bi = perm[start:start + BATCH_SIZE]
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
            loss = focal + RANKING_WEIGHT * rank
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

        # 验证
        if ep % 5 == 0 or ep == 1:
            model.eval()
            v_probs, v_labels = [], []
            with torch.no_grad():
                for i in va_idx:
                    d = train_data[i]
                    ga = graph_cache[d["ald"]]
                    gb = graph_cache[d["am"]]
                    ma = mol_cache[d["ald"]]
                    mb = mol_cache[d["am"]]
                    dp = compute_pair_descriptor_vector(ma, mb)
                    logit = model(collate_graphs([ga]), collate_graphs([gb]),
                                  torch.tensor([dp], dtype=torch.float))
                    v_probs.append(torch.sigmoid(logit).item())
                    v_labels.append(d["label"])
            val_pr = average_precision_score(v_labels, v_probs)
            logger.info(f"  ep {ep:3d}: val_PR={val_pr:.4f}")

            if val_pr > best_pr + 1e-4:
                best_pr = val_pr
                best_state = deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= PATIENCE:
                logger.info(f"  early stop @ ep {ep}, best_PR={best_pr:.4f}")
                break

    # ── 保存 ──
    if best_state is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        # 将 λ 写入文件名
        out_path = args.output.replace(".pt", f"_{args.lambda_chem:.4f}.pt")
        torch.save(best_state, out_path)
        logger.info(f"模型已保存: {out_path} (best PR={best_pr:.4f})")

    print(f"\nλ={args.lambda_chem} 训练完成, best val PR-AUC={best_pr:.4f}")


if __name__ == "__main__":
    main()
