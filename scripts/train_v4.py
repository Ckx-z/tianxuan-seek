"""v4 训练主脚本 — 缩小模型 + batch 支持 + 纯 Focal Loss。

Usage:
  python scripts/train_v4.py
  python scripts/train_v4.py --config config/model_v4.yaml --epochs 100
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
from rdkit import RDLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn_v4.model import V4Model
from src.screening.gnn_v4.v4_loss import FocalLoss
from src.screening.gnn_v4.v4_trainer import V4Trainer
from src.screening.gnn_v3.featurizer import smiles_to_graph
from src.utils.logger import setup_logger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = setup_logger("train_v4")

class PairDataset(torch.utils.data.Dataset):
    """醛胺配对数据集 — v4 版本，支持 batch 向量。"""

    def __init__(self, csv_path: str):
        with open(csv_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.graphs: list[dict | None] = []
        for r in self.rows:
            g_ald = smiles_to_graph(r["aldehyde_smiles"], role=0)
            g_amine = smiles_to_graph(r["amine_smiles"], role=1)
            self.graphs.append({"ald": g_ald, "amine": g_amine} if g_ald and g_amine else None)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        r = self.rows[idx]
        if g is None:
            return None
        return {
            "ald_x": g["ald"].x, "ald_edge_index": g["ald"].edge_index,
            "ald_edge_attr": g["ald"].edge_attr, "ald_num_atoms": g["ald"].x.shape[0],
            "amine_x": g["amine"].x, "amine_edge_index": g["amine"].edge_index,
            "amine_edge_attr": g["amine"].edge_attr, "amine_num_atoms": g["amine"].x.shape[0],
            "film_label": torch.tensor(float(r["is_film"]), dtype=torch.float),
            "quality_weight": torch.tensor(float(r.get("quality_weight", "1.0")), dtype=torch.float),
        }


def collate_fn(batch: list) -> dict | None:
    """支持 batch_size > 1 的 collate — 拼接图 + 构建 batch 向量。"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    bs = len(batch)
    ald_x = torch.cat([b["ald_x"] for b in batch])
    amine_x = torch.cat([b["amine_x"] for b in batch])
    ald_ei, amine_ei = [], []
    ald_ea, amine_ea = [], []
    ald_batch_list, amine_batch_list = [], []
    ald_off, amine_off = 0, 0
    for i, b in enumerate(batch):
        ald_ei.append(b["ald_edge_index"] + ald_off)
        amine_ei.append(b["amine_edge_index"] + amine_off)
        ald_ea.append(b["ald_edge_attr"])
        amine_ea.append(b["amine_edge_attr"])
        ald_batch_list.append(torch.full((b["ald_num_atoms"],), i, dtype=torch.long))
        amine_batch_list.append(torch.full((b["amine_num_atoms"],), i, dtype=torch.long))
        ald_off += b["ald_num_atoms"]
        amine_off += b["amine_num_atoms"]
    return {
        "ald_x": ald_x, "ald_edge_index": torch.cat(ald_ei, dim=1),
        "ald_edge_attr": torch.cat(ald_ea), "ald_batch": torch.cat(ald_batch_list),
        "amine_x": amine_x, "amine_edge_index": torch.cat(amine_ei, dim=1),
        "amine_edge_attr": torch.cat(amine_ea), "amine_batch": torch.cat(amine_batch_list),
        "batch_size": bs,
        "film_label": torch.stack([b["film_label"] for b in batch]),
        "quality_weight": torch.stack([b["quality_weight"] for b in batch]),
    }


def build_folds(csv_path: str, n_splits: int = 5, n_repeats: int = 3):
    """文献级 StratifiedKFold。"""
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_paper: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_paper.setdefault(r["paper_id"], []).append(i)

    paper_ids = list(by_paper.keys())
    paper_labels = np.array([
        int(any(rows[i]["is_film"] == "1" for i in by_paper[pid]))
        for pid in paper_ids
    ])

    folds = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + rep)
        for tp, vp in skf.split(paper_ids, paper_labels):
            train_p = {paper_ids[i] for i in tp}
            val_p = {paper_ids[i] for i in vp}
            folds.append((
                [i for pid in train_p for i in by_paper[pid]],
                [i for pid in val_p for i in by_paper[pid]],
            ))
    return folds


def main():
    parser = argparse.ArgumentParser(description="v4 训练")
    parser.add_argument("--config", type=str, default="config/model_v4.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="models/v4.0")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tc = cfg["training"]
    max_epochs = args.epochs or tc["max_epochs"]
    batch_size = args.batch_size or tc["batch_size"]
    csv_path = cfg["data"]["train_csv"]

    logger.info(f"加载数据: {csv_path}")
    full_ds = PairDataset(csv_path)
    logger.info(f"总样本: {len(full_ds)}")

    ec = cfg["evaluation"]
    folds = build_folds(csv_path, ec["cv_folds"], ec["cv_repeats"])
    logger.info(f"Folds: {len(folds)} ({ec['cv_folds']}x{ec['cv_repeats']})")

    tmp_model = V4Model(cfg)
    n_params = sum(p.numel() for p in tmp_model.parameters())
    logger.info(f"模型参数量: {n_params:,} ({n_params/1e6:.2f}M)")

    fold_pr_aucs = []
    for fi, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"=== Fold {fi+1}/{len(folds)} ===")

        train_ds = torch.utils.data.Subset(full_ds, train_idx)
        val_ds = torch.utils.data.Subset(full_ds, val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn)

        model = V4Model(cfg).to(args.device)
        loss_fn = FocalLoss(cfg["loss"]["focal_alpha"], cfg["loss"]["focal_gamma"])
        opt = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"],
                                weight_decay=tc["weight_decay"])
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max_epochs, eta_min=1e-6)

        trainer = V4Trainer(model, loss_fn, opt, lr_sched,
                            device=args.device, patience=tc["early_stop_patience"],
                            grad_clip=tc["grad_clip"])

        for epoch in range(max_epochs):
            metrics = trainer.step(train_loader, val_loader, epoch)
            if epoch % 10 == 0 or trainer.should_stop():
                logger.info(
                    f"  E{epoch:3d} loss={metrics.get('train_loss', 0):.4f} "
                    f"val_pr_auc={metrics['val_pr_auc']:.4f} "
                    f"best={metrics['best_pr_auc']:.4f}"
                )
            if trainer.should_stop():
                logger.info(f"  早停 @ {epoch}")
                break

        trainer.load_best()
        fold_pr_aucs.append(trainer.best_pr_auc)
        logger.info(f"  Fold {fi} PR-AUC: {trainer.best_pr_auc:.4f}")

    logger.info(f"CV PR-AUC: {np.mean(fold_pr_aucs):.4f} +/- {np.std(fold_pr_aucs):.4f}")

    os.makedirs(args.output, exist_ok=True)
    best_fold = int(np.argmax(fold_pr_aucs))
    logger.info(f"最佳 fold: {best_fold} (PR-AUC={fold_pr_aucs[best_fold]:.4f})")

    # 用最佳 fold 重新训练保存
    train_idx, val_idx = folds[best_fold]
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    model = V4Model(cfg).to(args.device)
    loss_fn = FocalLoss(cfg["loss"]["focal_alpha"], cfg["loss"]["focal_gamma"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"],
                            weight_decay=tc["weight_decay"])
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    trainer = V4Trainer(model, loss_fn, opt, lr_sched,
                        device=args.device, patience=tc["early_stop_patience"],
                        grad_clip=tc["grad_clip"])
    for epoch in range(max_epochs):
        trainer.step(train_loader, val_loader, epoch)
        if trainer.should_stop():
            break
    trainer.load_best()

    save_path = os.path.join(args.output, "v4_model.pt")
    torch.save({"model_state": trainer.best_state, "config": cfg,
                "fold_pr_aucs": fold_pr_aucs, "best_fold": best_fold}, save_path)
    logger.info(f"模型已保存: {save_path}")


if __name__ == "__main__":
    main()
