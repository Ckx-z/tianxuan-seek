"""v3 训练主脚本 — 数据加载 + 文献级 GroupKFold + 训练循环。

Usage:
  python scripts/train_v3.py
  python scripts/train_v3.py --config config/model_v3.yaml --epochs 100
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
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn_v3.model import V3Model
from src.screening.gnn_v3.v3_loss import V3Loss
from src.screening.gnn_v3.v3_scheduler import (
    ChemWarmupScheduler, build_optimizer, build_lr_scheduler,
)
from src.screening.gnn_v3.v3_trainer import V3Trainer
from src.screening.gnn_v3.featurizer import smiles_to_graph
from src.chemistry.chem_penalty import ViolationCache
from src.utils.logger import setup_logger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = setup_logger("train_v3")

COND_LABEL_MAPS = {
    "synthesis_route": {"solvothermal": 0, "interfacial": 1, "mechanochemical": 2,
                        "room-temperature": 3, "other": 4},
    "interface_type": {"liquid-solid": 0, "liquid-liquid": 1, "other": 2},
    "catalyst": {"acetic_acid": 0, "PTSA": 1, "TFA": 2, "Sc(OTf)3_Lewis": 3, "none_other": 4},
    "solvent": {"dioxane_mesitylene": 0, "DCB_BuOH": 1, "water_based": 2,
                "DMF_DMSO": 3, "acetonitrile": 4, "alcohol": 5, "other": 6},
    "temperature": {"rt": 0, "low": 1, "standard": 2, "high": 3, "unknown": 4},
}
COND_TASKS = list(COND_LABEL_MAPS.keys())


class PairDataset(torch.utils.data.Dataset):
    """醛胺配对数据集。"""

    def __init__(self, csv_path: str, vcache: ViolationCache | None = None):
        with open(csv_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.vcache = vcache
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

        cond_labels, cond_masks = {}, {}
        for task in COND_TASKS:
            raw = r.get(f"{task}_label", "")
            lm = COND_LABEL_MAPS[task]
            if raw in lm:
                cond_labels[task] = lm[raw]
                cond_masks[task] = 1.0
            else:
                cond_labels[task] = 0
                cond_masks[task] = 0.0

        cv = self.vcache.scores[idx] if self.vcache else 0.0

        return {
            "ald_x": g["ald"].x, "ald_edge_index": g["ald"].edge_index,
            "ald_edge_attr": g["ald"].edge_attr,
            "amine_x": g["amine"].x, "amine_edge_index": g["amine"].edge_index,
            "amine_edge_attr": g["amine"].edge_attr,
            "film_label": torch.tensor(float(r["is_film"]), dtype=torch.float),
            "quality_weight": torch.tensor(float(r.get("quality_weight", 1.0)), dtype=torch.float),
            "cond_labels": {k: torch.tensor(v, dtype=torch.long) for k, v in cond_labels.items()},
            "cond_masks": {k: torch.tensor(v, dtype=torch.float) for k, v in cond_masks.items()},
            "chem_violation": torch.tensor(cv, dtype=torch.float),
        }


def collate_fn(batch: list) -> dict | None:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    ald_x = torch.cat([b["ald_x"] for b in batch])
    amine_x = torch.cat([b["amine_x"] for b in batch])
    ald_ei, amine_ei = [], []
    ald_ea, amine_ea = [], []
    ald_off, amine_off = 0, 0
    for b in batch:
        ald_ei.append(b["ald_edge_index"] + ald_off)
        amine_ei.append(b["amine_edge_index"] + amine_off)
        ald_ea.append(b["ald_edge_attr"])
        amine_ea.append(b["amine_edge_attr"])
        ald_off += b["ald_x"].shape[0]
        amine_off += b["amine_x"].shape[0]

    return {
        "ald_x": ald_x, "ald_edge_index": torch.cat(ald_ei, dim=1),
        "ald_edge_attr": torch.cat(ald_ea),
        "amine_x": amine_x, "amine_edge_index": torch.cat(amine_ei, dim=1),
        "amine_edge_attr": torch.cat(amine_ea),
        "film_label": torch.stack([b["film_label"] for b in batch]),
        "quality_weight": torch.stack([b["quality_weight"] for b in batch]),
        "cond_labels": {k: torch.stack([b["cond_labels"][k] for b in batch]) for k in COND_TASKS},
        "cond_masks": {k: torch.stack([b["cond_masks"][k] for b in batch]) for k in COND_TASKS},
        "chem_violation": torch.stack([b["chem_violation"] for b in batch]),
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
    parser = argparse.ArgumentParser(description="v3 训练")
    parser.add_argument("--config", type=str, default="config/model_v3.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="models/v3.0")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tc = cfg["training"]
    max_epochs = args.epochs or tc["max_epochs"]
    batch_size = args.batch_size or tc["batch_size"]
    csv_path = cfg["data"]["train_csv"]

    # 化学违反度缓存
    logger.info("预计算化学违反度...")
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    vcache = ViolationCache([r["aldehyde_smiles"] for r in rows],
                            [r["amine_smiles"] for r in rows])
    logger.info(f"违反度 mean={vcache.mean_violation():.4f}")

    # Folds
    ec = cfg["evaluation"]
    folds = build_folds(csv_path, ec["cv_folds"], ec["cv_repeats"])
    logger.info(f"Folds: {len(folds)} ({ec['cv_folds']}x{ec['cv_repeats']})")

    fold_pr_aucs = []
    for fi, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"=== Fold {fi+1}/{len(folds)} ===")

        train_ds = torch.utils.data.Subset(PairDataset(csv_path, vcache), train_idx)
        val_ds = torch.utils.data.Subset(PairDataset(csv_path, vcache), val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn)

        model = V3Model(cfg).to(args.device)
        lc = cfg["loss"]
        loss_fn = V3Loss(lc["focal_alpha"], lc["focal_gamma"],
                         lc["cond_lambda"], lc["sym_lambda"],
                         lc["use_film_quality_weight"])
        opt = build_optimizer(model.parameters(), tc["learning_rate"],
                              tc["weight_decay"], tc["optimizer"])
        lr_sched = build_lr_scheduler(opt, max_epochs, tc["warmup_epochs"], tc["scheduler"])
        wc = cfg["chem_warmup"]
        chem_sched = ChemWarmupScheduler(max_epochs, wc["warmup_frac"],
                                         wc["ramp_frac"], wc["hold_frac"],
                                         lc["chem_lambda_max"])

        trainer = V3Trainer(model, loss_fn, opt, lr_sched, chem_sched,
                            device=args.device, patience=tc["early_stop_patience"],
                            grad_clip=tc["grad_clip"])

        for epoch in range(max_epochs):
            metrics = trainer.step(train_loader, val_loader, epoch)
            if epoch % 10 == 0 or trainer.should_stop():
                logger.info(
                    f"  E{epoch:3d} train_film={metrics.get('train_film',0):.4f} "
                    f"val_pr_auc={metrics['val_pr_auc']:.4f} "
                    f"best={metrics['best_pr_auc']:.4f} "
                    f"cl={metrics.get('train_chem_lambda',0):.4f}"
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
    logger.info(f"保存最佳 fold: {best_fold} (PR-AUC={fold_pr_aucs[best_fold]:.4f})")
    # 重新训练最佳 fold 并保存
    best_train_idx, best_val_idx = folds[best_fold]
    best_train_ds = torch.utils.data.Subset(PairDataset(csv_path, vcache), best_train_idx)
    best_train_loader = DataLoader(best_train_ds, batch_size=batch_size, shuffle=True,
                                    collate_fn=collate_fn)

    save_model = V3Model(cfg).to(args.device)
    save_opt = build_optimizer(save_model.parameters(), tc["learning_rate"],
                                tc["weight_decay"], tc["optimizer"])
    save_lr = build_lr_scheduler(save_opt, max_epochs, tc["warmup_epochs"], tc["scheduler"])
    save_chem = ChemWarmupScheduler(max_epochs, wc["warmup_frac"],
                                     wc["ramp_frac"], wc["hold_frac"],
                                     lc["chem_lambda_max"])
    save_trainer = V3Trainer(save_model, loss_fn, save_opt, save_lr, save_chem,
                              device=args.device, patience=tc["early_stop_patience"],
                              grad_clip=tc["grad_clip"])
    # 用最佳 fold 的训练集+验证集一起训练
    best_val_loader = DataLoader(best_train_ds, batch_size=batch_size, shuffle=False,
                                  collate_fn=collate_fn)
    best_epochs = max(fold_pr_aucs)  # 用最佳 PR-AUC 对应的 epoch 数
    for epoch in range(max_epochs):
        save_trainer.train_epoch(best_train_loader, epoch)
        if epoch % 10 == 0:
            val_m = save_trainer.validate(best_val_loader, epoch)
            logger.info(f"  全量训练 E{epoch}: val_pr_auc={val_m['pr_auc']:.4f}")

    torch.save(save_trainer.model.state_dict(), os.path.join(args.output, "v3_model.pt"))
    logger.info(f"模型已保存: {args.output}/v3_model.pt")


if __name__ == "__main__":
    main()
