"""v5 训练主脚本 — 三标签 Focal Loss + 3D + 规则向量 (方案 B)。

Usage:
  python scripts/train_v4.py
  python scripts/train_v4.py --config config/model_v4.yaml --epochs 100
  python scripts/train_v4.py --no-3d --no-rules  # 消融
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
from src.chemistry.conformer import DESCRIPTOR_NAMES
from src.chemistry.dimer import DIMER_DESCRIPTOR_NAMES
from src.chemistry.hard_rules import get_rule_vector, RULE_DIM
from src.utils.logger import setup_logger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = setup_logger("train_v4")

_MONOMER_3D_DIM = len(DESCRIPTOR_NAMES)  # 10 per monomer
_DIMER_3D_DIM = len(DIMER_DESCRIPTOR_NAMES)  # 10 per pair


class PairDataset(torch.utils.data.Dataset):
    """醛胺配对数据集 — v5，支持 3D + 规则向量 (方案 B)。"""

    def __init__(self, csv_path: str, use_3d: bool = True, use_rules: bool = True,
                 freq_weights: dict[tuple[str, str], float] | None = None):
        with open(csv_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.use_3d = use_3d
        self.use_rules = use_rules
        self.freq_weights = freq_weights or {}
        self.graphs: list[dict | None] = []
        self.rule_vectors: list[list[float]] = []
        for r in self.rows:
            g_ald = smiles_to_graph(r["aldehyde_smiles"], role=0)
            g_amine = smiles_to_graph(r["amine_smiles"], role=1)
            self.graphs.append({"ald": g_ald, "amine": g_amine} if g_ald and g_amine else None)
            if use_rules:
                self.rule_vectors.append(get_rule_vector(r["aldehyde_smiles"], r["amine_smiles"]))

    def __len__(self):
        return len(self.rows)

    def _get_3d(self, r: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ald_3d = [float(r.get(f"ald_3d_{name}", "0.0")) for name in DESCRIPTOR_NAMES]
        amine_3d = [float(r.get(f"amine_3d_{name}", "0.0")) for name in DESCRIPTOR_NAMES]
        dimer_3d = [float(r.get(name, "0.0")) for name in DIMER_DESCRIPTOR_NAMES]
        return (torch.tensor(ald_3d, dtype=torch.float),
                torch.tensor(amine_3d, dtype=torch.float),
                torch.tensor(dimer_3d, dtype=torch.float))

    def __getitem__(self, idx):
        g = self.graphs[idx]
        r = self.rows[idx]
        if g is None:
            return None
        freq_w = self.freq_weights.get(
            (r["aldehyde_smiles"], r["amine_smiles"]), 1.0)

        item = {
            "ald_x": g["ald"].x, "ald_edge_index": g["ald"].edge_index,
            "ald_edge_attr": g["ald"].edge_attr, "ald_num_atoms": g["ald"].x.shape[0],
            "amine_x": g["amine"].x, "amine_edge_index": g["amine"].edge_index,
            "amine_edge_attr": g["amine"].edge_attr, "amine_num_atoms": g["amine"].x.shape[0],
            "film_label": torch.tensor(float(r["is_film"]), dtype=torch.float),
            "quality_weight": torch.tensor(freq_w, dtype=torch.float),
        }
        if self.use_3d:
            ald_3d, amine_3d, dimer_3d = self._get_3d(r)
            item["ald_3d"] = ald_3d
            item["amine_3d"] = amine_3d
            item["dimer_3d"] = dimer_3d
        if self.use_rules:
            item["rule_vec"] = torch.tensor(self.rule_vectors[idx], dtype=torch.float)
        return item


def collate_fn(batch: list, use_3d: bool = True, use_rules: bool = True) -> dict | None:
    """支持 batch_size > 1 的 collate — 拼接图 + 3D + 规则向量。"""
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
    result = {
        "ald_x": ald_x, "ald_edge_index": torch.cat(ald_ei, dim=1),
        "ald_edge_attr": torch.cat(ald_ea), "ald_batch": torch.cat(ald_batch_list),
        "amine_x": amine_x, "amine_edge_index": torch.cat(amine_ei, dim=1),
        "amine_edge_attr": torch.cat(amine_ea), "amine_batch": torch.cat(amine_batch_list),
        "batch_size": bs,
        "film_label": torch.stack([b["film_label"] for b in batch]),
        "quality_weight": torch.stack([b["quality_weight"] for b in batch]),
    }
    if use_3d:
        result["ald_3d"] = torch.stack([b["ald_3d"] for b in batch])
        result["amine_3d"] = torch.stack([b["amine_3d"] for b in batch])
        result["dimer_3d"] = torch.stack([b["dimer_3d"] for b in batch])
    if use_rules and "rule_vec" in batch[0]:
        result["rule_vec"] = torch.stack([b["rule_vec"] for b in batch])
    return result


def build_folds(csv_path: str, n_splits: int = 5, n_repeats: int = 3):
    """文献级 StratifiedKFold。

    增广样本（source_db 包含 'augmented'）仅参与 train fold，不参与 val fold
    以避免数据泄漏（增广从原正样本衍生，若同 paper 一起进 val fold 则虚高）。
    """
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
            train_idx = [i for pid in train_p for i in by_paper[pid]]
            val_idx = [i for pid in val_p for i in by_paper[pid]]
            # 增广样本强制从 val 中移除（不参与验证）
            val_idx = [i for i in val_idx if "augmented" not in rows[i].get("source_db", "")]
            folds.append((train_idx, val_idx))
    return folds


def _compute_freq_weights(rows: list[dict]) -> dict[tuple[str, str], float]:
    """计算每对的频率权重 — 高频单体配对降权。
    weight = 1 / sqrt(freq_ald * freq_amine) 归一化到 [0.5, 1.5]。
    """
    ald_freq: dict[str, int] = {}
    am_freq: dict[str, int] = {}
    for r in rows:
        if r["is_film"] != "1":
            continue
        ald_freq[r["aldehyde_smiles"]] = ald_freq.get(r["aldehyde_smiles"], 0) + 1
        am_freq[r["amine_smiles"]] = am_freq.get(r["amine_smiles"], 0) + 1
    max_ald = max(ald_freq.values()) if ald_freq else 1
    max_am = max(am_freq.values()) if am_freq else 1
    weights: dict[tuple[str, str], float] = {}
    for r in rows:
        af = ald_freq.get(r["aldehyde_smiles"], 1)
        amf = am_freq.get(r["amine_smiles"], 1)
        raw = 1.0 / max((af / max_ald) * (amf / max_am), 0.1)
        weights[(r["aldehyde_smiles"], r["amine_smiles"])] = min(max(raw, 0.5), 1.5)
    return weights


def _load_pretrained(model: V4Model, path: str, device: str) -> None:
    """加载预训练 encoder 权重到 model.encoder.encoder (GINEncoder)。"""
    state = torch.load(path, map_location=device)
    model.encoder.encoder.load_state_dict(state)
    logger.info(f"已加载预训练 encoder: {path}")


def main():
    parser = argparse.ArgumentParser(description="v5 训练")
    parser.add_argument("--config", type=str, default="config/model_v4.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="models/v5.0")
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument("--no-rules", action="store_true")
    parser.add_argument("--pretrained-encoder", type=str, default=None)
    args = parser.parse_args()

    use_3d = not args.no_3d
    use_rules = not args.no_rules

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tc = cfg["training"]
    max_epochs = args.epochs or tc["max_epochs"]
    batch_size = args.batch_size or tc["batch_size"]
    csv_path = cfg["data"]["train_csv"]

    if "model" not in cfg:
        cfg["model"] = {}
    cfg["model"]["use_3d"] = use_3d
    cfg["model"]["use_rules"] = use_rules
    cfg["model"]["dim_rules"] = RULE_DIM
    if use_3d:
        cfg["model"]["monomer_3d_dim"] = _MONOMER_3D_DIM
        cfg["model"]["dimer_3d_dim"] = _DIMER_3D_DIM

    # 3D 归一化
    scaler_3d = None
    scaler_dimer = None
    if use_3d:
        with open(csv_path, "r", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
        ald_3d_all = np.array([
            [float(r.get(f"ald_3d_{name}", "0.0")) for name in DESCRIPTOR_NAMES]
            for r in all_rows
        ])
        amine_3d_all = np.array([
            [float(r.get(f"amine_3d_{name}", "0.0")) for name in DESCRIPTOR_NAMES]
            for r in all_rows
        ])
        combined = np.concatenate([ald_3d_all, amine_3d_all], axis=0)
        mean_3d = combined.mean(axis=0)
        std_3d = np.where(combined.std(axis=0) > 1e-8, combined.std(axis=0), 1.0)
        scaler_3d = {"mean": mean_3d.tolist(), "std": std_3d.tolist()}
        dimer_all = np.array([
            [float(r.get(name, "0.0")) for name in DIMER_DESCRIPTOR_NAMES]
            for r in all_rows
        ])
        mean_dimer = dimer_all.mean(axis=0)
        std_dimer = np.where(dimer_all.std(axis=0) > 1e-8, dimer_all.std(axis=0), 1.0)
        scaler_dimer = {"mean": mean_dimer.tolist(), "std": std_dimer.tolist()}

    logger.info(f"加载数据: {csv_path} (3D={'ON' if use_3d else 'OFF'}, rules={'ON' if use_rules else 'OFF'})")
    with open(csv_path, "r", encoding="utf-8") as f:
        all_rows_for_freq = list(csv.DictReader(f))
    freq_weights = _compute_freq_weights(all_rows_for_freq)
    full_ds = PairDataset(csv_path, use_3d=use_3d, use_rules=use_rules,
                          freq_weights=freq_weights)
    logger.info(f"总样本: {len(full_ds)}")

    ec = cfg["evaluation"]
    folds = build_folds(csv_path, ec["cv_folds"], ec["cv_repeats"])
    logger.info(f"Folds: {len(folds)} ({ec['cv_folds']}x{ec['cv_repeats']})")

    tmp_model = V4Model(cfg)
    n_params = sum(p.numel() for p in tmp_model.parameters())
    logger.info(f"模型参数量: {n_params:,} ({n_params/1e6:.2f}M)")

    _collate = lambda b: collate_fn(b, use_3d=use_3d, use_rules=use_rules)

    fold_pr_aucs = []
    for fi, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"=== Fold {fi+1}/{len(folds)} ===")

        train_ds = torch.utils.data.Subset(full_ds, train_idx)
        val_ds = torch.utils.data.Subset(full_ds, val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=_collate, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=_collate)

        model = V4Model(cfg).to(args.device)
        if args.pretrained_encoder:
            _load_pretrained(model, args.pretrained_encoder, args.device)
        if scaler_3d and scaler_dimer:
            model.set_3d_scaler(monomer_mean=scaler_3d["mean"],
                               monomer_std=scaler_3d["std"],
                               dimer_mean=scaler_dimer["mean"],
                               dimer_std=scaler_dimer["std"])
        loss_fn = FocalLoss(cfg["loss"]["focal_alpha"], cfg["loss"]["focal_gamma"])
        opt = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"],
                                weight_decay=tc["weight_decay"])
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max_epochs, eta_min=1e-6)

        trainer = V4Trainer(model, loss_fn, opt, lr_sched,
                            device=args.device, patience=tc["early_stop_patience"],
                            grad_clip=tc["grad_clip"],
                            max_epochs=max_epochs)

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

    # 最佳 fold 重训
    logger.info("用最佳 fold 重新训练...")
    train_idx, val_idx = folds[best_fold]
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=_collate)

    model = V4Model(cfg).to(args.device)
    if args.pretrained_encoder:
        _load_pretrained(model, args.pretrained_encoder, args.device)
    if scaler_3d and scaler_dimer:
        model.set_3d_scaler(monomer_mean=scaler_3d["mean"],
                           monomer_std=scaler_3d["std"],
                           dimer_mean=scaler_dimer["mean"],
                           dimer_std=scaler_dimer["std"])
    loss_fn = FocalLoss(cfg["loss"]["focal_alpha"], cfg["loss"]["focal_gamma"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"],
                            weight_decay=tc["weight_decay"])
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    trainer = V4Trainer(model, loss_fn, opt, lr_sched,
                        device=args.device, patience=tc["early_stop_patience"],
                        grad_clip=tc["grad_clip"],
                        max_epochs=max_epochs)
    for epoch in range(max_epochs):
        metrics = trainer.step(train_loader, val_loader, epoch)
        if epoch % 10 == 0 or trainer.should_stop():
            logger.info(
                f"  重训 E{epoch:3d} loss={metrics.get('train_loss', 0):.4f} "
                f"val_pr_auc={metrics['val_pr_auc']:.4f} "
                f"best={metrics['best_pr_auc']:.4f}"
            )
        if trainer.should_stop():
            logger.info(f"  重训早停 @ {epoch}")
            break
    trainer.load_best()
    logger.info(f"重训完成, best PR-AUC: {trainer.best_pr_auc:.4f}")

    save_path = os.path.join(args.output, "v5_model.pt")
    torch.save({"model_state": trainer.best_state, "config": cfg,
                "fold_pr_aucs": fold_pr_aucs, "best_fold": best_fold,
                "scaler_3d": scaler_3d, "scaler_dimer": scaler_dimer,
                "use_3d": use_3d, "use_rules": use_rules}, save_path)
    logger.info(f"模型已保存: {save_path}")


if __name__ == "__main__":
    main()
