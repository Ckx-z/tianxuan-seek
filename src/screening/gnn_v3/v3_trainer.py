"""v3 Trainer — 训练/验证循环 + 早停 + checkpoint 保存。"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.screening.gnn_v3.v3_loss import V3Loss
from src.screening.gnn_v3.v3_scheduler import ChemWarmupScheduler


class V3Trainer:
    """v3 模型训练器。"""

    def __init__(self, model: nn.Module, loss_fn: V3Loss,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: Any = None,
                 chem_scheduler: ChemWarmupScheduler | None = None,
                 device: str = "cpu", patience: int = 30,
                 grad_clip: float = 1.0):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.chem_scheduler = chem_scheduler
        self.device = device
        self.patience = patience
        self.grad_clip = grad_clip

        self.best_state = None
        self.best_pr_auc = 0.0
        self.best_epoch = 0
        self.no_improve = 0

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        comps_sum: dict[str, float] = {}
        cl = self.chem_scheduler.get_lambda(epoch) if self.chem_scheduler else 0.0

        for batch in loader:
            batch = batch.to(self.device)
            ald_data = Data(x=batch.ald_x, edge_index=batch.ald_edge_index,
                            edge_attr=batch.ald_edge_attr)
            amine_data = Data(x=batch.amine_x, edge_index=batch.amine_edge_index,
                              edge_attr=batch.amine_edge_attr)

            self.optimizer.zero_grad()
            result = self.model(ald_data, amine_data, return_attn=True)
            film_logits, cond_logits, (ald_attn, amine_attn) = result

            loss, comps = self.loss_fn(
                film_logits, batch.film_label,
                cond_logits, batch.cond_labels, batch.cond_masks,
                quality_weights=batch.quality_weight,
                ald_attn=ald_attn, amine_attn=amine_attn,
                chem_penalty=batch.chem_violation if hasattr(batch, "chem_violation") else None,
                chem_lambda=cl,
            )

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in comps.items():
                comps_sum[k] = comps_sum.get(k, 0.0) + v.item()

        n = len(loader)
        metrics = {k: v / n for k, v in comps_sum.items()}
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["chem_lambda"] = cl
        return metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()
        all_probs, all_labels = [], []
        cl = self.chem_scheduler.get_lambda(epoch) if self.chem_scheduler else 0.0

        for batch in loader:
            batch = batch.to(self.device)
            ald_data = Data(x=batch.ald_x, edge_index=batch.ald_edge_index,
                            edge_attr=batch.ald_edge_attr)
            amine_data = Data(x=batch.amine_x, edge_index=batch.amine_edge_index,
                              edge_attr=batch.amine_edge_attr)

            film_logits, cond_logits = self.model(ald_data, amine_data)
            probs = torch.sigmoid(film_logits)

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(batch.film_label.cpu().tolist())

        return {"pr_auc": average_precision_score(all_labels, all_probs)}

    def step(self, train_loader: DataLoader, val_loader: DataLoader,
             epoch: int) -> dict[str, float]:
        train_m = self.train_epoch(train_loader, epoch)
        val_m = self.validate(val_loader, epoch)

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_m["pr_auc"])
            else:
                self.lr_scheduler.step()

        pr_auc = val_m["pr_auc"]
        if pr_auc > self.best_pr_auc:
            self.best_pr_auc = pr_auc
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(self.model.state_dict())
            self.no_improve = 0
        else:
            self.no_improve += 1

        return {**{f"train_{k}": v for k, v in train_m.items()},
                **{f"val_{k}": v for k, v in val_m.items()},
                "best_pr_auc": self.best_pr_auc, "no_improve": self.no_improve}

    def should_stop(self) -> bool:
        return self.no_improve >= self.patience

    def load_best(self):
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
