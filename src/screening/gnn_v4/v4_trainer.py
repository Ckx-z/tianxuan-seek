"""v4 Trainer — 简化版训练器，纯 Focal Loss，支持 batch。"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from src.screening.gnn_v4.v4_loss import FocalLoss


class V4Trainer:
    """v4 模型训练器 — 纯成膜预测，无多任务。"""

    def __init__(self, model: nn.Module, loss_fn: FocalLoss,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: Any = None,
                 device: str = "cpu", patience: int = 30,
                 grad_clip: float = 1.0):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.patience = patience
        self.grad_clip = grad_clip

        self.best_state = None
        self.best_pr_auc = 0.0
        self.best_epoch = 0
        self.no_improve = 0

    def _to_device(self, batch: dict) -> tuple[Data, Data, torch.Tensor,
                                                 torch.Tensor, torch.Tensor,
                                                 torch.Tensor, int]:
        ald_data = Data(
            x=batch["ald_x"].to(self.device),
            edge_index=batch["ald_edge_index"].to(self.device),
            edge_attr=batch["ald_edge_attr"].to(self.device),
        )
        amine_data = Data(
            x=batch["amine_x"].to(self.device),
            edge_index=batch["amine_edge_index"].to(self.device),
            edge_attr=batch["amine_edge_attr"].to(self.device),
        )
        ald_batch = batch["ald_batch"].to(self.device)
        amine_batch = batch["amine_batch"].to(self.device)
        batch_size = batch["batch_size"]
        film_label = batch["film_label"].to(self.device)
        quality_weight = batch.get("quality_weight")
        if quality_weight is not None:
            quality_weight = quality_weight.to(self.device)
        return ald_data, amine_data, ald_batch, amine_batch, film_label, quality_weight, batch_size

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            ald_data, amine_data, ald_batch, amine_batch, film_label, qw, bs = self._to_device(batch)

            self.optimizer.zero_grad()
            logits = self.model(ald_data, amine_data, ald_batch, amine_batch, bs)
            loss = self.loss_fn(logits, film_label, qw)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

        return {"loss": total_loss / len(loader),
                "lr": self.optimizer.param_groups[0]["lr"]}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_probs, all_labels = [], []

        for batch in loader:
            ald_data, amine_data, ald_batch, amine_batch, film_label, qw, bs = self._to_device(batch)
            logits = self.model(ald_data, amine_data, ald_batch, amine_batch, bs)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(film_label.cpu().tolist())

        pr_auc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        return {"pr_auc": pr_auc}

    def step(self, train_loader: DataLoader, val_loader: DataLoader,
             epoch: int) -> dict[str, float]:
        train_m = self.train_epoch(train_loader)
        val_m = self.validate(val_loader)

        if self.lr_scheduler is not None:
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
