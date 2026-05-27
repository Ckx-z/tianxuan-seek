"""v3 Benchmark 评估 — 商业单体独立测试集。

Usage:
  python scripts/eval_v3.py --model models/v3.0/v3_model.pt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
import yaml
from rdkit import RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn_v3.model import V3Model
from src.screening.gnn_v3.featurizer import smiles_to_graph
from src.utils.logger import setup_logger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = setup_logger("eval_v3")


def main():
    parser = argparse.ArgumentParser(description="v3 Benchmark 评估")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/model_v3.yaml")
    parser.add_argument("--benchmark", type=str,
                        default="data/processed/benchmark_pairs.csv")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = V3Model(cfg).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device,
                          weights_only=True))
    model.eval()

    if not os.path.exists(args.benchmark):
        logger.warning(f"Benchmark 不存在: {args.benchmark}")
        return

    with open(args.benchmark, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    logger.info(f"Benchmark: {len(rows)} 对")

    probs, labels = [], []
    for r in rows:
        g_ald = smiles_to_graph(r["aldehyde_smiles"], role=0)
        g_amine = smiles_to_graph(r["amine_smiles"], role=1)
        if g_ald is None or g_amine is None:
            continue
        g_ald = g_ald.to(args.device)
        g_amine = g_amine.to(args.device)
        with torch.no_grad():
            logit = model.predict(g_ald, g_amine)
        probs.append(torch.sigmoid(logit).item())
        labels.append(float(r.get("is_film", 0)))

    if not probs:
        logger.warning("无可评估样本")
        return

    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    logger.info(f"PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    paired = sorted(zip(probs, labels), key=lambda x: x[0], reverse=True)
    for k in [20, 40, 100]:
        hit = sum(1 for _, l in paired[:k] if l == 1)
        logger.info(f"  Top {k}: {hit}/{k}")


if __name__ == "__main__":
    main()
