"""v3 筛选脚本 — 单体池 + 硬约束 + GNN 推理 + 多样性排序 + Top 40 + Bottom 10。

Usage:
  python scripts/screen_v3.py --model models/v3.0/v3_model.pt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, asdict

import torch
import yaml
from rdkit import RDLogger
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn_v3.model import V3Model
from src.screening.gnn_v3.featurizer import smiles_to_graph
from src.utils.logger import setup_logger
from scripts.pairing import load_monomer_pool, generate_pairs

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = setup_logger("screen_v3")


@dataclass
class ScreeningResult:
    aldehyde_smiles: str
    amine_smiles: str
    aldehyde_name: str
    amine_name: str
    gnn_prob: float
    gnn_logit: float
    pred_label: int
    hard_pass: bool
    topology: str
    aldehyde_source: str
    amine_source: str
    in_training_set: bool
    aldehyde_fg: int
    amine_fg: int


def diverse_top_k(results: list[ScreeningResult], k: int = 40,
                  max_per_monomer: int = 3) -> list[ScreeningResult]:
    results = sorted(results, key=lambda x: x.gnn_prob, reverse=True)
    selected = []
    ald_cnt: dict[str, int] = {}
    amine_cnt: dict[str, int] = {}
    for r in results:
        if not r.hard_pass:
            continue
        if ald_cnt.get(r.aldehyde_smiles, 0) >= max_per_monomer:
            continue
        if amine_cnt.get(r.amine_smiles, 0) >= max_per_monomer:
            continue
        selected.append(r)
        ald_cnt[r.aldehyde_smiles] = ald_cnt.get(r.aldehyde_smiles, 0) + 1
        amine_cnt[r.amine_smiles] = amine_cnt.get(r.amine_smiles, 0) + 1
        if len(selected) >= k:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description="v3 筛选")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/model_v3.yaml")
    parser.add_argument("--train-csv", type=str, default="data/processed/v3_train.csv")
    parser.add_argument("--commercial-csv", type=str,
                        default="data/processed/commercial_monomers_classified.csv")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--output", type=str, default="data/processed/v3_screening")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. 单体池
    logger.info("构建单体池...")
    aldehydes, amines = load_monomer_pool(args.train_csv, args.commercial_csv)
    logger.info(f"醛: {len(aldehydes)}, 胺: {len(amines)}")

    # 训练集已有配对
    train_pairs = set()
    with open(args.train_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            train_pairs.add((r["aldehyde_smiles"].strip(), r["amine_smiles"].strip()))

    # 2. 生成配对 + 硬约束
    logger.info("生成候选配对...")
    pairs = generate_pairs(aldehydes, amines, train_pairs)
    n_pass = sum(1 for p in pairs if p["hard_pass"])
    logger.info(f"候选: {len(pairs)} (通过硬约束: {n_pass})")

    # 3. GNN 推理
    logger.info("GNN 推理...")
    model = V3Model(cfg).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device, weights_only=True))
    model.eval()

    results = []
    batch_size = 32
    valid = [p for p in pairs if p["hard_pass"]]

    for i in range(0, len(valid), batch_size):
        batch = valid[i:i + batch_size]
        for p in batch:
            g_ald = smiles_to_graph(p["ald"].smiles, role=0)
            g_amine = smiles_to_graph(p["amine"].smiles, role=1)
            if g_ald is None or g_amine is None:
                continue
            g_ald = g_ald.to(args.device)
            g_amine = g_amine.to(args.device)
            with torch.no_grad():
                logit = model.predict(g_ald, g_amine)
            prob = torch.sigmoid(logit).item()
            results.append(ScreeningResult(
                aldehyde_smiles=p["ald"].smiles,
                amine_smiles=p["amine"].smiles,
                aldehyde_name=p["ald"].name,
                amine_name=p["amine"].name,
                gnn_prob=round(prob, 6),
                gnn_logit=round(logit.item(), 6),
                pred_label=1 if prob > 0.5 else 0,
                hard_pass=p["hard_pass"],
                topology=p["topology"],
                aldehyde_source=p["ald"].source,
                amine_source=p["amine"].source,
                in_training_set=p["in_training_set"],
                aldehyde_fg=p["ald"].n_fg,
                amine_fg=p["amine"].n_fg,
            ))
        if (i // batch_size) % 50 == 0:
            logger.info(f"  推理进度: {min(i + batch_size, len(valid))}/{len(valid)}")

    # 4. 多样性排序
    top_k = diverse_top_k(results, k=args.top)
    bottom_k = sorted(results, key=lambda x: x.gnn_prob)[:10]

    # 5. 输出
    os.makedirs(args.output, exist_ok=True)

    fields = list(asdict(top_k[0]).keys())
    top_path = os.path.join(args.output, f"Top{args.top}_v3.csv")
    with open(top_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in top_k:
            w.writerow(asdict(r))
    logger.info(f"Top {args.top}: {top_path}")

    bottom_path = os.path.join(args.output, "Bottom10_v3.csv")
    with open(bottom_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in bottom_k:
            w.writerow(asdict(r))
    logger.info(f"Bottom 10: {bottom_path}")

    top_train = sum(1 for r in top_k if r.in_training_set)
    top_comm = sum(1 for r in top_k
                   if r.aldehyde_source == "commercial" or r.amine_source == "commercial")
    logger.info(f"Top {args.top}: {top_train} 训练集已有, {top_comm} 含商业单体")


if __name__ == "__main__":
    main()
