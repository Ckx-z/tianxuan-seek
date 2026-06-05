"""Step 2 v5: GNN 推理 — 去掉 adj 归零，规则向量传入模型。
GNN 自己学会硬规则，输出概率直接作为最终分数。

输入 : data/processed/v4_cartesian_pairs.csv
模型 : models/v5.0/v5_model.pt
输出 : data/processed/v5_screening.csv
"""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.screen_v4 import (
    _canon_smiles, _run_attention_pooling, _pair_embedding,
    has_heterocycle, count_aromatic_rings,
)
from src.screening.gnn_v3.featurizer import smiles_to_graph
from src.chemistry.hard_rules import get_rule_vector, RULE_DIM
from src.screening.gnn_v4.model import V4Model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="data/processed/v4_cartesian_pairs.csv")
    p.add_argument("--model", default="models/v5.0/v5_model.pt")
    p.add_argument("--output", default="data/processed/v5_screening.csv")
    p.add_argument("--mc-samples", type=int, default=10)
    p.add_argument("--batch-flush", type=int, default=5000)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    print(f"=== 设备: {device} ===")

    print(f"\n=== Step 1: 加载笛卡尔积配对 ===")
    with open(args.pairs, "r", encoding="utf-8") as f:
        raw_pairs = list(csv.DictReader(f))
    if args.max_pairs > 0:
        raw_pairs = raw_pairs[:args.max_pairs]
    print(f"  加载 {len(raw_pairs)} 配对")

    print(f"\n=== Step 2: 加载模型 {args.model} ===")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    use_3d = ckpt.get("use_3d", False)
    use_rules = ckpt.get("use_rules", True)
    if use_3d:
        cfg["model"]["use_3d"] = True
    cfg["model"]["use_rules"] = use_rules
    cfg["model"]["dim_rules"] = RULE_DIM

    model = V4Model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if use_3d and ckpt.get("scaler_3d"):
        sd = ckpt.get("scaler_dimer")
        model.set_3d_scaler(
            monomer_mean=ckpt["scaler_3d"]["mean"],
            monomer_std=ckpt["scaler_3d"]["std"],
            dimer_mean=sd["mean"] if sd else None,
            dimer_std=sd["std"] if sd else None,
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,} ({n_params/1e6:.2f}M), 3D={'ON' if use_3d else 'OFF'}, rules={'ON' if use_rules else 'OFF'}")

    if use_3d:
        zero_3d = torch.zeros(1, 10, device=device)
    else:
        zero_3d = None

    print(f"\n=== Step 3: 构建图 + 单体编码 ===")
    unique_ald_smi = sorted(set(p["aldehyde_smiles"] for p in raw_pairs))
    unique_amine_smi = sorted(set(p["amine_smiles"] for p in raw_pairs))
    print(f"  独有醛: {len(unique_ald_smi)}, 独有胺: {len(unique_amine_smi)}")

    @torch.no_grad()
    def encode(graphs, label):
        cache = {}
        for smi, g in graphs.items():
            g_d = type(g)(x=g.x.to(device), edge_index=g.edge_index.to(device),
                         edge_attr=g.edge_attr.to(device))
            cache[smi] = model.encoder.encoder(g_d)
        print(f"    {label} 编码完成: {len(cache)} 个")
        return cache

    ald_graphs = {}
    for smi in unique_ald_smi:
        g = smiles_to_graph(smi, role=0)
        if g is not None:
            ald_graphs[smi] = g
    amine_graphs = {}
    for smi in unique_amine_smi:
        g = smiles_to_graph(smi, role=1)
        if g is not None:
            amine_graphs[smi] = g
    print(f"  图: {len(ald_graphs)} 醛, {len(amine_graphs)} 胺")

    t0 = time.time()
    ald_emb_cache = encode(ald_graphs, "醛")
    amine_emb_cache = encode(amine_graphs, "胺")
    print(f"  编码耗时: {time.time()-t0:.1f}s")

    print(f"\n=== Step 3.5: 预计算规则向量 ===")
    rule_cache = {}
    for p in raw_pairs:
        key = (p["aldehyde_smiles"], p["amine_smiles"])
        if key not in rule_cache:
            rule_cache[key] = get_rule_vector(p["aldehyde_smiles"], p["amine_smiles"])
    print(f"  规则向量: {len(rule_cache)} 对")

    print(f"\n=== Step 4: 推理 {len(raw_pairs)} 对 (MC={args.mc_samples}) ===")
    save_cols = [
        "aldehyde_smiles", "aldehyde_name", "ald_n", "ald_has_f", "ald_source",
        "amine_smiles", "amine_name", "am_n", "am_has_f", "am_source",
        "film_prob_mean", "film_prob_std",
        "hard_violations",
    ]

    results = []
    t0 = time.time()
    for pi, p in enumerate(raw_pairs):
        ald_s = p["aldehyde_smiles"]
        am_s = p["amine_smiles"]
        ald_emb = ald_emb_cache.get(ald_s)
        am_emb = amine_emb_cache.get(am_s)
        if ald_emb is None or am_emb is None:
            continue

        rv = torch.tensor(rule_cache[(ald_s, am_s)], dtype=torch.float, device=device).unsqueeze(0)

        model.enable_mc_dropout()
        mc_probs = []
        for _ in range(args.mc_samples):
            ea, eb, e_pair = _run_attention_pooling(model, ald_emb, am_emb, device)
            logit = model.film_head(ea, eb, e_pair, zero_3d_emb if use_3d else None, rv)
            mc_probs.append(torch.sigmoid(logit).item())
        model.eval()
        prob_mean = float(np.mean(mc_probs))
        prob_std = float(np.std(mc_probs))

        row = {
            "aldehyde_smiles": ald_s,
            "aldehyde_name": p.get("aldehyde_name", ""),
            "ald_n": p["ald_n"],
            "ald_has_f": p["ald_has_f"],
            "ald_source": p["ald_source"],
            "amine_smiles": am_s,
            "amine_name": p.get("amine_name", ""),
            "am_n": p["am_n"],
            "am_has_f": p["am_has_f"],
            "am_source": p["am_source"],
            "film_prob_mean": round(prob_mean, 6),
            "film_prob_std": round(prob_std, 6),
            "hard_violations": "",
        }
        results.append(row)

        if (pi + 1) % args.batch_flush == 0:
            elapsed = time.time() - t0
            rate = (pi + 1) / elapsed
            eta = (len(raw_pairs) - pi - 1) / rate
            print(f"  {pi+1}/{len(raw_pairs)} "
                  f"({100*(pi+1)/len(raw_pairs):.1f}%) "
                  f"速率 {rate:.0f}对/s ETA {eta:.0f}s "
                  f"通过 {len(results)}")

    print(f"\n=== Step 5: 排序 + 写出 {args.output} ===")
    results.sort(key=lambda x: x["film_prob_mean"], reverse=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=save_cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"  有效配对: {len(results)}/{len(raw_pairs)}")
    print(f"  写出: {args.output}")

    print(f"\n=== 摘要 ===")
    raw_scores = [r["film_prob_mean"] for r in results]
    print(f"  GNN prob: min={min(raw_scores):.4f} max={max(raw_scores):.4f} "
          f"mean={np.mean(raw_scores):.4f}±{np.std(raw_scores):.4f}")
    n_high = sum(1 for s in raw_scores if s >= 0.9)
    n_mid = sum(1 for s in raw_scores if 0.5 <= s < 0.9)
    n_low = sum(1 for s in raw_scores if s < 0.5)
    print(f"  >=0.9: {n_high} ({100*n_high/len(results):.1f}%)")
    print(f"  0.5-0.9: {n_mid} ({100*n_mid/len(results):.1f}%)")
    print(f"  <0.5: {n_low} ({100*n_low/len(results):.1f}%)")
    total_time = time.time() - t0
    print(f"  推理耗时: {total_time/60:.1f}min")


if __name__ == "__main__":
    main()
