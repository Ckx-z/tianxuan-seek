"""COF 因果属性预训练 v2 — 扩展形状/拓扑/电子特征。

预训练目标 (RDKit 零标注成本):
  原有 (8):
    1-2. 醛基数/胺基数 (回归)  — 连通性 C2/C3/C4
    3-4. 氟原子数/CF3 (回归/BCE) — 层间堆积
    5-7. 可旋转键/芳香环/sp3 (回归) — 刚性/共轭
    8. 单体类型 (3类) — COF 角色
  新增 — Benchmark 验证有用维度 (5):
    9. 非球面度 (回归)         — |r|=0.57, 分子形状 → 框架有序度
   10. 偏心率 (回归)           — |r|=0.43, 棒状 vs 盘状
   11. 平面性 RMSD (回归)      — |r|=0.32, 共面倾向 → π 堆叠
   12. 偶极矩 Gasteiger (回归)  — |r|=0.47, 极性匹配
   13. 反应位点总数 (回归)      — |r|=0.13* (但拓扑关键: C2/C3/C4)

完成后 → 两阶段微调到成膜任务 (487 标记样本)
"""
import argparse
import os
import sys
import warnings
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.gnn import (MoleculeEncoder, smiles_to_graph,
                               collate_graphs)
from src.chemistry.imine_check import ImineChecker
from src.chemistry.fluorination import FluorineDetector
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("cof_pretrain")

DEVICE = "cpu"
HIDDEN = 256
EPOCHS = 300
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-5


class COFPropertyPredictor(nn.Module):
    """多任务属性预测器 — 共享编码器 + 任务特定头。

    8 个辅助任务, 全部与 COF 成膜的化学因果相关。
    """

    def __init__(self, encoder: MoleculeEncoder, hidden: int = 256):
        super().__init__()
        self.encoder = encoder

        # 回归头 (MSE)
        reg_dim = hidden // 4  # 64
        self.head_n_aldehyde = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))
        self.head_n_amine = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))
        self.head_f_count = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))
        self.head_n_rot = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))
        self.head_n_arom = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))
        self.head_fraction_csp3 = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))

        # 二分类头 (BCE)
        self.head_has_cf3 = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))

        # 三分类头 (CE)
        self.head_monomer_type = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 3))

        # 新增: 形状/电子/拓扑 (benchmark 验证)
        self.head_asphericity = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))
        self.head_eccentricity = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))
        self.head_planarity = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))
        self.head_dipole = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))
        self.head_n_reactive = nn.Sequential(
            nn.Linear(hidden, reg_dim), nn.ReLU(), nn.Linear(reg_dim, 1))

    def forward(self, data: Data) -> dict[str, torch.Tensor]:
        h = self.encoder(data)
        return {
            "n_aldehyde": self.head_n_aldehyde(h).squeeze(-1),
            "n_amine": self.head_n_amine(h).squeeze(-1),
            "f_count": self.head_f_count(h).squeeze(-1),
            "n_rot": self.head_n_rot(h).squeeze(-1),
            "n_arom": self.head_n_arom(h).squeeze(-1),
            "fraction_csp3": self.head_fraction_csp3(h).squeeze(-1),
            "has_cf3": self.head_has_cf3(h).squeeze(-1),
            "monomer_type": self.head_monomer_type(h),
            "asphericity": self.head_asphericity(h).squeeze(-1),
            "eccentricity": self.head_eccentricity(h).squeeze(-1),
            "planarity": self.head_planarity(h).squeeze(-1),
            "dipole": self.head_dipole(h).squeeze(-1),
            "n_reactive": self.head_n_reactive(h).squeeze(-1),
        }


def _compute_shape_features(mol: Chem.Mol) -> dict:
    """从 ETKDG 3D 构象计算形状描述符 (无 xtb 依赖)。

    返回: asphericity, eccentricity, planarity, dipole
    """
    feats = {"asphericity": np.nan, "eccentricity": np.nan,
             "planarity": np.nan, "dipole": np.nan}
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.numThreads = 0
        cid = AllChem.EmbedMolecule(mol_h, params)
        if cid < 0:
            return feats
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        conf = mol_h.GetConformer()

        # 重原子坐标
        pts = []
        atoms_with_H = list(mol_h.GetAtoms())
        for atom in atoms_with_H:
            if atom.GetAtomicNum() > 1:
                p = conf.GetAtomPosition(atom.GetIdx())
                pts.append([p.x, p.y, p.z])
        if len(pts) < 4:
            return feats
        pts_arr = np.array(pts)
        pts_arr -= pts_arr.mean(axis=0)

        # 平面性: SVD 拟合面 RMSD
        _, s, vt = np.linalg.svd(pts_arr)
        dists = np.abs(pts_arr @ vt[2])
        feats["planarity"] = float(np.sqrt(np.mean(dists ** 2)))

        # 非球面度 & 偏心率: 惯性张量特征值
        I = pts_arr.T @ pts_arr / len(pts_arr)
        eigvals = np.sort(np.linalg.eigvalsh(I))
        if eigvals[2] > 0:
            feats["asphericity"] = float(
                1.5 * eigvals[2] / eigvals.sum() - 0.5)
            feats["eccentricity"] = float(
                np.sqrt(1 - eigvals[0] / eigvals[2]))
        else:
            feats["asphericity"] = 0.0
            feats["eccentricity"] = 0.0

        # 偶极矩 (Gasteiger 电荷)
        try:
            AllChem.ComputeGasteigerCharges(mol_h)
            d = np.zeros(3)
            for atom in atoms_with_H:
                p = conf.GetAtomPosition(atom.GetIdx())
                q = float(atom.GetProp("_GasteigerCharge"))
                d += q * np.array([p.x, p.y, p.z])
            feats["dipole"] = float(np.linalg.norm(d))
        except Exception:
            pass
    except Exception:
        pass
    return feats


def compute_properties(smiles: str) -> dict:
    """用 RDKit 计算分子属性 (预训练标签), 含形状/电子/拓扑。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    checker = ImineChecker()
    f_det = FluorineDetector()

    # 单体类型: 0=aldehyde, 1=amine, 2=other
    is_ald = checker.is_aldehyde(mol)
    is_am = checker.is_amine(mol)
    if is_ald and is_am:
        mono_type = 0
    elif is_ald:
        mono_type = 0
    elif is_am:
        mono_type = 1
    else:
        mono_type = 2

    n_ald = float(checker.count_aldehyde_groups(mol))
    n_am = float(checker.count_amine_groups(mol))

    props = {
        "n_aldehyde": n_ald,
        "n_amine": n_am,
        "f_count": float(f_det.count_fluorine(mol)),
        "has_cf3": 1.0 if f_det.has_cf3(mol) else 0.0,
        "n_rot": float(Descriptors.NumRotatableBonds(mol)),
        "n_arom": float(Descriptors.NumAromaticRings(mol)),
        "fraction_csp3": float(Descriptors.FractionCSP3(mol)),
        "monomer_type": mono_type,
        "n_reactive": n_ald + n_am,  # 拓扑: 总反应位点数
    }

    # 形状特征 (RDKit 3D 构象), NaN → 0.0 回退
    shape = _compute_shape_features(mol)
    for k in ["asphericity", "eccentricity", "planarity", "dipole"]:
        v = shape.get(k, np.nan)
        props[k] = 0.0 if np.isnan(v) else float(v)

    return props


def load_pretrain_data():
    """收集所有唯一 SMILES 并计算属性标签。"""
    import json
    import pandas as pd

    smiles_set = set()

    # 来源 1: 最新 label_metadata (v3, 1027 samples)
    for meta_path in ["data/processed/label_metadata_v3.csv",
                       "data/processed/label_metadata_v2.csv",
                       "data/processed/label_metadata.csv"]:
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path, encoding="utf-8-sig")
            for col in ["aldehyde_smiles", "amine_smiles"]:
                for s in meta[col].dropna():
                    mol = Chem.MolFromSmiles(s)
                    if mol:
                        smiles_set.add(Chem.MolToSmiles(mol, isomericSmiles=True))
            break  # 只用最新的

    # 来源 2: 三组 SMILES 缓存
    for cache_path in ["data/processed/monomer_smiles_cache.json",
                        "data/processed/monomer_smiles_new.json",
                        "data/processed/monomer_smiles_new3.json"]:
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
            for v in cache.values():
                if isinstance(v, str) and v:
                    mol = Chem.MolFromSmiles(v)
                    if mol:
                        smiles_set.add(Chem.MolToSmiles(mol, isomericSmiles=True))

    # 过滤并计算属性
    data = []
    for s in sorted(smiles_set):
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() < 3:
            continue
        g = smiles_to_graph(s)
        if g is None:
            continue
        props = compute_properties(s)
        if not props:
            continue
        data.append({"smiles": s, "graph": g, **props})

    logger.info(f"预训练数据: {len(data)} 个分子 (≥3 原子)")
    # 统计
    counts = defaultdict(int)
    for d in data:
        counts["monomer_type"] += 1
        for k in ["n_aldehyde", "n_amine", "f_count"]:
            if d[k] > 0:
                counts[f"{k}_positive"] += 1
    logger.info(f"  有醛基: {counts['n_aldehyde_positive']}, "
                f"有胺基: {counts['n_amine_positive']}, "
                f"含氟: {counts['f_count_positive']}")
    return data


def pretrain_cof(data: list[dict], model_dir: str):
    """COF 因果属性预训练。"""
    n = len(data)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"COF 属性预训练: {EPOCHS} epochs, {n_batches} batches/epoch")

    encoder = MoleculeEncoder(hidden=HIDDEN, dropout=0.1)
    model = COFPropertyPredictor(encoder, hidden=HIDDEN)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    # 任务权重: 醛/胺基 + 形状特征更重要
    task_weights = {
        "n_aldehyde": 2.0, "n_amine": 2.0, "n_reactive": 1.5,
        "f_count": 1.5, "has_cf3": 1.5,
        "n_rot": 0.5, "n_arom": 0.5, "fraction_csp3": 0.5,
        "monomer_type": 1.0,
        "asphericity": 1.5, "eccentricity": 1.0,
        "planarity": 1.0, "dipole": 1.0,
    }
    reg_keys = {"n_aldehyde", "n_amine", "n_reactive", "f_count",
                "n_rot", "n_arom", "fraction_csp3",
                "asphericity", "eccentricity", "planarity", "dipole"}
    bce_keys = {"has_cf3"}
    ce_keys = {"monomer_type"}

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = np.random.permutation(n)
        epoch_losses = defaultdict(float)

        for start in range(0, n, BATCH_SIZE):
            bi = idx[start:start + BATCH_SIZE]
            batch_graphs = [data[i]["graph"] for i in bi]
            batch = collate_graphs(batch_graphs)

            # 标签
            targets = {}
            for key in task_weights:
                vals = [data[i][key] for i in bi]
                t = torch.tensor(vals, dtype=torch.float)
                if key == "monomer_type":
                    t = t.long()
                targets[key] = t

            optimizer.zero_grad()
            preds = model(batch)

            loss = 0.0
            for key, w in task_weights.items():
                if key in reg_keys:
                    l = mse(preds[key], targets[key])
                elif key in bce_keys:
                    l = bce(preds[key], targets[key])
                elif key in ce_keys:
                    l = ce(preds[key], targets[key])
                loss = loss + w * l
                epoch_losses[key] += l.item()

            loss.backward()
            optimizer.step()

        scheduler.step()
        total = sum(epoch_losses.values()) / max(n_batches, 1)

        if epoch % 50 == 0 or epoch == 1:
            rmse_ald = np.sqrt(epoch_losses["n_aldehyde"] / max(n_batches, 1))
            rmse_am = np.sqrt(epoch_losses["n_amine"] / max(n_batches, 1))
            rmse_asp = np.sqrt(epoch_losses.get("asphericity", 0) / max(n_batches, 1))
            logger.info(f"  Epoch {epoch:3d}/{EPOCHS}: total={total:.4f}, "
                        f"RMSE_ald={rmse_ald:.3f}, RMSE_am={rmse_am:.3f}, "
                        f"RMSE_aspher={rmse_asp:.3f}")

        if total < best_loss - 1e-5:
            best_loss = total
            best_state = deepcopy(encoder.state_dict())

    # 保存
    path = os.path.join(model_dir, "gnn_encoder_cof_pretrained.pt")
    torch.save(best_state, path)
    logger.info(f"COF 预训练编码器已保存: {path} (best_loss={best_loss:.4f})")

    # 在留出集上评估
    model.eval()
    eval_metrics = defaultdict(list)
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = collate_graphs([data[i]["graph"] for i in range(start, end)])
            preds = model(batch)
            for key in task_weights:
                vals = np.array([data[i][key] for i in range(start, end)])
                p = preds[key].cpu().numpy()
                if key == "monomer_type":
                    acc = (p.argmax(axis=1) == vals).mean()
                    eval_metrics[key].append(acc)
                elif key == "has_cf3":
                    acc = ((p > 0).astype(int) == vals).mean()
                    eval_metrics[key].append(acc)
                else:
                    mae = np.abs(p.ravel() - vals).mean()
                    eval_metrics[key].append(mae)

    print("\n" + "=" * 60)
    print("  COF 属性预训练完成 — 留出集评估")
    print("=" * 60)
    for key, vals in eval_metrics.items():
        mu = np.mean(vals)
        label = "Acc" if key in ("monomer_type", "has_cf3") else "MAE"
        print(f"  {key:20s}: {label}={mu:.3f}")
    print(f"\n  编码器: {path}")
    print("=" * 60)

    return best_state


def main():
    parser = argparse.ArgumentParser(description="COF 因果属性预训练")
    parser.add_argument("--model-dir", default="models/v2.0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    global EPOCHS, BATCH_SIZE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("=== COF 因果属性预训练 (Route B) ===")
    logger.info(f"超参: HIDDEN={HIDDEN}, EPOCHS={EPOCHS}, LR={LR}")

    data = load_pretrain_data()
    pretrain_cof(data, args.model_dir)


if __name__ == "__main__":
    main()
