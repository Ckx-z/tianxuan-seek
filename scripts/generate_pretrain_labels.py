"""预训练标签生成 — 为 GNN 编码器扩展提供计算化学标签。

两类标签:
  1. 反应能: R-CHO + H₂N-R' → R-CH=N-R' + H₂O
     反映亚胺键形成的热力学驱动力 — 与成膜过程直接相关

  2. 堆积能: 二单体 π-π 堆叠结合能
     反映 COF 层间相互作用强度 — 与结晶度和膜稳定性相关

计算引擎: GFN2-xTB (单体/二聚体/产物几何优化)

输出:
  data/processed/pretrain_pair_labels.json  — 反应能 + 堆积能标签
  data/processed/pretrain_pair_labels.csv   — 同样内容，方便 pandas 读取
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("pretrain_labels")

N_CONFORMERS = 50
LAYER_DISTANCES = [3.2, 3.4, 3.6, 3.8]
STACK_MODES = {
    "AA": [0.0, 0.0],
    "AB": [1.5, 0.0],
}


def _safe_import_xtb():
    """尝试导入 xtb-python。"""
    try:
        from xtb.interface import Calculator
        from xtb.libxtb import VERBOSITY_MUTED
        return Calculator, VERBOSITY_MUTED
    except ImportError:
        return None, None


Calculator, VERBOSITY_MUTED = _safe_import_xtb()


# ── 构象生成 ──────────────────────────────────────────

def gen_best_conformer(smi: str):
    """RDKit ETKDGv3 → UFF 预优化 → 最低能构象。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.pruneRmsThresh = 0.5
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFORMERS, params=params)
    if not cids:
        return None, None
    results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=200)
    best_cid = min(results, key=lambda x: x[1])[0] if results else cids[0]
    return mol, best_cid


def mol_to_xyz_str(mol, cid: int) -> str:
    """RDKit Mol → XYZ 字符串。"""
    conf = mol.GetConformer(cid)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    lines = [str(len(atoms)), ""]
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol():2s}  {p.x:12.6f}  {p.y:12.6f}  {p.z:12.6f}")
    return "\n".join(lines)


# ── 反应产物生成 ──────────────────────────────────────

def build_imine_product(ald_smi: str, am_smi: str) -> Chem.Mol | None:
    """SMARTS 反应: 醛 + 胺 → 亚胺 + 水。

    使用显式氢无关的 SMILES 分子进行反应匹配。
    """
    ald_mol = Chem.MolFromSmiles(ald_smi)
    am_mol = Chem.MolFromSmiles(am_smi)
    if ald_mol is None or am_mol is None:
        return None

    rxn = AllChem.ReactionFromSmarts(
        "[CX3H1:1](=[O:2])-[#6:3].[NH2:4]-[#6:5]>>"
        "[CX3H1:1](=[N:4]-[#6:5])-[#6:3].[OH2:2]")
    try:
        products = rxn.RunReactants((ald_mol, am_mol))
        if products:
            for prod_set in products:
                for prod in prod_set:
                    try:
                        Chem.SanitizeMol(prod)
                        if prod.GetNumAtoms() > 3:
                            return prod
                    except Exception:
                        continue
    except Exception:
        pass
    return None


# ── xTB 能量计算 ──────────────────────────────────────

def _parse_energy_from_log(log_path: str) -> float:
    """从 xtbopt.log 读取最终优化能量。"""
    if not os.path.exists(log_path):
        return np.nan
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if "energy:" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        continue
    return np.nan


def _parse_energy_from_stdout(stdout: str) -> float:
    """从 xtb stdout 读取 TOTAL ENERGY。"""
    for line in stdout.split("\n"):
        if "TOTAL ENERGY" in line:
            try:
                return float(line.split()[-3])
            except (ValueError, IndexError):
                continue
    return np.nan


def xtb_sp_opt(xyz_str: str, workdir: str, label: str) -> dict:
    """GFN2-xTB 优化 + 能量读取 (用于预训练标签生成，跳过 Hessian)。"""
    workdir = os.path.abspath(workdir).replace(os.sep, "/")
    os.makedirs(workdir, exist_ok=True)
    xyz_path = os.path.join(workdir, f"{label}.xyz")
    with open(xyz_path, "w") as f:
        f.write(xyz_str)

    result = {"energy": np.nan, "success": False}

    try:
        r = subprocess.run(
            ["xtb", xyz_path, "--gfn", "2", "--chrg", "0", "--opt"],
            cwd=workdir, capture_output=True,
            encoding="utf-8", errors="replace", timeout=300)

        # 从 xtbopt.log 读取最终优化能量
        log_path = os.path.join(workdir, "xtbopt.log")
        result["energy"] = _parse_energy_from_log(log_path)

        if np.isnan(result["energy"]):
            result["energy"] = _parse_energy_from_stdout(r.stdout)

        if np.isnan(result["energy"]):
            # 单点能回退
            r = subprocess.run(
                ["xtb", xyz_path, "--gfn", "2", "--chrg", "0"],
                cwd=workdir, capture_output=True,
                encoding="utf-8", errors="replace", timeout=300)
            result["energy"] = _parse_energy_from_stdout(r.stdout)

        result["success"] = not np.isnan(result["energy"])
        return result

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout: {label}")
    except Exception as e:
        logger.warning(f"xTB error ({label}): {e}")

    return result


# ── 二聚体构建 ────────────────────────────────────────

def build_dimer_xyz(mol_a, cid_a, mol_b, cid_b,
                    dist: float, offset: tuple[float, float]) -> str:
    """将 B 置于 A 上方 dist Å 处 (XY 偏移 offset)，输出合并 XYZ。"""
    conf_a = mol_a.GetConformer(cid_a)
    conf_b = mol_b.GetConformer(cid_b)

    # A 的几何中心 (非 H)
    pos_a = []
    for atom in mol_a.GetAtoms():
        if atom.GetAtomicNum() > 1:
            p = conf_a.GetAtomPosition(atom.GetIdx())
            pos_a.append([p.x, p.y, p.z])
    cent_a = np.mean(pos_a, axis=0)

    # B 的几何中心
    pos_b = []
    for atom in mol_b.GetAtoms():
        if atom.GetAtomicNum() > 1:
            p = conf_b.GetAtomPosition(atom.GetIdx())
            pos_b.append([p.x, p.y, p.z])
    cent_b = np.mean(pos_b, axis=0)

    target = cent_a + np.array([offset[0], offset[1], dist])
    shift = target - cent_b

    atoms_combined = []
    for mol, tag in [(mol_a, "a"), (mol_b, "b")]:
        conf = mol.GetConformer(cid_a) if tag == "a" else mol.GetConformer(cid_b)
        for atom in mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            if tag == "b":
                p = [p.x + shift[0], p.y + shift[1], p.z + shift[2]]
            atoms_combined.append((atom.GetSymbol(), p))

    lines = [str(len(atoms_combined)), ""]
    for sym, pos in atoms_combined:
        lines.append(f"{sym:2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}")
    return "\n".join(lines)


# ── 主计算流程 ────────────────────────────────────────

def compute_reaction_energy(ald_smi: str, am_smi: str, workdir: str) -> dict:
    """计算亚胺键形成反应能: ΔE = E(imine) + E(H2O) - E(ald) - E(am)。

    使用 GFN2-xTB 几何优化 (不计算 Hessian/ZPE, 因预训练标签不需要 ±1 kcal 精度)。
    """
    # 反应物构象
    ald_mol, ald_cid = gen_best_conformer(ald_smi)
    am_mol, am_cid = gen_best_conformer(am_smi)
    if ald_mol is None or am_mol is None:
        return {"success": False, "error": "构象生成失败"}

    # 产物: 亚胺 + 水
    imine_mol_raw = build_imine_product(ald_smi, am_smi)
    if imine_mol_raw is None:
        return {"success": False, "error": "产物构建失败"}
    imine_mol, imine_cid = gen_best_conformer(Chem.MolToSmiles(imine_mol_raw))
    water_mol, water_cid = gen_best_conformer("O")
    if imine_mol is None or water_mol is None:
        return {"success": False, "error": "产物构象生成失败"}

    # xTB 优化 + 能量
    water_xyz = ("3\n\n"
                 "O    0.000000    0.000000    0.117279\n"
                 "H    0.000000    0.757160   -0.469117\n"
                 "H    0.000000   -0.757160   -0.469117\n")

    e_ald = xtb_sp_opt(mol_to_xyz_str(ald_mol, ald_cid),
                       os.path.join(workdir, "ald"), "ald")
    e_am = xtb_sp_opt(mol_to_xyz_str(am_mol, am_cid),
                      os.path.join(workdir, "am"), "am")
    e_imine = xtb_sp_opt(mol_to_xyz_str(imine_mol, imine_cid),
                          os.path.join(workdir, "imine"), "imine")
    e_water = xtb_sp_opt(water_xyz,
                          os.path.join(workdir, "water"), "water")

    if not all([e_ald["success"], e_am["success"],
                e_imine["success"], e_water["success"]]):
        return {"success": False, "error": "xTB 计算失败"}

    dE = e_imine["energy"] + e_water["energy"] - e_ald["energy"] - e_am["energy"]
    return {
        "success": True,
        "reaction_energy_hartree": dE,
        "reaction_energy_kcal": dE * 627.509,
        "e_aldehyde": e_ald["energy"],
        "e_amine": e_am["energy"],
        "e_imine": e_imine["energy"],
        "e_water": e_water["energy"],
    }


def compute_stacking_energy(ald_smi: str, am_smi: str, workdir: str) -> dict:
    """计算二单体堆叠结合能: 扫描 AA/AB × 多种距离，取最低。"""
    ald_mol, ald_cid = gen_best_conformer(ald_smi)
    am_mol, am_cid = gen_best_conformer(am_smi)
    if ald_mol is None or am_mol is None:
        return {"success": False, "error": "构象生成失败"}

    e_ald = xtb_sp_opt(mol_to_xyz_str(ald_mol, ald_cid),
                       os.path.join(workdir, "stack_ald"), "ald")
    e_am = xtb_sp_opt(mol_to_xyz_str(am_mol, am_cid),
                      os.path.join(workdir, "stack_am"), "am")
    if not e_ald["success"] or not e_am["success"]:
        return {"success": False, "error": "单体 xTB 计算失败"}

    best_binding = np.inf
    best_config = None
    all_results = []

    for mode, offset in STACK_MODES.items():
        for dist in LAYER_DISTANCES:
            dimer_xyz = build_dimer_xyz(ald_mol, ald_cid, am_mol, am_cid,
                                        dist, offset)
            e_dimer = xtb_sp_opt(dimer_xyz,
                                 os.path.join(workdir, f"stack_{mode}_{dist}"),
                                 f"dimer_{mode}_{dist}")
            if e_dimer["success"]:
                binding = e_dimer["energy"] - e_ald["energy"] - e_am["energy"]
                all_results.append({
                    "mode": mode, "distance": dist,
                    "binding_hartree": binding,
                    "binding_kcal": binding * 627.509,
                })
                if binding < best_binding:
                    best_binding = binding
                    best_config = {"mode": mode, "distance": dist}

    if best_config is None:
        return {"success": False, "error": "所有二聚体计算失败"}

    return {
        "success": True,
        "stacking_energy_hartree": best_binding,
        "stacking_energy_kcal": best_binding * 627.509,
        "best_config": best_config,
        "all_confs": all_results,
    }


def generate_all_labels(pairs_file: str, output_dir: str,
                        max_pairs: int = 0):
    """对所有单体对生成反应能 + 堆积能标签。"""
    with open(pairs_file, encoding="utf-8") as f:
        pairs = json.load(f)

    all_pairs = []
    for label_key in ["positive", "negative"]:
        for p in pairs[label_key]:
            all_pairs.append({
                "ald_smi": p["ald_smi"],
                "am_smi": p["am_smi"],
                "label": 1 if label_key == "positive" else 0,
            })

    if max_pairs > 0:
        all_pairs = all_pairs[:max_pairs]

    results = []
    for i, pair in enumerate(all_pairs):
        logger.info(f"[{i+1}/{len(all_pairs)}] {pair['ald_smi'][:20]} + {pair['am_smi'][:20]}")

        pair_workdir = os.path.join(output_dir, f"pair_{i:03d}")
        os.makedirs(pair_workdir, exist_ok=True)

        # 反应能
        t0 = time.time()
        rx = compute_reaction_energy(pair["ald_smi"], pair["am_smi"],
                                     os.path.join(pair_workdir, "reaction"))
        rx["walltime"] = round(time.time() - t0, 1)

        # 堆积能
        t0 = time.time()
        st = compute_stacking_energy(pair["ald_smi"], pair["am_smi"],
                                     os.path.join(pair_workdir, "stacking"))
        st["walltime"] = round(time.time() - t0, 1)

        results.append({
            "ald_smi": pair["ald_smi"],
            "am_smi": pair["am_smi"],
            "label": pair["label"],
            "reaction_energy_hartree": rx.get("reaction_energy_hartree"),
            "reaction_energy_kcal": rx.get("reaction_energy_kcal"),
            "stacking_energy_hartree": st.get("stacking_energy_hartree"),
            "stacking_energy_kcal": st.get("stacking_energy_kcal"),
            "stacking_mode": st.get("best_config", {}).get("mode"),
            "stacking_distance": st.get("best_config", {}).get("distance"),
        })

    # 保存
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "pretrain_pair_labels.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    import pandas as pd
    csv_path = os.path.join(output_dir, "pretrain_pair_labels.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")

    logger.info(f"标签已保存: {json_path}, {csv_path}")

    # 汇总
    valid = [r for r in results
             if r["reaction_energy_kcal"] is not None
             and r["stacking_energy_kcal"] is not None]
    if valid:
        rx_vals = [r["reaction_energy_kcal"] for r in valid]
        st_vals = [r["stacking_energy_kcal"] for r in valid]
        pos_rx = [r["reaction_energy_kcal"] for r in valid if r["label"] == 1]
        neg_rx = [r["reaction_energy_kcal"] for r in valid if r["label"] == 0]
        print(f"\n有效计算: {len(valid)}/{len(results)}")
        print(f"反应能: {np.mean(rx_vals):.1f} ± {np.std(rx_vals):.1f} kcal/mol")
        print(f"  正样本: {np.mean(pos_rx):.1f} kcal/mol")
        print(f"  负样本: {np.mean(neg_rx):.1f} kcal/mol")
        print(f"堆积能: {np.mean(st_vals):.1f} ± {np.std(st_vals):.1f} kcal/mol")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="预训练标签生成 (反应能 + 堆积能)")
    parser.add_argument("--pairs", default="data/processed/benchmark_pairs.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="最多计算 N 对 (0=全部)")
    args = parser.parse_args()

    logger.info("=== 预训练标签生成 ===")
    logger.info(f"输入: {args.pairs}")

    if Calculator is None:
        logger.warning("xtb-python 不可用，将尝试 CLI 回退")
        logger.warning("请确保 xtb 二进制在 PATH 中: conda install -c conda-forge xtb")

    generate_all_labels(args.pairs, args.output_dir, args.max_pairs)


if __name__ == "__main__":
    main()
