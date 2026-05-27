"""v3 数据集构建脚本 — 从 fluorofilm_v3.db 读取两库合并数据，应用过滤规则，输出 v3_train.csv。

过滤链:
  全量实验组 → is_imine_only → SMILES 完整 → 排金属 → is_film 非空 → 条件归并 → CSV

Usage:
  python scripts/build_v3_dataset.py                     # 全量构建
  python scripts/build_v3_dataset.py --dry-run           # 预检，只统计不输出
  python scripts/build_v3_dataset.py --output data/processed/v3_train.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from typing import Any, Optional

from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.condition_parser import ConditionParser
from src.chemistry.monomer import has_metal_smiles, has_metal_name, is_imine_only
from src.utils.logger import setup_logger

RDLogger.logger().setLevel(RDLogger.ERROR)
logger = setup_logger("build_v3_dataset")

DEFAULT_DB = "data/fluorofilm_v3.db"
DEFAULT_OUTPUT = "data/processed/v3_train.csv"


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip() or None


def _safe_bool(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        v = value.strip().lower()
        return 1 if v in ("true", "yes", "1") else 0
    return None


def load_experiments(db_path: str) -> list[dict]:
    """从数据库加载所有实验组，关联文献级 chemistry_type。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT e.*, p.chemistry_type, p.doi
        FROM experiments e
        JOIN papers p ON e.paper_id = p.paper_id
        ORDER BY e.paper_id, e.group_id
    """).fetchall()
    results = [dict(r) for r in rows]
    conn.close()
    logger.info(f"加载 {len(results)} 个实验组")
    return results


def build_dataset(db_path: str, output_path: str, dry_run: bool = False) -> dict:
    """执行过滤链，输出 CSV。"""
    experiments = load_experiments(db_path)
    parser = ConditionParser()
    stats: dict[str, int] = {
        "total": len(experiments),
        "imine_pass": 0, "imine_fail": 0,
        "smiles_pass": 0, "smiles_fail_missing": 0, "smiles_fail_parse": 0,
        "metal_pass": 0, "metal_fail_name": 0, "metal_fail_smiles": 0,
        "film_pass": 0, "film_fail_null": 0,
        "final": 0,
    }

    # ── 层 1: chemistry_type 过滤 ──
    layer1 = []
    for exp in experiments:
        ct = exp.get("chemistry_type", "")
        if is_imine_only(ct):
            stats["imine_pass"] += 1
            layer1.append(exp)
        else:
            stats["imine_fail"] += 1
    logger.info(f"层1 imine过滤: {stats['imine_pass']} 通过, {stats['imine_fail']} 排除")

    # ── 层 2: SMILES 完整 + 可解析 ──
    layer2 = []
    for exp in layer1:
        ald_smi = (exp.get("aldehyde_smiles") or "").strip()
        amine_smi = (exp.get("amine_smiles") or "").strip()
        if not ald_smi or not amine_smi:
            stats["smiles_fail_missing"] += 1
            continue

        mol_a = Chem.MolFromSmiles(ald_smi)
        mol_b = Chem.MolFromSmiles(amine_smi)

        # sanitize=False 降级挽救
        ald_source = "builtin"
        amine_source = "builtin"
        if mol_a is None:
            mol_a = Chem.MolFromSmiles(ald_smi, sanitize=False)
            ald_source = "sanitize_fallback" if mol_a is not None else "builtin"
        if mol_b is None:
            mol_b = Chem.MolFromSmiles(amine_smi, sanitize=False)
            amine_source = "sanitize_fallback" if mol_b is not None else "builtin"

        if mol_a is None or mol_b is None:
            stats["smiles_fail_parse"] += 1
            continue
        stats["smiles_pass"] += 1
        exp["_ald_smiles_source"] = ald_source
        exp["_amine_smiles_source"] = amine_source
        layer2.append(exp)
    logger.info(
        f"层2 SMILES过滤: {stats['smiles_pass']} 通过, "
        f"{stats['smiles_fail_missing']} 缺失, {stats['smiles_fail_parse']} 解析失败"
    )

    # ── 层 3: 排金属 ──
    layer3 = []
    for exp in layer2:
        ald_smi = (exp.get("aldehyde_smiles") or "").strip()
        amine_smi = (exp.get("amine_smiles") or "").strip()
        ald_name = exp.get("aldehyde_name") or ""
        amine_name = exp.get("amine_name") or ""

        if has_metal_smiles(ald_smi) or has_metal_smiles(amine_smi):
            stats["metal_fail_smiles"] += 1
            continue
        if has_metal_name(ald_name) or has_metal_name(amine_name):
            stats["metal_fail_name"] += 1
            continue
        stats["metal_pass"] += 1
        layer3.append(exp)
    logger.info(
        f"层3 金属过滤: {stats['metal_pass']} 通过, "
        f"{stats['metal_fail_smiles']} SMILES含金属, {stats['metal_fail_name']} 名称含金属"
    )

    # ── 层 4: is_film 非空 ──
    layer4 = []
    for exp in layer3:
        is_film = exp.get("is_film")
        if is_film is None:
            stats["film_fail_null"] += 1
            continue
        stats["film_pass"] += 1
        layer4.append(exp)
    logger.info(f"层4 is_film过滤: {stats['film_pass']} 通过, {stats['film_fail_null']} 缺失")

    # ── 层 5: 同文献条件补充 + 条件归并 ──
    by_paper: dict[str, list[dict]] = {}
    for exp in layer4:
        pid = exp["paper_id"]
        by_paper.setdefault(pid, []).append(exp)

    for pid, rows in by_paper.items():
        parser.fill_from_peers(rows)

    rows_out = []
    for exp in layer4:
        cond = parser.parse(
            synthesis_route=exp.get("synthesis_route"),
            interface_type=exp.get("interface_type"),
            catalyst=exp.get("catalyst"),
            solvent=exp.get("solvent"),
            temperature=exp.get("temperature"),
        )
        is_film_val = _safe_bool(exp.get("is_film"))
        film_quality = (exp.get("film_quality") or "").strip().lower()
        quality_weight = {"high": 1.5, "medium": 1.0, "low": 0.5}.get(film_quality, 1.0)

        row = {
            "paper_id": exp["paper_id"],
            "group_id": exp["group_id"],
            "source_db": "v3_merged",
            "aldehyde_smiles": (exp.get("aldehyde_smiles") or "").strip(),
            "amine_smiles": (exp.get("amine_smiles") or "").strip(),
            "aldehyde_smiles_source": exp.get("_ald_smiles_source", "builtin"),
            "amine_smiles_source": exp.get("_amine_smiles_source", "builtin"),
            "aldehyde_name": exp.get("aldehyde_name"),
            "amine_name": exp.get("amine_name"),
            "stoichiometry": _safe_str(exp.get("stoichiometry")),
            "solvent_raw": _safe_str(exp.get("solvent")),
            "temperature_raw": _safe_str(exp.get("temperature")),
            "catalyst_raw": _safe_str(exp.get("catalyst")),
            "synthesis_route_raw": _safe_str(exp.get("synthesis_route")),
            "interface_type_raw": _safe_str(exp.get("interface_type")),
            "solvent_label": cond["solvent"],
            "temperature_bin": cond["temperature"],
            "catalyst_label": cond["catalyst"],
            "synthesis_route_label": cond["synthesis_route"],
            "interface_type_label": cond["interface_type"],
            "is_film": is_film_val,
            "film_quality": film_quality or "unknown",
            "quality_weight": quality_weight,
            "has_fluorine_monomer": _safe_bool(exp.get("has_fluorine_monomer")),
            "has_n_heterocycle": _safe_bool(exp.get("has_n_heterocycle")),
            "confidence": _safe_str(exp.get("confidence")),
        }
        rows_out.append(row)

    stats["final"] = len(rows_out)

    pos = sum(1 for r in rows_out if r["is_film"] == 1)
    neg = sum(1 for r in rows_out if r["is_film"] == 0)
    logger.info(f"最终数据集: {stats['final']} 组 (正 {pos}, 负 {neg}, 正样本率 {100*pos/stats['final']:.1f}%)")

    if dry_run:
        logger.info("预检模式 — 不写入文件")
        return stats

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    logger.info(f"已写入 {output_path} ({len(rows_out)} 行)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="v3 数据集构建")
    parser.add_argument("--db", type=str, default=DEFAULT_DB, help="数据库路径")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="输出 CSV 路径")
    parser.add_argument("--dry-run", action="store_true", help="预检模式，不写入")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        logger.error(f"数据库不存在: {args.db}")
        sys.exit(1)

    stats = build_dataset(args.db, args.output, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print(f"  v3 数据集构建{' (预检)' if args.dry_run else ''}")
    print(f"  输入: {args.db}")
    print(f"  全量: {stats['total']}")
    print(f"  imine 过滤: {stats['imine_pass']} (+{stats['imine_fail']} 排除)")
    print(f"  SMILES 过滤: {stats['smiles_pass']} (+{stats['smiles_fail_missing']} 缺失, +{stats['smiles_fail_parse']} 解析失败)")
    print(f"  金属过滤: {stats['metal_pass']} (+{stats['metal_fail_smiles']} SMILES, +{stats['metal_fail_name']} 名称)")
    print(f"  is_film 过滤: {stats['film_pass']} (+{stats['film_fail_null']} 缺失)")
    print(f"  最终: {stats['final']} 组")
    if not args.dry_run:
        print(f"  输出: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
