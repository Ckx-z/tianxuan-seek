"""v3 SMILES 第二层回填 — 全局名称精确匹配 + PubChem 查询。

策略:
  1. 从已有 SMILES 的实验组建立 名称→SMILES 全局映射（取众数）
  2. 对缺失 SMILES 的实验组，精确名称匹配 → 直接复用
  3. 剩余未解决的，用 MonomerLibrary (内置字典+缓存+PubChem) 查询
  4. 将回填结果写回 YAML 文件

Usage:
  python scripts/backfill_smiles_v3.py              # 回填 + 保存
  python scripts/backfill_smiles_v3.py --dry-run    # 预检，不写入
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import Counter
from typing import Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.monomer import MonomerLibrary, _expand_name_candidates
from src.utils.logger import setup_logger

logger = setup_logger("backfill_smiles_v3")

DEFAULT_DB = "data/fluorofilm_v3.db"
STRUCTURED_DIRS = ["data/structured_v2", "data/structured_v3"]


def build_global_name_map(conn: sqlite3.Connection) -> tuple[dict[str, str], dict[str, str]]:
    """从已有 SMILES 的实验组建立全局 名称→SMILES 映射（取众数）。"""
    rows = conn.execute("""
        SELECT DISTINCT aldehyde_name, aldehyde_smiles, amine_name, amine_smiles
        FROM experiments
        WHERE aldehyde_smiles IS NOT NULL AND aldehyde_smiles != ''
        AND amine_smiles IS NOT NULL AND amine_smiles != ''
        AND aldehyde_name IS NOT NULL AND aldehyde_name != ''
        AND amine_name IS NOT NULL AND amine_name != ''
    """).fetchall()

    ald_map: dict[str, list[str]] = {}
    amine_map: dict[str, list[str]] = {}
    for ald_n, ald_s, amine_n, amine_s in rows:
        key_a = ald_n.strip().lower()
        key_b = amine_n.strip().lower()
        if key_a and ald_s:
            ald_map.setdefault(key_a, []).append(ald_s.strip())
        if key_b and amine_s:
            amine_map.setdefault(key_b, []).append(amine_s.strip())

    ald_result = {k: Counter(v).most_common(1)[0][0] for k, v in ald_map.items()}
    amine_result = {k: Counter(v).most_common(1)[0][0] for k, v in amine_map.items()}
    return ald_result, amine_result


def find_missing_experiments(conn: sqlite3.Connection) -> list[dict]:
    """查找有有效名称但 SMILES 缺失的实验组。"""
    rows = conn.execute("""
        SELECT e.paper_id, e.group_id, e.aldehyde_name, e.amine_name,
               e.aldehyde_smiles, e.amine_smiles
        FROM experiments e
        WHERE e.aldehyde_name IS NOT NULL AND e.aldehyde_name != ''
        AND e.aldehyde_name NOT LIKE '%Supporting Information%'
        AND e.aldehyde_name != 'N/A' AND e.aldehyde_name NOT LIKE '%Table S%'
        AND e.amine_name IS NOT NULL AND e.amine_name != ''
        AND e.amine_name NOT LIKE '%Supporting Information%'
        AND e.amine_name != 'N/A' AND e.amine_name NOT LIKE '%Table S%'
        AND ((e.aldehyde_smiles IS NULL OR e.aldehyde_smiles = '')
             OR (e.amine_smiles IS NULL OR e.amine_smiles = ''))
    """).fetchall()

    return [
        {
            "paper_id": r[0], "group_id": r[1],
            "aldehyde_name": r[2], "amine_name": r[3],
            "aldehyde_smiles": r[4], "amine_smiles": r[5],
        }
        for r in rows
    ]


def backfill(db_path: str, dry_run: bool = False) -> dict:
    """执行回填，将结果写回 YAML 文件。"""
    conn = sqlite3.connect(db_path)
    ald_map, amine_map = build_global_name_map(conn)
    logger.info(f"全局映射: 醛 {len(ald_map)}, 胺 {len(amine_map)}")

    missing = find_missing_experiments(conn)
    logger.info(f"缺失 SMILES 实验组: {len(missing)}")

    lib = MonomerLibrary(use_pubchem=False)  # 不调 PubChem，只用内置字典+缓存

    by_paper: dict[str, list[dict]] = {}
    for exp in missing:
        pid = exp["paper_id"]
        by_paper.setdefault(pid, []).append(exp)

    stats = {"total_missing": len(missing), "exact_match": 0,
             "candidate_match": 0, "pubchem_match": 0, "still_missing": 0,
             "yaml_updated": 0}

    updates: dict[str, dict[int, dict[str, str]]] = {}

    for pid, exps in by_paper.items():
        for exp in exps:
            gid = exp["group_id"]
            ald_smi = exp["aldehyde_smiles"]
            amine_smi = exp["amine_smiles"]
            ald_n = exp["aldehyde_name"]
            amine_n = exp["amine_name"]

            new_ald = None
            new_amine = None
            ald_source = None
            amine_source = None

            if not ald_smi or not ald_smi.strip():
                key = ald_n.strip().lower() if ald_n else ""
                if key in ald_map:
                    new_ald = ald_map[key]
                    ald_source = "exact_match"
                else:
                    candidates = _expand_name_candidates(ald_n)
                    for cand in candidates:
                        if cand.lower() in ald_map:
                            new_ald = ald_map[cand.lower()]
                            ald_source = "candidate_match"
                            break
                    if not new_ald:
                        for cand in candidates:
                            smi = lib.resolve(cand)
                            if smi:
                                new_ald = smi
                                ald_source = "pubchem"
                                break

            if not amine_smi or not amine_smi.strip():
                key = amine_n.strip().lower() if amine_n else ""
                if key in amine_map:
                    new_amine = amine_map[key]
                    amine_source = "exact_match"
                else:
                    candidates = _expand_name_candidates(amine_n)
                    for cand in candidates:
                        if cand.lower() in amine_map:
                            new_amine = amine_map[cand.lower()]
                            amine_source = "candidate_match"
                            break
                    if not new_amine:
                        for cand in candidates:
                            smi = lib.resolve(cand)
                            if smi:
                                new_amine = smi
                                amine_source = "pubchem"
                                break

            if new_ald or new_amine:
                updates.setdefault(pid, {})[gid] = {}
                if new_ald:
                    updates[pid][gid]["aldehyde_smiles"] = new_ald
                    stats[ald_source or "exact_match"] = stats.get(ald_source or "exact_match", 0) + 1
                if new_amine:
                    updates[pid][gid]["amine_smiles"] = new_amine
                    stats[amine_source or "exact_match"] = stats.get(amine_source or "exact_match", 0) + 1
            else:
                stats["still_missing"] += 1

    conn.close()
    lib.flush_cache()

    if not dry_run:
        yaml_updated = 0
        for pid, group_updates in updates.items():
            yaml_path = None
            for d in STRUCTURED_DIRS:
                candidate = os.path.join(d, f"{pid}.yaml")
                if os.path.exists(candidate):
                    yaml_path = candidate
                    break
            if not yaml_path:
                logger.warning(f"YAML 文件未找到: {pid}")
                continue

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            modified = False
            for exp in data.get("experiments", []):
                gid = exp.get("group_id")
                if gid in group_updates:
                    upd = group_updates[gid]
                    monomers = exp.setdefault("monomers", {})
                    if "aldehyde_smiles" in upd:
                        monomers["aldehyde_smiles"] = upd["aldehyde_smiles"]
                        modified = True
                    if "amine_smiles" in upd:
                        monomers["amine_smiles"] = upd["amine_smiles"]
                        modified = True

            if modified:
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                yaml_updated += 1

        stats["yaml_updated"] = yaml_updated
        logger.info(f"YAML 更新: {yaml_updated} 篇")

    logger.info(
        f"回填: 精确匹配 {stats.get('exact_match', 0)}, "
        f"候选匹配 {stats.get('candidate_match', 0)}, "
        f"PubChem {stats.get('pubchem_match', 0)}, "
        f"仍缺失 {stats['still_missing']}"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="v3 SMILES 第二层回填")
    parser.add_argument("--db", type=str, default=DEFAULT_DB, help="数据库路径")
    parser.add_argument("--dry-run", action="store_true", help="预检模式")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        logger.error(f"数据库不存在: {args.db}")
        sys.exit(1)

    stats = backfill(args.db, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print(f"  v3 SMILES 第二层回填{' (预检)' if args.dry_run else ''}")
    print(f"  缺失实验组: {stats['total_missing']}")
    print(f"  精确匹配: {stats.get('exact_match', 0)}")
    print(f"  候选匹配: {stats.get('candidate_match', 0)}")
    print(f"  PubChem: {stats.get('pubchem_match', 0)}")
    print(f"  仍缺失: {stats['still_missing']}")
    if not args.dry_run:
        print(f"  YAML 更新: {stats['yaml_updated']} 篇")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
