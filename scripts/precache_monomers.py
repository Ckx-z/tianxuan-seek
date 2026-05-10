"""单体 SMILES 预缓存脚本。

扫描所有 reagent 字段，提取唯一单体名称，通过 PubChem API 批量解析，
写入 JSON 缓存文件供 build_features.py 使用。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.monomer import MonomerLibrary, extract_monomer_names, _expand_name_candidates
from src.utils.db import init_db
from src.utils.logger import setup_logger

logger = setup_logger("precache")


def main():
    parser = argparse.ArgumentParser(description="预缓存单体 SMILES")
    parser.add_argument("--db", default="data/fluorofilm.db")
    parser.add_argument("--cache", default="data/processed/monomer_smiles_cache.json")
    parser.add_argument("--top", type=int, default=500,
                        help="最多查询 PubChem 的名称数")
    parser.add_argument("--min-freq", type=int, default=1,
                        help="最小出现频率")
    args = parser.parse_args()

    # 加载全部 reagent
    conn = init_db(args.db)
    rows = conn.execute(
        "SELECT reagent FROM literature WHERE reagent IS NOT NULL"
    ).fetchall()
    conn.close()
    logger.info(f"共 {len(rows)} 条试剂记录")

    # 收集所有唯一名称及其频率
    name_freq = {}
    for (reagent,) in rows:
        names = extract_monomer_names(reagent or "")
        for name in names:
            name_freq[name] = name_freq.get(name, 0) + 1

    logger.info(f"唯一名称数: {len(name_freq)}")

    # 先用内置字典筛选已解析的
    lib = MonomerLibrary(cache_path=args.cache, use_pubchem=False)
    resolved = set()
    unresolved = {}
    for name, freq in name_freq.items():
        # 展开候选名
        candidates = _expand_name_candidates(name)
        found = False
        for cand in candidates:
            smi = lib._resolve_single(cand)
            if smi:
                resolved.add(name)
                found = True
                break
        if not found:
            unresolved[name] = freq

    logger.info(f"已解析（内置+缓存）: {len(resolved)}, 待查询: {len(unresolved)}")

    # 按频率排序
    sorted_unresolved = sorted(unresolved.items(), key=lambda x: -x[1])
    candidates_to_query = [
        (name, freq) for name, freq in sorted_unresolved
        if freq >= args.min_freq and len(name) >= 3
    ][:args.top]

    logger.info(f"将查询 PubChem: {len(candidates_to_query)} 个名称")

    # 逐个查询 PubChem（限速）
    lib.use_pubchem = True
    queried, found = 0, 0
    for i, (name, freq) in enumerate(candidates_to_query):
        smi = lib._query_pubchem(name)
        queried += 1
        if smi:
            lib._cache[name.strip()] = smi
            found += 1
        if (i + 1) % 50 == 0:
            lib._save_cache()
            print(f"  进度: {i + 1}/{len(candidates_to_query)} "
                  f"(命中 {found}, 命中率 {100 * found / queried:.1f}%)")

    lib._save_cache()
    print(f"\n完成: 查询 {queried}, 命中 {found} "
          f"(命中率 {100 * found / max(queried, 1):.1f}%)")
    print(f"缓存已保存至: {args.cache}")


if __name__ == "__main__":
    main()
