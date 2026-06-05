"""Step 3: 拓扑硬排除 + 多样性约束 + adjusted 排序 → Top 40。

输入 : data/processed/v4_screening_v2_soft.csv  (232,740 配对)
输出 : data/processed/v4_top40_candidates.csv   (含 diversity_rank)

Pipeline:
  232,740 ─→ 拓扑硬排除(C3+C4/C4+C3/C3+C3) ─→ n>=2 硬过滤 ─→ sort by film_prob_adjusted
       ─→ diverse(Morgan Tanimoto<0.8) ─→ Top 40
            ↑
            醛和胺分别约束

硬规则:
  - C3+C4 / C4+C3: 非标准 2D 拓扑, 排除
  - C3+C3: 非标准 2D 拓扑, 排除
  - n_ald < 2 或 n_am < 2: 单官能团配对, 排除

多样性约束 (Tanimoto < 0.8):
  - 醛池: 与已选醛 Tanimoto < 0.8 才入选
  - 胺池: 与已选胺 Tanimoto < 0.8 才入选
  - 优先级: film_prob_adjusted 高的优先入

target 配对 (canonical SMILES):
  O=Cc1cc(-c2ccccc2)c(C=O)cc1-c1ccccc1  ×  Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1
"""
from __future__ import annotations

import os
import sys
import csv
import argparse
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

RDLogger.logger().setLevel(RDLogger.ERROR)

THRESHOLD_TANIMOTO = 0.8
TOP_K = 40
TARGET_ALD_CANON = "O=Cc1cc(-c2ccccc2)c(C=O)cc1-c1ccccc1"
TARGET_AM_CANON = "Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/v4_screening_v2_soft.csv")
    p.add_argument("--output", default="data/processed/v4_top40_chemfilt_raw.csv")
    p.add_argument("--candidates", default="data/processed/v4_top40_candidates.csv")
    p.add_argument("--tanimoto-threshold", type=float, default=THRESHOLD_TANIMOTO)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--include-target", action="store_true",
                   help="在 Top 40 之后追加 target 配对 (不计入多样性约束, 标记 <<< TARGET)")
    return p.parse_args()


def smiles_to_morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def main():
    args = parse_args()
    print("=== Step 3: 多样性 + Top 40 ===")

    with open(args.input, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"全表: {len(rows)}")

    # 硬规则已在 Step 2 (hard_rules.py) 完成，此处仅按 adj 排序 + 多样性筛选
    passed_sorted = sorted(rows,
                           key=lambda x: float(x["film_prob_adjusted"]),
                           reverse=True)
    print(f"按 film_prob_adjusted 排序后: {len(passed_sorted)} 配对")

    print("\n预计算 Morgan 指纹 (r=2, 2048 bit)...")
    ald_fps = {}
    am_fps = {}
    ald_smiles_in_pool = set(r["aldehyde_smiles"] for r in passed_sorted)
    am_smiles_in_pool = set(r["amine_smiles"] for r in passed_sorted)
    n_fail_ald = 0
    n_fail_am = 0
    for smi in ald_smiles_in_pool:
        fp = smiles_to_morgan_fp(smi)
        if fp is not None:
            ald_fps[smi] = fp
        else:
            n_fail_ald += 1
    for smi in am_smiles_in_pool:
        fp = smiles_to_morgan_fp(smi)
        if fp is not None:
            am_fps[smi] = fp
        else:
            n_fail_am += 1
    print(f"  醛: {len(ald_fps)}/{len(ald_smiles_in_pool)} (失败 {n_fail_ald})")
    print(f"  胺: {len(am_fps)}/{len(am_smiles_in_pool)} (失败 {n_fail_am})")

    print(f"\n=== 多样性贪心 (Tanimoto<{args.tanimoto_threshold}, Top {args.top_k}) ===")
    selected = []
    used_ald_fps = []
    used_am_fps = []

    for rank, r in enumerate(passed_sorted, 1):
        if len(selected) >= args.top_k:
            break
        ald_s = r["aldehyde_smiles"]
        am_s = r["amine_smiles"]

        ald_fp = ald_fps.get(ald_s)
        am_fp = am_fps.get(am_s)
        if ald_fp is None or am_fp is None:
            continue

        # 只从通过硬规则的配对中选 (adj>0)
        if float(r.get("film_prob_adjusted", 0)) <= 0:
            continue

        if used_ald_fps:
            sims = DataStructs.BulkTanimotoSimilarity(ald_fp, [fp for _, fp in used_ald_fps])
            if max(sims) >= args.tanimoto_threshold:
                continue
        if used_am_fps:
            sims = DataStructs.BulkTanimotoSimilarity(am_fp, [fp for _, fp in used_am_fps])
            if max(sims) >= args.tanimoto_threshold:
                continue

        selected.append({
            "diverse_rank": len(selected) + 1,
            "global_rank_adj": rank,
            **r,
        })
        used_ald_fps.append((ald_s, ald_fp))
        used_am_fps.append((am_s, am_fp))

    print(f"入选: {len(selected)} / Top {args.top_k}")

    # ── 可选: 追加 target 配对 ──
    if args.include_target:
        target_already = any(
            s["aldehyde_smiles"] == TARGET_ALD_CANON
            and s["amine_smiles"] == TARGET_AM_CANON
            for s in selected
        )
        if not target_already:
            # 找 target 在 passed_sorted 中的位置
            for i, r in enumerate(passed_sorted, 1):
                if (r["aldehyde_smiles"] == TARGET_ALD_CANON
                        and r["amine_smiles"] == TARGET_AM_CANON):
                    target_row = {
                        "diverse_rank": len(selected) + 1,
                        "global_rank_adj": i,
                        **r,
                    }
                    selected.append(target_row)
                    print(f"  >>> 追加 target 配对为 #{target_row['diverse_rank']} (raw={r['film_prob_mean']}, adj={r['film_prob_adjusted']})")
                    break
        else:
            print(f"  target 已在 Top {args.top_k} 中, 无需追加")

    print("\n=== Target 配对状态 ===")
    target_in_selected = any(
        s["aldehyde_smiles"] == TARGET_ALD_CANON
        and s["amine_smiles"] == TARGET_AM_CANON
        for s in selected
    )
    print(f"  target 入选 Top 40? {target_in_selected}")
    if not target_in_selected:
        for i, r in enumerate(passed_sorted, 1):
            if (r["aldehyde_smiles"] == TARGET_ALD_CANON
                    and r["amine_smiles"] == TARGET_AM_CANON):
                print(f"  target 在 passed_sorted 中排名: #{i}")
                print(f"  raw={r['film_prob_mean']}, adj={r['film_prob_adjusted']}")
                break
    else:
        for s in selected:
            if (s["aldehyde_smiles"] == TARGET_ALD_CANON
                    and s["amine_smiles"] == TARGET_AM_CANON):
                print(f"  target 在 Top 40 中 diverse_rank: #{s['diverse_rank']}")
                print(f"  raw={s['film_prob_mean']}, adj={s['film_prob_adjusted']}")
                break

    ald_use = Counter(s["aldehyde_smiles"] for s in selected)
    am_use = Counter(s["amine_smiles"] for s in selected)
    print(f"\n=== 醛/胺多样性 ===")
    print(f"  醛: {len(ald_use)} 个不同 (出现多次: {sum(1 for v in ald_use.values() if v>1)})")
    print(f"  胺: {len(am_use)} 个不同 (出现多次: {sum(1 for v in am_use.values() if v>1)})")

    print(f"\n=== 写出 ===")
    if not selected:
        print("  无选中配对, 跳过写文件")
        return
    out_fields = list(rows[0].keys()) + ["diverse_rank", "global_rank_adj"]
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for s in selected:
            w.writerow(s)
    print(f"  Top 40: {args.output} ({len(selected)} 配对)")

    cand_fields = list(rows[0].keys()) + ["diverse_rank"]
    with open(args.candidates, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cand_fields, extrasaction="ignore")
        w.writeheader()
        for s in selected:
            w.writerow(s)
    print(f"  候选: {args.candidates} ({len(selected)} 配对)")

    print(f"\n=== Top 40 摘要 (adj 排序, Tanimoto<{args.tanimoto_threshold}) ===")
    print("{:<4} {:<8} {:<8} {:<7} {:<6} {:<28} {:<28}".format(
        "#", "raw", "adj", "n_ald", "n_am", "aldehyde", "amine"))
    print("-" * 110)
    for s in selected:
        is_target = (s["aldehyde_smiles"] == TARGET_ALD_CANON
                     and s["amine_smiles"] == TARGET_AM_CANON)
        marker = " <<< TARGET" if is_target else ""
        print("{:<4} {:<8.4f} {:<8.4f} {:<7} {:<6} {:<28} {:<28}{}".format(
            s["diverse_rank"],
            float(s["film_prob_mean"]),
            float(s["film_prob_adjusted"]),
            s["ald_n"],
            s["am_n"],
            (s["aldehyde_name"] or s["aldehyde_smiles"][:28])[:28],
            (s["amine_name"] or s["amine_smiles"][:28])[:28],
            marker))


if __name__ == "__main__":
    main()
