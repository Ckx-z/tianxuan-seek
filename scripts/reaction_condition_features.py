"""从 YAML 提取反应条件特征 — 溶剂 (10-dim one-hot) + 温度 (1-dim)。

输出: data/processed/reaction_features.npz
  - feat: (N, 11) float32  — 溶剂10维 + 温度1维
  - mask: (N,) bool       — 该样本是否提取到有效反应条件
"""
import json
import os
import re
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("rxn_feat")

# ── YAML 目录与 source group 的映射 ──
YAML_DIRS = {
    "group1": "data/structured",
    "group1_unprocessed": "data/structured",
    "group2": "data/structured_new",
    "group3": "data/structured_new3",
}

# ── 溶剂类别定义 (优先级顺序) ──
SOLVENT_CATEGORIES = [
    "dioxane+mesitylene",
    "dioxane_pure",
    "dioxane+aqueous",
    "mesitylene_based",
    "DCB+BuOH",
    "DMF_based",
    "alcohol",
    "water_only",
    "water+organic",
    "not_specified",
    # 其余归入 "other"
]


def _find_yaml(literature_id: str, source: str) -> str | None:
    """根据 literature_id 和 source 定位 YAML 文件。"""
    base = YAML_DIRS.get(source, "data/structured")
    if not os.path.isdir(base):
        return None

    # 精确匹配
    exact = os.path.join(base, literature_id + ".yaml")
    if os.path.isfile(exact):
        return exact

    # 模糊匹配 (处理文件名前缀差异)
    for fn in os.listdir(base):
        if not fn.endswith(".yaml"):
            continue
        # 去除扩展名后比对
        stem = fn[:-5]
        if stem == literature_id or literature_id in stem or stem in literature_id:
            return os.path.join(base, fn)

    return None


def _classify_solvent(s: str) -> int:
    """将溶剂字符串映射到类别索引 (0-10, 10=other)。"""
    s_lower = s.lower()

    is_dioxane = "dioxane" in s_lower or "二氧六环" in s_lower
    is_mesitylene = "mesitylene" in s_lower or "均三甲苯" in s_lower
    is_dcb = "dcb" in s_lower or "dichlorobenzene" in s_lower or "二氯苯" in s_lower
    is_dmf = "dmf" in s_lower
    is_water = "water" in s_lower or "水" in s_lower or "aqueous" in s_lower
    is_alcohol = any(kw in s_lower for kw in [
        "ethanol", "methanol", "butanol", "etoh", "meoh", "buoh", "乙醇", "甲醇", "丁醇"
    ])

    if is_dioxane and is_mesitylene:
        return 0  # dioxane+mesitylene
    if is_dioxane and (is_water or "acetic" in s_lower or "醋酸" in s_lower or "乙酸" in s_lower):
        return 2  # dioxane+aqueous
    if is_dioxane:
        return 1  # dioxane_pure
    if is_mesitylene:
        return 3  # mesitylene_based
    if is_dcb and is_alcohol:
        return 4  # DCB+BuOH
    if is_dmf:
        return 5  # DMF_based
    if is_alcohol:
        return 6  # alcohol
    if is_water and ("organic" in s_lower or "有机" in s_lower or
                     "dcm" in s_lower or "dichloromethane" in s_lower or
                     "二氯甲烷" in s_lower):
        return 8  # water+organic
    if is_water:
        return 7  # water_only
    if any(kw in s_lower for kw in ["未明确", "未提及", "not specified"]):
        return 9  # not_specified
    return 10  # other


def _extract_temperature(t_str: str) -> float | None:
    """从温度字符串中提取数值 (摄氏度)。"""
    matches = re.findall(r'(\d+)\s*°?\s*[Cc]', str(t_str))
    if matches:
        return float(matches[0])
    # 尝试匹配 "室温" 类
    if any(kw in str(t_str).lower() for kw in [
        "room temperature", "室温", "ambient", "常温", "r.t.", "rt"
    ]):
        return 25.0
    return None


def _solvent_onehot(cat_idx: int) -> np.ndarray:
    """生成溶剂 one-hot 向量。other → 全零向量。"""
    n_explicit = len(SOLVENT_CATEGORIES)  # 10 个显式类别
    vec = np.zeros(n_explicit, dtype=np.float32)
    if cat_idx < n_explicit:
        vec[cat_idx] = 1.0
    # cat_idx == 10 (other) → all-zero
    return vec


def main():
    # ── 加载元数据 ──
    meta = pd.read_csv("data/processed/label_metadata_v3.csv", encoding="utf-8-sig")
    logger.info(f"加载 {len(meta)} 条样本")

    # ── 构建 YAML 缓存 (避免重复读取同一文献) ──
    yaml_cache = {}  # literature_id → dict

    n_solvent = len(SOLVENT_CATEGORIES)  # 10
    n_features = n_solvent + 1  # 10 solvent + 1 temp = 11
    features = np.zeros((len(meta), n_features), dtype=np.float32)
    mask = np.zeros(len(meta), dtype=bool)

    temp_values = []
    solvent_counts = Counter()
    yaml_hits = 0
    yaml_misses = 0

    for i, (_, row) in enumerate(meta.iterrows()):
        lid = row["literature_id"]
        src = row["source"]

        # 读取 YAML (优先缓存)
        if lid not in yaml_cache:
            yaml_path = _find_yaml(lid, src)
            if yaml_path:
                try:
                    with open(yaml_path, encoding="utf-8") as f:
                        yaml_cache[lid] = yaml.safe_load(f)
                    yaml_hits += 1
                except Exception:
                    yaml_cache[lid] = None
                    yaml_misses += 1
            else:
                yaml_cache[lid] = None
                yaml_misses += 1

        data = yaml_cache.get(lid)
        if data is None or not isinstance(data, dict):
            mask[i] = False
            continue

        # ── 溶剂 one-hot ──
        solvent_str = str(data.get("solvent", ""))
        if solvent_str and solvent_str.lower() not in ("nan", "none", "n/a", "null", ""):
            cat_idx = _classify_solvent(solvent_str)
            features[i, :n_solvent] = _solvent_onehot(cat_idx)
            solvent_counts[cat_idx] += 1
            mask[i] = True

        # ── 温度 (标准化在训练时做) ──
        temp_str = str(data.get("reaction_temperature", ""))
        temp_c = _extract_temperature(temp_str)
        if temp_c is not None:
            features[i, n_solvent] = temp_c
            temp_values.append(temp_c)
            mask[i] = True

    # ── 统计 ──
    logger.info(f"YAML 命中: {yaml_hits}, 未命中: {yaml_misses}")
    logger.info(f"反应条件有效样本: {mask.sum()}/{len(mask)} ({mask.sum()/len(mask)*100:.1f}%)")
    if temp_values:
        logger.info(f"温度提取: {len(temp_values)} 个, 范围 [{min(temp_values):.0f}, {max(temp_values):.0f}] C, "
                    f"中位数 {np.median(temp_values):.0f} C")
    else:
        logger.info("温度提取: 0 个")

    # ── 溶剂分布 ──
    label_map = {0: "dioxane+mesitylene", 1: "dioxane_pure", 2: "dioxane+aqueous",
                 3: "mesitylene_based", 4: "DCB+BuOH", 5: "DMF_based",
                 6: "alcohol", 7: "water_only", 8: "water+organic",
                 9: "not_specified", 10: "other"}
    logger.info("溶剂类别分布:")
    for idx, cnt in solvent_counts.most_common():
        pct = cnt / mask.sum() * 100 if mask.sum() > 0 else 0
        logger.info(f"  {label_map.get(idx, str(idx)):25s}: {cnt:4d} ({pct:.1f}%)")

    # ── 保存 ──
    out_path = "data/processed/reaction_features.npz"
    np.savez_compressed(out_path, feat=features, mask=mask)
    logger.info(f"特征已保存: {out_path} → feat={features.shape}, mask={mask.shape}")

    # ── 人均特征密度 ──
    n_nonzero_solvent = (features[:, :n_solvent].sum(axis=1) > 0).sum()
    n_nonzero_temp = (features[:, n_solvent] > 0).sum()
    logger.info(f"溶剂非零: {n_nonzero_solvent}/{len(mask)}, 温度非零: {n_nonzero_temp}/{len(mask)}")


if __name__ == "__main__":
    main()
