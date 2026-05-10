"""虚拟氟化修正 — 对非含氟组合评估加氟后的成膜提升。

工作流:
  1. 从 Route A 全量非F×非F 配对的 Top N 中取样本
  2. 对每个单体尝试虚拟氟化 (1-2 F)
  3. 生成: F-醛×原胺、原醛×F-胺、F-醛×F-胺
  4. 重新预测 margin，计算提升幅度 Δ
  5. 输出 fluorination_correction.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screening.features import FeatureEngineer
from src.chemistry.monomer import MonomerLibrary
from src.chemistry.fluorination import FluorineDetector, virtual_fluorination
from src.utils.logger import setup_logger

logger = setup_logger("virtual_f")


def predict_margin(model, scaler, selected_features, feature_eng, ald_smi, am_smi):
    """预测单体对的 margin score。"""
    from rdkit import Chem

    ald_mol = Chem.MolFromSmiles(ald_smi)
    am_mol = Chem.MolFromSmiles(am_smi)
    if ald_mol is None or am_mol is None:
        return np.nan
    try:
        feat = feature_eng.featurize_monomer_pair(ald_mol, am_mol)
        feat = feat.reshape(1, -1)
        if selected_features is not None and len(selected_features) > 0:
            feat = feat[:, selected_features]
        if scaler is not None:
            feat = scaler.transform(feat)
        return float(model.predict(feat, output_margin=True)[0])
    except Exception:
        return np.nan


def fluorinate_variants(smi: str, max_f: int = 2):
    """生成单体的氟化变体列表 [(label, f_smi), ...]。
    先尝试 +1F，成功后再在 +1F 基础上尝试 +2F。
    """
    variants = []
    for n in range(1, max_f + 1):
        f_smi = virtual_fluorination(smi, n_f=n)
        if f_smi and f_smi != smi:
            variants.append((f"+{n}F", f_smi))
    return variants


def main():
    parser = argparse.ArgumentParser(description="虚拟氟化修正")
    parser.add_argument("--llm", default="data/processed/monomer_smiles_llm.json")
    parser.add_argument("--model-dir", default="models/v1.0")
    parser.add_argument("--cache", default="data/processed/monomer_smiles_cache.json")
    parser.add_argument("--output", default="data/processed/fluorination_correction.csv")
    parser.add_argument("--top", type=int, default=50,
                       help="取 Top N 非F×非F 对进行虚拟氟化分析")
    parser.add_argument("--max-f", type=int, default=2,
                       help="最多添加 F 原子数")
    args = parser.parse_args()

    # 1. 加载模型
    import pickle
    from rdkit import Chem

    model_path = os.path.join(args.model_dir, "xgboost_model.pkl")
    if not os.path.exists(model_path):
        logger.error(f"模型不存在: {model_path}")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaler = None
    sp = os.path.join(args.model_dir, "scaler.pkl")
    if os.path.exists(sp):
        with open(sp, "rb") as f:
            scaler = pickle.load(f)

    selected_features = None
    ip = os.path.join(args.model_dir, "model_info.json")
    if os.path.exists(ip):
        with open(ip, "r", encoding="utf-8") as f:
            info = json.load(f)
            sf = info.get("selected_features", [])
            if sf and len(sf) > 0:
                selected_features = np.array(sf)

    # 2. 加载 2D 单体
    from scripts.screen_monomers_2d import extract_2d_monomers

    monomers = extract_2d_monomers(args.llm)
    logger.info(f"2D 单体: {len(monomers)}")

    aldehydes = monomers[monomers["is_aldehyde"]]
    amines = monomers[monomers["is_amine"]]

    # 3. 生成 非F×非F 配对（仅用文献数最多的单体，避免全组合爆炸）
    nf_ald = aldehydes[~aldehydes["has_fluorine"]].sort_values("n_papers", ascending=False)
    nf_am = amines[~amines["has_fluorine"]].sort_values("n_papers", ascending=False)

    # 取文献支持度最高的单体（各不超过 50 个）
    top_ald = nf_ald.head(50)
    top_am = nf_am.head(50)
    logger.info(f"非F-醛: {len(nf_ald)}→取{len(top_ald)}, 非F-胺: {len(nf_am)}→取{len(top_am)}")

    pairs = []
    for _, ald in top_ald.iterrows():
        for _, am in top_am.iterrows():
            if ald["smiles"] == am["smiles"]:
                continue
            pairs.append({
                "aldehyde_name": ald["name"],
                "amine_name": am["name"],
                "aldehyde_smiles": ald["smiles"],
                "amine_smiles": am["smiles"],
                "aldehyde_topo": ald.get("topology", "?"),
                "amine_topo": am.get("topology", "?"),
            })
    logger.info(f"非F×非F 配对 (预筛选): {len(pairs)}")

    # 4. 预测原始 margin
    monomer_lib = MonomerLibrary(cache_path=args.cache, use_pubchem=False)
    feature_eng = FeatureEngineer(monomer_lib)
    f_detector = FluorineDetector()

    logger.info("预测原始 margin...")
    for p in pairs:
        p["margin_orig"] = predict_margin(
            model, scaler, selected_features, feature_eng,
            p["aldehyde_smiles"], p["amine_smiles"],
        )

    # 取 Top N 有效对
    pairs_df = pd.DataFrame(pairs)
    valid_pairs = pairs_df.dropna(subset=["margin_orig"]).sort_values(
        "margin_orig", ascending=False
    )
    top_nf = valid_pairs.head(args.top).to_dict("records")
    logger.info(f"选取 Top {len(top_nf)} 非F对进行氟化修正")

    # 5. 虚拟氟化 & 重新预测
    from scripts.screen_monomers_2d import _topology_label

    results = []
    for i, p in enumerate(top_nf):
        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(top_nf)}")

        ald_smi = p["aldehyde_smiles"]
        am_smi = p["amine_smiles"]
        orig_margin = p["margin_orig"]

        # 醛氟化变体
        ald_variants = fluorinate_variants(ald_smi, args.max_f)
        # 胺氟化变体
        am_variants = fluorinate_variants(am_smi, args.max_f)

        # 策略 1: F-醛 × 原胺
        for v_label, f_ald_smi in ald_variants:
            margin = predict_margin(
                model, scaler, selected_features, feature_eng,
                f_ald_smi, am_smi,
            )
            if not np.isnan(margin):
                results.append({
                    "aldehyde_orig": p["aldehyde_name"],
                    "amine_orig": p["amine_name"],
                    "aldehyde_smiles_orig": ald_smi,
                    "amine_smiles_orig": am_smi,
                    "strategy": f"醛{v_label} × 原胺",
                    "fluorinated_smiles": f_ald_smi,
                    "partner_smiles": am_smi,
                    "fluorinated_side": "aldehyde",
                    "margin_orig": orig_margin,
                    "margin_fluorinated": margin,
                    "margin_delta": margin - orig_margin,
                })

        # 策略 2: 原醛 × F-胺
        for v_label, f_am_smi in am_variants:
            margin = predict_margin(
                model, scaler, selected_features, feature_eng,
                ald_smi, f_am_smi,
            )
            if not np.isnan(margin):
                results.append({
                    "aldehyde_orig": p["aldehyde_name"],
                    "amine_orig": p["amine_name"],
                    "aldehyde_smiles_orig": ald_smi,
                    "amine_smiles_orig": am_smi,
                    "strategy": f"原醛 × 胺{v_label}",
                    "fluorinated_smiles": f_am_smi,
                    "partner_smiles": ald_smi,
                    "fluorinated_side": "amine",
                    "margin_orig": orig_margin,
                    "margin_fluorinated": margin,
                    "margin_delta": margin - orig_margin,
                })

        # 策略 3: F-醛 × F-胺 (仅当两侧都有变体时)
        for va_label, f_ald_smi in ald_variants:
            for vb_label, f_am_smi in am_variants:
                margin = predict_margin(
                    model, scaler, selected_features, feature_eng,
                    f_ald_smi, f_am_smi,
                )
                if not np.isnan(margin):
                    results.append({
                        "aldehyde_orig": p["aldehyde_name"],
                        "amine_orig": p["amine_name"],
                        "aldehyde_smiles_orig": ald_smi,
                        "amine_smiles_orig": am_smi,
                        "strategy": f"醛{va_label} × 胺{vb_label}",
                        "fluorinated_smiles": f"{f_ald_smi}||{f_am_smi}",
                        "partner_smiles": "",
                        "fluorinated_side": "both",
                        "margin_orig": orig_margin,
                        "margin_fluorinated": margin,
                        "margin_delta": margin - orig_margin,
                    })

    # 6. 汇总 & 排序
    if not results:
        logger.error("无有效氟化结果（所有变体预测失败）")
        sys.exit(1)

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("margin_delta", ascending=False)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    res_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    # 7. 报告
    print("\n" + "=" * 86)
    print("  虚拟氟化修正 — 加氟对成膜预测 (margin) 的提升")
    print("=" * 86)

    # Top 15 提升最大的氟化方案
    print(f"\n  Top 15 氟化提升最大的单体对 (共 {len(res_df)} 种变体):\n")
    print(f"  {'#':3s}  {'ΔMargin':9s}  {'原始(margin)':13s}  {'策略':18s}  {'原始醛':24s}  {'原始胺':20s}")
    print("  " + "-" * 84)

    printed = set()
    count = 0
    for _, row in res_df.iterrows():
        key = (row["aldehyde_smiles_orig"], row["amine_smiles_orig"])
        if key in printed:
            continue
        if count >= 15:
            break
        printed.add(key)
        count += 1
        ald = str(row["aldehyde_orig"])[:22]
        am = str(row["amine_orig"])[:18]
        strategy = str(row["strategy"])[:17]
        print(
            f"  [{count:2d}]  {row['margin_delta']:+7.2f}    "
            f"{row['margin_orig']:6.2f} → {row['margin_fluorinated']:6.2f}    "
            f"{strategy:18s}  {ald:24s}  {am:20s}"
        )

    n_positive = int((res_df["margin_delta"] > 0).sum())
    mean_delta = res_df["margin_delta"].mean()
    max_delta = res_df["margin_delta"].max()

    # 按氟化侧统计
    side_stats = res_df.groupby("fluorinated_side")["margin_delta"].agg(["mean", "max", "count"])
    print(f"\n  氟化效果统计:")
    for side, row in side_stats.iterrows():
        labels = {"aldehyde": "氟化醛侧", "amine": "氟化胺侧", "both": "氟化双侧"}
        print(f"    {labels.get(side, side):12s}: 平均Δ={row['mean']:+.2f}, "
              f"最大Δ={row['max']:+.2f}, N={int(row['count'])}")

    print(f"\n  总览: {n_positive}/{len(res_df)} 变体有正向提升 ({100*n_positive/len(res_df):.0f}%)")
    print(f"  平均ΔMargin={mean_delta:+.2f}, 最大ΔMargin={max_delta:+.2f}")
    print(f"  结果已保存: {args.output}")
    print("=" * 86)

    monomer_lib.flush_cache()


if __name__ == "__main__":
    main()
