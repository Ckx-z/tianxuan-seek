"""路线 A 单体筛选脚本。

输入: data/fluorofilm.db + models/v1.0/
输出: data/processed/route_a_top20.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.monomer import MonomerLibrary
from src.screening.features import FeatureEngineer
from src.screening.predict import MonomerScreener
from src.utils.logger import setup_logger

logger = setup_logger("screen")


def main():
    parser = argparse.ArgumentParser(description="路线 A 单体筛选 → Top N")
    parser.add_argument("--db", default="data/fluorofilm.db")
    parser.add_argument("--model-dir", default="models/v1.0")
    parser.add_argument("--cache", default="data/processed/monomer_smiles_cache.json")
    parser.add_argument("--output", default="data/processed/route_a_top20.csv")
    parser.add_argument("--top", type=int, default=20, help="输出 Top N")
    parser.add_argument("--no-pubchem", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.model_dir, "xgboost_model.pkl")):
        logger.error(
            f"模型文件不存在: {args.model_dir}/xgboost_model.pkl\n"
            "请先运行: python scripts/build_features.py --no-pubchem && "
            "python scripts/train_model.py --no-tune"
        )
        sys.exit(1)

    # 初始化
    monomer_lib = MonomerLibrary(cache_path=args.cache,
                                  use_pubchem=not args.no_pubchem)
    feature_eng = FeatureEngineer(monomer_lib)
    screener = MonomerScreener(monomer_lib, feature_eng,
                                model_dir=args.model_dir)

    # 筛选
    top_pairs = screener.screen_top_n(n=args.top)

    if len(top_pairs) == 0:
        logger.error("未生成任何有效结果")
        sys.exit(1)

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    top_pairs.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nTop {args.top} 结果已保存至: {args.output}")

    # 打印表格
    print("\n" + "=" * 70)
    print(f"  Route A Top {min(args.top, len(top_pairs))} 单体对")
    print("=" * 70)
    cols = ["aldehyde", "amine", "pair_type", "film_probability",
            "fluorinated_score", "fluorination_gain"]
    display_cols = [c for c in cols if c in top_pairs.columns]
    for i, row in top_pairs[display_cols].iterrows():
        ald = row.get("aldehyde", "")
        am = row.get("amine", "")
        prob = row.get("film_probability", 0)
        fgain = row.get("fluorination_gain", 0)
        print(f"  [{i + 1:2d}] {prob:.3f} (ΔF={fgain:+.3f})  "
              f"{ald[:30]:30s} + {am[:30]:30s}")


if __name__ == "__main__":
    main()
