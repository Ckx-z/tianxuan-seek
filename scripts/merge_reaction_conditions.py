"""将结构化反应条件合并到特征矩阵，生成增强版 X_augmented.npz。"""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


CATEGORICAL_FIELDS = {
    "temperature_category": ["room_temp", "mild_heat", "solvothermal", "high_temp", "reflux", "unknown"],
    "solvent_system": ["monophasic", "biphasic", "solid_state", "unknown"],
    "solvent_main": ["DMF", "NMP", "dioxane", "mesitylene", "DMAc", "water", "EtOH", "MeOH",
                     "THF", "CHCl3", "CH2Cl2", "acetone", "toluene", "o-DCB", "n-BuOH",
                     "ethylene_glycol", "acetic_acid", "none", "other"],
    "catalyst_type": ["acetic_acid", "lewis_acid", "base", "none", "other", "unknown"],
    "synthesis_mode": ["solvothermal", "interfacial", "mechanochemical", "room_temp", "reflux",
                       "ionothermal", "other"],
    "interface_type": ["liquid_liquid", "liquid_solid", "gas_liquid", "solid_solid", "none", "unknown"],
}
# 去重 (处理 unknown 已在列表中的情况)
CATEGORICAL_FIELDS = {k: list(dict.fromkeys(v)) for k, v in CATEGORICAL_FIELDS.items()}

NUMERIC_FIELDS = ["temperature_c", "annealing_temp_c"]
BOOL_FIELDS = ["solvent_mixed", "has_annealing"]


def encode_conditions(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """编码反应条件为数值特征矩阵。"""
    encoded_parts = []
    feature_names = []

    # 数值字段
    for col in NUMERIC_FIELDS:
        vals = df[col].fillna(-999).values.reshape(-1, 1)
        encoded_parts.append(vals)
        feature_names.append(f"cond_{col}")

    # 布尔字段
    for col in BOOL_FIELDS:
        vals = df[col].fillna(False).astype(float).values.reshape(-1, 1)
        encoded_parts.append(vals)
        feature_names.append(f"cond_{col}")

    # is_homogeneous: True=1, False=0, null=0.5
    homo = df["is_homogeneous"].map({True: 1.0, False: 0.0}).fillna(0.5).values.reshape(-1, 1)
    encoded_parts.append(homo)
    feature_names.append("cond_is_homogeneous")

    # 类别字段 — one-hot
    for col, categories in CATEGORICAL_FIELDS.items():
        values = df[col].fillna("unknown").values.reshape(-1, 1)
        enc = OneHotEncoder(categories=[categories], sparse_output=False, handle_unknown="ignore")
        oh = enc.fit_transform(values)
        encoded_parts.append(oh)
        feature_names.extend(f"cond_{col}_{c}" for c in enc.categories_[0])

    X_cond = np.hstack(encoded_parts)
    return X_cond.astype(np.float32), feature_names


def main():
    parser = argparse.ArgumentParser(description="合并反应条件到特征矩阵")
    parser.add_argument("--features", default="data/processed/X_features.npz")
    parser.add_argument("--labels", default="data/processed/y_labels.npy")
    parser.add_argument("--meta", default="data/processed/label_metadata.csv")
    parser.add_argument("--conditions", default="data/processed/reaction_conditions.csv")
    parser.add_argument("--output", default="data/processed/X_augmented.npz")
    parser.add_argument("--output-names", default="data/processed/feature_names_augmented.csv")
    args = parser.parse_args()

    X = np.load(args.features)["X"]
    y = np.load(args.labels)
    meta = pd.read_csv(args.meta, encoding="utf-8-sig")
    conds = pd.read_csv(args.conditions, encoding="utf-8-sig")

    print(f"原始特征: {X.shape}")
    print(f"标签分布: 正={y.sum()}, 负={len(y) - y.sum()}")
    print(f"反应条件: {len(conds)} 条")

    # 按 literature_id 对齐
    cond_map = {}
    for _, row in conds.iterrows():
        lid = row["literature_id"]
        if pd.isna(lid):
            continue
        cond_map[str(lid)] = row

    # 为每个标记样本构建条件向量
    cond_rows = []
    matched = 0
    empty_template = conds.iloc[0].copy()
    for col in conds.columns:
        if col == "literature_id":
            continue
        empty_template[col] = np.nan
    for _, row in meta.iterrows():
        lid = str(row["literature_id"])
        if lid in cond_map:
            cond_rows.append(cond_map[lid])
            matched += 1
        else:
            empty = empty_template.copy()
            empty["literature_id"] = lid
            cond_rows.append(empty)

    cond_df = pd.DataFrame(cond_rows)
    print(f"匹配: {matched}/{len(meta)}")

    X_cond, cond_names = encode_conditions(cond_df)
    print(f"条件特征维度: {len(cond_names)}")

    X_aug = np.hstack([X, X_cond]).astype(np.float32)
    print(f"增强特征: {X_aug.shape}")

    # 验证无 NaN
    if np.any(np.isnan(X_aug)):
        nan_cols = np.where(np.isnan(X_aug).any(axis=0))[0]
        print(f"警告: {len(nan_cols)} 列含 NaN: {nan_cols[:10]}...")
    else:
        print("验证通过: 无 NaN")

    np.savez_compressed(args.output, X=X_aug)
    pd.DataFrame({"feature_name": cond_names}).to_csv(args.output_names, index=False, encoding="utf-8-sig")
    print(f"已保存: {args.output} ({X_aug.shape[1]} 维)")
    print(f"条件特征名: {args.output_names}")


if __name__ == "__main__":
    main()
