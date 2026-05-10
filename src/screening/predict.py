"""路线 A 筛选与预测模块。

流程：
  1. 从文献数据库提取所有唯一单体
  2. 筛选可形成亚胺键的（醛基/胺基）
  3. 按含氟/不含氟分为四组
  4. 路线 A: 四种配对方式
     - F-醛 × 非F-胺 + 非F-醛 × F-胺（标准含氟配对）
     - F-醛 × F-胺（双氟配对）
     - 非F-醛 × 非F-胺（无氟配对，后续虚拟氟化修正）
  5. 预测每对成膜概率 → 排名 → Top N
  6. 虚拟氟化修正：
     - 单F组合：非F单体 +1F
     - 无F组合：醛+1F / 胺+1F / 两者同时+1F → 取最佳
"""
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.chemistry.fluorination import FluorineDetector, virtual_fluorination
from src.chemistry.imine_check import ImineChecker
from src.chemistry.monomer import MonomerLibrary, extract_monomer_names
from src.screening.features import FeatureEngineer
from src.utils.db import get_all_records, init_db
from src.utils.logger import setup_logger

logger = setup_logger("predictor")


class MonomerScreener:
    """单体筛选器 — 路线 A 筛选 + 预测 + 虚拟氟化修正。"""

    def __init__(self, monomer_lib: MonomerLibrary,
                 feature_eng: FeatureEngineer,
                 model_dir: str = "models/v1.0"):
        self.monomer_lib = monomer_lib
        self.feature_eng = feature_eng
        self.imine_checker = ImineChecker()
        self.f_detector = FluorineDetector()
        self.model_dir = model_dir
        self._model = None
        self._loaded = False

    # -------------------------------------------------------------------
    # 单体提取
    # -------------------------------------------------------------------

    def extract_all_monomers_from_literature(
        self, db_path: str = "data/fluorofilm.db",
    ) -> pd.DataFrame:
        """从文献数据库提取所有唯一单体及其属性。"""
        conn = init_db(db_path)
        records = get_all_records(conn)
        conn.close()

        monomer_info: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            reagent = rec.get("reagent", "") or ""
            names = extract_monomer_names(reagent)
            for name in names:
                if name not in monomer_info:
                    mol = self.monomer_lib.get_mol(name)
                    if mol is None:
                        continue
                    info = {
                        "name": name,
                        "smiles": self.monomer_lib.resolve(name),
                        "is_aldehyde": self.imine_checker.is_aldehyde(mol),
                        "is_amine": self.imine_checker.is_amine(mol),
                        "has_fluorine": self.f_detector.has_fluorine(mol),
                        "n_f_atoms": self.f_detector.count_fluorine(mol),
                        "has_cf3": self.f_detector.has_cf3(mol),
                        "n_papers": 1,
                    }
                    monomer_info[name] = info
                elif name in monomer_info:
                    monomer_info[name]["n_papers"] += 1

        df = pd.DataFrame(monomer_info.values())
        logger.info(f"提取唯一单体: {len(df)} 个 (来自 {len(records)} 篇文献)")
        return df.sort_values("n_papers", ascending=False)

    # -------------------------------------------------------------------
    # 筛选
    # -------------------------------------------------------------------

    def filter_imine_capable(self, monomers: pd.DataFrame) -> pd.DataFrame:
        """筛选可形成亚胺键的单体。"""
        mask = monomers["is_aldehyde"] | monomers["is_amine"]
        result = monomers[mask].copy()
        logger.info(
            f"亚胺键可用单体: {len(result)}/{len(monomers)} "
            f"({100 * len(result) / max(len(monomers), 1):.1f}%)"
        )
        return result

    def split_by_fluorine(self, monomers: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """按氟含量和功能基分为四组。"""
        df = monomers.copy()
        df["group"] = "other"
        df.loc[df["is_aldehyde"] & df["has_fluorine"], "group"] = "f_aldehyde"
        df.loc[df["is_aldehyde"] & ~df["has_fluorine"], "group"] = "nonf_aldehyde"
        df.loc[df["is_amine"] & df["has_fluorine"], "group"] = "f_amine"
        df.loc[df["is_amine"] & ~df["has_fluorine"], "group"] = "nonf_amine"

        result = {g: df[df["group"] == g].copy() for g in df["group"].unique()}
        for g, sub in result.items():
            logger.info(f"  {g}: {len(sub)} 个单体")
        return result

    # -------------------------------------------------------------------
    # 路线 A 配对
    # -------------------------------------------------------------------

    def generate_route_a_pairs(self, groups: Dict[str, pd.DataFrame],
                               ) -> pd.DataFrame:
        """生成路线 A 全部四种组合。

        (1) F-醛 × 非F-胺    — 标准含氟配对
        (2) 非F-醛 × F-胺    — 标准含氟配对
        (3) F-醛 × F-胺      — 双氟配对
        (4) 非F-醛 × 非F-胺  — 无氟配对（后续通过虚拟氟化修正）
        """
        pairs = []
        f_ald = groups.get("f_aldehyde", pd.DataFrame())
        nonf_ald = groups.get("nonf_aldehyde", pd.DataFrame())
        f_am = groups.get("f_amine", pd.DataFrame())
        nonf_am = groups.get("nonf_amine", pd.DataFrame())

        pair_specs = [
            (f_ald, nonf_am, "F-aldehyde × nonF-amine"),
            (nonf_ald, f_am, "nonF-aldehyde × F-amine"),
            (f_ald, f_am, "F-aldehyde × F-amine"),
            (nonf_ald, nonf_am, "nonF-aldehyde × nonF-amine"),
        ]

        for ald_df, am_df, pair_type in pair_specs:
            if len(ald_df) == 0 or len(am_df) == 0:
                logger.info(f"  跳过 {pair_type}: 醛={len(ald_df)}, 胺={len(am_df)}（不足）")
                continue
            for _, ald in ald_df.iterrows():
                for _, am in am_df.iterrows():
                    pairs.append({
                        "aldehyde": ald["name"],
                        "amine": am["name"],
                        "aldehyde_f": ald["has_fluorine"],
                        "amine_f": am["has_fluorine"],
                        "pair_type": pair_type,
                    })
            logger.info(f"  {pair_type}: {len(ald_df)} × {len(am_df)} = {len(ald_df) * len(am_df)} 组合")

        df = pd.DataFrame(pairs)
        logger.info(f"路线 A 总组合数: {len(df)}")
        return df

    # -------------------------------------------------------------------
    # 预测
    # -------------------------------------------------------------------

    def _load_model(self):
        """加载训练好的模型、scaler 和特征选择索引。"""
        if self._loaded:
            return
        import json
        xgb_path = os.path.join(self.model_dir, "xgboost_model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        info_path = os.path.join(self.model_dir, "model_info.json")
        if not os.path.exists(xgb_path):
            raise FileNotFoundError(
                f"模型文件不存在: {xgb_path}，请先运行 train_model.py"
            )
        with open(xgb_path, "rb") as f:
            self._model = pickle.load(f)
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
        self._selected_features = None
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
                sf = info.get("selected_features", [])
                if sf and len(sf) > 0:
                    self._selected_features = np.array(sf)
        self._loaded = True

    def predict_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """对单体对预测成膜概率。"""
        self._load_model()
        results = []

        for _, row in pairs_df.iterrows():
            ald_name, am_name = row["aldehyde"], row["amine"]
            ald_mol = self.monomer_lib.get_mol(ald_name)
            am_mol = self.monomer_lib.get_mol(am_name)
            if ald_mol is None or am_mol is None:
                results.append(float("nan"))
                continue

            try:
                feat = self.feature_eng.featurize_monomer_pair(ald_mol, am_mol)
            except Exception:
                results.append(float("nan"))
                continue

            feat = feat.reshape(1, -1)
            # 应用训练时的特征选择
            if (hasattr(self, "_selected_features")
                    and self._selected_features is not None
                    and len(self._selected_features) > 0):
                feat = feat[:, self._selected_features]
            if hasattr(self, "_scaler") and self._scaler is not None:
                feat = self._scaler.transform(feat)
            prob = self._model.predict_proba(feat)[0, 1]
            results.append(prob)

        pairs_df["film_probability"] = results
        return pairs_df

    def screen_top_n(self, n: int = 20,
                     pre_fluorinate_nonf: bool = True) -> pd.DataFrame:
        """完整 Route A 筛选流程 → 返回 Top N。

        自动策略选择：
        - 若含氟单体总数 ≥ 60 且 F-醛 ≥ 15 且 F-胺 ≥ 15：
          使用四配对氟策略（F-醛×非F-胺 + 非F-醛×F-胺 + F-醛×F-胺 + 非F-醛×非F-胺）
        - 否则：简化为全量自由配对，仅按成膜概率排名（不区分是否含氟）

        参数:
            n: 输出的 Top N 数量
            pre_fluorinate_nonf: 仅氟策略模式生效，非F配对的 film_probability
                基于虚拟氟化结构预测，与含氟配对在同一基准竞争。
        """
        logger.info("=" * 50)
        logger.info("开始路线 A 筛选")

        # Step 1-2: 提取 + 过滤
        monomers = self.extract_all_monomers_from_literature()
        imine_monomers = self.filter_imine_capable(monomers)

        # Step 3: 分组
        groups = self.split_by_fluorine(imine_monomers)

        # 统计含氟单体数量，决定策略
        n_f_ald = len(groups.get("f_aldehyde", pd.DataFrame()))
        n_f_am = len(groups.get("f_amine", pd.DataFrame()))
        n_f_total = n_f_ald + n_f_am

        logger.info(
            f"含氟单体统计: F-醛={n_f_ald}, F-胺={n_f_am}, 总计={n_f_total}"
        )

        use_fluorine_strategy = (
            n_f_total >= 30 and n_f_ald >= 15 and n_f_am >= 15
        )

        if not use_fluorine_strategy:
            logger.warning(
                f"含氟单体数量不足 (需 F总计≥30 且 F-醛≥15 且 F-胺≥15，"
                f"当前 F总计={n_f_total} F-醛={n_f_ald} F-胺={n_f_am})，"
                f"切换为全量自由配对模式（不区分含氟）"
            )
            # 简化模式：所有亚胺单体自由配对
            pairs = self._generate_all_pairs(imine_monomers)
        else:
            logger.info("含氟单体充足，启用四配对氟策略")
            pairs = self.generate_route_a_pairs(groups)

        if len(pairs) == 0:
            logger.warning("未生成任何配对")
            return pd.DataFrame()

        # Step 5: 预测
        pairs = self.predict_pairs(pairs)

        # Step 5b: 仅氟策略模式下对非F×非F配对提前虚拟氟化
        if use_fluorine_strategy and pre_fluorinate_nonf:
            pairs = self._pre_fluorinate_nonf_pairs(pairs)

        valid = pairs.dropna(subset=["film_probability"])
        ranked = valid.sort_values("film_probability", ascending=False)
        top = ranked.head(n).reset_index(drop=True)

        # Step 6: 虚拟氟化修正（仅氟策略模式）
        if use_fluorine_strategy:
            top = self.virtual_fluorination_correction(top)
        else:
            # 简化模式：补充空的氟化字段以保持输出格式一致
            top["fluorinated_score"] = top["film_probability"]
            top["fluorination_gain"] = 0.0

        logger.info(f"Top {n} 筛选完成 (策略: {'氟策略' if use_fluorine_strategy else '全量自由配对'})")
        return top

    def _generate_all_pairs(self, monomers: pd.DataFrame) -> pd.DataFrame:
        """简化模式：所有醛 × 所有胺自由配对，不区分含氟。"""
        aldehydes = monomers[monomers["is_aldehyde"]]
        amines = monomers[monomers["is_amine"]]
        pairs = []
        for _, ald in aldehydes.iterrows():
            for _, am in amines.iterrows():
                pairs.append({
                    "aldehyde": ald["name"],
                    "amine": am["name"],
                    "aldehyde_f": ald["has_fluorine"],
                    "amine_f": am["has_fluorine"],
                    "pair_type": "all-pairs (free)",
                })
        df = pd.DataFrame(pairs)
        logger.info(
            f"全量自由配对: {len(aldehydes)} 醛 × {len(amines)} 胺 = {len(df)} 组合"
        )
        return df

    def _pre_fluorinate_nonf_pairs(self, pairs_df: pd.DataFrame,
                                   ) -> pd.DataFrame:
        """对非F×非F配对预先虚拟氟化再预测，使其评分基准与含氟配对一致。

        对非F醛×非F胺的每对组合：
          尝试醛+1F / 胺+1F / 双+1F → 取最佳氟化分数替换原始 film_probability
        """
        self._load_model()
        from rdkit import Chem

        nonf_mask = (pairs_df["pair_type"] == "nonF-aldehyde × nonF-amine")
        nonf_count = nonf_mask.sum()
        if nonf_count == 0:
            return pairs_df

        logger.info(f"预处理非F×非F配对虚拟氟化: {nonf_count} 对")

        new_probs = []
        for idx, row in pairs_df.iterrows():
            if not row["pair_type"] == "nonF-aldehyde × nonF-amine":
                new_probs.append(row.get("film_probability", np.nan))
                continue

            ald_name, am_name = row["aldehyde"], row["amine"]
            ald_smi = self.monomer_lib.resolve(ald_name)
            am_smi = self.monomer_lib.resolve(am_name)
            ald_mol = self.monomer_lib.get_mol(ald_name)
            am_mol = self.monomer_lib.get_mol(am_name)

            if ald_mol is None or am_mol is None:
                new_probs.append(row.get("film_probability", np.nan))
                continue

            best_score = row.get("film_probability", 0.0)
            if np.isnan(best_score):
                best_score = 0.0

            def _pred(a_mol, b_mol):
                try:
                    feat = self.feature_eng.featurize_monomer_pair(a_mol, b_mol)
                    feat = feat.reshape(1, -1)
                    if (hasattr(self, "_selected_features")
                            and self._selected_features is not None
                            and len(self._selected_features) > 0):
                        feat = feat[:, self._selected_features]
                    if hasattr(self, "_scaler") and self._scaler is not None:
                        feat = self._scaler.transform(feat)
                    return self._model.predict_proba(feat)[0, 1]
                except Exception:
                    return None

            # 策略1: 仅氟化醛
            if ald_smi:
                f_smi = virtual_fluorination(ald_smi, n_f=1)
                if f_smi:
                    f_mol = Chem.MolFromSmiles(f_smi)
                    if f_mol:
                        score = _pred(f_mol, am_mol)
                        if score is not None and score > best_score:
                            best_score = score

            # 策略2: 仅氟化胺
            if am_smi:
                f_smi = virtual_fluorination(am_smi, n_f=1)
                if f_smi:
                    f_mol = Chem.MolFromSmiles(f_smi)
                    if f_mol:
                        score = _pred(ald_mol, f_mol)
                        if score is not None and score > best_score:
                            best_score = score

            # 策略3: 双氟化
            if ald_smi and am_smi:
                f_ald_smi = virtual_fluorination(ald_smi, n_f=1)
                f_am_smi = virtual_fluorination(am_smi, n_f=1)
                if f_ald_smi and f_am_smi:
                    f_ald = Chem.MolFromSmiles(f_ald_smi)
                    f_am = Chem.MolFromSmiles(f_am_smi)
                    if f_ald and f_am:
                        score = _pred(f_ald, f_am)
                        if score is not None and score > best_score:
                            best_score = score

            new_probs.append(best_score)

        pairs_df = pairs_df.copy()
        pairs_df["film_probability"] = new_probs
        logger.info(
            f"非F×非F虚拟氟化预处理完成: "
            f"{nonf_count} 对 → 最佳氟化分数替换"
        )
        return pairs_df

    # -------------------------------------------------------------------
    # 虚拟氟化修正
    # -------------------------------------------------------------------

    def virtual_fluorination_correction(
        self, top_pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        """对 Top N 组合进行虚拟氟化修正。

        策略：
        - 双F组合：无需修正，直接使用原始分数
        - 单F组合：对非F单体虚拟加1F → 重新预测
        - 无F组合：分别尝试加F到醛、胺、两者同时 → 取最佳提升
        """
        self._load_model()
        from rdkit import Chem

        fluorinated_scores = []
        score_improvements = []

        for _, row in top_pairs.iterrows():
            ald_name, am_name = row["aldehyde"], row["amine"]
            ald_smi = self.monomer_lib.resolve(ald_name)
            am_smi = self.monomer_lib.resolve(am_name)

            ald_mol = self.monomer_lib.get_mol(ald_name)
            am_mol = self.monomer_lib.get_mol(am_name)
            ald_f = self.f_detector.has_fluorine(ald_mol) if ald_mol else False
            am_f = self.f_detector.has_fluorine(am_mol) if am_mol else False

            base_score = row.get("film_probability", 0.0)
            if np.isnan(base_score):
                base_score = 0.0

            # 双F：无需修正
            if ald_f and am_f:
                fluorinated_scores.append(base_score)
                score_improvements.append(0.0)
                continue

            best_score = base_score
            best_improv = 0.0

            def _score_pair(a_mol, b_mol):
                """预测 (aldehyde_mol, amine_mol) 的成膜概率。"""
                try:
                    feat = self.feature_eng.featurize_monomer_pair(a_mol, b_mol)
                    feat = feat.reshape(1, -1)
                    if (hasattr(self, "_selected_features")
                            and self._selected_features is not None
                            and len(self._selected_features) > 0):
                        feat = feat[:, self._selected_features]
                    if hasattr(self, "_scaler") and self._scaler is not None:
                        feat = self._scaler.transform(feat)
                    return self._model.predict_proba(feat)[0, 1]
                except Exception:
                    return None

            def _try_fluorinate(smi, n=1):
                """虚拟氟化 SMILES → Mol，失败返回 None。"""
                f_smi = virtual_fluorination(smi, n_f=n)
                if not f_smi:
                    return None
                mol = Chem.MolFromSmiles(f_smi)
                return mol if mol else None

            # 策略 1: 仅氟化醛（醛不含F时）
            if not ald_f and ald_smi:
                f_ald_mol = _try_fluorinate(ald_smi, n=1)
                if f_ald_mol and am_mol:
                    score = _score_pair(f_ald_mol, am_mol)
                    if score is not None and score - base_score > best_improv:
                        best_improv = score - base_score
                        best_score = score

            # 策略 2: 仅氟化胺（胺不含F时）
            if not am_f and am_smi:
                f_am_mol = _try_fluorinate(am_smi, n=1)
                if f_am_mol and ald_mol:
                    score = _score_pair(ald_mol, f_am_mol)
                    if score is not None and score - base_score > best_improv:
                        best_improv = score - base_score
                        best_score = score

            # 策略 3: 双氟化（两者都不含F时，同时加氟）
            if not ald_f and not am_f and ald_smi and am_smi:
                f_ald_mol = _try_fluorinate(ald_smi, n=1)
                f_am_mol = _try_fluorinate(am_smi, n=1)
                if f_ald_mol and f_am_mol:
                    score = _score_pair(f_ald_mol, f_am_mol)
                    if score is not None and score - base_score > best_improv:
                        best_improv = score - base_score
                        best_score = score

            fluorinated_scores.append(best_score)
            score_improvements.append(best_improv)

        top_pairs["fluorinated_score"] = fluorinated_scores
        top_pairs["fluorination_gain"] = score_improvements
        return top_pairs
