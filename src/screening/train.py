"""模型训练模块 — 标准 ML 流程。

流程: 数据分割 → 特征预处理 → 超参调优 → 训练 → 交叉验证 → 评估 → 持久化
支持: XGBoost (主模型)、Random Forest (对照)、Logistic Regression (基线)
"""
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate,
    learning_curve, train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.logger import setup_logger

logger = setup_logger("trainer")

RANDOM_STATE = 42
N_FOLDS = 5


class ModelTrainer:
    """模型训练器 — 训练 + 评估 + 持久化。"""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.selected_features: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.cv_results: Dict[str, Dict] = {}
        self.test_results: Dict[str, Dict] = {}

    # -------------------------------------------------------------------
    # 数据准备
    # -------------------------------------------------------------------

    def prepare_data(
        self, X: np.ndarray, y: np.ndarray,
        test_size: float = 0.20, val_size: float = 0.00,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """分层分割 train/val/test。

        默认仅 train/test (80/20)。若 val_size > 0 则 train/test/val。
        """
        # 先分出 test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y,
            random_state=self.random_state,
        )
        if val_size <= 0:
            logger.info(
                f"数据分割: train={X_temp.shape[0]}, test={X_test.shape[0]}"
            )
            return X_temp, X_test, y_temp, y_test

        # 再从 train 中分出 val
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp,
            random_state=self.random_state,
        )
        logger.info(
            f"数据分割: train={X_train.shape[0]}, "
            f"val={X_val.shape[0]}, test={X_test.shape[0]}"
        )
        return X_train, X_val, y_train, y_val, X_test, y_test

    # -------------------------------------------------------------------
    # 特征预处理
    # -------------------------------------------------------------------

    def preprocess_features(
        self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None,
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """特征预处理：去低方差 → 去高相关 → 标准化。

        仅对训练集 fit，测试集 transform（防止数据泄露）。
        """
        n_before = X_train.shape[1]

        # 1. 低方差特征过滤
        if variance_threshold > 0:
            vt = VarianceThreshold(threshold=variance_threshold)
            X_train = vt.fit_transform(X_train)
            if X_test is not None:
                X_test = vt.transform(X_test)
            logger.info(f"低方差过滤: {n_before} → {X_train.shape[1]} 维")

        # 2. 高相关特征过滤
        if correlation_threshold > 0 and X_train.shape[1] > 2:
            corr = np.corrcoef(X_train, rowvar=False)
            upper = np.abs(np.triu(corr, k=1))
            to_drop = set()
            for i in range(upper.shape[0]):
                for j in range(i + 1, upper.shape[1]):
                    if upper[i, j] > correlation_threshold:
                        to_drop.add(j)
            if to_drop:
                keep = [i for i in range(X_train.shape[1]) if i not in to_drop]
                X_train = X_train[:, keep]
                if X_test is not None:
                    X_test = X_test[:, keep]
                self.selected_features = np.array(keep)
                logger.info(f"高相关过滤: 移除 {len(to_drop)} 个特征, "
                            f"保留 {len(keep)} 维")

        # 3. 标准化
        X_train = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = self.scaler.transform(X_test)

        return X_train, X_test

    # -------------------------------------------------------------------
    # 超参调优
    # -------------------------------------------------------------------

    def tune_xgboost(
        self, X_train: np.ndarray, y_train: np.ndarray,
        n_iter: int = 40, cv: int = 3, n_jobs: int = 4,
    ) -> XGBClassifier:
        """XGBoost 随机搜索调参。

        搜索空间: max_depth, learning_rate, n_estimators, subsample,
                  colsample_bytree, reg_alpha, reg_lambda
        """
        from sklearn.model_selection import RandomizedSearchCV

        scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        param_dist = {
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "n_estimators": randint(100, 400),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": uniform(0, 1.0),
            "reg_lambda": uniform(0.5, 2.0),
        }
        base = XGBClassifier(
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=self.random_state,
            verbosity=0,
        )
        search = RandomizedSearchCV(
            base, param_dist, n_iter=n_iter, cv=cv,
            scoring="average_precision", random_state=self.random_state,
            n_jobs=n_jobs, verbose=0,
        )
        search.fit(X_train, y_train)
        logger.info(
            f"XGBoost 最佳参数 (CV AP={search.best_score_:.4f}): "
            f"{search.best_params_}"
        )
        return search.best_estimator_

    def tune_random_forest(
        self, X_train: np.ndarray, y_train: np.ndarray,
        n_iter: int = 30, cv: int = 3, n_jobs: int = 4,
    ) -> RandomForestClassifier:
        """RF 随机搜索调参。"""
        from sklearn.model_selection import RandomizedSearchCV

        param_dist = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(5, 30),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        }
        base = RandomForestClassifier(
            class_weight="balanced",
            random_state=self.random_state,
        )
        search = RandomizedSearchCV(
            base, param_dist, n_iter=n_iter, cv=cv,
            scoring="average_precision", random_state=self.random_state,
            n_jobs=n_jobs, verbose=0,
        )
        search.fit(X_train, y_train)
        logger.info(f"RF 最佳参数 (CV AP={search.best_score_:.4f}): {search.best_params_}")
        return search.best_estimator_

    # -------------------------------------------------------------------
    # 训练
    # -------------------------------------------------------------------

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      tune: bool = True) -> XGBClassifier:
        """训练 XGBoost 分类器。"""
        if tune and X_train.shape[0] >= 30:
            model = self.tune_xgboost(X_train, y_train)
        else:
            scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
            model = XGBClassifier(
                max_depth=6, learning_rate=0.1, n_estimators=200,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=self.random_state,
                verbosity=0,
            )
            model.fit(X_train, y_train)
        self.models["xgboost"] = model
        return model

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                            tune: bool = True) -> RandomForestClassifier:
        """训练 Random Forest 分类器。"""
        if tune and X_train.shape[0] >= 30:
            model = self.tune_random_forest(X_train, y_train)
        else:
            model = RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=self.random_state,
            )
            model.fit(X_train, y_train)
        self.models["random_forest"] = model
        return model

    def train_logistic(self, X_train: np.ndarray, y_train: np.ndarray,
                       ) -> LogisticRegression:
        """训练 Logistic Regression 基线。"""
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)
        self.models["logistic"] = model
        return model

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  tune: bool = True) -> Dict:
        """训练全部模型。"""
        logger.info("训练 XGBoost...")
        self.train_xgboost(X_train, y_train, tune=tune)
        logger.info("训练 Random Forest...")
        self.train_random_forest(X_train, y_train, tune=tune)
        logger.info("训练 Logistic Regression...")
        self.train_logistic(X_train, y_train)
        return self.models

    # -------------------------------------------------------------------
    # 交叉验证
    # -------------------------------------------------------------------

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       model_name: str = "xgboost",
                       n_folds: int = N_FOLDS,
                       ) -> Dict[str, float]:
        """分层 K 折交叉验证。"""
        model = self.models.get(model_name)
        if model is None:
            logger.error(f"模型 {model_name} 未训练")
            return {}

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=self.random_state)
        scoring = {
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        }
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)

        result = {}
        for metric, values in scores.items():
            if metric.startswith("test_"):
                key = metric[5:]
                result[key] = np.mean(values)
                result[f"{key}_std"] = np.std(values)

        self.cv_results[model_name] = result
        return result

    def cross_validate_all(self, X: np.ndarray, y: np.ndarray,
                           ) -> pd.DataFrame:
        """所有模型的交叉验证对比。"""
        rows = []
        for name in self.models:
            cv = self.cross_validate(X, y, model_name=name)
            if cv:
                rows.append({
                    "model": name,
                    "precision": cv.get("precision", 0),
                    "recall": cv.get("recall", 0),
                    "f1": cv.get("f1", 0),
                    "roc_auc": cv.get("roc_auc", 0),
                    "pr_auc": cv.get("average_precision", 0),
                })
        df = pd.DataFrame(rows)
        self._cv_df = df
        return df

    # -------------------------------------------------------------------
    # 评估
    # -------------------------------------------------------------------

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 model_name: str = "xgboost") -> Dict[str, Any]:
        """模型测试集评估。"""
        model = self.models.get(model_name)
        if model is None:
            return {}

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        result = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        self.test_results[model_name] = result
        return result

    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray,
                     ) -> pd.DataFrame:
        """所有模型测试集评估对比。"""
        rows = []
        for name in self.models:
            r = self.evaluate(X_test, y_test, model_name=name)
            if r:
                rows.append({
                    "model": name,
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "roc_auc": r["roc_auc"],
                    "pr_auc": r["pr_auc"],
                })
        df = pd.DataFrame(rows)
        self._test_df = df
        return df

    # -------------------------------------------------------------------
    # 特征重要性
    # -------------------------------------------------------------------

    def get_feature_importance(self, model_name: str = "xgboost",
                               top_n: int = 30) -> pd.DataFrame:
        """返回特征重要性排序 DataFrame。"""
        model = self.models.get(model_name)
        if model is None:
            return pd.DataFrame()

        if model_name == "xgboost":
            importances = model.feature_importances_
        elif model_name == "random_forest":
            importances = model.feature_importances_
        elif model_name == "logistic":
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()

        # 生成名称
        if self.selected_features is not None:
            importances = importances

        names = (self.feature_names if len(self.feature_names) == len(importances)
                 else [f"f{i}" for i in range(len(importances))])

        df = pd.DataFrame({"feature": names, "importance": importances})
        df = df.sort_values("importance", ascending=False).head(top_n)
        return df.reset_index(drop=True)

    # -------------------------------------------------------------------
    # 学习曲线
    # -------------------------------------------------------------------

    def plot_learning_curve(self, X: np.ndarray, y: np.ndarray,
                            model_name: str = "xgboost",
                            output_path: Optional[str] = None,
                            ) -> Tuple[plt.Figure, Dict]:
        """绘制学习曲线，诊断过拟合/欠拟合。"""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"模型 {model_name} 未训练")

        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=train_sizes,
            cv=StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=self.random_state),
            scoring="average_precision", n_jobs=1,
            random_state=self.random_state,
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(train_sizes_abs, train_mean - train_std,
                        train_mean + train_std, alpha=0.15, color="blue")
        ax.fill_between(train_sizes_abs, test_mean - test_std,
                        test_mean + test_std, alpha=0.15, color="orange")
        ax.plot(train_sizes_abs, train_mean, "o-", color="blue",
                label="Training PR-AUC")
        ax.plot(train_sizes_abs, test_mean, "o-", color="orange",
                label="Validation PR-AUC")
        ax.set_xlabel("Training Size")
        ax.set_ylabel("PR-AUC")
        ax.set_title(f"Learning Curve — {model_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"学习曲线保存至: {output_path}")

        return fig, {
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores_mean": train_mean.tolist(),
            "test_scores_mean": test_mean.tolist(),
        }

    def plot_feature_importance(self, model_name: str = "xgboost", top_n: int = 30,
                                output_path: Optional[str] = None,
                                ) -> Tuple[plt.Figure, pd.DataFrame]:
        """绘制特征重要性柱状图。"""
        df = self.get_feature_importance(model_name, top_n=top_n)
        if df.empty:
            raise ValueError("无法获取特征重要性")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(df)), df["importance"], color="steelblue")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["feature"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importance — {model_name}")

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"特征重要性图保存至: {output_path}")

        return fig, df

    # -------------------------------------------------------------------
    # 持久化
    # -------------------------------------------------------------------

    def save_models(self, output_dir: str):
        """保存全部模型、scaler、feature_names 到目录。"""
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(output_dir, f"{name}_model.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)

        with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        info = {
            "feature_names": self.feature_names,
            "selected_features": (self.selected_features.tolist()
                                  if self.selected_features is not None
                                  and len(self.selected_features) > 0
                                  else []),
            "cv_results": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                for kk, vv in v.items()}
                           for k, v in self.cv_results.items()},
            "test_results": self.test_results,
        }
        with open(os.path.join(output_dir, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"模型已保存至: {output_dir}/")

    def load_models(self, model_dir: str):
        """从目录加载已训练模型。"""
        for name in ["xgboost", "random_forest", "logistic"]:
            path = os.path.join(model_dir, f"{name}_model.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.models[name] = pickle.load(f)

        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        info_path = os.path.join(model_dir, "model_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
                self.feature_names = info.get("feature_names", [])
                self.cv_results = info.get("cv_results", {})
                self.test_results = info.get("test_results", {})

        logger.info(f"模型已加载: {list(self.models.keys())}")
