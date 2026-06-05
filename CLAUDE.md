# FluoroFilm — GNN 成膜预测 (AI 开发指南)

## 项目概述

从 ~840 篇 COF 文献 PDF 中用 LLM 提取亚胺键单体配对数据，训练 GNN 模型预测单体对能否形成连续薄膜。

**核心任务**：输入醛+胺 SMILES → 输出成膜概率 (0~1) + MC Dropout 不确定性。

**当前版本：v5** | PR-AUC 0.784 ± 0.06 | 6,201 条训练数据 | 四分类连续标签

---

## 环境搭建

```bash
# 1. 创建 conda 环境 (Python 3.12)
conda create -n dphuanjing python=3.12 -y
conda activate dphuanjing

# 2. 安装 PyTorch (CUDA 版本按需选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 PyTorch Geometric (注意版本匹配)
pip install torch_geometric

# 4. 安装其余依赖
pip install rdkit scikit-learn numpy pandas pyyaml python-docx

# 5. 验证
python -c "from src.screening.gnn_v4.model import V4Model; print('OK')"
```

**关键版本约束**：
- Python **3.12.7**（不要用 3.13，PyTorch Geometric 不兼容）
- RDKit **2026.03.1**（SMILES 解析行为对版本敏感）
- CUDA 可选，CPU 也能跑但推理慢

---

## 目录结构（带依赖关系）

```
├── predict_pair.py               # [入口] 单对预测 CLI
├── _build_cartesian.py           # [筛选 Step1] 笛卡尔积配对
├── _gnn_inference.py             # [筛选 Step2] GNN 推理 (依赖 screen_v4.py)
├── _filter_diverse.py            # [筛选 Step3] 多样性过滤
├── _generate_top40_report.py     # [筛选 Step4] Word 报告
│
├── src/
│   ├── screening/gnn_v4/         # ★ 当前模型 (v5 训练复用)
│   │   ├── model.py              # V4Model — 顶层组装，对外 API
│   │   ├── encoder.py            # GIN+GINE ×3 + JK-Net → 128维
│   │   ├── attention.py          # Bidirectional CrossGraphAttention (4头)
│   │   ├── pooling.py            # BatchedAttentionPool + PairPooling
│   │   ├── heads.py              # FilmHead — 拼接 [ea,eb,ea⊙eb,e_pair,rule_vec] → MLP → logit
│   │   ├── v4_loss.py            # FocalLoss (α=0.75, γ=2.0, 支持连续标签)
│   │   └── v4_trainer.py         # 训练循环 + 早停 + PR-AUC 监控
│   │
│   ├── screening/gnn_v3/         # v3 遗留 (仅一个文件被复用)
│   │   └── featurizer.py         # ★ smiles_to_graph() — 所有脚本都依赖它
│   │
│   ├── chemistry/
│   │   ├── hard_rules.py         # get_rule_vector() → 23维 0/1 向量
│   │   ├── linker_analyzer.py    # has_heterocycle / count_aromatic_rings / is_functionally_symmetric
│   │   ├── conformer.py          # 单体 3D 描述符 (10维, 可选, 消融表明无显著提升)
│   │   ├── dimer.py              # 二聚体 3D 描述符 (10维, 可选)
│   │   └── negative_sampler.py   # 化学规则负样本生成 (训练数据构建用)
│   │
│   └── utils/
│       └── logger.py             # setup_logger() — 统一日志格式
│
├── scripts/
│   ├── train_v4.py               # [入口] 训练主脚本 (15-fold CV + batch)
│   ├── build_v4_dataset.py       # 数据构建 (v3_db_full → 清洗 → 负样本 → v5_train_stage1)
│   ├── augment_v4_positives.py   # 正样本增广 (生成 0.7 标签数据)
│   ├── screen_v4.py              # ★ 批量筛选核心库 (被 _gnn_inference.py 和 _build_cartesian.py 导入)
│   ├── generate_v4_report.py     # Word 报告生成 (旧版)
│   ├── pretrain_v4.py            # GNN 预训练 (Masked Atom + 角色分类)
│   ├── diagnose_3d_signal.py     # 3D 信号 Cohen's d 诊断 (消融实验)
│   └── fix_monomer_pool.py       # 单体池修复工具 (一次性)
│
├── config/
│   └── model_v4.yaml             # 模型/训练超参数
│
├── data/
│   ├── processed/
│   │   ├── v5_train_stage1.csv   # ★ 训练集 (6,201 行)
│   │   ├── merged_monomer_pool.csv # 筛选池单体 (醛+胺 SMILES 库)
│   │   └── v4_cartesian_pairs.csv  # 笛卡尔积配对 (~232K)
│   ├── structured/               # LLM 提取的结构化 YAML (~955 篇文献)
│   ├── pdfs/                     # 原始 PDF (~840 篇)
│   └── *.db                      # SQLite 数据库 (提取中间结果)
│
├── models/
│   └── v5.0/                     # 模型权重 (v5_model.pt)
│
├── jiyi/                         # ★ 工作日志 — 按日期记录决策和结果
├── notebooks/                    # Jupyter 探索笔记
├── tests/                        # 测试
├── CLAUDE.md                     # 本文件 (给 AI 看)
└── README.md                     # 项目说明 (给人看)
```

### 关键依赖链

```
predict_pair.py
  └─→ src/screening/gnn_v4/model.py (V4Model)
  └─→ src/screening/gnn_v3/featurizer.py (smiles_to_graph)
  └─→ src/chemistry/hard_rules.py (get_rule_vector)

_gnn_inference.py
  └─→ scripts/screen_v4.py (_canon_smiles, _run_attention_pooling, _pair_embedding)
  └─→ src/screening/gnn_v3/featurizer.py
  └─→ src/chemistry/hard_rules.py

train_v4.py
  └─→ src/screening/gnn_v4/{model, v4_loss, v4_trainer}
  └─→ src/screening/gnn_v3/featurizer.py
  └─→ src/chemistry/{conformer, dimer, hard_rules}
  └─→ src/utils/logger.py
```

**注意**：`_gnn_inference.py` 和 `_build_cartesian.py` 从 `scripts.screen_v4` 导入辅助函数，这是有意为之——`screen_v4.py` 既是独立脚本也是库。

---

## 数据

### 训练数据 `v5_train_stage1.csv`

| 列名 | 类型 | 说明 |
|------|------|------|
| `paper_id` | str | 文献 ID（CV 拆分依据） |
| `group_id` | int | 文献内分组 |
| `source_db` | str | 数据来源 (v3_db_full / augmented_v4 / hard_rule_sampled / chem_rule_*) |
| `aldehyde_smiles` | str | 醛单体 SMILES |
| `amine_smiles` | str | 胺单体 SMILES |
| `aldehyde_name` | str | 醛单体名称 |
| `amine_name` | str | 胺单体名称 |
| `stoichiometry` | str | 投料比 |
| `solvent` | str | 溶剂 |
| `temperature` | str | 温度 |
| `catalyst` | str | 催化剂 |
| `synthesis_route` | str | 合成路线 |
| `interface_type` | str | 界面类型 |
| **`is_film`** | **float** | **★ 标签列 — 0.0 / 0.7 / 0.8 / 1.0** |
| `film_quality` | str | 膜质量描述 |
| `original_is_film` | str | 原始文献中的 is_film 值 |

**标签含义**：
- `1.0` — 文献确认连续成膜 (429 条)
- `0.8` — 粉末合成，成功但非连续膜 (291 条)
- `0.7` — 增广正样本，衍生自文献正样本 (1,308 条)
- `0.0` — 负样本：化学规则生成 + 文献内负样本 (4,173 条)

**CV 拆分**：按 `paper_id` 分层，严禁随机打散（同文献的配对必须在同一 fold）。

---

## 模型

### 推理流程

```
醛 SMILES ──→ smiles_to_graph(role=0) ──→ GIN Encoder ──→ ald_emb [N_ald, 128]
胺 SMILES ──→ smiles_to_graph(role=1) ──→ GIN Encoder ──→ amine_emb [N_am, 128]
                                                              │
                                              ┌───────────────┘
                                              ↓
                                   CrossGraphAttention (双向 4头)
                                              ↓
                                   PairPooling → e_pair [128]
                                              │
                              ┌───────────────┼───────────────┐
                              ↓                               ↓
                 [ea, eb, ea⊙eb, e_pair]              rule_vec [23]
                              │                               │
                              └───────────┬───────────────────┘
                                          ↓
                              FilmHead MLP [768→512→256→128→1]
                                          ↓
                                      sigmoid → prob [0~1]
```

### 关键 API

```python
from src.screening.gnn_v4.model import V4Model
from src.screening.gnn_v3.featurizer import smiles_to_graph
from src.chemistry.hard_rules import get_rule_vector

# 加载模型
ckpt = torch.load("models/v5.0/v5_model.pt", map_location=device, weights_only=False)
model = V4Model(ckpt["config"])
model.load_state_dict(ckpt["model_state"])
model.eval()

# 构建图 (role: 0=醛, 1=胺)
ald_g = smiles_to_graph("O=CC1=CC=C(C=O)C=C1", role=0)
amine_g = smiles_to_graph("NC1=CC=C(N)C=C1", role=1)

# 规则向量
rv = get_rule_vector(ald_smi, amine_smi)  # → list[float] 长度 23

# 单次推理
logit = model.predict_single(ald_g, amine_g, rule_vec=rv_tensor)
prob = torch.sigmoid(logit)

# MC Dropout (多次采样求不确定性)
model.enable_mc_dropout()
probs = [torch.sigmoid(model.predict_single(...)) for _ in range(10)]
mean, std = np.mean(probs), np.std(probs)
```

### Checkpoint 格式

```python
{
    "model_state": OrderedDict,     # 模型权重 (best PR-AUC)
    "config": dict,                 # 模型配置 (encoder/attention/pooling/heads 超参)
    "fold_pr_aucs": list[float],    # 各 fold 的 PR-AUC
    "best_fold": int,               # 最佳 fold 编号
    "scaler_3d": {"mean": [...], "std": [...]},   # 单体 3D 归一化参数 (可选)
    "scaler_dimer": {"mean": [...], "std": [...]}, # 二聚体 3D 归一化参数 (可选)
    "use_3d": bool,                 # 是否使用 3D 描述符
    "use_rules": bool,              # 是否使用规则向量
}
```

**加载注意事项**：
- 必须传 `weights_only=False`（checkpoint 包含 dict 非张量数据）
- 如果 `use_3d=True` 但当前模型不需要 3D，需设置 `cfg["model"]["use_3d"] = False` 重建模型

---

## 常用命令

```bash
conda activate dphuanjing

# === 核心：单对预测 ===
python predict_pair.py --ald "O=CC1=CC=C(C=O)C=C1" --amine "NC1=CC=C(N)C=C1"
python predict_pair.py --ald "SMILES" --amine "SMILES" --mc 20  # 更多 MC 采样

# === 训练 ===
python scripts/train_v4.py --config config/model_v4.yaml --output models/v5.0

# === 批量筛选管线 (从单体池大规模搜索，4 步) ===
python _build_cartesian.py                              # Step 1: 笛卡尔积
python _gnn_inference.py --model models/v5.0/v5_model.pt  # Step 2: GNN 推理
python _filter_diverse.py                               # Step 3: 多样性过滤
python _generate_top40_report.py                        # Step 4: 报告
```

---

## 工作日志 (`jiyi/`)

按日期组织的开发决策记录，AI 接手项目时应**优先查阅最近的日志**了解上下文：

| 文件 | 内容 |
|------|------|
| `2026-06-05.md` | ★ v5 数据修复 + 最终训练结果 |
| `2026-06-02.md` | v5 训练调试 |
| `2026-06-01.md` | 规则向量注入方案 |
| `2026-05-30.md` | 负样本生成策略 |
| `2026-05-28.md` | 数据清洗讨论 |
| `阶段*.md` | 各阶段计划文档 |
| `chemical-priors-regularization.md` | 化学先验正则化方案 |

---

## 常见坑

1. **`smiles_to_graph` 的 `role` 参数不能省** — `role=0` 标记醛基碳，`role=1` 标记胺基氮，影响位置编码
2. **RDKit 版本敏感** — `sanitize=False` 模式下的芳香性推断依赖 RDKit 版本，换版本后特征可能不同
3. **CKPT 加载** — 务必 `weights_only=False`，否则 `TypeError: 'dict' object is not callable`
4. **CUDA OOM** — 批量推理 ~232K 配对时，先编码所有独有单体再逐对推理，不要每对重新编码
5. **CV 拆分** — 必须按 `paper_id` 分层，同文献的配对分散到不同 fold 会导致标签泄漏
6. **`is_film` 是连续值** — 不是 0/1 二分类，用 BCE 而非 CrossEntropy，评估用 PR-AUC 而非 Accuracy
7. **`_gnn_inference.py` 从 `scripts.screen_v4` 导入** — 确保 `scripts/` 在 Python path 中（脚本自动处理）
8. **模型目录 `models/v5.0/` 当前为空** — 需要先训练生成 `v5_model.pt` 才能推理

---

## 开发规范

- Python 3.12，函数 type hints
- 模块 ≤ 800 行，函数 ≤ 50 行
- 不可变优先，避免 in-place 修改
- 功能块前中文注释说明意图
- 分支: `master` (稳定) / `develop` (主线) / `feature/*`
- 提交: Conventional Commits (feat/fix/refactor/docs/chore)
