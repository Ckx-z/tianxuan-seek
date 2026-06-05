# FluoroFilm — GNN 成膜预测筛选

## 项目概述

从 ~840 篇 PDF 文献中提取 COF 亚胺键单体配对数据，用 GNN 深度学习模型预测单体对能否成膜。
输入醛+胺 SMILES，输出成膜概率 (0~1)，附带 MC Dropout 不确定性。

**当前版本：v5** — 四分类连续标签 FocalLoss + 规则向量注入 + 化学规则负样本。

## 技术栈

| 层级 | 选型 | 版本 |
|------|------|------|
| 语言 | Python | **3.12.7** |
| 包管理 | conda (环境: `dphuanjing`) | — |
| 化学信息 | RDKit | **2026.03.1** |
| GNN | PyTorch + PyTorch Geometric | — |
| 报告生成 | python-docx | — |
| LLM 提取 | MiniMax API / MiMo Omni | — |

## 目录结构

```
FluoroFilm/
├── data/
│   ├── structured/              # LLM 提取的结构化 YAML
│   ├── processed/               # 训练数据、筛选结果、报告
│   │   ├── v5_train_stage1.csv  # 训练集 (6201 行, 4 标签)
│   │   ├── merged_monomer_pool.csv # 筛选池单体
│   │   └── v4_cartesian_pairs.csv  # 笛卡尔积配对 (~232K)
│   ├── fluorofilm_v3.db         # v3 数据库 (YAML → SQLite)
│   └── pdfs/                    # 原始 PDF
├── src/
│   ├── screening/gnn_v4/        # GNN 模型 (v5 共用 v4 架构)
│   │   ├── model.py             # V4Model — 端到端成膜预测
│   │   ├── encoder.py           # GIN+GINE x3 + JK-Net (128维)
│   │   ├── attention.py         # Bidirectional CrossGraphAttention (4头)
│   │   ├── pooling.py           # BatchedAttentionPool + PairPooling
│   │   ├── heads.py             # FilmHead + 23维规则向量注入
│   │   ├── v4_loss.py           # FocalLoss (α=0.75, γ=2.0, 支持连续标签)
│   │   └── v4_trainer.py        # 训练循环 + 早停
│   ├── screening/gnn_v3/        # v3 遗留 (仅 featurizer.py 被复用)
│   │   └── featurizer.py        # smiles_to_graph()
│   ├── chemistry/
│   │   ├── hard_rules.py        # 23维规则向量 (硬约束编码)
│   │   ├── conformer.py         # 单体 3D 描述符 (10维, 可选)
│   │   ├── dimer.py             # 二聚体 3D 描述符 (10维, 可选)
│   │   └── negative_sampler.py  # 化学规则负样本生成器
│   ├── extraction/              # LLM 提取
│   └── utils/                   # db.py, logger.py
├── scripts/
│   ├── train_v4.py              # 训练主入口 (15-fold CV + batch)
│   ├── build_v4_dataset.py      # 数据构建 (v4 → v5)
│   ├── augment_v4_positives.py  # 正样本增广 (0.7 标签)
│   ├── screen_v4.py             # 筛选主入口 (MC Dropout + 多样性)
│   └── generate_v4_report.py    # Word 报告生成
├── _gnn_inference.py            # Step 2: GNN 推理
├── _build_cartesian.py          # Step 1: 笛卡尔积配对构建
├── _filter_diverse.py           # Step 3: 多样性过滤 + Top 40
├── _generate_top40_report.py    # Step 4: Top 40 报告
├── config/
│   ├── model_v4.yaml            # 模型/训练超参数
│   └── model_v4_augmented.yaml  # 增广版配置
├── models/                      # 按版本管理 (权重不入库)
├── versions/                    # 历史版本归档
└── jiyi/                        # 工作日志
```

## 数据管线

### 训练数据 (`data/processed/v5_train_stage1.csv`)

| 来源 | 标签 | 数量 | 说明 |
|------|------|------|------|
| v3_db_full | 1.0 | 429 | 文献确认成膜 |
| v3_db_full | 0.8 | 291 | 粉末合成 (成功但非连续膜) |
| augmented_v4 | 0.7 | 1308 | 增广正样本 (文献正样本衍生) |
| hard_rule_sampled | 0.0 | 3104 | 硬规则确定性负样本 |
| chem_rule_* | 0.0 | 941 | 化学规则边界负样本 |
| v3_db_full (混合文献) | 0.0 | 128 | 文献内负样本 (同 paper 有成膜) |
| **合计** | | **6201** | 独立 paper_id: 5710 |

### 筛选管线 (3 步)

```
Step 1: _build_cartesian.py
  单体池 (醛×胺) → 笛卡尔积配对 (~232K) → 硬约束过滤 → v4_cartesian_pairs.csv

Step 2: _gnn_inference.py
  配对 → GNN 图编码 → Cross-Attention → FilmHead(+规则向量) → MC Dropout → 概率均值±std

Step 3: _filter_diverse.py
  GNN 概率 → 化学先验 → 嵌入多样性 MaxMin → Top 40 → Word 报告
```

## 模型架构

V4Model (0.70M 参数)，v5 训练复用：

```
醛 SMILES ──→ GIN Encoder (共享权重) ──→ ald_emb (128维)
胺 SMILES ──→ GIN Encoder (共享权重) ──→ amine_emb (128维)
                                              │
                              ┌───────────────┘
                              ↓
                   CrossGraphAttention (双向, 4头)
                              ↓
                   PairPooling → e_pair (128维)
                              │
              ┌───────────────┼───────────────┐
              ↓                               ↓
        ald_emb, amine_emb, e_pair      rule_vec (23维)
              │                               │
              └───────────┬───────────────────┘
                          ↓
              FilmHead [768→512→256→128] → logit → sigmoid → prob
```

- **规则向量 (23维)**：硬约束规则命中向量，注入 FilmHead，让模型学习硬规则模式
- **3D 描述符 (可选)**：单体 10维 + 二聚体 10维，消融实验表明无显著提升
- **FocalLoss 连续标签**：p_t = 1 - |target - prob|，支持 0.0/0.7/0.8/1.0 四级标签

## 训练结果

| 版本 | PR-AUC | 数据量 | 标签 |
|------|--------|--------|------|
| v4 (2026-05) | 0.76 ± 0.07 | 2093 | 二分类 |
| **v5 (2026-06)** | **0.784 ± 0.06** | 6201 | 四分类连续标签 |

v5 7-fold CV: 0.827, 0.797, 0.713, 0.833, 0.791, 0.856, 0.670

## 常用命令

```bash
conda activate dphuanjing

# 训练
python scripts/train_v4.py --config config/model_v4.yaml --output models/v5.0

# 筛选管线
python _build_cartesian.py                          # Step 1: 构建配对
python _gnn_inference.py --model models/v5.0/v5_model.pt  # Step 2: GNN 推理
python _filter_diverse.py                           # Step 3: 多样性过滤
python _generate_top40_report.py                    # Step 4: 报告
```

## 开发规范

- Python 3.12，函数 type hints
- 模块 ≤ 800 行，函数 ≤ 50 行
- 不可变优先，避免 in-place 修改
- 功能块前中文注释说明意图
- flake8 lint
- 分支: `master` (稳定) / `develop` (主线) / `feature/*`
- 提交: Conventional Commits

## 环境变量

```bash
export MINIMAX_API_KEY="your-api-key"
```
