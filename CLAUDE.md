# FluoroFilm

## 项目概述

从 ~840 篇 PDF 文献中提取结构化化学信息，针对**二维亚胺键 COF**（共价有机框架）
进行机器学习驱动的单体筛选。项目分两阶段推进：

**Phase 1–5 (已完成 v1.0)**：
1. PDF 解析 (PyMuPDF) → LLM 结构化提取 (MiniMax/MiMo, 21 字段/YAML)
2. 单体 SMILES 提取 + 分类 (醛/胺) + 氟检测 (SMARTS)
3. 手写特征 (ECFP4+MACCS+描述符 ≈2418 维) + XGBoost 成膜预测
4. Route A 严格 2D 筛选 (≥2 官能团, C3+C2→六方, C2+C2→四方)
5. 商业单体 (研伸科技 121 页目录) 通过 MiMo Omni 视觉提取接入
6. 虚拟氟化修正 (已弃用) + 知识图谱 + Word 报告

**Phase 6 (当前)**：用 GNN 深度学习替代手写特征，让模型自动从分子图学习
成膜相关的子结构模式，解决概率饱和和特征表达不足问题。当前推进到 **v4 (Route B)**。

## v1.0 已知问题 (重要)

| 问题 | 描述 | 影响 |
|------|------|------|
| 概率饱和 | 筛选集 margin 远超训练集，校准概率全 ~0.999 | 绝对概率不可信，只用 margin 排名 |
| 标签偏置 | 文献天然偏置——发表的大多是成功成膜案例，负样本不足 | PR-AUC 0.654 可能是数据瓶颈 |
| 特征-目标因果链过长 | Morgan 指纹只描述子结构共现，缺少空间/电子/协同效应 | Top 20 被简单小分子主导 |
| 虚拟氟化不合理 | 芳香 H→F 随机替换不可合成；ΔMargin 反映特征敏感度而非真实氟效应 | 该方向已暂缓 |
| 反应条件未利用 | 溶剂/温度/催化剂等 21 字段中的反应条件未进入特征空间 | 模型只看单体结构，不知道实验条件 |

## 技术栈

| 层级 | 选型 | 版本约束 |
|------|------|----------|
| 语言 | Python | **3.12.7** |
| 包管理 | conda | — |
| PDF 解析 | PyMuPDF | — |
| 化学信息 | RDKit | **2026.03.1** |
| ML (v1.0) | scikit-learn + xgboost | xgboost **3.2.0** |
| ML (Phase 6) | PyTorch + PyTorch Geometric | 待安装 |
| LLM | MiniMax API / MiMo Omni | — |
| 数据存储 | SQLite + YAML | — |
| 知识图谱 | NetworkX + pyvis | — |
| 报告生成 | python-docx | — |
| 可视化 | matplotlib + plotly | — |
| 交互开发 | Jupyter (VS Code 插件) | — |
| Lint | flake8 | — |

## 目录结构

```
FluoroFilm/
├── data/
│   ├── pdfs/                # 原始 PDF 文献
│   ├── extracted/           # PDF 解析后的原文 (.txt)
│   ├── structured/          # LLM 提取后的结构化信息 (.yaml)
│   ├── processed/           # 中间产物: 特征矩阵、筛选结果、报告
│   └── tmp/                 # PDF 页面渲染临时文件 (gitignored)
├── src/
│   ├── __init__.py
│   ├── pdf_parser/          # PDF 解析: parse_pdf.py
│   ├── extraction/          # LLM 提取: llm_extractor.py, minimax_client.py
│   ├── chemistry/           # 化学处理: monomer.py, imine_check.py, fluorination.py
│   ├── screening/           # ML 筛选 (v1.0): features.py, train.py, predict.py
│   └── utils/               # 工具: db.py, logger.py
├── notebooks/
│   └── phase6_gnn/          # Phase 6 GNN 探索 (Jupyter)
│       ├── 01_gnn_baseline.ipynb   # GNN 特征 vs Morgan 指纹对比
│       └── 02_explainability.ipynb # 子结构归因分析
├── tests/                   # 镜像 src/ 结构
├── models/                  # 按版本管理
│   ├── v1.0/                # XGBoost v1.0 (已完成)
│   └── v2.0/                # GNN 模型 (待建)
├── scripts/                 # 15 个运行脚本
├── CLAUDE.md
├── requirements.txt
└── .gitignore
```

## 常用命令

```bash
# 环境
conda activate fluoro-film             # 激活环境
pip install -r requirements.txt        # 安装依赖

# v1.0 ML 管道
python scripts/build_features.py       # 构建特征矩阵 (Morgan 指纹)
python scripts/train_model.py          # 训练 XGBoost/RF/LR
python scripts/screen_monomers_2d.py   # Route A 2D 筛选
python scripts/screen_monomers_2d.py --extra-monomers data/processed/commercial_monomers_classified.csv

# 商业单体提取 (独立支线)
python scripts/extract_commercial_monomers.py --pdf <path>
python scripts/merge_commercial_monomers.py

# 报告 & 图谱
python scripts/build_graphrag.py       # 知识图谱 HTML
python scripts/generate_report.py      # Word 报告

# Phase 6 GNN (Jupyter)
jupyter notebook notebooks/phase6_gnn/
```

## 开发规范

### 分支策略

| 分支 | 用途 |
|------|------|
| `master` | 稳定版本 |
| `develop` | 开发主线 |
| `feature/*` | 功能分支（如 `feature/pdf-parser`, `feature/ml-training`） |

### 提交规范 (Conventional Commits)

```
feat: 添加 PDF 批量解析模块
fix: 修复 RDKit SMILES 解析失败问题
refactor: 重构特征工程管道
test: 补充成膜预测模型单元测试
docs: 更新 CLAUDE.md 工作流说明
```

### 代码规范

- Python 3.12，所有函数必须 type hints
- 模块文件 ≤ 800 行，函数 ≤ 50 行
- 每个 `src/` 子模块最多 5 个 `.py` 文件
- 不可变优先，避免 in-place 修改
- 导出符号在 `__init__.py` 中显式声明
- 每个功能块前加中文注释说明意图（不需要逐行加，按函数/逻辑段加）
- 用 flake8 做 lint

### 测试规范

- 核心模块覆盖率 ≥ 80%
- AAA 模式（Arrange-Act-Assert）
- `tests/` 下镜像 `src/` 结构，每个模块一个测试文件
- 测试命名：`test_<行为描述>`

### 文件层级规则

- `data/pdfs/` 按年份组织，`data/extracted/` 和 `data/structured/` 与源 PDF 同名
- `notebooks/` 按阶段编号，分析过程中不做数据修改
- `models/` 按版本号管理，每个版本包含 scaler 和模型文件
- 跨模块共享的工具函数放 `src/utils/`
- 配置文件统一在项目根目录下作为 `.yaml` 或 `.json`

## v1.0 完整流水线 (已完成)

```
840 篇 PDF ──→ PyMuPDF 提取原文 ──→ MiniMax API 结构化提取 (21字段/YAML)
                                            │
                          ┌─────────────────┘
                          ↓
                  单体 SMILES 提取
                   ├─ ImineChecker (SMARTS 醛基/伯胺)
                   ├─ FluorineDetector (F/CF3)
                   └─ 商业单体 (MiMo Omni 视觉提取 → PubChem CAS→SMILES)
                          │
                          ↓
                  特征工程 (ECFP4+MACCS+描述符 ≈2418维)
                          │
                          ↓
                  模型训练 (XGBoost/RF/LR, 5折CV, SMOTE)
                          │
                          ↓
                  Route A 严格 2D 筛选
                   ├─ ≥2 官能团, ≥2 文献支持
                   ├─ C3+C2→六方, C2+C2→四方, C3+C4→排除
                   ├─ InChI Key 去重, 训练集配对排除
                   └─ 氟策略 ×4: F/非F 交叉配对
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
         Top 20.csv  知识图谱.html  Word报告.docx
```

## Phase 6 GNN 演进

### v2 已完成（预训练+微调）

**成果**：GNN + BilinearHead 端到端配对预测模型，PR-AUC 0.731。

| 指标 | XGBoost v1.0 | GNN v2 |
|------|-------------|--------|
| PR-AUC | 0.654 | 0.731 |
| 评估方式 | 5-fold CV | RepeatedStratifiedKFold (8折×8重复) |
| 特征 | ECFP4+MACCS+描述符 (2418维) | GNN 嵌入 (256维) + 机理描述符 (26维) |
| 训练方式 | — | GNN 预训练 + 微调 |

### v3 已弃置（冷启动 + 多任务 → PR-AUC 0.39）

5层 GIN+GINE + Cross-Graph Attention + ConditionHead (5条件多任务)。
PR-AUC 仅 0.39，根因：**93% 负样本来自全阴性文献（从未尝试成膜）→ 假负样本主导损失函数**。

### v4 Route B (当前) — 数据清洗 + 化学规则负样本

**核心洞察**：化学文献很少发表成膜失败的实验。597/656 篇文献从未尝试成膜（solvothermal 粉末合成），
它们的负样本是"未尝试"，不是"尝试过但失败"。模型学到的是"这篇文献做没做膜"而非"这个配对能不能成膜"。

**Route B 策略**：PU Learning 思路 — 丢弃全阴性文献，用化学规则生成确定性负样本。

#### 数据管线 (`scripts/build_v4_dataset.py`)

```
v3_train.csv (2268 pairs, 24% pos)
    │
    ├─ 1. 丢弃全阴性文献 (597 papers → 59 mixed papers)
    │     保留: 544 pos + 49 lit neg
    │
    ├─ 2. 化学规则负样本生成 (9 策略)
    │     ├─ 基础策略 (4): 不对称/多环/过取代/非平面
    │     └─ 边界策略 (5):
    │         ├─ 刚柔失配: 全刚性多环醛 × 长链脂肪胺
    │         ├─ C1+C3 边界: 单醛基醛 × 三官能团胺
    │         ├─ 非平面性: sp³ 碳桥醛 × 平面胺
    │         ├─ 弱亲核胺: C2醛 × 苯胺/吡啶胺
    │         └─ 过量氟: 全氟苯甲醛 × C2胺
    │     生成 1500 化学负样本 (去重: 与已有负样本不重复)
    │
    └─ 最终: v4_train.csv (2093 samples: 544 pos + 49 lit neg + 1500 chem neg)
```

#### 模型架构 (`src/screening/gnn_v4/`)

| 模块 | 文件 | 说明 |
|------|------|------|
| Encoder | `encoder.py` | GIN+GINE x3 + JK-Net mean pooling (128维), Siamese 共享权重 |
| Attention | `attention.py` | Bidirectional CrossGraphAttention (4头) |
| Pooling | `pooling.py` | BatchedAttentionPool + PairPooling，支持 batch 向量 |
| Head | `heads.py` | FilmHead 仅成膜预测 [512,256,128]→1 |
| Model | `model.py` | V4Model 端到端，`predict_single()` 单样本推理 |
| Loss | `v4_loss.py` | 纯 FocalLoss (α=0.75, γ=2.0)，移除所有多任务组件 |
| Trainer | `v4_trainer.py` | 简化训练循环 + 早停 |
| Train | `train_v4.py` | 15-fold 文献级 CV + collate_fn batch 拼接 |

**模型缩小**：3.19M (v3) → 0.67M (v4)，4.8x 缩减，适配 2093 样本避免过拟合。

#### 训练结果

| 指标 | v3 | v4 v1 (基础负样本) | v4 v2 (+边界负样本) |
|------|-----|-------------------|---------------------|
| CV PR-AUC | 0.39 | 0.7825 ± 0.049 | 0.7744 ± 0.065 |
| Best Fold | — | 0.871 | **0.879** |
| 参数量 | 3.19M | 0.67M | 0.67M |
| 训练数据 | 2268 (93%假负) | 2043 (544+49+1450) | 2093 (544+49+1500) |

**关键发现**：
- 丢弃 93% 假负样本后，PR-AUC 从 0.39 → 0.78 (+100%)
- 边界负样本提升 best fold (0.871→0.879) 但增加方差 (0.049→0.065)
- 仅 21/59 混合文献有统一反应条件 — 提供金标准负样本
- 距离 0.80 目标差 ~0.02，下一步方向：更多金标准负样本或架构调优

## 环境变量

```bash
export MINIMAX_API_KEY="your-api-key"
```
