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
成膜相关的子结构模式，解决概率饱和和特征表达不足问题。

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

## Phase 6 GNN 路线 (当前)

**目标**：用图神经网络替代手写 Morgan 指纹，让模型自动从分子图中学习成膜相关子结构。

**渐进式迁移** (不推倒重来)：

```
阶段 1 (2天): GNN 特征提取 vs Morgan 指纹
  ├── 分子图 → 3层GNN → 单体向量 (256维)
  ├── 醛向量 + 胺向量 → XGBoost 分类头
  └── 对比: GNN特征 vs Morgan指纹, AUC 谁高
      ⚠️ 如果 GNN AUC 比 Morgan 低 >5 个百分点 → 停止, 回查数据质量
         (可能原因: 数据噪音严重, 或成膜主要由简单特征决定)

阶段 2 (3天): 端到端训练 + 反应条件
  ├── GNN(醛) + GNN(胺) + MLP(反应条件) → 预测头
  └── 多任务: 成膜 + 结晶度 + 拓扑类型

阶段 3 (2天): 可解释性
  └── GNNExplainer / 注意力权重 → 模型学到了什么子结构
      与你心中的化学直觉对比验证
```

### 技术选型

- **框架**: PyTorch + PyTorch Geometric (PyG) — 社区最大,文档最好
- **开发方式**: Jupyter Notebook (VS Code 插件), 逐 cell 探索, 确认后转为 .py 脚本
- **预训练**: 考虑 ChemBERTa 分子预训练权重微调 (避免 487 条数据训大模型过拟合)

### 数据现状

| 项目 | 数量 |
|------|------|
| 结构化文献 | ~840 篇 YAML |
| 有标签样本 | 487 条 (正负不均衡, ~15% 正样本) |
| 特征维度 | 2418 (ECFP4 2048 + MACCS 334 + 描述符 24 + 配对 12) |
| 单体池 (含商业) | 1104 个, 753 个 2D 可用 |
| 反应条件字段 | 溶剂/温度/催化剂/合成路线/界面类型 (多数文献有空缺) |

### 模型 v1.0 基线

| 指标 | 值 |
|------|-----|
| XGBoost 5-fold CV PR-AUC | 0.654 |
| 概率校准 | Isotonic (筛选集饱和至全 ~0.999) |
| Top 20 配对类型 | 非F-醛×F-胺 11/20, 非F×非F 6/20 |
| 主导单体 | CF3-苯二胺 (商业) 出现 12/20 次 |

### 评估策略

- **内部验证**: 5 折分层 CV (PR-AUC, ROC-AUC, F1)
- **外部泛化**: 商业单体独立测试集 (模型从未见过的结构分布)
- **化学合理性**: 可解释性工具验证模型学到的子结构是否符合化学直觉

## 环境变量

```bash
export MINIMAX_API_KEY="your-api-key"
```
