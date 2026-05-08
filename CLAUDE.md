# FluoroFilm

## 项目概述

从 PDF 文献库中提取结构化信息，针对**二维亚胺键 COF**（共价有机框架）进行机器学习筛选。核心任务是：

1. 阅读约 1000 篇 PDF 文献，通过 LLM 提取结构化化学信息
2. 总结二维 COF 成膜关键控制因素理论（亚胺键影响、氟元素影响、成膜对液晶去除的应用）
3. 基于理论构建 ML 特征，用 DeepChem 训练成膜预测模型
4. **路线 A**：先筛选含氟醛/胺单体 → 与非含氟单体自由配对 → 预测成膜性 → 选出 Top 20
5. **补充修正**：对路线 A 筛出的非含氟组合，虚拟氟化后评估成膜提升幅度
6. 结果导出为 Word 可视化报告 + GraphRAG 知识图谱

## 技术栈

| 层级 | 选型 | 版本约束 |
|------|------|----------|
| 语言 | Python | **3.8.20** |
| 包管理 | conda | — |
| PDF 解析 | PyMuPDF + GROBID | GROBID 免费开源 (Apache 2.0) |
| 化学信息 | RDKit + DeepChem | DeepChem **2.8.0** |
| ML 框架 | scikit-learn + xgboost | — |
| LLM | MiniMax API | — |
| 数据存储 | SQLite | — |
| 知识图谱 | Microsoft GraphRAG | — |
| 报告生成 | python-docx | — |
| 可视化 | matplotlib + plotly | — |
| Lint | flake8 | — |

### GROBID

- 本地部署：`docker run -p 8070:8070 grobid/grobid`
- 或使用公开 API: `cloud.science-miner.com/grobid`
- 用途：学术论文结构化解析（标题、作者、摘要、参考文献分区）

## 目录结构

```
FluoroFilm/
├── data/
│   ├── pdfs/                # 原始 PDF 文献，按年份分组: pdfs/2020/, pdfs/2021/, ...
│   ├── extracted/           # 与 pdfs/ 同名对应的提取原文 (.txt)
│   ├── structured/          # LLM 提取后的结构化信息 (.yaml)
│   └── processed/           # 特征矩阵、训练集 (.csv, .npz)
├── src/
│   ├── __init__.py
│   ├── pdf_parser/          # PDF 解析: parse_pdf.py, grobid_client.py
│   ├── extraction/          # LLM 提取: llm_extractor.py, minimax_client.py
│   ├── chemistry/           # 化学处理: monomer.py, imine_check.py, fluorination.py
│   ├── screening/           # ML 筛选: features.py, train.py, predict.py
│   └── utils/               # 工具: db.py, logger.py
├── notebooks/               # 按阶段编号
│   ├── 01_explore_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_screening_results.ipynb
│   └── 05_fluorine_analysis.ipynb
├── tests/                   # 镜像 src/ 结构
├── models/                  # 按版本管理: v1.0/, v2.0/
├── scripts/                 # 运行脚本
├── .claude/
├── README.md
├── CLAUDE.md
├── environment.yml          # conda 环境定义
├── requirements.txt
└── .gitignore
```

## 常用命令

```bash
# 环境
conda activate fluoro-film             # 激活环境
conda env create -f environment.yml    # 从 yaml 创建环境
pip install -r requirements.txt        # 安装依赖

# 测试 & 检查
pytest tests/ -v                       # 运行测试
pytest tests/ --cov=src --cov-report=html  # 覆盖率报告
flake8 src/                            # 代码检查

# GROBID 服务
docker run -d -p 8070:8070 grobid/grobid   # 启动 GROBID

# PDF 解析管道
python scripts/parse_pdfs.py --input data/pdfs/ --output data/extracted/

# LLM 结构化提取
python scripts/extract_info.py --input data/extracted/ --output data/structured/

# ML 筛选
python scripts/build_features.py       # 构建特征矩阵
python scripts/train_model.py          # 训练 DeepChem 成膜预测模型
python scripts/screen_monomers.py --top 20  # 筛选 Top 20 单体对
python scripts/virtual_fluorination.py # 虚拟氟化修正

# 知识图谱
python scripts/build_graphrag.py       # 构建 GraphRAG

# 报告
python scripts/generate_report.py      # 生成 Word 可视化报告
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

- Python 3.8.20，所有函数必须 type hints（使用 `typing` 模块）
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

## 核心工作流 (Route A + 虚拟氟化修正)

```
1000篇文献 → PyMuPDF/GROBID 提取原文 → MiniMax API 结构化提取 → SQLite 存储
                                                                     ↓
         理论总结（人工 + LLM 辅助）：
           ├─ 二维 COF 成膜关键控制因素
           ├─ 亚胺键对成膜的影响
           ├─ 含氟单体对成膜的影响
           └─ 成膜如何应用于液晶去除
                                                                     ↓
         理论 → 转化为 ML 筛选规则/特征工程
                                                                     ↓
         筛选路线 A：
           Step 1: 筛选可形成亚胺键的醛/胺单体
           Step 2: 从中筛出含氟单体
           Step 3: 含氟 + 非含氟单体自由配对
           Step 4: DeepChem 预测成膜性 → Top 20
                                                                     ↓
         补充修正（虚拟氟化）：
           对路线 A 的非含氟组合预测加氟后成膜提升幅度
                                                                     ↓
         结果输出：
           ├─ Word 报告（排名表格 + 分子结构图 + 各维度得分）
           └─ GraphRAG 知识图谱（单体-性质-条件 实体关系）
```

### 筛选权重参考（后续实现时确定）

- 成膜预测：~50%
- 亚胺键稳定性：~次要
- 其他维度：后续讨论

## 环境变量

```bash
export MINIMAX_API_KEY="your-api-key"
export GROBID_URL="http://localhost:8070"  # 或 cloud.science-miner.com/grobid
```
