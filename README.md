# FluoroFilm — 二维 COF 成膜预测与单体筛选

基于机器学习的二维亚胺键 COF（共价有机框架）单体筛选工具，聚焦含氟/非含氟单体对成膜性能的影响。

## 核心任务

1. 从 ~1000 篇 PDF 文献中提取结构化化学信息
2. 总结二维 COF 成膜关键控制因素理论
3. 基于 XGBoost 训练成膜预测模型
4. **路线 A**：四种含氟/非含氟配对策略筛选 Top 20 亚胺键 COF 单体对
5. **虚拟氟化修正**：对非含氟组合评估加氟后成膜提升幅度

## 技术栈

| 层级 | 选型 | 版本 |
|------|------|------|
| 语言 | Python | 3.12 |
| PDF 解析 | PyMuPDF + GROBID | — |
| 化学信息 | RDKit | 2026.03 |
| ML 框架 | scikit-learn + XGBoost | 3.2.0 |
| LLM | MiniMax API (M2.7) | OpenAI SDK |
| 数据存储 | SQLite + YAML | — |
| 可视化 | matplotlib | 3.10 |

---

## 完整数据流向

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: PDF 解析                                                  │
│                                                                     │
│  data/pdfs/*.pdf (954篇)                                            │
│      │                                                              │
│      ├─[GROBID API]──→ TEI XML (学术结构化解析)                      │
│      └─[PyMuPDF]─────→ data/extracted/*.full.txt (纯文本)            │
│                                                                     │
│  关键文件: src/pdf_parser/parse_pdf.py, grobid_client.py             │
│  入口脚本: scripts/parse_pdfs.py                                     │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 2: LLM 结构化提取                                            │
│                                                                     │
│  data/extracted/*.full.txt                                          │
│      │                                                              │
│      └─[MiniMax API]──→ data/structured/*.yaml (21字段)             │
│                         data/fluorofilm.db (SQLite)                  │
│                                                                     │
│  关键文件: src/extraction/llm_extractor.py, minimax_client.py        │
│  入口脚本: scripts/extract_info.py                                   │
│  API 配置: config/extraction.yaml                                    │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 3: 化学信息学处理                                            │
│                                                                     │
│  reagent 字段 "Tp（1,3,5-三甲酰基间苯三酚）、Pa（对苯二胺）"         │
│      │                                                              │
│      ├─[monomer.py]──→ 名称 → SMILES → RDKit Mol                   │
│      │                 三级降级: 内置字典 → JSON缓存 → PubChem API   │
│      │                                                              │
│      ├─[imine_check.py]──→ SMARTS 判定: 醛基/胺基/亚胺键            │
│      │                                                              │
│      └─[fluorination.py]──→ 氟检测 (F/CF3) + 虚拟氟化 (H→F替换)     │
│                                                                     │
│  关键文件: src/chemistry/monomer.py, imine_check.py, fluorination.py │
│  扩展脚本: scripts/extract_monomer_smiles.py (LLM辅助SMILES提取)     │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 4: 特征工程                                                  │
│                                                                     │
│  文献记录 → 提取醛/胺单体对 → 四层特征向量                           │
│      │                                                              │
│      ├─ Layer 1: Morgan ECFP4 (1024) × 2 单体                       │
│      ├─ Layer 2: MACCS Keys (167) × 2 单体                          │
│      ├─ Layer 3: 分子描述符 (13维) × 2 单体                          │
│      └─ Layer 4: 配对特征 (9维) — F总数、醛胺比、MW等                │
│                                                                     │
│  总计 ~2418 维特征                                                  │
│  标签 y: film_crystallinity_fluorine → 成膜(1) / 不成膜(0)          │
│                                                                     │
│  输出: data/processed/X_features.npz, y_labels.npy                  │
│                                                                     │
│  关键文件: src/screening/features.py                                 │
│  入口脚本: scripts/build_features.py                                 │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 5: 模型训练                                                  │
│                                                                     │
│  X, y → 分层分割 → 特征预处理 → 超参调优 → 训练 → CV → 评估          │
│      │                                                              │
│      ├─ 80/20 分层分割 (train/test)                                  │
│      ├─ 低方差过滤 + 高相关过滤 + StandardScaler                     │
│      ├─ RandomizedSearchCV (XGBoost + RF)                           │
│      ├─ 5-fold Stratified CV (PR-AUC 主指标)                        │
│      └─ 测试集: Precision/Recall/F1/ROC-AUC/PR-AUC                  │
│                                                                     │
│  输出: models/v1.0/ (3模型 + scaler + 特征选择 + 评估结果)           │
│                                                                     │
│  关键文件: src/screening/train.py                                    │
│  入口脚本: scripts/train_model.py                                    │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 6: 路线 A 筛选 + 虚拟氟化修正                                  │
│                                                                     │
│  从文献数据库提取唯一单体                                            │
│      │                                                              │
│      ├─ 筛选亚胺键单体 (醛基/胺基)                                   │
│      ├─ 按含氟/不含氟 → 四组: F-醛, 非F-醛, F-胺, 非F-胺            │
│      │                                                              │
│      ├─ 四种配对策略:                                                │
│      │   ① F-醛 × 非F-胺    ② 非F-醛 × F-胺                        │
│      │   ③ F-醛 × F-胺      ④ 非F-醛 × 非F-胺                       │
│      │                                                              │
│      ├─ XGBoost 预测成膜概率 → 排名 → Top N                          │
│      │                                                              │
│      └─ 虚拟氟化修正:                                                │
│           · 单F组合: 对非F单体 +1F → 重预测                          │
│           · 无F组合: 醛+1F / 胺+1F / 双+1F → 取最佳                  │
│                                                                     │
│  输出: data/processed/route_a_top20.csv                              │
│                                                                     │
│  关键文件: src/screening/predict.py                                  │
│  入口脚本: scripts/screen_monomers.py                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 文件清单与说明

### 配置文件 (`config/`)

| 文件 | 用途 |
|------|------|
| `extraction.yaml` | MiniMax API 参数 (模型 M2.7, temperature 0.3, max_input_chars 8000) + 21 个提取字段定义 |
| `grobid.yaml` | GROBID 服务地址、超时设置、输入输出路径 |

### 核心源码 (`src/`)

#### `src/pdf_parser/` — PDF 解析层

| 文件 | 行数 | 功能 |
|------|------|------|
| `parse_pdf.py` | 31 | `extract_text()` — 用 PyMuPDF 逐页提取纯文本；`extract_metadata()` — 读取 PDF 元数据 |
| `grobid_client.py` | 93 | `GrobidClient` 类 — REST 客户端调用 GROBID API，将 PDF 转为 TEI XML 结构化文档（含标题、作者、摘要、参考文献分区） |

**数据流**: `data/pdfs/*.pdf` → PyMuPDF → `data/extracted/*.full.txt`；可选 GROBID → TEI XML

#### `src/extraction/` — LLM 提取层

| 文件 | 行数 | 功能 |
|------|------|------|
| `minimax_client.py` | 70 | `MiniMaxClient` — OpenAI 兼容 SDK 封装，对接 MiniMax API。`chat()` 方法含自动重试逻辑 (3次) |
| `llm_extractor.py` | 143 | `LLMExtractor` — COF 文献分析专家 System Prompt 构建 + YAML 响应解析。支持 YAML 块提取、正则兜底解析、缺失字段填充 |

**数据流**: `data/extracted/*.full.txt` → MiniMax M2.7 → `data/structured/*.yaml` (21 字段) + `data/fluorofilm.db` (SQLite)

**21 个提取字段**: `journal`, `system`, `conclusion_1/2/3`, `methods`, `reagent`, `catalyst`, `solvent`, `innovation`, `film_crystallinity_fluorine`, `reaction_temperature`, `schiff_base_kinetics`, `fluorine_effects`, `adsorption_mechanism`, `computational_methods`, `interface_type`, `annealing_conditions`, `synthesis_route`, `fluorine_monomer`, `synthesis_mode`

#### `src/chemistry/` — 化学信息层

| 文件 | 行数 | 功能 |
|------|------|------|
| `monomer.py` | 520 | `MonomerLibrary` — 名称→SMILES 三级降级引擎。内置 ~200 个 COF 常见单体字典 (Tp, TAPB, Pa, TA, TFTA, TAPT 等)；`extract_monomer_names()` 从 reagent 字段拆分单体名称（**仅按中文分隔符 `、；`** 拆分，避免切断 IUPAC 名）；`normalize_fluorine_monomer_field()` 关键字归一化判定含氟 |
| `imine_check.py` | 87 | `ImineChecker` — SMARTS 判定: 醛基 `[CX3H1](=O)[#6]`、伯胺 `[NH2]`、亚胺键 `[CX3](=[NX2])[#6]`。`classify_monomer()` 返回完整单体分类 |
| `fluorination.py` | 120 | `FluorineDetector` — 氟检测: `[F]` 和 CF3 `[F][C](F)(F)` SMARTS。`virtual_fluorination()` — 芳香环 H→F 启发式替换 (SMARTS `[cH]`)，用于评估加氟效果 |

**关键设计决策 — 名称拆分**:
- reagent 字段格式如 `"1,3,5-triformylphloroglucinol（Tp，三甲酰基间苯三酚）、p-phenylenediamine（Pa，对苯二胺）"`
- **不能**按英文逗号 `,` 拆分，因为 IUPAC 名本身含逗号 (`1,3,5-...`)
- 仅按中文分隔符 `、；` 拆分，再从括号内提取英文缩写

**SMILES 解析三级降级**:
1. **内置字典** (~200 常见单体，已验证 Canonical SMILES)
2. **JSON 缓存** (`monomer_smiles_cache.json`，运行时累积)
3. **LLM 辅助** (`monomer_smiles_llm.json`，批量从文献 reagent 提取)
4. (可选) PubChem PUG REST API (受网络限制，默认禁用)

#### `src/screening/` — ML 筛选层

| 文件 | 行数 | 功能 |
|------|------|------|
| `features.py` | 296 | `FeatureEngineer` — 四层特征构建 (Morgan ECFP4 + MACCS + 描述符 + 配对特征)，标签解析 `_extract_film_label()` (强阳性/强阴性关键词 → 二元标签) |
| `train.py` | 518 | `ModelTrainer` — 标准 ML 流水线: 分层分割 → 特征选择 (低方差+高相关过滤) → StandardScaler → RandomizedSearchCV 超参调优 → 训练 (XGBoost/RF/LR) → 5-fold CV → 测试集评估 → 学习曲线/特征重要性图 → 模型持久化 |
| `predict.py` | 351 | `MonomerScreener` — 路线 A 四配对策略 + XGBoost 成膜预测 + 虚拟氟化修正 (单F/双F/无F 三种策略) |

**模型评估指标** (PR-AUC 为主，适合不平衡分类):
- Precision / Recall / F1
- ROC-AUC
- PR-AUC (Precision-Recall AUC) — **主指标**

#### `src/utils/` — 工具层

| 文件 | 行数 | 功能 |
|------|------|------|
| `db.py` | 69 | SQLite 数据库: 建表 (`literature` 表, 21字段 + created_at, WAL模式), 增/查/导出 |
| `logger.py` | 19 | `setup_logger()` — 统一日志格式: `[时间] [级别] [模块]: 消息` |

---

### 运行脚本 (`scripts/`)

按管道阶段顺序执行：

| 序号 | 脚本 | 输入 | 输出 | 耗时 |
|------|------|------|------|------|
| 1 | `parse_pdfs.py` | `data/pdfs/` | `data/extracted/*.full.txt` | ~10 min |
| 2 | `extract_info.py` | `data/extracted/` | `data/structured/*.yaml` + SQLite | ~2-4 hrs |
| 3 | `extract_monomer_smiles.py` | `data/structured/` | `monomer_smiles_llm.json` + 更新缓存 | ~2-3 hrs |
| 4 | `build_features.py` | SQLite + 缓存 | `X_features.npz` + `y_labels.npy` | ~5 min |
| 5 | `train_model.py` | `X_features.npz` | `models/v1.0/*.pkl` | ~10 min |
| 6 | `screen_monomers.py` | SQLite + 模型 | `route_a_top20.csv` | ~2 min |

**并发说明**:
- `extract_info.py`: `--workers 3` 建议值（MiniMax API 3 线程并发）
- `extract_monomer_smiles.py`: `--workers 3` 同上
- 其余脚本为单线程

**断点续传**:
- `extract_info.py --skip-existing`: 跳过已有 YAML 的文件
- `extract_monomer_smiles.py --no-skip`: 重新处理已缓存的文献

---

### 数据文件 (`data/`)

```
data/
├── pdfs/                        # 原始 PDF (按年份分目录，gitignore)
│   ├── 2020/
│   ├── 2021/
│   └── ...
├── extracted/                   # 提取的纯文本 (.full.txt)
│   └── *.full.txt               # 与源 PDF 同名，~957 篇
├── structured/                  # LLM 结构化提取 (.yaml)
│   └── *.yaml                   # 与源 PDF 同名，21 字段，~955 篇
├── processed/                   # ML 特征与中间产物
│   ├── X_features.npz           # 特征矩阵 (n_samples × 2418)
│   ├── y_labels.npy             # 标签向量 (n_samples,)
│   ├── feature_names.json       # 特征名称列表
│   ├── label_metadata.csv       # 标签元数据 (文献ID+单体名+含氟)
│   ├── monomer_smiles_cache.json # 名称→SMILES 缓存
│   ├── monomer_smiles_llm.json  # LLM 辅助识别结果
│   └── route_a_top20.csv        # 路线 A 筛选输出
└── fluorofilm.db                # SQLite 主数据库 (WAL模式)
```

---

### 模型文件 (`models/v1.0/`)

```
models/v1.0/
├── xgboost_model.pkl            # 主模型 (XGBoost 二分类)
├── random_forest_model.pkl      # 对照模型
├── logistic_model.pkl           # 基线模型
├── scaler.pkl                   # StandardScaler
├── model_info.json              # 特征选择索引 + CV/测试评估结果
├── feature_importance.png       # Top 30 特征重要性图
└── learning_curve.png           # 学习曲线 (诊断过/欠拟合)
```

---

## 快速开始

```bash
# 1. 激活环境
conda activate dphuanjing

# 2. 完整管道 (从头运行)
# 步骤 1-2 已预先完成 (YAML + SQLite 已有数据)

# 步骤 3: 扩展单体库 (可选，已有内置字典)
python scripts/extract_monomer_smiles.py --workers 3 --no-skip

# 步骤 4: 构建特征矩阵
python scripts/build_features.py --no-pubchem

# 步骤 5: 训练模型
python scripts/train_model.py --no-tune  # 如数据量小可跳过调参

# 步骤 6: 路线 A 筛选
python scripts/screen_monomers.py --top 20

# 查看结果
cat data/processed/route_a_top20.csv
```

### 仅运行路线 A 筛选 (模型已训练)

```bash
python scripts/screen_monomers.py --top 20
```

---

## 路线 A 筛选策略详解

### 四种配对模式

| 模式 | 醛单体 | 胺单体 | 含氟 | 虚拟氟化 |
|------|--------|--------|------|----------|
| ① | F-醛 | 非F-胺 | 单F | 非F胺 +1F |
| ② | 非F-醛 | F-胺 | 单F | 非F醛 +1F |
| ③ | F-醛 | F-胺 | 双F | 无需修正 |
| ④ | 非F-醛 | 非F-胺 | 无F | 三者取最佳: 醛+1F / 胺+1F / 双+1F |

### 虚拟氟化修正

对于无氟或单氟组合，通过 SMARTS `[cH] → [cF]` 在芳香环上替换一个 H 为 F，位阻评分最小位点优先。氟化后重新预测成膜概率，与原分数比较得到 `fluorination_gain`。

### 输出表格字段

| 字段 | 含义 |
|------|------|
| `aldehyde` | 醛单体名称 |
| `amine` | 胺单体名称 |
| `pair_type` | 配对类型 |
| `film_probability` | 原始成膜预测概率 |
| `fluorinated_score` | 虚拟氟化后的成膜概率 |
| `fluorination_gain` | 氟化提升幅度 (Δ) |

---

## 安全与环境

```bash
export MINIMAX_API_KEY="your-api-key"    # 必需
export GROBID_URL="http://localhost:8070" # 可选
```

---

## 开发规范

- **分支**: `master`(稳定) / `develop`(开发) / `feature/*`
- **提交**: Conventional Commits (`feat:`, `fix:`, `refactor:`)
- **代码**: Python 3.8+ type hints，模块 ≤800行，函数 ≤50行
- **测试**: AAA 模式，覆盖率 ≥80%，`tests/` 镜像 `src/`
- **不可变优先**: 避免 in-place 修改
