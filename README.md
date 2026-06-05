# FluoroFilm — 二维 COF 成膜预测

基于图神经网络（GNN）预测亚胺键共价有机框架（COF）单体对的成膜概率。输入醛+胺 SMILES，输出成膜概率 (0~1) 及 MC Dropout 不确定性。

**当前版本：v5** | PR-AUC 0.784 ± 0.06 | 四分类连续标签 FocalLoss + 规则向量注入

---

## 项目目标

COF 薄膜在气体分离、离子传导、光电器件等领域有广泛应用前景。然而，从海量单体组合中实验筛选可成膜配对成本极高。本项目提供：

1. **单对预测** — 给定醛+胺 SMILES，预测其缩聚产物能否形成连续 COF 薄膜
2. **不确定性估计** — MC Dropout 给出预测置信区间，辅助判断可靠性
3. **批量筛选（可选）** — 从单体池中大规模搜索高成膜概率候选

---

## 快速开始

```bash
conda activate dphuanjing

# 单对预测
python predict_pair.py --ald "O=CC1=CC=C(C=O)C=C1" --amine "NC1=CC=C(N)C=C1"
# 输出: 成膜概率 0.xxxx ± 0.xxxx  |  判定: 高/中/低概率成膜

# 更多 MC 采样以提高不确定性估计精度
python predict_pair.py --ald "O=CC1=CC=C(C=O)C=C1" --amine "NC1=CC=C(N)C=C1" --mc 20
```

---

## 筛选管线

```
                        ┌──────────────────────────┐
                        │  醛 SMILES + 胺 SMILES    │
                        └──────────┬───────────────┘
                                   │
                        ┌──────────▼───────────────┐
                        │  smiles_to_graph()       │
                        │  分子图构建 (原子/键特征)   │
                        └──────────┬───────────────┘
                                   │
                        ┌──────────▼───────────────┐
                        │  GIN+GINE Encoder (共享)  │
                        │  3层 × 128维 + JK-Net     │
                        └──────────┬───────────────┘
                                   │
                        ┌──────────▼───────────────┐
                        │  CrossGraphAttention     │
                        │  双向 4头交叉注意力        │
                        └──────────┬───────────────┘
                                   │
                        ┌──────────▼───────────────┐
                        │  PairPooling → e_pair    │
                        │  + 23维规则向量注入        │
                        └──────────┬───────────────┘
                                   │
                        ┌──────────▼───────────────┐
                        │  FilmHead                │
                        │  [768→512→256→128] → logit│
                        └──────────┬───────────────┘
                                   │
                        ┌──────────▼───────────────┐
                        │  sigmoid → 成膜概率 (0~1) │
                        │  MC Dropout → 不确定性    │
                        └──────────────────────────┘
```

**批量筛选管线（可选）：**

```
单体池 (醛×胺) ──→ 笛卡尔积配对 (~232K) ──→ GNN 推理 (MC Dropout)
    ──→ 化学先验调分 ──→ Morgan Tanimoto < 0.8 多样性 ──→ Top 40 候选
```

---

## 模型架构

V4Model (0.70M 参数)，v5 训练复用：

```
醛 SMILES ──→ GIN Encoder (共享) ──→ ald_emb (128维)
胺 SMILES ──→ GIN Encoder (共享) ──→ amine_emb (128维)
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

关键设计：
- **共享 GIN 编码器**：醛和胺共享同一编码器，减少参数量，学习统一的分子表示
- **规则向量注入**：23 维化学硬约束规则向量直接注入 FilmHead，模型自行学习规则模式
- **FocalLoss 连续标签**：$p_t = 1 - |y - \hat{y}|$，支持 0.0 / 0.7 / 0.8 / 1.0 四级标签
- **MC Dropout**：推理时保持 Dropout 激活，多次采样给出不确定性估计

---

## 数据

### 训练集

| 标签 | 含义 | 数量 | 占比 |
|------|------|------|------|
| 1.0 | 文献确认连续成膜 | 429 | 6.9% |
| 0.8 | 粉末合成（成功但非连续膜） | 291 | 4.7% |
| 0.7 | 增广正样本（文献正样本衍生） | 1,308 | 21.1% |
| 0.0 | 负样本（化学规则 + 文献内负样本） | 4,173 | 67.3% |
| **合计** | | **6,201** | |

- 独立文献数：357 篇（来自 ~840 篇 PDF 的 LLM 结构化提取）
- 独立 paper_id：5,710（含合成负样本）
- 文献级 StratifiedKFold 交叉验证，防止同文献数据泄漏

### 数据来源

| 来源 | 数量 | 说明 |
|------|------|------|
| v3_db_full | 848 | LLM 从文献 PDF 提取的结构化配对数据 |
| augmented_v4 | 1,308 | 基于文献正样本的化学增广 |
| hard_rule_sampled | 3,104 | 硬规则确定性负样本 |
| chem_rule_* | 941 | 化学规则边界负样本（6 类子规则） |

---

## 训练结果

| 版本 | PR-AUC | 数据量 | 标签类型 | 备注 |
|------|--------|--------|----------|------|
| v4 (2026-05) | 0.76 ± 0.07 | 2,093 | 二分类 | Route B 基线 |
| **v5 (2026-06)** | **0.784 ± 0.06** | 6,201 | 四分类连续 | 当前版本 |

v5 7-fold CV: 0.827, 0.797, 0.713, 0.833, 0.791, 0.856, 0.670

---

## 目录结构

```
├── predict_pair.py               # 单对预测入口
├── src/
│   ├── screening/gnn_v4/         # GNN 模型 (encoder, attention, pooling, heads, loss, trainer)
│   ├── screening/gnn_v3/         # 分子图构建 (featurizer.py)
│   ├── chemistry/                # 化学规则、连接基团分析、3D 描述符、负样本生成
│   └── utils/                    # 日志工具
├── scripts/                      # 训练、数据构建、增广、批量筛选、报告生成
├── _build_cartesian.py           # 批量筛选 Step 1: 笛卡尔积配对
├── _gnn_inference.py             # 批量筛选 Step 2: GNN 推理
├── _filter_diverse.py            # 批量筛选 Step 3: 多样性过滤
├── _generate_top40_report.py     # 批量筛选 Step 4: Word 报告
├── config/model_v4.yaml          # 模型/训练超参数
├── data/processed/               # 训练数据、筛选结果
├── models/v5.0/                  # 模型权重
├── jiyi/                         # 工作日志
└── notebooks/                    # Jupyter 探索笔记
```

---

## 技术栈

| 层级 | 选型 |
|------|------|
| 语言 | Python 3.12 |
| 化学信息学 | RDKit |
| 图神经网络 | PyTorch + PyTorch Geometric |
| LLM 提取 | MiniMax API / MiMo Omni |
| 报告生成 | python-docx |

---

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{fluorofilm2026,
  title   = {FluoroFilm: GNN-Based Film Formation Prediction for 2D COF Imine Monomers},
  author  = {},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## 许可证

待定
