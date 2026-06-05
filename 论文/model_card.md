# 模型卡片 — FluoroFilm V4Model (v5 训练)

## 概述

V4Model 是一个基于图神经网络的成膜预测模型。输入醛+胺 SMILES 分子图，通过共享 GIN 编码器、交叉图注意力和规则向量注入，输出成膜概率 (0~1)。

- **参数量**：0.70M
- **训练版本**：v5（四分类连续标签 FocalLoss）
- **PR-AUC**：0.784 ± 0.06（7-fold CV）

---

## 模型架构

```
醛 SMILES ──→ smiles_to_graph(role=0) ──→ GIN+GINE Encoder (共享) ──→ ald_emb [N_ald, 128]
胺 SMILES ──→ smiles_to_graph(role=1) ──→ GIN+GINE Encoder (共享) ──→ amine_emb [N_am, 128]
                                                                          │
                                                          ┌───────────────┘
                                                          ↓
                                               CrossGraphAttention (双向, 4头)
                                                          ↓
                                               PairPooling → e_pair [128]
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ↓                               ↓
                               [ea, eb, ea*eb, e_pair]            rule_vec [23]
                                          │                               │
                                          └───────────┬───────────────────┘
                                                      ↓
                                          FilmHead MLP [768→512→256→128→1]
                                                      ↓
                                                  sigmoid → prob [0~1]
```

### 各模块详情

#### 1. 分子图构建 (`smiles_to_graph`)

- **原子特征 (37 维)**：元素类型 (10) + 杂化 (5) + 电荷 (5) + 度 (6) + H 数 (4) + 芳香性 (1) + 环内 (1) + 反应角色 (1) + 位置编码 (2) + 手性 (2)
- **边特征 (5 维)**：键类型 (4) + 共轭 (1)
- **位置编码**：BFS 距离到最近醛基/胺基，归一化到 [0, 1]

#### 2. SiameseEncoder（共享权重）

- **层数**：3 层 GINEConv
- **隐藏维度**：128
- **结构**：GINEConv(edge_mlp + node_mlp) → LayerNorm → 残差连接 → Dropout(0.15)
- **JK-Net**：3 层输出 mean pooling 拼接
- **参数量**：~0.3M

#### 3. CrossGraphAttention（双向交叉注意力）

- **头数**：4
- **头维度**：32 (128 / 4)
- **结构**：醛→胺 和 胺→醛 双向交叉注意力，各自残差 + LayerNorm
- **缩放因子**：1 / sqrt(32)

#### 4. PairPooling

- **BatchedAttentionPool**：4 个可学习 query 向量，多头注意力池化 → 拼接 → MLP 压缩
- **PairPooling**：醛池化向量 + 胺池化向量 + 差值的绝对值 → e_pair

#### 5. FilmHead

- **输入**：[ea (128), eb (128), ea*eb (128), e_pair (128), emb_3d (64), rule_vec (23)] = 599 维
- **3D 分支**：ConformerBranch 将单体 3D (10维) + 二聚体 3D (10维) → 64 维嵌入；消融表明单体 3D 无显著提升，当前默认关闭单体 3D，仅保留二聚体 3D
- **MLP 结构**：768→512→256→128→1（实际第一层 407→256）
- **激活**：ReLU
- **Dropout**：0.25
- **规则向量**：23 维 0/1 向量，直接拼接到输入层

---

## 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| hidden_dim | 128 | 编码器隐藏维度 |
| num_layers | 3 | GINE 层数 |
| num_heads | 4 | 注意力头数 |
| num_queries | 4 | 池化 query 数 |
| dropout (encoder/attention) | 0.15 | |
| dropout (FilmHead) | 0.25 | |
| batch_size | 32 | |
| max_epochs | 200 | |
| early_stop_patience | 30 | 30 epoch 无提升则停止 |
| learning_rate | 0.0005 | |
| weight_decay | 0.0001 | |
| optimizer | AdamW | |
| scheduler | CosineAnnealing | 带 5 epoch warmup |
| grad_clip | 1.0 | |
| focal_alpha | 0.75 | 正样本权重 |
| focal_gamma | 2.0 | 难样本聚焦强度 |

---

## 损失函数

**FocalLoss 连续版本**：

$$p_t = 1 - |y - \hat{y}|$$
$$\alpha_t = \alpha \cdot y + (1 - \alpha) \cdot (1 - y)$$
$$\mathcal{L} = \alpha_t \cdot (1 - p_t)^\gamma \cdot \text{BCE}(\hat{y}, y)$$

- 支持 0.0 / 0.7 / 0.8 / 1.0 四级连续标签
- 标签越接近 0.5，$p_t$ 越小，loss 权重越大（聚焦难样本）

---

## 训练配置

- **CV 策略**：StratifiedKFold，5 折 x 3 重复 = 15 折
- **分层依据**：paper_id（文献级，防止同文献数据泄漏）
- **最佳 fold 重训**：CV 完成后选 PR-AUC 最高的 fold，用该 fold 的训练集全量重训最终模型
- **评估指标**：PR-AUC（Precision-Recall AUC），因正负样本不平衡 (1:2.1)
- **3D 描述符**：可选，消融实验表明无显著提升，当前默认关闭

---

## 训练结果

| 版本 | PR-AUC | 数据量 | 标签类型 |
|------|--------|--------|----------|
| v4 (2026-05) | 0.76 ± 0.07 | 2,093 | 二分类 |
| **v5 (2026-06)** | **0.784 ± 0.06** | 6,201 | 四分类连续 |

v5 7-fold CV 详细：0.827, 0.797, 0.713, 0.833, 0.791, 0.856, 0.670

---

## 推理

### 单次推理

```python
model.eval()
with torch.no_grad():
    logit = model.predict_single(ald_graph, amine_graph, rule_vec=rv)
    prob = torch.sigmoid(logit)
```

### MC Dropout（不确定性估计）

```python
model.enable_mc_dropout()  # 强制 dropout 保持激活
probs = []
for _ in range(n_samples):
    logit = model.predict_single(...)
    probs.append(torch.sigmoid(logit).item())
mean, std = np.mean(probs), np.std(probs)
```

---

## 已知局限

1. **仅支持亚胺键 COF**：模型未在硼酸酯、酰亚胺等键合类型上训练
2. **需要 >=2 官能团**：单官能团单体无法形成 2D 网络，模型对此类输入行为未定义
3. **化学空间覆盖有限**：训练数据仅覆盖 954 种醛 x 1,415 种胺，对全新骨架的外推能力不确定
4. **单体 3D 描述符未使用**：消融实验表明单体 3D 描述符（10 维）对 PR-AUC 无显著提升，当前模型仅使用二聚体 3D 描述符（10 维）和 2D 拓扑信息
5. **MC Dropout 不确定性是近似的**：非贝叶斯推理，仅反映模型参数的局部敏感度
6. **规则向量与 GNN 的分工**：23 维规则向量编码化学硬约束（命中即不成膜），负责"排除不可能"；GNN 负责在可能范围内"判断好坏"。GNN 从数据中学到的软知识包括：
   - **分子拓扑隐式模式**：芳香环连接方式、取代基位置效应、共轭体系的电子效应等连续化学信息
   - **醛-胺配对兼容性**：CrossGraphAttention 学习哪些醛骨架与哪些胺骨架搭配效果好，超出简单的拓扑匹配
   - **取代基效应**：氟、甲基、甲氧基等取代基对成膜的细微影响（规则仅覆盖"过量氟取代"）
   - **灰色地带判断**：芳环数 4 个可能成膜也可能不成膜，GNN 学习这些连续边界
   - **文献隐含模式**：训练数据来自 357 篇文献，模型隐式学到了实验化学家的偏好和成功模式
7. **未校准**：输出的 sigmoid 概率未经过温度缩放或 Platt scaling 校准

---

## 环境要求

- Python 3.12.7
- PyTorch (CUDA 可选)
- PyTorch Geometric
- RDKit 2026.03.1

---

## 引用

```bibtex
@misc{fluorofilm_model_2026,
  title   = {V4Model: GNN-Based Film Formation Prediction for 2D Imine COF Monomer Pairs},
  author  = {},
  year    = {2026},
  note    = {0.70M parameters, GIN+GINE encoder + CrossGraphAttention + FilmHead with rule vector injection}
}
```
