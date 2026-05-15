---
name: chemical-priors-regularization
description: 化学先验转正则化的讨论结论——先验证规则准确性，再通过合成负样本+化学惩罚项注入训练
metadata:
  type: project
---

# 化学先验 → 正则化

## 当前状态

已实现四项化学先验规则（硬过滤/软降权，非训练中约束）：
- #0 苯环限制 (SMARTS c1ccccc1)
- #1 芳环数 ≤4
- #2 CanonicalRankAtoms 对称检测
- #4 C2 对位检查 (1,4-定位)

## 讨论结论

**先验证规则准确性，再考虑转为正则化。**

### 推荐方案：A + B 组合

**A) 合成负样本** — 自动生成违规配对 (label=0) 加入训练集，模型自己学到化学约束
**B) 化学惩罚项** — L_total = L_focal + λ·violation_score，类似 L2 正则化但惩罚违反化学定律

### 类比

```
L2:  L = L_data + λ‖w‖²           → 参数不要太大
Chem: L = L_data + λ·penalty(chem) → 预测不要违反化学定律
```
