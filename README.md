# FluoroFilm

基于机器学习的二维亚胺键 COF 单体筛选工具，聚焦含氟/非含氟单体对成膜性能的影响。

## 核心任务

1. 从 ~1000 篇 PDF 文献中提取结构化化学信息
2. 总结二维 COF 成膜关键控制因素理论
3. 基于 DeepChem 训练成膜预测模型
4. 筛选 Top 20 最佳含氟/非含氟亚胺键 COF 单体对
5. 产出 Word 可视化报告 + GraphRAG 知识图谱

## 技术栈

Python 3.8 · DeepChem 2.8 · RDKit · PyMuPDF · GROBID · MiniMax API · SQLite · scikit-learn

## 快速开始

```bash
# 1. 激活环境
conda activate dphuanjing

# 2. 安装新依赖
pip install -r requirements.txt

# 3. 验证
python -c "import deepchem; import rdkit; import pymupdf; print('OK')"
```

## 目录

```
data/        原始数据与处理产物
src/         核心代码（按模块划分）
notebooks/   分析笔记
tests/       测试
models/      DeepChem 模型
scripts/     运行脚本
```
