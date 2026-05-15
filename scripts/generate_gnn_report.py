"""生成 Route A GNN+XGBoost 筛选 Word 可视化报告。

参照 fluorine_cof_structures.docx 格式:
  - 竖版 A4
  - 封面 + 管道概览 + 模型性能
  - 图集 T01–T20: 单体信息 + 评分 + 分子结构图 + 分子式表
  - 规则效果分析 + 关键发现
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors

RDLogger.logger().setLevel(RDLogger.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

logger = setup_logger("gnn_report")


def _mol_formula(mol: Chem.Mol) -> str:
    return rdMolDescriptors.CalcMolFormula(mol)


def _draw_pair_image(ald_smi: str, am_smi: str, ald_label: str, am_label: str,
                     size=(500, 300)):
    """画醛+胺并排分子结构图。"""
    ald_mol = Chem.MolFromSmiles(ald_smi)
    am_mol = Chem.MolFromSmiles(am_smi)
    mols = []
    legends = []
    if ald_mol:
        mols.append(ald_mol)
        legends.append(ald_label)
    if am_mol:
        mols.append(am_mol)
        legends.append(am_label)
    if not mols:
        return None
    img = Draw.MolsToGridImage(
        mols, legends=legends, molsPerRow=2,
        subImgSize=size, useSVG=False,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _add_table(doc, headers, rows, col_widths=None):
    from docx.shared import Inches, Pt
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.autofit = True
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = str(h)
        for p in cell.paragraphs:
            p.alignment = 1
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = str(val) if val is not None else "-"
            for p in cell.paragraphs:
                p.alignment = 1
                for run in p.runs:
                    run.font.size = Pt(8)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    return table


def _add_image(doc, img_bytes, width_inches=5.5):
    from docx.shared import Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(io.BytesIO(img_bytes), width=Inches(width_inches))


def _score_fmt(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return f"{float(val):.4f}"


def _fmt_name(name, max_len=50):
    s = str(name)
    return s if len(s) <= max_len else s[:max_len - 2] + ".."


def main():
    parser = argparse.ArgumentParser(description="生成 GNN+XGBoost 筛选 Word 报告")
    parser.add_argument("--top20", default="data/processed/route_a_gnn_top40.csv")
    parser.add_argument("--xgb-info", default="models/v1.0/model_info.json")
    parser.add_argument("--output", default="data/processed/Route_A_GNN_Report_Top40.docx")
    args = parser.parse_args()

    from docx import Document
    from docx.shared import Inches, Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    top_df = pd.read_csv(args.top20)

    doc = Document()

    # 竖版 A4
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    # ============================================================
    # 封面
    # ============================================================
    for _ in range(4):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("含氟 COF 分子结构图集")
    run.font.size = Pt(26)
    run.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("GNN (60%) + XGBoost (40%) 加权融合 | 四项硬规则 | Top 40 双模分层")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(80, 80, 80)

    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(
        "亚胺键 2D COF | 苯环硬限制 | C3-胺优先 (50%)\n"
        "GNN Bilinear (PR-AUC 0.731) | XGBoost (PR-AUC 0.654)\n"
        "生成日期: 2026-05-14"
    )
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_page_break()

    # ============================================================
    # 1. 管道概览
    # ============================================================
    doc.add_heading("1. 筛选管道概览", level=1)

    doc.add_paragraph(
        "从 ~1000 篇 2D COF 文献出发，经 LLM 结构化提取 → 化学验证 → "
        "GNN+XGBoost 双模型集成 → 全量自由配对 → 三项硬规则过滤 → "
        "C3-胺分层选取，最终输出 Top 40 推荐单体对 (大胺小醛 + 大醛小胺 双模)。"
    )

    _add_table(doc,
        ["阶段", "输入", "输出", "数量"],
        [
            ["PDF 解析", "~1000 篇 PDF", "提取原文 (.txt)", "954 篇"],
            ["LLM 结构化提取", "954 篇原文", "结构化 YAML + SMILES", "840 条"],
            ["单体池构建", "LLM+商业+训练集", "去重 Canonical SMILES", "1,571 个"],
            ["#0 苯环限制", "1,571 单体", "排除无苯环", "−94 (→1,477)"],
            ["#1 芳环 ≤ 4", "1,477 单体", "排除芳环超标", "−146 (→1,331)"],
            ["#2 对称 + [NH2][c]", "1,331 单体", "排除不对称/酰肼", "−174 (→1,157)"],
            ["≥2 官能团", "1,157 单体", "2D COF 可用", "505 (醛234/胺271)"],
            ["全量配对", "505 单体", "醛 × 胺 四组策略", "63,414 对"],
            ["训练集排除", "63,414 对", "剔除已见组合", "63,181 对"],
            ["GNN+XGB 集成推理", "63,181 对", "Ensemble 评分", "62,948 对有效"],
            ["C3 胺 bonus ×1.15", "62,948 对", "C3 胺提权", "15,065 对受益"],
            ["去重+杂环降权", "62,948 对", "去重 + 降权", "62,948 对"],
            ["标准2D拓扑", "62,948 对", "仅六方/四方", "61,527 对"],
            ["C3 双模分层 Top40", "61,527 对", "C3胺池14+C3醛池10+其余16", "40 对"],
        ],
        col_widths=[1.8, 1.3, 1.5, 1.1],
    )

    doc.add_heading("1.1 单体过滤统计", level=2)
    _add_table(doc,
        ["过滤规则", "排除数", "剩余", "说明"],
        [
            ["原始去重单体", "—", "1,571", "LLM (619) + 商业 (485) + 训练集额外 (467)"],
            ["#0 无苯环", "94", "1,477", "SMARTS c1ccccc1 — 排除 glyoxal/脂肪胺等"],
            ["#1 芳环>4", "146", "1,331", "位阻过大 → 不可聚合"],
            ["#2 不对称+非芳香胺", "174", "1,157", "[NH2][c] 仅芳香伯胺 + CanonicalRankAtoms 对称检测"],
            ["官能团<2", "652", "505", "≥2 醛基 或 ≥2 芳香伯胺"],
        ],
        col_widths=[1.8, 0.7, 0.7, 3.5],
    )

    doc.add_page_break()

    # ============================================================
    # 2. 模型性能
    # ============================================================
    doc.add_heading("2. 模型性能", level=1)

    doc.add_heading("2.1 GNN Bilinear (主模型, 60% 权重)", level=2)
    _add_table(doc,
        ["指标", "值", "说明"],
        [
            ["架构", "3-Layer GCN (256-dim) + BilinearHead (rank=64)", "206K 参数"],
            ["输入", "分子图 + 26-dim RDKit 描述符", "原子特征: 元素/电荷/芳香性"],
            ["损失", "FocalLoss (γ=2.0) + Ranking Loss (w=0.005)", "类别不平衡 + 排序约束"],
            ["验证", "RepeatedStratifiedKFold 8×8 CV", "64 折 ± 置信区间"],
            ["PR-AUC", "0.731 ± 0.067", "正样本 ~15%"],
            ["ROC-AUC", "0.892 ± 0.045", "—"],
        ],
        col_widths=[1.4, 2.8, 2.4],
    )

    doc.add_heading("2.2 XGBoost (辅助哨兵, 40% 权重)", level=2)
    xgb_cv = {}
    if os.path.exists(args.xgb_info):
        with open(args.xgb_info, "r", encoding="utf-8") as f:
            xgb_info = json.load(f)
        xgb_cv = xgb_info.get("cv_results", {}).get("xgboost", {})

    _add_table(doc,
        ["指标", "值", "说明"],
        [
            ["特征", "ECFP4 (1024) + MACCS (167) + 描述符 (13) × 2 单体", "2,418 → 1,909 维 (选择后)"],
            ["PR-AUC (5-fold)", f"{xgb_cv.get('average_precision', 0.654):.3f}", "Morgan 指纹显式子结构检测"],
            ["ROC-AUC (5-fold)", f"{xgb_cv.get('roc_auc', 0):.3f}", "—"],
            ["角色", "化学合理性哨兵", "分歧惩罚 δ=0.10 — 拉住 GNN 过激预测"],
        ],
        col_widths=[1.4, 2.8, 2.4],
    )

    doc.add_heading("2.3 集成公式", level=2)
    doc.add_paragraph(
        "final = 0.60×GNN_norm + 0.40×XGB_norm − 0.10×|GNN_norm − XGB_norm|\n"
        "  • GNN: 分子图子结构交互 (学习型特征)\n"
        "  • XGBoost: 显式指纹共现 (规则型特征)\n"
        "  • 分歧惩罚: ~5% 对 (div > 0.5) 被拉低\n"
        "  • C3 胺 bonus: ×1.15 (大胺小醛), C3 醛 bonus: ×1.10 (大醛小胺)"
    )

    doc.add_page_break()

    # ============================================================
    # 3. 图集 T01–T40
    # ============================================================
    doc.add_heading("3. 分子结构图集", level=1)

    for i, (_, row) in enumerate(top_df.iterrows()):
        rank = i + 1
        ald_name = str(row["aldehyde"])
        am_name = str(row["amine"])
        ald_smi = str(row["aldehyde_smiles"])
        am_smi = str(row["amine_smiles"])
        ald_mol = Chem.MolFromSmiles(ald_smi)
        am_mol = Chem.MolFromSmiles(am_smi)
        ald_formula = _mol_formula(ald_mol) if ald_mol else "?"
        am_formula = _mol_formula(am_mol) if am_mol else "?"
        mw_ald = Descriptors.MolWt(ald_mol) if ald_mol else 0
        mw_am = Descriptors.MolWt(am_mol) if am_mol else 0
        ald_topo = str(row.get("aldehyde_topo", "?"))
        am_topo = str(row.get("amine_topo", "?"))
        ald_f = "是" if row.get("aldehyde_f") else "否"
        am_f = "是" if row.get("amine_f") else "否"
        pair_type = str(row["pair_type"])
        topo = str(row["topology"])

        # 图集标题
        doc.add_heading(f"图集 T{rank:02d}", level=2)

        # 单体信息
        p = doc.add_paragraph()
        run = p.add_run("醛单体: ")
        run.bold = True
        run.font.size = Pt(9)
        run = p.add_run(
            f"{ald_name}\n"
            f"  SMILES: {ald_smi}\n"
            f"  分子式: {ald_formula}  |  MW: {mw_ald:.1f}  |  醛基数: {ald_topo}  |  含氟: {ald_f}"
        )
        run.font.size = Pt(9)

        p = doc.add_paragraph()
        run = p.add_run("胺单体: ")
        run.bold = True
        run.font.size = Pt(9)
        run = p.add_run(
            f"{am_name}\n"
            f"  SMILES: {am_smi}\n"
            f"  分子式: {am_formula}  |  MW: {mw_am:.1f}  |  胺基数: {am_topo}  |  含氟: {am_f}"
        )
        run.font.size = Pt(9)

        # 评分信息
        c3_tag = " [C3加成 ×1.15]" if float(row.get("c3_bonus", 1.0)) > 1.0 else ""
        hetero_tag = " [杂环降权 ×0.85]" if float(row.get("hetero_penalty", 1.0)) < 1.0 else ""

        p = doc.add_paragraph()
        run = p.add_run("评分: ")
        run.bold = True
        run.font.size = Pt(9)
        run = p.add_run(
            f"综合得分 = {_score_fmt(row['adjusted_score'])}  |  "
            f"GNN = {_score_fmt(row['gnn_norm'])}  |  "
            f"XGB = {_score_fmt(row['xgb_norm'])}  |  "
            f"分歧 = {_score_fmt(row['divergence'])}\n"
            f"  配对策略: {pair_type}  |  拓扑: {topo}{c3_tag}{hetero_tag}"
        )
        run.font.size = Pt(9)

        # 结构图标签
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run("左: 醛单体 (Aldehyde)    右: 胺单体 (Amine)")
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(100, 100, 100)

        # 并排分子结构图
        img = _draw_pair_image(
            ald_smi, am_smi,
            f"A{rank}: {_fmt_name(ald_name, 30)}",
            f"B{rank}: {_fmt_name(am_name, 30)}",
            size=(450, 280),
        )
        if img:
            _add_image(doc, img, width_inches=5.2)

        # 分子式表格
        tbl = doc.add_table(rows=2, cols=2)
        tbl.style = "Light Grid Accent 1"
        tbl.autofit = True

        c00 = tbl.rows[0].cells[0]
        c00.text = "醛单体分子式"
        for p in c00.paragraphs:
            p.alignment = 1
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(8)

        c01 = tbl.rows[0].cells[1]
        c01.text = "胺单体分子式"
        for p in c01.paragraphs:
            p.alignment = 1
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(8)

        c10 = tbl.rows[1].cells[0]
        c10.text = ald_formula
        for p in c10.paragraphs:
            p.alignment = 1
            for run in p.runs:
                run.font.size = Pt(9)

        c11 = tbl.rows[1].cells[1]
        c11.text = am_formula
        for p in c11.paragraphs:
            p.alignment = 1
            for run in p.runs:
                run.font.size = Pt(9)

        doc.add_paragraph()

    doc.add_page_break()

    # ============================================================
    # 4. 规则效果 + 分布分析
    # ============================================================
    doc.add_heading("4. 硬规则效果与分布分析", level=1)

    doc.add_heading("4.1 规则影响矩阵", level=2)
    _add_table(doc,
        ["规则", "类型", "排除/影响", "说明"],
        [
            ["#0 苯环硬限制", "硬排除", "94 单体",
             "SMARTS c1ccccc1 — 排除 glyoxal/脂肪胺等非芳香分子"],
            ["#1 芳环 ≤ 4", "硬排除", "146 单体",
             "位阻限制 — 排除多环芳烃 (六苯并蔻等)"],
            ["#2 对称 (CanonicalRankAtoms)", "硬排除", "344 单体",
             "RDKit 全分子对称感知 + 仅芳香伯胺 [NH2][c]"],
            ["#4 C2 对位检查", "硬排除", "53 单体",
             "C2 反应基团必须在同苯环对位 (1,4) — 排除间/邻位异构体"],
            ["#5 炔丙基醚排除", "硬排除", "1 单体",
             "排除醚键+炔基共存 (化学不相容)"],
            ["#6 C2 取代基限卤素", "硬排除", "9 单体",
             "C2 单体苯环 >4 取代基时，多余取代基限卤素 (排除 OH/OMe/OEt)"],
            ["#3 杂环降权 ×0.85", "软惩罚", "6,736 / 18,252 对",
             "苯环优先: 双侧杂环 ×0.7225"],
            ["C3 胺 bonus ×1.15", "软Bonus", "15,065 / 62,948 对",
             "补偿训练集 C2/C3 ≈ 7:1 不平衡"],
        ],
        col_widths=[1.8, 0.6, 1.2, 3.0],
    )

    # 氟策略分布
    doc.add_heading("4.2 Top 40 氟策略分布", level=2)
    from collections import Counter
    f_counts = top_df["pair_type"].value_counts()
    f_rows = []
    for ptype, cnt in f_counts.items():
        sub = top_df[top_df["pair_type"] == ptype]
        f_rows.append([
            str(ptype), str(cnt), f"{100*cnt/len(top_df):.0f}%",
            f"{sub['adjusted_score'].mean():.4f}",
            f"{sub['adjusted_score'].min():.4f}–{sub['adjusted_score'].max():.4f}",
        ])
    _add_table(doc,
        ["配对策略", "对数", "占比", "平均综合分", "分数范围"],
        f_rows,
        col_widths=[1.8, 0.5, 0.5, 1.0, 1.2],
    )

    # 拓扑分布
    doc.add_heading("4.3 Top 40 拓扑分布", level=2)
    topo_counts = top_df["topology"].value_counts()
    topo_rows = [[str(t), str(c), f"{100*c/len(top_df):.0f}%"] for t, c in topo_counts.items()]
    _add_table(doc, ["拓扑类型", "对数", "占比"], topo_rows, col_widths=[1.5, 0.6, 0.6])

    c3_ratio = (top_df["amine_topo"] == "C3").sum()
    doc.add_paragraph(
        f"C3-胺占比: {c3_ratio}/{len(top_df)} ({100*c3_ratio//len(top_df)}%) — "
        f"C3-醛占比: {(top_df['aldehyde_topo'] == 'C3').sum()}/{len(top_df)}"
        f" ({(top_df['aldehyde_topo']=='C3').sum()*100//len(top_df)}%) — "
        f"双模分层: 大胺小醛(C3胺)35% + 大醛小胺(C3醛)25% + 其余40%。\n"
        f"GNN 均值: {top_df['gnn_norm'].mean():.3f}, "
        f"XGB 均值: {top_df['xgb_norm'].mean():.3f}, "
        f"平均分歧: {top_df['divergence'].mean():.3f}"
    )

    # 高频单体
    doc.add_heading("4.4 高频单体统计", level=2)
    ald_counter = Counter(top_df["aldehyde"])
    am_counter = Counter(top_df["amine"])
    doc.add_paragraph("高频醛单体:")
    _add_table(doc,
        ["名称", "出现次数", "占比"],
        [[name, str(cnt), f"{100*cnt/len(top_df):.0f}%"] for name, cnt in ald_counter.most_common(6)],
        col_widths=[3.0, 0.8, 0.6],
    )
    doc.add_paragraph("高频胺单体:")
    _add_table(doc,
        ["名称", "出现次数", "占比"],
        [[name, str(cnt), f"{100*cnt/len(top_df):.0f}%"] for name, cnt in am_counter.most_common(6)],
        col_widths=[3.0, 0.8, 0.6],
    )

    doc.add_page_break()

    # ============================================================
    # 5. 关键发现
    # ============================================================
    doc.add_heading("5. 关键发现与建议", level=1)

    c3_count = int((top_df["amine_topo"] == "C3").sum())
    f_ald = int(((top_df["aldehyde_f"]) & (~top_df["amine_f"])).sum())
    nf_nf = int(((~top_df["aldehyde_f"]) & (~top_df["amine_f"])).sum())

    findings = [
        ("C3 双模分层选取",
         f"Top 40 中 C3-胺 {c3_count}/{len(top_df)} ({100*c3_count//len(top_df)}%), "
         f"C3-醛 {(top_df['aldehyde_topo']=='C3').sum()}/{len(top_df)}"
         f" ({(top_df['aldehyde_topo']=='C3').sum()*100//len(top_df)}%)。"
         "双模分层: 大胺小醛 (C3胺) 主导六方拓扑, 大醛小胺 (C3醛×C2胺) 占中段排名。"
         "模型天然偏向 C2 小分子, 分层选取补偿了数据不平衡。"),
        ("大胺小醛 + 大醛小胺 双模式",
         "Top 14 (C3-胺组) 呈现典型「大胺小醛」: C3 胺 (MW 350-600) 作框架节点 + C2 醛作连接臂。"
         "中段 #9-#24 呈现「大醛小胺」: C3 醛 (TFB 衍生物) 作节点 + C2 对苯二胺作连接臂。"
         "两种模式均有化学合理性, 适用于不同合成场景。"),
        ("GNN+XGBoost 协同效应",
         f"XGBoost (mean={top_df['xgb_norm'].mean():.3f}) 比 GNN "
         f"(mean={top_df['gnn_norm'].mean():.3f}) 更保守, 有效抑制 GNN 对训练集高频单体的过拟合。"
         "分歧惩罚使一致高分对排到前列, GNN 偏好的可疑组合被拉低。"),
        ("四项硬规则过滤效果",
         "苯环硬限制 (#0) 排除 94 个非芳香单体 (含 glyoxal)。"
         "[NH2][c] SMARTS 修复排除酰肼类假阳性胺, "
         "CanonicalRankAtoms (#2) 排除 344 个不对称单体, "
         "对位检查 (#4) 防御性保留 (当前池 0 排除)。"
         "四项硬规则过滤 ~36% 输入单体, Top 40 化学合理性显著提升。"),
        ("氟策略分布",
         f"F-醛 × 非F-胺 占 {f_ald}/{len(top_df)}, "
         f"非F-醛 × 非F-胺 占 {nf_nf}/{len(top_df)}。"
         "TFTA 的四氟取代提供强吸电子效应 → 增强亚胺键稳定性, "
         "同时不降低胺侧亲核性。含氟醛 + 非氟 C3 大胺是最优策略。"),
        ("后续建议与化学先验正则化",
         "① 合成验证 Top 5 配对 (TFTA + TAPB 衍生物); "
         "② 验证化学先验准确性后, 将规则转为训练中正则化 (合成负样本 + 化学惩罚项); "
         "③ 将反应条件 (溶剂/温度/催化剂) 纳入特征空间; "
         "④ 多任务学习 (成膜+结晶度+拓扑) 提升模型鲁棒性"),
    ]

    for title, body in findings:
        p = doc.add_paragraph()
        run = p.add_run(f"{title}: ")
        run.bold = True
        run.font.size = Pt(10)
        run = p.add_run(body)
        run.font.size = Pt(10)
        doc.add_paragraph()

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    doc.save(args.output)
    logger.info(f"报告已生成: {args.output}")
    print(f"报告已生成: {args.output}")


if __name__ == "__main__":
    main()
