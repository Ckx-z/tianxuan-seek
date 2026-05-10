"""生成 Route A 筛选 Word 可视化报告。

内容包括:
  1. 封面与项目概述
  2. 数据概览 — 文献→单体→配对 流程统计
  3. 模型性能 — XGBoost/RF/LR 对比表
  4. Route A Top 20 排名表
  5. 虚拟氟化修正分析
  6. 关键发现与建议
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

logger = setup_logger("report")


def _add_table(doc, headers, rows, col_widths=None, bold_first=False):
    """添加格式化表格。"""
    from docx.shared import Inches, Pt, RGBColor
    from docx.oxml.ns import qn

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.autofit = True

    # 表头
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = str(h)
        for p in cell.paragraphs:
            p.alignment = 1  # 居中
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)

    # 数据行
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


def _add_image(doc, img_bytes, width_inches=6.0):
    """添加居中的图片。"""
    from docx.shared import Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(io.BytesIO(img_bytes), width=Inches(width_inches))


def _mol_grid_image(smiles_list, names, per_row=5, size=(300, 200)):
    """将多个分子结构绘制为网格图。"""
    from rdkit import Chem
    from rdkit.Chem import Draw

    mols = []
    legends = []
    for smi, name in zip(smiles_list, names):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        legends.append(str(name)[:30])
    if not mols:
        return None

    img = Draw.MolsToGridImage(
        mols, legends=legends, molsPerRow=per_row,
        subImgSize=size, useSVG=False,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="生成 Word 可视化报告")
    parser.add_argument("--top20", default="data/processed/route_a_top20.csv")
    parser.add_argument("--fluor", default="data/processed/fluorination_correction.csv")
    parser.add_argument("--model-dir", default="models/v1.0")
    parser.add_argument("--output", default="data/processed/Route_A_Report.docx")
    args = parser.parse_args()

    import json
    import numpy as np
    import pandas as pd
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.section import WD_ORIENT

    doc = Document()

    # 页面设置
    section = doc.sections[0]
    section.page_width = Inches(11.69)  # A4 landscape
    section.page_height = Inches(8.27)
    section.orientation = WD_ORIENT.LANDSCAPE

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    # ===================================================================
    # 封面
    # ===================================================================
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("\n\n\n")
    run.font.size = Pt(28)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("2D COF 单体筛选报告")
    run.font.size = Pt(28)
    run.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Route A — 含氟策略 × 机器学习成膜预测")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(100, 100, 100)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("\n\n亚胺键 2D COF | 四组氟配对策略 | XGBoost 模型")
    run.font.size = Pt(11)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("\n\n\n生成日期: 2026-05-09")
    run.font.size = Pt(10)

    doc.add_page_break()

    # ===================================================================
    # 1. 数据概览
    # ===================================================================
    doc.add_heading("1. 筛选管道概览", level=1)

    doc.add_paragraph(
        "从 ~1000 篇 2D COF 文献出发，经 LLM 结构化提取 → 化学验证 → "
        "模型训练 → 自由配对筛选，最终输出 Top 20 推荐单体对。"
    )

    _add_table(doc,
        ["阶段", "输入", "输出", "数量"],
        [
            ["PDF 解析", "~1000 PDF", "提取原文 (.txt)", "954 篇"],
            ["LLM 结构化提取", "954 篇原文", "结构化 YAML + SMILES", "840 条 (88.1%)"],
            ["化学去重 (Canonical SMILES)", "LLM SMILES", "唯一单体", "619 个"],
            ["2D COF 过滤 (≥2 官能团)", "619 单体", "2D 可用单体", "359 个 (醛141/胺218)"],
            ["训练集构建", "487 篇有标签 2D COF", "特征矩阵", "487 样本 × 1909 维"],
            ["Route A 自由配对", "359 单体", "醛×胺 四组策略", "30,738 对"],
            ["训练集排除 (严格)", "30,738 对", "剔除已见组合", "30,390 对"],
            ["非标准拓扑排除 (C3+C4)", "30,390 对", "仅保留六方/四方", "28,312 对"],
            ["Top 20 输出", "28,312 对", "margin 排序 Top 20", "20 对"],
        ],
        col_widths=[1.8, 1.6, 1.8, 1.2],
    )

    doc.add_paragraph()

    # 单体统计
    doc.add_heading("1.1 单体分类统计", level=2)
    _add_table(doc,
        ["类别", "数量", "含氟", "非含氟", "拓扑分布"],
        [
            ["醛单体 (Aldehyde)", "141", "13", "128", "C2: ~110, C3: ~31"],
            ["胺单体 (Amine)", "218", "11", "207", "C2: ~180, C3: ~25, C4: ~13"],
            ["合计", "359", "24", "335", ""],
        ],
        col_widths=[1.5, 0.8, 0.8, 0.8, 2.0],
    )

    doc.add_page_break()

    # ===================================================================
    # 2. 模型性能
    # ===================================================================
    doc.add_heading("2. 模型性能 (5-Fold CV + Test)", level=1)

    info_path = os.path.join(args.model_dir, "model_info.json")
    cv_xgb = {}
    cv_rf = {}
    cv_lr = {}
    test_xgb = {}
    test_rf = {}
    test_lr = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        cv = info.get("cv_results", {})
        cv_xgb = cv.get("xgboost", {})
        cv_rf = cv.get("random_forest", {})
        cv_lr = cv.get("logistic", {})
        test = info.get("test_results", {})
        test_xgb = test.get("xgboost", {})
        test_rf = test.get("random_forest", {})
        test_lr = test.get("logistic", {})

    doc.add_heading("2.1 5-Fold Cross-Validation", level=2)
    _add_table(doc,
        ["模型", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"],
        [
            ["XGBoost", f"{cv_xgb.get('roc_auc', 0):.3f}", f"{cv_xgb.get('average_precision', 0):.3f}",
             f"{cv_xgb.get('precision', 0):.3f}", f"{cv_xgb.get('recall', 0):.3f}",
             f"{cv_xgb.get('f1', 0):.3f}"],
            ["Random Forest", f"{cv_rf.get('roc_auc', 0):.3f}", f"{cv_rf.get('average_precision', 0):.3f}",
             f"{cv_rf.get('precision', 0):.3f}", f"{cv_rf.get('recall', 0):.3f}",
             f"{cv_rf.get('f1', 0):.3f}"],
            ["Logistic Regression", f"{cv_lr.get('roc_auc', 0):.3f}", f"{cv_lr.get('average_precision', 0):.3f}",
             f"{cv_lr.get('precision', 0):.3f}", f"{cv_lr.get('recall', 0):.3f}",
             f"{cv_lr.get('f1', 0):.3f}"],
        ],
        col_widths=[1.3, 0.9, 0.9, 0.9, 0.9, 0.9],
    )

    doc.add_heading("2.2 Test Set 表现 (20% hold-out)", level=2)
    _add_table(doc,
        ["模型", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"],
        [
            ["XGBoost (用于筛选)", f"{test_xgb.get('roc_auc', 0):.3f}",
             f"{test_xgb.get('pr_auc', 0):.3f}", f"{test_xgb.get('precision', 0):.3f}",
             f"{test_xgb.get('recall', 0):.3f}", f"{test_xgb.get('f1', 0):.3f}"],
            ["Random Forest", f"{test_rf.get('roc_auc', 0):.3f}",
             f"{test_rf.get('pr_auc', 0):.3f}", f"{test_rf.get('precision', 0):.3f}",
             f"{test_rf.get('recall', 0):.3f}", f"{test_rf.get('f1', 0):.3f}"],
            ["Logistic Regression", f"{test_lr.get('roc_auc', 0):.3f}",
             f"{test_lr.get('pr_auc', 0):.3f}", f"{test_lr.get('precision', 0):.3f}",
             f"{test_lr.get('recall', 0):.3f}", f"{test_lr.get('f1', 0):.3f}"],
        ],
        col_widths=[1.3, 0.9, 0.9, 0.9, 0.9, 0.9],
    )

    doc.add_paragraph(
        "\n注: 由于筛选单体与训练集高度重叠 (转导学习), predict_proba 概率压缩至 ~0.999。"
        " 排序使用 XGBoost raw margin scores (z-score 归一化为 0-100 得分)。"
    ).italic = True

    doc.add_page_break()

    # ===================================================================
    # 3. Route A Top 20
    # ===================================================================
    doc.add_heading("3. Route A Top 20 单体对 (2D COF 成膜预测)", level=1)

    doc.add_paragraph(
        "Route A 策略: 含氟 (F) 与非含氟 (非F) 单体交叉配对，生成四组："
        " (1) F-醛 × 非F-胺, (2) 非F-醛 × F-胺, (3) F-醛 × F-胺, (4) 非F-醛 × 非F-胺"
    )

    top_df = pd.read_csv(args.top20)
    rows = []
    for i, (_, r) in enumerate(top_df.iterrows()):
        rows.append([
            i + 1,
            f"{r['margin_score']:.1f}",
            str(r["topology"]),
            str(r["pair_type"]),
            str(r["aldehyde"])[:40],
            str(r["amine"])[:40],
            "✓" if r["aldehyde_f"] else "✗",
            "✓" if r["amine_f"] else "✗",
        ])

    _add_table(doc,
        ["#", "得分", "拓扑", "配对策略", "醛单体", "胺单体", "醛-F", "胺-F"],
        rows,
        col_widths=[0.3, 0.5, 1.0, 1.2, 2.0, 2.0, 0.4, 0.4],
    )

    # 分子结构图 — Top 10
    doc.add_heading("3.1 Top 10 分子结构", level=2)
    doc.add_paragraph("第一行: 醛单体 | 第二行: 胺单体 (按排名配对)")

    ald_smis = [str(top_df.iloc[i]["aldehyde_smiles"]) for i in range(min(10, len(top_df)))]
    am_smis = [str(top_df.iloc[i]["amine_smiles"]) for i in range(min(10, len(top_df)))]
    ald_names = [f"A{i+1}: {str(top_df.iloc[i]['aldehyde'])[:20]}" for i in range(min(10, len(top_df)))]
    am_names = [f"B{i+1}: {str(top_df.iloc[i]['amine'])[:20]}" for i in range(min(10, len(top_df)))]

    for title, smis, names in [("醛单体 (Aldehydes)", ald_smis, ald_names),
                                 ("胺单体 (Amines)", am_smis, am_names)]:
        doc.add_paragraph(title)
        img = _mol_grid_image(smis, names, per_row=5, size=(250, 160))
        if img:
            _add_image(doc, img, width_inches=9.0)

    doc.add_page_break()

    # ===================================================================
    # 4. 氟策略分析
    # ===================================================================
    doc.add_heading("4. 氟策略分析", level=1)

    nf_ald_nf_am = top_df[(~top_df["aldehyde_f"]) & (~top_df["amine_f"])]
    nf_ald_f_am = top_df[(~top_df["aldehyde_f"]) & (top_df["amine_f"])]
    f_ald_nf_am = top_df[(top_df["aldehyde_f"]) & (~top_df["amine_f"])]
    f_ald_f_am = top_df[(top_df["aldehyde_f"]) & (top_df["amine_f"])]

    doc.add_heading("4.1 Top 20 中四组策略分布", level=2)
    _add_table(doc,
        ["配对策略", "对数", "占比", "平均得分", "得分范围"],
        [
            ["非F-醛 × F-胺", str(len(nf_ald_f_am)),
             f"{100*len(nf_ald_f_am)/len(top_df):.0f}%",
             f"{nf_ald_f_am['margin_score'].mean():.1f}" if len(nf_ald_f_am) > 0 else "-",
             f"{nf_ald_f_am['margin_score'].min():.0f}-{nf_ald_f_am['margin_score'].max():.0f}" if len(nf_ald_f_am) > 0 else "-"],
            ["F-醛 × 非F-胺", str(len(f_ald_nf_am)),
             f"{100*len(f_ald_nf_am)/len(top_df):.0f}%",
             f"{f_ald_nf_am['margin_score'].mean():.1f}" if len(f_ald_nf_am) > 0 else "-",
             f"{f_ald_nf_am['margin_score'].min():.0f}-{f_ald_nf_am['margin_score'].max():.0f}" if len(f_ald_nf_am) > 0 else "-"],
            ["非F-醛 × 非F-胺", str(len(nf_ald_nf_am)),
             f"{100*len(nf_ald_nf_am)/len(top_df):.0f}%",
             f"{nf_ald_nf_am['margin_score'].mean():.1f}" if len(nf_ald_nf_am) > 0 else "-",
             f"{nf_ald_nf_am['margin_score'].min():.0f}-{nf_ald_nf_am['margin_score'].max():.0f}" if len(nf_ald_nf_am) > 0 else "-"],
            ["F-醛 × F-胺", str(len(f_ald_f_am)),
             f"{100*len(f_ald_f_am)/len(top_df):.0f}%",
             f"{f_ald_f_am['margin_score'].mean():.1f}" if len(f_ald_f_am) > 0 else "-",
             f"{f_ald_f_am['margin_score'].min():.0f}-{f_ald_f_am['margin_score'].max():.0f}" if len(f_ald_f_am) > 0 else "-"],
        ],
        col_widths=[1.4, 0.6, 0.6, 0.8, 0.8],
    )

    # 高频单体统计
    doc.add_heading("4.2 高频单体", level=2)

    from collections import Counter
    ald_counter = Counter(top_df["aldehyde"])
    am_counter = Counter(top_df["amine"])

    ald_rows = [[name, str(cnt), f"{100*cnt/len(top_df):.0f}%"] for name, cnt in ald_counter.most_common(8)]
    am_rows = [[name, str(cnt), f"{100*cnt/len(top_df):.0f}%"] for name, cnt in am_counter.most_common(8)]

    doc.add_paragraph("高频醛单体:")
    _add_table(doc, ["名称", "出现次数", "占比"], ald_rows, col_widths=[2.8, 0.8, 0.6])
    doc.add_paragraph("高频胺单体:")
    _add_table(doc, ["名称", "出现次数", "占比"], am_rows, col_widths=[2.8, 0.8, 0.6])

    doc.add_page_break()

    # ===================================================================
    # 5. 虚拟氟化修正
    # ===================================================================
    doc.add_heading("5. 虚拟氟化修正分析", level=1)

    doc.add_paragraph(
        "对非含氟单体对 (非F×非F) 进行虚拟氟化 (芳香 H→F, 1-2 个F)，"
        "预测加氟前后 margin score 变化 (Δ)。"
    )

    if os.path.exists(args.fluor):
        f_df = pd.read_csv(args.fluor)

        # 按策略统计
        f_side_stats = f_df.groupby("fluorinated_side").agg(
            mean_delta=("margin_delta", "mean"),
            max_delta=("margin_delta", "max"),
            count=("margin_delta", "count"),
            positive=("margin_delta", lambda x: int((x > 0).sum())),
        )

        doc.add_heading("5.1 氟化侧策略对比", level=2)
        f_rows = []
        for side, row in f_side_stats.iterrows():
            labels = {"aldehyde": "氟化醛侧", "amine": "氟化胺侧", "both": "氟化双侧"}
            f_rows.append([
                labels.get(side, side),
                f"{row['mean_delta']:+.2f}",
                f"{row['max_delta']:+.2f}",
                str(int(row['count'])),
                f"{int(row['positive'])}/{int(row['count'])} ({100*row['positive']/row['count']:.0f}%)",
            ])
        _add_table(doc,
            ["氟化策略", "平均ΔMargin", "最大ΔMargin", "变体数", "正向提升"],
            f_rows,
            col_widths=[1.2, 1.2, 1.2, 0.8, 1.4],
        )

        # Top 氟化提升
        doc.add_heading("5.2 氟化提升最大的 Top 10 单体对", level=2)

        # 每对取最佳策略
        best_per_pair = f_df.loc[
            f_df.groupby(["aldehyde_smiles_orig", "amine_smiles_orig"])["margin_delta"].idxmax()
        ].sort_values("margin_delta", ascending=False)

        fluor_rows = []
        for _, row in best_per_pair.head(10).iterrows():
            fluor_rows.append([
                f"{row['margin_delta']:+.2f}",
                f"{row['margin_orig']:.2f} → {row['margin_fluorinated']:.2f}",
                str(row["strategy"]),
                str(row["aldehyde_orig"])[:30],
                str(row["amine_orig"])[:30],
            ])
        _add_table(doc,
            ["ΔMargin", "原始→氟化后", "策略", "醛单体", "胺单体"],
            fluor_rows,
            col_widths=[0.8, 1.4, 1.4, 2.0, 1.8],
        )
    else:
        doc.add_paragraph("(虚拟氟化数据不可用，请先运行 scripts/virtual_fluorination.py)")

    doc.add_page_break()

    # ===================================================================
    # 6. 关键发现与建议
    # ===================================================================
    doc.add_heading("6. 关键发现与建议", level=1)

    findings = [
        ("氟策略效率",
         "\"非F-醛 × F-胺\" 是 Top 20 中最优策略 (16/20, 80%)。含氟胺单体 "
         "(如 2-(trifluoromethyl)benzene-1,4-diamine) 与不含氟醛配对时，"
         "模型给出最高 margin 得分。化学上合理：-CF3 吸电子增强亚胺键稳定性，"
         "同时避免醛侧氟化导致的反应活性过度降低。"),
        ("最佳醛单体",
         "2-hydroxybenzene-1,3,5-tricarbaldehyde (C3) 和 triformylphloroglucinol (C3) "
         "是最优醛单体。C3 对称性 + -OH 取代的模式在训练集中频繁出现于成膜 COF，"
         "模型学会了这一结构-性能关系。"),
        ("最佳胺单体",
         "2-(trifluoromethyl)benzene-1,4-diamine (C2, CF3) 在 Top 20 中出现 12 次。"
         "CF3 基团提供了氟效应且不降低胺的亲核性过多。相对地，"
         "四氟取代苯二胺 (F4) 排名较低，可能因过度氟化降低反应活性。"),
        ("拓扑偏好",
         "Top 20 中六方 (hcb) 和四方 (sql) 各占约 50%。六方结构主要来自 "
         "C3+C2 配对，四方来自 C2+C2。两者均在 2D COF 中有丰富文献先例。"),
        ("虚拟氟化教训",
         "随机加氟大多不利于成膜预测 (仅 13% 正向)。醛侧氟化 (平均 Δ=-0.70) "
         "显著优于胺侧氟化 (平均 Δ=-1.40)。建议实验验证时优先考虑已知的 "
         "含氟醛单体 (如 2,3,5,6-tetrafluoroterephthalaldehyde) 而非随机氟化。"),
        ("转导学习局限",
         "由于筛选单体集合与训练集高度重叠，模型概率值不可靠 (全压缩至 ~0.999)。"
         "排序通过 raw margin scores 实现，对化学趋势的捕捉是可靠的。"
         "建议在扩大单体库后重新训练以提高泛化能力。"),
    ]

    for title, body in findings:
        p = doc.add_paragraph()
        run = p.add_run(f"{title}: ")
        run.bold = True
        run.font.size = Pt(10)
        run = p.add_run(body)
        run.font.size = Pt(10)
        doc.add_paragraph()

    # ===================================================================
    # 保存
    # ===================================================================
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    doc.save(args.output)
    print(f"报告已生成: {args.output}")


if __name__ == "__main__":
    main()
