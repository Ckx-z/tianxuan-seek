"""生成 Route A Top 40 可视化 Word 报告。

用法:
  python scripts/generate_top40_report.py
  python scripts/generate_top40_report.py --input data/processed/route_a_gnn_top40.csv
"""
import argparse
import io
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger("top40_report")


def _mol_grid(smiles, names, mols_per_row=4, size=(400, 200)):
    mols = []
    legends = []
    for s, n in zip(smiles, names):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        mols.append(m)
        legends.append(n[:25])
    if not mols:
        return None
    return Draw.MolsToGridImage(mols, legends=legends,
                                molsPerRow=mols_per_row,
                                subImgSize=size)


def _add_image(doc, img, width=6.0):
    from docx.shared import Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    if img is None:
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    p.add_run().add_picture(buf, width=Inches(width))


def _add_table(doc, headers, rows, col_widths=None, font_size=8):
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
                    run.font.size = Pt(font_size)

    if col_widths:
        for i, w in enumerate(col_widths):
            for row_obj in table.rows:
                row_obj.cells[i].width = Inches(w)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/route_a_gnn_top40.csv")
    parser.add_argument("--full", default="data/processed/route_a_gnn_top40_full.csv")
    parser.add_argument("--output", default="data/processed/Top40_Report.docx")
    args = parser.parse_args()

    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn

    top = pd.read_csv(args.input, encoding="utf-8-sig")
    full = pd.read_csv(args.full, encoding="utf-8-sig") if os.path.exists(args.full) else None

    doc = Document()

    # 页边距
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # ── 封面 ──
    title = doc.add_heading("Route A Top 40 筛选报告", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("GNN + XGBoost 集成 · 七项化学硬规则 · C3 双模分层").font.size = Pt(12)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"模型: λ_chem=0.005 | 单体池: 272 可用 | 候选对: 18,432").font.size = Pt(10)
    doc.add_page_break()

    # ── 1. 数据概览 ──
    doc.add_heading("1. 数据概览", level=1)
    stats = [
        ["可用单体", "272 (醛=128, 胺=144)"],
        ["含氟单体", "F-醛=5, F-胺=6"],
        ["全量配对", "18,432 → 排除训练集 1,027 → 有效预测 18,342"],
        ["2D 拓扑分布", f"六方=7,702, 四方=10,069, 非标准=571"],
        ["高分歧 (>0.5)", f"{(full['divergence'] > 0.5).sum() if full is not None else 65} / 18,252"],
    ]
    _add_table(doc, ["指标", "值"], stats, col_widths=[2.5, 4.0])

    # ── 2. Top 40 排名表 ──
    doc.add_heading("2. Top 40 排名", level=1)

    headers = ["#", "综合分", "GNN", "XGB", "分歧", "醛", "胺", "拓扑", "氟策略"]
    rows = []
    for i, (_, r) in enumerate(top.iterrows()):
        ald_name = str(r["aldehyde"])[:28]
        am_name = str(r["amine"])[:28]
        rows.append([
            i + 1,
            f"{r['adjusted_score']:.3f}",
            f"{r['gnn_norm']:.3f}",
            f"{r['xgb_norm']:.3f}",
            f"{r['divergence']:.2f}",
            f"{ald_name} ({r['aldehyde_topo']})",
            f"{am_name} ({r['amine_topo']})",
            str(r["topology"]),
            str(r["pair_type"]),
        ])
    _add_table(doc, headers, rows, col_widths=[0.3, 0.55, 0.5, 0.5, 0.4, 2.0, 2.0, 0.7, 1.1],
               font_size=7)

    doc.add_page_break()

    # ── 3. 分子结构可视化 ──
    doc.add_heading("3. 分子结构可视化", level=1)

    # 提取去重单体
    ald_unique = top[["aldehyde_smiles", "aldehyde"]].drop_duplicates("aldehyde_smiles")
    am_unique = top[["amine_smiles", "amine"]].drop_duplicates("amine_smiles")

    doc.add_heading("3.1 醛单体", level=2)
    ald_smis = ald_unique["aldehyde_smiles"].tolist()
    ald_names = ald_unique["aldehyde"].tolist()
    for start in range(0, len(ald_smis), 8):
        img = _mol_grid(ald_smis[start:start+8], ald_names[start:start+8],
                        mols_per_row=4)
        if img:
            _add_image(doc, img, width=6.0)

    doc.add_heading("3.2 胺单体", level=2)
    am_smis = am_unique["amine_smiles"].tolist()
    am_names = am_unique["amine"].tolist()
    for start in range(0, len(am_smis), 8):
        img = _mol_grid(am_smis[start:start+8], am_names[start:start+8],
                        mols_per_row=4)
        if img:
            _add_image(doc, img, width=6.0)

    doc.add_page_break()

    # ── 4. 统计分析 ──
    doc.add_heading("4. 统计分析", level=1)

    # 4.1 拓扑分布
    doc.add_heading("4.1 拓扑分布 (Top 40)", level=2)
    topo_counts = top["topology"].value_counts()
    _add_table(doc, ["拓扑", "数量", "占比"],
               [[t, c, f"{c/40*100:.0f}%"] for t, c in topo_counts.items()],
               col_widths=[2.0, 1.0, 1.0])

    # 4.2 氟策略
    doc.add_heading("4.2 氟策略分布", level=2)
    f_counts = top["pair_type"].value_counts()
    _add_table(doc, ["氟策略", "数量", "占比"],
               [[t, c, f"{c/40*100:.0f}%"] for t, c in f_counts.items()],
               col_widths=[2.5, 1.0, 1.0])

    # 4.3 C3 占比
    doc.add_heading("4.3 C3 分层占比", level=2)
    c3_am = (top["amine_topo"] == "C3").sum()
    c3_ald = (top["aldehyde_topo"] == "C3").sum()
    _add_table(doc, ["指标", "数量", "目标"],
               [["C3-胺 (大胺小醛)", f"{c3_am}/40 ({c3_am*100/40:.0f}%)", "35%"],
                ["C3-醛 (大醛小胺)", f"{c3_ald}/40 ({c3_ald*100/40:.0f}%)", "25%"]],
               col_widths=[2.5, 2.0, 1.0])

    # 4.4 高频单体
    doc.add_heading("4.4 高频出现单体", level=2)
    ald_freq = Counter(top["aldehyde"])
    am_freq = Counter(top["amine"])
    freq_rows = []
    for name, cnt in ald_freq.most_common(5):
        freq_rows.append([name[:50], "醛", cnt])
    for name, cnt in am_freq.most_common(5):
        freq_rows.append([name[:50], "胺", cnt])
    _add_table(doc, ["单体名称", "类型", "出现次数"],
               freq_rows, col_widths=[3.5, 0.8, 1.0])

    # 4.5 分数分布
    doc.add_heading("4.5 分数分布", level=2)
    scores = top["adjusted_score"]
    _add_table(doc, ["统计量", "值"],
               [["均值", f"{scores.mean():.3f}"],
                ["中位数", f"{scores.median():.3f}"],
                ["最小值", f"{scores.min():.3f}"],
                ["最大值", f"{scores.max():.3f}"],
                ["标准差", f"{scores.std():.3f}"]],
               col_widths=[2.0, 2.0])

    # ── 5. 硬规则过滤统计 ──
    doc.add_heading("5. 化学硬规则过滤", level=1)
    _add_table(doc, ["规则", "过滤数", "说明"],
               [["#0 苯环必需", "94", "无苯环单体排除"],
                ["#1 芳环≤4", "146", "空间位阻约束"],
                ["#2 对称性", "344", "C2/C3 官能团对称"],
                ["#3 杂环降权", f"{(full['ald_has_heterocycle'] | full['am_has_heterocycle']).sum() if full is not None else 6736}", "×0.85 软惩罚"],
                ["#4 C2 对位", "53", "同环 1,4-位约束"],
                ["#5 炔丙基醚", "1", "醚+炔共存排除"],
                ["#6 C2 取代基", "9", ">4取代限卤素"],
                ["#7 官能团", "652", "≥2醛/胺基"],
                ["最终可用", "272", "过滤后单体数"]],
               col_widths=[1.5, 1.0, 3.0])

    doc.add_page_break()

    # ── 6. 六方 Top 10 结构 ──
    doc.add_heading("6. 六方 (hcb) Top 10 配对结构", level=1)
    hex_top = top[top["topology"].str.startswith("六方")].head(10)
    for i, (_, r) in enumerate(hex_top.iterrows()):
        ald_mol = Chem.MolFromSmiles(r["aldehyde_smiles"])
        am_mol = Chem.MolFromSmiles(r["amine_smiles"])
        if ald_mol and am_mol:
            img = Draw.MolsToGridImage(
                [ald_mol, am_mol],
                legends=[f"醛: {str(r['aldehyde'])[:30]}",
                         f"胺: {str(r['amine'])[:30]}"],
                molsPerRow=2, subImgSize=(400, 180))
            p = doc.add_paragraph()
            p.add_run(f"#{i+1} 综合分={r['adjusted_score']:.3f} | "
                      f"GNN={r['gnn_norm']:.3f} XGB={r['xgb_norm']:.3f}").bold = True
            _add_image(doc, img, width=5.5)

    # ── 7. 方法说明 ──
    doc.add_heading("7. 方法说明", level=1)
    methods = [
        ["模型架构", "GNN编码器 (v4 预训练, 256维) + BilinearHead (26维机理描述符)"],
        ["训练策略", "λ_chem=0.005 化学正则化, Focal Loss (α=0.75, γ=2.0)"],
        ["集成", "GNN 60% + XGBoost 40% − 0.10×分歧惩罚"],
        ["C3 分层", "大胺小醛 (C3-胺) 35% + 大醛小胺 (C3-醛) 25% + 其余 40%"],
        ["硬规则", "7 项化学约束 (苯环/芳环/对称/杂环/C2对位/炔丙基/取代基)"],
        ["去重", "InChI Key 配对去重, 保留最高分"],
    ]
    _add_table(doc, ["组件", "说明"], methods, col_widths=[1.5, 5.0])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    doc.save(args.output)
    logger.info(f"报告已保存: {args.output}")


if __name__ == "__main__":
    main()
