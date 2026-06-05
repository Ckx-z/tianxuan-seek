"""v4 Top 40 筛选报告 — Word 文档,适配新笛卡尔积管线。

输入 : data/processed/v4_top40_candidates.csv  (41 配对 = Top 40 + target)
       data/processed/v4_screening_v2_soft.csv  (232,740 配对, 全量统计)
输出 : data/processed/v4_Top40_Final_Report.docx
       + v4_top40_report_images/*.png  (结构图)

字段适配 (本管线):
  aldehyde_name / amine_name     (原脚本同)
  ald_n / am_n                  (本管线用 ald_n/am_n, 旧版用 n_aldehyde/n_amine)
  ald_has_f / am_has_f          (本管线用 ald_*, 旧版用 ald_has_f/amine_has_f)
  ald_source / am_source
  film_prob_mean / film_prob_std / film_prob_adjusted
  diverse_rank

章节:
  1. 封面
  2. 项目背景 (v4 Route B 简述)
  3. 筛选管线 (笛卡尔积 → GNN → chem_filt → 多样性)
  4. 统计摘要
  5. Top 40 + target 汇总表
  6. 候选详情 (41 张结构图卡片)
  7. 已知局限
"""
from __future__ import annotations

import os
import sys
import csv
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)

TARGET_ALD_CANON = "O=Cc1cc(-c2ccccc2)c(C=O)cc1-c1ccccc1"
TARGET_AM_CANON = "Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1"

OUTPUT_DIR = "data/processed/v4_top40_report_images"


def draw_mol(smiles: str, filename: str, size: tuple = (400, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    img = Draw.MolToImage(mol, size=size)
    img.save(path)
    return path


def set_cell_shading(cell, color: str):
    shading = cell._element.get_or_add_tcPr()
    shd = shading.makeelement(qn('w:shd'), {
        qn('w:fill'): color,
        qn('w:val'): 'clear',
    })
    shading.append(shd)


def add_colored_run(paragraph, text: str, bold: bool = False,
                    color: RGBColor | None = None, size: Pt | None = None):
    run = paragraph.add_run(text)
    run.bold = bold
    if color:
        run.font.color.rgb = color
    if size:
        run.font.size = size
    return run


def is_target(r: dict) -> bool:
    return (r["aldehyde_smiles"] == TARGET_ALD_CANON
            and r["amine_smiles"] == TARGET_AM_CANON)


def topology(n_ald: int, n_am: int) -> str:
    if n_ald < 2 or n_am < 2:
        return "non"
    if n_ald >= 4 or n_am >= 4:
        return "other"
    if (n_ald == 2 and n_am == 3) or (n_ald == 3 and n_am == 2):
        return "hex"
    if n_ald == 3 and n_am == 3:
        return "hex"
    if n_ald == 2 and n_am == 2:
        return "tet"
    return "other"


def generate(top_csv: str, full_csv: str, output_path: str):
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)

    with open(top_csv, "r", encoding="utf-8") as f:
        top_rows = list(csv.DictReader(f))
    with open(full_csv, "r", encoding="utf-8") as f:
        full_rows = list(csv.DictReader(f))

    n_top = sum(1 for r in top_rows if not is_target(r))
    n_target = sum(1 for r in top_rows if is_target(r))

    title = doc.add_heading("v4 GNN 成膜预测 — Top 40 筛选报告", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_colored_run(subtitle, "亚胺键 2D COF 单体筛选 — 笛卡尔积 + Route B", bold=True,
                    size=Pt(14), color=RGBColor(0x20, 0x60, 0xA0))

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run(
        f"模型: V4Model (GIN+GINE x3 + Cross-Graph Attention + FilmHead, 0.69M)\n"
        f"训练: 544 正样本 + 49 文献负样本 + 1968 化学规则负样本 (15-fold PR-AUC 0.78)\n"
        f"筛选池: 433 醛 × 538 胺 = 232,954 → 排除 4 自配对 + 210 训练集配对 = 232,740 笛卡尔积\n"
        f"Top 40 + target 配对 = {n_top + n_target} 候选 (Morgan Tanimoto<0.8)\n"
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    doc.add_page_break()

    doc.add_heading("1. 项目背景", level=1)
    doc.add_paragraph(
        "本项目从 ~840 篇 PDF 文献中提取结构化化学信息,针对二维亚胺键 COF "
        "(共价有机框架) 进行机器学习驱动的单体筛选。当前阶段为 Phase 6 — "
        "用 GNN (图神经网络) 替代手写特征,让模型自动从分子图学习成膜相关的"
        "子结构模式。"
    )
    doc.add_paragraph(
        "v4 Route B 核心策略: PU Learning — 丢弃全阴性文献 (597/656 篇从未尝试成膜),"
        "用化学规则生成确定性负样本 (基础 4 策略 + 边界 5 策略 = 1500 负样本)。"
        "数据配比: 544 正样本 + 49 文献负样本 + 1500 化学负样本 = 2093 总样本。"
    )

    doc.add_heading("2. 筛选管线", level=1)

    doc.add_heading("2.1 单体池合并", level=2)
    doc.add_paragraph(
        "醛池 = 训练集醛 (n_ald>=1) ∪ merged_monomer_pool 醛 (n_ald>=2) ∪ target 醛\n"
        "胺池 = 训练集胺 (n_am>=1) ∪ merged_monomer_pool 胺 (n_am>=2) ∪ target 胺\n"
        "  训练集醛/胺: 来自 v4_train_3d_dimer.csv (增广前), 强制进池\n"
        "  商业+LLM 池: n_ald>=2 / n_am>=2 严格 (chem_penalty 兜底)\n"
        "  target 醛: O=Cc1cc(-c2ccccc2)c(C=O)cc1-c1ccccc1 (无氟 3,3'-二苯基-[1,1'-联苯]-4,4'-二甲醛)\n"
        "  target 胺: Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1 (TAPB)\n"
        "  合并去重 (canonical SMILES): 醛 433, 胺 538"
    )

    doc.add_heading("2.2 笛卡尔积 + 训练集排除", level=2)
    doc.add_paragraph(
        "  理论笛卡尔积: 433 × 538 = 232,954\n"
        "  排除 (醛==胺 同一分子): 4\n"
        "  排除 (训练集 326 唯一配对, RDKit 命中 210): 210\n"
        "  实际筛选集: 232,740 配对"
    )

    doc.add_heading("2.3 GNN 推理", level=2)
    doc.add_paragraph(
        "  模型: models/v4.0_aug_v2/v4_model.pt (use_3d=True)\n"
        "  MC Dropout: 10 次前向传播, 取均值 ± 标准差\n"
        "  实际耗时: ~2h (CUDA, 单一 GPU)\n"
        "  输出字段: film_prob_mean, film_prob_std, film_prob_adjusted"
    )

    doc.add_heading("2.4 化学先验 + 硬过滤 + 多样性", level=2)
    doc.add_paragraph(
        "Step 1 (chem_filt): film_prob_adjusted >= 0.6 (63,523 候选)\n"
        "Step 2 (硬过滤): n_ald >= 2 AND n_am >= 2 (剩 49,407 候选)\n"
        "Step 3 (排序): 按 film_prob_mean (GNN raw) 降序\n"
        "Step 4 (多样性): Morgan Fingerprint (r=2, 2048 bit) Tanimoto<0.8, 醛和胺分别约束\n"
        "Step 5 (Top 40): 40 配对入选, target 配对追加为 #41 (不计入多样性约束)"
    )

    doc.add_heading("3. 统计摘要", level=1)

    full_adj = np.array([float(r["film_prob_adjusted"]) for r in full_rows])
    full_raw = np.array([float(r["film_prob_mean"]) for r in full_rows])
    top_raw = np.array([float(r["film_prob_mean"]) for r in top_rows])
    top_adj = np.array([float(r["film_prob_adjusted"]) for r in top_rows])
    top_std = np.array([float(r["film_prob_std"]) for r in top_rows])

    stats = doc.add_paragraph()
    stats.add_run(f"全表统计 (N={len(full_rows)}):\n").bold = True

    stats_table = doc.add_table(rows=8, cols=2)
    stats_table.style = "Light Grid Accent 1"
    stats_data = [
        ("GNN raw 中位数", f"{np.median(full_raw):.4f}"),
        ("GNN raw 均值 ± 标准差", f"{np.mean(full_raw):.4f} ± {np.std(full_raw):.4f}"),
        ("GNN raw 范围", f"{full_raw.min():.4f} ~ {full_raw.max():.4f}"),
        ("adj 中位数", f"{np.median(full_adj):.4f}"),
        ("adj 均值 ± 标准差", f"{np.mean(full_adj):.4f} ± {np.std(full_adj):.4f}"),
        ("adj >= 0.6 候选", f"{(full_adj >= 0.6).sum()} ({100*(full_adj >= 0.6).sum()/len(full_adj):.1f}%)"),
        ("adj >= 0.8 候选", f"{(full_adj >= 0.8).sum()} ({100*(full_adj >= 0.8).sum()/len(full_adj):.1f}%)"),
        ("adj >= 0.9 候选", f"{(full_adj >= 0.9).sum()} ({100*(full_adj >= 0.9).sum()/len(full_adj):.1f}%)"),
    ]
    for i, (k, v) in enumerate(stats_data):
        stats_table.rows[i].cells[0].text = k
        stats_table.rows[i].cells[1].text = v
        for cell in stats_table.rows[i].cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)

    doc.add_paragraph()
    stats2 = doc.add_paragraph()
    stats2.add_run(f"Top 40 + target 统计 (N={len(top_rows)}):\n").bold = True

    n_fluor = sum(1 for r in top_rows
                  if str(r["ald_has_f"]).lower() == "true"
                  or str(r["am_has_f"]).lower() == "true")
    n_train = sum(1 for r in top_rows
                  if "train" in r["ald_source"] or "train" in r["am_source"])
    n_com = sum(1 for r in top_rows
                if "commercial" in r["ald_source"] or "commercial" in r["am_source"])
    n_llm = sum(1 for r in top_rows
                if "llm" in r["ald_source"] or "llm" in r["am_source"])

    stats_table2 = doc.add_table(rows=6, cols=2)
    stats_table2.style = "Light Grid Accent 1"
    stats_data2 = [
        ("GNN raw 范围", f"{top_raw.min():.4f} ~ {top_raw.max():.4f}"),
        ("adj 范围", f"{top_adj.min():.4f} ~ {top_adj.max():.4f}"),
        ("MC σ 范围", f"{top_std.min():.4f} ~ {top_std.max():.4f}"),
        ("含氟配对", f"{n_fluor} ({100*n_fluor/len(top_rows):.1f}%)"),
        ("含训练集单体", f"{n_train} ({100*n_train/len(top_rows):.1f}%)"),
        ("含商业/llm 单体", f"{n_com}/{n_llm}"),
    ]
    for i, (k, v) in enumerate(stats_data2):
        stats_table2.rows[i].cells[0].text = k
        stats_table2.rows[i].cells[1].text = v
        for cell in stats_table2.rows[i].cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)

    doc.add_page_break()

    doc.add_heading(f"4. Top {n_top} + target ({n_target}) 汇总表", level=1)
    doc.add_paragraph(
        f"Top {n_top} 配对 (按 GNN raw 降序) + target 配对 (raw=0.940, 排名 #13,183, 不计入多样性约束)。"
        f"MC σ 为 Dropout 不确定性, 多样性由 Morgan Tanimoto<0.8 保证。"
    )

    table = doc.add_table(rows=1, cols=9)
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["排名", "醛单体", "胺单体", "C_醛+C_胺", "拓扑", "raw", "adj", "MC σ", "F/源"]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
        for p in table.rows[0].cells[i].paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(7)
        set_cell_shading(table.rows[0].cells[i], "D9E2F3")

    for r in top_rows:
        row = table.add_row()
        prob = float(r["film_prob_mean"])
        prob_adj = float(r["film_prob_adjusted"])
        prob_std = float(r["film_prob_std"])
        n_ald = int(r["ald_n"])
        n_am = int(r["am_n"])
        topo = topology(n_ald, n_am)
        ald_name = (r["aldehyde_name"] or r["aldehyde_smiles"][:25])[:25]
        am_name = (r["amine_name"] or r["amine_smiles"][:25])[:25]
        has_f = "F" if (str(r["ald_has_f"]).lower() == "true"
                        or str(r["am_has_f"]).lower() == "true") else "-"
        src = "/".join(sorted(set(
            [s for s in r["ald_source"].split("+") if s != "target"]
            + [s for s in r["am_source"].split("+") if s != "target"]
        )))

        values = [
            str(r["diverse_rank"]),
            ald_name, am_name,
            f"{n_ald}+{n_am}", topo,
            f"{prob:.4f}", f"{prob_adj:.4f}", f"{prob_std:.4f}",
            f"{has_f}/{src[:8]}",
        ]
        for j, v in enumerate(values):
            row.cells[j].text = v
            for p in row.cells[j].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(7)

        if is_target(r):
            for j in range(9):
                set_cell_shading(row.cells[j], "FFE599")
        elif prob > 0.97:
            for j in range(9):
                for p in row.cells[j].paragraphs:
                    for run in p.runs:
                        run.font.color.rgb = RGBColor(0x00, 0x80, 0x00)
        elif prob_std > 0.05:
            for p in row.cells[7].paragraphs:
                for run in p.runs:
                    run.font.color.rgb = RGBColor(0xFF, 0x80, 0x00)

    doc.add_page_break()

    doc.add_heading("5. 候选详情卡片", level=1)
    doc.add_paragraph(
        f"每张卡片包含: 醛/胺结构图 (RDKit 2D), GNN raw ± MC σ, adj 化学先验分, "
        f"拓扑 (hex/tet/other), 官能团数 (C_醛 + C_胺), 氟标记, 来源 (train/commercial/llm/target)。"
    )

    for idx, r in enumerate(top_rows):
        prob = float(r["film_prob_mean"])
        prob_adj = float(r["film_prob_adjusted"])
        prob_std = float(r["film_prob_std"])
        n_ald = int(r["ald_n"])
        n_am = int(r["am_n"])
        topo = topology(n_ald, n_am)
        rank = int(r["diverse_rank"])

        title_text = f"#{rank}  {(r['aldehyde_name'] or r['aldehyde_smiles'][:30])[:30]}"
        if is_target(r):
            title_text += "  <<< TARGET"
        title_text += f"  ×  {(r['amine_name'] or r['amine_smiles'][:30])[:30]}"
        doc.add_heading(title_text, level=2)

        metrics = doc.add_paragraph()
        add_colored_run(metrics, "GNN raw: ", bold=True, size=Pt(10))
        color = RGBColor(0x00, 0x80, 0x00) if prob > 0.97 else RGBColor(0x00, 0x00, 0x00)
        add_colored_run(metrics, f"{prob:.4f} ± {prob_std:.4f}", color=color)
        add_colored_run(metrics, f"  |  adj: {prob_adj:.4f}  |  "
                                  f"拓扑: {topo}  |  C{n_ald}+C{n_am}  |  "
                                  f"F: {r['ald_has_f']}/{r['am_has_f']}", size=Pt(9))
        metrics.add_run(f"\n来源: 醛={r['ald_source']}, 胺={r['am_source']}")

        try:
            ald_img = draw_mol(r["aldehyde_smiles"], f"rank{rank:02d}_ald.png")
            am_img = draw_mol(r["amine_smiles"], f"rank{rank:02d}_amine.png")
        except Exception:
            ald_img = am_img = None

        if ald_img and am_img:
            img_table = doc.add_table(rows=1, cols=2)
            img_table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for col_idx, (img, lbl) in enumerate([
                (ald_img, "醛单体"), (am_img, "胺单体"),
            ]):
                cell = img_table.rows[0].cells[col_idx]
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(img, width=Inches(2.5))
                cap = cell.add_paragraph(lbl)
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in cap.runs:
                    run.font.size = Pt(8)
                    run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

        if idx < len(top_rows) - 1:
            doc.add_paragraph("─" * 60)

    doc.add_page_break()
    doc.add_heading("6. 已知局限", level=1)

    doc.add_heading("6.1 数据局限", level=2)
    doc.add_paragraph(
        "- 化学文献天然偏置: 597/656 篇文献从未尝试成膜, 成功案例主导数据集。\n"
        "- 训练集正样本 544 条, 边界案例判断可能存在盲区。\n"
        "- 训练集 326 唯一配对中, 56 条 SMILES RDKit 解析失败, 仅 210 条能在笛卡尔积中匹配排除。"
    )

    doc.add_heading("6.2 模型局限", level=2)
    doc.add_paragraph(
        "- 模型仅评估单体对成膜潜力, 不考虑反应条件 (溶剂/温度/催化剂)。\n"
        "- GNN 对训练集未见过的大平面/新颖骨架给出保守低分, target 配对 (raw=0.940) 即此情形。\n"
        "- MC Dropout 仅覆盖 attention + head 层, 编码器不确定性未被捕获。\n"
        "- 可合成性、成本、纯度等实际因素未纳入筛选。"
    )

    doc.add_heading("6.3 筛选局限", level=2)
    doc.add_paragraph(
        "- 化学先验 (chem_penalty) 中含氟/特定结构有奖励, 可能使部分配对 adj>1.0。\n"
        "- Morgan Tanimoto<0.8 多样性约束在 chem_filt 之后应用, 可能因高分候选结构相近而漏选。\n"
        "- 训练集单体强制进池 (n>=1) 导致 1胺/1醛 配对进入笛卡尔积, 由后续硬过滤 (n>=2) 剔除。"
    )

    doc.save(output_path)
    print(f"报告已保存: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="v4 Top 40 筛选报告")
    parser.add_argument("--top-csv", default="data/processed/v4_top40_candidates.csv")
    parser.add_argument("--full-csv", default="data/processed/v4_screening_v2_soft.csv")
    parser.add_argument("--output", default="data/processed/v4_Top40_Final_Report.docx")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    if args.no_images:
        global OUTPUT_DIR
        OUTPUT_DIR = None

    generate(args.top_csv, args.full_csv, args.output)


if __name__ == "__main__":
    main()
