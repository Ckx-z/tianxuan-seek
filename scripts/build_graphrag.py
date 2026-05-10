"""构建 COF 知识图谱 — 从结构化 YAML/JSON 数据提取实体关系。

知识图谱内容:
  - 单体节点: 名称、SMILES、类型 (醛/胺)、含氟量、拓扑对称性
  - COF 配对边: 文献来源、成膜标签、氟策略
  - 属性关联: 氟含量 ↔ 成膜率、拓扑 ↔ 成膜率

输出:
  - data/processed/knowledge_graph.html (交互式 pyvis 可视化)
  - data/processed/graph_stats.json (图谱统计)
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.fluorination import FluorineDetector
from src.chemistry.imine_check import ImineChecker
from src.utils.logger import setup_logger

logger = setup_logger("graphrag")


def build_graph(llm_path: str, output_html: str, output_stats: str):
    from rdkit import Chem
    import networkx as nx
    from pyvis.network import Network

    with open(llm_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 去重文献
    seen = {}
    for r in records:
        lid = r.get("literature_id", "")
        if lid and lid not in seen:
            seen[lid] = r
    unique = list(seen.values())

    imine_checker = ImineChecker()
    f_detector = FluorineDetector()

    G = nx.Graph()

    # 统计
    monomer_stats = defaultdict(lambda: {
        "n_papers": 0, "n_film_positive": 0, "n_film_negative": 0,
        "types": set(), "smiles": "", "partners": Counter(),
    })

    # ---- 第一遍: 收集所有单体 ----
    for rec in unique:
        is_2d = rec.get("is_2d_cof", False)
        film_label = rec.get("film_label")
        monomers = [m for m in rec.get("monomers", []) if isinstance(m, dict)]

        for m in monomers:
            smi = m.get("canonical_smiles") or m.get("smiles", "")
            if not smi or smi.lower() in ("null", "none", ""):
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            can_smi = Chem.MolToSmiles(mol, canonical=True)
            mtype = m.get("monomer_type", "other")

            stats = monomer_stats[can_smi]
            stats["n_papers"] += 1
            stats["types"].add(mtype)
            stats["smiles"] = can_smi
            if film_label is True:
                stats["n_film_positive"] += 1
            elif film_label is False:
                stats["n_film_negative"] += 1

    # ---- 第二遍: 处理配对关系 ----
    pair_edges = []
    for rec in unique:
        if rec.get("is_2d_cof") is not True:
            continue
        film_label = rec.get("film_label")
        if film_label is None:
            continue

        monomers = [m for m in rec.get("monomers", []) if isinstance(m, dict)]
        valid = []
        for m in monomers:
            smi = m.get("canonical_smiles") or m.get("smiles", "")
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can_smi = Chem.MolToSmiles(mol, canonical=True)
            mtype = m.get("monomer_type", "other")
            valid.append({
                "name": m.get("name", "?"),
                "smiles": can_smi,
                "type": mtype,
                "has_f": f_detector.has_fluorine(mol),
                "n_f": f_detector.count_fluorine(mol),
                "n_ald": imine_checker.count_aldehyde_groups(mol),
                "n_am": imine_checker.count_amine_groups(mol),
            })

        aldehydes = [v for v in valid if v["type"] in ("aldehyde", "aldehyde-amine")]
        amines = [v for v in valid if v["type"] in ("amine", "aldehyde-amine")]

        if not aldehydes or not amines:
            continue

        for ald in aldehydes:
            for am in amines:
                if ald["smiles"] == am["smiles"]:
                    continue
                pair_edges.append({
                    "ald_smi": ald["smiles"],
                    "am_smi": am["smiles"],
                    "ald_name": ald["name"],
                    "am_name": am["name"],
                    "film_label": 1 if film_label else 0,
                    "ald_f": ald["has_f"],
                    "am_f": am["has_f"],
                    "ald_n_ald": ald["n_ald"],
                    "am_n_am": am["n_am"],
                    "lid": rec.get("literature_id", ""),
                })
                # 记录配对关系
                monomer_stats[ald["smiles"]]["partners"][am["smiles"]] += 1
                monomer_stats[am["smiles"]]["partners"][ald["smiles"]] += 1

    logger.info(
        f"实体: {len(monomer_stats)} 单体, {len(pair_edges)} 配对边"
    )

    # ---- 构建图 ----
    # 颜色映射
    def _node_color(mtype_set, has_f):
        if "aldehyde" in mtype_set and "amine" in mtype_set:
            base = "#9b59b6"  # 双功能紫色
        elif "aldehyde" in mtype_set:
            base = "#3498db" if not has_f else "#2980b9"  # 醛蓝
        elif "amine" in mtype_set:
            base = "#e74c3c" if not has_f else "#c0392b"  # 胺红
        else:
            base = "#95a5a6"
        return base

    def _node_size(n_papers, n_partners):
        return max(15, min(50, 10 + n_papers * 2 + n_partners * 0.5))

    # 添加单体节点
    for smi, stats in monomer_stats.items():
        has_f = f_detector.has_fluorine(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else False
        mtype_str = "/".join(sorted(stats["types"]))
        n_ald = imine_checker.count_aldehyde_groups(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0
        n_am = imine_checker.count_amine_groups(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0

        if n_ald >= 3:
            topo = "C3"
        elif n_ald >= 2:
            topo = "C2"
        elif n_am >= 4:
            topo = "C4"
        elif n_am >= 3:
            topo = "C3"
        elif n_am >= 2:
            topo = "C2"
        else:
            topo = "C1"

        total_labeled = stats["n_film_positive"] + stats["n_film_negative"]
        film_rate = stats["n_film_positive"] / max(total_labeled, 1)

        label = (
            f"{mtype_str} | {topo} | {'F' if has_f else '非F'}\n"
            f"文献: {stats['n_papers']} | 成膜率: {film_rate:.0%}"
        )

        G.add_node(
            smi,
            label=label[:60],
            title=f"SMILES: {stats['smiles']}\n类型: {mtype_str}\n"
                  f"拓扑: {topo}\n含氟: {'是' if has_f else '否'}\n"
                  f"文献数: {stats['n_papers']}\n"
                  f"成膜率: {film_rate:.1%} ({stats['n_film_positive']}/{total_labeled})",
            color=_node_color(stats["types"], has_f),
            size=_node_size(stats["n_papers"], len(stats["partners"])),
            font={"size": 10},
        )

    # 聚合边 (去重 SMILES 配对)
    edge_agg = defaultdict(lambda: {"count": 0, "film_pos": 0, "film_neg": 0})
    for e in pair_edges:
        key = tuple(sorted([e["ald_smi"], e["am_smi"]]))
        edge_agg[key]["count"] += 1
        if e["film_label"] == 1:
            edge_agg[key]["film_pos"] += 1
        else:
            edge_agg[key]["film_neg"] += 1

    # 添加边 (仅保留出现 ≥2 次的配对，避免图过密)
    for (smi_a, smi_b), agg in edge_agg.items():
        if agg["count"] < 2:
            continue
        total = agg["film_pos"] + agg["film_neg"]
        film_ratio = agg["film_pos"] / total if total > 0 else 0
        edge_color = "#27ae60" if film_ratio > 0.5 else "#e67e22"
        edge_width = min(5, 1 + agg["count"] * 0.5)
        G.add_edge(
            smi_a, smi_b,
            weight=agg["count"],
            title=f"配对次数: {agg['count']}\n成膜: {agg['film_pos']}\n不成膜: {agg['film_neg']}\n成膜率: {film_ratio:.1%}",
            color=edge_color,
            width=edge_width,
        )

    logger.info(f"图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # ---- 子图: 仅保留含氟 / 成膜相关关键节点 ----
    # 取度数 top 100 的核心节点
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:100]
    core = G.subgraph(top_nodes).copy()

    # ---- pyvis 可视化 ----
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.set_options("""
    var options = {
      "nodes": {"borderWidth": 1, "borderWidthSelected": 3},
      "edges": {"smooth": {"type": "continuous"}, "hoverWidth": 2},
      "physics": {"barnesHut": {"gravitationalConstant": -2000, "springLength": 150}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)

    net.from_nx(core)

    # 添加图例
    legend_html = """
    <div style="position:fixed;top:10px;right:10px;background:white;padding:10px;
                border:1px solid #ccc;border-radius:5px;font-size:12px;z-index:999">
      <b>图例</b><br/>
      <span style="color:#3498db">●</span> 醛单体 (非F)<br/>
      <span style="color:#2980b9">●</span> 醛单体 (F)<br/>
      <span style="color:#e74c3c">●</span> 胺单体 (非F)<br/>
      <span style="color:#c0392b">●</span> 胺单体 (F)<br/>
      <span style="color:#9b59b6">●</span> 双功能<br/>
      <hr/>
      <span style="color:#27ae60">—</span> 成膜率>50%<br/>
      <span style="color:#e67e22">—</span> 成膜率≤50%<br/>
      <span style="font-size:10px">节点大小 ∝ 文献数</span>
    </div>
    """
    net.html = net.html.replace("</body>", legend_html + "</body>")

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    net.save_graph(output_html)
    logger.info(f"交互式知识图谱已保存: {output_html}")

    # ---- 统计导出 ----
    stats = {
        "n_monomers": len(monomer_stats),
        "n_edges_total": len(edge_agg),
        "n_edges_core": core.number_of_edges(),
        "n_nodes_core": core.number_of_nodes(),
        "n_unique_papers": len(unique),
        "top_aldehydes": [
            {"smiles": s, "n_papers": v["n_papers"], "film_rate": v["n_film_positive"] / max(v["n_film_positive"] + v["n_film_negative"], 1)}
            for s, v in sorted(monomer_stats.items(),
                             key=lambda x: x[1]["n_papers"], reverse=True)
            if "aldehyde" in v["types"] and "amine" not in v["types"]
        ][:20],
        "top_amines": [
            {"smiles": s, "n_papers": v["n_papers"], "film_rate": v["n_film_positive"] / max(v["n_film_positive"] + v["n_film_negative"], 1)}
            for s, v in sorted(monomer_stats.items(),
                             key=lambda x: x[1]["n_papers"], reverse=True)
            if "amine" in v["types"] and "aldehyde" not in v["types"]
        ][:20],
        "top_pairs": [
            {"smiles_pair": list(k), "count": v["count"], "film_ratio": v["film_pos"] / (v["film_pos"] + v["film_neg"])}
            for k, v in sorted(edge_agg.items(), key=lambda x: x[1]["count"], reverse=True)[:30]
        ],
    }

    with open(output_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"图谱统计已保存: {output_stats}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="构建 COF 知识图谱")
    parser.add_argument("--input", default="data/processed/monomer_smiles_llm.json")
    parser.add_argument("--output-html", default="data/processed/knowledge_graph.html")
    parser.add_argument("--output-stats", default="data/processed/graph_stats.json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    stats = build_graph(args.input, args.output_html, args.output_stats)

    print("\n" + "=" * 60)
    print("  COF 知识图谱构建完成")
    print("=" * 60)
    print(f"  单体节点:     {stats['n_monomers']}")
    print(f"  配对边 (≥2次): {stats['n_edges_core']}")
    print(f"  核心节点:     {stats['n_nodes_core']}")
    print(f"  唯一文献:     {stats['n_unique_papers']}")
    print(f"  HTML 图谱:    {args.output_html}")
    print(f"  统计数据:     {args.output_stats}")
    print("=" * 60)


if __name__ == "__main__":
    main()
