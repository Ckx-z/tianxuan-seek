"""构建 COF 全量知识图谱 — 从 954 YAML + metadata CSV 提取实体关系。

图结构:
  - Monomer 节点: canonical SMILES, 类型/氟/拓扑/杂环
  - Literature 节点: literature_id, 成膜/拓扑/条件
  - PAIRED_WITH 边 (单体-单体): 文献来源, 成膜标签, 氟策略
  - APPEARS_IN 边 (单体-文献): 单体出现在哪篇文献

输出:
  - graphrag图谱辅助/graph.gml      (GML 格式, 可被 NetworkX/Cytoscape 读取)
  - graphrag图谱辅助/graph_nodes.json  (节点属性, 方便查询)
  - graphrag图谱辅助/graph_edges.json  (配对边, 方便查询)
"""
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chemistry.imine_check import ImineChecker
from src.chemistry.fluorination import FluorineDetector
from src.chemistry.linker_analyzer import count_aromatic_rings

warnings.filterwarnings("ignore")

# ── SMARTS ──
_ALD_SMARTS = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
_AM_SMARTS = Chem.MolFromSmarts("[NH2][c]")
_N_HETERO = Chem.MolFromSmarts("[n]")

imine_checker = ImineChecker()
f_detector = FluorineDetector()


def _canon(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def _topology_from_counts(n_ald: int, n_am: int) -> str:
    if n_am >= 4:
        return "C4"
    if n_ald >= 3 or n_am >= 3:
        return "C3"
    if n_ald >= 2 or n_am >= 2:
        return "C2"
    return "C1"


def _topology_label(ald: str, am: str) -> str:
    return _topology_pair(ald, am)


def _topology_pair(ald_topo: str, am_topo: str) -> str:
    ald_n = {"C1": 1, "C2": 2, "C3": 3, "C4": 4}.get(ald_topo, 0)
    am_n = {"C1": 1, "C2": 2, "C3": 3, "C4": 4}.get(am_topo, 0)
    if ald_n >= 3 and am_n >= 3:
        return "C3+C3"
    if ald_n >= 3 or am_n >= 3:
        return "hex (hcb)"
    if ald_n == 2 and am_n == 2:
        return "sql (四方)"
    if ald_n == 2 and am_n == 4:
        return "sql (四方)"
    if ald_n == 4 and am_n == 2:
        return "sql (四方)"
    return "other"


def _fluorine_strategy(ald_f: bool, am_f: bool) -> str:
    if ald_f and am_f:
        return "F×F"
    if ald_f:
        return "F×非F"
    if am_f:
        return "非F×F"
    return "非F×非F"


def _parse_yaml(yaml_path: str) -> Optional[Dict[str, Any]]:
    """读取单个 YAML, 提取关键字段。"""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return None
    return data


def build_graph(
    yaml_dir: str = "data/structured",
    meta_path: str = "data/processed/label_metadata_v4.csv",
    cache_path: str = "data/processed/monomer_smiles_cache.json",
    output_dir: str = "graphrag图谱辅助",
) -> tuple[nx.Graph, Dict[str, Any]]:
    """主函数: 构建全量 COF 知识图谱。"""

    # ── 1. 加载 SMILES 缓存 ──
    with open(cache_path, encoding="utf-8") as f:
        smiles_cache = json.load(f)

    # ── 2. 加载 metadata (配对 & 标签) ──
    meta = pd.read_csv(meta_path, encoding="utf-8-sig")

    # ── 3. 遍历 YAML, 积累实体信息 ──
    yaml_dir_path = Path(yaml_dir)
    yaml_files = sorted(yaml_dir_path.glob("*.yaml"))

    lit_nodes: Dict[str, Dict] = {}        # literature_id → attrs
    monomer_stats: Dict[str, Dict] = defaultdict(lambda: {
        "n_papers": 0, "types": set(), "film_pos": 0, "film_neg": 0,
        "n_ald": 0, "n_am": 0, "has_f": False, "n_f": 0,
        "partners": Counter(), "lit_ids": set(),
    })

    # 从 metadata 建立 literature_id → YAML 映射
    lit_to_yaml: Dict[str, str] = {}
    lid_pattern = set(meta["literature_id"].dropna().unique())

    for yf in yaml_files:
        lid = yf.stem  # filename without .yaml
        if lid in lid_pattern or True:  # 全量, 不限于 metadata
            lit_to_yaml[lid] = str(yf)

    # 遍历 metadata 中的配对
    pair_records: List[Dict] = []
    for _, row in meta.iterrows():
        lid = str(row["literature_id"])
        ald_smi = str(row["aldehyde_smiles"])
        am_smi = str(row["amine_smiles"])
        label = int(row["label"]) if pd.notna(row["label"]) else -1
        has_f = bool(row["has_f"]) if pd.notna(row["has_f"]) else False

        can_ald = _canon(ald_smi)
        can_am = _canon(am_smi)
        if not can_ald or not can_am:
            continue

        # 记录单体
        for smi, mtype in [(can_ald, "aldehyde"), (can_am, "amine")]:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            ms = monomer_stats[smi]
            ms["n_papers"] += 1
            ms["types"].add(mtype)
            ms["n_ald"] = imine_checker.count_aldehyde_groups(mol)
            ms["n_am"] = imine_checker.count_amine_groups(mol)
            ms["has_f"] = f_detector.has_fluorine(mol)
            ms["n_f"] = f_detector.count_fluorine(mol)
            ms["lit_ids"].add(lid)
            if label == 1:
                ms["film_pos"] += 1
            elif label == 0:
                ms["film_neg"] += 1

        # 记录配对
        monomer_stats[can_ald]["partners"][can_am] += 1
        monomer_stats[can_am]["partners"][can_ald] += 1

        pair_records.append({
            "ald_smi": can_ald, "am_smi": can_am,
            "lid": lid, "label": label,
            "has_f": has_f,
        })

    # ── 4. 补充 YAML 中的文献元信息 ──
    for lid, ypath in lit_to_yaml.items():
        try:
            data = _parse_yaml(ypath)
            if data is None:
                continue
        except Exception:
            continue

        lit_nodes[lid] = {
            "solvent": str(data.get("solvent", ""))[:200],
            "reaction_temperature": str(data.get("reaction_temperature", ""))[:200],
            "synthesis_mode": str(data.get("synthesis_mode", ""))[:200],
            "interface_type": str(data.get("interface_type", ""))[:200],
            "fluorine_monomer": str(data.get("fluorine_monomer", ""))[:100],
            "film_field": str(data.get("film_crystallinity_fluorine", ""))[:300],
            "has_yaml": True,
        }

    # ── 5. 构建 NetworkX 图 ──
    G = nx.Graph()

    # 单体节点
    for smi, stats in monomer_stats.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)
        n_ald = stats["n_ald"]
        n_am = stats["n_am"]
        topo = _topology_from_counts(n_ald, n_am)
        mtypes = stats["types"]
        if "aldehyde" in mtypes and "amine" in mtypes:
            mtype_str = "dual"
        elif "aldehyde" in mtypes:
            mtype_str = "aldehyde"
        else:
            mtype_str = "amine"

        has_n_hetero = mol.HasSubstructMatch(_N_HETERO)
        n_arom = count_aromatic_rings(mol)
        total_labeled = stats["film_pos"] + stats["film_neg"]
        film_rate = stats["film_pos"] / max(total_labeled, 1)

        G.add_node(smi, **{
            "node_type": "monomer",
            "monomer_type": mtype_str,
            "n_aldehyde": n_ald,
            "n_amine": n_am,
            "topology": topo,
            "has_fluorine": stats["has_f"],
            "n_fluorine": stats["n_f"],
            "has_n_heterocycle": has_n_hetero,
            "n_aromatic_rings": n_arom,
            "mw": round(mw, 2),
            "n_literatures": stats["n_papers"],
            "film_positive": stats["film_pos"],
            "film_negative": stats["film_neg"],
            "film_rate": round(film_rate, 3),
            "n_partners": len(stats["partners"]),
            "label": f"{mtype_str} | {topo} | {'F' if stats['has_f'] else '非F'} | {stats['n_papers']}篇",
        })

    # 文献节点
    for lid, attrs in lit_nodes.items():
        G.add_node(lid, **{"node_type": "literature", **attrs})

    # APPEARS_IN 边 (单体 → 文献)
    for smi, stats in monomer_stats.items():
        for lid in stats["lit_ids"]:
            if lid in lit_nodes:
                G.add_edge(smi, lid, edge_type="APPEARS_IN")

    # PAIRED_WITH 边 (单体 → 单体), 聚合
    pair_agg: Dict[tuple, Dict] = defaultdict(
        lambda: {"count": 0, "film_pos": 0, "film_neg": 0, "lids": []}
    )
    for pr in pair_records:
        if pr["ald_smi"] not in G or pr["am_smi"] not in G:
            continue
        key = tuple(sorted([pr["ald_smi"], pr["am_smi"]]))
        pair_agg[key]["count"] += 1
        if pr["label"] == 1:
            pair_agg[key]["film_pos"] += 1
        elif pr["label"] == 0:
            pair_agg[key]["film_neg"] += 1
        pair_agg[key]["lids"].append(pr["lid"])

    for (smi_a, smi_b), agg in pair_agg.items():
        total = agg["film_pos"] + agg["film_neg"]
        film_ratio = agg["film_pos"] / total if total > 0 else 0
        ald_smi = smi_a if G.nodes[smi_a].get("monomer_type") == "aldehyde" else smi_b
        am_smi = smi_b if G.nodes[smi_b].get("monomer_type") == "amine" else smi_a
        if G.nodes[ald_smi].get("monomer_type") != "aldehyde":
            ald_smi, am_smi = am_smi, ald_smi

        G.add_edge(smi_a, smi_b, **{
            "edge_type": "PAIRED_WITH",
            "ald_smi": ald_smi,
            "am_smi": am_smi,
            "count": agg["count"],
            "film_positive": agg["film_pos"],
            "film_negative": agg["film_neg"],
            "film_ratio": round(film_ratio, 3),
            "n_literatures": len(set(agg["lids"])),
            "literature_ids": list(set(agg["lids"])),
        })

    # ── 6. 导出 ──
    os.makedirs(output_dir, exist_ok=True)

    # GML
    gml_path = os.path.join(output_dir, "graph.gml")
    nx.write_gml(G, gml_path)
    print(f"GML 图谱: {gml_path}")

    # 节点 JSON
    nodes_out = []
    for nid, attrs in G.nodes(data=True):
        node_info = {"id": nid, **attrs}
        # 序列化 set
        for k, v in node_info.items():
            if isinstance(v, set):
                node_info[k] = list(v)
            elif isinstance(v, np.integer):
                node_info[k] = int(v)
            elif isinstance(v, np.floating):
                node_info[k] = float(v)
            elif isinstance(v, np.bool_):
                node_info[k] = bool(v)
        nodes_out.append(node_info)

    nodes_path = os.path.join(output_dir, "graph_nodes.json")
    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump(nodes_out, f, ensure_ascii=False, indent=2)
    print(f"节点 JSON: {nodes_path}")

    # 边 JSON (仅 PAIRED_WITH)
    edges_out = []
    for u, v, attrs in G.edges(data=True):
        if attrs.get("edge_type") != "PAIRED_WITH":
            continue
        edge_info = {"source": u, "target": v, **attrs}
        for k, val in edge_info.items():
            if isinstance(val, (np.integer,)):
                edge_info[k] = int(val)
            elif isinstance(val, (np.floating,)):
                edge_info[k] = float(val)
            elif isinstance(val, (np.bool_,)):
                edge_info[k] = bool(val)
        edges_out.append(edge_info)

    edges_path = os.path.join(output_dir, "graph_edges.json")
    with open(edges_path, "w", encoding="utf-8") as f:
        json.dump(edges_out, f, ensure_ascii=False, indent=2)
    print(f"边 JSON: {edges_path}")

    # 统计
    monomer_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "monomer"]
    lit_nodes_list = [n for n, d in G.nodes(data=True) if d.get("node_type") == "literature"]
    paired_edges = [e for e in G.edges(data=True) if e[2].get("edge_type") == "PAIRED_WITH"]
    appear_edges = [e for e in G.edges(data=True) if e[2].get("edge_type") == "APPEARS_IN"]

    stats = {
        "n_monomer_nodes": len(monomer_nodes),
        "n_literature_nodes": len(lit_nodes_list),
        "n_total_nodes": G.number_of_nodes(),
        "n_paired_edges": len(paired_edges),
        "n_appear_edges": len(appear_edges),
        "n_total_edges": G.number_of_edges(),
    }

    stats_path = os.path.join(output_dir, "graph_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n图谱统计:")
    print(f"  单体节点: {stats['n_monomer_nodes']}")
    print(f"  文献节点: {stats['n_literature_nodes']}")
    print(f"  配对边:   {stats['n_paired_edges']}")
    print(f"  归属边:   {stats['n_appear_edges']}")

    return G, stats


if __name__ == "__main__":
    G, stats = build_graph()
