"""COF 知识图谱交互式可视化 — pyvis 网页渲染。

输出:
  - graphrag图谱辅助/graph_visual.html  (全量图, 可交互浏览)
  - graphrag图谱辅助/graph_visual_core.html (核心子图, Top 200 节点)

颜色:
  蓝色系 — 醛单体 (深蓝=F, 浅蓝=非F)
  红色系 — 胺单体 (深红=F, 浅红=非F)
  紫色   — 双功能单体
  绿色边 — 成膜率>50%
  橙色边 — 成膜率≤50%
"""
import json
import os
import sys
from collections import Counter, defaultdict

import networkx as nx
from pyvis.network import Network

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _node_color(node: dict) -> str:
    """颜色编码: 类型 + 氟 + N杂环。"""
    mtype = node.get("monomer_type", "")
    has_f = node.get("has_fluorine", False)
    has_n = node.get("has_n_heterocycle", False)

    if mtype == "aldehyde":
        if has_n:
            return "#1a5276"   # N杂环醛 (深蓝)
        return "#2980b9" if has_f else "#85c1e9"   # F醛 / 非F醛
    elif mtype == "amine":
        if has_n:
            return "#922b21"   # N杂环胺 (深红)
        return "#c0392b" if has_f else "#f1948a"   # F胺 / 非F胺
    elif mtype == "dual":
        return "#8e44ad"   # 双功能 (紫色)
    return "#95a5a6"


def _node_size(node: dict) -> int:
    """节点大小: 文献数 + 配对度。"""
    n_lit = node.get("n_literatures", 0)
    n_part = node.get("n_partners", 0)
    return max(8, min(40, 6 + n_lit * 1.5 + n_part * 0.5))


def _node_title(node: dict, node_id: str) -> str:
    """悬停提示信息。"""
    mtype = node.get("monomer_type", "?")
    topo = node.get("topology", "?")
    has_f = "是" if node.get("has_fluorine") else "否"
    has_n = "是" if node.get("has_n_heterocycle") else "否"
    n_lit = node.get("n_literatures", 0)
    n_part = node.get("n_partners", 0)
    film_p = node.get("film_positive", 0)
    film_n = node.get("film_negative", 0)
    film_rate = node.get("film_rate", 0)
    n_ald = node.get("n_aldehyde", 0)
    n_am = node.get("n_amine", 0)
    n_arom = node.get("n_aromatic_rings", 0)
    mw = node.get("mw", 0)

    return (
        f"<b>SMILES:</b> {node_id[:50]}<br>"
        f"<b>类型:</b> {mtype} | 拓扑: {topo}<br>"
        f"<b>官能团:</b> {n_ald}醛基 {n_am}胺基<br>"
        f"<b>含氟:</b> {has_f} | N杂环: {has_n} | 芳环: {n_arom}<br>"
        f"<b>分子量:</b> {mw:.1f}<br>"
        f"<b>文献数:</b> {n_lit} | 配对邻居: {n_part}<br>"
        f"<b>成膜:</b> {film_p}/{film_p + film_n} ({film_rate:.1%})"
    )


def _edge_color(edge: dict) -> str:
    """边颜色: 成膜率 > 50% 绿色, 否则橙色。"""
    fr = edge.get("film_ratio", 0)
    if fr >= 0.5:
        return "#27ae60"
    if fr > 0:
        return "#f39c12"
    return "#bdc3c7"  # 灰 — 成膜率=0


def _edge_width(edge: dict) -> float:
    """边宽: 文献数。"""
    return min(5, 0.5 + edge.get("n_literatures", 1) * 0.8)


def build_visualization(
    gml_path: str = "graphrag图谱辅助/graph.gml",
    nodes_path: str = "graphrag图谱辅助/graph_nodes.json",
    edges_path: str = "graphrag图谱辅助/graph_edges.json",
    output_dir: str = "graphrag图谱辅助",
):
    G = nx.read_gml(gml_path)
    with open(nodes_path, encoding="utf-8") as f:
        nodes_data = {n["id"]: n for n in json.load(f)}
    with open(edges_path, encoding="utf-8") as f:
        edges_data = json.load(f)

    # ── 子图: 仅保留单体节点 + 配对边 ──
    monomers = {n for n, d in G.nodes(data=True) if d.get("node_type") == "monomer"}
    monomer_G = G.subgraph(monomers).copy()

    # 仅保留 PAIRED_WITH 边
    for u, v, d in list(monomer_G.edges(data=True)):
        if d.get("edge_type") != "PAIRED_WITH":
            monomer_G.remove_edge(u, v)

    # 移除孤立节点
    monomer_G.remove_nodes_from(list(nx.isolates(monomer_G)))

    print(f"可视化图: {monomer_G.number_of_nodes()} 节点, "
          f"{monomer_G.number_of_edges()} 边")

    # ═══════════════════════════════════════
    # 1. 核心子图 (Top 150 节点按度数)
    # ═══════════════════════════════════════
    degrees = dict(monomer_G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:150]
    core_G = monomer_G.subgraph(top_nodes).copy()

    core_net = Network(height="850px", width="100%", bgcolor="#f8f9fa",
                       font_color="#2c3e50", directed=False)
    core_net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 1.5,
        "borderWidthSelected": 4,
        "font": {"size": 11, "face": "Arial", "strokeWidth": 0}
      },
      "edges": {
        "smooth": {"type": "continuous", "forceDirection": "none"},
        "hoverWidth": 2,
        "selectionWidth": 2
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.3
        },
        "minVelocity": 0.75,
        "solver": "barnesHut"
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    for nid in core_G.nodes():
        nd = nodes_data.get(nid, {})
        core_net.add_node(
            nid,
            label=nd.get("label", nid[:30]),
            title=_node_title(nd, nid),
            color=_node_color(nd),
            size=_node_size(nd),
            borderWidth=2,
            borderWidthSelected=5,
        )

    for u, v, d in core_G.edges(data=True):
        core_net.add_edge(
            u, v,
            title=(
                f"文献: {d.get('n_literatures', 0)}<br>"
                f"成膜率: {d.get('film_ratio', 0):.1%}<br>"
                f"成膜: {d.get('film_positive', 0)} / "
                f"不成膜: {d.get('film_negative', 0)}"
            ),
            color=_edge_color(d),
            width=_edge_width(d),
        )

    core_path = os.path.join(output_dir, "graph_visual_core.html")
    core_net.save_graph(core_path)
    _add_legend(core_path)
    print(f"核心子图 (150节点): {core_path}")

    # ═══════════════════════════════════════
    # 2. 全量图 (全部单体)
    # ═══════════════════════════════════════
    full_net = Network(height="900px", width="100%", bgcolor="#ffffff",
                       font_color="#2c3e50", directed=False)
    full_net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 1,
        "borderWidthSelected": 3,
        "font": {"size": 9, "face": "Arial"}
      },
      "edges": {
        "smooth": {"type": "continuous", "forceDirection": "none"},
        "hoverWidth": 1.5
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.2,
          "springLength": 250,
          "springConstant": 0.02,
          "damping": 0.4
        },
        "minVelocity": 0.75,
        "solver": "barnesHut"
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 150,
        "navigationButtons": true
      }
    }
    """)

    for nid in monomer_G.nodes():
        nd = nodes_data.get(nid, {})
        full_net.add_node(
            nid,
            label=nd.get("label", nid[:25]),
            title=_node_title(nd, nid),
            color=_node_color(nd),
            size=_node_size(nd),
        )

    for u, v, d in monomer_G.edges(data=True):
        full_net.add_edge(
            u, v,
            title=(
                f"文献: {d.get('n_literatures', 0)} | "
                f"成膜率: {d.get('film_ratio', 0):.1%}"
            ),
            color=_edge_color(d),
            width=_edge_width(d),
        )

    full_path = os.path.join(output_dir, "graph_visual.html")
    full_net.save_graph(full_path)
    _add_legend(full_path)
    print(f"全量图: {full_path}")

    return core_path, full_path


def _add_legend(html_path: str):
    """在 HTML 中注入图例。"""
    legend = """
    <div style="position:fixed;top:10px;right:10px;background:rgba(255,255,255,0.95);
                padding:12px;border:1px solid #ccc;border-radius:6px;font-size:11px;
                z-index:999;font-family:Arial,sans-serif;line-height:1.6">
      <b style="font-size:13px">图例</b><br/>
      <span style="color:#2980b9">●</span> 醛 (F)&nbsp;&nbsp;
      <span style="color:#85c1e9">●</span> 醛 (非F)<br/>
      <span style="color:#c0392b">●</span> 胺 (F)&nbsp;&nbsp;
      <span style="color:#f1948a">●</span> 胺 (非F)<br/>
      <span style="color:#1a5276">●</span> N杂环醛&nbsp;
      <span style="color:#922b21">●</span> N杂环胺<br/>
      <span style="color:#8e44ad">●</span> 双功能<br/>
      <hr style="margin:4px 0"/>
      <span style="color:#27ae60">—</span> 成膜率 ≥ 50%<br/>
      <span style="color:#f39c12">—</span> 成膜率 0–50%<br/>
      <span style="color:#bdc3c7">—</span> 成膜率 = 0<br/>
      <hr style="margin:4px 0"/>
      <span>节点大小 ∝ 文献数</span><br/>
      <span>边宽 ∝ 配对文献数</span>
    </div>
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</body>", legend + "</body>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    build_visualization()
