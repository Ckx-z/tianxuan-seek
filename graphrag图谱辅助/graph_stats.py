"""COF 知识图谱统计分析 — 度分布、社区发现、中心性、聚合模式。

输出:
  - graphrag图谱辅助/graph_analysis.json   (统计结果)
  - graphrag图谱辅助/graph_communities.json (社区发现)
"""
import json
import os
from collections import Counter, defaultdict

import networkx as nx
import numpy as np


def analyze_graph(gml_path: str = "graphrag图谱辅助/graph.gml",
                  output_dir: str = "graphrag图谱辅助") -> dict:
    G = nx.read_gml(gml_path)

    monomers = {n for n, d in G.nodes(data=True) if d.get("node_type") == "monomer"}
    literatures = {n for n, d in G.nodes(data=True) if d.get("node_type") == "literature"}
    paired_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get("edge_type") == "PAIRED_WITH"]

    # ── 度分布 ──
    monomer_degrees = {n: G.degree(n) for n in monomers}
    deg_sorted = sorted(monomer_degrees.items(), key=lambda x: x[1], reverse=True)

    # ── 成膜率分布 ──
    film_rates = []
    for n in monomers:
        d = G.nodes[n]
        if d.get("film_rate") is not None:
            film_rates.append(d["film_rate"])
    film_rates = np.array(film_rates)

    # ── 氟分布 ──
    f_ald = sum(1 for n in monomers
                if G.nodes[n].get("monomer_type") == "aldehyde"
                and G.nodes[n].get("has_fluorine"))
    f_am = sum(1 for n in monomers
               if G.nodes[n].get("monomer_type") == "amine"
               and G.nodes[n].get("has_fluorine"))
    nonf_ald = sum(1 for n in monomers
                   if G.nodes[n].get("monomer_type") == "aldehyde"
                   and not G.nodes[n].get("has_fluorine"))
    nonf_am = sum(1 for n in monomers
                  if G.nodes[n].get("monomer_type") == "amine"
                  and not G.nodes[n].get("has_fluorine"))

    # ── 配对统计: 氟策略 × 成膜率 ──
    f_strategy_stats = defaultdict(lambda: {"count": 0, "film_pos": 0, "film_neg": 0})
    for u, v in paired_edges:
        d = G.get_edge_data(u, v)
        ald_f = G.nodes[u].get("has_fluorine", False)
        am_f = G.nodes[v].get("has_fluorine", False)
        if G.nodes[u].get("monomer_type") != "aldehyde":
            ald_f, am_f = am_f, ald_f
        f_strat = f"ald={'F' if ald_f else 'nF'}_am={'F' if am_f else 'nF'}"
        f_strategy_stats[f_strat]["count"] += 1
        f_strategy_stats[f_strat]["film_pos"] += d.get("film_positive", 0)
        f_strategy_stats[f_strat]["film_neg"] += d.get("film_negative", 0)

    # ── 拓扑组合统计 ──
    topo_pair_stats = defaultdict(lambda: {"count": 0, "film_pos": 0, "film_neg": 0})
    for u, v in paired_edges:
        d = G.get_edge_data(u, v)
        ald_t = G.nodes[u].get("topology", "?")
        am_t = G.nodes[v].get("topology", "?")
        if G.nodes[u].get("monomer_type") != "aldehyde":
            ald_t, am_t = am_t, ald_t
        combo = f"{ald_t}+{am_t}"
        topo_pair_stats[combo]["count"] += 1
        topo_pair_stats[combo]["film_pos"] += d.get("film_positive", 0)
        topo_pair_stats[combo]["film_neg"] += d.get("film_negative", 0)

    # ── 单体拓扑分布 ──
    topo_dist = Counter(G.nodes[n].get("topology", "?") for n in monomers)

    # ── 中心性 ──
    subgraph = G.subgraph(monomers)
    centrality = nx.degree_centrality(subgraph)
    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]

    # ── 社区发现 (Louvain) ──
    try:
        communities = nx.community.louvain_communities(subgraph)
    except Exception:
        communities = []

    community_summary = []
    for i, comm in enumerate(communities[:10]):
        comm_members = list(comm)[:30]
        comm_types = Counter(G.nodes[n].get("monomer_type", "?") for n in comm)
        comm_topo = Counter(G.nodes[n].get("topology", "?") for n in comm)
        comm_f = sum(1 for n in comm if G.nodes[n].get("has_fluorine"))
        community_summary.append({
            "community_id": i,
            "size": len(comm),
            "types": dict(comm_types),
            "topology": dict(comm_topo),
            "n_fluorine": comm_f,
            "sample_members": comm_members[:10],
        })

    # ── 组装结果 ──
    results = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_monomers": len(monomers),
        "n_literatures": len(literatures),
        "n_paired_edges": len(paired_edges),

        "degree_top20": [
            {"smiles": n, "degree": d, "monomer_type": G.nodes[n].get("monomer_type"),
             "n_literatures": G.nodes[n].get("n_literatures"),
             "film_rate": G.nodes[n].get("film_rate")}
            for n, d in deg_sorted[:20]
        ],

        "film_rate_stats": {
            "mean": float(np.mean(film_rates)) if len(film_rates) else 0,
            "median": float(np.median(film_rates)) if len(film_rates) else 0,
            "std": float(np.std(film_rates)) if len(film_rates) else 0,
            "p25": float(np.percentile(film_rates, 25)) if len(film_rates) else 0,
            "p75": float(np.percentile(film_rates, 75)) if len(film_rates) else 0,
        },

        "fluorine_distribution": {
            "f_aldehyde": f_ald, "f_amine": f_am,
            "nonf_aldehyde": nonf_ald, "nonf_amine": nonf_am,
        },

        "fluorine_strategy_stats": {
            k: {**v, "film_rate": round(v["film_pos"] / max(v["film_pos"] + v["film_neg"], 1), 3)}
            for k, v in f_strategy_stats.items()
        },

        "topology_pair_stats": {
            k: {**v, "film_rate": round(v["film_pos"] / max(v["film_pos"] + v["film_neg"], 1), 3)}
            for k, v in sorted(topo_pair_stats.items())
        },

        "monomer_topology_dist": dict(topo_dist),

        "centrality_top20": [
            {"smiles": n, "centrality": round(c, 4), "monomer_type": G.nodes[n].get("monomer_type")}
            for n, c in top_central
        ],

        "communities": community_summary,
    }

    # ── 导出 ──
    os.makedirs(output_dir, exist_ok=True)

    analysis_path = os.path.join(output_dir, "graph_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"分析结果: {analysis_path}")

    communities_path = os.path.join(output_dir, "graph_communities.json")
    with open(communities_path, "w", encoding="utf-8") as f:
        json.dump(community_summary, f, ensure_ascii=False, indent=2)
    print(f"社区发现: {communities_path}")

    # ── 打印摘要 ──
    print(f"\n═══ 图谱分析摘要 ═══")
    print(f"单体: {len(monomers)} | 文献: {len(literatures)} | 配对边: {len(paired_edges)}")
    print(f"成膜率: mean={results['film_rate_stats']['mean']:.2%}, "
          f"median={results['film_rate_stats']['median']:.2%}")
    print(f"氟分布: F-醛={f_ald}, F-胺={f_am}, 非F-醛={nonf_ald}, 非F-胺={nonf_am}")
    print(f"拓扑: {dict(topo_dist)}")
    print(f"社区数: {len(communities)}")
    print(f"\n氟策略成膜率:")
    for k, v in results["fluorine_strategy_stats"].items():
        print(f"  {k}: {v['count']}对, 成膜率 {v['film_rate']:.1%}")
    print(f"\n拓扑组合成膜率 Top 5:")
    for k, v in sorted(results["topology_pair_stats"].items(),
                       key=lambda x: x[1]["film_rate"], reverse=True)[:5]:
        print(f"  {k}: {v['count']}对, 成膜率 {v['film_rate']:.1%}")

    return results


if __name__ == "__main__":
    analyze_graph()
