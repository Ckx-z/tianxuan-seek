r"""COF 知识图谱自然语言查询接口。

支持的查询类型:
  /stats     — 图谱概览统计
  /monomer   — 单体详情 (SMILES, 文献数, 成膜率, 配对分布)
  /pair      — 配对查询 (醛+胺 → 成膜/条件/来源)
  /neighbors — 单体的全部配对邻居, 按文献数/成膜率排序
  /recommend — 推荐未尝试配对 (共同邻居 → 排除已知边)
  /filter    — 属性过滤 (含氟/拓扑/N杂环) + 聚合统计
  /compare   — 两个单体对比
  /ask       — 自然语言问题 (LLM 解析后调用上述函数)
"""
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from rdkit import Chem

_N_HETERO = Chem.MolFromSmarts("[n]")
_ALD_SMARTS = Chem.MolFromSmarts("[CX3H1](=O)[#6]")


def _canon(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


class GraphQuery:
    """轻量图查询引擎。"""

    def __init__(self, gml_path: str = "graphrag图谱辅助/graph.gml",
                 nodes_path: str = "graphrag图谱辅助/graph_nodes.json",
                 edges_path: str = "graphrag图谱辅助/graph_edges.json"):
        self.G = nx.read_gml(gml_path)
        with open(nodes_path, encoding="utf-8") as f:
            self.nodes_data = {n["id"]: n for n in json.load(f)}
        with open(edges_path, encoding="utf-8") as f:
            self.edges_data = json.load(f)

        # 索引
        self._ald_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("monomer_type") == "aldehyde" and d.get("n_aldehyde", 0) >= 2
        ]
        self._am_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("monomer_type") == "amine" and d.get("n_amine", 0) >= 2
        ]
        print(f"图谱已加载: {self.G.number_of_nodes()} 节点, "
              f"{self.G.number_of_edges()} 边")

    # ═══════════════════════════════════════════
    # 图谱概览
    # ═══════════════════════════════════════════
    def stats(self) -> str:
        G = self.G
        monomers = [d for _, d in G.nodes(data=True) if d.get("node_type") == "monomer"]
        lit = [d for _, d in G.nodes(data=True) if d.get("node_type") == "literature"]
        paired = [d for _, _, d in G.edges(data=True) if d.get("edge_type") == "PAIRED_WITH"]

        ald = [m for m in monomers if m.get("monomer_type") == "aldehyde" and m.get("n_aldehyde", 0) >= 2]
        am = [m for m in monomers if m.get("monomer_type") == "amine" and m.get("n_amine", 0) >= 2]
        f_ald = [m for m in ald if m.get("has_fluorine")]
        f_am = [m for m in am if m.get("has_fluorine")]
        n_hetero = [m for m in monomers if m.get("has_n_heterocycle")]

        topo_dist = Counter(m.get("topology") for m in monomers)
        film_pos = sum(1 for p in paired if p.get("film_positive", 0) > 0)
        film_neg = sum(1 for p in paired if p.get("film_negative", 0) > 0)

        return (
            f"═══ COF 知识图谱概览 ═══\n"
            f"单体节点: {len(monomers)} (醛={len(ald)}, 胺={len(am)})\n"
            f"  含氟醛: {len(f_ald)}, 含氟胺: {len(f_am)}\n"
            f"  含 N 杂环: {len(n_hetero)}\n"
            f"  拓扑分布: {dict(topo_dist)}\n"
            f"文献节点: {len(lit)}\n"
            f"配对边: {len(paired)} (成膜 {film_pos}, 不成膜 {film_neg})\n"
        )

    # ═══════════════════════════════════════════
    # 单体查询
    # ═══════════════════════════════════════════
    def monomer_info(self, smi: str) -> str:
        """查询单体详情。"""
        can = _canon(smi)
        if can not in self.G:
            return f"未找到单体: {smi}"

        d = dict(self.G.nodes[can])
        # 获取邻居配对
        neighbors = []
        for nb in self.G.neighbors(can):
            edge = self.G.get_edge_data(can, nb)
            if edge and edge.get("edge_type") == "PAIRED_WITH":
                nb_d = dict(self.G.nodes[nb])
                neighbors.append({
                    "smiles": nb,
                    "monomer_type": nb_d.get("monomer_type"),
                    "topology": nb_d.get("topology"),
                    "has_fluorine": nb_d.get("has_fluorine"),
                    "n_literatures": nb_d.get("n_literatures", 0),
                    "pair_count": edge.get("count", 0),
                    "film_ratio": edge.get("film_ratio", 0),
                })
        neighbors.sort(key=lambda x: x["pair_count"], reverse=True)

        lines = [
            f"═══ 单体: {can[:60]} ═══",
            f"类型: {d.get('monomer_type')} | 拓扑: {d.get('topology')} | "
            f"醛基数: {d.get('n_aldehyde')} | 胺基数: {d.get('n_amine')}",
            f"含氟: {d.get('has_fluorine')} (F原子数: {d.get('n_fluorine')}) | "
            f"N杂环: {d.get('has_n_heterocycle')} | 芳环: {d.get('n_aromatic_rings')}",
            f"分子量: {d.get('mw')}",
            f"文献数: {d.get('n_literatures')} | 成膜: {d.get('film_positive')} | "
            f"不成膜: {d.get('film_negative')} | 成膜率: {d.get('film_rate', 0):.1%}",
            f"配对邻居数: {d.get('n_partners')}",
            f"\nTop 10 配对邻居:",
        ]
        for nb in neighbors[:10]:
            lines.append(
                f"  {nb['monomer_type']:10s} {nb['topology']:3s} "
                f"{'F' if nb['has_fluorine'] else '非F':4s} "
                f"配对{nb['pair_count']}次 成膜率{nb['film_ratio']:.1%} "
                f"{nb['smiles'][:40]}"
            )
        return "\n".join(lines)

    # ═══════════════════════════════════════════
    # 配对查询
    # ═══════════════════════════════════════════
    def pair_info(self, ald_smi: str, am_smi: str) -> str:
        """查询特定醛-胺配对信息。"""
        can_ald = _canon(ald_smi)
        can_am = _canon(am_smi)
        if can_ald not in self.G or can_am not in self.G:
            return "单体不存在"

        edge = self.G.get_edge_data(can_ald, can_am)
        if not edge or edge.get("edge_type") != "PAIRED_WITH":
            # 检查是否有间接连接 (通过其他文献)
            ald_neighbors = set(self.G.neighbors(can_ald))
            am_neighbors = set(self.G.neighbors(can_am))
            common_lit = ald_neighbors & am_neighbors
            common_lit = [l for l in common_lit
                         if self.G.nodes[l].get("node_type") == "literature"]
            if common_lit:
                return (
                    f"该配对在图中无直接边, 但两单体共同出现在 "
                    f"{len(common_lit)} 篇文献中 (可能无直接配对待定)"
                )
            return "该配对无已知文献记录 (推荐候选!)"

        ald_d = dict(self.G.nodes[can_ald])
        am_d = dict(self.G.nodes[can_am])
        return (
            f"═══ 配对 ═══\n"
            f"醛: {can_ald[:40]} ({ald_d.get('topology')}, {'F' if ald_d.get('has_fluorine') else '非F'})\n"
            f"胺: {can_am[:40]} ({am_d.get('topology')}, {'F' if am_d.get('has_fluorine') else '非F'})\n"
            f"文献数: {edge.get('n_literatures')} | 总配对: {edge.get('count')}\n"
            f"成膜: {edge.get('film_positive')} | 不成膜: {edge.get('film_negative')} | "
            f"成膜率: {edge.get('film_ratio', 0):.1%}\n"
            f"来源文献: {edge.get('literature_ids', [])[:5]}"
        )

    # ═══════════════════════════════════════════
    # 属性过滤 + 聚合
    # ═══════════════════════════════════════════
    def filter_pairs(
        self,
        ald_has_f: Optional[bool] = None,
        am_has_f: Optional[bool] = None,
        ald_has_n: Optional[bool] = None,
        am_has_n: Optional[bool] = None,
        ald_topo: Optional[str] = None,
        am_topo: Optional[str] = None,
        min_film_count: int = 1,
        top_n: int = 20,
    ) -> str:
        """属性过滤配对, 返回统计 + Top N。"""
        results = []
        for edge in self.edges_data:
            ald = self.nodes_data.get(edge["source"], {})
            am = self.nodes_data.get(edge["target"], {})
            if ald.get("monomer_type") != "aldehyde":
                ald, am = am, ald

            if ald_has_f is not None and ald.get("has_fluorine") != ald_has_f:
                continue
            if am_has_f is not None and am.get("has_fluorine") != am_has_f:
                continue
            if ald_has_n is not None and ald.get("has_n_heterocycle") != ald_has_n:
                continue
            if am_has_n is not None and am.get("has_n_heterocycle") != am_has_n:
                continue
            if ald_topo is not None and ald.get("topology") != ald_topo:
                continue
            if am_topo is not None and am.get("topology") != am_topo:
                continue
            if edge["count"] < min_film_count:
                continue

            results.append(edge)

        total_pos = sum(r["film_positive"] for r in results)
        total_neg = sum(r["film_negative"] for r in results)
        total_labeled = total_pos + total_neg
        film_rate = total_pos / total_labeled if total_labeled > 0 else 0

        lines = [
            f"═══ 过滤结果 ═══",
            f"条件: aldF={ald_has_f} amF={am_has_f} aldN={ald_has_n} amN={am_has_n} "
            f"aldTopo={ald_topo} amTopo={am_topo}",
            f"匹配配对: {len(results)}",
            f"总成膜率: {film_rate:.1%} ({total_pos}/{total_labeled})",
            f"\nTop {min(top_n, len(results))}:",
        ]
        results.sort(key=lambda x: x["count"], reverse=True)
        for i, r in enumerate(results[:top_n]):
            lines.append(
                f"  [{i+1:2d}] {r['source'][:30]} + {r['target'][:30]} "
                f"文献{r['n_literatures']} 成膜率{r['film_ratio']:.1%}"
            )
        return "\n".join(lines)

    # ═══════════════════════════════════════════
    # 推荐未尝试配对
    # ═══════════════════════════════════════════
    def recommend(self, ald_smi: str, top_n: int = 10) -> str:
        """对给定醛, 推荐未尝试的胺配对 (基于共同邻居/相似结构)。"""
        can_ald = _canon(ald_smi)
        if can_ald not in self.G:
            return f"未找到醛: {ald_smi}"

        ald_d = dict(self.G.nodes[can_ald])
        if ald_d.get("monomer_type") != "aldehyde":
            return "该单体不是醛"

        known_amines = set()
        for nb in self.G.neighbors(can_ald):
            edge = self.G.get_edge_data(can_ald, nb)
            if edge and edge.get("edge_type") == "PAIRED_WITH":
                known_amines.add(nb)

        # 从已知配对的胺出发, 找它们的邻居 (结构相似的胺)
        candidates: Dict[str, float] = {}
        for known_am in list(known_amines)[:20]:
            am_d = dict(self.G.nodes[known_am])
            for nb in self.G.neighbors(known_am):
                nb_d = dict(self.G.nodes[nb])
                if (nb_d.get("monomer_type") == "amine"
                        and nb_d.get("n_amine", 0) >= 2
                        and nb not in known_amines
                        and nb != can_ald):
                    # 分数: 共享邻居数 + 成膜率加成
                    shared = len(set(self.G.neighbors(can_ald)) & set(self.G.neighbors(nb)))
                    score = shared + nb_d.get("film_rate", 0) * 3
                    candidates[nb] = max(candidates.get(nb, 0), score)

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        lines = [
            f"═══ 推荐配对 (醛: {can_ald[:50]}) ═══",
            f"已知胺配对: {len(known_amines)}",
            f"候选胺: {len(candidates)}",
            f"\nTop {min(top_n, len(sorted_candidates))} 推荐:",
        ]
        for i, (am_smi, score) in enumerate(sorted_candidates[:top_n]):
            am_d = dict(self.G.nodes[am_smi])
            lines.append(
                f"  [{i+1:2d}] {am_smi[:45]} "
                f"({am_d.get('topology')}, {'F' if am_d.get('has_fluorine') else '非F'}, "
                f"文献{am_d.get('n_literatures')}, 成膜率{am_d.get('film_rate', 0):.1%}) "
                f"score={score:.1f}"
            )
        return "\n".join(lines)

    # ═══════════════════════════════════════════
    # 自然语言查询 (简易关键词)
    # ═══════════════════════════════════════════
    def ask(self, question: str) -> str:
        """自然语言查询 — 关键词匹配 + 规则路由。"""
        q = question.lower()

        # /stats
        if any(w in q for w in ["概览", "统计", "总结", "summary", "stats", "/stats"]):
            return self.stats()

        # /filter
        if any(w in q for w in ["含氟", "氟", "f", "非f", "非氟"]):
            ald_f = None
            am_f = None
            if "含氟醛" in q or "f醛" in q or "f-醛" in q:
                ald_f = True
            if "非氟醛" in q or "非f醛" in q:
                ald_f = False
            if "含氟胺" in q or "f胺" in q or "f-胺" in q:
                am_f = True
            if "非氟胺" in q or "非f胺" in q:
                am_f = False
            # 默认: 含氟 vs 非氟 交叉
            if ald_f is None and am_f is None:
                if "非氟" in q or "非f" in q:
                    am_f = False
                    ald_f = False
                else:
                    ald_f = True

            ald_n = True if "三嗪" in q or "吡啶" in q or "杂环醛" in q else None
            am_n = True if "三嗪胺" in q or "杂环胺" in q else None

            return self.filter_pairs(
                ald_has_f=ald_f, am_has_f=am_f,
                ald_has_n=ald_n, am_has_n=am_n,
            )

        # /monomer
        if "单体" in q or "monomer" in q.lower():
            # 尝试提取 SMILES
            import re
            smiles_match = re.findall(r'[A-Za-z0-9\[\]\(\)\=\#\/\\@\-\+]{10,}', q)
            if smiles_match:
                return self.monomer_info(smiles_match[0])
            return "请提供 SMILES 字符串"

        # /recommend
        if any(w in q for w in ["推荐", "未尝试", "候选", "新配对", "recom"]):
            import re
            smiles_match = re.findall(r'[A-Za-z0-9\[\]\(\)\=\#\/\\@\-\+]{10,}', q)
            if smiles_match:
                return self.recommend(smiles_match[0])
            return "请提供醛的 SMILES"

        # /pair
        if "配对" in q or "pair" in q.lower():
            import re
            smiles_list = re.findall(r'[A-Za-z0-9\[\]\(\)\=\#\/\\@\-\+]{10,}', q)
            if len(smiles_list) >= 2:
                return self.pair_info(smiles_list[0], smiles_list[1])
            return "请提供醛和胺的 SMILES"

        # fallback
        return (
            f"无法解析问题: {question}\n"
            "支持的命令:\n"
            "  /stats     — 图谱概览\n"
            "  /monomer <SMILES> — 单体详情\n"
            "  /pair <醛SMILES> <胺SMILES> — 配对查询\n"
            "  /filter 含氟醛+非氟胺 — 属性过滤\n"
            "  /recommend <醛SMILES> — 推荐未尝试配对\n"
            "  或直接输入自然语言问题 (关键词匹配)"
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="COF 知识图谱查询")
    parser.add_argument("query", nargs="*", help="自然语言查询 (或用 /command)")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    args = parser.parse_args()

    gq = GraphQuery()

    if args.interactive:
        print("\nCOF 知识图谱查询 (输入 /help 查看命令, 输入 quit 退出)\n")
        while True:
            try:
                q = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            if q == "/help":
                print("命令: /stats /monomer <smi> /pair <smi1> <smi2>")
                print("      /filter <条件> /recommend <smi> /ask <问题>")
                continue
            print(gq.ask(q))
            print()
    else:
        query = " ".join(args.query) if args.query else "/stats"
        print(gq.ask(query))


if __name__ == "__main__":
    main()
