"""COF 知识图谱交互式可视化 — pyvis 网页渲染 + 搜索面板 + 文献详情。

输出:
  - graphrag图谱辅助/graph_visual.html  (全量图, 可交互浏览)
  - graphrag图谱辅助/graph_visual_core.html (核心子图, Top 150 节点)

颜色:
  蓝色系 — 醛单体 (深蓝=F/N杂环, 浅蓝=非F)
  红色系 — 胺单体 (深红=F/N杂环, 浅红=非F)
  紫色   — 双功能单体
  绿色边 — 成膜率>50%
  橙色边 — 成膜率≤50%
"""
import json
import os
import sys
from collections import defaultdict

import networkx as nx
from pyvis.network import Network

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _node_color(node: dict) -> str:
    mtype = node.get("monomer_type", "")
    has_f = node.get("has_fluorine", False)
    has_n = node.get("has_n_heterocycle", False)

    if mtype == "aldehyde":
        if has_n:
            return "#1a5276"
        return "#2980b9" if has_f else "#85c1e9"
    elif mtype == "amine":
        if has_n:
            return "#922b21"
        return "#c0392b" if has_f else "#f1948a"
    elif mtype == "dual":
        return "#8e44ad"
    return "#95a5a6"


def _node_size(node: dict) -> int:
    n_lit = node.get("n_literatures", 0)
    n_part = node.get("n_partners", 0)
    return max(8, min(40, 6 + n_lit * 1.5 + n_part * 0.5))


def _node_title(node: dict, node_id: str) -> str:
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
    fr = edge.get("film_ratio", 0)
    if fr >= 0.5:
        return "#27ae60"
    if fr > 0:
        return "#f39c12"
    return "#bdc3c7"


def _edge_width(edge: dict) -> float:
    return min(5, 0.5 + edge.get("n_literatures", 1) * 0.8)


def _build_literature_index(nodes_data: dict) -> dict:
    lit_index = {}
    for nid, nd in nodes_data.items():
        if nd.get("node_type") == "literature":
            lit_index[nid] = {
                "literature_id": nid,
                "solvent": str(nd.get("solvent", ""))[:300],
                "reaction_temperature": str(nd.get("reaction_temperature", ""))[:200],
                "synthesis_mode": str(nd.get("synthesis_mode", ""))[:200],
                "interface_type": str(nd.get("interface_type", ""))[:200],
                "fluorine_monomer": str(nd.get("fluorine_monomer", ""))[:150],
                "film_field": str(nd.get("film_field", ""))[:500],
            }
    return lit_index


def _build_monomer_lit_map(G, monomers: set) -> dict:
    mono_to_lits = defaultdict(list)
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") != "APPEARS_IN":
            continue
        if u in monomers and v not in monomers:
            mono_to_lits[u].append(v)
        elif v in monomers and u not in monomers:
            mono_to_lits[v].append(u)
    return {k: sorted(v) for k, v in mono_to_lits.items()}


def _inject_search_panel(html_path: str, monomer_to_lits: dict,
                         lit_index: dict, nodes_data: dict):
    mono_js = {}
    for smi, lit_ids in monomer_to_lits.items():
        nd = nodes_data.get(smi, {})
        mono_js[smi[:60]] = {
            "full_smi": smi,
            "label": nd.get("label", smi[:30]),
            "mtype": nd.get("monomer_type", "?"),
            "topo": nd.get("topology", "?"),
            "has_f": nd.get("has_fluorine", False),
            "has_n": nd.get("has_n_heterocycle", False),
            "n_arom": nd.get("n_aromatic_rings", 0),
            "mw": nd.get("mw", 0),
            "n_lit": nd.get("n_literatures", 0),
            "film_rate": nd.get("film_rate", 0),
            "lits": lit_ids,
        }

    mono_json = json.dumps(mono_js, ensure_ascii=False)
    lit_json = json.dumps(lit_index, ensure_ascii=False)

    panel_html = f"""<style>
#search-panel {{
    position:fixed; top:0; right:0; width:400px; height:100vh;
    background:#fff; border-left:2px solid #ddd; z-index:1000;
    display:flex; flex-direction:column; font-family:Arial,sans-serif;
    box-shadow:-4px 0 20px rgba(0,0,0,0.1);
}}
#search-panel.collapsed {{ right:-400px; }}
#panel-toggle {{
    position:fixed; top:10px; right:410px; z-index:1001;
    background:#2980b9; color:#fff; border:none; border-radius:4px 0 0 4px;
    padding:8px 12px; cursor:pointer; font-size:14px;
}}
#panel-toggle.collapsed {{ right:0; border-radius:4px; }}
#search-box {{
    padding:12px; border-bottom:1px solid #eee;
}}
#search-box input {{
    width:100%; padding:10px; border:2px solid #ddd; border-radius:6px;
    font-size:14px; outline:none; box-sizing:border-box;
}}
#search-box input:focus {{ border-color:#2980b9; }}
#search-results {{
    flex:1; overflow-y:auto; padding:8px;
}}
#search-results .result-item {{
    padding:10px; border-bottom:1px solid #f0f0f0; cursor:pointer;
    border-radius:4px; transition:background 0.15s;
}}
#search-results .result-item:hover {{ background:#eaf2f8; }}
#search-results .result-item.selected {{ background:#d4e6f1; border-left:3px solid #2980b9; }}
.result-meta {{ font-size:11px; color:#666; margin-top:4px; }}
.result-meta span {{ margin-right:8px; }}
.f-badge {{ padding:1px 6px; border-radius:3px; font-size:10px; font-weight:bold; }}
.f-badge.f-true {{ background:#e74c3c; color:#fff; }}
.f-badge.f-false {{ background:#95a5a6; color:#fff; }}
.n-badge {{ padding:1px 6px; border-radius:3px; font-size:10px; font-weight:bold; }}
.n-badge.n-true {{ background:#8e44ad; color:#fff; }}
.n-badge.n-false {{ background:#95a5a6; color:#fff; }}

#detail-panel {{
    position:fixed; top:0; right:400px; width:420px; height:100vh;
    background:#fefefe; border-left:1px solid #ccc; z-index:999;
    overflow-y:auto; font-family:Arial,sans-serif;
    display:none; box-shadow:-2px 0 15px rgba(0,0,0,0.08);
}}
#detail-header {{ padding:14px; background:#2c3e50; color:#fff; position:sticky; top:0; z-index:1; }}
#detail-header button {{
    float:right; background:none; border:1px solid rgba(255,255,255,0.5); color:#fff;
    border-radius:3px; cursor:pointer; padding:4px 10px; font-size:12px;
}}
#detail-body {{ padding:14px; }}
.detail-section {{ margin-bottom:16px; }}
.detail-section h4 {{
    font-size:13px; color:#2c3e50; border-bottom:2px solid #2980b9;
    padding-bottom:4px; margin-bottom:8px;
}}
.detail-table {{ width:100%; font-size:12px; border-collapse:collapse; }}
.detail-table td {{ padding:6px 8px; border-bottom:1px solid #eee; vertical-align:top; }}
.detail-table td:first-child {{ font-weight:bold; color:#555; width:100px; white-space:nowrap; }}
.lit-link {{ color:#2980b9; cursor:pointer; text-decoration:underline; }}

.mono-lit-item {{
    padding:8px 10px; border:1px solid #e0e0e0; margin:4px 0; border-radius:4px;
    cursor:pointer; transition:background 0.15s; font-size:12px;
}}
.mono-lit-item:hover {{ background:#eaf2f8; }}

.no-results {{ text-align:center; color:#999; padding:30px; font-size:14px; }}

/* Force full height for all ancestors */
html, body {{
    height: 100%;
    margin: 0;
    padding: 0;
    overflow: hidden;
}}

/* Override pyvis default layout — make room for side panel */
#mynetwork {{
    margin-right: 400px !important;
    width: calc(100% - 400px) !important;
    height: 100vh !important;
    float: none !important;
    position: absolute !important;
    top: 0;
    left: 0;
}}
</style>

<button id="panel-toggle" onclick="togglePanel()">◀ 搜索</button>

<div id="search-panel">
    <div id="search-box">
        <input type="text" id="search-input" placeholder="搜索单体: SMILES / 标签 / 属性..."
               oninput="doSearch()" autofocus>
    </div>
    <div id="search-results"><div class="no-results">输入关键词搜索单体</div></div>
</div>

<div id="detail-panel">
    <div id="detail-header">
        <span id="detail-title">单体详情</span>
        <button onclick="closeDetail()">✕</button>
    </div>
    <div id="detail-body"></div>
</div>

<script>
// ── Data ──
var MONOMERS = {mono_json};
var LITERATURES = {lit_json};

// ── Panel toggle ──
var panelVisible = true;
function togglePanel() {{
    var p = document.getElementById('search-panel');
    var btn = document.getElementById('panel-toggle');
    var netEl = document.getElementById('mynetwork');
    if (panelVisible) {{
        p.classList.add('collapsed'); btn.classList.add('collapsed');
        btn.textContent = '▶ 搜索';
        netEl.style.marginRight = '0';
        netEl.style.width = '100%';
    }} else {{
        p.classList.remove('collapsed'); btn.classList.remove('collapsed');
        btn.textContent = '◀ 搜索';
        netEl.style.marginRight = '400px';
        netEl.style.width = 'calc(100% - 400px)';
    }}
    panelVisible = !panelVisible;
    if (window.network) window.network.fit();
}}

// ── Search ──
function doSearch() {{
    var q = document.getElementById('search-input').value.toLowerCase().trim();
    var container = document.getElementById('search-results');
    if (!q) {{
        container.innerHTML = '<div class="no-results">输入关键词搜索单体</div>';
        resetHighlights();
        return;
    }}
    var results = [];
    for (var key in MONOMERS) {{
        var m = MONOMERS[key];
        var text = (m.full_smi + ' ' + m.label + ' ' + m.mtype + ' ' + m.topo).toLowerCase();
        if (text.indexOf(q) >= 0) results.push(m);
    }}
    if (results.length === 0) {{
        container.innerHTML = '<div class="no-results">无匹配结果</div>';
        resetHighlights();
        return;
    }}
    // sort by n_lit desc
    results.sort(function(a,b) {{ return b.n_lit - a.n_lit; }});
    var html = '';
    var matchSmiSet = new Set();
    for (var i = 0; i < Math.min(results.length, 80); i++) {{
        var m = results[i];
        matchSmiSet.add(m.full_smi);
        var fCls = m.has_f ? 'f-true' : 'f-false';
        var nCls = m.has_n ? 'n-true' : 'n-false';
        var smiShort = m.full_smi.length > 55 ? m.full_smi.substring(0, 52) + '...' : m.full_smi;
        html += '<div class="result-item" onclick="selectMonomer(this.getAttribute(\'data-smi\'))" data-smi="' +
                m.full_smi.replace(/"/g, '&quot;').replace(/'/g, '&#39;') + '">' +
                '<b>' + m.label + '</b> [' + m.mtype + ' | ' + m.topo + ']<br>' +
                '<span style="font-size:11px;color:#888">' + smiShort + '</span>' +
                '<div class="result-meta">' +
                '<span>文献:' + m.n_lit + '</span>' +
                '<span>成膜率:' + (m.film_rate * 100).toFixed(0) + '%</span>' +
                '<span class="f-badge ' + fCls + '">' + (m.has_f ? 'F' : '非F') + '</span>' +
                '<span class="n-badge ' + nCls + '">' + (m.has_n ? 'N杂环' : '无N杂') + '</span>' +
                '</div></div>';
    }}
    container.innerHTML = html;
    highlightNodes(matchSmiSet);
}}

// ── Highlight matching nodes in graph ──
var currentHighlight = new Set();
function highlightNodes(smiSet) {{
    if (!window.network) return;
    // Reset previous
    currentHighlight.forEach(function(id) {{
        window.monoNodes.forEach(function(n) {{
            if (window.nodeDataMap[n.id] === id || n.id === id) {{
                var color = window.nodeOrigColor[n.id] || '#95a5a6';
                network.body.data.nodes.update({{id: n.id, color: color, opacity: 1}});
            }}
        }});
    }});
    currentHighlight = new Set(smiSet);
    // Dim non-matching
    window.monoNodes.forEach(function(n) {{
        var smi = window.nodeDataMap[n.id] || n.id;
        if (smiSet.has(smi)) {{
            network.body.data.nodes.update({{id: n.id, opacity: 1, borderWidth: 4}});
        }} else {{
            network.body.data.nodes.update({{id: n.id, opacity: 0.25, borderWidth: 1}});
        }}
    }});
}}

function resetHighlights() {{
    if (!window.network) return;
    window.monoNodes.forEach(function(n) {{
        network.body.data.nodes.update({{id: n.id, opacity: 1, borderWidth: n.borderWidth || 1}});
    }});
}}

// ── Monomer selection ──
var selectedSmi = null;
function selectMonomer(smi) {{
    selectedSmi = smi;
    // highlight in search results
    var items = document.querySelectorAll('#search-results .result-item');
    items.forEach(function(el) {{ el.classList.remove('selected'); }});
    // find matching item by iterating (avoids CSS selector escaping issues)
    items.forEach(function(el) {{
        if (el.getAttribute('data-smi') === smi) el.classList.add('selected');
    }});

    // focus & highlight in graph
    if (window.network) {{
        focusNodeInGraph(smi);
    }}

    showDetail(smi);
}}

function focusNodeInGraph(smi) {{
    var nodeId = null;
    window.monoNodes.forEach(function(n) {{
        if (window.nodeDataMap[n.id] === smi || n.id === smi) nodeId = n.id;
    }});
    if (nodeId) {{
        var highlight = {{}}; highlight[nodeId] = true;
        currentHighlight.forEach(function(s) {{
            window.monoNodes.forEach(function(n) {{
                if (window.nodeDataMap[n.id] === s || n.id === s) highlight[n.id] = true;
            }});
        }});
        window.monoNodes.forEach(function(n) {{
            if (highlight[n.id]) {{
                network.body.data.nodes.update({{id: n.id, opacity: 1, borderWidth: 4}});
            }} else {{
                network.body.data.nodes.update({{id: n.id, opacity: 0.15, borderWidth: 1}});
            }}
        }});
        window.network.selectNodes([nodeId]);
        window.network.focus(nodeId, {{scale: 1.2, animation: true}});
    }}
}}

// ── Detail panel ──
function showDetail(smi) {{
    var m = MONOMERS[smi.substring(0, 60)];
    if (!m) {{
        // try longer key
        for (var k in MONOMERS) {{
            if (MONOMERS[k].full_smi === smi) {{ m = MONOMERS[k]; break; }}
        }}
    }}
    if (!m) return;

    var panel = document.getElementById('detail-panel');
    var body = document.getElementById('detail-body');
    document.getElementById('detail-title').textContent = '单体: ' + m.label;
    panel.style.display = 'block';

    var fStr = m.has_f ? '是' : '否';
    var nStr = m.has_n ? '是' : '否';
    var smiShort = m.full_smi.length > 60 ? m.full_smi.substring(0, 57) + '...' : m.full_smi;

    var html = '<div class="detail-section"><h4>基本属性</h4>' +
        '<table class="detail-table">' +
        '<tr><td>SMILES</td><td style="word-break:break-all;font-family:monospace;font-size:11px">' + smiShort + '</td></tr>' +
        '<tr><td>类型</td><td>' + m.mtype + '</td></tr>' +
        '<tr><td>拓扑</td><td>' + m.topo + '</td></tr>' +
        '<tr><td>含氟</td><td>' + fStr + '</td></tr>' +
        '<tr><td>N杂环</td><td>' + nStr + '</td></tr>' +
        '<tr><td>芳环数</td><td>' + m.n_arom + '</td></tr>' +
        '<tr><td>分子量</td><td>' + m.mw.toFixed(1) + '</td></tr>' +
        '<tr><td>文献数</td><td>' + m.n_lit + '</td></tr>' +
        '<tr><td>成膜率</td><td>' + (m.film_rate * 100).toFixed(1) + '%</td></tr>' +
        '</table></div>';

    html += '<div class="detail-section"><h4>来源文章 (' + m.lits.length + ' 篇)</h4>';

    for (var i = 0; i < m.lits.length; i++) {{
        var lid = m.lits[i];
        var lit = LITERATURES[lid];
        var lidShort = lid.length > 80 ? lid.substring(0, 77) + '...' : lid;
        if (lit) {{
            html += '<div class="mono-lit-item" onclick="showLitDetail(this.getAttribute(\'data-lid\'))" data-lid="' +
                lid.replace(/"/g, '&quot;').replace(/'/g, '&#39;') + '">' +
                '<b>#' + (i + 1) + '</b> ' + lidShort + '<br>' +
                '<span style="font-size:11px;color:#666">溶剂: ' +
                (lit.solvent || '?').substring(0, 60) + ' | 温度: ' +
                (lit.reaction_temperature || '?').substring(0, 30) + '</span></div>';
        }} else {{
            html += '<div class="mono-lit-item" style="color:#999">' +
                '<b>#' + (i + 1) + '</b> ' + lidShort + ' (无结构化数据)</div>';
        }}
    }}
    html += '</div>';

    body.innerHTML = html;
}}

function showLitDetail(lid) {{
    var lit = LITERATURES[lid];
    if (!lit) return;

    var panel = document.getElementById('detail-panel');
    var body = document.getElementById('detail-body');
    document.getElementById('detail-title').textContent = '文献详情';
    panel.style.display = 'block';

    var html = '<div class="detail-section"><h4>基本信息</h4>' +
        '<table class="detail-table">' +
        '<tr><td>文献 ID</td><td style="word-break:break-all;font-size:11px">' + lid + '</td></tr>' +
        '</table></div>';

    var fields = [
        ['溶剂', lit.solvent || '(无)'],
        ['反应温度', lit.reaction_temperature || '(无)'],
        ['合成方式', lit.synthesis_mode || '(无)'],
        ['界面类型', lit.interface_type || '(无)'],
        ['含氟单体', lit.fluorine_monomer || '(无)'],
        ['成膜/结晶/氟描述', lit.film_field || '(无)'],
    ];

    html += '<div class="detail-section"><h4>结构化信息 (21 字段关键项)</h4>' +
        '<table class="detail-table">';
    for (var i = 0; i < fields.length; i++) {{
        var val = fields[i][1].replace(/</g, '&lt;').replace(/>/g, '&gt;');
        html += '<tr><td>' + fields[i][0] + '</td><td>' + val + '</td></tr>';
    }}
    html += '</table></div>';

    if (selectedSmi) {{
        html += '<div class="detail-section">' +
            '<span class="lit-link" onclick="showDetail(selectedSmi)">&larr; 返回单体: ' +
            (MONOMERS[selectedSmi.substring(0,60)] || {{}}).label + '</span></div>';
    }}

    body.innerHTML = html;
}}

function closeDetail() {{
    document.getElementById('detail-panel').style.display = 'none';
    resetHighlights();
}}

// ── Init after pyvis loads ──
window.addEventListener('load', function() {{
    setTimeout(function() {{
        if (typeof network !== 'undefined') {{
            window.network = network;
            window.monoNodes = network.body.data.nodes.get().filter(function(n) {{
                return !n.id.startsWith('edge_');
            }});
            window.nodeDataMap = {{}};
            window.nodeOrigColor = {{}};
            window.monoNodes.forEach(function(n) {{
                window.nodeOrigColor[n.id] = n.color;
                // Check if n.id is a SMILES matching a MONOMERS key
                for (var k in MONOMERS) {{
                    if (MONOMERS[k].full_smi === n.id) {{
                        window.nodeDataMap[n.id] = MONOMERS[k].full_smi;
                        break;
                    }}
                }}
                if (!window.nodeDataMap[n.id]) {{
                    // Try partial match: SMILES prefix
                    for (var k in MONOMERS) {{
                        if (n.id.indexOf(k) >= 0 && k.length > 10) {{
                            window.nodeDataMap[n.id] = MONOMERS[k].full_smi;
                            break;
                        }}
                    }}
                }}
                if (!window.nodeDataMap[n.id]) window.nodeDataMap[n.id] = n.id;
            }});
            // Set size for side panel layout
            var netEl = document.getElementById('mynetwork');
            if (netEl) {{
                netEl.style.marginRight = '400px';
                netEl.style.width = 'calc(100% - 400px)';
                netEl.style.height = '100vh';
            }}
            network.fit();
        }}
    }}, 500);
}});
</script>"""

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("</body>", panel_html + "</body>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


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

    monomers = {n for n, d in G.nodes(data=True) if d.get("node_type") == "monomer"}
    monomer_G = G.subgraph(monomers).copy()

    for u, v, d in list(monomer_G.edges(data=True)):
        if d.get("edge_type") != "PAIRED_WITH":
            monomer_G.remove_edge(u, v)
    monomer_G.remove_nodes_from(list(nx.isolates(monomer_G)))

    print(f"可视化图: {monomer_G.number_of_nodes()} 节点, "
          f"{monomer_G.number_of_edges()} 边")

    # Build literature index & monomer→lit mapping
    lit_index = _build_literature_index(nodes_data)
    monomer_to_lits = _build_monomer_lit_map(G, monomers)

    # ═══════════════════════════════════════
    # 1. 核心子图 (Top 150 节点按度数)
    # ═══════════════════════════════════════
    degrees = dict(monomer_G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:150]
    core_G = monomer_G.subgraph(top_nodes).copy()

    core_net = Network(height="100%", width="100%", bgcolor="#f8f9fa",
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
    _inject_search_panel(core_path, monomer_to_lits, lit_index, nodes_data)
    print(f"核心子图 (150节点): {core_path}")

    # ═══════════════════════════════════════
    # 2. 全量图 (全部单体)
    # ═══════════════════════════════════════
    full_net = Network(height="100%", width="100%", bgcolor="#ffffff",
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
    _inject_search_panel(full_path, monomer_to_lits, lit_index, nodes_data)
    print(f"全量图: {full_path}")

    return core_path, full_path


def _add_legend(html_path: str):
    legend = """
    <div style="position:fixed;top:10px;left:10px;background:rgba(255,255,255,0.95);
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
