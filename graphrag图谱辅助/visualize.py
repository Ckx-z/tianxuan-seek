"""COF 知识图谱交互式可视化 — pyvis 网页渲染 + 搜索面板 + 文献详情 + 结构图。

输出:
  - graphrag图谱辅助/graph_visual.html  (全量图)
  - graphrag图谱辅助/graph_visual_core.html (核心子图, Top 150 节点)
"""
import json
import os
import sys
import hashlib
from collections import defaultdict
from pathlib import Path

import networkx as nx
import yaml
from pyvis.network import Network
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolDraw2DSVG

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chemistry.monomer import _BUILTIN_MONOMERS


# ── 反向名称字典: SMILES → 最佳名称 (优先缩写) ──
_NAME_PRIORITY = {}  # SMILES → name, 短的优先
for _name, _smi in _BUILTIN_MONOMERS.items():
    if not _smi:
        continue
    _smi = Chem.CanonSmiles(_smi) if Chem.MolFromSmiles(_smi) else _smi
    if _smi not in _NAME_PRIORITY or len(_name) < len(_NAME_PRIORITY[_smi]):
        _NAME_PRIORITY[_smi] = _name


def _chemical_name(smi: str) -> str:
    """SMILES → 最佳英文名称 (缩写优先)。"""
    try:
        csmi = Chem.CanonSmiles(smi)
    except Exception:
        return ""
    return _NAME_PRIORITY.get(csmi, "")


def _render_svg(smi: str, size: tuple = (240, 150)) -> str:
    """SMILES → 2D 结构 SVG 字符串。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        Chem.Kekulize(mol)
    except Exception:
        pass
    drawer = MolDraw2DSVG(size[0], size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


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


def _load_yaml_safe(yaml_path: str) -> dict:
    """安全读取 YAML，失败返回空字典。"""
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _build_literature_full(yaml_dir: str, lit_ids_set: set) -> dict:
    """从 YAML 按需读取文献完整 22 字段 (仅核心图关联文献)。"""
    full_db = {}
    yaml_dir_path = Path(yaml_dir)
    yaml_files = {f.stem: str(f) for f in yaml_dir_path.glob("*.yaml")}

    for lid in lit_ids_set:
        if lid not in yaml_files:
            continue
        data = _load_yaml_safe(yaml_files[lid])
        if not data:
            continue
        full_db[lid] = {
            "journal": str(data.get("journal", ""))[:300],
            "system": str(data.get("system", ""))[:300],
            "reagent": str(data.get("reagent", ""))[:300],
            "catalyst": str(data.get("catalyst", ""))[:200],
            "solvent": str(data.get("solvent", ""))[:300],
            "reaction_temperature": str(data.get("reaction_temperature", ""))[:200],
            "synthesis_mode": str(data.get("synthesis_mode", ""))[:200],
            "synthesis_route": str(data.get("synthesis_route", ""))[:300],
            "interface_type": str(data.get("interface_type", ""))[:200],
            "annealing_conditions": str(data.get("annealing_conditions", ""))[:200],
            "schiff_base_kinetics": str(data.get("schiff_base_kinetics", ""))[:300],
            "fluorine_effects": str(data.get("fluorine_effects", ""))[:300],
            "fluorine_monomer": str(data.get("fluorine_monomer", ""))[:150],
            "film_crystallinity_fluorine": str(data.get("film_crystallinity_fluorine", ""))[:500],
            "adsorption_mechanism": str(data.get("adsorption_mechanism", ""))[:300],
            "computational_methods": str(data.get("computational_methods", ""))[:300],
            "conclusion_1": str(data.get("conclusion_1", ""))[:400],
            "conclusion_2": str(data.get("conclusion_2", ""))[:400],
            "conclusion_3": str(data.get("conclusion_3", ""))[:400],
            "innovation": str(data.get("innovation", ""))[:400],
            "methods": str(data.get("methods", ""))[:300],
        }
    return full_db


def _inject_search_panel(html_path: str, monomer_to_lits: dict,
                         lit_full: dict, nodes_data: dict,
                         include_svg: bool = True,
                         svg_smiles: set = None):
    """注入搜索面板、JSON 数据、JS 逻辑。include_svg 只对核心图开启。"""
    svg_smiles = svg_smiles or set()
    # 构建 MONOMERS JS 数据
    mono_js = {}
    for smi, lit_ids in monomer_to_lits.items():
        nd = nodes_data.get(smi, {})
        entry = {
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
            "chem_name": _chemical_name(smi),
            "lits": lit_ids,
        }
        if include_svg and smi in svg_smiles:
            try:
                entry["svg"] = _render_svg(smi)
            except Exception:
                entry["svg"] = ""
        mono_js[smi[:60]] = entry

    mono_json = json.dumps(mono_js, ensure_ascii=False)
    lit_json = json.dumps(lit_full, ensure_ascii=False)

    # ── 渲染文献数量统计 ──
    svg_count = sum(1 for v in mono_js.values() if v.get("svg"))
    name_count = sum(1 for v in mono_js.values() if v.get("chem_name"))

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
.result-item {{
    padding:10px; border-bottom:1px solid #f0f0f0; cursor:pointer;
    border-radius:4px; transition:background 0.15s;
}}
.result-item:hover {{ background:#eaf2f8; }}
.result-item.selected {{ background:#d4e6f1; border-left:3px solid #2980b9; }}
.result-meta {{ font-size:11px; color:#666; margin-top:4px; }}
.result-meta span {{ margin-right:8px; }}
.f-badge {{ padding:1px 6px; border-radius:3px; font-size:10px; font-weight:bold; }}
.f-badge.f-true {{ background:#e74c3c; color:#fff; }}
.f-badge.f-false {{ background:#95a5a6; color:#fff; }}
.n-badge {{ padding:1px 6px; border-radius:3px; font-size:10px; font-weight:bold; }}
.n-badge.n-true {{ background:#8e44ad; color:#fff; }}
.n-badge.n-false {{ background:#95a5a6; color:#fff; }}

#detail-panel {{
    position:fixed; top:0; right:400px; width:460px; height:100vh;
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
.detail-table td:first-child {{ font-weight:bold; color:#555; width:90px; white-space:nowrap; }}
.lit-link {{ color:#2980b9; cursor:pointer; text-decoration:underline; }}
.mono-lit-item {{
    padding:8px 10px; border:1px solid #e0e0e0; margin:4px 0; border-radius:4px;
    cursor:pointer; transition:background 0.15s; font-size:12px;
}}
.mono-lit-item:hover {{ background:#eaf2f8; }}
.no-results {{ text-align:center; color:#999; padding:30px; font-size:14px; }}
.chem-structure {{ text-align:center; margin:8px 0; }}
.chem-structure svg {{ max-width:100%; height:auto; }}
.full-lit-table {{ width:100%; font-size:12px; border-collapse:collapse; }}
.full-lit-table td {{ padding:8px 10px; border-bottom:1px solid #eee; vertical-align:top; line-height:1.5; }}
.full-lit-table td:first-child {{ font-weight:bold; color:#3a539b; width:110px; white-space:nowrap; font-size:11px; }}

/* Force full height */
html, body {{
    height: 100%;
    margin: 0;
    padding: 0;
    overflow: hidden;
}}
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
        <input type="text" id="search-input" placeholder="搜索: SMILES / 标签 / film:top / f:yes / topo:C3 ..."
               autofocus>
        <div style="font-size:10px;color:#999;padding:4px 0 0 2px">
          快捷: <b>film:top</b> 成膜率最高 | <b>f:yes</b> 含氟 | <b>n:yes</b> N杂环 |
          <b>type:醛</b> | <b>topo:C3</b>
        </div>
    </div>
    <div id="search-results"><div class="no-results">输入关键词搜索单体</div></div>
</div>

<div id="detail-panel">
    <div id="detail-header">
        <span id="detail-title">单体详情</span>
        <button onclick="closeDetail()">X</button>
    </div>
    <div id="detail-body"></div>
</div>

<script>
var MONOMERS = {mono_json};
var LITERATURES = {lit_json};

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
    try {{
    var q = document.getElementById('search-input').value.toLowerCase().trim();
    var container = document.getElementById('search-results');
    if (!q) {{
        container.innerHTML = '<div class="no-results">输入关键词搜索单体</div>';
        resetHighlights();
        return;
    }}
    var sortMode = 'lit';
    var searchTerm = q;
    var specialCmds = {{
        'film:top': 'film', 'film:best': 'film', 'film:desc': 'film',
        'film:0': 'film0',
        'f:yes': 'f_yes', 'f:no': 'f_no',
        'n:yes': 'n_yes',
        'type:aldehyde': 'type_ald', 'type:amine': 'type_am',
        'topo:c1': 'topo_c1', 'topo:c2': 'topo_c2', 'topo:c3': 'topo_c3', 'topo:c4': 'topo_c4',
        'lit:desc': 'lit'
    }};
    for (var cmd in specialCmds) {{
        if (q === cmd || q.indexOf(cmd + ' ') === 0) {{
            sortMode = specialCmds[cmd];
            searchTerm = q.substring(cmd.length).trim();
            break;
        }}
    }}
    var results = [];
    for (var key in MONOMERS) {{
        var m = MONOMERS[key];
        if (sortMode === 'film') {{ results.push(m); continue; }}
        if (sortMode === 'film0') {{ if (m.film_rate === 0) results.push(m); continue; }}
        if (sortMode === 'f_yes') {{ if (m.has_f) results.push(m); continue; }}
        if (sortMode === 'f_no') {{ if (!m.has_f) results.push(m); continue; }}
        if (sortMode === 'n_yes') {{ if (m.has_n) results.push(m); continue; }}
        if (sortMode === 'type_ald') {{ if (m.mtype === 'aldehyde') results.push(m); continue; }}
        if (sortMode === 'type_am') {{ if (m.mtype === 'amine') results.push(m); continue; }}
        if (sortMode === 'topo_c1' || sortMode === 'topo_c2' ||
            sortMode === 'topo_c3' || sortMode === 'topo_c4') {{
            var t = sortMode.substring(5).toUpperCase();
            if (m.topo === t) results.push(m);
            continue;
        }}
        var text = (m.full_smi + ' ' + m.label + ' ' + m.mtype + ' ' + m.topo).toLowerCase();
        if (searchTerm && text.indexOf(searchTerm) >= 0) results.push(m);
        else if (!searchTerm) results.push(m);
    }}
    if (results.length === 0) {{
        container.innerHTML = '<div class="no-results">无匹配结果</div>';
        resetHighlights();
        return;
    }}
    if (sortMode === 'film') {{
        results.sort(function(a,b) {{ return b.film_rate - a.film_rate; }});
    }} else {{
        results.sort(function(a,b) {{ return b.n_lit - a.n_lit; }});
    }}
    var html = '';
    var matchSmiSet = new Set();
    for (var i = 0; i < Math.min(results.length, 80); i++) {{
        var m = results[i];
        matchSmiSet.add(m.full_smi);
        var fCls = m.has_f ? 'f-true' : 'f-false';
        var nCls = m.has_n ? 'n-true' : 'n-false';
        var smiShort = m.full_smi.length > 55 ? m.full_smi.substring(0, 52) + '...' : m.full_smi;
        html += '<div class="result-item" data-smi="' +
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
    }} catch(e) {{
        var container = document.getElementById('search-results');
        if (container) container.innerHTML = '<div class=\"no-results\" style=\"color:red\">JS Error: ' + e.message + '</div>';
    }}
}}

// ── Highlight (batch) ──
var currentHighlight = new Set();
function highlightNodes(smiSet) {{
    if (!window.network) return;
    var updates = [];
    window.monoNodes.forEach(function(n) {{
        var smi = window.nodeDataMap[n.id] || n.id;
        if (smiSet.has(smi)) {{
            updates.push({{id: n.id, opacity: 1, borderWidth: 4}});
        }} else {{
            updates.push({{id: n.id, opacity: 0.2, borderWidth: 1}});
        }}
    }});
    currentHighlight = new Set(smiSet);
    if (updates.length > 0) network.body.data.nodes.update(updates);
}}

function resetHighlights() {{
    if (!window.network) return;
    var updates = [];
    window.monoNodes.forEach(function(n) {{
        updates.push({{id: n.id, opacity: 1, borderWidth: n.borderWidth || 1}});
    }});
    if (updates.length > 0) network.body.data.nodes.update(updates);
}}

// ── Monomer selection ──
var selectedSmi = null;
function selectMonomer(smi) {{
    selectedSmi = smi;
    var items = document.querySelectorAll('#search-results .result-item');
    items.forEach(function(el) {{ el.classList.remove('selected'); }});
    items.forEach(function(el) {{
        if (el.getAttribute('data-smi') === smi) el.classList.add('selected');
    }});
    if (window.network) focusNodeInGraph(smi);
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
        var updates = [];
        window.monoNodes.forEach(function(n) {{
            if (highlight[n.id]) {{
                updates.push({{id: n.id, opacity: 1, borderWidth: 4}});
            }} else {{
                updates.push({{id: n.id, opacity: 0.15, borderWidth: 1}});
            }}
        }});
        if (updates.length > 0) network.body.data.nodes.update(updates);
        network.selectNodes([nodeId]);
        network.focus(nodeId, {{scale: 1.2, animation: true}});
    }}
}}

// ── Detail panel: monomer ──
function showDetail(smi) {{
    var m = MONOMERS[smi.substring(0, 60)];
    if (!m) {{
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

    var html = '';

    // ── 化学结构图 ──
    if (m.svg) {{
        html += '<div class="detail-section"><h4>化学结构</h4>';
        html += '<div class="chem-structure">' + m.svg + '</div></div>';
    }}

    // ── 基本属性 ──
    html += '<div class="detail-section"><h4>基本属性</h4>' +
        '<table class="detail-table">';
    if (m.chem_name) {{
        html += '<tr><td>名称</td><td><b>' + m.chem_name + '</b></td></tr>';
    }}
    html += '<tr><td>SMILES</td><td style="word-break:break-all;font-family:monospace;font-size:11px">' + smiShort + '</td></tr>' +
        '<tr><td>类型</td><td>' + m.mtype + '</td></tr>' +
        '<tr><td>拓扑</td><td>' + m.topo + '</td></tr>' +
        '<tr><td>含氟</td><td>' + fStr + '</td></tr>' +
        '<tr><td>N杂环</td><td>' + nStr + '</td></tr>' +
        '<tr><td>芳环数</td><td>' + m.n_arom + '</td></tr>' +
        '<tr><td>分子量</td><td>' + m.mw.toFixed(1) + '</td></tr>' +
        '<tr><td>文献数</td><td>' + m.n_lit + '</td></tr>' +
        '<tr><td>成膜率</td><td>' + (m.film_rate * 100).toFixed(1) + '%</td></tr>' +
        '</table></div>';

    // ── 来源文章列表 ──
    html += '<div class="detail-section"><h4>来源文章 (' + m.lits.length + ' 篇)</h4>';
    for (var i = 0; i < m.lits.length; i++) {{
        var lid = m.lits[i];
        var lit = LITERATURES[lid];
        var lidShort = lid.length > 80 ? lid.substring(0, 77) + '...' : lid;
        if (lit) {{
            html += '<div class="mono-lit-item" data-lid="' +
                lid.replace(/"/g, '&quot;').replace(/'/g, '&#39;') + '">' +
                '<b>#' + (i + 1) + '</b> ' + lidShort + '<br>' +
                '<span style="font-size:11px;color:#666">' +
                (lit.system || '').substring(0, 80) + '</span></div>';
        }} else {{
            html += '<div class="mono-lit-item" style="color:#999">' +
                '<b>#' + (i + 1) + '</b> ' + lidShort + ' (无结构化数据)</div>';
        }}
    }}
    html += '</div>';
    body.innerHTML = html;
}}

// ── Detail panel: literature (full 22 fields) ──
var LIT_FIELD_LABELS = {{
    'journal': '期刊',
    'system': '研究体系',
    'reagent': '试剂/单体',
    'catalyst': '催化剂',
    'solvent': '溶剂',
    'reaction_temperature': '反应温度',
    'synthesis_mode': '合成模式',
    'synthesis_route': '合成路线',
    'interface_type': '界面类型',
    'annealing_conditions': '退火条件',
    'schiff_base_kinetics': '席夫碱动力学',
    'fluorine_effects': '氟效应',
    'fluorine_monomer': '含氟单体',
    'film_crystallinity_fluorine': '成膜/结晶度/氟',
    'adsorption_mechanism': '吸附机理',
    'computational_methods': '计算方法',
    'conclusion_1': '实验结论 1',
    'conclusion_2': '实验结论 2',
    'conclusion_3': '实验结论 3',
    'innovation': '创新点',
    'methods': '表征方法'
}};
var LIT_FIELD_ORDER = [
    'journal', 'system', 'reagent', 'catalyst', 'solvent',
    'reaction_temperature', 'synthesis_mode', 'synthesis_route',
    'interface_type', 'annealing_conditions',
    'schiff_base_kinetics', 'fluorine_effects', 'fluorine_monomer',
    'film_crystallinity_fluorine', 'adsorption_mechanism',
    'computational_methods', 'conclusion_1', 'conclusion_2', 'conclusion_3',
    'innovation', 'methods'
];

function showLitDetail(lid) {{
    var lit = LITERATURES[lid];
    if (!lit) return;

    var panel = document.getElementById('detail-panel');
    var body = document.getElementById('detail-body');
    document.getElementById('detail-title').textContent = '文献详情';
    panel.style.display = 'block';

    var lidShort = lid.length > 80 ? lid.substring(0, 77) + '...' : lid;
    var html = '<div class="detail-section"><h4>文献 ID</h4>' +
        '<p style="font-size:11px;word-break:break-all;color:#666">' + lidShort + '</p></div>';

    html += '<div class="detail-section"><h4>结构化信息 (全部字段)</h4>' +
        '<table class="full-lit-table">';
    for (var i = 0; i < LIT_FIELD_ORDER.length; i++) {{
        var key = LIT_FIELD_ORDER[i];
        var val = lit[key];
        if (!val || val === 'None' || val === 'null' || val === '') continue;
        var label = LIT_FIELD_LABELS[key] || key;
        val = val.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        html += '<tr><td>' + label + '</td><td>' + val + '</td></tr>';
    }}
    html += '</table></div>';

    if (selectedSmi) {{
        var m = MONOMERS[selectedSmi.substring(0, 60)] || {{}};
        html += '<div class="detail-section">' +
            '<span class="lit-link" onclick="showDetail(selectedSmi)">&larr; 返回单体: ' +
            (m.label || selectedSmi.substring(0, 30)) + '</span></div>';
    }}

    body.innerHTML = html;
}}

function closeDetail() {{
    document.getElementById('detail-panel').style.display = 'none';
    resetHighlights();
}}

// ── Debug ──
var statusEl = document.getElementById('search-results');
function _status(msg) {{
    if (statusEl) statusEl.innerHTML = '<div class=\"no-results\">' + msg + '</div>';
}}

// ── Init ──
(function initGraphUI() {{
    _status('初始化中 (SVG: {svg_count}, 名称: {name_count}, 文献: ' + Object.keys(LITERATURES).length + ')...');
    if (typeof network === 'undefined') {{ _status('ERROR: pyvis network 未定义'); return; }}

    var attempts = 0;
    function wire() {{
        if (typeof network !== 'undefined' && network.body) {{
            _status('图谱就绪 — 输入关键词搜索单体');

            // Event delegation: search results
            document.getElementById('search-results').addEventListener('click', function(e) {{
                var item = e.target.closest('.result-item');
                if (item) selectMonomer(item.getAttribute('data-smi'));
            }});

            // Event delegation: literature items
            document.getElementById('detail-body').addEventListener('click', function(e) {{
                var item = e.target.closest('.mono-lit-item');
                if (item) showLitDetail(item.getAttribute('data-lid'));
            }});

            // Debounced search
            var searchTimer = null;
            document.getElementById('search-input').addEventListener('input', function() {{
                clearTimeout(searchTimer);
                searchTimer = setTimeout(doSearch, 250);
            }});

            window.network = network;
            window.monoNodes = network.body.data.nodes.get().filter(function(n) {{
                return !n.id.startsWith('edge_');
            }});
            window.nodeDataMap = {{}};
            window.nodeOrigColor = {{}};
            window.monoNodes.forEach(function(n) {{
                window.nodeOrigColor[n.id] = n.color;
                for (var k in MONOMERS) {{
                    if (MONOMERS[k].full_smi === n.id) {{
                        window.nodeDataMap[n.id] = MONOMERS[k].full_smi;
                        break;
                    }}
                }}
                if (!window.nodeDataMap[n.id]) {{
                    for (var k in MONOMERS) {{
                        if (n.id.indexOf(k) >= 0 && k.length > 10) {{
                            window.nodeDataMap[n.id] = MONOMERS[k].full_smi;
                            break;
                        }}
                    }}
                }}
                if (!window.nodeDataMap[n.id]) window.nodeDataMap[n.id] = n.id;
            }});

            var netEl = document.getElementById('mynetwork');
            if (netEl) {{
                netEl.style.marginRight = '400px';
                netEl.style.width = 'calc(100% - 400px)';
                netEl.style.height = '100vh';
            }}

            network.on('click', function(params) {{
                if (params.nodes.length === 1) {{
                    var nid = params.nodes[0];
                    var smi = window.nodeDataMap[nid];
                    if (smi) selectMonomer(smi);
                }}
            }});
            network.fit();
            return;
        }}
        attempts++;
        if (attempts < 50) setTimeout(wire, 200);
    }}
    wire();
}})();
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
    yaml_dir: str = "data/structured",
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

    monomer_to_lits = _build_monomer_lit_map(G, monomers)

    # ═══════════════════════════════════════
    # 1. 核心子图 (Top 150 节点) + SVG + 文献全文
    # ═══════════════════════════════════════
    degrees = dict(monomer_G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:150]
    core_G = monomer_G.subgraph(top_nodes).copy()

    # 核心子图关联的文献集合
    core_lit_ids = set()
    for nid in core_G.nodes():
        core_lit_ids.update(monomer_to_lits.get(nid, []))
    lit_full_core = _build_literature_full(yaml_dir, core_lit_ids)
    print(f"  核心图关联文献: {len(core_lit_ids)} 篇, 有效: {len(lit_full_core)}")

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
        "solver": "barnesHut",
        "stabilization": {"iterations": 200, "fit": true}
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
    _inject_search_panel(core_path, monomer_to_lits, lit_full_core,
                         nodes_data, include_svg=True,
                         svg_smiles=set(core_G.nodes()))
    print(f"核心子图 (150节点 + SVG + 全文): {core_path}")

    # ═══════════════════════════════════════
    # 2. 全量图 (无 SVG, 基础文献数据)
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
        "solver": "barnesHut",
        "stabilization": {"iterations": 200, "fit": true}
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

    # 全量图: 基础文献字段 (无 SVG, 无 YAML 全文)
    lit_basic = {}
    full_lit_ids = set()
    for nid in monomer_G.nodes():
        full_lit_ids.update(monomer_to_lits.get(nid, []))
    lit_full_all = _build_literature_full(yaml_dir, full_lit_ids)
    print(f"  全量图关联文献: {len(full_lit_ids)} 篇, 有效: {len(lit_full_all)}")

    full_path = os.path.join(output_dir, "graph_visual.html")
    full_net.save_graph(full_path)
    _add_legend(full_path)
    _inject_search_panel(full_path, monomer_to_lits, lit_full_all,
                         nodes_data, include_svg=False)
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
