"""COF 知识图谱交互式可视化 — pyvis + 搜索面板 + 结构图 + 商业单体 + 预测边。

输出:
  - graphrag图谱辅助/graph_visual.html  (全量图)
  - graphrag图谱辅助/graph_visual_core.html (核心子图, Top 150 节点)

颜色/样式:
  - 文献单体: 蓝=醛, 红=胺, 紫=双功能 (暗色调=F/N杂环)
  - 商业单体: 绿色节点
  - 文献配对边: 绿≥50%, 橙0-50%, 灰=0
  - 预测配对边: 红色虚线
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import yaml
from pyvis.network import Network
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2DSVG

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chemistry.monomer import _BUILTIN_MONOMERS

# ── 常量 ──
PREDICTED_EDGE_N = 250        # 筛选结果 Top N (核心图用)
SVG_SIZE_FULL = (200, 130)    # 全量图 SVG 尺寸 (小)
SVG_SIZE_CORE = (260, 170)    # 核心图 SVG 尺寸

# ── 反向名称字典 ──
_NAME_PRIORITY = {}
for _name, _smi in _BUILTIN_MONOMERS.items():
    if not _smi:
        continue
    try:
        _csmi = Chem.CanonSmiles(_smi)
        if _csmi not in _NAME_PRIORITY or len(_name) < len(_NAME_PRIORITY[_csmi]):
            _NAME_PRIORITY[_csmi] = _name
    except Exception:
        pass


def _canon(smi: str):
    try: return Chem.CanonSmiles(smi)
    except: return None


def _chemical_name(smi: str) -> str:
    csmi = _canon(smi)
    return _NAME_PRIORITY.get(csmi, "") if csmi else ""


def _render_svg(smi: str, size: tuple = (240, 150)) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try: Chem.Kekulize(mol)
    except: pass
    drawer = MolDraw2DSVG(size[0], size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ── 商业单体加载 ──
def _load_commercial(csv_path: str) -> dict:
    """返回 {canonical_smiles: {name, cas, type, ...}}"""
    df = pd.read_csv(csv_path)
    comm = {}
    for _, r in df.iterrows():
        smi = _canon(str(r.get("smiles", "")))
        if not smi: continue
        mtype = str(r.get("monomer_type", ""))
        # 规范类型名称
        if "aldehyde" in mtype: mtype = "aldehyde"
        elif "amine" in mtype: mtype = "amine"
        else: mtype = "dual" if "dual" in mtype else "?"
        comm[smi] = {
            "name": str(r.get("name", "")),
            "cas": str(r.get("cas", "")),
            "commercial_id": str(r.get("commercial_id", "")),
            "mtype": mtype,
            "has_f": bool(r.get("has_fluorine", False)),
        }
    return comm


# ── 筛选结果加载 ──
def _load_predicted_edges(screening_path: str, top_n: int) -> list:
    """返回 [(ald_smi, am_smi, margin_score), ...] 按 margin 降序取 top_n。"""
    df = pd.read_csv(screening_path)
    df = df.sort_values("margin_score", ascending=False).head(top_n)
    edges = []
    for _, r in df.iterrows():
        ald = _canon(str(r["aldehyde_smiles"]))
        am = _canon(str(r["amine_smiles"]))
        if ald and am:
            edges.append((ald, am, float(r["margin_score"])))
    return edges


# ── 节点颜色/样式 ──
def _node_color(nd: dict, is_commercial: bool = False) -> str:
    if is_commercial:
        return "#27ae60"  # 绿色 — 商业单体
    mtype = nd.get("monomer_type", "")
    has_f = nd.get("has_fluorine", False)
    has_n = nd.get("has_n_heterocycle", False)
    if mtype == "aldehyde":
        return "#1a5276" if has_n else ("#2980b9" if has_f else "#85c1e9")
    elif mtype == "amine":
        return "#922b21" if has_n else ("#c0392b" if has_f else "#f1948a")
    elif mtype == "dual":
        return "#8e44ad"
    return "#95a5a6"


def _node_size(nd: dict) -> int:
    n_lit = nd.get("n_literatures", 0)
    n_part = nd.get("n_partners", 0)
    return max(8, min(40, 6 + n_lit * 1.5 + n_part * 0.5))


def _node_title(nd: dict, nid: str, is_commercial: bool = False) -> str:
    mtype = nd.get("monomer_type", "?")
    topo = nd.get("topology", "?")
    has_f = "是" if nd.get("has_fluorine") else "否"
    has_n = "是" if nd.get("has_n_heterocycle") else "否"
    n_lit = nd.get("n_literatures", 0)
    n_part = nd.get("n_partners", 0)
    film_p = nd.get("film_positive", 0)
    film_n = nd.get("film_negative", 0)
    film_rate = nd.get("film_rate", 0)
    mw = nd.get("mw", 0)
    prefix = "[商业] " if is_commercial else ""
    return (
        f"{prefix}<b>SMILES:</b> {nid[:50]}<br>"
        f"<b>类型:</b> {mtype} | 拓扑: {topo}<br>"
        f"<b>含氟:</b> {has_f} | N杂环: {has_n}<br>"
        f"<b>分子量:</b> {mw:.1f}<br>"
        f"<b>文献数:</b> {n_lit} | 配对邻居: {n_part}<br>"
        f"<b>成膜:</b> {film_p}/{film_p + film_n} ({film_rate:.1%})"
    )


def _edge_color(edge: dict) -> str:
    fr = edge.get("film_ratio", 0)
    if fr >= 0.5: return "#27ae60"
    if fr > 0: return "#f39c12"
    return "#bdc3c7"


def _edge_width(edge: dict) -> float:
    return min(5, 0.5 + edge.get("n_literatures", 1) * 0.8)


# ── 图谱数据 ──
def _build_monomer_lit_map(G, monomers: set) -> dict:
    mono_to_lits = defaultdict(list)
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") != "APPEARS_IN": continue
        if u in monomers and v not in monomers: mono_to_lits[u].append(v)
        elif v in monomers and u not in monomers: mono_to_lits[v].append(u)
    return {k: sorted(v) for k, v in mono_to_lits.items()}


def _build_literature_full(yaml_dir: str, lit_ids_set: set) -> dict:
    yaml_dir_path = Path(yaml_dir)
    yaml_files = {f.stem: str(f) for f in yaml_dir_path.glob("*.yaml")}
    full_db = {}
    for lid in lit_ids_set:
        if lid not in yaml_files: continue
        try:
            with open(yaml_files[lid], encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(data, dict): continue
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


# ═══════════════════════════════════════════
# HTML 注入
# ═══════════════════════════════════════════
def _inject_panel(html_path: str, monomer_to_lits: dict,
                  lit_full: dict, nodes_data: dict,
                  commercial: dict, predicted_edges: list,
                  include_svg: bool, svg_size: tuple):
    # ── 构建 MONOMERS JS ──
    mono_js = {}
    # 文献单体
    for smi, lit_ids in monomer_to_lits.items():
        nd = nodes_data.get(smi, {})
        mono_js[smi[:60]] = {
            "full_smi": smi, "label": nd.get("label", smi[:30]),
            "mtype": nd.get("monomer_type", "?"), "topo": nd.get("topology", "?"),
            "has_f": nd.get("has_fluorine", False), "has_n": nd.get("has_n_heterocycle", False),
            "n_arom": nd.get("n_aromatic_rings", 0), "mw": nd.get("mw", 0),
            "n_lit": nd.get("n_literatures", 0), "film_rate": nd.get("film_rate", 0),
            "chem_name": _chemical_name(smi), "lits": lit_ids,
            "is_commercial": smi in commercial,
            "svg": _render_svg(smi, svg_size) if include_svg else "",
        }
        if smi in commercial:
            c = commercial[smi]
            mono_js[smi[:60]].update({"cas": c["cas"], "commercial_id": c["commercial_id"]})
    # 纯商业单体 (不在文献图中)
    for smi, c in commercial.items():
        if smi[:60] in mono_js: continue
        mono_js[smi[:60]] = {
            "full_smi": smi, "label": c.get("name", smi[:25]),
            "mtype": c.get("mtype", "?"), "topo": "?",
            "has_f": c.get("has_f", False), "has_n": False, "n_arom": 0,
            "mw": 0, "n_lit": 0, "film_rate": 0,
            "chem_name": c.get("name", ""), "lits": [],
            "is_commercial": True, "cas": c.get("cas", ""),
            "commercial_id": c.get("commercial_id", ""),
            "svg": _render_svg(smi, svg_size) if include_svg else "",
        }

    mono_json = json.dumps(mono_js, ensure_ascii=False)
    lit_json = json.dumps(lit_full, ensure_ascii=False)

    # 预测边 JS 数组
    pred_json = json.dumps(
        [{"ald": a, "am": b, "score": round(s, 4)} for a, b, s in predicted_edges],
        ensure_ascii=False
    )

    svg_cnt = sum(1 for v in mono_js.values() if v.get("svg"))
    comm_cnt = sum(1 for v in mono_js.values() if v.get("is_commercial"))

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
#search-box {{ padding:12px; border-bottom:1px solid #eee; }}
#search-box input {{
    width:100%; padding:10px; border:2px solid #ddd; border-radius:6px;
    font-size:14px; outline:none; box-sizing:border-box;
}}
#search-box input:focus {{ border-color:#2980b9; }}
#search-results {{ flex:1; overflow-y:auto; padding:8px; }}
.result-item {{
    padding:10px; border-bottom:1px solid #f0f0f0; cursor:pointer;
    border-radius:4px; transition:background 0.15s;
}}
.result-item:hover {{ background:#eaf2f8; }}
.result-item.selected {{ background:#d4e6f1; border-left:3px solid #2980b9; }}
.result-item.commercial {{ border-left:3px solid #27ae60; }}
.result-meta {{ font-size:11px; color:#666; margin-top:4px; }}
.result-meta span {{ margin-right:8px; }}
.badge {{ padding:1px 6px; border-radius:3px; font-size:10px; font-weight:bold; }}
.badge.f-true {{ background:#e74c3c; color:#fff; }}
.badge.f-false {{ background:#95a5a6; color:#fff; }}
.badge.n-true {{ background:#8e44ad; color:#fff; }}
.badge.n-false {{ background:#95a5a6; color:#fff; }}
.badge.comm {{ background:#27ae60; color:#fff; }}

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

html, body {{ height: 100%; margin: 0; padding: 0; overflow: hidden; }}
#mynetwork {{
    margin-right: 400px !important;
    width: calc(100% - 400px) !important;
    height: 100vh !important;
    float: none !important;
    position: absolute !important;
    top: 0; left: 0;
}}
</style>

<button id="panel-toggle" onclick="togglePanel()">◀ 搜索</button>

<div id="search-panel">
    <div id="search-box">
        <input type="text" id="search-input"
               placeholder="搜索: SMILES / f:yes / CAS / comm:yes ..." autofocus>
        <div style="font-size:10px;color:#999;padding:4px 0 0 2px">
          快捷: <b>f:yes</b> | <b>comm:yes</b> | <b>type:aldehyde</b> | <b>topo:C3</b>
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
var PREDICTED_EDGES = {pred_json};

var panelVisible = true;
function togglePanel() {{
    var p = document.getElementById('search-panel');
    var btn = document.getElementById('panel-toggle');
    var netEl = document.getElementById('mynetwork');
    if (panelVisible) {{
        p.classList.add('collapsed'); btn.classList.add('collapsed');
        btn.textContent = '▶ 搜索';
        netEl.style.marginRight = '0'; netEl.style.width = '100%';
    }} else {{
        p.classList.remove('collapsed'); btn.classList.remove('collapsed');
        btn.textContent = '◀ 搜索';
        netEl.style.marginRight = '400px'; netEl.style.width = 'calc(100% - 400px)';
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
        resetHighlights(); return;
    }}
    var sortMode = 'lit', searchTerm = q;
    var specialCmds = {{
        'film:0': 'film0',
        'f:yes': 'f_yes', 'f:no': 'f_no', 'n:yes': 'n_yes',
        'comm:yes': 'comm', 'type:aldehyde': 'type_ald', 'type:amine': 'type_am',
        'topo:c1': 'topo_c1', 'topo:c2': 'topo_c2', 'topo:c3': 'topo_c3', 'topo:c4': 'topo_c4'
    }};
    for (var cmd in specialCmds) {{
        if (q === cmd || q.indexOf(cmd + ' ') === 0) {{
            sortMode = specialCmds[cmd]; searchTerm = q.substring(cmd.length).trim(); break;
        }}
    }}
    var results = [];
    for (var key in MONOMERS) {{
        var m = MONOMERS[key];
        if (sortMode === 'film0') {{ if (m.film_rate === 0) results.push(m); continue; }}
        if (sortMode === 'f_yes') {{ if (m.has_f) results.push(m); continue; }}
        if (sortMode === 'f_no') {{ if (!m.has_f) results.push(m); continue; }}
        if (sortMode === 'n_yes') {{ if (m.has_n) results.push(m); continue; }}
        if (sortMode === 'comm') {{ if (m.is_commercial) results.push(m); continue; }}
        if (sortMode === 'type_ald') {{ if (m.mtype === 'aldehyde') results.push(m); continue; }}
        if (sortMode === 'type_am') {{ if (m.mtype === 'amine') results.push(m); continue; }}
        if (sortMode.substring(0,5) === 'topo_') {{
            if (m.topo === sortMode.substring(5).toUpperCase()) results.push(m); continue;
        }}
        // Text search: SMILES, label, name, CAS
        var txt = (m.full_smi + '|' + m.label + '|' + m.mtype + '|' + m.topo +
                   '|' + (m.chem_name||'') + '|' + (m.cas||'')).toLowerCase();
        if (searchTerm && txt.indexOf(searchTerm) >= 0) results.push(m);
        else if (!searchTerm) results.push(m);
    }}
    if (results.length === 0) {{
        container.innerHTML = '<div class="no-results">无匹配结果</div>';
        resetHighlights(); return;
    }}
    results.sort(function(a,b){{ return b.n_lit - a.n_lit; }});

    var html = '', matchSet = new Set();
    var limit = 80;
    for (var i = 0; i < Math.min(results.length, limit); i++) {{
        var m = results[i]; matchSet.add(m.full_smi);
        var cls = m.is_commercial ? ' commercial' : '';
        var smiShort = m.full_smi.length > 55 ? m.full_smi.substring(0,52)+'...' : m.full_smi;
        html += '<div class="result-item' + cls + '" data-smi="' +
            m.full_smi.replace(/"/g,'&quot;').replace(/'/g,'&#39;') + '">' +
            '<b>' + m.label + '</b> [' + m.mtype + ' | ' + m.topo + ']';
        if (m.is_commercial) html += ' <span class="badge comm">商业</span>';
        html += '<br><span style="font-size:11px;color:#888">' + smiShort + '</span>' +
            '<div class="result-meta">' +
            '<span>文献:' + m.n_lit + '</span>' +
            '<span>成膜率:' + (m.film_rate*100).toFixed(0) + '%</span>' +
            '<span class="badge ' + (m.has_f?'f-true':'f-false') + '">' + (m.has_f?'F':'非F') + '</span>';
        if (m.cas) html += '<span style="color:#888">CAS:' + m.cas + '</span>';
        html += '</div></div>';
    }}
    container.innerHTML = html;
    highlightNodes(matchSet);
    }} catch(e) {{
        var c = document.getElementById('search-results');
        if (c) c.innerHTML = '<div class="no-results" style="color:red">Error: '+e.message+'</div>';
    }}
}}

// ── Highlight (batch) ──
var currentHighlight = new Set();
function highlightNodes(smiSet) {{
    if (!window.network) return;
    var updates = [];
    window.monoNodes.forEach(function(n) {{
        var smi = window.nodeDataMap[n.id] || n.id;
        if (smiSet.has(smi)) updates.push({{id:n.id, opacity:1, borderWidth:4}});
        else updates.push({{id:n.id, opacity:0.2, borderWidth:1}});
    }});
    currentHighlight = new Set(smiSet);
    if (updates.length > 0) network.body.data.nodes.update(updates);
}}
function resetHighlights() {{
    if (!window.network) return;
    var updates = [];
    window.monoNodes.forEach(function(n) {{
        updates.push({{id:n.id, opacity:1, borderWidth:n.borderWidth||1}});
    }});
    if (updates.length > 0) network.body.data.nodes.update(updates);
}}

// ── Monomer selection ──
var selectedSmi = null;
function selectMonomer(smi) {{
    selectedSmi = smi;
    var items = document.querySelectorAll('#search-results .result-item');
    items.forEach(function(el){{el.classList.remove('selected');}});
    items.forEach(function(el){{if(el.getAttribute('data-smi')===smi)el.classList.add('selected');}});
    if (window.network) focusNodeInGraph(smi);
    showDetail(smi);
}}
function focusNodeInGraph(smi) {{
    var nodeId = null;
    window.monoNodes.forEach(function(n){{if(window.nodeDataMap[n.id]===smi||n.id===smi)nodeId=n.id;}});
    if (nodeId) {{
        var h={{}}; h[nodeId]=true;
        currentHighlight.forEach(function(s){{window.monoNodes.forEach(function(n){{if(window.nodeDataMap[n.id]===s||n.id===s)h[n.id]=true;}});}});
        var updates = [];
        window.monoNodes.forEach(function(n){{if(h[n.id])updates.push({{id:n.id,opacity:1,borderWidth:4}});else updates.push({{id:n.id,opacity:0.15,borderWidth:1}});}});
        if (updates.length>0) network.body.data.nodes.update(updates);
        network.selectNodes([nodeId]); network.focus(nodeId,{{scale:1.2,animation:true}});
    }}
}}

// ── Detail: monomer ──
function showDetail(smi) {{
    var m = MONOMERS[smi.substring(0,60)];
    if (!m) {{ for (var k in MONOMERS) {{ if (MONOMERS[k].full_smi===smi) {{ m=MONOMERS[k]; break; }} }} }}
    if (!m) return;
    var panel = document.getElementById('detail-panel');
    var body = document.getElementById('detail-body');
    document.getElementById('detail-title').textContent = '单体: ' + m.label;
    panel.style.display = 'block';
    var fStr = m.has_f ? '是' : '否', nStr = m.has_n ? '是' : '否';
    var smiShort = m.full_smi.length>60 ? m.full_smi.substring(0,57)+'...' : m.full_smi;
    var html = '';
    // 结构图
    if (m.svg) {{ html += '<div class="detail-section"><h4>化学结构</h4><div class="chem-structure">'+m.svg+'</div></div>'; }}
    // 属性
    html += '<div class="detail-section"><h4>基本属性</h4><table class="detail-table">';
    if (m.is_commercial) html += '<tr><td>标记</td><td><span class="badge comm">商业单体</span></td></tr>';
    if (m.chem_name) html += '<tr><td>名称</td><td><b>'+m.chem_name+'</b></td></tr>';
    if (m.cas) html += '<tr><td>CAS</td><td>'+m.cas+'</td></tr>';
    html += '<tr><td>SMILES</td><td style="word-break:break-all;font-family:monospace;font-size:11px">'+smiShort+'</td></tr>'+
        '<tr><td>类型</td><td>'+m.mtype+'</td></tr>'+
        '<tr><td>拓扑</td><td>'+m.topo+'</td></tr>'+
        '<tr><td>含氟</td><td>'+fStr+'</td></tr>'+
        '<tr><td>N杂环</td><td>'+nStr+'</td></tr>'+
        '<tr><td>芳环数</td><td>'+m.n_arom+'</td></tr>'+
        '<tr><td>分子量</td><td>'+(m.mw?m.mw.toFixed(1):'?')+'</td></tr>'+
        '<tr><td>文献数</td><td>'+m.n_lit+'</td></tr>'+
        '<tr><td>成膜率</td><td>'+(m.film_rate*100).toFixed(1)+'%</td></tr>'+
        '</table></div>';
    // 来源文章
    if (m.lits.length > 0) {{
        html += '<div class="detail-section"><h4>来源文章 ('+m.lits.length+' 篇)</h4>';
        for (var i=0;i<m.lits.length;i++) {{
            var lid=m.lits[i], lit=LITERATURES[lid];
            var lidShort = lid.length>80 ? lid.substring(0,77)+'...' : lid;
            if (lit) {{
                html += '<div class="mono-lit-item" data-lid="'+lid.replace(/"/g,'&quot;').replace(/'/g,'&#39;')+'">'+
                    '<b>#'+(i+1)+'</b> '+lidShort+'<br>'+
                    '<span style="font-size:11px;color:#666">'+(lit.system||'').substring(0,80)+'</span></div>';
            }} else {{
                html += '<div class="mono-lit-item" style="color:#999"><b>#'+(i+1)+'</b> '+lidShort+' (无数据)</div>';
            }}
        }}
        html += '</div>';
    }}
    body.innerHTML = html;
}}

// ── Detail: literature ──
var LIT_LABELS = {{
    'journal': '期刊', 'system': '研究体系', 'reagent': '试剂/单体',
    'catalyst': '催化剂', 'solvent': '溶剂', 'reaction_temperature': '反应温度',
    'synthesis_mode': '合成模式', 'synthesis_route': '合成路线',
    'interface_type': '界面类型', 'annealing_conditions': '退火条件',
    'schiff_base_kinetics': '席夫碱动力学', 'fluorine_effects': '氟效应',
    'fluorine_monomer': '含氟单体', 'film_crystallinity_fluorine': '成膜/结晶度/氟',
    'adsorption_mechanism': '吸附机理', 'computational_methods': '计算方法',
    'conclusion_1': '实验结论 1', 'conclusion_2': '实验结论 2', 'conclusion_3': '实验结论 3',
    'innovation': '创新点', 'methods': '表征方法'
}};
var LIT_ORDER = ['journal','system','reagent','catalyst','solvent','reaction_temperature',
    'synthesis_mode','synthesis_route','interface_type','annealing_conditions',
    'schiff_base_kinetics','fluorine_effects','fluorine_monomer','film_crystallinity_fluorine',
    'adsorption_mechanism','computational_methods','conclusion_1','conclusion_2','conclusion_3',
    'innovation','methods'];

function showLitDetail(lid) {{
    var lit = LITERATURES[lid]; if (!lit) return;
    var panel = document.getElementById('detail-panel');
    var body = document.getElementById('detail-body');
    document.getElementById('detail-title').textContent = '文献详情';
    panel.style.display = 'block';
    var lidShort = lid.length>80 ? lid.substring(0,77)+'...' : lid;
    var html = '<div class="detail-section"><h4>文献 ID</h4>'+
        '<p style="font-size:11px;word-break:break-all;color:#666">'+lidShort+'</p></div>'+
        '<div class="detail-section"><h4>结构化信息</h4><table class="full-lit-table">';
    for (var i=0;i<LIT_ORDER.length;i++) {{
        var key=LIT_ORDER[i], val=lit[key];
        if (!val||val==='None'||val==='null'||val==='') continue;
        html += '<tr><td>'+(LIT_LABELS[key]||key)+'</td><td>'+val.replace(/</g,'&lt;').replace(/>/g,'&gt;')+'</td></tr>';
    }}
    html += '</table></div>';
    if (selectedSmi) {{
        var m = MONOMERS[selectedSmi.substring(0,60)]||{{}};
        html += '<div class="detail-section"><span class="lit-link" onclick="showDetail(selectedSmi)">&larr; 返回: '+(m.label||'')+'</span></div>';
    }}
    body.innerHTML = html;
}}
function closeDetail() {{
    document.getElementById('detail-panel').style.display = 'none'; resetHighlights();
}}

// ── Debug ──
var statusEl = document.getElementById('search-results');
function _status(m){{ if (statusEl) statusEl.innerHTML = '<div class=\"no-results\">'+m+'</div>'; }}

// ── Init ──
(function initUI() {{
    _status('初始化中 (单体:'+Object.keys(MONOMERS).length+', SVG:{svg_cnt}, 商业:{comm_cnt}, 预测边:'+PREDICTED_EDGES.length+')...');
    if (typeof network==='undefined') {{ _status('ERROR: network 未定义'); return; }}
    var attempts=0;
    function wire() {{
        if (typeof network!=='undefined' && network.body) {{
            _status('图谱就绪 — 输入关键词搜索单体');

            document.getElementById('search-results').addEventListener('click', function(e){{
                var item = e.target.closest('.result-item'); if (item) selectMonomer(item.getAttribute('data-smi'));
            }});
            document.getElementById('detail-body').addEventListener('click', function(e){{
                var item = e.target.closest('.mono-lit-item'); if (item) showLitDetail(item.getAttribute('data-lid'));
            }});
            var searchTimer = null;
            document.getElementById('search-input').addEventListener('input', function(){{
                clearTimeout(searchTimer); searchTimer = setTimeout(doSearch, 250);
            }});

            window.network = network;
            window.monoNodes = network.body.data.nodes.get().filter(function(n){{return !n.id.startsWith('edge_');}});
            window.nodeDataMap = {{}}; window.nodeOrigColor = {{}};
            window.monoNodes.forEach(function(n){{
                window.nodeOrigColor[n.id] = n.color;
                for (var k in MONOMERS) {{
                    if (MONOMERS[k].full_smi === n.id) {{ window.nodeDataMap[n.id] = MONOMERS[k].full_smi; break; }}
                }}
                if (!window.nodeDataMap[n.id]) {{
                    for (var k in MONOMERS) {{ if (n.id.indexOf(k)>=0 && k.length>10) {{ window.nodeDataMap[n.id]=MONOMERS[k].full_smi; break; }} }}
                }}
                if (!window.nodeDataMap[n.id]) window.nodeDataMap[n.id] = n.id;
            }});

            var netEl = document.getElementById('mynetwork');
            if (netEl) {{
                netEl.style.marginRight = '400px'; netEl.style.width = 'calc(100% - 400px)'; netEl.style.height = '100vh';
            }}

            network.on('click', function(params){{
                if (params.nodes.length===1) {{ var smi=window.nodeDataMap[params.nodes[0]]; if (smi) selectMonomer(smi); }}
            }});
            network.fit();
            return;
        }}
        attempts++; if (attempts<50) setTimeout(wire, 200);
    }}
    wire();
}})();
</script>"""

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</body>", panel_html + "</body>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# ═══════════════════════════════════════════
# 主构建函数
# ═══════════════════════════════════════════
def build_visualization(
    gml_path: str = "graphrag图谱辅助/graph.gml",
    nodes_path: str = "graphrag图谱辅助/graph_nodes.json",
    edges_path: str = "graphrag图谱辅助/graph_edges.json",
    yaml_dir: str = "data/structured",
    commercial_csv: str = "data/processed/commercial_monomers_classified.csv",
    screening_csv: str = "data/processed/screening_top40_with_hard_full.csv",
    output_dir: str = "graphrag图谱辅助",
):
    G = nx.read_gml(gml_path)
    with open(nodes_path, encoding="utf-8") as f:
        nodes_data = {n["id"]: n for n in json.load(f)}
    with open(edges_path, encoding="utf-8") as f:
        json.load(f)  # edges_data — not needed directly

    monomers = {n for n, d in G.nodes(data=True) if d.get("node_type") == "monomer"}
    monomer_G = G.subgraph(monomers).copy()
    for u, v, d in list(monomer_G.edges(data=True)):
        if d.get("edge_type") != "PAIRED_WITH":
            monomer_G.remove_edge(u, v)
    monomer_G.remove_nodes_from(list(nx.isolates(monomer_G)))
    # 度≥2 过滤推迟到商业单体+预测边合并之后，统一筛选

    monomer_to_lits = _build_monomer_lit_map(G, monomers)

    # ── 商业单体 ──
    commercial = _load_commercial(commercial_csv)

    # ── 预测边 (Top 5000) ──
    pred_edges_all = _load_predicted_edges(screening_csv, PREDICTED_EDGE_N)
    # 收集预测边中涉及的所有 SMILES
    pred_smiles = set()
    for ald, am, _ in pred_edges_all:
        pred_smiles.add(ald); pred_smiles.add(am)

    # ── 全量图: Top 150 预测边节点 + 文献配对边 ──
    top_pred = pred_edges_all[:PREDICTED_EDGE_N]
    pred_node_ids = set()
    for ald, am, _ in top_pred:
        pred_node_ids.add(ald); pred_node_ids.add(am)

    full_G = nx.Graph()
    # 先加所有预测边节点 (含商业单体)
    for nid in pred_node_ids:
        if nid in monomer_G:
            nd = G.nodes[nid]
            full_G.add_node(nid, **{k: v for k, v in nd.items()})
        elif nid in commercial:
            c = commercial[nid]
            full_G.add_node(nid, node_type="commercial_monomer",
                           monomer_type=c.get("mtype", "?"),
                           has_fluorine=c.get("has_f", False),
                           has_n_heterocycle=False,
                           n_literatures=0, n_partners=0,
                           label=f"{c.get('name','')} [商业]")
        else:
            full_G.add_node(nid)

    # 加入节点间的文献配对边 (从 monomer_G 提取)
    for u, v, d in monomer_G.edges(data=True):
        if u in full_G and v in full_G:
            full_G.add_edge(u, v, **d)

    # 加入预测边 (不覆盖文献边)
    for ald, am, score in top_pred:
        if ald not in full_G or am not in full_G:
            continue
        if full_G.has_edge(ald, am):
            continue
        full_G.add_edge(ald, am, edge_type="PREDICTED",
                       margin_score=round(score, 4),
                       color="#e74c3c", dashes=True)

    # 过滤度<2 节点
    full_G.remove_nodes_from([n for n, d in full_G.degree() if d < 2])

    full_lit_ids = set()
    for nid in full_G.nodes():
        full_lit_ids.update(monomer_to_lits.get(nid, []))
    lit_full_all = _build_literature_full(yaml_dir, full_lit_ids)

    comm_in_full = sum(1 for s in commercial if s in full_G)
    pred_in_full = sum(1 for _, _, _ in top_pred if full_G.has_edge(_, _))
    lit_edge_cnt = sum(1 for _, _, d in full_G.edges(data=True) if d.get("edge_type") != "PREDICTED")
    print(f"全量图: {full_G.number_of_nodes()} 节点, {full_G.number_of_edges()} 边 "
          f"(文献边 {lit_edge_cnt}, 预测边 {pred_in_full}, 含商业 {comm_in_full})")
    print(f"  关联文献: {len(full_lit_ids)} 篇, 有效: {len(lit_full_all)}")

    # ═══════════════════════════════════════
    # 全量图渲染
    # ═══════════════════════════════════════
    full_net = Network(height="100%", width="100%", bgcolor="#ffffff",
                       font_color="#2c3e50", directed=False)
    full_net.set_options("""
    var options = {
      "nodes": {"borderWidth": 1, "borderWidthSelected": 3, "font": {"size": 9, "face": "Arial"}},
      "edges": {"smooth": {"type": "continuous", "forceDirection": "none"}, "hoverWidth": 1.5},
      "physics": {"barnesHut": {"gravitationalConstant": -2000, "centralGravity": 0.2,
        "springLength": 400, "springConstant": 0.01, "damping": 0.4},
        "minVelocity": 0.75, "solver": "barnesHut", "stabilization": {"iterations": 200, "fit": true}},
      "interaction": {"hover": true, "tooltipDelay": 150, "navigationButtons": true,
	       "hideEdgesOnDrag": true, "hideEdgesOnZoom": true}
    }
    """)

    for nid in full_G.nodes():
        nd = nodes_data.get(nid, {})
        is_comm = nid in commercial
        color = _node_color(nd) if not is_comm else "#27ae60"
        shape = "diamond" if (is_comm and nid not in monomer_G) else "dot"
        full_net.add_node(
            nid,
            label=nd.get("label", commercial.get(nid, {}).get("name", nid[:25])),
            title=_node_title(nd, nid, is_commercial=is_comm),
            color=color, size=_node_size(nd) if not is_comm else 10,
            borderWidth=2, borderWidthSelected=5,
            shape=shape,
        )

    pred_count = 0
    for u, v, d in full_G.edges(data=True):
        if d.get("edge_type") == "PREDICTED":
            pred_count += 1
            full_net.add_edge(u, v,
                title=f"预测配对 | margin: {d.get('margin_score', 0):.4f}",
                color="#e74c3c", width=0.8, dashes=[8, 4])
        else:
            full_net.add_edge(u, v,
                title=(f"文献: {d.get('n_literatures', 0)} | "
                       f"成膜率: {d.get('film_ratio', 0):.1%}"),
                color=_edge_color(d), width=_edge_width(d))

    full_path = os.path.join(output_dir, "graph_visual.html")
    full_net.save_graph(full_path)
    _add_legend(full_path)
    _inject_panel(full_path, monomer_to_lits, lit_full_all, nodes_data,
                  commercial, pred_edges_all, include_svg=True, svg_size=SVG_SIZE_FULL)
    print(f"全量图 ({full_G.number_of_nodes()}节点/{full_G.number_of_edges()}边, "
          f"其中预测边{pred_count}): {full_path}")

    # ═══════════════════════════════════════
    # 核心子图: 仅 Top 150 预测边网络
    # ═══════════════════════════════════════
    core_G_pred = nx.Graph()
    for ald, am, score in top_pred:
        core_G_pred.add_node(ald)
        core_G_pred.add_node(am)
        core_G_pred.add_edge(ald, am, edge_type="PREDICTED",
                            margin_score=round(score, 4),
                            color="#e74c3c", dashes=True)
    core_G_pred.remove_nodes_from([n for n, d in core_G_pred.degree() if d < 2])

    core_lit_ids = set()
    for nid in core_G_pred.nodes():
        core_lit_ids.update(monomer_to_lits.get(nid, []))
    lit_full_core = _build_literature_full(yaml_dir, core_lit_ids)
    print(f"  核心图关联文献: {len(core_lit_ids)} 篇, 有效: {len(lit_full_core)}")

    core_pred = [(a, b, s) for a, b, s in top_pred if core_G_pred.has_edge(a, b)]

    core_net = Network(height="100%", width="100%", bgcolor="#f8f9fa",
                       font_color="#2c3e50", directed=False)
    core_net.set_options("""
    var options = {
      "nodes": {"borderWidth": 1.5, "borderWidthSelected": 4, "font": {"size": 11, "face": "Arial", "strokeWidth": 0}},
      "edges": {"smooth": {"type": "continuous", "forceDirection": "none"}, "hoverWidth": 2, "selectionWidth": 2},
      "physics": {"barnesHut": {"gravitationalConstant": -3000, "centralGravity": 0.3,
        "springLength": 300, "springConstant": 0.025, "damping": 0.3},
        "minVelocity": 0.75, "solver": "barnesHut", "stabilization": {"iterations": 200, "fit": true}},
      "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true, "keyboard": true,
	       "hideEdgesOnDrag": true, "hideEdgesOnZoom": true}
    }
    """)

    for nid in core_G_pred.nodes():
        nd = nodes_data.get(nid, {})
        is_comm = nid in commercial
        color = _node_color(nd) if not is_comm else "#27ae60"
        core_net.add_node(nid, label=nd.get("label", nid[:30]),
                         title=_node_title(nd, nid, is_commercial=is_comm),
                         color=color, size=_node_size(nd) if not is_comm else 12,
                         borderWidth=2, borderWidthSelected=5)

    core_pred_count = 0
    for u, v, d in core_G_pred.edges(data=True):
        if d.get("edge_type") == "PREDICTED":
            core_pred_count += 1
            core_net.add_edge(u, v,
                title=f"预测 | margin: {d.get('margin_score', 0):.4f}",
                color="#e74c3c", width=0.8, dashes=[8, 4])
        else:
            core_net.add_edge(u, v,
                title=(f"文献: {d.get('n_literatures', 0)}<br>"
                       f"成膜率: {d.get('film_ratio', 0):.1%}<br>"
                       f"成膜: {d.get('film_positive', 0)}/{d.get('film_negative', 0)}"),
                color=_edge_color(d), width=_edge_width(d))

    core_path = os.path.join(output_dir, "graph_visual_core.html")
    core_net.save_graph(core_path)
    _add_legend(core_path)
    _inject_panel(core_path, monomer_to_lits, lit_full_core, nodes_data,
                  commercial, core_pred, include_svg=True, svg_size=SVG_SIZE_CORE)
    print(f"核心子图 ({core_G_pred.number_of_nodes()}节点/{core_G_pred.number_of_edges()}边, "
          f"其中预测边{core_pred_count}): {core_path}")

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
      <span style="color:#8e44ad">●</span> 双功能&nbsp;
      <span style="color:#27ae60">◆</span> 商业单体<br/>
      <hr style="margin:4px 0"/>
      <span style="color:#27ae60">—</span> 成膜率 ≥ 50%<br/>
      <span style="color:#f39c12">—</span> 成膜率 0–50%<br/>
      <span style="color:#bdc3c7">—</span> 成膜率 = 0<br/>
      <span style="color:#e74c3c">- -</span> 预测配对 (Top 250)<br/>
      <hr style="margin:4px 0"/>
      <span>节点大小 ∝ 文献数 | 边宽 ∝ 配对文献数</span>
    </div>
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</body>", legend + "</body>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    build_visualization()
