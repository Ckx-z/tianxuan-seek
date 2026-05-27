"""反应条件归并解析器 — 将 LLM 提取的文本条件字段标准化为分类标签。

5 个字段的归并规则：
  - synthesis_route: solvothermal / interfacial / mechanochemical / room-temperature / other
  - interface_type:  liquid-solid / liquid-liquid / other
  - catalyst:        acetic_acid / PTSA / TFA / Sc(OTf)3_Lewis / none_other
  - solvent:         dioxane_mesitylene / DCB_BuOH / water_based / DMF_DMSO / acetonitrile / alcohol / other
  - temperature:     rt / low / standard / high / unknown  (5 箱)

Usage:
  from src.chemistry.condition_parser import ConditionParser
  parser = ConditionParser()
  result = parser.parse(synthesis_route="solvothermal", solvent="...", ...)
"""
from __future__ import annotations

import re
from typing import Optional


# ── 同义词映射 ──────────────────────────────────────────────

_CATALYST_SYNONYMS = {
    "acetic acid": "acetic_acid",
    "醋酸": "acetic_acid",
    "乙酸": "acetic_acid",
    "acoh": "acetic_acid",
    "hac": "acetic_acid",
    "ch3cooh": "acetic_acid",
    "冰乙酸": "acetic_acid",
    "冰醋酸": "acetic_acid",
    "glacial acetic acid": "acetic_acid",
    "aqueous acetic acid": "acetic_acid",
    "p-toluenesulfonic acid": "PTSA",
    "ptsa": "PTSA",
    "对甲苯磺酸": "PTSA",
    "tsoh": "PTSA",
    "p-toluenesulfonic": "PTSA",
    "trifluoroacetic acid": "TFA",
    "tfa": "TFA",
    "三氟乙酸": "TFA",
    "sc(otf)3": "Sc(OTf)3_Lewis",
    "scandium triflate": "Sc(OTf)3_Lewis",
    "bf3": "Sc(OTf)3_Lewis",
    "lewis acid": "Sc(OTf)3_Lewis",
    "路易斯酸": "Sc(OTf)3_Lewis",
    "zncl2": "Sc(OTf)3_Lewis",
    "fecl3": "Sc(OTf)3_Lewis",
    "alcl3": "Sc(OTf)3_Lewis",
    "formic acid": "none_other",
    "甲酸": "none_other",
    "hcl": "none_other",
    "盐酸": "none_other",
    "none": "none_other",
    "无": "none_other",
    "no catalyst": "none_other",
    "不加催化剂": "none_other",
    "aniline": "none_other",
    "苯胺": "none_other",
}

_SOLVENT_NORMALIZE = {
    "dichlorobenzene": "DCB",
    "o-dichlorobenzene": "DCB",
    "o-dcb": "DCB",
    "1,2-dichlorobenzene": "DCB",
    "odcb": "DCB",
    "butanol": "BuOH",
    "n-butanol": "BuOH",
    "n-buoh": "BuOH",
    "nbuoh": "BuOH",
    "1-butanol": "BuOH",
    "二氧六环": "dioxane",
    "1,4-二氧六环": "dioxane",
    "均三甲苯": "mesitylene",
    "三甲基苯": "mesitylene",
    "甲苯": "toluene",
    "乙腈": "acetonitrile",
    "乙醇": "ethanol",
    "甲醇": "methanol",
    "水": "water",
    "去离子水": "water",
    "deionized water": "water",
    "di water": "water",
    "h2o": "water",
    "二氯甲烷": "DCM",
    "dichloromethane": "DCM",
    "三氯甲烷": "chloroform",
    "己烷": "hexane",
    "丙酮": "acetone",
    "苯": "benzene",
    "二甲亚砜": "DMSO",
    "氮甲基吡咯烷酮": "NMP",
    "四氢呋喃": "THF",
}

# ── 关键词归类 ──────────────────────────────────────────────

_SYNTHESIS_ROUTE_RULES = [
    (["interfacial", "界面", "liquid-liquid", "liquid-solid", "oil-water",
      "water-oil", "air-water", "air-liquid", "gas-liquid", "三相", "triphase",
      "langmuir", "supported on", "on pan", "on substrate", "on hopg", "on si",
      "界面聚合", "液液界面", "液固界面"], "interfacial"),
    (["solvothermal", "溶剂热", "solvothermal", "hydrothermal", "水热"], "solvothermal"),
    (["mechanochemical", "机械化学", "研磨", "ball-mill", "ball mill",
      "grinding", "lag"], "mechanochemical"),
    (["room-temperature", "room temperature", "室温", "rt", "ambient",
      "ultrasonication", "超声", "sonochemical"], "room-temperature"),
]


def _normalize_text(text: str) -> str:
    """小写化 + 去多余空格。"""
    return text.lower().strip()


def _match_any(text: str, keywords: list[str]) -> bool:
    """检查文本是否包含任一关键词。"""
    for kw in keywords:
        if kw in text:
            return True
    return False


def parse_synthesis_route(raw: Optional[str]) -> str:
    """归并 synthesis_route → 5 类。

    优先级: interfacial > solvothermal > mechanochemical > room-temperature > other
    """
    if not raw:
        return "other"
    text = _normalize_text(raw)

    # 排除非合成路线关键词
    exclude = ["post-modification", "post-synthetic", "后修饰", "post-plasma",
               "impregnation", "浸渍", "coating", "涂覆", "filtration", "过滤",
               "casting", "spin-coating", "drop-casting", "exfoliation", "剥离",
               "freeze-drying", "冻干", "computational", "计算"]
    for kw in exclude:
        if kw in text:
            return "other"

    for keywords, label in _SYNTHESIS_ROUTE_RULES:
        if _match_any(text, keywords):
            return label

    return "other"


def parse_interface_type(raw: Optional[str]) -> str:
    """归并 interface_type → 3 类。

    优先级: liquid-solid > liquid-liquid > other
    """
    if not raw:
        return "other"
    text = _normalize_text(raw)

    if _match_any(text, ["liquid-solid", "液-固", "液固界面", "solid-liquid",
                          "固-液", "on substrate", "on hopg", "on si", "on pan",
                          "on fe", "on mx", "in solution", "in stainless",
                          "反应釜", "pyrex tube", "centrifuge tube"]):
        return "liquid-solid"

    if _match_any(text, ["liquid-liquid", "液-液", "液液界面", "oil-water",
                          "水-油", "油-水", "water-oil", "triphase", "三相",
                          "through support membrane", "aao界面", "界面分隔"]):
        return "liquid-liquid"

    if _match_any(text, ["gas-liquid", "气-液", "air-liquid", "空气-液",
                          "gas-solid", "气-固", "liquid-gas", "液-气"]):
        return "other"

    return "other"


def parse_catalyst(raw: Optional[str]) -> str:
    """归并 catalyst → 5 类。

    先做同义词标准化，再按关键词归类。
    """
    if not raw:
        return "none_other"
    text = _normalize_text(raw)

    # 精确同义词匹配
    for syn, label in _CATALYST_SYNONYMS.items():
        if syn in text:
            return label

    # 模糊匹配
    if _match_any(text, ["acetic acid", "醋酸", "乙酸", "acoh", "hac", "ch3cooh"]):
        return "acetic_acid"
    if _match_any(text, ["ptsa", "p-toluenesulfonic", "对甲苯磺酸", "tsoh"]):
        return "PTSA"
    if _match_any(text, ["trifluoroacetic", "tfa", "三氟乙酸"]):
        return "TFA"
    if _match_any(text, ["sc(otf)3", "scandium triflate", "bf3", "lewis acid",
                          "路易斯酸", "zncl2", "fecl3", "alcl3"]):
        return "Sc(OTf)3_Lewis"
    if _match_any(text, ["none", "无", "no catalyst", "不加催化剂"]):
        return "none_other"

    return "none_other"


def parse_solvent(raw: Optional[str]) -> str:
    """归并 solvent → 7 类。

    先做名称标准化，再按主溶剂体系归类。
    """
    if not raw:
        return "other"
    text = _normalize_text(raw)

    # 名称标准化
    for old, new in _SOLVENT_NORMALIZE.items():
        text = text.replace(old, new)

    # 无溶剂
    if _match_any(text, ["无溶剂", "固相反应", "no solvent", "solvent-free",
                          "solid-state", "neat"]):
        return "other"

    # dioxane + mesitylene 体系
    has_dioxane = _match_any(text, ["dioxane"])
    has_mesitylene = _match_any(text, ["mesitylene"])
    if has_dioxane or has_mesitylene:
        return "dioxane_mesitylene"

    # DCB + BuOH 体系
    has_dcb = _match_any(text, ["dcb", "dichlorobenzene"])
    has_buoh = _match_any(text, ["buoh", "butanol"])
    if has_dcb or has_buoh:
        return "DCB_BuOH"

    # 水体系
    if _match_any(text, ["water", "aqueous", "h2o", "水相", "水溶液",
                          "buffer", "缓冲", "去离子水"]):
        return "water_based"

    # DMF/DMSO 体系
    if _match_any(text, ["dmf", "dmso", "nmp", "dimethylformamide",
                          "dimethyl sulfoxide", "氮甲基吡咯烷酮"]):
        return "DMF_DMSO"

    # 乙腈
    if _match_any(text, ["acetonitrile", "乙腈", "ch3cn"]):
        return "acetonitrile"

    # 醇类
    if _match_any(text, ["ethanol", "methanol", "乙醇", "甲醇", "alcohol",
                          "etoh", "meoh", "甘油", "glycerol", "乙二醇",
                          "ethylene glycol", "丁醇"]):
        return "alcohol"

    return "other"


def parse_temperature(raw: Optional[str]) -> str:
    """归并 temperature → 5 箱。

    rt(<30°C) / low(30-80) / standard(80-130) / high(130-200) / unknown
    正则提取数字，范围取最高温（多步骤时最苛刻条件）。
    """
    if not raw:
        return "unknown"
    text = raw.lower().strip()

    # 室温
    if _match_any(text, ["room temperature", "室温", "rt", "ambient",
                          "ultrasonication", "超声"]):
        return "rt"

    # 回流 → standard
    if _match_any(text, ["回流", "reflux"]):
        return "standard"

    # solvothermal 无温度 → standard
    if _match_any(text, ["solvothermal", "溶剂热"]) and not re.search(r"\d+", text):
        return "standard"

    # 提取所有数字+°C/℃/K
    temps = re.findall(r"(\d+)\s*(?:°c|℃|°|k)", text, re.IGNORECASE)
    if not temps:
        bare = re.findall(r"(\d{2,3})\b", text)
        if bare:
            temps = bare

    if not temps:
        return "unknown"

    temps_c = []
    for t in temps:
        val = int(t)
        temps_c.append(val)

    if not temps_c:
        return "unknown"

    max_temp = max(temps_c)

    if max_temp < 30:
        return "rt"
    elif max_temp < 80:
        return "low"
    elif max_temp < 130:
        return "standard"
    elif max_temp < 200:
        return "high"
    else:
        return "high"


class ConditionParser:
    """反应条件归并解析器。

    支持从同文献其他实验组补充缺失的 temperature 和 interface_type。
    """

    # 非反应温度关键词（后处理/测试/负载温度）
    _NON_REACTION_TEMP_KW = [
        "负载", "熔融", "干燥", "测试", "还原", "后处理",
        "drying", "impregnation", "melt", "test", "load",
        "fabrication", "separation", "性能测试", "活化",
        "退火", "annealing", "烘干", "蒸发", "evaporation",
    ]

    def parse(self,
              synthesis_route: Optional[str] = None,
              interface_type: Optional[str] = None,
              catalyst: Optional[str] = None,
              solvent: Optional[str] = None,
              temperature: Optional[str] = None) -> dict:
        """解析所有条件字段，返回标准化标签。"""
        return {
            "synthesis_route": parse_synthesis_route(synthesis_route),
            "interface_type": parse_interface_type(interface_type),
            "catalyst": parse_catalyst(catalyst),
            "solvent": parse_solvent(solvent),
            "temperature": parse_temperature(temperature),
        }

    def parse_row(self, row: dict) -> dict:
        """从数据库行 dict 解析条件。"""
        return self.parse(
            synthesis_route=row.get("synthesis_route"),
            interface_type=row.get("interface_type"),
            catalyst=row.get("catalyst"),
            solvent=row.get("solvent"),
            temperature=row.get("temperature"),
        )

    def is_reaction_temperature(self, temp_text: str) -> bool:
        """判断温度文本是否为反应温度（而非后处理/测试温度）。"""
        text = temp_text.lower()
        for kw in self._NON_REACTION_TEMP_KW:
            if kw in text:
                return False
        return True

    def fill_from_peers(self, rows: list[dict]) -> list[dict]:
        """从同文献其他实验组补充缺失的 temperature 和 interface_type。

        Args:
            rows: 同一文献的实验组列表，每行含 paper_id, group_id,
                  temperature, interface_type 等字段。

        Returns:
            补充后的 rows（原地修改）。
        """
        # 收集同文献的有效 temperature（排除非反应温度）
        valid_temps = []
        valid_interfaces = []
        for r in rows:
            t = r.get("temperature")
            if t and t.strip():
                parsed = parse_temperature(t)
                if parsed != "unknown" and self.is_reaction_temperature(t):
                    valid_temps.append(t)
            it = r.get("interface_type")
            if it and it.strip():
                valid_interfaces.append(it)

        # 取众数
        peer_temp = max(set(valid_temps), key=valid_temps.count) if valid_temps else None
        peer_interface = max(set(valid_interfaces), key=valid_interfaces.count) if valid_interfaces else None

        # 补充缺失
        for r in rows:
            t = r.get("temperature")
            if (not t or not t.strip()) and peer_temp:
                r["temperature"] = peer_temp
            it = r.get("interface_type")
            if (not it or not it.strip()) and peer_interface:
                r["interface_type"] = peer_interface

        return rows
