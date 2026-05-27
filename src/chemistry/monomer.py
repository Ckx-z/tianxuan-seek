"""单体名称 → SMILES 解析模块。

通过内置 COF 常见单体字典 + PubChem API 三级降级，
将 reagent 字段中的化学名称转换为 RDKit Mol 对象。
"""
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

from rdkit import Chem
from src.utils.logger import setup_logger

logger = setup_logger("monomer")

# ---------------------------------------------------------------------------
# 内置 COF 单体字典：常见缩写/名称 → 已验证 Canonical SMILES
# ---------------------------------------------------------------------------
_BUILTIN_MONOMERS: Dict[str, str] = {
    # === 三醛单体（Tri-aldehydes）===
    "Tp": "O=Cc1c(O)c(C=O)c(O)c(C=O)c1O",
    "1,3,5-triformylphloroglucinol": "O=Cc1c(O)c(C=O)c(O)c(C=O)c1O",
    "1,3,5-三甲酰基间苯三酚": "O=Cc1c(O)c(C=O)c(O)c(C=O)c1O",
    "TFP": "O=Cc1cc(C=O)cc(C=O)c1",
    "TFB": "O=Cc1cc(C=O)cc(C=O)c1",
    "1,3,5-triformylbenzene": "O=Cc1cc(C=O)cc(C=O)c1",
    "1,3,5-benzenetricarboxaldehyde": "O=Cc1cc(C=O)cc(C=O)c1",
    "1,3,5-三甲酰基苯": "O=Cc1cc(C=O)cc(C=O)c1",
    "TFPB": "O=Cc1ccc(-c2cc(-c3ccc(C=O)cc3)cc(-c3ccc(C=O)cc3)c2)cc1",
    "1,3,5-tris(4-formylphenyl)benzene": "O=Cc1ccc(-c2cc(-c3ccc(C=O)cc3)cc(-c3ccc(C=O)cc3)c2)cc1",
    "1,3,5-三(4-甲酰基苯基)苯": "O=Cc1ccc(-c2cc(-c3ccc(C=O)cc3)cc(-c3ccc(C=O)cc3)c2)cc1",
    "TPAL": "O=Cc1ccc(N(c2ccc(C=O)cc2)c2ccc(C=O)cc2)cc1",
    "tris(4-formylphenyl)amine": "O=Cc1ccc(N(c2ccc(C=O)cc2)c2ccc(C=O)cc2)cc1",
    "三(4-甲酰基苯基)胺": "O=Cc1ccc(N(c2ccc(C=O)cc2)c2ccc(C=O)cc2)cc1",

    # === 二醛单体（Di-aldehydes）===
    "TA": "O=Cc1ccc(C=O)cc1",
    "terephthalaldehyde": "O=Cc1ccc(C=O)cc1",
    "对苯二甲醛": "O=Cc1ccc(C=O)cc1",
    "TFTA": "O=Cc1c(F)c(F)c(C=O)c(F)c1F",
    "2,3,5,6-tetrafluoroterephthalaldehyde": "O=Cc1c(F)c(F)c(C=O)c(F)c1F",
    "2,3,5,6-四氟对苯二甲醛": "O=Cc1c(F)c(F)c(C=O)c(F)c1F",
    "DVA": "O=Cc1cc(OC)c(C=O)cc1OC",
    "2,5-dimethoxyterephthalaldehyde": "O=Cc1cc(OC)c(C=O)cc1OC",
    "2,5-二甲氧基对苯二甲醛": "O=Cc1cc(OC)c(C=O)cc1OC",
    "DHTA": "O=Cc1cc(O)c(C=O)cc1O",
    "2,5-dihydroxyterephthalaldehyde": "O=Cc1cc(O)c(C=O)cc1O",
    "2,5-二羟基对苯二甲醛": "O=Cc1cc(O)c(C=O)cc1O",
    "DCTP": "O=Cc1cc(Cl)c(C=O)cc1Cl",
    "2,5-dichloroterephthalaldehyde": "O=Cc1cc(Cl)c(C=O)cc1Cl",
    "BPDA": "O=Cc1ccc(-c2ccc(C=O)cc2)cc1",
    "biphenyl-4,4'-dicarboxaldehyde": "O=Cc1ccc(-c2ccc(C=O)cc2)cc1",
    "4,4'-联苯二甲醛": "O=Cc1ccc(-c2ccc(C=O)cc2)cc1",
    "TFBPDA": "O=Cc1c(F)c(F)c(-c2c(F)c(F)c(C=O)c(F)c2F)c(F)c1F",
    "2,3,5,6,2',3',5',6'-octafluorobiphenyl-4,4'-dicarboxaldehyde": "O=Cc1c(F)c(F)c(-c2c(F)c(F)c(C=O)c(F)c2F)c(F)c1F",
    "2-fluoro-terephthalaldehyde": "O=Cc1ccc(C=O)c(F)c1",
    "2-氟对苯二甲醛": "O=Cc1ccc(C=O)c(F)c1",

    # === 三胺单体（Tri-amines）===
    "TAPB": "Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1",
    "TAPB-1": "Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1",
    "1,3,5-tris(4-aminophenyl)benzene": "Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1",
    "1,3,5-三(4-氨基苯基)苯": "Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1",
    "TAPT": "Nc1ccc(-c2nc(-c3ccc(N)cc3)nc(-c3ccc(N)cc3)n2)cc1",
    "2,4,6-tris(4-aminophenyl)-1,3,5-triazine": "Nc1ccc(-c2nc(-c3ccc(N)cc3)nc(-c3ccc(N)cc3)n2)cc1",
    "2,4,6-三(4-氨基苯基)-1,3,5-三嗪": "Nc1ccc(-c2nc(-c3ccc(N)cc3)nc(-c3ccc(N)cc3)n2)cc1",
    "TAPA": "Nc1ccc(N(c2ccc(N)cc2)c2ccc(N)cc2)cc1",
    "tris(4-aminophenyl)amine": "Nc1ccc(N(c2ccc(N)cc2)c2ccc(N)cc2)cc1",
    "三(4-氨基苯基)胺": "Nc1ccc(N(c2ccc(N)cc2)c2ccc(N)cc2)cc1",
    "TAPM": "Nc1ccc(C(c2ccc(N)cc2)(c2ccc(N)cc2)c2ccc(N)cc2)cc1",
    "tetrakis(4-aminophenyl)methane": "Nc1ccc(C(c2ccc(N)cc2)(c2ccc(N)cc2)c2ccc(N)cc2)cc1",
    "四(4-氨基苯基)甲烷": "Nc1ccc(C(c2ccc(N)cc2)(c2ccc(N)cc2)c2ccc(N)cc2)cc1",

    # === 二胺单体（Di-amines）===
    "Pa": "Nc1ccc(N)cc1",
    "PDA": "Nc1ccc(N)cc1",
    "p-phenylenediamine": "Nc1ccc(N)cc1",
    "对苯二胺": "Nc1ccc(N)cc1",
    "1,4-苯二胺": "Nc1ccc(N)cc1",
    "BD": "Nc1ccc(-c2ccc(N)cc2)cc1",
    "benzidine": "Nc1ccc(-c2ccc(N)cc2)cc1",
    "联苯胺": "Nc1ccc(-c2ccc(N)cc2)cc1",
    "DABP": "Nc1ccc(-c2ccc(N)cc2)cc1",
    "4,4'-diaminobiphenyl": "Nc1ccc(-c2ccc(N)cc2)cc1",
    "DAB": "Nc1ccc(-c2ccc(N)c(N)c2)cc1N",
    "3,3'-diaminobenzidine": "Nc1ccc(-c2ccc(N)c(N)c2)cc1N",
    "TFDA": "Nc1c(F)c(F)c(N)c(F)c1F",
    "2,3,5,6-tetrafluoro-p-phenylenediamine": "Nc1c(F)c(F)c(N)c(F)c1F",
    "2,3,5,6-四氟对苯二胺": "Nc1c(F)c(F)c(N)c(F)c1F",
    "OFB": "Nc1c(F)c(F)c(-c2c(F)c(F)c(N)c(F)c2F)c(F)c1F",
    "octafluoro-4,4'-biphenyldiamine": "Nc1c(F)c(F)c(-c2c(F)c(F)c(N)c(F)c2F)c(F)c1F",
    "o-tolidine": "Nc1cc(C)c(-c2cc(C)c(N)cc2)cc1",
    "3,3'-dimethylbenzidine": "Nc1cc(C)c(-c2cc(C)c(N)cc2)cc1",
    "o-dianisidine": "Nc1cc(OC)c(-c2cc(OC)c(N)cc2)cc1",
    "DAMP": "Nc1cc(C)c(N)cc1C",
    "2,3,5,6-tetramethyl-p-phenylenediamine": "Nc1c(C)c(C)c(N)c(C)c1C",

    # === 特殊单体 ===
    "melamine": "Nc1nc(N)nc(N)n1",
    "三聚氰胺": "Nc1nc(N)nc(N)n1",
    "EB": "CC[N+](CC)(CC)c1cc2cc(N)ccc2c2c1cc(N)cc2",
    "ethidium bromide": "CC[N+](CC)(CC)c1cc2cc(N)ccc2c2c1cc(N)cc2",
    "溴化乙锭": "CC[N+](CC)(CC)c1cc2cc(N)ccc2c2c1cc(N)cc2",
    "TTA": "Nc1nc(N)nc(-c2ccc(N)cc2)n1",
    "2,4-diamino-6-(4-aminophenyl)-1,3,5-triazine": "Nc1nc(N)nc(-c2ccc(N)cc2)n1",
    "TpPa": None,  # 这是 COF 名称，不是单体
    "COF": None,
    "iCOF": None,

    # === 更多常见单体 ===
    "TFP": "O=Cc1cc(C=O)cc(C=O)c1",
    "TAPP": "Nc1ccc(C2=C3C=CC(=C3)N=C3C=CC(=C3)N=C3C=CC(=C3)N2)cc1",
    "tetrakis(4-aminophenyl)porphyrin": "Nc1ccc(C2=C3C=CC(=C3)N=C3C=CC(=C3)N=C3C=CC(=C3)N2)cc1",
    "TPB": "Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1",  # 同 TAPB
    "aniline": "Nc1ccccc1",
    "苯胺": "Nc1ccccc1",
    "benzaldehyde": "O=Cc1ccccc1",
    "苯甲醛": "O=Cc1ccccc1",
    "1,4-苯二甲醛": "O=Cc1ccc(C=O)cc1",
    "terephthaldehyde": "O=Cc1ccc(C=O)cc1",  # 常见拼写变体
    "1,4-phthalaldehyde": "O=Cc1ccc(C=O)cc1",
    "triformylphloroglucinol": "O=Cc1c(O)c(C=O)c(O)c(C=O)c1O",
    "2,4,6-triformylphloroglucinol": "O=Cc1c(O)c(C=O)c(O)c(C=O)c1O",
    "2,4,6-trihydroxybenzene-1,3,5-tricarbaldehyde": "O=Cc1c(O)c(C=O)c(O)c(C=O)c1O",
    "PTSA": None,  # 催化剂
    "p-toluenesulfonic acid": None,

    # === 含氟单体补充 ===
    "TFB-aldehyde": "O=Cc1c(F)c(F)c(C=O)c(F)c1F",  # 同 TFTA
    "4-fluorobenzaldehyde": "O=Cc1ccc(F)cc1",
    "2,4-difluorobenzaldehyde": "O=Cc1ccc(F)c(F)c1",
    "3,5-difluorobenzaldehyde": "O=Cc1cc(F)cc(F)c1",
    "pentafluorobenzaldehyde": "O=Cc1c(F)c(F)c(F)c(F)c1F",
    "3,5-bis(trifluoromethyl)aniline": "Nc1cc(C(F)(F)F)cc(C(F)(F)F)c1",
    "2,3,4,5,6-pentafluoroaniline": "Nc1c(F)c(F)c(F)c(F)c1F",

    # === 二醇/多酚单体（用于硼酸酯 COF）===
    "HHTP": "Oc1cc(O)c2c(O)c(O)c(O)c(O)c2c1",
    "2,3,6,7,10,11-hexahydroxytriphenylene": "Oc1cc(O)c2c(O)c(O)c(O)c(O)c2c1",
    "THB": "Oc1cc(O)c(O)cc1O",
    "1,2,4,5-tetrahydroxybenzene": "Oc1cc(O)c(O)cc1O",

    # === 其他功能单体 ===
    "hydrazine": "NN",
    "hydrazide": None,  # 官能团不是单体
    "tetrathiafulvalene": "S1C=C2SC=CS2S1",
    "TTF": "S1C=C2SC=CS2S1",
    "porphyrin": None,  # 骨架名
    "phthalocyanine": None,
    "triphenylamine": "N(c1ccccc1)(c1ccccc1)c1ccccc1",
    "三苯胺": "N(c1ccccc1)(c1ccccc1)c1ccccc1",
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "芘": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "tetraphenylethylene": "C(=C(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1",
    "TPE": "C(=C(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1",
    "spirobifluorene": "C12(c3ccccc3-c3ccccc13)c1ccccc1-c1ccccc21",
    "carbazole": "c1ccc2[nH]c3ccccc3c2c1",
    "thiophene": "c1ccsc1",
    "dibenzothiophene": "c1ccc2sc3ccccc3c2c1",
    "benzothiadiazole": "c1ccc2nsnc2c1",
    "BT": "c1ccc2nsnc2c1",
    "benzotrithiophene": None,  # BTT 骨架名，非具体单体
    "naphthalenediimide": "O=C1C2=CC=CC=C2C(=O)N1",
    "NDI": "O=C1C2=CC=CC=C2C(=O)N1",
    "perylenediimide": "O=C1C2=CC=C3C4=CC=C5C(=O)N(C6=CC=CC=C6)C(=O)C6=CC=C(C4=C56)C4=C3C2=C2C(=O)N(C3=CC=CC=C3)C(=O)C3=CC=C1C4=C32",
    "PDI": None,  # 骨架名
    "viologen": "C1=C[N+](C)=CC=C1",
    "bipyridine": "c1ccnc(-c2ncccc2)c1",
    "bpy": "c1ccnc(-c2ncccc2)c1",
    "phenanthroline": "c1cnc2c(c1)ccc1cccnc21",

    # === 常见溶剂和催化剂（标记为 None 避免误匹配）===
    "acetic acid": None,
    "醋酸": None,
    "DMF": None,
    "dimethylformamide": None,
    "DMSO": None,
    "THF": None,
    "dioxane": None,
    "mesitylene": None,
    "均三甲苯": None,
    "o-DCB": None,
    "n-BuOH": None,
    "methanol": None,
    "ethanol": None,
    "acetone": None,
    "acetonitrile": None,
    "dichloromethane": None,
    "DCM": None,
    "chloroform": None,
    "ethyl acetate": None,
    "EtOAc": None,
    "diethyl ether": None,
    "hexane": None,
    "toluene": None,
    "甲苯": None,
    "tetrahydrofuran": None,

    # === 氟化醛单体 ===
    "4-trifluoroacetylbenzaldehyde": "O=Cc1ccc(C(=O)C(F)(F)F)cc1",
    "4-三氟乙酰基苯甲醛": "O=Cc1ccc(C(=O)C(F)(F)F)cc1",
    "TFA": "O=Cc1ccc(C(=O)C(F)(F)F)cc1",

    # === 氟化胺单体 ===
    "4-fluoroaniline": "Nc1ccc(F)cc1",
    "4-氟苯胺": "Nc1ccc(F)cc1",
    "2-fluoroaniline": "Nc1ccccc1F",
    "2,4-difluoroaniline": "Nc1ccc(F)cc1F",
    "3,5-difluoroaniline": "Nc1cc(F)cc(F)c1",
    "3-(trifluoromethyl)aniline": "Nc1cccc(C(F)(F)F)c1",
    "4-(trifluoromethyl)aniline": "Nc1ccc(C(F)(F)F)cc1",
}

# 溶剂和催化剂黑名单（不参与单体提取）
_NON_MONOMER_KEYWORDS = [
    "acetic acid", "醋酸", "乙酸", "PTSA", "对甲苯磺酸", "Sc(OTf)",
    "三氟乙酸", "trifluoroacetic acid", "mesitylene", "均三甲苯",
    "dioxane", "二氧六环", "n-BuOH", "正丁醇", "butanol",
    "o-DCB", "邻二氯苯", "o-dichlorobenzene", "DMF", "DMAc",
    "THF", "四氢呋喃", "ethanol", "乙醇", "acetone", "丙酮",
    "acetonitrile", "乙腈", "CH3CN", "methanol", "甲醇",
    "dichloromethane", "DCM", "二氯甲烷", "chloroform", "三氯甲烷",
    "DMSO", "二甲基亚砜", "NMP", "N-甲基吡咯烷酮",
    "water", "水", "deionized water", "去离子水",
    "HCl", "盐酸", "NaOH", "氢氧化钠", "KOH", "氢氧化钾",
    "EtOH", "MeOH", "Et3N", "三乙胺", "triethylamine",
    "Pd", "Pd(PPh3)4", "Pd(OAc)2", "CuI", "CuBr",
    "NaBH4", "Na2SO4", "MgSO4", "K2CO3", "碳酸钾",
    "NaCl", "NaHCO3", "NH4Cl", "NH3", "氨水",
]


class MonomerLibrary:
    """COF 单体字典 — SMILES 解析、PubChem 缓存。"""

    def __init__(self, cache_path: str = "data/processed/monomer_smiles_cache.json",
                 use_pubchem: bool = True):
        self.cache_path = cache_path
        self.use_pubchem = use_pubchem
        self._cache: Dict[str, Optional[str]] = {}
        self._load_cache()
        self._pubchem_last_call = 0.0

    # -----------------------------------------------------------------------
    # 公共接口
    # -----------------------------------------------------------------------

    def resolve(self, name: str) -> Optional[str]:
        """名称 → SMILES，三级降级：内置字典 → 缓存 → PubChem API。

        会尝试从混合格式中提取多种候选名：
        'English（缩写，中文）' → 先试 English，再试缩写，再试中文
        """
        if not name:
            return None
        candidates = _expand_name_candidates(name)
        for cand in candidates:
            smi = self._resolve_single(cand)
            if smi:
                return smi
        return None

    def _resolve_single(self, clean: str) -> Optional[str]:
        """单一候选名的三级解析。"""
        if not clean:
            return None
        # 1. 内置字典
        smi = self._lookup_builtin(clean)
        if smi is not None:
            return smi if smi else None
        # 2. JSON 缓存
        if clean in self._cache:
            return self._cache[clean]
        # 3. PubChem（可选）
        if self.use_pubchem:
            smi = self._query_pubchem(clean)
            if smi:
                self._cache[clean] = smi
                self._save_cache()
                return smi
        return None

    def get_mol(self, name: str) -> Optional[Chem.Mol]:
        """名称 → RDKit Mol，解析失败返回 None。"""
        smi = self.resolve(name)
        if not smi:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning(f"SMILES 无法生成 Mol: {smi} (name={name})")
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        return mol

    def normalize_name(self, raw: str) -> str:
        """清洗：去首尾空白、去掉括号外前缀碎片、统一半角。"""
        if not raw:
            return ""
        s = raw.strip()
        # 去掉外侧中英文括号标注
        s = re.sub(r'^[（(]\s*', '', s)
        s = re.sub(r'\s*[）)]$', '', s)
        # 去掉前导碎片（逗号、数字、点开头的残留）
        s = re.sub(r'^[,，、.\d\s]+', '', s)
        # 全角转半角
        s = s.replace("（", "(").replace("）", ")").replace("，", ",").replace("；", ";")
        s = s.replace("′", "'").replace("″", "\"").replace("‴", "'''")
        # 去掉末尾句号、逗号、分号
        s = s.rstrip(".。,，;；")
        # 去掉多余的 "单体" "monomer" 后缀（但不影响缩写本身）
        s = re.sub(r'\s*[（(]?\s*(单体|monomer)\s*[）)]?\s*$', '', s, flags=re.I)
        return s.strip()

    def flush_cache(self):
        """强制保存缓存。"""
        self._save_cache()

    # -----------------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------------

    def _lookup_builtin(self, name: str) -> Optional[Optional[str]]:
        """内置字典查表，尝试精确匹配和驼峰变体。"""
        # 精确匹配
        if name in _BUILTIN_MONOMERS:
            return _BUILTIN_MONOMERS[name]
        # 小写匹配
        if name.lower() in _BUILTIN_MONOMERS:
            return _BUILTIN_MONOMERS[name.lower()]
        # 去掉括号内容后匹配
        base = re.sub(r'\s*\(.*?\)\s*', '', name).strip()
        if base in _BUILTIN_MONOMERS:
            return _BUILTIN_MONOMERS[base]
        if base.lower() in _BUILTIN_MONOMERS:
            return _BUILTIN_MONOMERS[base.lower()]
        return None  # 未命中字典，返回 None（非 "no SMILES"）

    def _query_pubchem(self, name: str) -> Optional[str]:
        """通过 PubChem PUG REST API 按名称查询 Canonical SMILES。"""
        # 限速：两次调用间隔 ≥ 0.6 秒
        elapsed = time.time() - self._pubchem_last_call
        if elapsed < 0.6:
            time.sleep(0.6 - elapsed)
        self._pubchem_last_call = time.time()

        encoded = quote(name, safe="")
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{encoded}/property/CanonicalSMILES/JSON"
        )
        try:
            resp = urlopen(url, timeout=10)
            data = json.loads(resp.read().decode())
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props and "CanonicalSMILES" in props[0]:
                smi = props[0]["CanonicalSMILES"]
                logger.info(f"PubChem 命中: {name} → {smi}")
                return smi
        except HTTPError as e:
            if e.code == 404:
                logger.debug(f"PubChem 未找到: {name}")
            else:
                logger.warning(f"PubChem 请求失败: {e}")
        except (URLError, OSError, ValueError, UnicodeError) as e:
            logger.warning(f"PubChem 网络/编码错误: {e}")
        return None

    def _load_cache(self):
        """从 JSON 文件加载缓存。"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        """保存缓存到 JSON 文件。"""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def extract_monomer_names(reagent_text: str) -> List[str]:
    """从 reagent 字段提取单体名称列表。

    按中文顿号/分号拆分，每个片段尝试提取英文 IUPAC 名（括号外）
    或缩写（括号内），优先匹配字典。
    """
    if not reagent_text:
        return []
    # 仅按中文分隔符拆分（避免英文逗号切断 IUPAC 名）
    parts = re.split(r'[、；]', reagent_text)
    monomers = []
    for p in parts:
        p = p.strip().strip("'\"")
        if not p or len(p) < 2:
            continue
        # 过滤溶剂/催化剂黑名单（完整匹配时才跳过）
        is_skip = any(
            kw.lower() == p.lower() or
            (len(kw) > 6 and kw.lower() in p.lower())
            for kw in _NON_MONOMER_KEYWORDS
        )
        if is_skip:
            continue

        # 尝试提取英文名（括号前的主文本）
        m = re.match(r'^([^(（]+)', p)
        name = m.group(1).strip() if m else p
        # 去掉首部常见的编号前缀如 "1. " "① "
        name = re.sub(r'^[\d①②③④⑤]+[.\s、．]*', '', name).strip()
        if name and len(name) >= 2:
            monomers.append(name)
    # 去重保序
    seen = set()
    result = []
    for m in monomers:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _expand_name_candidates(raw: str) -> List[str]:
    """从混合格式 'English（缩写，中文）' 提取候选名列表。

    策略：按优先级排列
    1. 英文全名（去除末尾括号标注后的主文本）
    2. 括号内的英文缩写（2-6 个大写字母）
    3. 中文名（全角括号内）
    4. 原始文本本身

    注意：IUPAC 名称内部含括号（如 tetrakis(4-formylphenyl)methane），
    不能简单在第一个 '(' 处截断。只去除末尾空格后紧跟的括号标注。
    """
    candidates = []
    raw = raw.strip()
    candidates.append(raw)  # 完整文本作为兜底

    # 提取末尾括号标注（空格 + 括号 + 缩写/中文）
    # 匹配末尾的 " (XXX)" 或 "（XXX）" 格式，其中 XXX 是缩写或中文
    tail_paren = re.search(r'\s+[（(]([^）)]+)[）)]?\s*$', raw)
    if tail_paren:
        main = raw[:tail_paren.start()].strip()
        if main and main != raw:
            candidates.insert(0, main)

        # 提取括号内缩写
        pc = tail_paren.group(1).strip()
        abbrs = re.findall(r'\b([A-Z][A-Za-z0-9]{1,7})\b', pc)
        for ab in abbrs:
            if ab not in candidates:
                candidates.insert(1, ab)
        if re.search(r'[一-鿿]', pc):
            candidates.append(pc)
    else:
        # 没有末尾括号标注时，尝试提取全角括号内的中文
        paren_content = re.findall(r'[（(]([^）)]*)[）)]', raw)
        for pc in paren_content:
            pc = pc.strip()
            if not pc:
                continue
            if re.search(r'[一-鿿]', pc):
                candidates.append(pc)

    # 去重保序
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ── v3 单体过滤扩展 ──────────────────────────────────────────────

# SMILES 中视为「有机」的元素（原子序数），不在白名单内的金属/半金属视为污染物
_ORGANIC_ATOMIC_NUMBERS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 52, 53}

# 单体名称中提示含金属的关键词（骨架/配体类）
_METAL_SKELETON_KEYWORDS = [
    "porphyrin", "卟啉", "phthalocyanine", "酞菁", "salen", "salophen",
    "TAPP", "Pc", "corrole", "咔咯", "MOF", "金属有机框架",
    "配合物", "络合物", "complex", "metalloligand", "金属配体",
    "bpy", "bipyridine", "phenanthroline", "菲咯啉", "terpyridine",
    "三联吡啶", "crown ether", "冠醚",
]

# 单体名称中的金属元素关键词（需词边界匹配，避免 diamine 误匹配 In）
_METAL_ELEMENT_PATTERN = re.compile(
    r'\b(Fe|Co|Ni|Cu|Zn|Mn|Ru|Rh|Pd|Pt|Ir|Os|Re|Cr|Mo|W|V|Ti|Zr|Hf|'
    r'Ag|Au|Cd|Hg|Al|Ga|In|Sn|Pb|Sb|Bi|Mg|Ca|Sr|Ba|'
    r'La|Ce|Eu|Tb)\b',
    re.IGNORECASE,
)


def has_metal_smiles(smiles: Optional[str]) -> bool:
    """从 SMILES 检测是否含金属/半金属原子。

    RDKit 解析后遍历所有原子，检查原子序数 > 18 且不在有机白名单中。
    同时检查 SMILES 字符串中的 [Metal] 标记（RDKit 可能无法解析某些金属配合物）。
    """
    if not smiles or not smiles.strip():
        return False
    s = smiles.strip()

    # 字符串级预检：[X] 方括号标记的非有机原子
    bracket_atoms = re.findall(r'\[([A-Z][a-z]?)', s)
    for atom in bracket_atoms:
        if atom not in {"C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
                        "Si", "Se", "Te", "B", "H", "He", "Li", "Be",
                        "Ne", "Ar", "Kr", "Xe", "Rn"}:
            return True

    # RDKit 原子序数检测
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num > 18 and atomic_num not in _ORGANIC_ATOMIC_NUMBERS:
            return True
    return False


def has_metal_name(name: Optional[str]) -> bool:
    """从单体名称检测是否含金属相关关键词。

    覆盖 porphyrin/phthalocyanine/salen/TAPP/corrode 等常见金属配体骨架，
    以及配合物/络合物/MOF 等金属有机体系。
    金属元素符号使用词边界匹配，避免 diamine→In 误匹配。
    """
    if not name or not name.strip():
        return False
    text = name.strip()

    # 骨架/配体关键词
    for kw in _METAL_SKELETON_KEYWORDS:
        if kw.lower() in text.lower():
            return True

    # 金属元素符号（词边界）
    if _METAL_ELEMENT_PATTERN.search(text):
        return True

    return False


def is_imine_only(chemistry_type: Optional[str]) -> bool:
    """检查 chemistry_type 是否为纯亚胺体系。

    排除：hydrazone/imide/boronate/olefin/triazine/phenazine/azine/
          squaraine/carbamate/urea/amide/mixed/hybrid/post-modification 等。
    """
    if not chemistry_type or not chemistry_type.strip():
        return False
    ct = chemistry_type.lower().strip()

    # 先检查黑名单（排除包含 imine 但实际是混合体系的情况）
    non_imine = [
        "hydrazone", "imide", "boronate", "boroxine", "boronic ester",
        "olefin", "vinylene", "triazine", "phenazine", "azine",
        "squaraine", "carbamate", "urea", "amide", "ester",
        "keto", "ketone", "ether", "thioether",
        "mixed", "hybrid", "dual", "hetero", "heterogeneous",
        "post-modification", "post-synthetic", "后修饰", "psm",
        "metal", "coordination", "配位",
        "not cof", "non-cof", "非cof",
        "thiazole", "oxazole", "reduced", "reduction",
        "polymer", "dynamic", "transformation", "conversion",
        "keto-enamine",
    ]
    for kw in non_imine:
        if kw in ct:
            return False

    # 白名单：明确为 imine/Schiff-base
    imine_whitelist = ["imine", "schiff base", "schiff-base", "亚胺", "席夫碱"]
    for w in imine_whitelist:
        if w in ct:
            return True

    return False


def normalize_fluorine_monomer_field(text: str) -> Optional[bool]:
    """将 fluorine_monomer 字段归一化为 True/False/None。

    模式匹配：
    - 阳性：是/含有含氟/CF3/F-substituted/F-containing
    - 阴性：否/不含/无氟/不含氟/非氟/none/false/null
    - 模糊：其他情况返回 None
    """
    if not text or text.strip().lower() in ("null", "none", "n/a", ""):
        return None
    t = text.strip().lower()

    positive_patterns = [
        r'^是', r'含有.*氟', r'含氟', r'存在.*氟', r'cf3',
        r'f[- ]?(substituted|containing)', r'fluorinated',
        r'^true$', r'^yes$',
        r'使用.*含氟', r'引入.*氟', r'含.*f\b', r'\bf\b.*单体',
        r'即有含氟', r'存在含氟', r'涉及.*含氟',
    ]
    for pat in positive_patterns:
        if re.search(pat, t):
            return True

    negative_patterns = [
        r'^否', r'不含氟', r'不含.*氟', r'无氟', r'无含氟',
        r'非氟', r'不含f', r'未使用.*含氟', r'不涉及.*氟',
        r'^false$', r'^no$',
        r'未提及含氟', r'没有.*氟', r'不含有氟',
        r'不含氟元素', r'无氟单体',
    ]
    for pat in negative_patterns:
        if re.search(pat, t):
            return False

    # 含「否」且无「是」→ 模糊，但大概率阴性
    if "否" in t and "是" not in t:
        return False

    return None
