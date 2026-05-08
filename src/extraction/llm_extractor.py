import re
from typing import Optional, Dict, List

import yaml

from src.extraction.minimax_client import MiniMaxClient
from src.utils.logger import setup_logger

logger = setup_logger("extractor")


# System Prompt 模板 — COF 材料文献分析专家
SYSTEM_PROMPT_TEMPLATE = """你是一位专业的COF（共价有机框架）材料研究文献分析专家，尤其擅长二维亚胺键COF。
请仔细阅读提供的文献内容，严格按以下格式生成 YAML 结构化总结。

输出要求：
1. 以 YAML 格式输出，每行 字段key: 内容
2. 所有字段的值必须翻译为中文撰写
3. 期刊名、试剂名、化学物质名等专有名词保留英文原文，并在括号内加中文注释
   例如 journal: 'Angew. Chem. Int. Ed.（德国应用化学）'
   例如 reagent: 'ethidium bromide（溴化乙锭）、1,3,5-triformylphloroglucinol（1,3,5-三甲酰基间苯三酚）'
4. 文献中明确提及的内容才填写，未提及的字段填 null
5. 内容基于原文，不做无依据扩展
6. 结晶性需说明晶型（单晶/多晶/无定形）
7. 氟相关信息需说明是否含F或CF3基团

输出格式（严格按此结构，仅输出YAML块，不输出任何解释性文字）：
```yaml
{field_entries}
```

YAML格式注意：
- 字段从行首开始（不要添加前导空格，不要缩进），与上述顺序一致
- 所有值用单引号括起来
- 空值填 null
- 多行文本值用竖线 | 语法另起一行缩进"""


class LLMExtractor:
    """结构化文献提取器 — 调用 LLM 从全文提取 21 个字段"""

    def __init__(self, client: MiniMaxClient, fields: List[Dict],
                 max_input_chars: int = 8000):
        self.client = client
        self.fields = fields          # [{key, label}, ...]
        self.max_input_chars = max_input_chars
        self.field_keys = [f["key"] for f in fields]
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """根据字段定义构建 COF 专家 System Prompt"""
        entries = []
        for f in self.fields:
            entries.append(f"  {f['key']}: [{f['label']}]")
        field_block = "\n".join(entries)
        return SYSTEM_PROMPT_TEMPLATE.format(field_entries=field_block)

    def extract(self, txt_path: str) -> Optional[Dict]:
        """读取全文文本，调用 LLM 提取结构化信息"""
        # 读取文本
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        except Exception as e:
            logger.error(f"读取文本失败: {txt_path} - {e}")
            return None

        if not full_text or len(full_text) < 100:
            logger.warning(f"文本内容过短: {txt_path}")
            return None

        # 截断至 max_input_chars，优先保留开头（标题+摘要+引言最关键）
        text_truncated = full_text[:self.max_input_chars]

        # 构建用户提示词
        user_prompt = f"请分析以下文献内容并生成结构化 YAML：\n\n---\n{text_truncated}\n---"

        # 调用 LLM
        response = self.client.chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )
        if not response:
            logger.error(f"LLM 返回空响应: {txt_path}")
            return None

        # 解析 YAML
        data = self._parse_yaml_response(response)
        if data is None:
            logger.warning(f"YAML 解析失败: {txt_path}")
            return None

        # 补齐缺失字段
        result = self._validate_fields(data)
        return result

    def _parse_yaml_response(self, response: str) -> Optional[Dict]:
        """从 LLM 响应中提取 YAML 块并解析，失败时逐行正则兜底"""
        import textwrap

        # 尝试提取 ```yaml ... ``` 代码块
        yaml_match = re.search(r"```yaml\s*(.*?)```", response, re.DOTALL)
        if yaml_match:
            yaml_str = yaml_match.group(1).strip()
        else:
            yaml_str = response.strip()

        # 去除整段 YAML 的共同前导空格（LLM 可能无意中缩进）
        yaml_str = textwrap.dedent(yaml_str)

        # 先尝试标准 YAML 解析
        try:
            data = yaml.safe_load(yaml_str)
            if isinstance(data, dict):
                return data
        except yaml.YAMLError:
            pass

        # YAML 解析失败时，逐行正则提取 "key: value" 对
        result = {}
        for line in yaml_str.split("\n"):
            m = re.match(r"^\s*(\w+):\s*(.+)", line)
            if m:
                key = m.group(1)
                value = m.group(2).strip()
                # 去掉首尾单引号（LLM 可能按提示加上）
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                # 过滤掉明显不是字段的行
                if len(key) > 2 and value and value.lower() != "null":
                    result[key] = value
                elif value.lower() == "null":
                    result[key] = None
        if result:
            return result
        return None

    def _validate_fields(self, data: Dict) -> Dict:
        """确保所有字段存在，缺失的填 null"""
        result = {}
        for key in self.field_keys:
            result[key] = data.get(key) if data.get(key) else None
        return result
