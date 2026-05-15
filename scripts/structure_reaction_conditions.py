"""将 YAML 中的反应条件中文自由文本结构化，输出 ML-ready 特征 CSV。

处理字段: reaction_temperature, solvent, catalyst, synthesis_mode,
         interface_type, annealing_conditions, synthesis_route
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

logger = setup_logger("struct_cond")

SYSTEM_PROMPT = """你是化学实验条件提取助手。输入多篇 COF 文献的反应条件原文（中文），
对每篇提取以下结构化字段，输出 JSON 数组：

```json
[{
  "literature_id": "原样返回",
  "temperature_c": null 或数字(只填数值),
  "temperature_category": "room_temp|mild_heat(60-100)|solvothermal(100-200)|high_temp(>200)|reflux|unknown",
  "solvent_system": "monophasic|biphasic|solid_state|unknown",
  "solvent_main": "从列表选一个最匹配: DMF|NMP|dioxane|mesitylene|DMAc|water|EtOH|MeOH|THF|CHCl3|CH2Cl2|acetone|toluene|o-DCB|n-BuOH|ethylene_glycol|acetic_acid|none|other",
  "solvent_mixed": true或false,
  "catalyst_type": "acetic_acid|lewis_acid(Sc(OTf)3/BF3/ZnCl2等)|base|none|other|unknown",
  "synthesis_mode": "solvothermal|interfacial|mechanochemical|room_temp|reflux|ionothermal|other",
  "interface_type": "liquid_liquid|liquid_solid|gas_liquid|solid_solid|none|unknown",
  "has_annealing": true或false,
  "annealing_temp_c": null或数字,
  "is_homogeneous": true或false或null
}]
```

规则：
- 温度: "室温"→temperature_c=25, category="room_temp"; "120°C"→temperature_c=120; "120-150°C"→取均值135
- 溶剂: 如含DMSO也归入other; 水/有机两相→solvent_system="biphasic"; 无溶剂→"solid_state"
- 催化剂: 乙酸水溶液→"acetic_acid"; Sc(OTf)3/BF3/ZnCl2→"lewis_acid"; 明确说无催化剂→"none"
- synthesis_mode: 溶剂热(密封容器高温)→"solvothermal"; 界面聚合→"interfacial"; 研磨→"mechanochemical"
- interface_type: 两种不混溶液体接触→"liquid_liquid"; 在基底上生长→"liquid_solid"
- has_annealing: 有退火/加热后处理步骤→true
- is_homogeneous: 均相合成→true; 异相→false; 无法判断→null
- 字段无法判断时填 null"""


def load_yaml_conditions(yaml_dir: str) -> list[dict]:
    """读取所有 YAML，提取反应条件字段。"""
    rows = []
    yaml_files = sorted(f for f in os.listdir(yaml_dir) if f.endswith(".yaml"))

    for yf in yaml_files:
        path = os.path.join(yaml_dir, yf)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        lid = data.get("literature_id", yf.replace(".yaml", ""))
        rows.append({
            "literature_id": lid,
            "filename": yf,
            "reaction_temperature": str(data.get("reaction_temperature") or ""),
            "solvent": str(data.get("solvent") or ""),
            "catalyst": str(data.get("catalyst") or ""),
            "synthesis_mode": str(data.get("synthesis_mode") or ""),
            "interface_type": str(data.get("interface_type") or ""),
            "annealing_conditions": str(data.get("annealing_conditions") or ""),
            "synthesis_route": str(data.get("synthesis_route") or ""),
        })

    logger.info(f"读取 {len(rows)} 个 YAML 的条件字段")
    return rows


def _build_user_prompt(batch: list[dict]) -> str:
    """为一批文献构建 MiMo 用户提示。"""
    lines = []
    for i, r in enumerate(batch):
        text = (
            f"反应温度: {r['reaction_temperature'][:200]}\n"
            f"溶剂: {r['solvent'][:200]}\n"
            f"催化剂: {r['catalyst'][:200]}\n"
            f"合成方式: {r['synthesis_mode'][:200]}\n"
            f"界面类型: {r['interface_type'][:200]}\n"
            f"退火条件: {r['annealing_conditions'][:200]}\n"
            f"合成路线: {r['synthesis_route'][:200]}"
        )
        lines.append(f"--- 文献 {r['literature_id']} ---\n{text}")
    return "\n\n".join(lines)


def _parse_batch_response(content: str) -> list[dict]:
    """从 LLM 回复中解析 JSON 数组。处理 markdown 代码块和截断。"""
    if not content or not content.strip():
        return []

    # 提取 markdown 代码块
    code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    json_str = code_match.group(1) if code_match else content

    # 定位 JSON 数组
    start = json_str.find("[")
    if start == -1:
        logger.warning(f"未找到 JSON 数组起始: {content[:200]}")
        return []

    # 用 raw_decode 尝试逐步解析
    from json import JSONDecodeError, loads
    decoder = __import__("json").JSONDecoder()
    json_str = json_str[start:]

    try:
        result, _ = decoder.raw_decode(json_str)
        if isinstance(result, list):
            return result
    except (JSONDecodeError, ValueError):
        pass

    # 截断修复：找最后一个完整对象
    last_complete = json_str.rfind('"}')
    if last_complete > 0:
        try:
            fixed = json_str[:last_complete + 2] + "]"
            result, _ = decoder.raw_decode(fixed)
            if isinstance(result, list):
                return result
        except (JSONDecodeError, ValueError):
            pass

    # 最后尝试：去掉最后一个不完整对象
    last_obj_start = json_str.rfind('\n    {')
    if last_obj_start == -1:
        last_obj_start = json_str.rfind('\n  {')
    if last_obj_start == -1:
        last_obj_start = json_str.rfind('{')
    if last_obj_start > 0:
        try:
            fixed = json_str[:last_obj_start].rstrip().rstrip(",") + "\n]"
            result, _ = decoder.raw_decode(fixed)
            if isinstance(result, list):
                return result
        except (JSONDecodeError, ValueError):
            pass

    logger.warning(f"JSON 解析失败: {json_str[:200]}")
    return []


def main():
    parser = argparse.ArgumentParser(description="结构化反应条件字段")
    parser.add_argument("--yaml-dir", default="data/structured")
    parser.add_argument("--output", default="data/processed/reaction_conditions.csv")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="每批发送的文献数 (5 安全)")
    parser.add_argument("--limit", type=int, default=0,
                       help="限制处理篇数 (0=全部)")
    parser.add_argument("--provider", default="minimax",
                       choices=["minimax", "mimo"],
                       help="LLM 提供商 (default: minimax)")
    args = parser.parse_args()

    # 加载配置
    cfg_path = "config/extraction.yaml"
    if not os.path.exists(cfg_path):
        logger.error(f"配置文件不存在: {cfg_path}")
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.provider == "minimax":
        api_cfg = cfg["fallback"]
    else:
        api_cfg = cfg["minimax"]

    from openai import OpenAI

    client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg["api_base"])
    model = api_cfg["model"]
    logger.info(f"使用 {args.provider}: {model}")

    # 加载 YAML
    all_rows = load_yaml_conditions(args.yaml_dir)
    if args.limit > 0:
        all_rows = all_rows[:args.limit]

    # 加载已完成的结果（断点续跑）
    results = []
    done_ids = set()
    tmp_path = args.output.replace(".csv", "_tmp.csv")
    if os.path.exists(tmp_path):
        done_df = pd.read_csv(tmp_path, encoding="utf-8-sig")
        results = done_df.to_dict("records")
        done_ids = set(done_df["literature_id"])
        logger.info(f"断点续跑: 已有 {len(results)} 条, 跳过 {len(done_ids)} 篇")

    all_rows = [r for r in all_rows if r["literature_id"] not in done_ids]
    total = len(all_rows)
    batch_size = args.batch_size
    n_batches = (total + batch_size - 1) // batch_size

    logger.info(f"开始 MiniMax 结构化提取: 剩余 {total} 篇, {n_batches} 批次")

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, total)
        batch = all_rows[start:end]

        logger.info(f"  [{bi+1}/{n_batches}] 处理 {start+1}-{end}...")

        prompt = _build_user_prompt(batch)
        max_retry = 3
        parsed = []
        for attempt in range(1, max_retry + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=16384,
                )
                content = resp.choices[0].message.content
                parsed = _parse_batch_response(content)
                if parsed:
                    break
                logger.warning(f"    解析为空, 重试 {attempt}/{max_retry}")
            except Exception as e:
                logger.warning(f"    API 失败 (attempt {attempt}): {e}")
                if attempt < max_retry:
                    time.sleep(5)

        if parsed:
            results.extend(parsed)
            # 增量保存
            pd.DataFrame(results).to_csv(tmp_path, index=False, encoding="utf-8-sig")
            logger.info(f"    提取 {len(parsed)} 条 (累计 {len(results)})")
        else:
            logger.error(f"    批次 {bi+1} 完全失败")

        if bi < n_batches - 1:
            time.sleep(1.5)

    # 合并结果
    if not results:
        logger.error("未提取到任何结构化条件")
        sys.exit(1)

    df = pd.DataFrame(results)
    logger.info(f"结构化完成: {len(df)} 条")

    # 与原始文献 ID 对齐
    original_ids = {r["literature_id"] for r in load_yaml_conditions(args.yaml_dir)}
    matched = df[df["literature_id"].isin(original_ids)]
    logger.info(f"与原始文献匹配: {len(matched)}/{len(original_ids)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    matched.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {args.output}")

    # 统计
    print("\n" + "=" * 60)
    print("  反应条件结构化提取完成")
    print("=" * 60)
    for col in matched.columns:
        if col == "literature_id":
            continue
        non_null = matched[col].notna().sum()
        print(f"  {col}: {non_null}/{len(matched)} ({100*non_null/max(len(matched),1):.0f}%)")
    print(f"\n  输出: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
