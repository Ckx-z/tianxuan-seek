"""从 YAML 自由文本中提取结晶度和拓扑类型结构化标签。

输入字段: film_crystallinity_fluorine, system, synthesis_route
输出字段: crystallinity, topology_type, film_morphology
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

logger = setup_logger("struct_labels")

SYSTEM_PROMPT = """你是 COF 化学文献分析助手。输入多篇 COF 文献的原文片段，
对每篇提取以下结构化标签，输出 JSON 数组：

```json
[{
  "literature_id": "原样返回",
  "crystallinity": "high|medium|low|amorphous|unknown",
  "topology_type": "hcb|sql|kgm|dia|other|unknown",
  "film_morphology": "film|membrane|powder|bulk|nanosheet|composite|unknown"
}]
```

规则：
- crystallinity: PXRD 峰尖锐、高结晶性→"high"; 多晶粉末、中等结晶→"medium";
  结晶性差、宽峰→"low"; 无定形→"amorphous"; 无法判断→"unknown"
- topology_type: 根据单体连接方式判断:
  C3+C2 三连接+二连接→"hcb" (六方); C2+C2→"sql" (四方);
  C3+C1→"kgm" (Kagome); 金刚石拓扑→"dia"; 其他/混合→"other"; 无法判断→"unknown"
- film_morphology: 自支撑膜、连续薄膜、free-standing→"film";
  在基底上的膜、复合膜→"membrane"; 粉末、微晶粉末→"powder";
  块状→"bulk"; 纳米片→"nanosheet"; 复合材料→"composite"; 无法判断→"unknown"
- 字段无法判断时填 "unknown"
- 仅输出 JSON 数组，不要添加任何解释"""


def load_yaml_fields(yaml_dir: str) -> list[dict]:
    """读取所有 YAML，提取相关字段。"""
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
            "film_crystallinity_fluorine": str(data.get("film_crystallinity_fluorine") or "")[:300],
            "system": str(data.get("system") or "")[:200],
            "synthesis_route": str(data.get("synthesis_route") or "")[:200],
        })

    logger.info(f"读取 {len(rows)} 个 YAML")
    return rows


def _build_user_prompt(batch: list[dict]) -> str:
    """为一批文献构建用户提示。"""
    lines = []
    for r in batch:
        text = (
            f"成膜/结晶/氟: {r['film_crystallinity_fluorine']}\n"
            f"体系: {r['system']}\n"
            f"合成路线: {r['synthesis_route']}"
        )
        lines.append(f"--- 文献 {r['literature_id']} ---\n{text}")
    return "\n\n".join(lines)


def _parse_batch_response(content: str) -> list[dict]:
    """从 LLM 回复中解析 JSON 数组。处理 markdown 代码块和截断。"""
    if not content or not content.strip():
        return []

    code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    json_str = code_match.group(1) if code_match else content

    start = json_str.find("[")
    if start == -1:
        logger.warning(f"未找到 JSON 数组起始: {content[:200]}")
        return []

    decoder = __import__("json").JSONDecoder()
    json_str = json_str[start:]

    try:
        result, _ = decoder.raw_decode(json_str)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # 截断修复
    last_complete = json_str.rfind('"}')
    if last_complete > 0:
        try:
            fixed = json_str[:last_complete + 2] + "]"
            result, _ = decoder.raw_decode(fixed)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    last_obj = json_str.rfind('\n    {')
    if last_obj == -1:
        last_obj = json_str.rfind('\n  {')
    if last_obj == -1:
        last_obj = json_str.rfind('{')
    if last_obj > 0:
        try:
            fixed = json_str[:last_obj].rstrip().rstrip(",") + "\n]"
            result, _ = decoder.raw_decode(fixed)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning(f"JSON 解析失败: {json_str[:200]}")
    return []


def main():
    parser = argparse.ArgumentParser(description="提取结晶度、拓扑类型标签")
    parser.add_argument("--yaml-dir", default="data/structured")
    parser.add_argument("--output", default="data/processed/crystallinity_topology.csv")
    parser.add_argument("--batch-size", type=int, default=30,
                       help="每批发送的文献数 (30 推荐)")
    parser.add_argument("--limit", type=int, default=0,
                       help="限制处理篇数 (0=全部)")
    parser.add_argument("--provider", default="minimax",
                       choices=["minimax", "mimo"])
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
    all_rows = load_yaml_fields(args.yaml_dir)
    if args.limit > 0:
        all_rows = all_rows[:args.limit]

    # 断点续跑
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

    # 时间预估
    est_seconds = n_batches * 5  # ~3s API + 1.5s sleep
    est_min = est_seconds / 60
    logger.info(f"开始提取: 剩余 {total} 篇, {n_batches} 批次, 预估 {est_min:.0f} 分钟")

    t_start = time.time()

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, total)
        batch = all_rows[start:end]

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
                    max_tokens=8192,
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
            pd.DataFrame(results).to_csv(tmp_path, index=False, encoding="utf-8-sig")
            elapsed = time.time() - t_start
            pct = (bi + 1) / n_batches * 100
            eta = elapsed / (bi + 1) * (n_batches - bi - 1)
            logger.info(
                f"  [{bi+1}/{n_batches}] {len(parsed)} 条 (累计 {len(results)}), "
                f"进度 {pct:.0f}%, 预计剩余 {eta:.0f}s"
            )
        else:
            logger.error(f"    批次 {bi+1} 完全失败")

        if bi < n_batches - 1:
            time.sleep(1.5)

    # 保存
    if not results:
        logger.error("未提取到任何标签")
        sys.exit(1)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {args.output} ({len(df)} 条)")

    # 统计
    print("\n" + "=" * 60)
    print("  结晶度 & 拓扑类型标签提取完成")
    print("=" * 60)
    for col in ["crystallinity", "topology_type", "film_morphology"]:
        if col in df.columns:
            counts = df[col].value_counts()
            total = len(df)
            print(f"\n  {col}:")
            for val, cnt in counts.items():
                print(f"    {val}: {cnt} ({100*cnt/total:.0f}%)")
    print(f"\n  输出: {args.output}")
    print(f"  耗时: {time.time() - t_start:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
