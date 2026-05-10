"""通过 LLM 从 reagent 字段提取单体 SMILES。

输入: data/structured/*.yaml (954篇)
输出: data/processed/monomer_smiles_llm.json

流程:
  1. 读取每篇 YAML 的 reagent + system + film_crystallinity_fluorine
  2. LLM 识别单体名称 → 生成 SMILES → 分类醛/胺 → 判定含氟
  3. RDKit 验证 SMILES 有效性
  4. 汇总去重到 JSON 库
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from rdkit import Chem

from src.extraction.minimax_client import MiniMaxClient, load_extraction_config, create_minimax_client, create_fallback_client
from src.utils.logger import setup_logger

logger = setup_logger("monomer_smiles")

SYSTEM_PROMPT = """你是一位精通有机化学和COF（共价有机框架）材料的化学信息学专家。
你的任务是从文献的试剂信息中，识别出用于合成COF的单体分子，并提供其SMILES表示。

## 关键要求

1. **识别单体**：从reagent文本中提取实际参与COF骨架构建的单体（醛类、胺类、硼酸类等），忽略溶剂、催化剂
2. **提供SMILES**：为每个单体提供标准的Canonical SMILES字符串
3. **分类**：判断每个单体是醛类(aldehyde)、胺类(amine)、醛胺双功能(aldehyde-amine)还是其他(other)
4. **含氟判定**：判断每个单体是否含有氟原子
5. **2D COF判定**：根据研究体系描述，判断该文献研究的COF是否为二维(2D)结构
6. **成膜判定**：根据film_crystallinity_fluorine字段，判断该文献是否报道成功制备了COF薄膜

## 输出格式

仅输出以下JSON格式（不要包含任何其他文字、解释、或markdown标记）：

{
  "is_2d_cof": true,
  "monomers": [
    {
      "name": "1,3,5-triformylphloroglucinol",
      "smiles": "O=Cc1c(O)c(C=O)c(O)c(C=O)c1O",
      "monomer_type": "aldehyde",
      "has_fluorine": false
    }
  ],
  "film_label": true
}

字段说明:
- is_2d_cof: true=二维COF, false=三维COF或其他材料, null=无法判断
- monomers: 每个参与COF骨架构建的单体信息数组
  - name: 英文化学名（尽量使用IUPAC名称）
  - smiles: Canonical SMILES（必须提供有效SMILES，无法确定时填null）
  - monomer_type: "aldehyde"(含醛基), "amine"(含伯胺基), "aldehyde-amine"(同时含醛和胺), "other"(其他)
  - has_fluorine: 分子中是否含氟原子
- film_label: true=成功制备COF薄膜, false=未成膜/粉末状, null=未提及

## SMILES注意事项

- 确保SMILES语法正确，原子价态合理
- 醛基写作 C=O 而非 CO
- 伯胺写作 N 而非 [N]
- 对于常见COF单体，务必使用准确的SMILES"""

USER_PROMPT_TEMPLATE = """请分析以下COF文献信息，提取单体的SMILES：

## 研究体系
{system}

## 试剂原料
{reagent}

## 成膜与结晶性
{film_info}

请识别所有参与COF骨架构建的单体，提供SMILES并分类。"""


# 进度线程锁
_progress_lock = threading.Lock()
_result_lock = threading.Lock()


def _log_failed_response(log_path: str, paper_id: str, response: str):
    """将 JSON 解析失败的原始 LLM 响应追加到 JSONL 日志文件。"""
    import json as _json
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        _json.dump({"literature_id": paper_id, "response": response}, f, ensure_ascii=False)
        f.write("\n")


def _check_2d_from_system(system: str) -> Optional[bool]:
    """从 system 字段快速预判是否为 2D COF（在调用 LLM 前的粗筛）。"""
    if not system:
        return None
    s = system.lower()
    # 明确 3D 关键词
    if any(kw in s for kw in ["3d cof", "三维cof", "three-dimensional cof",
                               "3d covalent", "三维共价"]):
        return False
    # 明确 2D 关键词
    if any(kw in s for kw in ["2d cof", "二维cof", "two-dimensional cof",
                               "2d covalent", "二维共价", "nanosheet",
                               "纳米片", "film", "薄膜", "membrane"]):
        return True
    return None


def _call_llm(
    client: MiniMaxClient,
    system_prompt: str,
    user_prompt: str,
    paper_name: str = "",
) -> Optional[str]:
    """调用 LLM，失败时返回 None（不抛异常）。"""
    try:
        return client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retry=3,
            sleep_between=3,
        )
    except Exception as e:
        logger.warning(f"LLM 调用异常 ({paper_name}): {e}")
        return None


def _process_one(
    yaml_path: Path,
    client: MiniMaxClient,
    index: int,
    total: int,
    failed_log_path: Optional[str] = None,
    fallback_client: Optional[MiniMaxClient] = None,
) -> Tuple[bool, Optional[Dict]]:
    """处理单篇文献：读取 YAML → LLM 提取 → 返回结构化数据。

    主 LLM 失败时自动降级到 fallback_client（若已配置）。"""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"[{index}/{total}] 读取失败: {yaml_path.name} - {e}")
        return (False, None)

    reagent = data.get("reagent", "") or ""
    system = data.get("system", "") or ""
    film_info = data.get("film_crystallinity_fluorine", "") or ""

    if not reagent.strip():
        return (False, None)

    # 快速筛选：跳过明确 3D COF
    is_2d_hint = _check_2d_from_system(system)
    if is_2d_hint is False:
        with _progress_lock:
            logger.info(f"[{index}/{total}] 跳过 (3D COF): {yaml_path.name}")
        return (True, {"is_2d_cof": False, "monomers": [], "film_label": None})

    user_prompt = USER_PROMPT_TEMPLATE.format(
        system=system or "未提及",
        reagent=reagent,
        film_info=film_info or "未提及",
    )

    # 主 LLM 调用
    response = _call_llm(client, SYSTEM_PROMPT, user_prompt, yaml_path.name)
    # 主 LLM 失败 → 尝试备用
    if response is None and fallback_client is not None:
        logger.info(f"[{index}/{total}] 主 LLM 失败，尝试备用: {yaml_path.name}")
        response = _call_llm(fallback_client, SYSTEM_PROMPT, user_prompt, yaml_path.name)
    if not response:
        logger.warning(f"[{index}/{total}] LLM 空响应: {yaml_path.name}")
        return (False, None)

    # 解析 JSON
    parsed = _parse_json_response(response)
    if parsed is None:
        # 保存失败响应供后续分析
        if failed_log_path:
            _log_failed_response(failed_log_path, yaml_path.stem, response)

        # 一次 JSON 修复重试（带原始上下文）
        fix_prompt = (
            f"请分析以下COF文献信息，提取单体的SMILES：\n\n"
            f"## 研究体系\n{system or '未提及'}\n\n"
            f"## 试剂原料\n{reagent}\n\n"
            f"## 成膜与结晶性\n{film_info or '未提及'}\n\n"
            f"请仅输出JSON格式结果，不要添加<think>标签、markdown标记或任何其他文字：\n"
            f'{{"is_2d_cof": true/false/null, "monomers": ['
            f'{{"name": "...", "smiles": "...", "monomer_type": "aldehyde/amine/other", "has_fluorine": true/false}}'
            f'], "film_label": true/false/null}}'
        )
        response2 = _call_llm(
            client,
            "你是一位精通有机化学的化学信息学专家。只输出JSON，不要输出<think>标签或任何其他内容。",
            fix_prompt,
            yaml_path.name,
        )
        if response2 is None and fallback_client is not None:
            response2 = _call_llm(
                fallback_client,
                "只输出JSON，不要其他内容。",
                fix_prompt,
                yaml_path.name,
            )
        if response2:
            parsed = _parse_json_response(response2)

        if parsed is None:
            logger.warning(f"[{index}/{total}] JSON 解析失败: {yaml_path.name}")
            snippet = response[:200].replace("\n", " ")
            logger.debug(f"  原始响应: {snippet}...")
            return (False, None)

    # 过滤掉无效 SMILES 的单体
    valid_monomers = []
    for m in parsed.get("monomers", []):
        if not isinstance(m, dict):
            logger.debug(f"  跳过非dict单体元素: {str(m)[:100]}")
            continue
        smi = m.get("smiles", "")
        if smi and smi != "null" and smi.lower() != "none":
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
                m["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
                valid_monomers.append(m)
            else:
                logger.debug(f"  RDKit 无效 SMILES: {m.get('name', '?')} → {smi}")
        elif m.get("name"):
            valid_monomers.append(m)  # 保留无SMILES但有名称的单体

    parsed["monomers"] = valid_monomers
    parsed["literature_id"] = yaml_path.stem

    with _progress_lock:
        logger.info(
            f"[{index}/{total}] OK ({len(valid_monomers)} monomers): {yaml_path.name}"
        )

    return (True, parsed)


def _parse_json_response(response: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON 块并解析，多层降级。"""
    if not response:
        return None

    # 0. 去掉 LLM thinking 标签 (<think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # 1. 提取代码块 (```json ... ``` 或 ``` ... ```)
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    json_str = m.group(1).strip() if m else cleaned.strip()
    json_str = json_str.strip().lstrip("﻿")

    # 2. 标准 JSON 解析
    parsed = _try_parse_json(json_str)
    if parsed is not None:
        return parsed

    # 3. 提取 { 到 } 范围
    m2 = re.search(r"\{.*\}", json_str, re.DOTALL)
    if m2:
        parsed = _try_parse_json(m2.group(0))
        if parsed is not None:
            return parsed

    # 4. 常见修复后重试
    fixed = _repair_json(json_str)
    if fixed:
        parsed = _try_parse_json(fixed)
        if parsed is not None:
            return parsed

    return None


def _try_parse_json(s: str) -> Optional[Dict]:
    """尝试解析 JSON，失败返回 None。"""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def _repair_json(s: str) -> Optional[str]:
    """尝试修复 LLM 常见 JSON 错误：尾部逗号、单引号、未转义字符。"""
    # 去掉尾随逗号（在 } 或 ] 之前）
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # 单引号替换为双引号（谨慎：值内单引号不做替换）
    # 只替换键和值周围的单引号
    s = re.sub(r"'([^']*)':", r'"\1":', s)  # key: 'xxx':
    s = re.sub(r":\s*'([^']*)'", r': "\1"', s)  # : 'value'
    # 修复 true/false/null 大小写
    s = re.sub(r':\s*True\b', ': true', s)
    s = re.sub(r':\s*False\b', ': false', s)
    s = re.sub(r':\s*Null\b', ': null', s)
    s = re.sub(r':\s*None\b', ': null', s)
    return s


def _validate_monomer_smiles(smiles: str) -> Optional[str]:
    """验证 SMILES 有效性，返回 Canonical SMILES 或 None。"""
    if not smiles or smiles.lower() in ("null", "none", ""):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True)


def run(
    input_dir: str = "data/structured",
    output_path: str = "data/processed/monomer_smiles_llm.json",
    cache_path: str = "data/processed/monomer_smiles_cache.json",
    limit: int = 0,
    workers: int = 3,
    skip_existing: bool = True,
):
    """批量提取单体 SMILES。

    参数:
        input_dir: YAML 结构化文件目录
        output_path: 输出 JSON 路径（LLM 识别的完整结果）
        cache_path: 并入已有缓存文件
        limit: 限制处理数量（0=全部）
        workers: 并发线程数
        skip_existing: 是否跳过已在缓存中的文献
    """
    yaml_dir = Path(input_dir)
    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    if not yaml_files:
        logger.error(f"未找到 YAML 文件: {yaml_dir}")
        sys.exit(1)

    total = min(len(yaml_files), limit) if limit else len(yaml_files)
    yaml_files = yaml_files[:total]
    logger.info(f"共 {len(yaml_files)} 篇文献待处理，并发数: {workers}")

    # 加载已有结果（断点续传），并预先去重
    previous_results: List[Dict] = []
    existing_ids: set = set()
    if skip_existing and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # 去重（防止之前中间保存产生的重复）
            seen = {}
            for r in raw:
                lid = r.get("literature_id", "")
                if lid and lid not in seen:
                    seen[lid] = r
            previous_results = list(seen.values())
            existing_ids = set(seen.keys())
            if len(raw) != len(previous_results):
                logger.info(f"已去重: {len(raw)} → {len(previous_results)} 条唯一记录")
            logger.info(f"已有 {len(existing_ids)} 条记录，将跳过已处理文献")
        except Exception:
            pass

    # 过滤未处理的文件
    task_list = []
    for p in yaml_files:
        if p.stem not in existing_ids:
            task_list.append(p)
    logger.info(f"待处理: {len(task_list)}, 已跳过: {len(yaml_files) - len(task_list)}")

    if not task_list:
        logger.info("所有文献已处理完毕")
        return

    # 初始化
    config = load_extraction_config()
    client = create_minimax_client(config)
    # MiMo 推理模型需大量 reasoning tokens，8192 确保有空间输出
    client.max_tokens = 8192

    # 备用 LLM（MiniMax 不可用时启用）
    fallback_client = create_fallback_client(config)
    if fallback_client:
        fallback_client.max_tokens = 8192
        logger.info(f"备用 LLM 已配置: {fallback_client.model}")
    else:
        logger.info("未配置备用 LLM")

    failed_log = os.path.join(os.path.dirname(output_path), "failed_json_responses.jsonl")
    # 轮转旧的失败日志
    if os.path.exists(failed_log):
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")
        bak = failed_log.replace(".jsonl", f"_{ts}.jsonl")
        os.rename(failed_log, bak)
        logger.info(f"旧失败日志已轮转: {os.path.basename(bak)}")

    results = []
    count_ok, count_fail = 0, 0
    total_tasks = len(task_list)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for idx, yaml_path in enumerate(task_list, start=1):
            future = executor.submit(
                _process_one, yaml_path, client, idx, total_tasks, failed_log, fallback_client,
            )
            futures[future] = yaml_path

        save_ticks = 0
        last_save_time = time.time()
        for future in as_completed(futures):
            success, data = future.result()
            if success and data is not None:
                with _result_lock:
                    results.append(data)
                count_ok += 1
            else:
                count_fail += 1

            save_ticks += 1
            done = count_ok + count_fail
            elapsed = time.time() - last_save_time
            # 每50篇 或 超过5分钟 保存一次中间结果
            if save_ticks >= 50 or elapsed > 300:
                try:
                    with _progress_lock:
                        logger.info(f"进度: {count_ok} OK, {count_fail} fail, {done}/{total_tasks}")
                    with _result_lock:
                        _save_intermediate(output_path, previous_results + list(results))
                except Exception as e:
                    logger.error(f"中间保存失败: {e}")
                save_ticks = 0
                last_save_time = time.time()

    # 合并：先前结果 + 当前新结果
    all_results = previous_results + list(results)

    # 去重（安全网）
    seen = {}
    for r in all_results:
        lid = r.get("literature_id", "")
        if lid and lid not in seen:
            seen[lid] = r
    final = list(seen.values())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # 统计
    total_monomers = sum(len(r.get("monomers", [])) for r in final)
    monomer_with_smiles = sum(
        1 for r in final for m in r.get("monomers", []) if m.get("smiles")
    )
    is_2d_count = sum(1 for r in final if r.get("is_2d_cof") is True)
    film_count = sum(1 for r in final if r.get("film_label") is True)

    logger.info(f"完成: {count_ok} OK, {count_fail} fail")
    logger.info(f"总计: {len(final)} 篇文献, {total_monomers} 个单体实例")
    logger.info(f"  含有效 SMILES: {monomer_with_smiles}")
    logger.info(f"  2D COF: {is_2d_count}")
    logger.info(f"  成功成膜: {film_count}")
    logger.info(f"结果保存至: {output_path}")

    # 同步更新缓存
    _update_monomer_cache(output_path, cache_path)


def _load_existing(output_path: str) -> List[Dict]:
    """加载已有结果。"""
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_intermediate(output_path: str, data: List[Dict]):
    """保存中间结果（原子写入，失败不影响主流程）。"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp = output_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, output_path)
    except Exception as e:
        logger.error(f"中间保存异常: {e}")


def _update_monomer_cache(llm_output: str, cache_path: str):
    """将 LLM 识别的 SMILES 同步到 MonomerLibrary 缓存。"""
    if not os.path.exists(llm_output):
        return
    with open(llm_output, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 汇总所有 monomer name → SMILES
    smi_map = {}
    for r in results:
        for m in r.get("monomers", []):
            name = m.get("name", "").strip()
            smi = m.get("canonical_smiles", "") or m.get("smiles", "")
            if name and smi and smi.lower() not in ("null", "none", ""):
                # 验证 SMILES
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    smi_map[name.lower()] = Chem.MolToSmiles(mol, canonical=True)

    # 合并到现有缓存
    existing = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    existing.update(smi_map)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    logger.info(f"单体缓存更新: {len(smi_map)} 个新 SMILES (总计 {len(existing)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="通过 LLM 从文献 reagent 字段提取单体 SMILES"
    )
    parser.add_argument("--input", default="data/structured")
    parser.add_argument("--output", default="data/processed/monomer_smiles_llm.json")
    parser.add_argument("--cache", default="data/processed/monomer_smiles_cache.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理数量（0=全部）")
    parser.add_argument("--workers", type=int, default=3,
                        help="并发数（默认 3）")
    parser.add_argument("--no-skip", action="store_true",
                        help="不跳过已处理文献")
    args = parser.parse_args()

    run(
        input_dir=args.input,
        output_path=args.output,
        cache_path=args.cache,
        limit=args.limit,
        workers=args.workers,
        skip_existing=not args.no_skip,
    )
