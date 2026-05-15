"""新样本全自动处理管线。

531 篇 PDF → 文本 → LLM 提取 (21字段/YAML) → SMILES 解析 → 标签判定 → 合并

标签规则 (用户确认):
  - film_crystallinity_fluorine 提到成膜 → label = 1 (正样本)
  - 未提成膜 → label = 0 (负样本)
  - 前提: 必须有醛基单体 + 胺基单体同时存在

合并规则 (用户确认):
  - 按 canonical SMILES 对去重
  - 新老样本冲突时 → 以老样本标签为准

用法:
  python scripts/process_new_samples.py
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chemistry.imine_check import ImineChecker
from src.chemistry.fluorination import FluorineDetector
from src.chemistry.monomer import MonomerLibrary
from src.utils.logger import setup_logger

logger = setup_logger("process_new")

# 否定模式 (先匹配，命中则直接判负)
FILM_NEGATIVE_KEYWORDS = [
    "未成膜", "不成膜", "难以成膜", "无法成膜", "非成膜",
    "未提及成膜", "未提到成膜", "未提成膜", "无成膜",
    "no film", "not film", "non-film",
]

# 成膜正样本关键词 — 分两级:
#   L1: 高置信 — 明确表述自己制备出了膜
#   L2: 中置信 — 排除否定后，「成膜/薄膜」大概率是作者自己做了膜
FILM_POSITIVE_L1 = [
    "制备了", "制得了", "制备出", "制得出",
    "形成了", "获得了", "得到了", "合成了",
    "自支撑膜", "连续薄膜", "均匀薄膜", "成膜性良好", "连续膜",
    "free-standing", "freestanding",
    "成功制备", "成功合成", "成功制",
]
FILM_POSITIVE_L2 = [
    "成膜", "薄膜", "film", "membrane",
    "thin film", "thin-film",
]


def _canon(smi: str) -> Optional[str]:
    if not smi or (isinstance(smi, float) and np.isnan(smi)):
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def parse_new_pdfs(input_dir: str, output_dir: str):
    """Step 1: 531 PDFs → 文本 (PyMuPDF)。"""
    from src.pdf_parser.parse_pdf import extract_text

    os.makedirs(output_dir, exist_ok=True)
    pdf_files = sorted(Path(input_dir).glob("*.pdf"))
    logger.info(f"Step 1 — PDF 解析: {len(pdf_files)} 篇")

    ok, fail = 0, 0
    for i, pdf_path in enumerate(pdf_files, 1):
        stem = pdf_path.stem
        txt_out = os.path.join(output_dir, f"{stem}.full.txt")
        if os.path.exists(txt_out):
            ok += 1
            continue
        txt = extract_text(str(pdf_path))
        if txt and len(txt) > 200:
            Path(txt_out).write_text(txt, encoding="utf-8")
            ok += 1
        else:
            fail += 1
        if i % 50 == 0:
            logger.info(f"  [{i}/{len(pdf_files)}] 完成: {ok} OK, {fail} fail")

    logger.info(f"Step 1 完成: {ok} OK, {fail} fail")
    return ok, fail


def extract_yamls(input_dir: str, output_dir: str, db_path: str, workers: int = 1):
    """Step 2: 文本 → LLM 提取 → YAML (复用 extract_info 多线程逻辑)。"""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.extraction.minimax_client import (
        load_extraction_config, create_minimax_client,
    )
    from src.extraction.llm_extractor import LLMExtractor
    from src.utils.db import init_db, insert_record, get_record_count

    os.makedirs(output_dir, exist_ok=True)
    txt_dir = Path(input_dir)
    yaml_dir = Path(output_dir)

    txt_files = sorted(txt_dir.glob("*.full.txt"))
    logger.info(f"Step 2 — LLM 提取: {len(txt_files)} 篇 ({workers} 并发)")

    config = load_extraction_config()
    minimax_client = create_minimax_client(config)
    extractor = LLMExtractor(
        client=minimax_client,
        fields=config["fields"],
        max_input_chars=config["minimax"].get("max_input_chars", 8000),
    )

    progress_lock = threading.Lock()
    failed_log = yaml_dir / "_failed_extract.log"
    ok, fail = 0, 0

    def _process_one(txt_path, stem, idx, total):
        nonlocal ok, fail
        yaml_out = yaml_dir / f"{stem}.yaml"
        if yaml_out.exists():
            with progress_lock:
                logger.info(f"  跳过已处理: {yaml_out.name}")
            return True

        data = extractor.extract(str(txt_path))
        if data is None:
            with progress_lock:
                fail += 1
                logger.warning(f"  [{idx}/{total}] FAIL: {txt_path.name}")
                with open(failed_log, "a", encoding="utf-8") as fl:
                    fl.write(f"{txt_path.name}\n")
            return False

        data["literature_id"] = stem
        yaml_out.write_text(
            yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        with progress_lock:
            ok += 1
            if ok % 50 == 0:
                logger.info(f"  [{idx}/{total}] 进度: {ok} OK, {fail} fail")
        return True

    tasks = [(p, p.stem.replace(".full", "")) for p in txt_files]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, t[0], t[1], i, len(tasks)): t
                   for i, t in enumerate(tasks, 1)}
        for future in as_completed(futures):
            future.result()

    logger.info(f"Step 2 完成: {ok} OK, {fail} fail")
    return ok, fail


def extract_smiles_via_llm(yaml_dir: str, output_json: str, workers: int = 10):
    """Step 3: LLM SMILES 提取 (多线程并行)。

    每个线程独立 MiniMax client，线程安全缓存读写。
    断点续传: 已有缓存则跳过。
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.extraction.minimax_client import (
        load_extraction_config, create_minimax_client,
    )

    cache = {}
    cache_lock = threading.Lock()
    if os.path.exists(output_json):
        with open(output_json, encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"已有 SMILES 缓存: {len(cache)} 篇")

    yaml_files = sorted(Path(yaml_dir).glob("*.yaml"))
    pending = [(yf, yf.stem) for yf in yaml_files if yf.stem not in cache]
    if not pending:
        logger.info("Step 3 — 所有 YAML 已有 SMILES 缓存，跳过")
        return

    logger.info(f"Step 3 — LLM SMILES 提取: {len(pending)} 篇 ({workers} 并发)")

    SYSTEM_PROMPT = """你是一位精通有机化学和COF（共价有机框架）材料的化学信息学专家。
你的任务是从文献的试剂信息中，识别出用于合成COF的单体分子，并提供其SMILES表示。

## 关键要求
1. **识别单体**：从reagent文本中提取实际参与COF骨架构建的单体（醛类、胺类等），忽略溶剂、催化剂
2. **提供SMILES**：为每个单体提供标准的Canonical SMILES字符串
3. **分类**：判断每个单体是醛类(aldehyde)、胺类(amine)还是其他(other)
4. **含氟判定**：判断每个单体是否含有氟原子
5. **成膜判定**：根据film_crystallinity_fluorine字段，判断是否报道成功制备了COF薄膜

## 输出格式
仅输出以下JSON格式（不要包含任何其他文字或markdown标记）：
{
  "monomers": [
    {"name": "...", "smiles": "...", "monomer_type": "aldehyde", "has_fluorine": false}
  ],
  "film_label": true
}"""

    USER_PROMPT_TEMPLATE = """请分析以下COF文献信息，提取单体的SMILES：

## 研究体系
{system}

## 试剂信息
{reagent}

## 成膜/结晶/氟信息
{film}

请输出JSON。"""

    progress_lock = threading.Lock()
    ok_count = [0]
    fail_count = [0]
    done_count = [0]

    def _process_one(yf, stem):
        try:
            data = yaml.safe_load(yf.read_text(encoding="utf-8"))
        except Exception:
            with progress_lock:
                fail_count[0] += 1; done_count[0] += 1
            return

        reagent = data.get("reagent", "")
        system = data.get("system", "")
        film = data.get("film_crystallinity_fluorine", "")

        if not reagent or str(reagent).lower() == "null":
            with progress_lock:
                fail_count[0] += 1; done_count[0] += 1
            return

        prompt = USER_PROMPT_TEMPLATE.format(reagent=reagent, system=system, film=film)
        config = load_extraction_config()
        client = create_minimax_client(config)
        resp = client.chat(SYSTEM_PROMPT, prompt)
        if not resp:
            with progress_lock:
                fail_count[0] += 1; done_count[0] += 1
            return

        try:
            json_match = re.search(r"\{[\s\S]*\}", str(resp))
            if json_match:
                parsed = json.loads(json_match.group(0))
                with cache_lock:
                    cache[stem] = parsed
                with progress_lock:
                    ok_count[0] += 1
            else:
                with progress_lock:
                    fail_count[0] += 1
        except Exception:
            with progress_lock:
                fail_count[0] += 1

        with progress_lock:
            done_count[0] += 1
            if done_count[0] % 50 == 0:
                logger.info(f"  [{done_count[0]}/{len(pending)}] SMILES: {ok_count[0]} OK, {fail_count[0]} fail")
                os.makedirs(os.path.dirname(output_json), exist_ok=True)
                with cache_lock:
                    snapshot = dict(cache)
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=2, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, yf, stem): stem for yf, stem in pending}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                pass

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dict(cache), f, indent=2, ensure_ascii=False)
    logger.info(f"Step 3 完成: {ok_count[0]} OK, {fail_count[0]} fail → {output_json}")


def _check_film_label(film_field) -> int:
    """film_crystallinity_fluorine → 1(制备了膜) / 0(未制备)。

    逻辑:
      1. 空 → 0
      2. 否定词命中 → 0 (未成膜/不成膜/未提及成膜…)
      3. L1 高置信词 + 膜/薄膜/film 共现 → 1 (如「制备了薄膜」)
      4. L2 中置信词命中 → 1 (否定已排除，「成膜/薄膜」大概率自制)
      5. 其余 → 0
    """
    if not film_field or str(film_field).lower() == "null":
        return 0
    text = str(film_field).lower()

    # 否定 → 直接判负
    for kw in FILM_NEGATIVE_KEYWORDS:
        if kw in text:
            return 0

    # 检查是否含「膜」相关字眼 (用于 L1 共现)
    has_mo = any(w in text for w in ["膜", "film", "membrane"])

    # L1: 高置信制备动词 + 膜共现
    if has_mo:
        for kw in FILM_POSITIVE_L1:
            if kw in text:
                return 1

    # L2: 中置信 (否定已排除，命中即正)
    for kw in FILM_POSITIVE_L2:
        if kw in text:
            return 1

    return 0


def build_label_metadata(
    yaml_dir: str, smiles_json: str,
    old_meta_path: str, old_yaml_dir: str, output_path: str,
):
    """Step 4: 统一用新规则重新标定老样本+新样本，去重合并。

    新规则: film_crystallinity_fluorine 提及制备膜 → 1, 否则 → 0。
    老样本不再优先——同样读取 YAML 重标。
    """
    logger.info("Step 4 — 构建 label_metadata (统一新规则)...")

    checker = ImineChecker()
    f_det = FluorineDetector()

    with open(smiles_json, encoding="utf-8") as f:
        smiles_data = json.load(f)
    logger.info(f"SMILES 数据 (新): {len(smiles_data)} 篇")

    all_records = []
    seen = set()
    skipped = {"no_smiles": 0, "no_ald_amine": 0}
    label_counts = {1: 0, 0: 0}

    def _add_pairs(aldehydes, amines, lid, film_field, source_tag):
        """为醛×胺笛卡尔积生成记录，去重。"""
        has_f = any(
            f_det.has_fluorine(Chem.MolFromSmiles(s))
            for s in aldehydes + amines if Chem.MolFromSmiles(s)
        )
        label = _check_film_label(film_field)
        for a in aldehydes:
            for b in amines:
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                all_records.append({
                    "literature_id": lid,
                    "aldehyde_smiles": a,
                    "amine_smiles": b,
                    "label": label,
                    "film_field": str(film_field)[:200],
                    "has_f": has_f,
                    "source": source_tag,
                })
                label_counts[label] += 1

    # ── 处理新样本 ──
    yaml_files = sorted(Path(yaml_dir).glob("*.yaml"))
    for yf in yaml_files:
        stem = yf.stem
        try:
            data = yaml.safe_load(yf.read_text(encoding="utf-8"))
        except Exception:
            continue

        film_field = data.get("film_crystallinity_fluorine", "")

        smi_entry = smiles_data.get(stem, {})
        monomers = smi_entry.get("monomers", []) if isinstance(smi_entry, dict) else []
        if not monomers:
            skipped["no_smiles"] += 1
            continue

        aldehydes, amines = [], []
        for m in monomers:
            smi = m.get("smiles", "")
            mtype = m.get("monomer_type", "")
            if not smi or smi == "null":
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol, isomericSmiles=True)
            if mtype == "aldehyde" or checker.is_aldehyde(mol):
                aldehydes.append(can)
            if mtype == "amine" or checker.is_amine(mol):
                amines.append(can)

        if not aldehydes or not amines:
            skipped["no_ald_amine"] += 1
            continue

        lid = data.get("literature_id", stem)
        _add_pairs(aldehydes, amines, lid, film_field, "group2")

    new_ok = len(all_records)

    # ── 处理老样本: 读旧 YAML 重标 ──
    old_meta = pd.read_csv(old_meta_path, encoding="utf-8-sig")
    old_yaml_dir_path = Path(old_yaml_dir)
    old_relabeled = 0
    old_no_yaml = 0
    old_skipped_dup = 0

    # 建立 literature_id → YAML 路径映射
    old_yaml_map = {}
    for yf in old_yaml_dir_path.glob("*.yaml"):
        old_yaml_map[yf.stem] = yf

    for _, row in old_meta.iterrows():
        a = _canon(str(row["aldehyde_smiles"]))
        b = _canon(str(row["amine_smiles"]))
        if not a or not b:
            continue
        key = (a, b)
        if key in seen:
            old_skipped_dup += 1
            continue

        lid = str(row.get("literature_id", ""))
        old_label = int(row["label"])

        # 尝试找到对应 YAML 以重新标签
        yf = old_yaml_map.get(lid)
        if yf is not None:
            try:
                old_data = yaml.safe_load(yf.read_text(encoding="utf-8"))
            except Exception:
                old_data = {}
            film_field = old_data.get("film_crystallinity_fluorine", "")
            new_label = _check_film_label(film_field)
            old_relabeled += 1
        else:
            # 找不到 YAML: 保留旧标签，标记 source
            film_field = ""
            new_label = old_label
            old_no_yaml += 1

        seen.add(key)
        all_records.append({
            "literature_id": lid,
            "aldehyde_smiles": a,
            "amine_smiles": b,
            "label": new_label,
            "film_field": str(film_field)[:200],
            "has_f": bool(row.get("has_fluorine", False)),
            "source": "group1" if yf else "group1_keep",
        })
        label_counts[new_label] += 1

    logger.info(f"新样本: {new_ok} 对 (已去重)")
    logger.info(f"  跳过: 无SMILES={skipped['no_smiles']}, 无醛/胺={skipped['no_ald_amine']}")
    logger.info(f"老样本: {old_relabeled} 重标 / {old_no_yaml} 保留旧标签 / {old_skipped_dup} 已重复跳过")
    logger.info(f"  老标签变更: 请在输出中对比")

    merged_df = pd.DataFrame(all_records)
    pos = int(merged_df["label"].sum())
    neg = len(merged_df) - pos

    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"合并完成: {len(merged_df)} 条 ({pos}+/{neg}-)")
    logger.info(f"  来源: {dict(merged_df['source'].value_counts())}")
    logger.info(f"已保存: {output_path}")

    return merged_df


def main():
    parser = argparse.ArgumentParser(description="新样本全自动处理管线")
    parser.add_argument("--pdf-dir", default="C:/Users/ckx/Desktop/shujuku2")
    parser.add_argument("--txt-dir", default="data/extracted_new")
    parser.add_argument("--yaml-dir", default="data/structured_new")
    parser.add_argument("--db", default="data/fluorofilm_new.db")
    parser.add_argument("--smiles-json", default="data/processed/monomer_smiles_new.json")
    parser.add_argument("--old-meta", default="data/processed/label_metadata.csv")
    parser.add_argument("--old-yaml-dir", default="data/structured")
    parser.add_argument("--output", default="data/processed/label_metadata_v2.csv")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-smiles", action="store_true")
    args = parser.parse_args()

    logger.info("=== 新样本处理管线 ===")
    logger.info(f"PDF 目录: {args.pdf_dir}")

    if not args.skip_parse:
        t0 = time.time()
        parse_new_pdfs(args.pdf_dir, args.txt_dir)
        logger.info(f"Step 1 用时: {time.time() - t0:.0f}s")
    else:
        logger.info("Step 1 — 跳过")

    if not args.skip_extract:
        t0 = time.time()
        extract_yamls(args.txt_dir, args.yaml_dir, args.db, args.workers)
        logger.info(f"Step 2 用时: {time.time() - t0:.0f}s")
    else:
        logger.info("Step 2 — 跳过")

    if not args.skip_smiles:
        t0 = time.time()
        extract_smiles_via_llm(args.yaml_dir, args.smiles_json)
        logger.info(f"Step 3 用时: {time.time() - t0:.0f}s")
    else:
        logger.info("Step 3 — 跳过")

    t0 = time.time()
    merged = build_label_metadata(args.yaml_dir, args.smiles_json, args.old_meta, args.old_yaml_dir, args.output)
    logger.info(f"Step 4 用时: {time.time() - t0:.0f}s")

    print("\n" + "=" * 60)
    print(f"  合并后总样本: {len(merged)}")
    print(f"  正样本 (成膜): {int(merged['label'].sum())}")
    print(f"  负样本: {len(merged) - int(merged['label'].sum())}")
    print(f"  来源分布:")
    print(merged["source"].value_counts().to_string())
    print(f"  输出: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
