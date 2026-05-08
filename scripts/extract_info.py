import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.utils.logger import setup_logger
from src.utils.db import init_db, insert_record, get_record_count
from src.extraction.minimax_client import (
    load_extraction_config, create_minimax_client,
)
from src.extraction.llm_extractor import LLMExtractor

logger = setup_logger("extract_info")

# 共享状态的线程锁
_progress_lock = threading.Lock()
_failed_lock = threading.Lock()


def _process_one(
    txt_path: Path,
    stem: str,
    yaml_dir: Path,
    db_path: str,
    extractor: LLMExtractor,
    index: int,
    total: int,
) -> Tuple[bool, str]:
    """
    处理单篇文献的完整流程：API 提取 → YAML 写入 → SQLite 写入
    每个线程独立调用，拥有自己的 DB 连接
    """
    yaml_out = yaml_dir / f"{stem}.yaml"

    # Step 1: LLM 提取
    data = extractor.extract(str(txt_path))
    if data is None:
        return (False, txt_path.name)

    # Step 2: 写入 YAML
    data["literature_id"] = stem
    yaml_out.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False,
                  sort_keys=False),
        encoding="utf-8",
    )

    # Step 3: 写入 SQLite（每个线程独立连接，WAL 模式自动处理并发）
    conn = init_db(db_path)
    try:
        insert_record(conn, stem, data)
    finally:
        conn.close()

    with _progress_lock:
        logger.info(f"[{index}/{total}] OK: {txt_path.name}")

    return (True, "")


def run(input_dir: str, output_dir: str, db_path: str, limit: int,
        skip_existing: bool, workers: int):
    """批量 LLM 提取（多线程并行）"""
    txt_dir = Path(input_dir)
    yaml_dir = Path(output_dir)
    yaml_dir.mkdir(parents=True, exist_ok=True)
    failed_log = yaml_dir / "_failed_extract.log"

    # 收集所有 .full.txt
    txt_files = sorted(txt_dir.rglob("*.full.txt"))
    if not txt_files:
        logger.error(f"No .full.txt files found under: {txt_dir}")
        sys.exit(1)

    total = min(len(txt_files), limit) if limit else len(txt_files)
    txt_files = txt_files[:total]

    # 断点续传：提前过滤已处理文件
    task_list: List[Tuple[Path, str]] = []
    count_skip = 0
    for p in txt_files:
        stem = p.stem.replace(".full", "")
        yaml_out = yaml_dir / f"{stem}.yaml"
        if skip_existing and yaml_out.exists():
            logger.info(f"  跳过已处理: {yaml_out.name}")
            count_skip += 1
        else:
            task_list.append((p, stem))

    logger.info(
        f"待处理: {len(task_list)} 篇, 已跳过: {count_skip}, "
        f"并发数: {workers}"
    )

    if not task_list:
        logger.info("没有需要处理的文件")
        return

    # 加载配置 + 初始化共享组件
    config = load_extraction_config()
    minimax_client = create_minimax_client(config)
    extractor = LLMExtractor(
        client=minimax_client,
        fields=config["fields"],
        max_input_chars=config["minimax"]["max_input_chars"],
    )

    # 多线程并行处理
    count_ok, count_fail = 0, 0
    total_tasks = len(task_list)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        futures = {}
        for idx, (txt_path, stem) in enumerate(task_list, start=1):
            future = executor.submit(
                _process_one,
                txt_path, stem, yaml_dir, db_path, extractor,
                idx, total_tasks,
            )
            futures[future] = txt_path

        # 收集结果
        for future in as_completed(futures):
            success, fname = future.result()
            if success:
                count_ok += 1
            else:
                count_fail += 1
                with _failed_lock:
                    with open(failed_log, "a", encoding="utf-8") as f:
                        f.write(f"{fname}\n")

            # 每 10 篇输出进度统计
            done = count_ok + count_fail
            if done % 10 == 0 or count_ok % 10 == 0:
                with _progress_lock:
                    logger.info(
                        f"进度: {count_ok} OK, {count_fail} failed, "
                        f"{done}/{total_tasks} done"
                    )

    # 数据库统计
    conn = init_db(db_path)
    record_count = get_record_count(conn)
    conn.close()

    logger.info(
        f"Done: {count_ok} OK, {count_fail} failed, "
        f"{count_skip} skipped (of {total})"
    )
    logger.info(f"Database records: {record_count}")
    if count_fail:
        logger.info(f"Failures logged to: {failed_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM structured extraction for FluoroFilm"
    )
    parser.add_argument("--input", default="data/extracted",
                        help="Directory containing .full.txt files")
    parser.add_argument("--output", default="data/structured",
                        help="Output directory for .yaml files")
    parser.add_argument("--db", default="data/fluorofilm.db",
                        help="SQLite database path")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of files (0=all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip already-processed files (resume)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    run(args.input, args.output, args.db, args.limit,
        args.skip_existing, args.workers)
