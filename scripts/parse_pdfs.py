import argparse
import os
import sys
import time
from pathlib import Path

# 将项目根目录加入 path，支持直接 python scripts/parse_pdfs.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logger
from src.pdf_parser.grobid_client import load_grobid_config, create_grobid_client
from src.pdf_parser.parse_pdf import extract_text

logger = setup_logger("parse_pdfs")


def run(input_dir: str, output_dir: str, limit: int, skip_grobid: bool):
    """批量解析 PDF：遍历目录，依次 GROBID(可选) + PyMuPDF，输出 XML/TXT"""
    pdf_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_log = out_dir / "_failed.log"

    # 收集所有 PDF 并按名称排序
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found under: {pdf_dir}")
        sys.exit(1)

    total = min(len(pdf_files), limit) if limit else len(pdf_files)
    count_ok, count_fail = 0, 0

    # GROBID 客户端初始化（可选）
    grobid_client = None
    if not skip_grobid:
        config = load_grobid_config()
        grobid_client = create_grobid_client(config)

    for i, pdf_path in enumerate(pdf_files[:total], start=1):
        stem = pdf_path.stem
        logger.info(f"[{i}/{total}] Processing: {pdf_path.name}")

        ok = True

        # Step 1: GROBID 解析（可选），产出 TEI XML
        if grobid_client:
            xml_text = grobid_client.process_fulltext(str(pdf_path))
            if xml_text:
                xml_out = out_dir / f"{stem}.grobid.xml"
                xml_out.write_text(xml_text, encoding="utf-8")
            else:
                ok = False

        # Step 2: PyMuPDF 全文提取，产出纯文本
        txt = extract_text(str(pdf_path))
        if txt:
            txt_out = out_dir / f"{stem}.full.txt"
            txt_out.write_text(txt, encoding="utf-8")
        else:
            ok = False

        # 统计成功/失败
        if ok:
            count_ok += 1
        else:
            count_fail += 1
            with open(failed_log, "a", encoding="utf-8") as f_log:
                f_log.write(f"{pdf_path.name}\n")

    logger.info(f"Done: {count_ok} OK, {count_fail} failed (of {total})")
    if count_fail:
        logger.info(f"Failures logged to: {failed_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch PDF parser for FluoroFilm")
    parser.add_argument("--input", default="data/pdfs", help="PDF input directory")
    parser.add_argument("--output", default="data/extracted", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs (0=all)")
    parser.add_argument("--skip-grobid", action="store_true", help="Skip GROBID, run PyMuPDF only")
    args = parser.parse_args()

    run(args.input, args.output, args.limit, args.skip_grobid)
