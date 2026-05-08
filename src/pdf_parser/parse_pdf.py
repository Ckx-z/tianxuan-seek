from typing import Optional

import fitz

from src.utils.logger import setup_logger

logger = setup_logger("pymupdf")


def extract_text(pdf_path: str) -> Optional[str]:
    """用 PyMuPDF 逐页提取 PDF 全文，以空行分隔各页"""
    try:
        doc = fitz.open(pdf_path)
        pages = [doc[i].get_text() for i in range(len(doc))]
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        logger.error(f"PyMuPDF failed to extract text: {pdf_path} - {e}")
        return None


def extract_metadata(pdf_path: str) -> dict:
    """提取 PDF 内嵌元数据（标题、作者等，作为备选信息源）"""
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata
        doc.close()
        return dict(meta) if meta else {}
    except Exception as e:
        logger.warning(f"PyMuPDF failed to read metadata: {pdf_path} - {e}")
        return {}
