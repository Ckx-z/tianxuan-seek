import time
from typing import Optional

import requests
import urllib3
import yaml

# 关闭 HTTPS 证书验证警告（GROBID 免费端点证书可能无效）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src.utils.logger import setup_logger

logger = setup_logger("grobid")


class GrobidClient:
    """GROBID 学术论文结构化解析客户端"""

    def __init__(self, url: str, timeout: int = 120, retry: int = 3,
                 sleep_between: int = 5):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retry = retry
        self.sleep_between = sleep_between

    def process_fulltext(self, pdf_path: str) -> Optional[str]:
        """上传 PDF 到 GROBID，返回 TEI XML 全文"""
        endpoint = f"{self.url}/api/processFulltextDocument"
        for attempt in range(1, self.retry + 1):
            try:
                with open(pdf_path, "rb") as f:
                    response = requests.post(
                        endpoint,
                        files={"input": f},
                        headers={"Accept": "application/xml"},
                        timeout=self.timeout,
                        verify=False,
                    )
                if response.status_code == 200:
                    logger.info(f"GROBID OK: {pdf_path}")
                    return response.text
                else:
                    logger.warning(
                        f"GROBID HTTP {response.status_code} for {pdf_path}"
                        f" (attempt {attempt}/{self.retry})"
                    )
            except requests.RequestException as e:
                logger.warning(
                    f"GROBID error for {pdf_path}: {e}"
                    f" (attempt {attempt}/{self.retry})"
                )
            if attempt < self.retry:
                time.sleep(self.sleep_between)

        logger.error(f"GROBID failed after {self.retry} retries: {pdf_path}")
        return None

    def process_metadata(self, pdf_path: str) -> Optional[dict]:
        """仅提取标题/作者/摘要等元数据（轻量模式，备选）"""
        endpoint = f"{self.url}/api/processHeaderDocument"
        for attempt in range(1, self.retry + 1):
            try:
                with open(pdf_path, "rb") as f:
                    response = requests.post(
                        endpoint,
                        files={"input": f},
                        headers={"Accept": "application/xml"},
                        timeout=self.timeout,
                        verify=False,
                    )
                if response.status_code == 200:
                    return {"raw_xml": response.text}
            except requests.RequestException:
                pass
            if attempt < self.retry:
                time.sleep(self.sleep_between)
        return None


def load_grobid_config(config_path: str = "config/grobid.yaml") -> dict:
    """读取 GROBID 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_grobid_client(config: dict) -> GrobidClient:
    """根据配置字典创建 GrobidClient 实例"""
    return GrobidClient(
        url=config["grobid"]["url"],
        timeout=config["grobid"]["timeout"],
        retry=config["grobid"]["retry"],
        sleep_between=config["grobid"]["sleep_between"],
    )
