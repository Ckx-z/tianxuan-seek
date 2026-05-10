import os
import time
from typing import Optional

import yaml
from openai import OpenAI

from src.utils.logger import setup_logger

logger = setup_logger("minimax")


class MiniMaxClient:
    """用 OpenAI 兼容 SDK 对接 MiniMax API"""

    def __init__(self, api_key: str, base_url: str, model: str,
                 temperature: float = 0.3, max_tokens: int = 4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 初始化 OpenAI 客户端，指向 MiniMax 端点
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"MiniMax 客户端初始化: {base_url}, model={model}")

    def chat(self, system_prompt: str, user_prompt: str,
             retry: int = 3, sleep_between: int = 3) -> Optional[str]:
        """发送对话请求，含重试逻辑"""
        for attempt in range(1, retry + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(
                    f"MiniMax API 调用失败 (attempt {attempt}/{retry}): {e}"
                )
            if attempt < retry:
                time.sleep(sleep_between)
        logger.error(f"MiniMax API 重试 {retry} 次后仍失败")
        return None


def load_extraction_config(config_path: str = "config/extraction.yaml") -> dict:
    """读取提取配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_minimax_client(config: dict) -> MiniMaxClient:
    """根据配置创建 MiniMaxClient 实例。

    API key 优先级：环境变量 MINIMAX_API_KEY > config 中的 api_key 字段。
    """
    mc = config["minimax"]
    api_key = mc.get("api_key", "") or os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError(
            "环境变量 MINIMAX_API_KEY 未设置，且配置中无 api_key 字段"
        )
    return MiniMaxClient(
        api_key=api_key,
        base_url=mc["api_base"],
        model=mc["model"],
        temperature=mc.get("temperature", 0.3),
        max_tokens=mc.get("max_tokens", 4096),
    )


def create_fallback_client(config: dict) -> Optional[MiniMaxClient]:
    """根据配置创建备用 LLM 客户端（MiMo 等）。

    API key 优先级：环境变量 FALLBACK_API_KEY > config 中的 api_key 字段。
    若配置中无 fallback 段，返回 None。
    """
    fb = config.get("fallback")
    if not fb:
        return None
    api_key = (os.getenv("FALLBACK_API_KEY")
               or fb.get("api_key", "")
               or os.getenv("MINIMAX_API_KEY", ""))
    if not api_key:
        logger.warning("备用 LLM 未配置 API key，跳过")
        return None
    return MiniMaxClient(
        api_key=api_key,
        base_url=fb["api_base"],
        model=fb["model"],
        temperature=fb.get("temperature", 0.3),
        max_tokens=fb.get("max_tokens", 4096),
    )
