"""运行配置：从环境变量读取，方便在本地、评测和 Redis 之间切换。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    catalog_path: Path = PROJECT_ROOT / "data" / "products.json"
    # 模拟数据的快照日期。证据时效按它计算，保证评测结果可复现。
    snapshot_date: date = date(2026, 10, 1)
    stale_days: int = 30
    top_k: int = 5  # 检索阶段的候选池大小
    show_k: int = 3  # 最终展示给用户的商品数
    max_replans: int = 2
    min_confidence: float = 0.75  # 低于该值走“带提示推荐”
    redis_url: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "qwen-plus"


def load_settings() -> Settings:
    return Settings(
        redis_url=os.getenv("REDIS_URL") or None,
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_model=os.getenv("LLM_MODEL", "qwen-plus"),
    )
