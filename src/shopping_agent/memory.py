"""双层记忆。

1. 会话记忆（短期）：LangGraph checkpointer，按 thread_id 保存整个 AgentState，
   生产环境用 Redis（RedisSaver），本地和评测用内存版。
2. 用户画像记忆（长期）：LangGraph Store，按 user_id 跨会话保存偏好，
   比如“不喜欢的品牌”“各品类的预算”。新会话开始时读出来预填约束。

两层的区别：checkpointer 管“这次对话进行到哪了”，Store 管“这个用户是什么样的人”。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore

PROFILE_NS = ("profiles",)


def load_profile(store: BaseStore, user_id: str) -> dict[str, Any]:
    item = store.get(PROFILE_NS, user_id)
    return dict(item.value) if item else {}


def update_profile(profile: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    """把本轮确认的约束沉淀进画像，按品类分开记：不喜欢小米耳机，不代表不看小米手机。"""
    cat = constraints.get("category")
    p = {
        "brand_dislikes": {k: list(v) for k, v in profile.get("brand_dislikes", {}).items()},
        "budgets": dict(profile.get("budgets", {})),
        "liked_features": {k: list(v) for k, v in profile.get("liked_features", {}).items()},
    }
    if not cat:
        return p
    dislikes = set(p["brand_dislikes"].get(cat, [])) | set(constraints.get("brands_exclude", []))
    # 用户明确指定过的品牌，不再视为“不喜欢”
    dislikes -= set(constraints.get("brands_include", []))
    p["brand_dislikes"][cat] = sorted(dislikes)
    if constraints.get("budget_max") is not None:
        p["budgets"][cat] = constraints["budget_max"]
    feats = set(p["liked_features"].get(cat, [])) | set(constraints.get("features", []))
    p["liked_features"][cat] = sorted(feats)
    return p


def save_profile(store: BaseStore, user_id: str, profile: dict[str, Any]) -> None:
    store.put(PROFILE_NS, user_id, profile)


@contextmanager
def make_checkpointer(redis_url: str | None) -> Iterator[Any]:
    """有 REDIS_URL 就用 Redis 持久化会话状态，否则用内存。"""
    if not redis_url:
        yield InMemorySaver()
        return
    from langgraph.checkpoint.redis import RedisSaver  # 可选依赖

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        yield saver
