"""记忆入口。

1. 会话记忆（短期）：LangGraph checkpointer，按 thread_id 保存整个 AgentState，
   生产环境用 Redis（RedisSaver），本地和评测用内存版。会话内的上下文压缩见 context.py。
2. 长期记忆：LangGraph Store，按 user_id 跨会话保存。内部分 L0–L3 四层（见 agent_memory/），
   最上层的 L3 画像写在 PROFILE_NS 下，新会话开始时读出来预填约束。

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
