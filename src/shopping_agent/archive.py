"""L0 原文归档：被压缩掉的对话、超长工具结果都先完整存一份，上下文里只留摘要和 ref_id。

上下文压缩（context.py）和长期记忆（agent_memory/）共用这一层：
- 压缩时被切掉的历史 -> 归档，摘要里带 ref_id，需要时 recall_context(ref_id) 取回；
- 每轮用户原话 -> 归档，L1 事实的 source_ref 指向这里，任何一条记忆都能追溯到原话。

存在 LangGraph Store 里，key 空间是 ("archive", tenant, user_id)。读取时做权限校验：
- 不同租户、不同用户的归档互相不可见；
- visibility="session" 的记录（压缩归档、工具结果）只有同一个会话能读；
- visibility="user" 的记录（用户原话）同一用户跨会话可读，用于长期记忆溯源。
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.store.base import BaseStore

Visibility = Literal["session", "user"]

RECALL_MAX_CHARS = 4000


class ArchiveAccessError(PermissionError):
    pass


@dataclass(frozen=True)
class Caller:
    """谁在读：权限校验用。"""

    user_id: str
    thread_id: str
    tenant: str = "default"


class ArchiveStore:
    def __init__(self, store: BaseStore):
        self.store = store

    @staticmethod
    def _ns(tenant: str, user_id: str) -> tuple[str, ...]:
        return ("archive", tenant, user_id)

    def put(
        self,
        caller: Caller,
        kind: str,
        text: str,
        visibility: Visibility = "session",
        meta: dict[str, Any] | None = None,
    ) -> str:
        ref_id = f"{kind}_{uuid.uuid4().hex[:10]}"
        self.store.put(
            self._ns(caller.tenant, caller.user_id),
            ref_id,
            {
                "ref_id": ref_id,
                "kind": kind,
                "thread_id": caller.thread_id,
                "visibility": visibility,
                "text": text,
                "meta": meta or {},
                "ts": time.time(),
            },
        )
        return ref_id

    def get(self, caller: Caller, ref_id: str) -> dict[str, Any]:
        item = self.store.get(self._ns(caller.tenant, caller.user_id), ref_id)
        # 读不到和无权限返回同一种错误，不泄露“别人的 ref_id 是否存在”
        if item is None:
            raise ArchiveAccessError(f"ref {ref_id} not found or not accessible")
        rec = dict(item.value)
        if rec["visibility"] == "session" and rec["thread_id"] != caller.thread_id:
            raise ArchiveAccessError(f"ref {ref_id} not found or not accessible")
        return rec


def recall_context(
    archive: ArchiveStore,
    caller: Caller,
    ref_id: str,
    mode: Literal["grep", "tail", "range"] = "tail",
    pattern: str = "",
    start: int = 0,
    length: int = RECALL_MAX_CHARS,
) -> dict[str, Any]:
    """按需取回归档原文。三种读法，每次最多返回 4000 字符，避免一次把上下文又撑爆。

    - grep：返回包含 pattern 的行（带行号），适合“刚才那款多少钱”；
    - tail：返回末尾一段；
    - range：从 start 开始读 length 个字符，配合 next_start 分段读完。
    """
    try:
        rec = archive.get(caller, ref_id)
    except ArchiveAccessError as e:
        return {"ok": False, "error": str(e)}
    text: str = rec["text"]
    length = max(1, min(length, RECALL_MAX_CHARS))
    if mode == "grep":
        if not pattern:
            return {"ok": False, "error": "grep 需要 pattern"}
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
        hits = [f"{i}: {line}" for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]
        body = "\n".join(hits)
        return {"ok": True, "ref_id": ref_id, "hits": len(hits), "text": body[:length], "truncated": len(body) > length}
    if mode == "tail":
        return {"ok": True, "ref_id": ref_id, "text": text[-length:], "truncated": len(text) > length}
    start = max(0, start)
    end = start + length
    return {
        "ok": True,
        "ref_id": ref_id,
        "text": text[start:end],
        "next_start": end if end < len(text) else None,
        "total_chars": len(text),
    }
