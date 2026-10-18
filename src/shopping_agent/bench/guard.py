"""循环护栏：一条任务里的工具调用预算 + 重复调用去重。

ShoppingBench 上最常见的“空转”有两种：
1. 重规划换了策略，但新的检索和之前某次一模一样（比如 query 本来就不超过 4 个词，
   “缩短 query”之后还是原 query；同店定向检索又搜了一遍已经搜过的店）；
2. 多件商品 × 多个候选店铺 × 翻页，调用次数随件数相乘，没有上限。

CallGuard 在调用沙箱之前拦一道：
- 同一条任务里，参数完全相同的只读调用直接复用第一次的结果，不再发请求，也不写新的 rollout step；
- 真实调用次数超过 max_calls 就不再调用，返回空结果，由编排层用已有候选收尾。

多个子 agent 并行时共用同一个 guard，所以加了锁。
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardStats:
    calls: int = 0  # 真正发到沙箱的调用
    cached: int = 0  # 被去重拦下的重复调用
    blocked: int = 0  # 预算用完后被拦下的调用


@dataclass
class CallGuard:
    max_calls: int | None = 40
    dedup: bool = True
    stats: GuardStats = field(default_factory=GuardStats)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @staticmethod
    def key(tool: str, params: dict[str, Any]) -> str:
        return tool + "\n" + json.dumps(params, sort_keys=True, ensure_ascii=False)

    def before(self, tool: str, params: dict[str, Any]) -> tuple[str, Any]:
        """返回 ("run", None) / ("cached", 结果) / ("blocked", None)。"run" 会预占一次预算。"""
        k = self.key(tool, params)
        with self._lock:
            if self.dedup and k in self._cache:
                self.stats.cached += 1
                return "cached", self._cache[k]
            if self.max_calls is not None and self.stats.calls >= self.max_calls:
                self.stats.blocked += 1
                return "blocked", None
            self.stats.calls += 1
            return "run", None

    def after(self, tool: str, params: dict[str, Any], results: Any, ok: bool) -> None:
        """只缓存成功的结果：失败的调用下次还应该再试。"""
        if not (self.dedup and ok):
            return
        with self._lock:
            self._cache.setdefault(self.key(tool, params), results)

    @property
    def exhausted(self) -> bool:
        return self.max_calls is not None and self.stats.calls >= self.max_calls
