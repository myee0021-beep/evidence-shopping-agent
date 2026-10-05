"""Agent 的结构化工作状态。

LangGraph 会在每个节点执行后把 State 存进 checkpointer（本项目用 Redis），
同一个 thread_id 的下一轮对话会接着这份状态继续，这就是“会话记忆”。

字段的合并方式由 reducer 决定：
- messages / trace 用 operator.add 追加；
- 其余字段默认覆盖。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # 输入
    user_id: str
    query: str

    # 会话层
    turn: int
    messages: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]

    # 当前任务的工作状态
    intent: str  # search | compare | detail | clarify
    constraints: dict[str, Any]  # 用户的结构化约束，跨轮继承
    relaxed: list[str]  # 本轮重规划时放宽的软约束
    refs: list[int]  # 指代：“第二个” -> [2]
    focus: list[str]  # 指代解析后的商品 ID
    shown: list[str]  # 上一次展示给用户的商品，指代消解的依据
    candidates: list[str]  # 检索得到的候选池
    evidence: dict[str, dict]  # 商品 ID -> 证据记录
    comparison: dict[str, Any]
    confidence: float
    route: str  # recommend | caveat | compare | detail | clarify | replan | fallback
    replan_count: int
    notes: list[str]  # 需要告诉用户的说明（沿用偏好、放宽条件等）
    response: str
