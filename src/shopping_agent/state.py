"""Agent 的结构化工作状态。

LangGraph 会在每个节点执行后把 State 存进 checkpointer（本项目用 Redis），
同一个 thread_id 的下一轮对话会接着这份状态继续，这就是“会话记忆”。

字段的合并方式由 reducer 决定：
- trace 用 operator.add 追加；
- messages 默认追加，上下文压缩时用 ReplaceMessages 整体替换成“摘要 + 最近几轮”；
- 其余字段默认覆盖。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ReplaceMessages(list):
    """节点返回它而不是普通 list 时，messages 整体替换（上下文压缩用）。"""


def add_or_replace(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    if isinstance(right, ReplaceMessages):
        return list(right)
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    # 输入
    user_id: str
    query: str

    # 会话层
    turn: int
    thread_id: str  # 由 harness 从 config 写入，归档权限校验用
    messages: Annotated[list[dict], add_or_replace]
    trace: Annotated[list[dict], operator.add]

    # 当前任务的工作状态
    intent: str  # search | compare | detail | clarify
    constraints: dict[str, Any]  # 用户的结构化约束，跨轮继承
    constraint_diff: dict[str, Any]  # 本轮用户说的话带来的约束变化 {added, changed, removed}
    relaxed: list[str]  # 本轮重规划时放宽的软约束
    refs: list[int]  # 指代：“第二个” -> [2]
    focus: list[str]  # 指代解析后的商品 ID
    shown: list[str]  # 上一次展示给用户的商品，指代消解的依据
    shown_snapshot: dict[str, Any]  # 展示时的快照：版本号 + 序号 -> (ID, 标题, 当时的价格)
    candidates: list[str]  # 检索得到的候选池
    evidence: dict[str, dict]  # 商品 ID -> 证据记录
    comparison: dict[str, Any]
    review_evidence: dict[str, dict]  # 商品 ID -> 评论证据（各方面的结论和引用片段）
    confidence: float
    route: str  # recommend | caveat | compare | detail | clarify | replan | fallback
    replan_count: int
    notes: list[str]  # 需要告诉用户的说明（沿用偏好、放宽条件等）
    response: str
    context: dict[str, Any]  # 最近一次上下文压缩的统计
