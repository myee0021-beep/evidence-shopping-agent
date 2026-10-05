"""Trace 回放与错误归因。

每个节点、每次 skill 调用都会在 AgentState.trace 里留下一条记录。
- replay()：按轮次把决策轨迹打印出来，复盘 bad case；
- attribute()：根据轨迹判断一轮对话错在哪个阶段（理解 / 检索 / 证据 / 决策）。

命令行用法：
    python -m shopping_agent.trace eval/results/cases_hard/failed_traces.jsonl
"""

from __future__ import annotations

import json
import sys
from typing import Any


def turn_events(trace: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    return [e for e in trace if e.get("turn") == turn]


def replay(trace: list[dict[str, Any]], turn: int | None = None) -> str:
    lines = []
    for e in trace:
        if turn is not None and e.get("turn") != turn:
            continue
        if e["type"] == "skill":
            mark = "ok" if e["ok"] else f"FAIL {e['error']}"
            lines.append(f"  [t{e['turn']}]   └─ skill {e['skill']}({_short(e['args'])}) -> {mark} {e['latency_ms']}ms")
        else:
            extra = {k: v for k, v in e.items() if k not in ("turn", "node", "type", "status", "latency_ms")}
            status = "" if e["status"] == "ok" else f" ERROR {e.get('error')}"
            lines.append(f"  [t{e['turn']}] {e['node']:<16}{e['latency_ms']:>8}ms{status} {_short(extra)}")
    return "
".join(lines)


FAILURE_STAGE = (
    ("constraints.", "understanding"),
    ("memory.", "understanding"),
    ("recall=", "retrieval"),
    ("skills=", "tool_call"),
    ("relaxed=", "replan"),
    ("违反硬约束", "evidence"),
    ("不应被推荐", "evidence"),
    ("route=", "routing"),
)


def attribute(trace: list[dict[str, Any]], turn: int, failures: list[str] | None = None) -> str:
    """错误归因：先看评测失败项指向哪个阶段，再看轨迹里第一个“看起来不对”的阶段。"""
    for f in failures or []:
        for prefix, stage in FAILURE_STAGE:
            if f.startswith(prefix) or prefix in f:
                return stage
    events = turn_events(trace, turn)
    nodes = {e["node"]: e for e in events if e["type"] == "node"}
    if any(e.get("status") == "error" for e in nodes.values()):
        bad = next(n for n, e in nodes.items() if e.get("status") == "error")
        return f"exception@{bad}"
    if any(e["type"] == "skill" and not e["ok"] for e in events):
        return "tool_failure"
    u = nodes.get("understand", {})
    if u.get("intent") == "clarify":
        return "understanding"
    r = nodes.get("retrieve")
    if r is not None and not r.get("candidates"):
        return "retrieval"
    g = nodes.get("gather_evidence")
    if g is not None and not g.get("usable"):
        return "evidence"
    return "decision_or_generation"


def _short(obj: Any, limit: int = 160) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[: limit - 3] + "..."


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            print(f"== {rec['case_id']} turn {rec['turn']} | 归因：{rec['attribution']} | 失败项：{rec['failures']}")
            print(replay(rec["trace"], rec["turn"]))
            print()
