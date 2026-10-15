"""长对话上下文策略对比：固定保留最近 5 轮 vs 按 token 预算压缩 + 状态固定注入。

    python eval/run_context_eval.py --turns 10 20 --budget 2000

对话由真实 harness 跑出来，每轮额外插入一条“工具结果”消息（本轮涉及商品的详情 JSON），
模拟一个把工具结果放进上下文的 LLM Agent 实际会看到的历史。

指标：
- 约束保持率：第 1 轮说的约束（预算 500、不要小米）在最后一轮发给模型的窗口里还在不在；
- 平均 / 最大窗口 token；
- 超预算轮数：窗口超过预算、必须截断或报错的轮数。

注意：这是机制验证，数据是合成的，只能说明“为什么要这么做”，不能当作效果数字写进简历。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from shopping_agent.archive import ArchiveStore, Caller  # noqa: E402
from shopping_agent.catalog import load_catalog  # noqa: E402
from shopping_agent.config import load_settings  # noqa: E402
from shopping_agent.context import compact, count_tokens, turn_starts  # noqa: E402
from shopping_agent.harness import build_agent, pinned_state  # noqa: E402

SCRIPT = ["想买个500以内的蓝牙耳机，不要小米", "第一个怎么样", "第二个和第三个对比一下", "要降噪的", "第二个怎么样"]


def dialogue(n_turns: int, settings, catalog):
    """返回 (逐轮的完整消息历史, 最后一轮的 state)。"""
    agent = build_agent(catalog, settings, InMemoryStore(), InMemorySaver())
    msgs, out = [], None
    for i in range(n_turns):
        q = SCRIPT[i % len(SCRIPT)] if i < len(SCRIPT) else SCRIPT[1 + i % (len(SCRIPT) - 1)]
        out = agent.invoke({"user_id": "u", "query": q}, {"configurable": {"thread_id": "t"}})
        ids = out.get("focus") or out.get("shown") or []
        detail = [catalog[p].model_dump(mode="json") for p in ids]
        msgs += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "function": {"name": "get_product_detail"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(detail, ensure_ascii=False)},
            {"role": "assistant", "content": out["response"]},
        ]
        yield list(msgs), out


def has_constraints(window_text: str) -> bool:
    return "500" in window_text and "小米" in window_text


def run(n_turns: int, budget: int) -> dict:
    settings = load_settings()
    catalog = load_catalog(settings.catalog_path)
    archive = ArchiveStore(InMemoryStore())
    caller = Caller("u", "t")

    fixed_tokens, budget_tokens = [], []
    fixed_over = budget_over = 0
    compacted: list[dict] = []
    fixed_ok = budget_ok = False
    for msgs, state in dialogue(n_turns, settings, catalog):
        # A：固定保留最近 5 轮原文，不注入结构化状态
        starts = turn_starts(msgs)
        window_a = msgs[starts[-5]:] if len(starts) > 5 else msgs
        ta = count_tokens(window_a)
        fixed_tokens.append(ta)
        fixed_over += ta > budget
        fixed_ok = has_constraints(json.dumps(window_a, ensure_ascii=False))

        # B：按预算压缩（增量：在上一轮压缩结果上追加本轮新消息）+ 状态固定注入
        new_part = msgs[len(msgs) - 4 :]
        hist = (compacted + new_part) if compacted else msgs
        pinned = pinned_state(state)
        res = compact(hist, budget=budget, pinned_tokens=count_tokens(pinned), archive=archive, caller=caller)
        compacted = res.messages
        window_b = pinned + compacted
        tb = count_tokens(window_b)
        budget_tokens.append(tb)
        budget_over += tb > budget
        budget_ok = has_constraints(json.dumps(window_b, ensure_ascii=False))

    return {
        "turns": n_turns,
        "fixed_5_turns": {
            "最后一轮约束仍在窗口": fixed_ok,
            "平均窗口 tokens": round(mean(fixed_tokens)),
            "最大窗口 tokens": max(fixed_tokens),
            "超预算轮数": fixed_over,
        },
        "budget_compaction": {
            "最后一轮约束仍在窗口": budget_ok,
            "平均窗口 tokens": round(mean(budget_tokens)),
            "最大窗口 tokens": max(budget_tokens),
            "超预算轮数": budget_over,
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--budget", type=int, default=2000)
    args = ap.parse_args()
    results = [run(n, args.budget) for n in args.turns]
    print(json.dumps({"budget": args.budget, "note": "合成数据，仅用于机制验证", "results": results}, ensure_ascii=False, indent=2))
