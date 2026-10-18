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
import re
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


def dialogue(n_turns: int, settings, catalog, payload: str = "detail"):
    """返回 (逐轮的完整消息历史, 最后一轮的 state)。

    payload 决定每轮插进上下文的工具结果：
    - detail：本轮关注 / 展示的 1–3 个商品详情（get_product_detail）；
    - search：同品类最多 10 个商品的一页检索结果，更接近 Products Finder 一次检索带回的量。
    """
    agent = build_agent(catalog, settings, InMemoryStore(), InMemorySaver())
    msgs, out = [], None
    for i in range(n_turns):
        q = SCRIPT[i % len(SCRIPT)] if i < len(SCRIPT) else SCRIPT[1 + i % (len(SCRIPT) - 1)]
        out = agent.invoke({"user_id": "u", "query": q}, {"configurable": {"thread_id": "t"}})
        ids = out.get("focus") or out.get("shown") or []
        if payload == "search":
            cat = out.get("constraints", {}).get("category")
            ids = [pid for pid, p in catalog.items() if p.category == cat][:10] or ids
        detail = [catalog[p].model_dump(mode="json") for p in ids]
        msgs += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "function": {"name": "get_product_detail"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(detail, ensure_ascii=False)},
            {"role": "assistant", "content": out["response"]},
        ]
        yield list(msgs), out


def visible_ids(window: list[dict], ids: set[str]) -> int:
    """窗口里还能同时看到 ID 和价格的商品数：问“刚才那个 E05 多少钱”时不用 recall 就能答。"""
    text = "\n".join(str(m.get("content") or "") for m in window)
    return sum(bool(re.search(rf'(id=|"id": "){pid}\b[^\n]*?price', text)) for pid in ids)


def has_constraints(window_text: str) -> bool:
    return "500" in window_text and "小米" in window_text


def run(n_turns: int, budget: int, payload: str = "detail") -> dict:
    settings = load_settings()
    catalog = load_catalog(settings.catalog_path)
    archive = ArchiveStore(InMemoryStore())
    caller = Caller("u", "t")

    fixed_tokens: list[int] = []
    fixed_over = 0
    fixed_ok = False
    strat = {k: {"hist": [], "tokens": [], "over": 0, "ok": False, "summarized": 0, "raw_turns": 0, "visible": 0}
             for k in ("no_digest", "digest")}
    seen_ids: set[str] = set()
    for msgs, state in dialogue(n_turns, settings, catalog, payload):
        # A：固定保留最近 5 轮原文，不注入结构化状态
        starts = turn_starts(msgs)
        window_a = msgs[starts[-5]:] if len(starts) > 5 else msgs
        ta = count_tokens(window_a)
        fixed_tokens.append(ta)
        fixed_over += ta > budget
        fixed_ok = has_constraints(json.dumps(window_a, ensure_ascii=False))

        # B / C：按预算压缩（增量：在上一轮压缩结果上追加本轮新消息）+ 状态固定注入
        #   B 是旧行为：超长工具结果只留前 300 字符预览；
        #   C 是商品摘要：超长结果和更早轮次的商品结果都留“ID + 价格 + 库存 + 关键属性”
        new_part = msgs[len(msgs) - 4 :]
        seen_ids |= set(re.findall(r'"id": "(\w+)"', new_part[2]["content"]))
        pinned = pinned_state(state)
        for key, digest in (("no_digest", False), ("digest", True)):
            st = strat[key]
            hist = (st["hist"] + new_part) if st["hist"] else msgs
            res = compact(hist, budget=budget, pinned_tokens=count_tokens(pinned), archive=archive, caller=caller,
                          product_digest=digest)
            st["hist"] = res.messages
            st["summarized"] += res.turns_summarized
            window = pinned + res.messages
            t = count_tokens(window)
            st["tokens"].append(t)
            st["over"] += t > budget
            st["ok"] = has_constraints(json.dumps(window, ensure_ascii=False))
            st["raw_turns"] = len(turn_starts(res.messages))
            st["visible"] = visible_ids(window, seen_ids)

    return {
        "turns": n_turns,
        "fixed_5_turns": {
            "最后一轮约束仍在窗口": fixed_ok,
            "平均窗口 tokens": round(mean(fixed_tokens)),
            "最大窗口 tokens": max(fixed_tokens),
            "超预算轮数": fixed_over,
        },
        **{name: {
            "最后一轮约束仍在窗口": strat[k]["ok"],
            "平均窗口 tokens": round(mean(strat[k]["tokens"])),
            "最大窗口 tokens": max(strat[k]["tokens"]),
            "超预算轮数": strat[k]["over"],
            "被摘要掉的轮数（累计）": strat[k]["summarized"],
            "最后一轮窗口里的原文轮数": strat[k]["raw_turns"],
            "工具返回过的商品在最后窗口里还能看到 ID+价格": f"{strat[k]['visible']}/{len(seen_ids)}",
        } for name, k in (("budget_compaction(preview)", "no_digest"), ("budget_compaction(product_digest)", "digest"))},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--payload", choices=["detail", "search"], default="detail",
                    help="每轮插入的工具结果：detail=1–3 个商品详情，search=一页最多 10 个商品")
    args = ap.parse_args()
    results = [run(n, args.budget, args.payload) for n in args.turns]
    print(json.dumps({"budget": args.budget, "payload": args.payload, "note": "合成数据，仅用于机制验证",
                      "results": results}, ensure_ascii=False, indent=2))
