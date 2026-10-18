"""Agent Harness：受控执行链 + 决策编排。

执行链：
    用户查询 -> understand（意图识别 / 约束抽取 / 指代消解）
             -> retrieve（商品检索）
             -> gather_evidence（证据聚合）
             -> judge（置信度判断，决定响应路径）
             -> respond（响应生成）
             -> update_memory（L0 原话归档 + L1 事实 + 重算 L2 / L3 画像）
             -> compact_context（按 token 预算压缩对话历史）

judge 的出口：
    recommend / caveat / compare / detail / clarify / fallback -> respond
    replan -> 放宽一个软约束后回到 retrieve（最多 max_replans 次）

Harness 负责的事：
- 状态流转：节点只读写 AgentState，由 LangGraph 管理；
- Skills 调用：统一经过 skills.call_skill，参数校验、耗时、失败都可观测；
- 异常兜底：任何节点抛异常都会被 guarded() 接住，记录 trace 并走安全路径；
- 决策轨迹：每个节点写一条 trace（输入要点、输出要点、耗时），用于 bad case 回放和错误归因。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from .agent_memory import LayeredMemory
from .archive import ArchiveStore, Caller
from .catalog import Catalog
from .config import Settings
from .context import Summarizer, compact, count_tokens, rule_summarize
from .evidence import describe_issues, fact_check
from .memory import load_profile
from .privacy import GUARD_REPLY, guard_reason
from .reviews import citation_check, load_reviews, reason_lines, select_aspects, snippet_index
from .skills import SkillContext, call_skill
from .state import AgentState, ReplaceMessages
from .understanding import TurnParse, diff_constraints, merge_constraints, rule_parse

Parser = Callable[[str, dict[str, Any], list[str]], TurnParse | None]
# 回复润色（比如交给 LLM 改写措辞）：输入模板回复和状态，输出新回复。改写后必须通过事实校验
Rewriter = Callable[[str, AgentState], str]

# 消融开关：关掉某个模块，看它到底贡献了多少
ABLATIONS = {
    "state": "不做跨轮约束继承（每轮只看这一句话）",
    "evidence": "不做证据校验（相信列表页价格和库存）",
    "replan": "不重规划（检索为空直接兜底）",
    "memory": "不读写长期记忆",
    "reviews": "不检索评论证据（推荐理由只用商品字段）",
}

# 软约束的放宽顺序：先放宽属性，再放宽指定品牌。预算、品类、排除品牌属于硬约束，从不自动放宽。
RELAX_ORDER = ("features", "brands_include")
RELAX_LABEL = {"features": "属性要求", "brands_include": "品牌限定"}

# 节点出错时的安全输出
NODE_FALLBACK: dict[str, dict[str, Any]] = {
    "understand": {"intent": "clarify", "notes": ["没能理解这句话"]},
    "retrieve": {"candidates": []},
    "gather_evidence": {"evidence": {}},
    "judge": {"route": "fallback", "confidence": 0.0},
    "replan": {"route": "fallback"},
    "respond": {"response": "抱歉，系统暂时出了点问题，请稍后再试。", "messages": []},
    "update_memory": {},
    "compact_context": {},
}

SYSTEM_PROMPT = "你是一个基于证据做推荐的购物助手。只推荐证据层校验通过的商品，不编造价格和库存。"


def _effective(state: AgentState) -> dict[str, Any]:
    """本轮实际用于检索的约束 = 用户约束 - 本轮放宽的软约束。"""
    relaxed = set(state.get("relaxed", []))
    return {k: v for k, v in state.get("constraints", {}).items() if k not in relaxed}


def _next_relaxation(state: AgentState) -> str | None:
    c = state.get("constraints", {})
    relaxed = state.get("relaxed", [])
    for key in RELAX_ORDER:
        if key not in relaxed and c.get(key):
            return key
    return None


def _skill_event(turn: int, node: str, name: str, args: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn": turn,
        "node": node,
        "type": "skill",
        "skill": name,
        "args": args,
        "ok": res["ok"],
        "error": res.get("error"),
        "latency_ms": res["latency_ms"],
    }


def pinned_state(state: AgentState) -> list[dict[str, Any]]:
    """每轮固定注入、不参与压缩的部分：系统指令 + 结构化购物状态。"""
    snap = state.get("shown_snapshot", {})
    items = [f"{it['ord']}. {it['title']}（{it['price']} 元，id={it['id']}）" for it in snap.get("items", [])]
    body = {
        "约束": state.get("constraints", {}),
        "上次展示（序号 -> 商品，第 %s 轮快照）" % snap.get("version", "-"): items,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "name": "shopping_state", "content": json.dumps(body, ensure_ascii=False)},
    ]


def build_agent(
    catalog: Catalog,
    settings: Settings,
    store: BaseStore,
    checkpointer: Any,
    parser: Parser | None = None,
    summarizer: Summarizer = rule_summarize,
    rewriter: Rewriter | None = None,
    ablate: frozenset[str] = frozenset(),
):
    unknown = set(ablate) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"unknown ablation: {sorted(unknown)}")
    ctx = SkillContext(catalog, settings, load_reviews(settings.reviews_path) if "reviews" not in ablate else {})
    memory = LayeredMemory(store)
    archive = ArchiveStore(store)

    def caller_of(state: AgentState) -> Caller:
        return Caller(user_id=state["user_id"], thread_id=state.get("thread_id", "default"))

    def guarded(name: str, fn: Callable[[AgentState], dict[str, Any]]):
        """包一层：计时、记录 trace、异常兜底。"""

        def wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            start = time.perf_counter()
            turn = state.get("turn", 0) + (1 if name == "understand" else 0)
            if name == "understand":
                # 归档和一次性记忆按会话隔离，需要知道当前 thread_id
                state = {**state, "thread_id": config.get("configurable", {}).get("thread_id", "default")}
            try:
                out = fn(state)
                status, error = "ok", None
            except Exception as e:  # noqa: BLE001
                out = dict(NODE_FALLBACK[name])
                status, error = "error", f"{type(e).__name__}: {e}"
                if name == "understand":
                    out.update(turn=turn, messages=[{"role": "user", "content": state.get("query", "")}])
            ms = round((time.perf_counter() - start) * 1000, 2)
            event = {"turn": turn, "node": name, "type": "node", "status": status, "latency_ms": ms}
            if error:
                event["error"] = error
            event.update(out.pop("_trace", {}))
            out["trace"] = out.pop("trace", []) + [event]
            return out

        return wrapper

    # ---------- 节点 ----------

    def understand(state: AgentState) -> dict[str, Any]:
        turn = state.get("turn", 0) + 1
        query = state["query"]
        prev = {} if "state" in ablate else state.get("constraints", {})
        shown = state.get("shown", [])
        notes: list[str] = []

        # 套话类意图（供应商、进货价、系统提示词、他人信息）：不解析、不检索、不调 skill
        if reason := guard_reason(query):
            return {
                "turn": turn,
                "thread_id": state["thread_id"],
                "messages": [{"role": "user", "content": query}],
                "intent": "guard",
                "constraint_diff": {"added": {}, "changed": {}, "removed": []},
                "relaxed": [],
                "replan_count": 0,
                "candidates": [],
                "focus": [],
                "notes": [],
                "_trace": {"intent": "guard", "guard": reason},
            }

        parsed, source = None, "rules"
        if parser is not None:
            titles = [catalog[i].title for i in shown]
            if getattr(parser, "wants_history", False):
                # 历史已经过压缩：摘要 + 最近几轮，约束和展示快照在 prev / titles 里单独给
                parsed = parser(query, prev, titles, history=state.get("messages", []))
            else:
                parsed = parser(query, prev, titles)
            source = "llm" if parsed else "rules(llm_failed)"
        if parsed is None:
            parsed = rule_parse(query)

        constraints = merge_constraints(prev, parsed)
        # 只记录用户这句话带来的变化；后面的画像预填、“更便宜的”不算用户原话
        diff = diff_constraints(prev, constraints)

        # 长期记忆：新会话第一次出现品类时，用 L3 画像预填约束，并预取相关的 L1 / L2
        memory_hits: list[dict[str, Any]] = []
        if "memory" not in ablate and not prev.get("category") and constraints.get("category"):
            profile = load_profile(store, state["user_id"])
            memory_hits = memory.search(caller_of(state), query, k=3)
            dislikes = [
                b
                for b in profile.get("brand_dislikes", {}).get(constraints["category"], [])
                if b not in constraints["brands_include"] and b not in constraints["brands_exclude"]
            ]
            if dislikes:
                constraints["brands_exclude"] = sorted(set(constraints["brands_exclude"]) | set(dislikes))
                notes.append(f"沿用你之前的偏好：不看{'、'.join(dislikes)}")

        # 指代消解：序号 -> 上一次展示的商品
        intent = parsed.intent
        refs = list(parsed.refs)
        if intent == "compare" and not refs and len(shown) >= 2:
            refs = [1, 2]
        focus = [shown[r - 1] for r in refs if 1 <= r <= len(shown)]

        # “有没有更便宜的”：以指代商品（没有就用上次的第一个）为锚点，收紧预算上限
        if parsed.cheaper and shown:
            anchor = focus[0] if focus else shown[0]
            ev = state.get("evidence", {}).get(anchor)
            price = ev["verified_price"] if ev else catalog[anchor].price
            constraints["budget_max"] = min(constraints.get("budget_max") or price, price - 1)
            notes.append(f"在「{catalog[anchor].title}」（{price} 元）的基础上找更便宜的")
            intent, focus = "search", []

        if intent in ("compare", "detail") and len(focus) < (2 if intent == "compare" else 1):
            intent = "clarify"
            notes.append("没找到你指的是哪个商品")
        if intent == "search" and not constraints.get("category"):
            intent = "clarify"

        return {
            "turn": turn,
            "thread_id": state["thread_id"],
            "messages": [{"role": "user", "content": query}],
            "intent": intent,
            "constraints": constraints,
            "constraint_diff": diff,
            "refs": refs,
            "focus": focus,
            "relaxed": [],
            "replan_count": 0,
            "candidates": [],
            "comparison": {},
            "notes": notes,
            "_trace": {
                "parser": source,
                "intent": intent,
                "constraints": constraints,
                "diff": diff,
                "focus": focus,
                "memory_hits": [h["id"] for h in memory_hits],
            },
        }

    def retrieve(state: AgentState) -> dict[str, Any]:
        args = {**_effective(state), "limit": settings.top_k}
        res = call_skill("search_products", args, ctx)
        ids = res["data"] if res["ok"] else []
        turn = state["turn"]
        return {
            "candidates": ids,
            "trace": [_skill_event(turn, "retrieve", "search_products", args, res)],
            "_trace": {"candidates": ids},
        }

    def evidence_for(turn: int, pid: str, constraints: dict[str, Any]) -> tuple[dict | None, list[dict]]:
        """单个商品：拉详情 -> 校验证据。各商品之间互不依赖，可以并行。"""
        events = []
        r1 = call_skill("get_product_detail", {"product_id": pid}, ctx)
        events.append(_skill_event(turn, "gather_evidence", "get_product_detail", {"product_id": pid}, r1))
        if not r1["ok"]:
            return None, events
        if "evidence" in ablate:
            # 消融：直接相信列表页的价格和库存
            d = r1["data"]
            ev = {
                "product_id": pid, "verified_price": d["price"], "list_price": d["price"], "in_stock": True,
                "seller": d["seller"], "seller_rating": d["seller_rating"], "features": d["features"],
                "age_days": 0, "coverage": 1.0, "issues": [], "confidence": 1.0, "violations": [], "usable": True,
            }
            return ev, events
        args = {"product_id": pid, "constraints": constraints}
        r2 = call_skill("verify_evidence", args, ctx)
        events.append(_skill_event(turn, "gather_evidence", "verify_evidence", args, r2))
        return (r2["data"] if r2["ok"] else None), events

    def gather_evidence(state: AgentState) -> dict[str, Any]:
        turn = state["turn"]
        compare_like = state["intent"] in ("compare", "detail")
        targets = state["focus"] if compare_like else state["candidates"]
        constraints = {} if compare_like else _effective(state)
        if settings.parallel_evidence and len(targets) > 1:
            # 详情服务是下游 RPC，串行时总耗时 = 候选数 × 单次时延；并行后约等于最慢的一次
            with ThreadPoolExecutor(max_workers=min(len(targets), settings.max_workers)) as pool:
                results = list(pool.map(lambda pid: evidence_for(turn, pid, constraints), targets))
        else:
            results = [evidence_for(turn, pid, constraints) for pid in targets]
        evidence: dict[str, dict] = {}
        events: list[dict] = []
        for pid, (ev, evs) in zip(targets, results):  # 按候选顺序汇总，trace 顺序稳定
            events += evs
            if ev is not None:
                evidence[pid] = ev
        out: dict[str, Any] = {"evidence": evidence, "trace": events}
        if state["intent"] == "compare":
            args = {"product_ids": state["focus"]}
            r3 = call_skill("compare_products", args, ctx)
            events.append(_skill_event(turn, "gather_evidence", "compare_products", args, r3))
            out["comparison"] = r3["data"] if r3["ok"] else {}
        out["_trace"] = {
            "usable": [p for p, e in evidence.items() if e["usable"]],
            "rejected": {p: e["violations"] for p, e in evidence.items() if not e["usable"]},
        }
        return out

    def judge(state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        evidence = state.get("evidence", {})
        if intent == "guard":
            return {"route": "guard", "confidence": 0.0, "_trace": {"route": "guard"}}
        if intent == "clarify":
            return {"route": "clarify", "confidence": 0.0, "_trace": {"route": "clarify"}}
        if intent in ("compare", "detail"):
            ok = all(p in evidence for p in state["focus"])
            route = intent if ok else "fallback"
            return {"route": route, "confidence": 1.0 if ok else 0.0, "_trace": {"route": route}}

        usable = [p for p in state["candidates"] if evidence.get(p, {}).get("usable")]
        if not usable:
            key = None if "replan" in ablate else _next_relaxation(state)
            if key and state.get("replan_count", 0) < settings.max_replans:
                route = "replan"
            else:
                route = "fallback"
            return {"route": route, "confidence": 0.0, "_trace": {"route": route, "reason": "no_usable_candidate"}}

        conf = round(mean(evidence[p]["confidence"] for p in usable[: settings.show_k]), 2)
        route = "recommend" if conf >= settings.min_confidence else "caveat"
        return {"route": route, "confidence": conf, "_trace": {"route": route, "confidence": conf}}

    def replan(state: AgentState) -> dict[str, Any]:
        key = _next_relaxation(state)
        if key is None:
            return {"route": "fallback"}
        label = RELAX_LABEL[key]
        return {
            "relaxed": state.get("relaxed", []) + [key],
            "replan_count": state.get("replan_count", 0) + 1,
            "notes": state.get("notes", []) + [f"没有同时满足{label}的商品，先放宽{label}"],
            "_trace": {"relaxed": key},
        }

    def reviews_for(state: AgentState, pids: list[str]) -> tuple[dict[str, dict], list[dict]]:
        """给要展示的商品拉评论证据。商品之间互不依赖，和证据聚合一样并行。"""
        if "reviews" in ablate or not ctx.reviews or not pids:
            return {}, []
        user_features = state.get("constraints", {}).get("features", [])

        def one(pid: str) -> tuple[str, dict | None, dict]:
            aspects = select_aspects(state.get("query", ""), user_features, catalog[pid].features)
            if not aspects:
                return pid, None, {}
            args = {"product_id": pid, "aspects": aspects}
            res = call_skill("get_review_evidence", args, ctx)
            return pid, (res["data"] if res["ok"] else None), _skill_event(
                state["turn"], "review_evidence", "get_review_evidence", args, res)

        if settings.parallel_evidence and len(pids) > 1:
            with ThreadPoolExecutor(max_workers=min(len(pids), settings.max_workers)) as pool:
                results = list(pool.map(one, pids))
        else:
            results = [one(p) for p in pids]
        evs = {pid: ev for pid, ev, _ in results if ev is not None}
        return evs, [e for _, _, e in results if e]

    def respond(state: AgentState) -> dict[str, Any]:
        route = state["route"]
        notes = state.get("notes", [])
        evidence = state.get("evidence", {})
        shown = state.get("shown", [])
        lines = [f"（{n}）" for n in notes]
        evidence = dict(evidence)
        review_evs: dict[str, dict] = {}
        events: list[dict] = []

        if route == "guard":
            lines.append(GUARD_REPLY)
        elif route == "clarify":
            if not state.get("constraints", {}).get("category"):
                lines.append("想买哪类商品？比如蓝牙耳机、手机、跑鞋或机械键盘，预算大概多少？")
            else:
                lines.append("你指的是哪一个？可以说“第一个”“第二个”。")
        elif route in ("recommend", "caveat"):
            usable = [p for p in state["candidates"] if evidence.get(p, {}).get("usable")][: settings.show_k]
            if route == "caveat":
                lines.append("以下商品部分信息不完整或可能已过期，下单前请以商品页为准：")
            else:
                lines.append("根据你的要求，推荐这几款：")
            review_evs, events = reviews_for(state, usable)
            for i, pid in enumerate(usable, 1):
                lines += _card(i, pid, catalog, evidence[pid], reason_lines(review_evs.get(pid, {})))
            shown = usable
        elif route == "compare":
            cmp = state.get("comparison", {})
            lines.append("对比如下：")
            for it in cmp.get("items", []):
                feats = "、".join(it["features"]) or "无"
                stock = "有货" if it["in_stock"] else "缺货"
                lines.append(f"- {it['title']}：{it['price']} 元｜评分 {it['rating']}｜{stock}｜特性：{feats}")
            if cmp:
                lines.append(
                    f"更便宜：{catalog[cmp['cheapest']].title}；评分更高：{catalog[cmp['best_rated']].title}"
                )
        elif route == "detail":
            pid = state["focus"][0]
            p, ev = catalog[pid], evidence[pid]
            lines.append(f"{p.title}：")
            lines.append(f"- 价格 {ev['verified_price']} 元，{'有货' if ev['in_stock'] else '缺货'}")
            lines.append(f"- 评分 {p.rating}，销量 {p.sales}，商家 {p.seller}（评分 {p.seller_rating or '暂无'}）")
            lines.append(f"- 特性：{'、'.join(p.features) or '无'}")
            lines += [f"- 注意：{t}" for t in describe_issues(ev)]
            review_evs, events = reviews_for(state, [pid])
            lines += [ln.replace("   ", "- ", 1) for ln in reason_lines(review_evs.get(pid, {}))]
            lines.append(f"- 链接：{p.deep_link}")
        else:  # fallback
            c = state.get("constraints", {})
            lines.append("在你的条件下没有找到可以放心推荐的商品。")
            if c.get("category") and c.get("budget_max") is not None:
                relaxed_budget = {k: v for k, v in c.items() if k not in ("budget_max", "features", "brands_include")}
                res = call_skill("search_products", {**relaxed_budget, "limit": 20}, ctx)
                if res["ok"] and res["data"]:
                    nearest = min(res["data"], key=lambda pid: catalog[pid].price)
                    # 兜底推荐也要按详情页价格报价，不能直接报列表价
                    ev, _ = evidence_for(state["turn"], nearest, {})
                    if ev is not None:
                        evidence[nearest] = ev
                        p = catalog[nearest]
                        lines.append(f"最接近的是「{p.title}」，{ev['verified_price']} 元。要不要放宽一下预算？")

        text = "\n".join(lines)
        # 回复里出现的每个价格都必须来自证据层；改写后对不上就退回模板回复
        allowed = {e["verified_price"] for e in evidence.values()}
        # 上一轮快照里的价格也是核验过的（“在 X（499 元）的基础上找更便宜的”）
        allowed |= {it["price"] for it in state.get("shown_snapshot", {}).get("items", [])}
        # 评论引用只能指向本轮检索到的片段，原话要对得上，差评不能当卖点讲
        snippets = snippet_index(review_evs)
        check: dict[str, Any] = {"ok": True, "rewritten": False}
        if rewriter is not None and route in ("recommend", "caveat", "compare", "detail"):
            candidate = rewriter(text, state)
            bad = fact_check(candidate, allowed)
            bad_cites = citation_check(candidate, snippets)
            if bad or bad_cites:
                check = {"ok": False, "rewritten": False}
                if bad:
                    check["unverified_prices"] = bad
                if bad_cites:
                    check["citation_problems"] = bad_cites
            else:
                text, check = candidate, {"ok": True, "rewritten": True}
        elif (bad := fact_check(text, allowed)) or (bad_cites := citation_check(text, snippets)):
            check = {"ok": False, "unverified_prices": bad, "citation_problems": bad_cites if not bad else []}
        snapshot = state.get("shown_snapshot", {})
        if route in ("recommend", "caveat"):
            # 快照：序号 -> 商品 ID 的映射和展示时的价格固定下来，
            # 下一轮“第二个”按快照解析，不受候选重排影响；“刚才那款多少钱”也读快照
            snapshot = {
                "version": state["turn"],
                "items": [
                    {"ord": i, "id": pid, "title": catalog[pid].title, "price": evidence[pid]["verified_price"]}
                    for i, pid in enumerate(shown, 1)
                ],
            }
        review_trace = {pid: {a["aspect"]: a["verdict"] for a in ev["aspects"]} for pid, ev in review_evs.items()}
        out: dict[str, Any] = {
            "response": text,
            "shown": shown,
            "shown_snapshot": snapshot,
            "review_evidence": review_evs,
            "messages": [{"role": "assistant", "content": text}],
            "_trace": {"route": route, "shown": shown, "fact_check": check,
                       **({"reviews": review_trace} if review_trace else {})},
        }
        if events:
            out["trace"] = events
        return out

    def update_memory(state: AgentState) -> dict[str, Any]:
        if "memory" in ablate or state.get("route") in ("clarify", "guard"):
            return {}
        if not state.get("constraints", {}).get("category"):
            return {}
        res = memory.record_turn(
            caller_of(state), state["query"], state.get("constraint_diff", {}), state["constraints"], state["turn"]
        )
        return {
            "_trace": {
                "source_ref": res["source_ref"],
                "facts": [f"{f['op']}:{f['text']}({f['scope']})" for f in res["facts"]],
                "profile": {k: v for k, v in res["profile"].items() if k != "provenance"},
            }
        }

    def compact_context(state: AgentState) -> dict[str, Any]:
        msgs = state.get("messages", [])
        res = compact(
            msgs,
            budget=settings.context_budget_tokens,
            pinned_tokens=count_tokens(pinned_state(state)),
            archive=archive,
            caller=caller_of(state),
            summarizer=summarizer,
        )
        stats = {
            "status": res.status,
            "tokens_before": res.tokens_before,
            "tokens_after": res.tokens_after,
            "refs": res.ref_ids,
        }
        out: dict[str, Any] = {"context": stats, "_trace": stats}
        if res.changed:
            out["messages"] = ReplaceMessages(res.messages)
        return out

    # ---------- 图 ----------

    g = StateGraph(AgentState)
    for name, fn in [
        ("understand", understand),
        ("retrieve", retrieve),
        ("gather_evidence", gather_evidence),
        ("judge", judge),
        ("replan", replan),
        ("respond", respond),
        ("update_memory", update_memory),
        ("compact_context", compact_context),
    ]:
        g.add_node(name, guarded(name, fn))

    g.add_edge(START, "understand")
    g.add_conditional_edges(
        "understand",
        lambda s: {"clarify": "judge", "guard": "judge", "compare": "gather_evidence", "detail": "gather_evidence"}.get(
            s["intent"], "retrieve"
        ),
        ["judge", "gather_evidence", "retrieve"],
    )
    g.add_edge("retrieve", "gather_evidence")
    g.add_edge("gather_evidence", "judge")
    g.add_conditional_edges("judge", lambda s: "replan" if s["route"] == "replan" else "respond", ["replan", "respond"])
    g.add_conditional_edges("replan", lambda s: "respond" if s.get("route") == "fallback" else "retrieve", ["retrieve", "respond"])
    g.add_edge("respond", "update_memory")
    g.add_edge("update_memory", "compact_context")
    g.add_edge("compact_context", END)
    return g.compile(checkpointer=checkpointer, store=store)


def _card(i: int, pid: str, catalog: Catalog, ev: dict[str, Any], reasons: list[str] | None = None) -> list[str]:
    p = catalog[pid]
    out = [f"{i}. {p.title}｜{ev['verified_price']} 元｜评分 {p.rating}｜{p.seller}"]
    if p.features:
        out.append(f"   特性：{'、'.join(p.features)}")
    out += reasons or []
    out += [f"   注意：{t}" for t in describe_issues(ev)]
    out.append(f"   {p.deep_link}")
    return out
