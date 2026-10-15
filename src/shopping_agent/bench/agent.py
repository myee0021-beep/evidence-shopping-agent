"""在 ShoppingBench 沙箱上跑同一套 harness 链路。

    understand（任务解析）
      -> retrieve（search_products -> 沙箱 find_product，价格 / 服务下推给沙箱过滤）
      -> gather_evidence（get_product_detail -> view_product_information；verify_evidence 本地逐项核验属性词）
      -> judge（有没有属性全命中的候选）
      -> replan（换检索策略：补缺失属性词 -> 翻页 -> 缩短 query；硬约束不放宽）
      -> align_shop（同店约束：先求交集，没有就按店铺定向检索）
      -> plan_basket（多件商品 + 优惠券的组合求解）
      -> respond（recommend_product + terminate）

skill 名沿用主链路，trace 格式也一样，可以直接用 shopping_agent.trace.replay 回放。
和主链路的一处区别：主链路重规划是“放宽软约束”，这里是“换检索策略”。
ShoppingBench 按属性逐项打分，放宽属性等于直接丢分，所以宁可多搜几次。
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .sandbox import Sandbox
from .task import ItemSpec, Task, TaskSpec, rule_parse_task
from .voucher import plan_basket

TaskParser = Callable[[str, Task], TaskSpec | None]

MAX_REPLANS = 2
DETAIL_K = 10  # 每次检索拿详情的候选数（沙箱一页 10 条）


def tool_call_id(name: str, parameters: dict[str, Any], length: int = 8) -> str:
    """和官方 util/message.generate_tool_call_id 相同的算法，rollout 里的 id 风格一致。"""
    digest = hashlib.md5(f"{name}\n{parameters}".encode("utf-8")).digest()
    s = base64.urlsafe_b64encode(digest).decode("utf-8").replace("=", "").replace("+", "").replace("/", "")
    return s[:length]


# ---------- 证据核验 ----------


def product_text(p: dict[str, Any]) -> tuple[set[tuple[str, str]], set[str], str]:
    pairs: set[tuple[str, str]] = set()
    for kv in (p.get("sku_options") or {}).values():
        pairs |= {(k.lower(), str(v).lower()) for k, v in kv.items()}
    for k, vs in (p.get("attributes") or {}).items():
        pairs |= {(k.lower(), str(v).lower()) for v in vs}
    values = {v for _, v in pairs}
    text = " ".join(str(p.get(k) or "") for k in ("title", "short_description")).lower()
    return pairs, values, text


def verify(p: dict[str, Any], item: ItemSpec) -> dict[str, Any]:
    """逐项核验：价格、服务是硬约束（违反即不可用）；属性词算覆盖率。

    属性词命中分三档：(key, value) 精确命中 > 值出现在 SKU / 属性里 > 值出现在标题描述里。
    官方评测只认前两档，所以第三档单独记为 weak，方便在 trace 里看出“标题里有、属性里没有”的情况。
    """
    violations = []
    price = float(p.get("price", 0))
    if item.price_min is not None and price < item.price_min:
        violations.append("price_below_min")
    if item.price_max is not None and price > item.price_max:
        violations.append("price_above_max")
    for s in item.services:
        if s not in (p.get("service") or []):
            violations.append(f"missing_service:{s}")

    pairs, values, text = product_text(p)
    matched, weak, missing = [], [], []
    for t in item.terms:
        v = t.value.lower()
        if (t.key and (t.key.lower(), v) in pairs) or v in values:
            matched.append(t.value)
        elif v in text:
            weak.append(t.value)
        else:
            missing.append(t.value)
    n = len(item.terms)
    score = (len(matched) + 0.5 * len(weak)) / n if n else 1.0
    return {
        "product_id": str(p["product_id"]),
        "shop_id": str(p.get("shop_id")),
        "price": price,
        "sold_count": p.get("sold_count", 0),
        "score": round(score, 3),
        "matched": matched,
        "weak": weak,
        "missing": missing,
        "violations": violations,
        "usable": not violations,
    }


# ---------- Agent ----------


@dataclass
class RunResult:
    query: str
    spec: TaskSpec
    recommended: list[str]
    steps: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    products: dict[str, dict[str, Any]] = field(default_factory=dict)
    basket: dict[str, Any] | None = None


class BenchAgent:
    def __init__(self, sandbox: Sandbox, parser: TaskParser | None = None):
        self.sandbox = sandbox
        self.parser = parser

    def run(self, query: str, task: Task) -> RunResult:
        return _Episode(self.sandbox, self.parser, query, task).run()


class _Episode:
    def __init__(self, sandbox: Sandbox, parser: TaskParser | None, query: str, task: Task):
        self.sb, self.parser, self.query, self.task = sandbox, parser, query, task
        self.trace: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []

    # ---- 记录 ----

    def node(self, name: str, start: float, **extra: Any) -> None:
        ms = round((time.perf_counter() - start) * 1000, 2)
        self.trace.append({"turn": 1, "node": name, "type": "node", "status": "ok", "latency_ms": ms, **extra})

    def call(self, node: str, skill: str, tool: str, params: dict[str, Any], think: str) -> Any:
        """一次沙箱调用 = 一条 skill trace + 一个官方格式的 rollout step。"""
        start = time.perf_counter()
        try:
            fn = getattr(self.sb, tool)
            results, ok, err = fn(**params), True, None
        except Exception as e:  # noqa: BLE001 - 工具失败交给编排层：当作空结果继续
            results, ok, err = [], False, f"{type(e).__name__}: {e}"
        ms = round((time.perf_counter() - start) * 1000, 2)
        self.trace.append(
            {"turn": 1, "node": node, "type": "skill", "skill": skill, "args": {"tool": tool, **params},
             "ok": ok, "error": err, "latency_ms": ms}
        )
        self.step(think, [(tool, params, results)])
        return results

    def step(self, think: str, calls: list[tuple[str, dict[str, Any], Any]]) -> None:
        tool_calls, obs = [], []
        for name, params, results in calls:
            cid = tool_call_id(name, params)
            tool_calls.append({"name": name, "parameters": params, "tool_call_id": cid})
            if results is not None:
                obs.append({"tool_call_id": cid, "results": results})
        content = (
            f"<think>{think}</think>\n<tool_call>"
            + json.dumps([{"name": c["name"], "parameters": c["parameters"]} for c in tool_calls], ensure_ascii=False)
            + "</tool_call>"
        )
        message: dict[str, Any] = {"think": think, "tool_call": tool_calls}
        if obs:
            message["obs"] = obs
        self.steps.append(
            {
                "prompt": [{"role": "user", "content": self.query}] if not self.steps else [],
                "completion": {"reasoning_content": "", "content": content, "message": message},
                "extra_info": {"step": len(self.steps) + 1, "query": self.query, "timestamp": int(time.time() * 1000)},
            }
        )

    # ---- 节点 ----

    def understand(self) -> TaskSpec:
        start = time.perf_counter()
        spec, source = None, "rules"
        if self.parser is not None:
            try:
                spec = self.parser(self.query, self.task)
            except Exception:  # noqa: BLE001 - LLM 失败走规则兜底
                spec = None
            source = "llm" if spec else "rules(llm_failed)"
        if spec is None:
            spec = rule_parse_task(self.query, self.task)
        self.node("understand", start, parser=source, items=[i.model_dump(exclude_none=True) for i in spec.items],
                  voucher=spec.voucher.model_dump() if spec.voucher else None)
        return spec

    def search_and_verify(self, idx: int, item: ItemSpec, q: str, page: int, shop_id: str | None = None) -> list[dict]:
        params: dict[str, Any] = {"q": q, "page": page}
        if price := item.price_param():
            params["price"] = price
        if item.services:
            params["service"] = ",".join(item.services)
        if shop_id:
            params["shop_id"] = shop_id
        start = time.perf_counter()
        rows = self.call("retrieve", "search_products", "find_product", params,
                         f"Search candidates for item {idx + 1}: {q!r} (page {page}).")
        self.node("retrieve", start, item=idx + 1, q=q, page=page, shop_id=shop_id, candidates=len(rows))
        if not rows:
            return []
        start = time.perf_counter()
        ids = ",".join(str(r["product_id"]) for r in rows[:DETAIL_K])
        self.call("gather_evidence", "get_product_detail", "view_product_information", {"product_ids": ids},
                  f"Check SKU options and attributes of item {idx + 1} candidates.")
        checked = [verify(self.sb.seen[str(r["product_id"])], item) for r in rows[:DETAIL_K]]
        usable = [c for c in checked if c["usable"]]
        usable.sort(key=lambda c: (-c["score"], -c["sold_count"]))
        self.node(
            "gather_evidence", start, item=idx + 1,
            usable=[c["product_id"] for c in usable[:5]],
            best={k: usable[0][k] for k in ("product_id", "score", "missing", "weak")} if usable else None,
            rejected={c["product_id"]: c["violations"] for c in checked if not c["usable"]},
        )
        return usable

    def find_item(self, idx: int, item: ItemSpec) -> list[dict]:
        pool: dict[str, dict] = {}
        q, page, tried = item.q, 1, []
        for attempt in range(MAX_REPLANS + 1):
            for c in self.search_and_verify(idx, item, q, page):
                pool.setdefault(c["product_id"], c)
            ranked = sorted(pool.values(), key=lambda c: (-c["score"], -c["sold_count"]))
            start = time.perf_counter()
            done = bool(ranked) and ranked[0]["score"] >= 1.0
            route = "accept" if done or attempt == MAX_REPLANS else "replan"
            self.node("judge", start, item=idx + 1, route=route, best_score=ranked[0]["score"] if ranked else None)
            if route == "accept":
                break
            # 重规划：换检索策略，不放宽硬约束
            start = time.perf_counter()
            missing = ranked[0]["missing"] if ranked else []
            if missing and "add_terms" not in tried:
                strategy, q, page = "add_terms", f"{item.q} {' '.join(missing[:3])}", 1
            elif ranked and "next_page" not in tried:
                strategy, page = "next_page", page + 1
            else:
                strategy, q, page = "shorten_query", " ".join(item.q.split()[:4]), 1
            tried.append(strategy)
            self.node("replan", start, item=idx + 1, strategy=strategy, q=q, page=page)
        return sorted(pool.values(), key=lambda c: (-c["score"], -c["sold_count"]))

    def align_shop(self, spec: TaskSpec, cands: list[list[dict]]) -> list[list[dict]]:
        """同店约束：先看各件商品的候选有没有共同店铺；没有就拿第一件的店铺做定向检索。"""
        start = time.perf_counter()
        common = set.intersection(*({c["shop_id"] for c in cs} for cs in cands)) if all(cands) else set()
        if common:
            self.node("align_shop", start, strategy="intersection", shops=sorted(common)[:5])
            return [[c for c in cs if c["shop_id"] in common] for cs in cands]
        self.node("align_shop", start, strategy="shop_scoped_search")
        seeds = list(dict.fromkeys(c["shop_id"] for cs in cands for c in cs))[:3]
        for shop in seeds:
            scoped = []
            for i, item in enumerate(spec.items):
                own = [c for c in cands[i] if c["shop_id"] == shop]
                scoped.append(own or self.search_and_verify(i, item, item.q, 1, shop_id=shop))
            if all(scoped):
                return scoped
        return cands

    def run(self) -> RunResult:
        spec = self.understand()
        cands = [self.find_item(i, item) for i, item in enumerate(spec.items)]
        if spec.same_shop and len(spec.items) > 1:
            cands = self.align_shop(spec, cands)

        start = time.perf_counter()
        basket = plan_basket(cands, spec.voucher, spec.same_shop and len(spec.items) > 1)
        v = spec.voucher
        if v and v.voucher_type == "shop" and len(spec.items) > 1 and not (basket and basket["budget_ok"]):
            # 跨店组合超预算：店铺券要求同店才能用，按店铺定向检索再求一次
            self.node("plan_basket", start, chosen=None, note="cross-shop basket over budget, try same-shop")
            aligned = plan_basket(self.align_shop(spec, cands), v, same_shop=True)
            if aligned and (basket is None or aligned["budget_ok"]):
                basket = aligned
            start = time.perf_counter()
        if basket is None:
            # 某件商品没有可用候选：其余照常推荐，缺的那件留空位，评测按缺失计分
            chosen = [cs[0]["product_id"] if cs else "" for cs in cands]
        else:
            chosen = [c["product_id"] for c in basket["items"]]
        self.node("plan_basket", start, chosen=chosen,
                  basket={k: v for k, v in (basket or {}).items() if k != "items"} or None)

        start = time.perf_counter()
        ids = ",".join(chosen)
        if any(chosen):
            self.step("Recommend the verified products in the requested order.",
                      [("recommend_product", {"product_ids": ids}, f"Having recommended the products to the user: {ids}.")])
        status = "success" if all(chosen) else "failure"
        self.step("Task finished.", [("terminate", {"status": status}, f"The interaction has been completed with status: {status}")])
        self.node("respond", start, route="recommend" if any(chosen) else "fallback", recommended=chosen)

        products = {pid: dict(self.sb.seen.get(pid, {})) for pid in chosen if pid}
        return RunResult(self.query, spec, chosen, self.steps, self.trace, products, basket)
