"""在 ShoppingBench 沙箱上跑同一套 harness 链路。

    understand（任务解析）
      -> 每件商品一个子 agent（ItemFinder，独立上下文，可并行）：
           retrieve（search_products -> 沙箱 find_product，价格 / 服务下推给沙箱过滤）
        -> gather_evidence（get_product_detail -> view_product_information；verify_evidence 本地逐项核验属性词）
        -> judge（有没有属性全命中的候选）
        -> replan（换检索策略：补缺失属性词 -> 翻页 -> 缩短 query；硬约束不放宽）
         子 agent 只把候选列表交回主 agent，自己的检索过程不进主 agent 的决策
      -> 循环护栏（CallGuard）：同一任务里参数相同的调用复用结果；工具调用总数有上限；
         重规划拿不出新的 (query, page) 就停
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..tooling import ToolArgsError
from .guard import CallGuard
from .sandbox import Sandbox
from .task import ItemSpec, Task, TaskSpec, rule_parse_task
from .tools import canonical_params
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
class EpisodeConfig:
    """一条任务的运行配置。默认值就是评测时用的配置；消融时逐项关掉。"""

    max_tool_calls: int | None = 40  # 一条任务最多真实调用沙箱多少次（去重命中不算）
    dedup: bool = True  # 参数相同的只读调用复用结果
    stop_on_no_progress: bool = True  # 重规划拿不出新的 (query, page) 就停，而不是重复检索
    workers: int = 4  # 多件商品的子 agent 并行数；1 = 串行


@dataclass
class RunResult:
    query: str
    spec: TaskSpec
    recommended: list[str]
    steps: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    products: dict[str, dict[str, Any]] = field(default_factory=dict)
    basket: dict[str, Any] | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class BenchAgent:
    def __init__(self, sandbox: Sandbox, parser: TaskParser | None = None, config: EpisodeConfig | None = None):
        self.sandbox = sandbox
        self.parser = parser
        self.config = config or EpisodeConfig()

    def run(self, query: str, task: Task) -> RunResult:
        return _Episode(self.sandbox, self.parser, query, task, self.config).run()


class _Worker:
    """有自己 trace 和 rollout 缓冲区的执行单元。主 agent 和每个子 agent 各是一个 _Worker。

    缓冲区彼此独立：子 agent 跑的时候看不到别的商品的检索过程，主 agent 也不读子 agent 的中间结果，
    只拿它交回的候选列表。rollout step 先记成 (think, calls)，最后由主 agent 按商品顺序统一渲染，
    所以并行执行时 rollout 的顺序也和串行一致。
    """

    def __init__(self, name: str, sb: Sandbox, guard: CallGuard, config: EpisodeConfig):
        self.name, self.sb, self.guard, self.config = name, sb, guard, config
        self.trace: list[dict[str, Any]] = []
        self.pending: list[tuple[str, list[tuple[str, dict[str, Any], Any]]]] = []

    # ---- 记录 ----

    def node(self, name: str, start: float, **extra: Any) -> None:
        ms = round((time.perf_counter() - start) * 1000, 2)
        self.trace.append({"turn": 1, "node": name, "type": "node", "status": "ok", "latency_ms": ms,
                           "agent": self.name, **extra})

    def call(self, node: str, skill: str, tool: str, params: dict[str, Any], think: str) -> Any:
        """一次沙箱调用 = 一条 skill trace + 一个官方格式的 rollout step（去重命中和被预算拦下的不写 step）。"""
        start = time.perf_counter()
        event: dict[str, Any] = {"turn": 1, "node": node, "type": "skill", "skill": skill, "agent": self.name}
        try:
            params = canonical_params(tool, params)
        except ToolArgsError as e:  # 参数不合法：本地拦下，不浪费一次沙箱调用
            self.trace.append({**event, "args": {"tool": tool, **params}, "ok": False, "error": str(e),
                               "latency_ms": 0.0})
            return []
        event["args"] = {"tool": tool, **params}
        decision, cached = self.guard.before(tool, params)
        if decision == "cached":
            self.trace.append({**event, "ok": True, "error": None, "cached": True, "latency_ms": 0.0})
            return cached
        if decision == "blocked":
            self.trace.append({**event, "ok": False, "error": "tool_budget_exhausted", "latency_ms": 0.0})
            return []
        try:
            results, ok, err = getattr(self.sb, tool)(**params), True, None
        except Exception as e:  # noqa: BLE001 - 工具失败交给编排层：当作空结果继续
            results, ok, err = [], False, f"{type(e).__name__}: {e}"
        self.guard.after(tool, params, results, ok)
        ms = round((time.perf_counter() - start) * 1000, 2)
        self.trace.append({**event, "ok": ok, "error": err, "latency_ms": ms})
        self.pending.append((think, [(tool, params, results)]))
        return results

    # ---- 检索 + 核验 ----

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


@dataclass
class SubResult:
    """子 agent 交回主 agent 的全部内容：候选列表 + 一行摘要。检索过程留在子 agent 自己的缓冲区里。"""

    idx: int
    candidates: list[dict[str, Any]]
    worker: _Worker
    stop_reason: str
    latency_ms: float

    def summary(self) -> dict[str, Any]:
        best = self.candidates[0] if self.candidates else None
        return {
            "item": self.idx + 1,
            "candidates": len(self.candidates),
            "best": {k: best[k] for k in ("product_id", "score", "missing")} if best else None,
            "tool_calls": sum(1 for e in self.worker.trace if e["type"] == "skill" and not e.get("cached")),
            "stop_reason": self.stop_reason,
        }


class ItemFinder(_Worker):
    """单件商品的子 agent：检索 -> 核验 -> 判断 -> 重规划，直到属性全命中或策略用完。"""

    def __init__(self, idx: int, item: ItemSpec, sb: Sandbox, guard: CallGuard, config: EpisodeConfig):
        super().__init__(f"item-{idx + 1}", sb, guard, config)
        self.idx, self.item = idx, item

    def next_strategy(self, ranked: list[dict], q: str, page: int, tried: list[str],
                      seen: set[tuple[str, int]]) -> tuple[str, str, int] | None:
        """按顺序给出下一个检索策略。开了护栏时跳过会重复已搜过的 (query, page) 的策略。"""
        item = self.item
        missing = ranked[0]["missing"] if ranked else []
        options = []
        if missing:
            options.append(("add_terms", f"{item.q} {' '.join(missing[:3])}", 1))
        if ranked:
            options.append(("next_page", q, page + 1))
        options.append(("shorten_query", " ".join(item.q.split()[:4]), 1))
        for name, nq, npage in options:
            if name in tried:
                continue
            if self.config.stop_on_no_progress and (nq, npage) in seen:
                tried.append(name)  # 这个策略不会带来新结果，记为已试过
                continue
            return name, nq, npage
        return None if self.config.stop_on_no_progress else ("shorten_query", options[-1][1], 1)

    def run(self) -> SubResult:
        t0 = time.perf_counter()
        pool: dict[str, dict] = {}
        q, page, tried = self.item.q, 1, []
        seen: set[tuple[str, int]] = set()
        stop = "max_replans"
        for attempt in range(MAX_REPLANS + 1):
            seen.add((q, page))
            for c in self.search_and_verify(self.idx, self.item, q, page):
                pool.setdefault(c["product_id"], c)
            ranked = sorted(pool.values(), key=lambda c: (-c["score"], -c["sold_count"]))
            start = time.perf_counter()
            done = bool(ranked) and ranked[0]["score"] >= 1.0
            nxt = None
            if not done and attempt < MAX_REPLANS and not self.guard.exhausted:
                nxt = self.next_strategy(ranked, q, page, tried, seen)
            route = "replan" if nxt else "accept"
            if done:
                stop = "all_terms_matched"
            elif self.guard.exhausted:
                stop = "tool_budget_exhausted"
            elif nxt is None and attempt < MAX_REPLANS:
                stop = "no_new_strategy"
            self.node("judge", start, item=self.idx + 1, route=route,
                      best_score=ranked[0]["score"] if ranked else None,
                      **({} if route == "replan" else {"stop_reason": stop}))
            if nxt is None:
                break
            # 重规划：换检索策略，不放宽硬约束
            start = time.perf_counter()
            strategy, q, page = nxt
            tried.append(strategy)
            self.node("replan", start, item=self.idx + 1, strategy=strategy, q=q, page=page)
        cands = sorted(pool.values(), key=lambda c: (-c["score"], -c["sold_count"]))
        return SubResult(self.idx, cands, self, stop, round((time.perf_counter() - t0) * 1000, 2))


class _Episode(_Worker):
    """主 agent：解析任务、派发子 agent、同店对齐、组合求解、推荐。"""

    def __init__(self, sandbox: Sandbox, parser: TaskParser | None, query: str, task: Task, config: EpisodeConfig):
        guard = CallGuard(max_calls=config.max_tool_calls, dedup=config.dedup)
        super().__init__("main", sandbox, guard, config)
        self.parser, self.query, self.task = parser, query, task
        self.steps: list[dict[str, Any]] = []

    def render(self, pending: list[tuple[str, list[tuple[str, dict[str, Any], Any]]]]) -> None:
        for think, calls in pending:
            self.step(think, calls)

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

    def dispatch(self, spec: TaskSpec) -> list[list[dict]]:
        """每件商品交给一个子 agent。多件商品时并行跑，结果按商品顺序合并。"""
        start = time.perf_counter()
        finders = [ItemFinder(i, item, self.sb, self.guard, self.config) for i, item in enumerate(spec.items)]
        workers = min(self.config.workers, len(finders))

        def safe_run(f: ItemFinder) -> SubResult:
            try:
                return f.run()
            except Exception as e:  # noqa: BLE001 - 一件商品失败不拖垮整条任务
                f.trace.append({"turn": 1, "node": "subagent", "type": "node", "status": "error",
                                "error": f"{type(e).__name__}: {e}", "latency_ms": 0.0, "agent": f.name})
                return SubResult(f.idx, [], f, "error", 0.0)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="item") as pool:
                results = list(pool.map(safe_run, finders))
        else:
            results = [safe_run(f) for f in finders]
        for r in results:  # 按商品顺序合并：rollout 和串行时完全一致
            self.trace.extend(r.worker.trace)
            self.render(r.worker.pending)
            self.trace.append({"turn": 1, "node": "subagent", "type": "node", "status": "ok",
                               "latency_ms": r.latency_ms, "agent": r.worker.name, **r.summary()})
        self.node("dispatch", start, subagents=len(finders), parallel=workers)
        return [r.candidates for r in results]

    def align_shop(self, spec: TaskSpec, cands: list[list[dict]]) -> list[list[dict]]:
        """同店约束：先看各件商品的候选有没有共同店铺；没有就拿前几个店铺做定向检索。"""
        start = time.perf_counter()
        common = set.intersection(*({c["shop_id"] for c in cs} for cs in cands)) if all(cands) else set()
        if common:
            self.node("align_shop", start, strategy="intersection", shops=sorted(common)[:5])
            return [[c for c in cs if c["shop_id"] in common] for cs in cands]
        self.node("align_shop", start, strategy="shop_scoped_search")
        seeds = list(dict.fromkeys(c["shop_id"] for cs in cands for c in cs))[:3]
        result = cands
        for shop in seeds:
            if self.guard.exhausted:
                break
            scoped = []
            for i, item in enumerate(spec.items):
                own = [c for c in cands[i] if c["shop_id"] == shop]
                scoped.append(own or self.search_and_verify(i, item, item.q, 1, shop_id=shop))
            if all(scoped):
                result = scoped
                break
        self.render(self.pending)
        self.pending = []
        return result

    def run(self) -> RunResult:
        t0 = time.perf_counter()
        spec = self.understand()
        cands = self.dispatch(spec)
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
        g = self.guard.stats
        stats = {"tool_calls": g.calls, "cached": g.cached, "blocked": g.blocked,
                 "wall_ms": round((time.perf_counter() - t0) * 1000, 2)}
        return RunResult(self.query, spec, chosen, self.steps, self.trace, products, basket, stats)
