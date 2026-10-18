import json
import re

from langgraph.store.memory import InMemoryStore

from shopping_agent.archive import ArchiveStore, Caller
from shopping_agent.bench.guard import CallGuard
from shopping_agent.bench.tools import BENCH_TOOLS, canonical_params
from shopping_agent.context import digest_products, offload_tool_results
from shopping_agent.skills import REGISTRY
from shopping_agent.tooling import ToolArgsError, ToolSpec

ME = Caller("u", "t")


def products(n):
    return [{"id": f"E{i:02d}", "title": f"商品 {i}", "brand": "华为", "price": 100 + i, "detail_price": 120 + i,
             "stock": i, "rating": 4.5, "sales": 1000, "seller": "官方店", "features": ["降噪", "长续航"],
             "updated_at": "2026-09-25"} for i in range(n)]


def test_digest_keeps_id_both_prices_stock_and_attrs():
    d = digest_products(json.dumps(products(3), ensure_ascii=False))
    lines = d.splitlines()
    assert len(lines) == 3
    assert "id=E01" in lines[1] and "price=101" in lines[1] and "detail_price=121" in lines[1]
    assert "stock=1" in lines[1] and "attrs=降噪/长续航" in lines[1]
    assert "seller=" not in d and "sales=" not in d


def test_digest_handles_skill_envelope_and_bench_shape():
    env = json.dumps({"ok": True, "data": products(1)}, ensure_ascii=False)
    assert "id=E00" in digest_products(env)
    bench = json.dumps([{"product_id": "W1", "shop_id": "s1", "price": 350.0,
                         "sku_options": {"1": {"size": "12 inch"}}, "attributes": {"material": ["rubber"]}}])
    d = digest_products(bench)
    assert "id=W1" in d and "shop_id=s1" in d and "size=12 inch" in d and "material=rubber" in d
    assert digest_products("not json") is None and digest_products('{"answer": 1}') is None


def test_old_product_results_are_digested_but_latest_turn_kept_raw():
    archive = ArchiveStore(InMemoryStore())
    raw = json.dumps(products(3), ensure_ascii=False)  # 不到 1200 字符，按长度规则不会卸载
    msgs = [
        {"role": "user", "content": "q1"}, {"role": "tool", "tool_call_id": "a", "content": raw},
        {"role": "user", "content": "q2"}, {"role": "tool", "tool_call_id": "b", "content": raw},
    ]
    out, refs = offload_tool_results(msgs, archive, ME, max_chars=10_000, digest_before=2)
    assert out[1]["offload_kind"] == "products" and "id=E02" in out[1]["content"] and len(refs) == 1
    assert out[3]["content"] == raw  # 最近一轮保持原文
    old, _ = offload_tool_results(msgs, archive, ME, max_chars=10_000, digest_before=2, product_digest=False)
    assert old[1]["content"] == raw  # 关掉摘要 = 旧行为


def test_skills_and_bench_tools_share_one_contract():
    assert isinstance(REGISTRY["search_products"], ToolSpec)
    assert BENCH_TOOLS["find_product"].schema()["function"]["parameters"]["properties"]["page"]["maximum"] == 5
    assert canonical_params("find_product", {"q": "wheel", "page": 1, "price": None}) == {"q": "wheel", "page": 1}
    for bad in ({"q": "", "page": 1}, {"q": "x", "page": 9}):
        try:
            canonical_params("find_product", bad)
            raise AssertionError("should reject")
        except ToolArgsError as e:
            assert "invalid args" in str(e)


def test_call_guard_dedups_and_enforces_budget():
    g = CallGuard(max_calls=2)
    p = {"q": "a", "page": 1}
    assert g.before("find_product", p) == ("run", None)
    g.after("find_product", p, ["r"], ok=True)
    assert g.before("find_product", p) == ("cached", ["r"])
    assert g.before("find_product", {"q": "b", "page": 1})[0] == "run"
    assert g.before("find_product", {"q": "c", "page": 1})[0] == "blocked"
    assert (g.stats.calls, g.stats.cached, g.stats.blocked) == (2, 1, 1) and g.exhausted
    g2 = CallGuard()
    g2.before("find_product", p)
    g2.after("find_product", p, [], ok=False)  # 失败结果不缓存
    assert g2.before("find_product", p)[0] == "run"


# ---------- ShoppingBench：子 agent、并行、循环护栏 ----------

import threading  # noqa: E402
import time  # noqa: E402

from shopping_agent.bench.agent import BenchAgent, EpisodeConfig  # noqa: E402
from shopping_agent.bench.run import efficiency, select_cases  # noqa: E402
from shopping_agent.bench.sandbox import LatencySandbox, OfflineSandbox  # noqa: E402
from shopping_agent.bench.task import CachedParser, ItemSpec, Term, rule_parse_task  # noqa: E402
from test_bench import DOCS, VOUCHER, stub  # noqa: E402

NO_GUARD = EpisodeConfig(dedup=False, stop_on_no_progress=False, max_tool_calls=None, workers=1)
SHOP_Q = "Find a shop that offers these items: (1) oil pastel crayons; (2) a sketch pad drawing paper."


def rollout(res):
    return [[(c["name"], c["parameters"]) for c in s["completion"]["message"]["tool_call"]] for s in res.steps]


def run(q, task, cfg, items=None, sb=None):
    return BenchAgent(sb or OfflineSandbox.from_documents(DOCS), stub(items) if items else None, cfg).run(q, task)


def test_parallel_subagents_produce_same_rollout_as_serial():
    q = "I'm looking for oil pastel crayons, and a sketch pad." + VOUCHER
    serial = run(q, "voucher", EpisodeConfig(workers=1))
    parallel = run(q, "voucher", EpisodeConfig(workers=4))
    assert rollout(serial) == rollout(parallel) and serial.recommended == parallel.recommended == ["X2", "Y2"]
    steps = [s["extra_info"]["step"] for s in parallel.steps]
    assert steps == list(range(1, len(steps) + 1)) and parallel.steps[0]["prompt"]  # 编号连续，只有第一步带 prompt


def test_subagents_are_isolated_and_report_summary_only():
    res = run(SHOP_Q, "shop", EpisodeConfig())
    subs = [e for e in res.trace if e["node"] == "subagent"]
    assert [s["item"] for s in subs] == [1, 2] and all("stop_reason" in s and "tool_calls" in s for s in subs)
    # 每个子 agent 的检索事件都打了自己的标签，不会混进别的商品
    for e in res.trace:
        if e["node"] in ("retrieve", "gather_evidence") and e["type"] == "node" and e["agent"] != "main":
            assert e["agent"] == f"item-{e['item']}"


def test_parallel_subagents_actually_overlap():
    seen, lock, active = [], threading.Lock(), [0]

    class Slow(OfflineSandbox):
        def find_product(self, **params):
            with lock:
                active[0] += 1
                seen.append(active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return super().find_product(**params)

    run(SHOP_Q, "shop", EpisodeConfig(workers=4), sb=Slow.from_documents(DOCS))
    assert max(seen) >= 2


def test_guard_skips_replans_that_repeat_a_search():
    items = [ItemSpec(q="pushcart wheel", terms=[Term(key="color", value="glow in the dark")])]
    off = run("wheel", "product", NO_GUARD, items)
    on = run("wheel", "product", EpisodeConfig(workers=1), items)
    assert off.recommended == on.recommended
    assert on.stats["tool_calls"] < len([e for e in off.trace if e["type"] == "skill"])
    judge = [e for e in on.trace if e["node"] == "judge"][-1]
    assert judge["route"] == "accept" and judge["stop_reason"] in ("no_new_strategy", "max_replans")


def test_guard_off_matches_original_behaviour():
    items = [ItemSpec(q="rubber wheel pushcart", terms=[Term(key="color", value="glow in the dark")])]
    res = run("rubber wheel", "product", NO_GUARD, items)
    assert [e["strategy"] for e in res.trace if e["node"] == "replan"] == ["add_terms", "next_page"]
    assert len(res.steps) == 7  # 和改造前的 rollout 快照一致


def test_tool_budget_stops_calls_and_still_terminates():
    items = [ItemSpec(q="pushcart wheel", terms=[Term(value="glow in the dark")])]
    res = run("wheel", "product", EpisodeConfig(max_tool_calls=1, workers=1), items)
    assert res.stats["tool_calls"] == 1 and res.stats["blocked"] >= 1
    assert any(e.get("error") == "tool_budget_exhausted" for e in res.trace)
    assert res.steps[-1]["completion"]["message"]["tool_call"][0]["name"] == "terminate"


def test_invalid_tool_args_rejected_locally():
    calls = []

    class Spy(OfflineSandbox):
        def find_product(self, **params):
            calls.append(params)
            return super().find_product(**params)

    res = run("x", "product", EpisodeConfig(), [ItemSpec(q="")], sb=Spy.from_documents(DOCS))
    assert calls == [] and any("invalid args" in (e.get("error") or "") for e in res.trace)


def test_eval_mode_helpers(tmp_path):
    cases = [{"query": str(i)} for i in range(10)]
    assert [i for i, _ in select_cases(cases, 0, 50, 7)] == [7]
    assert [i for i, _ in select_cases(cases, 3, 2, None)] == [4, 5]
    cache = tmp_path / "parse.json"
    calls = []

    def fake_llm(q, t):
        calls.append(q)
        return rule_parse_task(q, t)

    p1 = CachedParser(fake_llm, cache)
    spec = p1(SHOP_Q, "shop")
    p1.save()
    p2 = CachedParser(None, cache)  # 第二次不需要模型
    assert p2(SHOP_Q, "shop") == spec and calls == [SHOP_Q] and p2.hits == 1
    res = run(SHOP_Q, "shop", EpisodeConfig(), sb=LatencySandbox(OfflineSandbox.from_documents(DOCS), 1))
    eff = efficiency([res])
    assert eff["avg_tool_calls"] == res.stats["tool_calls"] and eff["latency_ms_p50"] > 0


# ---------- 评论证据 ----------

from shopping_agent.config import load_settings  # noqa: E402
from shopping_agent.reviews import (  # noqa: E402
    aspect_evidence,
    citation_check,
    load_reviews,
    reason_lines,
    select_aspects,
    snippet_index,
)

REVIEWS = load_reviews(load_settings().reviews_path)


def test_select_aspects_priority():
    assert select_aspects("想要降噪的，送男朋友", ["降噪"], ["长续航", "降噪"]) == ["降噪", "送礼", "长续航"]
    assert select_aspects("随便看看", [], []) == []


def test_review_verdicts_supported_and_disputed():
    ok = aspect_evidence(REVIEWS["E03"], "降噪")
    assert ok["verdict"] == "supported" and all(s["rating"] >= 4 for s in ok["snippets"])
    bad = aspect_evidence(REVIEWS["E07"], "降噪")
    assert bad["verdict"] == "disputed" and bad["neg"] > bad["pos"] and bad["snippets"][0]["rating"] < 4
    assert aspect_evidence(REVIEWS["E03"], "热插拔")["verdict"] == "no_evidence"


def test_template_reasons_pass_citation_check_and_attacks_fail():
    ev = {"E03": {"aspects": [aspect_evidence(REVIEWS["E03"], "降噪")]},
          "E07": {"aspects": [aspect_evidence(REVIEWS["E07"], "降噪")]}}
    snippets = snippet_index(ev)
    text = "\n".join(reason_lines(ev["E03"]) + reason_lines(ev["E07"]))
    assert "评价有分歧" in text and citation_check(text, snippets) == []
    good_id = ev["E03"]["aspects"][0]["snippets"][0]["id"]
    kinds = lambda t: {p["type"] for p in citation_check(t, snippets)}  # noqa: E731
    assert kinds(text.replace(good_id, "R-E03-99")) == {"unknown_citation"}
    assert "unsupported_quote" in kinds(re.sub("「[^」]+」", "「全网最好」", text, count=1))
    assert kinds(re.sub(r"\[R-[^\]]+\]", "", text)) == {"missing_citation"}
    assert "polarity_mismatch" in kinds(text.replace("评价有分歧：", "买家评价：").replace("条差评", "条"))
    assert citation_check("最接近的是「华为 FreeBuds 6i」，要不要放宽预算？", snippets) == []  # 商品名不算评论引用


def test_agent_recommendation_cites_reviews_and_rejects_fabricated_rewrite():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from shopping_agent.catalog import load_catalog
    from shopping_agent.harness import build_agent

    s = load_settings()
    cat = load_catalog(s.catalog_path)

    def ask(rewriter=None, ablate=frozenset()):
        a = build_agent(cat, s, InMemoryStore(), InMemorySaver(), rewriter=rewriter, ablate=ablate)
        return a.invoke({"user_id": "u", "query": "想买个500以内的降噪蓝牙耳机，不要小米，送男朋友的"},
                        {"configurable": {"thread_id": "t"}})

    out = ask()
    assert "买家评价：" in out["response"] and "[R-E03-" in out["response"]
    assert any(e.get("skill") == "get_review_evidence" for e in out["trace"])
    fabricated = ask(rewriter=lambda t, st: t.replace("买家评价：", "买家评价：音质「同价位第一」[R-E03-77]；"))
    respond = next(e for e in fabricated["trace"] if e["node"] == "respond")
    assert respond["fact_check"]["ok"] is False and respond["fact_check"]["citation_problems"]
    assert "R-E03-77" not in fabricated["response"]  # 退回模板
    assert "买家评价" not in ask(ablate=frozenset({"reviews"}))["response"]
