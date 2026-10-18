import json

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
