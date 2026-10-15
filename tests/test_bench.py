import json
import re
from pathlib import Path

from shopping_agent.bench.agent import BenchAgent, verify
from shopping_agent.bench.metrics import jaccard_sim, rule_score, score_case
from shopping_agent.bench.sandbox import OfflineSandbox, clean_params
from shopping_agent.bench.task import ItemSpec, TaskSpec, Term, parse_voucher, rule_parse_task
from shopping_agent.bench.voucher import price_after_voucher, within_budget
from shopping_agent.trace import replay

DOCS = Path(__file__).parent / "fixtures" / "bench_documents.jsonl"
VOUCHER = (
    "\n\nMy budget is only `380`, but I have a voucher with the following rules:\n"
    "1. The voucher only applies to the products from the same shop.\n"
    "2. It is valid only when the total price of the products exceeds `400`.\n"
    "3. It provides a fixed discount of `100`."
)


def sandbox():
    return OfflineSandbox.from_documents(DOCS)


def stub(items):
    """离线测试用的“LLM 解析器”：固定返回给定的 items。"""
    return lambda query, task: rule_parse_task(query, task).model_copy(update={"items": items})


# ---------- 官方 rollout 格式（复刻 format_reward / extract_recommed_product / length_reward 的检查） ----------


def format_ok(content):
    for role in ("think", "tool_call"):
        if len(re.findall(f"<{role}>", content)) != 1 or len(re.findall(f"</{role}>", content)) != 1:
            return False
    calls = json.loads(content.split("<tool_call>")[1].split("</tool_call>")[0])
    return isinstance(calls, list) and all(isinstance(c["parameters"], dict) for c in calls)


def recommended_from(steps):
    ids = ""
    for st in steps:
        for c in st["completion"]["message"].get("tool_call", []):
            if c["name"] == "recommend_product":
                ids = c["parameters"]["product_ids"]
    return ids


def test_product_task_end_to_end_and_rollout_format():
    q = "Looking for a 12 inch heavy duty rubber wheel for a pushcart with cash on delivery."
    res = BenchAgent(sandbox()).run(q, "product")
    assert res.spec.items[0].services == ["COD"]
    assert res.recommended == ["W1"]
    assert all(format_ok(s["completion"]["content"]) for s in res.steps)
    assert all(s["extra_info"]["query"] == q for s in res.steps)
    assert recommended_from(res.steps) == "W1"
    assert res.steps[-1]["completion"]["message"]["tool_call"][0]["name"] == "terminate"
    # trace 和主链路同一格式，可以直接回放
    assert "search_products" in replay(res.trace)


def test_replan_switches_strategy_without_relaxing():
    items = [ItemSpec(q="rubber wheel pushcart", terms=[Term(key="color", value="glow in the dark")])]
    res = BenchAgent(sandbox(), stub(items)).run("rubber wheel", "product")
    strategies = [e["strategy"] for e in res.trace if e["node"] == "replan"]
    assert strategies == ["add_terms", "next_page"]
    assert res.recommended[0] in ("W1", "W2")  # 仍然推荐最接近的，缺失项记在 trace 里


def test_verify_attribute_levels():
    p = {"product_id": "W1", "price": 350, "service": ["COD"], "title": "Rubber Wheel",
         "sku_options": {"1": {"size": "12 inch"}}, "attributes": {"material": ["rubber"]}}
    item = ItemSpec(q="wheel", price_max=300, services=["COD"],
                    terms=[Term(key="size", value="12 inch"), Term(value="wheel"), Term(value="steel")])
    v = verify(p, item)
    assert v["matched"] == ["12 inch"] and v["weak"] == ["wheel"] and v["missing"] == ["steel"]
    assert v["violations"] == ["price_above_max"] and not v["usable"]


def test_voucher_basket_prefers_same_shop_when_voucher_needed():
    q = "I'm looking for oil pastel crayons, and a sketch pad." + VOUCHER
    res = BenchAgent(sandbox()).run(q, "voucher")
    assert res.spec.voucher.voucher_type == "shop" and len(res.spec.items) == 2
    assert res.recommended == ["X2", "Y2"]
    assert res.basket["budget_ok"] and res.basket["final"] == 300


def test_shop_task_returns_single_shop():
    q = "Find a shop that offers these items: (1) oil pastel crayons; (2) a sketch pad drawing paper."
    res = BenchAgent(sandbox()).run(q, "shop")
    shops = {res.products[p]["shop_id"] for p in res.recommended}
    assert len(res.recommended) == 2 and len(shops) == 1


def test_sandbox_failure_is_contained():
    class Down(OfflineSandbox):
        def find_product(self, **params):
            raise ConnectionError("search engine down")

    res = BenchAgent(Down([])).run("rubber wheel", "product")
    assert any(e["type"] == "skill" and not e["ok"] for e in res.trace)
    last = res.steps[-1]["completion"]["message"]["tool_call"][0]
    assert last == {**last, "name": "terminate", "parameters": {"status": "failure"}}


# ---------- 计价、参数、评分与官方一致 ----------


def test_voucher_math_matches_official_evaluator():
    v = parse_voucher(
        "My budget is only `1807`, but ... same shop ... exceeds `1563`.\n"
        "3. It provides a percentage discount of `14%` with a cap of `1465`."
    )
    assert v.voucher_type == "shop" and v.discount == 0.14 and v.cap == 1465
    assert price_after_voucher([1600.0], ["s"], v) == 1600 * 0.86
    assert price_after_voucher([1000.0, 700.0], ["a", "b"], v) == 1700  # 跨店不能用店铺券
    assert within_budget([1000.0, 700.0], ["a", "b"], v)  # 但原价已在预算内，照样满足


def test_clean_params_matches_official_preprocessing():
    p = clean_params({"q": "x", "page": 1, "sort": "default", "service": "default,COD", "price": ""})
    assert p == {"q": "x", "page": 1, "service": "COD"}


def test_rule_parse_prices_and_services():
    it = rule_parse_task("Show me jeans priced between 100 and 300 PHP with free shipping and LazMall.", "product").items[0]
    assert (it.price_min, it.price_max) == (100, 300) and sorted(it.services) == ["freeShipping", "official"]
    it = rule_parse_task("Show me single-pack bags of chili powder priced above 375 pesos.", "product").items[0]
    assert (it.price_min, it.price_max) == (375, None)


def test_rule_score_and_case_score():
    prod = {"product_id": "X", "shop_id": "A", "title": "Oil Pastel Crayons", "price": 300, "service": ["official"],
            "sku_options": {"1": {"size": "l"}}, "attributes": {"type_of_inks": ["oil-based"]}}
    reward = {"product_id": "other", "title": ["Oil Pastel Crayons"], "price": [{"less than": [None, 350]}],
              "service": ["official"], "sku_options": [{"size": "l"}], "attributes": [{"type_of_inks": ["oil-based"]}]}
    r, total, hit = rule_score(prod, reward, jaccard_sim)
    assert r == 1.0 and total["sku & attrs"] == 2
    s = score_case("product", {"reward": reward}, ["X"], {"X": prod}, jaccard_sim)
    assert s["success"] == 1 and s["gt"] == 0
