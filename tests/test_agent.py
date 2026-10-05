from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from shopping_agent.catalog import load_catalog
from shopping_agent.config import load_settings
from shopping_agent.evidence import build_evidence
from shopping_agent.harness import build_agent
from shopping_agent.memory import update_profile
from shopping_agent.skills import REGISTRY, SkillContext, call_skill
from shopping_agent.understanding import TurnParse, merge_constraints, rule_parse

SETTINGS = load_settings()
CATALOG = load_catalog(SETTINGS.catalog_path)
CTX = SkillContext(CATALOG, SETTINGS)


def make_agent():
    return build_agent(CATALOG, SETTINGS, InMemoryStore(), InMemorySaver())


def ask(agent, text, thread="t", user="u"):
    return agent.invoke({"user_id": user, "query": text}, {"configurable": {"thread_id": thread}})


# ---------- 需求理解 ----------


def test_rule_parse_budget_and_exclude():
    p = rule_parse("想买个500以内的蓝牙耳机，不要小米")
    assert p.category == "蓝牙耳机"
    assert p.budget_max == 500
    assert p.brands_exclude == ["小米"]
    assert p.brands_include == []


def test_rule_parse_refs_and_compare():
    assert rule_parse("第二个怎么样").refs == [2]
    assert rule_parse("第二个怎么样").intent == "detail"
    p = rule_parse("第一个和第三个对比一下")
    assert p.intent == "compare" and p.refs == [1, 3]


def test_merge_resets_on_category_switch():
    prev = {"category": "蓝牙耳机", "budget_max": 500, "brands_exclude": ["小米"]}
    merged = merge_constraints(prev, TurnParse(category="跑鞋"))
    assert merged["category"] == "跑鞋"
    assert "budget_max" not in merged and merged["brands_exclude"] == []


def test_merge_include_removes_exclude():
    prev = {"category": "蓝牙耳机", "brands_exclude": ["索尼"], "brands_include": []}
    merged = merge_constraints(prev, TurnParse(brands_include=["索尼"]))
    assert merged["brands_include"] == ["索尼"] and merged["brands_exclude"] == []


# ---------- Skills 与证据 ----------


def test_skill_schema_and_validation():
    assert REGISTRY["search_products"].schema()["function"]["name"] == "search_products"
    bad = call_skill("search_products", {"limit": 5}, CTX)  # 缺 category
    assert not bad["ok"] and "invalid args" in bad["error"]
    assert not call_skill("no_such_skill", {}, CTX)["ok"]


def test_evidence_flags_conflict_stale_stock():
    assert "price_conflict" in build_evidence(CATALOG["E12"], SETTINGS)["issues"]
    assert "stale" in build_evidence(CATALOG["E14"], SETTINGS)["issues"]
    assert not build_evidence(CATALOG["E13"], SETTINGS)["in_stock"]


def test_verified_price_breaks_budget():
    r = call_skill("verify_evidence", {"product_id": "E12", "constraints": {"budget_max": 500}}, CTX)
    assert r["ok"] and r["data"]["violations"] == ["over_budget"]


# ---------- 端到端 ----------


def test_end_to_end_multi_turn():
    agent = make_agent()
    out = ask(agent, "想买个500以内的蓝牙耳机，不要小米")
    assert out["route"] == "recommend"
    assert all(CATALOG[p].brand != "小米" for p in out["shown"])
    assert "E12" not in out["shown"] and "E13" not in out["shown"]

    out = ask(agent, "第二个怎么样")
    assert out["route"] == "detail"

    out = ask(agent, "有没有更便宜的")
    assert out["constraints"]["brands_exclude"] == ["小米"]
    top_price = out["constraints"]["budget_max"]
    assert all(out["evidence"][p]["verified_price"] <= top_price for p in out["shown"])


def test_replan_then_fallback():
    agent = make_agent()
    out = ask(agent, "降噪耳机，300以内", thread="a")
    assert out["route"] == "recommend" and out["relaxed"] == ["features"]
    out = ask(agent, "200以内的机械键盘", thread="b")
    assert out["route"] == "fallback"


def test_profile_memory_across_threads():
    agent = make_agent()
    ask(agent, "耳机不要小米", thread="a", user="u1")
    out = ask(agent, "看看耳机", thread="b", user="u1")
    assert "小米" in out["constraints"]["brands_exclude"]
    out = ask(agent, "看看手机", thread="c", user="u1")
    assert out["constraints"]["brands_exclude"] == []


def test_node_exception_is_contained(monkeypatch):
    import shopping_agent.harness as h

    def boom(*a, **k):
        raise RuntimeError("search service down")

    monkeypatch.setattr(h, "call_skill", boom)
    out = ask(make_agent(), "500以内的耳机")
    errors = [e for e in out["trace"] if e.get("status") == "error"]
    assert errors and out["response"]


def test_update_profile_is_category_scoped():
    p = update_profile({}, {"category": "手机", "brands_exclude": ["小米"], "budget_max": 3000})
    assert p["brand_dislikes"] == {"手机": ["小米"]} and p["budgets"] == {"手机": 3000}
