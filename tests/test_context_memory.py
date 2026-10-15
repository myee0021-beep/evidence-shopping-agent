from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from shopping_agent.agent_memory import LayeredMemory
from shopping_agent.agent_memory.search import hybrid_search, rrf, tokenize
from shopping_agent.archive import ArchiveStore, Caller, recall_context
from shopping_agent.catalog import load_catalog
from shopping_agent.config import load_settings
from shopping_agent.context import (
    ContextOverflowError,
    build_window,
    compact,
    count_message_tokens,
    count_tokens,
    safe_cut_index,
    unpaired_tool_messages,
)
from shopping_agent.harness import build_agent

SETTINGS = load_settings()
CATALOG = load_catalog(SETTINGS.catalog_path)
ME = Caller("u1", "t1")


def turn(i, tool_chars=0, parallel=False):
    """一轮：用户 -> 助手发起工具调用 -> 工具结果 -> 助手回复。"""
    msgs = [{"role": "user", "content": f"第{i}轮：预算{i * 100}元以内的耳机"}]
    if tool_chars:
        ids = [f"c{i}a", f"c{i}b"] if parallel else [f"c{i}a"]
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": x, "function": {"name": "search"}} for x in ids]})
        msgs += [{"role": "tool", "tool_call_id": x, "content": "详情" * tool_chars} for x in ids]
    msgs.append({"role": "assistant", "content": f"第{i}轮推荐了 E0{i % 9 + 1}"})
    return msgs


def history(n, **kw):
    return [m for i in range(1, n + 1) for m in turn(i, **kw)]


# ---------- token 计数与安全切分 ----------


def test_token_count_includes_tool_calls():
    plain = {"role": "assistant", "content": ""}
    with_call = {**plain, "tool_calls": [{"id": "x", "function": {"name": "search_products", "arguments": "{}"}}]}
    assert count_message_tokens(with_call) > count_message_tokens(plain)


def test_safe_cut_never_splits_tool_pairs():
    msgs = history(6, tool_chars=50, parallel=True)
    for keep in range(0, count_tokens(msgs), 37):
        cut = safe_cut_index(msgs, keep)
        if cut:
            assert msgs[cut]["role"] == "user"
            assert unpaired_tool_messages(msgs[cut:]) == []


def test_safe_cut_keeps_last_turn_even_if_huge():
    msgs = history(2) + turn(3, tool_chars=3000)
    cut = safe_cut_index(msgs, keep_tokens=10)
    assert msgs[cut]["content"].startswith("第3轮")


def test_single_turn_cannot_be_cut():
    assert safe_cut_index(turn(1, tool_chars=10), 0) == 0


# ---------- 压缩主流程 ----------


def test_noop_under_budget():
    archive = ArchiveStore(InMemoryStore())
    res = compact(history(2), budget=10_000, pinned_tokens=50, archive=archive, caller=ME)
    assert res.status == "noop" and not res.changed


def test_compact_archives_then_summarizes_and_recalls():
    archive = ArchiveStore(InMemoryStore())
    msgs = history(12)
    res = compact(msgs, budget=300, pinned_tokens=50, archive=archive, caller=ME)
    assert res.status == "compacted" and res.tokens_after < res.tokens_before
    assert res.tokens_after <= 300 * 0.4 + 300 * 0.2 + 20
    summary = res.messages[0]
    assert summary["name"] == "history_summary" and summary["refs"]
    # 数字按原文保留在摘要里；最早的摘要行超出上限被省略，但原文仍可取回
    assert "预算900元以内" in summary["content"] and "预算100元以内" not in summary["content"]
    got = recall_context(archive, ME, summary["refs"][0], mode="grep", pattern="第1轮")
    assert got["ok"] and "预算100元" in got["text"]


def test_incremental_summary_accumulates_refs():
    archive = ArchiveStore(InMemoryStore())
    msgs = history(12)
    first = compact(msgs, budget=300, pinned_tokens=50, archive=archive, caller=ME)
    more = first.messages + history(20)[len(history(12)) :]
    second = compact(more, budget=300, pinned_tokens=50, archive=archive, caller=ME)
    assert len(second.messages[0]["refs"]) == 2
    assert second.messages[0]["turns"] > first.messages[0]["turns"]


def test_large_tool_result_is_offloaded_first():
    archive = ArchiveStore(InMemoryStore())
    msgs = turn(1, tool_chars=2500)
    res = compact(msgs, budget=1500, pinned_tokens=0, archive=archive, caller=ME)
    tool = next(m for m in res.messages if m["role"] == "tool")
    assert tool["offloaded"] and len(tool["content"]) < 600
    assert recall_context(archive, ME, tool["offloaded"], mode="range", length=10_000)["next_start"] == 4000


def test_summary_failure_marks_gap_and_keeps_ref():
    def broken(prev, head):
        raise TimeoutError("llm timeout")

    archive = ArchiveStore(InMemoryStore())
    res = compact(history(12), budget=300, pinned_tokens=50, archive=archive, caller=ME, summarizer=broken)
    assert res.status == "summary_failed"
    assert "摘要生成失败" in res.messages[0]["content"] and res.messages[0]["refs"][0] in res.messages[0]["content"]


def test_archive_failure_stops_trimming():
    class Down(ArchiveStore):
        def put(self, *a, **k):
            raise ConnectionError("store down")

    msgs = history(12)
    res = compact(msgs, budget=300, pinned_tokens=50, archive=Down(InMemoryStore()), caller=ME)
    assert res.status == "archive_failed" and res.messages == msgs
    with pytest.raises(ContextOverflowError):
        build_window([], res.messages, budget=300)


# ---------- recall 权限 ----------


def test_recall_permissions():
    archive = ArchiveStore(InMemoryStore())
    ref = archive.put(ME, "history", "secret")
    assert recall_context(archive, ME, ref)["ok"]
    assert not recall_context(archive, Caller("u1", "other_thread"), ref)["ok"]
    assert not recall_context(archive, Caller("u2", "t1"), ref)["ok"]
    assert not recall_context(archive, Caller("u1", "t1", tenant="t2"), ref)["ok"]
    shared = archive.put(ME, "utterance", "不要小米", visibility="user")
    assert recall_context(archive, Caller("u1", "other_thread"), shared)["ok"]


# ---------- 检索 ----------


def test_rrf_and_hybrid_search():
    assert rrf([[0, 1], [1, 2]])[0][0] == 1
    assert "小米" in tokenize("不要小米")
    docs = ["蓝牙耳机：不要小米", "跑鞋：想要缓震", "手机：预算 3000 元以内"]
    assert hybrid_search("小米耳机", docs)[0][0] == 0
    assert hybrid_search("缓震跑鞋", docs)[0][0] == 1


# ---------- 分层记忆 ----------


def diff_of(c):
    return {"added": c, "changed": {}, "removed": []}


def test_one_off_fact_stays_in_session():
    mem = LayeredMemory(InMemoryStore())
    c = {"category": "跑鞋", "brands_exclude": ["耐克"]}
    res = mem.record_turn(Caller("u", "a"), "这次给我爸买跑鞋，不要耐克", diff_of(c), c, 1)
    assert res["facts"][0]["scope"] == "session"
    assert res["profile"]["brand_dislikes"] == {}
    assert mem.facts(Caller("u", "b")) == []  # 新会话看不到


def test_confirmation_threshold_and_supersede():
    mem = LayeredMemory(InMemoryStore())
    c = {"category": "蓝牙耳机", "budget_max": 500}
    mem.record_turn(Caller("u", "a"), "500以内的耳机", diff_of(c), c, 1)
    assert mem.profile(Caller("u", "a"))["budgets"] == {}  # 只说过一次
    mem.record_turn(Caller("u", "b"), "耳机还是500以内", diff_of(c), c, 1)
    assert mem.profile(Caller("u", "b"))["budgets"] == {"蓝牙耳机": 500}

    c2 = {"category": "蓝牙耳机", "budget_max": 800}
    res = mem.record_turn(Caller("u", "c"), "耳机预算800", diff_of(c2), c2, 1)
    assert res["facts"][0]["op"] == "supersede"
    assert mem.profile(Caller("u", "c"))["budgets"] == {}  # 新值还没确认够
    chain = mem.trace(Caller("u", "c"), res["facts"][0]["id"])["chain"]
    assert [x["status"] for x in chain] == ["active", "superseded"]
    assert chain[1]["sources"][0]["text"] == "500以内的耳机"  # 追溯到原话


def test_include_overrides_dislike():
    mem = LayeredMemory(InMemoryStore())
    c = {"category": "蓝牙耳机", "brands_exclude": ["小米"]}
    mem.record_turn(ME, "耳机不要小米", diff_of(c), c, 1)
    assert mem.profile(ME)["brand_dislikes"] == {"蓝牙耳机": ["小米"]}
    c2 = {"category": "蓝牙耳机", "brands_include": ["小米"]}
    mem.record_turn(ME, "还是看看小米吧", diff_of(c2), c2, 2)
    assert mem.profile(ME)["brand_dislikes"] == {}


# ---------- 端到端 ----------


def ask(agent, text, thread="t", user="u"):
    return agent.invoke({"user_id": user, "query": text}, {"configurable": {"thread_id": thread}})


def test_long_dialogue_compacts_but_keeps_constraints():
    settings = replace(SETTINGS, context_budget_tokens=800)
    agent = build_agent(CATALOG, settings, InMemoryStore(), InMemorySaver())
    out = ask(agent, "想买个500以内的蓝牙耳机，不要小米")
    statuses = []
    for _ in range(8):
        out = ask(agent, "第一个怎么样")
        out = ask(agent, "第二个和第三个对比一下")
        statuses.append(out["context"]["status"])
        assert count_tokens(out["messages"]) <= settings.context_budget_tokens
    assert "compacted" in statuses
    assert out["messages"][0]["name"] == "history_summary"
    # 第 1 轮说的约束还在结构化状态里
    assert out["constraints"]["budget_max"] == 500 and out["constraints"]["brands_exclude"] == ["小米"]
    # 指代按快照解析
    assert [it["id"] for it in out["shown_snapshot"]["items"]] == out["shown"]


def test_one_off_request_does_not_leak_across_sessions():
    agent = build_agent(CATALOG, SETTINGS, InMemoryStore(), InMemorySaver())
    ask(agent, "这次给我爸买耳机，不要小米", thread="a", user="u9")
    out = ask(agent, "看看耳机", thread="b", user="u9")
    assert out["constraints"]["brands_exclude"] == []


def test_constraint_diff_in_trace():
    agent = build_agent(CATALOG, SETTINGS, InMemoryStore(), InMemorySaver())
    ask(agent, "500以内的蓝牙耳机")
    out = ask(agent, "不要小米")
    assert out["constraint_diff"]["added"] == {"brands_exclude": ["小米"]}
