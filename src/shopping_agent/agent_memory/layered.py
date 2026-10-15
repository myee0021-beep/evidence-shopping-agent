"""L0–L3 分层长期记忆（思路参考 TencentDB Agent Memory 的语义金字塔，按购物场景改造）。

    L3 用户画像   按品类：不喜欢的品牌、预算、想要的属性        <- 从 L1 规则聚合，带 provenance
    L2 场景块     每个品类一块 Markdown，人可读，可直接拼进 prompt  <- 从 L1 规则派生
    L1 原子事实   “蓝牙耳机：不要小米”，带 source_ref / scope / 确认次数 / 覆盖链
    L0 原始对话   用户原话归档（archive.py），任何一条记忆都能追溯回来

和原方案的取舍：
- L1 -> L2 -> L3 都用规则聚合，不调 LLM：写入零成本、结果确定、可测试，也就没有“画像漂移”；
- 不做不可逆压缩：L3 每个字段都记着来自哪些 L1 事实，L1 记着来自哪条 L0 原话；
- 不做 Mermaid 任务画布：购物链路是固定的几步，结构化 AgentState 已经承担了“导航”的作用。

存储全部走 LangGraph Store，本地用内存，生产可以换成 Redis / Postgres Store。
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from langgraph.store.base import BaseStore

from ..archive import ArchiveStore, Caller
from ..memory import PROFILE_NS
from ..privacy import redact_pii
from .facts import CONFIRM_THRESHOLD, FactDraft, extract_facts
from .search import Embedder, hashing_embed, hybrid_search

class LayeredMemory:
    def __init__(self, store: BaseStore, embed: Embedder = hashing_embed):
        self.store = store
        self.archive = ArchiveStore(store)
        self.embed = embed

    # ---------- 存储 ----------

    @staticmethod
    def _facts_ns(caller: Caller) -> tuple[str, ...]:
        return ("memory", caller.tenant, caller.user_id, "facts")

    def facts(self, caller: Caller, include_inactive: bool = False) -> list[dict[str, Any]]:
        """当前可见的 L1 事实：长期事实 + 本会话的一次性事实。"""
        items = [i.value for i in self.store.search(self._facts_ns(caller), limit=10_000)]
        out = []
        for f in items:
            if f["scope"] == "session" and f["thread_id"] != caller.thread_id:
                continue
            if not include_inactive and f["status"] != "active":
                continue
            out.append(dict(f))
        return sorted(out, key=lambda f: f["created_at"])

    def _put_fact(self, caller: Caller, fact: dict[str, Any]) -> None:
        self.store.put(self._facts_ns(caller), fact["id"], fact)

    # ---------- 写入 ----------

    def record_turn(
        self, caller: Caller, query: str, diff: dict[str, Any], constraints: dict[str, Any], turn: int
    ) -> dict[str, Any]:
        """一轮结束后：原话进 L0，抽事实进 L1，重算 L2 / L3。"""
        drafts = extract_facts(query, diff, constraints)
        if not drafts:
            return {"source_ref": None, "facts": [], "profile": self.profile(caller)}
        # 跨会话可读的原话归档一律先脱敏：手机号、邮箱、证件号、卡号不进长期记忆
        clean, pii = redact_pii(query)
        source_ref = self.archive.put(caller, "utterance", clean, visibility="user", meta={"turn": turn, "pii": pii})
        written = [self._upsert(caller, d, source_ref) for d in drafts]
        profile = self.profile(caller)
        # 写到 memory.load_profile 读的位置，harness 预填约束时直接读 L3
        self.store.put(PROFILE_NS, caller.user_id, profile)
        return {"source_ref": source_ref, "facts": written, "profile": profile}

    def _upsert(self, caller: Caller, d: FactDraft, source_ref: str) -> dict[str, Any]:
        """同一个 key：值相同累加确认，值不同则新事实覆盖旧事实（旧的保留，标记 superseded）。"""
        current = next(
            (f for f in self.facts(caller) if f["key"] == d.key and f["scope"] == d.scope),
            None,
        )
        now = time.time()
        if current and current["value"] == d.value:
            current["confirmations"] += 1
            current["source_refs"].append(source_ref)
            current["updated_at"] = now
            self._put_fact(caller, current)
            return {"op": "confirm", **current}
        fact = {
            "id": f"f_{uuid.uuid4().hex[:10]}",
            "kind": d.kind,
            "key": d.key,
            "value": d.value,
            "text": d.text,
            "category": d.category,
            "scope": d.scope,
            "thread_id": caller.thread_id,
            "source_refs": [source_ref],
            "confirmations": 1,
            "status": "active",
            "supersedes": current["id"] if current else None,
            "created_at": now,
            "updated_at": now,
        }
        self._put_fact(caller, fact)
        if current:
            current["status"] = "superseded"
            current["superseded_by"] = fact["id"]
            self._put_fact(caller, current)
        return {"op": "supersede" if current else "add", **fact}

    # ---------- L2 / L3 ----------

    def scenes(self, caller: Caller) -> dict[str, str]:
        """L2：每个品类一块 Markdown。"""
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for f in self.facts(caller):
            by_cat.setdefault(f["category"], []).append(f)
        out = {}
        for cat, fs in by_cat.items():
            lines = [f"## {cat}"]
            for f in fs:
                if f["kind"] == "brand_dislike" and f["value"] is False:
                    continue
                tag = "本次会话" if f["scope"] == "session" else f"确认 {f['confirmations']} 次"
                lines.append(f"- {f['text']}（{tag}）")
            out[cat] = "\n".join(lines)
        return out

    def profile(self, caller: Caller) -> dict[str, Any]:
        """L3：只聚合长期事实，并且确认次数达到阈值。每个字段都记录来源事实。"""
        p: dict[str, Any] = {"brand_dislikes": {}, "budgets": {}, "liked_features": {}, "provenance": {}}
        for f in self.facts(caller):
            if f["scope"] != "long_term" or f["confirmations"] < CONFIRM_THRESHOLD[f["kind"]]:
                continue
            cat, kind = f["category"], f["kind"]
            if kind == "brand_dislike" and f["value"] is True:
                brand = f["key"].rsplit("|", 1)[1]
                p["brand_dislikes"].setdefault(cat, []).append(brand)
                p["provenance"][f"brand_dislikes.{cat}.{brand}"] = f["id"]
            elif kind == "budget_max":
                p["budgets"][cat] = f["value"]
                p["provenance"][f"budgets.{cat}"] = f["id"]
            elif kind == "feature":
                feat = f["key"].rsplit("|", 1)[1]
                p["liked_features"].setdefault(cat, []).append(feat)
                p["provenance"][f"liked_features.{cat}.{feat}"] = f["id"]
        for d in (p["brand_dislikes"], p["liked_features"]):
            for k in d:
                d[k] = sorted(d[k])
        return p

    # ---------- 召回 ----------

    def search(self, caller: Caller, query: str, k: int = 5) -> list[dict[str, Any]]:
        """search_memory 工具：L1 事实和 L2 场景一起做 BM25 + 向量检索，RRF 融合。"""
        facts = self.facts(caller)
        scenes = self.scenes(caller)
        docs = [f["text"] for f in facts] + list(scenes.values())
        hits = []
        for idx, score in hybrid_search(query, docs, k=k, embed=self.embed):
            if idx < len(facts):
                f = facts[idx]
                hits.append({"layer": "L1", "id": f["id"], "text": f["text"], "scope": f["scope"], "score": round(score, 4)})
            else:
                cat = list(scenes)[idx - len(facts)]
                hits.append({"layer": "L2", "id": f"scene:{cat}", "text": scenes[cat], "score": round(score, 4)})
        return hits

    def trace(self, caller: Caller, fact_id: str) -> dict[str, Any]:
        """trace_memory 工具：事实 -> 覆盖链 -> L0 原话。"""
        all_facts = {f["id"]: f for f in self.facts(caller, include_inactive=True)}
        if fact_id not in all_facts:
            return {"ok": False, "error": f"fact {fact_id} not found"}
        chain = []
        cur: dict[str, Any] | None = all_facts[fact_id]
        while cur is not None:
            sources = []
            for ref in cur["source_refs"]:
                try:
                    sources.append({"ref": ref, "text": self.archive.get(caller, ref)["text"]})
                except PermissionError:
                    sources.append({"ref": ref, "text": None})
            chain.append({"id": cur["id"], "text": cur["text"], "status": cur["status"], "sources": sources})
            cur = all_facts.get(cur.get("supersedes") or "")
        return {"ok": True, "chain": chain}

    def prefetch(self, caller: Caller, query: str, k: int = 3) -> dict[str, Any]:
        """请求前主动预取：L3 画像 + 和这句话相关的 L1 / L2，拼进上下文。"""
        return {"profile": self.profile(caller), "hits": self.search(caller, query, k)}


MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "检索用户的历史偏好和事实（BM25 + 向量，RRF 融合）",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 5}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trace_memory",
            "description": "追溯一条记忆的来源：覆盖链和用户原话",
            "parameters": {
                "type": "object",
                "properties": {"fact_id": {"type": "string"}},
                "required": ["fact_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_context",
            "description": "取回被压缩归档的对话或工具结果原文，每次最多 4000 字符",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref_id": {"type": "string"},
                    "mode": {"type": "string", "enum": ["grep", "tail", "range"]},
                    "pattern": {"type": "string"},
                    "start": {"type": "integer"},
                    "length": {"type": "integer"},
                },
                "required": ["ref_id"],
            },
        },
    },
]
