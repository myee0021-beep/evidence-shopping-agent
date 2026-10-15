"""Skills / Tools：把电商能力封装成可组合、可校验、可观测的工具。

每个 skill 都有：
- 名字和描述（可以直接转成 LLM 的 function calling schema）；
- 一个 pydantic 参数模型，调用前统一校验；
- 统一的返回格式 {"ok", "data" | "error", "latency_ms", "attempts"}，失败不抛异常，
  而是把错误交回给编排层决定 replan 还是兜底；
- 下游调用的超时和重试：只读 skill（idempotent=True）超时或异常时重试，参数错误不重试。

skill 本身是薄封装，真正的能力在下游服务（搜索、详情、价格、库存、店铺……），见 docs/skills.md。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from . import retrieval
from .catalog import Catalog
from .config import Settings
from .evidence import build_evidence, hard_violations


@dataclass
class SkillContext:
    catalog: Catalog
    settings: Settings


@dataclass
class Skill:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel, SkillContext], Any]
    idempotent: bool = True  # 只读 skill 可以安全重试；加购、下单这类写操作不能

    def schema(self) -> dict[str, Any]:
        """导出成 OpenAI 兼容的 function calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


REGISTRY: dict[str, Skill] = {}


def skill(name: str, description: str, args_model: type[BaseModel]):
    def deco(fn):
        REGISTRY[name] = Skill(name, description, args_model, fn)
        return fn

    return deco


_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="skill")


def call_skill(name: str, args: dict[str, Any], ctx: SkillContext) -> dict[str, Any]:
    start = time.perf_counter()
    result: dict[str, Any]
    attempts = 0
    if name not in REGISTRY:
        result = {"ok": False, "error": f"unknown skill: {name}"}
    else:
        s = REGISTRY[name]
        try:
            parsed = s.args_model(**args)
        except ValidationError as e:  # 参数错误重试也没用，直接返回
            parsed, result = None, {"ok": False, "error": f"invalid args: {e.errors()[0]['msg']}"}
        if parsed is not None:
            retries = ctx.settings.skill_retries if s.idempotent else 0
            timeout = ctx.settings.skill_timeout_ms / 1000
            for attempts in range(1, retries + 2):
                try:
                    data = _POOL.submit(s.fn, parsed, ctx).result(timeout=timeout)
                    result = {"ok": True, "data": data}
                    break
                except FutureTimeout:
                    result = {"ok": False, "error": f"timeout after {ctx.settings.skill_timeout_ms}ms"}
                except KeyError as e:  # 业务错误（商品不存在），重试也没用
                    result = {"ok": False, "error": f"KeyError: {e}"}
                    break
                except Exception as e:  # noqa: BLE001 - 工具失败交给编排层处理
                    result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    result["attempts"] = attempts
    return result


# ---------- 具体的 skills ----------


class SearchArgs(BaseModel):
    category: str
    budget_min: int | None = None
    budget_max: int | None = None
    brands_include: list[str] = []
    brands_exclude: list[str] = []
    features: list[str] = []
    limit: int = Field(5, ge=1, le=20)


@skill("search_products", "按品类、预算、品牌和属性检索商品，返回排序后的商品 ID", SearchArgs)
def search_products(a: SearchArgs, ctx: SkillContext) -> list[str]:
    return retrieval.search(ctx.catalog, a.model_dump(exclude={"limit"}), a.limit)


class FilterArgs(BaseModel):
    product_ids: list[str]
    features: list[str]


@skill("filter_by_attributes", "在给定商品中筛出具备全部指定属性的商品", FilterArgs)
def filter_by_attributes(a: FilterArgs, ctx: SkillContext) -> list[str]:
    return [
        pid for pid in a.product_ids if all(f in ctx.catalog[pid].features for f in a.features)
    ]


class DetailArgs(BaseModel):
    product_id: str


@skill("get_product_detail", "获取商品详情页信息（价格、库存、商家、属性、更新时间）", DetailArgs)
def get_product_detail(a: DetailArgs, ctx: SkillContext) -> dict[str, Any]:
    if ctx.settings.detail_latency_ms:
        time.sleep(ctx.settings.detail_latency_ms / 1000)  # 模拟详情服务的 RPC 时延，压测并行效果用
    if a.product_id not in ctx.catalog:
        raise KeyError(f"product {a.product_id} not found")
    return ctx.catalog[a.product_id].model_dump(mode="json")


class VerifyArgs(BaseModel):
    product_id: str
    constraints: dict[str, Any] = {}


@skill("verify_evidence", "聚合并校验商品证据，复核硬约束，给出置信度", VerifyArgs)
def verify_evidence(a: VerifyArgs, ctx: SkillContext) -> dict[str, Any]:
    p = ctx.catalog[a.product_id]
    ev = build_evidence(p, ctx.settings)
    ev["violations"] = hard_violations(p, ev, a.constraints)
    ev["usable"] = not ev["violations"]
    return ev


class CompareArgs(BaseModel):
    product_ids: list[str] = Field(min_length=2)


@skill("compare_products", "对比多个商品的价格、评分、商家和属性", CompareArgs)
def compare_products(a: CompareArgs, ctx: SkillContext) -> dict[str, Any]:
    items = []
    for pid in a.product_ids:
        p = ctx.catalog[pid]
        ev = build_evidence(p, ctx.settings)
        items.append(
            {
                "id": pid,
                "title": p.title,
                "price": ev["verified_price"],
                "rating": p.rating,
                "seller_rating": p.seller_rating,
                "features": p.features,
                "in_stock": ev["in_stock"],
                "issues": ev["issues"],
            }
        )
    cheapest = min(items, key=lambda x: x["price"])["id"]
    best_rated = max(items, key=lambda x: x["rating"])["id"]
    return {"items": items, "cheapest": cheapest, "best_rated": best_rated}
