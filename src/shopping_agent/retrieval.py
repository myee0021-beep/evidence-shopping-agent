"""商品过滤与候选排序：硬约束过滤 -> 相关度排序 -> 收缩到候选池。"""

from __future__ import annotations

import math
from typing import Any

from .catalog import Catalog, Product


def passes_hard_filter(p: Product, c: dict[str, Any]) -> bool:
    """检索阶段的硬过滤，用列表页价格。详情页价格在证据层再校验一次。"""
    if c.get("category") and p.category != c["category"]:
        return False
    if c.get("budget_max") is not None and p.price > c["budget_max"]:
        return False
    if c.get("budget_min") is not None and p.price < c["budget_min"]:
        return False
    if p.brand in c.get("brands_exclude", []):
        return False
    include = c.get("brands_include") or []
    if include and p.brand not in include:
        return False
    required = c.get("features") or []
    if any(f not in p.features for f in required):
        return False
    return True


def score(p: Product, c: dict[str, Any]) -> float:
    """相关度打分：评分为主，属性命中和销量为辅。"""
    wanted = c.get("features") or []
    hits = sum(1 for f in wanted if f in p.features)
    return p.rating + 0.3 * hits + 0.15 * math.log10(p.sales + 1)


def search(catalog: Catalog, c: dict[str, Any], limit: int) -> list[str]:
    pool = [p for p in catalog.values() if passes_hard_filter(p, c)]
    pool.sort(key=lambda p: (-score(p, c), p.price, p.id))
    return [p.id for p in pool[:limit]]
