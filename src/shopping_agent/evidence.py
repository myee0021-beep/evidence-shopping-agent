"""证据层：对价格、库存、商家、属性和信息时效做统一聚合与校验。

决策节点只看证据，不看模型“觉得”：
- 详情页价格是准的，列表页价格只用于粗筛；
- 缺货、按详情页价格超预算的商品不可推荐；
- 证据缺失、冲突、过期会降低置信度。
"""

from __future__ import annotations

from typing import Any

from .catalog import Product
from .config import Settings

ISSUE_TEXT = {
    "price_conflict": "列表页和详情页价格不一致，已按详情页价格计算",
    "out_of_stock": "当前缺货",
    "stale": "商品信息超过 {days} 天未更新",
    "missing_seller_rating": "缺少商家评分",
}


def build_evidence(p: Product, settings: Settings) -> dict[str, Any]:
    verified_price = p.detail_price if p.detail_price is not None else p.price
    issues: list[str] = []
    if p.detail_price is not None and p.detail_price != p.price:
        issues.append("price_conflict")
    if p.stock <= 0:
        issues.append("out_of_stock")
    age_days = (settings.snapshot_date - p.updated_at).days
    if age_days > settings.stale_days:
        issues.append("stale")
    if p.seller_rating is None:
        issues.append("missing_seller_rating")

    fields = [verified_price, p.stock, p.seller_rating, p.features, p.updated_at]
    coverage = sum(v is not None for v in fields) / len(fields)

    confidence = coverage
    if "price_conflict" in issues:
        confidence -= 0.3
    if "stale" in issues:
        confidence -= 0.3
    return {
        "product_id": p.id,
        "verified_price": verified_price,
        "list_price": p.price,
        "in_stock": p.stock > 0,
        "seller": p.seller,
        "seller_rating": p.seller_rating,
        "features": list(p.features),
        "age_days": age_days,
        "coverage": round(coverage, 2),
        "issues": issues,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
    }


def hard_violations(p: Product, ev: dict[str, Any], c: dict[str, Any]) -> list[str]:
    """用证据（详情页价格、库存）复核硬约束。"""
    v: list[str] = []
    if c.get("category") and p.category != c["category"]:
        v.append("category")
    if c.get("budget_max") is not None and ev["verified_price"] > c["budget_max"]:
        v.append("over_budget")
    if c.get("budget_min") is not None and ev["verified_price"] < c["budget_min"]:
        v.append("under_budget")
    if p.brand in c.get("brands_exclude", []):
        v.append("excluded_brand")
    include = c.get("brands_include") or []
    if include and p.brand not in include:
        v.append("brand_not_included")
    if not ev["in_stock"]:
        v.append("out_of_stock")
    return v


def describe_issues(ev: dict[str, Any]) -> list[str]:
    return [ISSUE_TEXT[i].format(days=ev["age_days"]) for i in ev["issues"] if i != "out_of_stock"]
