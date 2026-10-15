"""优惠券计价和组合求解（plan_basket）。

计价逻辑逐条对齐官方 run_evaluate.eval_voucher：
1. 原价总和 <= 预算：直接满足；
2. 平台券，或店铺券且所有商品同店：总价 >= 门槛才可用；
   - 满减：总价 - 面额；
   - 折扣：max(总价 × (1 - 折扣率), 总价 - 封顶)，即优惠额不超过封顶；
3. 用券后 <= 预算才算满足。

组合求解：每件商品取前 k 个通过证据核验的候选，枚举组合（k=5、4 件商品也只有 625 种），
先保证属性核验分高，再保证预算满足，最后选更便宜的。
"""

from __future__ import annotations

import itertools
from typing import Any

from .task import Voucher


def price_after_voucher(prices: list[float], shop_ids: list[str], v: Voucher) -> float:
    total = sum(prices)
    applicable = v.voucher_type == "platform" or len(set(shop_ids)) == 1
    if not applicable or total < v.threshold:
        return total
    if v.discount_type == "fixed":
        return total - (v.face_value or 0)
    cap = v.cap if v.cap is not None else float("inf")
    return max(total * (1 - (v.discount or 0)), total - cap)


def within_budget(prices: list[float], shop_ids: list[str], v: Voucher) -> bool:
    return sum(prices) <= v.budget or price_after_voucher(prices, shop_ids, v) <= v.budget


def plan_basket(
    candidates: list[list[dict[str, Any]]], voucher: Voucher | None, same_shop: bool, k: int = 5
) -> dict[str, Any] | None:
    """candidates[i] 是第 i 件商品的候选（带 product_id / shop_id / price / score）。

    返回 {"items": [...], "total", "final", "budget_ok"}；没有可行组合时返回 None。
    """
    pools = [c[:k] for c in candidates]
    if not pools or any(not p for p in pools):
        return None
    best, best_key = None, None
    for combo in itertools.product(*pools):
        ids = [c["product_id"] for c in combo]
        if len(set(ids)) < len(ids):
            continue
        shops = [str(c["shop_id"]) for c in combo]
        if same_shop and len(set(shops)) != 1:
            continue
        prices = [float(c["price"]) for c in combo]
        ok = within_budget(prices, shops, voucher) if voucher else True
        final = price_after_voucher(prices, shops, voucher) if voucher else sum(prices)
        key = (sum(c["score"] for c in combo), ok, -final)
        if best_key is None or key > best_key:
            best_key = key
            best = {"items": list(combo), "total": sum(prices), "final": round(final, 2), "budget_ok": ok}
    return best
