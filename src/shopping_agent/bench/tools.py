"""ShoppingBench 沙箱工具的契约：和主链路 skill 共用 tooling.ToolSpec。

调用前统一校验参数（页码越界、空 query 这类错误在本地就拦下，不浪费一次沙箱调用），
schema 可以直接导出给 LLM 做 function calling。执行按名字分派到 Sandbox 的同名方法。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..tooling import ToolSpec
from .sandbox import MAX_PAGE


class FindProductArgs(BaseModel):
    q: str = Field(min_length=1, description="检索词")
    page: int = Field(1, ge=1, le=MAX_PAGE, description=f"页码，1–{MAX_PAGE}，每页 10 条")
    price: str | None = Field(None, description="价格区间，如 '100-300'、'375-'")
    service: str | None = Field(None, description="服务，逗号分隔：official / freeShipping / COD / flashsale")
    shop_id: str | None = Field(None, description="只在这个店铺里检索")
    sort: str | None = Field(None, description="default / order / priceasc / pricedesc")


class ViewInfoArgs(BaseModel):
    product_ids: str = Field(min_length=1, description="逗号分隔的商品 ID")


BENCH_TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in (
        ToolSpec("find_product", "在沙箱商品库里检索商品，价格、服务、店铺直接下推过滤", FindProductArgs),
        ToolSpec("view_product_information", "查看商品详情：SKU 选项、属性、描述", ViewInfoArgs),
    )
}


def canonical_params(tool: str, params: dict[str, Any]) -> dict[str, Any]:
    """校验参数并去掉空值，得到写进 rollout 和去重缓存的规范形式。参数非法时抛 ToolArgsError。"""
    spec = BENCH_TOOLS[tool]
    return spec.validate(params).model_dump(exclude_none=True)


def schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in BENCH_TOOLS.values()]
