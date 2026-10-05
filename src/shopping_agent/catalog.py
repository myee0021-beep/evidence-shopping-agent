"""模拟商品库。真实系统里这一层是搜索服务和商品详情服务。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel


class Product(BaseModel):
    id: str
    title: str
    category: str
    brand: str
    price: int  # 列表页价格，可能和详情页不一致
    detail_price: int | None = None  # 详情页价格；None 表示与列表页一致
    stock: int
    rating: float
    sales: int
    seller: str
    seller_rating: float | None = None  # None 表示该字段缺失
    features: list[str] = []
    updated_at: date

    @property
    def deep_link(self) -> str:
        return f"https://example.com/item/{self.id}"


Catalog = dict[str, Product]


def load_catalog(path: Path) -> Catalog:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["id"]: Product(**item) for item in items}
