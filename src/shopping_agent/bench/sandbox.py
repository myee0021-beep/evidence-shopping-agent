"""ShoppingBench 沙箱的两种接法。

- HttpSandbox：官方搜索服务（./init_env.sh 启动，默认 127.0.0.1:5631，背后是 250 万 Lazada 商品的
  Lucene 索引）。请求参数的预处理和官方 toolkit/find_product.py 保持一致。
- OfflineSandbox：内存版，语义相同（店铺 / 价格 / 服务过滤、10 条一页、最多 5 页），
  用 documents.jsonl 的一个子集或测试夹具即可运行，不需要 Java。

两者都会把见过的商品字段（搜索字段 + 详情字段）合并缓存在 seen 里，
本地 metrics 用它复核推荐结果，不必再查一遍 Lucene 索引。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from ..agent_memory.search import BM25

PAGE_SIZE = 10
MAX_PAGE = 5
SERVICES = ("official", "freeShipping", "COD", "flashsale")
SEARCH_FIELDS = ("product_id", "shop_id", "title", "price", "service", "sold_count")
INFO_FIELDS = ("product_id", "short_description", "description", "sku_options", "attributes")


class Sandbox:
    def __init__(self) -> None:
        self.seen: dict[str, dict[str, Any]] = {}
        self._seen_lock = threading.Lock()  # 多个子 agent 并行调用时共用 seen

    def _remember(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with self._seen_lock:
            for r in rows:
                self.seen.setdefault(str(r["product_id"]), {}).update(r)
        return rows

    def find_product(self, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    def view_product_information(self, product_ids: str) -> list[dict[str, Any]]:
        raise NotImplementedError


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """和官方 FindProduct._request 一样的预处理：去掉空值和 default。"""
    out = {"q": params["q"], "page": params.get("page", 1)}
    for k in ("shop_id", "price"):
        if params.get(k):
            out[k] = params[k]
    if params.get("sort") and params["sort"] != "default":
        out["sort"] = params["sort"]
    service = params.get("service")
    if service and service != "default":
        out["service"] = ",".join(x for x in service.split(",") if x != "default")
    return out


class HttpSandbox(Sandbox):
    def __init__(self, base_url: str = "http://127.0.0.1:5631", timeout: float = 60, retries: int = 3):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.base_url}/{path}?{query}"
        last: Exception | None = None
        for i in range(self.retries):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:  # noqa: BLE001 - 重试后仍失败再抛给 skill 层
                last = e
                time.sleep(min(2**i, 8))
        raise ConnectionError(f"sandbox {path} failed: {last}")

    def find_product(self, **params: Any) -> list[dict[str, Any]]:
        p = clean_params(params)
        p["q"] = quote_plus(p["q"])
        return self._remember(self._get("find_product", p) or [])

    def view_product_information(self, product_ids: str) -> list[dict[str, Any]]:
        return self._remember(self._get("view_product_information", {"product_ids": product_ids}) or [])


class OfflineSandbox(Sandbox):
    def __init__(self, products: list[dict[str, Any]]):
        super().__init__()
        self.products = {str(p["product_id"]): p for p in products}
        self._ids = list(self.products)
        self._bm25 = BM25([self._index_text(self.products[i]) for i in self._ids])

    @classmethod
    def from_documents(cls, path: str | Path, limit: int | None = None) -> "OfflineSandbox":
        """读 ShoppingBench 的 documents.jsonl（每行 {"id", "contents", "product"}）或纯商品 jsonl。"""
        products = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if limit is not None and len(products) >= limit:
                    break
                obj = json.loads(line)
                products.append(obj.get("product", obj))
        return cls(products)

    @staticmethod
    def _index_text(p: dict[str, Any]) -> str:
        sku = " ".join(v for kv in (p.get("sku_options") or {}).values() for v in kv.values())
        attrs = " ".join(v for vs in (p.get("attributes") or {}).values() for v in vs)
        return f"{p.get('title', '')} {sku} {attrs}"

    def find_product(self, **params: Any) -> list[dict[str, Any]]:
        p = clean_params(params)
        page = int(p["page"]) if str(p["page"]).isdigit() else 0
        if not 1 <= page <= MAX_PAGE:
            return []
        low, high = _price_range(p.get("price"))
        services = [s for s in (p.get("service") or "").split(",") if s in SERVICES]
        scores = self._bm25.scores(p["q"])
        ranked = sorted((i for i in range(len(self._ids)) if scores[i] > 0), key=lambda i: -scores[i])
        hits = []
        for i in ranked:
            prod = self.products[self._ids[i]]
            if p.get("shop_id") and str(prod.get("shop_id")) != str(p["shop_id"]):
                continue
            if low is not None and prod["price"] < low or high is not None and prod["price"] > high:
                continue
            if any(s not in prod.get("service", []) for s in services):
                continue
            hits.append(prod)
            if len(hits) >= MAX_PAGE * PAGE_SIZE:
                break
        sort = p.get("sort")
        if sort == "order":
            hits.sort(key=lambda x: -x["sold_count"])
        elif sort == "priceasc":
            hits.sort(key=lambda x: x["price"])
        elif sort == "pricedesc":
            hits.sort(key=lambda x: -x["price"])
        rows = [{k: h[k] for k in SEARCH_FIELDS} for h in hits[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]]
        return self._remember(rows)

    def view_product_information(self, product_ids: str) -> list[dict[str, Any]]:
        rows = [{k: self.products[i].get(k) for k in INFO_FIELDS} for i in product_ids.split(",") if i in self.products]
        return self._remember(rows)


def _price_range(price: str | None) -> tuple[float | None, float | None]:
    if not price or "-" not in price:
        return None, None
    parts = price.split("-")
    if len(parts) != 2:
        return None, None

    def num(x: str) -> float | None:
        try:
            return float(x)
        except ValueError:
            return None

    return num(parts[0]), num(parts[1])


class LatencySandbox(Sandbox):
    """给任意沙箱的每次调用加一段固定时延，模拟线上搜索 / 详情服务的 RPC 耗时（压测并行效果用）。"""

    def __init__(self, inner: Sandbox, latency_ms: float):
        super().__init__()
        self.inner, self.latency_ms = inner, latency_ms
        self.seen = inner.seen
        self._seen_lock = inner._seen_lock

    def find_product(self, **params: Any) -> list[dict[str, Any]]:
        time.sleep(self.latency_ms / 1000)
        return self.inner.find_product(**params)

    def view_product_information(self, product_ids: str) -> list[dict[str, Any]]:
        time.sleep(self.latency_ms / 1000)
        return self.inner.view_product_information(product_ids)
