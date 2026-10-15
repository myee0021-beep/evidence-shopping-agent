"""本地评测：口径对齐官方 run_evaluate.py / rewards/orm.py，但不需要 Java 和 Lucene 索引。

推荐商品的完整字段在跑的时候就从沙箱缓存下来了（trace 文件里的 products），这里直接用。

和官方的唯一差别是标题相似度：
- 官方用 Qwen3-Embedding-0.6B 算余弦相似度，>= 0.5 算命中；
- 装了 sentence-transformers 就用同一个模型（--title-sim st），结果与官方一致；
- 否则退化成词级 Jaccard（--title-sim jaccard），输出里会标注 approx，最终数字以官方脚本为准。

指标：
- ASR（成功率）：product 要 rule >= 1；shop 还要同店；voucher 还要用券后不超预算；
- GT 命中率：推荐的正好是生成该任务的那个商品；
- 分字段匹配：title / price / service / sku & attrs。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable

from .task import Voucher
from .voucher import within_budget

FIELDS = ("title", "price", "service", "sku & attrs")
TitleSim = Callable[[str, str], float]


def jaccard_sim(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def make_title_sim(kind: str = "auto") -> tuple[TitleSim, str]:
    if kind in ("auto", "st"):
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

            def st_sim(a: str, b: str) -> float:
                emb = model.encode([a, b])
                return float(model.similarity(emb, emb)[0][1])

            return st_sim, "st"
        except Exception:  # noqa: BLE001
            if kind == "st":
                raise
    return jaccard_sim, "jaccard(approx)"


def rule_score(product: dict[str, Any], reward: dict[str, Any], title_sim: TitleSim) -> tuple[float, Counter, Counter]:
    """与官方 rule_score_reward 同样的计分规则。"""
    total, hit = Counter(), Counter()
    if str(product.get("product_id")) == str(reward["product_id"]):
        return 1.0, total, hit
    for title in reward.get("title", []):
        total["title"] += 1
        hit["title"] += title_sim(product.get("title", ""), title) >= 0.5
    for rng in reward.get("price", []):
        for mode, (lo, hi) in rng.items():
            total["price"] += 1
            p = product.get("price", 0)
            hit["price"] += (
                (mode == "less than" and p <= hi) or (mode == "greater than" and p >= lo) or (mode == "between" and lo <= p <= hi)
            )
    for s in reward.get("service", []):
        total["service"] += 1
        hit["service"] += s in (product.get("service") or [])

    skus = [set()] + [set(kv.items()) for kv in (product.get("sku_options") or {}).values()]
    attrs = {(k, v) for k, vs in (product.get("attributes") or {}).items() for v in vs}
    best_t = best_h = 0
    for sku in skus:
        t = h = 0
        for opt in reward.get("sku_options", []):
            for kv in opt.items():
                t += 1
                h += kv in sku or kv in attrs
        for attr in reward.get("attributes", []):
            for k, vs in attr.items():
                for v in vs:
                    t += 1
                    h += (k, v) in sku or (k, v) in attrs
        best_t, best_h = max(best_t, t), max(best_h, h)
    total["sku & attrs"] += best_t
    hit["sku & attrs"] += best_h
    n = sum(total.values())
    return (sum(hit.values()) / n if n else 1.0), total, hit


def score_case(task: str, case: dict[str, Any], recommended: list[str], products: dict[str, dict], title_sim: TitleSim) -> dict[str, Any]:
    rewards = case["reward"] if isinstance(case["reward"], list) else [case["reward"]]
    if task == "product":
        recommended = recommended[:1]
    s: dict[str, float] = Counter()
    hits, shops, prices = 0, [], []
    for i, rw in enumerate(rewards):
        pid = recommended[i] if i < len(recommended) else ""
        prod = products.get(pid)
        if not prod:
            continue
        hits += 1
        shops.append(str(prod.get("shop_id")))
        prices.append(float(prod.get("price", 0)))
        r, total, hit = rule_score(prod, rw, title_sim)
        s["product"] += 1
        s["gt"] += str(prod.get("product_id")) == str(rw["product_id"])
        s["rule"] += r
        for f in FIELDS:
            s[f] += hit[f] / total[f] if total[f] else 1
    n = len(rewards)
    out = {k: v / n for k, v in s.items()}
    out.setdefault("rule", 0.0)
    out.setdefault("gt", 0.0)
    out["shop"] = int(hits == n and len(set(shops)) == 1)
    if task == "voucher":
        v = Voucher(**{k: case["voucher"][k] for k in Voucher.model_fields})
        out["budget"] = int(hits == n and within_budget(prices, shops, v))
    out["success"] = int(
        out["rule"] >= 1
        and (task != "shop" or out["shop"] == 1)
        and (task != "voucher" or out["budget"] == 1)
    )
    return out


def summarize(task: str, scores: list[dict[str, Any]], title_sim_name: str) -> dict[str, Any]:
    n = len(scores) or 1

    def avg(k: str) -> float:
        return round(sum(s.get(k, 0) for s in scores) / n, 4)

    out = {
        "task": task,
        "cases": len(scores),
        "title_sim": title_sim_name,
        "ASR": avg("success"),
        "gt_rate": round(sum(s["gt"] >= 1 for s in scores) / n, 4),
        "rule_score": avg("rule"),
        "field_match": {f: avg(f) for f in FIELDS},
    }
    if task == "shop":
        out["shop_match"] = avg("shop")
    if task == "voucher":
        out["budget_match"] = avg("budget")
    return out
