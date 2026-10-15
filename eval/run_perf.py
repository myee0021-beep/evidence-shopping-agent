"""证据聚合并行化的时延对比：串行 vs 并行，在模拟的下游详情服务时延下跑同一批用例。

    python eval/run_perf.py --latency-ms 30

本地商品库是内存读取，耗时可以忽略，所以用 DETAIL_LATENCY_MS 给 get_product_detail 加一段固定的
RPC 时延来模拟详情服务。数字说明的是“并行能把多少串行等待省掉”，不是线上真实时延。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from shopping_agent.catalog import load_catalog  # noqa: E402
from shopping_agent.config import load_settings  # noqa: E402
from shopping_agent.harness import build_agent  # noqa: E402
from shopping_agent.trace import turn_events  # noqa: E402


def p95(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def run(parallel: bool, latency_ms: float, cases: list[dict]) -> dict:
    settings = replace(load_settings(), parallel_evidence=parallel, detail_latency_ms=latency_ms)
    catalog = load_catalog(settings.catalog_path)
    e2e, gather, routes = [], [], []
    for case in cases:
        agent = build_agent(catalog, settings, InMemoryStore(), InMemorySaver())
        for turn in case["turns"]:
            cfg = {"configurable": {"thread_id": f"{case['id']}-{turn.get('thread', 'main')}"}}
            start = time.perf_counter()
            out = agent.invoke({"user_id": case.get("user_id", "u"), "query": turn["user"]}, cfg)
            e2e.append((time.perf_counter() - start) * 1000)
            routes.append(out["route"])
            gather += [e["latency_ms"] for e in turn_events(out["trace"], out["turn"])
                       if e["type"] == "node" and e["node"] == "gather_evidence"]
    return {
        "mode": "parallel" if parallel else "serial",
        "end_to_end_p50": round(median(e2e), 1),
        "end_to_end_p95": round(p95(e2e), 1),
        "gather_evidence_p50": round(median(gather), 1),
        "gather_evidence_p95": round(p95(gather), 1),
        "routes": routes,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-ms", type=float, default=30)
    ap.add_argument("--cases", default="cases.jsonl")
    args = ap.parse_args()
    lines = (ROOT / "eval" / args.cases).read_text(encoding="utf-8").splitlines()
    cases = [json.loads(x) for x in lines if x.strip()]
    serial, parallel = run(False, args.latency_ms, cases), run(True, args.latency_ms, cases)
    assert serial.pop("routes") == parallel.pop("routes"), "并行改变了决策结果"
    print(json.dumps({"detail_latency_ms": args.latency_ms, "note": "下游时延为模拟值",
                      "results": [serial, parallel]}, ensure_ascii=False, indent=2))
