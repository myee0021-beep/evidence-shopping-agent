"""分层评测：检索 -> 硬约束 -> 工具调用 -> 记忆 -> 路由 -> 端到端，外加分阶段时延。

    python eval/run_eval.py                              # 基础用例，规则解析器
    python eval/run_eval.py --cases cases_hard.jsonl     # 口语化难例
    python eval/run_eval.py --k 3 --llm                  # LLM 解析器，每个用例跑 3 次，算 pass^3

产出（按用例文件分目录）：
    eval/results/<用例文件名>/summary.json        各项指标
    eval/results/<用例文件名>/failed_traces.jsonl 失败轮次的完整轨迹 + 错误归因
                                                 （用 python -m shopping_agent.trace 回放）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from shopping_agent.catalog import load_catalog  # noqa: E402
from shopping_agent.cli import make_parser  # noqa: E402
from shopping_agent.config import load_settings  # noqa: E402
from shopping_agent.harness import build_agent  # noqa: E402
from shopping_agent.trace import attribute, turn_events  # noqa: E402


def p95(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else 0.0


def rate(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def same(actual, expected) -> bool:
    if isinstance(expected, list):
        return sorted(actual or []) == sorted(expected)
    return actual == expected


def hard_ok(pid: str, out: dict, catalog) -> bool:
    """展示出来的商品是否满足硬约束（按证据里的详情页价格算）。"""
    c = {k: v for k, v in out["constraints"].items() if k not in out.get("relaxed", [])}
    p, ev = catalog[pid], out["evidence"][pid]
    price = ev["verified_price"]
    return (
        p.category == c.get("category")
        and (c.get("budget_max") is None or price <= c["budget_max"])
        and (c.get("budget_min") is None or price >= c["budget_min"])
        and p.brand not in c.get("brands_exclude", [])
        and (not c.get("brands_include") or p.brand in c["brands_include"])
        and ev["in_stock"]
    )


def run(args) -> dict:
    settings = load_settings()
    catalog = load_catalog(settings.catalog_path)
    parser = make_parser(settings) if args.llm else None
    lines = (ROOT / "eval" / args.cases).read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]

    m = defaultdict(list)  # 指标名 -> list[bool | float]
    node_lat = defaultdict(list)
    e2e_lat: list[float] = []
    case_runs: dict[str, list[bool]] = defaultdict(list)
    failed = []

    for run_idx in range(args.k):
        for case in cases:
            agent = build_agent(catalog, settings, InMemoryStore(), InMemorySaver(), parser)
            user = case.get("user_id", "eval_user")
            prev_top_price = None
            case_ok = True
            for t_idx, turn in enumerate(case["turns"], 1):
                thread = f"{case['id']}-{run_idx}-{turn.get('thread', 'main')}"
                start = time.perf_counter()
                out = agent.invoke({"user_id": user, "query": turn["user"]}, {"configurable": {"thread_id": thread}})
                e2e_lat.append((time.perf_counter() - start) * 1000)
                exp = turn.get("expect", {})
                events = turn_events(out["trace"], out["turn"])
                for e in events:
                    if e["type"] == "node":
                        node_lat[e["node"]].append(e["latency_ms"])
                fails = []

                if "route" in exp:
                    ok = out["route"] == exp["route"]
                    m["route_accuracy"].append(ok)
                    ok or fails.append(f"route={out['route']} 期望 {exp['route']}")

                for key, val in exp.get("constraints", {}).items():
                    ok = same(out["constraints"].get(key), val)
                    m["constraint_extraction"].append(ok)
                    ok or fails.append(f"constraints.{key}={out['constraints'].get(key)} 期望 {val}")

                for key, val in exp.get("memory", {}).items():
                    ok = same(out["constraints"].get(key), val)
                    m["memory_recall"].append(ok)
                    ok or fails.append(f"memory.{key}={out['constraints'].get(key)} 期望 {val}")

                if "relevant" in exp:
                    got = set(out.get("candidates", []))
                    rec = len(got & set(exp["relevant"])) / len(exp["relevant"])
                    m["retrieval_recall"].append(rec)
                    rec == 1 or fails.append(f"recall={rec:.2f}")

                if "skills" in exp:
                    called = {e["skill"] for e in events if e["type"] == "skill"}
                    all_ok = all(e["ok"] for e in events if e["type"] == "skill")
                    ok = set(exp["skills"]) <= called and all_ok
                    m["tool_call_accuracy"].append(ok)
                    ok or fails.append(f"skills={sorted(called)} 期望包含 {exp['skills']}")

                if "relaxed" in exp:
                    ok = same(out.get("relaxed"), exp["relaxed"])
                    m["replan_accuracy"].append(ok)
                    ok or fails.append(f"relaxed={out.get('relaxed')} 期望 {exp['relaxed']}")

                shown = out.get("shown", []) if out["route"] in ("recommend", "caveat") else []
                for pid in shown:
                    ok = hard_ok(pid, out, catalog)
                    m["hard_constraint_satisfaction"].append(ok)
                    ok or fails.append(f"{pid} 违反硬约束")
                for pid in exp.get("shown_exclude", []):
                    if pid in shown:
                        fails.append(f"{pid} 不应被推荐")
                for b in exp.get("shown_brands_exclude", []):
                    if any(catalog[p].brand == b for p in shown):
                        fails.append(f"推荐里出现了排除品牌 {b}")
                if "shown_brands_only" in exp and any(catalog[p].brand not in exp["shown_brands_only"] for p in shown):
                    fails.append(f"推荐里出现了限定之外的品牌")
                if exp.get("cheaper_than_prev_top") and prev_top_price is not None:
                    prices = [out["evidence"][p]["verified_price"] for p in shown]
                    if not prices or max(prices) >= prev_top_price:
                        fails.append(f"没有比上一轮首推（{prev_top_price} 元）更便宜")

                turn_ok = not fails
                m["turn_success"].append(turn_ok)
                case_ok &= turn_ok
                if not turn_ok:
                    failed.append(
                        {
                            "case_id": case["id"],
                            "run": run_idx,
                            "turn": out["turn"],
                            "user": turn["user"],
                            "failures": fails,
                            "attribution": attribute(out["trace"], out["turn"], fails),
                            "trace": events,
                        }
                    )
                if shown:
                    prev_top_price = out["evidence"][shown[0]]["verified_price"]
            case_runs[case["id"]].append(case_ok)

    summary = {
        "case_file": args.cases,
        "parser": "llm" if parser else "rules",
        "k": args.k,
        "cases": len(cases),
        "turns": len(m["turn_success"]) // args.k,
        "metrics": {
            "检索召回 retrieval_recall": round(mean(m["retrieval_recall"]), 4) if m["retrieval_recall"] else None,
            "硬约束满足率 hard_constraint_satisfaction": rate(m["hard_constraint_satisfaction"]),
            "约束抽取准确率 constraint_extraction": rate(m["constraint_extraction"]),
            "工具调用正确率 tool_call_accuracy": rate(m["tool_call_accuracy"]),
            "跨轮记忆召回 memory_recall": rate(m["memory_recall"]),
            "重规划正确率 replan_accuracy": rate(m["replan_accuracy"]),
            "路由准确率 route_accuracy": rate(m["route_accuracy"]),
            "单轮成功率 turn_success": rate(m["turn_success"]),
            "端到端任务成功率 pass@1": round(mean(mean(v) for v in case_runs.values()), 4),
            f"稳定性 pass^{args.k}": rate([all(v) for v in case_runs.values()]),
        },
        "latency_ms": {
            "end_to_end_p50": round(median(e2e_lat), 2),
            "end_to_end_p95": round(p95(e2e_lat), 2),
            "per_node_p50": {k: round(median(v), 3) for k, v in node_lat.items()},
        },
        "failures_by_stage": dict(_count(f["attribution"] for f in failed)),
    }

    out_dir = ROOT / "eval" / "results" / Path(args.cases).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "failed_traces.jsonl", "w", encoding="utf-8") as f:
        for rec in failed:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return summary


def _count(xs):
    d = defaultdict(int)
    for x in xs:
        d[x] += 1
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1, help="每个用例重复次数，用于计算 pass^k")
    ap.add_argument("--llm", action="store_true", help="使用 LLM 解析器（需要 LLM_API_KEY）")
    ap.add_argument("--cases", default="cases.jsonl", help="eval/ 下的用例文件，如 cases_hard.jsonl")
    s = run(ap.parse_args())
    print(json.dumps(s, ensure_ascii=False, indent=2))
