"""在 ShoppingBench 上跑 harness，输出官方格式的 rollout 和本地评测结果。

    # 官方沙箱（先在 ShoppingBench 目录下 ./init_env.sh 启动搜索服务）
    python -m shopping_agent.bench.run --bench-dir ../ShoppingBench --task product --limit 50 --llm
    python -m shopping_agent.bench.run --bench-dir ../ShoppingBench --task voucher --limit 50 --llm

    # 不装 Java：用 documents.jsonl 的子集做离线沙箱（只能验证链路，数字不可比）
    python -m shopping_agent.bench.run --bench-dir ../ShoppingBench --task product --limit 20 \\
        --offline ../ShoppingBench/resources/documents.jsonl --offline-limit 200000

    # 只复现第 17 条（bad case 排查）；固定解析结果，重跑时只有改动的模块会带来差异
    python -m shopping_agent.bench.run --bench-dir ../ShoppingBench --task product --only 17 \\
        --llm --parse-cache eval/results/shoppingbench/parse_cache_product.json

评测模式（非交互、可复现）：
    --seed            传给 LLM 解析器（temperature 固定为 0）
    --parse-cache     LLM 解析结果落盘，之后的重跑直接复用，不再受模型输出波动影响
    --only / --offset 单条任务入口和分段跑
    --max-tool-calls  每条任务的沙箱调用上限（循环护栏）；--no-guard 关掉去重和无进展停止，做消融
    --workers         多件商品的子 agent 并行数，1 = 串行；并行和串行产出的 rollout 一致
    每次运行的参数、git commit、时间都写进 summary 的 run 字段，数字能追溯到具体代码和配置。

产出（eval/results/shoppingbench/）：
    rollout_<task>.jsonl          官方格式，可以直接交给官方 src/agent/run_evaluate.py
    official_config_<task>.json   对应的官方评测配置
    trace_<task>.jsonl            每条任务的解析结果、推荐、组合、逐节点 trace、得分和失败归因
    summary_<task>.json           ASR、GT 命中率、分字段匹配、失败阶段分布
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Callable

from ..config import PROJECT_ROOT, load_settings
from .agent import BenchAgent, EpisodeConfig, RunResult
from .metrics import make_title_sim, score_case, summarize
from .sandbox import HttpSandbox, LatencySandbox, OfflineSandbox
from .task import CachedParser, LLMTaskParser


def attribute(task: str, case: dict[str, Any], res: RunResult, score: dict[str, Any]) -> str:
    """失败归因：解析 -> 检索 -> 证据 -> 同店 -> 预算。"""
    if score["success"]:
        return "ok"
    n = len(case["reward"]) if isinstance(case["reward"], list) else 1
    if task != "product" and len(res.spec.items) != n:
        return "understanding(item_split)"
    if not all(res.recommended[:n]) or len(res.recommended) < n:
        return "retrieval(no_candidate)"
    gathered = [e for e in res.trace if e["node"] == "gather_evidence" and e["type"] == "node" and e.get("best")]
    if any(e["best"]["missing"] for e in gathered[-n:]):
        return "evidence(attribute_mismatch)"
    if task == "shop" and not score.get("shop"):
        return "constraint(same_shop)"
    if task == "voucher" and not score.get("budget"):
        return "constraint(budget)"
    return "decision(wrong_product)"


def p95(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else 0.0


def run_cases(
    task: str,
    cases: list[dict[str, Any]],
    agent: BenchAgent,
    title_sim: Callable[[str, str], float] | None = None,
    on_case: Callable[[int, dict[str, Any], RunResult, dict[str, Any] | None, str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[RunResult], Counter]:
    """跑一批任务。有标注（case["reward"]）就评分和归因，没有就只跑链路（ablation 的演示模式用）。"""
    scores, results, stages = [], [], Counter()
    for i, case in enumerate(cases, 1):
        res = agent.run(case["query"], task)
        score, stage = None, "unlabeled"
        if "reward" in case and title_sim is not None:
            score = score_case(task, case, res.recommended, res.products, title_sim)
            stage = attribute(task, case, res, score)
            scores.append(score)
        stages[stage] += 1
        results.append(res)
        if on_case:
            on_case(i, case, res, score, stage)
    return scores, results, stages


def efficiency(results: list[RunResult]) -> dict[str, Any]:
    """工具调用和时延：消融表里和 ASR 并列的另一半。"""
    if not results:
        return {}
    calls = [r.stats.get("tool_calls", 0) for r in results]
    wall = [r.stats.get("wall_ms", 0.0) for r in results]
    stops = Counter(e.get("stop_reason") for r in results for e in r.trace if e["node"] == "subagent")
    return {
        "avg_tool_calls": round(sum(calls) / len(calls), 2),
        "max_tool_calls": max(calls),
        "dedup_hits": sum(r.stats.get("cached", 0) for r in results),
        "budget_blocked": sum(r.stats.get("blocked", 0) for r in results),
        "latency_ms_p50": round(median(wall), 1),
        "latency_ms_p95": round(p95(wall), 1),
        "subagent_stop_reasons": dict(stops),
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def select_cases(cases: list[dict[str, Any]], offset: int, limit: int, only: int | None) -> list[tuple[int, dict]]:
    """返回 (原始序号, case)。序号从 1 开始，和日志、trace 里的编号一致，方便单条复现。"""
    indexed = list(enumerate(cases, 1))
    if only is not None:
        if not 1 <= only <= len(cases):
            raise SystemExit(f"--only {only} 超出范围（共 {len(cases)} 条）")
        return [indexed[only - 1]]
    return indexed[offset : offset + limit]


def build_parser(args: argparse.Namespace):
    inner = None
    if args.llm:
        s = load_settings()
        if not s.llm_api_key and not args.parse_cache:
            raise SystemExit("--llm 需要设置 LLM_API_KEY（或提供已有的 --parse-cache）")
        if s.llm_api_key:
            inner = LLMTaskParser(s.llm_api_key, s.llm_base_url, s.llm_model, seed=args.seed)
    if args.parse_cache:
        return CachedParser(inner, args.parse_cache)
    return inner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True, help="ShoppingBench 仓库目录（读 data/synthesize_<task>_test.jsonl）")
    ap.add_argument("--task", choices=["product", "shop", "voucher"], required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0, help="从第几条开始（0 起）")
    ap.add_argument("--only", type=int, default=None, help="只跑第 N 条（1 起，和日志编号一致）")
    ap.add_argument("--llm", action="store_true", help="用 LLM 解析任务（需要 LLM_API_KEY），否则用规则")
    ap.add_argument("--seed", type=int, default=0, help="传给 LLM 解析器的 seed")
    ap.add_argument("--parse-cache", help="解析结果缓存文件：存在就复用，跑完写回")
    ap.add_argument("--sandbox-url", default="http://127.0.0.1:5631")
    ap.add_argument("--offline", help="documents.jsonl 路径：用离线沙箱代替官方搜索服务")
    ap.add_argument("--offline-limit", type=int, default=None)
    ap.add_argument("--latency-ms", type=float, default=0, help="给每次沙箱调用加固定时延（压测并行用）")
    ap.add_argument("--max-tool-calls", type=int, default=40, help="每条任务的沙箱调用上限；0 = 不限")
    ap.add_argument("--no-guard", action="store_true", help="关掉去重和无进展停止（消融用）")
    ap.add_argument("--workers", type=int, default=4, help="多件商品子 agent 的并行数，1 = 串行")
    ap.add_argument("--title-sim", choices=["auto", "st", "jaccard"], default="auto")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "eval" / "results" / "shoppingbench"))
    args = ap.parse_args()

    bench = Path(args.bench_dir)
    data_file = bench / "data" / f"synthesize_{args.task}_test.jsonl"
    all_cases = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = select_cases(all_cases, args.offset, args.limit, args.only)
    cases = [c for _, c in picked]

    sandbox = OfflineSandbox.from_documents(args.offline, args.offline_limit) if args.offline else HttpSandbox(args.sandbox_url)
    if args.latency_ms:
        sandbox = LatencySandbox(sandbox, args.latency_ms)
    parser = build_parser(args)
    config = EpisodeConfig(
        max_tool_calls=args.max_tool_calls or None,
        dedup=not args.no_guard,
        stop_on_no_progress=not args.no_guard,
        workers=args.workers,
    )
    agent = BenchAgent(sandbox, parser, config)
    title_sim, sim_name = make_title_sim(args.title_sim)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.only}" if args.only is not None else ""
    rollout_path = out / f"rollout_{args.task}{suffix}.jsonl"
    with open(rollout_path, "w", encoding="utf-8") as fr, \
            open(out / f"trace_{args.task}{suffix}.jsonl", "w", encoding="utf-8") as ft:

        def write(i: int, case: dict, res: RunResult, score: dict | None, stage: str) -> None:
            idx = picked[i - 1][0]
            fr.write(json.dumps(res.steps, ensure_ascii=False) + "\n")
            ft.write(json.dumps({
                "index": idx,
                "query": res.query,
                "spec": res.spec.model_dump(exclude_none=True),
                "recommended": res.recommended,
                "basket": {k: v for k, v in (res.basket or {}).items() if k != "items"},
                "products": res.products,
                "score": score,
                "attribution": stage,
                "stats": res.stats,
                "trace": res.trace,
            }, ensure_ascii=False, default=str) + "\n")
            ok = score["success"] if score else "-"
            rule = f"{score['rule']:.2f}" if score else "-"
            print(f"[#{idx} {i}/{len(cases)}] success={ok} rule={rule} calls={res.stats['tool_calls']} {stage}")

        scores, results, stages = run_cases(args.task, cases, agent, title_sim, write)

    if isinstance(parser, CachedParser):
        parser.save()
    # 没有标注时不报 ASR，免得 0.0 被误读成“全部失败”
    summary = summarize(args.task, scores, sim_name) if scores else {"task": args.task, "cases": len(results),
                                                                     "labeled": 0, "ASR": None}
    summary["parser"] = ("llm" if args.llm else "rules") + ("+cache" if args.parse_cache else "")
    summary["sandbox"] = "offline" if args.offline else args.sandbox_url
    summary["failures_by_stage"] = dict(stages)
    summary["efficiency"] = efficiency(results)
    summary["run"] = {
        "git_commit": git_commit(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "case_indices": [i for i, _ in picked][:20] + (["..."] if len(picked) > 20 else []),
        "config": {**vars(args)},
        "parse_cache": {"hits": parser.hits, "misses": parser.misses} if isinstance(parser, CachedParser) else None,
    }
    (out / f"summary_{args.task}{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    official = {
        "task": args.task,
        "synthesize_file": str(data_file.resolve()),
        "rollout_file": str(rollout_path.resolve()),
        "model_config": {"model": "evidence-shopping-agent"},
    }
    (out / f"official_config_{args.task}{suffix}.json").write_text(json.dumps(official, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
