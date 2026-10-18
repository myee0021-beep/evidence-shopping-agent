"""ShoppingBench 链路的消融：循环护栏、子 agent 并行各贡献了多少。

    # 真实评测（需要按 docs/shoppingbench.md 准备 ShoppingBench 和搜索服务）
    python eval/run_bench_ablation.py --bench-dir ../ShoppingBench --task product --limit 50 \\
        --llm --parse-cache eval/results/shoppingbench/parse_cache_product.json

    # 演示模式：仓库自带的离线夹具（7 个商品、9 条任务），没有标注，只比较调用次数和时延。
    # 解析结果用手写的 tests/fixtures/bench_parse_cache.json（模拟 LLM 抽出的属性词），
    # 这样属性没命中时才会触发重规划，护栏才有东西可拦
    python eval/run_bench_ablation.py --demo --latency-ms 30

对比的配置：
    full            默认配置：去重 + 无进展停止 + 调用预算 + 子 agent 并行
    -guard          关掉去重和无进展停止（预算仍在）
    -parallel       子 agent 串行
    -guard-parallel 两个都关，等价于改造前的链路
每组用同一份解析结果（--parse-cache 或规则解析），差异只来自被关掉的模块。
演示模式的数字只能说明机制，不能当效果写进简历。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.bench.agent import BenchAgent, EpisodeConfig  # noqa: E402
from shopping_agent.bench.metrics import make_title_sim, summarize  # noqa: E402
from shopping_agent.bench.run import efficiency, run_cases  # noqa: E402
from shopping_agent.bench.sandbox import HttpSandbox, LatencySandbox, OfflineSandbox  # noqa: E402
from shopping_agent.bench.task import CachedParser  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
CONFIGS = {
    "full": EpisodeConfig(),
    "-guard": EpisodeConfig(dedup=False, stop_on_no_progress=False),
    "-parallel": EpisodeConfig(workers=1),
    "-guard-parallel": EpisodeConfig(dedup=False, stop_on_no_progress=False, workers=1),
}


def load(args) -> list[tuple[str, list[dict]]]:
    if args.demo:
        rows = [json.loads(x) for x in (FIXTURES / "bench_tasks.jsonl").read_text(encoding="utf-8").splitlines() if x]
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r["task"], []).append(r)
        return list(groups.items())
    path = Path(args.bench_dir) / "data" / f"synthesize_{args.task}_test.jsonl"
    cases = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()][: args.limit]
    return [(args.task, cases)]


def make_sandbox(args):
    if args.demo:
        sb = OfflineSandbox.from_documents(FIXTURES / "bench_documents.jsonl")
    elif args.offline:
        sb = OfflineSandbox.from_documents(args.offline, args.offline_limit)
    else:
        sb = HttpSandbox(args.sandbox_url)
    return LatencySandbox(sb, args.latency_ms) if args.latency_ms else sb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="用仓库自带的离线夹具跑（无标注，只看调用和时延）")
    ap.add_argument("--bench-dir")
    ap.add_argument("--task", choices=["product", "shop", "voucher"], default="product")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--parse-cache", help="先用 bench.run --parse-cache 生成；所有配置共用同一份解析")
    ap.add_argument("--sandbox-url", default="http://127.0.0.1:5631")
    ap.add_argument("--offline")
    ap.add_argument("--offline-limit", type=int)
    ap.add_argument("--latency-ms", type=float, default=0)
    ap.add_argument("--title-sim", choices=["auto", "st", "jaccard"], default="jaccard")
    args = ap.parse_args()
    if not args.demo and not args.bench_dir:
        raise SystemExit("需要 --bench-dir，或者用 --demo 跑离线夹具")

    cache = args.parse_cache or (FIXTURES / "bench_parse_cache.json" if args.demo else None)
    parser = CachedParser(None, cache) if cache else None
    title_sim, sim_name = make_title_sim(args.title_sim)
    table = []
    for name, cfg in CONFIGS.items():
        row: dict = {"config": name}
        for task, cases in load(args):
            # 每个配置一个新沙箱：seen 缓存不跨配置共享，各组互不影响
            agent = BenchAgent(make_sandbox(args), parser, cfg)
            scores, results, stages = run_cases(task, cases, agent, None if args.demo else title_sim)
            eff = efficiency(results)
            if scores:
                row[f"{task}.ASR"] = summarize(task, scores, sim_name)["ASR"]
            row[f"{task}.avg_calls"] = eff["avg_tool_calls"]
            row[f"{task}.dedup_hits"] = eff["dedup_hits"]
            row[f"{task}.p50_ms"] = eff["latency_ms_p50"]
            row[f"{task}.recommended"] = [r.recommended for r in results]
        table.append(row)

    keys = [k for k in table[0] if not k.endswith(".recommended")]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for row in table:
        print("| " + " | ".join(str(row[k]) for k in keys) + " |")
    same = all(row[k] == table[0][k] for row in table for k in row if k.endswith(".recommended"))
    print(f"\n各配置推荐结果是否一致：{same}")
    if args.demo:
        print("演示模式：离线夹具、无标注、模拟时延，只说明机制，不是效果数字。")


if __name__ == "__main__":
    main()
