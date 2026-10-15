"""在 ShoppingBench 上跑 harness，输出官方格式的 rollout 和本地评测结果。

    # 官方沙箱（先在 ShoppingBench 目录下 ./init_env.sh 启动搜索服务）
    python -m shopping_agent.bench.run --bench-dir ../ShoppingBench --task product --limit 50 --llm
    python -m shopping_agent.bench.run --bench-dir ../ShoppingBench --task voucher --limit 50 --llm

    # 不装 Java：用 documents.jsonl 的子集做离线沙箱（只能验证链路，数字不可比）
    python -m shopping_agent.bench.run --bench-dir ../ShoppingBench --task product --limit 20 \\
        --offline ../ShoppingBench/resources/documents.jsonl --offline-limit 200000

产出（eval/results/shoppingbench/）：
    rollout_<task>.jsonl          官方格式，可以直接交给官方 src/agent/run_evaluate.py
    official_config_<task>.json   对应的官方评测配置
    trace_<task>.jsonl            每条任务的解析结果、推荐、组合、逐节点 trace、得分和失败归因
    summary_<task>.json           ASR、GT 命中率、分字段匹配、失败阶段分布
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, load_settings
from .agent import BenchAgent, RunResult
from .metrics import make_title_sim, score_case, summarize
from .sandbox import HttpSandbox, OfflineSandbox
from .task import LLMTaskParser


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True, help="ShoppingBench 仓库目录（读 data/synthesize_<task>_test.jsonl）")
    ap.add_argument("--task", choices=["product", "shop", "voucher"], required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--llm", action="store_true", help="用 LLM 解析任务（需要 LLM_API_KEY），否则用规则")
    ap.add_argument("--sandbox-url", default="http://127.0.0.1:5631")
    ap.add_argument("--offline", help="documents.jsonl 路径：用离线沙箱代替官方搜索服务")
    ap.add_argument("--offline-limit", type=int, default=None)
    ap.add_argument("--title-sim", choices=["auto", "st", "jaccard"], default="auto")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "eval" / "results" / "shoppingbench"))
    args = ap.parse_args()

    bench = Path(args.bench_dir)
    data_file = bench / "data" / f"synthesize_{args.task}_test.jsonl"
    cases = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()][: args.limit]

    sandbox = OfflineSandbox.from_documents(args.offline, args.offline_limit) if args.offline else HttpSandbox(args.sandbox_url)
    parser = None
    if args.llm:
        s = load_settings()
        if not s.llm_api_key:
            raise SystemExit("--llm 需要设置 LLM_API_KEY")
        parser = LLMTaskParser(s.llm_api_key, s.llm_base_url, s.llm_model)
    agent = BenchAgent(sandbox, parser)
    title_sim, sim_name = make_title_sim(args.title_sim)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rollout_path = out / f"rollout_{args.task}.jsonl"
    scores, stages = [], Counter()
    with open(rollout_path, "w", encoding="utf-8") as fr, open(out / f"trace_{args.task}.jsonl", "w", encoding="utf-8") as ft:
        for i, case in enumerate(cases, 1):
            res = agent.run(case["query"], args.task)
            score = score_case(args.task, case, res.recommended, res.products, title_sim)
            stage = attribute(args.task, case, res, score)
            scores.append(score)
            stages[stage] += 1
            fr.write(json.dumps(res.steps, ensure_ascii=False) + "\n")
            ft.write(json.dumps({
                "query": res.query,
                "spec": res.spec.model_dump(exclude_none=True),
                "recommended": res.recommended,
                "basket": {k: v for k, v in (res.basket or {}).items() if k != "items"},
                "products": res.products,
                "score": score,
                "attribution": stage,
                "trace": res.trace,
            }, ensure_ascii=False, default=str) + "\n")
            print(f"[{i}/{len(cases)}] success={score['success']} rule={score['rule']:.2f} {stage}")

    summary = summarize(args.task, scores, sim_name)
    summary["parser"] = "llm" if parser else "rules"
    summary["sandbox"] = "offline" if args.offline else args.sandbox_url
    summary["failures_by_stage"] = dict(stages)
    (out / f"summary_{args.task}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    official = {
        "task": args.task,
        "synthesize_file": str(data_file.resolve()),
        "rollout_file": str(rollout_path.resolve()),
        "model_config": {"model": "evidence-shopping-agent"},
    }
    (out / f"official_config_{args.task}.json").write_text(json.dumps(official, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
