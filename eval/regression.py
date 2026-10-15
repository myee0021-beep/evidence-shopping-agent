"""回归准入：改动后在固定用例上全量重跑，和基线逐项对比，任何一层指标下降超过阈值就判失败。

    python eval/regression.py                        # 两套用例都对比基线，失败时退出码 1
    python eval/regression.py --update-baseline      # 确认改动符合预期后，更新基线

为什么要分层对比、而不是只看端到端：
- 改了理解层，路由准确率涨了，但硬约束满足率跌了，说明改出了新问题；只看端到端可能互相抵消；
- 失败阶段分布也要对比：某一类失败变多了，就是新引入的问题，即使总数没变。

LLM 解析器有随机性，用 --k 3 跑 pass^3，基线也用同样的 k 生成。时延波动大，只提示不拦截。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_eval import run  # noqa: E402

BASELINE_DIR = ROOT / "baselines"
CASE_FILES = ("cases.jsonl", "cases_hard.jsonl")


def compare(base: dict, cur: dict, tolerance: float) -> tuple[list[str], list[str]]:
    """返回 (退步项, 提示项)。"""
    regressions, notes = [], []
    for name, b in base["metrics"].items():
        c = cur["metrics"].get(name)
        if b is None or c is None:
            continue
        if c < b - tolerance:
            regressions.append(f"{name}: {b} -> {c}")
        elif c > b:
            notes.append(f"提升 {name}: {b} -> {c}")
    bf, cf = base.get("failures_by_stage", {}), cur.get("failures_by_stage", {})
    for stage in sorted(set(bf) | set(cf)):
        if cf.get(stage, 0) > bf.get(stage, 0):
            regressions.append(f"失败阶段 {stage}: {bf.get(stage, 0)} -> {cf.get(stage, 0)} 例")
    bt, ct = base.get("cost", {}).get("avg_window_tokens"), cur.get("cost", {}).get("avg_window_tokens")
    if bt and ct and ct > bt * 1.2:
        notes.append(f"提示：平均窗口 tokens {bt} -> {ct}（+{(ct / bt - 1):.0%}）")
    bl, cl = base["latency_ms"]["end_to_end_p95"], cur["latency_ms"]["end_to_end_p95"]
    if bl and cl > bl * 2 and cl - bl > 20:
        notes.append(f"提示：端到端 p95 时延 {bl}ms -> {cl}ms（时延波动大，不拦截）")
    return regressions, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.0, help="允许的指标下降幅度，LLM 解析时可设 0.02")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    BASELINE_DIR.mkdir(exist_ok=True)
    failed = False
    for cases in CASE_FILES:
        cur = run(SimpleNamespace(cases=cases, k=args.k, llm=args.llm, hybrid=False, ablate=""))
        path = BASELINE_DIR / f"{Path(cases).stem}{'_llm' if args.llm else ''}.json"
        if args.update_baseline or not path.exists():
            keep = {k: cur[k] for k in ("case_file", "parser", "k", "cases", "turns", "metrics", "latency_ms", "cost", "failures_by_stage")}
            path.write_text(json.dumps(keep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[{cases}] 基线已写入 {path.relative_to(ROOT.parent)}")
            continue
        regressions, notes = compare(json.loads(path.read_text(encoding="utf-8")), cur, args.tolerance)
        status = "FAIL" if regressions else "PASS"
        print(f"[{cases}] {status}")
        for r in regressions:
            print(f"  ✗ {r}")
        for n in notes:
            print(f"  · {n}")
        failed |= bool(regressions)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
