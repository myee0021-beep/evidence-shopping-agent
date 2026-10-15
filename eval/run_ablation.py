"""消融：每次关掉一个模块，看它贡献了多少。

    python eval/run_ablation.py

在两套用例上跑完整版和 4 个消融版本，输出 Markdown 表。用例是模拟商品库上手写的，
数字只说明“每个模块在这些场景里起什么作用”，不是线上效果。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

from run_eval import run  # noqa: E402

from shopping_agent.harness import ABLATIONS  # noqa: E402

COLS = (
    ("硬约束满足", "硬约束满足率 hard_constraint_satisfaction"),
    ("跨轮记忆", "跨轮记忆召回 memory_recall"),
    ("路由准确", "路由准确率 route_accuracy"),
    ("单轮成功", "单轮成功率 turn_success"),
    ("端到端 pass@1", "端到端任务成功率 pass@1"),
)


def fmt(v) -> str:
    return "—" if v is None else f"{v:.0%}"


def main() -> None:
    variants = [("完整版", "")] + [(f"去掉 {k}（{v}）", k) for k, v in ABLATIONS.items()]
    for cases in ("cases.jsonl", "cases_hard.jsonl"):
        print(f"\n### {cases}\n")
        print("| 版本 | " + " | ".join(c for c, _ in COLS) + " |")
        print("|" + " --- |" * (len(COLS) + 1))
        for label, ab in variants:
            s = run(SimpleNamespace(cases=cases, k=1, llm=False, hybrid=False, ablate=ab))
            row = " | ".join(fmt(s["metrics"][key]) for _, key in COLS)
            print(f"| {label} | {row} |")


if __name__ == "__main__":
    main()
