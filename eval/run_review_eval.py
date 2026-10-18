"""评论证据的评测：检索准不准、结论对不对、引用校验能不能拦住改写里的幻觉。

    python eval/run_review_eval.py

1. 检索精度：对每个商品的每个卖点，检索到的片段里有多少条确实在讲这个卖点
   （用数据里的 aspect 标签当标注；运行时不读这个标签）。
2. 结论准确率：supported / disputed 和数据生成时设定的结果是否一致。
3. 引用校验召回：对推荐回复做 6 类“模型改写”扰动，看 citation_check + fact_check 拦下多少，
   以及正常的措辞改写会不会被误拦（误拦率）。

数据是模拟的，扰动也是规则构造的，数字只说明机制在设计的场景下有效，不是线上幻觉率。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from gen_reviews import DISPUTED  # noqa: E402
from shopping_agent.catalog import load_catalog  # noqa: E402
from shopping_agent.config import load_settings  # noqa: E402
from shopping_agent.evidence import fact_check  # noqa: E402
from shopping_agent.harness import build_agent  # noqa: E402
from shopping_agent.reviews import (  # noqa: E402
    aspect_evidence,
    citation_check,
    load_reviews,
    snippet_index,
)

QUERIES = [
    "想买个500以内的降噪蓝牙耳机，不要小米，送男朋友的",
    "漫步者的降噪耳机",
    "2000以内的5G手机，要能无线充电",
    "跑鞋要缓震的，送朋友",
    "热插拔的机械键盘",
    "1000以内的轻量跑鞋",
]


def retrieval_and_verdicts(catalog, reviews) -> dict:
    hits = total = correct = n = 0
    for pid, p in catalog.items():
        for f in p.features:
            ev = aspect_evidence(reviews.get(pid, []), f)
            tag = {r["id"]: r["aspect"] for r in reviews.get(pid, [])}
            hits += sum(tag[s["id"]] == f for s in ev["snippets"])
            total += len(ev["snippets"])
            expected = "disputed" if DISPUTED.get(pid) == f else "supported"
            correct += ev["verdict"] == expected
            n += 1
    return {"snippet_precision": f"{hits}/{total} = {hits / total:.0%}", "verdict_accuracy": f"{correct}/{n} = {correct / n:.0%}"}


# ---------- 扰动：模拟 LLM 改写可能引入的问题 ----------


def fabricate_id(text: str) -> str:
    return re.sub(r"\[R-([A-Za-z0-9]+)-\d+\]", r"[R-\1-99]", text, count=1)


def alter_quote(text: str) -> str:
    # 只改评论原话（带引用的那一行），不改商品名
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "[R-" in ln:
            lines[i] = re.sub(r"「([^」]+)」", "「音质是同价位最好的，强烈推荐」", ln, count=1)
            break
    return "\n".join(lines)


def drop_citation(text: str) -> str:
    return re.sub(r"\[R-[A-Za-z0-9]+-\d+\]", "", text)


def negative_as_positive(text: str) -> str:
    return text.replace("评价有分歧：", "买家评价：").replace("条差评", "条评价")


def wrong_price(text: str) -> str:
    return re.sub(r"(\d+) 元", lambda m: f"{int(m.group(1)) - 50} 元", text, count=1)


def swap_citation(text: str, snippets: dict) -> str:
    """把一条引用换成另一个真实存在、但原话对不上的评论 ID。"""
    cited = re.findall(r"\[(R-[A-Za-z0-9]+-\d+)\]", text)
    if not cited:
        return text
    first = cited[0]
    other = next((sid for sid in snippets if sid != first and snippets[sid]["text"] != snippets[first]["text"]), None)
    return text.replace(f"[{first}]", f"[{other}]", 1) if other else text


def benign(text: str) -> str:
    return text.replace("根据你的要求，推荐这几款", "按你的条件挑了这几款").replace("买家评价：", "买家怎么说：")


ATTACKS = {
    "编造评论 ID": fabricate_id,
    "篡改评论原话": alter_quote,
    "删掉引用": drop_citation,
    "差评当卖点": negative_as_positive,
    "改价格": wrong_price,
}


def main() -> None:
    settings = load_settings()
    catalog = load_catalog(settings.catalog_path)
    reviews = load_reviews(settings.reviews_path)
    result = {"retrieval": retrieval_and_verdicts(catalog, reviews)}

    agent = build_agent(catalog, settings, InMemoryStore(), InMemorySaver())
    caught = {name: [0, 0] for name in [*ATTACKS, "引用换成别的评论"]}
    benign_blocked = benign_total = 0
    for i, q in enumerate(QUERIES):
        out = agent.invoke({"user_id": "u", "query": q}, {"configurable": {"thread_id": f"r{i}"}})
        if out["route"] not in ("recommend", "caveat") or not out.get("review_evidence"):
            continue
        text = out["response"]
        snippets = snippet_index(out["review_evidence"])
        allowed = {e["verified_price"] for e in out["evidence"].values()}

        def blocked(t: str) -> bool:
            return bool(fact_check(t, allowed) or citation_check(t, snippets))

        assert not blocked(text), f"模板回复没通过校验：{q}"
        attacks = {**ATTACKS, "引用换成别的评论": lambda t: swap_citation(t, snippets)}
        for name, fn in attacks.items():
            mutated = fn(text)
            if mutated == text:
                continue  # 这条回复里没有可扰动的内容（比如没有分歧项）
            caught[name][1] += 1
            caught[name][0] += blocked(mutated)
        benign_total += 1
        benign_blocked += blocked(benign(text))

    result["rewrite_attacks"] = {k: f"{c}/{t}" for k, (c, t) in caught.items()}
    result["benign_rewrites_blocked"] = f"{benign_blocked}/{benign_total}"
    result["note"] = "模拟评论 + 规则构造的扰动，只说明机制，不是线上幻觉率"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
