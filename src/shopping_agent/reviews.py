"""评论证据：按卖点检索评论片段，生成必须带引用的推荐理由，并校验引用。

商品字段只能说明“商品宣称有降噪”，评论才能说明“买家觉得降噪好不好”。这里做三件事：

1. 选方面（aspect）：用户本轮明确要求的属性（“要降噪的”）优先，其次是查询里的场景词
   （“送人”-> 送礼），最后补商品自己标注的卖点，最多 max_aspects 个；
2. 检索 + 聚合：每个方面在该商品的评论里做 BM25 + 向量混合检索（复用记忆检索的 RRF），
   再按关键词确认相关性，统计好评 / 差评，给出结论：
     supported    有好评，且好评多于差评，可以当卖点讲；
     disputed     差评不少于好评，不能当卖点，只能作为提示；
     no_evidence  没有相关评论，不提；
3. 引用约束：推荐理由的每一句都要带 [评论 ID]，引号里的原话必须能在被引用的评论里找到；
   引用差评的句子必须带“分歧 / 差评 / 但”这类转折词，不能把差评当卖点。
   模板生成的理由天然满足；LLM 改写后由 citation_check 校验，不满足就退回模板。

数据是模拟的（data/reviews.json，由 data/gen_reviews.py 生成）。真实评论没有现成的方面标签，
所以运行时只用评论正文做检索和相关性判断，数据里的 aspect 字段只给评测用。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .agent_memory.search import hybrid_search

# 方面 -> 检索词。第一个词是展示名；评论里出现其中任一个词，才算和这个方面相关
ASPECT_TERMS: dict[str, tuple[str, ...]] = {
    "降噪": ("降噪", "安静", "吵"),
    "长续航": ("续航", "充电器", "充一次", "电池", "掉电", "有电"),
    "轻量": ("轻", "重量", "分量", "偏重"),
    "防水": ("防水", "进水", "淋湿", "出汗"),
    "缓震": ("缓震", "脚感", "膝盖", "落地"),
    "热插拔": ("热插拔", "换轴"),
    "5G": ("5G", "信号", "网速"),
    "无线充电": ("无线充电",),
    "音质": ("音质", "低音", "高音", "人声"),
    "佩戴": ("佩戴", "耳塞", "戴久"),
    "拍照": ("拍照", "夜景", "人像"),
    "手感": ("手感", "键程", "敲击"),
    "做工": ("做工", "毛刺", "毛边", "细节"),
    "送礼": ("送", "礼物", "包装", "盒子"),
    "性价比": ("性价比", "价位", "值", "贵"),
}

# 查询里的场景词 -> 方面
SCENARIO_WORDS: dict[str, tuple[str, ...]] = {
    "送礼": ("送人", "送礼", "礼物", "送给", "送朋友", "送男朋友", "送女朋友", "生日", "送爸", "送妈"),
    "性价比": ("性价比", "划算", "便宜点", "实惠"),
    "音质": ("音质", "听歌"),
    "佩戴": ("佩戴", "跑步戴", "不掉"),
    "拍照": ("拍照", "摄影", "拍照好"),
    "手感": ("手感", "打字"),
}

HEDGE_WORDS = ("分歧", "差评", "但", "不过", "注意", "有人反映", "也有")
# 讲评论的行：这些行里的引号原话必须带引用。商品名也用「」，所以不能只看引号
REVIEW_WORDS = ("评价", "评论", "买家", "好评", "差评", "用户说", "反馈")
CITE_RE = re.compile(r"\[(R-[A-Za-z0-9]+-\d+)\]")
QUOTE_RE = re.compile(r"「([^」]+)」")
POSITIVE_MIN_RATING = 4


def load_reviews(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for r in json.loads(p.read_text(encoding="utf-8")):
        out.setdefault(r["product_id"], []).append(r)
    return out


def select_aspects(query: str, user_features: list[str], product_features: list[str], max_aspects: int = 3) -> list[str]:
    """用户明确要求的 > 场景词推出来的 > 商品自己的卖点。"""
    aspects = [f for f in user_features if f in ASPECT_TERMS]
    aspects += [a for a, words in SCENARIO_WORDS.items() if any(w in query for w in words)]
    aspects += [f for f in product_features if f in ASPECT_TERMS]
    return list(dict.fromkeys(aspects))[:max_aspects]


def is_positive(r: dict[str, Any]) -> bool:
    return r["rating"] >= POSITIVE_MIN_RATING


def aspect_evidence(reviews: list[dict[str, Any]], aspect: str, k: int = 2) -> dict[str, Any]:
    terms = ASPECT_TERMS.get(aspect, (aspect,))
    relevant = [r for r in reviews if any(t in r["text"] for t in terms)]
    pos = [r for r in relevant if is_positive(r)]
    neg = [r for r in relevant if not is_positive(r)]
    if not relevant:
        verdict = "no_evidence"
    elif pos and len(pos) > len(neg):
        verdict = "supported"
    else:
        verdict = "disputed"
    # 检索排序：混合检索给相关评论排名；supported 优先展示好评，disputed 优先展示差评
    texts = [r["text"] for r in relevant]
    order = [i for i, _ in hybrid_search(" ".join(terms), texts, k=len(texts))] if texts else []
    order += [i for i in range(len(relevant)) if i not in order]
    ranked = [relevant[i] for i in order]
    prefer_pos = verdict == "supported"
    ranked.sort(key=lambda r: is_positive(r) != prefer_pos)  # 稳定排序：同极性内保持检索名次
    return {
        "aspect": aspect,
        "verdict": verdict,
        "pos": len(pos),
        "neg": len(neg),
        "snippets": [{"id": r["id"], "rating": r["rating"], "text": r["text"]} for r in ranked[:k]],
    }


def review_evidence(reviews: list[dict[str, Any]], aspects: list[str], k: int = 2) -> dict[str, Any]:
    items = [aspect_evidence(reviews, a, k) for a in aspects]
    return {"aspects": items, "total_reviews": len(reviews)}


def reason_lines(ev: dict[str, Any]) -> list[str]:
    """模板版推荐理由：好评支撑的卖点一行，有分歧的一行。每句都带引用。"""
    good, mixed = [], []
    for a in ev.get("aspects", []):
        if not a["snippets"]:
            continue
        s = a["snippets"][0]
        if a["verdict"] == "supported":
            good.append(f"{a['aspect']}「{s['text']}」[{s['id']}]")
        elif a["verdict"] == "disputed":
            mixed.append(f"{a['aspect']} {a['pos']} 条好评 / {a['neg']} 条差评，如「{s['text']}」[{s['id']}]")
    out = []
    if good:
        out.append("   买家评价：" + "；".join(good))
    if mixed:
        out.append("   评价有分歧：" + "；".join(mixed))
    return out


def snippet_index(evs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """本轮检索到的所有评论片段：ID -> 片段。引用只能指向这里面的评论。"""
    return {s["id"]: s for ev in evs.values() for a in ev.get("aspects", []) for s in a["snippets"]}


def citation_check(text: str, snippets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """逐行检查评论引用，返回问题列表（空列表 = 通过）。

    - unknown_citation：引用的评论 ID 不在本轮检索结果里（模型编造的引用）；
    - missing_citation：讲评论的一行（带“评价 / 买家 / 好评”等词）有引号原话但没有任何引用；
    - unsupported_quote：引号里的话在这一行引用的评论里都找不到；
    - polarity_mismatch：引用了差评，但这一行没有任何转折 / 提示词，等于把差评当卖点讲。
    """
    problems: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines(), 1):
        cited = CITE_RE.findall(line)
        quotes = QUOTE_RE.findall(line)
        for cid in cited:
            if cid not in snippets:
                problems.append({"line": n, "type": "unknown_citation", "id": cid})
        known = [snippets[c] for c in cited if c in snippets]
        if quotes and not cited and any(w in line for w in REVIEW_WORDS):
            problems.append({"line": n, "type": "missing_citation", "quote": quotes[0]})
        for q in quotes:
            if known and not any(q in s["text"] for s in known):
                problems.append({"line": n, "type": "unsupported_quote", "quote": q})
        if any(s["rating"] < POSITIVE_MIN_RATING for s in known) and not any(w in line for w in HEDGE_WORDS):
            problems.append({"line": n, "type": "polarity_mismatch"})
    return problems
