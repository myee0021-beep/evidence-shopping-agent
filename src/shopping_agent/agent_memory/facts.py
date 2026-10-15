"""L1 原子事实的抽取规则。

只从“本轮约束 diff”里抽，不从完整约束里抽：
- 约束被跨轮沿用，不算用户又说了一遍，不能拿来累加确认次数；
- 从画像预填进来的约束、“更便宜的”推出来的预算，都不是用户的原话，不进记忆。

作用范围（scope）：
- long_term：用户本人的稳定偏好，确认够次数后进入画像；
- session：一次性需求，比如“这次给我爸买”，只在本次会话生效，不进画像，避免污染以后的推荐。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

Scope = Literal["long_term", "session"]

ONE_OFF_RE = re.compile(
    r"(?:给|帮|替|送)(?:我)?(?:爸|妈|父亲|母亲|老公|老婆|男朋友|女朋友|朋友|同事|孩子|儿子|女儿|爷爷|奶奶|长辈|老人|家人)"
    r"|送人|送礼|礼物|这次|这一次|临时"
)

# 显式的否定偏好（“不要小米”）说一次就算数；预算、属性这类可能是一次性的，要确认 2 次才进画像
CONFIRM_THRESHOLD = {"brand_dislike": 1, "brand_like": 2, "budget_max": 2, "budget_min": 2, "feature": 2}


@dataclass
class FactDraft:
    kind: str
    key: str
    value: Any
    text: str
    category: str
    scope: Scope


def detect_scope(query: str) -> Scope:
    return "session" if ONE_OFF_RE.search(query) else "long_term"


def extract_facts(query: str, diff: dict[str, Any], constraints: dict[str, Any]) -> list[FactDraft]:
    cat = constraints.get("category")
    if not cat:
        return []
    scope = detect_scope(query)
    out: list[FactDraft] = []
    if "category" in diff.get("changed", {}):
        # 切换品类时约束整体重置，新品类下的约束都算本轮新说的
        diff = {"added": {k: v for k, v in constraints.items() if v not in (None, [], "")}}

    def new_items(k: str) -> list[Any]:
        if k in diff.get("added", {}):
            return list(diff["added"][k])
        if k in diff.get("changed", {}):
            ch = diff["changed"][k]
            return [x for x in ch["to"] if x not in (ch["from"] or [])]
        return []

    def new_value(k: str) -> Any:
        if k in diff.get("added", {}):
            return diff["added"][k]
        if k in diff.get("changed", {}):
            return diff["changed"][k]["to"]
        return None

    for b in new_items("brands_exclude"):
        out.append(FactDraft("brand_dislike", f"brand_dislike|{cat}|{b}", True, f"{cat}：不要{b}", cat, scope))
    for b in new_items("brands_include"):
        # 明确指定了某品牌：之前的“不喜欢”被覆盖
        out.append(FactDraft("brand_dislike", f"brand_dislike|{cat}|{b}", False, f"{cat}：可以考虑{b}", cat, scope))
        out.append(FactDraft("brand_like", f"brand_like|{cat}|{b}", True, f"{cat}：想要{b}", cat, scope))
    for f in new_items("features"):
        out.append(FactDraft("feature", f"feature|{cat}|{f}", True, f"{cat}：想要{f}", cat, scope))
    if (v := new_value("budget_max")) is not None:
        out.append(FactDraft("budget_max", f"budget_max|{cat}", v, f"{cat}：预算 {v} 元以内", cat, scope))
    if (v := new_value("budget_min")) is not None:
        out.append(FactDraft("budget_min", f"budget_min|{cat}", v, f"{cat}：预算 {v} 元以上", cat, scope))
    return out
