"""需求理解：意图识别 + 约束抽取 + 跨轮约束合并。

两种解析器输出同一个 TurnParse：
- rule_parse：规则版，离线可跑、结果确定，用作基线和 LLM 失败时的兜底；
- llm.LLMParser：大模型版（见 llm.py），覆盖更多口语表达。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel

Intent = Literal["search", "compare", "detail", "clarify"]

CATEGORIES: dict[str, tuple[str, ...]] = {
    "蓝牙耳机": ("耳机", "耳塞", "airpods"),
    "手机": ("手机", "iphone"),
    "跑鞋": ("跑鞋", "跑步鞋", "运动鞋"),
    "机械键盘": ("键盘",),
}

BRANDS: dict[str, tuple[str, ...]] = {
    "小米": ("小米", "红米", "redmi", "xiaomi"),
    "华为": ("华为", "huawei"),
    "苹果": ("苹果", "apple", "iphone", "airpods"),
    "索尼": ("索尼", "sony"),
    "漫步者": ("漫步者", "edifier"),
    "倍思": ("倍思", "baseus"),
    "OPPO": ("oppo",),
    "vivo": ("vivo", "iqoo"),
    "一加": ("一加", "oneplus"),
    "耐克": ("耐克", "nike"),
    "阿迪达斯": ("阿迪达斯", "阿迪", "adidas"),
    "李宁": ("李宁",),
    "安踏": ("安踏",),
    "罗技": ("罗技", "logitech"),
    "樱桃": ("樱桃", "cherry"),
    "达尔优": ("达尔优",),
    "阿米洛": ("阿米洛", "varmilo"),
}

FEATURES = ("降噪", "防水", "轻量", "缓震", "热插拔", "无线充电", "5G", "长续航")

CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}

COMPARE_WORDS = ("对比", "比较", "哪个好", "哪个更", "区别", "差别", "vs", "pk")
CHEAPER_WORDS = ("更便宜", "便宜点", "便宜一些", "再便宜", "便宜一点")
EXCLUDE_RE = re.compile(r"(?:不要|别要|不考虑|除了|排除|不喜欢|不买)([^，,。！!？?；;]*)")
RANGE_RE = re.compile(r"(\d+)\s*(?:元|块)?\s*(?:到|至|-|~)\s*(\d+)")
MAX_RES = (
    re.compile(r"(\d+)\s*(?:元|块)?\s*(?:以内|以下|之内|内)"),
    re.compile(r"(?:不超过|不高于|低于|预算)\s*(\d+)"),
)
MIN_RE = re.compile(r"(\d+)\s*(?:元|块)?\s*以上")
REF_RE = re.compile(r"第\s*([一二两三四五1-5])\s*(?:个|款|台|双|把)?")


class TurnParse(BaseModel):
    """单轮解析结果（只描述这一句话，跨轮合并在 merge_constraints 里做）。"""

    intent: Intent = "search"
    category: str | None = None
    budget_min: int | None = None
    budget_max: int | None = None
    brands_include: list[str] = []
    brands_exclude: list[str] = []
    features: list[str] = []
    refs: list[int] = []  # 1-based，指向上一次展示的商品
    cheaper: bool = False


def _find_brands(text: str) -> list[str]:
    low = text.lower()
    return [b for b, aliases in BRANDS.items() if any(a in low for a in aliases)]


def rule_parse(text: str) -> TurnParse:
    low = text.lower()
    p = TurnParse()

    for cat, kws in CATEGORIES.items():
        if any(k in low for k in kws):
            p.category = cat
            break

    if m := RANGE_RE.search(text):
        p.budget_min, p.budget_max = sorted((int(m.group(1)), int(m.group(2))))
    else:
        for r in MAX_RES:
            if m := r.search(text):
                p.budget_max = int(m.group(1))
                break
        if m := MIN_RE.search(text):
            p.budget_min = int(m.group(1))

    excluded: list[str] = []
    for m in EXCLUDE_RE.finditer(text):
        span = re.split(r"要|想|只", m.group(1), maxsplit=1)[0]
        excluded += _find_brands(span)
    p.brands_exclude = sorted(set(excluded))
    p.brands_include = [b for b in _find_brands(text) if b not in p.brands_exclude]
    p.features = [f for f in FEATURES if f.lower() in low]

    for m in REF_RE.finditer(text):
        tok = m.group(1)
        p.refs.append(CN_NUM.get(tok) or int(tok))
    if not p.refs and any(w in text for w in ("这两个", "这俩", "前两个")):
        p.refs = [1, 2]

    p.cheaper = any(w in text for w in CHEAPER_WORDS)

    if any(w in low for w in COMPARE_WORDS):
        p.intent = "compare"
    elif len(p.refs) == 1 and not p.cheaper:
        p.intent = "detail"
    else:
        p.intent = "search"
    return p


def merge_constraints(prev: dict[str, Any], p: TurnParse) -> dict[str, Any]:
    """把本轮解析结果合并进跨轮约束。

    - 切换品类：旧约束（预算、品牌、属性）不再适用，整体重置；
    - 预算：本轮给了就覆盖；
    - 排除品牌：取并集（用户说过不要的，后面一直不要）；
    - 指定品牌：本轮给了就覆盖，并从排除列表里移除；
    - 属性：取并集。
    """
    if p.category and prev.get("category") and p.category != prev["category"]:
        base: dict[str, Any] = {}
    else:
        base = {k: (list(v) if isinstance(v, list) else v) for k, v in prev.items()}

    if p.category:
        base["category"] = p.category
    if p.budget_min is not None:
        base["budget_min"] = p.budget_min
    if p.budget_max is not None:
        base["budget_max"] = p.budget_max

    exclude = set(base.get("brands_exclude", [])) | set(p.brands_exclude)
    if p.brands_include:
        base["brands_include"] = list(p.brands_include)
        exclude -= set(p.brands_include)
    base["brands_exclude"] = sorted(exclude)
    base["features"] = sorted(set(base.get("features", [])) | set(p.features))
    base.setdefault("brands_include", [])
    return base
