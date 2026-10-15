"""ShoppingBench 任务解析：一句英文需求 -> 结构化的 TaskSpec。

和主链路一样分两套解析器，输出同一个结构：
- rule_parse_task：正则版。优惠券规则是模板生成的，正则就能全部解析；价格、服务词也比较固定；
  多商品拆分和属性词抽取只做了启发式，作为 LLM 失败时的兜底；
- LLMTaskParser：拆分商品、抽取属性词（size / color / material ...），JSON + pydantic 校验 + 重试。

硬约束和软约束的划分沿用主链路：
- 价格区间、服务（LazMall / 包邮 / 货到付款 / 闪购）、同店、优惠后预算 -> 硬约束，交给沙箱过滤或组合求解；
- 属性词（尺码、颜色、材质……）-> 证据层逐项核验，没命中就重规划检索，而不是放宽。
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

Task = Literal["product", "shop", "voucher"]


class Term(BaseModel):
    key: str | None = None  # 如 "size"、"color_family"；不确定时留空
    value: str


class ItemSpec(BaseModel):
    q: str
    price_min: float | None = None
    price_max: float | None = None
    services: list[str] = []
    terms: list[Term] = []

    def price_param(self) -> str | None:
        if self.price_min is None and self.price_max is None:
            return None
        lo = "" if self.price_min is None else f"{self.price_min:g}"
        hi = "" if self.price_max is None else f"{self.price_max:g}"
        return f"{lo}-{hi}"


class Voucher(BaseModel):
    voucher_type: Literal["platform", "shop"]
    threshold: float
    discount_type: Literal["fixed", "percentage"]
    face_value: float | None = None
    discount: float | None = None
    cap: float | None = None
    budget: float


class TaskSpec(BaseModel):
    task: Task
    items: list[ItemSpec]
    voucher: Voucher | None = None

    @property
    def same_shop(self) -> bool:
        """只有 shop 任务硬性要求同店。店铺券任务不要求同店：同店才能用券，但不用券也在预算内同样算成功。"""
        return self.task == "shop"


# ---------- 优惠券（模板化，正则全覆盖） ----------

BUDGET_RE = re.compile(r"budget is only `([\d.]+)`")
THRESHOLD_RE = re.compile(r"exceeds `([\d.]+)`")
FIXED_RE = re.compile(r"fixed discount of `([\d.]+)`")
PCT_RE = re.compile(r"percentage discount of `([\d.]+)%`(?: with a cap of `([\d.]+)`)?")


def split_voucher(query: str) -> tuple[str, str]:
    """返回 (需求正文, 预算和优惠券说明)。"""
    idx = query.find("My budget is only")
    if idx < 0:
        return query.strip(), ""
    return query[:idx].strip(), query[idx:]


def parse_voucher(text: str) -> Voucher | None:
    if not (b := BUDGET_RE.search(text)) or not (t := THRESHOLD_RE.search(text)):
        return None
    vtype = "shop" if "same shop" in text else "platform"
    if m := FIXED_RE.search(text):
        return Voucher(voucher_type=vtype, threshold=float(t.group(1)), discount_type="fixed",
                       face_value=float(m.group(1)), budget=float(b.group(1)))
    if m := PCT_RE.search(text):
        return Voucher(voucher_type=vtype, threshold=float(t.group(1)), discount_type="percentage",
                       discount=float(m.group(1)) / 100, cap=float(m.group(2)) if m.group(2) else None,
                       budget=float(b.group(1)))
    return None


# ---------- 规则解析 ----------

NUM = r"(\d+(?:\.\d+)?)"
CUR = r"(?:\s*(?:php|pesos?|₱))?"
BETWEEN_RES = (
    re.compile(rf"between\s*(?:php\s*|₱)?{NUM}{CUR}\s*(?:and|to|-)\s*(?:php\s*|₱)?{NUM}", re.I),
    re.compile(rf"from\s*(?:php\s*|₱)?{NUM}{CUR}\s*to\s*(?:php\s*|₱)?{NUM}", re.I),
)
MIN_RE = re.compile(rf"(?:above|over|more than|greater than|at least|exceeding|higher than)\s*(?:php\s*|₱)?{NUM}", re.I)
MAX_RE = re.compile(rf"(?:below|under|less than|up to|at most|no more than|cheaper than|lower than)\s*(?:php\s*|₱)?{NUM}{CUR}(?!\s*(?:kg|g|ml|l|cm|mm|m|months?|years?|pcs))", re.I)
SERVICE_WORDS = {
    "official": ("lazmall", "official store", "authenticity"),
    "freeShipping": ("free shipping", "free delivery", "complimentary shipping", "ships for free"),
    "COD": ("cash on delivery", "cod"),
    "flashsale": ("lazflash", "flash sale", "flashsale", "flash deal"),
}
NUMBERED_RE = re.compile(r"\(\d\)\s*")
ORDINAL_RE = re.compile(r"\b(?:first|second|third|fourth|fifth|lastly|finally)\s*,\s*", re.I)
# 没有显式序号时的启发式切分：“, a …”“ and an …”“; …”“another item is …”
# 排除 “a price / a cap / a plain design” 这类修饰语，避免把同一件商品切开
NOT_ITEM = r"(?!price|cap|budget|deal|plain|simple|relaxed|regular|single|set of|pack of|size|color|length|weight)"
LOOSE_SPLIT_RE = re.compile(
    rf",?\s*(?:and\s+)?(?:also\s+(?:(?:searching|looking)\s+for\s+)?|another (?:item )?is\s+|the other is\s+)"
    rf"|,\s*(?:and\s+)?(?:plus\s+)?(?=(?:an?|one)\s+{NOT_ITEM})"
    rf"|\s+and\s+(?=(?:an?|another)\s+{NOT_ITEM})"
    r"|;\s*|\.\s+(?:also|another)\s+",
    re.I,
)
NEW_ITEM_SENT_RE = re.compile(
    r"^(?:also|additionally|plus|and|lastly|finally|i'?m also|i also|i am also)\b,?\s*"
    r"(?:(?:i'?m |i am )?(?:searching for|looking for|interested in|need|want)\s+)?"
    r"|^(?:i'?m |i am |i )?(?:also )?(?:searching for|looking for|interested in|need|want)\s+",
    re.I,
)
PREAMBLE_RE = re.compile(
    r"^.*?\b(?:these (?:items|products)|the following|following items|items|that include|offering|offers?|sells?|selling|"
    r"where|looking for|searching for|need|want)\b\s*:?\s*(?:both\s+)?(?:one item (?:has|is)\s+)?",
    re.I,
)


def split_items(body: str) -> list[str]:
    head = re.split(r"\s+(?:The shop|Please|The first is|For the \w+, )", body, maxsplit=1)[0]
    if ":" in head[:120]:
        head = head.split(":", 1)[1]
    else:
        head = PREAMBLE_RE.sub("", head, count=1)
    if NUMBERED_RE.search(head):
        parts = NUMBERED_RE.split(head)
    elif ORDINAL_RE.search(head):
        parts = ORDINAL_RE.split(head)
    else:
        # 先按句子分：以 Also / Additionally / I'm also 开头的句子是新商品，
        # 其余句子（“It should also allow COD”）是对上一件商品的补充
        groups: list[str] = []
        for sent in re.split(r"(?<=[.!?])\s+", head):
            if groups and not NEW_ITEM_SENT_RE.match(sent):
                groups[-1] += " " + sent
            else:
                groups.append(NEW_ITEM_SENT_RE.sub("", sent))
        parts = [x for g in groups for x in LOOSE_SPLIT_RE.split(g)]
    return [p.strip(" ,.;") for p in parts if p and len(p.strip(" ,.;")) > 3]


LEAD_RE = re.compile(
    r"^(?:i'?m |i am )?(?:looking for|find|show me|searching for|search for|i need|i want|need|want)\s+"
    r"(?:a |an |some |shops? (?:that |which )?(?:offers?|sells?|offering|selling) (?:these items:?\s*|both\s*)?)?",
    re.I,
)


def _price(text: str) -> tuple[float | None, float | None]:
    for rx in BETWEEN_RES:
        if m := rx.search(text):
            lo, hi = sorted((float(m.group(1)), float(m.group(2))))
            return lo, hi
    lo = float(m.group(1)) if (m := MIN_RE.search(text)) else None
    hi = float(m.group(1)) if (m := MAX_RE.search(text)) else None
    return lo, hi


def _services(text: str) -> list[str]:
    low = text.lower()
    return [s for s, words in SERVICE_WORDS.items() if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words)]


def _clean_q(text: str) -> str:
    t = LEAD_RE.sub("", text.strip())
    for rx in (*BETWEEN_RES, MIN_RE, MAX_RE):
        t = rx.sub(" ", t)
    t = re.sub(r"\b(priced|costing|that costs?|with a price|price|php|pesos?|and available|available with|with)\b", " ", t, flags=re.I)
    for words in SERVICE_WORDS.values():
        for w in words:
            t = re.sub(rf"\b{re.escape(w)}\b", " ", t, flags=re.I)
    t = re.sub(r"[^\w\s/'.-]", " ", t)
    return re.sub(r"\s+", " ", t).strip(" .,-")[:120] or text[:120]


def _item(text: str) -> ItemSpec:
    lo, hi = _price(text)
    return ItemSpec(q=_clean_q(text), price_min=lo, price_max=hi, services=_services(text))


def rule_parse_task(query: str, task: Task) -> TaskSpec:
    body, voucher_text = split_voucher(query)
    voucher = parse_voucher(voucher_text) if voucher_text else None
    if task == "product":
        return TaskSpec(task=task, items=[_item(body)], voucher=voucher)
    items = [_item(p) for p in split_items(body)] or [_item(body)]
    return TaskSpec(task=task, items=items, voucher=voucher)


# ---------- LLM 解析 ----------

SYSTEM_PROMPT = """You parse a shopping request into JSON for a product search agent.

Output: {"items": [{"q": str, "price_min": number|null, "price_max": number|null,
                    "services": [str], "terms": [{"key": str|null, "value": str}]}]}

Rules:
- One entry in "items" per distinct product the user wants, in the order they are mentioned.
- "q": a short keyword search query for that product (3-8 words, no prices, no service words).
- price_min / price_max: only if the user states a price bound for THAT product.
- services: subset of ["official" (LazMall), "freeShipping", "COD" (cash on delivery), "flashsale" (LazFlash)].
- terms: concrete attribute or SKU values the product must have (size, color, material, brand, pattern,
  model, capacity, age range, flavor ...). Lowercase. Use a key like "size", "color_family", "material",
  "brand" if obvious, else null. Do not repeat words already covered by price or services.
- Ignore any budget / voucher paragraph; it is handled elsewhere.
Output JSON only."""


class LLMTaskParser:
    def __init__(self, api_key: str, base_url: str | None, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def __call__(self, query: str, task: Task) -> TaskSpec | None:
        body, voucher_text = split_voucher(query)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body}]
        for _ in range(2):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0, response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content or ""
            try:
                data: dict[str, Any] = json.loads(raw)
                items = [ItemSpec(**it) for it in data["items"]]
                if not items:
                    raise ValueError("no items")
                if task == "product":
                    items = items[:1]
                voucher = parse_voucher(voucher_text) if voucher_text else None
                return TaskSpec(task=task, items=items, voucher=voucher)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as e:
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Invalid output: {e}. Output JSON matching the schema only."},
                ]
        return None
