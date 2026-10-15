"""隐私与安全：敏感信息脱敏、套话类意图拦截。

三道关：
1. 写入长期记忆前脱敏（redact_pii）：手机号、邮箱、身份证号、银行卡号不进跨会话的 L0 归档。
   会话内的压缩归档保留原文（只有本会话能读，recall 需要原文），长期记忆一律脱敏；
2. 套话类意图拦截（guard_reason）：问供应商、进货价、佣金、系统提示词、其他用户信息的，
   在理解阶段直接走 guard 路由，不进检索，也不调任何 skill；
3. 输出只能来自证据：见 evidence.fact_check，回复里的价格必须和证据层逐项一致。

另外两道在别处：
- 数据隔离：归档和记忆按 (tenant, user_id) 分命名空间，session 级记录只有同一会话能读（archive.py）；
- 字段白名单：商品详情 skill 只返回 Product 模型里的字段，供应商、成本价这类内部字段模型根本拿不到。
"""

from __future__ import annotations

import re

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("身份证号", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("银行卡号", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("邮箱", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
)

GUARD_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("商业信息", re.compile(r"供应商|供货商|进货价|成本价|出厂价|佣金|返点|毛利|采购价|底价")),
    ("系统信息", re.compile(r"系统提示|提示词|system prompt|prompt|你的指令|忽略(?:之前|上面|以上)的?(?:指令|要求)", re.I)),
    ("他人信息", re.compile(r"(?:别人|其他用户|其他人|他人)的?(?:订单|地址|手机|电话|偏好|记录|购买)")),
)

GUARD_REPLY = "这个问题我没法回答。我可以帮你按预算、品牌和功能挑商品，或者对比你感兴趣的几款。"


def redact_pii(text: str) -> tuple[str, list[str]]:
    """返回 (脱敏后的文本, 命中的类型)。"""
    hits = []
    for label, rx in PII_PATTERNS:
        if rx.search(text):
            hits.append(label)
            text = rx.sub(f"[{label}]", text)
    return text, hits


def guard_reason(text: str) -> str | None:
    for label, rx in GUARD_RULES:
        if rx.search(text):
            return label
    return None
