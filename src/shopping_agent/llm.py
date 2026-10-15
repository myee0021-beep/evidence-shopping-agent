"""LLM 版意图识别与约束抽取（OpenAI 兼容接口，可接 Qwen / DeepSeek 等）。

这里用的是“提示词 + JSON 输出 + pydantic 校验”的方式，而不是原生 function calling：
- 任何兼容 OpenAI 接口的模型都能用；
- 解析失败时把错误回传给模型重试一次，仍失败就返回 None，由规则解析兜底。
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .understanding import BRANDS, CATEGORIES, FEATURES, TurnParse, rule_parse

SYSTEM_PROMPT = f"""你是购物助手的需求理解模块。把用户这一句话解析成 JSON，只描述这一句话本身。

字段：
- intent: "search" | "compare" | "detail" | "clarify"
  - compare：要求对比多个商品；detail：追问某一个商品；clarify：信息不足以检索
- category: {list(CATEGORIES)} 之一，或 null
- budget_min / budget_max: 整数（元）或 null
- brands_include / brands_exclude: 品牌列表，取值范围 {list(BRANDS)}
- features: 取值范围 {list(FEATURES)}
- refs: 用户指代的商品序号（1 开始），如“第二个” -> [2]，“这两个” -> [1, 2]
- cheaper: 用户是否要求“更便宜的”

只输出 JSON，不要输出其他内容。"""


class LLMParser:
    def __init__(self, api_key: str, base_url: str | None, model: str):
        from openai import OpenAI  # 可选依赖，只有启用 LLM 时才需要

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    # harness 看到这个标记，会把压缩后的对话窗口（摘要 + 最近几轮）一起传进来
    wants_history = True

    def __call__(
        self,
        text: str,
        prev: dict[str, Any],
        shown_titles: list[str],
        history: list[dict[str, Any]] | None = None,
    ) -> TurnParse | None:
        context = {
            "已有约束": prev,
            "上一次展示的商品": [f"{i + 1}. {t}" for i, t in enumerate(shown_titles)],
        }
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 只带 role / content，工具调用细节和归档引用不需要给解析器看
        messages += [{"role": m["role"], "content": str(m.get("content", ""))} for m in history or []]
        messages.append(
            {"role": "user", "content": f"上下文：{json.dumps(context, ensure_ascii=False)}\n用户：{text}"}
        )
        for _ in range(2):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            _add_usage(self.usage, resp)
            raw = resp.choices[0].message.content or ""
            try:
                return TurnParse(**json.loads(raw))
            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"输出不合法：{e}。请只输出符合字段要求的 JSON。"},
                ]
        return None


def _add_usage(usage: dict[str, int], resp: Any) -> None:
    usage["calls"] += 1
    u = getattr(resp, "usage", None)
    if u is not None:
        usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0


# 规则搞不定的口语信号：中文数字 / 3k、口语否定、按品牌或位置指代
COLLOQUIAL_RE = re.compile(
    r"[一二两三四五六七八九十百千万]+\s*(?:块|元|千|百|万)|\d+(?:\.\d+)?\s*[kKwW千万]"
    r"|别给|不是|不需要|不用|不想要|不太要|那个.{0,6}的|最后一个|倒数|上一个|刚才"
)


class HybridParser:
    """规则优先，规则搞不定才调 LLM：省成本、省时延，命中规则的句子零延迟。

    什么时候调 LLM：
    - 规则什么都没解析出来（没有品类、预算、品牌、属性、指代）；
    - 句子里有规则处理不了的口语信号（“两千”“3k”“别给我推”“不是苹果的”“那个华为的”）。
    llm_rate 记录 LLM 调用占比，评测时一起看成本。
    """

    def __init__(self, llm: Any):
        self.llm = llm
        self.wants_history = getattr(llm, "wants_history", False)
        self.stats = {"turns": 0, "llm_calls": 0}

    @property
    def usage(self) -> dict[str, Any]:
        u = dict(getattr(self.llm, "usage", {}))
        u["llm_rate"] = round(self.stats["llm_calls"] / self.stats["turns"], 4) if self.stats["turns"] else 0.0
        return {**u, **self.stats}

    @staticmethod
    def needs_llm(text: str, p: TurnParse) -> bool:
        empty = not (p.category or p.budget_max or p.budget_min or p.brands_include or p.brands_exclude
                     or p.features or p.refs or p.cheaper or p.intent == "compare")
        return empty or bool(COLLOQUIAL_RE.search(text))

    def __call__(self, text: str, prev: dict[str, Any], shown_titles: list[str], **kw: Any) -> TurnParse | None:
        self.stats["turns"] += 1
        p = rule_parse(text)
        if not self.needs_llm(text, p):
            return p
        self.stats["llm_calls"] += 1
        if not self.wants_history:
            kw.pop("history", None)
        return self.llm(text, prev, shown_titles, **kw) or p


SUMMARY_PROMPT = """你在为购物助手压缩对话历史。把“已有摘要”和“新增对话”合并成一份新摘要，分四类，每条一行，以“- ”开头：
- 已确认事实：用户明确说过的信息
- 用户偏好：喜欢 / 不喜欢 / 预算 / 属性要求
- 已做决定：已推荐、已对比、已排除的商品
- 未完成事项：用户还在等的回答

要求：数字、品牌、型号、价格按原文保留，不要改写或取整；不要编造对话里没有的内容；只输出摘要正文。"""


class LLMSummarizer:
    """增量摘要：已有摘要 + 新切掉的对话 -> 新摘要。失败直接抛异常，由 context.compact 标出缺口。"""

    def __init__(self, api_key: str, base_url: str | None, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def __call__(self, prev_summary: str, head: list[dict[str, Any]]) -> str:
        from .context import render

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": f"已有摘要：\n{prev_summary or '（无）'}\n\n新增对话：\n{render(head)}"},
            ],
        )
        return resp.choices[0].message.content or ""
