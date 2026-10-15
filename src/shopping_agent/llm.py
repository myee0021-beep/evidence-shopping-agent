"""LLM 版意图识别与约束抽取（OpenAI 兼容接口，可接 Qwen / DeepSeek 等）。

这里用的是“提示词 + JSON 输出 + pydantic 校验”的方式，而不是原生 function calling：
- 任何兼容 OpenAI 接口的模型都能用；
- 解析失败时把错误回传给模型重试一次，仍失败就返回 None，由规则解析兜底。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .understanding import BRANDS, CATEGORIES, FEATURES, TurnParse

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
            raw = resp.choices[0].message.content or ""
            try:
                return TurnParse(**json.loads(raw))
            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"输出不合法：{e}。请只输出符合字段要求的 JSON。"},
                ]
        return None


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
