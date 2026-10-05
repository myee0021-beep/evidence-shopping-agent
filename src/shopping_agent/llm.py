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

    def __call__(self, text: str, prev: dict[str, Any], shown_titles: list[str]) -> TurnParse | None:
        context = {
            "已有约束": prev,
            "上一次展示的商品": [f"{i + 1}. {t}" for i, t in enumerate(shown_titles)],
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"上下文：{json.dumps(context, ensure_ascii=False)}
用户：{text}"},
        ]
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
