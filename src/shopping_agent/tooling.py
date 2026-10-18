"""工具契约的公共基类：主链路的 skill 和 ShoppingBench 沙箱工具共用一套声明方式。

一个工具只需要声明四样东西：
- name / description：给模型看的名字和用途；
- args_model：pydantic 参数模型，调用前统一校验，也用来导出 JSON schema；
- idempotent：只读工具才允许重试和缓存去重，写操作（加购、下单）两样都不能做。

执行方式由各自的调用层决定：主链路 skill 带 fn（见 skills.Skill），沙箱工具按名字分派到
Sandbox 的同名方法（见 bench.tools）。这样加一个新意图对应的工具，只需要写一个参数模型和一个
实现，校验、schema 导出、trace 记录都不用重复写。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


class ToolArgsError(ValueError):
    """参数不合法。重试也没用，调用层应直接返回错误，而不是重试。"""


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    idempotent: bool = True  # 只读工具可以安全重试 / 去重；写操作不能

    def schema(self) -> dict[str, Any]:
        """导出成 OpenAI 兼容的 function calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    def validate(self, args: dict[str, Any]) -> BaseModel:
        try:
            return self.args_model(**args)
        except ValidationError as e:
            err = e.errors()[0]
            loc = ".".join(str(x) for x in err.get("loc", ())) or "args"
            raise ToolArgsError(f"invalid args: {loc}: {err['msg']}") from e
