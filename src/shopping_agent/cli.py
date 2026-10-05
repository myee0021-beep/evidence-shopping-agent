"""命令行对话。

    python -m shopping_agent.cli --user u1
    python -m shopping_agent.cli --user u1 --trace     # 每轮打印决策轨迹

对话中可用的命令：
    /new    开一个新会话（新的 thread_id，长期画像保留）
    /state  查看当前的结构化约束
    /trace  打开 / 关闭轨迹打印
    /quit   退出
"""

from __future__ import annotations

import argparse
import json
import uuid

from langgraph.store.memory import InMemoryStore

from .catalog import load_catalog
from .config import load_settings
from .harness import build_agent
from .memory import make_checkpointer
from .trace import replay


def make_parser(settings):
    if not settings.llm_api_key:
        return None
    from .llm import LLMParser

    return LLMParser(settings.llm_api_key, settings.llm_base_url, settings.llm_model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="demo_user")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    settings = load_settings()
    catalog = load_catalog(settings.catalog_path)
    store = InMemoryStore()
    show_trace = args.trace

    with make_checkpointer(settings.redis_url) as checkpointer:
        agent = build_agent(catalog, settings, store, checkpointer, make_parser(settings))
        thread = uuid.uuid4().hex[:8]
        backend = "Redis" if settings.redis_url else "内存"
        parser_name = "LLM" if settings.llm_api_key else "规则"
        print(f"会话 {thread}｜会话记忆：{backend}｜需求理解：{parser_name}。输入 /quit 退出。")
        while True:
            try:
                text = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/new":
                thread = uuid.uuid4().hex[:8]
                print(f"已开启新会话 {thread}")
                continue
            config = {"configurable": {"thread_id": thread}}
            if text == "/state":
                values = agent.get_state(config).values
                print(json.dumps(values.get("constraints", {}), ensure_ascii=False, indent=2))
                continue
            if text == "/trace":
                show_trace = not show_trace
                print(f"轨迹打印：{'开' if show_trace else '关'}")
                continue
            out = agent.invoke({"user_id": args.user, "query": text}, config)
            print(f"\n助手：{out['response']}")
            if show_trace:
                print("\n" + replay(out["trace"], out["turn"]))


if __name__ == "__main__":
    main()
