"""上下文压缩：按 token 预算触发，而不是“固定保留最近 N 轮”。

固定保留 5 轮的两个问题：
1. 一轮对比就可能带回几个商品的完整详情，5 轮还没到，窗口先满了；
2. 第 6 轮时第 1 轮说的“预算 500 以内”被整轮删掉，约束直接丢失。

这里的做法分三层：
1. 固定注入，不参与压缩：系统指令、工具定义、结构化购物状态（约束、已展示商品快照）。
   约束放在 state 里每轮原样带着走，从根上不会被摘要丢掉。
2. 对话历史按预算压缩：总量超过 trigger 才动手；从后往前保留最近一段，切点只能落在
   某个用户轮次的开头，所以工具调用和它的结果永远不会被拆开；被切掉的部分先归档（L0），
   再做增量摘要。
3. 超长工具结果卸载：完整原文归档，上下文里只留预览和 ref_id，需要时 recall_context 取回。

失败处理（宁可不压，也不能压错）：
- 归档失败：停止裁剪，原样返回；如果原样放不下，build_window 直接拒绝发送；
- 摘要失败：不把半份摘要当完整结果，在窗口里明确标出“第 a–b 轮摘要缺失”，并保留原文 ref；
- 压缩后仍放不下：抛 ContextOverflowError，由调用方降级（比如只带状态、不带历史）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .archive import ArchiveStore, Caller

Message = dict[str, Any]
Summarizer = Callable[[str, list[Message]], str]

PER_MESSAGE_OVERHEAD = 4  # role、分隔符等格式开销
CJK_RE = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")
SUMMARY_NAME = "history_summary"


class ContextOverflowError(RuntimeError):
    pass


# ---------- token 计数 ----------


def count_text_tokens(text: str) -> int:
    """近似计数：中文按 1 字 1 token，其余按 4 字符 1 token。

    只用来做触发判断，不需要和计费完全一致；接真实模型时可以换成对应的 tokenizer。
    """
    if not text:
        return 0
    cjk = len(CJK_RE.findall(text))
    rest = len(text) - cjk
    return cjk + (rest + 3) // 4


def count_message_tokens(m: Message) -> int:
    """角色字段、工具调用字段都要算进去，否则带 tool_calls 的消息会被严重低估。"""
    n = PER_MESSAGE_OVERHEAD + count_text_tokens(str(m.get("content") or ""))
    if m.get("tool_calls"):
        n += count_text_tokens(json.dumps(m["tool_calls"], ensure_ascii=False))
    if m.get("tool_call_id"):
        n += count_text_tokens(m["tool_call_id"])
    if m.get("name"):
        n += count_text_tokens(m["name"])
    return n


def count_tokens(messages: list[Message]) -> int:
    return sum(count_message_tokens(m) for m in messages)


# ---------- 安全切分 ----------


def is_summary(m: Message) -> bool:
    return m.get("role") == "system" and m.get("name") == SUMMARY_NAME


def turn_starts(messages: list[Message]) -> list[int]:
    """每个用户轮次开头的下标。用户消息之后到下一条用户消息之前，都属于同一轮。"""
    return [i for i, m in enumerate(messages) if m.get("role") == "user"]


def safe_cut_index(messages: list[Message], keep_tokens: int) -> int:
    """从后往前保留约 keep_tokens 的消息，返回第一条保留消息的下标。

    切点只落在用户轮次开头：
    - assistant 的 tool_calls 和对应的 tool 结果都在同一轮里，不会被拆开；
    - 并行工具调用（一条 assistant 带多个 tool_calls）同理；
    - 最后一轮永远保留，哪怕它本身就超过 keep_tokens。
    返回 0 表示没有可以安全切掉的部分。
    """
    starts = [i for i in turn_starts(messages) if i > 0]
    if not starts:
        return 0
    suffix = 0
    cut = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        suffix += count_message_tokens(messages[i])
        if i in starts:
            cut = i
            if suffix >= keep_tokens:
                break
    if cut == len(messages):  # 只有一轮
        return 0
    # 最后一轮必须完整保留
    return min(cut, starts[-1])


def unpaired_tool_messages(messages: list[Message]) -> list[int]:
    """校验用：返回没有对应 tool_call 的 tool 消息下标（切分正确时应为空）。"""
    seen: set[str] = set()
    bad = []
    for i, m in enumerate(messages):
        for tc in m.get("tool_calls") or []:
            seen.add(tc.get("id", ""))
        if m.get("role") == "tool" and m.get("tool_call_id") not in seen:
            bad.append(i)
    return bad


# ---------- 工具结果卸载 ----------


def offload_tool_results(
    messages: list[Message], archive: ArchiveStore, caller: Caller, max_chars: int = 1200, preview_chars: int = 300
) -> tuple[list[Message], list[str]]:
    """超长工具结果先完整归档，上下文里只留预览 + ref_id。归档失败就保留原文。"""
    out, refs = [], []
    for m in messages:
        content = str(m.get("content") or "")
        if m.get("role") == "tool" and len(content) > max_chars and not m.get("offloaded"):
            try:
                ref = archive.put(caller, "tool", content, meta={"tool_call_id": m.get("tool_call_id")})
            except Exception:  # noqa: BLE001 - 归档失败宁可不卸载
                out.append(m)
                continue
            refs.append(ref)
            preview = content[:preview_chars]
            m = {
                **m,
                "content": f"{preview}…\n[完整结果已归档 ref={ref}，共 {len(content)} 字符，"
                f"需要时调用 recall_context(ref_id=\"{ref}\")]",
                "offloaded": ref,
            }
        out.append(m)
    return out, refs


# ---------- 摘要 ----------


def render(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        role = m.get("role")
        content = str(m.get("content") or "")
        if m.get("tool_calls"):
            calls = ", ".join(tc.get("function", {}).get("name", tc.get("name", "?")) for tc in m["tool_calls"])
            content = f"{content} [调用工具: {calls}]".strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def rule_summarize(prev_summary: str, head: list[Message], max_line_chars: int = 80) -> str:
    """规则版增量摘要：每轮保留用户原话（截断）和助手回复的第一行。

    数字和品牌这类关键信息按原文保留，不做改写；真正需要精确约束的地方以结构化 state 为准。
    LLM 版见 llm.LLMSummarizer。
    """
    lines = [ln for ln in prev_summary.splitlines() if ln.startswith("- ")]
    for m in head:
        if is_summary(m):
            continue
        content = str(m.get("content") or "").strip()
        if m.get("role") == "user":
            lines.append(f"- 用户：{content[:max_line_chars]}")
        elif m.get("role") == "assistant" and content:
            lines.append(f"- 助手：{content.splitlines()[0][:max_line_chars]}")
    return "\n".join(lines)


# ---------- 压缩主流程 ----------


@dataclass
class CompactResult:
    status: str  # noop | compacted | summary_failed | archive_failed | nothing_to_cut
    messages: list[Message]
    tokens_before: int
    tokens_after: int
    ref_ids: list[str] = field(default_factory=list)
    turns_summarized: int = 0

    @property
    def changed(self) -> bool:
        return self.status in ("compacted", "summary_failed") or bool(self.ref_ids)


def compact(
    messages: list[Message],
    *,
    budget: int,
    pinned_tokens: int,
    archive: ArchiveStore,
    caller: Caller,
    summarizer: Summarizer = rule_summarize,
    trigger_ratio: float = 0.8,
    keep_ratio: float = 0.4,
    summary_ratio: float = 0.2,
    max_tool_chars: int = 1200,
) -> CompactResult:
    """预算 = 固定注入（pinned） + 对话历史。超过 budget × trigger_ratio 才压缩。

    摘要本身也有上限（budget × summary_ratio）：短对话的逐轮摘要加上表头，可能比原文还长，
    超出时丢掉最早的摘要行，并注明去归档里找，原文一个字都不会丢。
    """
    before = count_tokens(messages)
    if pinned_tokens + before <= budget * trigger_ratio:
        return CompactResult("noop", messages, before, before)

    # 先卸载超长工具结果，常常这一步就够了
    msgs, refs = offload_tool_results(messages, archive, caller, max_tool_chars)
    if pinned_tokens + count_tokens(msgs) <= budget * trigger_ratio:
        return CompactResult("compacted", msgs, before, count_tokens(msgs), refs)

    prev = msgs[0] if msgs and is_summary(msgs[0]) else None
    body = msgs[1:] if prev else msgs
    keep_tokens = max(0, int(budget * keep_ratio) - pinned_tokens)
    cut = safe_cut_index(body, keep_tokens)
    if cut == 0:
        return CompactResult("nothing_to_cut", msgs, before, count_tokens(msgs), refs)

    head, tail = body[:cut], body[cut:]
    try:
        ref = archive.put(caller, "history", render(head), meta={"messages": head})
    except Exception:  # noqa: BLE001 - 归档失败：停止裁剪，原样返回
        return CompactResult("archive_failed", messages, before, before)
    refs.append(ref)

    prev_text = prev["content"] if prev else ""
    prev_refs = list(prev.get("refs", [])) if prev else []
    covered = (prev.get("turns", 0) if prev else 0) + len(turn_starts(head))
    status = "compacted"
    try:
        summary_body = summarizer(_strip_header(prev_text), head)
        if not summary_body.strip():
            raise ValueError("empty summary")
    except Exception:  # noqa: BLE001 - 摘要失败：明确标出缺口，保留原文引用
        status = "summary_failed"
        start = (prev.get("turns", 0) if prev else 0) + 1
        summary_body = (
            _strip_header(prev_text)
            + f"\n- [第 {start}–{covered} 轮摘要生成失败，原文见 ref={ref}，需要时调用 recall_context]"
        ).strip()

    all_refs = prev_refs + [ref]
    summary_body = _cap_lines(summary_body, int(budget * summary_ratio))
    header = f"【历史摘要｜覆盖前 {covered} 轮｜原文归档：{', '.join(all_refs)}】"
    summary_msg = {
        "role": "system",
        "name": SUMMARY_NAME,
        "content": f"{header}\n{summary_body}",
        "refs": all_refs,
        "turns": covered,
    }
    new_msgs = [summary_msg] + tail
    return CompactResult(status, new_msgs, before, count_tokens(new_msgs), refs, len(turn_starts(head)))


def build_window(pinned: list[Message], messages: list[Message], budget: int) -> list[Message]:
    """组装最终发给模型的窗口。放不下就拒绝发送，而不是悄悄截断。"""
    window = pinned + messages
    total = count_tokens(window)
    if total > budget:
        raise ContextOverflowError(f"context {total} tokens > budget {budget}")
    return window


def _cap_lines(body: str, max_tokens: int) -> str:
    lines = [ln for ln in body.splitlines() if ln.strip() and not ln.startswith("- [更早的")]
    dropped = 0
    while len(lines) > 1 and count_text_tokens("\n".join(lines)) > max_tokens:
        lines.pop(0)
        dropped += 1
    prev_dropped = re.search(r"更早的 (\d+) 条", body)
    dropped += int(prev_dropped.group(1)) if prev_dropped else 0
    if dropped:
        lines.insert(0, f"- [更早的 {dropped} 条摘要已省略，原文见归档 ref]")
    return "\n".join(lines)


def _strip_header(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("【历史摘要"))
