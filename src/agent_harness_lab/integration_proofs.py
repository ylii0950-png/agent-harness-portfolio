"""Ephemeral tool-evidence checks. Never serialize private message bodies."""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from agent_harness_lab.contracts import ToolCall, ToolResult
from agent_harness_lab.providers import ConversationItem


def paired_tools(messages: Sequence[ConversationItem]) -> list[tuple[ToolCall, ToolResult]]:
    """Only accept adjacent, matching runtime call/results, in execution order."""
    return [
        (call, result)
        for call, result in zip(messages, messages[1:], strict=False)
        if isinstance(call, ToolCall)
        and isinstance(result, ToolResult)
        and call.call_id == result.call_id
        and call.name == result.name
    ]


def arguments(call: ToolCall) -> dict[str, Any]:
    try:
        value = json.loads(call.arguments)
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def successful_read(messages: Sequence[ConversationItem], path: str, content: str) -> bool:
    return any(
        call.name == "read_file"
        and arguments(call).get("path") == path
        and result.ok
        and result.content == content
        for call, result in paired_tools(messages)
    )


def final_readback(
    messages: Sequence[ConversationItem], path: str, content: str, *, after_tool: str | None = None
) -> bool:
    """A correct successful read must follow the last successful write to this path."""
    written = read = False
    ready = after_tool is None
    for call, result in paired_tools(messages):
        if call.name == after_tool and result.ok:
            ready = True
        args = arguments(call)
        if args.get("path") != path or not result.ok:
            continue
        if call.name == "write_file":
            written = ready and args.get("content") == content
            read = False  # A later overwrite invalidates every earlier readback.
        elif call.name == "read_file":
            read = written and result.content == content
    return read


def decision_after_delegate(
    messages: Sequence[ConversationItem], events: list[dict[str, Any]], path: str
) -> bool:
    """A write decision needs a later Parent model round, not a same-response guess."""
    parent_calls = {
        row.get("call_ref"): row
        for row in events
        if row.get("event") == "tool.call" and row.get("depth") == 0
    }
    delegated_round: int | None = None
    valid = False
    for call, result in paired_tools(messages):
        row = parent_calls.get(hashlib.sha256(call.call_id.encode()).hexdigest()[:16], {})
        current_round = row.get("round")
        if call.name == "delegate_task" and result.ok and type(current_round) is int:
            delegated_round = current_round
        if call.name == "write_file" and arguments(call).get("path") == path and result.ok:
            valid = (
                delegated_round is not None
                and type(current_round) is int
                and current_round > delegated_round
            )
    return valid
