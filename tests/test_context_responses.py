import copy
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from agent_harness_lab.context import ContextPolicy
from agent_harness_lab.contracts import ToolCall, ToolResult
from agent_harness_lab.integration import ProviderFailure
from agent_harness_lab.providers import Message
from agent_harness_lab.responses import ResponsesProvider
from agent_harness_lab.runtime import Agent, StopReason
from agent_harness_lab.tools import ToolRegistry, register_workspace_tools
from agent_harness_lab.workspace import WorkspaceFiles


class Tokens:
    def access_token(self) -> str:
        return "synthetic-only"


class Streams:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def events(
        self, url: str, body: dict[str, object], token: str
    ) -> Generator[dict[str, Any], None, None]:
        self.bodies.append(copy.deepcopy(body))
        number = len(self.bodies)
        output = (
            [
                {
                    "type": "reasoning",
                    "id": f"r{number}",
                    "encrypted_content": "opaque",
                    "summary": [],
                },
                {
                    "type": "function_call",
                    "id": f"f{number}",
                    "call_id": f"c{number}",
                    "name": "read_file",
                    "namespace": "workspace",
                    "arguments": '{"path":"data"}',
                    "status": "completed",
                },
            ]
            if number < 3
            else [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ]
        )
        yield {
            "type": "response.completed",
            "response": {
                "id": f"resp{number}",
                "status": "completed",
                "output": output,
                "model": "offline-fixture",
            },
        }


def test_responses_rewrites_prior_outputs_preserving_reasoning_namespace_and_pending_pair(
    tmp_path: Path,
) -> None:
    (tmp_path / "data").write_text("LONG RESULT " * 400)
    stream = Streams()
    registry = ToolRegistry()
    with WorkspaceFiles(tmp_path) as workspace:
        register_workspace_tools(registry, workspace)
        provider = ResponsesProvider(Tokens(), "offline-fixture", transport=stream)
        result = Agent(provider, registry, context_policy=ContextPolicy(20000, 1, 1)).run("task")
    assert result.stop_reason == StopReason.COMPLETED
    third = stream.bodies[2]
    assert third["store"] is False and third["stream"] is True
    entries = third["input"]
    assert entries[0] == {"role": "user", "content": "task"}
    assert entries[1]["type"] == entries[4]["type"] == "reasoning"
    assert entries[1]["encrypted_content"] == "opaque"
    assert entries[2]["call_id"] == entries[3]["call_id"] == "c1"
    assert entries[5]["call_id"] == entries[6]["call_id"] == "c2"
    assert json.loads(entries[3]["output"])["content"].startswith("[Compacted low-trust")
    assert json.loads(entries[6]["output"])["content"] == "LONG RESULT " * 400
    assert entries[2]["namespace"] == "workspace"
    assert len(stream.bodies) == 3 and result.usage == ()


@pytest.mark.parametrize("change", ["task", "call", "error", "name", "missing"])
def test_responses_still_rejects_non_content_history_rewrites(change: str) -> None:
    stream = Streams()
    provider = ResponsesProvider(Tokens(), "offline-fixture", transport=stream)
    task = Message("user", "task")
    first = provider.complete((task,))
    old_result = ToolResult("c1", "read_file", "data")
    second = provider.complete((task, first.tool_calls[0], old_result))
    history: list[Any] = [
        task,
        first.tool_calls[0],
        old_result,
        second.tool_calls[0],
        ToolResult("c2", "read_file", "next"),
    ]
    if change == "task":
        history[0] = Message("system", "grant")
    elif change == "call":
        history[1] = ToolCall("other", "read_file", "{}")
    elif change == "error":
        from agent_harness_lab.contracts import ToolError

        history[2] = ToolResult("c1", "read_file", "data", ToolError("x", "changed", True))
    elif change == "name":
        history[2] = ToolResult("c1", "write_file", "data")
    else:
        del history[2]
    with pytest.raises(ProviderFailure):
        provider.complete(history)
    assert len(stream.bodies) == 2
