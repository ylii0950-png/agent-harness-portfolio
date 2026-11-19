from typing import Any

import pytest

from agent_harness_lab.contracts import ToolCall, ToolDefinition, ToolResult
from agent_harness_lab.providers import (
    ConversationItem,
    Message,
    MockInvalidResponse,
    MockProvider,
    ModelResponse,
    ProviderExhaustedError,
)


def test_explicit_faults_are_deterministic_and_consume_one_accepted_request() -> None:
    for _ in range(2):
        provider = MockProvider(
            [RuntimeError("offline failure"), MockInvalidResponse(None), "recovered"]
        )
        messages = [Message("user", "task")]
        with pytest.raises(RuntimeError, match="offline failure"):
            provider.complete(messages)
        assert provider.complete(messages) is None
        assert provider.complete(messages) == ModelResponse("recovered")
        with pytest.raises(ProviderExhaustedError):
            provider.complete(messages)
        assert len(provider.calls) == 3


def test_tool_records_and_catalog_snapshots_are_supported() -> None:
    provider = MockProvider([ModelResponse(tool_calls=(ToolCall("one", "read", "{}"),)), "done"])
    catalog = [ToolDefinition("read", "Read.")]
    first = provider.complete([Message("user", "task")], tools=catalog)
    catalog.clear()
    history: list[ConversationItem] = [
        Message("user", "task"),
        first.tool_calls[0],
        ToolResult("one", "read", "content"),
    ]
    assert provider.complete(history).content == "done"
    history.clear()
    assert len(provider.calls[1]) == 3
    assert provider.tool_catalogs[0] == (ToolDefinition("read", "Read."),)


@pytest.mark.parametrize("tools", [None, "bad", [None], ["read"]])
def test_bad_catalog_does_not_consume_script(tools: Any) -> None:
    provider = MockProvider(["ok"])
    with pytest.raises(TypeError):
        provider.complete([Message("user", "task")], tools=tools)
    assert not provider.calls
    assert provider.complete([Message("user", "task")]).content == "ok"
