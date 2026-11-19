import copy
import io
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from agent_harness_lab.contracts import (
    ParameterType,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from agent_harness_lab.http_transport import HTTPTransport, parse_sse
from agent_harness_lab.integration import FailureKind, ProviderFailure, api_failure
from agent_harness_lab.providers import Message
from agent_harness_lab.responses import ResponsesProvider
from agent_harness_lab.runtime import Agent, StopReason
from agent_harness_lab.tools import Permission, ToolRegistry, register_workspace_tools
from agent_harness_lab.workspace import WorkspaceFiles


def message(text: str = "done") -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text}],
    }


def call(arguments: Any = '{"path":"file"}', **overrides: Any) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc_one",
        "call_id": "one",
        "name": "read_file",
        "namespace": "workspace",
        "arguments": arguments,
        "status": "completed",
        **overrides,
    }


def completed(output: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_test",
            "status": "completed",
            "output": output,
            "model": "test-model",
            **extra,
        },
    }


class Tokens:
    def access_token(self) -> str:
        return "synthetic-access"


class Streams:
    def __init__(self, batches: list[list[dict[str, Any]]]) -> None:
        self.batches = iter(batches)
        self.bodies: list[dict[str, Any]] = []
        self.closed = 0

    def events(
        self, url: str, body: dict[str, object], token: str
    ) -> Generator[dict[str, Any], None, None]:
        assert url == "https://api.openai.com/v1/responses" and token == "synthetic-access"
        self.bodies.append(copy.deepcopy(body))
        try:
            yield from next(self.batches)
        finally:
            self.closed += 1


def test_sse_multiline_comments_utf8_and_completed_only() -> None:
    source = io.BytesIO(
        b':heartbeat\r\nevent: response.completed\r\ndata: {"type":\r\n'
        b'data: "response.completed"}\r\n\r\ndata: [DONE]\n\n'
    )
    assert list(parse_sse(source)) == [{"type": "response.completed"}]


@pytest.mark.parametrize(
    "raw,kind",
    [
        (b"data: {\n\n", FailureKind.PROTOCOL),
        (b"data: null\n\n", FailureKind.PROTOCOL),
        (b'data: {"type":"response.created"}', FailureKind.INTERRUPTED),
        (b"data: \xff\n\n", FailureKind.PROTOCOL),
        (b"data: " + b"x" * (8 * 1024 * 1024) + b"\n\n", FailureKind.INTERRUPTED),
    ],
)
def test_sse_malformed_truncated_or_oversized_fails(raw: bytes, kind: FailureKind) -> None:
    with pytest.raises(ProviderFailure) as error:
        list(parse_sse(io.BytesIO(raw)))
    assert error.value.info.kind == kind


def test_stateless_namespace_request_and_reasoning_history_are_preserved() -> None:
    reasoning = {
        "type": "reasoning",
        "id": "rs_one",
        "encrypted_content": "opaque-data",
        "summary": [],
    }
    first = call()
    transport = Streams(
        [
            [
                {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
                completed([reasoning, first]),
            ],
            [completed([message()])],
        ]
    )
    provider = ResponsesProvider(Tokens(), "test-model", transport=transport)
    tool = ToolDefinition("read_file", "Read.", (ToolParameter("path", ParameterType.STRING),))
    task = Message("user", "read the file")
    result = provider.complete([task], tools=[tool])
    assert result.tool_calls == (ToolCall("one", "read_file", '{"path":"file"}'),)
    returned = ToolResult("one", "read_file", "content")
    assert provider.complete([task, result.tool_calls[0], returned], tools=[tool]).content == "done"
    first_body, second_body = transport.bodies
    assert first_body["store"] is False and first_body["stream"] is True
    assert (
        not {"previous_response_id", "conversation", "max_output_tokens", "temperature"}
        & first_body.keys()
    )
    assert first_body["tools"][0]["type"] == "namespace"
    assert first_body["tools"][0]["tools"][0]["strict"] is False
    assert second_body["input"] == [
        {"role": "user", "content": task.content},
        reasoning,
        first,
        {"type": "function_call_output", "call_id": "one", "output": returned.output_json()},
    ]
    assert transport.closed == 2
    assert provider.served_models == {"test-model"}


def test_system_maps_to_developer_and_new_task_resets_conversation() -> None:
    transport = Streams([[completed([message()])], [completed([message()])]])
    provider = ResponsesProvider(Tokens(), "test-model", transport=transport)
    provider.complete([Message("system", "policy"), Message("user", "one")])
    assert transport.bodies[0]["input"][0]["role"] == "developer"
    provider.complete([Message("user", "new task")])
    assert transport.bodies[1]["input"] == [{"role": "user", "content": "new task"}]


@pytest.mark.parametrize(
    "events,kind",
    [
        ([], FailureKind.INTERRUPTED),
        ([{"type": "response.incomplete"}], FailureKind.INTERRUPTED),
        (
            [
                {
                    "type": "response.failed",
                    "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}},
                }
            ],
            FailureKind.QUOTA,
        ),
        (
            [{"type": "error", "error": {"code": "subscription_sharing_user_not_eligible"}}],
            FailureKind.AUTHORIZATION,
        ),
        ([completed([call(namespace="other")])], FailureKind.PROTOCOL),
        ([completed([call(arguments=None)])], FailureKind.PROTOCOL),
        ([completed([call(), call()])], FailureKind.PROTOCOL),
        ([completed([{"type": "shell_call"}])], FailureKind.PROTOCOL),
        ([completed([], status="incomplete")], FailureKind.PROTOCOL),
        ([completed([])], FailureKind.PROTOCOL),
        ([completed([message()], usage={"total_tokens": True})], FailureKind.PROTOCOL),
        (
            [
                {"type": "response.output_item.done", "output_index": 0, "item": message("other")},
                completed([message()]),
            ],
            FailureKind.PROTOCOL,
        ),
        (
            [
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": "{}",
                },
                completed([call()]),
            ],
            FailureKind.PROTOCOL,
        ),
        (
            [
                {"type": "response.output_text.delta", "output_index": 0, "delta": "other"},
                completed([message()]),
            ],
            FailureKind.PROTOCOL,
        ),
        ([completed([call(status="incomplete")])], FailureKind.PROTOCOL),
    ],
)
def test_failed_or_invalid_stream_never_returns_success(
    events: list[dict[str, Any]], kind: FailureKind
) -> None:
    transport = Streams([events])
    provider = ResponsesProvider(Tokens(), "test-model", transport=transport)
    with pytest.raises(ProviderFailure) as error:
        provider.complete([Message("user", "task")])
    assert error.value.info.kind == kind and transport.closed == 1


def test_real_usage_only_when_provided_and_delta_assembly() -> None:
    transport = Streams(
        [
            [
                {"type": "response.output_text.delta", "output_index": 0, "delta": "do"},
                {"type": "response.output_text.delta", "output_index": 0, "delta": "ne"},
                completed(
                    [message()], usage={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}
                ),
            ]
        ]
    )
    response = ResponsesProvider(Tokens(), "test-model", transport=transport).complete(
        [Message("user", "task")]
    )
    assert response.content == "done" and response.usage is not None
    assert response.usage.total_tokens == 13


def test_unpaired_feedback_rejected_before_request_and_budget_not_exceeded() -> None:
    transport = Streams([[completed([call()])]])
    provider = ResponsesProvider(Tokens(), "test-model", transport=transport, max_requests=1)
    task = Message("user", "read")
    response = provider.complete([task])
    with pytest.raises(ProviderFailure, match="unpaired_history"):
        provider.complete([task, response.tool_calls[0]])
    with pytest.raises(ProviderFailure, match="request_budget_exhausted"):
        provider.complete([task, response.tool_calls[0], ToolResult("one", "read_file", "ok")])
    assert provider.requests == len(transport.bodies) == 1


def test_bad_stream_cannot_dispatch_a_valid_looking_write(tmp_path: Path) -> None:
    transport = Streams(
        [
            [
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": "{}",
                },
                completed(
                    [call(name="write_file", arguments='{"path":"written","content":"bad"}')]
                ),
            ]
        ]
    )
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
        register_workspace_tools(registry, workspace)
        result = Agent(
            ResponsesProvider(Tokens(), "test-model", transport=transport), registry
        ).run("write")
    assert result.stop_reason in {StopReason.MODEL_PROTOCOL_ERROR, StopReason.PROVIDER_ERROR}
    assert result.tool_calls == 0 and not (tmp_path / "written").exists()


@pytest.mark.parametrize(
    "status,body,kind",
    [
        (401, {"detail": "sensitive text"}, FailureKind.AUTHORIZATION),
        (
            403,
            {"error": {"code": "subscription_sharing_user_not_eligible"}},
            FailureKind.AUTHORIZATION,
        ),
        (429, {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}, FailureKind.QUOTA),
        (503, {"error": {"code": "subscription_sharing_usage_unavailable"}}, FailureKind.API),
        (400, {"error": {"code": {"malformed": True}}}, FailureKind.API),
    ],
)
def test_api_error_categories_are_safe(status: int, body: object, kind: FailureKind) -> None:
    error = api_failure(status, body, "req_synthetic")
    assert error.info.kind == kind and error.info.status == status
    assert error.info.request_id == "req_synthetic"
    assert "sensitive text" not in str(error)


def test_transport_rejects_arbitrary_endpoint_before_connecting() -> None:
    with pytest.raises(ValueError):
        HTTPTransport().json("GET", "https://unapproved.example")


@pytest.mark.parametrize("mime", ["", "text/event-stream; charset=utf-8", "Text/Event-Stream"])
def test_valid_sse_is_accepted_even_when_content_type_is_missing_or_case_varies(
    mime: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = ("data: " + json.dumps(completed([message()])) + "\n\n").encode()

    class Response(io.BytesIO):
        headers = {"Content-Type": mime}

    transport = HTTPTransport()
    monkeypatch.setattr(transport, "_open", lambda *args: Response(payload))
    provider = ResponsesProvider(Tokens(), "test-model", transport=transport)
    assert provider.complete([Message("user", "task")]).content == "done"


@pytest.mark.parametrize("payload", [b'{"output_text":"not SSE"}', b"data: [DONE]\n\n"])
def test_missing_header_does_not_allow_non_sse_or_missing_completion(
    payload: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response(io.BytesIO):
        headers: dict[str, str] = {}

    transport = HTTPTransport()
    monkeypatch.setattr(transport, "_open", lambda *args: Response(payload))
    with pytest.raises(ProviderFailure):
        ResponsesProvider(Tokens(), "test-model", transport=transport).complete(
            [Message("user", "task")]
        )


def test_completed_envelope_uses_finalized_stream_items_and_preserves_feedback() -> None:
    item = call()
    stream = Streams(
        [
            [
                {"type": "response.output_item.added", "output_index": 0, "item": item},
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                completed([]),
            ],
            [
                {"type": "response.output_item.done", "output_index": 0, "item": message()},
                completed([]),
            ],
        ]
    )
    provider = ResponsesProvider(Tokens(), "test-model", transport=stream)
    task = Message("user", "read")
    response = provider.complete([task])
    assert response.tool_calls == (ToolCall("one", "read_file", '{"path":"file"}'),)
    assert (
        provider.complete(
            [task, response.tool_calls[0], ToolResult("one", "read_file", "content")]
        ).content
        == "done"
    )
    assert stream.bodies[1]["input"][1] == item


@pytest.mark.parametrize("indices", [[1], [0, 2]])
def test_terminal_envelope_with_unfinished_or_gapped_items_is_rejected(indices: list[int]) -> None:
    events = [
        {
            "type": "response.output_item.done",
            "output_index": index,
            "item": call(call_id=str(index)),
        }
        for index in indices
    ]
    events.append(completed([]))
    with pytest.raises(ProviderFailure):
        ResponsesProvider(Tokens(), "test-model", transport=Streams([events])).complete(
            [Message("user", "task")]
        )
