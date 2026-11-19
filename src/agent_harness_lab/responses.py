"""ChatGPT-plan-only Responses adapter. Transport never executes model tools."""

import copy
from collections.abc import Generator, Sequence
from typing import Any, Protocol

from agent_harness_lab.contracts import ToolCall, ToolDefinition, ToolResult
from agent_harness_lab.http_transport import HTTPTransport
from agent_harness_lab.integration import (
    FailureKind,
    TokenUsage,
    api_failure,
    failure,
)
from agent_harness_lab.oauth import RESOURCE, OAuthConnection
from agent_harness_lab.providers import ConversationItem, Message, ModelResponse
from agent_harness_lab.tracing import TraceSink, emit

_NAMESPACE = "workspace"
_STREAM_EVENTS = frozenset(
    {
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.completed",
        "response.failed",
        "response.incomplete",
        "error",
        "response.refusal.delta",
        "response.refusal.done",
    }
)


class AccessTokenSource(Protocol):
    def access_token(self) -> str: ...


class EventTransport(Protocol):
    def events(
        self, url: str, body: dict[str, object], token: str
    ) -> Generator[dict[str, Any], None, None]: ...


def _initial_history(messages: tuple[ConversationItem, ...]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Message):
            raise failure(FailureKind.PROTOCOL, "unpaired_history")
        history.append(
            {
                "role": "developer" if message.role == "system" else message.role,
                "content": message.content,
            }
        )
    return history


class ResponsesProvider:
    """Synchronous adapter, one conversation at a time, no automatic retry/billing fallback.

    A fresh task-only history resets per-run state. Continuations must include all
    exact call/result pairs from the previous response; raw reasoning/output items
    are replayed in memory, including encrypted reasoning. No previous_response_id.
    """

    def __init__(
        self,
        connection: AccessTokenSource,
        model: str,
        *,
        max_requests: int = 8,
        transport: EventTransport | None = None,
        trace: TraceSink | None = None,
    ) -> None:
        if not isinstance(model, str) or not model or len(model) > 100:
            raise ValueError("model must be an explicit catalog ID")
        if type(max_requests) is not int or not 0 <= max_requests <= 100:
            raise ValueError("request budget must be an integer between 0 and 100")
        self._connection = connection
        self.model = model
        self.max_requests = max_requests
        self.requests = 0
        self._transport = transport or HTTPTransport()
        self._trace = trace
        self._last_history: tuple[ConversationItem, ...] = ()
        self._pending: tuple[ToolCall, ...] = ()
        self._wire: list[dict[str, Any]] = []
        self.served_models: set[str] = set()

    def _history(self, messages: tuple[ConversationItem, ...]) -> list[dict[str, Any]]:
        if all(isinstance(item, Message) for item in messages):
            return _initial_history(messages)
        prefix = len(self._last_history)
        if not self._pending or len(messages) < prefix:
            raise failure(FailureKind.PROTOCOL, "history_mismatch")
        # A context pass may shorten completed outputs, never requests/messages/status.
        rewritten: dict[str, ToolResult] = {}
        for old, new in zip(self._last_history, messages[:prefix], strict=True):
            if old == new:
                continue
            if (
                type(old) is not ToolResult
                or type(new) is not ToolResult
                or old.call_id != new.call_id
                or old.name != new.name
                or old.error != new.error
                or type(new.content) is not str
            ):
                raise failure(FailureKind.PROTOCOL, "history_mismatch")
            rewritten[old.call_id] = new
        tail = messages[prefix:]
        if len(tail) != 2 * len(self._pending):
            raise failure(FailureKind.PROTOCOL, "unpaired_history")
        result = copy.deepcopy(self._wire)
        for entry in result:
            if entry.get("type") == "function_call_output":
                replacement = rewritten.pop(str(entry.get("call_id")), None)
                if replacement is not None:
                    entry["output"] = replacement.output_json()
        if rewritten:
            raise failure(FailureKind.PROTOCOL, "history_mismatch")
        for index, call in enumerate(self._pending):
            submitted, output = tail[2 * index : 2 * index + 2]
            if (
                submitted != call
                or not isinstance(output, ToolResult)
                or output.call_id != call.call_id
                or output.name != call.name
            ):
                raise failure(FailureKind.PROTOCOL, "history_mismatch")
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output.output_json(),
                }
            )
        return result

    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        snapshot = tuple(messages)
        if not snapshot:
            raise ValueError("history must not be empty")
        wire = self._history(snapshot)
        definitions = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.json_schema(),
                "strict": False,
            }
            for tool in tools
        ]
        body: dict[str, object] = {
            "model": self.model,
            "input": wire,
            "store": False,
            "stream": True,
        }
        if definitions:
            body["tools"] = [
                {
                    "type": "namespace",
                    "name": _NAMESPACE,
                    "description": "Capability-checked workspace tools.",
                    "tools": definitions,
                }
            ]
        if self.requests >= self.max_requests:
            raise failure(FailureKind.REQUEST_BUDGET, "request_budget_exhausted")
        token = self._connection.access_token()
        self.requests += 1  # Failed HTTP attempts count; no implicit retry.
        events = self._transport.events(RESOURCE + "/responses", body, token)
        completed: dict[str, Any] | None = None
        done_items: dict[int, dict[str, Any]] = {}
        started_items: set[int] = set()
        deltas: dict[tuple[str, int], str] = {}
        try:
            for event in events:
                event_type = event.get("type")
                emit(
                    self._trace,
                    "model.event",
                    {"event_type": event_type if event_type in _STREAM_EVENTS else "other"},
                )
                if event_type in {
                    "response.output_text.delta",
                    "response.function_call_arguments.delta",
                }:
                    index, delta = event.get("output_index"), event.get("delta")
                    if type(index) is not int or index < 0 or not isinstance(delta, str):
                        raise failure(FailureKind.PROTOCOL, "invalid_stream_delta")
                    # Content-part index prevents merging separate text parts.
                    part = event.get("content_index", 0)
                    if type(part) is not int or part < 0:
                        raise failure(FailureKind.PROTOCOL, "invalid_stream_delta")
                    key = (f"{event_type}:{part}", index)
                    deltas[key] = deltas.get(key, "") + delta
                elif event_type == "response.output_item.added":
                    index = event.get("output_index")
                    if type(index) is not int or index < 0 or index in started_items:
                        raise failure(FailureKind.PROTOCOL, "invalid_started_item")
                    started_items.add(index)
                elif event_type == "response.output_item.done":
                    index, item = event.get("output_index"), event.get("item")
                    if (
                        type(index) is not int
                        or index < 0
                        or not isinstance(item, dict)
                        or index in done_items
                    ):
                        raise failure(FailureKind.PROTOCOL, "invalid_completed_item")
                    done_items[index] = item
                elif event_type in {"response.failed", "error"}:
                    error = event.get("response", event)
                    if not isinstance(error, dict):
                        error = {}
                    raise api_failure(None, error)
                elif event_type == "response.incomplete":
                    raise failure(FailureKind.INTERRUPTED, "response_incomplete")
                elif event_type == "response.completed":
                    if not isinstance(event.get("response"), dict):
                        raise failure(FailureKind.PROTOCOL, "missing_completed_response")
                    completed = event["response"]
                    break
        finally:
            events.close()
        if completed is None:
            raise failure(FailureKind.INTERRUPTED, "missing_completed_event")
        if completed.get("status") != "completed" or not isinstance(completed.get("output"), list):
            raise failure(FailureKind.PROTOCOL, "invalid_completed_response")
        output = completed["output"]
        # Some official streams use response.completed as a terminal envelope:
        # the finalized output is carried only by output_item.done events.
        if not output and done_items:
            if (
                set(done_items) != set(range(len(done_items)))
                or not started_items <= done_items.keys()
            ):
                raise failure(FailureKind.PROTOCOL, "unfinished_output_item")
            output = [done_items[index] for index in range(len(done_items))]
        for _delta_type, index in deltas:
            if index >= len(output):
                raise failure(FailureKind.PROTOCOL, "unmatched_stream_delta")
        for index, item in done_items.items():
            if index >= len(output) or item != output[index]:
                raise failure(FailureKind.PROTOCOL, "completed_item_mismatch")
        calls: list[ToolCall] = []
        text: list[str] = []
        for index, item in enumerate(output):
            if not isinstance(item, dict):
                raise failure(FailureKind.PROTOCOL, "invalid_output_item")
            if item.get("status") not in (None, "completed"):
                raise failure(FailureKind.PROTOCOL, "incomplete_output_item")
            kind = item.get("type")
            if kind == "function_call":
                if item.get("namespace") not in (None, _NAMESPACE):
                    raise failure(FailureKind.PROTOCOL, "unknown_tool_namespace")
                if any(
                    not isinstance(item.get(key), str) for key in ("call_id", "name", "arguments")
                ):
                    raise failure(FailureKind.PROTOCOL, "invalid_tool_call")
                call = ToolCall(item["call_id"], item["name"], item["arguments"])
                try:
                    call.validate()
                except (TypeError, ValueError):
                    raise failure(FailureKind.PROTOCOL, "invalid_tool_call") from None
                calls.append(call)
                accumulated = deltas.get(("response.function_call_arguments.delta:0", index))
                if accumulated is not None and accumulated != call.arguments:
                    raise failure(FailureKind.PROTOCOL, "argument_delta_mismatch")
            elif kind == "message":
                if item.get("role") != "assistant" or not isinstance(item.get("content"), list):
                    raise failure(FailureKind.PROTOCOL, "invalid_output_message")
                for part_index, part in enumerate(item["content"]):
                    if not isinstance(part, dict):
                        raise failure(FailureKind.PROTOCOL, "invalid_output_message")
                    value = (
                        part.get("text")
                        if part.get("type") == "output_text"
                        else part.get("refusal")
                    )
                    if not isinstance(value, str):
                        raise failure(FailureKind.PROTOCOL, "unsupported_message_content")
                    accumulated = deltas.get((f"response.output_text.delta:{part_index}", index))
                    if accumulated is not None and accumulated != value:
                        raise failure(FailureKind.PROTOCOL, "text_delta_mismatch")
                    text.append(value)
            elif kind != "reasoning":
                raise failure(FailureKind.PROTOCOL, "unsupported_output_item")
        if not calls and not text:
            raise failure(FailureKind.PROTOCOL, "missing_final_answer")
        if len({call.call_id for call in calls}) != len(calls):
            raise failure(FailureKind.PROTOCOL, "duplicate_tool_id")
        usage: TokenUsage | None = None
        raw_usage = completed.get("usage")
        if raw_usage is not None:
            if not isinstance(raw_usage, dict):
                raise failure(FailureKind.PROTOCOL, "invalid_usage")
            try:
                usage = TokenUsage(
                    *(
                        raw_usage.get(name)
                        for name in ("input_tokens", "output_tokens", "total_tokens")
                    )
                )
            except ValueError:
                raise failure(FailureKind.PROTOCOL, "invalid_usage") from None
        served = completed.get("model")
        if isinstance(served, str) and served:
            self.served_models.add(served)
        self._last_history, self._pending = snapshot, tuple(calls)
        self._wire = copy.deepcopy(wire + output)
        return ModelResponse("" if calls else "".join(text), tuple(calls), usage)


def available_models(connection: OAuthConnection) -> tuple[str, ...]:
    payload = connection.transport.json(
        "GET",
        RESOURCE + "/models",
        headers={"Authorization": "Bearer " + connection.access_token()},
    )
    models = payload.get("models")
    if not isinstance(models, list):
        raise failure(FailureKind.PROTOCOL, "invalid_model_catalog")
    return tuple(
        item["slug"]
        for item in models
        if isinstance(item, dict)
        and item.get("visibility") == "list"
        and isinstance(item.get("slug"), str)
    )
