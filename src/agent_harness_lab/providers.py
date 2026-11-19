"""Synchronous provider boundary and deterministic, network-free test scripts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from agent_harness_lab.contracts import ToolCall, ToolDefinition, ToolResult
from agent_harness_lab.integration import TokenUsage

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("role must be system, user, or assistant")
        if not isinstance(self.content, str):
            raise TypeError("message content must be text")


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("response content must be text")


ConversationItem = Message | ToolCall | ToolResult


class ModelProvider(Protocol):
    """Return a final text response or tool calls; the Agent owns execution."""

    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse: ...


class ProviderExhaustedError(RuntimeError):
    """The deterministic response script has no responses left."""


@dataclass(frozen=True, slots=True)
class MockInvalidResponse:
    """Explicit fault injection; ordinary invalid script entries still fail early."""

    value: object


MockStep = str | ModelResponse | Exception | MockInvalidResponse


class MockProvider:
    """Consume scripted responses/faults and retain immutable request snapshots.

    Construct a new instance with the same script to reproduce a run. Invalid
    calls and exhaustion do not advance or record a request. Scripted exceptions
    consume one step and record the accepted request, just like a provider failure.
    Instances are synchronous and are not intended for concurrent use.
    """

    def __init__(self, responses: Sequence[MockStep]) -> None:
        if isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence):
            raise TypeError("responses must be a sequence of text responses")
        self._responses = tuple(responses)
        if any(
            not isinstance(response, (str, ModelResponse, Exception, MockInvalidResponse))
            for response in self._responses
        ):
            raise TypeError("expected text, ModelResponse, or an explicit mock fault")
        self._calls: list[tuple[ConversationItem, ...]] = []
        self._tool_catalogs: list[tuple[ToolDefinition, ...]] = []
        self._cursor = 0

    @property
    def calls(self) -> tuple[tuple[ConversationItem, ...], ...]:
        return tuple(self._calls)

    @property
    def tool_catalogs(self) -> tuple[tuple[ToolDefinition, ...], ...]:
        return tuple(self._tool_catalogs)

    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise TypeError("messages must be a sequence of Message objects")
        snapshot = tuple(messages)
        if not snapshot:
            raise ValueError("at least one message is required")
        if any(not isinstance(message, (Message, ToolCall, ToolResult)) for message in snapshot):
            raise TypeError("expected Message, ToolCall, or ToolResult")
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
            raise TypeError("tools must be a sequence of ToolDefinition objects")
        catalog = tuple(tools)
        if any(type(tool) is not ToolDefinition for tool in catalog):
            raise TypeError("expected ToolDefinition")
        if self._cursor == len(self._responses):
            raise ProviderExhaustedError("mock response script exhausted")
        step = self._responses[self._cursor]
        self._cursor += 1
        self._calls.append(snapshot)
        self._tool_catalogs.append(catalog)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, MockInvalidResponse):
            return cast(ModelResponse, step.value)  # Deliberate protocol fault for runtime tests.
        return ModelResponse(step) if isinstance(step, str) else step
