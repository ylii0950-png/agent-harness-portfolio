"""Finite synchronous model/tool execution; no SDK, background tasks, or shell."""

import hashlib
import time
from dataclasses import asdict, dataclass
from enum import StrEnum

from agent_harness_lab.context import (
    ByteEstimator,
    Compactor,
    ContextController,
    ContextError,
    ContextEstimator,
    ContextPolicy,
    ContextStats,
    DeterministicCompactor,
)
from agent_harness_lab.contracts import ToolCall, ToolError, ToolResult
from agent_harness_lab.execution import ExecutionFrame, RunBudget, RunTrace, TotalLimits
from agent_harness_lab.integration import FailureInfo, FailureKind, ProviderFailure, TokenUsage
from agent_harness_lab.providers import ConversationItem, Message, ModelProvider, ModelResponse
from agent_harness_lab.tools import ToolRegistry
from agent_harness_lab.tracing import TraceSink, emit


class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_MODEL_ROUNDS = "max_model_rounds"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_TOTAL_MODEL_CALLS = "max_total_model_calls"
    MAX_TOTAL_TOOL_CALLS = "max_total_tool_calls"
    MODEL_PROTOCOL_ERROR = "model_protocol_error"
    PROVIDER_ERROR = "provider_error"
    TOOL_ERROR = "tool_error"
    INTERRUPTED = "interrupted"
    CONTEXT_ERROR = "context_error"


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_model_rounds: int = 10
    max_tool_calls: int = 20

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.max_model_rounds, self.max_tool_calls)
        ):
            raise ValueError("run limits must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class AgentResult:
    stop_reason: StopReason
    final_answer: str | None
    model_rounds: int
    tool_calls: int
    messages: tuple[ConversationItem, ...]
    failure: FailureInfo | None = None
    usage: tuple[TokenUsage, ...] = ()
    total_model_calls: int = 0
    total_tool_calls: int = 0
    context_stats: tuple[ContextStats, ...] = ()

    @property
    def tool_results(self) -> tuple[ToolResult, ...]:
        return tuple(item for item in self.messages if isinstance(item, ToolResult))


def _validate_response(response: ModelResponse, seen_ids: set[str]) -> None:
    if type(response) is not ModelResponse or type(response.content) is not str:
        raise ValueError("expected ModelResponse")
    if type(response.tool_calls) is not tuple:
        raise ValueError("tool calls must be an immutable tuple")
    if response.usage is not None and type(response.usage) is not TokenUsage:
        raise ValueError("invalid token usage")
    if response.tool_calls and response.content:
        raise ValueError("response cannot mix a final answer and tool requests")
    batch_ids: set[str] = set()
    for call in response.tool_calls:
        if type(call) is not ToolCall:
            raise ValueError("expected ToolCall")
        call.validate()
        if call.call_id in seen_ids or call.call_id in batch_ids:
            raise ValueError("tool call IDs must be unique within a run")
        batch_ids.add(call.call_id)


_DEFAULT_LIMITS = RunLimits()


class Agent:
    """One task per run. Registry/handlers/providers are trusted application code.

    Validate the entire model decision before dispatch. Refuse a whole batch if
    it exceeds the remaining tool budget. Reserve a model turn for tool feedback:
    a tool request on the last allowed model turn causes no tool side effects.
    """

    def __init__(
        self,
        provider: ModelProvider,
        registry: ToolRegistry,
        *,
        limits: RunLimits = _DEFAULT_LIMITS,
        trace: TraceSink | None = None,
        total_limits: TotalLimits | None = None,
        context_policy: ContextPolicy | None = None,
        context_estimator: ContextEstimator | None = None,
        compactor: Compactor | None = None,
    ) -> None:
        if type(limits) is not RunLimits:
            raise TypeError("expected RunLimits")
        if total_limits is not None and type(total_limits) is not TotalLimits:
            raise TypeError("expected TotalLimits")
        self._total_limits = total_limits or TotalLimits(
            limits.max_model_rounds, limits.max_tool_calls
        )
        if context_policy is not None and type(context_policy) is not ContextPolicy:
            raise TypeError("expected ContextPolicy")
        if context_policy is None and (context_estimator is not None or compactor is not None):
            raise ValueError("context hooks require an explicit context policy")
        self._context_policy = context_policy
        self._context_estimator = context_estimator or ByteEstimator()
        self._compactor = compactor or DeterministicCompactor()
        self._provider = provider
        self._registry = registry
        self._limits = limits
        self._trace = trace

    def run(self, task: str) -> AgentResult:
        budget = RunBudget(self._total_limits)
        budget.claim_provider(self._provider)
        return self._run_with_frame(task, ExecutionFrame(budget))

    def _run_with_frame(self, task: str, frame: ExecutionFrame) -> AgentResult:
        trace = RunTrace(self._trace, frame) if self._trace is not None else None
        with self._registry.execution_scope(frame), self._registry.run_scope():
            return self._run(task, frame, trace)

    def _run(self, task: str, frame: ExecutionFrame, trace: TraceSink | None) -> AgentResult:
        if not isinstance(task, str):
            raise TypeError("task must be text")
        if not task.strip():
            raise ValueError("task must not be empty")
        messages: list[ConversationItem] = [Message("user", task)]
        active = list(messages)
        context = (
            ContextController(self._context_policy, self._context_estimator, self._compactor)
            if self._context_policy is not None
            else None
        )

        def append(item: ConversationItem) -> None:
            messages.append(item)  # Full in-memory execution audit, never re-sent after trimming.
            active.append(item)

        model_rounds = 0
        tool_calls = 0
        seen_ids: set[str] = set()
        run_failure: FailureInfo | None = None
        usage: list[TokenUsage] = []
        started = time.monotonic()
        emit(trace, "run.started", {})

        def finish(reason: StopReason, answer: str | None = None) -> AgentResult:
            emit(
                trace,
                "run.stopped",
                {
                    "stop_reason": reason.value,
                    "model_rounds": model_rounds,
                    "tool_calls": tool_calls,
                    "total_model_calls": frame.budget.model_calls,
                    "total_tool_calls": frame.budget.tool_calls,
                    "duration_ms": (time.monotonic() - started) * 1000,
                },
            )
            return AgentResult(
                reason,
                answer,
                model_rounds,
                tool_calls,
                tuple(messages),
                run_failure,
                tuple(usage),
                frame.budget.model_calls,
                frame.budget.tool_calls,
                tuple(context.stats) if context is not None else (),
            )

        for _ in range(self._limits.max_model_rounds):
            if frame.budget.models_remaining <= 0:
                return finish(StopReason.MAX_TOTAL_MODEL_CALLS)
            if context is not None:
                try:
                    active = list(context.prepare(tuple(active), self._registry.definitions, trace))
                except ContextError as exc:
                    run_failure = FailureInfo(FailureKind.CONTEXT, exc.code)
                    return finish(StopReason.CONTEXT_ERROR)
            model_rounds += 1
            frame.budget.model_calls += 1
            emit(trace, "model.request", {"round": model_rounds, "item_count": len(active)})
            request_started = time.monotonic()
            try:
                response = self._provider.complete(tuple(active), tools=self._registry.definitions)
            except KeyboardInterrupt:
                run_failure = FailureInfo(FailureKind.INTERRUPTED, "user_interrupted")
                emit(
                    trace,
                    "model.error",
                    {
                        "round": model_rounds,
                        "error_kind": run_failure.kind.value,
                        "error_code": run_failure.code,
                    },
                )
                return finish(StopReason.INTERRUPTED)
            except Exception as exc:
                run_failure = (
                    exc.info
                    if isinstance(exc, ProviderFailure)
                    else FailureInfo(FailureKind.API, "unexpected_provider_failure")
                )
                emit(
                    trace,
                    "model.error",
                    {
                        "round": model_rounds,
                        "error_kind": run_failure.kind.value,
                        "error_code": run_failure.code,
                        "status": run_failure.status,
                        "request_id": run_failure.request_id,
                        "duration_ms": (time.monotonic() - request_started) * 1000,
                    },
                )
                return finish(
                    StopReason.INTERRUPTED
                    if run_failure.kind == FailureKind.INTERRUPTED
                    else (
                        StopReason.MODEL_PROTOCOL_ERROR
                        if run_failure.kind == FailureKind.PROTOCOL
                        else StopReason.PROVIDER_ERROR
                    )
                )
            try:
                _validate_response(response, seen_ids)
            except (ValueError, TypeError, AttributeError):
                run_failure = FailureInfo(FailureKind.PROTOCOL, "invalid_model_response")
                emit(
                    trace,
                    "model.error",
                    {
                        "round": model_rounds,
                        "error_kind": run_failure.kind.value,
                        "error_code": run_failure.code,
                    },
                )
                return finish(StopReason.MODEL_PROTOCOL_ERROR)
            if response.usage is not None:
                usage.append(response.usage)
            emit(
                trace,
                "model.response",
                {
                    "round": model_rounds,
                    "item_count": len(response.tool_calls),
                    "duration_ms": (time.monotonic() - request_started) * 1000,
                    **(asdict(response.usage) if response.usage is not None else {}),
                },
            )
            if not response.tool_calls:
                append(Message("assistant", response.content))
                return finish(StopReason.COMPLETED, response.content)
            if model_rounds == self._limits.max_model_rounds:
                return finish(StopReason.MAX_MODEL_ROUNDS)
            if len(response.tool_calls) > self._limits.max_tool_calls - tool_calls:
                return finish(StopReason.MAX_TOOL_CALLS)
            if frame.budget.models_remaining <= 0:
                return finish(StopReason.MAX_TOTAL_MODEL_CALLS)
            if not frame.budget.tools_fit(len(response.tool_calls)):
                return finish(StopReason.MAX_TOTAL_TOOL_CALLS)
            seen_ids.update(call.call_id for call in response.tool_calls)
            for call in response.tool_calls:
                # A nested delegation may have consumed budget since the batch precheck.
                if frame.budget.models_remaining <= 0:
                    return finish(StopReason.MAX_TOTAL_MODEL_CALLS)
                if not frame.budget.tools_fit(1):
                    return finish(StopReason.MAX_TOTAL_TOOL_CALLS)
                frame.budget.tool_calls += 1
                call_fields: dict[str, object] = {
                    "round": model_rounds,
                    "tool_attempt": tool_calls + 1,
                    "tool_name": call.name if self._registry.get(call.name) else "unregistered",
                    "call_ref": hashlib.sha256(
                        call.call_id.encode("utf-8", errors="surrogatepass")
                    ).hexdigest()[:16],
                }
                tool_started = time.monotonic()
                append(call)
                tool_calls += 1  # Charge every dispatch attempt, including denied/invalid calls.
                try:
                    result = self._registry.invoke(
                        call,
                        trace=trace,
                        round_number=model_rounds,
                        tool_attempt=tool_calls,
                    )
                except KeyboardInterrupt:
                    run_failure = FailureInfo(FailureKind.INTERRUPTED, "user_interrupted")
                    result = ToolResult(
                        call.call_id,
                        call.name,
                        error=ToolError("interrupted", "tool execution interrupted", False),
                    )
                except Exception:
                    result = ToolResult(
                        call.call_id,
                        call.name,
                        error=ToolError("dispatch_error", "tool dispatch failed", False),
                    )
                append(result)
                emit(
                    trace,
                    "tool.result",
                    {
                        **call_fields,
                        "success": result.ok,
                        "content_bytes": len(result.content.encode("utf-8", errors="replace")),
                        "error_code": result.error.code if result.error else None,
                        "duration_ms": (time.monotonic() - tool_started) * 1000,
                    },
                )
                if run_failure is not None and run_failure.kind == FailureKind.INTERRUPTED:
                    return finish(StopReason.INTERRUPTED)
                if result.error is not None and not result.error.recoverable:
                    return finish(StopReason.TOOL_ERROR)
        return finish(StopReason.MAX_MODEL_ROUNDS)
