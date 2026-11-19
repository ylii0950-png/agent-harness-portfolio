"""One-level, synchronous delegation through the existing Agent and permission gate."""

import inspect
import json
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from typing import Protocol, cast

from agent_harness_lab.context import ContextPolicy
from agent_harness_lab.contracts import ParameterType, ToolDefinition, ToolParameter
from agent_harness_lab.permissions import Argument, Permission
from agent_harness_lab.providers import ModelProvider
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.tools import ToolExecutionError, ToolRegistry
from agent_harness_lab.tracing import emit


class ProviderFactory(Protocol):
    """Trusted factory must return a new, independent provider on every invocation."""

    def __call__(self) -> ModelProvider: ...


@dataclass(frozen=True, slots=True, repr=False)
class SubagentResult:
    status: str
    stop_reason: str
    result: str | None
    model_calls: int
    tool_calls: int
    error_code: str | None = None
    truncated: bool = False

    def __repr__(self) -> str:
        return f"SubagentResult(status={self.status!r}, stop_reason={self.stop_reason!r})"

    def output_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


_LIMITS = frozenset(
    {
        StopReason.MAX_MODEL_ROUNDS,
        StopReason.MAX_TOOL_CALLS,
        StopReason.MAX_TOTAL_MODEL_CALLS,
        StopReason.MAX_TOTAL_TOOL_CALLS,
    }
)


_DEFAULT_CHILD_LIMITS = RunLimits(6, 10)


class SubagentDispatcher:
    """Trusted configuration, no histories or providers cached across root runs."""

    def __init__(
        self,
        registry: ToolRegistry,
        provider_factory: ProviderFactory,
        *,
        child_tools: Collection[str],
        child_limits: RunLimits = _DEFAULT_CHILD_LIMITS,
        child_context_policy: ContextPolicy | None = None,
    ) -> None:
        if (
            not callable(provider_factory)
            or inspect.iscoroutinefunction(provider_factory)
            or inspect.iscoroutinefunction(provider_factory.__call__)
            or type(child_limits) is not RunLimits
        ):
            raise TypeError("expected synchronous provider factory and RunLimits")
        if isinstance(child_tools, (str, bytes)) or any(
            type(name) is not str or registry.get(name) is None
            for name in child_tools
            if name != "delegate_task"
        ):
            raise ValueError("child tools must be registered tool names")
        if child_context_policy is not None and type(child_context_policy) is not ContextPolicy:
            raise TypeError("expected child ContextPolicy")
        self._context_policy = child_context_policy
        self._registry = registry
        self._factory = provider_factory
        self._tools = frozenset(child_tools)
        self._limits = child_limits

    def _dispatch(self, arguments: Mapping[str, Argument]) -> str:
        frame = self._registry.execution
        if frame is None:
            raise ToolExecutionError("run_scope_required", "delegation requires an Agent run")
        task = cast(str, arguments["task"])
        context = cast(str, arguments.get("context", ""))
        try:
            if not task.strip() or any(len(s.encode("utf-8")) > 65536 for s in (task, context)):
                raise ValueError("invalid task")
        except (ValueError, UnicodeError):
            raise ToolExecutionError("invalid_subtask", "subtask text is invalid") from None
        child = frame.child()
        trace = self._registry.invocation_trace
        metadata: dict[str, object] = {
            "run_id": child.run_id,
            "parent_run_id": frame.run_id,
            "depth": child.depth,
        }

        def output(result: SubagentResult) -> str:
            event = {
                "completed": "subagent_completed",
                "failed": "subagent_failed",
                "limit_reached": "subagent_limit_reached",
            }[result.status]
            emit(
                trace,
                event,
                {
                    **metadata,
                    "status": result.status,
                    "stop_reason": result.stop_reason,
                    "model_rounds": result.model_calls,
                    "tool_calls": result.tool_calls,
                    "total_model_calls": frame.budget.model_calls,
                    "total_tool_calls": frame.budget.tool_calls,
                    "error_code": result.error_code,
                },
            )
            return result.output_json()

        if frame.depth >= frame.budget.limits.max_depth:
            return output(SubagentResult("limit_reached", "max_delegation_depth", None, 0, 0))
        if frame.budget.models_remaining <= 0:
            return output(SubagentResult("limit_reached", "max_total_model_calls", None, 0, 0))
        emit(trace, "subagent_started", metadata)
        before_models, before_tools = frame.budget.model_calls, frame.budget.tool_calls
        try:
            provider = self._factory()
            if inspect.iscoroutine(provider):
                provider.close()
                raise TypeError("provider factory must be synchronous")
            frame.budget.claim_provider(provider)
            # Shared immutable Engine, narrowed catalog/dispatch, fresh Todo scope/history.
            with self._registry.restrict_tools(self._tools):
                result = Agent(
                    provider,
                    self._registry,
                    limits=self._limits,
                    trace=trace,
                    context_policy=self._context_policy,
                )._run_with_frame(
                    task + ("\n\nExplicit context:\n" + context if context else ""), child
                )
        except (Exception, KeyboardInterrupt, SystemExit):
            return output(
                SubagentResult(
                    "failed",
                    "child_exception",
                    None,
                    frame.budget.model_calls - before_models,
                    frame.budget.tool_calls - before_tools,
                    "child_execution_failed",
                )
            )
        status = (
            "completed"
            if result.stop_reason == StopReason.COMPLETED
            else "limit_reached"
            if result.stop_reason in _LIMITS
            else "failed"
        )
        answer = result.final_answer if status == "completed" else None
        return output(
            SubagentResult(
                status,
                result.stop_reason.value,
                answer[:8192] if answer is not None else None,
                result.model_rounds,
                result.tool_calls,
                None if status != "failed" else "child_" + result.stop_reason.value,
                answer is not None and len(answer) > 8192,
            )
        )


def register_subagent_tool(
    registry: ToolRegistry,
    provider_factory: ProviderFactory,
    *,
    child_tools: Collection[str],
    child_limits: RunLimits = _DEFAULT_CHILD_LIMITS,
    child_context_policy: ContextPolicy | None = None,
) -> None:
    dispatcher = SubagentDispatcher(
        registry,
        provider_factory,
        child_tools=child_tools,
        child_limits=child_limits,
        child_context_policy=child_context_policy,
    )
    registry.register(
        ToolDefinition(
            "delegate_task",
            "Delegate one focused task in fresh context; return only status and final summary.",
            (
                ToolParameter("task", ParameterType.STRING),
                ToolParameter("context", ParameterType.STRING, required=False),
            ),
        ),
        dispatcher._dispatch,
        permission=Permission.READ,
    )
