import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from agent_harness_lab.contracts import ToolCall, ToolDefinition, ToolResult
from agent_harness_lab.execution import TotalLimits
from agent_harness_lab.permissions import Argument, Permission
from agent_harness_lab.providers import ConversationItem, MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.subagents import register_subagent_tool
from agent_harness_lab.tools import ToolRegistry
from agent_harness_lab.tracing import JSONLTrace


def request(name: str, identity: str = "one") -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(identity, name, '{"task":"focused"}' if name == "delegate_task" else "{}"),
        )
    )


def configure(
    child: MockProvider,
    effects: list[str],
    *,
    child_limits: RunLimits | None = None,
) -> tuple[ToolRegistry, list[str]]:
    registry = ToolRegistry()

    def effect(arguments: Mapping[str, Argument]) -> str:
        effects.append("executed")
        return "effect result"

    registry.register(
        ToolDefinition("effect", "Record an effect."), effect, permission=Permission.READ
    )
    created: list[str] = []

    def factory() -> MockProvider:
        created.append("child")
        return child

    register_subagent_tool(
        registry,
        factory,
        child_tools=("effect", "delegate_task"),
        child_limits=child_limits or RunLimits(4, 5),
    )
    return registry, created


def summary(result: Any) -> dict[str, Any]:
    return dict(json.loads(result.tool_results[0].content))


@pytest.mark.parametrize("model_limit,expected_effects", [(0, 0), (1, 0), (2, 1), (3, 2)])
def test_child_local_model_limit_before_extra_side_effect(
    model_limit: int, expected_effects: int
) -> None:
    effects: list[str] = []
    child = MockProvider([request("effect", str(i)) for i in range(6)])
    registry, _ = configure(child, effects, child_limits=RunLimits(model_limit, 20))
    result = Agent(
        MockProvider([request("delegate_task"), "parent done"]),
        registry,
        total_limits=TotalLimits(20, 30),
    ).run("parent")
    assert result.stop_reason == StopReason.COMPLETED
    assert (
        summary(result)["status"] == "limit_reached"
        and summary(result)["stop_reason"] == "max_model_rounds"
    )
    assert len(child.calls) == model_limit and len(effects) == expected_effects


@pytest.mark.parametrize("tool_limit,batch", [(0, 1), (1, 2), (2, 3)])
def test_child_whole_batch_over_local_tool_limit_has_no_side_effect(
    tool_limit: int, batch: int
) -> None:
    effects: list[str] = []
    calls = tuple(ToolCall(str(i), "effect", "{}") for i in range(batch))
    child = MockProvider([ModelResponse(tool_calls=calls)])
    registry, _ = configure(child, effects, child_limits=RunLimits(4, tool_limit))
    result = Agent(MockProvider([request("delegate_task"), "done"]), registry).run("parent")
    assert summary(result)["stop_reason"] == "max_tool_calls" and effects == []
    assert summary(result)["tool_calls"] == 0


@pytest.mark.parametrize("limit", [0, 1])
def test_root_total_model_limit_prevents_child_factory(limit: int) -> None:
    effects: list[str] = []
    child = MockProvider(["never"])
    registry, created = configure(child, effects)
    parent = MockProvider([request("delegate_task")])
    result = Agent(parent, registry, total_limits=TotalLimits(limit, 20)).run("parent")
    assert result.stop_reason == StopReason.MAX_TOTAL_MODEL_CALLS
    assert result.model_rounds == len(parent.calls) == limit and created == []
    assert result.total_tool_calls == 0


def test_shared_model_budget_stops_child_before_last_turn_tool_or_approval() -> None:
    effects: list[str] = []
    child = MockProvider([request("effect")])
    registry, _ = configure(child, effects)
    parent = MockProvider([request("delegate_task"), "must not request this"])
    result = Agent(parent, registry, total_limits=TotalLimits(2, 20)).run("parent")
    assert summary(result)["stop_reason"] == "max_total_model_calls"
    assert result.stop_reason == StopReason.MAX_TOTAL_MODEL_CALLS and effects == []
    assert result.total_model_calls == 2 and result.total_tool_calls == 1
    assert len(parent.calls) == len(child.calls) == 1


def test_child_can_finish_at_exact_total_model_limit_but_parent_cannot_call_again() -> None:
    effects: list[str] = []
    child = MockProvider(["child finished"])
    registry, _ = configure(child, effects)
    parent = MockProvider([request("delegate_task"), "unused"])
    result = Agent(parent, registry, total_limits=TotalLimits(2, 20)).run("parent")
    assert summary(result)["status"] == "completed" and result.total_model_calls == 2
    assert result.stop_reason == StopReason.MAX_TOTAL_MODEL_CALLS and len(parent.calls) == 1


@pytest.mark.parametrize("total_tools", [0, 1])
def test_shared_tool_budget_includes_delegation_and_child_attempts(total_tools: int) -> None:
    effects: list[str] = []
    child = MockProvider([request("effect")])
    registry, created = configure(child, effects)
    parent = MockProvider([request("delegate_task"), "parent done"])
    result = Agent(parent, registry, total_limits=TotalLimits(10, total_tools)).run("parent")
    assert effects == [] and result.total_tool_calls == total_tools
    if total_tools == 0:
        assert created == [] and result.stop_reason == StopReason.MAX_TOTAL_TOOL_CALLS
    else:
        assert summary(result)["stop_reason"] == "max_total_tool_calls"
        assert result.stop_reason == StopReason.COMPLETED


def test_nested_consumption_rechecks_parent_pending_batch_before_side_effect() -> None:
    effects: list[str] = []
    child = MockProvider([request("effect"), "child done"])
    registry, _ = configure(child, effects)
    parent = MockProvider(
        [
            ModelResponse(
                tool_calls=(
                    request("delegate_task").tool_calls[0],
                    ToolCall("after", "effect", "{}"),
                )
            )
        ]
    )
    result = Agent(parent, registry, total_limits=TotalLimits(10, 2)).run("parent")
    assert result.stop_reason == StopReason.MAX_TOTAL_TOOL_CALLS
    assert result.total_tool_calls == 2 and result.tool_calls == 1 and effects == ["executed"]
    assert summary(result)["tool_calls"] == 1


def test_parent_batch_precheck_rejects_all_before_any_child_or_handler() -> None:
    effects: list[str] = []
    child = MockProvider(["unused"])
    registry, created = configure(child, effects)
    parent = MockProvider(
        [
            ModelResponse(
                tool_calls=(
                    request("delegate_task").tool_calls[0],
                    ToolCall("after", "effect", "{}"),
                )
            )
        ]
    )
    result = Agent(parent, registry, total_limits=TotalLimits(10, 1)).run("parent")
    assert result.stop_reason == StopReason.MAX_TOTAL_TOOL_CALLS
    assert effects == [] and created == [] and result.total_tool_calls == 0


@pytest.mark.parametrize("depth", [0, 1])
def test_depth_limit_blocks_recursion_without_new_provider(depth: int, tmp_path: Path) -> None:
    effects: list[str] = []
    child = MockProvider([request("delegate_task"), "child survived"])
    registry, created = configure(child, effects)
    path = tmp_path / "trace.jsonl"
    with JSONLTrace(path) as trace:
        result = Agent(
            MockProvider([request("delegate_task"), "parent done"]),
            registry,
            total_limits=TotalLimits(10, 10, depth),
            trace=trace,
        ).run("parent")
    if depth == 0:
        assert summary(result)["stop_reason"] == "max_delegation_depth" and created == []
    else:
        assert created == ["child"] and len(child.calls) == 2
        nested = json.loads(cast(ToolResult, child.calls[1][-1]).content)
        assert (
            nested["status"] == "limit_reached" and nested["stop_reason"] == "max_delegation_depth"
        )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(
        e["event"] == "subagent_limit_reached" and e["stop_reason"] == "max_delegation_depth"
        for e in events
    )
    assert registry.execution is None and "effect" in registry.names


@pytest.mark.parametrize("value", [-1, True, "2", None, 1.2])
def test_total_limit_configuration_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(ValueError):
        TotalLimits(value, 2)
    with pytest.raises(ValueError):
        TotalLimits(2, value)
    with pytest.raises(ValueError):
        TotalLimits(2, 2, value)


def test_deeper_recursion_configuration_is_not_implemented() -> None:
    with pytest.raises(ValueError):
        TotalLimits(10, 10, 2)


def test_tool_scope_intersection_restores_after_exception_and_never_expands() -> None:
    effects: list[str] = []
    registry, _ = configure(MockProvider(["unused"]), effects)
    with pytest.raises(RuntimeError), registry.restrict_tools(("effect",)):
        with registry.restrict_tools(("effect", "delegate_task")):
            assert registry.names == ("effect",)
            assert registry.invoke(request("delegate_task").tool_calls[0]).error is not None
        raise RuntimeError("test cleanup")
    assert registry.get("delegate_task") is not None and effects == []


class InterruptProvider:
    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        raise KeyboardInterrupt


def test_child_interrupt_is_structured_and_parent_state_remains_usable() -> None:
    registry = ToolRegistry()
    register_subagent_tool(registry, InterruptProvider, child_tools=())
    result = Agent(MockProvider([request("delegate_task"), "parent done"]), registry).run("parent")
    assert summary(result)["stop_reason"] == "interrupted"
    assert result.stop_reason == StopReason.COMPLETED and registry.execution is None


@pytest.mark.parametrize(
    "failure", [RuntimeError("private factory error"), SystemExit("private exit")]
)
def test_factory_exception_returns_sanitized_failure_and_parent_continues(
    failure: BaseException,
) -> None:
    def factory() -> MockProvider:
        raise failure

    registry = ToolRegistry()
    register_subagent_tool(registry, factory, child_tools=())
    result = Agent(MockProvider([request("delegate_task"), "done"]), registry).run("parent")
    assert (
        summary(result)["error_code"] == "child_execution_failed"
        and summary(result)["model_calls"] == 0
    )
    assert "private" not in str(summary(result)) and registry.execution is None


def test_unrecoverable_child_handler_failure_is_not_parent_fatal() -> None:
    def broken(arguments: Mapping[str, Argument]) -> str:
        raise RuntimeError("private child exception")

    registry = ToolRegistry()
    registry.register(ToolDefinition("broken", "Fail."), broken, permission=Permission.READ)
    register_subagent_tool(
        registry, lambda: MockProvider([request("broken")]), child_tools=("broken",)
    )
    result = Agent(MockProvider([request("delegate_task"), "done"]), registry).run("parent")
    assert summary(result)["stop_reason"] == "tool_error" and summary(result)["status"] == "failed"
    assert result.stop_reason == StopReason.COMPLETED and "private" not in str(summary(result))


def test_child_logging_failure_blocks_handler_and_scope_is_released() -> None:
    class BrokenTrace:
        def record(self, event: str, **fields: object) -> None:
            if event == "tool.call" and fields["depth"] == 1:
                raise OSError("private trace failure")

    effects: list[str] = []
    registry, _ = configure(MockProvider([request("effect")]), effects)
    result = Agent(
        MockProvider([request("delegate_task"), "done"]), registry, trace=BrokenTrace()
    ).run("parent")
    assert effects == [] and summary(result)["status"] == "failed"
    assert registry.execution is None and registry.invocation_trace is None


@pytest.mark.parametrize(
    "event,step,limits",
    [
        ("subagent_failed", RuntimeError("private"), RunLimits(3, 2)),
        ("subagent_limit_reached", request("effect"), RunLimits(1, 2)),
    ],
)
def test_child_terminal_trace_event_is_formal_and_non_sensitive(
    event: str, step: Any, limits: RunLimits, tmp_path: Path
) -> None:
    effects: list[str] = []
    registry, _ = configure(MockProvider([step]), effects, child_limits=limits)
    path = tmp_path / "trace.jsonl"
    with JSONLTrace(path) as trace:
        result = Agent(MockProvider([request("delegate_task"), "done"]), registry, trace=trace).run(
            "private root"
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    terminal = [row for row in rows if row["event"] == event]
    assert len(terminal) == 1 and terminal[0]["depth"] == 1
    assert terminal[0]["model_rounds"] == summary(result)["model_calls"] == 1
    assert "private" not in path.read_text() and effects == []


def test_async_factory_is_rejected_and_sync_coroutine_wrapper_is_closed() -> None:
    async def asynchronous() -> MockProvider:
        return MockProvider(["never"])

    registry = ToolRegistry()
    with pytest.raises(TypeError):
        register_subagent_tool(registry, cast(Any, asynchronous), child_tools=())
    register_subagent_tool(registry, cast(Any, lambda: asynchronous()), child_tools=())
    result = Agent(MockProvider([request("delegate_task"), "done"]), registry).run("parent")
    assert summary(result)["status"] == "failed"


@pytest.mark.parametrize("names", ["read_file", ("not_registered",), (True,)])
def test_invalid_child_tool_configuration_is_rejected(names: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        register_subagent_tool(ToolRegistry(), lambda: MockProvider(["never"]), child_tools=names)


def test_default_child_catalog_cannot_invoke_delegation() -> None:
    child = MockProvider([request("delegate_task"), "denied"])
    registry = ToolRegistry()
    created: list[str] = []

    def factory() -> MockProvider:
        created.append("one")
        return child

    register_subagent_tool(registry, factory, child_tools=())
    result = Agent(MockProvider([request("delegate_task"), "done"]), registry).run("parent")
    rejected = cast(ToolResult, child.calls[1][-1])
    assert rejected.error is not None and rejected.error.code == "unknown_tool"
    assert child.tool_catalogs == ((), ()) and created == ["one"]
    assert summary(result)["status"] == "completed"


class SlottedProvider:
    __slots__ = ()

    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        return ModelResponse("slotted summary")


def test_slotted_provider_needs_no_weak_reference_or_mutable_attributes() -> None:
    assert Agent(SlottedProvider(), ToolRegistry()).run("direct").final_answer == "slotted summary"
    registry = ToolRegistry()
    register_subagent_tool(registry, SlottedProvider, child_tools=())
    result = Agent(MockProvider([request("delegate_task"), "parent done"]), registry).run("parent")
    assert summary(result)["status"] == "completed"
    assert summary(result)["result"] == "slotted summary" and summary(result)["model_calls"] == 1
