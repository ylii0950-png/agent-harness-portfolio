import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from agent_harness_lab.contracts import (
    ParameterType,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from agent_harness_lab.providers import Message, MockInvalidResponse, MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.tools import (
    Argument,
    Permission,
    ToolExecutionError,
    ToolRegistry,
    register_workspace_tools,
)
from agent_harness_lab.workspace import WorkspaceFiles


def echo_registry(effects: list[str]) -> ToolRegistry:
    registry = ToolRegistry()

    def echo(arguments: Mapping[str, Argument]) -> str:
        value = str(arguments["text"])
        effects.append(value)
        return value

    registry.register(
        ToolDefinition("echo", "Echo text.", (ToolParameter("text", ParameterType.STRING),)),
        echo,
        permission=Permission.READ,
    )
    return registry


def request(call_id: str = "call_1", text: str = "hello") -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, "echo", json.dumps({"text": text})),))


def test_direct_final_answer() -> None:
    provider = MockProvider(["done"])
    result = Agent(provider, ToolRegistry()).run("task")
    assert result.stop_reason == StopReason.COMPLETED
    assert result.final_answer == "done"
    assert (result.model_rounds, result.tool_calls) == (1, 0)
    assert result.messages == (Message("user", "task"), Message("assistant", "done"))


@pytest.mark.parametrize("count", [1, 3])
def test_consecutive_tools_feedback_and_deterministic_result(count: int) -> None:
    def execute() -> tuple[object, list[str]]:
        effects: list[str] = []
        registry = echo_registry(effects)
        provider = MockProvider([*[request(f"call_{i}", str(i)) for i in range(count)], "done"])
        result = Agent(provider, registry).run("task")
        assert result.stop_reason == StopReason.COMPLETED
        assert result.tool_calls == count and result.model_rounds == count + 1
        for index, tool_result in enumerate(result.tool_results):
            assert tool_result.call_id == f"call_{index}"
            assert tool_result.content == str(index) and tool_result.ok
            assert provider.calls[index + 1][-1] == tool_result
            assert provider.tool_catalogs[index] == registry.definitions
        return result, effects

    first, effects = execute()
    assert execute() == (first, effects)
    assert effects == [str(i) for i in range(count)]


def test_batch_runs_in_order_and_feedback_is_paired() -> None:
    effects: list[str] = []
    first, second = request("one", "a").tool_calls[0], request("two", "b").tool_calls[0]
    provider = MockProvider([ModelResponse(tool_calls=(first, second)), "done"])
    result = Agent(provider, echo_registry(effects), limits=RunLimits(2, 2)).run("task")
    assert result.stop_reason == StopReason.COMPLETED
    assert effects == ["a", "b"]
    assert provider.calls[1] == (
        Message("user", "task"),
        first,
        ToolResult("one", "echo", "a"),
        second,
        ToolResult("two", "echo", "b"),
    )


@pytest.mark.parametrize(
    "name,arguments,code",
    [
        ("unknown", "{}", "unknown_tool"),
        ("echo", '{"text":4}', "invalid_arguments"),
        ("echo", "{", "invalid_arguments"),
    ],
)
def test_recoverable_tool_error_can_be_corrected(name: str, arguments: str, code: str) -> None:
    effects: list[str] = []
    provider = MockProvider(
        [
            ModelResponse(tool_calls=(ToolCall("bad", name, arguments),)),
            request("fixed", "corrected"),
            "done",
        ]
    )
    result = Agent(provider, echo_registry(effects)).run("task")
    assert result.stop_reason == StopReason.COMPLETED
    error = result.tool_results[0].error
    assert error is not None and error.code == code and error.recoverable
    assert provider.calls[1][-1] == result.tool_results[0]
    assert result.tool_calls == 2 and effects == ["corrected"]


@pytest.mark.parametrize("recoverable", [False, True])
def test_curated_error_recoverability_controls_continuation(recoverable: bool) -> None:
    registry = ToolRegistry()

    def handler(arguments: Mapping[str, Argument]) -> str:
        raise ToolExecutionError("temporary", "curated error", recoverable=recoverable)

    registry.register(
        ToolDefinition("fail", "Fail deliberately."), handler, permission=Permission.READ
    )
    provider = MockProvider([ModelResponse(tool_calls=(ToolCall("one", "fail", "{}"),)), "done"])
    result = Agent(provider, registry).run("task")
    assert result.stop_reason == (StopReason.COMPLETED if recoverable else StopReason.TOOL_ERROR)
    assert len(provider.calls) == (2 if recoverable else 1)
    assert result.tool_results[0].error is not None
    assert result.tool_results[0].error.recoverable is recoverable


def test_unexpected_handler_exception_is_fatal_and_stops_remaining_batch() -> None:
    effects: list[str] = []
    registry = echo_registry(effects)

    def fail(arguments: Mapping[str, Argument]) -> str:
        raise RuntimeError("private exception text")

    registry.register(ToolDefinition("fail", "Fail."), fail, permission=Permission.READ)
    provider = MockProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall("bad", "fail", "{}"),
                    request("not_run").tool_calls[0],
                )
            ),
            "not requested",
        ]
    )
    result = Agent(provider, registry).run("task")
    assert result.stop_reason == StopReason.TOOL_ERROR
    assert result.final_answer is None and result.tool_calls == 1
    assert effects == [] and len(provider.calls) == 1
    assert len(result.messages) == 3
    assert result.tool_results[0].error is not None
    assert result.tool_results[0].error.code == "execution_error"
    assert "private exception text" not in repr(result)


@pytest.mark.parametrize(
    "step", [RuntimeError("private provider detail"), MockInvalidResponse(None)]
)
def test_provider_failure_and_protocol_failure_are_distinct(step: Any) -> None:
    provider = MockProvider([step])
    result = Agent(provider, ToolRegistry()).run("task")
    assert result.stop_reason == (
        StopReason.PROVIDER_ERROR
        if isinstance(step, Exception)
        else StopReason.MODEL_PROTOCOL_ERROR
    )
    assert result.model_rounds == 1 and result.tool_calls == 0
    assert result.messages == (Message("user", "task"),)
    assert "private provider detail" not in repr(result)


def test_exhausted_mock_stops_as_provider_error() -> None:
    result = Agent(MockProvider([]), ToolRegistry()).run("task")
    assert result.stop_reason == StopReason.PROVIDER_ERROR


@pytest.mark.parametrize(
    "response",
    [
        MockInvalidResponse({"content": "not our protocol"}),
        ModelResponse("ambiguous", tool_calls=(ToolCall("one", "echo", "{}"),)),
        ModelResponse(tool_calls=cast(Any, [ToolCall("one", "echo", "{}")])),
        ModelResponse(tool_calls=cast(Any, (None,))),
        ModelResponse(tool_calls=(ToolCall("", "echo", "{}"),)),
        ModelResponse(tool_calls=(ToolCall("one", "", "{}"),)),
        ModelResponse(tool_calls=(ToolCall("one", "echo", cast(Any, {})),)),
        ModelResponse(tool_calls=(ToolCall("same", "echo", "{}"), ToolCall("same", "echo", "{}"))),
    ],
)
def test_entire_malformed_decision_is_rejected_before_side_effects(response: Any) -> None:
    effects: list[str] = []
    result = Agent(MockProvider([response]), echo_registry(effects)).run("task")
    assert result.stop_reason == StopReason.MODEL_PROTOCOL_ERROR
    assert result.tool_calls == 0 and effects == []


def test_duplicate_id_in_later_round_does_not_dispatch_again() -> None:
    effects: list[str] = []
    provider = MockProvider([request("same"), request("same")])
    result = Agent(provider, echo_registry(effects)).run("task")
    assert result.stop_reason == StopReason.MODEL_PROTOCOL_ERROR
    assert result.tool_calls == 1 and effects == ["hello"]


def test_bad_second_call_prevents_first_valid_call_side_effect() -> None:
    effects: list[str] = []
    decision = ModelResponse(
        tool_calls=(request().tool_calls[0], ToolCall("bad", "echo", cast(Any, None)))
    )
    result = Agent(MockProvider([decision]), echo_registry(effects)).run("task")
    assert result.stop_reason == StopReason.MODEL_PROTOCOL_ERROR
    assert result.tool_calls == 0 and effects == []


@pytest.mark.parametrize("rounds", [0, 1, 4])
def test_model_round_boundary_stops_before_extra_tool_effects(rounds: int) -> None:
    effects: list[str] = []
    provider = MockProvider([request(str(i)) for i in range(20)])
    result = Agent(provider, echo_registry(effects), limits=RunLimits(rounds, 100)).run("task")
    assert result.stop_reason == StopReason.MAX_MODEL_ROUNDS
    assert result.model_rounds == len(provider.calls) == rounds
    assert result.tool_calls == len(effects) == max(0, rounds - 1)


def test_final_answer_allowed_on_last_model_round_and_zero_tool_budget() -> None:
    result = Agent(MockProvider(["done"]), ToolRegistry(), limits=RunLimits(1, 0)).run("task")
    assert result.stop_reason == StopReason.COMPLETED


@pytest.mark.parametrize("budget,batch_size", [(0, 1), (1, 2)])
def test_whole_batch_refused_before_tool_budget_side_effects(budget: int, batch_size: int) -> None:
    effects: list[str] = []
    batch = ModelResponse(
        tool_calls=tuple(request(str(i)).tool_calls[0] for i in range(batch_size))
    )
    result = Agent(MockProvider([batch]), echo_registry(effects), limits=RunLimits(5, budget)).run(
        "task"
    )
    assert result.stop_reason == StopReason.MAX_TOOL_CALLS
    assert result.tool_calls == 0 and effects == []


def test_later_batch_exceeding_remaining_budget_does_not_partially_execute() -> None:
    effects: list[str] = []
    batch = ModelResponse(tool_calls=(request("two").tool_calls[0], request("three").tool_calls[0]))
    result = Agent(
        MockProvider([request(), batch]), echo_registry(effects), limits=RunLimits(5, 2)
    ).run("task")
    assert result.stop_reason == StopReason.MAX_TOOL_CALLS
    assert result.tool_calls == 1 and len(effects) == 1


def test_denied_and_unknown_dispatches_still_consume_tool_budget() -> None:
    script = [ModelResponse(tool_calls=(ToolCall(str(i), "unknown", "{}"),)) for i in range(8)]
    provider = MockProvider(script)
    result = Agent(provider, ToolRegistry(), limits=RunLimits(10, 2)).run("task")
    assert result.stop_reason == StopReason.MAX_TOOL_CALLS
    assert result.tool_calls == 2 and result.model_rounds == 3
    assert len(result.tool_results) == 2


@pytest.mark.parametrize("value", [-1, True, 1.5, "3", None])
def test_invalid_limits(value: Any) -> None:
    with pytest.raises(ValueError):
        RunLimits(value, 2)
    with pytest.raises(ValueError):
        RunLimits(2, value)


@pytest.mark.parametrize("task", [None, 3, "", "  "])
def test_bad_task_does_not_call_provider(task: Any) -> None:
    provider = MockProvider(["never"])
    with pytest.raises((TypeError, ValueError)):
        Agent(provider, ToolRegistry()).run(task)
    assert provider.calls == ()


def test_run_transcripts_and_call_id_sets_do_not_leak_between_tasks() -> None:
    effects: list[str] = []
    provider = MockProvider([request(), "one", request(), "two"])
    agent = Agent(provider, echo_registry(effects))
    first, second = agent.run("first task"), agent.run("second task")
    assert first.stop_reason == second.stop_reason == StopReason.COMPLETED
    assert provider.calls[2] == (Message("user", "second task"),)
    assert first.messages[0] == Message("user", "first task")


@pytest.mark.parametrize(
    "path", ["../outside.txt", "sub/../../outside.txt", ".git/config", "alias", "linked/file"]
)
def test_runtime_file_tools_cannot_bypass_workspace_containment(tmp_path: Path, path: str) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("unchanged")
    (root / "alias").symlink_to(outside)
    (root / "linked").symlink_to(tmp_path, target_is_directory=True)
    provider = MockProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "attempt", "write_file", json.dumps({"path": path, "content": "attack"})
                    ),
                )
            ),
            "denial handled",
        ]
    )
    with WorkspaceFiles(root, allow_write=True) as workspace:
        registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
        register_workspace_tools(registry, workspace)
        result = Agent(provider, registry).run("task")
    assert result.stop_reason == StopReason.COMPLETED
    assert result.tool_results[0].error is not None
    assert result.tool_results[0].error.code == "workspace_path_denied"
    assert outside.read_text() == "unchanged" and not (tmp_path / "file").exists()


@pytest.mark.parametrize(
    "registry_write,workspace_write", [(False, False), (False, True), (True, False), (True, True)]
)
def test_both_registry_and_workspace_write_permissions_are_required(
    tmp_path: Path, registry_write: bool, workspace_write: bool
) -> None:
    provider = MockProvider(
        [
            ModelResponse(
                tool_calls=(ToolCall("write", "write_file", '{"path":"file","content":"hello"}'),)
            ),
            "done",
        ]
    )
    with WorkspaceFiles(tmp_path, allow_write=workspace_write) as workspace:
        registry = ToolRegistry(
            permissions=(Permission.READ, Permission.WRITE)
            if registry_write
            else (Permission.READ,)
        )
        register_workspace_tools(registry, workspace)
        result = Agent(provider, registry).run("task")
    assert result.tool_results[0].ok is (registry_write and workspace_write)
    assert (tmp_path / "file").exists() is (registry_write and workspace_write)
    if not (registry_write and workspace_write):
        error = result.tool_results[0].error
        assert error is not None
        assert error.code == (
            "workspace_permission_denied" if registry_write else "permission_denied"
        )


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "alias", "linked/secret"])
def test_read_escape_is_denied_before_content_reaches_provider(tmp_path: Path, path: str) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("outside private text")
    (root / "alias").symlink_to(secret)
    (root / "linked").symlink_to(tmp_path, target_is_directory=True)
    provider = MockProvider(
        [
            ModelResponse(tool_calls=(ToolCall("read", "read_file", json.dumps({"path": path})),)),
            "denied",
        ]
    )
    with WorkspaceFiles(root) as workspace:
        registry = ToolRegistry()
        register_workspace_tools(registry, workspace)
        result = Agent(provider, registry).run("read the requested file")
    error = result.tool_results[0].error
    assert error is not None and error.code == "workspace_path_denied"
    assert result.stop_reason == StopReason.COMPLETED
    assert "outside private text" not in repr(provider.calls)
