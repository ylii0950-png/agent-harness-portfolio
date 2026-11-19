import json

import pytest

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.permissions import Permission
from agent_harness_lab.providers import MockInvalidResponse, MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.todo_tools import register_todo_tools
from agent_harness_lab.tools import ToolRegistry


def create() -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                "create",
                "todo_write",
                json.dumps(
                    {
                        "operation": "create",
                        "items_json": json.dumps(
                            [
                                {"id": "one", "description": "private first task"},
                                {"id": "two", "description": "private second task"},
                            ]
                        ),
                    }
                ),
            ),
        )
    )


def read(identity: str) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(identity, "todo_read", "{}"),))


def update(identity: str, task_id: str, status: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                identity,
                "todo_write",
                json.dumps({"operation": "update", "task_id": task_id, "status": status}),
            ),
        )
    )


def registry() -> ToolRegistry:
    tools = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
    register_todo_tools(tools)
    return tools


def test_full_lifecycle_is_explicit_and_each_tool_result_reaches_the_model() -> None:
    provider = MockProvider(
        [
            create(),
            read("initial"),
            update("a", "one", "in_progress"),
            update("b", "one", "completed"),
            update("c", "two", "in_progress"),
            update("d", "two", "completed"),
            read("final"),
            "completed",
        ]
    )
    result = Agent(provider, registry(), limits=RunLimits(8, 7)).run("perform two steps")
    assert result.stop_reason == StopReason.COMPLETED and result.tool_calls == 7
    assert all(output.ok for output in result.tool_results)
    assert json.loads(result.tool_results[-1].content)["counts"] == {
        "pending": 0,
        "in_progress": 0,
        "completed": 2,
    }
    for index, output in enumerate(result.tool_results):
        assert provider.calls[index + 1][-1] == output


def test_final_answer_claim_does_not_complete_or_replace_pending_tasks() -> None:
    provider = MockProvider([create(), "I completed every task."])
    result = Agent(provider, registry()).run("make a plan")
    assert result.stop_reason == StopReason.COMPLETED
    plan = json.loads(result.tool_results[-1].content)
    assert plan["counts"] == {"pending": 2, "in_progress": 0, "completed": 0}
    assert all(task["status"] == "pending" for task in plan["tasks"])


def test_same_agent_and_registry_have_no_plan_state_in_the_next_run() -> None:
    provider = MockProvider(
        [create(), "first", read("second_read"), update("old_id", "one", "in_progress"), "second"]
    )
    agent = Agent(provider, registry())
    first = agent.run("first run with private task content")
    second = agent.run("second run")
    assert json.loads(first.tool_results[0].content)["counts"]["pending"] == 2
    assert json.loads(second.tool_results[0].content)["tasks"] == []
    assert (
        second.tool_results[1].error is not None
        and second.tool_results[1].error.code == "todo_unknown_id"
    )
    assert "private" not in repr(provider.calls[2])
    assert second.stop_reason == StopReason.COMPLETED


def test_different_agents_sharing_a_registry_cannot_observe_each_others_plan() -> None:
    shared = registry()
    Agent(MockProvider([create(), "first"]), shared).run("first")
    second = Agent(MockProvider([read("read"), "second"]), shared).run("second")
    assert json.loads(second.tool_results[0].content)["tasks"] == []


@pytest.mark.parametrize("broken", [False, True])
def test_run_scope_is_released_after_protocol_or_provider_failure(broken: bool) -> None:
    step = MockInvalidResponse(None) if broken else RuntimeError("private provider failure")
    provider = MockProvider([create(), step, read("new"), "new final"])
    agent = Agent(provider, registry())
    failed = agent.run("old")
    assert failed.stop_reason in (StopReason.PROVIDER_ERROR, StopReason.MODEL_PROTOCOL_ERROR)
    next_run = agent.run("new")
    assert json.loads(next_run.tool_results[0].content)["tasks"] == []


@pytest.mark.parametrize("limits", [RunLimits(1, 10), RunLimits(3, 0)])
def test_limits_prevent_plan_creation_and_later_runs_still_start_empty(limits: RunLimits) -> None:
    shared = registry()
    failed = Agent(MockProvider([create()]), shared, limits=limits).run("blocked")
    assert failed.tool_calls == 0
    later = Agent(MockProvider([read("read"), "done"]), shared).run("later")
    assert json.loads(later.tool_results[0].content)["tasks"] == []
