import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from agent_harness_lab.contracts import ToolCall, ToolDefinition, ToolResult
from agent_harness_lab.execution import TotalLimits
from agent_harness_lab.permissions import ApprovalRequest, Decision, Permission, PermissionEngine
from agent_harness_lab.providers import Message, MockInvalidResponse, MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.subagents import register_subagent_tool
from agent_harness_lab.todo_tools import register_todo_tools
from agent_harness_lab.tools import Argument, ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles


def call(name: str = "delegate_task", identity: str = "one", **arguments: str) -> ModelResponse:
    if name == "delegate_task" and not arguments:
        arguments = {"task": "explicit child task"}
    return ModelResponse(tool_calls=(ToolCall(identity, name, json.dumps(arguments)),))


def payload(result: Any, index: int = 0) -> dict[str, Any]:
    output = result.tool_results[index]
    assert output.ok
    return dict(json.loads(output.content))


_DEFAULT_TEST_LIMITS = RunLimits()


def setup(
    children: list[MockProvider],
    *,
    registry: ToolRegistry | None = None,
    child_tools: tuple[str, ...] = (),
    child_limits: RunLimits = _DEFAULT_TEST_LIMITS,
) -> tuple[ToolRegistry, list[int]]:
    tools = registry if registry is not None else ToolRegistry()
    created: list[int] = []

    def factory() -> MockProvider:
        created.append(len(created))
        return children[len(created) - 1]

    register_subagent_tool(tools, factory, child_tools=child_tools, child_limits=child_limits)
    return tools, created


def test_parent_child_context_isolation_feedback_and_final_summary_only() -> None:
    effects: list[str] = []
    registry = ToolRegistry()

    def private(arguments: Mapping[str, Argument]) -> str:
        effects.append("private child tool content")
        return effects[-1]

    registry.register(
        ToolDefinition("private", "Read private fixture."), private, permission=Permission.READ
    )
    child = MockProvider([call("private"), "public child summary"])
    tools, created = setup([child], registry=registry, child_tools=("private",))
    parent = MockProvider(
        [call(task="focused request", context="explicit allowed context"), "parent done"]
    )
    result = Agent(parent, tools).run("PRIVATE parent original task/history")
    summary = payload(result)
    assert summary == {
        "status": "completed",
        "stop_reason": "completed",
        "result": "public child summary",
        "model_calls": 2,
        "tool_calls": 1,
        "error_code": None,
        "truncated": False,
    }
    assert created == [0] and effects == ["private child tool content"]
    assert child.calls[0] == (
        Message("user", "focused request\n\nExplicit context:\nexplicit allowed context"),
    )
    assert "PRIVATE" not in repr(child.calls)
    assert "private child tool content" not in repr(parent.calls)
    assert parent.calls[1][-1] == result.tool_results[0]
    assert result.final_answer == "parent done"
    assert (result.model_rounds, result.tool_calls) == (2, 1)
    assert (result.total_model_calls, result.total_tool_calls) == (4, 2)
    assert tools.execution is None and tools.invocation_trace is None


def test_multiple_children_and_independent_runs_do_not_share_context_or_todos() -> None:
    registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
    register_todo_tools(registry)
    plan = json.dumps([{"id": "parent", "description": "private parent plan"}])
    child_plan = json.dumps([{"id": "child", "description": "private child plan"}])
    first_child = MockProvider(
        [
            call("todo_read"),
            call("todo_write", "plan", operation="create", items_json=child_plan),
            "first summary",
        ]
    )
    second_child = MockProvider([call("todo_read"), "second summary"])
    tools, _ = setup(
        [first_child, second_child], registry=registry, child_tools=("todo_read", "todo_write")
    )
    parent = MockProvider(
        [
            call("todo_write", operation="create", items_json=plan),
            call(identity="child_one"),
            call(identity="child_two"),
            call("todo_read", "parent_read"),
            "done",
            call("todo_read", "new_read"),
            "new done",
        ]
    )
    agent = Agent(parent, tools, total_limits=TotalLimits(20, 20))
    result = agent.run("first root")
    assert json.loads(cast(ToolResult, first_child.calls[1][-1]).content)["tasks"] == []
    assert json.loads(cast(ToolResult, second_child.calls[1][-1]).content)["tasks"] == []
    assert "first summary" not in repr(second_child.calls)
    assert payload(result, 3)["tasks"][0]["id"] == "parent"
    assert payload(agent.run("second root"))["tasks"] == []


@pytest.mark.parametrize("policy", ["capability", "deny", "ask_missing", "ask_false", "ask_error"])
def test_child_cannot_elevate_write_permissions_or_modify_files(
    tmp_path: Path, policy: str
) -> None:
    approvals: list[ApprovalRequest] = []

    def approval(request: ApprovalRequest) -> bool:
        approvals.append(request)
        if policy == "ask_error":
            raise RuntimeError("private approval details")
        return False

    rules = {"write_file": Decision.DENY} if policy == "deny" else {"write_file": Decision.ASK}
    registry = ToolRegistry(
        permission_engine=PermissionEngine(
            permissions=(Permission.READ,)
            if policy == "capability"
            else (Permission.READ, Permission.WRITE),
            rules=rules,
            approval_handler=None if policy == "ask_missing" else approval,
        )
    )
    child = MockProvider([call("write_file", path="never.txt", content="private"), "denied safely"])
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        register_workspace_tools(registry, workspace)
        tools, _ = setup([child], registry=registry, child_tools=("write_file",))
        result = Agent(MockProvider([call(), "parent continues"]), tools).run("parent")
    assert result.stop_reason == StopReason.COMPLETED and not (tmp_path / "never.txt").exists()
    failed = cast(ToolResult, child.calls[1][-1])
    assert failed.error is not None and failed.error.code == "permission_denied"
    assert len(approvals) == (1 if policy in ("ask_false", "ask_error") else 0)
    assert "private approval details" not in result.tool_results[0].content


def test_each_child_write_requires_its_own_exact_approval(tmp_path: Path) -> None:
    seen: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        seen.append(request)
        return len(seen) == 1

    registry = ToolRegistry(
        permission_engine=PermissionEngine(
            permissions=(Permission.READ, Permission.WRITE),
            rules={"write_file": Decision.ASK},
            approval_handler=approve,
        )
    )
    child = MockProvider(
        [
            call("write_file", "a", path="one.txt", content="one"),
            call("write_file", "b", path="two.txt", content="two"),
            "done",
        ]
    )
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        register_workspace_tools(registry, workspace)
        tools, _ = setup([child], registry=registry, child_tools=("write_file",))
        Agent(MockProvider([call(), "done"]), tools).run("parent")
    assert (tmp_path / "one.txt").read_text() == "one" and not (tmp_path / "two.txt").exists()
    assert [r.call_id for r in seen] == ["a", "b"]


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "linked.txt"])
def test_child_cannot_escape_workspace_even_if_write_is_approved(tmp_path: Path, path: str) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("untouched")
    (tmp_path / "linked.txt").symlink_to(outside)
    registry = ToolRegistry(
        permission_engine=PermissionEngine(
            permissions=(Permission.READ, Permission.WRITE),
            rules={"write_file": Decision.ASK},
            approval_handler=lambda request: True,
        )
    )
    # overwrite is a boolean schema: use a genuine well-typed attack rather than fail on types.
    child = MockProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "x",
                        "write_file",
                        json.dumps({"path": path, "content": "bad", "overwrite": True}),
                    ),
                )
            ),
            "denied",
        ]
    )
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        register_workspace_tools(registry, workspace)
        tools, _ = setup([child], registry=registry, child_tools=("write_file",))
        Agent(MockProvider([call(), "done"]), tools).run("parent")
    rejected = cast(ToolResult, child.calls[1][-1])
    assert rejected.error is not None and rejected.error.code.startswith("workspace_")
    assert outside.read_text() == "untouched"


@pytest.mark.parametrize("name", ["unregistered", "write_file"])
def test_unknown_or_not_allowlisted_tool_never_reaches_handler(tmp_path: Path, name: str) -> None:
    registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
    child = MockProvider([call(name, path="no.txt", content="no"), "done"])
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        register_workspace_tools(registry, workspace)
        tools, _ = setup([child], registry=registry, child_tools=("read_file",))
        Agent(MockProvider([call(), "done"]), tools).run("parent")
    error = cast(ToolResult, child.calls[1][-1]).error
    assert error is not None and error.code == "unknown_tool" and not (tmp_path / "no.txt").exists()
    assert [d.name for d in child.tool_catalogs[0]] == ["read_file"]
    assert tools.get("write_file") is not None


@pytest.mark.parametrize("decision", [Decision.DENY, Decision.ASK])
def test_delegation_itself_is_gated_before_provider_factory(decision: Decision) -> None:
    registry = ToolRegistry(permission_engine=PermissionEngine(rules={"delegate_task": decision}))
    tools, created = setup([MockProvider(["must not run"])], registry=registry)
    result = Agent(MockProvider([call(), "denied"]), tools).run("parent")
    assert result.tool_results[0].error is not None and created == []


@pytest.mark.parametrize("broken", [RuntimeError("private failure"), MockInvalidResponse(None)])
def test_child_provider_failure_is_sanitized_and_parent_continues(broken: Any) -> None:
    child = MockProvider([broken])
    tools, _ = setup([child])
    result = Agent(MockProvider([call(), "parent survived"]), tools).run("parent")
    assert result.stop_reason == StopReason.COMPLETED and result.final_answer == "parent survived"
    summary = payload(result)
    assert summary["status"] == "failed" and summary["result"] is None
    assert summary["stop_reason"] in ("provider_error", "model_protocol_error")
    assert "private failure" not in str(summary)


def test_reusing_a_provider_between_siblings_is_rejected_without_private_context() -> None:
    child = MockProvider(["first", "must not run"])
    registry = ToolRegistry()
    register_subagent_tool(registry, lambda: child, child_tools=())
    parent = MockProvider([call(identity="a"), call(identity="b"), "done"])
    result = Agent(parent, registry).run("root")
    assert payload(result)["status"] == "completed"
    assert payload(result, 1)["status"] == "failed" and len(child.calls) == 1


def test_fresh_provider_factory_keeps_independent_root_runs_isolated() -> None:
    children: list[MockProvider] = []

    def factory() -> MockProvider:
        child = MockProvider(["summary"])
        children.append(child)
        return child

    registry = ToolRegistry()
    register_subagent_tool(registry, factory, child_tools=())
    agent = Agent(
        MockProvider([call(task="first private task"), "one", call(task="second task"), "two"]),
        registry,
    )
    assert payload(agent.run("private root"))["status"] == "completed"
    assert payload(agent.run("other root"))["status"] == "completed"
    assert len(children) == 2 and children[0] is not children[1]
    assert children[1].calls[0] == (Message("user", "second task"),)
    assert "private" not in repr(children[1].calls)


def test_parent_provider_cannot_be_reused_as_child() -> None:
    parent = MockProvider([call(), "parent result"])
    registry = ToolRegistry()
    register_subagent_tool(registry, lambda: parent, child_tools=())
    result = Agent(parent, registry).run("private parent")
    assert payload(result)["status"] == "failed" and len(parent.calls) == 2


@pytest.mark.parametrize("task", ["", " ", "x" * 65537, "\ud800"])
def test_invalid_subtask_creates_no_child(task: str) -> None:
    tools, created = setup([MockProvider(["never"])])
    result = Agent(MockProvider([call(task=task), "recovered"]), tools).run("parent")
    assert result.tool_results[0].error is not None and created == []


def test_final_summary_is_bounded_and_no_history_is_returned() -> None:
    child = MockProvider(["x" * 9000])
    tools, _ = setup([child])
    result = Agent(MockProvider([call(), "done"]), tools).run("parent")
    summary = payload(result)
    assert summary["truncated"] is True and len(summary["result"]) == 8192
    assert "messages" not in summary and "history" not in summary


def test_trace_links_root_child_and_omits_private_task_context_results(tmp_path: Path) -> None:
    child = MockProvider(["private child result"])
    tools, _ = setup([child])
    path = tmp_path / "trace.jsonl"
    with JSONLTrace(path) as trace:
        result = Agent(
            MockProvider(
                [call(task="private child task", context="private context"), "private parent final"]
            ),
            tools,
            trace=trace,
        ).run("private parent history")
    text = path.read_text()
    assert "private" not in text and "arguments" not in text
    events = [json.loads(line) for line in text.splitlines()]
    started = next(r for r in events if r["event"] == "subagent_started")
    completed = next(r for r in events if r["event"] == "subagent_completed")
    root = events[0]["run_id"]
    assert started["parent_run_id"] == root and started["depth"] == 1
    assert completed["run_id"] == started["run_id"] and completed["stop_reason"] == "completed"
    child_events = [r for r in events if r["run_id"] == started["run_id"]]
    assert {r["event"] for r in child_events} >= {"run.started", "model.request", "run.stopped"}
    assert (
        events[-1]["run_id"] == root
        and events[-1]["total_model_calls"] == result.total_model_calls == 3
    )
