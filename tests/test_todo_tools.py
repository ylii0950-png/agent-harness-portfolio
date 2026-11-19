import json
from pathlib import Path
from typing import Any, cast

import pytest

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.permissions import ApprovalRequest, Decision, Permission, PermissionEngine
from agent_harness_lab.todo_tools import register_todo_tools
from agent_harness_lab.todos import TodoStore
from agent_harness_lab.tools import ToolRegistry
from agent_harness_lab.tracing import JSONLTrace


def write(**arguments: str) -> ToolCall:
    return ToolCall("write", "todo_write", json.dumps(arguments))


def create(description: str = "Private description") -> ToolCall:
    return write(
        operation="create", items_json=json.dumps([{"id": "one", "description": description}])
    )


def query(registry: ToolRegistry) -> dict[str, Any]:
    result = registry.invoke(ToolCall("query", "todo_read", "{}"))
    assert result.ok
    return dict(json.loads(result.content))


def configured() -> ToolRegistry:
    registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
    register_todo_tools(registry)
    return registry


def test_tools_create_read_and_update_use_structured_results() -> None:
    registry = configured()
    with registry.run_scope():
        assert query(registry)["tasks"] == []
        assert registry.invoke(create()).ok
        assert query(registry)["counts"] == {"pending": 1, "in_progress": 0, "completed": 0}
        assert registry.invoke(write(operation="update", task_id="one", status="in_progress")).ok
        result = registry.invoke(write(operation="update", task_id="one", status="completed"))
        assert result.ok and json.loads(result.content)["tasks"][0]["status"] == "completed"
        assert query(registry)["counts"]["completed"] == 1


@pytest.mark.parametrize(
    "items",
    [
        "{",
        "[]",
        "null",
        "{}",
        "[1]",
        '[{"id":"one"}]',
        '[{"id":"one","description":"x","status":"completed"}]',
        '[{"id":"one","id":"two","description":"x"}]',
        '[{"id":1,"description":"x"}]',
        '[{"id":"one","description":false}]',
        '[{"id":"one","description":"x"},{"id":"one","description":"y"}]',
        "[" * 1100 + "]" * 1100,
    ],
)
def test_invalid_json_plan_never_creates_or_replaces_state(items: str) -> None:
    registry = configured()
    with registry.run_scope():
        failed = registry.invoke(write(operation="create", items_json=items))
        assert failed.error is not None and failed.error.recoverable
        assert query(registry)["tasks"] == []
        assert registry.invoke(create()).ok


@pytest.mark.parametrize(
    "arguments",
    [
        {"operation": "delete", "task_id": "one"},
        {"operation": "create"},
        {"operation": "update", "task_id": "one"},
        {"operation": "update", "task_id": "one", "status": "done"},
        {"operation": "update", "task_id": "missing", "status": "in_progress"},
        {"operation": "update", "task_id": "one", "status": "completed"},
        {"operation": "update", "task_id": "one", "status": "in_progress", "items_json": "[]"},
    ],
)
def test_wrong_operation_and_update_leave_entire_plan_unchanged(arguments: dict[str, str]) -> None:
    registry = configured()
    with registry.run_scope():
        assert registry.invoke(create()).ok
        before = query(registry)
        rejected = registry.invoke(write(**arguments))
        assert rejected.error is not None and rejected.error.recoverable
        assert query(registry) == before


def test_default_read_allows_query_and_denied_write_cannot_even_construct_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry()
    register_todo_tools(registry)
    with registry.run_scope():

        def fail(self: TodoStore) -> None:
            raise AssertionError("denied handler must not allocate state")

        monkeypatch.setattr(TodoStore, "__init__", fail)
        result = registry.invoke(create())
        assert result.error is not None and result.error.code == "permission_denied"
    monkeypatch.undo()
    with registry.run_scope():
        assert query(registry)["tasks"] == []


@pytest.mark.parametrize("reply", [False, None, "yes", RuntimeError("private approval failure")])
def test_ask_refusal_or_failure_never_changes_previously_created_plan(reply: Any) -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        requests.append(request)
        if len(requests) == 1:
            return True
        if isinstance(reply, Exception):
            raise reply
        return cast(bool, reply)

    registry = ToolRegistry(
        permission_engine=PermissionEngine(
            permissions=(Permission.READ, Permission.WRITE),
            rules={"todo_write": Decision.ASK},
            approval_handler=approve,
        )
    )
    register_todo_tools(registry)
    with registry.run_scope():
        assert registry.invoke(create()).ok
        before = query(registry)
        result = registry.invoke(write(operation="update", task_id="one", status="in_progress"))
        assert result.error is not None and result.error.code == "permission_denied"
        assert "private approval failure" not in result.output_json()
        assert query(registry) == before
    assert len(requests) == 2


def test_tool_state_requires_explicit_scope_and_nested_scopes_restore_outer_plan() -> None:
    registry = configured()
    failed = registry.invoke(create())
    assert failed.error is not None and failed.error.code == "run_scope_required"
    with registry.run_scope():
        assert registry.invoke(create("outer private plan")).ok
        before = query(registry)
        with registry.run_scope():
            assert query(registry)["tasks"] == []
            assert registry.invoke(create("inner private plan")).ok
        assert query(registry) == before
    with registry.run_scope():
        assert query(registry)["tasks"] == []


def test_todo_trace_has_only_counts_hashed_ids_status_and_fixed_errors(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    registry = configured()
    with JSONLTrace(path) as trace, registry.run_scope():
        assert registry.invoke(create("private content secret"), trace=trace).ok
        assert registry.invoke(
            write(operation="update", task_id="one", status="in_progress"), trace=trace
        ).ok
        result = registry.invoke(
            write(operation="update", task_id="private_identity", status="completed"), trace=trace
        )
        assert result.error is not None and result.error.code == "todo_unknown_id"
    text = path.read_text()
    assert "private" not in text and "description" not in text and "items_json" not in text
    events = [json.loads(line) for line in text.splitlines()]
    selected = [row for row in events if row["event"].startswith("todo.")]
    assert [r["event"] for r in selected] == ["todo.created", "todo.updated", "todo.error"]
    assert selected[0]["pending_count"] == 1
    assert (
        selected[1]["previous_status"] == "pending" and selected[1]["todo_status"] == "in_progress"
    )
    assert selected[1]["todo_ref"] != "one"
    assert selected[-1]["error_code"] == "todo_unknown_id"
    assert registry.invocation_trace is None


class FailUpdateTrace:
    def record(self, event: str, **fields: object) -> None:
        if event == "todo.updated":
            raise OSError("private logging failure")


def test_failed_change_logging_returns_error_and_keeps_old_state() -> None:
    registry = configured()
    with registry.run_scope():
        assert registry.invoke(create()).ok
        before = query(registry)
        rejected = registry.invoke(
            write(operation="update", task_id="one", status="in_progress"), trace=FailUpdateTrace()
        )
        assert rejected.error is not None and rejected.error.code == "execution_error"
        assert "private logging failure" not in rejected.output_json()
        assert query(registry) == before
        assert registry.invocation_trace is None
