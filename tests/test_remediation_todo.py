import json

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.permissions import Permission
from agent_harness_lab.providers import MockProvider
from agent_harness_lab.runtime import Agent
from agent_harness_lab.todo_tools import register_todo_tools
from agent_harness_lab.tools import ToolRegistry


def test_model_catalog_describes_exact_operations_and_parameter_combinations() -> None:
    registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
    register_todo_tools(registry)
    provider = MockProvider(["done"])
    Agent(provider, registry).run("Inspect tool instructions")
    definition = next(t for t in provider.tool_catalogs[0] if t.name == "todo_write")
    for text in (
        "operation='create'",
        "items_json",
        "operation='update'",
        "task_id",
        "status",
        "pending -> in_progress -> completed",
        "at most one",
        "only",
    ):
        assert text in definition.description
    assert "AURORA" not in definition.description and "NOVA" not in definition.description


def test_invalid_operation_does_not_change_or_auto_correct_plan() -> None:
    registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
    register_todo_tools(registry)
    with registry.run_scope():
        registry.invoke(
            ToolCall(
                "c",
                "todo_write",
                json.dumps(
                    {
                        "operation": "create",
                        "items_json": json.dumps([{"id": "step", "description": "Synthetic task"}]),
                    }
                ),
            )
        )
        before = registry.invoke(ToolCall("r1", "todo_read", "{}"))
        error = registry.invoke(
            ToolCall(
                "bad",
                "todo_write",
                json.dumps({"operation": "set_status", "task_id": "step", "status": "completed"}),
            )
        )
        after = registry.invoke(ToolCall("r2", "todo_read", "{}"))
        assert error.error is not None and error.error.code == "todo_invalid_operation"
        assert before.content == after.content
