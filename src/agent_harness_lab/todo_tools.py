"""TodoWrite / query adapters; every model operation enters the existing Registry."""

import json
from collections.abc import Mapping
from typing import Any, cast

from agent_harness_lab.contracts import ParameterType, ToolDefinition, ToolParameter
from agent_harness_lab.permissions import Argument, Permission
from agent_harness_lab.todos import MAX_TASKS, TodoError, TodoErrorCode, TodoStatus, TodoStore
from agent_harness_lab.tools import ToolExecutionError, ToolRegistry
from agent_harness_lab.tracing import emit


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("duplicate key")
        value[name] = item
    return value


def _drafts(payload: str) -> tuple[tuple[str, str], ...]:
    try:
        data = json.loads(payload, object_pairs_hook=_object)
        if type(data) is not list or not 1 <= len(data) <= MAX_TASKS:
            raise ValueError("expected bounded array")
        drafts: list[tuple[str, str]] = []
        for item in data:
            if (
                type(item) is not dict
                or item.keys() != {"id", "description"}
                or type(item["id"]) is not str
                or type(item["description"]) is not str
            ):
                raise ValueError("invalid item")
            drafts.append((item["id"], item["description"]))
        return tuple(drafts)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise TodoError(TodoErrorCode.INVALID_PLAN) from None


def register_todo_tools(registry: ToolRegistry) -> None:
    """No process-global plan: lazily allocate inside this Agent.run's scope only."""
    key = object()

    def write(arguments: Mapping[str, Argument]) -> str:
        operation = arguments["operation"]
        trace = registry.invocation_trace
        try:
            store = registry.run_resource(key, TodoStore)
            if operation == "create":
                if arguments.keys() != {"operation", "items_json"}:
                    raise TodoError(TodoErrorCode.INVALID_ARGUMENTS)
                store.create(_drafts(cast(str, arguments["items_json"])), trace=trace)
            elif operation == "update":
                if arguments.keys() != {"operation", "task_id", "status"}:
                    raise TodoError(TodoErrorCode.INVALID_ARGUMENTS)
                try:
                    status = TodoStatus(cast(str, arguments["status"]))
                except ValueError:
                    raise TodoError(TodoErrorCode.INVALID_STATUS) from None
                store.update(cast(str, arguments["task_id"]), status, trace=trace)
            else:
                raise TodoError(TodoErrorCode.INVALID_OPERATION)
            return store.output_json()
        except TodoError as exc:
            emit(
                trace,
                "todo.error",
                {
                    "operation": operation if operation in ("create", "update") else "unknown",
                    "error_code": "todo_" + exc.code.value,
                },
            )
            raise ToolExecutionError("todo_" + exc.code.value, "todo operation rejected") from None

    def read(arguments: Mapping[str, Argument]) -> str:
        store = registry.run_resource(key, TodoStore)
        emit(registry.invocation_trace, "todo.queried", {"task_count": len(store.snapshot())})
        return store.output_json()

    registry.register(
        ToolDefinition(
            "todo_write",
            "Create or update this run's plan. operation must be exactly 'create' or 'update'. "
            "Create: supply only operation='create' and items_json, a JSON array of objects "
            "with unique string id and description; new tasks start pending. "
            "Update: supply only operation='update', task_id (an existing ID), and status "
            "('in_progress' or 'completed'). Transitions: pending -> in_progress -> completed; "
            "at most one task in_progress. Check actual tool success before marking completion. "
            "Do not supply unused fields or auto-correct rejected operations.",
            (
                ToolParameter("operation", ParameterType.STRING),
                ToolParameter("items_json", ParameterType.STRING, required=False),
                ToolParameter("task_id", ParameterType.STRING, required=False),
                ToolParameter("status", ParameterType.STRING, required=False),
            ),
        ),
        write,
        permission=Permission.WRITE,
    )
    registry.register(
        ToolDefinition("todo_read", "Read this run's plan and status counts."),
        read,
        permission=Permission.READ,
    )
