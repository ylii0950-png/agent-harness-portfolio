"""Atomic, bounded in-memory plans; only explicit forward transitions change state."""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from agent_harness_lab.tracing import TraceSink, emit

MAX_TASKS = 20


class TodoStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TodoErrorCode(StrEnum):
    INVALID_PLAN = "invalid_plan"
    DUPLICATE_ID = "duplicate_id"
    PLAN_EXISTS = "plan_exists"
    UNKNOWN_ID = "unknown_id"
    INVALID_STATUS = "invalid_status"
    INVALID_TRANSITION = "invalid_transition"
    ACTIVE_TASK_EXISTS = "active_task_exists"
    INVALID_OPERATION = "invalid_operation"
    INVALID_ARGUMENTS = "invalid_arguments"


class TodoError(Exception):
    def __init__(self, code: TodoErrorCode) -> None:
        self.code = code
        super().__init__("todo operation rejected: " + code.value)


@dataclass(frozen=True, slots=True, repr=False)
class TodoItem:
    task_id: str
    description: str
    status: TodoStatus = TodoStatus.PENDING

    def __repr__(self) -> str:
        return "TodoItem(<protected>)"


def _counts(items: tuple[TodoItem, ...]) -> dict[str, object]:
    return {
        "task_count": len(items),
        **{
            status.value + "_count": sum(item.status == status for item in items)
            for status in TodoStatus
        },
    }


class TodoStore:
    """One plan, IDs unique within this run, one active task, no replacement/reopen.

    Construct/validate the whole proposal and record metadata before committing.
    Returned tuples/items are immutable; updating never mutates earlier snapshots.
    This class is trusted application state; model access is through gated tools.
    """

    def __init__(self) -> None:
        self._items: tuple[TodoItem, ...] = ()

    def snapshot(self) -> tuple[TodoItem, ...]:
        return self._items

    def create(
        self, drafts: Sequence[tuple[str, str]], *, trace: TraceSink | None = None
    ) -> tuple[TodoItem, ...]:
        if self._items:
            raise TodoError(TodoErrorCode.PLAN_EXISTS)
        if not isinstance(drafts, (list, tuple)) or not 1 <= len(drafts) <= MAX_TASKS:
            raise TodoError(TodoErrorCode.INVALID_PLAN)
        proposed: list[TodoItem] = []
        seen: set[str] = set()
        for draft in drafts:
            if type(draft) is not tuple or len(draft) != 2:
                raise TodoError(TodoErrorCode.INVALID_PLAN)
            task_id, description = draft
            if (
                type(task_id) is not str
                or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", task_id)
                or type(description) is not str
                or not description.strip()
            ):
                raise TodoError(TodoErrorCode.INVALID_PLAN)
            try:
                if len(description.encode("utf-8")) > 4096:
                    raise TodoError(TodoErrorCode.INVALID_PLAN)
            except UnicodeError:
                raise TodoError(TodoErrorCode.INVALID_PLAN) from None
            if task_id in seen:
                raise TodoError(TodoErrorCode.DUPLICATE_ID)
            seen.add(task_id)
            proposed.append(TodoItem(task_id, description))
        snapshot = tuple(proposed)
        emit(trace, "todo.created", _counts(snapshot))
        self._items = snapshot
        return snapshot

    def update(
        self, task_id: str, status: TodoStatus, *, trace: TraceSink | None = None
    ) -> tuple[TodoItem, ...]:
        if type(status) is not TodoStatus:
            raise TodoError(TodoErrorCode.INVALID_STATUS)
        if type(task_id) is not str:
            raise TodoError(TodoErrorCode.UNKNOWN_ID)
        index = next((i for i, item in enumerate(self._items) if item.task_id == task_id), None)
        if index is None:
            raise TodoError(TodoErrorCode.UNKNOWN_ID)
        current = self._items[index]
        if (current.status, status) not in {
            (TodoStatus.PENDING, TodoStatus.IN_PROGRESS),
            (TodoStatus.IN_PROGRESS, TodoStatus.COMPLETED),
        }:
            raise TodoError(TodoErrorCode.INVALID_TRANSITION)
        if status == TodoStatus.IN_PROGRESS and any(
            item.status == TodoStatus.IN_PROGRESS for item in self._items
        ):
            raise TodoError(TodoErrorCode.ACTIVE_TASK_EXISTS)
        snapshot = tuple(
            replace(item, status=status) if i == index else item
            for i, item in enumerate(self._items)
        )
        emit(
            trace,
            "todo.updated",
            {
                **_counts(snapshot),
                "todo_ref": hashlib.sha256(task_id.encode()).hexdigest()[:16],
                "previous_status": current.status.value,
                "todo_status": status.value,
            },
        )
        self._items = snapshot
        return snapshot

    def output_json(self) -> str:
        return json.dumps(
            {
                "tasks": [
                    {
                        "id": item.task_id,
                        "description": item.description,
                        "status": item.status.value,
                    }
                    for item in self._items
                ],
                "counts": {
                    status.value: sum(i.status == status for i in self._items)
                    for status in TodoStatus
                },
            },
            sort_keys=True,
        )
