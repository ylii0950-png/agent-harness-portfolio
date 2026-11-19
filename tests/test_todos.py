from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from agent_harness_lab.todos import TodoError, TodoErrorCode, TodoStatus, TodoStore


def test_create_query_and_explicit_forward_lifecycle_preserve_snapshots() -> None:
    store = TodoStore()
    assert store.snapshot() == ()
    initial = store.create([("read", "Read source"), ("write", "Write receipt")])
    assert [task.task_id for task in initial] == ["read", "write"]
    assert all(task.status == TodoStatus.PENDING for task in initial)
    active = store.update("read", TodoStatus.IN_PROGRESS)
    assert active[0].status == TodoStatus.IN_PROGRESS and initial[0].status == TodoStatus.PENDING
    store.update("read", TodoStatus.COMPLETED)
    store.update("write", TodoStatus.IN_PROGRESS)
    final = store.update("write", TodoStatus.COMPLETED)
    assert all(task.status == TodoStatus.COMPLETED for task in final)
    assert all(task.status == TodoStatus.PENDING for task in initial)
    assert "Read source" not in repr(initial)
    with pytest.raises(FrozenInstanceError):
        cast(Any, final[0]).status = TodoStatus.PENDING


@pytest.mark.parametrize(
    "drafts",
    [
        None,
        "text",
        {},
        [],
        [("", "x")],
        [("Bad", "x")],
        [("a", "")],
        [("a", " \n")],
        [("a", 3)],
        [(True, "x")],
        [["a", "x"]],
        [("a",)],
        [("a", "x", "completed")],
        [("a" * 65, "x")],
        [("a", "x" * 4097)],
        [("a", "\ud800")],
    ],
)
def test_invalid_creation_is_atomic_and_does_not_poison_future_creation(drafts: Any) -> None:
    store = TodoStore()
    with pytest.raises(TodoError) as caught:
        store.create(drafts)
    assert caught.value.code == TodoErrorCode.INVALID_PLAN and store.snapshot() == ()
    assert store.create([("valid", "Valid plan")])[0].status == TodoStatus.PENDING


def test_duplicate_ids_and_replacement_are_rejected_without_losing_state() -> None:
    store = TodoStore()
    with pytest.raises(TodoError, match="duplicate_id"):
        store.create([("one", "First"), ("one", "Second")])
    assert store.snapshot() == ()
    initial = store.create([("one", "Original")])
    with pytest.raises(TodoError, match="plan_exists"):
        store.create([("other", "Must not replace")])
    assert store.snapshot() is initial


def test_plan_and_description_bounds_include_actual_utf8_bytes() -> None:
    store = TodoStore()
    assert len(store.create([(f"task_{i}", "x") for i in range(20)])) == 20
    with pytest.raises(TodoError, match="invalid_plan"):
        TodoStore().create([(f"task_{i}", "x") for i in range(21)])
    assert TodoStore().create([("unicode", "é" * 2048)])[0].description == "é" * 2048
    with pytest.raises(TodoError, match="invalid_plan"):
        TodoStore().create([("unicode", "é" * 2049)])


@pytest.mark.parametrize("status", ["done", "completed", None, True, 1, {}])
def test_invalid_status_never_mutates_any_item(status: Any) -> None:
    store = TodoStore()
    previous = store.create([("one", "Original")])
    with pytest.raises(TodoError, match="invalid_status"):
        store.update("one", status)
    assert store.snapshot() is previous


@pytest.mark.parametrize("identity", ["missing", "", "one ", None, True])
def test_unknown_task_id_is_rejected_without_mutation(identity: Any) -> None:
    store = TodoStore()
    previous = store.create([("one", "Original")])
    with pytest.raises(TodoError, match="unknown_id"):
        store.update(identity, TodoStatus.IN_PROGRESS)
    assert store.snapshot() is previous


@pytest.mark.parametrize(
    "old,new",
    [
        (TodoStatus.PENDING, TodoStatus.PENDING),
        (TodoStatus.PENDING, TodoStatus.COMPLETED),
        (TodoStatus.IN_PROGRESS, TodoStatus.PENDING),
        (TodoStatus.IN_PROGRESS, TodoStatus.IN_PROGRESS),
        (TodoStatus.COMPLETED, TodoStatus.PENDING),
        (TodoStatus.COMPLETED, TodoStatus.IN_PROGRESS),
        (TodoStatus.COMPLETED, TodoStatus.COMPLETED),
    ],
)
def test_illegal_transition_leaves_previous_state_unchanged(
    old: TodoStatus, new: TodoStatus
) -> None:
    store = TodoStore()
    store.create([("one", "Original")])
    if old != TodoStatus.PENDING:
        store.update("one", TodoStatus.IN_PROGRESS)
    if old == TodoStatus.COMPLETED:
        store.update("one", TodoStatus.COMPLETED)
    previous = store.snapshot()
    with pytest.raises(TodoError, match="invalid_transition"):
        store.update("one", new)
    assert store.snapshot() is previous


def test_only_one_active_task_and_failed_update_does_not_reorder_or_replace_tasks() -> None:
    store = TodoStore()
    store.create([("one", "First"), ("two", "Second")])
    previous = store.update("one", TodoStatus.IN_PROGRESS)
    with pytest.raises(TodoError, match="active_task_exists"):
        store.update("two", TodoStatus.IN_PROGRESS)
    assert store.snapshot() is previous
    store.update("one", TodoStatus.COMPLETED)
    assert store.update("two", TodoStatus.IN_PROGRESS)[1].status == TodoStatus.IN_PROGRESS


class BrokenTrace:
    def record(self, event: str, **fields: object) -> None:
        raise OSError("trace unavailable")


def test_trace_failure_before_commit_preserves_create_and_update_state() -> None:
    store = TodoStore()
    with pytest.raises(OSError):
        store.create([("one", "Private description")], trace=BrokenTrace())
    assert store.snapshot() == ()
    previous = store.create([("one", "Private description")])
    with pytest.raises(OSError):
        store.update("one", TodoStatus.IN_PROGRESS, trace=BrokenTrace())
    assert store.snapshot() is previous
