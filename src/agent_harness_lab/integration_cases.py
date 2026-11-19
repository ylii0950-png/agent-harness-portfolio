"""Fixed synthetic fixtures and deterministic decisions, not a replacement Agent."""

import json
from importlib.resources import files
from typing import Any

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.providers import MockProvider, MockStep, ModelResponse


def cases() -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(files("agent_harness_lab").joinpath("integration_tasks.json").read_text())
    )


def _batch(*items: tuple[str, dict[str, str | bool]]) -> ModelResponse:
    return ModelResponse(
        tool_calls=tuple(
            ToolCall(f"fixture_{i}_{name}", name, json.dumps(args))
            for i, (name, args) in enumerate(items)
        )
    )


def mock_for(case: str, role: str) -> MockProvider:
    # Unique IDs across a script; runtime executes the actual handlers and decisions.
    def batch(index: int, *items: tuple[str, dict[str, str | bool]]) -> ModelResponse:
        raw = _batch(*items)
        return ModelResponse(
            tool_calls=tuple(
                ToolCall(f"{index}_{c.call_id}", c.name, c.arguments) for c in raw.tool_calls
            )
        )

    script: list[MockStep]
    if role == "child":
        script = (
            [
                batch(0, ("list_skills", {})),
                batch(1, ("load_skill", {"name": "extract"}), ("read_file", {"path": "input.txt"})),
                "NOVA-52",
            ]
            if case == "B"
            else [
                batch(0, ("write_file", {"path": "child-denied.txt", "content": "denied\n"})),
                "DENIED",
            ]
        )
    elif case == "A":
        script = [
            batch(
                0,
                (
                    "todo_write",
                    {
                        "operation": "create",
                        "items_json": json.dumps(
                            [
                                {"id": "inspect", "description": "Read source"},
                                {"id": "receipt", "description": "Save receipt"},
                            ]
                        ),
                    },
                ),
                (
                    "todo_write",
                    {"operation": "update", "task_id": "inspect", "status": "in_progress"},
                ),
                ("read_file", {"path": "source.txt"}),
            ),
            batch(
                1,
                (
                    "todo_write",
                    {"operation": "update", "task_id": "inspect", "status": "completed"},
                ),
                (
                    "todo_write",
                    {"operation": "update", "task_id": "receipt", "status": "in_progress"},
                ),
                ("write_file", {"path": "receipt.txt", "content": "AURORA-26\n"}),
            ),
            batch(
                2,
                ("write_file", {"path": "forbidden.txt", "content": "denied\n"}),
                (
                    "todo_write",
                    {"operation": "update", "task_id": "receipt", "status": "completed"},
                ),
                ("todo_read", {}),
            ),
            "Completed",
        ]
    elif case == "B":
        script = [
            batch(
                0,
                (
                    "delegate_task",
                    {"task": "Discover skills, load extract, read input.txt; return only code."},
                ),
            ),
            batch(
                1,
                ("write_file", {"path": "receipt.txt", "content": "NOVA-52\n"}),
                ("read_file", {"path": "receipt.txt"}),
            ),
            "Completed",
        ]
    elif case in {"C_off", "C_on"}:
        script = [
            batch(0, ("read_file", {"path": "first.txt"})),
            batch(1, ("read_file", {"path": "second.txt"})),
            batch(2, ("write_file", {"path": "receipt.txt", "content": "ORBIT-41|LUMEN-73\n"})),
            "Completed",
        ]
    else:
        script = [
            batch(0, ("read_file", {"path": "../outside.txt"}), ("list_skills", {})),
            batch(
                1,
                ("load_skill", {"name": "adversarial"}),
                (
                    "delegate_task",
                    {
                        "task": "Attempt write_file child-denied.txt content denied newline once; "
                        "report refusal."
                    },
                ),
            ),
            batch(2, ("request_write", {"path": "ask-denied.txt", "content": "denied\n"})),
            "DENIED",
        ]
    return MockProvider(script)
