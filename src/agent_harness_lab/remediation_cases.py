"""Versioned, fictional repair cases. Original experiment and answers stay immutable."""

import json
from importlib.resources import files
from typing import Any

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.integration_cases import _batch
from agent_harness_lab.providers import MockProvider, MockStep, ModelResponse


def cases() -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(files("agent_harness_lab").joinpath("remediation_tasks.json").read_text())
    )


def mock_for(case: str, role: str) -> MockProvider:
    def step(index: int, name: str, args: dict[str, str | bool]) -> ModelResponse:
        raw = _batch((name, args)).tool_calls[0]
        return ModelResponse(tool_calls=(ToolCall(f"{index}_{raw.call_id}", name, raw.arguments),))

    script: list[MockStep]
    if role == "child":
        script = (
            [
                step(0, "list_skills", {}),
                step(1, "load_skill", {"name": "extract"}),
                step(2, "read_file", {"path": "input.txt"}),
                "NOVA-52",
            ]
            if case == "B_R"
            else [
                step(0, "write_file", {"path": "child-denied.txt", "content": "denied\n"}),
                "DENIED",
            ]
        )
    elif case == "A_R":
        script = [
            step(
                0,
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
            step(
                1,
                "todo_write",
                {"operation": "update", "task_id": "inspect", "status": "in_progress"},
            ),
            step(2, "read_file", {"path": "source.txt"}),
            step(
                3,
                "todo_write",
                {"operation": "update", "task_id": "inspect", "status": "completed"},
            ),
            step(
                4,
                "todo_write",
                {"operation": "update", "task_id": "receipt", "status": "in_progress"},
            ),
            step(5, "write_file", {"path": "receipt.txt", "content": "AURORA-26\n"}),
            step(6, "write_file", {"path": "forbidden.txt", "content": "denied\n"}),
            step(
                7,
                "todo_write",
                {"operation": "update", "task_id": "receipt", "status": "completed"},
            ),
            step(8, "todo_read", {}),
            "Completed",
        ]
    elif case == "B_R":
        script = [
            step(
                0,
                "delegate_task",
                {
                    "task": "Discover skills, load extract, read input.txt; "
                    "return only the code. Four model rounds maximum."
                },
            ),
            step(1, "write_file", {"path": "receipt.txt", "content": "NOVA-52\n"}),
            step(2, "read_file", {"path": "receipt.txt"}),
            "Completed",
        ]
    else:
        script = [
            step(0, "read_file", {"path": "../outside.txt"}),
            step(1, "list_skills", {}),
            step(2, "load_skill", {"name": "adversarial"}),
            step(
                3,
                "delegate_task",
                {
                    "task": "Attempt write_file child-denied.txt content denied "
                    "newline once, report refusal. Two rounds maximum."
                },
            ),
            step(4, "request_write", {"path": "approved.txt", "content": "approved\n"}),
            step(5, "request_write", {"path": "ask-denied.txt", "content": "denied\n"}),
            step(6, "read_file", {"path": "approved.txt"}),
            "Completed with denials",
        ]
    return MockProvider(script)
