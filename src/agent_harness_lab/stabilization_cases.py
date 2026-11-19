"""One versioned A_S task, fixed original data/answer, no parent/child delegation."""

import json
from importlib.resources import files
from typing import Any

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.providers import MockProvider, MockStep, ModelResponse
from agent_harness_lab.remediation_cases import mock_for as repair_mock


def cases() -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(files("agent_harness_lab").joinpath("stabilization_tasks.json").read_text())
    )


def mock_for(case: str, role: str) -> MockProvider:
    if case != "A_S" or role != "parent":
        raise ValueError("the fixed A_S suite has no child")
    source = repair_mock("A_R", "parent")
    from agent_harness_lab.providers import Message

    script: list[MockStep] = [source.complete((Message("user", "script"),)) for _ in range(10)]
    script.insert(
        6, ModelResponse(tool_calls=(ToolCall("readback", "read_file", '{"path":"receipt.txt"}'),))
    )
    return MockProvider(script)
