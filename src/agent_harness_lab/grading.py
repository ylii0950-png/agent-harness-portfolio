"""Independent deterministic grader: inspect answers, actual calls, and artifacts."""

import json
from dataclasses import dataclass
from pathlib import Path

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.runtime import AgentResult, StopReason
from agent_harness_lab.workspace import FileToolError, WorkspaceFiles


@dataclass(frozen=True, slots=True)
class Expected:
    answer: str
    tool: str
    error: str | None = None
    file: str | None = None
    file_content: str | None = None
    call_path: str | None = None


def grade(result: AgentResult, workspace: Path, expected: Expected) -> tuple[bool, str]:
    if result.stop_reason != StopReason.COMPLETED:
        return False, "agent_not_completed"
    if result.final_answer != expected.answer:
        return False, "answer_mismatch"
    calls = [item for item in result.messages if isinstance(item, ToolCall)]
    results = result.tool_results
    if not any(call.name == expected.tool for call in calls):
        return False, "required_tool_not_called"
    if len(calls) != len(results) or any(
        call.call_id != item.call_id for call, item in zip(calls, results, strict=True)
    ):
        return False, "unpaired_tool_results"
    if expected.call_path is not None:
        try:
            if len(calls) != 1 or json.loads(calls[0].arguments).get("path") != expected.call_path:
                return False, "tool_path_mismatch"
        except (ValueError, AttributeError):
            return False, "tool_path_mismatch"
    if expected.error:
        if len(results) != 1 or results[0].error is None or results[0].error.code != expected.error:
            return False, "expected_denial_missing"
    elif any(not item.ok for item in results):
        return False, "tool_failed"
    if expected.file is not None:
        try:
            with WorkspaceFiles(workspace) as files:
                actual = files.read_text(expected.file)
        except FileToolError:
            return False, "artifact_missing_or_unsafe"
        if actual != expected.file_content:
            return False, "artifact_mismatch"
    return True, "accepted"
