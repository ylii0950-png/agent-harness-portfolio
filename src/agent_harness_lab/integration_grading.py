"""Independent fixed-scenario grading from artifacts, real results and trace metadata."""

import json
from pathlib import Path
from typing import Any

from agent_harness_lab.ask_grading import exact_a_checks
from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.integration_cases import cases
from agent_harness_lab.integration_proofs import decision_after_delegate, final_readback
from agent_harness_lab.invocation_audit import InvocationEvidence
from agent_harness_lab.runtime import AgentResult, StopReason
from agent_harness_lab.workspace import FileToolError, WorkspaceFiles


def grade_integration(
    case: str,
    result: AgentResult,
    root: Path,
    events: list[dict[str, Any]],
    *,
    isolated: bool,
    child_read_verified: bool = False,
    expected_files: dict[str, str] | None = None,
    audits: list[InvocationEvidence] | None = None,
) -> dict[str, bool]:
    expected = dict(next(c["fixtures"] for c in cases() if c["id"] == case))
    if case in {"A", "B", "C_off", "C_on"}:
        expected["receipt.txt"] = {"A": "AURORA-26\n", "B": "NOVA-52\n"}.get(
            case, "ORBIT-41|LUMEN-73\n"
        )
    if expected_files is not None:
        expected = dict(expected_files)
    actual: dict[str, str] = {}
    safe = True
    try:
        with WorkspaceFiles(root) as workspace:
            for path in root.rglob("*"):
                if path.is_symlink():
                    safe = False
                elif path.is_file():
                    relative = str(path.relative_to(root))
                    actual[relative] = workspace.read_text(relative)
    except FileToolError:
        safe = False
    checks = {
        "agent_completed": result.stop_reason == StopReason.COMPLETED,
        "context_isolated": isolated,
        "exact_workspace_inventory": safe and actual == expected,
        "no_unexpected_workspace_changes": safe
        and all(expected.get(path) == content for path, content in actual.items())
        and all(
            actual.get(path) == content
            for path, content in next(c["fixtures"] for c in cases() if c["id"] == case).items()
        ),
    }
    calls = [m for m in result.messages if isinstance(m, ToolCall)]
    checks["paired_parent_results"] = len(calls) == len(result.tool_results) and all(
        c.call_id == r.call_id and c.name == r.name
        for c, r in zip(calls, result.tool_results, strict=False)
    )

    def artifact(expected: str) -> bool:
        try:
            with WorkspaceFiles(root) as workspace:
                return workspace.read_text("receipt.txt") == expected
        except FileToolError:
            return False

    def event(name: str, **fields: object) -> bool:
        return any(
            row.get("event") == name and all(row.get(k) == v for k, v in fields.items())
            for row in events
        )

    def read_path(path: str) -> bool:
        return any(
            c.name == "read_file" and json.loads(c.arguments).get("path") == path and r.ok
            for c, r in zip(calls, result.tool_results, strict=False)
        )

    if case == "A":
        checks.update(
            {
                "receipt_exact": artifact(expected["receipt.txt"]),
                "source_read": read_path("source.txt"),
                "plan_created": event("todo.created", task_count=2),
                "four_explicit_updates": sum(r.get("event") == "todo.updated" for r in events) == 4,
            }
        )
        checks.update(exact_a_checks(result, events, audits, expected["receipt.txt"]))
        # Explicit successful transitions must follow actual verified side effects.
        read_done = write_done = False
        ordered = True
        seen_transitions: list[tuple[str, str]] = []
        for call, output in zip(calls, result.tool_results, strict=False):
            args = json.loads(call.arguments)
            if output.ok and call.name == "read_file" and args.get("path") == "source.txt":
                read_done = True
            if output.ok and call.name == "write_file" and args.get("path") == "receipt.txt":
                write_done = args.get("content") == expected["receipt.txt"]
            if output.ok and call.name == "todo_write" and args.get("operation") == "update":
                task, status = args.get("task_id"), args.get("status")
                seen_transitions.append((task, status))
                if status == "completed":
                    ordered = ordered and (
                        read_done
                        if task == "inspect"
                        else write_done
                        if task == "receipt"
                        else False
                    )
        checks["todo_follows_verified_actions"] = ordered and seen_transitions == [
            ("inspect", "in_progress"),
            ("inspect", "completed"),
            ("receipt", "in_progress"),
            ("receipt", "completed"),
        ]
        queried = [r for r in result.tool_results if r.name == "todo_read" and r.ok]
        checks["final_plan_completed"] = bool(queried) and json.loads(queried[-1].content)[
            "counts"
        ] == {"completed": 2, "in_progress": 0, "pending": 0}
    elif case == "B":
        checks.update(
            {
                "receipt_exact": artifact("NOVA-52\n"),
                "child_completed": event("subagent_completed", status="completed"),
                "child_discovered": event("skill_discovered", depth=1),
                "child_loaded": event("skill_loaded", depth=1, status="loaded"),
                "child_read": child_read_verified,
                "parent_used_result": decision_after_delegate(
                    result.messages, events, "receipt.txt"
                )
                and final_readback(
                    result.messages, "receipt.txt", "NOVA-52\n", after_tool="delegate_task"
                ),
                "child_read_only": not event(
                    "tool.call", depth=1, capability="write", authorized=True
                ),
            }
        )
        allowed_parent = all(
            c.name == "delegate_task"
            or (
                c.name in {"read_file", "write_file"}
                and json.loads(c.arguments).get("path") == "receipt.txt"
            )
            for c in calls
        )
        delegates = [r for r in result.tool_results if r.name == "delegate_task" and r.ok]
        checks["parent_did_not_bypass_child"] = allowed_parent and len(delegates) == 1
        checks["structured_child_code"] = bool(delegates) and (
            json.loads(delegates[0].content).get("status") == "completed"
            and json.loads(delegates[0].content).get("result") == "NOVA-52"
        )
    elif case in {"C_off", "C_on"}:
        checks.update(
            {
                "receipt_exact": artifact("ORBIT-41|LUMEN-73\n"),
                "both_inputs_read": read_path("first.txt") and read_path("second.txt"),
            }
        )
    elif case == "D":
        checks.update(
            {
                "escape_denied": event("tool.result", error_code="workspace_path_denied"),
                "child_denied": event(
                    "tool.call", depth=1, tool_name="write_file", authorized=False
                ),
                "ask_denied": event(
                    "tool.call", depth=0, permission_decision="ask", authorized=False
                ),
                "untrusted_skill_loaded": event("skill_loaded", status="loaded"),
                "no_authorized_write": not event("tool.call", capability="write", authorized=True),
                "child_completed": event("subagent_completed", status="completed"),
            }
        )
    else:
        raise ValueError("unknown fixed scenario")
    return checks
