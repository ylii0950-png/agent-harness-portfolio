"""Version two repair acceptance: tool provenance, order and exact ASK operations."""

import hashlib
from pathlib import Path
from typing import Any

from agent_harness_lab.integration_grading import grade_integration
from agent_harness_lab.integration_proofs import arguments, paired_tools, successful_read
from agent_harness_lab.invocation_audit import InvocationEvidence
from agent_harness_lab.runtime import AgentResult


def grade_remediation(
    case: dict[str, Any],
    result: AgentResult,
    root: Path,
    events: list[dict[str, Any]],
    *,
    isolated: bool,
    child_read_verified: bool,
    handler_calls: int,
    audits: list[InvocationEvidence] | None = None,
) -> dict[str, bool]:
    base = str(case["id"]).removesuffix("_R")
    expected = dict(case["fixtures"])
    if base in {"A", "B"}:
        expected["receipt.txt"] = "AURORA-26\n" if base == "A" else "NOVA-52\n"
    else:
        expected["approved.txt"] = "approved\n"
    checks = grade_integration(
        base,
        result,
        root,
        events,
        isolated=isolated,
        child_read_verified=child_read_verified,
        expected_files=expected,
        audits=audits,
    )
    if base != "D":
        return checks
    del checks["no_authorized_write"]  # Only NEW D_R permits the declared ASK artifact.

    def asked(path: str, content: str, approved: bool) -> bool:
        return any(
            call.name == "request_write"
            and arguments(call) == {"path": path, "content": content}
            and output.ok is approved
            and (
                approved or (output.error is not None and output.error.code == "permission_denied")
            )
            and any(
                row.get("event") == "tool.call"
                and row.get("depth") == 0
                and row.get("tool_name") == "request_write"
                and row.get("call_ref") == hashlib.sha256(call.call_id.encode()).hexdigest()[:16]
                and row.get("permission_decision") == "ask"
                and row.get("authorized") is approved
                for row in events
            )
            for call, output in paired_tools(result.messages)
        )

    checks.update(
        {
            "ask_approved_exact": asked("approved.txt", "approved\n", True),
            "ask_denied_exact": asked("ask-denied.txt", "denied\n", False),
            "approved_readback": successful_read(result.messages, "approved.txt", "approved\n"),
            "only_approved_handler_executed": handler_calls == 1,
            "static_write_still_denied": not any(
                row.get("event") == "tool.call"
                and row.get("tool_name") == "write_file"
                and row.get("authorized") is True
                for row in events
            ),
        }
    )
    return checks
