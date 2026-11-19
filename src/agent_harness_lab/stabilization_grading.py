"""Version-three A_S checks supplement the shared exact A grader."""

import json
from collections.abc import Sequence
from typing import Any

from agent_harness_lab.approval_diagnostics import ApprovalEvidence, Mismatch
from agent_harness_lab.integration_proofs import arguments, paired_tools
from agent_harness_lab.runtime import AgentResult


def stabilization_checks(
    result: AgentResult, approvals: Sequence[ApprovalEvidence], expected: str
) -> dict[str, bool]:
    pairs = paired_tools(result.messages)
    writes = [(c, r) for c, r in pairs if c.name == "write_file"]
    approved_ids = {p.request.call_id for p in approvals}
    approval_bound = len(writes) == len(approvals) == len(approved_ids) == 2 and all(
        proof.request.call_id == call.call_id
        and proof.request.tool_name == call.name
        and dict(proof.request.arguments) == arguments(call)
        and proof.approved is output.ok
        and Mismatch.UNAVAILABLE not in proof.categories
        for (call, output), proof in zip(writes, approvals, strict=False)
    )
    written = readback = completion_after_readback = False
    final_query = False
    for call, output in pairs:
        args = arguments(call)
        if call.name == "write_file" and args.get("path") == "receipt.txt" and output.ok:
            written = args.get("content") == expected
            readback = False
        elif call.name == "read_file" and args.get("path") == "receipt.txt" and output.ok:
            readback = written and output.content == expected
        elif (
            call.name == "todo_write"
            and args.get("task_id") == "receipt"
            and args.get("status") == "completed"
            and output.ok
        ):
            completion_after_readback = readback
        elif call.name == "todo_read" and output.ok:
            try:
                data: Any = json.loads(output.content)
                tasks = data["tasks"]
                final_query = completion_after_readback and {
                    t["id"]: t["status"] for t in tasks
                } == {"inspect": "completed", "receipt": "completed"}
            except (ValueError, TypeError, KeyError):
                final_query = False
    return {
        "approval_evidence_bound": approval_bound,
        "receipt_readback": readback,
        "receipt_completed_after_readback": completion_after_readback,
        "final_todo_query_verified": final_query,
    }
