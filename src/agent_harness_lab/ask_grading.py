"""Exact A authorization provenance. Inputs are trusted in-memory execution evidence."""

import hashlib
from collections.abc import Sequence
from typing import Any

from agent_harness_lab.contracts import ToolCall, ToolResult
from agent_harness_lab.integration_proofs import arguments, paired_tools
from agent_harness_lab.invocation_audit import InvocationEvidence
from agent_harness_lab.runtime import AgentResult


def exact_a_checks(
    result: AgentResult,
    events: list[dict[str, Any]],
    evidence: Sequence[InvocationEvidence] | None,
    content: str,
) -> dict[str, bool]:
    pairs = paired_tools(result.messages)
    tools = [m for m in result.messages if isinstance(m, (ToolCall, ToolResult))]
    proofs = list(evidence or ())
    calls = [r for r in events if r.get("event") == "tool.call" and r.get("depth") == 0]
    outputs = [r for r in events if r.get("event") == "tool.result" and r.get("depth") == 0]
    starts = [r for r in events if r.get("event") == "run.started" and r.get("depth") == 0]
    seq = [r.get("seq") for r in events]
    integrity = (
        bool(pairs)
        and len(tools) == 2 * len(pairs)
        and len(pairs) == len(proofs) == len(calls) == len(outputs)
        and len(starts) == 1
        and all(type(n) is int for n in seq)
        and all(
            type(a) is int and type(b) is int and a < b for a, b in zip(seq, seq[1:], strict=False)
        )
        and len({c.call_id for c, _ in pairs}) == len(pairs)
    )
    if not integrity:
        return {
            k: False
            for k in ("a_trace_integrity", "ask_approved", "ask_denied", "denied_no_side_effects")
        }
    approved = denied = False
    denied_safe = True
    previous = 0
    for index, ((call, output), proof, event, outcome) in enumerate(
        zip(pairs, proofs, calls, outputs, strict=False), 1
    ):
        ref = hashlib.sha256(call.call_id.encode()).hexdigest()[:16]
        integrity = integrity and (
            proof.call == call
            and proof.result == output
            and proof.run_id == starts[0].get("run_id")
            and all(
                row.get("run_id") == proof.run_id
                and row.get("call_ref") == ref
                and row.get("tool_name") == call.name
                and row.get("round") == proof.round_number
                and row.get("tool_attempt") == proof.attempt == index
                for row in (event, outcome)
            )
            and previous < event["seq"] < outcome["seq"]
            and outcome.get("success") is output.ok
            and outcome.get("error_code") == (output.error.code if output.error else None)
        )
        previous = outcome["seq"]
        if not event.get("authorized"):
            denied_safe = (
                denied_safe
                and proof.handler_calls == 0
                and proof.files_unchanged
                and proof.outside_unchanged
            )
        else:
            integrity = integrity and proof.handler_calls == 1
        if call.name != "write_file":
            continue
        args = arguments(call)
        create = (
            set(args) <= {"path", "content", "overwrite"} and args.get("overwrite", False) is False
        )
        ask = event.get("permission_decision") == "ask"
        if args.get("path") == "receipt.txt" and args.get("content") == content and create:
            approved = approved or (
                ask
                and event.get("authorized") is True
                and event.get("permission_reason") == "approved"
                and output.ok
                and proof.handler_calls == 1
            )
        if args.get("path") == "forbidden.txt" and args.get("content") == "denied\n" and create:
            denied = denied or (
                ask
                and event.get("authorized") is False
                and event.get("permission_reason") == "approval_denied"
                and not output.ok
                and output.error is not None
                and output.error.code == "permission_denied"
                and proof.handler_calls == 0
                and proof.files_unchanged
                and proof.outside_unchanged
            )
    return {
        "a_trace_integrity": bool(integrity),
        "ask_approved": bool(integrity and approved),
        "ask_denied": bool(integrity and denied),
        "denied_no_side_effects": bool(integrity and denied_safe),
    }
