"""Independent grading of actual bounded Mock Agent evidence, never model prose.

No scenario identifier or expected answer selects an outcome. The caller may
compare the observed outcome/state to a separately authored public expectation.
This checks a trusted application's execution receipts, not adversarial Python
code or independent access isolation. No formal scorer/Gold is imported.
"""

from dataclasses import asdict, dataclass
from typing import Protocol

from agent_harness_lab.contracts import ToolCall, ToolResult
from agent_harness_lab.permissions import Permission
from agent_harness_lab.providers import Message, MockProvider
from agent_harness_lab.runtime import AgentResult, StopReason
from knowledge_ticket_agent.agent_tools import (
    SCHEMA,
    DemoApplication,
    ToolObservation,
    call_reference,
    transition_arguments,
    verified_parent_citations,
)
from knowledge_ticket_agent.citations import Citation, cite, verify_citation
from knowledge_ticket_agent.evidence_adapter import Outcome
from knowledge_ticket_agent.strict import FixtureError, decode, parse_json
from knowledge_ticket_agent.ticket_store import TicketState


class RunEvidence(Protocol):
    @property
    def app(self) -> DemoApplication: ...

    @property
    def result(self) -> AgentResult: ...

    @property
    def provider(self) -> MockProvider: ...

    @property
    def children(self) -> tuple[MockProvider, ...]: ...


@dataclass(frozen=True)
class IntegrationGrade:
    outcome: str
    issues: tuple[str, ...]
    parent_model_calls: int
    child_model_calls: int
    parent_tool_calls: int
    child_tool_calls: int
    agent_search_calls: int
    agent_get_calls: int
    agent_full_source_chunks: int
    required_policy_chunks: int
    backend_search_calls: int
    backend_complete: bool
    handler_entries: int
    commits: int
    final_revision: int
    final_status: str
    readback_verified: bool
    todo_tasks: int
    todo_complete: int
    todo_final_read: bool
    skill_loads: int
    compactions: int
    child_valid: bool


def _object(text: str, limit: int = 65536) -> dict[str, object]:
    if type(text) is not str or len(text.encode()) > limit:
        raise ValueError("bounded_json")
    value = parse_json(text)
    if type(value) is not dict:
        raise ValueError("json_object")
    return value


def _events(
    app: DemoApplication, record: ToolObservation, event: str
) -> list[tuple[int, dict[str, object]]]:
    return [
        (n, fields)
        for n, (kind, fields) in enumerate(app.trace.rows)
        if kind == event
        and fields.get("run_id") == record.run_id
        and fields.get("depth") == record.depth
        and fields.get("call_ref") == call_reference(record.call.call_id)
    ]


def _canonical_get(app: DemoApplication, record: ToolObservation) -> Citation:
    raw = _object(record.result.content)
    if set(raw) != {"schema", "citation"} or raw["schema"] != SCHEMA:
        raise ValueError("get_schema")
    citation = decode(Citation, raw["citation"])
    verify_citation(
        citation,
        app.data,
        app.retriever,
        category=app.category,
        snapshot=app.snapshot,
        require_authority=False,
    )
    if citation != record.citation or citation != cite(
        app.retriever, citation.chunk_id, category=app.category, snapshot=app.snapshot
    ):
        raise ValueError("get_canonical_binding")
    arguments = _object(record.call.arguments)
    if arguments != {"chunk_id": citation.chunk_id, "doc_ref": citation.doc_ref}:
        raise ValueError("get_arguments")
    # A prior actual search by the same run must have returned this candidate.
    found = False
    for previous in app.registry.records:
        if (
            previous.ordinal >= record.ordinal
            or previous.run_id != record.run_id
            or previous.depth != record.depth
            or previous.call.name != "kb_search"
            or not previous.result.ok
        ):
            continue
        search = _object(previous.result.content)
        result = search.get("result")
        if search.get("schema") != SCHEMA or type(result) is not dict:
            raise ValueError("search_schema")
        hits = result.get("hits")
        if type(hits) is not list:
            raise ValueError("search_schema")
        found = found or any(
            type(h) is dict and h.get("chunk_id") == citation.chunk_id for h in hits
        )
    if not found:
        raise ValueError("get_without_actual_search")
    return citation


def _todo(records: tuple[ToolObservation, ...]) -> tuple[int, int, bool]:
    previous: dict[str, str] = {}
    created = False
    last_name = ""
    for record in records:
        if record.call.name not in {"todo_write", "todo_read"} or not record.result.ok:
            continue
        raw = _object(record.result.content)
        if set(raw) != {"tasks", "counts"} or type(raw["tasks"]) is not list:
            raise ValueError("todo_output")
        tasks = raw["tasks"]
        current: dict[str, str] = {}
        for item in tasks:
            if (
                type(item) is not dict
                or set(item) != {"id", "description", "status"}
                or type(item["id"]) is not str
                or type(item["description"]) is not str
                or item["status"] not in {"pending", "in_progress", "completed"}
                or item["id"] in current
            ):
                raise ValueError("todo_output")
            current[item["id"]] = item["status"]
        if not 0 <= len(current) <= 20 or sum(s == "in_progress" for s in current.values()) > 1:
            raise ValueError("todo_bounds")
        if raw["counts"] != {
            s: sum(v == s for v in current.values())
            for s in ("pending", "in_progress", "completed")
        }:
            raise ValueError("todo_counts")
        if record.call.name == "todo_read":
            if current != previous:
                raise ValueError("todo_read_state")
        else:
            args = _object(record.call.arguments)
            if args.get("operation") == "create":
                if created or not current or any(v != "pending" for v in current.values()):
                    raise ValueError("todo_creation")
                created = True
            elif args.get("operation") == "update":
                if current.keys() != previous.keys():
                    raise ValueError("todo_identity")
                changes = [(previous[k], v) for k, v in current.items() if previous[k] != v]
                if len(changes) != 1 or changes[0] not in {
                    ("pending", "in_progress"),
                    ("in_progress", "completed"),
                }:
                    raise ValueError("todo_transition")
            else:
                raise ValueError("todo_operation")
        previous, last_name = current, record.call.name
    return (
        len(previous),
        sum(v == "completed" for v in previous.values()),
        last_name == "todo_read",
    )


def _children(app: DemoApplication, records: tuple[ToolObservation, ...]) -> tuple[bool, bool]:
    valid = True
    limited = any(
        event == "subagent_limit_reached"
        or (
            event == "run.stopped"
            and fields.get("depth") != 0
            and fields.get("stop_reason") == "context_error"
        )
        for event, fields in app.trace.rows
    )
    for record in records:
        if record.call.name != "delegate_task":
            continue
        if not record.result.ok:
            valid = False
            continue
        try:
            wrapper = _object(record.result.content, 16384)
            if set(wrapper) != {
                "status",
                "stop_reason",
                "result",
                "model_calls",
                "tool_calls",
                "error_code",
                "truncated",
            }:
                raise ValueError("child_wrapper")
            if wrapper["status"] == "limit_reached" or wrapper["stop_reason"] == "context_error":
                limited = True
                valid = False
                continue
            if (
                wrapper["status"] != "completed"
                or wrapper["stop_reason"] != "completed"
                or wrapper["error_code"] is not None
                or wrapper["truncated"] is not False
                or type(wrapper["model_calls"]) is not int
                or not 1 <= wrapper["model_calls"] <= 4
                or type(wrapper["tool_calls"]) is not int
                or not 0 <= wrapper["tool_calls"] <= 10
                or type(wrapper["result"]) is not str
            ):
                raise ValueError("child_status")
            calls = _events(app, record, "tool.call")
            returns = _events(app, record, "tool.result")
            if len(calls) != 1 or len(returns) != 1:
                raise ValueError("child_dispatch")
            starts = [
                fields
                for kind, fields in app.trace.rows[calls[0][0] + 1 : returns[0][0]]
                if kind == "run.started"
                and fields.get("depth") == 1
                and fields.get("parent_run_id") == record.run_id
            ]
            if len(starts) != 1:
                raise ValueError("child_frame")
            child_id = starts[0]["run_id"]
            child_records = [r for r in app.registry.records if r.parent_ordinal == record.ordinal]
            models = [
                fields
                for kind, fields in app.trace.rows
                if kind == "model.request" and fields.get("run_id") == child_id
            ]
            if (
                wrapper["model_calls"] != len(models)
                or wrapper["tool_calls"] != len(child_records)
                or any(r.depth != 1 or r.run_id != child_id for r in child_records)
            ):
                raise ValueError("child_accounting")
            answer = _object(wrapper["result"], 2048)
            ids = answer.get("chunk_ids")
            if (
                set(answer) != {"schema", "chunk_ids"}
                or answer["schema"] != SCHEMA
                or type(ids) is not list
                or not 1 <= len(ids) <= 8
                or any(type(i) is not str for i in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("child_claims")
            obtained = {
                _canonical_get(app, r).chunk_id
                for r in child_records
                if r.call.name == "kb_get" and r.result.ok
            }
            if not set(ids) <= obtained:
                raise ValueError("child_unsupported_claim")
        except (FixtureError, TypeError, ValueError, AttributeError, KeyError):
            valid = False
    return valid, limited


def grade_run(run: RunEvidence) -> IntegrationGrade:
    """Classify actual execution independently of public scenario IDs/expectations."""
    app, result = run.app, run.result
    issues: list[str] = []
    records = tuple(sorted(app.registry.records, key=lambda r: r.ordinal))
    root_id = app.workflow.run_id
    parent = tuple(r for r in records if r.depth == 0)
    child = tuple(r for r in records if r.depth != 0)
    final_snapshot = app.workflow.store.snapshot
    state = next(s for s in final_snapshot.tickets if s.ticket_id == app.ticket_id)
    initial = app.initial
    commits = (
        final_snapshot.commits[len(initial.commits) :]
        if initial is not None
        else final_snapshot.commits
    )
    if initial is None:
        issues.append("missing_initial_snapshot")
    elif tuple(s for s in initial.tickets if s.ticket_id != app.ticket_id) != tuple(
        s for s in final_snapshot.tickets if s.ticket_id != app.ticket_id
    ):
        issues.append("unrelated_ticket_mutated")
    if len(commits) > 1 or (state.revision == 1) != bool(commits):
        issues.append("state_journal_mismatch")
    if any(type(r.ordinal) is not int or type(r.depth) is not int for r in records):
        issues.append("observation_type")
    if tuple(r.ordinal for r in records) != tuple(range(1, len(records) + 1)):
        issues.append("observation_sequence")
    if any(r.run_id != root_id or r.parent_ordinal is not None for r in parent):
        issues.append("root_frame_mismatch")
    if initial is not None and (
        (parent and (parent[0].before != initial or parent[-1].after != final_snapshot))
        or any(a.after != b.before for a, b in zip(parent, parent[1:], strict=False))
        or (not parent and initial != final_snapshot)
    ):
        issues.append("unobserved_state_change")
    transcript_calls = tuple(m for m in result.messages if type(m) is ToolCall)
    transcript_results = tuple(m for m in result.messages if type(m) is ToolResult)
    if transcript_calls != tuple(r.call for r in parent) or transcript_results != tuple(
        r.result for r in parent
    ):
        issues.append("agent_registry_pair_mismatch")
    model_rows = [(event, fields) for event, fields in app.trace.rows if event == "model.request"]
    root_models = sum(
        fields.get("run_id") == root_id and fields.get("depth") == 0 for _, fields in model_rows
    )
    child_models = len(model_rows) - root_models
    if (
        len(run.provider.calls) != result.model_rounds
        or root_models != result.model_rounds
        or result.tool_calls != len(parent)
        or result.total_model_calls != len(model_rows)
        or result.total_tool_calls != len(records)
        or sum(len(p.calls) for p in run.children) != child_models
        or any(p is run.provider for p in run.children)
        or len({id(p) for p in run.children}) != len(run.children)
    ):
        issues.append("runtime_accounting_mismatch")
    if any(
        p.calls
        and (
            len(p.calls[0]) != 1
            or type(p.calls[0][0]) is not Message
            or p.calls[0][0].role != "user"
        )
        for p in run.children
    ):
        issues.append("child_history_not_fresh")
    starts = [f for e, f in app.trace.rows if e == "run.started" and f.get("depth") == 0]
    stops = [f for e, f in app.trace.rows if e == "run.stopped" and f.get("depth") == 0]
    if (
        len(starts) != 1
        or starts[0].get("run_id") != root_id
        or len(stops) != 1
        or stops[0].get("run_id") != root_id
        or stops[0].get("stop_reason") != result.stop_reason.value
        or stops[0].get("model_rounds") != result.model_rounds
        or stops[0].get("tool_calls") != result.tool_calls
        or stops[0].get("total_model_calls") != result.total_model_calls
        or stops[0].get("total_tool_calls") != result.total_tool_calls
    ):
        issues.append("root_lifecycle_trace_mismatch")
    compactions = [
        f for e, f in app.trace.rows if e == "compaction_completed" and f.get("depth") == 0
    ]
    if [
        {
            k: f.get(k)
            for k in ("estimated_before", "estimated_after", "compacted_results", "summary_chars")
        }
        for f in compactions
    ] != [asdict(stat) for stat in result.context_stats]:
        issues.append("context_stats_trace_mismatch")
    forbidden_child = {
        "ticket_transition",
        "ticket_assess",
        "ticket_read",
        "todo_write",
        "write_file",
    }
    if any(
        forbidden_child & {t.name for t in catalog}
        for p in run.children
        for catalog in p.tool_catalogs
    ):
        issues.append("child_catalog_escalated")
    transitions = tuple(r for r in parent if r.call.name == "ticket_transition")
    for record in records:
        calls, returns = _events(app, record, "tool.call"), _events(app, record, "tool.result")
        if len(calls) != 1 or len(returns) != 1 or calls[0][0] >= returns[0][0]:
            issues.append("trace_pair_mismatch")
            continue
        permission, returned = calls[0][1], returns[0][1]
        if (
            type(permission.get("authorized")) is not bool
            or record.handler_entries not in (0, 1)
            or record.handler_entries != int(permission["authorized"] is True)
            or returned.get("success") is not record.result.ok
            or returned.get("error_code")
            != (record.result.error.code if record.result.error else None)
        ):
            issues.append("permission_handler_result_mismatch")
        if record.depth != 0 and (
            permission.get("capability") == "write" and permission.get("authorized") is True
        ):
            issues.append("child_authorized_write")
        if record.before != record.after and (
            record.depth != 0
            or record.call.name != "ticket_transition"
            or permission.get("authorized") is not True
            or permission.get("capability") != "write"
            or permission.get("permission_decision") != "ask"
            or permission.get("permission_reason") != "approved"
        ):
            issues.append("unattributed_mutation")
        if record.call.name == "ticket_transition" and permission.get("authorized") is True:
            approvals = [a for a in app.approval_calls if a.call_id == record.call.call_id]
            args = _object(record.call.arguments)
            if (
                len(approvals) != 1
                or approvals[0].tool_name != "ticket_transition"
                or approvals[0].permission is not Permission.WRITE
                or approvals[0].arguments.keys() != args.keys()
                or any(
                    type(v) is not type(args[k]) or v != args[k]
                    for k, v in approvals[0].arguments.items()
                )
            ):
                issues.append("fresh_approval_receipt_missing")
            if (
                permission.get("capability") != "write"
                or permission.get("permission_decision") != "ask"
                or permission.get("permission_reason") != "approved"
            ):
                issues.append("write_not_exact_ask")
    if app.workflow.handler_invocations > sum(
        r.handler_entries for r in transitions
    ) or app.workflow.handler_invocations != sum(
        e.kind == "handler" for e in app.workflow.audit.events
    ):
        issues.append("business_handler_audit_mismatch")
    for record in parent:
        if record.call.name in {"load_skill", "list_skills"} and record.result.ok:
            try:
                skill = _object(record.result.content)
                if skill.get("trust") != "low" or type(skill.get("notice")) is not str:
                    raise ValueError("skill_trust")
            except (FixtureError, TypeError, ValueError):
                issues.append("skill_trust_receipt_invalid")
    obtained: set[str] = set()
    try:
        from_helper = verified_parent_citations(app)
        own = tuple(
            _canonical_get(app, r) for r in parent if r.call.name == "kb_get" and r.result.ok
        )
        if own != from_helper:
            raise ValueError("independent_get_mismatch")
        obtained = {c.chunk_id for c in own}
    except (FixtureError, TypeError, ValueError, AttributeError, KeyError):
        issues.append("agent_source_provenance_invalid")
    authority = app.retriever.selection(
        category=app.category, snapshot=app.snapshot
    ).authoritative_refs
    required = {c.chunk_id for c in app.retriever.index.chunks if c.doc_ref in authority}
    backend_complete = bool(app.backend) and all(
        b.outcome == Outcome.SUFFICIENT.value for b in app.backend
    )
    for check in app.backend:
        assessment = next(
            (
                r
                for r in parent
                if r.ordinal == check.ordinal and r.call.name == "ticket_assess" and r.result.ok
            ),
            None,
        )
        before_gets = {
            r.citation.chunk_id
            for r in parent
            if r.ordinal < check.ordinal
            and r.call.name == "kb_get"
            and r.result.ok
            and r.citation is not None
        }
        if (
            assessment is None
            or set(check.required_chunks) != required
            or set(check.parent_chunks) != before_gets
        ):
            issues.append("backend_agent_attribution_mismatch")
        if check.proposal_ref is not None:
            proposal = app.proposals.get(check.proposal_ref)
            if (
                proposal is None
                or proposal.run_id != root_id
                or proposal.ticket_id != app.ticket_id
                or proposal.evidence_identity != check.evidence_identity
                or check.outcome != Outcome.SUFFICIENT.value
                or not required <= before_gets
            ):
                issues.append("proposal_evidence_binding_mismatch")
    for commit in commits:
        matching = [
            r
            for r in transitions
            if len(r.after.commits) > len(r.before.commits) and commit in r.after.commits
        ]
        if len(matching) != 1:
            issues.append("journal_without_actual_write")
            continue
        arguments = _object(matching[0].call.arguments)
        reference = arguments.get("proposal_ref")
        proposal = app.proposals.get(reference) if type(reference) is str else None
        if (
            proposal is None
            or type(reference) is not str
            or arguments != transition_arguments(proposal, reference)
            or commit.action_id != proposal.idempotency_id
            or commit.binding != proposal.binding
            or commit.after != state
            or commit.before.revision != 0
            or commit.after.revision != 1
            or commit.before.tenant_id != commit.after.tenant_id
        ):
            issues.append("journal_proposal_binding_mismatch")
    last_transition = max((r.ordinal for r in transitions), default=0)
    readback = False
    readback_ordinal = 0
    read_error = False
    for record in parent:
        if record.call.name != "ticket_read" or record.ordinal <= last_transition:
            continue
        if not record.result.ok:
            read_error = True
            continue
        try:
            payload = _object(record.result.content)
            observed = decode(TicketState, payload["state"])
            if (
                payload.get("schema") != SCHEMA
                or payload.get("root_run_id") != root_id
                or observed != record.read_state
                or observed != state
            ):
                raise ValueError("readback_binding")
            readback = bool(commits)
            readback_ordinal = record.ordinal
        except (FixtureError, TypeError, ValueError, AttributeError, KeyError):
            issues.append("readback_binding_mismatch")
    todo_tasks, todo_complete, todo_final_read = 0, 0, False
    try:
        todo_tasks, todo_complete, todo_final_read = _todo(parent)
        if commits:
            reads = [r.ordinal for r in parent if r.call.name == "todo_read" and r.result.ok]
            completions = [
                r.ordinal
                for r in parent
                if r.call.name == "todo_write"
                and r.result.ok
                and _object(r.call.arguments).get("status") == "completed"
            ]
            todo_final_read = todo_final_read and bool(reads) and reads[-1] > readback_ordinal
            todo_final_read = (
                todo_final_read and bool(completions) and completions[-1] > readback_ordinal
            )
    except (FixtureError, TypeError, ValueError, AttributeError, KeyError):
        issues.append("todo_evidence_invalid")
    children_valid, child_limited = _children(app, parent)
    if not children_valid and (app.proposals or commits):
        issues.append("invalid_child_granted_authority")
    answer: dict[str, object] | None = None
    claims: set[str] = set()
    if result.stop_reason is StopReason.COMPLETED:
        if (
            not result.messages
            or type(result.messages[-1]) is not Message
            or result.messages[-1].role != "assistant"
            or result.messages[-1].content != result.final_answer
        ):
            issues.append("final_answer_transcript_mismatch")
        try:
            answer = _object(result.final_answer or "", 8192)
            ids = answer.get("citation_ids")
            if (
                set(answer) != {"schema", "status", "ticket_id", "revision", "citation_ids"}
                or answer["schema"] != SCHEMA
                or answer["ticket_id"] != app.ticket_id
                or type(answer["revision"]) is not int
                or answer["revision"] != state.revision
                or answer["status"]
                not in {
                    "resolved",
                    "awaiting_review",
                    "awaiting_information",
                    "blocked",
                    "read_only",
                }
                or type(ids) is not list
                or len(ids) > 32
                or any(type(i) is not str or not 0 < len(i) <= 256 for i in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("final_schema")
            claims = set(ids)
            if not claims <= obtained:
                issues.append("unsupported_final_citation")
            if answer["status"] not in {"blocked", "read_only"} and (
                not commits or answer["status"] != state.status.value
            ):
                issues.append("false_business_completion")
            if answer["status"] == "read_only" and (commits or transitions):
                issues.append("false_read_only_claim")
        except (FixtureError, TypeError, ValueError, AttributeError, KeyError):
            issues.append("final_claim_invalid")
    budget_reasons = {
        StopReason.MAX_MODEL_ROUNDS,
        StopReason.MAX_TOOL_CALLS,
        StopReason.MAX_TOTAL_MODEL_CALLS,
        StopReason.MAX_TOTAL_TOOL_CALLS,
        StopReason.CONTEXT_ERROR,
    }
    errors = tuple(r for r in parent if not r.result.ok)
    denied = any(
        r.call.name == "ticket_transition"
        and r.result.error is not None
        and r.result.error.code == "permission_denied"
        for r in parent
    )
    transition_error = any(not r.result.ok for r in transitions)
    if commits and (not readback or read_error or transition_error):
        outcome = "postcommit_verification_error"
    elif result.stop_reason in budget_reasons or child_limited:
        outcome = "budget_exhaustion"
    elif issues or result.stop_reason is not StopReason.COMPLETED:
        outcome = "agent_failure"
    elif denied:
        outcome = "permission_denial"
    elif errors:
        outcome = "tool_error"
    elif answer is not None and answer["status"] == "blocked" and not commits:
        outcome = "expected_denial"
    elif commits:
        if (
            not required <= obtained
            or not required <= claims
            or not backend_complete
            or not readback
            or not todo_tasks
            or todo_complete != todo_tasks
            or not todo_final_read
            or answer is None
            or answer["status"] != state.status.value
        ):
            issues.append("business_success_unproven")
            outcome = "agent_failure"
        else:
            outcome = "business_success"
    elif answer is not None and answer["status"] == "read_only" and claims:
        outcome = "read_only_success"
    else:
        issues.append("completion_unclassified")
        outcome = "agent_failure"
    return IntegrationGrade(
        outcome,
        tuple(dict.fromkeys(issues)),
        root_models,
        child_models,
        len(parent),
        len(child),
        sum(r.call.name == "kb_search" for r in parent),
        sum(r.call.name == "kb_get" for r in parent),
        len(obtained),
        len(required),
        sum(b.backend_search_calls for b in app.backend),
        backend_complete,
        sum(r.handler_entries for r in transitions),
        len(commits),
        state.revision,
        state.status.value,
        readback,
        todo_tasks,
        todo_complete,
        todo_final_read,
        sum(r.call.name == "load_skill" and r.result.ok for r in parent),
        sum(
            event == "compaction_completed" and fields.get("depth") == 0
            for event, fields in app.trace.rows
        ),
        children_valid,
    )
