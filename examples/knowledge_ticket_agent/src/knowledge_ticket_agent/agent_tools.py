"""Bounded Demo adapters around the unchanged Core Registry and M3 backend.

Runtime observations are ephemeral public-development evidence, not a new
permission system. Trusted application objects are not process isolation.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from agent_harness_lab.contracts import (
    ParameterType,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from agent_harness_lab.permissions import (
    ApprovalHandler,
    ApprovalRequest,
    Argument,
    Decision,
    Permission,
    PermissionEngine,
)
from agent_harness_lab.skill_tools import register_skill_tools
from agent_harness_lab.skills import SkillRegistry
from agent_harness_lab.todo_tools import register_todo_tools
from agent_harness_lab.tools import ToolExecutionError, ToolHandler, ToolRegistry
from agent_harness_lab.tracing import TraceSink
from agent_harness_lab.workspace import WorkspaceFiles
from knowledge_ticket_agent.approvals import (
    BusinessWorkflow,
    Proposal,
    ScriptedApproval,
    collect_evidence,
    documentary_requirements,
    invocation_arguments,
)
from knowledge_ticket_agent.citations import Citation, cite, verify_citation
from knowledge_ticket_agent.evaluation_scope import ArtificialCaseScope, execution_identity
from knowledge_ticket_agent.formal_scope import SealedScope
from knowledge_ticket_agent.relevance_evidence import source_consistent
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever
from knowledge_ticket_agent.rules import evaluate_rules
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.strict import FixtureError, decode, parse_json
from knowledge_ticket_agent.ticket_store import FaultHook, StoreSnapshot, TicketState

SCHEMA = "lqd-m4-1.0.0"


def encoded(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def exact(left: Mapping[str, Argument], right: Mapping[str, Argument]) -> bool:
    return left.keys() == right.keys() and all(
        type(v) is type(right[k]) and v == right[k] for k, v in left.items()
    )


@dataclass(frozen=True)
class ToolObservation:
    ordinal: int
    parent_ordinal: int | None
    run_id: str
    depth: int
    call: ToolCall
    result: ToolResult
    handler_entries: int
    before: StoreSnapshot
    after: StoreSnapshot
    citation: Citation | None = None
    read_state: TicketState | None = None


@dataclass
class _Active:
    ordinal: int
    call: ToolCall
    handler_entries: int = 0
    citation: Citation | None = None
    read_state: TicketState | None = None


@dataclass(frozen=True)
class BackendCheck:
    ordinal: int
    outcome: str
    backend_search_calls: int
    required_chunks: tuple[str, ...]
    parent_chunks: tuple[str, ...]
    proposal_ref: str | None
    evidence_identity: str | None


class RuntimeTrace:
    def __init__(self, app: "DemoApplication") -> None:
        self.app = app
        self.rows: list[tuple[str, dict[str, object]]] = []

    def record(self, event: str, **fields: object) -> None:
        if len(self.rows) >= 2048:
            raise ValueError("trace_capacity")
        self.rows.append((event, dict(fields)))
        if event == "run.started" and fields.get("depth") == 0:
            self.app._start(cast(str, fields["run_id"]))


class DemoRegistry(ToolRegistry):
    """Instrumentation only: all schema/permission/handler dispatch uses super."""

    def __init__(self, engine: PermissionEngine, app: "DemoApplication") -> None:
        super().__init__(permission_engine=engine)
        self.app = app
        self.records: list[ToolObservation] = []
        self._stack: list[_Active] = []
        self._ordinal = 0

    @property
    def current(self) -> _Active:
        if not self._stack:
            raise ToolExecutionError("invocation_required", "actual Agent invocation required")
        return self._stack[-1]

    def register(
        self, definition: ToolDefinition, handler: ToolHandler, *, permission: Permission
    ) -> None:
        def counted(arguments: Mapping[str, Argument]) -> str:
            self.current.handler_entries += 1
            return handler(arguments)

        super().register(definition, counted, permission=permission)

    def invoke(
        self,
        call: ToolCall,
        *,
        trace: TraceSink | None = None,
        round_number: int | None = None,
        tool_attempt: int | None = None,
    ) -> ToolResult:
        frame = self.execution
        if frame is None or round_number is None or tool_attempt is None:
            raise ToolExecutionError("agent_scope_required", "actual Agent dispatch required")
        if self._ordinal >= 64:
            raise ToolExecutionError(
                "observation_capacity", "bounded invocation limit", recoverable=False
            )
        self._ordinal += 1
        active = _Active(self._ordinal, call)
        parent = self._stack[-1].ordinal if self._stack else None
        before = self.app.workflow.store.snapshot
        self._stack.append(active)
        try:
            result = super().invoke(
                call, trace=trace, round_number=round_number, tool_attempt=tool_attempt
            )
            self.records.append(
                ToolObservation(
                    active.ordinal,
                    parent,
                    frame.run_id,
                    frame.depth,
                    call,
                    result,
                    active.handler_entries,
                    before,
                    self.app.workflow.store.snapshot,
                    active.citation,
                    active.read_state,
                )
            )
            return result
        finally:
            self._stack.pop()


def transition_arguments(proposal: Proposal, reference: str) -> dict[str, Argument]:
    return {
        "proposal_ref": reference,
        "ticket_id": proposal.ticket_id,
        "tenant_id": proposal.tenant_id,
        "actor_id": proposal.actor_id,
        "action": proposal.action,
        "destination": proposal.destination.value,
        "expected_revision": proposal.expected_revision,
        "snapshot": proposal.snapshot,
    }


class DemoApplication:
    """One fresh application per actual root Agent.run, no global mutable state."""

    def __init__(
        self,
        data: Dataset,
        ticket_id: str,
        *,
        actor_id: str = "SYN-STAFF-02",
        snapshot: str = "normal",
        approval: str = "approve",
        skills_root: Path | None = None,
        fault: FaultHook | None = None,
        approval_handler: ApprovalHandler | None = None,
        case_scope: ArtificialCaseScope | SealedScope | None = None,
        transition_policy: Decision = Decision.ASK,
        retriever: Retriever | None = None,
    ) -> None:
        if transition_policy not in {Decision.ASK, Decision.DENY}:
            raise ValueError("transition_policy_must_restrict")
        if approval not in {"approve", "refuse", "missing", "invalid", "throw"}:
            raise ValueError("approval_strategy")
        self.case_scope = case_scope
        self.data, self.ticket_id, self.actor_id = data, ticket_id, actor_id
        self.snapshot, self.approval, self.fault = snapshot, approval, fault
        self.category = evaluate_rules(
            data, ticket_id, snapshot=snapshot, case_scope=case_scope
        ).category
        if self.category is Category.OTHER and not isinstance(case_scope, SealedScope):
            raise ValueError("unsupported_demo_ticket")
        original = Retriever(data)
        if retriever is not None:
            if retriever.index != original.index:
                raise ValueError("application_retrieval_corpus")
        self.retriever = original if retriever is None else retriever
        self._workflow: BusinessWorkflow | None = None
        self.initial: StoreSnapshot | None = None
        self.proposals: dict[str, Proposal] = {}
        self.backend: list[BackendCheck] = []
        self.approval_calls: list[ApprovalRequest] = []
        self._searched: dict[str, set[str]] = {}
        self._workspace: WorkspaceFiles | None = None
        handler = (
            approval_handler
            if approval_handler is not None
            else (None if approval == "missing" else self._approve)
        )
        engine = PermissionEngine(
            permissions=(Permission.READ, Permission.WRITE),
            rules={"ticket_transition": transition_policy},
            approval_handler=handler,
        )
        self.registry = DemoRegistry(engine, self)
        self.trace = RuntimeTrace(self)
        self._register()
        register_todo_tools(self.registry)
        if skills_root is not None:
            self._workspace = WorkspaceFiles(skills_root)
            register_skill_tools(self.registry, SkillRegistry(self._workspace, ("skills",)))

    @property
    def workflow(self) -> BusinessWorkflow:
        if self._workflow is None:
            raise ValueError("root_run_not_started")
        return self._workflow

    def _start(self, run_id: str) -> None:
        if self._workflow is not None:
            raise ValueError("application_already_consumed")
        self._workflow = BusinessWorkflow(self.data, run_id, case_scope=self.case_scope)
        self.initial = self.workflow.store.snapshot

    def close(self) -> None:
        if self._workspace is not None:
            self._workspace.close()

    def _root(self) -> None:
        frame = self.registry.execution
        if frame is None or frame.depth != 0 or frame.run_id != self.workflow.run_id:
            raise ToolExecutionError("root_only", "operation requires this root run")

    def _target(self, arguments: Mapping[str, Argument]) -> None:
        self._root()
        if arguments["ticket_id"] != self.ticket_id:
            raise ToolExecutionError("ticket_scope", "ticket outside this task")

    def _approve(self, request: ApprovalRequest) -> bool:
        self._root()
        self.approval_calls.append(request)
        reference = request.arguments.get("proposal_ref")
        proposal = self.proposals.get(reference) if type(reference) is str else None
        if (
            proposal is None
            or not self._singleton_transition()
            or not self._current_proposal(proposal)
            or not exact(request.arguments, transition_arguments(proposal, cast(str, reference)))
        ):
            return False
        expected = ApprovalRequest(
            self.registry.current.call.call_id,
            "ticket_transition",
            Permission.WRITE,
            transition_arguments(proposal, cast(str, reference)),
        )
        if not ScriptedApproval(expected, True)(request):
            return False
        if self.approval == "invalid":
            return cast(bool, "yes")  # Explicit public fault injection; Core must reject.
        if self.approval == "throw":
            raise RuntimeError("scripted_approval_failure")
        return self.approval == "approve"

    def _singleton_transition(self) -> bool:
        """A WRITE decision cannot share dynamic delegation/other tool effects."""
        responses = [
            fields
            for event, fields in self.trace.rows
            if event == "model.response"
            and fields.get("run_id") == self.workflow.run_id
            and fields.get("depth") == 0
        ]
        return bool(responses) and responses[-1].get("item_count") == 1

    def _current_proposal(self, proposal: Proposal) -> bool:
        """Approval cannot freeze a superseded application policy/fact context."""
        return (
            proposal.run_id == self.workflow.run_id
            and proposal.ticket_id == self.ticket_id
            and proposal.actor_id == self.actor_id
            and proposal.snapshot == self.snapshot
            and type(self.data) is Dataset
            and execution_identity(self.data, self.case_scope) == proposal.context_identity
            and self.category
            is evaluate_rules(
                self.data, self.ticket_id, snapshot=self.snapshot, case_scope=self.case_scope
            ).category
            and self.workflow._valid(proposal)
        )

    def required_chunks(self) -> tuple[str, ...]:
        if self.category is Category.OTHER and isinstance(self.case_scope, SealedScope):
            return ()  # Unsupported action has no executable policy obligations.
        required = {
            (c.doc_ref, c.section_id)
            for r in documentary_requirements(self.data, self.category, self.snapshot)
            for c in r.components
        }
        return tuple(
            sorted(
                c.chunk_id
                for c in self.retriever.index.chunks
                if (c.doc_ref, c.section_id) in required
            )
        )

    def parent_chunks(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    r.citation.chunk_id
                    for r in self.registry.records
                    if r.depth == 0
                    and r.result.ok
                    and r.call.name == "kb_get"
                    and r.citation is not None
                }
            )
        )

    def children_valid(self) -> bool:
        """Claims count only if bounded output matches that actual Child's full gets."""
        for record in self.registry.records:
            if record.depth != 0 or record.call.name != "delegate_task":
                continue
            try:
                wrapper = parse_json(record.result.content)
                if type(wrapper) is not dict:
                    return False
                if set(wrapper) != {
                    "status",
                    "stop_reason",
                    "result",
                    "model_calls",
                    "tool_calls",
                    "error_code",
                    "truncated",
                }:
                    return False
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
                    or len(wrapper["result"]) > 2048
                ):
                    return False
                claims = parse_json(wrapper["result"])
                if (
                    type(claims) is not dict
                    or set(claims) != {"schema", "chunk_ids"}
                    or claims["schema"] != SCHEMA
                ):
                    return False
                ids = claims["chunk_ids"]
                child_records = [
                    r for r in self.registry.records if r.parent_ordinal == record.ordinal
                ]
                obtained = {
                    r.citation.chunk_id
                    for r in child_records
                    if r.result.ok and r.citation is not None
                }
                if (
                    type(ids) is not list
                    or not 1 <= len(ids) <= 8
                    or len(set(ids)) != len(ids)
                    or any(type(i) is not str or i not in obtained for i in ids)
                    or wrapper["tool_calls"] != len(child_records)
                ):
                    return False
            except (FixtureError, TypeError, ValueError):
                return False
        return True

    def _register(self) -> None:
        def read(arguments: Mapping[str, Argument]) -> str:
            self._target(arguments)
            if self.fault and self.workflow.store.snapshot.commits:
                self.fault("ticket_read")
            state = self.workflow.store.read(self.ticket_id)
            self.registry.current.read_state = state
            ticket = next(t for t in self.data.tickets if t.ticket_id == self.ticket_id)
            return encoded(
                {
                    "schema": SCHEMA,
                    "state": asdict(state),
                    "root_run_id": self.workflow.run_id,
                    "trusted_facts": asdict(ticket.facts),
                }
            )

        def search(arguments: Mapping[str, Argument]) -> str:
            result = self.retriever.search(
                cast(str, arguments["query"]),
                category=self.category,
                snapshot=self.snapshot,
                top_k=5,
            )
            frame = self.registry.execution
            assert frame is not None
            self._searched.setdefault(frame.run_id, set()).update(h.chunk_id for h in result.hits)
            return encoded({"schema": SCHEMA, "result": asdict(result)})

        def get(arguments: Mapping[str, Argument]) -> str:
            frame = self.registry.execution
            assert frame is not None
            chunk_id = cast(str, arguments["chunk_id"])
            try:
                if chunk_id not in self._searched.get(frame.run_id, set()):
                    raise ValueError("no_actual_search")
                citation = cite(
                    self.retriever, chunk_id, category=self.category, snapshot=self.snapshot
                )
                if citation.doc_ref != arguments["doc_ref"]:
                    raise ValueError("source_version_mismatch")
                verify_citation(
                    citation,
                    self.data,
                    self.retriever,
                    category=self.category,
                    snapshot=self.snapshot,
                    require_authority=False,
                )
                result = encoded({"schema": SCHEMA, "citation": asdict(citation)})
            except (FixtureError, ValueError):
                raise ToolExecutionError(
                    "source_rejected", "indexed source binding rejected"
                ) from None
            self.registry.current.citation = citation  # Only after actual canonical output exists.
            return result

        def assess(arguments: Mapping[str, Argument]) -> str:
            self._target(arguments)
            if len(self.backend) >= 16:
                raise ToolExecutionError("assessment_capacity", "bounded assessment limit")
            if self.category is Category.OTHER and isinstance(self.case_scope, SealedScope):
                decision = evaluate_rules(
                    self.data, self.ticket_id, snapshot=self.snapshot, case_scope=self.case_scope
                ).decision.value
                if decision != "unsupported":
                    raise ToolExecutionError("unsupported_context", "invalid trusted context")
                self.backend.append(
                    BackendCheck(
                        self.registry.current.ordinal,
                        "NOT_ASSESSABLE",
                        0,
                        (),
                        self.parent_chunks(),
                        None,
                        None,
                    )
                )
                return encoded(
                    {
                        "schema": SCHEMA,
                        "status": "blocked",
                        "reason": "unsupported_action",
                        "backend_outcome": "NOT_ASSESSABLE",
                        "decision": decision,
                        "proposal": None,
                    }
                )
            evidence = collect_evidence(self.data, self.category, self.snapshot)
            # Backend retrieval is kept separate even when Agent provenance is insufficient.
            from knowledge_ticket_agent.approvals import evidence_gate

            outcome, _ = evidence_gate(self.data, self.category, self.snapshot, evidence)
            required, obtained = self.required_chunks(), self.parent_chunks()
            prepared = None
            reason = "agent_evidence_missing"
            if not parent_sources_consistent(self):
                reason = "source_directive_conflict"
            elif set(required) <= set(obtained) and self.children_valid():
                prepared = self.workflow.prepare(
                    self.ticket_id, self.actor_id, evidence, snapshot=self.snapshot
                )
                reason = prepared.reason
            elif not self.children_valid():
                reason = "child_provenance_rejected"
            proposal = prepared.proposal if prepared is not None else None
            reference = f"proposal-{len(self.proposals) + 1}" if proposal else None
            if proposal is not None and reference is not None:
                self.proposals[reference] = proposal
            self.backend.append(
                BackendCheck(
                    self.registry.current.ordinal,
                    outcome.value,
                    len(evidence),
                    required,
                    obtained,
                    reference,
                    proposal.evidence_identity if proposal else None,
                )
            )
            return encoded(
                {
                    "schema": SCHEMA,
                    "status": "proposal" if proposal else "blocked",
                    "reason": reason,
                    "backend_outcome": outcome.value,
                    "decision": evaluate_rules(
                        self.data,
                        self.ticket_id,
                        snapshot=self.snapshot,
                        case_scope=self.case_scope,
                    ).decision.value,
                    "proposal": transition_arguments(proposal, cast(str, reference))
                    if proposal
                    else None,
                }
            )

        def transition(arguments: Mapping[str, Argument]) -> str:
            if not parent_sources_consistent(self):
                raise ToolExecutionError(
                    "source_directive_conflict", "conflicting canonical evidence", recoverable=False
                )
            self._target(arguments)
            proposal = self.proposals.get(cast(str, arguments["proposal_ref"]))
            if (
                proposal is None
                or not self._singleton_transition()
                or not exact(
                    arguments, transition_arguments(proposal, cast(str, arguments["proposal_ref"]))
                )
                or not self._current_proposal(proposal)
            ):
                raise ToolExecutionError("proposal_scope", "issued proposal scope rejected")
            call_id = self.registry.current.call.call_id
            if (
                self.workflow._active
                or call_id in self.workflow._calls
                or len(self.workflow._calls) >= 16
            ):
                raise ToolExecutionError("invocation_invalid", "invocation already used or active")
            self.workflow.audit.reserve(10)
            self.workflow._calls.add(call_id)
            self.workflow._active = True
            try:
                self.workflow._apply_transition(
                    proposal, invocation_arguments(proposal), self.fault
                )
                # Transition acknowledgement is never final completion or readback proof.
                return encoded(
                    {
                        "schema": SCHEMA,
                        "status": "journal_recorded",
                        "verification": "ticket_read_required",
                    }
                )
            finally:
                self.workflow._active = False

        def register(
            name: str,
            handler: ToolHandler,
            fields: tuple[str, ...],
            permission: Permission = Permission.READ,
        ) -> None:
            self.registry.register(
                ToolDefinition(
                    name,
                    "Bounded fictional Demo operation",
                    tuple(
                        ToolParameter(
                            f,
                            ParameterType.INTEGER
                            if f == "expected_revision"
                            else ParameterType.STRING,
                        )
                        for f in fields
                    ),
                ),
                handler,
                permission=permission,
            )

        register("ticket_read", read, ("ticket_id",))
        register("kb_search", search, ("query",))
        register("kb_get", get, ("chunk_id", "doc_ref"))
        register("ticket_assess", assess, ("ticket_id",))
        register(
            "ticket_transition",
            transition,
            (
                "proposal_ref",
                "ticket_id",
                "tenant_id",
                "actor_id",
                "action",
                "destination",
                "expected_revision",
                "snapshot",
            ),
            Permission.WRITE,
        )


def verified_parent_citations(app: DemoApplication) -> tuple[Citation, ...]:
    """Reverify actual successful call/output pairs, not a claimed citation list."""
    verified = []
    for r in app.registry.records:
        if r.depth != 0 or r.call.name != "kb_get" or not r.result.ok:
            continue
        raw = parse_json(r.result.content)
        if type(raw) is not dict or set(raw) != {"schema", "citation"} or raw["schema"] != SCHEMA:
            raise ValueError("invalid_get_output")
        citation = decode(Citation, raw["citation"])
        verify_citation(
            citation,
            app.data,
            app.retriever,
            category=app.category,
            snapshot=app.snapshot,
            require_authority=False,
        )
        if citation != r.citation or citation != cite(
            app.retriever, citation.chunk_id, category=app.category, snapshot=app.snapshot
        ):
            raise ValueError("get_receipt_mismatch")
        verified.append(citation)
    return tuple(verified)


def call_reference(call_id: str) -> str:
    return hashlib.sha256(call_id.encode()).hexdigest()[:16]


def parent_sources_consistent(app: DemoApplication) -> bool:
    if not isinstance(app.retriever, RelevanceRetriever):
        return True
    return source_consistent(
        app.data,
        app.retriever,
        verified_parent_citations(app),
        category=app.category,
        snapshot=app.snapshot,
    )


class PolicyEvidenceSession:
    """Explicit read-only domain QA tools. Observations come only from actual handlers.

    The same proof engine is used by the business adapters, whose separate frozen
    M3 contract cannot be replaced by this session's query or semantic verdict.
    """

    def __init__(
        self,
        data: Dataset,
        retriever: RelevanceRetriever,
        query: str,
        *,
        category: Category,
        snapshot: str = "normal",
    ) -> None:
        from knowledge_ticket_agent.relevance_evidence import select_sources, verify_support

        self.data, self.retriever, self.query = data, retriever, query
        self.category, self.snapshot = category, snapshot
        self.searched: set[str] = set()
        self.reads: dict[str, Citation] = {}
        self.registry = ToolRegistry(permissions=(Permission.READ,))

        def fresh() -> tuple[set[str], dict[str, Citation]]:
            return set(), {}

        def state() -> tuple[set[str], dict[str, Citation]]:
            current = self.registry.run_resource(self, fresh)
            self.searched, self.reads = current  # Audit view only, not the next run's input.
            return current

        def search(arguments: Mapping[str, Argument]) -> str:
            selection = select_sources(
                data, retriever, cast(str, arguments["query"]), category=category, snapshot=snapshot
            )
            result = selection.result
            searched, _ = state()
            searched.update(h.chunk_id for h in result.hits)
            searched.update(selection.read_dependencies)
            return encoded(
                {"result": asdict(result), "read_dependencies": selection.read_dependencies}
            )

        def get(arguments: Mapping[str, Argument]) -> str:
            identifier = cast(str, arguments["chunk_id"])
            searched, reads = state()
            if identifier not in searched:
                raise ToolExecutionError("get_without_search", "Actual source candidate required")
            citation = cite(retriever, identifier, category=category, snapshot=snapshot)
            verify_citation(citation, data, retriever, category=category, snapshot=snapshot)
            reads[identifier] = citation
            return encoded({"citation": asdict(citation), "method": retriever.method_identity})

        def verify(arguments: Mapping[str, Argument]) -> str:
            _, reads = state()
            verdict = verify_support(
                data,
                retriever,
                query,
                tuple(reads.values()),
                category=category,
                snapshot=snapshot,
            )
            return encoded(
                {
                    "status": verdict.status,
                    "basis": "source_assertion",
                    "method": verdict.method_identity,
                    "retrieval_method": verdict.retrieval_method,
                    "missing": verdict.missing,
                    "unknown": verdict.unknown,
                    "conflicts": [
                        {
                            "reason": c.reason,
                            "sources": [
                                claim.span.chunk_id for claim in (c.left, c.right, *c.additional)
                            ],
                        }
                        for c in verdict.conflicts
                    ],
                    "bindings": [
                        {
                            "obligation": b.obligation_id,
                            "key": b.requirement.key,
                            "sources": [
                                {
                                    "chunk": c.span.chunk_id,
                                    "start": c.span.start,
                                    "end": c.span.end,
                                    "sha256": c.span.quote_sha256,
                                }
                                for c in b.claims
                            ],
                            "qualifiers": [c.span.chunk_id for c in b.qualifiers],
                        }
                        for b in verdict.bindings
                    ],
                }
            )

        for name, handler, parameters in (
            ("kb_search", search, (ToolParameter("query", ParameterType.STRING),)),
            ("kb_get", get, (ToolParameter("chunk_id", ParameterType.STRING),)),
            ("evidence_verify", verify, ()),
        ):
            self.registry.register(
                ToolDefinition(name, "Read-only canonical policy evidence", parameters),
                handler,
                permission=Permission.READ,
            )
