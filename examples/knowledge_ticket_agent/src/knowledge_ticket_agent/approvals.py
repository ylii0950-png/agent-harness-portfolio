"""Finite synchronous workflow with documentary checks and Core exact ASK dispatch.

Trusted application objects/configuration are not a Python security boundary.
Only public source navigation is performed; no Gold or formal runner is imported.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from agent_harness_lab.contracts import ParameterType, ToolCall, ToolDefinition, ToolParameter
from agent_harness_lab.permissions import (
    ApprovalHandler,
    ApprovalRequest,
    Argument,
    Decision,
    Permission,
    PermissionEngine,
)
from agent_harness_lab.tools import ToolExecutionError, ToolRegistry
from knowledge_ticket_agent.audit import BusinessAudit
from knowledge_ticket_agent.citations import cite
from knowledge_ticket_agent.evaluation_scope import (
    ArtificialCaseScope,
    execution_identity,
    validate_scope,
)
from knowledge_ticket_agent.evidence_adapter import (
    EvidenceRequest,
    EvidenceRequirement,
    Outcome,
    SectionComponent,
    assess_evidence,
    scope_for,
)
from knowledge_ticket_agent.formal_scope import SealedScope
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.rules import RuleEvaluation, evaluate_rules
from knowledge_ticket_agent.schemas import Category, Dataset, Split, Status
from knowledge_ticket_agent.strict import FixtureError
from knowledge_ticket_agent.ticket_store import FaultHook, TicketState, TicketStore

TOOL = "ticket_transition"
WORKFLOW_VERSION = "lqd-m3-1.0.0"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def documentary_requirements(
    data: Dataset, category: Category, snapshot: str = "normal"
) -> tuple[EvidenceRequirement, ...]:
    """All sections of all current authoritative policies, including restrictions.

    One section per adapter request keeps its existing five-hit limit explicit.
    This catalog is rebuilt by the application, never supplied by the requester.
    """
    retriever = Retriever(data)
    scope = scope_for(data, category=category, snapshot=snapshot)
    selection = retriever.selection(category=category, snapshot=snapshot)
    sections = sorted(
        (d.metadata.ref, s.section_id)
        for d in data.documents
        if d.metadata.ref in selection.authoritative_refs
        for s in d.sections
    )
    if not 0 < len(sections) <= 16:
        raise ValueError("documentary_capacity")
    return tuple(
        EvidenceRequirement(
            f"m3-section-{n}",
            WORKFLOW_VERSION,
            scope,
            (SectionComponent(f"section-{n}", ref, section),),
        )
        for n, (ref, section) in enumerate(sections, 1)
    )


def collect_evidence(
    data: Dataset, category: Category, snapshot: str = "normal"
) -> tuple[EvidenceRequest, ...]:
    """Public catalog-directed lexical navigation, not a new ranking benchmark.

    Use genuine unchanged Retriever.search results and canonical citations. Never
    synthesize hits or fill missing sections: a retrieval miss blocks execution.
    No query/score optimization or frozen evaluation input is involved.
    """
    retriever = Retriever(data)
    requests = []
    for requirement in documentary_requirements(data, category, snapshot):
        component = requirement.components[0]
        document = next(d for d in data.documents if d.metadata.ref == component.doc_ref)
        section = next(s for s in document.sections if s.section_id == component.section_id)
        result = retriever.search(
            (section.heading + " " + document.metadata.title)[:256],
            category=category,
            snapshot=snapshot,
            top_k=5,
        )
        requests.append(
            EvidenceRequest(
                requirement.requirement_id,
                requirement.version,
                requirement.scope,
                result,
                tuple(
                    cite(retriever, h.chunk_id, category=category, snapshot=snapshot)
                    for h in result.hits
                ),
            )
        )
    return tuple(requests)


def evidence_gate(
    data: Dataset, category: Category, snapshot: str, evidence: tuple[EvidenceRequest, ...]
) -> tuple[Outcome, str]:
    try:
        requirements = documentary_requirements(data, category, snapshot)
        if type(evidence) is not tuple or len(evidence) != len(requirements):
            return Outcome.INSUFFICIENT, "documentary_set_incomplete"
        if any(type(r) is not EvidenceRequest for r in evidence):
            return Outcome.NOT_ASSESSABLE, "documentary_request_invalid"
        if tuple(r.requirement_id for r in evidence) != tuple(
            r.requirement_id for r in requirements
        ):
            return Outcome.NOT_ASSESSABLE, "documentary_identity_mismatch"
        outcomes = tuple(
            assess_evidence(data, requirements, request).outcome for request in evidence
        )
        for status in (Outcome.NOT_ASSESSABLE, Outcome.CONFLICT, Outcome.INSUFFICIENT):
            if status in outcomes:
                return status, "documentary_gate_blocked"
        return Outcome.SUFFICIENT, _digest(
            [asdict(r) for r in requirements] + [asdict(e) for e in evidence]
        )
    except (FixtureError, ValueError, TypeError, AttributeError, KeyError):
        return Outcome.NOT_ASSESSABLE, "documentary_configuration_invalid"


@dataclass(frozen=True)
class Proposal:
    run_id: str
    proposal_id: str
    ticket_id: str
    tenant_id: str
    actor_id: str
    expected_revision: int
    action: str
    destination: Status
    rule_version: str
    snapshot: str
    decision: str
    context_identity: str
    evidence_identity: str
    required_permission: str
    idempotency_id: str

    @property
    def binding(self) -> str:
        return _digest(asdict(self))


def invocation_arguments(proposal: Proposal) -> dict[str, Argument]:
    # Match the flat primitive schema Core parses from JSON, including enum text.
    values = {**asdict(proposal), "binding": proposal.binding}
    return {k: v.value if isinstance(v, Status) else v for k, v in values.items()}


def approval_request(proposal: Proposal, call_id: str) -> ApprovalRequest:
    """Trusted application constructs the exact current invocation expectation."""
    return ApprovalRequest(call_id, TOOL, Permission.WRITE, invocation_arguments(proposal))


@dataclass(frozen=True)
class ScriptedApproval:
    expected: ApprovalRequest
    allow: bool

    def __call__(self, request: ApprovalRequest) -> bool:
        return (
            type(self.allow) is bool
            and self.allow is True
            and type(request) is ApprovalRequest
            and type(self.expected) is ApprovalRequest
            and type(request.call_id) is type(self.expected.call_id) is str
            and request.call_id == self.expected.call_id
            and type(request.tool_name) is type(self.expected.tool_name) is str
            and request.tool_name == self.expected.tool_name
            and self.expected.permission is Permission.WRITE
            and request.permission is Permission.WRITE
            and request.arguments.keys() == self.expected.arguments.keys()
            and all(
                type(value) in {str, int, bool}
                and type(value) is type(self.expected.arguments[name])
                and value == self.expected.arguments[name]
                for name, value in request.arguments.items()
            )
        )


@dataclass(frozen=True)
class Preparation:
    evaluation: RuleEvaluation
    evidence_outcome: Outcome
    reason: str
    proposal: Proposal | None


@dataclass(frozen=True)
class WorkflowResult:
    code: str
    authorized: bool
    handler_invocations: int
    committed: bool
    readback_verified: bool
    final_state: TicketState | None
    mutated: bool = False


class _PermissionTrace:
    def __init__(self, audit: BusinessAudit, proposal: Proposal) -> None:
        self.audit, self.proposal = audit, proposal
        self.authorized = False

    def record(self, event: str, **fields: object) -> None:
        if event == "tool.call":
            self.authorized = fields.get("authorized") is True
            self.audit.record(
                "permission", str(fields["permission_decision"]), self.proposal.ticket_id
            )
            self.audit.record("approval", str(fields["permission_reason"]), self.proposal.ticket_id)


class BusinessWorkflow:
    """One trusted configuration, store and bounded audit per root run.

    Proposal IDs are not credentials. Issued-object equality, current context,
    full documentary checks, actor scope and Core ASK all remain required.
    No parallel or nested execute calls are accepted by this synchronous demo.
    """

    def __init__(
        self,
        data: Dataset,
        run_id: str,
        *,
        case_scope: ArtificialCaseScope | SealedScope | None = None,
        store: TicketStore | None = None,
    ) -> None:
        if type(run_id) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", run_id):
            raise ValueError("run_id_invalid")
        if case_scope is not None:
            validate_scope(data, case_scope.ticket_id, case_scope)
        self._case_scope = case_scope
        self._data, self.run_id = data, run_id
        tickets = tuple(
            t
            for t in data.tickets
            if (t.split is Split.DEV if case_scope is None else t.ticket_id == case_scope.ticket_id)
        )
        if store is not None and (
            case_scope is not None
            or {(s.ticket_id, s.tenant_id) for s in store.snapshot.tickets}
            != {(t.ticket_id, t.tenant_id) for t in tickets}
        ):
            raise ValueError("workflow_store_scope")
        self.store = store if store is not None else TicketStore(tickets)
        self.audit = BusinessAudit()
        self._issued: dict[str, tuple[Proposal, tuple[EvidenceRequest, ...]]] = {}
        self._calls: set[str] = set()
        self._active = False
        self.handler_invocations = 0

    def _actor(self, proposal: Proposal) -> bool:
        return any(
            s.actor_id == proposal.actor_id
            and proposal.ticket_id in s.allowed_ticket_ids
            and proposal.action in s.allowed_actions
            for s in self._data.staff
        )

    def prepare(
        self,
        ticket_id: str,
        actor_id: str,
        evidence: tuple[EvidenceRequest, ...],
        *,
        snapshot: str = "normal",
    ) -> Preparation:
        if self._active or len(self._issued) >= 16:
            raise ValueError("workflow_capacity")
        self.audit.reserve(3)
        evaluation = evaluate_rules(
            self._data, ticket_id, snapshot=snapshot, case_scope=self._case_scope
        )
        safe_id = (
            ticket_id
            if any(
                t.ticket_id == ticket_id
                and (
                    t.split is Split.DEV
                    if self._case_scope is None
                    else t.ticket_id == self._case_scope.ticket_id
                )
                for t in self._data.tickets
            )
            else "unknown"
        )
        self.audit.record("recommendation", evaluation.decision.value, safe_id)
        outcome, identity = evidence_gate(self._data, evaluation.category, snapshot, evidence)
        if evaluation.destination is None or outcome is not Outcome.SUFFICIENT:
            self.audit.record("failure", "no_executable_proposal", safe_id)
            return Preparation(evaluation, outcome, "no_executable_proposal", None)
        ticket = next(t for t in self._data.tickets if t.ticket_id == ticket_id)
        state = self.store.read(ticket_id)
        if state.status is not Status.OPEN or state.revision != 0:
            return Preparation(evaluation, outcome, "stale_state", None)
        ordinal = len(self._issued) + 1
        identifier = _digest([self.run_id, ticket_id, ordinal])
        proposal = Proposal(
            self.run_id,
            identifier,
            ticket_id,
            ticket.tenant_id,
            actor_id,
            0,
            evaluation.destination.value,
            evaluation.destination,
            evaluation.rule_version,
            snapshot,
            evaluation.decision.value,
            execution_identity(self._data, self._case_scope),
            identity,
            Permission.WRITE.value,
            identifier,
        )
        if not self._actor(proposal):
            self.audit.record("failure", "actor_scope_denied", safe_id)
            return Preparation(evaluation, outcome, "actor_scope_denied", None)
        self._issued[identifier] = (proposal, evidence)
        self.audit.record("proposal", proposal.action, safe_id, identifier)
        return Preparation(evaluation, outcome, "proposal_created", proposal)

    def _valid(self, proposal: Proposal) -> bool:
        issued = self._issued.get(proposal.proposal_id)
        if (
            issued is None
            or issued[0] != proposal
            or type(proposal.expected_revision) is not int
            or proposal.context_identity != execution_identity(self._data, self._case_scope)
            or not self._actor(proposal)
        ):
            return False
        evaluation = evaluate_rules(
            self._data, proposal.ticket_id, snapshot=proposal.snapshot, case_scope=self._case_scope
        )
        outcome, identity = evidence_gate(
            self._data, evaluation.category, proposal.snapshot, issued[1]
        )
        if (
            evaluation.destination is not proposal.destination
            or evaluation.decision.value != proposal.decision
            or evaluation.rule_version != proposal.rule_version
            or outcome is not Outcome.SUFFICIENT
            or identity != proposal.evidence_identity
        ):
            return False
        state = self.store.read(proposal.ticket_id)
        replay = self.store.replay(proposal.idempotency_id, proposal.binding)
        return state.tenant_id == proposal.tenant_id and (
            (state.status is Status.OPEN and state.revision == proposal.expected_revision)
            or (replay is not None and replay.after == state)
        )

    def _apply_transition(
        self,
        proposal: Proposal,
        arguments: Mapping[str, Argument],
        fault: FaultHook | None = None,
    ) -> str:
        """Trusted Registry handler primitive; never an alternative permission gate.

        Both standalone M3 and the M4 WRITE adapter call this only after Core
        authorization. Callers must reserve audit capacity/invocation scope.
        This private Python method is not exposed as a model-visible tool.
        """
        self.handler_invocations += 1
        self.audit.record("handler", "entered", proposal.ticket_id, proposal.idempotency_id)
        if dict(arguments) != invocation_arguments(proposal) or not self._valid(proposal):
            raise ToolExecutionError("proposal_changed", "current proposal validation failed")
        changed = self.store._commit(
            ticket_id=proposal.ticket_id,
            tenant_id=proposal.tenant_id,
            expected_revision=proposal.expected_revision,
            destination=proposal.destination,
            action_id=proposal.idempotency_id,
            binding=proposal.binding,
            fault=fault,
        )
        self.audit.record(
            "commit" if changed else "replay",
            "new_transition" if changed else "existing_transition",
            proposal.ticket_id,
            proposal.idempotency_id,
            1,
        )
        return "simulated_transition_recorded"

    def execute(
        self,
        proposal: Proposal,
        call_id: str,
        *,
        approver: ApprovalHandler | None,
        permissions: tuple[Permission, ...] = (Permission.WRITE,),
        fault: FaultHook | None = None,
    ) -> WorkflowResult:
        if self._active:
            return WorkflowResult("invocation_active", False, 0, False, False, None)
        if (
            type(proposal) is not Proposal
            or type(call_id) is not str
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", call_id)
            or call_id in self._calls
            or len(self._calls) >= 16
        ):
            return WorkflowResult("invocation_invalid", False, 0, False, False, None)
        self.audit.reserve(10)
        self._calls.add(call_id)
        self._active = True
        initial_count = self.handler_invocations
        initial_snapshot = self.store.snapshot
        trace = _PermissionTrace(self.audit, proposal)
        code = "preflight_failed"
        verified = False
        try:
            if not self._valid(proposal):
                code = "proposal_invalid"
            else:
                engine = PermissionEngine(
                    permissions=permissions, rules={TOOL: Decision.ASK}, approval_handler=approver
                )
                registry = ToolRegistry(permission_engine=engine)

                def handler(arguments: Mapping[str, Argument]) -> str:
                    return self._apply_transition(proposal, arguments, fault)

                arguments = invocation_arguments(proposal)
                registry.register(
                    ToolDefinition(
                        TOOL,
                        "Simulated ticket state transition only",
                        tuple(
                            ToolParameter(
                                k, ParameterType.INTEGER if type(v) is int else ParameterType.STRING
                            )
                            for k, v in arguments.items()
                        ),
                    ),
                    handler,
                    permission=Permission.WRITE,
                )
                result = registry.invoke(
                    ToolCall(call_id, TOOL, json.dumps(arguments)), trace=trace
                )
                code = (
                    "completed"
                    if result.ok
                    else result.error.code
                    if result.error
                    else "execution_error"
                )
                if fault:
                    fault("readback")
                # This read happens after dispatch; callback/return text cannot establish success.
                actual = self.store.read(proposal.ticket_id)
                commit = self.store.replay(proposal.idempotency_id, proposal.binding)
                verified = result.ok and commit is not None and commit.after == actual
                if result.ok and not verified:
                    code = "readback_mismatch"
                self.audit.record(
                    "readback",
                    "verified" if verified else "unverified",
                    proposal.ticket_id,
                    proposal.idempotency_id,
                    actual.revision,
                )
        except Exception:
            code = "workflow_error"
        finally:
            self._active = False
        # Diagnostic state is truthful even when a post-publication hook/readback failed.
        known = self._issued.get(proposal.proposal_id)
        final_state = (
            next(s for s in self.store.snapshot.tickets if s.ticket_id == known[0].ticket_id)
            if known
            else None
        )
        committed = (
            known is not None
            and self.store.replay(known[0].idempotency_id, known[0].binding) is not None
        )
        if code != "completed":
            self.audit.record("failure", code, known[0].ticket_id if known else "unknown")
        return WorkflowResult(
            code,
            trace.authorized,
            self.handler_invocations - initial_count,
            committed,
            verified,
            final_state,
            self.store.snapshot != initial_snapshot,
        )
