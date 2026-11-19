"""Prospective public M3 boundaries, never sealed evaluation cases."""

from dataclasses import replace
from datetime import timedelta
from typing import Any, cast

import pytest
from knowledge_ticket_agent.approvals import (
    BusinessWorkflow,
    Proposal,
    ScriptedApproval,
    approval_request,
    collect_evidence,
    documentary_requirements,
    evidence_gate,
)
from knowledge_ticket_agent.evidence_adapter import Outcome
from knowledge_ticket_agent.grading import grade_transition
from knowledge_ticket_agent.loaders import utc
from knowledge_ticket_agent.rules import evaluate_rules
from knowledge_ticket_agent.schemas import Category, Dataset, Status
from knowledge_ticket_agent.strict import FixtureError

from agent_harness_lab.permissions import ApprovalHandler, ApprovalRequest, Permission

from .m3_public_oracle import PUBLIC_M3_EXPECTATIONS, PublicFacts


def authored(data: Dataset, facts: PublicFacts) -> Dataset:
    """Map independently authored facts onto DEV D01; no Gold fixture involved."""
    ticket = next(t for t in data.tickets if t.ticket_id == "KT-D01")
    customer = next(c for c in data.customers if c.customer_id == ticket.customer_id)
    invoice = next(i for i in data.invoices if i.invoice_id == ticket.facts.invoice_id)
    customer = replace(
        customer,
        plan=facts.plan,
        role=facts.role,
        identity_verified=facts.identity_verified,
        security_hold=facts.security_hold,
    )
    invoice = replace(
        invoice,
        purchased_at=(utc(ticket.submitted_at) - timedelta(days=facts.age_days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        amount_minor=facts.amount_minor,
        first_purchase=facts.first_purchase,
        already_credited=facts.already_credited,
    )
    action = {
        Category.BILLING: "credit_guidance",
        Category.ACCESS: "access_guidance",
        Category.EXPORT: "export_guidance",
    }[facts.category]
    ticket = replace(
        ticket,
        facts=replace(
            ticket.facts,
            requested_action=action,
            exception_id=facts.exception_id,
            request_scope=facts.request_scope,
        ),
        body=facts.ticket_body,
        claimed_category=facts.claimed_category,
    )
    if not facts.relationship_valid:
        ticket = replace(ticket, tenant_id="SYN-TENANT-01")
    return replace(
        data,
        tickets=tuple(ticket if t.ticket_id == ticket.ticket_id else t for t in data.tickets),
        customers=tuple(
            customer if c.customer_id == customer.customer_id else c for c in data.customers
        ),
        invoices=tuple(invoice if i.invoice_id == invoice.invoice_id else i for i in data.invoices),
    )


def prepared(dataset: Dataset, *, ticket: str = "KT-D01") -> tuple[BusinessWorkflow, Proposal]:
    workflow = BusinessWorkflow(dataset, "public-run")
    category = evaluate_rules(dataset, ticket).category
    result = workflow.prepare(ticket, "SYN-STAFF-02", collect_evidence(dataset, category))
    assert result.proposal is not None, result
    return workflow, result.proposal


@pytest.mark.parametrize("case", PUBLIC_M3_EXPECTATIONS, ids=lambda c: c.case_id)
def test_independent_public_oracle(dataset: Dataset, case: object) -> None:
    from .m3_public_oracle import PublicExpectation

    assert isinstance(case, PublicExpectation)
    data = authored(dataset, case.facts)
    evaluation = evaluate_rules(data, "KT-D01", snapshot=case.facts.snapshot)
    if case.decision is not None:
        assert evaluation.decision is case.decision
    assert evaluation.destination is case.destination
    workflow = BusinessWorkflow(data, case.case_id)
    initial = workflow.store.snapshot
    try:
        evidence = collect_evidence(data, case.facts.category, case.facts.snapshot)
    except FixtureError:
        evidence = ()
    preparation = workflow.prepare("KT-D01", "SYN-STAFF-02", evidence, snapshot=case.facts.snapshot)
    if case.destination is None:
        assert preparation.proposal is None
        assert workflow.handler_invocations == 0
        assert workflow.store.snapshot == initial
    else:
        proposal = preparation.proposal
        assert proposal is not None
        result = workflow.execute(
            proposal,
            "call-1",
            approver=ScriptedApproval(approval_request(proposal, "call-1"), True),
        )
        assert grade_transition(
            initial,
            workflow.store.snapshot,
            result,
            workflow.audit.events,
            ticket_id="KT-D01",
            expected=case.destination,
        ).passed


@pytest.mark.parametrize(
    "kind",
    (
        "missing",
        "empty_hits",
        "version",
        "scope",
        "quote",
        "stale_index",
        "duplicate",
        "requester_subset",
    ),
)
def test_documentary_failures_never_propose(dataset: Dataset, kind: str) -> None:
    evidence = collect_evidence(dataset, Category.BILLING)
    first = evidence[0]
    if kind == "missing":
        evidence = evidence[:-1]
    elif kind == "empty_hits":
        evidence = (
            replace(first, result=replace(first.result, hits=(), status="no_match"), citations=()),
        ) + evidence[1:]
    elif kind == "version":
        evidence = (replace(first, requirement_version="different"),) + evidence[1:]
    elif kind == "scope":
        evidence = (replace(first, scope=replace(first.scope, snapshot="conflict")),) + evidence[1:]
    elif kind == "quote":
        evidence = (
            replace(
                first,
                citations=(replace(first.citations[0], quote="forged"),) + first.citations[1:],
            ),
        ) + evidence[1:]
    elif kind == "stale_index":
        evidence = (
            replace(first, result=replace(first.result, index_sha256="0" * 64)),
        ) + evidence[1:]
    elif kind == "duplicate":
        evidence = (evidence[1],) + evidence[1:]
    else:
        # A complete permissive-policy request alone cannot define requirements.
        evidence = evidence[:1]
    workflow = BusinessWorkflow(dataset, "evidence-failure")
    initial = workflow.store.snapshot
    result = workflow.prepare("KT-D01", "SYN-STAFF-02", evidence)
    assert result.proposal is None
    assert result.evidence_outcome in {Outcome.INSUFFICIENT, Outcome.NOT_ASSESSABLE}
    assert workflow.store.snapshot == initial and workflow.handler_invocations == 0


def test_conflict_blocks_without_choosing_winner(dataset: Dataset) -> None:
    evidence = collect_evidence(dataset, Category.EXPORT, "conflict")
    assert evidence_gate(dataset, Category.EXPORT, "conflict", evidence)[0] is Outcome.CONFLICT
    workflow = BusinessWorkflow(dataset, "conflict")
    assert (
        workflow.prepare("KT-D10", "SYN-STAFF-02", evidence, snapshot="conflict").proposal is None
    )
    assert workflow.handler_invocations == 0


def test_trusted_requirements_include_all_authority(dataset: Dataset) -> None:
    requirements = documentary_requirements(dataset, Category.BILLING)
    sections = {(c.doc_ref, c.section_id) for r in requirements for c in r.components}
    assert {
        ("POL-BILL-02@1.0", "restrictions"),
        ("EX-BILL-TEAM@1.0", "team_window"),
        ("POL-CASE-01@1.0", "states"),
        ("POL-ESCALATE-01@1.0", "routing"),
    } <= sections


@pytest.mark.parametrize(
    "kind",
    (
        "deny",
        "missing",
        "invalid",
        "exception",
        "no_write",
        "cross_call",
        "cross_ticket",
        "cross_tenant",
        "cross_action",
        "cross_revision",
    ),
)
def test_approval_failures_zero_handler_and_state(dataset: Dataset, kind: str) -> None:
    workflow, proposal = prepared(dataset)
    initial = workflow.store.snapshot
    request = approval_request(proposal, "call-1")
    approver: ApprovalHandler | None = ScriptedApproval(request, False)
    permissions = (Permission.WRITE,)
    if kind == "missing":
        approver = None
    elif kind == "invalid":
        approver = cast(ApprovalHandler, lambda _: "yes")
    elif kind == "exception":

        def throwing(request: ApprovalRequest) -> bool:
            raise RuntimeError("private-like-content-must-not-appear")

        approver = throwing
    elif kind == "no_write":
        permissions = (Permission.READ,)
        approver = ScriptedApproval(request, True)
    elif kind == "cross_call":
        approver = ScriptedApproval(replace(request, call_id="prior-call"), True)
    elif kind.startswith("cross_"):
        field, value = {
            "cross_ticket": ("ticket_id", "KT-D03"),
            "cross_tenant": ("tenant_id", "SYN-TENANT-01"),
            "cross_action": ("action", "awaiting_review"),
            "cross_revision": ("expected_revision", 1),
        }[kind]
        changed = dict(request.arguments)
        changed[field] = cast(str | int, value)
        approver = ScriptedApproval(replace(request, arguments=changed), True)
    result = workflow.execute(proposal, "call-1", approver=approver, permissions=permissions)
    assert not result.authorized and result.handler_invocations == 0
    assert workflow.handler_invocations == 0 and workflow.store.snapshot == initial
    assert grade_transition(
        initial,
        workflow.store.snapshot,
        result,
        workflow.audit.events,
        ticket_id=proposal.ticket_id,
        expected=None,
    ).passed
    assert "private-like-content" not in repr(workflow.audit.events)


@pytest.mark.parametrize(
    "field,value",
    (
        ("ticket_id", "KT-D03"),
        ("tenant_id", "SYN-TENANT-01"),
        ("actor_id", "SYN-STAFF-01"),
        ("action", "awaiting_review"),
        ("destination", Status.REVIEW),
        ("snapshot", "conflict"),
        ("evidence_identity", "0" * 64),
        ("expected_revision", 1),
        ("expected_revision", False),
        ("proposal_id", "forged"),
        ("run_id", "other"),
        ("idempotency_id", "other"),
        ("rule_version", "unknown"),
        ("required_permission", "read"),
    ),
)
def test_forged_proposal_has_no_authority(dataset: Dataset, field: str, value: object) -> None:
    workflow, proposal = prepared(dataset)
    initial = workflow.store.snapshot
    forged = replace(proposal, **cast(Any, {field: value}))
    result = workflow.execute(
        forged, "call-1", approver=ScriptedApproval(approval_request(forged, "call-1"), True)
    )
    assert result.code == "proposal_invalid"
    assert result.handler_invocations == 0 and workflow.store.snapshot == initial


def test_read_only_actor_cannot_prepare(dataset: Dataset) -> None:
    workflow = BusinessWorkflow(dataset, "actor")
    assert (
        workflow.prepare(
            "KT-D01", "SYN-STAFF-01", collect_evidence(dataset, Category.BILLING)
        ).proposal
        is None
    )
    assert workflow.handler_invocations == 0


@pytest.mark.parametrize(
    "phase", ("before_commit", "audit_prepared", "before_publish", "after_commit", "readback")
)
def test_fault_atomicity_and_truthful_postcommit_status(dataset: Dataset, phase: str) -> None:
    workflow, proposal = prepared(dataset)
    initial = workflow.store.snapshot

    def fault(at: str) -> None:
        if at == phase:
            raise RuntimeError("injected")

    result = workflow.execute(
        proposal,
        "call-1",
        approver=ScriptedApproval(approval_request(proposal, "call-1"), True),
        fault=fault,
    )
    assert result.authorized and result.handler_invocations == 1
    assert not result.readback_verified
    if phase in {"after_commit", "readback"}:
        assert result.committed and result.final_state is not None
        assert result.final_state.revision == 1 and len(workflow.store.snapshot.commits) == 1
    else:
        assert not result.committed and workflow.store.snapshot == initial


def test_approval_time_context_change_is_revalidated(dataset: Dataset) -> None:
    workflow, proposal = prepared(dataset)
    before = workflow.store.snapshot

    def change_context(request: ApprovalRequest) -> bool:
        workflow._data = replace(
            dataset, staff=tuple(replace(s, allowed_actions=("read",)) for s in dataset.staff)
        )
        return True

    result = workflow.execute(proposal, "call-1", approver=change_context)
    assert result.authorized and result.handler_invocations == 1
    assert result.code == "proposal_changed" and workflow.store.snapshot == before


def test_replay_needs_fresh_exact_approval_and_never_rewrites(dataset: Dataset) -> None:
    workflow, proposal = prepared(dataset)
    old = workflow.prepare(
        "KT-D01", "SYN-STAFF-02", collect_evidence(dataset, Category.BILLING)
    ).proposal
    assert old is not None
    approval = ScriptedApproval(approval_request(proposal, "call-1"), True)
    first = workflow.execute(proposal, "call-1", approver=approval)
    assert first.readback_verified
    assert first.mutated
    after = workflow.store.snapshot
    second = workflow.execute(proposal, "call-2", approver=approval)
    assert (
        not second.authorized
        and second.handler_invocations == 0
        and workflow.store.snapshot == after
    )
    replay = workflow.execute(
        proposal, "call-3", approver=ScriptedApproval(approval_request(proposal, "call-3"), True)
    )
    assert replay.authorized and replay.handler_invocations == 1 and replay.readback_verified
    assert not replay.mutated
    assert any(
        e.kind == "replay" and e.code == "existing_transition" for e in workflow.audit.events
    )
    assert workflow.store.snapshot is after
    stale = workflow.execute(
        old, "call-4", approver=ScriptedApproval(approval_request(old, "call-4"), True)
    )
    assert stale.code == "proposal_invalid" and stale.handler_invocations == 0
    assert workflow.store.snapshot == after and len(after.commits) == 1
    duplicate = workflow.execute(proposal, "call-3", approver=None)
    assert duplicate.code == "invocation_invalid" and workflow.store.snapshot == after


def test_postwrite_grader_rejects_prewrite_read_or_missing_audit(dataset: Dataset) -> None:
    workflow, proposal = prepared(dataset)
    before = workflow.store.snapshot
    result = workflow.execute(
        proposal, "call-1", approver=ScriptedApproval(approval_request(proposal, "call-1"), True)
    )
    for bad in (
        replace(result, final_state=before.tickets[0]),
        replace(result, readback_verified=False),
    ):
        assert not grade_transition(
            before,
            workflow.store.snapshot,
            bad,
            workflow.audit.events,
            ticket_id=proposal.ticket_id,
            expected=Status.RESOLVED,
        ).passed
    assert not grade_transition(
        before,
        workflow.store.snapshot,
        result,
        workflow.audit.events[:-1],
        ticket_id=proposal.ticket_id,
        expected=Status.RESOLVED,
    ).passed


def test_isolated_deterministic_runs_and_nested_execution(dataset: Dataset) -> None:
    first, proposal = prepared(dataset)
    second, equivalent = prepared(dataset)
    assert proposal == equivalent and first.store is not second.store
    nested_codes = []

    def approve(request: ApprovalRequest) -> bool:
        nested_codes.append(first.execute(proposal, "nested", approver=None).code)
        return ScriptedApproval(approval_request(proposal, "call-1"), True)(request)

    one = first.execute(proposal, "call-1", approver=approve)
    assert second.store.read("KT-D01").revision == 0
    two = second.execute(
        equivalent,
        "call-1",
        approver=ScriptedApproval(approval_request(equivalent, "call-1"), True),
    )
    assert one == two and first.store.snapshot == second.store.snapshot
    assert first.audit.events == second.audit.events and nested_codes == ["invocation_active"]


def test_grader_rejects_fabricated_success_without_mutation(dataset: Dataset) -> None:
    from knowledge_ticket_agent.approvals import WorkflowResult

    workflow, proposal = prepared(dataset)
    before = workflow.store.snapshot
    for forged in (
        WorkflowResult("completed", False, 1, False, True, before.tickets[0]),
        WorkflowResult("completed", True, 1, False, True, before.tickets[0]),
        WorkflowResult("permission_denied", False, 1, False, False, before.tickets[0]),
        WorkflowResult("permission_denied", False, 0, True, False, before.tickets[0]),
    ):
        assert not grade_transition(
            before, before, forged, (), ticket_id=proposal.ticket_id, expected=None
        ).passed


def test_exact_approval_rejects_bool_revision_expectation(dataset: Dataset) -> None:
    workflow, proposal = prepared(dataset)
    before = workflow.store.snapshot
    request = approval_request(proposal, "call-1")
    arguments = dict(request.arguments)
    arguments["expected_revision"] = False
    result = workflow.execute(
        proposal,
        "call-1",
        approver=ScriptedApproval(replace(request, arguments=arguments), True),
    )
    assert not result.authorized and result.handler_invocations == 0
    assert workflow.store.snapshot == before


def test_grader_checks_denial_audit_against_actual_result(dataset: Dataset) -> None:
    from knowledge_ticket_agent.audit import AuditEvent

    workflow, proposal = prepared(dataset)
    before = workflow.store.snapshot
    result = workflow.execute(proposal, "call-1", approver=None)
    assert not grade_transition(
        before, before, result, (), ticket_id=proposal.ticket_id, expected=None
    ).passed
    fabricated = workflow.audit.events + (
        AuditEvent(len(workflow.audit.events) + 1, "handler", "entered", proposal.ticket_id),
    )
    assert not grade_transition(
        before, before, result, fabricated, ticket_id=proposal.ticket_id, expected=None
    ).passed


def test_actual_stale_postwrite_read_cannot_claim_completion(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from knowledge_ticket_agent.ticket_store import TicketState

    workflow, proposal = prepared(dataset)
    before = workflow.store.snapshot
    original = workflow.store.read
    cached = original(proposal.ticket_id)

    def stale_read(ticket_id: str) -> TicketState:
        return cached if workflow.store.snapshot.commits else original(ticket_id)

    monkeypatch.setattr(workflow.store, "read", stale_read)
    result = workflow.execute(
        proposal,
        "call-1",
        approver=ScriptedApproval(approval_request(proposal, "call-1"), True),
    )
    assert result.code == "readback_mismatch" and not result.readback_verified
    assert result.committed and result.mutated and result.final_state is not None
    assert result.final_state.revision == 1 and result.final_state.status is Status.RESOLVED
    assert not grade_transition(
        before,
        workflow.store.snapshot,
        result,
        workflow.audit.events,
        ticket_id=proposal.ticket_id,
        expected=Status.RESOLVED,
    ).passed
