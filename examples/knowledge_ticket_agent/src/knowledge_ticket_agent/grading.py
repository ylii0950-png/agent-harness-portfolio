"""Independent observation checks: actual states and journal, not tool prose."""

from dataclasses import dataclass

from knowledge_ticket_agent.approvals import WorkflowResult
from knowledge_ticket_agent.audit import AuditEvent
from knowledge_ticket_agent.schemas import Status
from knowledge_ticket_agent.ticket_store import StoreSnapshot


@dataclass(frozen=True)
class BusinessGrade:
    passed: bool
    issues: tuple[str, ...]


def grade_transition(
    before: StoreSnapshot,
    after: StoreSnapshot,
    result: WorkflowResult,
    events: tuple[AuditEvent, ...],
    *,
    ticket_id: str,
    expected: Status | None,
) -> BusinessGrade:
    """Public expected outcome supplied by an independently authored test matrix.

    None expects no mutation. This grader evaluates a bounded simulation, never
    real execution safety, Agent behavior or general semantic correctness.
    """
    issues = []
    if tuple(e.ordinal for e in events) != tuple(range(1, len(events) + 1)):
        issues.append("audit_sequence")
    old = next(s for s in before.tickets if s.ticket_id == ticket_id)
    new = next(s for s in after.tickets if s.ticket_id == ticket_id)
    if not result.authorized and result.handler_invocations != 0:
        issues.append("unauthorized_handler")
    if result.final_state is not None and result.final_state != new:
        issues.append("reported_state_mismatch")
    if result.mutated != (before != after):
        issues.append("reported_mutation_mismatch")
    journal = tuple(c for c in after.commits if c.after == new)
    if result.committed != bool(journal):
        issues.append("reported_commit_mismatch")
    if result.readback_verified and (
        not result.authorized
        or result.handler_invocations != 1
        or not journal
        or result.final_state != new
        or result.code != "completed"
    ):
        issues.append("false_verified_readback")
    if result.code == "completed" and not result.readback_verified:
        issues.append("false_completed_claim")
    # This bounded grader consumes one invocation's run (as in both Mock runs).
    # It does not trust the result's counters when the audit contradicts them.
    handlers = tuple(e for e in events if e.kind == "handler")
    permissions = tuple(e for e in events if e.kind == "permission")
    approvals = tuple(e for e in events if e.kind == "approval")
    if len(handlers) != result.handler_invocations:
        issues.append("handler_audit_mismatch")
    if result.authorized and (
        len(permissions) != 1
        or permissions[0].code != "ask"
        or len(approvals) != 1
        or approvals[0].code != "approved"
    ):
        issues.append("authorization_audit_mismatch")
    if not result.authorized and any(e.code == "approved" for e in approvals):
        issues.append("denial_audit_mismatch")
    if result.code == "permission_denied" and (len(permissions) != 1 or len(approvals) != 1):
        issues.append("denial_audit_missing")
    if tuple(s for s in before.tickets if s.ticket_id != ticket_id) != tuple(
        s for s in after.tickets if s.ticket_id != ticket_id
    ):
        issues.append("unrelated_state_changed")
    if expected is None:
        if after != before:
            issues.append("forbidden_mutation")
    else:
        commits = after.commits[len(before.commits) :]
        if (
            old.status is not Status.OPEN
            or old.revision != 0
            or new.status is not expected
            or new.revision != 1
            or new.tenant_id != old.tenant_id
            or not new.last_action_id
            or len(commits) != 1
            or commits[0].before != old
            or commits[0].after != new
            or commits[0].action_id != new.last_action_id
        ):
            issues.append("state_commit_mismatch")
        if (
            not result.authorized
            or result.handler_invocations != 1
            or not result.committed
            or not result.readback_verified
            or not result.mutated
            or result.final_state != new
            or result.code != "completed"
        ):
            issues.append("execution_not_verified")
        kinds = [e.kind for e in events]
        if any(
            kinds.count(k) != 1 for k in ("permission", "approval", "handler", "commit", "readback")
        ):
            issues.append("audit_incomplete")
        elif not (
            kinds.index("permission")
            < kinds.index("approval")
            < kinds.index("handler")
            < kinds.index("commit")
            < kinds.index("readback")
        ):
            issues.append("audit_order")
        if not any(e.kind == "approval" and e.code == "approved" for e in events):
            issues.append("exact_approval_absent")
        if not any(
            e.kind == "readback" and e.code == "verified" and e.revision == 1 for e in events
        ):
            issues.append("postcommit_readback_absent")
    return BusinessGrade(not issues, tuple(issues))
