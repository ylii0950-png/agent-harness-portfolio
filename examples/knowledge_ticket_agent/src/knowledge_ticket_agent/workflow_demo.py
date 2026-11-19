"""Two isolated offline Mock scenarios; not an Agent evaluation."""

import json
from dataclasses import asdict

from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.approvals import (
    BusinessWorkflow,
    ScriptedApproval,
    approval_request,
    collect_evidence,
)
from knowledge_ticket_agent.grading import grade_transition
from knowledge_ticket_agent.schemas import Category, Status


def main() -> None:
    for label, allow in (("approved", True), ("refused", False)):
        data = load_dataset()
        workflow = BusinessWorkflow(data, f"public-mock-{label}")
        before = workflow.store.snapshot
        preparation = workflow.prepare(
            "KT-D01", "SYN-STAFF-02", collect_evidence(data, Category.BILLING)
        )
        proposal = preparation.proposal
        if proposal is None:
            raise RuntimeError("public_mock_preparation_blocked")
        result = workflow.execute(
            proposal,
            "call-1",
            approver=ScriptedApproval(approval_request(proposal, "call-1"), allow),
        )
        grade = grade_transition(
            before,
            workflow.store.snapshot,
            result,
            workflow.audit.events,
            ticket_id=proposal.ticket_id,
            expected=Status.RESOLVED if allow else None,
        )
        print(
            json.dumps(
                {"scenario": label, "result": asdict(result), "grade": asdict(grade)},
                sort_keys=True,
            )
        )
        if not grade.passed:
            raise RuntimeError("public_mock_grade_failed")


if __name__ == "__main__":
    main()
