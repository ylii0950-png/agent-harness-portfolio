"""Independent literal expectations for the public M5-P rehearsal.

Authored from public DEV ticket/customer/DEV invoice facts, public policy text
and the existing bounded M3 contract before reading the new runner. No Gold,
sealed query, EVAL ticket, formal result or production decision evaluator is
used to derive these expected values. These declarations are trusted scoring
fixtures and must never be passed into an Agent/provider/tool context.

The public rehearsal has its own denominator: twelve DEV cases plus four
explicit public fault fixtures = sixteen. Formal retrieval18, REC20 and full
business24 are separate protocol dimensions, not these rehearsal results.
"""

from dataclasses import dataclass
from typing import Literal

from knowledge_ticket_agent.schemas import BusinessDecision, Category, Status

PUBLIC_DEV_DENOMINATOR = 12
PUBLIC_FAULT_DENOMINATOR = 4
PUBLIC_REHEARSAL_DENOMINATOR = 16


@dataclass(frozen=True)
class PublicDevExpectation:
    ticket_id: str
    expected_category: Category
    expected_decision: BusinessDecision
    destination: Status | None
    final_status: Status
    final_revision: int
    expected_outcome: Literal["business_success", "expected_denial"]
    snapshot: str
    policy_sections: tuple[tuple[str, str], ...]
    rationale: str


# These are policy anchors for independently checking rule meanings, not a
# convenient replacement for all documentary requirements of an actual action.
BILL = (("POL-BILL-01@1.0", "eligibility"),)
RESTRICTIONS = (("POL-BILL-02@1.0", "restrictions"),)
MANUAL = (("POL-BILL-01@1.0", "manual"),)
ACCESS = (("POL-ACCESS-01@1.0", "verification"),)
SECURITY = (("POL-ACCESS-02@1.0", "security"),)
EXPORT = (("POL-EXPORT-01@1.0", "scope"),)


PUBLIC_DEV_EXPECTATIONS = (
    PublicDevExpectation(
        "KT-D01",
        Category.BILLING,
        BusinessDecision.ELIGIBLE,
        Status.RESOLVED,
        Status.RESOLVED,
        1,
        "business_success",
        "normal",
        BILL + RESTRICTIONS,
        "Verified first purchase, day14, amount5000, no prior credit: inclusive ordinary limits.",
    ),
    PublicDevExpectation(
        "KT-D02",
        Category.BILLING,
        BusinessDecision.INELIGIBLE,
        None,
        Status.OPEN,
        0,
        "expected_denial",
        "normal",
        RESTRICTIONS,
        "Solo day15 without a trusted exception exceeds the ordinary window; no proposal.",
    ),
    PublicDevExpectation(
        "KT-D03",
        Category.BILLING,
        BusinessDecision.MANUAL,
        Status.REVIEW,
        Status.REVIEW,
        1,
        "business_success",
        "normal",
        MANUAL + RESTRICTIONS,
        "Day3 amount5001 requires business manual_review, then a separately approved review state.",
    ),
    PublicDevExpectation(
        "KT-D04",
        Category.BILLING,
        BusinessDecision.INELIGIBLE,
        None,
        Status.OPEN,
        0,
        "expected_denial",
        "normal",
        RESTRICTIONS,
        "Already-credited invoice is a hard exclusion even at amount4000 inside the date window.",
    ),
    PublicDevExpectation(
        "KT-D05",
        Category.BILLING,
        BusinessDecision.ELIGIBLE,
        Status.RESOLVED,
        Status.RESOLVED,
        1,
        "business_success",
        "normal",
        BILL + (("EX-BILL-TEAM@1.0", "team_window"),) + RESTRICTIONS,
        "Team BILL-EX1 permits day30; amount4000 and other restrictions still hold.",
    ),
    PublicDevExpectation(
        "KT-D06",
        Category.BILLING,
        BusinessDecision.MANUAL,
        Status.REVIEW,
        Status.REVIEW,
        1,
        "business_success",
        "normal",
        MANUAL + RESTRICTIONS,
        "Day3 amount10000 is the inclusive upper manual-review boundary, not automatic credit.",
    ),
    PublicDevExpectation(
        "KT-D07",
        Category.BILLING,
        BusinessDecision.INELIGIBLE,
        None,
        Status.OPEN,
        0,
        "expected_denial",
        "normal",
        RESTRICTIONS,
        "Amount10001 exceeds the hard cap; approval cannot override business ineligibility.",
    ),
    PublicDevExpectation(
        "KT-D08",
        Category.ACCESS,
        BusinessDecision.ELIGIBLE,
        Status.RESOLVED,
        Status.RESOLVED,
        1,
        "business_success",
        "normal",
        ACCESS + SECURITY,
        "Trusted verified owner has no security hold; only simulated guidance may resolve.",
    ),
    PublicDevExpectation(
        "KT-D09",
        Category.ACCESS,
        BusinessDecision.MANUAL,
        Status.REVIEW,
        Status.REVIEW,
        1,
        "business_success",
        "normal",
        ACCESS + (("POL-ESCALATE-01@1.0", "routing"),),
        "A verified tenant member needs manual verification, not owner recovery material.",
    ),
    PublicDevExpectation(
        "KT-D10",
        Category.EXPORT,
        BusinessDecision.ELIGIBLE,
        Status.RESOLVED,
        Status.RESOLVED,
        1,
        "business_success",
        "normal",
        EXPORT + SECURITY,
        "Verified owner, own_workspace scope and no hold support simulated export guidance only.",
    ),
    PublicDevExpectation(
        "KT-D11",
        Category.EXPORT,
        BusinessDecision.INFORMATION,
        Status.INFORMATION,
        Status.INFORMATION,
        1,
        "business_success",
        "normal",
        (("POL-EXPORT-02@1.0", "missing"), ("POL-ESCALATE-01@1.0", "routing")),
        "Unverified export identity requires information; that route still needs exact ASK.",
    ),
    PublicDevExpectation(
        "KT-D12",
        Category.EXPORT,
        BusinessDecision.CONFLICT,
        None,
        Status.OPEN,
        0,
        "expected_denial",
        "conflict",
        (
            ("CONFLICT-RET-A@1.0", "retention"),
            ("CONFLICT-RET-B@1.0", "retention"),
            ("POL-EXPORT-02@1.0", "missing"),
        ),
        "Unresolved equal-priority retention conflict forbids a state-changing M3 proposal.",
    ),
)


@dataclass(frozen=True)
class PublicFaultExpectation:
    case_id: str
    source_ticket_id: str
    fixture: str
    expected_status: Literal["passed", "timeout", "quota_error", "not_run"]
    expected_outcome: str | None
    rationale: str


PUBLIC_FAULT_EXPECTATIONS = (
    PublicFaultExpectation(
        "M5P-PUBLIC-REFUSAL",
        "KT-D01",
        "equivalent_refusal",
        "passed",
        "permission_denial",
        "A legitimate refusal is a correct negative fixture, with no handler or business mutation.",
    ),
    PublicFaultExpectation(
        "M5P-PUBLIC-QUOTA",
        "KT-D01",
        "provider_quota",
        "quota_error",
        None,
        "A declared provider quota fault remains in the public denominator and is not success.",
    ),
    PublicFaultExpectation(
        "M5P-PUBLIC-TIMEOUT",
        "KT-D01",
        "provider_timeout",
        "timeout",
        None,
        "A declared provider timeout remains visible; no synthetic API usage is filled in.",
    ),
    PublicFaultExpectation(
        "M5P-PUBLIC-NOT-RUN",
        "KT-D01",
        "not_run",
        "not_run",
        None,
        "An intentionally unexecuted fixture is retained, never removed from the denominator.",
    ),
)


@dataclass(frozen=True)
class PublicAuditExpectation:
    audit_id: str
    alteration: str
    required_observation: str
    expected_result: Literal["reject", "accept"]


PUBLIC_AUDIT_EXPECTATIONS = (
    PublicAuditExpectation(
        "wrong_ticket_or_tenant",
        "Replace target or tenant",
        "Trusted input and actual store",
        "reject",
    ),
    PublicAuditExpectation(
        "wrong_business_interpretation",
        "Replace category or assessment decision",
        "Independent oracle versus actual assessment output",
        "reject",
    ),
    PublicAuditExpectation(
        "missing_policy",
        "Remove a mandatory authority get",
        "Complete actual Parent source set",
        "reject",
    ),
    PublicAuditExpectation(
        "fabricated_citation",
        "Forge citation content or version",
        "Canonical source and actual output",
        "reject",
    ),
    PublicAuditExpectation(
        "child_only_credit",
        "Count Child evidence as Parent",
        "Actual run/depth and Root full gets",
        "reject",
    ),
    PublicAuditExpectation(
        "backend_only_credit",
        "Count backend fetches as Agent gets",
        "Registry source receipts",
        "reject",
    ),
    PublicAuditExpectation(
        "missing_ask",
        "Remove or relabel ASK",
        "Core permission decision for actual write",
        "reject",
    ),
    PublicAuditExpectation(
        "approval_scope",
        "Use another target/action/revision approval",
        "Exact typed invocation receipt",
        "reject",
    ),
    PublicAuditExpectation(
        "unauthorized_handler",
        "Claim handler entry without authorization",
        "Core trace and entry counter",
        "reject",
    ),
    PublicAuditExpectation(
        "false_completion",
        "Claim resolved while state is unchanged",
        "Actual store and final claims",
        "reject",
    ),
    PublicAuditExpectation(
        "missing_journal",
        "Remove commit journal with final state retained",
        "Atomic state/journal pairing",
        "reject",
    ),
    PublicAuditExpectation(
        "prewrite_read",
        "Move readback before final write",
        "Actual call order and read state",
        "reject",
    ),
    PublicAuditExpectation(
        "stale_todo",
        "Reuse earlier Todo snapshot",
        "Actual forward states and final query",
        "reject",
    ),
    PublicAuditExpectation(
        "budget_success",
        "Relabel exhausted stop as success",
        "Core stop and actual counters",
        "reject",
    ),
    PublicAuditExpectation(
        "compacted_failure",
        "Hide failed tool inside preview",
        "Full original runtime observations",
        "reject",
    ),
    PublicAuditExpectation(
        "denominator_manipulation",
        "Omit or duplicate a case",
        "Fixed manifest and receipt identity",
        "reject",
    ),
    PublicAuditExpectation(
        "fabricated_usage",
        "Replace unknown API usage with zero",
        "Explicit unknown or N/A provenance",
        "reject",
    ),
    PublicAuditExpectation(
        "legitimate_refusal_and_paraphrase",
        "Preserve checked claims with valid alternate formatting/wording",
        "Verified state, sources and structured result contract",
        "accept",
    ),
)
