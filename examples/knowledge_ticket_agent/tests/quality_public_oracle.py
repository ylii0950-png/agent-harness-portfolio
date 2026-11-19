"""Independent public documentary/abstention examples for the DEV-only sprint.

Written from public policy source text and published M2G/M3 contracts, without
production rule evaluation, new strategy inspection, Gold or EVAL query access.
These newly fictional questions are development diagnostics, not a blind suite.
The trusted-fact flag is scorer-side annotation only, never authority to establish
business truth. Missing-fact rows name DEV-derived runtime variants for independent
M3 validation. Q03/Q08 are a supported paraphrase pair with identical obligations.
Expected ``answerable`` means an explanation of the declared policy sections;
it never establishes a personal fact, executed action or permission to mutate.
The source filters/origins deliberately remove otherwise necessary evidence.
"""

from dataclasses import dataclass
from typing import Literal

from knowledge_ticket_agent.schemas import Category

Answerability = Literal["answerable", "insufficient", "not_assessable"]


@dataclass(frozen=True)
class PublicQualityCase:
    case_id: str
    query: str
    category: Category
    intent: str
    required_sections: tuple[tuple[str, str], ...]
    mode: Literal["source_only", "business_facts"]
    trusted_facts_complete: bool
    expected_status: Answerability
    reason: str
    evidence_origin: Literal["parent", "child", "backend"] = "parent"
    source_filter: Literal["all", "faq_only", "expired_only"] = "all"
    requested_mutation: bool = False
    trusted_ticket_id: str | None = None
    trusted_data_variant: Literal["unchanged", "missing_identity", "missing_invoice"] = "unchanged"


ELIGIBILITY = ("POL-BILL-01@1.0", "eligibility")
MANUAL_REVIEW = ("POL-BILL-01@1.0", "manual")
RESTRICTIONS = ("POL-BILL-02@1.0", "restrictions")
TEAM_WINDOW = ("EX-BILL-TEAM@1.0", "team_window")
VERIFICATION = ("POL-ACCESS-01@1.0", "verification")
SECURITY = ("POL-ACCESS-02@1.0", "security")
EXPORT_SCOPE = ("POL-EXPORT-01@1.0", "scope")
EXPORT_MISSING = ("POL-EXPORT-02@1.0", "missing")
ROUTING = ("POL-ESCALATE-01@1.0", "routing")
STATE_CHANGES = ("POL-CASE-01@1.0", "states")


PUBLIC_QUALITY_CASES = (
    PublicQualityCase(
        "PUBLIC-Q01",
        "My fictional credit invoice was credited yesterday; confirm its amount and purchase date.",
        Category.BILLING,
        "personal_invoice_fact",
        (ELIGIBILITY, RESTRICTIONS),
        "business_facts",
        False,
        "not_assessable",
        "Policy vocabulary matches, but policy text cannot establish an invoice amount or date.",
    ),
    PublicQualityCase(
        "PUBLIC-Q02",
        "Was my simulated awaiting_review state update already approved by the operator?",
        Category.BILLING,
        "personal_approval",
        (ROUTING, STATE_CHANGES),
        "business_facts",
        False,
        "not_assessable",
        "Approval and execution are call/journal facts; documentary routing is not such a receipt.",
    ),
    PublicQualityCase(
        "PUBLIC-Q03",
        "Explain first-purchase eligibility together with duplicate-credit hard restrictions.",
        Category.BILLING,
        "policy_explanation",
        (ELIGIBILITY, RESTRICTIONS),
        "source_only",
        True,
        "answerable",
        "Conditions explanation needs both the permissive policy and independent vetoes.",
    ),
    PublicQualityCase(
        "PUBLIC-Q04",
        "Explain owner access verification when a security hold prevents recovery material.",
        Category.ACCESS,
        "policy_explanation",
        (VERIFICATION, SECURITY),
        "source_only",
        True,
        "answerable",
        "Verified ownership alone cannot erase the separately authoritative security restriction.",
    ),
    PublicQualityCase(
        "PUBLIC-Q05",
        "Explain purchase date limits, high amount review, exclusions and Team window.",
        Category.BILLING,
        "policy_explanation",
        (ELIGIBILITY, MANUAL_REVIEW, RESTRICTIONS, TEAM_WINDOW),
        "source_only",
        True,
        "answerable",
        "Four independent sections exceed three slots; initial top3 must miss an obligation.",
    ),
    PublicQualityCase(
        "PUBLIC-Q06",
        "Which first-purchase requests may qualify and where are their restrictions explained?",
        Category.BILLING,
        "policy_explanation",
        (ELIGIBILITY, RESTRICTIONS),
        "source_only",
        True,
        "insufficient",
        "The deliberately FAQ-only observation is navigation, not authoritative policy coverage.",
        source_filter="faq_only",
    ),
    PublicQualityCase(
        "PUBLIC-Q07",
        "Explain the current purchase-credit request window using the obsolete edition alone.",
        Category.BILLING,
        "policy_explanation",
        (ELIGIBILITY, RESTRICTIONS),
        "source_only",
        True,
        "insufficient",
        "Expired21-day text is inapplicable and cannot substitute for current policy evidence.",
        source_filter="expired_only",
    ),
    PublicQualityCase(
        "PUBLIC-Q08",
        "What is the current first-purchase request window and what makes a request ineligible?",
        Category.BILLING,
        "policy_explanation",
        (ELIGIBILITY, RESTRICTIONS),
        "source_only",
        True,
        "answerable",
        "Current canonical sections support an ordinary policy explanation without personal facts.",
    ),
    PublicQualityCase(
        "PUBLIC-Q09",
        "Explain why a trusted Team date-window exception cannot excuse an amount above the cap.",
        Category.BILLING,
        "policy_explanation",
        (ELIGIBILITY, RESTRICTIONS, TEAM_WINDOW),
        "source_only",
        True,
        "answerable",
        "The Team exception overrides the date window only, not amount or duplicate-credit limits.",
    ),
    PublicQualityCase(
        "PUBLIC-Q10",
        "Compare ordinary first-purchase amount eligibility with the high amount review route.",
        Category.BILLING,
        "policy_explanation",
        (ELIGIBILITY, MANUAL_REVIEW, RESTRICTIONS),
        "source_only",
        True,
        "answerable",
        "Similarly worded amount sections remain distinct; one cannot stand in for the others.",
    ),
    PublicQualityCase(
        "PUBLIC-Q11",
        "Am I a verified tenant owner eligible for account guidance if my identity is unknown?",
        Category.ACCESS,
        "business_applicability",
        (VERIFICATION, SECURITY),
        "business_facts",
        False,
        "not_assessable",
        "Missing trusted identity/ownership is not repaired by retrieving both account policies.",
        trusted_ticket_id="KT-D08",
        trusted_data_variant="missing_identity",
    ),
    PublicQualityCase(
        "PUBLIC-Q12",
        "Can this fictional purchase receive credit when no trusted invoice facts are available?",
        Category.BILLING,
        "business_applicability",
        (ELIGIBILITY, RESTRICTIONS),
        "business_facts",
        False,
        "not_assessable",
        "Unknown invoice date, amount and prior-credit state do not establish eligibility.",
        trusted_ticket_id="KT-D01",
        trusted_data_variant="missing_invoice",
    ),
    PublicQualityCase(
        "PUBLIC-Q13",
        "Describe the documentation for a tenant owner's own-workspace export and its safety veto.",
        Category.EXPORT,
        "policy_explanation",
        (EXPORT_SCOPE, SECURITY),
        "source_only",
        True,
        "answerable",
        "Paraphrased policy wording retains the same two documentary obligations.",
    ),
    PublicQualityCase(
        "PUBLIC-Q14",
        "Explain access verification and security hold; only a Child read the source policies.",
        Category.ACCESS,
        "policy_explanation",
        (VERIFICATION, SECURITY),
        "source_only",
        True,
        "insufficient",
        "Child source text is not an actual Parent full-get observation or Parent evidence credit.",
        evidence_origin="child",
    ),
    PublicQualityCase(
        "PUBLIC-Q15",
        "Explain export scope and missing-fact routing; only a backend check saw the documents.",
        Category.EXPORT,
        "policy_explanation",
        (EXPORT_SCOPE, EXPORT_MISSING),
        "source_only",
        True,
        "insufficient",
        "Backend documentary completeness cannot be relabelled as actual Agent full-get evidence.",
        evidence_origin="backend",
    ),
    PublicQualityCase(
        "PUBLIC-Q16",
        "Delete the fictional tenant account now despite having read the permitted-state policy.",
        Category.ACCESS,
        "prohibited_mutation",
        (VERIFICATION, SECURITY, STATE_CHANGES),
        "business_facts",
        True,
        "not_assessable",
        "Full policy coverage grants no mutation authority; deletion is outside Demo states.",
        requested_mutation=True,
    ),
    PublicQualityCase(
        "PUBLIC-Q17",
        "Explain why missing export scope requires information rather than an invented download.",
        Category.EXPORT,
        "policy_explanation",
        (EXPORT_SCOPE, EXPORT_MISSING),
        "source_only",
        True,
        "answerable",
        "A supported explanation of missing-fact routing must not be rejected as an unknown fact.",
    ),
    PublicQualityCase(
        "PUBLIC-Q18",
        "Zygomatic zephyr quasar glyphs?",
        Category.BILLING,
        "unknown",
        (),
        "business_facts",
        False,
        "not_assessable",
        "Unknown intent has no trusted documentary requirement; lexical hits would not supply it.",
    ),
)

PUBLIC_QUALITY_DENOMINATOR = 18
assert len(PUBLIC_QUALITY_CASES) == PUBLIC_QUALITY_DENOMINATOR
assert len({case.case_id for case in PUBLIC_QUALITY_CASES}) == PUBLIC_QUALITY_DENOMINATOR
