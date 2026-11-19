"""Finite fictional business predicates; no DSL, Gold, IO or permission grants.

Only public DEV tickets may be assessed by this M3 implementation. Documentary
completeness and current state/authorization are separate workflow gates.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TypeVar

from knowledge_ticket_agent.evaluation_scope import ArtificialCaseScope, validate_scope
from knowledge_ticket_agent.formal_scope import SealedScope
from knowledge_ticket_agent.loaders import markdown, utc
from knowledge_ticket_agent.retrieval import catalog
from knowledge_ticket_agent.schemas import (
    BusinessDecision,
    Category,
    Customer,
    Dataset,
    Invoice,
    Plan,
    Role,
    Split,
    Staff,
    Status,
    Ticket,
)
from knowledge_ticket_agent.strict import FixtureError, decode, require
from knowledge_ticket_agent.validation import validate_dataset

RULE_VERSION = "lqd-rules-1.0.0"
# Canonical JSON digests of the existing PUBLIC M1 manifest and declarations.
# Facts are deliberately not pinned: independent DEV fixtures may vary them.
_MANIFEST_CONTRACT = "bef8addc6fa10d9ac45014b6e26a16cb1cfe8fa1280ef03cb6633c46c41a2380"
_RULE_CONTRACT = "83421f0e1fe045057f8e32bf2ee65b0e992e465bffb9cb3fdc3b87c47f5d78cc"
_MANIFEST_SOURCE = "52a13a1b0b0eae835fde7f27612b14dea57d8ecbf457aa21dd49d26127d856c9"
_ACTION_CATEGORIES = {
    "credit_guidance": Category.BILLING,
    "access_guidance": Category.ACCESS,
    "export_guidance": Category.EXPORT,
}
T = TypeVar("T", Customer, Invoice, Staff, Ticket)


@dataclass(frozen=True)
class RuleEvaluation:
    category: Category
    decision: BusinessDecision
    reasons: tuple[str, ...]
    applied_rules: tuple[str, ...]
    destination: Status | None
    snapshot: str
    rule_version: str = RULE_VERSION


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _strict(model: type[T], value: T) -> None:
    """The dataclass constructors alone do not enforce bool/int/enum types."""
    require(type(value) is model, "business_record_type")
    decode(model, json.loads(json.dumps(asdict(value))))


def validate_rule_contract(data: Dataset) -> None:
    """Bind the finite predicates to exact public policy inputs, not rule text execution.

    Existing M1 structural validation remains authoritative. Reparse documents
    rather than trusting a caller-supplied Section body detached from its source.
    This function reads only the supplied in-memory Dataset.
    """
    require(type(data) is Dataset, "business_dataset_type")
    require(data.manifest_sha256 == _MANIFEST_SOURCE, "business_manifest_binding")
    require(_digest(asdict(data.manifest)) == _MANIFEST_CONTRACT, "business_manifest_version")
    require(data.manifest.ruleset_version == RULE_VERSION, "business_rule_version")
    require(_digest([asdict(rule) for rule in data.rules]) == _RULE_CONTRACT, "business_rules")
    require(len(data.documents) == 20, "business_documents")
    resources = {resource.path: resource for resource in data.manifest.resources}
    for policy in data.documents:
        resource = resources.get(policy.metadata.path)
        require(resource is not None, "business_source_binding")
        assert resource is not None
        encoded = policy.text.encode()
        require(
            len(encoded) == resource.bytes
            and hashlib.sha256(encoded).hexdigest() == resource.sha256,
            "business_source_binding",
        )
        require(markdown(policy.metadata, policy.text) == policy, "business_section_binding")
    validate_dataset(data)
    # All staff records feed the later authorization gate, including read-only staff.
    for actor in data.staff:
        _strict(Staff, actor)


def _result(
    category: Category,
    decision: BusinessDecision,
    reason: str,
    rules: tuple[str, ...],
    destination: Status | None,
    snapshot: str,
) -> RuleEvaluation:
    return RuleEvaluation(
        category,
        decision,
        (reason,),
        rules + (("CASE-01",) if destination is not None else ()),
        destination,
        snapshot,
    )


def _billing(data: Dataset, ticket: Ticket, customer: Customer, snapshot: str) -> RuleEvaluation:
    category = Category.BILLING
    require(ticket.facts.invoice_id is not None, "business_missing_invoice")
    matches = tuple(i for i in data.invoices if i.invoice_id == ticket.facts.invoice_id)
    require(len(matches) == 1, "business_invoice_relationship")
    invoice = matches[0]
    _strict(Invoice, invoice)
    require(
        invoice.customer_id == ticket.customer_id
        and invoice.tenant_id == ticket.tenant_id
        and invoice.unit == "DEMO_CREDIT",
        "business_invoice_relationship",
    )
    purchased, submitted = utc(invoice.purchased_at), utc(ticket.submitted_at)
    require(purchased <= submitted, "business_future_purchase")
    age = (submitted.date() - purchased.date()).days
    window = 14
    applied: tuple[str, ...] = ()
    if ticket.facts.exception_id is not None:
        require(
            ticket.facts.exception_id == "BILL-EX1" and customer.plan is Plan.TEAM,
            "business_exception_scope",
        )
        window = 30
        applied = ("BILL-EX1",)
    if (
        invoice.amount_minor > 10000
        or invoice.already_credited
        or not invoice.first_purchase
        or not 0 <= age <= window
    ):
        return _result(
            category,
            BusinessDecision.INELIGIBLE,
            "BILLING_HARD_RESTRICTION",
            applied + ("BILL-03",),
            None,
            snapshot,
        )
    if not customer.identity_verified:
        # A known False verification flag is not an eligible credit request.
        # M1 does not clearly specify its executable billing route: abstain.
        return _result(
            category,
            BusinessDecision.INFORMATION,
            "BILLING_IDENTITY_UNVERIFIED",
            applied + ("BILL-01",),
            None,
            snapshot,
        )
    if invoice.amount_minor >= 5001:
        return _result(
            category,
            BusinessDecision.MANUAL,
            "BILLING_AMOUNT_REVIEW",
            applied + ("BILL-02",),
            Status.REVIEW,
            snapshot,
        )
    return _result(
        category,
        BusinessDecision.ELIGIBLE,
        "BILLING_ELIGIBLE",
        applied + ("BILL-01",),
        Status.RESOLVED,
        snapshot,
    )


def _access_export(
    ticket: Ticket, customer: Customer, category: Category, snapshot: str
) -> RuleEvaluation:
    require(ticket.facts.exception_id is None, "business_exception_scope")
    if customer.security_hold:
        return _result(
            category,
            BusinessDecision.ESCALATE,
            "SECURITY_HOLD",
            ("ACCESS-02",),
            Status.REVIEW,
            snapshot,
        )
    if category is Category.ACCESS:
        if not customer.identity_verified or customer.role is not Role.OWNER:
            return _result(
                category,
                BusinessDecision.MANUAL,
                "ACCESS_MANUAL_VERIFICATION",
                ("ACCESS-01",),
                Status.REVIEW,
                snapshot,
            )
        return _result(
            category,
            BusinessDecision.ELIGIBLE,
            "ACCESS_ELIGIBLE",
            ("ACCESS-01",),
            Status.RESOLVED,
            snapshot,
        )
    if (
        not customer.identity_verified
        or customer.role is not Role.OWNER
        or ticket.facts.request_scope != "own_workspace"
    ):
        return _result(
            category,
            BusinessDecision.INFORMATION,
            "EXPORT_MISSING_REQUIREMENTS",
            ("EXPORT-02",),
            Status.INFORMATION,
            snapshot,
        )
    return _result(
        category,
        BusinessDecision.ELIGIBLE,
        "EXPORT_ELIGIBLE",
        ("EXPORT-01",),
        Status.RESOLVED,
        snapshot,
    )


def evaluate_rules(
    data: Dataset,
    ticket_id: str,
    *,
    snapshot: str = "normal",
    case_scope: ArtificialCaseScope | SealedScope | None = None,
) -> RuleEvaluation:
    """Assess trusted loaded facts; never derive privilege from subject/body/category.

    A recommendation is not an approval or proof of documentary completeness.
    Any unknown/invalid input returns no executable destination. Only the
    prospective M3 workflow may bind an assessment to an immutable proposal.
    """
    category = Category.OTHER
    try:
        require(type(ticket_id) is str and 1 <= len(ticket_id) <= 64, "business_ticket_id")
        require(type(snapshot) is str and snapshot in {"normal", "conflict"}, "business_snapshot")
        validate_rule_contract(data)
        matches = tuple(t for t in data.tickets if t.ticket_id == ticket_id)
        require(len(matches) == 1, "business_ticket_missing")
        ticket = matches[0]
        _strict(Ticket, ticket)
        require(
            ticket.split is Split.DEV
            if case_scope is None
            else ticket.split is Split.EVAL and validate_scope(data, ticket_id, case_scope),
            "business_dev_only",
        )
        requested = ticket.facts.requested_action
        if requested not in _ACTION_CATEGORIES:
            return _result(
                category, BusinessDecision.UNSUPPORTED, "UNSUPPORTED_ACTION", (), None, snapshot
            )
        assert requested is not None
        category = _ACTION_CATEGORIES[requested]
        customers = tuple(c for c in data.customers if c.customer_id == ticket.customer_id)
        require(len(customers) == 1, "business_customer_relationship")
        customer = customers[0]
        _strict(Customer, customer)
        require(
            customer.tenant_id == ticket.tenant_id
            and ticket.facts.trusted_identity_ref == customer.customer_id,
            "business_identity_relationship",
        )
        selection = catalog(data, snapshot=snapshot, category=category)
        if selection.conflicts:
            return _result(
                category,
                BusinessDecision.CONFLICT,
                "UNRESOLVED_POLICY_CONFLICT",
                ("EXPORT-02",) if category is Category.EXPORT else (),
                None,
                snapshot,
            )
        rule_ids = {
            rule_id
            for policy in data.manifest.documents
            if policy.ref in selection.authoritative_refs
            for rule_id in policy.rule_ids
        }
        required = {
            Category.BILLING: {"BILL-01", "BILL-02", "BILL-03", "BILL-EX1", "CASE-01"},
            Category.ACCESS: {"ACCESS-01", "ACCESS-02", "CASE-01"},
            Category.EXPORT: {"ACCESS-02", "EXPORT-01", "EXPORT-02", "CASE-01"},
        }[category]
        require(required <= rule_ids, "business_policy_applicability")
        if category is Category.BILLING:
            return _billing(data, ticket, customer, snapshot)
        return _access_export(ticket, customer, category, snapshot)
    except (FixtureError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
        return _result(
            category, BusinessDecision.LOAD_ERROR, "INVALID_TRUSTED_INPUT", (), None, "invalid"
        )
