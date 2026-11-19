"""Immutable M1 contracts only; no tools, retrieval or state-changing operations."""

from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    BILLING = "billing_refund"
    ACCESS = "account_access"
    EXPORT = "data_export"
    OTHER = "other"


class Split(StrEnum):
    DEV = "dev"
    EVAL = "eval"


class Status(StrEnum):
    OPEN = "open"
    INFORMATION = "awaiting_information"
    REVIEW = "awaiting_review"
    RESOLVED = "resolved"


class DocKind(StrEnum):
    POLICY = "policy"
    LEGACY = "legacy"
    EXCEPTION = "exception"
    CONFLICT = "conflict"
    FAQ = "faq"


class Role(StrEnum):
    OWNER = "owner"
    MEMBER = "member"


class Plan(StrEnum):
    SOLO = "Solo"
    TEAM = "Team"


class StaffRole(StrEnum):
    SUPPORT = "support"
    REVIEWER = "reviewer"


class BusinessDecision(StrEnum):
    ELIGIBLE = "eligible"
    MANUAL = "manual_review"
    INELIGIBLE = "ineligible"
    INFORMATION = "needs_information"
    ESCALATE = "escalate"
    CONFLICT = "policy_conflict"
    UNSUPPORTED = "unsupported"
    LOAD_ERROR = "load_error"


class ApprovalExpectation(StrEnum):
    APPROVE = "approve"
    REFUSE = "deny"
    STATIC = "static_deny"
    NONE = "not_requested"
    MISMATCH = "scope_mismatch"


@dataclass(frozen=True)
class Synthetic:
    synthetic: bool
    data_origin: str


@dataclass(frozen=True)
class SectionSpec:
    section_id: str
    heading: str


@dataclass(frozen=True)
class Document(Synthetic):
    doc_id: str
    version: str
    title: str
    policy_family: str
    kind: DocKind
    effective_from: str
    effective_to: str | None
    scope: tuple[Category, ...]
    supersedes: tuple[str, ...]
    rule_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    path: str
    sections: tuple[SectionSpec, ...]

    @property
    def ref(self) -> str:
        return f"{self.doc_id}@{self.version}"


@dataclass(frozen=True)
class Resource:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Snapshot:
    name: str
    documents: tuple[str, ...]


@dataclass(frozen=True)
class Counts:
    documents: int
    customers: int
    staff: int
    invoices: int
    dev: int
    eval: int
    rules: int


@dataclass(frozen=True)
class Manifest(Synthetic):
    schema_version: int
    corpus_version: str
    ruleset_version: str
    profile: str
    as_of: str
    resources: tuple[Resource, ...]
    documents: tuple[Document, ...]
    snapshots: tuple[Snapshot, ...]
    counts: Counts


@dataclass(frozen=True)
class Customer(Synthetic):
    customer_id: str
    display_name: str
    tenant_id: str
    plan: Plan
    role: Role
    identity_verified: bool
    security_hold: bool


@dataclass(frozen=True)
class Staff(Synthetic):
    actor_id: str
    role: StaffRole
    allowed_ticket_ids: tuple[str, ...]
    allowed_actions: tuple[str, ...]


@dataclass(frozen=True)
class Invoice(Synthetic):
    invoice_id: str
    customer_id: str
    tenant_id: str
    purchased_at: str
    amount_minor: int
    unit: str
    first_purchase: bool
    already_credited: bool


@dataclass(frozen=True)
class Facts:
    invoice_id: str | None
    trusted_identity_ref: str
    requested_action: str | None
    request_scope: str | None
    exception_id: str | None


@dataclass(frozen=True)
class Ticket(Synthetic):
    ticket_id: str
    split: Split
    customer_id: str
    tenant_id: str
    submitted_at: str
    revision: int
    status: Status
    last_action_id: str | None
    subject: str
    body: str
    claimed_category: Category | None
    facts: Facts


@dataclass(frozen=True)
class Condition:
    field: str
    operator: str
    value: str
    policy_ref: str
    section_id: str
    statement: str


@dataclass(frozen=True)
class RuleBranch:
    effect: BusinessDecision
    conditions: tuple[Condition, ...]


@dataclass(frozen=True)
class Rule(Synthetic):
    rule_id: str
    version: str
    scope: tuple[Category, ...]
    overrides: tuple[str, ...]
    branches: tuple[RuleBranch, ...]


@dataclass(frozen=True)
class CustomersFile(Synthetic):
    version: str
    customers: tuple[Customer, ...]


@dataclass(frozen=True)
class StaffFile(Synthetic):
    version: str
    staff: tuple[Staff, ...]


@dataclass(frozen=True)
class InvoicesFile(Synthetic):
    version: str
    invoices: tuple[Invoice, ...]


@dataclass(frozen=True)
class TicketsFile(Synthetic):
    version: str
    tickets: tuple[Ticket, ...]


@dataclass(frozen=True)
class RulesFile(Synthetic):
    version: str
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class Section:
    section_id: str
    heading: str
    body: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class PolicyText:
    metadata: Document
    text: str
    sections: tuple[Section, ...]


@dataclass(frozen=True)
class Dataset:
    manifest: Manifest
    manifest_sha256: str
    documents: tuple[PolicyText, ...]
    customers: tuple[Customer, ...]
    staff: tuple[Staff, ...]
    invoices: tuple[Invoice, ...]
    tickets: tuple[Ticket, ...]
    rules: tuple[Rule, ...]
