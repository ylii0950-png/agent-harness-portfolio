"""Explicit trusted M1 BusinessGold reader. Never auto-loaded with public corpus."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from agent_harness_lab import WorkspaceFiles
from knowledge_ticket_agent.loaders import MAX_JSON_BYTES, inventory, read_model, read_text
from knowledge_ticket_agent.schemas import (
    ApprovalExpectation,
    BusinessDecision,
    Category,
    Customer,
    Dataset,
    Facts,
    Invoice,
    Split,
    Status,
    Synthetic,
)
from knowledge_ticket_agent.strict import decode, parse_json, require
from knowledge_ticket_agent.validation import normalized


@dataclass(frozen=True)
class Budget:
    parent_rounds: int
    child_rounds: int
    total_rounds: int
    total_tools: int
    max_depth: int


@dataclass(frozen=True)
class Anchor:
    unit_id: str
    policy_ref: str
    section_id: str
    start_char: int
    end_char: int
    quote: str


@dataclass(frozen=True)
class GoldCase(Synthetic):
    case_id: str
    split: Split
    query_id: str
    query: str
    positive_retrieval: bool
    required_coverage: bool
    expected_category: Category
    customer_facts: Customer
    invoice_facts: Invoice | None
    ticket_facts: Facts
    business_decision: BusinessDecision
    allowed_action: str
    approval_expected: ApprovalExpectation
    final_status: Status
    final_revision: int
    snapshot: str
    fixture_mode: str
    retrieval_units: tuple[str, ...]  # Semantic anchors only; NOT chunk/retrieval mappings.
    required_rule_ids: tuple[str, ...]
    permitted_actions: tuple[str, ...]
    anchors: tuple[Anchor, ...]
    forbidden_effects: tuple[str, ...]
    necessary_verification: tuple[str, ...]
    budget: Budget


@dataclass(frozen=True)
class BusinessGold(Synthetic):
    schema_version: int
    gold_version: str
    corpus_version: str
    ruleset_version: str
    as_of: str
    manifest_sha256: str
    cases: tuple[GoldCase, ...]


@dataclass(frozen=True)
class Freeze(Synthetic):
    gold_version: str
    sha256: str
    bytes: int
    cases: int
    retrieval_mapping: str


def load_gold(root: Path, data: Dataset, *, public_root: Path) -> BusinessGold:
    require(not root.resolve().is_relative_to(public_root.resolve()), "gold_in_public_corpus")
    require(inventory(root) == {"business_gold.json", "freeze.json"}, "gold_inventory")
    with WorkspaceFiles(root, max_bytes=MAX_JSON_BYTES) as workspace:
        freeze = read_model(workspace, "freeze.json", Freeze)
        text = read_text(workspace, "business_gold.json")
        require(
            len(text.encode()) == freeze.bytes
            and hashlib.sha256(text.encode()).hexdigest() == freeze.sha256,
            "gold_integrity",
        )
        gold = decode(BusinessGold, parse_json(text))
    require(
        gold.synthetic is True
        and freeze.synthetic is True
        and gold.data_origin == freeze.data_origin == "authored_fiction",
        "synthetic_label",
    )
    require(
        gold.schema_version == 1 and freeze.retrieval_mapping == "not_created_M2_only",
        "gold_schema",
    )
    require(gold.gold_version == freeze.gold_version == "lqd-business-gold-1.0.0", "gold_version")
    require(
        gold.manifest_sha256 == data.manifest_sha256
        and gold.corpus_version == data.manifest.corpus_version
        and gold.ruleset_version == data.manifest.ruleset_version
        and gold.as_of == data.manifest.as_of,
        "gold_binding",
    )
    tickets = {t.ticket_id: t for t in data.tickets}
    customers = {c.customer_id: c for c in data.customers}
    invoices = {i.invoice_id: i for i in data.invoices}
    docs = {d.metadata.ref: d for d in data.documents}
    require(len(gold.cases) == freeze.cases == len(tickets) == 36, "gold_count")
    require({g.case_id for g in gold.cases} == set(tickets), "gold_case_ids")
    require(
        len({g.query_id for g in gold.cases}) == 36
        and len({normalized(g.query) for g in gold.cases}) == 36,
        "gold_query_duplicates",
    )
    positive = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 21, 23, 24}
    for g in gold.cases:
        t = tickets[g.case_id]
        require(g.synthetic is True and g.data_origin == "authored_fiction", "synthetic_label")
        require(
            g.split == t.split and g.query == t.body and g.query_id == "Q-" + t.ticket_id[3:],
            "gold_input",
        )
        require(
            len(g.query) <= 256
            and g.customer_facts == customers[t.customer_id]
            and g.ticket_facts == t.facts
            and g.invoice_facts == invoices.get(t.facts.invoice_id or ""),
            "gold_facts",
        )
        if g.split == Split.EVAL:
            index = int(g.case_id[-2:])
            require(g.positive_retrieval == (index in positive), "gold_positive_set")
            require(g.required_coverage == (index in positive | {16, 22}), "gold_coverage_set")
        else:
            require(g.positive_retrieval and g.required_coverage, "gold_dev_set")
        require(
            (g.snapshot == "conflict") == (g.case_id in {"KT-D12", "KT-E19"}), "gold_conflict_scope"
        )
        require(g.snapshot in {"normal", "conflict"}, "gold_snapshot")
        require(
            g.fixture_mode
            in {
                "normal",
                "conflict",
                "path_probe",
                "child_write",
                "missing_document",
                "false_child",
                "readback_order",
                "injection",
                "compaction",
            },
            "fixture_mode",
        )
        require(g.budget == Budget(12, 4, 16, 28, 1), "gold_budget")
        require(len(set(g.retrieval_units)) == len(g.retrieval_units), "duplicate_unit")
        require(bool(g.anchors) == g.required_coverage, "gold_required_evidence")
        require(set(g.retrieval_units) <= {a.unit_id for a in g.anchors}, "gold_retrieval_anchors")
        require(
            len(set(g.required_rule_ids)) == len(g.required_rule_ids)
            and set(g.required_rule_ids) <= {r.rule_id for r in data.rules},
            "gold_rule_ids",
        )
        require(
            set(g.permitted_actions)
            == {"read", "propose"} | ({g.allowed_action} if g.allowed_action != "none" else set()),
            "gold_permitted_actions",
        )
        snapshot = next(s for s in data.manifest.snapshots if s.name == g.snapshot)
        for anchor in g.anchors:
            require(
                anchor.policy_ref in snapshot.documents and anchor.policy_ref in docs,
                "gold_policy_reference",
            )
            policy = docs[anchor.policy_ref]
            require(policy.metadata.kind.value not in {"legacy", "faq"}, "gold_authority")
            sections = {s.section_id: s for s in policy.sections}
            require(anchor.section_id in sections, "gold_section")
            section = sections[anchor.section_id]
            require(
                section.start_char <= anchor.start_char < anchor.end_char <= section.end_char
                and bool(anchor.quote)
                and policy.text[anchor.start_char : anchor.end_char] == anchor.quote,
                "gold_anchor",
            )
        if g.approval_expected == ApprovalExpectation.APPROVE:
            require(
                g.final_revision == 1
                and g.final_status != Status.OPEN
                and g.allowed_action == g.final_status.value,
                "gold_final_state",
            )
        else:
            require(
                g.final_revision == 0
                and g.final_status == Status.OPEN
                and g.allowed_action == "none",
                "gold_refusal_state",
            )
        require(
            set(g.forbidden_effects)
            == {
                "real_payment",
                "real_account_change",
                "real_export",
                "unrelated_ticket_change",
                "unauthorized_state_change",
            },
            "gold_forbidden_effects",
        )
        require(bool(g.necessary_verification), "gold_verification")
    return gold
